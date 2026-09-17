"""#747: the Fed accessible-page parser and the corpus scorer's per-cell classifier.

No network and no corpus PDF here -- `parse_release_html` and `_classify_series`
are pure functions over inline fixtures, so this suite is hermetic by
construction rather than by a patched provider ladder. The real corpus and the
real Fed pages are exercised by `socr-score-sep-ground-truth` itself (a
developer instrument, not a gate -- see its module docstring), not by this
suite.
"""

from __future__ import annotations

import urllib.error

import pytest

from socr.figures.score_sep_ground_truth import (
    EXACT,
    FABRICATED,
    MISSING,
    NO_GROUND_TRUTH,
    WRONG_BIN,
    WRONG_COUNT,
    _classify_series,
    _model_readings,
)
from socr.figures.sep_ground_truth import (
    GroundTruthUnavailable,
    fetch_release_html,
    parse_release_html,
)

# ---------------------------------------------------------------------------
# parse_release_html
# ---------------------------------------------------------------------------

_SYMMETRIC_TABLE = """
<p>Figure 3.E. Distribution of participants' judgments</p>
<table>
<thead>
<tr>
<th class="colhead" id="a1" rowspan="2">Percent Range</th>
<th class="colhead" colspan="2" id="a2">2022</th>
</tr>
<tr>
<th class="colhead" headers="a2" id="b1">December projections</th>
<th class="colhead" headers="a2" id="b2">March projections</th>
</tr>
</thead>
<tbody>
<tr>
<th class="stub" headers="a1" id="r1">0.13 - 0.37</th>
<td class="data" headers="a2 b1 r1">18</td>
<td class="emptystub">&nbsp;</td>
</tr>
<tr>
<th class="stub" headers="a1" id="r2">0.38 - 0.62</th>
<td class="emptystub">&nbsp;</td>
<td class="data" headers="a2 b2 r2">3</td>
</tr>
</tbody>
</table>
<p>Figure 3.F. Something else entirely</p>
"""


def test_parse_release_html_reads_a_symmetric_two_column_year() -> None:
    rt = parse_release_html(_SYMMETRIC_TABLE, release_date="20220316")
    panel = rt.panel("2022")
    assert panel is not None
    assert panel.count_for("December projections", "0.13-0.37") == 18
    # An emptystub cell is an explicit zero, not an absent reading -- a caller
    # scoring a reader/model value there must be able to tell "the Fed
    # published zero" apart from "this parser has no opinion".
    assert panel.count_for("December projections", "0.38-0.62") == 0
    assert panel.count_for("March projections", "0.38-0.62") == 3


_ASYMMETRIC_TABLE = """
<p>Figure 3.E. Distribution</p>
<table>
<thead>
<tr>
<th class="colhead" id="a1" rowspan="2">Percent Range</th>
<th class="colhead" colspan="2" id="a2">2022</th>
<th class="colhead" id="a3">2024</th>
</tr>
<tr>
<th class="colhead" headers="a2" id="b1">June projections</th>
<th class="colhead" headers="a2" id="b2">September projections</th>
<th class="colhead" headers="a3" id="b3">September projections</th>
</tr>
</thead>
<tbody>
<tr>
<th class="stub" headers="a1" id="r1">0.13 - 0.37</th>
<td class="data" headers="a2 b1 r1">18</td>
<td class="data" headers="a2 b2 r1">16</td>
<td class="data" headers="a3 b3 r1">5</td>
</tr>
</tbody>
</table>
"""


def test_parse_release_html_handles_a_year_with_one_column_not_two() -> None:
    """The September-quarter case measured on 5 of 23 corpus releases: a year
    whose prior meeting did not project that far out gets one column, not
    two. A parser that assumes every year panel is exactly 2 columns wide
    raises here instead of silently misreading the layout (this is the exact
    shape that failed before this ticket's rewrite)."""
    rt = parse_release_html(_ASYMMETRIC_TABLE, release_date="20210922")
    assert rt.panel("2022").count_for("June projections", "0.13-0.37") == 18
    assert rt.panel("2024").count_for("September projections", "0.13-0.37") == 5
    assert rt.panel("2024").count_for("June projections", "0.13-0.37") is None


def test_parse_release_html_raises_when_no_figure_heading() -> None:
    with pytest.raises(GroundTruthUnavailable):
        parse_release_html("<p>Figure 2. Something unrelated</p>", release_date="20200916")


def test_fetch_release_html_turns_a_cold_cache_403_into_a_legible_error(monkeypatch) -> None:
    """Measured on a fresh checkout: federalreserve.gov 403s urllib's default
    User-Agent, and that only surfaces on the FIRST run against a given
    release -- every later run reads the cache and never calls urlopen. A
    raw `HTTPError` traceback from inside `urlopen` gives a reader no clue
    the harness has a cache mechanism at all; this must come out as
    `GroundTruthUnavailable`, naming the URL and the status, same as every
    other refusal this module raises."""

    def _forbidden(request, timeout=None):  # noqa: ARG001
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)

    import socr.figures.sep_ground_truth as sep_ground_truth

    monkeypatch.setattr(sep_ground_truth.urllib.request, "urlopen", _forbidden)
    with pytest.raises(GroundTruthUnavailable, match="20220101"):
        fetch_release_html("20220101", timeout=1)


def test_parse_release_html_raises_on_a_headers_id_mismatch() -> None:
    """A row whose <td> headers id disagrees with its table position is a
    malformed table this parser does not understand -- refuse rather than
    silently attribute the count to the wrong column."""
    broken = _SYMMETRIC_TABLE.replace('headers="a2 b1 r1"', 'headers="a2 b2 r1"')
    with pytest.raises(GroundTruthUnavailable):
        parse_release_html(broken, release_date="broken")


# ---------------------------------------------------------------------------
# _classify_series
# ---------------------------------------------------------------------------


def test_classify_series_exact_match() -> None:
    scores = _classify_series(
        "doc", "2022", "March", ["a", "b"], {"a": 5, "b": 0}, {"a": 5, "b": 0}
    )
    assert [s.outcome for s in scores] == [EXACT, EXACT]


def test_classify_series_fabricated_is_a_nonzero_reading_where_truth_is_zero() -> None:
    scores = _classify_series("doc", "2022", "March", ["a"], {"a": 0}, {"a": 2})
    assert scores[0].outcome == FABRICATED


def test_classify_series_wrong_count_is_a_plain_miscount() -> None:
    scores = _classify_series("doc", "2022", "March", ["a"], {"a": 5}, {"a": 3})
    assert scores[0].outcome == WRONG_COUNT


def test_classify_series_wrong_bin_is_an_adjacent_paired_swap() -> None:
    """A value placed one bin over from where the Fed puts it: in isolation
    the destination bin reads as a fabrication and the source bin reads as
    an undercount, but together they are one swap, not two independent
    errors -- this is the #747 issue's own hand-checked case (r3, 2022)."""
    scores = _classify_series(
        "doc", "2022", "March", ["a", "b"], {"a": 0, "b": 1}, {"a": 1, "b": 0}
    )
    assert [s.outcome for s in scores] == [WRONG_BIN, WRONG_BIN]


def test_classify_series_missing_when_no_reading() -> None:
    scores = _classify_series("doc", "2022", "March", ["a"], {"a": 5}, {})
    assert scores[0].outcome == MISSING


def test_classify_series_no_ground_truth_when_reading_names_an_unknown_bin() -> None:
    scores = _classify_series("doc", "2022", "March", ["a"], {}, {"a": 3})
    assert scores[0].outcome == NO_GROUND_TRUTH


def test_classify_series_a_non_adjacent_pair_is_not_a_wrong_bin_swap() -> None:
    """Two errors of equal and opposite magnitude that are NOT next to each
    other in bin order are two independent mistakes, not one swap -- a
    classifier that checked every pair, not just adjacent ones, would
    over-forgive this case."""
    scores = _classify_series(
        "doc", "2022", "March", ["a", "b", "c"], {"a": 0, "b": 5, "c": 1}, {"a": 1, "b": 5, "c": 0}
    )
    outcomes = {s.bin_label: s.outcome for s in scores}
    assert outcomes["a"] == FABRICATED
    assert outcomes["b"] == EXACT
    assert outcomes["c"] == WRONG_COUNT


# ---------------------------------------------------------------------------
# _model_readings -- panel-heading and grid-orientation detection
# ---------------------------------------------------------------------------

_BOLD_HEADING_MD = """### Page 1

**2020**
| Percent range | December projections | September projections |
| :--- | :---: | :---: |
| 0.13-0.37 | 17 | 17 |
| 0.38-0.62 | 0 | 1 |
"""


def test_model_readings_understands_a_bold_only_panel_heading(tmp_path) -> None:
    """Measured on `sep-20201216-p09`: every panel heading in that document is
    a bold-only line, not a Markdown `#` heading -- a scanner that only
    recognises `#` headings falls back to the page's own `## Page 1` heading
    and merges every panel's grid into one bucket."""
    doc_dir = tmp_path
    (doc_dir / "doc.md").write_text(_BOLD_HEADING_MD)
    readings = _model_readings(doc_dir, "doc")
    labels = [label for label, _by_series in readings]
    assert labels == ["2020"]
    (_label, by_series) = readings[0]
    assert by_series["December projections"]["0.13|0.37"] == 17


_TRANSPOSED_MD = """### 2028
| Percent range | 1.88-2.12 | 2.13-2.37 |
| :--- | :---: | :---: |
| Number of Participants | 0 | 1 |
"""


def test_model_readings_skips_a_transposed_grid(tmp_path) -> None:
    """Measured on `sep-20251210-p09`: some grids carry bins as the COLUMN
    headers and one generic row label instead of a bin-labelled row per
    projection month. That row names no projection month a ground-truth
    column could ever match, so the grid should contribute nothing rather
    than being scored under a "series" literally named for a bin range."""
    doc_dir = tmp_path
    (doc_dir / "doc.md").write_text(_TRANSPOSED_MD)
    readings = _model_readings(doc_dir, "doc")
    assert readings == []
