"""#1184: the three real Table 1 layouts, replayed from frozen word boxes.

The 43/43, 30 and 28 results were measured in ``scratch/`` against PDFs that
are not in this repository, so until now nothing in the suite held the real
geometry the binding tolerance was derived from (PR #1195 review). The
fixtures beside this file are exactly what PyMuPDF reports for those pages --
word boxes and the text direction the page records -- so replaying one drives
the parser over the real layout with no PDF, no network and no model.

Regenerate with ``scratch/routing-baseline/leads/1184/probes/freeze_pages.py``.

The 2015-09-17 expectations are the Fed's OWN published figures for that
release (``fomcprojtabl20150917.htm``, "Advance release of table 1"), carried
here as literal data with that provenance because CI has no network. All 43 of
its published range cells were compared in
``probes/score_2015.py``: 43 agree, 0 wrong, 0 missing.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from socr.figures.chart_witness import (
    LAYOUT_SIDE_BY_SIDE_GROUPS,
    LAYOUT_STACKED_BLOCKS,
    parse_table1_witness,
)

_FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "gh1184"


class _FrozenPage:
    """A page that answers the two questions the parser asks of one."""

    def __init__(self, payload):
        self._words = [tuple(w) for w in payload["words"]]
        self.rotation = 0

    def get_text(self, kind):
        assert kind == "words", kind
        return list(self._words)


@pytest.fixture
def replay(monkeypatch):
    def _replay(name):
        payload = json.loads((_FIXTURES / f"{name}.json").read_text())
        import socr.core.born_digital as born_digital

        monkeypatch.setattr(born_digital, "upright_rotation_for", lambda page: payload["rotation"])
        return payload, parse_table1_witness(_FrozenPage(payload))

    return _replay


def test_2010_reads_every_row_including_the_one_the_count_rule_rejected(replay):
    """2010-04-28 p11. Upright, groups side by side, no median group.

    The count rule rejected both core PCE rows here -- six cells against a
    header of eight -- because the Fed does not collect longer-run core PCE
    (that release's own footnote 3). Those rows are the point of this fixture.
    """
    payload, witness = replay("2010-04-28")
    assert payload["rotation"] == 0
    assert witness.refusal == ""
    assert witness.layout == LAYOUT_SIDE_BY_SIDE_GROUPS
    assert len(witness.rows) == 30
    assert witness.rejected == ()
    assert witness.range_for("Change in real GDP", "current", "2010") == (2.7, 4.0)
    assert witness.range_for("Core PCE inflation", "current", "2012") == (0.6, 2.2)
    assert witness.range_for("Core PCE inflation", "current", "Longer run") is None, (
        "the column the release does not print yields no witness, and costs nothing else"
    )
    assert witness.prior_months == frozenset({"January"})


def test_2008_reads_a_table_split_into_stacked_blocks(replay):
    """2008-10-29 p12. One year header, a central-tendency block over a range block.

    Every row carries four cells against a header of eight, so the count rule
    refused the whole table. This page also draws a prose column beside the
    table, and one of its bands carries both.
    """
    payload, witness = replay("2008-10-29")
    assert witness.refusal == ""
    assert witness.layout == LAYOUT_STACKED_BLOCKS
    assert len(witness.rows) == 28
    assert witness.range_for("Change in real GDP", "current", "2008") == (-0.3, 0.5), (
        "and a negative projection keeps its sign"
    )
    assert witness.range_for("Unemployment rate", "prior", "2010") == (5.0, 5.8)
    assert witness.range_for("Unemployment rate", "prior", "2011") is None, (
        "the n/a cells of a prior row yield no witness"
    )
    reasons = [reason for _, _, reason in witness.rejected]
    assert len(reasons) == 1, "only the prose band is refused"


#: The Fed's own published range figures for 2015-09-17, variable by year.
#: Source: fomcprojtabl20150917.htm, "Advance release of table 1".
_FED_2015_RANGE = {
    ("Change in real GDP", "current"): {
        "2015": (1.9, 2.5),
        "2016": (2.1, 2.8),
        "2017": (1.9, 2.6),
        "2018": (1.6, 2.4),
        "Longer run": (1.8, 2.7),
    },
    ("Unemployment rate", "current"): {
        "2015": (4.9, 5.2),
        "2016": (4.5, 5.0),
        "2017": (4.5, 5.0),
        "2018": (4.6, 5.3),
        "Longer run": (4.7, 5.8),
    },
    ("PCE inflation", "current"): {
        "2015": (0.3, 1.0),
        "2016": (1.5, 2.4),
        "2017": (1.7, 2.2),
        "2018": (1.8, 2.1),
        "Longer run": (2.0, 2.0),
    },
    ("Federal funds rate", "current"): {
        "2015": (-0.1, 0.9),
        "2016": (-0.1, 2.9),
        "2017": (1.0, 3.9),
        "2018": (2.9, 3.9),
        "Longer run": (3.0, 4.0),
    },
}


def test_2015_reads_a_table_drawn_rotated_and_matches_the_published_figures(replay):
    """2015-09-17 p13. Drawn at 90 degrees inside an unrotated page.

    ``page.rotation`` is 0 here, which is why the direction is taken from what
    the page records. This release also prints a collapsed range as one figure
    (longer-run PCE inflation, 2.0), negative federal-funds projections, and a
    core PCE row whose label wraps so the data band reads only "Core PCE".
    """
    payload, witness = replay("2015-09-17")
    assert payload["rotation"] == 90
    assert witness.refusal == ""
    assert witness.layout == LAYOUT_SIDE_BY_SIDE_GROUPS
    assert len(witness.rows) == 43
    assert witness.rejected == ()
    for (variable, series_row), by_year in _FED_2015_RANGE.items():
        for year, published in by_year.items():
            assert witness.range_for(variable, series_row, year) == published, (
                f"{variable} {series_row} {year}"
            )
    assert witness.range_for("PCE inflation", "current", "Longer run") == (2.0, 2.0), (
        "a collapsed range is read as the degenerate interval it is"
    )
    assert witness.range_for("Core PCE inflation", "current", "Longer run") is None
    assert witness.variable_for("Core PCE inflation") == "core pce", (
        "the wrapped label is matched by prefix, not reassembled"
    )


def test_the_tolerance_the_real_pages_get_is_the_one_that_was_measured(replay):
    """The geometry the binding rests on, pinned on real pages rather than fixtures."""
    from socr.figures.chart_witness import _bands, _cells_of

    for name in ("2010-04-28", "2008-10-29", "2015-09-17"):
        payload, _ = replay(name)
        _, bands = _bands(_FrozenPage(payload))
        spans = sorted(
            c.span for tokens in bands.values() for c in _cells_of(tokens) if c.span is not None
        )
        separation = spans[len(spans) // 2]
        # Measured over the three releases when the rule was chosen:
        # 20.20pt on 2015-09-17 and 22.19-22.20pt on the other two.
        assert 20.0 <= separation <= 22.5, (name, separation)
        assert 5.0 <= separation / 4 <= 5.6, (name, separation / 4)


@pytest.mark.parametrize(
    "name,lost,variable,year,others",
    [
        ("2010-04-28", "2.7", "Change in real GDP", "2010", 30),
        ("2015-09-17", "1.9", "Change in real GDP", "2015", 43),
    ],
)
def test_losing_one_printed_figure_of_a_real_cell_refuses_that_row(
    replay, monkeypatch, name, lost, variable, year, others
):
    """The claim the tolerance exists to make, tested on the real pages.

    A cell that loses one of its two figures is no longer printed centred in
    its column -- the survivor sits a whole figure off -- so it binds to
    nothing and the row is refused rather than shipping half a range. Fixtures
    drawn to a grid cannot show that the real pages leave enough room for it;
    these are the real word boxes, with one token deleted.
    """
    import socr.core.born_digital as born_digital

    payload = json.loads((_FIXTURES / f"{name}.json").read_text())
    monkeypatch.setattr(born_digital, "upright_rotation_for", lambda page: payload["rotation"])
    clean = parse_table1_witness(_FrozenPage(payload))
    assert len(clean.rows) == others
    assert clean.range_for(variable, "current", year) is not None

    hits = [w for w in payload["words"] if w[4] == lost]
    assert len(hits) >= 1, lost
    payload["words"] = [w for w in payload["words"] if w is not hits[0]]
    damaged = parse_table1_witness(_FrozenPage(payload))

    assert damaged.range_for(variable, "current", year) is None, (
        "the row that lost a figure ships nothing"
    )
    assert len(damaged.rows) < len(clean.rows)
    assert damaged.rejected, "and says so by name"
