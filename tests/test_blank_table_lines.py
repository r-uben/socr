"""Blank rows/columns in a model table, and the escaped-star header token.

Fed 2020 Tealbook p27 shipped as a failed-table marker although every value was
read correctly: the model wrote an empty header row, an empty spacer column
between column groups, and ``**Historical\\***`` for the native ``Historical*``.
Hermetic: native words and rules are a stored snapshot, no provider is called.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from socr.tables.header_attribution import HeaderVerdict
from socr.tables.header_cut import _fold, header_cut_verdict
from socr.tables.reconcile import drop_blank_table_lines as _drop
from socr.tables.reconcile import find_table_blocks


def drop_blank_table_lines(markdown):
    return _drop(markdown).text


FIX = Path(__file__).parent / "fixtures" / "tb2020_p27"


def _p27():
    md = (FIX / "qwen_raw.md").read_text()
    words = [tuple(w) for w in json.loads((FIX / "words.json").read_text())]
    rules = [tuple(r) for r in json.loads((FIX / "rules.json").read_text())]
    return md, words, rules


def test_blank_column_and_row_are_dropped():
    md = "| | A | | B |\n| --- | --- | --- | --- |\n| | | | |\n| x | 1 | | 2 |\n"
    out = drop_blank_table_lines(md)
    assert out == "|  | A | B |\n| --- | --- | --- |\n| x | 1 | 2 |\n"


def test_blank_header_row_moves_separator_under_first_content_row():
    md = "| | |\n| --- | --- |\n| | H |\n| a | 1 |\n"
    out = drop_blank_table_lines(md)
    assert out.splitlines()[:2] == ["|  | H |", "| --- | --- |"]


@pytest.mark.parametrize("blank", ["", " ", "&nbsp;", "&nbsp; &nbsp;", "****", "____"])
def test_every_blank_spelling_is_dropped(blank):
    md = f"| A | {blank} | B |\n| --- | --- | --- |\n| 1 | {blank} | 2 |\n"
    assert drop_blank_table_lines(md) == "| A | B |\n| --- | --- |\n| 1 | 2 |\n"


@pytest.mark.parametrize("content", ["*", "-", "0", "n/a", "**0**", "&nbsp;x"])
def test_any_content_keeps_its_column(content):
    md = f"| A | B | C |\n| --- | --- | --- |\n| 1 | {content} | 2 |\n"
    assert drop_blank_table_lines(md) == md


def test_clean_ragged_and_non_table_text_are_unchanged():
    clean = "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
    ragged = "| A | B | |\n| --- | --- |\n| 1 | | |\n"
    prose = "No table here | just a pipe.\n"
    for md in (clean, ragged, prose, ""):
        assert drop_blank_table_lines(md) == md


def test_an_all_blank_table_is_left_alone():
    md = "| | |\n| --- | --- |\n| | |\n"
    assert drop_blank_table_lines(md) == md


def test_surrounding_text_and_a_second_table_survive():
    md = "intro\n\n| A | |\n| --- | --- |\n| 1 | |\n\n| X | Y |\n| --- | --- |\n| 3 | 4 |\n\nend\n"
    out = drop_blank_table_lines(md)
    assert out.startswith("intro\n\n| A |\n| --- |\n| 1 |\n\n| X | Y |")
    assert out.endswith("| 3 | 4 |\n\nend\n")


def test_p27_shipped_table_has_no_phantom_column_or_row():
    md, _, _ = _p27()
    before = find_table_blocks(md)[0].grid
    after = find_table_blocks(drop_blank_table_lines(md))[0].grid
    assert len(before[0]) == 10 and len(after[0]) == 9
    assert all(len(r) == 9 for r in after)
    flat = lambda g: sorted(c for r in g for c in r if c.strip())  # noqa: E731
    assert flat(before) == flat(after)  # no content lost


def test_escaped_star_folds_to_the_native_token():
    assert _fold("**Historical\\***") == _fold("Historical*") == "historical"
    assert _fold("a\\_b") == _fold("a_b")


def test_p27_header_is_attributed_only_because_of_the_escaped_star_fold(monkeypatch):
    """Pins a DIFFERENCE: the same grid is HARD without the fold and OK with it."""
    from socr.tables import header_cut

    md, words, rules = _p27()
    grid = find_table_blocks(drop_blank_table_lines(md))[0].grid
    assert header_cut_verdict(grid, words, rules) is HeaderVerdict.OK
    monkeypatch.setattr(header_cut, "_MD_ESCAPED_EMPHASIS_RE", re.compile(r"(x)(?!)"))
    assert header_cut_verdict(grid, words, rules) is HeaderVerdict.HARD


def test_p27_spacer_column_no_longer_shifts_the_lanes():
    """Binding: the raw table's spacer column leaves columns unverifiable."""
    from socr.tables.binding import bind

    md, words, _ = _p27()
    region = (71.95, 145.98, 552.09, 531.94)
    raw = bind(words, md, region=region, spans=[])
    fixed = bind(words, drop_blank_table_lines(md), region=region, spans=[])
    assert raw.column_binding_unverifiable and not fixed.column_binding_unverifiable
    assert len(fixed.matched_cells) > len(raw.matched_cells)


# --- a label the PAGE wraps onto a second baseline (numbers on the first) ----

_LABEL = "Reserve balances held by depository institutions"


def _spans():
    return json.loads((FIX / "spans.json").read_text())


def _bind_p27(md, *, spans):
    from socr.tables.binding import bind

    _, words, _ = _p27()
    return bind(
        words, drop_blank_table_lines(md), region=(71.95, 145.98, 552.09, 531.94), spans=spans
    )


def _labels(result):
    return [c.candidate_label for c in result.row_label_contradictions]


def test_wrapped_native_label_is_not_a_contradiction_with_font_evidence():
    md, _, _ = _p27()
    assert _LABEL not in _labels(_bind_p27(md, spans=_spans()))


def test_wrapped_native_label_without_font_evidence_stays_a_contradiction():
    """Fail closed: no spans, no merge. Pins the DIFFERENCE spans make."""
    md, _, _ = _p27()
    assert _LABEL in _labels(_bind_p27(md, spans=[]))


def test_a_different_model_label_is_still_a_contradiction():
    md, _, _ = _p27()
    wrong = md.replace(_LABEL, "Reserve balances held by banks")
    assert wrong != md
    assert "Reserve balances held by banks" in _labels(_bind_p27(wrong, spans=_spans()))


def _row(label, *, parent, x0, multiset=(), font=("F", 10, False), size=10.0):
    from collections import Counter

    from socr.tables.binding import _NativeRow

    return _NativeRow(
        y=0.0,
        row_path=(label,),
        is_parent=parent,
        lane_tokens={},
        multiset=Counter(multiset),
        band_ambiguous=False,
        label_bbox=(x0, 0.0, x0 + 50, 10.0),
        label_font=font,
        label_font_size=size,
    )


def test_continuation_helper_merges_only_a_valueless_same_font_unindented_row():
    from socr.tables.binding import _native_label_with_trailing_wrap as wrap

    head = _row("held by", parent=False, x0=97, multiset=["1"])
    assert wrap([head, _row("banks", parent=True, x0=99)], 0) == "held by banks"
    # a number on the second line: it is a data row, not a continuation
    assert wrap([head, _row("banks", parent=False, x0=99, multiset=["2"])], 0) == ""
    # a value-less row that kept a number-bearing multiset is not a continuation
    assert wrap([head, _row("banks", parent=True, x0=99, multiset=["2"])], 0) == ""
    # nested more than one em: a child heading, not a wrap
    assert wrap([head, _row("banks", parent=True, x0=97 + 11)], 0) == ""
    # different font, or no font data
    assert wrap([head, _row("banks", parent=True, x0=99, font=("G", 10, False))], 0) == ""
    assert (
        wrap(
            [_row("x", parent=False, x0=97, font=None), _row("y", parent=True, x0=97, font=None)], 0
        )
        == ""
    )
    # first row is itself a heading
    assert wrap([_row("x", parent=True, x0=97), _row("y", parent=True, x0=97)], 0) == ""
    # nothing below
    assert wrap([head], 0) == ""


# --- a heading tier written as a body row; a footnote below the closing rule --

_REGION = (71.95297241210938, 145.97998046875, 552.09326171875, 531.9361572265625)


class _Page:
    """The slice of a PyMuPDF page the region check reads: its words."""

    def __init__(self, words):
        self._words = words

    def get_text(self, kind):
        assert kind == "words"
        return self._words


def _check(md, *, page):
    from socr.tables.native_cell_check import check_table_cell_binding

    _, words, _ = _p27()
    return check_table_cell_binding(
        table_id="t", words=words, markdown=md, region_bbox=_REGION, spans=_spans(), page=page
    )


def test_year_tier_row_is_ambiguous_not_contradicted_and_not_verified():
    from socr.tables.native_cell_check import NativeCellOutcome

    md, words, _ = _p27()
    fixed = drop_blank_table_lines(md)
    bound = _bind_p27(md, spans=_spans())
    assert bound.contradicted_cells == []
    assert bound.ambiguous_count >= 8  # the tier is counted unverified
    assert len(bound.matched_cells) > 100  # the data rows still bind
    result = _check(fixed, page=_Page(words))
    assert result.outcome is NativeCellOutcome.AMBIGUOUS


def test_a_year_tier_with_a_different_year_is_still_a_contradiction():
    md, _, _ = _p27()
    wrong = md.replace("Sep 2017", "Sep 2018")
    assert wrong != md
    assert _bind_p27(wrong, spans=_spans()).contradicted_cells


def test_header_tier_predicate_is_narrow():
    from collections import Counter

    from socr.tables.binding import _NativeRow, model_row_is_header_tier

    native = _NativeRow(
        y=0.0,
        row_path=("",),
        is_parent=False,
        lane_tokens={0: ("2017", False), 1: ("2030", False)},
        multiset=Counter(["2017", "2030"]),
        band_ambiguous=False,
    )
    assert model_row_is_header_tier(native, ("", "Sep 2017", "**Dec 2030**"))
    assert not model_row_is_header_tier(native, ("", "Sep 2017", "Dec 2031"))  # other year
    assert not model_row_is_header_tier(native, ("", "2017", "2030"))  # numbers: ordinary path
    assert not model_row_is_header_tier(native, ("Total", "Sep 2017", "Dec 2030"))  # labelled
    assert not model_row_is_header_tier(native, ("", "Sep 2017", "Dec 2030", "x"))  # stray cell
    assert not model_row_is_header_tier(native, ("", "Sep 2017", ""))  # a lane unread


def test_footnote_below_the_closing_rule_is_not_pulled_into_the_region(monkeypatch):
    """Pins a DIFFERENCE: only the closing-rule test changes the region."""
    from socr.tables import locate
    from socr.tables.native_contradiction import extend_check_region_for_trailing_words as ext

    _, words, rules = _p27()
    page = _Page(words)
    monkeypatch.setattr(locate, "_horizontal_rules", lambda _p: rules)
    assert ext(page, _REGION, "| a |") == _REGION
    monkeypatch.setattr(locate, "_horizontal_rules", lambda _p: [])
    assert ext(page, _REGION, "| a |")[3] > _REGION[3]  # no closing rule: extended as before
    far = [(_REGION[3] - 40, _REGION[0], _REGION[2])]  # a rule that is not the bottom edge
    monkeypatch.setattr(locate, "_horizontal_rules", lambda _p: far)
    assert ext(page, _REGION, "| a |")[3] > _REGION[3]


# --- review round: counts, kept header column, escaped pipe, outdent, closing rule ---


def test_counts_are_reported():
    md = "| | A | | B |\n| --- | --- | --- | --- |\n| | | | |\n| x | 1 | | 2 |\n"
    r = _drop(md)
    assert (r.rows, r.columns) == (1, 1)
    clean = "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
    assert _drop(clean) == (clean, 0, 0)


def test_a_header_over_a_blank_body_column_is_kept():
    md = "| A | B | C |\n| --- | --- | --- |\n| 1 |  | 2 |\n| 3 |  | 4 |\n"
    assert _drop(md).text == md


def test_a_block_with_an_escaped_pipe_is_left_alone():
    # Split on the escaped pipe the block looks rectangular (3 cells) with a blank last
    # column; dropping it would corrupt the first header cell.
    md = "| A \\| B | |\n| --- | --- | --- |\n| 1 | 2 | |\n"
    assert _drop(md).text == md


def test_the_page_gate_records_the_drop_and_ships_the_cleaned_text():
    from unittest.mock import MagicMock, patch

    import fitz

    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.agentic import NativeTableVerifierJudge
    from socr.tables.reconcile import TABLE_BLANK_LINES_DROPPED_KIND

    def run(text):
        events = []
        judge = NativeTableVerifierJudge(
            inner=MagicMock(),
            get_fitz_page=lambda pn: fitz.open().new_page(),
            is_table_page=lambda pn: True,
            record_event=events.append,
        )
        out = PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, engine="qwen")
        with patch(
            "socr.tables.native_verifier.verify_native_table", side_effect=RuntimeError("stop")
        ):
            judge.assess(out, MagicMock())
        return out, [e for e in events if e.kind == TABLE_BLANK_LINES_DROPPED_KIND]

    out, ev = run("| A | | B |\n| --- | --- | --- |\n| 1 | | 2 |\n")
    assert out.text == "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
    assert len(ev) == 1 and ev[0].data == {"rows_dropped": 0, "columns_dropped": 1}
    out, ev = run("| A | B |\n| --- | --- |\n| 1 | 2 |\n")
    assert ev == []


def test_outdented_continuation_is_not_a_wrap():
    from socr.tables.binding import _native_label_with_trailing_wrap as wrap

    head = _row("held by", parent=False, x0=97, multiset=["1"])
    assert wrap([head, _row("banks", parent=True, x0=97)], 0) == "held by banks"  # dx == 0
    assert wrap([head, _row("banks", parent=True, x0=90)], 0) == ""  # starts left of its line


def _total_row_page(*, total_numeric=True):
    """Body rows on lanes, a full-width rule, then a row beneath it."""
    body = []
    for i, y in enumerate((100.0, 112.0, 124.0)):
        for x, t in ((200.0, f"{i}1"), (260.0, f"{i}2"), (320.0, f"{i}3")):
            body.append((x, y, x + 20, y + 9, t, 0, i))
    tail = (
        [(200.0, 148.0, 220.0, 157.0, "99", 0, 9), (260.0, 148.0, 280.0, 157.0, "98", 0, 9)]
        if total_numeric
        else [
            (100.0, 148.0, 130.0, 157.0, "since", 1, 9),
            (137.0, 148.0, 160.0, 157.0, "2017", 1, 9),
        ]
    )
    return _Page(body + tail)


def test_a_rule_above_a_lane_aligned_total_row_does_not_close_the_region(monkeypatch):
    from socr.tables import locate
    from socr.tables.native_contradiction import extend_check_region_for_trailing_words as ext

    region = (150.0, 90.0, 400.0, 140.0)
    monkeypatch.setattr(locate, "_horizontal_rules", lambda _p: [(140.0, 150.0, 400.0)])
    assert ext(_total_row_page(), region, "| a |")[3] > region[3]  # Total row is checked
    assert ext(_total_row_page(total_numeric=False), region, "| a |") == region  # footnote prose


def test_a_wrong_total_under_a_midrule_is_still_contradicted(monkeypatch):
    """The Total row stays inside the region, so a wrong Total is a contradiction."""
    from socr.tables import locate
    from socr.tables.binding import bind
    from socr.tables.native_contradiction import extend_check_region_for_trailing_words as ext

    page = _total_row_page()
    region = (150.0, 90.0, 400.0, 140.0)
    monkeypatch.setattr(locate, "_horizontal_rules", lambda _p: [(140.0, 150.0, 400.0)])
    wide = ext(page, region, "| a |")
    md = (
        "| item | a | b | c |\n| --- | --- | --- | --- |\n"
        "| r0 | 01 | 02 | 03 |\n| r1 | 11 | 12 | 13 |\n| r2 | 21 | 22 | 23 |\n"
        "| Total | 99 | 55 | |\n"
    )
    words = page._words
    wrong = bind(words, md, region=wide, spans=[])
    assert any(c.native_token == "98" for c in wrong.contradicted_cells)
    closed = bind(words, md, region=region, spans=[])
    assert not any(c.native_token == "98" for c in closed.contradicted_cells)


def test_a_kept_candidate_with_a_blank_column_can_be_rejudged(tmp_path, monkeypatch):
    """#1013: the judge chain drops blank columns; the re-judge must hash the cleaned bytes."""
    import test_gh1013_rejudge_timed_out_candidate as t
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline.orchestrator import UnifiedPipeline

    monkeypatch.setattr(
        UnifiedPipeline, "_available_engines_for_agentic", lambda self: [PROFILE_QWEN_LOCAL]
    )
    monkeypatch.setattr(UnifiedPipeline, "_resolve_judge_model", lambda self, *a, **kw: "")

    text = "| A | | B |\n| --- | --- | --- |\n| 1 | | 2 |\n"

    class Cleaning(t._Judge):
        def assess(self, output, provider):
            self.calls.append(output.text)
            output.text = drop_blank_table_lines(output.text)
            return t.AcceptDecision(
                accept=True, reason="ok", raw_verdict=t.JudgeVerdict(faithful=True)
            )

    h = t._Harness(tmp_path)
    h.ocr_text = text
    t._run1_times_out(h)
    h.judge = Cleaning()
    h.ocr_text = "| other | x |\n| --- | --- |\n| 9 | 9 |\n"
    h.run()
    kinds = t._kinds(h)
    assert "rejudge_accepted" in kinds and "rejudge_error" not in kinds
    assert h.ocr_calls == []
