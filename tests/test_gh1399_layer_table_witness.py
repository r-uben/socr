"""#1399: the stored old-OCR layer as a witness for a table the pixel check REJECTED.

Hermetic: no tesseract, no ollama, no provider. Pages are tiny PDFs built in memory whose layer
words are inserted by hand; the image reader is an injected function. Every guard is a
DIFFERENCE (the same input with the layer witness off and on, or with one thing changed), never
an outcome measured on one machine (CLAUDE.md, the no-provider trap).

Scope under test: the layer acts ONLY on a scanned, suspect-layer table the pixel check
rejected; the best it can do is WARNING with the table; a table the pixel check accepts is
untouched.
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import fitz
import pytest

from socr.core.result import PageOutput, PageStatus
from socr.tables import layer_witness as lw
from socr.tables import source_evidence as se

TABLE = "| Item | Q1 | Q2 |\n|---|---|---|\n| a | -0.7 | 1.5 |\n| b | 2.5 | 3.0 |\n"
GOOD_LAYER = ["Item", "Q1", "Q2", "-0.7", "1.5", "2.5", "3.0"]
GOOD_OCR = "Item Q1 Q2 a -0.7 1.5 b 2.5 3.0"
LOST_OCR = "Item Q1 Q2 a 07 1.5 b 2.5 3.0"  # tesseract lost the sign and the point


def _page(words: list[str]):
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    for i, w in enumerate(words):
        page.insert_text((50 + 60 * (i % 8), 100 + 30 * (i // 8)), w)
    return page


@pytest.fixture(autouse=True)
def _crop_and_no_chart(monkeypatch):
    """One located crop over the whole page; the chart detectors say 'not a chart'."""
    monkeypatch.setattr(
        "socr.tables.locate.locate_tables",
        lambda page: [SimpleNamespace(bbox=(0.0, 0.0, 600.0, 800.0))],
    )
    monkeypatch.setattr(lw, "page_is_chart", lambda page: False)


def _verify(words, reading, ocr_text, *, witness=True, layer_is_evidence=False, trusted=False):
    return se.verify_scanned_table(
        _page(words),
        reading,
        ocr_image_fn=lambda pix: ocr_text,
        native_trusted=trusted,
        layer_is_evidence=layer_is_evidence,
        layer_witness=witness,
    )


def _key(res):
    return (res.verifiable, res.passed, res.reason, res.deferred, res.disputed_values, res.unproven)


# ---- scope: a table the pixel check accepts is untouched -----------------------------------


@pytest.mark.parametrize(
    "layer",
    [GOOD_LAYER, [*GOOD_LAYER, "9.9"], ["Item", "Q1", "Q2", "-7", "1.5"], ["Item"]],
    ids=["agrees", "layer-has-more", "layer-disagrees", "no-layer-numbers"],
)
def test_a_table_the_pixel_check_accepts_is_identical_with_the_witness_on_or_off(layer):
    on = _verify(layer, TABLE, GOOD_OCR)
    off = _verify(layer, TABLE, GOOD_OCR, witness=False)
    assert off.passed, "premise: the pixel check accepts this table"
    assert _key(on) == _key(off)
    assert not on.unproven


@pytest.mark.parametrize(
    "layer_is_evidence,trusted",
    [(True, False), (True, None), (True, True)],
)
def test_outside_the_suspect_scanned_scope_the_gate_is_byte_identical(layer_is_evidence, trusted):
    for ocr in (GOOD_OCR, LOST_OCR):
        on = _verify(GOOD_LAYER, TABLE, ocr, layer_is_evidence=layer_is_evidence, trusted=trusted)
        off = _verify(
            GOOD_LAYER,
            TABLE,
            ocr,
            witness=False,
            layer_is_evidence=layer_is_evidence,
            trusted=trusted,
        )
        assert _key(on) == _key(off)


# ---- the rescue: rejected by the pixels, agreed by the layer: WARNING, never SUCCESS --------


def test_lost_mark_confirmed_by_the_layer_ships_marked_never_success():
    off = _verify(GOOD_LAYER, TABLE, LOST_OCR, witness=False)
    on = _verify(GOOD_LAYER, TABLE, LOST_OCR)
    assert not off.passed
    assert on.passed and on.unproven, "ships, marked"
    assert "placement" in on.unproven
    assert on.unproven_values == off.disputed_values and on.unproven_values


def test_no_path_ends_in_success_from_the_layer():
    assert not hasattr(lw, "LAYER_CONFIRMS_ALONE")
    assert _verify(GOOD_LAYER, TABLE, LOST_OCR).unproven


# ---- B: marks, both sides, all reading tokens ----------------------------------------------


def test_tesseract_holding_a_sign_the_reading_lacks_vetoes_even_when_the_layer_agrees():
    reading = TABLE.replace("-0.7", "0.7")
    layer = ["Item", "Q1", "Q2", "0.7", "1.5", "2.5", "3.0"]
    ocr = "Item Q1 Q2 a -0.7 1.5 b 2.5 3.0"
    on = _verify(layer, reading, ocr)
    off = _verify(layer, reading, ocr, witness=False)
    assert not off.passed and not on.passed and not on.unproven
    # the contrast: tesseract that merely lost the sign does not veto
    assert _verify(layer, reading, "Item Q1 Q2 a 07 1.5 b 2.5 3.0").passed


def test_layer_holding_other_marks_vetoes_the_reading():
    layer = ["Item", "Q1", "Q2", "-0.7", "1.5", "2.5", "3.0", "0.7"]
    reading = TABLE.replace("-0.7", "0.7")
    assert not _verify(layer, reading, LOST_OCR).passed


def test_layer_occurrence_the_reading_does_not_consume_keeps_the_rejection():
    """The layer side of B, carried by D: the layer's extra -0.7 is not in the reading."""
    layer = ["Item", "Q1", "Q2", "0.7", "-0.7", "1.5", "2.5", "3.0"]
    reading = TABLE.replace("-0.7", "0.7")
    assert not _verify(layer, reading, LOST_OCR).passed
    prose = "Change was -0.7 overall.\n\n" + reading
    assert _verify(layer, prose, LOST_OCR).passed, "a page number consumes its own occurrence"


def test_dash_led_layer_digits_veto_a_reading_token_sharing_them():
    layer = [*GOOD_LAYER, "\u20140.7"]
    witness = SimpleNamespace(
        get_text=lambda kind: [
            (0.0, 0.0, 9.0, 9.0, "\u20140.7", 0, 0, 0),
            (20.0, 0.0, 29.0, 9.0, "-0.7", 0, 0, 1),
        ]
    )
    assert layer and lw.table_layer_witness(witness, boxes=[(0, 0, 100, 100)]).dash_keys == {"7"}
    reading = Counter({"-0.7": 1})
    verdict = lw.judge_against_layer(
        TABLE,
        se.TableTokens(numeric=reading, has_numeric=True),
        lw.TableLayerWitness(
            in_crop=reading, whole=reading, dash_keys=frozenset({"7"}), crops_found=True
        ),
    )
    assert "-0.7" in verdict.b and not verdict.accepted


def test_witness_side_rule_is_directional_for_tesseract_only():
    reading = Counter({"-0.7": 1})
    assert lw.witness_conflicts(reading, Counter({"7": 1}), any_difference=False) == []
    assert lw.witness_conflicts(
        Counter({"0.7": 1}), Counter({"-0.7": 1}), any_difference=False
    ) == ["0.7"]
    assert lw.witness_conflicts(Counter({"0.7": 1}), Counter({"7": 1}), any_difference=True) == [
        "0.7"
    ]
    # an exact match consumes the occurrence
    assert (
        lw.witness_conflicts(
            Counter({"0.7": 1, "-0.7": 1}), Counter({"-0.7": 1}), any_difference=True
        )
        == []
    )


def test_mark_conflict_alone_blocks_the_rescue():
    """One numeric cell per row: the format check is silent, B is the only guard."""
    one = "| Item | Q1 |\n|---|---|\n| a | -7 |\n"
    assert lw.format_outliers(one) == []
    on = _verify(["Item", "Q1", "-0.7"], one, "Item Q1 a 7")
    assert not on.passed and not on.unproven


# ---- C ------------------------------------------------------------------------------------


def test_format_outlier_blocks_a_dropped_point_all_witnesses_share():
    reading = TABLE.replace("-0.7", "-7")
    layer = ["Item", "Q1", "Q2", "-7", "1.5", "2.5", "3.0"]
    on = _verify(layer, reading, "Item Q1 Q2 a 7 1.5 b 2.5 3.0")
    assert not on.passed and not on.unproven


def test_format_outliers_excludes_headers_integer_columns_and_zero():
    md = (
        "| Year | A | B |\n|---|---|---|\n"
        "| 2001 | 1.5 | 2.5 |\n| 2002 | 3.5 | 4 |\n| 2003 | 5.5 | 6.5 |\n"
    )
    assert lw.format_outliers(md) == ["4"]
    zero = "| Item | A | B |\n|---|---|---|\n| x | 1.5 | 0 |\n| y | 2.5 | 3.5 |\n"
    assert lw.format_outliers(zero) == []
    assert lw.format_outliers(zero.replace("| 0 |", "| 100 |")) == ["100"]


# ---- D always runs ------------------------------------------------------------------------


def test_a_layer_number_missing_from_the_reading_keeps_the_rejection():
    on = _verify([*GOOD_LAYER, "9.9"], TABLE, LOST_OCR)
    assert not on.passed and not on.unproven


def test_with_no_crop_d_runs_against_the_whole_page_layer(monkeypatch):
    monkeypatch.setattr("socr.tables.locate.locate_tables", lambda page: [])
    complete = _verify(GOOD_LAYER, TABLE, LOST_OCR)
    dropped_row = _verify([*GOOD_LAYER, "4.4", "5.5"], TABLE, LOST_OCR)
    assert complete.passed and complete.unproven
    assert not dropped_row.passed and not dropped_row.unproven


def test_the_layer_is_scoped_to_the_crop_when_one_exists(monkeypatch):
    # the first row of the page (8 words) is the crop; the reading's numbers sit outside it
    words = ["a", "b", "c", "d", "e", "f", "g", "h", "-0.7", "1.5", "2.5", "3.0"]
    box = SimpleNamespace(bbox=(0.0, 0.0, 600.0, 110.0))
    monkeypatch.setattr("socr.tables.locate.locate_tables", lambda page: [box])
    scoped = _verify(words, TABLE, LOST_OCR)
    monkeypatch.setattr("socr.tables.locate.locate_tables", lambda page: [])
    unscoped = _verify(words, TABLE, LOST_OCR)
    assert not scoped.passed
    assert unscoped.passed and unscoped.unproven


def test_dash_led_layer_words_are_collected_inside_the_crop_only():
    words = [
        (0.0, 0.0, 10.0, 10.0, "\u20147", 0, 0, 0),  # inside the crop
        (0.0, 300.0, 10.0, 310.0, "\u20148", 0, 0, 1),  # outside it
        (20.0, 0.0, 30.0, 10.0, "1.5", 0, 0, 2),
    ]
    page = SimpleNamespace(get_text=lambda kind: words)
    witness = lw.table_layer_witness(page, boxes=[(0.0, 0.0, 100.0, 100.0)])
    assert witness is not None and witness.dash_keys == {"7"}


# ---- pre-gates ----------------------------------------------------------------------------


def test_chart_page_and_truncated_reading_keep_the_rejection(monkeypatch):
    assert _verify(GOOD_LAYER, TABLE, LOST_OCR).passed, "premise: rescued without the gates"
    monkeypatch.setattr(lw, "page_is_chart", lambda page: True)
    chart = _verify(GOOD_LAYER, TABLE, LOST_OCR)
    assert not chart.passed and not chart.unproven
    monkeypatch.setattr(lw, "page_is_chart", lambda page: False)
    cut = TABLE.rstrip("\n").rsplit("|", 1)[0].rstrip()
    truncated = _verify(GOOD_LAYER, cut, LOST_OCR)
    assert not truncated.passed and not truncated.unproven


# ---- HTML reading -------------------------------------------------------------------------


def test_an_html_table_reading_is_converted_before_the_layer_step():
    html = (
        "<table><tr><th>Item</th><th>Q1</th><th>Q2</th></tr>"
        "<tr><td>a</td><td>-0.7</td><td>1.5</td></tr>"
        "<tr><td>b</td><td>2.5</td><td>3.0</td></tr></table>"
    )
    md, _ = se.html_table_for_check(html)
    tokens = se.collect_table_tokens(md)
    bundle = se.SourceEvidenceBundle(numeric=Counter({"1.5": 1}), has_content_evidence=True)
    rejected = se.SourceEvidenceResult(
        verifiable=True, passed=False, reason="r", disputed_values=("-0.7",)
    )

    def run(layer):
        return se._apply_layer_witness(
            _page(layer), html, tokens, bundle, rejected, chart_page=False
        )

    assert run(GOOD_LAYER).unproven, "premise: the raw HTML is read as its table"
    assert not run([*GOOD_LAYER, "9.9"]).passed, "D sees the layer number the HTML reading lacks"


# ---- the judge: wiring, and never "pixel confirmed" -----------------------------------------


class _Inner:
    def __init__(self):
        self._pixel_confirmed_candidates: dict = {}

    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        return AcceptDecision(accept=True, reason="inner", confidence=1.0)


def _judge(words, ocr):
    from socr.pipeline.agentic import SourceEvidenceTableJudge

    inner = _Inner()
    page = _page(words)
    judge = SourceEvidenceTableJudge(
        inner=inner,
        get_fitz_page=lambda n: page,
        ocr_image_fn=lambda pix: ocr,
        native_trusted=lambda n: False,
        layer_is_evidence=lambda n: False,
    )
    out = PageOutput(page_num=1, text=TABLE, status=PageStatus.SUCCESS, engine="qwen")
    return judge, inner, out


def test_judge_wires_the_witness_and_does_not_record_the_candidate_as_pixel_confirmed():
    judge, inner, out = _judge(GOOD_LAYER, LOST_OCR)
    assert judge.assess(out, SimpleNamespace()).accept, (
        "the rescued candidate reaches the inner judge"
    )
    assert out.table_layer_unproven and out.table_layer_unproven_values
    assert id(out) not in inner._pixel_confirmed_candidates
    # the contrast: a candidate the pixels themselves confirmed IS recorded
    judge2, inner2, out2 = _judge(GOOD_LAYER, GOOD_OCR)
    judge2.assess(out2, SimpleNamespace())
    assert id(out2) in inner2._pixel_confirmed_candidates
    assert not out2.table_layer_unproven


# ---- the mark ------------------------------------------------------------------------------


def test_the_mark_says_what_was_and_was_not_checked_and_lists_every_value():
    from socr.core.manifest import _apply_label_unverified_guard
    from socr.core.marks import CheckResult, page_unchecked_clause_mark
    from socr.core.verdict import _table_layer_clause

    values = tuple(f"{i}.{i}" for i in range(1, 12))
    out = PageOutput(page_num=3, text=TABLE, status=PageStatus.SUCCESS, engine="qwen")
    assert _table_layer_clause(out) is None
    assert _apply_label_unverified_guard(out).status is PageStatus.SUCCESS
    out.table_layer_unproven = "x"
    out.table_layer_unproven_values = values
    assert _apply_label_unverified_guard(out).status is PageStatus.WARNING
    clause = _table_layer_clause(out)
    assert clause is not None and clause.result is CheckResult.UNKNOWN
    line = page_unchecked_clause_mark(3, clause.check, clause.result, clause.scope, clause.reason)
    assert "agree with the page's old stored text layer" in line
    assert "not an independent reading" in line
    assert "placement" in line and "not verified" in line
    assert all(v in line for v in values), "every value is named, no silent cut"
    back = PageOutput.from_dict(out.to_dict())
    assert back.table_layer_unproven_values == values
    assert "table_layer_unproven" not in PageOutput(page_num=1, text="x").to_dict()
