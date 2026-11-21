"""#1399: the stored old-OCR layer as the witness of a model-read table on a scanned page.

Hermetic: no tesseract, no ollama, no provider. Pages are tiny PDFs built in memory whose layer
words are inserted by hand; the image reader is an injected function. Every guard is pinned as
a DIFFERENCE (the same input with the layer witness off and on, or with one thing changed), never
as an outcome measured on one machine (CLAUDE.md, the no-provider trap).
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import fitz
import pytest

from socr.core.result import PageOutput, PageStatus
from socr.tables import layer_witness as lw
from socr.tables import source_evidence as se
from socr.tables.source_evidence import collect_table_tokens

TABLE = "| Item | Q1 | Q2 |\n|---|---|---|\n| a | -0.7 | 1.5 |\n| b | 2.5 | 3.0 |\n"


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


def _verify(words, reading, ocr_text, *, witness=True):
    return se.verify_scanned_table(
        _page(words),
        reading,
        ocr_image_fn=lambda pix: ocr_text,
        native_trusted=False,
        layer_is_evidence=False,
        layer_witness=witness,
    )


GOOD_LAYER = ["Item", "Q1", "Q2", "-0.7", "1.5", "2.5", "3.0"]
GOOD_OCR = "Item Q1 Q2 a -0.7 1.5 b 2.5 3.0"


def test_agreeing_layer_and_image_check_is_success_and_identical_to_witness_off():
    on = _verify(GOOD_LAYER, TABLE, GOOD_OCR)
    off = _verify(GOOD_LAYER, TABLE, GOOD_OCR, witness=False)
    assert on.passed and not on.unproven
    assert (on.passed, on.unproven) == (off.passed, off.unproven)


def test_lost_point_confirmed_by_the_layer_ships_marked_not_rejected():
    # tesseract lost the point and the sign (the audit's dominant cause): off rejects, on ships marked
    ocr = "Item Q1 Q2 a 07 1.5 b 2.5 3.0"
    off = _verify(GOOD_LAYER, TABLE, ocr, witness=False)
    on = _verify(GOOD_LAYER, TABLE, ocr)
    assert not off.passed
    assert on.passed and on.unproven
    assert "image check" in on.unproven


def test_layer_alone_confirms_only_under_the_policy_switch(monkeypatch):
    ocr = "Item Q1 Q2 a 07 1.5 b 2.5 3.0"
    before = _verify(GOOD_LAYER, TABLE, ocr)
    monkeypatch.setattr(lw, "LAYER_CONFIRMS_ALONE", True)
    after = _verify(GOOD_LAYER, TABLE, ocr)
    assert before.unproven and not after.unproven and after.passed


def test_dropped_point_repeated_by_the_image_check_is_disputed_by_the_layer():
    reading = TABLE.replace("-0.7", "-7")
    ocr = "Item Q1 Q2 a -7 1.5 b 2.5 3.0"  # tesseract repeats the model's error
    off = _verify(GOOD_LAYER, reading, ocr, witness=False)
    on = _verify(GOOD_LAYER, reading, ocr)
    assert off.passed, "premise: the image check alone confirms the wrong number"
    assert not on.passed and "-7" in on.disputed_values


def test_layer_that_repeats_the_dropped_point_is_caught_by_the_format_check():
    reading = TABLE.replace("-0.7", "-7")
    layer = ["Item", "Q1", "Q2", "-7", "1.5", "2.5", "3.0"]
    on = _verify(layer, reading, "Item Q1 Q2 a -7 1.5 b 2.5 3.0")
    assert not on.passed and "-7" in on.disputed_values
    # the same cell where the row is not decimal: nothing to depart from
    plain = TABLE.replace("| -0.7 | 1.5 |", "| -7 | 15 |").replace("| 2.5 | 3.0 |", "| 25 | 30 |")
    assert lw.format_outliers(plain) == []


def test_layer_number_missing_from_the_reading_ships_marked():
    layer = [*GOOD_LAYER, "9.9"]
    on = _verify(layer, TABLE, GOOD_OCR)
    off = _verify(layer, TABLE, GOOD_OCR, witness=False)
    assert off.passed and not off.unproven
    assert on.passed and on.unproven and on.unproven_values == ("9.9",)


def test_reading_number_absent_from_the_layer_is_not_success():
    layer = ["Item", "Q1", "Q2", "-0.7", "1.5", "2.5"]  # 3.0 missing from the layer
    on = _verify(layer, TABLE, GOOD_OCR)
    assert on.passed and on.unproven
    assert "3.0" in on.unproven


def test_no_crop_located_caps_at_marked(monkeypatch):
    monkeypatch.setattr("socr.tables.locate.locate_tables", lambda page: [])
    on = _verify(GOOD_LAYER, TABLE, GOOD_OCR)
    assert on.passed and on.unproven and "crop" in on.unproven


def test_chart_page_and_truncated_reading_are_refused(monkeypatch):
    monkeypatch.setattr(lw, "page_is_chart", lambda page: True)
    chart = _verify(GOOD_LAYER, TABLE, GOOD_OCR)
    assert not chart.passed and chart.reason == lw.REASON_CHART
    monkeypatch.setattr(lw, "page_is_chart", lambda page: False)
    cut = TABLE.rstrip("\n").rsplit("|", 1)[0].rstrip()  # last row loses its closing pipe
    truncated = _verify(GOOD_LAYER, cut, GOOD_OCR)
    assert not truncated.passed and truncated.reason == lw.REASON_TRUNCATED


def test_page_without_layer_numbers_or_reading_is_untouched():
    nothing = ["Item", "Q1", "Q2"]
    on = _verify(nothing, TABLE, "Item Q1 Q2 a 07 1.5 b 2.5 3.0")
    off = _verify(nothing, TABLE, "Item Q1 Q2 a 07 1.5 b 2.5 3.0", witness=False)
    assert (on.passed, on.reason, on.unproven) == (off.passed, off.reason, off.unproven)
    blank = _verify(GOOD_LAYER, TABLE, "")
    blank_off = _verify(GOOD_LAYER, TABLE, "", witness=False)
    assert (blank.passed, blank.reason) == (blank_off.passed, blank_off.reason)


def test_layer_stays_excluded_when_the_caller_did_not_ask_for_the_witness():
    ocr = "Item Q1 Q2 a 07 1.5 b 2.5 3.0"
    default = se.verify_scanned_table(
        _page(GOOD_LAYER),
        TABLE,
        ocr_image_fn=lambda pix: ocr,
        native_trusted=False,
        layer_is_evidence=False,
    )
    assert not default.passed and not default.unproven


def test_swapped_cells_pass_the_multiset_by_design():
    swapped = TABLE.replace("| -0.7 | 1.5 |", "| 1.5 | -0.7 |")
    on = _verify(GOOD_LAYER, swapped, GOOD_OCR)
    assert on.passed and not on.unproven  # position-blind; strict row binding is the net


# ---- pure parts -------------------------------------------------------------------------


def test_mark_conflict_needs_the_same_digits_with_other_marks():
    left = Counter({"-7": 1, "4.4": 1})
    layer = Counter({"-0.7": 1})
    assert lw.mark_conflicts(left, layer, frozenset(), frozenset()) == ["-7"]
    assert lw.mark_conflicts(left, layer, frozenset(), frozenset({"44"})) == ["-7", "4.4"]
    assert lw.mark_conflicts(Counter({"4.4": 1}), layer, frozenset({"44"}), frozenset()) == ["4.4"]


def test_format_outliers_excludes_header_rows_and_integer_columns():
    md = (
        "| Year | A | B |\n|---|---|---|\n"
        "| 2001 | 1.5 | 2.5 |\n| 2002 | 3.5 | 4 |\n| 2003 | 5.5 | 6.5 |\n"
    )
    assert lw.format_outliers(md) == ["4"]  # the year column is all integers: not an outlier


def test_unproven_survives_the_sidecar_and_demotes_the_page():
    from socr.core.manifest import _apply_label_unverified_guard
    from socr.core.verdict import _table_layer_clause

    out = PageOutput(page_num=1, text=TABLE, status=PageStatus.SUCCESS, engine="qwen")
    assert _apply_label_unverified_guard(out).status is PageStatus.SUCCESS
    out.table_layer_unproven = "the layer lacks 3.0"
    out.table_layer_unproven_values = ("3.0",)
    assert _apply_label_unverified_guard(out).status is PageStatus.WARNING
    assert _table_layer_clause(out) is not None
    back = PageOutput.from_dict(out.to_dict())
    assert back.table_layer_unproven == out.table_layer_unproven
    assert back.table_layer_unproven_values == ("3.0",)
    clean = PageOutput(page_num=1, text=TABLE, status=PageStatus.SUCCESS, engine="qwen")
    assert "table_layer_unproven" not in clean.to_dict()


def test_collect_tokens_premise():
    assert collect_table_tokens(TABLE).numeric["-0.7"] == 1


def test_mark_conflict_alone_blocks_the_image_check_from_rescuing():
    """One numeric cell per row: the format check has nothing to say, B is the only guard."""
    one = "| Item | Q1 |\n|---|---|\n| a | -7 |\n"
    assert lw.format_outliers(one) == []
    layer = ["Item", "Q1", "-0.7"]
    ocr = "Item Q1 a -7"
    off = _verify(layer, one, ocr, witness=False)
    on = _verify(layer, one, ocr)
    assert off.passed, "premise: the image check repeats the error and confirms it"
    assert not on.passed and on.disputed_values == ("-7",)
