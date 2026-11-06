"""#1218 follow-up: the review items on merged PR #1277 (cubic), each pinned as a difference."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from socr.core.verdict import page_dropped_numbers
from socr.tables.layer_body import classify_layer_numbers
from socr.tables.reading_agreement import dropped_numbers

from test_gh1218_drops_clause import (
    MISSING_ROW,
    _doc_dir,
    _out,
    _p,
    _pdf_dir,
    _rotated_pdf,
    _run,
    _without_the_clause,
)

RECT = (0.0, 0.0, 612.0, 792.0)


# --- 1. a #1218 pass merges into the entry an earlier pass wrote ---------------------------


def test_refinalize_merges_into_an_entry_an_earlier_pass_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from socr.pipeline.refinalize_own_text import LOG_NAME, refinalize_document

    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    old = tmp_path / "old"
    with monkeypatch.context() as m:
        _without_the_clause(m)
        _run(old, pdf, monkeypatch, drop=True, pages_read=[])
    doc = _doc_dir(old)
    earlier = {
        "before": {"status": "success"},
        "after": {"status": "warning", "text_layer_numbers": "FAIL"},
        "socr_source_digest": "an earlier pass",
    }
    (doc / LOG_NAME).write_text(
        json.dumps({"tool": "socr-refinalize-own-text", "pages": {"1": earlier}})
    )
    changes = {c.page_num: c for c in refinalize_document(doc, _pdf_dir(tmp_path, pdf))}
    assert set(changes) == {1, 2}
    pages = json.loads((doc / LOG_NAME).read_text())["pages"]
    # page 1 had an entry: its original `before` and numbers field survive, the drops are added
    assert pages["1"]["before"] == earlier["before"]
    assert pages["1"]["after"]["text_layer_numbers"] == "FAIL"
    assert pages["1"]["after"]["text_layer_drops"] == "FAIL"
    assert pages["1"]["after"]["status"] == changes[1].after
    # page 2 had none: written as a first entry is
    assert pages["2"]["before"] == {"status": changes[2].before}
    assert pages["2"]["after"]["text_layer_drops"] == "FAIL"


# --- 2. the reading direction of a rotated page that gives no votes --------------------------


def _single_digit_lines(rotation: int) -> list[str]:
    """Four single-digit words per line (square boxes: no multi-character evidence, so only
    /Rotate says the page is sideways), laid out the way /Rotate ``rotation`` displays them
    left to right: 90 reads toward decreasing y, 270 toward increasing y."""
    step = -12.0 if rotation == 90 else 12.0
    words = []
    for line, x in enumerate((100.0, 130.0)):
        y0 = 700.0 if rotation == 90 else 100.0
        for i in range(4):
            y = y0 + i * step
            words.append((x, y, x + 10.0, y + 10.0, str(line * 4 + i + 1)))
    entries = classify_layer_numbers(words, RECT, None, rotation=rotation)
    return sorted({e.line_text for e in entries})


@pytest.mark.parametrize("rotation", [90, 270])
def test_a_rotated_page_with_no_votes_groups_its_words_into_rows_in_reading_order(
    rotation: int,
) -> None:
    # the two rows, each read in the direction the page's /Rotate displays
    assert _single_digit_lines(rotation) == ["1 2 3 4", "5 6 7 8"]


# --- 3. a word character before the leading digits is not a thousands group ------------------


def test_a_digit_that_ends_a_token_does_not_start_a_space_thousands_group() -> None:
    assert dropped_numbers("1,234", "Q1 234").dropped == ("1,234",)
    # difference: the same digits after a space are the writing 1 234 of 1,234
    assert dropped_numbers("1,234", "Total 1 234").dropped == ()
    assert dropped_numbers("1,234", "1 234").dropped == ()


# --- 4. a consensus winner is a model reading ------------------------------------------------


def test_a_consensus_winner_is_checked_for_dropped_numbers_like_its_engine() -> None:
    p = _p()
    owed = page_dropped_numbers(_out(MISSING_ROW, "qwen"), p)
    assert owed, "fixture premise: the single engine drops the second row"
    assert page_dropped_numbers(_out(MISSING_ROW, "consensus(qwen)"), p) == owed
    # difference: an engine that is no model reading is not checked
    assert page_dropped_numbers(_out(MISSING_ROW, "native"), p) == ()


# --- 5. the decimal-fragment chain filter follows the pairs it keeps ----------------------


def _tail(items):
    words = [(x0, 100.0, x1, 108.5, t) for t, x0, x1 in items]
    return [e.written for e in classify_layer_numbers(words, RECT, None) if e.row is not None]


def test_a_chain_of_close_integers_keeps_every_pair_that_does_not_overlap_a_kept_one() -> None:
    chain = [
        ("Mean", 37.6, 58.6),
        ("1", 100.0, 104.0),
        ("23", 105.0, 115.0),
        ("45", 116.0, 126.0),
        ("67", 127.0, 137.0),
        ("12.5", 300.0, 320.0),
    ]
    # words 1-2 and 3-4 are two non-overlapping pairs; the pair 2-3 shares words with both
    assert _tail(chain) == ["1.23", "45.67", "12.5"]
    # difference: with the closeness removed nothing is joined
    spread = [(t, x0 + 40.0 * i, x1 + 40.0 * i) for i, (t, x0, x1) in enumerate(chain)]
    assert _tail(spread) == ["1", "23", "45", "67", "12.5"]


# --- 6. a chart box is grown once ------------------------------------------------------------


def _step_row():
    return [
        (50.0 + 60.0 * i, 100.0, 90.0 + 60.0 * i, 110.0, t)
        for i, t in enumerate(["Percent", "0.75", "0.50", "0.25", "0.00"])
    ]


def _step_tail(**boxes):
    return [
        e.written
        for e in classify_layer_numbers(_step_row(), RECT, None, **boxes)
        if e.row is not None
    ]


def test_a_chart_box_is_not_grown_again_but_a_raw_image_box_is() -> None:
    box = [(40.0, 20.0, 400.0, 95.0)]  # ends 5 pt above the row: closer than one text height
    # a chart box arrives already grown (chart_region_bboxes): the row is NOT at it
    assert len(_step_tail(excluded_boxes=box)) == 4
    # an image box is raw: its tick labels sit just outside it, so the same box DOES claim the row
    assert _step_tail(image_boxes=box) == []
    # difference: a box touching the row claims it either way
    touching = [(40.0, 20.0, 400.0, 101.0)]
    assert _step_tail(excluded_boxes=touching) == []
