"""GH-145: a one-point overlap with a table region deleted the whole text block.

`extract_structured` suppressed any block for which `block_rect.intersects(r)`
was true. `intersects()` fires on a single point of contact, and a table region
routinely ends a few points below the table's last rule — clipping the top line
of the notes paragraph beneath it and deleting the entire paragraph.

On the reference page the notes block was 22.8% inside the region and 77%
outside. It took with it the sample sizes, the sample period, and the sentence
saying the parentheses hold standard errors. The page shipped SUCCESS.

Nothing caught it: born-digital pages get no quality gate, and the deterministic
table verifier compares NUMERIC multisets — every number on that page was present
and correct. The guard that catches a wrong number is blind to a missing
paragraph.

The fix replaces geometry with content. A line is dropped only when the region's
markdown demonstrably already contains its words, because absence of the text in
the replacement is proof that dropping it would lose content. Worst case is now a
duplicated line instead of a deleted one — the correct direction to fail for a
citation corpus.
"""

from __future__ import annotations

import re
from collections import Counter

import fitz
import pytest

from socr.core.born_digital import (
    BornDigitalDetector,
    _line_is_in_region_text,
    _REGION_COVERAGE_DROP,
    _rect_coverage,
    _region_token_index,
    block_is_page_furniture,
    dominant_text_direction,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _line(text, dir_=(1.0, 0.0), bbox=(0, 0, 100, 10)):
    return {"dir": dir_, "bbox": bbox, "spans": [{"text": text}]}


def _block(*lines, bbox=(0, 0, 100, 40)):
    return {"type": 0, "bbox": bbox, "lines": list(lines)}


# ---------------------------------------------------------------------------
# Step 0 — page furniture
# ---------------------------------------------------------------------------


def test_rotated_publisher_stamp_is_furniture():
    """ "Downloaded from https://academic.oup.com/..." rides the margin at 90 degrees."""
    page_blocks = [_block(_line("body text")), _block(_line("more body"))]
    dominant = dominant_text_direction(page_blocks)
    stamp = _block(_line("Downloaded from https://academic.oup.com/qje", dir_=(0.0, 1.0)))

    assert block_is_page_furniture(stamp, dominant) is True


def test_body_text_is_never_furniture():
    blocks = [_block(_line("body text")), _block(_line("more body"))]
    dominant = dominant_text_direction(blocks)

    assert block_is_page_furniture(blocks[0], dominant) is False


def test_dominant_direction_is_derived_not_assumed():
    """A rotated page must not have every one of its lines called furniture.

    An earlier version of this test also asserted that a HORIZONTAL block on such
    a page is furniture. That assertion was wrong and is now inverted: on the
    reference document the sole horizontal block on a landscape table page is the
    running head carrying the page number, and deleting it would lose citable
    content. See ``test_horizontal_text_survives_a_rotated_page``.
    """
    rotated = [
        _block(_line("column one", dir_=(0.0, 1.0))),
        _block(_line("column two", dir_=(0.0, 1.0))),
        _block(_line("counter-rotated stamp", dir_=(0.0, -1.0))),
    ]

    dominant = dominant_text_direction(rotated)

    assert dominant == (0.0, 1.0)
    assert block_is_page_furniture(rotated[0], dominant) is False
    assert block_is_page_furniture(rotated[2], dominant) is True


def test_block_without_lines_is_not_furniture():
    """Absence of evidence must never delete text."""
    assert (
        block_is_page_furniture({"type": 0, "bbox": (0, 0, 1, 1), "lines": []}, (1.0, 0.0)) is False
    )


# ---------------------------------------------------------------------------
# Step 4 — drop on content, not geometry
# ---------------------------------------------------------------------------


def test_line_already_in_the_region_markdown_is_droppable():
    idx = _region_token_index([(None, "| 3M Treasury yield | 0.67 |")])

    assert _line_is_in_region_text(_line("3M Treasury yield 0.67"), idx) is True


def test_line_carrying_one_unrepresented_word_is_kept():
    """One token the replacement lacks is enough — that token would be lost."""
    idx = _region_token_index([(None, "| 3M Treasury yield | 0.67 |")])

    assert (
        _line_is_in_region_text(_line("Notes. Each estimate comes from a regression"), idx) is False
    )


def test_the_notes_line_that_regressed_is_kept():
    """The exact sentence deleted from the reference page."""
    idx = _region_token_index([(None, "| 2Y Treasury yield | 1.10 | 1.06 | 0.04 |")])

    assert (
        _line_is_in_region_text(_line("Robust standard errors are in parentheses."), idx) is False
    )


def test_a_table_title_inside_the_region_bbox_is_kept():
    """Region rectangles span their own titles; the markdown does not contain them."""
    idx = _region_token_index([(None, "| 3M Treasury yield | 0.67 |")])

    assert _line_is_in_region_text(_line("RESPONSE OF INTEREST RATES"), idx) is False


def test_punctuation_only_line_is_droppable():
    idx = _region_token_index([(None, "| a | b |")])

    assert _line_is_in_region_text(_line("———"), idx) is True


def test_numbers_count_as_content():
    """A stray value not in the replacement keeps its line — numbers are the point."""
    idx = _region_token_index([(None, "| row | 0.67 |")])

    assert _line_is_in_region_text(_line("0.99"), idx) is False


# ---------------------------------------------------------------------------
# Geometry helper (still used to decide whether the content test applies)
# ---------------------------------------------------------------------------


def test_partial_coverage_is_measured_not_boolean():
    """The old code asked 'do these touch'; the failure needed 'how much'."""
    block = fitz.Rect(0, 361, 100, 416)  # the notes paragraph
    region = fitz.Rect(0, 129, 100, 374)  # the table region, ending 13pt too low

    cov = _rect_coverage(block, region)

    assert 0.0 < cov < 0.5, f"expected a small partial overlap, got {cov:.2%}"


def test_zero_area_block_does_not_divide_by_zero():
    assert _rect_coverage(fitz.Rect(5, 5, 5, 5), fitz.Rect(0, 0, 10, 10)) == 0.0


# ---------------------------------------------------------------------------
# End to end on a synthesised page
# ---------------------------------------------------------------------------


@pytest.fixture
def table_page(tmp_path):
    """A whitespace-gutter table with notes tight beneath it — the regression shape.

    Deliberately UNRULED. A first version of this fixture drew ruling lines, so
    ``find_tables()`` succeeded and returned a tight bbox that never reached the
    notes — and the test passed against the pre-fix code, guarding nothing. The
    real failure needs the word-geometry rowizer, which is only reached when both
    earlier strategies return nothing, and whose region overruns the last data
    row. Verified to fail before the fix and pass after.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 70), "TABLE I", fontsize=10)
    page.insert_text((60, 86), "Nominal      Real      Inflation", fontsize=9)
    y = 104
    rows = [
        ("3M Treasury yield", "0.67", "0.61", "0.06"),
        ("6M Treasury yield", "0.85", "0.80", "0.05"),
        ("1Y Treasury yield", "1.00", "0.94", "0.06"),
        ("2Y Treasury yield", "1.10", "1.06", "0.04"),
        ("3Y Treasury yield", "1.06", "1.02", "0.04"),
        ("5Y Treasury yield", "0.73", "0.64", "0.09"),
    ]
    for lab, a, b, c in rows:
        page.insert_text((60, y), f"{lab}      {a}      {b}      {c}", fontsize=9)
        y += 16
        page.insert_text((60, y), f"      ({a})      ({b})      ({c})", fontsize=9)
        y += 16
    page.insert_text((60, y + 4), "Notes. Robust standard errors in parentheses.", fontsize=7)
    page.insert_text((60, y + 16), "The sample size is 106 observations.", fontsize=7)
    path = tmp_path / "t.pdf"
    doc.save(str(path))
    doc.close()
    return path


def test_notes_beneath_a_table_survive_extraction(table_page):
    with fitz.open(table_page) as doc:
        out = BornDigitalDetector().extract_structured(doc[0])

    assert "Robust standard errors" in out
    assert "106 observations" in out


@pytest.mark.xfail(
    strict=True,
    reason=(
        "GH-144: the word-geometry rowizer splits values while building the grid — "
        "'3M Treasury yield 0.67' becomes '| 3M | Treasury | yield 0 | .67 |', "
        "destroying 0.67. A separate and worse defect than GH-145: this loses "
        "NUMBERS, and no amount of composition repair can restore a value the "
        "reconstruction never emitted. This fixture is the minimal reproduction; "
        "flip to passing when GH-144 is fixed."
    ),
)
def test_no_table_value_is_lost(table_page):
    with fitz.open(table_page) as doc:
        page = doc[0]
        raw = page.get_text("text")
        out = BornDigitalDetector().extract_structured(page)

    nums = lambda s: Counter(re.findall(r"\d+\.\d+", s))
    assert not (nums(raw) - nums(out)), "extraction lost a numeric value"


def test_horizontal_text_survives_a_rotated_page():
    """A landscape table makes the page's dominant direction rotated.

    Without the horizontal exemption, the one horizontal block on such a page is
    deleted as "furniture" — on the reference document that block is the running
    head carrying the page number (1324), which is citable content. Caught by
    inspecting a rotated page before this shipped.
    """
    rotated_body = [_block(_line(f"col {i}", dir_=(0.0, -1.0))) for i in range(8)]
    running_head = _block(_line("1324 THE QUARTERLY JOURNAL OF ECONOMICS", dir_=(1.0, 0.0)))
    stamp = _block(_line("Downloaded from https://academic.oup.com/qje", dir_=(0.0, 1.0)))
    dominant = dominant_text_direction(rotated_body + [running_head, stamp])

    assert dominant == (0.0, -1.0)
    assert block_is_page_furniture(running_head, dominant) is False
    assert block_is_page_furniture(stamp, dominant) is True


# ---------------------------------------------------------------------------
# Review corrections (#145 PR review)
# ---------------------------------------------------------------------------


def test_counter_directional_text_is_relegated_not_deleted(tmp_path):
    """Rotated text keeps its content; only its POSITION is corrected.

    Deleting it would also discard legitimately rotated content — a chart's axis
    labels, a rotated table header, a marginal note. The problem is reading
    order, and relegation solves that completely without losing a word.
    """
    doc = fitz.open()
    page = doc.new_page()
    for i in range(6):
        page.insert_text((60, 90 + i * 16), "body prose line that runs across", fontsize=9)
    page.insert_text((560, 300), "Downloaded from https://example.com/x", fontsize=7, rotate=90)
    path = tmp_path / "stamp.pdf"
    doc.save(str(path))
    doc.close()

    with fitz.open(path) as d:
        out = BornDigitalDetector().extract_structured(d[0])

    assert "Downloaded from" in out, "counter-directional text must not be deleted"
    body = "\n".join(out.strip().split("\n")[:-1])
    assert "Downloaded from" not in body, "it must not interrupt the prose either"


def test_tokens_are_matched_against_the_covering_region_only():
    """A line must not be deleted because ANOTHER table happens to contain its words.

    Two regions on one page: a line overlapping region A whose words appear only
    in region B is not represented by A's replacement, and dropping it loses text.
    """
    region_a = [(None, "| alpha | 1.00 |")]
    region_b = [(None, "| beta | 2.00 |")]

    assert _line_is_in_region_text(_line("beta 2.00"), _region_token_index(region_a)) is False
    assert _line_is_in_region_text(_line("beta 2.00"), _region_token_index(region_b)) is True
    combined = _region_token_index(region_a + region_b)
    assert _line_is_in_region_text(_line("beta 2.00"), combined) is True, (
        "a combined index is exactly the bug: region A would delete region B's line"
    )


def test_a_block_barely_touching_a_region_keeps_every_line():
    """Below the coverage threshold the region does not represent the block at all.

    Without this, a caption reading '0.67' beneath a table containing 0.67 is
    deleted on a one-point overlap — the original defect in a new costume.
    """
    block = fitz.Rect(0, 361, 100, 416)
    region = fitz.Rect(0, 129, 100, 374)

    assert _rect_coverage(block, region) < _REGION_COVERAGE_DROP
