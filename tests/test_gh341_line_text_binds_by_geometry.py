"""GH-341 hole 1: the span path must bind a link by geometry, not by content.

`_line_text` resolved an anchor with `text.index(anchor)` -- the FIRST
occurrence in the span, whatever the link rectangle actually covered. A
uniform-font line is a single span, so a page citing "Smith 2020" in the body
and again in the references stamped the bibliography's DOI on the in-text
mention.

A wrong URI on the wrong citation is worse than a dropped one: a missing link is
visible, a misattributed one is not. So an anchor that matches no occurrence
under the rectangle is SKIPPED rather than guessed.

Holes 2 and 3 of this ticket (the `text.find` fallback and partial resolves)
landed in #417; this is the remaining one.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.born_digital import _line_text  # noqa: E402

LINE = "See Smith 2020 here; cite Smith 2020 there."
SPAN = {"text": LINE, "bbox": (72.0, 90.0, 400.0, 102.0)}
URI = "https://doi.org/10.1000/x"
ANCHOR = "Smith 2020"


def _wrapped_offset(out: str) -> int:
    """Character offset of the wrapped occurrence in the emitted line."""
    assert "[" in out, f"nothing was wrapped: {out}"
    return out.index("[")


def test_the_link_wraps_the_occurrence_its_rectangle_covers() -> None:
    """The defect: a rectangle over the SECOND mention wrapped the first."""
    rect = fitz.Rect(240.0, 90.0, 300.0, 102.0)  # right half of the span
    out = _line_text([SPAN], [(rect, URI, ANCHOR)])

    assert out.count("](") == 1, f"expected exactly one link: {out}"
    assert _wrapped_offset(out) > out.index("cite"), (
        f"the link wrapped the FIRST mention though its rectangle covers the second: {out}"
    )


def test_a_rectangle_over_the_first_mention_still_wraps_the_first() -> None:
    """Control: the fix must not simply invert the choice.

    Without this, always taking the LAST occurrence would satisfy the test
    above while being just as wrong.
    """
    rect = fitz.Rect(90.0, 90.0, 150.0, 102.0)  # left part of the span
    out = _line_text([SPAN], [(rect, URI, ANCHOR)])

    assert out.count("](") == 1, f"expected exactly one link: {out}"
    assert _wrapped_offset(out) < out.index("cite"), (
        f"the link wrapped the SECOND mention though its rectangle covers the first: {out}"
    )


def test_a_rectangle_over_neither_mention_wraps_nothing() -> None:
    """Prefer skipping to guessing.

    GH-465: the first version of this test put the rectangle PAST the span
    (x 500-560 against a span ending at 400). `_line_text` drops that at
    `span_rect.intersects(rect)` and never reaches the offset resolver, so the
    test re-pinned the pre-existing intersects filter and passed with the skip
    path deleted -- vacuous.

    The rectangle now sits INSIDE the span, over the gap BETWEEN the two
    mentions, which is the path the skip actually guards: the link intersects
    the span, the anchor is present, and no occurrence lies under the rect.
    """
    rect = fitz.Rect(180.0, 90.0, 230.0, 102.0)  # inside the span, between the two mentions
    assert rect.intersects(fitz.Rect(SPAN["bbox"])), (
        "the rectangle must intersect the span, or this pins the intersects "
        "filter rather than the skip path"
    )
    out = _line_text([SPAN], [(rect, URI, ANCHOR)])

    assert out == LINE, f"a link with no covered occurrence was still stamped: {out}"


def test_a_line_with_no_links_is_byte_identical() -> None:
    """The golden-fragment contract this function has always carried."""
    assert _line_text([SPAN], []) == LINE


# GH-465 hole 2 asked for an `extract_structured` pin on a table/span page.
# I could not build one, and the reason is worth recording rather than replacing
# with a fixture that passes for the wrong reason.
#
# The page must DETECT a table for the prose line to reach `_line_text`. A
# two-column table is not detected at all, so the page silently falls back to
# the flat path -- which #417 already fixes, and the pins then passed under BOTH
# a `text.index` and a `text.rindex` revert, i.e. measured nothing.
#
# Giving it four numeric lanes does detect a table, but then the prose line is
# swallowed INTO the grid:
#
#     '| See Smith 2020 here; | cite Smith 2020 there. |  |  |'
#
# and the link is gone before binding is reached. Moving the prose 460pt from
# the table does not change it. That is the prose-gridding defect (#150/#213
# family), not this ticket's, and it makes the fixture unbuildable today.
#
# The binding itself is pinned above at `_line_text`, in both directions and on
# the skip path. Measured on #465.
