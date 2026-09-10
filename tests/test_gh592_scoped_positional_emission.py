"""GH-592 round 3 (Astra review of PR #704): positional emission must be SCOPED.

Round 2 replaced the aligned-run assembler's block-order emission with a
page-wide row-major (visual row, then x) emission. That fixed the displaced
declined rows on the real Fed pages, but it reordered *every* unconsumed line
on the page: a page carrying a genuine attendee roster AND, elsewhere, an
unrelated two-column prose block had that prose interleaved line by line
(LEFT 1, RIGHT 1, LEFT 2, ...) even though the four geometric guards in
``_try_aligned_run`` correctly refused to merge it. Tokens all survive; the
reading order does not.

These tests pin the scoping invariant: finding a run anywhere on a page never
reorders content that is not entangled with that run.

Derived from the reviewer's reproducers (``test_run_does_not_interleave_...``
and ``test_real_extractor_preserves_unrelated_column_order``) plus the three
controls shipped alongside them.
"""

from __future__ import annotations

import ast
import subprocess

import fitz
import pytest
from test_born_digital_aligned_runs import _build_attendee_list_page

from socr.core import born_digital as bd

#: The unrelated two-column prose placed below the roster. Two independent
#: paragraphs at x=72 and x=330 -- a genuine column gutter, an order of
#: magnitude wider than ``ALIGNED_RUN_GAP_MAX_WORD_SPACES`` allows, so the
#: guards decline it and every line stays unconsumed.
_PROSE_COLUMNS = ((72, "LEFT"), (330, "RIGHT"))

#: Reading order the caller's own ``page.get_text("text")`` produces for those
#: two paragraphs: each column whole, in block order. Pinned rather than
#: recomputed from ``origin/main`` so the assertion is hermetic in CI, where a
#: shallow checkout has no ``origin/main`` ref.
_EXPECTED_PROSE_ORDER = [
    "LEFT paragraph line 1",
    "LEFT paragraph line 2",
    "LEFT paragraph line 3",
    "RIGHT paragraph line 1",
    "RIGHT paragraph line 2",
    "RIGHT paragraph line 3",
]


def _add_prose_columns(page: fitz.Page) -> None:
    for x, prefix in _PROSE_COLUMNS:
        page.insert_textbox(
            fitz.Rect(x, 400, x + 190, 500),
            "\n".join(f"{prefix} paragraph line {i}" for i in range(1, 4)),
            fontsize=10,
        )


def _prose_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.startswith(("LEFT", "RIGHT"))]


def _previous_assembler():
    """The pre-#704 (``origin/main``) assembler, for old-vs-new comparisons.

    Loads the three functions the branch changed out of ``origin/main`` and
    executes them against this module's live namespace, so the PDF, the
    measurement helpers and the detector are shared and only the
    implementation under test differs. ``_ALIGNED_RUN_MIN_ROWS`` is restored to
    the previous value of two, which the old guards assumed.
    """
    source = subprocess.check_output(
        ["git", "show", "origin/main:src/socr/core/born_digital.py"], text=True
    )
    tree = ast.parse(source)
    names = {"_assemble_prose_with_aligned_runs", "_find_aligned_runs", "_try_aligned_run"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    env = dict(vars(bd))
    env["_ALIGNED_RUN_MIN_ROWS"] = 2
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "previous_born_digital.py", "exec"), env)
    return env["_assemble_prose_with_aligned_runs"]


def _origin_main_available() -> bool:
    try:
        subprocess.check_output(
            ["git", "rev-parse", "--verify", "origin/main"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return False
    return True


def test_run_does_not_interleave_unrelated_two_column_prose():
    """A run in the roster must not reorder an unrelated column pair below it."""
    page = _build_attendee_list_page()
    _add_prose_columns(page)

    out = bd._assemble_prose_with_aligned_runs(page)

    assert out is not None, "the roster is a genuine run; the assembler must engage"
    assert _prose_lines(out) == _EXPECTED_PROSE_ORDER, (
        "the declined prose columns must keep their own block reading order; "
        f"got {_prose_lines(out)}"
    )


def test_real_extractor_preserves_unrelated_column_order():
    """Same invariant through the production caller, not just the assembler."""
    page = _build_attendee_list_page()
    _add_prose_columns(page)

    out = bd.BornDigitalDetector().extract_structured(page)

    assert _prose_lines(out) == _EXPECTED_PROSE_ORDER, (
        f"production extraction interleaved the unrelated columns: {_prose_lines(out)}"
    )


def test_unconsumed_lines_keep_the_assembler_off_ordering():
    """Every line no run consumed appears in the same relative order as when
    the assembler is forced off entirely.

    The general form of the finding: with no block shared between a run and an
    unconsumed line (true of this page -- the roster's two columns are their
    own text boxes), scoped emission has nothing to reposition, so the
    non-roster text must be exactly the caller's own order.
    """
    page = _build_attendee_list_page()
    _add_prose_columns(page)

    with_assembler = bd.BornDigitalDetector().extract_structured(page)
    baseline = page.get_text("text")

    roster_names = ("Greenspan", "Corrigan", "Angell", "Black", "Seger")

    def unconsumed(text: str) -> list[str]:
        # Drop both halves of every roster row: the name lines, and the bare
        # honorific label lines the run merges them with.
        return [
            line.strip()
            for line in text.splitlines()
            if line.strip()
            and line.strip() not in ("Mr.", "Ms.")
            and not any(name in line for name in roster_names)
        ]

    assert unconsumed(with_assembler) == unconsumed(baseline)


def test_no_run_page_is_left_untouched():
    """A page with no aligned run returns ``None`` -- the caller's path is kept."""
    doc = fitz.open()
    page = doc.new_page()
    _add_prose_columns(page)
    page.insert_text((72, 200), "A sentence with a reference", fontsize=10)
    page.insert_text((220, 196), "1", fontsize=6)

    assert bd._assemble_prose_with_aligned_runs(page) is None


def test_orphan_is_emitted_once_in_position():
    """A value with no label is emitted exactly once, below the paired names."""
    page = _build_attendee_list_page()
    page.insert_text((120, 320), "ORPHAN VALUE", fontsize=10)

    out = bd._assemble_prose_with_aligned_runs(page)

    assert out is not None
    assert out.count("ORPHAN VALUE") == 1
    assert out.index("ORPHAN VALUE") > out.index("Mr. Black")


@pytest.mark.skipif(
    not _origin_main_available(), reason="origin/main ref not present (shallow checkout)"
)
def test_two_row_decline_is_an_actual_behavior_change():
    """The ``_ALIGNED_RUN_MIN_ROWS`` floor of three declines a real two-row pair.

    The documented conservative trade-off from round 2, pinned as a DIFFERENCE
    between the two implementations rather than as an absolute: the previous
    assembler merged this pair, this one declines and keeps the unmerged
    extraction. No tokens are lost either way.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Some ordinary running prose establishes the word space measurement here.",
        fontsize=10,
    )
    right = (
        90 + fitz.get_text_length("Mr.", fontsize=10) + 1.2 * fitz.get_text_length(" ", fontsize=10)
    )
    page.insert_textbox(fitz.Rect(90, 104, 130, 244), "Mr.\nMr.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, 104, 560, 244),
        "Angell\nCorrigan, Vice Chairman of Committee",
        fontsize=10,
    )

    assert _previous_assembler()(page) is not None
    assert bd._assemble_prose_with_aligned_runs(page) is None
