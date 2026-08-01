"""#123 TICKET-A1: a standing metamorphic corruption battery over ``score_page``.

Seven defects in ``table_exactness`` were each found by accident, after they had
already produced published numbers (``docs/plans/metric-blind-spots/TICKETS.md``).
All seven share one shape: a known perturbation of the input moved the score the
wrong way, or failed to move it at all. Nothing tested the metric itself - it was
the doer grading its own work, one level up.

This module asserts two metamorphic properties over a fitz-generated fixture pair
(native PDF + emitted markdown), rather than pinning specific percentages:

- **benign transforms leave the score unchanged** - the perturbation changes
  presentation, not content, so a correct engine should be scored identically
  before and after.
- **corrupting transforms make the score strictly worse** - the perturbation
  changes content, so the score MUST move down, not stay flat.

Every fixture clears the grid predicate in ``native_rows.rows_establish_grid``
(#123 TICKET-B1: at least two value columns, at least two rows sharing that
width). A fixture that does not is silently not-scorable - ``score_page`` returns
``pct=None`` - and every "strictly worse" assertion would then compare against
``None`` rather than failing loudly. ``test_base_fixture_is_scorable`` pins this
before anything else runs.
"""

from __future__ import annotations

import fitz
import pytest

from socr.benchmark.table_exactness import score_page

# ----------------------------------------------------------------------
# The fixture pair: a born-digital page + its ground-truth row shape.
#
# 5 rows, 2 value columns. "Other adjustments" is deliberately sparse (one
# value, one gap) - the empty cell the shift-into-empty-cell corruption needs.
# Grid predicate: modal width 2, satisfied by 4 of 5 rows - clears
# ``rows_establish_grid`` with margin to spare.
# ----------------------------------------------------------------------

_ROWS: list[tuple[str, tuple[str, ...]]] = [
    ("Total spending", ("42.8", "30.5")),
    ("Departmental spending", ("24.8", "18.2")),
    ("Welfare spending", ("18.4", "12.0")),
    ("Debt interest", ("3.8", "2.6")),
    ("Other adjustments", ("0.5",)),
]
_VALUE_XS = (300.0, 360.0)
_WIDTH = 2


def _build_pdf(path, rows=_ROWS, value_xs=_VALUE_XS):
    doc = fitz.open()
    page = doc.new_page()
    y = 200.0
    for label, values in rows:
        page.insert_text((60.0, y), label, fontsize=9)
        for x, value in zip(value_xs, values):
            page.insert_text((x, y), value, fontsize=9)
        y += 18.0
    page.draw_line(fitz.Point(50, 190), fitz.Point(470, 190))
    page.draw_line(fitz.Point(50, y), fitz.Point(470, y))
    doc.save(path)
    doc.close()
    return fitz.open(path)


@pytest.fixture(scope="module")
def battery_page(tmp_path_factory):
    path = tmp_path_factory.mktemp("battery") / "battery.pdf"
    opened = _build_pdf(path)
    yield opened[0]
    opened.close()


# ----------------------------------------------------------------------
# Markdown rendering, and small row-list transforms. Transforms operate on the
# logical (label, values) rows rather than raw markdown text, so each one states
# its intent directly instead of via string surgery.
# ----------------------------------------------------------------------


def _row_line(label: str, values: tuple[str, ...], width: int = _WIDTH) -> str:
    padded = list(values) + [""] * (width - len(values))
    return "| " + label + " | " + " | ".join(padded) + " |"


def _table_md(rows: list[tuple[str, tuple[str, ...]]], width: int = _WIDTH) -> str:
    header = "| | " + " | ".join(f"c{i + 1}" for i in range(width)) + " |"
    sep = "| " + " | ".join(["---"] * (width + 1)) + " |"
    body = [_row_line(label, values, width) for label, values in rows]
    return "\n".join([header, sep, *body])


def _replace_row(rows, index, *, label=None, values=None):
    new_rows = list(rows)
    old_label, old_values = new_rows[index]
    new_rows[index] = (
        label if label is not None else old_label,
        values if values is not None else old_values,
    )
    return new_rows


def _append_row(rows, label, values=()):
    return [*rows, (label, values)]


def _swap_values_between_rows(rows):
    """Swap column-0 between the first two rows - same digits, wrong rows."""
    new_rows = list(rows)
    (label0, values0), (label1, values1) = new_rows[0], new_rows[1]
    new_rows[0] = (label0, (values1[0], values0[1]))
    new_rows[1] = (label1, (values0[0], values1[1]))
    return new_rows


# ----------------------------------------------------------------------
# The battery, table-driven: each entry perturbs the emitted markdown while the
# native PDF (the ground truth) stays fixed.
# ----------------------------------------------------------------------

BENIGN = [
    pytest.param(
        "bold_a_value_cell",
        lambda rows: _replace_row(rows, 1, values=("**24.8**", "18.2")),
        id="bold_a_value_cell",
    ),
    # The five footnote-marker spellings ``normalize_label`` already folds.
    # #123 TICKETS.md "Known landmines": do not add a sixth special case here -
    # this exercises the existing fold, it does not extend it.
    pytest.param(
        "footnote_bare_digit",
        lambda rows: _replace_row(rows, 0, label="Total spending1"),
        id="footnote_bare_digit",
    ),
    pytest.param(
        "footnote_latex_dollar",
        lambda rows: _replace_row(rows, 0, label="Total spending$^1$"),
        id="footnote_latex_dollar",
    ),
    pytest.param(
        "footnote_latex_brace",
        lambda rows: _replace_row(rows, 0, label="Total spending^{1}"),
        id="footnote_latex_brace",
    ),
    pytest.param(
        "footnote_html_sup",
        lambda rows: _replace_row(rows, 0, label="Total spending<sup>1,2</sup>"),
        id="footnote_html_sup",
    ),
    pytest.param(
        "footnote_unicode",
        lambda rows: _replace_row(rows, 0, label="Total spending¹"),
        id="footnote_unicode",
    ),
    pytest.param(
        "reflow_whitespace",
        lambda rows: _replace_row(
            rows, 0, label="  Total spending  ", values=("  42.8  ", " 30.5 ")
        ),
        id="reflow_whitespace",
    ),
    pytest.param(
        "add_note_row",
        lambda rows: _append_row(rows, "Note: figures may not sum due to rounding"),
        id="add_note_row",
    ),
]

CORRUPTING = [
    pytest.param(
        "shift_into_adjacent_empty_cell",
        lambda rows: _replace_row(rows, 4, values=("", "0.5")),
        marks=pytest.mark.xfail(
            reason=(
                "#123 TICKET-B2: markdown_rows filters empty cells out of a row "
                "before comparing, so a value shifted into an adjacent empty "
                "cell still reduces to the same value multiset and scores as "
                "correct. Fixed by TICKET-B2 (preserve cell positions instead "
                "of compacting them)."
            ),
            strict=True,
        ),
        id="shift_into_adjacent_empty_cell",
    ),
    pytest.param(
        "swap_values_between_rows",
        _swap_values_between_rows,
        id="swap_values_between_rows",
    ),
    pytest.param(
        "drop_a_value",
        lambda rows: _replace_row(rows, 3, values=("3.8",)),
        id="drop_a_value",
    ),
    pytest.param(
        "perturb_a_digit",
        lambda rows: _replace_row(rows, 3, values=("3.9", "2.6")),
        id="perturb_a_digit",
    ),
]


def test_base_fixture_is_scorable(battery_page):
    """Constraint from B1: every fixture below must clear the grid predicate.

    A battery that silently degenerates to comparing ``None`` to ``None`` is
    worse than no battery - pin the base fixture's scorability before trusting
    any "strictly worse" assertion built on top of it.
    """
    report = score_page(battery_page, _table_md(_ROWS))

    assert report.scorable is True
    assert report.pct is not None
    assert report.pct == 100.0


@pytest.mark.parametrize("name,transform", BENIGN)
def test_benign_transform_leaves_score_unchanged(battery_page, name, transform):
    base = score_page(battery_page, _table_md(_ROWS))
    assert base.scorable is True and base.pct is not None

    perturbed = score_page(battery_page, _table_md(transform(_ROWS)))

    assert perturbed.scorable is True
    assert perturbed.pct == base.pct, name


@pytest.mark.parametrize("name,transform", CORRUPTING)
def test_corrupting_transform_makes_score_strictly_worse(battery_page, name, transform):
    base = score_page(battery_page, _table_md(_ROWS))
    assert base.scorable is True and base.pct is not None

    corrupted = score_page(battery_page, _table_md(transform(_ROWS)))

    assert corrupted.scorable is True
    assert corrupted.pct is not None
    assert corrupted.pct < base.pct, name


# ----------------------------------------------------------------------
# Wrap a label across two lines.
#
# Not folded into BENIGN above: it perturbs the *native PDF*, not the markdown,
# so it needs its own fixture pair.
#
# This is a SECOND defect found by this battery, distinct from TICKET-B2 and not
# tracked by an existing ticket. #123's own commit 270cdab ("don't merge a
# wrapped label with the line above it") stops the row parser from
# INTERLEAVING a wrapped label's words with an unrelated neighbouring line, but
# it does not reconstruct a label that is genuinely split across two visual
# bands: whichever band carries the row's values keeps only the text on that
# band, and the other line is silently dropped from the ground truth entirely.
# A perfect transcription of a page with a two-line-wrapped row label is
# therefore scored as if that row were missing - the same wrong-direction shape
# as all seven defects this battery exists to catch.
# ----------------------------------------------------------------------


@pytest.fixture
def wrapped_label_page(tmp_path):
    """ "Central government net debt" wrapped across two lines, plus one plain row."""
    path = tmp_path / "wrapped_label.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60.0, 200.0), "Central government net", fontsize=9)
    page.insert_text((60.0, 209.0), "debt", fontsize=9)
    for x, value in zip(_VALUE_XS, ("1.0", "2.0")):
        page.insert_text((x, 209.0), value, fontsize=9)
    page.insert_text((60.0, 230.0), "Other row", fontsize=9)
    for x, value in zip(_VALUE_XS, ("4.0", "5.0")):
        page.insert_text((x, 230.0), value, fontsize=9)
    page.draw_line(fitz.Point(50, 190), fitz.Point(470, 190))
    page.draw_line(fitz.Point(50, 245), fitz.Point(470, 245))
    doc.save(path)
    doc.close()

    opened = fitz.open(path)
    yield opened[0]
    opened.close()


@pytest.mark.xfail(
    reason=(
        "Second defect found by this battery, not tracked by an existing "
        "ticket: native_rows_from_page never reconstructs a label genuinely "
        "split across two visual bands - it keeps only the text on the "
        "value-bearing band and silently drops the other line, so a perfect "
        "transcription of a wrapped-label row scores as if the row were "
        "missing."
    ),
    strict=True,
)
def test_wrapped_label_is_scored_the_same_as_unwrapped(wrapped_label_page):
    md = _table_md([("Central government net debt", ("1.0", "2.0")), ("Other row", ("4.0", "5.0"))])

    report = score_page(wrapped_label_page, md)

    assert report.scorable is True
    assert report.pct == 100.0
