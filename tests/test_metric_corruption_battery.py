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


# ----------------------------------------------------------------------
# The invariant the battery was missing.
#
# Every assertion above is RELATIVE (``corrupted.pct < base.pct``), and the base
# fixture's sparse row has a TRAILING gap ("| 0.5 |  |"), which stays aligned even
# when the two sides of the comparison disagree about what a gap is. So the whole
# battery could pass while a perfect transcription scored well under 100%.
#
# It did. TICKET-B2's first attempt (reverted in 717914d) made the markdown side
# positional while the ground truth stayed compacted, and a faithful transcription
# of a LEADING-gap sparse row scored 80% while a sloppy one that dropped the column
# entirely scored 100% - the metric rewarding the engine that destroyed the table's
# structure. Nothing in the battery moved.
#
# This pins the most basic metamorphic property of all, with the leading gap that
# exposes the asymmetry: a perfect transcription scores 100%, full stop.
# ----------------------------------------------------------------------


@pytest.fixture
def leading_gap_page(tmp_path):
    """Three rows, two value columns; the middle row's only value is in column 2."""
    path = tmp_path / "leading_gap.pdf"
    doc = fitz.open()
    page = doc.new_page()
    placements = [
        ("Alpha", (("1.0", 0), ("2.0", 1))),
        ("Sparse", (("0.5", 1),)),
        ("Beta", (("4.0", 0), ("5.0", 1))),
    ]
    y = 200.0
    for label, cells in placements:
        page.insert_text((60.0, y), label, fontsize=9)
        for value, column in cells:
            page.insert_text((_VALUE_XS[column], y), value, fontsize=9)
        y += 18.0
    page.draw_line(fitz.Point(50, 190), fitz.Point(470, 190))
    page.draw_line(fitz.Point(50, y), fitz.Point(470, y))
    doc.save(path)
    doc.close()

    opened = fitz.open(path)
    yield opened[0]
    opened.close()


def test_a_perfect_transcription_scores_100(leading_gap_page):
    """A faithful transcription, gap included, must score 100%."""
    md = _table_md([("Alpha", ("1.0", "2.0")), ("Sparse", ("", "0.5")), ("Beta", ("4.0", "5.0"))])

    report = score_page(leading_gap_page, md)

    assert report.scorable is True
    assert report.pct == 100.0, f"faithful transcription scored {report.pct}%, not 100%"


def test_dropping_a_column_never_beats_keeping_the_gap(leading_gap_page):
    """The metric must never reward an engine for destroying column structure.

    The direction matters more than either absolute score: if omitting the empty
    cell outscores reproducing it, the production accept rule in
    ``escalation_decision`` prefers the worse engine.
    """
    faithful = _table_md(
        [("Alpha", ("1.0", "2.0")), ("Sparse", ("", "0.5")), ("Beta", ("4.0", "5.0"))]
    )
    sloppy = _table_md([("Alpha", ("1.0", "2.0")), ("Beta", ("4.0", "5.0"))], width=_WIDTH)
    sloppy = sloppy.replace("| Beta |", "| Sparse | 0.5 |\n| Beta |", 1)

    faithful_report = score_page(leading_gap_page, faithful)
    sloppy_report = score_page(leading_gap_page, sloppy)

    assert faithful_report.exact >= sloppy_report.exact, (
        f"dropping the column scored {sloppy_report.exact} exact cells vs "
        f"{faithful_report.exact} for keeping the gap - the metric is rewarding "
        "structure-destroying output"
    )


# ----------------------------------------------------------------------
# #123 TICKET-B2 design note, section 6: the alignment is fitted independently to
# whichever markdown is being scored, so a wider candidate has strictly more
# admissible maps (C(M, L)) than a narrower one. Nothing above catches that -
# every fixture pair above has the SAME column count on both sides of a
# comparison. Pin the asymmetry directly: an incumbent and a same-shift candidate
# that additionally pads on a spurious column must score the same, not better.
# ----------------------------------------------------------------------


def test_same_shift_plus_spurious_column_never_beats_the_incumbent(leading_gap_page):
    """A wider candidate must not win on extra map freedom alone.

    Both markdowns apply the identical shift (the sparse row's value lands in
    column 1 instead of column 2); the candidate additionally appends a spurious
    third column filled with a low-entropy repeated value. If the extra column
    bought the candidate a better map, ``accept candidate iff candidate.exact >
    incumbent.exact`` would prefer it over an equally-wrong incumbent for free.
    """
    incumbent = _table_md(
        [("Alpha", ("1.0", "2.0")), ("Sparse", ("0.5", "")), ("Beta", ("4.0", "5.0"))],
        width=2,
    )
    candidate = _table_md(
        [
            ("Alpha", ("1.0", "2.0", "0.0")),
            ("Sparse", ("0.5", "", "0.0")),
            ("Beta", ("4.0", "5.0", "0.0")),
        ],
        width=3,
    )

    incumbent_report = score_page(leading_gap_page, incumbent)
    candidate_report = score_page(leading_gap_page, candidate)

    assert candidate_report.exact <= incumbent_report.exact, (
        f"same-shift-plus-spurious-column candidate scored {candidate_report.exact} "
        f"exact cells vs {incumbent_report.exact} for the incumbent it should not beat"
    )


def test_padding_with_low_entropy_columns_never_beats_a_faithful_transcription(tmp_path):
    """Padding with junk columns must not outscore a faithful, narrower transcription.

    Three dense rows, two real value columns. The padded candidate reproduces both
    real columns faithfully and then appends three extra columns of repeated
    low-entropy values (``0.0``, ``-``, blank) that could never be the map target
    for any ground-truth lane. Padding must not raise the score above the faithful
    transcription's.
    """
    value_xs = (300.0, 360.0)
    path = tmp_path / "padding.pdf"
    doc = fitz.open()
    page = doc.new_page()
    rows = [
        ("Alpha", ("1.0", "2.0")),
        ("Beta", ("4.0", "5.0")),
        ("Gamma", ("6.0", "7.0")),
    ]
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

    opened = fitz.open(path)
    page = opened[0]

    faithful = _table_md(rows, width=2)
    padded_rows = [(label, (*values, "0.0", "-", "")) for label, values in rows]
    padded = _table_md(padded_rows, width=5)

    faithful_report = score_page(page, faithful)
    padded_report = score_page(page, padded)
    opened.close()

    assert padded_report.exact <= faithful_report.exact, (
        f"padding with junk columns scored {padded_report.exact} exact cells vs "
        f"{faithful_report.exact} for the faithful narrower transcription"
    )


# ----------------------------------------------------------------------
# Width coverage.
#
# Every fixture above uses exactly TWO value columns, which is the one width the
# lane splitter handled when it first landed (27aa432): a regular grid of three or
# more evenly-spaced columns collapsed into a single lane, and a faithful
# transcription of a dense 4-column table scored 25% instead of 100% - worse than
# the pre-lane positional comparison it replaced. The suite was green throughout
# because nothing exercised width >= 3.
#
# Regular, evenly-spaced columns are the canonical financial-table geometry, so
# this parametrises the perfect-transcription invariant across widths.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("width", [2, 3, 4, 5])
def test_a_perfect_transcription_of_a_regular_grid_scores_100(tmp_path, width):
    """Evenly-spaced columns must not collapse into one lane."""
    xs = tuple(250.0 + 50.0 * i for i in range(width))
    labels = ("Alpha", "Beta", "Gamma")
    values = [tuple(f"{r + 1}.{c + 1}" for c in range(width)) for r in range(len(labels))]

    path = tmp_path / f"regular_{width}.pdf"
    doc = fitz.open()
    page = doc.new_page()
    y = 200.0
    for label, row_values in zip(labels, values):
        page.insert_text((60.0, y), label, fontsize=9)
        for x, value in zip(xs, row_values):
            page.insert_text((x, y), value, fontsize=9)
        y += 18.0
    page.draw_line(fitz.Point(50, 190), fitz.Point(470, 190))
    page.draw_line(fitz.Point(50, y), fitz.Point(470, y))
    doc.save(path)
    doc.close()

    opened = fitz.open(path)
    try:
        report = score_page(opened[0], _table_md(list(zip(labels, values)), width=width))
    finally:
        opened.close()

    assert report.scorable is True
    assert report.pct == 100.0, (
        f"faithful transcription of a {width}-column regular grid scored "
        f"{report.pct}% ({report.exact}/{report.cells})"
    )


# ----------------------------------------------------------------------
# Offset invariance.
#
# The regular-grid test above passes at one particular page offset. Shifting the
# same table 20pt left changes nothing structurally, but glyph rendering emits a
# third distinct gap magnitude from floating-point noise (50.0 vs 50.000015).
# The ratio-jump splitter then picks that meaningless 50.0 -> 50.000015 jump as
# its elbow instead of the real 0 -> 50 one, and four of five columns collapse
# into a single lane.
#
# Lane assignment must be a function of the table's structure, not of where the
# table happens to sit on the page or of sub-micrometre rendering noise. Real
# PDFs carry this noise constantly; the fixtures above were simply lucky.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("start_x", [230.0, 240.0, 250.0, 260.0])
def test_lane_assignment_is_invariant_to_page_offset(tmp_path, start_x):
    """The same grid, shifted horizontally, must produce the same lanes."""
    width = 5
    xs = tuple(start_x + 50.0 * i for i in range(width))
    labels = ("Alpha", "Beta", "Gamma")
    values = [tuple(f"{r + 1}.{c + 1}" for c in range(width)) for r in range(len(labels))]

    path = tmp_path / f"offset_{int(start_x)}.pdf"
    doc = fitz.open()
    page = doc.new_page()
    y = 200.0
    for label, row_values in zip(labels, values):
        page.insert_text((60.0, y), label, fontsize=9)
        for x, value in zip(xs, row_values):
            page.insert_text((x, y), value, fontsize=9)
        y += 18.0
    page.draw_line(fitz.Point(50, 190), fitz.Point(560, 190))
    page.draw_line(fitz.Point(50, y), fitz.Point(560, y))
    doc.save(path)
    doc.close()

    opened = fitz.open(path)
    try:
        report = score_page(opened[0], _table_md(list(zip(labels, values)), width=width))
    finally:
        opened.close()

    assert report.pct == 100.0, (
        f"a faithful transcription of the same grid at x={start_x} scored "
        f"{report.pct}% ({report.exact}/{report.cells}) — lane assignment is "
        "sensitive to page offset or rendering noise"
    )
