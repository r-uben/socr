"""#734 Stage A: a FILLED chart grid is reconciled against the region's geometry.

The defect this answers: #635 Stage 0 acts only where the model left an EMPTY
grid, so a model that writes NUMBERS into a chart region's grid bypasses the
geometric reader entirely and the page ships those numbers unchecked. On the
corpus page the shipped grid carries ``| 0.13-0.37 | 17 | 17 |`` while the
reader independently derives ``17, 0, 0, ...`` for the same panel, and nothing
compares the two.

Every pin here is a DIFFERENCE: the same grid against two panels, or the same
panel against two grids, with exactly one thing changed. No test asserts an
absolute outcome measured on one machine, and nothing here needs a provider --
the functions under test are pure.

The rule being pinned (``docs/plans/chart-data/DESIGN.md``, Stage 2): reconcile
by cell identity; never publish a model number contradicted by geometry; never
average competing counts; never use totals to force agreement; and geometry
having no opinion is not a contradiction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from test_gh735_sep_reader import BINS, SOLID, read_panel  # noqa: E402

from socr.figures.chart_data import (  # noqa: E402
    find_empty_skeletons,
    find_filled_grids,
)
from socr.figures.chart_reader import (  # noqa: E402
    INTEGER,
    PRESENT,
    UNRESOLVED,
    Bin,
    Cell,
    Frame,
    PanelReading,
    SeriesReading,
    YCalibration,
)
from socr.figures.chart_reconcile import (  # noqa: E402
    AGREED,
    BINS_IN_COLUMN,
    BINS_IN_HEADER,
    CAUSE_BIN_UNMATCHED,
    CAUSE_CELL_ABSENT,
    CAUSE_NEITHER_MATCHED,
    CAUSE_READER_UNRESOLVED,
    CAUSE_SERIES_UNMATCHED,
    CONTRADICTED,
    NOT_A_COUNT,
    UNKNOWN_TO_GEOMETRY,
    CellVerdict,
    GridReconciliation,
    _bin_key,
    reconcile_grid,
)

# The panel these fixtures are about: three printed percent-range bins and one
# series, which is the shape of the Fed SEP panels the ticket is about.
RANGES = ("0.13-0.37", "0.38-0.62", "0.63-0.87")
DASHED = "December projections"
#: Two series' worth of counts over RANGES, distinct so no cell can be matched
#: to the wrong one by value.
TRIPLE_A = {RANGES[0]: 1, RANGES[1]: 2, RANGES[2]: 3}
TRIPLE_B = {RANGES[0]: 4, RANGES[1]: 5, RANGES[2]: 6}


def _rows_for(*series: dict[str, int]) -> list[list[str]]:
    """Grid rows: one per bin, one column per series, in RANGES order."""
    return [[b, *[str(s[b]) for s in series]] for b in RANGES]


def _cell(label: str, count: int | None) -> Cell:
    """One reader cell. ``None`` is UNRESOLVED -- geometry with no opinion."""
    status = INTEGER if count is not None else UNRESOLVED
    return Cell(bin_label=label, status=status, count=count, interval=(0.0, 0.0), detail="")


def make_panel(
    series: dict[str, dict[str, int | None]],
    *,
    bins: tuple[str, ...] = RANGES,
    page_num: int = 1,
    region_index: int = 1,
) -> PanelReading:
    """A PanelReading holding exactly the counts asked for, and nothing else."""
    return PanelReading(
        page_num=page_num,
        region_index=region_index,
        label="December 2020",
        bins=tuple(
            Bin(label=b, centre=float(i), lo=float(i), hi=float(i) + 1.0)
            for i, b in enumerate(bins)
        ),
        series=tuple(
            SeriesReading(
                name=name,
                style="solid_fill",
                presence=PRESENT,
                cells=tuple(_cell(b, counts[b]) for b in bins if b in counts),
            )
            for name, counts in series.items()
        ),
        calibration=YCalibration(
            points_per_unit=10.0, zero_y=400.0, pairs=((400.0, 0.0),), residual=0.1, checked_ticks=2
        ),
        frame=Frame(baseline=400.0, x0=100.0, x1=400.0, tick_ys=(380.0, 360.0)),
    )


def grid_text(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def make_grid(header: list[str], rows: list[list[str]]):
    found = find_filled_grids(grid_text(header, rows))
    assert len(found) == 1, f"expected one filled grid, got {len(found)}"
    return found[0]


def by_bin(result) -> dict[str, tuple[str, int | None]]:
    return {c.bin_label: (c.status, c.published) for c in result.cells}


# ---------------------------------------------------------------------------
# The finder: the complement of Stage 0's, over the same parse
# ---------------------------------------------------------------------------


def test_one_filled_cell_moves_a_grid_from_one_finder_to_the_other() -> None:
    """The ONLY difference is whether a data cell says anything."""
    empty = grid_text(["Percent range", "Participants"], [["0.13-0.37", ""]])
    filled = grid_text(["Percent range", "Participants"], [["0.13-0.37", "17"]])

    assert len(find_empty_skeletons(empty)) == 1
    assert find_filled_grids(empty) == []
    assert find_empty_skeletons(filled) == []
    assert len(find_filled_grids(filled)) == 1


def test_a_grid_with_no_data_position_is_neither_empty_nor_filled() -> None:
    one_column = "| Percent range |\n| --- |\n| 0.13-0.37 |"
    assert find_empty_skeletons(one_column) == []
    assert find_filled_grids(one_column) == []


def test_a_fenced_grid_is_invisible_to_both_finders() -> None:
    """The literal-context mask is shared, so a code SAMPLE is neither."""
    body = grid_text(["Percent range", "Participants"], [["0.13-0.37", "17"]])
    assert len(find_filled_grids(body)) == 1
    assert find_filled_grids(f"```\n{body}\n```") == []
    assert find_empty_skeletons("```\n" + body.replace("17", "") + "\n```") == []


def test_the_filled_grid_keeps_the_labels_the_page_gave_each_cell() -> None:
    grid = make_grid(["Percent range", "Sep", "Dec"], [["0.13-0.37", "1", "17"]])
    assert grid.label_header == "Percent range"
    assert grid.data_headers == ["Sep", "Dec"]
    assert grid.rows == (("0.13-0.37", ("1", "17")),)


# ---------------------------------------------------------------------------
# Agreement, contradiction, and geometry with no opinion
# ---------------------------------------------------------------------------


def test_the_same_number_against_two_panels_agrees_with_one_and_not_the_other() -> None:
    grid = make_grid(["Percent range", SOLID], [[RANGES[0], "17"]])

    agrees = reconcile_grid(grid, make_panel({SOLID: {RANGES[0]: 17}}))
    differs = reconcile_grid(grid, make_panel({SOLID: {RANGES[0]: 16}}))

    assert by_bin(agrees) == {RANGES[0]: (AGREED, 17)}
    assert by_bin(differs) == {RANGES[0]: (CONTRADICTED, None)}


def test_geometry_with_no_number_is_not_a_contradiction() -> None:
    """UNRESOLVED vs a DIFFERENT integer -- the one thing changed is the cell."""
    grid = make_grid(["Percent range", SOLID], [[RANGES[0], "17"]])

    silent = reconcile_grid(grid, make_panel({SOLID: {RANGES[0]: None}}))
    speaks = reconcile_grid(grid, make_panel({SOLID: {RANGES[0]: 16}}))

    assert by_bin(silent) == {RANGES[0]: (UNKNOWN_TO_GEOMETRY, None)}
    assert by_bin(speaks) == {RANGES[0]: (CONTRADICTED, None)}
    # Unverified, not impeached: the model's value survives on the record.
    assert silent.cells[0].model_count == 17
    assert silent.cells[0].reader_count is None


def test_a_bin_geometry_never_read_is_unknown_not_contradicted() -> None:
    grid = make_grid(["Percent range", SOLID], [[RANGES[0], "17"], [RANGES[1], "3"]])

    both = reconcile_grid(grid, make_panel({SOLID: {RANGES[0]: 17, RANGES[1]: 0}}))
    one = reconcile_grid(grid, make_panel({SOLID: {RANGES[0]: 17}}))

    assert by_bin(both)[RANGES[1]] == (CONTRADICTED, None)
    assert by_bin(one)[RANGES[1]] == (UNKNOWN_TO_GEOMETRY, None)
    assert one.unmatched_bins == (RANGES[1],)


def test_a_cell_carrying_no_count_is_not_a_contradiction_either() -> None:
    """Same panel, same identity; the only change is what the grid cell says."""
    panel = make_panel({SOLID: {RANGES[0]: 17}})

    number = reconcile_grid(make_grid(["Percent range", SOLID], [[RANGES[0], "16"]]), panel)
    marker = reconcile_grid(make_grid(["Percent range", SOLID], [[RANGES[0], "n/a"]]), panel)

    assert by_bin(number) == {RANGES[0]: (CONTRADICTED, None)}
    assert by_bin(marker) == {RANGES[0]: (NOT_A_COUNT, None)}


# ---------------------------------------------------------------------------
# The three prohibitions
# ---------------------------------------------------------------------------


def test_a_contradicted_number_is_published_by_nothing_in_the_result() -> None:
    grid = make_grid(["Percent range", SOLID], [[RANGES[0], "17"]])
    result = reconcile_grid(grid, make_panel({SOLID: {RANGES[0]: 3}}))

    verdict = result.cells[0]
    assert verdict.status == CONTRADICTED
    assert verdict.published is None
    # Both numbers are on the record, and neither is presented as the count.
    assert (verdict.model_count, verdict.reader_count) == (17, 3)
    assert [c.published for c in result.cells] == [None]


def test_competing_counts_are_never_averaged() -> None:
    grid = make_grid(["Percent range", SOLID], [[RANGES[0], "17"]])
    result = reconcile_grid(grid, make_panel({SOLID: {RANGES[0]: 3}}))

    mean = (17 + 3) // 2
    assert mean not in {c.published for c in result.cells}
    assert mean not in {c.reader_count for c in result.cells}
    assert mean not in {c.model_count for c in result.cells}


def test_a_matching_total_cannot_force_a_single_cell_to_agree() -> None:
    """Same multiset, same total, different binding -- every cell contradicts."""
    counts = {RANGES[0]: 17, RANGES[1]: 0, RANGES[2]: 0}
    permuted = {RANGES[0]: 0, RANGES[1]: 0, RANGES[2]: 17}
    grid = make_grid(["Percent range", SOLID], [[b, str(counts[b])] for b in RANGES])

    same = reconcile_grid(grid, make_panel({SOLID: counts}))
    shifted = reconcile_grid(grid, make_panel({SOLID: permuted}))

    assert sum(counts.values()) == sum(permuted.values())
    assert [c.status for c in same.cells] == [AGREED, AGREED, AGREED]
    assert [c.status for c in shifted.cells] == [CONTRADICTED, AGREED, CONTRADICTED]
    assert [c.published for c in shifted.cells] == [None, 0, None]


# ---------------------------------------------------------------------------
# Identity, not position
# ---------------------------------------------------------------------------


def test_the_transpose_of_a_grid_reconciles_to_the_same_verdicts() -> None:
    """Bins down the side or across the top: the cells are the same cells."""
    panel = make_panel({SOLID: {RANGES[0]: 17, RANGES[1]: 0}})
    down = make_grid(["Percent range", SOLID], [[RANGES[0], "17"], [RANGES[1], "5"]])
    across = make_grid(["Series", RANGES[0], RANGES[1]], [[SOLID, "17", "5"]])

    a, b = reconcile_grid(down, panel), reconcile_grid(across, panel)

    assert (a.orientation, b.orientation) == (BINS_IN_COLUMN, BINS_IN_HEADER)
    assert by_bin(a) == by_bin(b) == {RANGES[0]: (AGREED, 17), RANGES[1]: (CONTRADICTED, None)}


def test_a_bin_written_with_an_en_dash_is_the_bin_the_chart_printed() -> None:
    """One atom changed makes it a different bin; the dash alone does not."""
    panel = make_panel({SOLID: {"0.13-0.37": 17, "0.38-0.62": 4}})
    spaced = make_grid(["Percent range", SOLID], [["0.13 \u2013 0.37", "17"], ["0.38-0.62", "4"]])
    other = make_grid(["Percent range", SOLID], [["0.13-9.99", "17"], ["0.38-0.62", "4"]])

    a, b = reconcile_grid(spaced, panel), reconcile_grid(other, panel)

    assert [c.status for c in a.cells] == [AGREED, AGREED]
    assert [c.status for c in b.cells] == [UNKNOWN_TO_GEOMETRY, AGREED]
    assert (a.unmatched_bins, b.unmatched_bins) == ((), ("0.13-9.99",))


def test_an_indented_entity_series_label_is_the_same_series() -> None:
    """``decode_label_cell``'s view, which is the binder's -- not raw bytes."""
    panel = make_panel({SOLID: {RANGES[0]: 17}})
    across = ["Series", RANGES[0]]

    padded = reconcile_grid(make_grid(across, [[f"&nbsp;&nbsp;**{SOLID}**", "17"]]), panel)
    renamed = reconcile_grid(make_grid(across, [["June projections", "17"]]), panel)

    assert [c.status for c in padded.cells] == [AGREED]
    assert [c.status for c in renamed.cells] == [UNKNOWN_TO_GEOMETRY]
    assert renamed.unmatched_series == ("June projections",)


# ---------------------------------------------------------------------------
# Refusals: no cell identity means no verdict, in either direction
# ---------------------------------------------------------------------------


def test_a_repeated_label_refuses_the_grid_rather_than_guessing() -> None:
    panel = make_panel({SOLID: {RANGES[0]: 17, RANGES[1]: 0}})
    unique = make_grid(["Percent range", SOLID], [[RANGES[0], "17"], [RANGES[1], "0"]])
    repeated = make_grid(["Percent range", SOLID], [[RANGES[0], "17"], [RANGES[0], "0"]])

    ok, refused = reconcile_grid(unique, panel), reconcile_grid(repeated, panel)

    assert ok.refusal == "" and len(ok.cells) == 2
    assert refused.refusal and refused.cells == ()
    assert (refused.agreed, refused.contradicted, refused.unknown) == (0, 0, 0)


def test_a_grid_sharing_no_identity_with_the_panel_is_refused() -> None:
    panel = make_panel({SOLID: {RANGES[0]: 17}})
    bound = make_grid(["Percent range", SOLID], [[RANGES[0], "17"]])
    foreign = make_grid(["Quarter", "Revenue"], [["Q1", "17"]])

    assert reconcile_grid(bound, panel).refusal == ""
    assert reconcile_grid(foreign, panel).refusal != ""
    assert reconcile_grid(foreign, panel).cells == ()


def test_a_panel_with_nothing_read_reconciles_nothing() -> None:
    grid = make_grid(["Percent range", SOLID], [[RANGES[0], "17"]])

    read = reconcile_grid(grid, make_panel({SOLID: {RANGES[0]: 17}}))
    unread = reconcile_grid(grid, make_panel({SOLID: {}}))

    assert read.cells and read.refusal == ""
    assert unread.cells == () and unread.refusal != ""


# ---------------------------------------------------------------------------
# Against the reader the pipeline actually runs
# ---------------------------------------------------------------------------


def test_the_real_reader_contradicts_the_one_number_that_was_changed(tmp_path: Path) -> None:
    """A drawn chart, read by ``read_chart_page``; only the grid's cell moves."""
    drawn = {BINS[0]: 3, BINS[1]: 5, BINS[2]: 0, BINS[3]: 2}
    panel, refusal = read_panel(tmp_path, "gh734", bars=drawn)
    assert refusal is None and panel is not None

    header = ["Series", *BINS]
    honest = make_grid(header, [[SOLID, *[str(drawn[b]) for b in BINS]]])
    tampered = make_grid(header, [[SOLID, "3", "17", "0", "2"]])

    good, bad = reconcile_grid(honest, panel), reconcile_grid(tampered, panel)

    # The drawn counts are never contradicted; the tampered one always is, and
    # it is the only cell whose verdict moves. What the reader makes of the
    # empty bin is not pinned -- only the difference between the two grids is.
    assert good.contradicted == 0
    assert bad.contradicted == 1
    moved = [b for g, b in zip(good.cells, bad.cells, strict=True) if g.status != b.status]
    assert [(c.bin_label, c.model_count, c.published) for c in moved] == [(BINS[1], 17, None)]
    assert moved[0].reader_count == drawn[BINS[1]]


# ---------------------------------------------------------------------------
# Why a cell is unknown: four failures that wear one status
# ---------------------------------------------------------------------------


def test_the_cause_separates_geometrys_own_limit_from_a_cell_never_compared() -> None:
    """One grid, two panels. The status is identical; the CAUSE is not.

    The distinction the disclosure turns on: geometry read this cell and could
    not resolve it, versus geometry holds a reading for this bin that the
    series label never met. The second is the laundering surface.
    """
    grid = make_grid(["Percent range", SOLID], [[RANGES[0], "8"]])

    silent = reconcile_grid(grid, make_panel({SOLID: {RANGES[0]: None}}))
    missed = reconcile_grid(grid, make_panel({"December projections": {RANGES[0]: 7}}))

    assert [c.status for c in silent.cells] == [c.status for c in missed.cells]
    assert silent.cells[0].cause == CAUSE_READER_UNRESOLVED
    assert missed.cells[0].cause == CAUSE_SERIES_UNMATCHED


def test_a_cell_never_compared_does_not_claim_geometry_has_no_such_cell() -> None:
    """The detail must be true of THIS cell: the bin matched, and was read."""
    grid = make_grid(["Percent range", SOLID], [[RANGES[0], "8"]])
    missed = reconcile_grid(grid, make_panel({"December projections": {RANGES[0]: 7}}))

    detail = missed.cells[0].detail
    assert missed.unmatched_bins == ()
    assert "no such cell" not in detail
    assert "never compared" in detail


def test_each_way_of_missing_has_its_own_cause() -> None:
    grid = make_grid(["Percent range", SOLID], [[RANGES[0], "1"], [RANGES[1], "2"]])
    other = make_grid(
        ["Percent range", "December projections"], [[RANGES[0], "1"], ["9.9-9.99", "2"]]
    )

    reader_limit = reconcile_grid(grid, make_panel({SOLID: {RANGES[0]: None}}))
    label_miss = reconcile_grid(other, make_panel({SOLID: {RANGES[0]: 1}}))
    absent = reconcile_grid(
        make_grid(["Percent range", "B"], [[RANGES[0], "1"]]),
        make_panel({"A": {RANGES[0]: 1}, "B": {RANGES[1]: 2}}),
    )

    assert [c.cause for c in reader_limit.cells] == [CAUSE_READER_UNRESOLVED, CAUSE_BIN_UNMATCHED]
    assert [c.cause for c in label_miss.cells] == [CAUSE_SERIES_UNMATCHED, CAUSE_NEITHER_MATCHED]
    assert [c.cause for c in absent.cells] == [CAUSE_CELL_ABSENT]


def test_a_cause_is_recorded_for_unknown_cells_and_for_nothing_else() -> None:
    panel = make_panel({SOLID: {RANGES[0]: 7, RANGES[1]: None}})
    grid = make_grid(["Percent range", SOLID], [[RANGES[0], "7"], [RANGES[1], "2"]])

    result = reconcile_grid(grid, panel)

    assert [(c.status, bool(c.cause)) for c in result.cells] == [
        (AGREED, False),
        (UNKNOWN_TO_GEOMETRY, True),
    ]


# ---------------------------------------------------------------------------
# The inverse asymmetry: a reading the grid publishes no column for
# ---------------------------------------------------------------------------


def test_a_collapse_that_keeps_a_real_series_name_is_not_a_clean_sheet() -> None:
    """The dangerous shape: cells DO publish, and half the chart is unconsulted.

    Both grids name a real series and agree on every cell they carry. The
    second simply publishes one column where the panel holds two, so every
    model→reader field reports perfectly: no unmatched bin, no unmatched
    series, no contradiction, no refusal. Only the reader→model direction can
    say that three proven readings were never asked about.
    """
    panel = make_panel({SOLID: TRIPLE_A, DASHED: TRIPLE_B})
    both = make_grid(["Percent range", SOLID, DASHED], _rows_for(TRIPLE_A, TRIPLE_B))
    one = make_grid(["Percent range", SOLID], _rows_for(TRIPLE_A))

    full, collapsed = reconcile_grid(both, panel), reconcile_grid(one, panel)

    assert (full.agreed, collapsed.agreed) == (6, 3)
    assert (full.contradicted, collapsed.contradicted) == (0, 0)
    assert collapsed.unmatched_bins == () and collapsed.unmatched_series == ()
    assert collapsed.refusal == ""
    assert (full.uncovered_count, collapsed.uncovered_count) == (0, 3)
    assert (full.verified, collapsed.verified) == (True, False)


def test_a_dropped_bin_row_is_uncovered_though_no_series_was_dropped() -> None:
    """Why the field is per identity: nothing dropped here IS a series.

    Both series appear in the grid and every cell present agrees. One printed
    bin row is missing, so two proven readings go unaddressed -- and the
    derived series summary is correctly EMPTY, because no series was dropped.
    An axis-level field cannot express this case at all.
    """
    panel = make_panel({SOLID: TRIPLE_A, DASHED: TRIPLE_B})
    every_row = make_grid(["Percent range", SOLID, DASHED], _rows_for(TRIPLE_A, TRIPLE_B))
    short = make_grid(["Percent range", SOLID, DASHED], _rows_for(TRIPLE_A, TRIPLE_B)[:2])

    whole, clipped = reconcile_grid(every_row, panel), reconcile_grid(short, panel)

    assert (whole.agreed, clipped.agreed) == (6, 4)
    assert clipped.unmatched_bins == () and clipped.unmatched_series == ()
    assert (whole.uncovered_count, clipped.uncovered_count) == (0, 2)
    assert clipped.unpublished_series == ()
    assert [u.bin_label for u in clipped.uncovered] == [RANGES[2], RANGES[2]]


def test_part_of_a_series_going_unaddressed_is_counted_per_cell() -> None:
    """A partial drop: the identities are counted, not the axis."""
    panel = make_panel({SOLID: {RANGES[0]: 1, RANGES[1]: 2}, DASHED: {RANGES[0]: 4}})
    grid = make_grid(["Percent range", SOLID], [[RANGES[0], "1"], [RANGES[1], "2"]])

    result = reconcile_grid(grid, panel)

    assert result.agreed == 2
    assert result.uncovered_count == 1
    assert [(u.series_name, u.bin_label) for u in result.uncovered] == [(DASHED, RANGES[0])]


def test_an_unaddressed_reading_is_split_by_whether_geometry_had_a_number() -> None:
    """Only the cell's own status changes; the identity is unaddressed either way."""
    read = make_panel({SOLID: {RANGES[0]: 1}, DASHED: {RANGES[0]: 4}})
    unread = make_panel({SOLID: {RANGES[0]: 1}, DASHED: {RANGES[0]: None}})
    grid = make_grid(["Percent range", SOLID], [[RANGES[0], "1"]])

    proven, silent = reconcile_grid(grid, read), reconcile_grid(grid, unread)

    assert proven.uncovered_count == silent.uncovered_count == 1
    assert (proven.uncovered_with_count, silent.uncovered_with_count) == (1, 0)
    assert (proven.uncovered[0].reader_count, silent.uncovered[0].reader_count) == (4, None)


def test_the_series_summary_is_derived_and_cannot_disagree_with_the_cells() -> None:
    """A partly-addressed series is not "unpublished"; a wholly missed one is."""
    panel = make_panel({SOLID: TRIPLE_A, DASHED: TRIPLE_B})
    row_short = make_grid(["Percent range", SOLID, DASHED], _rows_for(TRIPLE_A, TRIPLE_B)[:2])
    col_short = make_grid(["Percent range", SOLID], _rows_for(TRIPLE_A))

    partial, whole_series = reconcile_grid(row_short, panel), reconcile_grid(col_short, panel)

    assert partial.uncovered_count == 2 and partial.unpublished_series == ()
    assert whole_series.uncovered_count == 3 and whole_series.unpublished_series == (DASHED,)
    # Derived from the same set: every named series is wholly uncovered.
    missed = {u.series_name for u in whole_series.uncovered}
    assert set(whole_series.unpublished_series) <= missed


def test_uncovered_geometry_is_ranked_by_what_the_grid_published_not_by_volume() -> None:
    """Six uncovered beside nothing published is safer than three beside cells.

    The caption-headed grid matches no series, publishes no number, and leaves
    the whole panel uncovered -- recall loss, and nothing fabricated ships.
    The collapse that keeps a real name publishes three agreed cells beside
    three proven readings nobody consulted, which is a page that reads as
    checked and is half a chart.
    """
    panel = make_panel({SOLID: TRIPLE_A, DASHED: TRIPLE_B})
    caption = make_grid(["Percent range", "Number of participants"], _rows_for(TRIPLE_A))
    collapse = make_grid(["Percent range", SOLID], _rows_for(TRIPLE_A))

    quiet, shipping = reconcile_grid(caption, panel), reconcile_grid(collapse, panel)

    assert (quiet.uncovered_count, shipping.uncovered_count) == (6, 3)
    assert (quiet.published_cells, shipping.published_cells) == (0, 3)
    assert quiet.uncovered_beside_published == 0
    assert shipping.uncovered_beside_published == 3


# ---------------------------------------------------------------------------
# Verification is withheld when the comparison could not see everything
# ---------------------------------------------------------------------------


def test_renaming_the_series_that_would_contradict_earns_no_clean_bill() -> None:
    """The laundering route, closed. No cell verdict moves; the CLAIM does.

    Both grids publish the same agreeing column. The second renames the series
    that carries the disagreement, so its cells vanish from the comparison
    instead of contradicting -- and that must not read as a verified grid.
    """
    panel = make_panel({SOLID: {RANGES[0]: 3}, "December projections": {RANGES[0]: 9}})
    honest = make_grid(["Percent range", SOLID, "December projections"], [[RANGES[0], "3", "8"]])
    laundered = make_grid(["Percent range", SOLID, "Participants"], [[RANGES[0], "3", "8"]])

    caught, hidden = reconcile_grid(honest, panel), reconcile_grid(laundered, panel)

    assert caught.contradicted == 1 and caught.verified is False
    # The rename removed the cells, not the disagreement.
    assert hidden.contradicted == 0
    assert hidden.verified is False
    assert hidden.coverage_complete is False


def test_withholding_verification_never_moves_a_cell_or_adds_a_contradiction() -> None:
    panel = make_panel({SOLID: {RANGES[0]: 3}, "December projections": {RANGES[0]: 1}})
    covered = make_grid(["Percent range", SOLID, "December projections"], [[RANGES[0], "3", "1"]])
    partial = make_grid(["Percent range", SOLID], [[RANGES[0], "3"]])

    full, short = reconcile_grid(covered, panel), reconcile_grid(partial, panel)

    assert full.verified is True and short.verified is False
    assert short.contradicted == 0
    # The cell both grids share is judged identically, and still publishes.
    assert [(c.status, c.published) for c in short.cells] == [(AGREED, 3)]


def test_a_refused_grid_is_not_verified_by_having_no_contradictions() -> None:
    """A refusal reached no verdict, so it corroborates nothing."""
    panel = make_panel({SOLID: {RANGES[0]: 3}})
    refused = reconcile_grid(make_grid(["Quarter", "Revenue"], [["Q1", "17"]]), panel)

    assert refused.refusal != ""
    assert refused.contradicted == 0
    assert refused.verified is False


def test_a_reconciliation_holding_no_verdict_at_all_is_not_verified() -> None:
    """The empty result must not read as a clean bill.

    ``reconcile_grid`` cannot reach this state -- a grid that is not refused
    always yields at least one verdict -- but ``GridReconciliation`` is Stage
    B's to construct and its defaults are ``cells=()``, ``refusal=""``. The
    guard is on the type, not on the pass, which is why it is pinned here
    rather than through a grid.
    """
    blank = GridReconciliation(
        page_num=1, region_index=1, table_index=1, orientation=BINS_IN_COLUMN
    )

    assert (blank.refusal, blank.cells, blank.contradicted) == ("", (), 0)
    assert blank.coverage_complete is True
    assert blank.verified is False


def test_a_refusal_overrides_any_verdicts_the_result_happens_to_carry() -> None:
    """A refusal is not merely "no cells": it disqualifies what is there.

    ``reconcile_grid`` never emits a refusal alongside verdicts -- a refused
    grid returns none -- so the two clauses overlap on everything the pass
    itself produces, and a guard written against a refused grid tests nothing
    about the refusal clause. ``GridReconciliation`` is Stage B's to construct,
    and a result carrying both must not read as a clean bill.
    """
    verdict = CellVerdict(
        bin_label=RANGES[0],
        series_name=SOLID,
        status=AGREED,
        model_text="3",
        model_count=3,
        reader_count=3,
    )
    clean = GridReconciliation(
        page_num=1, region_index=1, table_index=1, orientation=BINS_IN_COLUMN, cells=(verdict,)
    )
    disqualified = GridReconciliation(
        page_num=1,
        region_index=1,
        table_index=1,
        orientation=BINS_IN_COLUMN,
        cells=(verdict,),
        refusal="the panel repeats a series name",
    )

    assert clean.verified is True
    assert disqualified.contradicted == 0 and disqualified.coverage_complete is True
    assert disqualified.verified is False


# ---------------------------------------------------------------------------
# The panel's own identity axes are no more exempt than the grid's
# ---------------------------------------------------------------------------


def test_a_bin_repeated_within_one_read_series_refuses_the_panel() -> None:
    """Unguarded, the index keeps the LAST cell and judges the model on it."""
    grid = make_grid(["Percent range", SOLID], [[RANGES[0], "5"]])
    clean = make_panel({SOLID: {RANGES[0]: 5}})
    doubled = PanelReading(
        page_num=clean.page_num,
        region_index=clean.region_index,
        label=clean.label,
        bins=clean.bins,
        series=(
            SeriesReading(
                name=SOLID,
                style="solid_fill",
                presence=PRESENT,
                cells=(_cell(RANGES[0], 5), _cell(RANGES[0], 9)),
            ),
        ),
        calibration=clean.calibration,
        frame=clean.frame,
    )

    ok, refused = reconcile_grid(grid, clean), reconcile_grid(grid, doubled)

    assert [c.status for c in ok.cells] == [AGREED]
    assert refused.refusal and refused.cells == ()


# ---------------------------------------------------------------------------
# The orientation rule's precondition, recorded
# ---------------------------------------------------------------------------


def test_bins_that_could_head_either_axis_refuse_rather_than_transpose() -> None:
    """The precondition: no header cell carries one of the panel's own bins.

    Range bins cannot collide with a prose header, which is why the corpus
    never exercises this. Single-value bins can, and when both axes name the
    panel's bins equally the grid is refused -- never silently transposed.
    """
    ranges = make_panel({SOLID: {RANGES[0]: 1, RANGES[1]: 2}})
    single = make_panel({"16": {"16": 1, "17": 2}, "17": {"16": 3, "17": 4}}, bins=("16", "17"))

    safe = reconcile_grid(
        make_grid(["Percent range", SOLID], [[RANGES[0], "1"], [RANGES[1], "2"]]), ranges
    )
    collides = reconcile_grid(
        make_grid(["Bin", "16", "17"], [["16", "1", "2"], ["17", "3", "4"]]), single
    )

    assert safe.refusal == "" and safe.orientation == BINS_IN_COLUMN
    assert collides.refusal != "" and collides.cells == ()


def test_no_header_of_a_reconciled_grid_carries_one_of_the_panels_bins() -> None:
    """The property the corpus satisfies, stated as a check rather than a count.

    Membership in the panel's own bins is the filter -- prose canonicalises to
    itself, so "bins are ranges" is not the property. Where a header DOES carry
    a bin label, the tie above is what happens.
    """
    panel = make_panel({SOLID: {RANGES[0]: 1, RANGES[1]: 2}})
    grid = make_grid(["Percent range", SOLID], [[RANGES[0], "1"], [RANGES[1], "2"]])

    result = reconcile_grid(grid, panel)
    panel_bins = {_bin_key(b.label) for b in panel.bins}

    assert result.orientation == BINS_IN_COLUMN
    assert not [h for h in grid.data_headers if _bin_key(h) in panel_bins]
