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
    CONTRADICTED,
    NOT_A_COUNT,
    UNKNOWN_TO_GEOMETRY,
    reconcile_grid,
)

# The panel these fixtures are about: three printed percent-range bins and one
# series, which is the shape of the Fed SEP panels the ticket is about.
RANGES = ("0.13-0.37", "0.38-0.62", "0.63-0.87")


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
