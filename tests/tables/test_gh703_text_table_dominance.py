"""GH-703: A2's row-shortfall term must not floor a complete TEXT table.

A2's term (b) (``structure_check._truncated_row_shortfall``) reconciles the
candidate's numeric body-row count against the native table-shaped row count,
with ``row_shape_min`` derived from the candidate itself. On a text table --
a comparison box whose cells are sentences carrying zero or one number each --
that minimum collapses to 1, so every prose line on the page that mentions a
figure counts as a native "table row", and a complete, ladder-accepted
candidate reads as a massive shortfall and is discarded.

Round 1 gated term (b) on the CANDIDATE's own row widths. Astra's review
falsified that: a numeric table truncated down to two legitimately sparse rows
loses its dominance together with the missing rows, term (b) abstains, and the
truncated reading wins selection over the complete one -- exactly what A2
exists to prevent. Round 2 asks the eligibility question on the NATIVE side,
which a model cannot truncate: does the page show recurring numeric column
lanes? A table's numerals recur in shared x-lanes down the page; prose figures
scatter.

Difference pins below monkeypatch ``_native_page_has_column_lanes`` to a
constant ``True``, which restores the pre-#703 behaviour exactly (the gate is
the only change to term (b)), and assert the one outcome that flips.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from socr.core.manifest import _strict_grid_authored_pool, structure_class_grid_winner
from socr.tables import structure_check
from socr.tables.reconcile import raw_table_block_lines
from socr.tables.row_corroboration import numeric_body_rows, table_blocks
from socr.tables.structure_check import (
    DEFECT_TABLE_TRUNCATED,
    _final_row_truncated,
    _native_page_has_column_lanes,
    table_output_defect,
    table_truncated,
)

from test_structure_check_truncated import (  # noqa: I001  (pytest rootdir import)
    BULLETIN_P2_COMPLETE,
    _grid_reading_output,
    _strict_grid_output,
    _strict_page,
)


def _row_words(y: float, tokens: list[str]) -> list[tuple]:
    """Native words shaped like ``page.get_text("words")``, one baseline band."""
    words = []
    x = 0.0
    for tok in tokens:
        words.append((x, y, x + 8.0, y + 10.0, tok))
        x += 12.0
    return words


def _scattered_words(y: float, tokens: list[str], *, offset: float) -> list[tuple]:
    """A prose line: tokens laid out by their own widths from *offset*, so
    numerals land where the sentence happens to put them rather than in a
    shared lane. This is what separates prose from a table.
    """
    words = []
    x = offset
    for tok in tokens:
        width = 5.0 * len(tok)
        words.append((x, y, x + width, y + 10.0, tok))
        x += width + 4.0
    return words


# ---------------------------------------------------------------------------
# The lane rule itself
# ---------------------------------------------------------------------------


def test_aligned_numeric_grid_has_column_lanes() -> None:
    words: list[tuple] = []
    for i in range(6):
        words += _row_words(20.0 * i, [f"20{10 + i}", str(100 + i), str(200 + i)])
    assert _native_page_has_column_lanes(words) is True


def test_scattered_prose_figures_have_no_column_lanes() -> None:
    """Prose lines that each mention a figure: many numerals, no lane reuse."""
    lines = [
        ["Inflation", "rose", "to", "2.7%", "in", "the", "quarter"],
        ["Unemployment", "fell", "to", "4%", "by", "year", "end"],
        ["Average", "weekly", "hours", "worked", "were", "32"],
        ["Participation", "held", "just", "under", "63.7", "percent"],
        ["Productivity", "growth", "averaged", "0.25", "over", "the", "period"],
    ]
    words: list[tuple] = []
    for i, line in enumerate(lines):
        words += _scattered_words(20.0 * i, line, offset=float(3 * i))
    assert _native_page_has_column_lanes(words) is False


def test_no_words_has_no_column_lanes() -> None:
    assert _native_page_has_column_lanes([]) is False


# ---------------------------------------------------------------------------
# Hermetic text table
# ---------------------------------------------------------------------------

# A two-column comparison box, cells are sentences. Two of the eight body rows
# carry exactly one number; the rest carry none. This is the BoE Table 3.B
# shape, reduced.
TEXT_TABLE_MD = (
    "| Developments anticipated in February | Developments now anticipated |\n"
    "| :--- | :--- |\n"
    "| **Unemployment** | **Revised down slightly** |\n"
    "| Unemployment rate to remain around four percent. | Unemployment rate to fall to 4% "
    "by the end of the year. |\n"
    "| **Participation** | **Broadly unchanged** |\n"
    "| Participation rate to remain just above the recent average. | Participation rate to "
    "remain just under that average. |\n"
    "| **Average hours** | **Broadly unchanged** |\n"
    "| Average weekly hours worked to remain around 32. | Average weekly hours worked to "
    "fall slightly. |\n"
    "| **Productivity** | **Broadly unchanged** |\n"
    "| Quarterly hourly labour productivity growth to average just over a quarter of a "
    "percent. | Unchanged from February. |\n"
)

# The page around that box: prose lines, most of which mention exactly one
# figure, laid out as running text so the numerals do not share x-lanes. At
# row_shape_min == 1 every one of these bands counts as a native
# "table-shaped row" -- which is the whole defect.
TEXT_TABLE_WORDS: list[tuple] = []
for _i, _line in enumerate(
    [
        ["Inflation", "Report", "May", "2018", "Section", "3"],
        ["Chart", "3.7", "Productivity", "growth", "remains", "subdued"],
        ["Percentage", "changes", "on", "a", "year", "earlier"],
        ["Output", "per", "hour", "since", "2002"],
        ["Sources:", "ONS", "and", "Bank", "calculations", "2018"],
        ["Unemployment", "rate", "to", "fall", "to", "4%"],
        ["Average", "weekly", "hours", "worked", "around", "32"],
        ["For", "more", "details", "see", "Tenreyro", "2018"],
        ["De-globalisation", "and", "inflation", "Carney", "2017"],
        ["Productivity", "growth", "in", "the", "year", "2016"],
        ["Business", "investment", "growth", "in", "2015"],
        ["Total", "factor", "productivity", "since", "2014"],
    ]
):
    TEXT_TABLE_WORDS += _scattered_words(10.0 + _i * 20.0, _line, offset=float(2 * _i))


def test_text_table_is_not_truncated() -> None:
    """A complete text table must survive the shipping gate."""
    candidate_rows = [
        row for rows in table_blocks(TEXT_TABLE_MD) for row in numeric_body_rows(rows) if row
    ]
    assert candidate_rows, "fixture must have numeric body rows for term (b) to be reachable"

    assert _native_page_has_column_lanes(TEXT_TABLE_WORDS) is False
    assert table_truncated(TEXT_TABLE_MD, TEXT_TABLE_WORDS) is False
    assert table_output_defect(TEXT_TABLE_MD, TEXT_TABLE_WORDS) != DEFECT_TABLE_TRUNCATED


def test_text_table_difference_pin_term_b_ungated_vs_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same page, same words, same term (a): the ONLY thing that changes is
    whether term (b)'s native-lane gate is consulted.
    """
    gated = table_truncated(TEXT_TABLE_MD, TEXT_TABLE_WORDS)

    monkeypatch.setattr(structure_check, "_native_page_has_column_lanes", lambda words: True)
    ungated = table_truncated(TEXT_TABLE_MD, TEXT_TABLE_WORDS)

    assert (ungated, gated) == (True, False)
    # and term (a) is not what changed -- it never fired on this page
    assert not any(_final_row_truncated(block) for block in raw_table_block_lines(TEXT_TABLE_MD))


def test_numeric_table_shortfall_is_unchanged_by_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other side of the difference pin: on a page with column lanes the
    gate is inert -- gated and ungated verdicts are identical, truncated and
    complete alike.
    """
    rows = [(2000 + i, float(100 + i), float(200 + i)) for i in range(20)]
    header = "| Year | A | B |\n|---|---|---|\n"
    complete_md = header + "".join(f"| {y} | {a} | {b} |\n" for y, a, b in rows)
    truncated_md = header + "".join(f"| {y} | {a} | {b} |\n" for y, a, b in rows[:-2])
    words: list[tuple] = []
    for i, (year, a, b) in enumerate(rows):
        words += _row_words(10.0 + i * 20.0, [str(year), str(a), str(b)])

    gated = (table_truncated(complete_md, words), table_truncated(truncated_md, words))
    monkeypatch.setattr(structure_check, "_native_page_has_column_lanes", lambda words: True)
    ungated = (table_truncated(complete_md, words), table_truncated(truncated_md, words))

    assert gated == ungated == (False, True)


# ---------------------------------------------------------------------------
# Astra's round-1 counterexample: a numeric table truncated to sparse rows
# ---------------------------------------------------------------------------


def _sparse_prefix_fixture() -> tuple[str, str, list[tuple]]:
    """A numeric table with two legitimate one-number rows followed by
    eighteen dense ones, and the same table truncated to only those two sparse
    rows. Round 1's candidate-side dominance test called the truncated reading
    a text table; the native words are an aligned grid throughout.
    """
    rows = [("Opening", "17", ""), ("Closing", "19", "")] + [
        (f"Item{i}", str(100 + i), str(200 + i)) for i in range(18)
    ]
    header = "| Item | A | B |\n| --- | --- | --- |\n"
    complete = header + "".join("| " + " | ".join(row) + " |\n" for row in rows)
    truncated = header + "".join("| " + " | ".join(row) + " |\n" for row in rows[:2])
    words: list[tuple] = []
    for i, row in enumerate(rows):
        words += _row_words(20.0 * i, [tok for tok in row if tok])
    return complete, truncated, words


def test_numeric_table_sparse_prefix_is_still_truncated() -> None:
    """Astra reproducer 1: the surviving rows are sparse, the page is not."""
    complete, truncated, words = _sparse_prefix_fixture()

    surviving = [row for rows in table_blocks(truncated) for row in numeric_body_rows(rows) if row]
    assert all(len(row) == 1 for row in surviving), "the truncated rows must look like prose rows"
    assert _native_page_has_column_lanes(words) is True

    assert table_truncated(complete, words) is False
    assert table_truncated(truncated, words) is True


def test_sparse_prefix_truncation_not_kept_in_the_strict_pool() -> None:
    """Astra reproducer 2: it must still be dropped from S1's strict pool."""
    complete, truncated, words = _sparse_prefix_fixture()
    short = _strict_grid_output("qwen", truncated)
    full = _grid_reading_output("gemini", complete)
    page = _strict_page([short, full])
    page.native_words = words

    assert short not in _strict_grid_authored_pool(page)


def test_sparse_prefix_truncation_does_not_win_over_complete_reading() -> None:
    """Astra reproducer 3: the complete gemini reading still wins selection.

    Round 1 flipped this winner to the truncated qwen reading.
    """
    complete, truncated, words = _sparse_prefix_fixture()
    short = _strict_grid_output("qwen", truncated)
    full = _grid_reading_output("gemini", complete)
    page = _strict_page([short, full])
    page.native_words = words

    winner = structure_class_grid_winner(page)
    assert winner is not None
    assert winner.engine == "gemini"


def test_ecb_p2_rows_are_all_wide() -> None:
    """Astra's control: the ECB fixture cannot supply the sparse-prefix shape
    by row deletion alone, which is why the synthetic one above is the
    counterexample of record.
    """
    rows = [row for block in table_blocks(BULLETIN_P2_COMPLETE) for row in numeric_body_rows(block)]
    assert rows
    assert all(len(row) >= 2 for row in rows)


# ---------------------------------------------------------------------------
# The real BoE fixture (corpus-skipped)
# ---------------------------------------------------------------------------

_CENSUS = Path.home() / "Data/socr/census-boe-2026-09-10"
BOE_2018_PDF = _CENSUS / "in/boe-meetings-2018-scan-p28-30.pdf"
# The cached qwen candidate for p1 shipped in the census run: 23/23 of the
# page's numbers, 0 extras, 10 pipe rows, table_ladder_accepted -- and floored
# to 0/23 by candidate_truncated. See docs/log/2026-09-10_third-institution-census.md.
BOE_2018_P1_QWEN = (
    _CENSUS
    / "out/boe-meetings-2018-scan-p28-30/cache/c6"
    / "c6270478f9d5cd36d245a5663cd19285c35e1f104e0dc133a6d8e5d3ec90a098.json"
)


def _boe_p1() -> tuple[str, list]:
    import pymupdf

    markdown = json.loads(BOE_2018_P1_QWEN.read_text())["text"]
    with pymupdf.open(BOE_2018_PDF) as doc:
        words = doc[0].get_text("words")
    return markdown, list(words)


@pytest.mark.skipif(
    not (BOE_2018_PDF.exists() and BOE_2018_P1_QWEN.exists()),
    reason="BoE census corpus not present on this machine",
)
def test_real_boe_p1_text_table_ships() -> None:
    """The ticket's own page, on the real PDF and the real cached attempt."""
    markdown, words = _boe_p1()

    # grounding: this IS the complete candidate the ladder accepted
    assert "Table 3.B Monitoring the MPC's key judgements" in markdown
    assert "Unemployment rate to fall to 4% by the end of the year." in markdown

    candidate_rows = [
        row for rows in table_blocks(markdown) for row in numeric_body_rows(rows) if row
    ]
    assert candidate_rows == [("4%",), ("32.",)]
    assert _native_page_has_column_lanes(words) is False

    assert table_truncated(markdown, words) is False
    assert table_output_defect(markdown, words) != DEFECT_TABLE_TRUNCATED


@pytest.mark.skipif(
    not (BOE_2018_PDF.exists() and BOE_2018_P1_QWEN.exists()),
    reason="BoE census corpus not present on this machine",
)
def test_real_boe_p1_difference_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression, pinned as a difference on the real page: ungated term
    (b) truncates it (the census outcome, 0/23 shipped); gated, it does not.
    """
    markdown, words = _boe_p1()

    gated = table_truncated(markdown, words)
    monkeypatch.setattr(structure_check, "_native_page_has_column_lanes", lambda words: True)
    ungated = table_truncated(markdown, words)

    assert (ungated, gated) == (True, False)


# ---------------------------------------------------------------------------
# Astra's round-2 counterexample: one unrelated numeral bridging two columns
# ---------------------------------------------------------------------------
#
# Round 2 asked the detector its lane question with its own greedy adjacency
# clustering, in which an x position joins the running lane whenever it is
# within the tolerance of the PREVIOUS one. A single footnote value printed
# between two real columns is then inside the tolerance of both and chains
# them into one lane, the gate returns False, term (b) abstains, and the
# truncated candidate wins selection over the complete one. Round 3 seeds
# lanes on recurrence instead: a position that occurs once can join a lane but
# can never found or bridge one.


def _bridged_sparse_prefix() -> tuple[str, str, list[tuple]]:
    """``_sparse_prefix_fixture`` plus one unrelated footnote value.

    Astra's geometry verbatim: ``999`` at x0=18, x1=26, on its own band, whose
    left edge is 6pt from both column x0s (12 and 24) and whose right edge is
    6pt from both column x1s (20 and 32). Neither the table nor either
    candidate changes.
    """
    complete, truncated, words = _sparse_prefix_fixture()
    return (
        complete,
        truncated,
        words + [(0.0, 500.0, 8.0, 510.0, "Note"), (18.0, 500.0, 26.0, 510.0, "999")],
    )


def test_one_off_bridge_does_not_collapse_recurring_lanes() -> None:
    """The clustering difference itself, pinned on the two lane builders.

    Adjacency chains the two recurring columns through the one-off position;
    recurrence-seeded clustering keeps them apart because ``18`` never founds
    a lane and the two centres are further apart than the tolerance.
    """
    from socr.tables.reconstruct import _adjacent_lane_of, _seeded_lane_of

    _, _, words = _bridged_sparse_prefix()
    nums = [(w[0], round(w[1])) for w in words if w[4].replace(".", "").isdigit()]
    xs = sorted({x for x, _ in nums})

    adjacent = _adjacent_lane_of(xs)
    seeded = _seeded_lane_of(nums, xs)

    assert adjacent[12.0] == adjacent[24.0], "the bridge is what round 2 tripped over"
    assert seeded[12.0] != seeded[24.0]
    # the one-off position is absorbed, never a lane of its own
    assert seeded[18.0] in {seeded[12.0], seeded[24.0]}


def test_detector_entry_point_keeps_adjacency_clustering() -> None:
    """``has_numeric_columns``' own answer is unchanged: the new clustering is
    opt-in, so GH-248/GH-349's callers keep the behaviour they were measured on.
    """
    from socr.tables.reconstruct import has_recurring_numeric_columns

    _, _, words = _bridged_sparse_prefix()

    assert has_recurring_numeric_columns(words, 2) is False  # adjacency, as before
    assert has_recurring_numeric_columns(words, 2, seeded_lanes=True) is True


def test_one_off_numeric_bridge_does_not_disable_shortfall() -> None:
    """Astra reproducer 4: the bridged page is still a numeric table, so the
    two-row truncation is still caught.
    """
    _, truncated, bridged = _bridged_sparse_prefix()

    assert _native_page_has_column_lanes(bridged) is True
    assert table_truncated(truncated, bridged) is True


def test_bridge_does_not_restore_truncated_winner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Astra reproducer 5, as a difference: with the gate forced open and with
    the real gate, the complete gemini reading wins on the bridged page alike.
    """
    complete, truncated, bridged = _bridged_sparse_prefix()

    def _winner():
        page = _strict_page(
            [_strict_grid_output("qwen", truncated), _grid_reading_output("gemini", complete)]
        )
        page.native_words = bridged
        winner = structure_class_grid_winner(page)
        return None if winner is None else winner.engine

    gated = _winner()
    monkeypatch.setattr(structure_check, "_native_page_has_column_lanes", lambda words: True)
    ungated = _winner()

    assert (ungated, gated) == ("gemini", "gemini")


BOE_2003_PDF = Path.home() / "Data/socr/census-boe-2026-09-10/in/boe-meetings-2003-table-p15-17.pdf"


@pytest.mark.skipif(not BOE_2003_PDF.exists(), reason="BoE census corpus not present")
def test_real_boe_2003_pages_are_prose_not_tables() -> None:
    """Astra's coverage probe, with its premise measured.

    The three pages close the gate on both clusterings. That is not lost
    coverage on a numeric table: the pages are the Bank's narrative annex, and
    their 9/17/10 bands "at width two" are prose lines quoting two figures
    each. No band puts a numeral in two recurring lanes on either anchor, which
    is the shape term (b) needs to reconcile anything at all.
    """
    import pymupdf

    from socr.tables.reconstruct import has_recurring_numeric_columns

    with pymupdf.open(BOE_2003_PDF) as doc:
        pages = [(p.get_text(), list(p.get_text("words"))) for p in doc]

    assert "ANNEX:  SUMMARY OF DATA PRESENTED BY BANK STAFF" in pages[0][0]
    assert "|" not in pages[1][0], "no pipe table, and no ruled table to read as one"

    for text, words in pages:
        assert _native_page_has_column_lanes(words) is False
        assert has_recurring_numeric_columns(words, 2) is False


# ---------------------------------------------------------------------------
# Astra's round-3 counterexamples: seeding by neighbourhood, merging by tolerance
# ---------------------------------------------------------------------------
#
# Round 3 stopped a one-off position from CHAINING two columns but not from
# FOUNDING the lane that swallows both: seed recurrence counted the union of
# bands anywhere in the tolerance neighbourhood, so the bridge borrowed both
# columns' support and outranked each of them. And two genuine columns closer
# together than the tolerance were merged unconditionally, however many rows
# carried a cell in each. Round 4 qualifies a seed by its OWN occupancy and
# lets same-band co-occurrence override the tolerance.


def _bridge_with_complementary_row() -> tuple[str, str, list[tuple]]:
    """Astra's geometry: the sparse-prefix table plus a legitimate final row
    holding only the second column's value, then one unrelated numeral at
    x=18. The final row is what stops column 12's bands from subsuming column
    24's, so neighbourhood support makes the bridge the strongest seed.
    """
    complete, truncated, words = _sparse_prefix_fixture()
    complete += "| Final | | 999 |\n"
    words = words + [(0.0, 400.0, 8.0, 410.0, "Final"), (24.0, 400.0, 32.0, 410.0, "999")]
    return complete, truncated, words + [(18.0, 500.0, 26.0, 510.0, "777")]


def test_one_off_bridge_cannot_found_a_lane() -> None:
    """The seeding difference itself: a position occupying one band founds no
    lane however much support its neighbourhood has.
    """
    from socr.tables.reconstruct import _seeded_lane_of

    _, _, words = _bridge_with_complementary_row()
    nums = [(w[0], round(w[1])) for w in words if w[4].isdigit()]
    lanes = _seeded_lane_of(nums, sorted({x for x, _ in nums}))

    assert lanes[12.0] != lanes[24.0], "the two real columns must stay apart"
    assert lanes[18.0] in {lanes[12.0], lanes[24.0]}, "the bridge joins, never founds"


def test_one_off_bridge_with_complementary_sparse_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Astra reproducer 6: the complete gemini reading wins with the gate real
    and with it forced open alike.
    """
    complete, truncated, bridged = _bridge_with_complementary_row()

    def _winner():
        page = _strict_page(
            [_strict_grid_output("qwen", truncated), _grid_reading_output("gemini", complete)]
        )
        page.native_words = bridged
        winner = structure_class_grid_winner(page)
        return None if winner is None else winner.engine

    assert _native_page_has_column_lanes(bridged) is True
    gated = _winner()
    monkeypatch.setattr(structure_check, "_native_page_has_column_lanes", lambda words: True)
    assert (_winner(), gated) == ("gemini", "gemini")


def _tight_columns() -> tuple[str, str, list[tuple]]:
    """Astra's second geometry: the same table with its horizontal geometry
    uniformly scaled by 5/12, so the columns start 5pt apart -- inside
    ``_LANE_X_TOL_PT`` -- while the boxes still do not overlap.
    """
    complete, truncated, words = _sparse_prefix_fixture()
    return (
        complete,
        truncated,
        [(w[0] * 5 / 12, w[1], w[2] * 5 / 12, w[3], w[4]) for w in words],
    )


def test_co_occurrence_overrides_the_merge_tolerance() -> None:
    """Two recurring positions carrying cells of the SAME row are separate
    columns by direct evidence, whatever their x distance.
    """
    from socr.tables.reconstruct import _seeded_lane_of

    _, _, words = _tight_columns()
    nums = [(w[0], round(w[1])) for w in words if w[4].isdigit()]
    xs = sorted({x for x, _ in nums})
    lanes = _seeded_lane_of(nums, xs)

    left, right = 5.0, 10.0
    assert abs(right - left) < 6.0, "the fixture must sit inside the tolerance"
    assert lanes[left] != lanes[right]


def test_tight_disjoint_columns_keep_shortfall(monkeypatch: pytest.MonkeyPatch) -> None:
    """Astra reproducer 7: term (b) still catches the truncation at 5pt pitch."""
    complete, truncated, words = _tight_columns()

    def _winner():
        page = _strict_page(
            [_strict_grid_output("qwen", truncated), _grid_reading_output("gemini", complete)]
        )
        page.native_words = words
        winner = structure_class_grid_winner(page)
        return None if winner is None else winner.engine

    assert _native_page_has_column_lanes(words) is True
    assert table_truncated(truncated, words) is True
    gated = _winner()
    monkeypatch.setattr(structure_check, "_native_page_has_column_lanes", lambda words: True)
    assert (_winner(), gated) == ("gemini", "gemini")


def test_right_aligned_digit_widths_still_seed() -> None:
    """Astra's GH-349 control: a right-aligned column whose x0 moves with the
    digit count is still found, because the x1 anchor is attempted too.
    """
    from socr.tables.reconstruct import has_recurring_numeric_columns

    words = []
    for i, number in enumerate(["1", "22", "333", "4444", "55555"]):
        for right in (100.0, 200.0):
            words.append((right - 5.0 * len(number), 20.0 * i, right, 20.0 * i + 10.0, number))

    assert has_recurring_numeric_columns(words, 2, seeded_lanes=True) is True


@pytest.mark.skipif(not BOE_2018_PDF.exists(), reason="BoE census corpus not present")
def test_real_boe_2018_chart_pages_close_the_gate() -> None:
    """Round 4 flips these two pages from True to False. They are not tables.

    Pages 2 and 3 of the Inflation Report excerpt are prose sections carrying
    vector fan charts; their recurring numeric positions are y-axis tick
    labels (``180/160/140`` down one axis, ``90/80/70/60/50`` down another),
    each on a band of its own, and almost no band holds a cell in two of them.
    Round 3 read the axes as columns through neighbourhood support.
    """
    import pymupdf

    with pymupdf.open(BOE_2018_PDF) as doc:
        pages = [(p.get_text(), p.get_drawings(), list(p.get_text("words"))) for p in doc]

    for text, drawings, words in pages[1:]:
        assert drawings, "these pages carry vector charts"
        assert "|" not in text, "and no table"
        assert _native_page_has_column_lanes(words) is False


# ---------------------------------------------------------------------------
# Astra's round-4 counterexample: half-point jitter across the rounding boundary
# ---------------------------------------------------------------------------
#
# Round 4 required two positions to co-occur on _MIN_TABLE_ROWS bands before
# the tolerance would stop merging them. That count divides one column's
# evidence exactly the way the quantisation does: a column whose anchor jitters
# across the rounding boundary is two positions, and each half co-occurs with
# the neighbouring column on only half of the shared rows. Co-occurrence needs
# no count -- one band holding two distinct numerals already settles that one
# column cannot hold both, because a column holds one cell per row.


def _jittered_tight_columns(*, jitter: bool) -> list[tuple]:
    """Astra's geometry: six rows, the first two carrying only the first
    column, the last four dense. The first column sits at x=12.49 and its tight
    neighbour at x=17.49, with 2pt boxes. With *jitter*, alternate first-column
    cells move 0.02pt to 12.51 -- across the whole-point rounding boundary.
    """
    _, _, original = _sparse_prefix_fixture()
    words = []
    for x0, y0, x1, y1, text in [w for w in original if w[1] < 120]:
        if text.isdigit():
            if x0 == 12:
                x0 = 12.51 if jitter and int(y0 / 20) % 2 else 12.49
            else:
                x0 = 17.49
            x1 = x0 + 2
        words.append((x0, y0, x1, y1, text))
    return words


def test_jittered_column_halves_share_a_lane_with_its_neighbour_split() -> None:
    """The lane mapping itself: the two halves of the jittered column land in
    one lane, and the tight neighbour keeps its own.
    """
    from socr.tables.reconstruct import _seeded_lane_of

    words = _jittered_tight_columns(jitter=True)
    nums = [(w[0], round(w[1])) for w in words if w[4].isdigit()]
    lanes = _seeded_lane_of(nums, sorted({x for x, _ in nums}))

    assert lanes[12.49] == lanes[12.51], "one printed column, two rounded positions"
    assert lanes[17.49] != lanes[12.49], "the neighbour shares four rows, so it is a column"


def test_half_point_jitter_preserves_shortfall() -> None:
    """Astra reproducer 8, as a difference: the 0.02pt move must not change
    whether the truncation is caught.
    """
    _, truncated, _ = _sparse_prefix_fixture()

    fixed = _jittered_tight_columns(jitter=False)
    jittered = _jittered_tight_columns(jitter=True)

    assert _native_page_has_column_lanes(fixed) is True
    assert _native_page_has_column_lanes(jittered) is True
    assert table_truncated(truncated, fixed) == table_truncated(truncated, jittered) is True


def test_one_shared_band_is_enough_to_split_two_positions() -> None:
    """Co-occurrence carries no count of its own.

    Two recurring positions inside the merge tolerance that share a single
    band are already two columns; requiring the sharing to recur is what lost
    the jittered column above.
    """
    from socr.tables.reconstruct import _seeded_lane_of

    words: list[tuple] = []
    for i in range(4):  # four bands, both positions, but only one shared band
        words.append((10.0, 20.0 * i, 12.0, 20.0 * i + 10.0, str(i)))
    for i in range(4, 8):
        words.append((14.0, 20.0 * i, 16.0, 20.0 * i + 10.0, str(i)))
    words.append((14.0, 0.0, 16.0, 10.0, "9"))  # the single shared band

    nums = [(w[0], round(w[1])) for w in words]
    lanes = _seeded_lane_of(nums, sorted({x for x, _ in nums}))
    assert abs(14.0 - 10.0) < 6.0, "inside the merge tolerance"
    assert lanes[10.0] != lanes[14.0]


def test_citation_rows_on_a_text_page_are_a_known_scope_limitation() -> None:
    """DOCUMENTED LIMITATION, deliberately pinned rather than fixed (#703).

    The gate asks its question of the WHOLE page. Three ordinary numbered
    source citations -- markers at one x, years at another -- are three bands
    populating two recurring lanes, which is all the gate asks for, so a text
    table sharing the page with them is once again exposed to term (b). The
    citations are not part of the candidate's table, and page-wide alignment
    does not establish that they are.

    This predates the gate (the same citations qualified under every revision)
    and narrowing it needs the region of the candidate's own table, which this
    change does not have. Recorded so the limitation is visible and any future
    change to it is deliberate.
    """
    assert table_truncated(TEXT_TABLE_MD, TEXT_TABLE_WORDS) is False

    citations: list[tuple] = []
    for i in range(3):
        y = 400.0 + 20.0 * i
        citations += [
            (50.0, y, 55.0, y + 10.0, str(i + 1)),
            (65.0, y, 140.0, y + 10.0, "Source citation"),
            (180.0, y, 200.0, y + 10.0, str(2016 + i)),
        ]

    assert _native_page_has_column_lanes(TEXT_TABLE_WORDS + citations) is True
    assert table_truncated(TEXT_TABLE_MD, TEXT_TABLE_WORDS + citations) is True
