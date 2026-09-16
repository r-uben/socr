"""GH-152: two tables printed side by side must not be merged into one region.

``rowize_from_word_list`` segmented rows by y across the FULL page width, so
two tables sitting at the same vertical extent in different x-bands were read
as single rows spanning both. Worse than losing structure: with incompatible
schemas (a different column count per side), the merged row attributes the
RIGHT table's values to the LEFT table's labels -- a wrong number shipping
under someone else's label, the misattribution class this repo forbids.

Fix: ``_detect_column_gutter`` looks for a single wide x-interval crossed by
no word anywhere in the word list (reusing ``ALIGNED_RUN_GAP_MAX_WORD_SPACES``
as the width yardstick, not a new constant). If found, the words are split at
the gutter and each side is rowized independently
(``_rowize_word_group``) -- but ONLY when both sides independently (a) yield
a valid table and (b) keep their own label column (``_has_row_labels``).
That second gate is the one that matters: without it, a single wide table's
own label-to-value gap, or a gap between grouped value columns, would also
look like a gutter and the split would tear ONE real table into two -- a new
misattribution of the same kind this ticket fixes. See
``test_wide_single_table_with_large_label_gap_is_not_split`` and
``test_wide_single_table_between_value_column_groups_is_not_split``.

Hermetic: pure synthetic word geometry (PyMuPDF ``get_text("words")`` tuple
shape), no PDF read, no provider, no corpus content.
"""

from __future__ import annotations

import pytest

from socr.tables.reconstruct import rowize_from_word_list

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

CHAR_W = 6.0
ROW_H = 14.0


def _w(x: float, y: float, text: str, block: int, line: int, word_no: int = 0) -> tuple:
    """A PyMuPDF word tuple with a text-length-proportional width.

    ``block``/``line`` mirror PyMuPDF's own grouping: two tables printed side
    by side are two different text blocks even when a row of each shares a
    y-coordinate, which is what ``_median_word_gap`` (GH-152's word-space
    yardstick) relies on to avoid contaminating the intra-row gap measurement
    with the inter-table gutter.
    """
    x1 = x + CHAR_W * len(text)
    return (x, y, x1, y + 9.0, text, block, line, word_no)


def _cells(md: str) -> list[list[str]]:
    """Parse a markdown table's rows, dropping the separator row and a blank
    HEADER row (``_grid_to_markdown`` emits an empty header when row 0 is
    itself a data row, so the first non-separator line here can be blank)."""
    rows = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in md.splitlines()
        if line.lstrip().startswith("|") and "---" not in line
    ]
    return [row for row in rows if any(c for c in row)]


def _two_tables(n_rows: int = 5) -> list:
    """Left: label + 2 numeric lanes. Right: label + 3 numeric lanes, offset
    far enough (x=350 vs. left's last lane ending ~214) to be a genuine
    column gutter, not a within-row seam."""
    words: list = []
    y = 100.0
    for i in range(n_rows):
        row_y = y + i * ROW_H
        words.append(_w(60.0, row_y, f"LeftLab{i}", block=0, line=i))
        words.append(_w(140.0, row_y, f"{i}.11", block=0, line=i))
        words.append(_w(190.0, row_y, f"{i}.22", block=0, line=i))
        words.append(_w(350.0, row_y, f"RightLab{i}", block=1, line=i))
        words.append(_w(430.0, row_y, f"{i}.33", block=1, line=i))
        words.append(_w(480.0, row_y, f"{i}.44", block=1, line=i))
        words.append(_w(530.0, row_y, f"{i}.55", block=1, line=i))
    return words


class TestSideBySideTablesAreSeparated:
    """The headline defect: misattribution, not just flattening."""

    def test_main_reproduces_the_misattribution(self):
        """Sanity check on the fixture itself, run through THIS worktree's
        code but with the split disabled -- i.e. exactly what shipped before
        GH-152. If this stops reproducing the bug, the fixture no longer
        proves what the test below claims to fix."""
        from socr.tables import reconstruct

        words = _two_tables()
        monkey = pytest.MonkeyPatch()
        monkey.setattr(reconstruct, "_detect_column_gutter", lambda _words: None)
        try:
            regions = rowize_from_word_list(words)
        finally:
            monkey.undo()

        assert len(regions) == 1, "fixture must merge into one region without the fix"
        grid = _cells(regions[0][1])
        labels = [row[0] for row in grid]
        assert all(lab.startswith("LeftLab") for lab in labels), labels
        assert not any("RightLab" in " ".join(row) for row in grid), (
            "RightLab tokens must be entirely absent pre-fix -- that is the defect"
        )
        # The misattribution itself: LeftLab0's row picks up a value that is
        # really RightLab0's ("0.33"), under LeftLab0's own label.
        assert grid[0][0] == "LeftLab0"
        assert "0.33" in grid[0], (
            "expected the merged row to misattribute RightLab0's first value "
            f"(0.33) onto LeftLab0's row: {grid[0]}"
        )

    def test_two_regions_each_with_correct_labels_and_values(self):
        words = _two_tables()
        regions = rowize_from_word_list(words)
        assert len(regions) == 2, f"expected two separate table regions, got {len(regions)}"

        by_first_label = {}
        for _rect, md in regions:
            grid = _cells(md)
            by_first_label[grid[0][0]] = grid

        left = by_first_label["LeftLab0"]
        right = by_first_label["RightLab0"]

        for i, row in enumerate(left):
            assert row[0] == f"LeftLab{i}"
            assert row[1:] == [f"{i}.11", f"{i}.22"], row

        for i, row in enumerate(right):
            assert row[0] == f"RightLab{i}"
            assert row[1:] == [f"{i}.33", f"{i}.44", f"{i}.55"], row

    def test_no_value_ships_under_the_wrong_tables_label(self):
        """The misattribution guarantee, stated directly rather than via the
        two grids' shape: no LEFT row may carry a RIGHT-table value, and vice
        versa."""
        words = _two_tables()
        regions = rowize_from_word_list(words)
        grids = [_cells(md) for _rect, md in regions]

        left_grid = next(g for g in grids if g[0][0].startswith("LeftLab"))
        right_grid = next(g for g in grids if g[0][0].startswith("RightLab"))

        left_values = {c for row in left_grid for c in row[1:] if c}
        right_values = {c for row in right_grid for c in row[1:] if c}
        # Left table's own values are i.11 / i.22; right's are i.33/i.44/i.55.
        assert not (left_values & right_values), (left_values, right_values)
        assert all(v.endswith(("11", "22")) for v in left_values), left_values
        assert all(v.endswith(("33", "44", "55")) for v in right_values), right_values


class TestFalsePositiveGuards:
    """A single wide table must never be torn in two."""

    def test_wide_single_table_with_large_label_gap_is_not_split(self):
        """A table's own label -> value gap can be wider than an ordinary
        intra-row seam. Splitting on it would strand the labels (0 numeric
        lanes, never a valid table) beside a bare, unlabelled numeric grid --
        exactly the misattribution-adjacent loss GH-152 exists to prevent,
        just relocated. Must stay ONE region."""
        words: list = []
        y = 100.0
        labels = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
        for i, lab in enumerate(labels):
            row_y = y + i * ROW_H
            words.append(_w(60.0, row_y, lab, block=0, line=i))
            # Wide gap: label ends ~60+30=90-ish, first value starts at 300.
            words.append(_w(300.0, row_y, f"{i}.11", block=0, line=i))
            words.append(_w(350.0, row_y, f"{i}.22", block=0, line=i))
            words.append(_w(400.0, row_y, f"{i}.33", block=0, line=i))

        regions = rowize_from_word_list(words)
        assert len(regions) == 1, (
            f"a single table's label gap must not be split into {len(regions)} regions"
        )
        grid = _cells(regions[0][1])
        assert [row[0] for row in grid] == labels
        for i, row in enumerate(grid):
            assert row[1:] == [f"{i}.11", f"{i}.22", f"{i}.33"]

    def test_wide_single_table_between_value_column_groups_is_not_split(self):
        """The gutter can also land BETWEEN two groups of value columns (e.g.
        a "Mean" group and an "SD" group) rather than at the label boundary.
        The right group would have no label words at all -- caught by
        ``_has_row_labels``, not by the numeric-lane gate the label-gap case
        above relies on."""
        words: list = []
        y = 100.0
        for i in range(5):
            row_y = y + i * ROW_H
            words.append(_w(60.0, row_y, f"Row{i}", block=0, line=i))
            words.append(_w(110.0, row_y, f"{i}.11", block=0, line=i))
            words.append(_w(150.0, row_y, f"{i}.22", block=0, line=i))
            # Wide gap between the two value-column groups.
            words.append(_w(350.0, row_y, f"{i}.33", block=0, line=i))
            words.append(_w(400.0, row_y, f"{i}.44", block=0, line=i))

        regions = rowize_from_word_list(words)
        assert len(regions) == 1, (
            f"a gap between two value-column groups of ONE table must not be "
            f"split into {len(regions)} regions"
        )
        grid = _cells(regions[0][1])
        for i, row in enumerate(grid):
            assert row[0] == f"Row{i}"
            assert row[1:] == [f"{i}.11", f"{i}.22", f"{i}.33", f"{i}.44"]


class TestSingleColumnByteIdentity:
    """Most pages are not two-column; the split path must be provably inert
    on them -- a DIFFERENCE-pin (split on vs. forced off), not an absolute
    string, per this repo's no-pinned-absolutes convention."""

    def test_no_gutter_found_is_byte_identical_to_split_disabled(self):
        """Direct proof that when no gutter is detected, the split code path
        contributes nothing: run the same prose-shaped word list through
        ``rowize_from_word_list`` normally, and again with
        ``_detect_column_gutter`` forced to always return ``None`` (the
        pre-GH-152 code path). The two must be identical."""
        from socr.tables import reconstruct

        # A single-column table with generous internal spacing: no x-band
        # candidate should ever appear crossed by no word (every column's
        # gap is well under a real gutter).
        words: list = []
        y = 100.0
        for i in range(6):
            row_y = y + i * ROW_H
            words.append(_w(60.0, row_y, f"Item{i}", block=0, line=i))
            words.append(_w(150.0, row_y, f"{i}.10", block=0, line=i))
            words.append(_w(200.0, row_y, f"{i}.20", block=0, line=i))
            words.append(_w(250.0, row_y, f"{i}.30", block=0, line=i))

        normal = rowize_from_word_list(words)

        monkey = pytest.MonkeyPatch()
        monkey.setattr(reconstruct, "_detect_column_gutter", lambda _words: None)
        try:
            forced_off = rowize_from_word_list(words)
        finally:
            monkey.undo()

        assert [md for _r, md in normal] == [md for _r, md in forced_off]
        assert [(r.x0, r.y0, r.x1, r.y1) for r, _m in normal] == [
            (r.x0, r.y0, r.x1, r.y1) for r, _m in forced_off
        ]

    def test_single_column_fixture_page_unaffected(self):
        """Same difference-pin, run against the real (non-synthetic) TR-0
        fixture PDF used across this module's other tests."""
        from pathlib import Path

        from socr.tables import reconstruct

        fixture = Path(__file__).parent / "fixtures" / "table_repair" / "ce_like_p4.pdf"
        doc = fitz.open(str(fixture))
        page = doc[0]
        words = list(page.get_text("words"))
        doc.close()

        normal = rowize_from_word_list(words)

        monkey = pytest.MonkeyPatch()
        monkey.setattr(reconstruct, "_detect_column_gutter", lambda _words: None)
        try:
            forced_off = rowize_from_word_list(words)
        finally:
            monkey.undo()

        assert [md for _r, md in normal] == [md for _r, md in forced_off]


class TestUnhandledLayouts:
    """Layouts GH-152 explicitly leaves unhandled must fail closed to
    today's single-merged-region behaviour, not guess."""

    def test_three_columns_falls_back_to_one_merged_region(self):
        """Three x-bands yield two gutters, not one -- ``_detect_column_gutter``
        refuses rather than picking a pairing."""
        words: list = []
        y = 100.0
        for i in range(5):
            row_y = y + i * ROW_H
            words.append(_w(60.0, row_y, f"A{i}", block=0, line=i))
            words.append(_w(110.0, row_y, f"{i}.11", block=0, line=i))
            words.append(_w(160.0, row_y, f"{i}.22", block=0, line=i))
            words.append(_w(300.0, row_y, f"B{i}", block=1, line=i))
            words.append(_w(350.0, row_y, f"{i}.33", block=1, line=i))
            words.append(_w(400.0, row_y, f"{i}.44", block=1, line=i))
            words.append(_w(550.0, row_y, f"C{i}", block=2, line=i))
            words.append(_w(600.0, row_y, f"{i}.55", block=2, line=i))
            words.append(_w(650.0, row_y, f"{i}.66", block=2, line=i))

        regions = rowize_from_word_list(words)
        assert len(regions) == 1, (
            f"three-column pages are unhandled; expected the unsplit fallback, "
            f"got {len(regions)} regions"
        )

    def test_full_width_caption_disables_the_split(self):
        """A word whose bbox spans the candidate gutter (a caption crossing
        both columns) rules the gutter out entirely -- fail closed to the
        merged behaviour, verified identical to the split-disabled path."""
        from socr.tables import reconstruct

        words = _two_tables()
        # A word bbox spanning the gutter interval (left ends ~214, right
        # starts at 350).
        words.append(
            _w(180.0, 100.0 + 5 * ROW_H, "ThisCaptionSpansTheWholeGutterX", block=2, line=0)
        )

        normal = rowize_from_word_list(words)

        monkey = pytest.MonkeyPatch()
        monkey.setattr(reconstruct, "_detect_column_gutter", lambda _words: None)
        try:
            forced_off = rowize_from_word_list(words)
        finally:
            monkey.undo()

        assert len(normal) == 1, "a spanning caption must disable the split"
        assert [md for _r, md in normal] == [md for _r, md in forced_off]
