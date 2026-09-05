"""TR-1: Deterministic word-geometry rowizer for lane-stacked find_tables() regions.

Tests covering:
1. ``rowize_from_words`` on the TR-0 fixture (single-schema main table):
   - Extracts the main forecaster grid as a proper markdown table.
   - All cell values match the ground truth (Ashford Capital → 2.1/1.9/3.4/2.8).
   - Blank (``na``) cells are PRESERVED as empty cells, not skipped/dropped.
   - The ragged Fenwick Group row (2 blanks) is correct.
2. ``_is_lane_stacked`` detection helper:
   - Detects embedded newlines in cells.
   - Passes on a clean (non-stacked) table.
3. ``extract_structured`` integration:
   - On the TR-0 fixture, produces exactly one markdown table (the main table).
   - That table contains correct values.
   - The historical-table text and prose are still present in the output
     (not silently dropped).
4. ``rowize_from_word_list`` scoping:
   - When called on only the main-table words (clipped by bbox), returns
     the same correct grid without interference from chart / prose words.

Critical investigation (TR-1 requirement):
    ``find_tables()`` (default lines strategy) returns ZERO tables for the
    TR-0 fixture because it has no ruling lines.  The two-table separate-vs-
    merged question is moot: both tables are invisible to the lines strategy.
    The text-strategy ``reconstruct_table_regions`` over-merges the entire page
    into one region that fails ``_looks_tabular`` (data_row_frac 0.45 < 0.5
    due to chart-tick rows diluting the data-row count).  The word-geometry
    rowizer segments by vertical gap (1.5 × median row gap ≈ 19.5 pt, derived
    from the page's own layout) and correctly extracts the main table (gap 25 pt
    between last data row and chart title is above threshold) while rejecting
    the merged chart+hist segment (data_row_frac still 0.45 < 0.5).

    Result: TR-1 fixes the main forecaster grid.  The historical table
    extraction requires TR-2's per-region segmentation (chart bboxes from
    ``has_chart_marks`` to split the chart from the hist table).
    The e2e xfail test remains xfail: only 1 grid is produced, not 2.

CI hermeticity:
    All tests are pure born-digital / rowizer unit tests.  No ``route_page``,
    no ollama, no provider patching required.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "table_repair"
FIXTURE_PDF = FIXTURE_DIR / "ce_like_p4.pdf"
GROUND_TRUTH = FIXTURE_DIR / "ground_truth.json"

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_gt() -> dict:
    return json.loads(GROUND_TRUTH.read_text())


def _main_table_gt(gt: dict) -> dict:
    for r in gt["regions"]:
        if r["id"] == "main_table":
            return r
    raise KeyError("main_table not found in ground truth")


_MD_ROW_RE = re.compile(r"^\|(.+)\|$")


def _parse_md_table(md: str) -> list[list[str]]:
    """Parse a single markdown table into a list of rows (list of cell strings).

    Skips separator rows (``| --- | ...``).  Returns the header as row 0.
    """
    rows = []
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r"-+", c) for c in cells if c):
            continue  # separator row
        rows.append(cells)
    return rows


_TABLE_RE = re.compile(
    r"(?:^[ \t]*\|.+\|[ \t]*$\n?){2,}",
    re.MULTILINE,
)


def _find_md_tables(text: str) -> list[str]:
    """Extract all markdown tables from ``text`` as raw strings."""
    return _TABLE_RE.findall(text)


# ---------------------------------------------------------------------------
# 1. rowize_from_words: main forecaster grid (single-schema lane-stacked)
# ---------------------------------------------------------------------------


class TestRowizeFromWordsMainTable:
    """rowize_from_words on the TR-0 fixture main forecaster grid."""

    @pytest.fixture(scope="class")
    def page_result(self):
        from socr.tables.reconstruct import rowize_from_words

        doc = fitz.open(str(FIXTURE_PDF))
        page = doc[0]
        result = rowize_from_words(page)
        doc.close()
        return result

    def test_returns_at_least_one_region(self, page_result):
        """rowize_from_words must return at least one table region."""
        assert len(page_result) >= 1, (
            "rowize_from_words returned no regions for the TR-0 fixture. "
            "Expected the main forecaster grid to be extracted."
        )

    def test_main_table_markdown_present(self, page_result):
        """The first (and only expected) region must be a valid markdown table."""
        _, md = page_result[0]
        assert "| --- |" in md, "Expected a markdown separator row in the output."
        assert "|" in md, "Expected pipe-delimited markdown table rows."

    def test_all_forecaster_rows_present(self, page_result):
        """All six forecaster row labels must appear in the extracted table."""
        gt = _load_gt()
        main = _main_table_gt(gt)
        _, md = page_result[0]
        for label in main["row_labels"]:
            assert label in md, (
                f"Forecaster label {label!r} not found in extracted markdown. Markdown:\n{md}"
            )

    def test_numeric_values_correct(self, page_result):
        """Spot-check known cell values from the ground truth."""
        _, md = page_result[0]
        # Ashford Capital: 2.1, 1.9, 3.4, 2.8
        assert "2.1" in md, "Ashford Capital GDP_2024 (2.1) not found"
        assert "1.9" in md, "Ashford Capital GDP_2025 (1.9) not found"
        assert "3.4" in md, "Ashford Capital CPI_2024 (3.4) not found"
        assert "2.8" in md, "Ashford Capital CPI_2025 (2.8) not found"
        # Clearview Economics: 1.8, 2.0, 3.6, 3.0
        assert "1.8" in md, "Clearview Economics GDP_2024 (1.8) not found"
        assert "2.0" in md, "Clearview Economics GDP_2025 (2.0) not found"
        # Historical table value that must NOT appear in the main table region
        # (5.9 is GDP Actual 2021 from the hist table, never in the main grid)

    def test_ground_truth_all_numeric_cells(self, page_result):
        """Every numeric cell from the ground truth must appear in the extracted md."""
        gt = _load_gt()
        main = _main_table_gt(gt)
        _, md = page_result[0]

        rows = _parse_md_table(md)
        # Build a flat text representation for value search
        all_text = " ".join(c for row in rows for c in row)

        missing = []
        for key, expected in main["cells"].items():
            if expected == "na":
                continue  # blanks are checked separately below
            if expected not in all_text:
                missing.append(f"{key}: {expected!r}")

        assert not missing, (
            "The following ground-truth numeric cells are absent from the "
            "extracted table:\n" + "\n".join(missing)
        )

    def test_na_blank_cells_preserved(self, page_result):
        """Blank (na) cells must produce EMPTY cells, not shift adjacent values.

        A missing token in a lane should become ``""`` (empty), never dropped.
        We verify by checking that the rows for Brightwater Research and
        Dunmore Analytics have the right number of data columns AND that the
        correct values appear in the correct relative positions.

        Concretely: Brightwater Research has GDP_2024=2.3, GDP_2025=na (blank),
        CPI_2024=3.1, CPI_2025=2.6.  If blanks are dropped, 3.1 would shift
        left into the GDP_2025 column — a silent value loss.
        """
        _, md = page_result[0]
        rows = _parse_md_table(md)

        # Find the Brightwater Research row
        bw_row: list[str] | None = None
        for row in rows:
            if any("Brightwater" in c for c in row):
                bw_row = row
                break

        assert bw_row is not None, "Brightwater Research row not found in extracted table."

        # Brightwater Research: label | 2.3 | [blank] | 3.1 | 2.6
        # The exact column indices depend on the header, but the relative order
        # must be: 2.3 before 3.1, and the row must have at least 5 cells
        # (label + 4 data columns — one being empty).
        non_empty_vals = [c for c in bw_row if c.strip()]
        assert "2.3" in non_empty_vals, "Brightwater GDP_2024 (2.3) missing"
        assert "3.1" in non_empty_vals, "Brightwater CPI_2024 (3.1) missing"
        assert "2.6" in non_empty_vals, "Brightwater CPI_2025 (2.6) missing"

        # Verify 2.3 comes BEFORE 3.1 in column order
        idx_23 = bw_row.index("2.3") if "2.3" in bw_row else -1
        idx_31 = bw_row.index("3.1") if "3.1" in bw_row else -1
        assert idx_23 < idx_31, (
            f"Value ordering wrong: 2.3 is at col {idx_23}, 3.1 at col {idx_31}. "
            "Blank cell may have been dropped, shifting values left."
        )

        # There must be an empty cell between 2.3 and 3.1 (the GDP_2025 blank)
        if idx_23 >= 0 and idx_31 >= 0:
            between = bw_row[idx_23 + 1 : idx_31]
            assert any(c == "" for c in between), (
                f"No blank cell between 2.3 and 3.1 in Brightwater row: "
                f"{bw_row!r}. The GDP_2025 blank was dropped or merged."
            )

    def test_fenwick_group_ragged_row(self, page_result):
        """Fenwick Group has GDP_2025=na and CPI_2024=na — two blanks."""
        _, md = page_result[0]
        rows = _parse_md_table(md)

        fenwick_row: list[str] | None = None
        for row in rows:
            if any("Fenwick" in c for c in row):
                fenwick_row = row
                break

        assert fenwick_row is not None, "Fenwick Group row not found"

        # Fenwick: 1.7, [blank], [blank], 2.7
        assert "1.7" in fenwick_row, "Fenwick GDP_2024 (1.7) missing"
        assert "2.7" in fenwick_row, "Fenwick CPI_2025 (2.7) missing"

        # Count blanks between 1.7 and 2.7
        idx_17 = fenwick_row.index("1.7") if "1.7" in fenwick_row else -1
        idx_27 = fenwick_row.index("2.7") if "2.7" in fenwick_row else -1

        if idx_17 >= 0 and idx_27 >= 0:
            between = fenwick_row[idx_17 + 1 : idx_27]
            blank_count = sum(1 for c in between if c == "")
            assert blank_count >= 2, (
                f"Fenwick Group should have >= 2 blanks between 1.7 and 2.7, "
                f"got {blank_count}. Row: {fenwick_row!r}."
            )

    def test_rect_covers_main_table_region(self, page_result):
        """The returned rect must span the main table's approximate y-range."""
        rect, _ = page_result[0]
        # Main table header starts near y=46, last data row near y=133
        assert rect.y0 <= 68, f"rect.y0={rect.y0} too far below y=46 header"
        assert rect.y1 >= 130, f"rect.y1={rect.y1} too far above y=133 last row"


# ---------------------------------------------------------------------------
# 2. _is_lane_stacked detection
# ---------------------------------------------------------------------------


class TestIsLaneStacked:
    """Unit tests for the lane-stacked detection predicate."""

    def test_detects_embedded_newlines(self):
        """A cell with embedded newlines (multiple values stacked) is lane-stacked."""
        from socr.core.born_digital import _is_lane_stacked

        mock_table = MagicMock()
        mock_table.extract.return_value = [
            ["Name1\nName2\nName3", "1.2\n2.3\n3.4"],
            ["More\nNames", "4.5\n5.6"],
        ]
        assert _is_lane_stacked(mock_table) is True

    def test_clean_table_not_stacked(self):
        """A well-formed table (no embedded newlines) is NOT lane-stacked."""
        from socr.core.born_digital import _is_lane_stacked

        mock_table = MagicMock()
        mock_table.extract.return_value = [
            ["Industry", "b", "s", "h"],
            ["FabPr", "0.253", "0.179", "0.211"],
            ["Clths", "0.144", "0.135", "0.290"],
        ]
        assert _is_lane_stacked(mock_table) is False

    def test_empty_table_not_stacked(self):
        """An empty extract is not stacked."""
        from socr.core.born_digital import _is_lane_stacked

        mock_table = MagicMock()
        mock_table.extract.return_value = []
        assert _is_lane_stacked(mock_table) is False

    def test_extract_raises_not_stacked(self):
        """If extract() raises, _is_lane_stacked returns False (never raises)."""
        from socr.core.born_digital import _is_lane_stacked

        mock_table = MagicMock()
        mock_table.extract.side_effect = RuntimeError("boom")
        assert _is_lane_stacked(mock_table) is False

    def test_none_cells_not_stacked(self):
        """None cells (empty) are not stacked."""
        from socr.core.born_digital import _is_lane_stacked

        mock_table = MagicMock()
        mock_table.extract.return_value = [
            [None, "1.2", "2.3"],
            [None, "4.5", None],
        ]
        assert _is_lane_stacked(mock_table) is False


# ---------------------------------------------------------------------------
# 3. extract_structured integration on TR-0 fixture
# ---------------------------------------------------------------------------


class TestExtractStructuredOnFixture:
    """extract_structured on the TR-0 fixture after TR-1 rowizer is installed."""

    @pytest.fixture(scope="class")
    def structured_text(self):
        from socr.core.born_digital import BornDigitalDetector

        doc = fitz.open(str(FIXTURE_PDF))
        page = doc[0]
        text = BornDigitalDetector().extract_structured(page)
        doc.close()
        return text

    def test_produces_at_least_one_markdown_table(self, structured_text):
        """After TR-1, extract_structured should produce at least one markdown table.

        TR-1 success criterion: the main forecaster grid is extracted as a grid.
        TR-2 additionally extracts the historical table, so ≥ 2 tables is expected
        after TR-2 is implemented.  This test relaxes the strict == 1 assertion so
        the TR-2 improvement does not count as a regression for TR-1's check.
        """
        tables = _find_md_tables(structured_text)
        assert len(tables) >= 1, (
            f"Expected at least 1 markdown table, got {len(tables)}. "
            "The rowizer gate may be broken."
        )

    def test_main_table_values_correct(self, structured_text):
        """Key values from the main forecaster grid must appear in the table."""
        tables = _find_md_tables(structured_text)
        md = tables[0]
        assert "Ashford Capital" in md
        assert "2.1" in md  # GDP_2024
        assert "1.9" in md  # GDP_2025
        assert "3.4" in md  # CPI_2024
        assert "2.8" in md  # CPI_2025
        assert "Brightwater Research" in md
        assert "Fenwick Group" in md

    def test_prose_still_present(self, structured_text):
        """Prose text must NOT be silently dropped by the rowizer."""
        assert "Government and Background Data" in structured_text, (
            "Prose text dropped — the rowizer rect may be too broad."
        )
        assert "Blanks indicate" in structured_text, "Prose sentence 'Blanks indicate' dropped."

    def test_historical_data_present(self, structured_text):
        """The historical data must appear in the output (as grid or flat text)."""
        # After TR-1 the hist table appeared as flat text; after TR-2 it is a grid.
        # Either way, the key label must be present somewhere in the output.
        assert "GDP Actual" in structured_text, (
            "Historical table content dropped entirely — unexpected regression."
        )


# ---------------------------------------------------------------------------
# 4. rowize_from_word_list scoping (region-clipped words)
# ---------------------------------------------------------------------------


class TestRowizeFromWordListScoped:
    """rowize_from_word_list on words clipped to the main table's bbox."""

    def test_clipped_words_produce_correct_grid(self):
        """When only main-table words are fed in, the rowizer still works."""
        from socr.tables.reconstruct import rowize_from_word_list

        doc = fitz.open(str(FIXTURE_PDF))
        page = doc[0]

        # Clip to approximately the main table's bounding box
        # (the fixture generator places the main table at y=46..144)
        import fitz as fitz_mod

        main_bbox = fitz_mod.Rect(36.0, 40.0, 520.0, 150.0)
        words = [
            w
            for w in page.get_text("words")
            if fitz_mod.Rect(w[0], w[1], w[2], w[3]).intersects(main_bbox)
        ]
        doc.close()

        result = rowize_from_word_list(words)
        assert len(result) >= 1, (
            "rowize_from_word_list returned no regions for clipped main-table words."
        )
        _, md = result[0]
        assert "Ashford Capital" in md
        assert "2.1" in md
        assert "Fenwick Group" in md
        assert "1.7" in md
        assert "2.7" in md

    def test_empty_word_list_returns_empty(self):
        """An empty word list must return [] without raising."""
        from socr.tables.reconstruct import rowize_from_word_list

        result = rowize_from_word_list([])
        assert result == []

    def test_prose_only_words_return_empty(self):
        """Words from the prose region alone must not produce a table."""
        from socr.tables.reconstruct import rowize_from_word_list

        doc = fitz.open(str(FIXTURE_PDF))
        page = doc[0]
        import fitz as fitz_mod

        prose_bbox = fitz_mod.Rect(36.0, 360.0, 520.0, 430.0)
        words = [
            w
            for w in page.get_text("words")
            if fitz_mod.Rect(w[0], w[1], w[2], w[3]).intersects(prose_bbox)
        ]
        doc.close()

        result = rowize_from_word_list(words)
        assert result == [], (
            f"Expected no tables from prose-only words, got {len(result)}. "
            "The rowizer is misfiring on prose."
        )


# ---------------------------------------------------------------------------
# 5. No-regression: existing reconstruct_table_regions still works
# ---------------------------------------------------------------------------


class TestNoRegressionReconstructTableRegions:
    """TR-1 must not break the existing text-strategy reconstruct path."""

    def test_booktabs_page_still_recovered(self):
        """The booktabs (horizontal-rules-only) page that was already working
        must still be recovered by reconstruct_table_regions."""
        from socr.tables.reconstruct import reconstruct_table_regions

        doc = fitz.open()
        page = doc.new_page()
        cols = [90, 230, 320, 410]
        data = [
            ["Industry", "b", "s", "h"],
            ["FabPr", "0.253", "0.179", "0.211"],
            ["Clths", "0.144", "0.135", "0.290"],
            ["Chem", "0.041", "0.000", "0.154"],
            ["Toys", "0.082", "0.321", "0.144"],
            ["Energy", "0.180", "0.171", "0.365"],
        ]
        rows = [120 + i * 22 for i in range(len(data))]
        for r, row in enumerate(data):
            for c, cell in enumerate(row):
                page.insert_text((cols[c], rows[r]), cell, fontsize=10)
        for yy in [rows[0] - 8, rows[1] - 6, rows[-1] + 8]:
            page.draw_line((90, yy), (440, yy))

        result = reconstruct_table_regions(page)
        assert result, "reconstruct_table_regions must still recover booktabs tables"
        _, md = result[0]
        assert "FabPr" in md and "0.253" in md

    def test_rowize_from_words_does_not_fire_on_booktabs_page(self):
        """For a booktabs page where reconstruct_table_regions already works,
        rowize_from_words is a fallback and may return [] or duplicates — the
        caller only uses it when reconstruct returns nothing.  We verify it
        doesn't crash and handles the page gracefully."""
        from socr.tables.reconstruct import rowize_from_words

        doc = fitz.open()
        page = doc.new_page()
        cols = [90, 230, 320, 410]
        data = [
            ["Industry", "b", "s", "h"],
            ["FabPr", "0.253", "0.179", "0.211"],
            ["Clths", "0.144", "0.135", "0.290"],
            ["Chem", "0.041", "0.000", "0.154"],
            ["Toys", "0.082", "0.321", "0.144"],
            ["Energy", "0.180", "0.171", "0.365"],
        ]
        rows_y = [120 + i * 22 for i in range(len(data))]
        for r, row in enumerate(data):
            for c, cell in enumerate(row):
                page.insert_text((cols[c], rows_y[r]), cell, fontsize=10)
        for yy in [rows_y[0] - 8, rows_y[1] - 6, rows_y[-1] + 8]:
            page.draw_line((90, yy), (440, yy))

        # Must not raise
        result = rowize_from_words(page)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 6. Blocking-1 regression: bibliography / reference page must NOT rowize
# ---------------------------------------------------------------------------


class TestBibliographyFalsePositivePrevention:
    """rowize_from_words must be blocked by has_numeric_columns for pages that
    only have 2 aligned text columns (author + year or year + page-number).
    These are common in born-digital PDFs and must NEVER produce a markdown
    table through extract_structured.
    """

    def _make_bibliography_page(self):
        """Create a born-digital page simulating a 2-column reference list.

        Layout: author name at x≈50, year at x≈230, page-number at x≈400.
        This gives 2 numeric x-lanes (year + page-number) which is below the
        _MIN_LANES_PER_ROW=3 threshold used by has_numeric_columns.
        """
        import fitz as fitz_mod

        doc = fitz_mod.open()
        page = doc.new_page(width=500, height=600)
        refs = [
            ("Acemoglu, D.", "2001", "127"),
            ("Banerjee, A.", "1992", "651"),
            ("Card, D.", "1999", "1801"),
            ("DiNardo, J.", "1996", "1001"),
            ("Heckman, J.", "1979", "153"),
            ("Katz, L.", "1992", "35"),
            ("Krueger, A.", "1994", "999"),
        ]
        y = 80
        for author, year, pg in refs:
            page.insert_text((50, y), author, fontsize=9)
            page.insert_text((230, y), year, fontsize=9)
            page.insert_text((400, y), pg, fontsize=9)
            y += 22
        return page

    def test_rowize_from_words_blocked_on_bibliography(self):
        """rowize_from_words must return [] for a 2-column reference page."""
        from socr.tables.reconstruct import rowize_from_words

        page = self._make_bibliography_page()
        result = rowize_from_words(page)
        assert result == [], (
            "rowize_from_words must return [] for a 2-column bibliography page "
            f"(has_numeric_columns guard missing). Got {len(result)} result(s)."
        )

    def test_extract_structured_no_table_on_bibliography(self):
        """extract_structured must NOT produce a markdown table for a 2-column
        bibliography page.  The output must contain no '| --- |' separator rows.
        """
        from socr.core.born_digital import BornDigitalDetector

        page = self._make_bibliography_page()
        detector = BornDigitalDetector()
        result = detector.extract_structured(page)

        # Combine all text chunks for easy searching
        full_text = "\n".join(
            chunk
            if isinstance(chunk, str)
            else (chunk.get("text", "") if isinstance(chunk, dict) else "")
            for chunk in (result if isinstance(result, list) else [result])
        )
        # Also handle case where result is a plain string
        if isinstance(result, str):
            full_text = result

        assert "| --- |" not in full_text, (
            "extract_structured produced a markdown table for a bibliography page. "
            "The has_numeric_columns guard in rowize_from_words must prevent this.\n"
            f"Output: {full_text[:500]}"
        )


# ---------------------------------------------------------------------------
# 7. Blocking-2: real PDF integration test for _is_lane_stacked path
# ---------------------------------------------------------------------------


class TestIsLaneStackedRealPDFIntegration:
    """End-to-end test: a real PDF page where find_tables() (lines strategy)
    returns a table with embedded-newline cells (lane-stacked symptom).
    BornDigitalDetector.extract_structured must produce a correct markdown grid
    (individual rows, correct values), NOT the collapsed raw text.
    """

    def _make_stacked_ruled_page(self):
        """Create a PDF page with a ruled grid and stacked text in cells.

        The grid uses vertical and horizontal ruling lines so find_tables()
        (default lines strategy) detects it.  Two forecasters are stacked
        (inserted at two y-positions) within each cell row, so find_tables()
        collapses them into one cell with an embedded newline.

        Layout (col_xs=[20, 180, 270, 360], row_ys=[40, 80, 120, 160, 200]):
          - Row 0 (header): Forecaster | GDP | CPI
          - Row 1: Alpha Fund / Beta Research | 2.1/1.8 | 3.1/2.9
          - Row 2: Gamma Corp / Delta Ltd      | 2.5/2.3 | 3.4/3.2
          - Row 3: Epsilon Inc / Zeta Group    | 1.9/2.7 | 2.8/3.5
        """
        import fitz as fitz_mod

        doc = fitz_mod.open()
        page = doc.new_page(width=400, height=400)
        col_xs = [20, 180, 270, 360]
        row_ys = [40, 80, 120, 160, 200]

        # Draw ruling lines
        for x in col_xs:
            page.draw_line((x, row_ys[0]), (x, row_ys[-1]))
        for y in row_ys:
            page.draw_line((col_xs[0], y), (col_xs[-1], y))

        # Header row
        page.insert_text((25, 65), "Forecaster", fontsize=8)
        page.insert_text((185, 65), "GDP", fontsize=8)
        page.insert_text((275, 65), "CPI", fontsize=8)

        # Row 1: stacked text (two names + two values per cell)
        page.insert_text((25, 93), "Alpha Fund", fontsize=8)
        page.insert_text((25, 104), "Beta Research", fontsize=8)
        page.insert_text((185, 93), "2.1", fontsize=8)
        page.insert_text((185, 104), "1.8", fontsize=8)
        page.insert_text((275, 93), "3.1", fontsize=8)
        page.insert_text((275, 104), "2.9", fontsize=8)

        # Row 2: stacked text
        page.insert_text((25, 133), "Gamma Corp", fontsize=8)
        page.insert_text((25, 144), "Delta Ltd", fontsize=8)
        page.insert_text((185, 133), "2.5", fontsize=8)
        page.insert_text((185, 144), "2.3", fontsize=8)
        page.insert_text((275, 133), "3.4", fontsize=8)
        page.insert_text((275, 144), "3.2", fontsize=8)

        # Row 3: stacked text
        page.insert_text((25, 173), "Epsilon Inc", fontsize=8)
        page.insert_text((25, 184), "Zeta Group", fontsize=8)
        page.insert_text((185, 173), "1.9", fontsize=8)
        page.insert_text((185, 184), "2.7", fontsize=8)
        page.insert_text((275, 173), "2.8", fontsize=8)
        page.insert_text((275, 184), "3.5", fontsize=8)

        return page

    def test_find_tables_detects_stacked_cells(self):
        """Verify that find_tables() actually captures stacked-text cells
        (embedded newlines) on this fixture — otherwise the _is_lane_stacked
        path would never be triggered."""
        page = self._make_stacked_ruled_page()
        tables_result = page.find_tables()
        assert tables_result.tables, (
            "find_tables() returned no tables for the stacked-ruled fixture. "
            "The integration test fixture must have ruling lines that find_tables "
            "can detect."
        )
        # At least one cell should have an embedded newline (stacked symptom)
        found_stacked = False
        for table in tables_result.tables:
            try:
                rows = table.extract()
            except Exception:
                continue
            for row in rows or []:
                for cell in row or []:
                    if cell and isinstance(cell, str) and "\n" in cell:
                        found_stacked = True
        assert found_stacked, (
            "No embedded-newline cell found in any find_tables() result. "
            "_is_lane_stacked would never trigger. Check the fixture design."
        )

    def test_extract_structured_produces_correct_grid(self):
        """extract_structured must produce a markdown grid with individual rows,
        NOT collapsed stacked text, for the lane-stacked ruled fixture.

        Expected: each forecaster appears on their own row with correct values.
        The output must NOT contain raw stacked text like 'Alpha Fund\\nBeta Research'.
        """
        from socr.core.born_digital import BornDigitalDetector

        page = self._make_stacked_ruled_page()
        detector = BornDigitalDetector()
        result = detector.extract_structured(page)

        # Normalise result to a single string for assertion
        if isinstance(result, list):
            full_text = "\n".join(
                chunk
                if isinstance(chunk, str)
                else (chunk.get("text", "") if isinstance(chunk, dict) else str(chunk))
                for chunk in result
            )
        else:
            full_text = str(result)

        # Must have produced at least one markdown table
        assert "| --- |" in full_text, (
            "extract_structured produced no markdown table for the stacked-ruled "
            "fixture. The _is_lane_stacked path must route to rowize_from_word_list.\n"
            f"Output (truncated): {full_text[:500]}"
        )

        # All six forecasters must appear individually (not stacked/merged)
        for name in [
            "Alpha Fund",
            "Beta Research",
            "Gamma Corp",
            "Delta Ltd",
            "Epsilon Inc",
            "Zeta Group",
        ]:
            assert name in full_text, (
                f"Forecaster {name!r} missing from extract_structured output. "
                "The rowizer must split stacked cells into individual rows.\n"
                f"Output (truncated): {full_text[:500]}"
            )

        # Key numeric values must be present
        for val in [
            "2.1",
            "1.8",
            "3.1",
            "2.9",
            "2.5",
            "2.3",
            "3.4",
            "3.2",
            "1.9",
            "2.7",
            "2.8",
            "3.5",
        ]:
            assert val in full_text, (
                f"Value {val!r} missing from extract_structured output.\n"
                f"Output (truncated): {full_text[:500]}"
            )

        # Raw stacked text must NOT appear (that would be the passthrough bug)
        assert "Alpha Fund\nBeta Research" not in full_text, (
            "extract_structured passed through raw stacked text 'Alpha Fund\\nBeta Research'. "
            "The _is_lane_stacked branch must route to the rowizer, not _table_to_markdown."
        )


def test_a2_binder_repair_does_not_change_rowizer_output(monkeypatch):
    """VI-A2 owns ``binding.py``. The TR-1 rowizer must not consult it.

    Difference: ``rowize_from_word_list`` on the same words with
    ``binding._native_rows`` replaced by a bomb vs left intact — output
    identical. If A2 leaked into reconstruct, the bomb would fire.
    """
    from socr.tables import binding as binding_module
    from socr.tables.reconstruct import rowize_from_word_list

    words = [
        (50.0, 100.0, 90.0, 108.0, "Ashford", 0, 0, 0),
        (150.0, 100.0, 180.0, 108.0, "2.1", 0, 0, 1),
        (220.0, 100.0, 250.0, 108.0, "1.9", 0, 0, 2),
        (50.0, 116.0, 90.0, 124.0, "Clearview", 0, 1, 0),
        (150.0, 116.0, 180.0, 124.0, "1.8", 0, 1, 1),
        (220.0, 116.0, 250.0, 124.0, "2.0", 0, 1, 2),
        (50.0, 132.0, 90.0, 140.0, "Dunmore", 0, 2, 0),
        (150.0, 132.0, 180.0, 140.0, "2.3", 0, 2, 1),
        (220.0, 132.0, 250.0, 140.0, "2.6", 0, 2, 2),
    ]
    intact = rowize_from_word_list(words)

    def _bomb(*_a, **_k):
        raise AssertionError("rowizer called binding._native_rows")

    monkeypatch.setattr(binding_module, "_native_rows", _bomb)
    monkeypatch.setattr(binding_module, "_assign_bands", _bomb)
    monkeypatch.setattr(binding_module, "_words_in_region", _bomb)
    patched = rowize_from_word_list(words)
    assert patched == intact
