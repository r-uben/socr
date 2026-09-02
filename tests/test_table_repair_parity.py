"""TR-0: License-clean CE-like fixture + cell-by-cell parity harness.

This module covers:
  1. Fixture sanity checks — the synthesized PDF is born-digital, has a text
     layer, and ``has_chart_marks`` returns True for the vector chart region.
  2. Ground-truth JSON integrity — every expected cell key and value is
     well-formed; ``na`` blanks and ragged rows are enumerated.
  3. ``assert_table_parity`` unit tests — the helper itself correctly reports
     exact mismatches on a known-good vs known-bad table. These tests pass NOW
     (no xfail).
  4. End-to-end agentic test — drives ``UnifiedPipeline.process()`` on the
     fixture and asserts full cell-parity for BOTH tables, chart as image-asset,
     and prose as prose, all in reading order.
     Marked ``@pytest.mark.xfail(strict=True)`` because TR-1…TR-3 are not yet
     implemented; the pipeline still collapses the whitespace-gutter tables.

CI hermeticity (critical):
  The end-to-end test patches ``_available_engines_for_agentic`` and
  ``route_page`` so no ollama / real provider is needed.  The born-digital
  path (native text extraction) is exercised, not the VLM path — which is
  fine: the fixture IS born-digital, so native extraction is the code path
  that should work once TR-1 fixes the rowizer gate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "table_repair"
FIXTURE_PDF = FIXTURE_DIR / "ce_like_p4.pdf"
GROUND_TRUTH = FIXTURE_DIR / "ground_truth.json"


# ---------------------------------------------------------------------------
# Helpers — load ground truth
# ---------------------------------------------------------------------------


def _load_gt() -> dict:
    return json.loads(GROUND_TRUTH.read_text())


def _region(gt: dict, region_id: str) -> dict:
    for r in gt["regions"]:
        if r["id"] == region_id:
            return r
    raise KeyError(f"region {region_id!r} not in ground truth")


# ---------------------------------------------------------------------------
# Core parity helper
# ---------------------------------------------------------------------------

_MD_TABLE_RE = re.compile(
    r"(?:^[ \t]*\|.+\|[ \t]*$\n?){2,}",
    re.MULTILINE,
)

_IMAGE_REF_RE = re.compile(r"!\[.*?\]\(.*?\)")


class ParityError(AssertionError):
    """Raised by ``assert_table_parity`` with a full per-cell diff."""


def _parse_markdown_tables(md: str) -> list[list[list[str]]]:
    """Parse all markdown tables from *md* into a list of grids.

    Each grid is a list of rows; each row is a list of cell strings (stripped,
    pipe-delimited columns).  The separator row (``| --- |``) is dropped.
    """
    tables = []
    for match in _MD_TABLE_RE.finditer(md):
        raw = match.group(0)
        rows = []
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            # Skip separator rows (e.g. | --- | --- |)
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.fullmatch(r"-+", c) for c in cells if c):
                continue
            rows.append(cells)
        if rows:
            tables.append(rows)
    return tables


def assert_table_parity(
    extracted_md: str,
    ground_truth: dict[str, Any],
) -> None:
    """Assert that *extracted_md* matches *ground_truth* cell-by-cell.

    Parameters
    ----------
    extracted_md:
        The full markdown string produced by the pipeline for the fixture page.
    ground_truth:
        The parsed ground-truth dict (from ``ground_truth.json``).

    Raises
    ------
    ParityError
        If any of the following conditions fail:
        - Each table region is a SEPARATE markdown grid (not merged).
        - Cell values match for every ``row_label|column`` key.
        - The chart region has no transcribed data values (no markdown table
          overlapping its column set, no raw numbers in a table-looking block).
        - The prose region appears as plain text (not a markdown table).
        - Regions appear in reading order (main_table → chart → historical_table
          → prose) in the document.

    Notes
    -----
    The helper reports ALL mismatching cells in one error, not just the first,
    so a caller can see the complete diff in one shot.
    """
    regions = ground_truth["regions"]
    table_regions = [r for r in regions if r["kind"] == "table"]
    chart_regions = [r for r in regions if r["kind"] == "chart"]
    prose_regions = [r for r in regions if r["kind"] == "text"]

    errors: list[str] = []

    # ------------------------------------------------------------------
    # 1. Count separate markdown tables
    # ------------------------------------------------------------------
    md_tables = _parse_markdown_tables(extracted_md)
    if len(md_tables) < len(table_regions):
        errors.append(
            f"SEPARATE-GRIDS: expected >= {len(table_regions)} separate markdown tables "
            f"(one per table region), got {len(md_tables)}. "
            f"Tables found: {len(md_tables)}. "
            "This usually means the tables are collapsed or merged."
        )
    # Upper-bound guard: over-segmentation (rowizer splits one table into many)
    # produces more grids than expected and may cause cell-parity mismatches by
    # scattering rows across fragmented tables.  We allow at most one extra table
    # per expected table region (e.g. a multi-header that splits on collapse).
    max_tables = len(table_regions) + 1
    if len(md_tables) > max_tables:
        errors.append(
            f"OVER-SEGMENTED: expected at most {max_tables} markdown tables "
            f"(len(table_regions)={len(table_regions)} + 1 tolerance), "
            f"got {len(md_tables)}. "
            "This usually means the rowizer over-split one table into multiple grids."
        )

    # ------------------------------------------------------------------
    # 2. Per-table cell parity
    # ------------------------------------------------------------------
    for t_idx, table_region in enumerate(table_regions):
        region_id = table_region["id"]
        expected_cells = table_region["cells"]  # {"{label}|{col}": value}
        columns = table_region["columns"]
        row_labels = table_region["row_labels"]

        # Try to find this table's grid among the parsed markdown tables.
        # Heuristic: the grid whose first data column matches at least one
        # expected row label is "this" table.
        matched_grid: list[list[str]] | None = None
        for grid in md_tables:
            if not grid:
                continue
            # Check if any row in the grid contains one of our expected labels
            for row in grid[1:]:  # skip header row
                row_text = " ".join(row)
                if any(lbl in row_text for lbl in row_labels):
                    matched_grid = grid
                    break
            if matched_grid is not None:
                break

        if matched_grid is None:
            errors.append(
                f"TABLE-MISSING [{region_id}]: could not find a markdown table "
                f"containing any of the expected row labels: {row_labels[:3]}…"
            )
            # Report all expected cells as missing
            for key, expected_val in expected_cells.items():
                errors.append(f"  MISSING [{region_id}] {key}: expected {expected_val!r}")
            continue

        # Build a lookup from the matched grid.
        # First row is the header; subsequent rows are data rows.
        if len(matched_grid) < 2:
            errors.append(
                f"TABLE-EMPTY [{region_id}]: matched grid has fewer than 2 rows "
                f"(need a header + at least 1 data row)."
            )
            continue

        header_row = matched_grid[0]
        # Map column names to grid column indices.
        # The first cell in each data row is the row label (forecaster name etc.)
        # Normalize underscores to spaces before comparing so "GDP_2024" matches
        # a header cell produced as "GDP 2024" by the multi-line-header collapser.
        col_idx: dict[str, int] = {}
        for col in columns:
            col_norm = col.replace("_", " ")
            for ci, hdr_cell in enumerate(header_row):
                hdr_norm = hdr_cell.replace("_", " ")
                if col_norm in hdr_norm or hdr_norm in col_norm:
                    col_idx[col] = ci
                    break

        # Map row labels to grid row indices.
        label_row_idx: dict[str, int] = {}
        for ri, row in enumerate(matched_grid[1:], start=1):
            row_text = " ".join(row)
            for lbl in row_labels:
                if lbl in row_text:
                    label_row_idx[lbl] = ri
                    break

        # Cell-by-cell comparison
        for key, expected_val in expected_cells.items():
            label, col = key.split("|", 1)
            ri = label_row_idx.get(label)
            ci = col_idx.get(col)
            if ri is None:
                errors.append(
                    f"ROW-MISSING [{region_id}] {key}: "
                    f"row label {label!r} not found in extracted table."
                )
                continue
            if ci is None:
                errors.append(
                    f"COL-MISSING [{region_id}] {key}: "
                    f"column {col!r} not found in extracted header row {header_row}."
                )
                continue
            grid_row = matched_grid[ri]
            if ci >= len(grid_row):
                errors.append(
                    f"CELL-MISSING [{region_id}] {key}: "
                    f"row {ri} has only {len(grid_row)} cells (need index {ci}); "
                    f"row content: {grid_row!r}."
                )
                continue
            actual_val = grid_row[ci].strip()
            # Normalise: treat empty string as "na"
            if actual_val == "":
                actual_val = "na"
            if actual_val != expected_val:
                errors.append(
                    f"CELL-MISMATCH [{region_id}] {key}: "
                    f"expected {expected_val!r}, got {actual_val!r}."
                )

    # ------------------------------------------------------------------
    # 3. Chart region — no transcribed data values
    # ------------------------------------------------------------------
    for chart_region in chart_regions:
        # Gather numeric tokens from the chart's column labels so we can
        # detect if any table in the markdown accidentally contains them.
        # For the fixture, the chart region has no expected cells ({}), so
        # we just assert no extra markdown table overlapping the chart exists.
        # Additionally, verify the chart is represented as an image ref.
        if not _IMAGE_REF_RE.search(extracted_md):
            errors.append(
                "CHART-NO-IMAGE-REF: the chart region must be represented as an "
                "image asset (e.g. ![chart](path/to/chart.png)) — no raw data values "
                "should be transcribed for this region. No image reference found in output."
            )

    # ------------------------------------------------------------------
    # 4. Prose region — appears as plain text (not a markdown table)
    # ------------------------------------------------------------------
    # We check that the prose keyword appears somewhere outside a table block.
    # Simple heuristic: strip all markdown table blocks, then check prose text.
    prose_keywords = ["Government", "Background Data", "official sources", "Blanks indicate"]
    stripped_md = _MD_TABLE_RE.sub("", extracted_md)
    for prose_region in prose_regions:
        desc = prose_region.get("description", "")
        found_prose = any(kw in stripped_md for kw in prose_keywords)
        if not found_prose:
            errors.append(
                f"PROSE-MISSING [{prose_region['id']}]: expected prose text "
                f"(e.g. {prose_keywords[0]!r}) outside markdown tables. "
                f"Description: {desc!r}."
            )

    # ------------------------------------------------------------------
    # 5. Reading order — main_table before chart before historical_table before prose
    # ------------------------------------------------------------------
    order_checks = [
        ("main_table", "Ashford Capital"),
        ("chart", None),  # image ref — positional check only
        ("historical_table", "GDP Actual"),
        ("prose", "Government"),
    ]
    positions: list[tuple[str, int]] = []
    for region_id, keyword in order_checks:
        if keyword is not None:
            pos = extracted_md.find(keyword)
        else:
            m = _IMAGE_REF_RE.search(extracted_md)
            pos = m.start() if m else -1
        positions.append((region_id, pos))

    for i in range(len(positions) - 1):
        rid_a, pos_a = positions[i]
        rid_b, pos_b = positions[i + 1]
        if pos_a == -1 or pos_b == -1:
            # Can't check order if one or both are absent (already reported above)
            continue
        if pos_a >= pos_b:
            errors.append(
                f"READING-ORDER: {rid_a!r} (pos {pos_a}) must appear before "
                f"{rid_b!r} (pos {pos_b}) in the extracted markdown."
            )

    if errors:
        bullet = "\n  - "
        raise ParityError(
            f"Parity check failed ({len(errors)} issue(s)):{bullet}" + bullet.join(errors)
        )


# ---------------------------------------------------------------------------
# Fixture sanity tests (pass NOW, no xfail)
# ---------------------------------------------------------------------------


fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")


class TestFixtureSanity:
    """Verify the synthesized fixture meets TR-0's physical requirements."""

    def test_pdf_exists(self):
        assert FIXTURE_PDF.exists(), f"Fixture PDF missing: {FIXTURE_PDF}"

    def test_ground_truth_exists(self):
        assert GROUND_TRUTH.exists(), f"Ground truth JSON missing: {GROUND_TRUTH}"

    def test_born_digital_text_layer(self):
        """The fixture PDF must have a real text layer (born-digital)."""
        doc = fitz.open(str(FIXTURE_PDF))
        page = doc[0]
        words = page.get_text("words")
        doc.close()
        assert len(words) >= 60, (
            f"Expected >= 60 words in text layer, got {len(words)}. "
            "The fixture must be born-digital."
        )

    def test_chart_marks_detectable(self):
        """The vector chart region must be detectable by ``has_chart_marks``."""
        from socr.figures.extractor import has_chart_marks

        doc = fitz.open(str(FIXTURE_PDF))
        page = doc[0]
        result = has_chart_marks(page)
        doc.close()
        assert result, (
            "has_chart_marks returned False — the fixture must have a vector "
            "chart region (filled rects / lines) so the chart lane can detect it."
        )

    def test_no_ruling_lines(self):
        """The TABLE regions must use whitespace gutters, not ruling lines.

        This is the whole point of the CE-like fixture: ``find_tables()`` with
        the default 'lines' strategy has no grid lines to latch onto, exactly
        reproducing the CE p.4 whitespace-gutter failure mode. The chart region
        intentionally DOES contain vector strokes (axis lines), so its vertical
        band is excluded — only the table regions must be line-free.
        """
        import importlib.util

        # Reference the chart band by the generator's own constants (not a
        # hardcoded y-range) so this stays correct if the layout shifts.
        spec = importlib.util.spec_from_file_location(
            "tr0_generate_fixture", FIXTURE_DIR / "generate_fixture.py"
        )
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)

        doc = fitz.open(str(FIXTURE_PDF))
        page = doc[0]
        paths = page.get_drawings()
        doc.close()

        # A ruling line is a long, thin, horizontal STROKE spanning the table
        # width. PyMuPDF ``get_drawings()`` reports stroked paths as type "s" /
        # "fs" (NEVER "l" — the prior filter matched nothing and the assert was
        # a no-op). Exclude the chart's vertical band, whose axis strokes are
        # intentional vector drawing.
        table_col_span_min = 200.0  # wider than any glyph, shorter than the table
        ruling_lines = []
        for p in paths:
            if p.get("type") not in ("s", "fs"):
                continue
            rect = p.get("rect")
            if rect is None:
                continue
            x0, y0, x1, y1 = rect[0], rect[1], rect[2], rect[3]
            is_horizontal = abs(y1 - y0) < 2.0
            spans_table_width = (x1 - x0) > table_col_span_min
            in_chart_band = gen.CHART_TOP <= y0 <= gen.CHART_BOTTOM
            if is_horizontal and spans_table_width and not in_chart_band:
                ruling_lines.append(tuple(rect))
        assert ruling_lines == [], (
            f"Found {len(ruling_lines)} long horizontal ruling line(s) outside the "
            f"chart band: {ruling_lines}. Tables must use whitespace gutters only."
        )

    def test_ground_truth_structure(self):
        """The ground truth JSON must have 4 regions in the expected order."""
        gt = _load_gt()
        assert gt["version"] == 1
        regions = gt["regions"]
        assert len(regions) == 4, f"Expected 4 regions, got {len(regions)}"
        kinds = [r["kind"] for r in regions]
        assert kinds == ["table", "chart", "table", "text"], (
            f"Expected [table, chart, table, text], got {kinds}"
        )

    def test_ground_truth_cell_count(self):
        """Every cell in both tables must be enumerated in the ground truth."""
        gt = _load_gt()
        main = _region(gt, "main_table")
        hist = _region(gt, "historical_table")

        expected_main = len(main["row_labels"]) * len(main["columns"])
        expected_hist = len(hist["row_labels"]) * len(hist["columns"])

        assert len(main["cells"]) == expected_main, (
            f"main_table: expected {expected_main} cells, got {len(main['cells'])}"
        )
        assert len(hist["cells"]) == expected_hist, (
            f"historical_table: expected {expected_hist} cells, got {len(hist['cells'])}"
        )

    def test_na_blanks_present(self):
        """At least 2 blank cells must be marked 'na' in the main table."""
        gt = _load_gt()
        main = _region(gt, "main_table")
        na_cells = [k for k, v in main["cells"].items() if v == "na"]
        assert len(na_cells) >= 2, (
            f"Expected at least 2 'na' cells in main_table, got {len(na_cells)}: {na_cells}"
        )

    def test_ragged_row_present(self):
        """At least one ragged/short row must be declared in the main table."""
        gt = _load_gt()
        main = _region(gt, "main_table")
        assert len(main["ragged_rows"]) >= 1, (
            "Expected at least 1 ragged row in main_table; got none. "
            "The fixture must include a deliberately short/ragged source row."
        )

    def test_chart_has_no_cells(self):
        """The chart region must enumerate zero cells (no transcribed values)."""
        gt = _load_gt()
        chart = _region(gt, "chart")
        assert chart["cells"] == {}, f"chart region must have no cells, got {chart['cells']}"

    def test_prose_has_no_cells(self):
        """The prose region must enumerate zero cells."""
        gt = _load_gt()
        prose = _region(gt, "prose")
        assert prose["cells"] == {}, f"prose region must have no cells, got {prose['cells']}"


# ---------------------------------------------------------------------------
# Parity helper unit tests (pass NOW, no xfail)
# ---------------------------------------------------------------------------


class TestAssertTableParity:
    """Test the parity helper itself on known-good and known-bad markdown."""

    def _gt(self) -> dict:
        return _load_gt()

    def _good_md(self) -> str:
        """Construct a markdown string that satisfies all parity checks."""
        # Main table
        main_header = "| Forecaster | GDP_2024 | GDP_2025 | CPI_2024 | CPI_2025 |"
        main_sep = "| --- | --- | --- | --- | --- |"
        main_rows = [
            "| Ashford Capital | 2.1 | 1.9 | 3.4 | 2.8 |",
            "| Brightwater Research | 2.3 | na | 3.1 | 2.6 |",
            "| Clearview Economics | 1.8 | 2.0 | 3.6 | 3.0 |",
            "| Dunmore Analytics | 2.5 | 2.2 | 3.2 | na |",
            "| Eastfield Partners | 2.0 | 1.8 | 3.5 | 2.9 |",
            "| Fenwick Group | 1.7 | na | na | 2.7 |",
        ]
        main_table = "\n".join([main_header, main_sep] + main_rows)

        # Chart image ref
        chart_ref = "![Federal Budget Balance](figures/chart_page_1.png)"

        # Historical table
        hist_header = "| Indicator | 2021 | 2022 | 2023 |"
        hist_sep = "| --- | --- | --- | --- |"
        hist_rows = [
            "| GDP Actual | 5.9 | 2.1 | 2.5 |",
            "| CPI Actual | 4.7 | 8.0 | 4.1 |",
            "| Unemployment | 5.4 | 3.7 | 3.5 |",
        ]
        hist_table = "\n".join([hist_header, hist_sep] + hist_rows)

        # Prose
        prose = (
            "Government and Background Data\n"
            "The following table draws on official sources including the IMF.\n"
            "Blanks indicate data not available.\n"
        )

        return "\n\n".join([main_table, chart_ref, hist_table, prose])

    def test_good_md_passes(self):
        """A perfectly correct extracted markdown raises no error."""
        gt = self._gt()
        # Should not raise
        assert_table_parity(self._good_md(), gt)

    def test_wrong_cell_value_detected(self):
        """A single wrong cell value must be reported in the error."""
        md = self._good_md().replace("| Ashford Capital | 2.1 |", "| Ashford Capital | 9.9 |")
        gt = self._gt()
        with pytest.raises(ParityError) as exc_info:
            assert_table_parity(md, gt)
        err = str(exc_info.value)
        assert "Ashford Capital|GDP_2024" in err, (
            f"Expected 'Ashford Capital|GDP_2024' in error message; got: {err}"
        )
        assert "9.9" in err or "2.1" in err, (
            f"Expected the mismatched value(s) in error message; got: {err}"
        )

    def test_missing_row_detected(self):
        """A missing data row must be reported as ROW-MISSING."""
        # Remove the Clearview Economics row INCLUDING its trailing newline, so
        # the table stays one contiguous grid (a blank line would instead split
        # it and trip SEPARATE-GRIDS — passing the test for the wrong reason).
        md = self._good_md().replace("| Clearview Economics | 1.8 | 2.0 | 3.6 | 3.0 |\n", "")
        assert "Clearview Economics" not in md  # the row is genuinely gone
        gt = self._gt()
        with pytest.raises(ParityError) as exc_info:
            assert_table_parity(md, gt)
        err = str(exc_info.value)
        assert "ROW-MISSING" in err, (
            f"Expected a ROW-MISSING error on a contiguous table; got: {err}"
        )
        assert "Clearview Economics" in err, (
            f"Expected 'Clearview Economics' in the ROW-MISSING error; got: {err}"
        )

    def test_collapsed_tables_detected(self):
        """A single collapsed grid (tables merged) must trigger SEPARATE-GRIDS."""
        gt = self._gt()
        # Produce a markdown with only ONE table (both tables merged, a collapsed artifact)
        collapsed = (
            "| Forecaster | Value |\n"
            "| --- | --- |\n"
            "| Ashford Capital\\nBrightwater Research | 2.1\\n2.3 |\n"
        )
        with pytest.raises(ParityError) as exc_info:
            assert_table_parity(collapsed, gt)
        err = str(exc_info.value)
        assert "SEPARATE-GRIDS" in err, (
            f"Expected 'SEPARATE-GRIDS' error for merged tables; got: {err}"
        )

    def test_chart_without_image_ref_detected(self):
        """Absence of an image reference must raise CHART-NO-IMAGE-REF."""
        # Build a good MD but remove the chart image ref
        md = self._good_md().replace("![Federal Budget Balance](figures/chart_page_1.png)", "")
        gt = self._gt()
        with pytest.raises(ParityError) as exc_info:
            assert_table_parity(md, gt)
        err = str(exc_info.value)
        assert "CHART-NO-IMAGE-REF" in err, f"Expected 'CHART-NO-IMAGE-REF' error; got: {err}"

    def test_na_blank_mismatch_detected(self):
        """A 'na' cell filled with a number must be reported as CELL-MISMATCH."""
        # Replace 'na' in Brightwater GDP_2025 with a plausible number
        md = self._good_md().replace(
            "| Brightwater Research | 2.3 | na | 3.1 | 2.6 |",
            "| Brightwater Research | 2.3 | 1.5 | 3.1 | 2.6 |",
        )
        gt = self._gt()
        with pytest.raises(ParityError) as exc_info:
            assert_table_parity(md, gt)
        err = str(exc_info.value)
        assert "Brightwater Research|GDP_2025" in err, (
            f"Expected 'Brightwater Research|GDP_2025' in error; got: {err}"
        )

    def test_multiple_mismatches_reported_together(self):
        """All mismatching cells must be in a single ParityError, not just the first."""
        # Corrupt two cells in two different rows
        md = (
            self._good_md()
            .replace("| Ashford Capital | 2.1 |", "| Ashford Capital | 0.0 |")
            .replace("| Clearview Economics | 1.8 |", "| Clearview Economics | 0.0 |")
        )
        gt = self._gt()
        with pytest.raises(ParityError) as exc_info:
            assert_table_parity(md, gt)
        err = str(exc_info.value)
        # Both mismatches must appear in the same error
        assert "Ashford Capital" in err and "Clearview Economics" in err, (
            f"Expected both mismatches in one error; got: {err}"
        )


# ---------------------------------------------------------------------------
# End-to-end agentic test  (TR-2 implemented — xfail removed)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TR-4a: Dense-fixture (real-CE-geometry) — the failing gate for TR-4
# ---------------------------------------------------------------------------

DENSE_FIXTURE_PDF = FIXTURE_DIR / "ce_like_p4_dense.pdf"
DENSE_GROUND_TRUTH = FIXTURE_DIR / "ground_truth_dense.json"


def _load_dense_gt() -> dict:
    return json.loads(DENSE_GROUND_TRUTH.read_text())


def _dense_region(gt: dict) -> dict:
    """Return the single table region from the dense ground truth."""
    for r in gt["regions"]:
        if r["kind"] == "table":
            return r
    raise KeyError("no table region in dense ground truth")


def assert_dense_table_parity(extracted_md: str, ground_truth: dict) -> None:
    """Assert the dense fixture's table is correctly extracted.

    The dense fixture has exactly ONE table region (the forecaster grid).
    This helper checks that EVERY row label is paired with its correct values
    — which FAILS today because the name/value y-offset causes interleaved
    rows in the rowizer output (name row has no data; data row has no label).

    Raises ``ParityError`` with a per-cell diff on any mismatch.
    """
    region = _dense_region(ground_truth)
    row_labels = region["row_labels"]
    columns = region["columns"]
    expected_cells = region["cells"]

    errors: list[str] = []

    # Parse markdown tables from extracted output
    md_tables = _parse_markdown_tables(extracted_md)
    if not md_tables:
        errors.append(
            "NO-TABLE: no markdown table found in the dense fixture output. "
            "Expected at least one table with forecaster rows."
        )
        for key, val in list(expected_cells.items())[:5]:
            errors.append(f"  MISSING {key}: expected {val!r}")
        if errors:
            bullet = "\n  - "
            raise ParityError(
                f"Dense parity check failed ({len(errors)} issue(s)):{bullet}" + bullet.join(errors)
            )

    # Collect all rows from all parsed tables (the interleaved output splits
    # the forecasters into multiple regions separated by the summary gap).
    all_rows: list[list[str]] = []
    for grid in md_tables:
        all_rows.extend(grid)

    # Build a mapping from label to its data cells (by scanning all rows for
    # label matches in column 0).
    label_to_data: dict[str, list[str]] = {}
    for row in all_rows:
        if not row:
            continue
        label_text = row[0].strip()
        for lbl in row_labels:
            if lbl in label_text and label_text not in ("", "---"):
                # Found a row with this label; record its data cells
                data = [c.strip() for c in row[1:]]
                label_to_data[lbl] = data
                break

    # Per-cell check: for each expected cell, verify the label row has the
    # correct value at the right column position.
    # The header row detection: find a row containing any column name token.
    col_tokens = [c.replace("_", " ").split()[-1] for c in columns]  # e.g. "2024"
    header_row: list[str] = []
    for grid in md_tables:
        for row in grid[:3]:  # header is in the first few rows
            row_text = " ".join(row)
            if any(tok in row_text for tok in col_tokens):
                header_row = row
                break
        if header_row:
            break

    # Map column names to header indices
    col_idx: dict[str, int] = {}
    for col in columns:
        col_norm = col.replace("_", " ")
        for ci, hdr_cell in enumerate(header_row):
            hdr_norm = hdr_cell.replace("_", " ")
            if col_norm in hdr_norm or hdr_norm in col_norm:
                col_idx[col] = ci
                break

    for key, expected_val in expected_cells.items():
        label, col = key.split("|", 1)
        data = label_to_data.get(label)
        if data is None:
            errors.append(
                f"ROW-MISSING {key}: label {label!r} not found as a row header "
                f"in the extracted output.  This indicates the name↔value binding "
                f"is broken (the label row has no data, or is absent entirely)."
            )
            continue

        # The data cells are relative to the label column (index 0 of the row).
        # col_idx gives position in the full row; data starts at index 1.
        ci = col_idx.get(col)
        if ci is None:
            errors.append(
                f"COL-MISSING {key}: column {col!r} not found in header row {header_row!r}."
            )
            continue

        # data[ci - 1] because data is row[1:] (skips label cell)
        data_idx = ci - 1
        if data_idx < 0 or data_idx >= len(data):
            errors.append(
                f"CELL-OOB {key}: data index {data_idx} out of range "
                f"(data has {len(data)} cells: {data!r})."
            )
            continue

        actual_val = data[data_idx] if data[data_idx] else "na"
        if actual_val != expected_val:
            errors.append(f"CELL-MISMATCH {key}: expected {expected_val!r}, got {actual_val!r}.")

    if errors:
        bullet = "\n  - "
        raise ParityError(
            f"Dense parity check failed ({len(errors)} issue(s)):{bullet}" + bullet.join(errors)
        )


class TestDenseFixtureSanity:
    """Verify the TR-4a dense fixture meets its physical requirements.

    These tests PASS now (sanity checks on the fixture itself).
    """

    def test_dense_pdf_exists(self) -> None:
        assert DENSE_FIXTURE_PDF.exists(), f"Dense fixture PDF missing: {DENSE_FIXTURE_PDF}"

    def test_dense_ground_truth_exists(self) -> None:
        assert DENSE_GROUND_TRUTH.exists(), f"Dense GT missing: {DENSE_GROUND_TRUTH}"

    def test_dense_born_digital(self) -> None:
        """Dense fixture must have a real text layer (born-digital)."""
        doc = fitz.open(str(DENSE_FIXTURE_PDF))
        page = doc[0]
        words = page.get_text("words")
        doc.close()
        assert len(words) >= 40, (
            f"Expected >= 40 words in dense fixture text layer, got {len(words)}."
        )

    def test_dense_gt_version_and_regions(self) -> None:
        gt = _load_dense_gt()
        assert gt["version"] == 1
        regions = gt["regions"]
        assert len(regions) == 1, f"Expected 1 region in dense GT, got {len(regions)}"
        assert regions[0]["kind"] == "table"

    def test_dense_gt_cell_count(self) -> None:
        gt = _load_dense_gt()
        region = _dense_region(gt)
        expected = len(region["row_labels"]) * len(region["columns"])
        assert len(region["cells"]) == expected, (
            f"dense_table: expected {expected} cells, got {len(region['cells'])}"
        )

    def test_dense_geometry_fields_present(self) -> None:
        """Ground truth must record the geometry metadata for traceability."""
        gt = _load_dense_gt()
        geo = gt.get("geometry", {})
        assert "name_y_offset_pt" in geo, "geometry.name_y_offset_pt missing"
        assert "row_spacing_pt" in geo, "geometry.row_spacing_pt missing"
        assert geo["name_y_offset_pt"] >= 0.5, (
            f"name_y_offset_pt={geo['name_y_offset_pt']} is too small; "
            "must be >= 0.5 pt to produce distinct y-groups in the rowizer."
        )

    def test_dense_fixture_produces_lane_stacked_table(self) -> None:
        """The dense fixture MUST trigger find_tables() returning a lane-stacked result.

        This is the critical prerequisite: the fixture must route through
        ``_is_lane_stacked`` -> ``rowize_from_word_list``, which is where the
        name/value y-offset failure occurs.  If find_tables() returns 0 tables,
        the pipeline falls through to reconstruct_table_regions (text-strategy)
        which handles the offset correctly -> false green.
        """
        from socr.core.born_digital import _is_lane_stacked

        doc = fitz.open(str(DENSE_FIXTURE_PDF))
        page = doc[0]
        tables_result = page.find_tables()
        doc.close()

        assert len(tables_result.tables) >= 1, (
            "Dense fixture must produce >= 1 table from find_tables() — "
            "the table border structure must trigger the lines-strategy detector.  "
            "If find_tables() returns 0 tables the failure mode is not reproduced "
            "(the pipeline uses reconstruct_table_regions instead which handles "
            "the name/value offset correctly)."
        )
        lane_stacked = [t for t in tables_result.tables if _is_lane_stacked(t)]
        assert len(lane_stacked) >= 1, (
            "Dense fixture must produce at least one lane-stacked table from "
            "find_tables() — the name/value y-offset must cause cells with "
            "embedded newlines (\\n-stacked tokens), which is the trigger for "
            "the rowize_from_word_list path where the interleaving failure occurs."
        )

    def test_dense_extraction_reproduces_interleaved_failure(self) -> None:
        """extract_structured MUST produce the interleaved name/value failure.

        This test CONFIRMS the fixture faithfully reproduces real CE's failure:
        the output must contain label-only rows (names) interleaved with
        data-only rows (values), broken name↔value binding.

        This test itself PASSES (it asserts the failure IS present in the
        output).  The xfail is on the PARITY test that requires the failure
        to be FIXED.
        """
        from socr.core.born_digital import BornDigitalDetector

        doc = fitz.open(str(DENSE_FIXTURE_PDF))
        page = doc[0]
        detector = BornDigitalDetector()
        extracted = detector.extract_structured(page)
        doc.close()

        # The interleaved output has rows like:
        #   |  | 2.3 | 1.9 | 2.1 | 2.4 |   ← data with EMPTY label
        #   | Alpha Economics |  |  |  |  |  ← label with NO data
        # Parse the markdown tables to find these patterns.
        md_tables = _parse_markdown_tables(extracted)
        assert md_tables, (
            "extract_structured must produce at least one markdown table on the "
            "dense fixture (after _is_lane_stacked routes to the rowizer).  "
            f"Got empty output. Full extraction:\n{extracted[:500]}"
        )

        # Look for label-only rows: label non-empty, ALL data cells empty
        label_only_rows = []
        data_only_rows = []
        for grid in md_tables:
            for row in grid[1:]:  # skip header
                if not row:
                    continue
                label = row[0].strip()
                data = [c.strip() for c in row[1:]]
                has_label = bool(label) and not re.fullmatch(r"-+", label)
                has_data = any(c for c in data)
                if has_label and not has_data:
                    label_only_rows.append(row)
                elif not has_label and has_data:
                    data_only_rows.append(row)

        assert label_only_rows, (
            "Expected label-only rows (name present, no data values) in the dense "
            "fixture extraction — this is the interleaved failure mode.  "
            "If no label-only rows exist the fixture does NOT reproduce real CE's "
            "failure and must be revised.  "
            f"Full extraction:\n{extracted}"
        )
        assert data_only_rows, (
            "Expected data-only rows (data values present, no label) in the dense "
            "fixture extraction — this is the interleaved failure mode.  "
            f"Full extraction:\n{extracted}"
        )


# ---------------------------------------------------------------------------
# TR-4a xfail parity test — the FAILING GATE for TR-4
# ---------------------------------------------------------------------------


class TestDenseFixtureParityXfail:
    """Dense fixture parity — XFAIL until TR-4 fixes the row-structure problem.

    This test reproduces the real-CE failure (top block merged, names offset
    from values by one row) on the synthesized dense fixture.  It is marked
    ``xfail(strict=True)`` so CI stays green and the test turns red the moment
    the underlying bug is fixed (reminding the implementer to remove the xfail).

    The test will PASS (and the xfail will be satisfied) once TR-4
    (value-guarded VLM-for-structure) correctly assigns each name to its
    corresponding value row.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "TR-4 value-guard implemented (label-binding correctly detects the "
            "interleaved row pattern) but the RECOVERY step is not yet done: "
            "the TR-1 rowizer still produces interleaved rows on the dense fixture "
            "(label rows with no data, data rows with no label), and the VLM "
            "re-ask that would propose the correct structure is a later step.  "
            "The value-guard rejects the broken output as intended (hard-fail via "
            "label-binding).  Remove this xfail once the VLM re-ask + structure "
            "recovery produces a parity-passing output."
        ),
    )
    def test_dense_parity_fails_today(self) -> None:
        """Dense fixture parity must FAIL today (reproducing real-CE row-offset bug).

        This test drives ``BornDigitalDetector().extract_structured()`` directly
        (no agentic pipeline needed — the failure is in the born-digital path)
        and asserts cell-by-cell parity against the dense ground truth.

        It is expected to FAIL (xfail) because the interleaved output breaks the
        name↔value binding: each forecaster row produces two grid rows —
        one with the label and no data, one with the data and no label.

        CI hermeticity: no ollama/provider is needed — this exercises only the
        native text extraction path (BornDigitalDetector), which runs entirely
        in PyMuPDF without any model call.
        """
        from socr.core.born_digital import BornDigitalDetector

        doc = fitz.open(str(DENSE_FIXTURE_PDF))
        page = doc[0]
        detector = BornDigitalDetector()
        extracted = detector.extract_structured(page)
        doc.close()

        gt = _load_dense_gt()
        # This assertion IS expected to raise ParityError today — names are
        # in rows by themselves (no values), so every cell will be MISSING.
        assert_dense_table_parity(extracted, gt)


class TestEndToEndParity:
    """Drive ``UnifiedPipeline.process()`` on the fixture and assert full parity.

    TR-2 (chart-clip + per-region verifier + token coverage + reading-order
    reassembly) is now implemented, so the native reconstruction must reach
    full cell parity.

    P2 / GH-317 retarget. The fixture's page carries tables, so it is excluded
    from the free native lane and enters the provider ladder; the route stub
    below accepts nothing, so the ladder is exhausted with no attempt authoring
    a grid. Before P2 that ending shipped the native reconstruction, and this
    test measured TR-2 parity on the assembled markdown. P2 rules that the
    native grid the verifier refused must never be the consolation prize, so
    the page now ships the fail-closed floor and the grid is withheld.

    The measurement is therefore taken where the reconstruction still exists --
    the page's own native text, captured from the live ``DocumentState`` -- and
    is NOT weakened: the same ``assert_table_parity`` over the same ground
    truth, cell for cell, separate grids and reading order included. What is
    added is the P2 guarantee: those bytes must not appear in the shipped
    document.
    """

    def test_agentic_parity_on_ce_like_fixture(self, tmp_path: Path) -> None:
        """Native reconstruction reaches full cell parity; P2 withholds it from the ship.

        Hermeticity:
          - ``_available_engines_for_agentic`` is patched → no ollama required.
          - ``route_page`` is patched → no VLM call; the page uses the
            born-digital / native text path (the path TR-2 fixes).
          - ``probe_ollama_idle`` is patched → no network call.
        """
        from socr.core.config import EngineType, PipelineConfig
        from socr.core.manifest import _native_text_with_appends
        from socr.core.providers import PROFILE_QWEN_LOCAL
        from socr.pipeline.orchestrator import UnifiedPipeline

        config = PipelineConfig(
            primary_engine=EngineType.QWEN,
            agentic=True,
            judge_backend="heuristic",
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            save_figures=False,
            write_manifest=False,
        )
        pipeline = UnifiedPipeline(config)

        captured_state: dict[str, object] = {}
        orig_assemble = pipeline._phase_assemble

        def _capture_assemble(state, output_dir):
            captured_state["state"] = state
            return orig_assemble(state, output_dir)

        with (
            patch.object(
                pipeline,
                "_available_engines_for_agentic",
                return_value=[PROFILE_QWEN_LOCAL],
            ),
            patch("socr.pipeline.orchestrator.route_page") as mock_route,
            patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True),
            # CI has no ollama: `_phase_judge_hard_pages` builds an
            # OllamaVisionJudge and POSTs to it regardless of `judge_backend`
            # unless the judge model resolves empty.
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch.object(pipeline, "_phase_assemble", side_effect=_capture_assemble),
        ):
            # route_page should NOT be called for a native (born-digital) page,
            # but we patch it defensively so CI does not attempt a real VLM call.
            from socr.core.result import PageOutput, PageStatus
            from socr.pipeline.agentic import PageDecision, ProviderAttempt

            def _native_route(page_num, ladder, run_provider, judge, **kwargs):
                out = PageOutput(
                    page_num=page_num,
                    text="[native route stub — should not be called]",
                    status=PageStatus.SUCCESS,
                    engine="qwen",
                    audit_passed=False,
                )
                prof = ladder[0]
                att = ProviderAttempt(
                    engine=prof.engine,
                    output=out,
                    cost_usd=0.0,
                    accepted=False,
                    reason="stub",
                    provider_id=prof.id,
                    model=prof.model,
                    backend=prof.backend,
                )
                return PageDecision(page_num=page_num, final_output=out, attempts=[att])

            mock_route.side_effect = _native_route

            result = pipeline.process(FIXTURE_PDF, tmp_path)

        gt = _load_gt()

        # TR-1..TR-3: the native reconstruction must reach full cell parity.
        # Measured on the reconstruction itself because P2 stops it shipping.
        state = captured_state["state"]
        native_md = "\n\n".join(
            _native_text_with_appends(state.pages[n]) for n in sorted(state.pages)
        )
        assert_table_parity(native_md, gt)

        # P2 / GH-317: the ladder was exhausted with no rung authoring a grid,
        # so that reconstruction is withheld and the whole-page floor ships.
        # This fixture is ONE page and it is floored, so the document carries no
        # text at all: assemble writes no markdown, and the page's marker and
        # image live in the sidecar and the figures directory. See the
        # second-order-consequence note in
        # docs/log/2026-09-01_p2-structure-class-floor.md.
        extracted_md = result.markdown or ""
        for row_label in ("Ashford Capital", "Brightwater Research", "GDP Actual"):
            assert row_label in native_md, f"fixture no longer produces {row_label!r}"
            assert row_label not in extracted_md, (
                f"refused native grid row {row_label!r} shipped despite the P2 floor"
            )

        # Withheld, not silently lost: the failure surfaces on the document.
        from socr.core.result import DocumentStatus

        assert result.status is DocumentStatus.ERROR, result.status
        assert "structure-class ladder exhausted" in (result.error or "")
