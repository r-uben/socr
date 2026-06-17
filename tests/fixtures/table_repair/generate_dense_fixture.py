"""Generate the TR-4a dense-fixture PDF (ce_like_p4_dense.pdf).

PURPOSE
-------
This generator reproduces the GEOMETRY of real Consensus Economics page 4
(202401.pdf p.4) so that the fixture faithfully exercises the row-structure
failure mode that TR-1...TR-3 could NOT expose (the clean TR-0 fixture gave
false confidence).

LICENSE-CLEAN PROCESS
---------------------
The row y-positions and column x-positions used below were DERIVED from the
real CE PDF (via ``page.get_text("words")`` on a locally-licensed copy) at
AUTHORING TIME.  Only the COORDINATES were copied — every forecaster name
and every numeric value is INVENTED.  No real CE text, firm name, or figure
appears in the generated PDF or in this file.  The real CE PDF is NOT
accessed at test runtime (the generator is run once; only the committed
ce_like_p4_dense.pdf is a test dependency).

GEOMETRY EXTRACTED (read-only, not committed)
---------------------------------------------
From 202401.pdf page 3 (0-indexed, i.e. page 4 in 1-indexed):
  - Value row y-positions: alternating pattern starting at ~143.4 pt,
    spacing ~10.5 pt between value rows.
  - Name baseline appears ~1.0 pt BELOW its corresponding value row
    (e.g. values at y=143.4, name at y=144.4).
  - Column x-positions: name label at x≈37.4; data columns at
    x≈149.0, 169.9, 189.4, 210.2 (first four indicator-year pairs).
  - Summary rows (Consensus, High, Low) appear after a larger gap
    (~21 pt) following the last forecaster row.

FAILURE MODE REPRODUCED
-----------------------
The TR-1 rowizer groups words by rounded y0. When name baseline is 1 pt
below value baseline, the rounded y-groups are:
  y_val=143 → data row  (empty label, data cells filled)
  y_name=144 → name row (label='FirmA', all data cells empty)
This interleaves name and value rows, breaking the name↔value binding:
parity checker sees values orphaned from their labels and names orphaned
from their values.  The median inter-row gap in the rounded sequence is
~1 pt, so the split threshold stays at _SPLIT_GAP_MIN_PT (10.0 pt) and the
whole table ends in ONE segment — but with interleaved rows.

XFAIL INTENT
------------
This fixture is the acceptance gate for TR-4 (value-guarded VLM-for-structure).
The dense-fixture parity test is marked xfail(strict=True) until TR-4 fixes
the rowizer.  Once TR-4 is implemented the test must be un-xfailed and must
PASS (not just xpass).
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# Geometry constants (derived from real CE 202401.pdf p.4 at authoring time)
# ---------------------------------------------------------------------------
# These coordinates reproduce the CE page's STRUCTURE; the CONTENT (names,
# numbers) is entirely synthesized.  All units are PDF points (1 pt = 1/72 in).

PAGE_W = 595.0  # A4 width (close to real CE page size)
PAGE_H = 842.0  # A4 height

# Name label x-position (left of the data columns)
LABEL_X = 37.4

# Data column x-positions (indicator-year pairs): GDP_2024, GDP_2025, CPI_2024, CPI_2025
# Derived from the first four column pairs of the real CE table.
DATA_COL_XS = [149.0, 169.9, 189.4, 210.2]

# Column names (synthesized — not real CE indicator names)
COLUMNS = ["GDP_2024", "GDP_2025", "CPI_2024", "CPI_2025"]

# Real CE geometry: first value row at y≈143.4, spacing ≈10.5 pt.
# Name baseline is ≈1.0 pt below the corresponding value row.
FIRST_VALUE_Y = 143.4  # y0 of the first data value row
ROW_SPACING = 10.5  # pt between consecutive value rows (real CE ~10.5)
NAME_Y_OFFSET = 1.0  # pt: name baseline = value_y + NAME_Y_OFFSET (the critical offset)

# Gap between last forecaster row and summary rows (real CE ~21 pt after last row)
SUMMARY_GAP = 21.0

# Font sizes
FS_HDR = 7.5
FS_DATA = 7.5

# Header y positions (above first data row)
HDR_INDICATOR_Y = 82.0  # indicator row (GDP, CPI)
HDR_YEAR_Y = 127.1  # year row (2024, 2025)
LABEL_HDR_Y = 126.3  # "Economic Forecasters" label

# ---------------------------------------------------------------------------
# Synthesized ground-truth data
# ---------------------------------------------------------------------------
# 8 forecaster rows + 3 summary rows.  Invented names and numbers — no real
# CE content.  Values are chosen to be plausible (economic forecast range)
# but are deliberately "round-ish" so they are easy to spot in test failures.

FORECASTER_ROWS: list[dict] = [
    {
        "label": "Alpha Economics",
        "GDP_2024": "2.3",
        "GDP_2025": "1.9",
        "CPI_2024": "2.1",
        "CPI_2025": "2.4",
    },
    {
        "label": "Beta Research",
        "GDP_2024": "2.0",
        "GDP_2025": "1.4",
        "CPI_2024": "2.2",
        "CPI_2025": "1.8",
    },
    {
        "label": "Gamma Partners",
        "GDP_2024": "1.9",
        "GDP_2025": "2.1",
        "CPI_2024": "1.5",
        "CPI_2025": "2.0",
    },
    {
        "label": "Delta Analytics",
        "GDP_2024": "1.9",
        "GDP_2025": "1.6",
        "CPI_2024": "2.0",
        "CPI_2025": "1.7",
    },
    {
        "label": "Epsilon Capital",
        "GDP_2024": "1.7",
        "GDP_2025": "1.1",
        "CPI_2024": "1.6",
        "CPI_2025": "1.5",
    },
    {
        "label": "Zeta Forecasting",
        "GDP_2024": "1.7",
        "GDP_2025": "1.5",
        "CPI_2024": "1.8",
        "CPI_2025": "1.6",
    },
    {
        "label": "Eta Institute",
        "GDP_2024": "1.7",
        "GDP_2025": "1.7",
        "CPI_2024": "1.6",
        "CPI_2025": "1.9",
    },
    {
        "label": "Theta Associates",
        "GDP_2024": "1.6",
        "GDP_2025": "1.5",
        "CPI_2024": "1.4",
        "CPI_2025": "1.3",
        "_ragged": True,
    },
]

SUMMARY_ROWS: list[dict] = [
    {
        "label": "Consensus (Mean)",
        "GDP_2024": "1.9",
        "GDP_2025": "1.6",
        "CPI_2024": "1.8",
        "CPI_2025": "1.8",
    },
    {"label": "High", "GDP_2024": "2.3", "GDP_2025": "2.1", "CPI_2024": "2.2", "CPI_2025": "2.4"},
    {"label": "Low", "GDP_2024": "1.6", "GDP_2025": "1.1", "CPI_2024": "1.4", "CPI_2025": "1.3"},
]


def _value_y(row_index: int) -> float:
    """Y position of the data values for forecaster row *row_index*."""
    return FIRST_VALUE_Y + row_index * ROW_SPACING


def _name_y(row_index: int) -> float:
    """Y position of the forecaster NAME for row *row_index*.

    The name baseline is NAME_Y_OFFSET pt BELOW the corresponding value
    baseline — reproducing the real CE typesetting where the name glyph
    is set slightly lower than the numeric glyphs in the same logical row.
    This is the source of the interleaved-row failure in the TR-1 rowizer.
    """
    return _value_y(row_index) + NAME_Y_OFFSET


def _summary_value_y(summary_index: int) -> float:
    """Y for the data values of summary row *summary_index*."""
    last_forecaster_y = _value_y(len(FORECASTER_ROWS) - 1)
    return last_forecaster_y + SUMMARY_GAP + summary_index * ROW_SPACING


def _summary_name_y(summary_index: int) -> float:
    return _summary_value_y(summary_index) + NAME_Y_OFFSET


def build_ground_truth() -> dict:
    """Build the canonical ground-truth dict for the dense fixture."""
    cells: dict[str, str] = {}
    for row in FORECASTER_ROWS + SUMMARY_ROWS:
        label = row["label"]
        for col in COLUMNS:
            cells[f"{label}|{col}"] = row[col]

    all_rows = FORECASTER_ROWS + SUMMARY_ROWS
    row_labels = [r["label"] for r in all_rows]
    ragged_rows = [r["label"] for r in FORECASTER_ROWS if r.get("_ragged")]
    na_cells = [k for k, v in cells.items() if v == "na"]

    return {
        "version": 1,
        "fixture": "ce_like_p4_dense.pdf",
        "description": (
            "Dense TR-4a fixture: geometry derived from real CE 202401.pdf p.4 "
            "(row y-positions and column x-positions only — all names and values "
            "are INVENTED).  Reproduces the name/value y-offset (≈1 pt) and "
            "dense row spacing (≈10.5 pt) that breaks the TR-1 rowizer's "
            "gap-segmentation: names and values interleave in the grid output, "
            "breaking the name↔value binding.  This is the TR-4 acceptance gate."
        ),
        "regions": [
            {
                "id": "dense_table",
                "kind": "table",
                "order": 0,
                "schema": "forecaster_grid_dense",
                "columns": COLUMNS,
                "row_labels": row_labels,
                "cells": cells,
                "ragged_rows": ragged_rows,
                "na_cells": na_cells,
            }
        ],
        "geometry": {
            "note": (
                "Geometry derived from real CE 202401.pdf p.4 at authoring time. "
                "Coordinates only — no CE content."
            ),
            "first_value_y_pt": FIRST_VALUE_Y,
            "row_spacing_pt": ROW_SPACING,
            "name_y_offset_pt": NAME_Y_OFFSET,
            "data_col_xs": DATA_COL_XS,
            "label_x": LABEL_X,
        },
    }


def generate_pdf(out_path: Path) -> None:
    """Write the dense fixture PDF to *out_path*.

    Places synthesized forecaster names and invented numbers at the REAL CE
    page's geometric coordinates.  No real CE content is written.

    The table has a thin outer border box (a single stroke rectangle spanning
    the header-to-summary area) derived from real CE geometry.  This border is
    what makes PyMuPDF's ``find_tables(strategy="lines")`` detect a table —
    reproducing the real CE trigger that routes the region to ``_is_lane_stacked``
    → ``rowize_from_word_list``, where the name/value y-offset produces the
    interleaved failure.  Without this border the fixture would route to
    ``reconstruct_table_regions`` (text-strategy) which handles the offset
    correctly and gives a false green.
    """
    import fitz  # PyMuPDF

    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)

    # ------------------------------------------------------------------
    # Table border structure (stroke-only, thin)
    #
    # The real CE table has:
    #  1. An outer border rectangle (the "frame" around the whole table)
    #  2. Vertical column-separator lines spanning top-to-bottom of the table
    #  3. Horizontal separator lines (between header and data, between data and
    #     summary rows)
    #
    # These strokes are what makes PyMuPDF's ``find_tables(strategy="lines")``
    # detect the region as a table.  Without them, find_tables returns 0 tables
    # and the pipeline falls through to reconstruct_table_regions (text-strategy)
    # which handles the name/value offset correctly — giving a false green.
    #
    # Geometry derived from real CE 202401.pdf p.4 (coordinates only, no content):
    #  - Outer box: x0≈34, y0≈63, x1≈562, y1≈563 (full-page border)
    #  - Column separators: at x≈145.9, 186.7, 225.8, 271.9 (between data cols)
    # ------------------------------------------------------------------
    # In our fixture we place the right boundary just past the last data column
    last_data_x = DATA_COL_XS[-1]
    border_x0 = LABEL_X - 5.0  # left of label column
    border_y0 = 58.0  # above the header
    border_x1 = last_data_x + 22.0  # right of last data column
    last_summary_y = _summary_value_y(len(SUMMARY_ROWS) - 1) + 8.0
    border_y1 = last_summary_y + 6.0

    import fitz as _fitz

    # Outer border rectangle
    border_rect = _fitz.Rect(border_x0, border_y0, border_x1, border_y1)
    page.draw_rect(border_rect, color=(0.0, 0.38, 0.66), width=0.48, fill=None)

    # Vertical column-separator lines (between data columns, and between label
    # and first data column).  These make find_tables() see distinct columns —
    # but because the name-label text (left of the first separator) and value
    # text (right of the separators) interleave by y, the result is lane-stacked.
    col_sep_x_positions = [
        DATA_COL_XS[0] - 7.0,  # left of first data column (label|data boundary)
        DATA_COL_XS[1] - 7.0,  # between col 0 and col 1
        DATA_COL_XS[2] - 7.0,  # between col 1 and col 2
        DATA_COL_XS[3] - 7.0,  # between col 2 and col 3
    ]
    for sep_x in col_sep_x_positions:
        page.draw_line(
            (sep_x, border_y0),
            (sep_x, border_y1),
            color=(0.70, 0.70, 0.71),
            width=0.48,
        )

    # Horizontal separator after the header (below the year row)
    sep_y_header = HDR_YEAR_Y + 4.0  # just below year header
    page.draw_line(
        (border_x0, sep_y_header),
        (border_x1, sep_y_header),
        color=(0.0, 0.38, 0.66),
        width=0.48,
    )

    # Horizontal separator before summary rows (above Consensus)
    sep_y_summary = _summary_value_y(0) - 8.0
    page.draw_line(
        (border_x0, sep_y_summary),
        (border_x1, sep_y_summary),
        color=(0.70, 0.70, 0.71),
        width=0.48,
    )

    # ------------------------------------------------------------------
    # Header rows (two-line: indicator / year)
    # ------------------------------------------------------------------
    # Indicator row (real CE has multi-line stacked headers)
    for x, ind in zip(DATA_COL_XS, ["GDP", "GDP", "CPI", "CPI"]):
        page.insert_text((x, HDR_INDICATOR_Y), ind, fontsize=FS_HDR, fontname="helv")

    # Year row
    page.insert_text(
        (LABEL_X, LABEL_HDR_Y), "Economic Forecasters", fontsize=FS_HDR, fontname="helv"
    )
    for x, yr in zip(DATA_COL_XS, ["2024", "2025", "2024", "2025"]):
        page.insert_text((x, HDR_YEAR_Y), yr, fontsize=FS_HDR, fontname="helv")

    # ------------------------------------------------------------------
    # Forecaster data rows
    # ------------------------------------------------------------------
    # CRITICAL: values appear at y_val; name appears at y_val + NAME_Y_OFFSET.
    # This is the key geometric property from real CE — the name glyph is
    # set ~1 pt lower than the numeric glyphs in the same logical row.
    # The 1 pt offset makes y-groups distinct when rounded (e.g. y_val rounds
    # to 135 and y_name rounds to 136), causing the rowizer to produce
    # INTERLEAVED rows: data-row with empty label, then name-row with no data.
    for ri, row in enumerate(FORECASTER_ROWS):
        y_val = _value_y(ri)
        y_name = _name_y(ri)

        # Name at y_name (slightly BELOW value row — the critical offset)
        page.insert_text(
            (LABEL_X, y_name),
            row["label"],
            fontsize=FS_DATA,
            fontname="helv",
        )
        # Data values at y_val
        for col, x in zip(COLUMNS, DATA_COL_XS):
            val = row[col]
            if val != "na":
                page.insert_text((x, y_val), val, fontsize=FS_DATA, fontname="helv")

    # ------------------------------------------------------------------
    # Summary rows (Consensus, High, Low) — larger gap after last forecaster
    # ------------------------------------------------------------------
    for si, row in enumerate(SUMMARY_ROWS):
        y_val = _summary_value_y(si)
        y_name = _summary_name_y(si)

        page.insert_text(
            (LABEL_X, y_name),
            row["label"],
            fontsize=FS_DATA,
            fontname="helv",
        )
        for col, x in zip(COLUMNS, DATA_COL_XS):
            val = row[col]
            if val != "na":
                page.insert_text((x, y_val), val, fontsize=FS_DATA, fontname="helv")

    doc.save(str(out_path))
    doc.close()


def main() -> None:
    pdf_path = HERE / "ce_like_p4_dense.pdf"
    gt_path = HERE / "ground_truth_dense.json"

    generate_pdf(pdf_path)
    gt = build_ground_truth()
    gt_path.write_text(json.dumps(gt, indent=2) + "\n")

    print(f"Generated: {pdf_path}")
    print(f"Generated: {gt_path}")

    # Sanity check: born-digital text layer
    import fitz

    doc = fitz.open(str(pdf_path))
    page = doc[0]
    words = page.get_text("words")
    doc.close()
    print(f"Text-layer word count: {len(words)} (expected >= 40)")
    assert len(words) >= 40, f"Too few words: {len(words)}"

    # Verify the geometry produces the failure: name and value y-groups
    # are distinct (not merged into one rounded-y group).
    y0s = sorted({round(w[1]) for w in words})
    print(f"Unique rounded y0 values: {y0s[:20]} ...")

    # Each forecaster should contribute TWO distinct y-groups (value row + name row).
    # The name and value for the same forecaster must NOT round to the same y.
    # Find the first data value by looking at words with x in data column range.
    data_words = [w for w in words if DATA_COL_XS[0] - 2 <= w[0] <= DATA_COL_XS[-1] + 20]
    name_words = [w for w in words if w[0] < DATA_COL_XS[0] - 2 and 100 < w[1] < 250]

    if data_words and name_words:
        # Find the first forecaster's data y and name y
        first_data_y = round(min(w[1] for w in data_words if w[1] > 100))
        first_name_y = round(min(w[1] for w in name_words))
        print(
            f"Geometry check: first data row y≈{first_data_y}, "
            f"first name row y≈{first_name_y}, offset={first_name_y - first_data_y} pt"
        )
        assert first_name_y != first_data_y, (
            f"Name y and value y round to the same value ({first_data_y}) — "
            "the offset is too small to reproduce the failure mode. "
            f"NAME_Y_OFFSET={NAME_Y_OFFSET} is insufficient."
        )
        assert abs(first_name_y - first_data_y) <= 2, (
            f"Name/value offset too large ({first_name_y - first_data_y} pt); "
            "the fixture should have a small (~1 pt) offset to match real CE geometry."
        )

    print("OK: dense fixture born-digital and geometry confirmed.")


if __name__ == "__main__":
    main()
