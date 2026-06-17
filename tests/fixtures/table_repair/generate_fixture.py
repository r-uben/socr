"""Generate the TR-0 license-clean CE-like fixture PDF.

This script is SEMANTICALLY idempotent: running it always produces the same text
layer, geometry, and ``ground_truth.json``. The raw ``ce_like_p4.pdf`` bytes are
NOT identical run-to-run — PyMuPDF embeds a random document ID in the PDF trailer
on every ``doc.save()`` — but the content (words, coordinates, drawings) is.

Synthesized content only — no real Consensus Economics data.
All forecaster names and numbers are invented.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# Ground-truth data (synthesized)
# ---------------------------------------------------------------------------

# Main forecaster grid: 6 rows × 4 indicator-year columns.
# Columns: GDP_2024, GDP_2025, CPI_2024, CPI_2025
# One cell is deliberately blank (None → "na") and one row is "ragged"
# (marked in the metadata but still fully specified in the ground truth).

MAIN_TABLE_COLUMNS = ["GDP_2024", "GDP_2025", "CPI_2024", "CPI_2025"]

MAIN_TABLE_ROWS: list[dict] = [
    {
        "label": "Ashford Capital",
        "GDP_2024": "2.1",
        "GDP_2025": "1.9",
        "CPI_2024": "3.4",
        "CPI_2025": "2.8",
    },
    {
        "label": "Brightwater Research",
        "GDP_2024": "2.3",
        "GDP_2025": "na",  # deliberate blank
        "CPI_2024": "3.1",
        "CPI_2025": "2.6",
    },
    {
        "label": "Clearview Economics",
        "GDP_2024": "1.8",
        "GDP_2025": "2.0",
        "CPI_2024": "3.6",
        "CPI_2025": "3.0",
    },
    {
        "label": "Dunmore Analytics",
        "GDP_2024": "2.5",
        "GDP_2025": "2.2",
        "CPI_2024": "3.2",
        "CPI_2025": "na",  # deliberate blank
    },
    {
        "label": "Eastfield Partners",
        "GDP_2024": "2.0",
        "GDP_2025": "1.8",
        "CPI_2024": "3.5",
        "CPI_2025": "2.9",
    },
    {
        # Deliberately ragged / short source row — GDP_2025 and CPI_2024 missing
        "label": "Fenwick Group",
        "GDP_2024": "1.7",
        "GDP_2025": "na",
        "CPI_2024": "na",
        "CPI_2025": "2.7",
        "_ragged": True,
    },
]

# Historical table: different schema — 3 year columns (2021, 2022, 2023).
HIST_TABLE_COLUMNS = ["2021", "2022", "2023"]

HIST_TABLE_ROWS: list[dict] = [
    {
        "label": "GDP Actual",
        "2021": "5.9",
        "2022": "2.1",
        "2023": "2.5",
    },
    {
        "label": "CPI Actual",
        "2021": "4.7",
        "2022": "8.0",
        "2023": "4.1",
    },
    {
        "label": "Unemployment",
        "2021": "5.4",
        "2022": "3.7",
        "2023": "3.5",
    },
]


def build_ground_truth() -> dict:
    """Build the canonical ground-truth dict."""
    main_cells = {}
    for row in MAIN_TABLE_ROWS:
        label = row["label"]
        for col in MAIN_TABLE_COLUMNS:
            main_cells[f"{label}|{col}"] = row[col]

    hist_cells = {}
    for row in HIST_TABLE_ROWS:
        label = row["label"]
        for col in HIST_TABLE_COLUMNS:
            hist_cells[f"{label}|{col}"] = row[col]

    return {
        "version": 1,
        "fixture": "ce_like_p4.pdf",
        "description": (
            "Synthesized CE-p4-like fixture: born-digital, no ruling lines. "
            "Two numeric tables (main forecaster grid + historical data), "
            "one vector chart region, one prose text box."
        ),
        "regions": [
            {
                "id": "main_table",
                "kind": "table",
                "order": 0,
                "schema": "forecaster_grid",
                "columns": MAIN_TABLE_COLUMNS,
                "row_labels": [r["label"] for r in MAIN_TABLE_ROWS],
                "cells": main_cells,
                "ragged_rows": [r["label"] for r in MAIN_TABLE_ROWS if r.get("_ragged")],
                "na_cells": [k for k, v in main_cells.items() if v == "na"],
            },
            {
                "id": "chart",
                "kind": "chart",
                "order": 1,
                "description": "Synthesized Federal Budget chart (vector drawing, no data values)",
                "cells": {},  # no transcribed values — image-asset lane
            },
            {
                "id": "historical_table",
                "kind": "table",
                "order": 2,
                "schema": "historical_data",
                "columns": HIST_TABLE_COLUMNS,
                "row_labels": [r["label"] for r in HIST_TABLE_ROWS],
                "cells": hist_cells,
                "ragged_rows": [],
                "na_cells": [],
            },
            {
                "id": "prose",
                "kind": "text",
                "order": 3,
                "description": "Government and Background Data text box",
                "cells": {},  # prose — no tabular cells
            },
        ],
    }


# ---------------------------------------------------------------------------
# PDF layout constants (points; 1 pt = 1/72 inch)
# Page: US Letter 612 × 792 pt
# ---------------------------------------------------------------------------

PAGE_W = 612.0
PAGE_H = 792.0

# Margins
LEFT = 36.0
RIGHT = PAGE_W - 36.0

# Font sizes
FS_TITLE = 11.0
FS_HEADER = 8.5
FS_CELL = 8.0

# Column positions for main table (label + 4 data columns)
# We use explicit x-coordinates so the PDF has a real text layer with
# well-separated columns (whitespace gutters, NO ruling lines).
MAIN_COL_X = {
    "label": LEFT,
    "GDP_2024": 200.0,
    "GDP_2025": 285.0,
    "CPI_2024": 370.0,
    "CPI_2025": 455.0,
}
MAIN_HDR_Y = 60.0
MAIN_ROW_H = 13.0
MAIN_FIRST_ROW_Y = MAIN_HDR_Y + MAIN_ROW_H + 4.0

# Chart region bounds (drawn below the main table)
CHART_TOP = MAIN_FIRST_ROW_Y + len(MAIN_TABLE_ROWS) * MAIN_ROW_H + 18.0
CHART_BOTTOM = CHART_TOP + 110.0
CHART_LEFT = LEFT
CHART_RIGHT = RIGHT

# Historical table (below chart)
HIST_TOP = CHART_BOTTOM + 18.0
HIST_COL_X = {
    "label": LEFT,
    "2021": 240.0,
    "2022": 330.0,
    "2023": 420.0,
}
HIST_HDR_Y = HIST_TOP
HIST_ROW_H = 13.0
HIST_FIRST_ROW_Y = HIST_HDR_Y + HIST_ROW_H + 4.0

# Prose text box (below historical table)
PROSE_TOP = HIST_FIRST_ROW_Y + len(HIST_TABLE_ROWS) * HIST_ROW_H + 20.0


def generate_pdf(out_path: Path) -> None:
    """Write the fixture PDF to *out_path*."""
    import fitz  # PyMuPDF

    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)

    # ------------------------------------------------------------------
    # Region 0: Main forecaster grid header
    # ------------------------------------------------------------------
    # Column headers (two-line: indicator / year)
    for col, x in MAIN_COL_X.items():
        if col == "label":
            page.insert_text(
                (x, MAIN_HDR_Y),
                "Forecaster",
                fontsize=FS_HEADER,
                fontname="helv",
            )
        else:
            indicator, year = col.split("_")
            page.insert_text(
                (x, MAIN_HDR_Y - 6),
                indicator,
                fontsize=FS_HEADER - 1,
                fontname="helv",
            )
            page.insert_text(
                (x, MAIN_HDR_Y + 4),
                year,
                fontsize=FS_HEADER - 1,
                fontname="helv",
            )

    # Main table rows
    for i, row in enumerate(MAIN_TABLE_ROWS):
        y = MAIN_FIRST_ROW_Y + i * MAIN_ROW_H
        page.insert_text(
            (MAIN_COL_X["label"], y),
            row["label"],
            fontsize=FS_CELL,
            fontname="helv",
        )
        for col in MAIN_TABLE_COLUMNS:
            val = row[col]
            if val != "na":
                page.insert_text(
                    (MAIN_COL_X[col], y),
                    val,
                    fontsize=FS_CELL,
                    fontname="helv",
                )
            # "na" blanks are represented by absence of text — whitespace gap

    # ------------------------------------------------------------------
    # Region 1: Vector chart (draw filled rects + lines so has_chart_marks
    # can detect at least one vector chart cluster)
    # ------------------------------------------------------------------
    # Title above chart
    page.insert_text(
        (LEFT, CHART_TOP - 6),
        "Federal Budget Balance (% of GDP)",
        fontsize=FS_HEADER,
        fontname="helv",
    )

    # Axis lines
    axis_x = CHART_LEFT + 40
    axis_y_bottom = CHART_BOTTOM - 10
    axis_y_top = CHART_TOP + 10

    # Vertical axis
    page.draw_line(
        (axis_x, axis_y_bottom),
        (axis_x, axis_y_top),
        color=(0, 0, 0),
        width=0.5,
    )
    # Horizontal axis
    page.draw_line(
        (axis_x, axis_y_bottom),
        (CHART_RIGHT - 10, axis_y_bottom),
        color=(0, 0, 0),
        width=0.5,
    )

    # Bar chart — 4 bars representing years 2020-2023
    bar_years = ["2020", "2021", "2022", "2023"]
    bar_values = [-3.8, -7.5, -5.2, -4.1]  # synthesized values, not real
    chart_w = CHART_RIGHT - 10 - axis_x
    bar_w = chart_w / (len(bar_years) * 2)
    bar_gap = bar_w
    scale = (axis_y_bottom - axis_y_top - 10) / 10.0  # pts per % unit

    for j, (yr, val) in enumerate(zip(bar_years, bar_values)):
        bx = axis_x + bar_gap + j * (bar_w + bar_gap)
        bar_height = abs(val) * scale
        rect = fitz.Rect(bx, axis_y_bottom - bar_height, bx + bar_w, axis_y_bottom)
        page.draw_rect(rect, color=(0.2, 0.2, 0.8), fill=(0.4, 0.4, 0.9), width=0.3)

        # Year label below bar
        page.insert_text(
            (bx, axis_y_bottom + 8),
            yr,
            fontsize=6,
            fontname="helv",
        )

    # Axis tick labels (so has_chart_marks sees numeric content near lines)
    for tick_val in [0, -2, -4, -6, -8]:
        ty = axis_y_bottom + tick_val * scale
        page.draw_line(
            (axis_x - 3, ty),
            (axis_x, ty),
            color=(0, 0, 0),
            width=0.3,
        )
        page.insert_text(
            (axis_x - 22, ty + 2),
            str(tick_val),
            fontsize=6,
            fontname="helv",
        )

    # ------------------------------------------------------------------
    # Region 2: Historical data table
    # ------------------------------------------------------------------
    # Section title
    page.insert_text(
        (LEFT, HIST_TOP - 6),
        "Historical Data",
        fontsize=FS_HEADER,
        fontname="helv",
    )

    # Column headers
    page.insert_text(
        (HIST_COL_X["label"], HIST_HDR_Y),
        "Indicator",
        fontsize=FS_HEADER,
        fontname="helv",
    )
    for col in HIST_TABLE_COLUMNS:
        page.insert_text(
            (HIST_COL_X[col], HIST_HDR_Y),
            col,
            fontsize=FS_HEADER,
            fontname="helv",
        )

    # Historical data rows
    for i, row in enumerate(HIST_TABLE_ROWS):
        y = HIST_FIRST_ROW_Y + i * HIST_ROW_H
        page.insert_text(
            (HIST_COL_X["label"], y),
            row["label"],
            fontsize=FS_CELL,
            fontname="helv",
        )
        for col in HIST_TABLE_COLUMNS:
            page.insert_text(
                (HIST_COL_X[col], y),
                row[col],
                fontsize=FS_CELL,
                fontname="helv",
            )

    # ------------------------------------------------------------------
    # Region 3: Prose text box
    # ------------------------------------------------------------------
    page.insert_text(
        (LEFT, PROSE_TOP),
        "Government and Background Data",
        fontsize=FS_HEADER,
        fontname="helv",
    )

    prose_lines = [
        "The following table draws on official sources including the IMF, OECD, and",
        "national statistical offices. Forecasts are point estimates as of the survey",
        "date. Blanks indicate data not available or not applicable for that period.",
        "Historical figures may be subject to revision as new data become available.",
    ]
    for k, line in enumerate(prose_lines):
        page.insert_text(
            (LEFT, PROSE_TOP + 12 + k * 11),
            line,
            fontsize=7.5,
            fontname="helv",
        )

    doc.save(str(out_path))
    doc.close()


def main() -> None:
    pdf_path = HERE / "ce_like_p4.pdf"
    gt_path = HERE / "ground_truth.json"

    generate_pdf(pdf_path)
    gt = build_ground_truth()
    gt_path.write_text(json.dumps(gt, indent=2) + "\n")

    print(f"Generated: {pdf_path}")
    print(f"Generated: {gt_path}")

    # Quick sanity-check: the PDF has a text layer
    import fitz

    doc = fitz.open(str(pdf_path))
    page = doc[0]
    words = page.get_text("words")
    doc.close()
    print(f"Text layer word count: {len(words)} (expected >= 60)")
    assert len(words) >= 60, f"Too few words in text layer: {len(words)}"
    print("OK: born-digital text layer confirmed.")


if __name__ == "__main__":
    main()
