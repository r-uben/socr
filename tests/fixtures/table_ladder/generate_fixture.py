"""Generate the H1 table-judge-ladder e2e fixture: a two-page, born-digital PDF.

This script is SEMANTICALLY idempotent: running it always produces the same
text layer, geometry, and ``ground_truth.json``. The raw PDF bytes are NOT
identical run-to-run (PyMuPDF embeds a random document ID in the PDF trailer
on every ``doc.save()``) but the content (words, coordinates, ruling lines)
is — mirroring ``tests/fixtures/table_repair/generate_fixture.py``'s own
contract.

Synthesized content only — no real data.

Two pages, two ruled (fully boxed) tables, so B0's witness locator finds
exactly one box per page regardless of which candidate markdown a test
supplies for that page:

- Page 1 ("clean_table"): a 3-row x 2-column grid with distinct values, used
  as the ladder's control page — its native words always agree with the
  correct candidate markdown, so the mechanical binding check
  (``tables/binding.py bind()``) never contradicts it.
- Page 2 ("shift_table"): the GH-273 shape — 4 rows x 2 columns, DISTINCT
  per-row value pairs (so a row-label permutation is unambiguous), such that
  a candidate markdown that keeps every value but shifts the row labels by
  one position produces an IDENTICAL value multiset with a WRONG binding —
  exactly the defect two frontier judges both missed (GH-273; see
  ``docs/log/2026-08-30_table-judge-ladder.md``).
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Ground-truth data (synthesized)
# ---------------------------------------------------------------------------

CLEAN_COLUMNS = ["Value"]
CLEAN_ROWS: list[dict] = [
    {"label": "Revenue", "Value": "120"},
    {"label": "Costs", "Value": "45"},
    {"label": "Margin", "Value": "75"},
]

SHIFT_COLUMNS = ["OLS", "IV"]
SHIFT_ROWS: list[dict] = [
    {"label": "RowA", "OLS": "11", "IV": "21"},
    {"label": "RowB", "OLS": "32", "IV": "42"},
    {"label": "RowC", "OLS": "53", "IV": "63"},
    {"label": "RowD", "OLS": "74", "IV": "84"},
]


def build_ground_truth() -> dict:
    """Build the canonical ground-truth dict for both pages."""
    clean_cells = {f"{row['label']}|{col}": row[col] for row in CLEAN_ROWS for col in CLEAN_COLUMNS}
    shift_cells = {f"{row['label']}|{col}": row[col] for row in SHIFT_ROWS for col in SHIFT_COLUMNS}

    return {
        "version": 1,
        "fixture": "binding_shift_doc.pdf",
        "description": (
            "Synthesized two-page fixture for TICKET-H1 (GH-353): page 1 is a "
            "clean, unambiguous ruled table; page 2 reproduces the GH-273 shape "
            "(identical value multiset, row labels permutable without changing "
            "any value) for the mechanical binding check."
        ),
        "regions": [
            {
                "id": "clean_table",
                "page": 1,
                "kind": "table",
                "columns": CLEAN_COLUMNS,
                "row_labels": [r["label"] for r in CLEAN_ROWS],
                "cells": clean_cells,
            },
            {
                "id": "shift_table",
                "page": 2,
                "kind": "table",
                "columns": SHIFT_COLUMNS,
                "row_labels": [r["label"] for r in SHIFT_ROWS],
                "cells": shift_cells,
                # A row-label rotation (each row keeps its own values, but is
                # relabeled with the NEXT row's label) — every value from
                # ``cells`` is still present, only the (row, value) pairing
                # changes. This is the exact "shifted" candidate markdown the
                # e2e tests feed the pipeline to trigger the mechanical check.
                "shifted_row_label_order": [
                    SHIFT_ROWS[(i + 1) % len(SHIFT_ROWS)]["label"] for i in range(len(SHIFT_ROWS))
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# PDF layout constants (points; 1 pt = 1/72 inch)
# ---------------------------------------------------------------------------

LEFT = 100.0
COL_W = 120.0
ROW_H = 22.0
TABLE_TOP = 100.0


def _draw_ruled_table(page, columns: list[str], rows: list[dict], top: float) -> None:
    """Draw a ruled (fully boxed) table: a label column + one column per
    entry in ``columns``, one row per entry in ``rows``, plus a header row."""
    import fitz

    n_cols = 1 + len(columns)
    col_xs = [LEFT + i * COL_W for i in range(n_cols + 1)]
    row_ys = [top + i * ROW_H for i in range(len(rows) + 2)]  # +1 header, +1 closing edge

    # Header row
    page.insert_text((col_xs[0] + 4, row_ys[0] + 15), "Label", fontsize=9)
    for c, col in enumerate(columns):
        page.insert_text((col_xs[c + 1] + 4, row_ys[0] + 15), col, fontsize=9)

    # Data rows
    for r, row in enumerate(rows):
        y = row_ys[r + 1]
        page.insert_text((col_xs[0] + 4, y + 15), row["label"], fontsize=9)
        for c, col in enumerate(columns):
            page.insert_text((col_xs[c + 1] + 4, y + 15), row[col], fontsize=9)

    # Ruling lines: horizontal for every row boundary, vertical for every column.
    for yy in row_ys:
        page.draw_line((col_xs[0], yy), (col_xs[-1], yy))
    for xx in col_xs:
        page.draw_line((xx, row_ys[0]), (xx, row_ys[-1]))


def generate_pdf(out_path: Path) -> None:
    """Write the two-page fixture PDF to *out_path*."""
    import fitz  # PyMuPDF

    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((LEFT, TABLE_TOP - 12), "Clean table (control page)", fontsize=10)
    _draw_ruled_table(page1, CLEAN_COLUMNS, CLEAN_ROWS, TABLE_TOP)

    page2 = doc.new_page()
    page2.insert_text((LEFT, TABLE_TOP - 12), "Shift table (GH-273 shape)", fontsize=10)
    _draw_ruled_table(page2, SHIFT_COLUMNS, SHIFT_ROWS, TABLE_TOP)

    doc.save(str(out_path))
    doc.close()


def main() -> None:
    pdf_path = HERE / "binding_shift_doc.pdf"
    gt_path = HERE / "ground_truth.json"

    generate_pdf(pdf_path)
    gt = build_ground_truth()
    gt_path.write_text(json.dumps(gt, indent=2) + "\n")

    print(f"Generated: {pdf_path}")
    print(f"Generated: {gt_path}")

    # Sanity-check: both pages have a real, ruled, born-digital text layer.
    import fitz

    doc = fitz.open(str(pdf_path))
    assert doc.page_count == 2, f"Expected 2 pages, got {doc.page_count}"
    for i in range(2):
        words = doc[i].get_text("words")
        assert len(words) >= 6, f"page {i + 1}: too few words in text layer: {len(words)}"
        tables = doc[i].find_tables()
        assert len(tables.tables) >= 1, f"page {i + 1}: find_tables() found no ruled table"
    doc.close()
    print("OK: two-page born-digital, ruled-table fixture confirmed.")


if __name__ == "__main__":
    main()
