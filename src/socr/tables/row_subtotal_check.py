"""Advisory five-year subtotal checks for fiscal markdown tables.

JCT/CBO revenue tables often expose 2004..2008 plus a ``2004-08`` five-year
subtotal column. This module checks whether that subtotal matches the sum of
the five year columns within a rounding tolerance (values are in millions).

Advisory only: rounding, ``---`` negligible cells, spanned text rows, and
panel-specific layouts produce frequent false positives, so callers must not
gate shipping on this alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from socr.core.table_grid import markdown_table_cells
from socr.tables.native_verifier import _normalize_numeric_token, is_numeric_token

_YEAR_RE = re.compile(r"^20\d{2}$")
_RANGE_5Y_RE = re.compile(r"^20\d{2}-20\d{2}$")


@dataclass(frozen=True)
class SubtotalAdvisory:
    row_label: str
    detail: str
    delta: float | None = None


def _parse_numeric_cell(cell: str) -> float | None:
    cell = cell.strip()
    if not cell or cell in ("---", "—", "-", "–"):
        return 0.0
    if not is_numeric_token(cell):
        return None
    norm = _normalize_numeric_token(cell)
    if norm is None:
        return None
    try:
        return float(norm.replace(",", ""))
    except ValueError:
        return None


def advise_five_year_subtotals(markdown: str, *, tolerance: float = 2.0) -> list[SubtotalAdvisory]:
    """Return subtotal mismatches for each markdown table block in *markdown*."""
    advisories: list[SubtotalAdvisory] = []
    for block in markdown.split("\n\n"):
        if "|" not in block or "---" not in block:
            continue
        rows = markdown_table_cells(block)
        if len(rows) < 3:
            continue
        header = [c.strip() for c in rows[0]]
        year_cols: list[tuple[int, str]] = []
        subtotal_cols: list[tuple[int, str]] = []
        for idx, label in enumerate(header):
            lab = label.replace("**", "").strip()
            if _YEAR_RE.match(lab):
                year_cols.append((idx, lab))
            elif _RANGE_5Y_RE.match(lab):
                subtotal_cols.append((idx, lab))
        if len(year_cols) < 5 or not subtotal_cols:
            continue
        year_cols.sort(key=lambda t: t[1])
        five = year_cols[:5]
        for sub_idx, sub_label in subtotal_cols:
            for row in rows[2:]:
                if not any(cell.strip() for cell in row):
                    continue
                label = row[0].strip() if row else ""
                nums = [_parse_numeric_cell(row[i] if i < len(row) else "") for i, _ in five]
                if any(n is None for n in nums):
                    continue
                sub_val = _parse_numeric_cell(row[sub_idx] if sub_idx < len(row) else "")
                if sub_val is None:
                    continue
                total = sum(nums)  # type: ignore[arg-type]
                delta = abs(total - sub_val)
                if delta > tolerance:
                    advisories.append(
                        SubtotalAdvisory(
                            row_label=label[:80],
                            detail=(
                                f"{sub_label}: row sum {total:.0f} vs subtotal "
                                f"{sub_val:.0f} (delta {delta:.0f}, tol {tolerance:.0f})"
                            ),
                            delta=delta,
                        )
                    )
    return advisories
