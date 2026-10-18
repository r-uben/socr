"""Default-on per-cell table check against born-digital native word geometry.

Uses ``tables.binding.bind()`` as the primary oracle. When column geometry is
lane-mismatched (common on dense French budget tables), falls back to row-anchor
comparison: for each row ``bind()`` already paired, compare x-sorted native
numeric tokens to the candidate row's data cells (order + N2 normalization).

Outcomes (closed set):

* ``verified`` — structural agreement from ``bind()``, or the row-anchor
  fallback found no mismatches on every paired row.
* ``contradicted`` — a contradicted cell, row-label contradiction, or a
  significant (column-bound) native/model unbound cell, or a row-anchor mismatch.
* ``ambiguous`` — insufficient geometry to compare (no paired rows, parse failure).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from socr.tables.binding import BindingResult, bind, parse_grid
from socr.tables.native_verifier import (
    _normalize_binding_numeric_token,
    _normalize_cell,
    is_numeric_token,
)


class NativeCellOutcome(str, Enum):
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class TableCellCheckResult:
    table_id: str
    outcome: NativeCellOutcome
    binding: BindingResult | None = None
    detail: str = ""
    contradicted_samples: list[dict[str, str]] = field(default_factory=list)


def _significant_unbound(result: BindingResult) -> bool:
    return bool([c for c in result.native_unbound if c.col_path]) or bool(
        [c for c in result.model_unbound if c.col_path]
    )


def _active_contradictions(result: BindingResult, markdown: str) -> bool:
    if result.contradicted_cells:
        return True
    if _significant_unbound(result):
        return True
    if not result.row_label_contradictions:
        return False
    if result.column_binding_unverifiable:
        return bool(_row_anchor_mismatch(result, markdown))
    return True


def _row_anchor_mismatch(result: BindingResult, markdown: str) -> list[tuple[str, str, str]]:
    """Return (row_label, native_token, model_token) tuples that disagree."""
    grid = parse_grid(markdown)
    if grid is None or not result.row_binding:
        return []
    mismatches: list[tuple[str, str, str]] = []
    for cand_idx, native_idx in result.row_binding.items():
        if cand_idx >= len(grid.rows) or native_idx >= len(result.native_rows):
            continue
        native_row = result.native_rows[native_idx]
        if native_row.is_parent:
            continue
        cand_row = grid.rows[cand_idx]
        native_nums = [
            _normalize_binding_numeric_token(text)
            for _lane, (text, ambiguous) in sorted(native_row.lane_tokens.items())
            if not ambiguous and is_numeric_token(_normalize_cell(text))
        ]
        model_nums = []
        for cell in cand_row[1:]:
            cell = _normalize_cell(cell.strip())
            if not cell or cell == "-":
                continue
            if is_numeric_token(cell):
                model_nums.append(_normalize_binding_numeric_token(cell))
        stub = cand_row[0].strip() if cand_row else ""
        if not native_nums:
            # Native label wrapped onto the next line — no numeric evidence on this band.
            continue
        if not model_nums:
            mismatches.append((stub, ",".join(native_nums), ""))
            continue
        if len(native_nums) != len(model_nums):
            mismatches.append((stub, ",".join(native_nums), ",".join(model_nums)))
            continue
        for n_tok, m_tok in zip(native_nums, model_nums, strict=True):
            if n_tok != m_tok:
                mismatches.append((stub, n_tok, m_tok))
    return mismatches


def classify_binding_for_cell_check(result: BindingResult, markdown: str) -> NativeCellOutcome:
    """Map one ``bind()`` result to the three consumer outcomes."""
    if _active_contradictions(result, markdown):
        return NativeCellOutcome.CONTRADICTED
    if result.structural_agreement:
        return NativeCellOutcome.VERIFIED
    if result.matched_cells and result.ambiguous_count == 0 and not _significant_unbound(result):
        return NativeCellOutcome.VERIFIED
    mismatches = _row_anchor_mismatch(result, markdown)
    if mismatches:
        return NativeCellOutcome.CONTRADICTED
    if result.row_binding:
        return NativeCellOutcome.VERIFIED
    return NativeCellOutcome.AMBIGUOUS


def check_table_cell_binding(
    *,
    table_id: str,
    words: list,
    markdown: str,
    region_bbox,
    spans: list | None = None,
) -> TableCellCheckResult:
    """Run ``bind()`` and classify for one table region."""
    try:
        binding = bind(words, markdown, region=region_bbox, spans=spans or [])
    except Exception as exc:
        return TableCellCheckResult(
            table_id=table_id,
            outcome=NativeCellOutcome.AMBIGUOUS,
            detail=f"bind failed: {type(exc).__name__}: {exc}",
        )
    outcome = classify_binding_for_cell_check(binding, markdown)
    samples: list[dict[str, str]] = []
    for cell in binding.contradicted_cells[:5]:
        samples.append(
            {
                "row": "/".join(cell.row_path),
                "native": cell.native_token,
                "model": cell.model_token,
            }
        )
    if outcome is NativeCellOutcome.CONTRADICTED and not samples:
        for stub, n_tok, m_tok in _row_anchor_mismatch(binding, markdown)[:5]:
            samples.append({"row": stub, "native": n_tok, "model": m_tok})
    detail = (
        f"outcome={outcome.value}; fully_checked={binding.fully_checked}; "
        f"contradicted={len(binding.contradicted_cells)}; "
        f"matched={len(binding.matched_cells)}; ambiguous={binding.ambiguous_count}"
    )
    return TableCellCheckResult(
        table_id=table_id,
        outcome=outcome,
        binding=binding,
        detail=detail,
        contradicted_samples=samples,
    )


def binding_summary_payload(result: TableCellCheckResult) -> dict[str, Any]:
    """JSON-serializable sidecar fragment for one table."""
    b = result.binding
    payload: dict[str, Any] = {
        "table_id": result.table_id,
        "outcome": result.outcome.value,
        "detail": result.detail,
    }
    if b is not None:
        payload.update(
            {
                "fully_checked": b.fully_checked,
                "structural_agreement": b.structural_agreement,
                "contradicted_cells": len(b.contradicted_cells),
                "row_label_contradictions": len(b.row_label_contradictions),
                "ambiguous_count": b.ambiguous_count,
                "column_binding_unverifiable": b.column_binding_unverifiable,
                "row_binding_unverifiable": b.row_binding_unverifiable,
            }
        )
    if result.contradicted_samples:
        payload["samples"] = list(result.contradicted_samples)
    return payload
