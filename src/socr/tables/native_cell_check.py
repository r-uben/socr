"""Default-on per-cell table check against born-digital native word geometry.

Uses ``tables.binding.bind()`` as the primary oracle. When column geometry is
lane-mismatched (common on dense French budget tables), falls back to row-anchor
comparison on ``bind()``'s ``row_binding`` map only (no substring label guessing).

Outcomes (closed set):

* ``verified`` — ``structural_agreement``, or a documented French-table fallback
  with real per-row numeric comparisons and no anchor mismatches.
* ``contradicted`` — contradicted cells, row-label contradictions (when columns
  are verifiable), or row-anchor mismatches.
* ``ambiguous`` — any unbound cell (when columns are verifiable), incomplete
  coverage, ambiguous locale, or insufficient comparisons.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from socr.tables.binding import BindingResult, UnboundCell, bind, parse_grid
from socr.tables.native_verifier import label_key
from socr.tables.native_verifier import (
    _THOUSANDS_SPACE_RE,
    _normalize_cell,
    binding_numeric_tokens_equivalent,
    is_numeric_token,
    strip_presentation,
)

_YEAR_HEADER_RE = re.compile(r"^20\d{2}$")
_PAGE_NUMBER_RE = re.compile(r"^\d{1,3}$")


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


def _prepare_numeric_cell(raw: str) -> str:
    """Presentation + interior thousand-space strip before numeric classification."""
    cell = _normalize_cell(strip_presentation(raw.strip()))
    return _THOUSANDS_SPACE_RE.sub("", cell)


def _is_year_header_row(cand_row: list[str]) -> bool:
    stub = (cand_row[0] if cand_row else "").strip()
    if stub and not _YEAR_HEADER_RE.match(stub):
        return False
    data = [_prepare_numeric_cell(c) for c in cand_row[1:] if c.strip() and c.strip() != "-"]
    return bool(data) and all(_YEAR_HEADER_RE.match(c) for c in data)


def _model_row_numeric_tokens(cand_row: list[str]) -> list[str]:
    out: list[str] = []
    for cell in cand_row[1:]:
        cell = _prepare_numeric_cell(cell)
        if not cell or cell == "-":
            continue
        if is_numeric_token(cell):
            out.append(cell)
    return out


def _native_row_numeric_tokens(native_row) -> list[str]:
    return [
        text
        for _lane, (text, ambiguous) in sorted(native_row.lane_tokens.items())
        if not ambiguous and is_numeric_token(_prepare_numeric_cell(text))
    ]


def _model_row_slots(cand_row: list[str]) -> list[str | None]:
    slots: list[str | None] = []
    for cell in cand_row[1:]:
        cell = _prepare_numeric_cell(cell)
        if not cell or cell == "-":
            slots.append(None)
        elif is_numeric_token(cell):
            slots.append(cell)
        else:
            slots.append("__non_numeric__")
    return slots


def _native_row_slots(native_row) -> list[str | None]:
    slots: list[str | None] = []
    for _lane, (text, ambiguous) in sorted(native_row.lane_tokens.items()):
        if ambiguous:
            return []
        text = _prepare_numeric_cell(text)
        if not text or text == "-":
            slots.append(None)
        elif is_numeric_token(text):
            slots.append(text)
        else:
            slots.append("__non_numeric__")
    return slots


def _slots_equivalent(
    native_slots: list[str | None],
    model_slots: list[str | None],
) -> bool | None:
    if len(native_slots) != len(model_slots):
        return False
    for n_raw, m_raw in zip(native_slots, model_slots, strict=True):
        if n_raw is None and m_raw is None:
            continue
        if n_raw is None or m_raw is None:
            return False
        if n_raw == "__non_numeric__" or m_raw == "__non_numeric__":
            if n_raw != m_raw:
                return False
            continue
        equiv = binding_numeric_tokens_equivalent(n_raw, m_raw)
        if equiv is None:
            return None
        if not equiv:
            return False
    return True


def _compare_bound_row(
    native_row,
    cand_row: list[str],
    *,
    column_binding_unverifiable: bool = False,
) -> bool | None:
    """``True``/``False`` on agreement; ``None`` = no comparison or ambiguous locale."""
    model_slots = _model_row_slots(cand_row)
    native_slots = _native_row_slots(native_row)
    if model_slots and native_slots and len(model_slots) == len(native_slots):
        return _slots_equivalent(native_slots, model_slots)
    model_nums = _model_row_numeric_tokens(cand_row)
    native_nums = _native_row_numeric_tokens(native_row)
    if not native_nums:
        return None
    if not model_nums:
        return False
    if len(native_nums) != len(model_nums):
        return False
    for n_raw, m_raw in zip(native_nums, model_nums, strict=True):
        equiv = binding_numeric_tokens_equivalent(n_raw, m_raw)
        if equiv is None:
            return None
        if not equiv:
            return False
    return True


def _row_anchor_scan(
    result: BindingResult, markdown: str
) -> tuple[list[tuple[str, str, str]], int]:
    """Return (mismatches, successful_numeric_comparisons)."""
    grid = parse_grid(markdown)
    if grid is None:
        return [("", "", "parse_failed")], 0
    mismatches: list[tuple[str, str, str]] = []
    comparisons = 0

    for cand_idx, native_idx in result.row_binding.items():
        if cand_idx >= len(grid.rows) or native_idx >= len(result.native_rows):
            continue
        cand_row = grid.rows[cand_idx]
        if _is_year_header_row(cand_row):
            continue
        native_row = result.native_rows[native_idx]
        if native_row.is_parent:
            continue
        stub = cand_row[0].strip() if cand_row else ""
        if result.column_binding_unverifiable:
            native_label = native_row.row_path[-1].strip() if native_row.row_path else ""
            cand_label = stub
            if label_key(cand_label) != label_key(native_label):
                model_nums = _model_row_numeric_tokens(cand_row)
                native_nums = _native_row_numeric_tokens(native_row)
                if model_nums and native_nums and len(model_nums) == len(native_nums):
                    equiv = all(
                        binding_numeric_tokens_equivalent(n, m) is True
                        for n, m in zip(native_nums, model_nums, strict=True)
                    )
                    if equiv:
                        mismatches.append((stub, native_label, cand_label))
                        continue
        cmp = _compare_bound_row(
            native_row,
            cand_row,
            column_binding_unverifiable=result.column_binding_unverifiable,
        )
        if cmp is True and _native_row_numeric_tokens(native_row):
            model_slots = _model_row_slots(cand_row)
            native_slots = _native_row_slots(native_row)
            slot_aligned = model_slots and native_slots and len(model_slots) == len(native_slots)
            if (
                result.column_binding_unverifiable
                and model_slots
                and native_slots
                and not slot_aligned
            ):
                pass
            else:
                comparisons += 1
        elif cmp is False:
            mismatches.append(
                (
                    stub,
                    ",".join(_native_row_numeric_tokens(native_row)),
                    ",".join(_model_row_numeric_tokens(cand_row)),
                )
            )
        elif (
            cmp is None
            and _native_row_numeric_tokens(native_row)
            and _model_row_numeric_tokens(cand_row)
        ):
            mismatches.append((stub, "ambiguous_locale", "ambiguous_locale"))

    bound_cand = set(result.row_binding.keys())
    for cand_idx, cand_row in enumerate(grid.rows):
        if cand_idx in bound_cand or _is_year_header_row(cand_row):
            continue
        if not any(
            is_numeric_token(_prepare_numeric_cell(c))
            for c in cand_row[1:]
            if c.strip() and c.strip() != "-"
        ):
            continue
        stub = cand_row[0].strip() if cand_row else ""
        mismatches.append((stub, "native_unbound", "model_data_row"))

    return mismatches, comparisons


def _native_unbound_cell_is_excludable(cell: UnboundCell) -> bool:
    """Only year column headers, page numbers, and footnote bands are ignorable."""
    for seg in cell.row_path:
        if seg.strip().startswith("**"):
            return True
    tok = _prepare_numeric_cell(cell.token.strip())
    if not tok or not is_numeric_token(tok):
        return False
    if not cell.col_path:
        year_core = tok.rstrip(",.;:")
        if _YEAR_HEADER_RE.match(year_core):
            return True
        if _PAGE_NUMBER_RE.match(year_core) and len(cell.row_path) <= 1:
            return True
    return False


def _numeric_unbound_blocks_verify(result: BindingResult) -> bool:
    """Any unmatched side carrying a numeric token blocks VERIFIED."""
    for cell in result.model_unbound:
        if is_numeric_token(_prepare_numeric_cell(cell.token)):
            return True
    for cell in result.native_unbound:
        if _native_unbound_cell_is_excludable(cell):
            continue
        tok = _prepare_numeric_cell(cell.token)
        if is_numeric_token(tok):
            return True
    return False


def _row_binding_covers_model_data_rows(result: BindingResult, markdown: str) -> bool:
    grid = parse_grid(markdown)
    if grid is None:
        return False
    for cand_idx, cand_row in enumerate(grid.rows):
        if _is_year_header_row(cand_row):
            continue
        if not any(
            is_numeric_token(_prepare_numeric_cell(c))
            for c in cand_row[1:]
            if c.strip() and c.strip() != "-"
        ):
            continue
        if cand_idx not in result.row_binding:
            return False
    return True


def classify_binding_for_cell_check(result: BindingResult, markdown: str) -> NativeCellOutcome:
    mismatches, comparisons = _row_anchor_scan(result, markdown)
    if result.contradicted_cells:
        return NativeCellOutcome.CONTRADICTED
    if result.row_label_contradictions and not result.column_binding_unverifiable:
        return NativeCellOutcome.CONTRADICTED
    hard_mismatches = [m for m in mismatches if m[1] != "ambiguous_locale"]
    if hard_mismatches:
        return NativeCellOutcome.CONTRADICTED
    if result.ambiguous_count and not result.column_binding_unverifiable:
        return NativeCellOutcome.AMBIGUOUS
    if _numeric_unbound_blocks_verify(result):
        return NativeCellOutcome.AMBIGUOUS
    if result.structural_agreement:
        return NativeCellOutcome.VERIFIED
    if _row_binding_covers_model_data_rows(result, markdown) and comparisons > 0:
        return NativeCellOutcome.VERIFIED
    if mismatches:
        return NativeCellOutcome.AMBIGUOUS
    return NativeCellOutcome.AMBIGUOUS


def check_table_cell_binding(
    *,
    table_id: str,
    words: list,
    markdown: str,
    region_bbox,
    spans: list | None = None,
) -> TableCellCheckResult:
    if not words:
        return TableCellCheckResult(
            table_id=table_id,
            outcome=NativeCellOutcome.AMBIGUOUS,
            detail="no native words in region",
        )
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
        mismatches, _ = _row_anchor_scan(binding, markdown)
        for stub, n_tok, m_tok in mismatches[:5]:
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


def skipped_check_result(table_id: str, reason: str) -> TableCellCheckResult:
    return TableCellCheckResult(
        table_id=table_id,
        outcome=NativeCellOutcome.AMBIGUOUS,
        detail=f"skipped: {reason}",
    )


def binding_summary_payload(result: TableCellCheckResult) -> dict[str, Any]:
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
