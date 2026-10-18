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
_FOOTNOTE_PAREN_NUM_RE = re.compile(r"^\(\d+\)")
_DATA_ROW_LABEL_RE = re.compile(
    r"(?i)\b(Total|Sous-total|Solde|Ensemble|Recettes|Dépenses|Budget|Effectifs|Impôts|Taxes|Autres)\b"
)
_FOOTNOTE_LABEL_PREFIXES = (
    "Source :",
    "Source:",
    "Note :",
    "Note:",
    "Lecture :",
    "Lecture:",
    "Champ :",
    "Champ:",
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


def _prepare_numeric_cell(raw: str) -> str:
    """Presentation + interior thousand-space strip before numeric classification."""
    cell = _normalize_cell(strip_presentation(raw.strip()))
    return _THOUSANDS_SPACE_RE.sub("", cell)


def _is_year_header_row(cand_row: list[str]) -> bool:
    """Column-header row only: empty stub and every data cell is a year token."""
    stub = (cand_row[0] if cand_row else "").strip()
    if stub:
        return False
    data = [_prepare_numeric_cell(c) for c in cand_row[1:] if c.strip() and c.strip() != "-"]
    return bool(data) and all(_YEAR_HEADER_RE.match(c) for c in data)


def _row_has_numeric_data(cand_row: list[str]) -> bool:
    return bool(_model_row_numeric_tokens(cand_row))


def _bound_rows_requiring_coverage(result: BindingResult, cand_rows: list[list[str]]) -> list[int]:
    """Every bound candidate row that carries numeric model content."""
    out: list[int] = []
    for cand_idx in result.row_binding:
        if cand_idx >= len(cand_rows):
            continue
        cand_row = cand_rows[cand_idx]
        if _row_has_numeric_data(cand_row):
            out.append(cand_idx)
    return out


def _model_row_numeric_tokens(cand_row: list[str]) -> list[str]:
    out: list[str] = []
    for cell in cand_row[1:]:
        cell = _prepare_numeric_cell(cell)
        if not cell or cell == "-":
            continue
        if is_numeric_token(cell):
            out.append(cell)
    return out


def _native_row_numeric_tokens(native_row, *, include_ambiguous_lanes: bool = False) -> list[str]:
    return [
        text
        for _lane, (text, ambiguous) in sorted(native_row.lane_tokens.items())
        if (include_ambiguous_lanes or not ambiguous)
        and is_numeric_token(_prepare_numeric_cell(text))
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


def _native_row_slots(native_row, *, include_ambiguous_lanes: bool = False) -> list[str | None]:
    slots: list[str | None] = []
    for _lane, (text, ambiguous) in sorted(native_row.lane_tokens.items()):
        if ambiguous and not include_ambiguous_lanes:
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
    incl = column_binding_unverifiable
    model_slots = _model_row_slots(cand_row)
    native_slots = _native_row_slots(native_row, include_ambiguous_lanes=incl)
    if model_slots and native_slots and len(model_slots) == len(native_slots):
        slot_cmp = _slots_equivalent(native_slots, model_slots)
        if slot_cmp is not None or not incl:
            return slot_cmp
    model_nums = _model_row_numeric_tokens(cand_row)
    native_nums = _native_row_numeric_tokens(native_row, include_ambiguous_lanes=incl)
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


def _candidate_rows_for_scan(result: BindingResult, markdown: str) -> list[list[str]] | None:
    if result.candidate_rows:
        return [list(row) for row in result.candidate_rows]
    grid = parse_grid(markdown)
    if grid is None:
        return None
    rows = list(grid.rows)
    if grid.spacer_row_indices:
        rows = [row for i, row in enumerate(rows) if i not in grid.spacer_row_indices]
    return rows


_bound_data_row_indices = _bound_rows_requiring_coverage


def _native_row_label_text(native_row) -> str:
    return " ".join(p.strip() for p in native_row.row_path if p and p.strip())


def _row_path_is_footnote_prose(row_path: tuple[str, ...]) -> bool:
    if not row_path:
        return False
    head = row_path[0].strip()
    if head.startswith("**") and not head.startswith("** dont"):
        return True
    for seg in row_path:
        if _native_leaf_is_excludable_footnote_label(seg.strip()):
            return True
    return False


def _native_leaf_is_excludable_footnote_label(leaf: str) -> bool:
    """True only for prose footnote markers, never data subtotals/totals."""
    if not leaf.strip():
        return False
    if _DATA_ROW_LABEL_RE.search(leaf):
        return False
    if leaf.startswith("** dont"):
        return False
    if leaf.startswith("**") or (leaf.startswith("*") and not leaf.startswith("**")):
        return True
    if _FOOTNOTE_PAREN_NUM_RE.match(leaf.strip()):
        return True
    for pref in _FOOTNOTE_LABEL_PREFIXES:
        if leaf.strip().startswith(pref):
            return True
    return False


def _label_key_literal(key) -> str:
    if not key:
        return ""
    return key[0][1] if key[0] else ""


def _is_truncated_label_variant(full_label: str, partial_label: str) -> bool:
    full_k = _label_key_literal(label_key(full_label))
    part_k = _label_key_literal(label_key(partial_label))
    if not full_k or not part_k:
        return False
    return full_k.startswith(part_k) or part_k.startswith(full_k)


def _labels_disagree_under_unverifiable_columns(native_row, cand_label: str) -> bool:
    if not cand_label.strip():
        return False
    cand_k = label_key(cand_label)
    if not cand_k:
        return False
    if native_row.row_path:
        last = native_row.row_path[-1].strip()
        if last and _is_truncated_label_variant(cand_label, last):
            return False
        joined = _native_row_label_text(native_row)
        if joined and _is_truncated_label_variant(cand_label, joined):
            return False
    native_keys: list = []
    if native_row.row_path:
        last = native_row.row_path[-1].strip()
        if last:
            native_keys.append(label_key(last))
        joined = _native_row_label_text(native_row)
        if joined and joined != last:
            native_keys.append(label_key(joined))
    return all(cand_k != k for k in native_keys if k)


def _native_row_index_for_unbound(cell: UnboundCell, result: BindingResult) -> int | None:
    for idx, native_row in enumerate(result.native_rows):
        if native_row.row_path == cell.row_path:
            return idx
    return None


def _table_bound_data_y_max(result: BindingResult) -> float | None:
    bound_native = set(result.row_binding.values())
    y_maxs: list[float] = []
    for idx, native_row in enumerate(result.native_rows):
        if idx not in bound_native or native_row.is_parent:
            continue
        if native_row.label_bbox is not None:
            y_maxs.append(native_row.label_bbox[3])
        for bbox in native_row.lane_bboxes.values():
            y_maxs.append(bbox[3])
    if not y_maxs:
        return None
    return max(y_maxs)


def _table_bound_data_y_min(result: BindingResult) -> float | None:
    bound_native = set(result.row_binding.values())
    y_mins: list[float] = []
    for idx, native_row in enumerate(result.native_rows):
        if idx not in bound_native or native_row.is_parent:
            continue
        if native_row.label_bbox is not None:
            y_mins.append(native_row.label_bbox[1])
        for bbox in native_row.lane_bboxes.values():
            y_mins.append(bbox[1])
    if not y_mins:
        return None
    return min(y_mins)


def _median_bound_row_pitch(result: BindingResult) -> float | None:
    bound_native = set(result.row_binding.values())
    ys: list[float] = []
    for idx, native_row in enumerate(result.native_rows):
        if idx not in bound_native or native_row.is_parent:
            continue
        if native_row.label_bbox is not None:
            ys.append((native_row.label_bbox[1] + native_row.label_bbox[3]) / 2.0)
    ys.sort()
    if len(ys) < 2:
        return None
    diffs = [ys[i + 1] - ys[i] for i in range(len(ys) - 1) if ys[i + 1] - ys[i] > 1.0]
    if not diffs:
        return None
    diffs.sort()
    mn = diffs[0]
    core = [d for d in diffs if d <= max(mn * 2.5, mn + 12.0)]
    if not core:
        core = diffs
    q1_idx = min(len(core) - 1, max(0, len(core) // 4))
    return core[q1_idx]


def _footnote_separation_threshold(result: BindingResult) -> float:
    pitch = _median_bound_row_pitch(result)
    if pitch is None:
        return 40.0
    return max(pitch * 2.5, pitch + 8.0)


def _is_whole_unbound_native_row(cell: UnboundCell, result: BindingResult) -> bool:
    idx = _native_row_index_for_unbound(cell, result)
    if idx is None:
        return False
    return idx not in set(result.row_binding.values())


def _bbox_outside_region(
    bbox: tuple[float, float, float, float],
    region: tuple[float, float, float, float],
    *,
    margin: float = 1.0,
) -> bool:
    x0, y0, x1, y1 = region
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return cx < x0 - margin or cx > x1 + margin or cy < y0 - margin or cy > y1 + margin


def _native_unbound_cell_is_excludable(
    cell: UnboundCell,
    result: BindingResult,
    region: tuple[float, float, float, float] | None,
) -> bool:
    """Position-only: margin/outside region, header band, or footnote band below data."""
    bbox = cell.token_bbox
    if bbox is None:
        return False
    if region is not None:
        try:
            box = tuple(float(v) for v in region)
        except (TypeError, ValueError):
            box = None
        if box is not None and len(box) == 4 and _bbox_outside_region(bbox, box):
            return True
    if _is_whole_unbound_native_row(cell, result):
        if not cell.col_path:
            tok = _prepare_numeric_cell(cell.token)
            if _YEAR_HEADER_RE.match(tok):
                data_y = result.native_data_y_min
                header_y = result.native_header_band_y_max
                if header_y is not None and bbox[3] <= header_y:
                    return True
                if data_y is not None and bbox[3] < data_y - 1.5:
                    return True
        if _row_path_is_footnote_prose(cell.row_path):
            return True
        leaf = cell.row_path[-1].strip() if cell.row_path else ""
        if _native_leaf_is_excludable_footnote_label(leaf):
            return True
        return False
    header_y = result.native_header_band_y_max
    data_y = result.native_data_y_min
    if not cell.col_path and header_y is not None and bbox[3] <= header_y:
        return True
    if not cell.col_path and data_y is not None and bbox[3] < data_y - 1.5:
        return True
    return False


def _row_anchor_scan(
    result: BindingResult, markdown: str
) -> tuple[list[tuple[str, str, str]], int, dict[int, int]]:
    """Return (mismatches, successful_numeric_comparisons, comparisons_per_cand_row)."""
    cand_rows = _candidate_rows_for_scan(result, markdown)
    if cand_rows is None:
        return [("", "", "parse_failed")], 0, {}
    mismatches: list[tuple[str, str, str]] = []
    comparisons = 0
    per_row: dict[int, int] = {}

    for cand_idx, native_idx in result.row_binding.items():
        if cand_idx >= len(cand_rows) or native_idx >= len(result.native_rows):
            continue
        cand_row = cand_rows[cand_idx]
        native_row = result.native_rows[native_idx]
        if native_row.is_parent:
            continue
        stub = cand_row[0].strip() if cand_row else ""
        cmp = _compare_bound_row(
            native_row,
            cand_row,
            column_binding_unverifiable=result.column_binding_unverifiable,
        )
        if cmp is True and _native_row_numeric_tokens(
            native_row, include_ambiguous_lanes=result.column_binding_unverifiable
        ):
            comparisons += 1
            per_row[cand_idx] = per_row.get(cand_idx, 0) + 1
            if result.column_binding_unverifiable and _labels_disagree_under_unverifiable_columns(
                native_row, stub
            ):
                model_nums = _model_row_numeric_tokens(cand_row)
                native_nums = _native_row_numeric_tokens(native_row, include_ambiguous_lanes=True)
                if model_nums and native_nums and len(model_nums) == len(native_nums):
                    equiv = all(
                        binding_numeric_tokens_equivalent(n, m) is True
                        for n, m in zip(native_nums, model_nums, strict=True)
                    )
                    if equiv:
                        native_label = _native_row_label_text(native_row)
                        mismatches.append((stub, native_label, stub))
            continue
        if result.column_binding_unverifiable and _labels_disagree_under_unverifiable_columns(
            native_row, stub
        ):
            model_nums = _model_row_numeric_tokens(cand_row)
            native_nums = _native_row_numeric_tokens(native_row, include_ambiguous_lanes=True)
            if model_nums and native_nums and len(model_nums) == len(native_nums):
                equiv = all(
                    binding_numeric_tokens_equivalent(n, m) is True
                    for n, m in zip(native_nums, model_nums, strict=True)
                )
                if equiv:
                    native_label = _native_row_label_text(native_row)
                    mismatches.append((stub, native_label, stub))
                    continue
        if cmp is False:
            mismatches.append(
                (
                    stub,
                    ",".join(_native_row_numeric_tokens(native_row)),
                    ",".join(_model_row_numeric_tokens(cand_row)),
                )
            )
        elif (
            cmp is None
            and _native_row_numeric_tokens(
                native_row, include_ambiguous_lanes=result.column_binding_unverifiable
            )
            and _model_row_numeric_tokens(cand_row)
        ):
            mismatches.append((stub, "ambiguous_locale", "ambiguous_locale"))

    bound_cand = set(result.row_binding.keys())
    for cand_idx, cand_row in enumerate(cand_rows):
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

    return mismatches, comparisons, per_row


def _numeric_unbound_blocks_verify(
    result: BindingResult,
    region: tuple[float, float, float, float] | None,
) -> bool:
    """Any unmatched side carrying a numeric token blocks VERIFIED."""
    for cell in result.model_unbound:
        if is_numeric_token(_prepare_numeric_cell(cell.token)):
            return True
    for cell in result.native_unbound:
        if _native_unbound_cell_is_excludable(cell, result, region):
            continue
        tok = _prepare_numeric_cell(cell.token)
        if is_numeric_token(tok):
            return True
    return False


def _fallback_verified_allowed(
    result: BindingResult,
    markdown: str,
    region: tuple[float, float, float, float] | None,
    *,
    comparisons_per_row: dict[int, int],
) -> bool:
    if _numeric_unbound_blocks_verify(result, region):
        return False
    cand_rows = _candidate_rows_for_scan(result, markdown)
    if cand_rows is None:
        return False
    data_rows = _bound_rows_requiring_coverage(result, cand_rows)
    if not data_rows:
        return False
    return all(comparisons_per_row.get(idx, 0) >= 1 for idx in data_rows)


def classify_binding_for_cell_check(
    result: BindingResult,
    markdown: str,
    *,
    region: tuple[float, float, float, float] | None = None,
) -> NativeCellOutcome:
    if result.contradicted_cells:
        return NativeCellOutcome.CONTRADICTED
    if result.row_label_contradictions and not result.column_binding_unverifiable:
        return NativeCellOutcome.CONTRADICTED

    if result.structural_agreement:
        return NativeCellOutcome.VERIFIED

    mismatches, comparisons, per_row = _row_anchor_scan(result, markdown)
    if any(m[2] == "parse_failed" for m in mismatches):
        return NativeCellOutcome.AMBIGUOUS

    hard_mismatches = [
        m for m in mismatches if m[1] != "ambiguous_locale" and m[2] != "parse_failed"
    ]
    if hard_mismatches:
        return NativeCellOutcome.CONTRADICTED
    if result.ambiguous_count and not result.column_binding_unverifiable:
        return NativeCellOutcome.AMBIGUOUS
    if _numeric_unbound_blocks_verify(result, region):
        return NativeCellOutcome.AMBIGUOUS
    if _fallback_verified_allowed(result, markdown, region, comparisons_per_row=per_row):
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
    region_tuple: tuple[float, float, float, float] | None = None
    if region_bbox is not None:
        try:
            region_tuple = tuple(float(v) for v in region_bbox)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            region_tuple = None
    try:
        binding = bind(words, markdown, region=region_tuple, spans=spans or [])
    except Exception as exc:
        return TableCellCheckResult(
            table_id=table_id,
            outcome=NativeCellOutcome.AMBIGUOUS,
            detail=f"bind failed: {type(exc).__name__}: {exc}",
        )
    outcome = classify_binding_for_cell_check(binding, markdown, region=region_tuple)
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
        mismatches, _, _ = _row_anchor_scan(binding, markdown)
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
