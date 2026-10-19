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
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from socr.tables.binding import (
    BindingResult,
    UnboundCell,
    _cell_binding_numeric_tokens,
    bind,
    parse_grid,
)
from socr.tables.native_rows import normalize_label
from socr.tables.native_verifier import label_key
from socr.tables.native_verifier import (
    _THOUSANDS_SPACE_RE,
    _normalize_binding_numeric_token,
    _normalize_cell,
    binding_numeric_tokens_equivalent,
    is_numeric_token,
    strip_presentation,
)

_YEAR_HEADER_RE = re.compile(r"^20\d{2}$")
_FOOTNOTE_PAREN_NUM_RE = re.compile(r"^\(\d{1,2}\)$")
_LABEL_FOOTNOTE_MARK_RE = re.compile(r"\(\d{1,2}\)\s*$")
_STUB_HEADER_LABELS = frozenset({"libellé", "libelle", "label", "poste"})
_LATEX_TABULAR_BEGIN_RE = re.compile(r"\\begin\{(?:tabular\*?|array)\}", re.IGNORECASE)
_LATEX_TABULAR_END_RE = re.compile(r"\\end\{(?:tabular\*?|array)\}", re.IGNORECASE)


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
    if cell.startswith("+") and len(cell) > 1 and cell[1].isdigit():
        cell = cell[1:]
    return _THOUSANDS_SPACE_RE.sub("", cell)


def _is_year_header_row(cand_row: list[str]) -> bool:
    """Column-header row only: empty stub and every data cell is a year token."""
    stub = (cand_row[0] if cand_row else "").strip()
    if stub:
        return False
    data = [_prepare_numeric_cell(c) for c in cand_row[1:] if c.strip() and c.strip() != "-"]
    return bool(data) and all(_YEAR_HEADER_RE.match(c) for c in data)


def _is_column_header_row(cand_row: list[str]) -> bool:
    """Header band row: optional Libellé stub and year tokens in data lanes."""
    if not cand_row:
        return False
    stub = cand_row[0].strip()
    if stub and normalize_label(stub) not in _STUB_HEADER_LABELS:
        return False
    data = [_prepare_numeric_cell(c) for c in cand_row[1:] if c.strip() and c.strip() != "-"]
    return bool(data) and all(_YEAR_HEADER_RE.match(c) for c in data)


def _model_header_year_tokens(cand_rows: list[list[str]]) -> list[str] | None:
    for row in cand_rows[:4]:
        if _is_column_header_row(row):
            return [
                _prepare_numeric_cell(c)
                for c in row[1:]
                if c.strip() and _YEAR_HEADER_RE.match(_prepare_numeric_cell(c))
            ]
    return None


def _native_header_year_tokens(result: BindingResult) -> list[str] | None:
    years: list[str] = []
    for chp in result.column_header_paths:
        for seg in chp.path:
            tok = _prepare_numeric_cell(seg.strip())
            if _YEAR_HEADER_RE.match(tok):
                years.append(tok)
    if years:
        return years
    for nr in result.native_rows:
        if not nr.multiset:
            continue
        vals = [str(v) for v in nr.multiset.elements()]
        if vals and all(_YEAR_HEADER_RE.match(v.replace(".0", "")) for v in vals):
            years.extend(v.replace(".0", "") for v in vals)
            break
    return years or None


def _model_header_years_from_markdown(markdown: str) -> list[str] | None:
    for line in markdown.splitlines()[:10]:
        if not line.startswith("|") or "---" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if _is_column_header_row(cells):
            return _model_header_year_tokens([cells])
    return None


def _header_year_mismatch(result: BindingResult, markdown: str) -> bool:
    model_years = _model_header_years_from_markdown(markdown)
    if not model_years:
        return False
    if not model_years:
        return False
    native_years = _native_header_year_tokens(result)
    if not native_years or len(native_years) != len(model_years):
        return False
    return list(model_years) != list(native_years)


def _strip_label_footnote_decorations(label: str) -> str:
    s = label.strip()
    s = re.sub(r"\*+$", "", s).strip()
    return _LABEL_FOOTNOTE_MARK_RE.sub("", s).strip()


def _soft_row_label_mismatch(candidate_label: str, native_path: tuple[str, ...]) -> bool:
    """Wrapped/truncated native labels or unverifiable columns → doubt, not contradiction."""
    native_text = " ".join(p.strip() for p in native_path if p.strip())
    if not native_text and len(native_path) > 1:
        native_text = " ".join(p.strip() for p in native_path[:-1] if p.strip())
    cand_k = normalize_label(_strip_label_footnote_decorations(candidate_label))
    native_k = normalize_label(native_text)
    last_k = normalize_label(native_path[-1].strip()) if native_path else ""
    if not cand_k or not (native_k or last_k):
        return True
    keys = [k for k in (native_k, last_k) if k]
    if any(cand_k == k for k in keys):
        return False
    if native_path and not native_path[-1].strip() and len(native_path) > 1:
        parent_k = normalize_label(
            _strip_label_footnote_decorations(
                " ".join(p.strip() for p in native_path[:-1] if p.strip()).rstrip("-")
            )
        )
        if parent_k and len(parent_k) >= 10:
            if cand_k.startswith(parent_k) or parent_k.startswith(
                cand_k[: min(len(cand_k), len(parent_k))]
            ):
                return True
    if len(native_path) >= 2 and native_path[-1].strip():
        cont = native_path[-1].strip()
        if cont and cont[0].islower():
            joined = normalize_label(
                _strip_label_footnote_decorations(
                    " ".join(p.strip() for p in native_path if p.strip())
                )
            )
            if joined and cand_k and (joined == cand_k or cand_k.startswith(joined[:40])):
                return True
    return False


def _row_label_contradictions_are_soft(result: BindingResult) -> bool:
    if not result.row_label_contradictions:
        return True
    if result.column_binding_unverifiable:
        return True
    return all(
        _soft_row_label_mismatch(rl.candidate_label, rl.row_path)
        for rl in result.row_label_contradictions
    )


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


def _model_cell_numeric_values(cell: str) -> list[str]:
    raw = cell.strip()
    if not raw or raw == "-":
        return []
    merged = _cell_binding_numeric_tokens(raw)
    if merged:
        return merged
    cell = _prepare_numeric_cell(raw)
    if cell and is_numeric_token(cell):
        return [cell]
    return []


def _model_row_numeric_tokens(cand_row: list[str]) -> list[str]:
    out: list[str] = []
    for cell in cand_row[1:]:
        out.extend(_model_cell_numeric_values(cell))
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
        merged = _model_cell_numeric_values(cell)
        if not cell.strip() or cell.strip() == "-":
            slots.append(None)
        elif len(merged) == 1:
            slots.append(merged[0])
        elif merged:
            slots.append("__non_numeric__")
        else:
            cell = _prepare_numeric_cell(cell)
            if not cell:
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
        model_counter = Counter(
            _normalize_binding_numeric_token(_prepare_numeric_cell(m)) for m in model_nums
        )
        native_counter = getattr(native_row, "multiset", None) or Counter()
        if model_counter and model_counter == native_counter:
            return None
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
    """True only for prose footnote markers on rows with no data-lane numbers."""
    if not leaf.strip():
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


def _labels_disagree_under_unverifiable_columns(native_row, cand_label: str) -> bool:
    if not cand_label.strip():
        return bool(native_row.row_path and any(part.strip() for part in native_row.row_path))
    cand_k = label_key(cand_label)
    if not cand_k:
        return False
    native_keys: list = []
    if native_row.row_path:
        joined = _native_row_label_text(native_row)
        if joined:
            native_keys.append(label_key(joined))
        last = native_row.row_path[-1].strip()
        if last:
            k = label_key(last)
            if k and k not in native_keys:
                native_keys.append(k)
    if not native_keys:
        return True
    return all(cand_k != k for k in native_keys if k)


def _native_row_index_for_unbound(cell: UnboundCell, result: BindingResult) -> int | None:
    for idx, native_row in enumerate(result.native_rows):
        if native_row.row_path == cell.row_path:
            return idx
    return None


def _native_row_has_data_lane_numbers(result: BindingResult, row_path: tuple[str, ...]) -> bool:
    for native_row in result.native_rows:
        if native_row.row_path != row_path:
            continue
        for text, _ambig in native_row.lane_tokens.values():
            if is_numeric_token(_prepare_numeric_cell(text)):
                return True
    for cell in result.native_unbound:
        if cell.row_path != row_path or not cell.col_path:
            continue
        if is_numeric_token(_prepare_numeric_cell(cell.token)):
            return True
    return False


def _unbound_cell_has_data_lane_number(cell: UnboundCell) -> bool:
    if not cell.col_path:
        return False
    return is_numeric_token(_prepare_numeric_cell(cell.token))


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


def _native_row_is_year_column_header(result: BindingResult, row_path: tuple[str, ...]) -> bool:
    leaf = row_path[-1].strip() if row_path else ""
    if leaf and not _YEAR_HEADER_RE.match(_prepare_numeric_cell(leaf)):
        return False
    for nr in result.native_rows:
        if nr.row_path != row_path:
            continue
        multiset = getattr(nr, "multiset", None)
        if not multiset:
            continue
        elems = [str(v).replace(".0", "") for v in multiset.elements()]
        return bool(elems) and all(_YEAR_HEADER_RE.match(e) for e in elems)
    return False


def _latex_tabular_to_markdown(markdown: str) -> str:
    """Best-effort 1:1 conversion of a model LaTeX tabular block to markdown."""
    begin = _LATEX_TABULAR_BEGIN_RE.search(markdown)
    end = _LATEX_TABULAR_END_RE.search(markdown)
    if not begin or not end or end.start() <= begin.start():
        raise ValueError("unclosed or missing LaTeX tabular environment")
    body = markdown[begin.end() : end.start()]
    rows: list[list[str]] = []
    for chunk in re.split(r"\\\\", body):
        line = chunk.strip()
        if not line:
            continue
        line = re.sub(r"\\hline", "", line)
        line = re.sub(r"\\(top|mid|bottom)rule", "", line)
        if "&" not in line:
            continue
        cells = []
        for cell in line.split("&"):
            cell = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell)
            cell = re.sub(r"\\textit\{([^}]*)\}", r"\1", cell)
            cell = cell.replace("\\%", "%").replace("$", "").strip()
            cells.append(cell)
        if cells:
            rows.append(cells)
    if not rows:
        raise ValueError("LaTeX tabular contained no rows")
    width = max(len(r) for r in rows)
    norm = [r + [""] * (width - len(r)) for r in rows]
    lines = [
        "| " + " | ".join(norm[0]) + " |",
        "| " + " | ".join("---" for _ in norm[0]) + " |",
    ]
    for r in norm[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines) + "\n"


def _prepare_markdown_for_cell_check(markdown: str) -> tuple[str, str | None]:
    if not _LATEX_TABULAR_BEGIN_RE.search(markdown):
        return markdown, None
    try:
        return _latex_tabular_to_markdown(markdown), "latex_tabular_converted"
    except (ValueError, re.error) as exc:
        return markdown, f"latex_tabular_unparsed: {exc}"


def _native_unbound_cell_is_excludable(
    cell: UnboundCell,
    result: BindingResult,
    region: tuple[float, float, float, float] | None,
) -> bool:
    """Position-only: margin/outside region, header band, or footnote band below data."""
    if _native_row_is_year_column_header(result, cell.row_path):
        tok = _prepare_numeric_cell(cell.token)
        if _YEAR_HEADER_RE.match(tok):
            return True
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
    if _unbound_cell_has_data_lane_number(cell):
        return False
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
        if _native_row_has_data_lane_numbers(result, cell.row_path):
            return False
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
                        mismatches.append((stub, native_label, "ambiguous_label"))
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
                    mismatches.append((stub, native_label, "ambiguous_label"))
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
        kind = "model_data_row"
        if result.column_binding_unverifiable:
            kind = "ambiguous_label"
        mismatches.append((stub, "native_unbound", kind))

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


def _native_row_counts_as_data(nr) -> bool:
    multiset = getattr(nr, "multiset", None)
    if not multiset or getattr(nr, "is_parent", False) or not getattr(nr, "row_path", ()):
        return False
    vals = [v for v in multiset.elements()]
    if vals and all(_YEAR_HEADER_RE.match(str(v).replace(".0", "")) for v in vals):
        return False
    return True


def _model_covers_native_data_rows(result: BindingResult, markdown: str) -> bool:
    """False when the native grid has substantially more numeric rows than the model bound."""
    cand_rows = _candidate_rows_for_scan(result, markdown)
    if cand_rows is None:
        return True
    bound = len(_bound_rows_requiring_coverage(result, cand_rows))
    native_data = sum(1 for nr in result.native_rows if _native_row_counts_as_data(nr))
    return bound >= native_data


def classify_binding_for_cell_check(
    result: BindingResult,
    markdown: str,
    *,
    region: tuple[float, float, float, float] | None = None,
) -> NativeCellOutcome:
    if result.contradicted_cells:
        footnote_only = all(
            _FOOTNOTE_PAREN_NUM_RE.match(_prepare_numeric_cell(c.native_token or ""))
            or _FOOTNOTE_PAREN_NUM_RE.match(_prepare_numeric_cell(c.model_token or ""))
            for c in result.contradicted_cells
        )
        if not footnote_only:
            return NativeCellOutcome.CONTRADICTED
    if result.row_label_contradictions and not result.column_binding_unverifiable:
        if not _row_label_contradictions_are_soft(result):
            return NativeCellOutcome.CONTRADICTED
    if _header_year_mismatch(result, markdown):
        return NativeCellOutcome.CONTRADICTED

    if result.structural_agreement:
        if result.row_label_contradictions and _row_label_contradictions_are_soft(result):
            return NativeCellOutcome.AMBIGUOUS
        if not _model_covers_native_data_rows(result, markdown):
            return NativeCellOutcome.AMBIGUOUS
        return NativeCellOutcome.VERIFIED

    mismatches, comparisons, per_row = _row_anchor_scan(result, markdown)
    if any(m[2] == "parse_failed" for m in mismatches):
        return NativeCellOutcome.AMBIGUOUS

    hard_mismatches = [
        m
        for m in mismatches
        if m[1] != "ambiguous_locale" and m[2] not in ("parse_failed", "ambiguous_label")
    ]
    if hard_mismatches:
        return NativeCellOutcome.CONTRADICTED
    if any(m[2] == "ambiguous_label" for m in mismatches):
        return NativeCellOutcome.AMBIGUOUS
    if result.ambiguous_count and not result.column_binding_unverifiable:
        return NativeCellOutcome.AMBIGUOUS
    if _numeric_unbound_blocks_verify(result, region):
        return NativeCellOutcome.AMBIGUOUS
    if _fallback_verified_allowed(result, markdown, region, comparisons_per_row=per_row):
        if not _model_covers_native_data_rows(result, markdown):
            return NativeCellOutcome.AMBIGUOUS
        return NativeCellOutcome.VERIFIED
    if result.row_label_contradictions and result.column_binding_unverifiable:
        return NativeCellOutcome.AMBIGUOUS
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
    peer_region_bboxes: list | None = None,
    locator_boxes: list | None = None,
    page=None,
) -> TableCellCheckResult:
    if not words:
        return TableCellCheckResult(
            table_id=table_id,
            outcome=NativeCellOutcome.AMBIGUOUS,
            detail="no native words in region",
        )
    markdown, latex_note = _prepare_markdown_for_cell_check(markdown)
    if latex_note and latex_note.startswith("latex_tabular_unparsed"):
        return TableCellCheckResult(
            table_id=table_id,
            outcome=NativeCellOutcome.AMBIGUOUS,
            detail=latex_note,
        )
    region_tuple: tuple[float, float, float, float] | None = None
    if region_bbox is not None:
        try:
            region_tuple = tuple(float(v) for v in region_bbox)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            region_tuple = None
    if region_tuple is not None and peer_region_bboxes:
        from socr.tables.native_contradiction import refine_check_region

        peers = []
        for peer in peer_region_bboxes:
            try:
                peers.append(tuple(float(v) for v in peer))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
        region_tuple = refine_check_region(region_tuple, peers)
    if region_tuple is not None and page is not None:
        from socr.tables.native_contradiction import extend_check_region_for_trailing_words

        region_tuple = extend_check_region_for_trailing_words(page, region_tuple, markdown)
    try:
        binding = bind(words, markdown, region=region_tuple, spans=spans or [])
    except Exception as exc:
        return TableCellCheckResult(
            table_id=table_id,
            outcome=NativeCellOutcome.AMBIGUOUS,
            detail=f"bind failed: {type(exc).__name__}: {exc}",
        )
    outcome = classify_binding_for_cell_check(binding, markdown, region=region_tuple)
    if (
        outcome is NativeCellOutcome.VERIFIED
        and region_tuple is not None
        and locator_boxes
        and page is not None
    ):
        from socr.tables.native_contradiction import (
            _bbox_tuple,
            _box_area_tuple,
            _box_contains,
            _box_has_numbers_beyond_markdown,
            _markdown_number_counter,
        )

        boxes = [_bbox_tuple(b) for b in locator_boxes]
        if len(boxes) == 2:
            outer = max(boxes, key=_box_area_tuple)
            inner = min(boxes, key=_box_area_tuple)
            if (
                _box_contains(outer, inner)
                and region_tuple == inner
                and _box_area_tuple(outer) > _box_area_tuple(inner) * 1.8
            ):
                cand_rows = _candidate_rows_for_scan(binding, markdown)
                model_data_rows = (
                    len(_bound_rows_requiring_coverage(binding, cand_rows)) if cand_rows else 0
                )
                if model_data_rows <= 4 and _box_has_numbers_beyond_markdown(
                    page, outer, _markdown_number_counter([markdown])
                ):
                    outcome = NativeCellOutcome.AMBIGUOUS
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
    if latex_note:
        detail = f"{latex_note}; {detail}"
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
