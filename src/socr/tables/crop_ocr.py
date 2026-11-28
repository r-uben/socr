"""Table-crop chart filtering, canonical HTML->Markdown normalization, and two-reader verification.

Designed for tight table crops (e.g. sliced by ``ocria-nano-1`` or a layout
detector) read by two independent free vision models (such as OCRia-1 +
GLM-OCR or LightOnOCR-2).

Fail-closed invariants:
  * Chart-overlapping candidate table boxes are dropped before OCR so chart
    regions are never mis-cropped as tables.
  * Spanned-header normalization ONLY deduplicates repeated colspan/rowspan
    4-digit year tokens inside pure leading header rows. Data rows, minus
    signs, decimals, fractions, and cell counts are NEVER weakened.
  * Two-reader verification runs :func:`socr.tables.second_reader.compare_readings`
    in BOTH directions (``a -> b`` and ``b -> a``) with ``chart_picture=False``
    and fails closed on any data-row discrepancy, truncation, unclosed tag, or
    reader error.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from socr.core.html_tables import convert_html_tables
from socr.tables.second_reader import (
    MINUS_CHARS,
    TILDE,
    Agreement,
    Difference,
    compare_readings,
    row_numbers,
)

BoxT = TypeVar("BoxT")

#: Default IoU cutoff above which a candidate table box is treated as overlapping
#: a detected chart/figure region and dropped.
DEFAULT_CHART_OVERLAP_MAX_IOU = 0.15
#: Default coverage cutoff (``intersection_area / table_box_area``) above which a
#: candidate table box lying substantially inside a chart/figure region is dropped.
DEFAULT_CHART_OVERLAP_MAX_COVER = 0.25
#: Maximum leading rows of a Markdown table eligible for pure-header inspection.
_MAX_HEADER_TIERS = 4

_FENCE_LINE_RE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*$")
_FURNITURE_BLOCK_RE = re.compile(
    r"<(page_header|page_footer|page_number|watermark)>(.*?)</\1>",
    re.S | re.I,
)
_TABLE_BLOCK_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.S | re.I)
_TABLE_OPEN_RE = re.compile(r"<table[\s>]", re.I)
_TABLE_CLOSE_RE = re.compile(r"</table>", re.I)
_TR_OPEN_RE = re.compile(r"<tr[\s>]", re.I)
_TR_CLOSE_RE = re.compile(r"</tr>", re.I)

_MD_SEP_CELL_RE = re.compile(r"^:?-{2,}:?$")
_YEAR_TOKEN_RE = re.compile(r"(?<!\d)(1[89]\d{2}|20\d{2})(?!\d)")
_PERIOD_TOKEN_RE = re.compile(r"\b(?:Q[1-4]|H[12])\b", re.I)
_SIGN_BEFORE_DIGIT_RE = re.compile(rf"[{re.escape(MINUS_CHARS)}{re.escape(TILDE)}]\s*\.?\d")
_DECIMAL_OR_FRAC_RE = re.compile(r"\d+\.\d+|\.\d+|\d+\s*/\s*\d+")
_STANDALONE_YEAR_RE = re.compile(r"^(1[89]\d{2}|20\d{2})$")


class TableCropReader(Protocol):
    """Protocol satisfied by :class:`socr.engines.ocria.OCRiaEngine` and reader callables."""

    def read_image(self, image: Path | str | bytes) -> tuple[str, str | None]: ...


@dataclass(frozen=True)
class CropAgreementResult:
    """Verdict of running two independent free readers on the same tight table crop."""

    agreed: bool
    #: Verified canonical Markdown when ``agreed`` is True; empty string when failed closed.
    markdown: str
    differences: tuple[Difference, ...]
    reader_a_model: str
    reader_b_model: str
    reader_a_markdown: str = ""
    reader_b_markdown: str = ""
    unknown_reason: str = ""


# ---------------------------------------------------------------------------
# 1. Chart-box overlap filtering
# ---------------------------------------------------------------------------


def _as_bbox(box: Any) -> tuple[float, float, float, float]:
    """Extract ``(x0, y0, x1, y1)`` from a 4-sequence, rect-like object, or ``.bbox`` carrier."""
    if hasattr(box, "bbox"):
        box = box.bbox
    if all(hasattr(box, attr) for attr in ("x0", "y0", "x1", "y1")):
        x0, y0, x1, y1 = float(box.x0), float(box.y0), float(box.x1), float(box.y1)
    elif isinstance(box, Sequence) and len(box) == 4:
        x0, y0, x1, y1 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    else:
        raise TypeError(f"Unsupported bounding box representation: {box!r}")
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _box_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersection_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def filter_chart_overlapping_boxes(
    table_boxes: Sequence[BoxT],
    chart_boxes: Sequence[Any],
    *,
    max_iou: float = DEFAULT_CHART_OVERLAP_MAX_IOU,
    max_cover: float = DEFAULT_CHART_OVERLAP_MAX_COVER,
) -> list[BoxT]:
    """Drop candidate table crop boxes that overlap any detected chart/figure box.

    A table box is dropped when its area is zero, or when for any chart box either:
      * ``intersection / union >= max_iou``, or
      * ``intersection / table_area >= max_cover`` (fraction of the candidate
        table crop covered by the chart region).

    Items are returned in their original order and representation.
    """
    if not table_boxes:
        return []
    parsed_charts = [
        (cb, area) for cb in (_as_bbox(c) for c in chart_boxes) if (area := _box_area(cb)) > 0.0
    ]
    kept: list[BoxT] = []
    for item in table_boxes:
        tb = _as_bbox(item)
        t_area = _box_area(tb)
        if t_area <= 0.0:
            continue
        overlaps_chart = False
        for cb, c_area in parsed_charts:
            inter = _intersection_area(tb, cb)
            if inter <= 0.0:
                continue
            union = t_area + c_area - inter
            iou = inter / union if union > 0.0 else 0.0
            cover = inter / t_area
            if iou >= max_iou or cover >= max_cover:
                overlaps_chart = True
                break
        if not overlaps_chart:
            kept.append(item)
    return kept


# ---------------------------------------------------------------------------
# 2. Clean HTML->Markdown extraction & spanned-header normalization
# ---------------------------------------------------------------------------


def _unfenced_if_closed(text: str) -> tuple[str, bool]:
    """Strip balanced markdown code fences. Leave an unclosed fence untouched."""
    lines = text.strip().split("\n")
    fences = [i for i, line in enumerate(lines) if _FENCE_LINE_RE.match(line)]
    if not fences:
        return text, False
    if len(fences) % 2 == 1:
        return text, True
    return "\n".join(line for i, line in enumerate(lines) if i not in fences), False


def _has_unclosed_html_table(text: str) -> bool:
    return len(_TABLE_OPEN_RE.findall(text)) != len(_TABLE_CLOSE_RE.findall(text)) or len(
        _TR_OPEN_RE.findall(text)
    ) != len(_TR_CLOSE_RE.findall(text))


def _has_unclosed_markup(text: str) -> bool:
    """True when *text* has an unclosed markdown code fence or unclosed HTML table/row tag."""
    if not (text or "").strip():
        return False
    unfenced, open_fence = _unfenced_if_closed(text)
    if open_fence:
        return True
    cleaned = _FURNITURE_BLOCK_RE.sub("\n", unfenced)
    return _has_unclosed_html_table(cleaned)


def _split_md_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("|") or stripped.count("|") < 2:
        return None
    inner = stripped[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    # Split on unescaped pipes only so literal ``\|`` inside a cell stays intact.
    cells = re.split(r"(?<!\\)\|", inner)
    return [c.strip() for c in cells]


def _is_md_separator_row(cells: Sequence[str]) -> bool:
    return (
        bool(cells)
        and all(_MD_SEP_CELL_RE.match(c.strip()) for c in cells if c.strip())
        and any(c.strip() for c in cells)
    )


def _format_md_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _cell_is_pure_header_cell(cell: str) -> bool:
    """True when *cell* carries no data values (only empty, prose header labels, years, or Q/H).

    Any decimal number, signed/negative number, parenthesized number, fraction,
    currency/percent value, or bare non-year number immediately disqualifies the
    row from header normalization so a data row can NEVER be touched.
    """
    text = cell.strip()
    if not text:
        return True
    if _SIGN_BEFORE_DIGIT_RE.search(text) or _DECIMAL_OR_FRAC_RE.search(text):
        return False
    if "$" in text or "%" in text or re.search(r"\(\s*\d", text):
        return False
    without_periods = _PERIOD_TOKEN_RE.sub(" ", text)
    nums = row_numbers(without_periods)
    if not nums:
        return True
    if all(_STANDALONE_YEAR_RE.match(tok) is not None for tok in nums):
        return True
    # Allow descriptive header prose such as "From 12 months earlier" or
    # "12 months ending in Dec." ONLY when the cell has alphabetic words (never
    # a bare numeric data cell like "160" or "12").
    return bool(re.search(r"[A-Za-z]{2,}", without_periods))


def _row_is_pure_header_row(cells: Sequence[str]) -> bool:
    """True iff every value-column cell (index >= 1) is pure header content and
    no cell carries bare non-year numeric data."""
    if len(cells) < 2:
        return False
    return all(_cell_is_pure_header_cell(c) for c in cells[1:])


def _extract_single_year(cell: str) -> str | None:
    years = _YEAR_TOKEN_RE.findall(cell)
    if len(years) == 1:
        return years[0]
    return None


def _strip_year_from_header_cell(cell: str, year: str) -> str:
    """Remove one duplicated colspan *year* token from a pure header cell."""
    updated = re.sub(rf"(?<!\d){re.escape(year)}(?!\d)", " ", cell, count=1)
    return " ".join(updated.split())


def _normalize_spanned_headers_in_table(table_md: str) -> str:
    """Deduplicate repeated colspan/rowspan 4-digit year tokens in leading pure header rows.

    Never alters cell counts, data rows, minus signs, decimals, or non-year tokens.
    """
    raw_lines = [line for line in table_md.strip().splitlines() if line.strip()]
    parsed: list[tuple[bool, list[str]]] = []
    for line in raw_lines:
        cells = _split_md_row(line)
        if cells is None:
            return table_md
        parsed.append((_is_md_separator_row(cells), cells))

    # Identify leading pure header rows (stopping at the first non-header row).
    # Once past the Markdown separator row (``| --- |``), a row is only eligible
    # as a secondary header tier if its first cell is empty or repeats row 0's
    # top-left stub label (vertical rowspan). Any new non-empty row label (e.g.
    # ``| Total loans | 1950 | 1950 |``) begins data rows and stops immediately.
    header_indices: list[int] = []
    non_sep_seen = 0
    seen_sep = False
    header0_stub = ""
    for idx, (is_sep, cells) in enumerate(parsed):
        if is_sep:
            seen_sep = True
            continue
        if non_sep_seen >= _MAX_HEADER_TIERS:
            break
        if seen_sep:
            stub = cells[0].strip()
            if stub and stub != header0_stub:
                break
        if not _row_is_pure_header_row(cells):
            break
        if non_sep_seen == 0:
            header0_stub = cells[0].strip()
        header_indices.append(idx)
        non_sep_seen += 1

    # 1. Horizontal consecutive colspan year deduplication within each header row.
    #    E.g. ``| | 1999 July | 1999 Aug. | 2000 Jan. | 2000 Feb. |`` ->
    #         ``| | 1999 July | Aug. | 2000 Jan. | Feb. |``
    #    and  ``| | 2000 | 2000 | 2000 |`` -> ``| | 2000 |  |  |``.
    #    Non-consecutive groups like ``| 1999 | 2000 | Percent change | 1999 | 2000 |``
    #    are preserved intact.
    for idx in header_indices:
        is_sep, cells = parsed[idx]
        new_cells = list(cells)
        prev_year: str | None = None
        for col in range(1, len(new_cells)):
            cell = new_cells[col]
            if not cell.strip():
                # Empty cells inside a header row are colspan padding from
                # ``convert_html_tables``; they do not end a repeated year span.
                continue
            year = _extract_single_year(cell)
            if year is None:
                prev_year = None
                continue
            if year == prev_year:
                new_cells[col] = _strip_year_from_header_cell(cell, year)
            else:
                prev_year = year
        parsed[idx] = (is_sep, new_cells)

    # 2. Vertical rowspan year deduplication in the exact same column of adjacent header tiers.
    #    Only clears a cell whose ENTIRE text is the exact repeated 4-digit year.
    for pos in range(1, len(header_indices)):
        prev_idx = header_indices[pos - 1]
        curr_idx = header_indices[pos]
        prev_cells = parsed[prev_idx][1]
        curr_cells = list(parsed[curr_idx][1])
        changed = False
        for col in range(1, min(len(prev_cells), len(curr_cells))):
            curr_val = curr_cells[col].strip()
            if not _STANDALONE_YEAR_RE.match(curr_val):
                continue
            if _extract_single_year(prev_cells[col]) == curr_val:
                curr_cells[col] = ""
                changed = True
        if changed:
            parsed[curr_idx] = (parsed[curr_idx][0], curr_cells)

    out_lines: list[str] = []
    for raw_line, (is_sep, cells) in zip(raw_lines, parsed, strict=True):
        if is_sep:
            out_lines.append(raw_line.strip())
        else:
            out_lines.append(_format_md_row(cells))
    return "\n".join(out_lines)


def _extract_markdown_table_blocks(text: str) -> list[str]:
    """Extract contiguous Markdown pipe-table blocks from *text*."""
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if _split_md_row(line) is not None:
            current.append(line.strip())
        else:
            if len(current) >= 2:
                blocks.append("\n".join(current))
            current = []
    if len(current) >= 2:
        blocks.append("\n".join(current))
    return blocks


def normalize_html_table_for_comparison(html_or_md: str) -> str:
    """Extract table blocks, convert HTML to Markdown, and normalize spanned year headers.

    Surrounding page headers, footers, page numbers, and chart captions are
    stripped when ``<table>...</table>`` or Markdown pipe-table blocks are
    present. Truncated/unclosed code fences or HTML tables are returned
    unconverted so downstream comparison fails closed as ``UNKNOWN``.
    """
    if not (html_or_md or "").strip():
        return ""
    unfenced, open_fence = _unfenced_if_closed(html_or_md)
    if open_fence:
        return html_or_md
    cleaned = _FURNITURE_BLOCK_RE.sub("\n", unfenced)
    if _has_unclosed_html_table(cleaned):
        return cleaned

    html_tables = _TABLE_BLOCK_RE.findall(cleaned)
    if html_tables:
        md_tables: list[str] = []
        for raw_table in html_tables:
            converted = convert_html_tables(raw_table).strip()
            if converted:
                md_tables.append(_normalize_spanned_headers_in_table(converted))
        return "\n\n".join(md_tables)

    md_blocks = _extract_markdown_table_blocks(cleaned)
    if md_blocks:
        return "\n\n".join(_normalize_spanned_headers_in_table(b) for b in md_blocks)

    return cleaned.strip()


# ---------------------------------------------------------------------------
# 3. Two-reader table crop verification
# ---------------------------------------------------------------------------


def _reader_model_name(reader: Any) -> str:
    for attr in ("model",):
        val = getattr(reader, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    cfg = getattr(reader, "config", None)
    if cfg is not None:
        val = getattr(cfg, "model", None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    for attr in ("name", "__name__"):
        val = getattr(reader, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return type(reader).__name__


def _invoke_reader(
    reader: TableCropReader | Callable[[Path], str | tuple[str, str | None]],
    crop_path: Path,
) -> tuple[str, str | None]:
    if hasattr(reader, "read_image") and callable(reader.read_image):
        result = reader.read_image(crop_path)
    elif callable(reader):
        result = reader(crop_path)
    else:
        raise TypeError(f"Table crop reader must provide .read_image() or be callable: {reader!r}")
    if isinstance(result, tuple) and len(result) == 2:
        raw_text, finish_reason = result
        return (str(raw_text or ""), str(finish_reason) if finish_reason is not None else None)
    return (str(result or ""), None)


def verify_table_crop_two_readers(
    crop_path: Path | str,
    *,
    reader_a: TableCropReader | Callable[[Path], str | tuple[str, str | None]],
    reader_b: TableCropReader | Callable[[Path], str | tuple[str, str | None]],
) -> CropAgreementResult:
    """Run two independent free readers on *crop_path* and compare both directions.

    Fails closed (``agreed=False``, ``markdown=""``) on:
      * any reader exception, empty output, truncation (``finish_reason == "length"``),
        or unclosed code fence / HTML table;
      * ``Agreement.UNKNOWN`` (e.g. no numeric tokens on either side);
      * any data-row discrepancy (digit swap, sign flip, missing/extra value) in
        either ``a -> b`` or ``b -> a``.
    """
    path = Path(crop_path)
    model_a = _reader_model_name(reader_a)
    model_b = _reader_model_name(reader_b)

    try:
        raw_a, finish_a = _invoke_reader(reader_a, path)
    except Exception as exc:
        return CropAgreementResult(
            agreed=False,
            markdown="",
            differences=(),
            reader_a_model=model_a,
            reader_b_model=model_b,
            unknown_reason=f"reader_a_error: {type(exc).__name__}: {exc}",
        )

    try:
        raw_b, finish_b = _invoke_reader(reader_b, path)
    except Exception as exc:
        return CropAgreementResult(
            agreed=False,
            markdown="",
            differences=(),
            reader_a_model=model_a,
            reader_b_model=model_b,
            unknown_reason=f"reader_b_error: {type(exc).__name__}: {exc}",
        )

    trunc_a = finish_a == "length"
    trunc_b = finish_b == "length"
    norm_a = normalize_html_table_for_comparison(raw_a)
    norm_b = normalize_html_table_for_comparison(raw_b)

    if trunc_a or trunc_b:
        which = "both" if (trunc_a and trunc_b) else ("reader_a" if trunc_a else "reader_b")
        return CropAgreementResult(
            agreed=False,
            markdown="",
            differences=(),
            reader_a_model=model_a,
            reader_b_model=model_b,
            reader_a_markdown=norm_a,
            reader_b_markdown=norm_b,
            unknown_reason=f"{which}_truncated",
        )

    unclosed_a = _has_unclosed_markup(raw_a)
    unclosed_b = _has_unclosed_markup(raw_b)
    if unclosed_a or unclosed_b:
        which = (
            "both" if (unclosed_a and unclosed_b) else ("reader_a" if unclosed_a else "reader_b")
        )
        return CropAgreementResult(
            agreed=False,
            markdown="",
            differences=(),
            reader_a_model=model_a,
            reader_b_model=model_b,
            reader_a_markdown=norm_a,
            reader_b_markdown=norm_b,
            unknown_reason=f"{which}_unclosed_markup",
        )

    if not norm_a.strip() or not norm_b.strip():
        which = "reader_a" if not norm_a.strip() else "reader_b"
        return CropAgreementResult(
            agreed=False,
            markdown="",
            differences=(),
            reader_a_model=model_a,
            reader_b_model=model_b,
            reader_a_markdown=norm_a,
            reader_b_markdown=norm_b,
            unknown_reason=f"{which}_empty",
        )

    verdict_ab = compare_readings(norm_a, norm_b, chart_picture=False, reader_truncated=trunc_b)
    verdict_ba = compare_readings(norm_b, norm_a, chart_picture=False, reader_truncated=trunc_a)

    agreed = verdict_ab.outcome is Agreement.AGREE and verdict_ba.outcome is Agreement.AGREE
    combined_diffs: list[Difference] = []
    seen_diffs: set[Difference] = set()
    for diff in (*verdict_ab.differences, *verdict_ba.differences):
        if diff not in seen_diffs:
            seen_diffs.add(diff)
            combined_diffs.append(diff)

    unknown_reason = ""
    if not agreed:
        reasons = [r for r in (verdict_ab.unknown_reason, verdict_ba.unknown_reason) if r]
        unknown_reason = "; ".join(dict.fromkeys(reasons))

    return CropAgreementResult(
        agreed=agreed,
        markdown=norm_a if agreed else "",
        differences=tuple(combined_diffs),
        reader_a_model=model_a,
        reader_b_model=model_b,
        reader_a_markdown=norm_a,
        reader_b_markdown=norm_b,
        unknown_reason=unknown_reason,
    )
