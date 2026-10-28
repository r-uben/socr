"""Layout-aware crops for the Gemma OCR rung (300 dpi)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from socr.core.born_digital import upright_rotation_for


@dataclass(frozen=True)
class GemmaCropPlan:
    """How to rasterize one page for Gemma."""

    clips: tuple[tuple[str, object], ...]  # (label, fitz.Rect)
    full_page_unverified: bool = False


def _median_word_gap(words: list) -> float | None:
    gaps = sorted(
        words[i + 1][0] - words[i][2]
        for i in range(len(words) - 1)
        if words[i + 1][0] - words[i][2] > 0
    )
    if not gaps:
        return None
    return gaps[len(gaps) // 2]


def _detect_page_column_gutter(words: list) -> float | None:
    """Single vertical gutter between two x-bands (GH-152 yardstick, local copy)."""
    from socr.core.born_digital import ALIGNED_RUN_GAP_MAX_WORD_SPACES

    if len(words) < 2:
        return None
    word_gap = _median_word_gap(words)
    if word_gap is None:
        return None
    min_gutter_pt = ALIGNED_RUN_GAP_MAX_WORD_SPACES * word_gap
    intervals = sorted((w[0], w[2]) for w in words)
    merged: list[list[float]] = []
    for x0, x1 in intervals:
        if merged and x0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])
    gutters = [
        (merged[i][1], merged[i + 1][0])
        for i in range(len(merged) - 1)
        if merged[i + 1][0] - merged[i][1] >= min_gutter_pt
    ]
    if len(gutters) != 1:
        return None
    gap_start, gap_end = gutters[0]
    return (gap_start + gap_end) / 2.0


def _forbidden_rects(page) -> list:
    """Table and equation regions that must not be cut."""
    import fitz

    rects: list[fitz.Rect] = []
    try:
        for t in page.find_tables() or []:
            rects.append(fitz.Rect(t.bbox))
    except Exception:
        pass
    try:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_LIGATURES)["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            text = " ".join(
                span.get("text", "")
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            )
            if "$" in text or "\\(" in text or "\\[" in text:
                rects.append(fitz.Rect(block["bbox"]))
    except Exception:
        pass
    return rects


def _split_intersects_forbidden(split_is_vertical: bool, coord: float, forbidden: list) -> bool:
    import fitz

    for rect in forbidden:
        if split_is_vertical:
            if rect.x0 < coord < rect.x1:
                return True
        elif rect.y0 < coord < rect.y1:
            return True
    return False


def _two_column_clips(
    page, gutter_x: float, forbidden: list
) -> tuple[tuple[str, object], ...] | None:
    import fitz

    rect = page.rect
    if _split_intersects_forbidden(True, gutter_x, forbidden):
        return None
    return (
        ("L", fitz.Rect(rect.x0, rect.y0, gutter_x, rect.y1)),
        ("R", fitz.Rect(gutter_x, rect.y0, rect.x1, rect.y1)),
    )


def _single_column_clips(page, forbidden: list) -> tuple[tuple[str, object], ...] | None:
    import fitz

    rect = page.rect
    height = rect.y1 - rect.y0
    if height <= 0:
        return None
    try:
        blocks = page.get_text("dict")["blocks"]
    except Exception:
        return None
    y_edges = sorted(
        {
            y
            for block in blocks
            if block.get("type") == 0
            for y in (block["bbox"][1], block["bbox"][3])
        }
    )
    best_gap = 0.0
    split_y = None
    mid_lo = rect.y0 + height * 0.25
    mid_hi = rect.y0 + height * 0.75
    for i in range(len(y_edges) - 1):
        y0, y1 = y_edges[i], y_edges[i + 1]
        gap = y1 - y0
        mid = (y0 + y1) / 2.0
        if mid_lo <= mid <= mid_hi and gap > best_gap:
            best_gap = gap
            split_y = mid
    if split_y is None or best_gap < height * 0.04:
        return None
    if _split_intersects_forbidden(False, split_y, forbidden):
        return None
    overlap = height * 0.03
    return (
        ("T", fitz.Rect(rect.x0, rect.y0, rect.x1, split_y + overlap)),
        ("B", fitz.Rect(rect.x0, split_y - overlap, rect.x1, rect.y1)),
    )


def plan_gemma_page_crops(page) -> GemmaCropPlan:
    """Choose gutter, horizontal band, or full-page (unverified) crops."""
    import fitz

    rect = page.rect
    forbidden = _forbidden_rects(page)
    clips: tuple[tuple[str, object], ...] | None = None
    try:
        words = page.get_text("words")
        gutter = _detect_page_column_gutter(words) if words else None
        if gutter is not None:
            clips = _two_column_clips(page, gutter, forbidden)
    except Exception:
        clips = None
    if clips is None:
        clips = _single_column_clips(page, forbidden)
    if clips is None:
        return GemmaCropPlan(clips=(("full", fitz.Rect(rect)),), full_page_unverified=True)
    return GemmaCropPlan(clips=clips, full_page_unverified=False)


def render_gemma_crops(
    pdf_path: Path,
    page_num: int,
    dest_dir: Path,
    *,
    dpi: int,
) -> tuple[list[tuple[str, Path]], bool]:
    """Render planned crops; return ``(stem, path)`` pairs and unverified flag."""
    import fitz
    from PIL import Image

    from socr.core.pdf import open_pdf

    dest_dir.mkdir(parents=True, exist_ok=True)
    stems: list[tuple[str, Path]] = []
    with open_pdf(pdf_path) as doc:
        page = doc[page_num - 1]
        plan = plan_gemma_page_crops(page)
        rotation = upright_rotation_for(page)
        for label, clip in plan.clips:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            if rotation:
                mat = mat.prerotate(rotation)
            pix = page.get_pixmap(matrix=mat, clip=clip)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            stem = f"page_{page_num:04d}_{label}"
            path = dest_dir / f"{stem}.png"
            img.save(path)
            stems.append((stem, path))
        return stems, plan.full_page_unverified


def render_page_half_crops(
    pdf_path: Path,
    page_num: int,
    dest_dir: Path,
    *,
    dpi: int,
) -> list[tuple[str, Path]]:
    """Backward-compatible entry: layout-aware crops (not always left/right)."""
    stems, _ = render_gemma_crops(pdf_path, page_num, dest_dir, dpi=dpi)
    return stems
