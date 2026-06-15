"""Figure extraction from PDF files.

Consolidates the ~400 lines duplicated between processor.py and
hpc_sequential_pipeline.py into a single shared module.

Three extraction strategies (applied per page, in order):
  0. Vector figure clustering (union-find on drawing bounding boxes)
  1. IMAGE blocks from page text dict
  2. Raw embedded images via xref

Logo/letterhead filtering
--------------------------
Title-page logos and letterhead banners share a distinctive geometry: they sit
at the very top of the page, span most of the page width, and are short relative
to the page height.  The filter below gates on three conditions derived from page
geometry (no magic literals -- all values are named config constants):

  1. ``y0 < page_height * LOGO_TOP_MARGIN_RATIO``  -- image top is in the top band
  2. ``(x1 - x0) / page_width > LOGO_WIDE_WIDTH_RATIO``  -- image is wide (banner-like)
  3. ``(y1 - y0) / page_height < LOGO_HEIGHT_RATIO``  -- image is short relative to page

All three must be true together; a genuine top-of-page figure (chart inset,
pull-quote graphic) typically violates at least one -- either it is not wide
enough to span the page or it occupies a substantial fraction of the page height.
The filter is deliberately conservative: a single failing condition lets the
candidate through.
"""

from __future__ import annotations

import logging
import signal
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

# Timeout for per-page figure extraction (seconds).
# Protects against malformed PDFs that hang in PyMuPDF calls.
PAGE_EXTRACTION_TIMEOUT = 30


@contextmanager
def _timeout_guard(seconds: int, label: str = ""):
    """Context manager that raises TimeoutError after `seconds`.

    Uses SIGALRM on Unix. Falls through as a no-op on Windows or if
    seconds <= 0 (caller should handle gracefully).
    """
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"Figure extraction timed out after {seconds}s: {label}")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@dataclass
class ExtractedFigure:
    """A figure image extracted from a PDF page.

    ``bbox`` is the figure's bounding rectangle in page-space coordinates as a
    4-float tuple ``(x0, y0, x1, y1)``, matching PyMuPDF convention.  It is used
    downstream (GH-47C) to filter native word tokens inside the figure region via
    ``page.get_text("words")``.  ``None`` for Strategy 2 xref images whose
    placement rect could not be determined (rare; treated as inapplicable).
    """

    figure_num: int
    page_num: int
    image: Image.Image  # PIL Image (lazy import)
    saved_path: str | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class ExtractionResult:
    """Returned by ``FigureExtractor.extract()``.

    ``figures`` holds every figure that passed all filters.
    ``cap_reached`` is ``True`` when extraction halted because the
    ``max_total`` figure cap was hit before the last page was processed;
    callers should surface this fact to the operator (console + audit log)
    so silently dropped later figures are not invisible.
    """

    figures: list[ExtractedFigure] = field(default_factory=list)
    cap_reached: bool = False


# --- Defaults ---
RENDER_DPI = 150
MAX_DIM = 1024
MIN_AREA = 80 * 80
MIN_DRAWINGS_FOR_VECTOR = 5
MIN_VECTOR_AREA_RATIO = 0.05
MAX_VECTOR_AREA_RATIO = 0.85
HEADER_FOOTER_MARGIN = 0.1
CLUSTER_GAP = 30
FINE_CLUSTER_GAP = 3
VECTOR_SPLIT_AREA_RATIO = 0.50
DENSE_VECTOR_MIN_AREA_RATIO = MIN_VECTOR_AREA_RATIO * 0.5
TABLE_GRID_MIN_AREA_RATIO = MIN_VECTOR_AREA_RATIO * 4
MIN_DATA_MARKS = 3
DATA_STROKE_MIN_WIDTH = 1.0

# --- Logo / letterhead filter (IMAGE blocks and raw xref images) ---
# Fraction of page height defining the "top band" where logos/letterheads live.
LOGO_TOP_MARGIN_RATIO: float = 0.20
# Minimum fraction of page width that a banner-style image must span.
LOGO_WIDE_WIDTH_RATIO: float = 0.50
# Maximum fraction of page height that a logo/letterhead may occupy.
# Images taller than this are presumed to be substantive content, not a header.
LOGO_HEIGHT_RATIO: float = 0.15


class FigureExtractor:
    """Extracts figure images from a PDF file."""

    def __init__(
        self,
        max_total: int = 25,
        max_per_page: int = 3,
        save_dir: Path | None = None,
    ) -> None:
        self.max_total = max_total
        self.max_per_page = max_per_page
        self.save_dir = save_dir

    def extract(self, pdf_path: Path, skip_pages: set[int] | None = None) -> ExtractionResult:
        """Extract all figures from a PDF. Returns an :class:`ExtractionResult`.

        ``skip_pages`` (1-indexed) drops those pages entirely. Meant for
        scanned pages, where the only extractable "figure" is the full-page
        raster itself — noise that crowds out real figures and burns vision
        calls; the OCR engine's inline description already covers them.

        If the ``max_total`` cap is reached before all pages are processed,
        ``ExtractionResult.cap_reached`` is set to ``True`` and a ``WARNING``
        is logged so the operator knows later figures were silently dropped.
        Callers should additionally record this in the durable audit log.
        """
        import fitz

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        figures: list[ExtractedFigure] = []
        counter = 1
        cap_reached = False
        if skip_pages:
            logger.info(f"Figure extraction skipping {len(skip_pages)} page(s) of {pdf_path.name}")

        try:
            with fitz.open(pdf_path) as pdf:
                total_pages = len(pdf)
                for page_index in range(total_pages):
                    if counter > self.max_total:
                        remaining = total_pages - page_index
                        logger.warning(
                            "Figure cap reached (%d figures, max_total=%d): "
                            "stopping after page %d, %d page(s) not processed for %s",
                            self.max_total,
                            self.max_total,
                            page_index,  # 0-indexed == 1-indexed last-processed page
                            remaining,
                            pdf_path.name,
                        )
                        cap_reached = True
                        break

                    page = pdf[page_index]
                    page_num = page_index + 1
                    if skip_pages and page_num in skip_pages:
                        continue
                    per_page = 0
                    processed: set[tuple[int, int, int, int]] = set()

                    page_width = page.rect.width
                    page_height = page.rect.height
                    page_area = page_width * page_height
                    is_landscape = page_width > page_height

                    # Landscape adjustments
                    min_area_ratio = (
                        MIN_VECTOR_AREA_RATIO * 0.5 if is_landscape else MIN_VECTOR_AREA_RATIO
                    )
                    max_area_ratio = 0.98 if is_landscape else MAX_VECTOR_AREA_RATIO
                    min_drawings = 3 if is_landscape else MIN_DRAWINGS_FOR_VECTOR

                    try:
                        with _timeout_guard(PAGE_EXTRACTION_TIMEOUT, f"page {page_num}"):
                            counter, per_page = self._extract_page_figures(
                                page,
                                page_num,
                                pdf,
                                figures,
                                counter,
                                per_page,
                                processed,
                                page_width,
                                page_height,
                                page_area,
                                is_landscape,
                                min_area_ratio,
                                max_area_ratio,
                                min_drawings,
                            )
                    except TimeoutError:
                        logger.warning(
                            f"Figure extraction timed out on page {page_num} "
                            f"of {pdf_path.name}, skipping page"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Figure extraction failed on page {page_num} "
                            f"of {pdf_path.name}: {type(e).__name__}: {e}"
                        )

        except Exception as e:
            logger.error(f"Figure extraction failed for {pdf_path.name}: {e}")

        logger.info(f"Extracted {len(figures)} figures from {pdf_path.name}")
        return ExtractionResult(figures=figures, cap_reached=cap_reached)

    def _extract_page_figures(
        self,
        page,
        page_num: int,
        pdf,
        figures: list[ExtractedFigure],
        counter: int,
        per_page: int,
        processed: set[tuple[int, int, int, int]],
        page_width: float,
        page_height: float,
        page_area: float,
        is_landscape: bool,
        min_area_ratio: float,
        max_area_ratio: float,
        min_drawings: int,
    ) -> tuple[int, int]:
        """Extract figures from a single page using all three strategies.

        Returns updated (counter, per_page).
        """
        # --- Strategy 0: Vector figures ---
        try:
            drawings = page.get_drawings()
            if page.rotation:
                # get_drawings() reports rects in unrotated (mediabox) space,
                # while page.rect and get_pixmap(clip=...) use rotated page
                # space. Map the rects so clustering, the area/margin gates,
                # and the render clip all share the rotated frame (#41).
                rot = page.rotation_matrix
                for d in drawings:
                    rect = d.get("rect")
                    if rect is not None:
                        d["rect"] = (rect * rot).normalize()
            if len(drawings) >= min_drawings:
                regions = _vector_regions(drawings, page_width, page_height, page_area)

                for region_drawings, bbox in regions:
                    if counter > self.max_total or per_page >= self.max_per_page:
                        break

                    x0, y0, x1, y1 = bbox
                    w, h = x1 - x0, y1 - y0
                    area = w * h
                    ratio = area / page_area
                    has_data_marks = _has_vector_data_marks(region_drawings)

                    if area < MIN_AREA or w < 50 or h < 50:
                        continue
                    if ratio > max_area_ratio:
                        continue
                    if ratio < min_area_ratio:
                        dense_min_ratio = min(min_area_ratio, DENSE_VECTOR_MIN_AREA_RATIO)
                        if ratio < dense_min_ratio or not has_data_marks:
                            continue
                    if len(region_drawings) < min_drawings:
                        continue
                    if _looks_like_table_grid(region_drawings, bbox, ratio, has_data_marks):
                        continue

                    # Skip header/footer
                    if not is_landscape:
                        cy = (y0 + y1) / 2
                        in_margin = cy < page_height * HEADER_FOOTER_MARGIN or cy > page_height * (
                            1 - HEADER_FOOTER_MARGIN
                        )
                        if in_margin and len(region_drawings) < 20:
                            continue

                    key = (int(x0), int(y0), int(x1), int(y1))
                    if key in processed:
                        continue
                    processed.add(key)

                    img = _render_region(page, x0, y0, x1, y1, page_width, page_height)
                    if img is None:
                        continue

                    fig = ExtractedFigure(
                        figure_num=counter,
                        page_num=page_num,
                        image=img,
                        bbox=(x0, y0, x1, y1),
                    )
                    if self.save_dir:
                        fig.saved_path = str(self._save(img, counter, page_num))
                    figures.append(fig)
                    counter += 1
                    per_page += 1

                # Presentation fallback
                if is_landscape and per_page == 0 and len(drawings) >= 10:
                    fb_x0, fb_y0 = page_width * 0.05, page_height * 0.15
                    fb_x1, fb_y1 = page_width * 0.95, page_height * 0.90
                    img = _render_region(
                        page,
                        fb_x0,
                        fb_y0,
                        fb_x1,
                        fb_y1,
                        page_width,
                        page_height,
                    )
                    if img:
                        fig = ExtractedFigure(
                            figure_num=counter,
                            page_num=page_num,
                            image=img,
                            bbox=(fb_x0, fb_y0, fb_x1, fb_y1),
                        )
                        if self.save_dir:
                            fig.saved_path = str(self._save(img, counter, page_num))
                        figures.append(fig)
                        counter += 1
                        per_page += 1
        except Exception as e:
            logger.debug(f"Vector figure extraction failed on page {page_num}: {e}")

        # --- Strategy 1: IMAGE blocks ---
        try:
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if counter > self.max_total or per_page >= self.max_per_page:
                    break
                if block.get("type") != 1:
                    continue

                bbox = block.get("bbox")
                if not bbox:
                    continue

                x0, y0, x1, y1 = bbox
                w, h = x1 - x0, y1 - y0
                area = w * h
                aspect = w / max(h, 1)
                if area < MIN_AREA or aspect > 8 or aspect < 0.125:
                    continue

                # Logo/letterhead filter: suppress wide short images at the very
                # top of the page (title-page banners, institutional logos).
                if _is_logo_or_letterhead(x0, y0, x1, y1, page_width, page_height):
                    logger.debug(
                        "Suppressing likely logo/letterhead at (%.0f,%.0f,%.0f,%.0f) on page %d",
                        x0,
                        y0,
                        x1,
                        y1,
                        page_num,
                    )
                    continue

                key = (int(x0), int(y0), int(x1), int(y1))
                if key in processed:
                    continue
                processed.add(key)

                img = _render_region(page, x0, y0, x1, y1, page_width, page_height, padding=0)
                if img is None:
                    continue

                fig = ExtractedFigure(
                    figure_num=counter,
                    page_num=page_num,
                    image=img,
                    bbox=(x0, y0, x1, y1),
                )
                if self.save_dir:
                    fig.saved_path = str(self._save(img, counter, page_num))
                figures.append(fig)
                counter += 1
                per_page += 1
        except Exception as e:
            logger.debug(f"IMAGE block extraction failed on page {page_num}: {e}")

        # --- Strategy 2: Raw embedded images ---
        for img_info in page.get_images(full=True):
            if counter > self.max_total or per_page >= self.max_per_page:
                break

            xref = img_info[0]
            w, h = img_info[2], img_info[3]
            area = w * h
            aspect = w / max(h, 1)
            if area < MIN_AREA or aspect > 8 or aspect < 0.125:
                continue

            # Dedupe against placement-based captures: the same raster placed
            # on the page reports its bbox via get_image_rects, which matches
            # the IMAGE-block bbox Strategy 1 recorded in `processed` (#42).
            try:
                placement_rects = list(page.get_image_rects(img_info))
            except Exception:
                placement_rects = []
            placement_keys = [(int(r.x0), int(r.y0), int(r.x1), int(r.y1)) for r in placement_rects]
            if any(k in processed for k in placement_keys):
                continue
            processed.update(placement_keys)

            # Logo/letterhead filter: when any placement rect looks like a banner,
            # suppress this image.  Strategy 1 already deduped placed images that
            # matched an IMAGE block, so reaching here means the image was not in
            # the text dict — still apply the position/geometry check.
            if any(
                _is_logo_or_letterhead(r.x0, r.y0, r.x1, r.y1, page_width, page_height)
                for r in placement_rects
            ):
                logger.debug(
                    "Suppressing likely logo/letterhead (xref %d) on page %d",
                    xref,
                    page_num,
                )
                continue

            try:
                raw = pdf.extract_image(xref)
                if len(raw.get("image", b"")) < 5000:
                    continue
            except Exception as e:
                logger.debug(f"Image xref {xref} extraction failed on page {page_num}: {e}")
                continue

            img = _extract_xref_image(pdf, xref)
            if img is None:
                continue

            # Derive page-space bbox from the first placement rect when available
            # so downstream label recovery can filter native words by region.
            xref_bbox: tuple[float, float, float, float] | None = None
            if placement_rects:
                r = placement_rects[0]
                xref_bbox = (r.x0, r.y0, r.x1, r.y1)

            fig = ExtractedFigure(
                figure_num=counter,
                page_num=page_num,
                image=img,
                bbox=xref_bbox,
            )
            if self.save_dir:
                fig.saved_path = str(self._save(img, counter, page_num))
            figures.append(fig)
            counter += 1
            per_page += 1

        return counter, per_page

    def _save(self, img: Image.Image, fig_num: int, page_num: int) -> Path:
        path = self.save_dir / f"figure_{fig_num}_page{page_num}.png"
        img.save(path)
        return path


# --- Helpers ---


def _render_region(
    page,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    page_width: float,
    page_height: float,
    padding: int = 10,
) -> Image.Image | None:
    """Render a rectangular region of a PDF page to PIL Image."""
    import fitz
    from PIL import Image

    clip = fitz.Rect(
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(page_width, x1 + padding),
        min(page_height, y1 + padding),
    )
    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    try:
        pix = page.get_pixmap(matrix=mat, clip=clip)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        if max(img.size) > MAX_DIM:
            img.thumbnail((MAX_DIM, MAX_DIM))
        return img
    except Exception:
        return None


def _extract_xref_image(pdf, xref: int) -> Image.Image | None:
    """Extract an embedded image by xref and convert to RGB PIL Image."""
    import fitz
    from PIL import Image

    pix = None
    rgb = None
    try:
        pix = fitz.Pixmap(pdf, xref)
        if pix.colorspace is None:
            return None
        if pix.colorspace != fitz.csRGB or pix.alpha or pix.n != 3:
            rgb = fitz.Pixmap(fitz.csRGB, pix)
            pix = rgb
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        if max(img.size) > MAX_DIM:
            img.thumbnail((MAX_DIM, MAX_DIM))
        return img
    except Exception:
        return None
    finally:
        del rgb
        del pix


def _cluster_drawings(
    drawings: list[dict],
    page_width: float,
    page_height: float,
    cluster_gap: float,
) -> list[tuple[list[dict], tuple[float, float, float, float]]]:
    """Cluster drawings into figure regions using union-find on bounding boxes."""
    if not drawings:
        return []

    boxes = []
    for d in drawings:
        rect = d.get("rect")
        boxes.append((rect.x0, rect.y0, rect.x1, rect.y1) if rect else None)

    valid = [(i, boxes[i]) for i in range(len(boxes)) if boxes[i] is not None]
    if not valid:
        return []

    # Union-Find
    parent = {i: i for i, _ in valid}

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, (idx_i, box_i) in enumerate(valid):
        for j, (idx_j, box_j) in enumerate(valid):
            if i >= j:
                continue
            x0_i, y0_i, x1_i, y1_i = box_i
            x0_j, y0_j, x1_j, y1_j = box_j

            h_gap = (
                max(0, x0_j - x1_i) if x1_i < x0_j else max(0, x0_i - x1_j) if x1_j < x0_i else 0
            )
            v_gap = (
                max(0, y0_j - y1_i) if y1_i < y0_j else max(0, y0_i - y1_j) if y1_j < y0_i else 0
            )

            if h_gap <= cluster_gap and v_gap <= cluster_gap:
                union(idx_i, idx_j)

    clusters: dict[int, list[int]] = {}
    for idx, _ in valid:
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    results = []
    for indices in clusters.values():
        cboxes = [boxes[i] for i in indices if boxes[i] is not None]
        if not cboxes:
            continue
        x0 = min(b[0] for b in cboxes)
        y0 = min(b[1] for b in cboxes)
        x1 = max(b[2] for b in cboxes)
        y1 = max(b[3] for b in cboxes)
        results.append(([drawings[i] for i in indices], (x0, y0, x1, y1)))

    results.sort(key=lambda r: (r[1][1], r[1][0]))
    return results


def _vector_regions(
    drawings: list[dict],
    page_width: float,
    page_height: float,
    page_area: float,
) -> list[tuple[list[dict], tuple[float, float, float, float]]]:
    """Return vector regions, splitting page-spanning bridged clusters.

    A coarse gap keeps legitimate multi-part figures together, but dense
    dashboard pages can bridge unrelated table/chart panels into one near-page
    crop. Oversized coarse regions are reclustered with a fine gap; ordinary
    regions keep the historical behavior.
    """
    coarse = _cluster_drawings(drawings, page_width, page_height, CLUSTER_GAP)
    regions: list[tuple[list[dict], tuple[float, float, float, float]]] = []
    for region_drawings, bbox in coarse:
        ratio = _bbox_area_ratio(bbox, page_area)
        if ratio > VECTOR_SPLIT_AREA_RATIO:
            fine = _cluster_drawings(region_drawings, page_width, page_height, FINE_CLUSTER_GAP)
            if len(fine) > 1:
                regions.extend(fine)
                continue
        regions.append((region_drawings, bbox))
    return _dedupe_regions(regions)


def _dedupe_regions(
    regions: list[tuple[list[dict], tuple[float, float, float, float]]],
) -> list[tuple[list[dict], tuple[float, float, float, float]]]:
    seen: set[tuple[int, int, int, int]] = set()
    out = []
    for drawings, bbox in sorted(regions, key=lambda r: (r[1][1], r[1][0])):
        key = tuple(int(v) for v in bbox)
        if key in seen:
            continue
        seen.add(key)
        out.append((drawings, bbox))
    return out


def _bbox_area_ratio(bbox: tuple[float, float, float, float], page_area: float) -> float:
    x0, y0, x1, y1 = bbox
    return ((x1 - x0) * (y1 - y0)) / max(page_area, 1)


def _is_logo_or_letterhead(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    page_width: float,
    page_height: float,
) -> bool:
    """Return True when an image block looks like a title-page logo or letterhead banner.

    Three geometry conditions — all derived from page dimensions, no hard-coded
    pixel values — must hold simultaneously:

    1. The image top sits inside the top ``LOGO_TOP_MARGIN_RATIO`` band of the page.
    2. The image spans at least ``LOGO_WIDE_WIDTH_RATIO`` of the page width.
    3. The image height is at most ``LOGO_HEIGHT_RATIO`` of the page height.

    A genuine substantive figure near the top of the page typically violates at
    least one condition (it is too tall, or it does not span the full width).
    The filter is therefore conservative: when any single condition is false the
    candidate is passed through and treated as a real figure.
    """
    in_top_band = y0 < page_height * LOGO_TOP_MARGIN_RATIO
    is_wide = (x1 - x0) >= page_width * LOGO_WIDE_WIDTH_RATIO
    is_short = (y1 - y0) <= page_height * LOGO_HEIGHT_RATIO
    return in_top_band and is_wide and is_short


def _has_vector_data_marks(drawings: list[dict]) -> bool:
    """True for clusters with chart-like bars/series rather than pure ruling.

    Small vector charts in reports are often below the normal area gate, so they
    need a positive visual signal: colored fills (bars/points) or thick strokes
    (line series). Thin gray/black ruling alone is treated as table/layout.
    """
    marks = 0
    for d in drawings:
        fill = d.get("fill")
        if fill is not None and not _is_neutral_color(fill):
            marks += 1
        width = float(d.get("width") or 0)
        color = d.get("color")
        if width >= DATA_STROKE_MIN_WIDTH and (
            not _is_neutral_color(color) or width >= DATA_STROKE_MIN_WIDTH * 2
        ):
            marks += 1
        if marks >= MIN_DATA_MARKS:
            return True
    return False


def _is_neutral_color(color) -> bool:
    if color is None:
        return True
    try:
        r, g, b = (float(c) for c in color[:3])
    except Exception:
        return True
    return max(r, g, b) - min(r, g, b) < 0.08


def _looks_like_table_grid(
    drawings: list[dict],
    bbox: tuple[float, float, float, float],
    ratio: float,
    has_data_marks: bool,
) -> bool:
    """Reject large ruling-only grids so tables are not emitted as figures."""
    if has_data_marks or ratio < TABLE_GRID_MIN_AREA_RATIO:
        return False

    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    horizontal = 0
    vertical = 0
    for d in drawings:
        for item in d.get("items", []):
            kind = item[0] if item else None
            if kind == "l" and len(item) >= 3:
                p1, p2 = item[1], item[2]
                dx = abs(p2.x - p1.x)
                dy = abs(p2.y - p1.y)
                if dy <= 1 and dx >= width * 0.4:
                    horizontal += 1
                elif dx <= 1 and dy >= height * 0.4:
                    vertical += 1
            elif kind == "re" and len(item) >= 2:
                rect = item[1]
                if rect.width >= width * 0.4:
                    horizontal += 2
                if rect.height >= height * 0.4:
                    vertical += 2
    return horizontal >= 3 and vertical >= 2
