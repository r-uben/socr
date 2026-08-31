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
    ``cap_page`` is the 1-indexed page number where extraction was stopped
    by the cap (the first page that was not processed); ``None`` when the
    cap was not hit.
    """

    figures: list[ExtractedFigure] = field(default_factory=list)
    cap_reached: bool = False
    cap_page: int | None = None


@dataclass
class PageExtractionResult:
    """Returned by ``FigureExtractor.extract_page()`` (PP-4 / GH-86).

    ``next_counter`` is the doc-global figure counter after this page finishes.
    ``cap_reached`` is ``True`` when ``max_total`` was hit on or during this page.
    """

    figures: list[ExtractedFigure] = field(default_factory=list)
    next_counter: int = 1
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

# --- Chart-lane detection (PP-7) ---
# Minimum bounding-box area (pts²) a vector cluster must span before it is
# eligible for the chart-asset lane.  Derived from the existing MIN_AREA gate
# (80×80 = 6400 pts²) scaled up to require a meaningfully-sized chart region
# rather than a small decorative mark (e.g. a bullet or horizontal rule).
# 120×120 = 14 400 pts² ≈ ~4.2 cm × 4.2 cm at 72 dpi — roughly the smallest
# chart that can convey information.  Kept as a named constant (not a magic
# literal) so it is tunable from data.
CHART_MIN_CLUSTER_AREA: float = 120.0 * 120.0

# Minimum fraction of a cluster's bbox area that a single drawing must cover to
# be treated as an enclosing plot frame (axes box) rather than a datum inside
# the plot. Derived from two measured populations on real corpus pages: plot
# frames cover 0.85-0.99 of their cluster bbox (Heston p10 / page index 9: a
# 932-drawing cluster whose largest single mark covers 0.876 of the cluster
# area, with 928 marks interior to that frame; `_has_vector_data_marks` is
# False on this cluster because the strokes are thin and neutral-coloured).
# The largest single mark in measured table-ruling clusters covers <= 0.06 of
# its cluster area. 0.5 sits far from both populations: a mark covering half
# its cluster IS the frame, not a datum.
CHART_FRAME_MIN_CLUSTER_COVERAGE: float = 0.5

# Tolerance (pts) below which a line segment counts as horizontal/vertical.
# Same convention as the axis tests in `_looks_like_table_grid` (dx/dy <= 1pt):
# absorbs sub-point float noise from PDF transform matrices without admitting
# genuinely diagonal strokes.
AXIS_LINE_TOLERANCE_PT: float = 1.0

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
        from socr.core.pdf import open_pdf

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        figures: list[ExtractedFigure] = []
        counter = 1
        cap_reached = False
        cap_page: int | None = None
        if skip_pages:
            logger.info(f"Figure extraction skipping {len(skip_pages)} page(s) of {pdf_path.name}")

        try:
            with open_pdf(pdf_path) as pdf:
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
                        # 1-indexed page where extraction was stopped (first skipped page).
                        cap_page = page_index + 1
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
        return ExtractionResult(figures=figures, cap_reached=cap_reached, cap_page=cap_page)

    def extract_page(
        self,
        page,
        page_num: int,
        pdf,
        counter: int,
    ) -> PageExtractionResult:
        """Extract figures from a single open fitz page (PP-4 per-page entry).

        The caller threads ``counter`` doc-globally across pages so
        ``figure_<N>_page<P>.png`` filenames stay monotonic.  Returns an
        empty figure list (``cap_reached=True``) when ``counter`` already
        exceeds ``max_total``.
        """
        if counter > self.max_total:
            return PageExtractionResult(figures=[], next_counter=counter, cap_reached=True)

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        figures: list[ExtractedFigure] = []
        per_page = 0
        processed: set[tuple[int, int, int, int]] = set()

        page_width = page.rect.width
        page_height = page.rect.height
        page_area = page_width * page_height
        is_landscape = page_width > page_height

        min_area_ratio = MIN_VECTOR_AREA_RATIO * 0.5 if is_landscape else MIN_VECTOR_AREA_RATIO
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
                "Figure extraction timed out on page %d, skipping page figures",
                page_num,
            )
        except Exception as e:
            logger.warning(
                "Figure extraction failed on page %d: %s: %s",
                page_num,
                type(e).__name__,
                e,
            )

        return PageExtractionResult(
            figures=figures,
            next_counter=counter,
            cap_reached=counter > self.max_total,
        )

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
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

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


def _has_framed_data_cluster(
    cluster_drawings: list[dict], bbox: tuple[float, float, float, float]
) -> bool:
    """True when the cluster looks like a plot frame enclosing many data marks.

    Detector-only companion to ``_has_vector_data_marks`` (which is NOT
    modified here). Thin-stroke, monochrome spike plots fail the coloured-
    fill/thick-stroke test above, but they do draw an enclosing axes rectangle
    around dozens to hundreds of interior marks. Two measured populations on
    real corpus pages separate cleanly on coverage of a single mark's area
    relative to its cluster's bbox area: plot frames cover 0.85-0.99, while
    the largest single mark in table-ruling clusters covers <= 0.06 — see
    ``CHART_FRAME_MIN_CLUSTER_COVERAGE`` for the derivation.

    Only rect geometry plus item *kinds* are consulted (no width/color/fill)
    so this stays a spatial-containment signal, independent of
    ``_has_vector_data_marks``. Two structural guards keep the gate honest
    without new tuned thresholds:

    - A frame candidate must be built entirely of axis-aligned strokes
      (``re`` items and horizontal/vertical ``l`` segments, per
      ``AXIS_LINE_TOLERANCE_PT``). A large polyline, curve, or diagonal
      decoration whose *bounding box* happens to span the cluster must not
      qualify as a frame. (The frame need not be a single ``re`` primitive:
      real axes boxes — e.g. Heston p10 — are drawn as line segments plus
      tick marks in one path.)
    - An interior drawing that itself covers
      >= ``CHART_FRAME_MIN_CLUSTER_COVERAGE`` of the cluster is frame-like:
      by the same definitional bound that selects the frame, it *is* a frame,
      not a datum. It does not count toward the interior-mark threshold, so a
      duplicate copy of the frame (fill + stroke emitted as two drawings)
      cannot count itself.
    """
    x0, y0, x1, y1 = bbox
    cluster_area = (x1 - x0) * (y1 - y0)
    if cluster_area <= 0:
        return False

    frames: list[tuple[float, float, float, float]] = []
    marks: list[tuple[float, float, float, float]] = []
    for d in cluster_drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        mark_area = (rect.x1 - rect.x0) * (rect.y1 - rect.y0)
        if mark_area / cluster_area >= CHART_FRAME_MIN_CLUSTER_COVERAGE:
            if _is_axis_aligned_path(d):
                frames.append((rect.x0, rect.y0, rect.x1, rect.y1))
            continue
        marks.append((rect.x0, rect.y0, rect.x1, rect.y1))

    for fx0, fy0, fx1, fy1 in frames:
        interior = 0
        for ox0, oy0, ox1, oy1 in marks:
            if ox0 >= fx0 and oy0 >= fy0 and ox1 <= fx1 and oy1 <= fy1:
                interior += 1
                if interior >= MIN_DRAWINGS_FOR_VECTOR:
                    return True

    return False


def _is_axis_aligned_path(drawing: dict) -> bool:
    """True when every item is a rect or a horizontal/vertical line segment."""
    items = drawing.get("items", [])
    if not items:
        return False
    for item in items:
        kind = item[0] if item else None
        if kind == "re":
            continue
        if kind == "l" and len(item) >= 3:
            p1, p2 = item[1], item[2]
            if (
                abs(p2.x - p1.x) <= AXIS_LINE_TOLERANCE_PT
                or abs(p2.y - p1.y) <= AXIS_LINE_TOLERANCE_PT
            ):
                continue
        return False
    return True


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
                if dy <= AXIS_LINE_TOLERANCE_PT and dx >= width * 0.4:
                    horizontal += 1
                elif dx <= AXIS_LINE_TOLERANCE_PT and dy >= height * 0.4:
                    vertical += 1
            elif kind == "re" and len(item) >= 2:
                rect = item[1]
                if rect.width >= width * 0.4:
                    horizontal += 2
                if rect.height >= height * 0.4:
                    vertical += 2
    return horizontal >= 3 and vertical >= 2


#: GH-369 (cubic P2): the smallest run of consecutive bare-number lines that
#: can be an axis scale. Two, because a scale is a sequence of ticks and a lone
#: number between sentences is content -- a year, a headline stat, a footnote
#: marker. Structural, not tuned: no page-level ratio is consulted anywhere.
_MIN_AXIS_SCALE_TICKS = 2

_CHART_AXIS_FENCE_OPEN = "<!-- socr:chart-axis-residue"
_CHART_AXIS_FENCE_CLOSE = "socr:end-chart-axis-residue -->"


def _is_bare_number_line(line: str) -> bool:
    """Whether *line* is nothing but a number.

    GH-369. Deliberately narrow: only a line whose ENTIRE stripped content is a
    numeric literal (optionally signed, decimal, or percent). Axis tick scales
    render exactly this way -- ``2``, ``4``, ``18``, ``-0.5``, ``25%`` each
    alone on a line -- and it is these that are indistinguishable from a series
    of data values downstream.

    Word-shaped tick labels ("Lower", "Broadly similar", "Higher") are NOT
    matched. They are ordinary prose to any reader, they do not read as data,
    and a classifier confident enough to catch them would be confident enough
    to swallow a real sentence. There is no share-of-page threshold anywhere in
    this path: each line is judged on its own content.
    """
    text = line.strip()
    if not text:
        return False
    if text.endswith("%"):
        text = text[:-1].strip()
    if text[:1] in "+-":
        text = text[1:]
    if not text:
        return False
    return text.replace(".", "", 1).isdigit()


def split_chart_axis_residue(native_text: str) -> tuple[str, list[str]]:
    """Split *native_text* into (body, axis-residue lines).

    GH-369. A page routed to the chart-asset lane saves the chart as an image
    and records "data values not transcribed" -- then ships the native text
    layer whole, axis tick scales included. On the reported page 52% of
    non-empty lines were a lone number, which is indistinguishable from a
    series of real values and contradicts the lane's own declaration.

    The residue is SEPARATED, never dropped: a wrong or dropped number is worse
    than a missing one, so every input line is returned in one of the two
    halves and the caller fences the residue rather than deleting it. Order is
    preserved WITHIN each half, but the interleaving between them is not: the
    residue moves to a trailing fence, so the original line order is
    deliberately lost. Nothing is lost as content; the page is not
    byte-reconstructible from the output.

    Returns ``(body, residue)``. ``residue`` is empty for any page with no bare
    numeric lines, which leaves every non-chart page byte-identical.
    """
    lines = native_text.splitlines()
    body: list[str] = []
    residue: list[str] = []

    index = 0
    while index < len(lines):
        if not _is_bare_number_line(lines[index]):
            body.append(lines[index])
            index += 1
            continue
        run_end = index
        while run_end < len(lines) and _is_bare_number_line(lines[run_end]):
            run_end += 1
        run = lines[index:run_end]
        # A scale is a SEQUENCE of ticks. One number standing alone between
        # sentences is a year, a headline stat, or a footnote marker -- content,
        # not chart furniture -- so it stays visible in the body. This is a
        # structural minimum drawn from what an axis IS, not a tuned cutoff:
        # there is no such thing as a one-tick scale.
        (residue if len(run) >= _MIN_AXIS_SCALE_TICKS else body).extend(run)
        index = run_end

    return "\n".join(body), residue


def fence_chart_axis_residue(native_text: str) -> str:
    """Body text with bare-numeric axis lines moved into a fenced block.

    GH-369. The fence is an HTML comment, so the residue survives verbatim in
    the file (auditable, greppable, machine-distinguishable) while not
    rendering as prose beside the chart image it belongs to. Returns
    *native_text* unchanged when there is nothing to fence.
    """
    body, residue = split_chart_axis_residue(native_text)
    if not residue:
        return native_text
    fenced = "\n".join(
        [
            _CHART_AXIS_FENCE_OPEN,
            "axis tick labels from the chart on this page; not data values.",
            *residue,
            _CHART_AXIS_FENCE_CLOSE,
        ]
    )
    body = body.rstrip()
    return f"{body}\n\n{fenced}" if body else fenced


def has_chart_marks(page) -> bool:
    """Cluster-first chart detector for the PP-7 chart-asset routing lane.

    Returns True when the page contains at least one spatially-coherent vector
    cluster that (a) meets the minimum bounding-box area gate
    (``CHART_MIN_CLUSTER_AREA``) and (b) passes ``_has_vector_data_marks``
    (coloured fills or thick strokes) OR ``_has_framed_data_cluster`` (an
    axis-aligned frame path covering >= ``CHART_FRAME_MIN_CLUSTER_COVERAGE``
    of the cluster bbox plus >= ``MIN_DRAWINGS_FOR_VECTOR`` interior marks —
    catches thin-stroke monochrome spike plots that have no coloured fill or
    thick stroke).
    Also returns True when the page carries at least one embedded raster image
    (``page.get_images()``), because an embedded PNG/JPEG chart is visually lost
    as native word-salad just like a vector one.

    Note on ``_looks_like_table_grid``: this function is intentionally NOT called
    here.  Its first-line short-circuit ``if has_data_marks: return False`` means
    it always returns False once the data-marks gate passes — calling it would be
    dead code.  Table pages are excluded upstream by ``_page_has_tables`` in
    ``_is_chart_asset_page``, which is the correct structural boundary.

    Detection is fully deterministic and model-free; it reuses the same named-
    constant thresholds already trusted by ``FigureExtractor``.

    NOTE: Calling ``get_drawings()`` on every born-digital non-table page is a
    moderate cost, but it is the SAME call the figure-extraction phase already
    makes for the same page — no new I/O.

    Cluster-first (load-bearing): we do NOT count isolated marks on the flat
    ``get_drawings()`` list, because scattered decorative rules (e.g. three thin
    horizontal lines on a title page) can reach ``MIN_DATA_MARKS`` and false-
    trigger the chart lane.  Requiring a spatially-coherent cluster first
    suppresses those false positives.

    Logging: logs mark counts, cluster counts, and rejection reasons at DEBUG
    level so operators can trace routing decisions without re-running the pipeline.
    """
    page_width = page.rect.width
    page_height = page.rect.height
    page_area = page_width * page_height

    # Fast path: embedded raster image present → raster chart.
    try:
        raster_images = page.get_images()
        if raster_images:
            logger.debug(
                "has_chart_marks p%s: raster path — %d embedded image(s)",
                getattr(page, "number", "?") + 1,
                len(raster_images),
            )
            return True
    except Exception as exc:
        logger.debug("has_chart_marks: get_images() failed: %s", exc)

    # Vector path: cluster drawings, then gate each cluster.
    try:
        drawings = page.get_drawings()
    except Exception as exc:
        logger.debug("has_chart_marks: get_drawings() failed: %s", exc)
        return False

    page_num_label = getattr(page, "number", "?")
    try:
        page_num_label = page.number + 1
    except Exception:
        pass

    if not drawings:
        logger.debug("has_chart_marks p%s: no drawings", page_num_label)
        return False

    clusters = _cluster_drawings(drawings, page_width, page_height, CLUSTER_GAP)
    logger.debug(
        "has_chart_marks p%s: %d drawings → %d cluster(s)",
        page_num_label,
        len(drawings),
        len(clusters),
    )

    for cluster_drawings, bbox in clusters:
        x0, y0, x1, y1 = bbox
        area = (x1 - x0) * (y1 - y0)

        # (a) Minimum bounding-box area gate.
        if area < CHART_MIN_CLUSTER_AREA:
            logger.debug(
                "has_chart_marks p%s: cluster bbox=%.0f×%.0f area=%.0f < "
                "CHART_MIN_CLUSTER_AREA=%.0f — skip (too small)",
                page_num_label,
                x1 - x0,
                y1 - y0,
                area,
                CHART_MIN_CLUSTER_AREA,
            )
            continue

        # (b) Must pass a positive gate: coloured/thick data marks, or (only
        # when those are absent) a framed cluster of thin-stroke interior marks.
        has_data_marks = _has_vector_data_marks(cluster_drawings)
        has_framed_cluster = not has_data_marks and _has_framed_data_cluster(cluster_drawings, bbox)
        if not (has_data_marks or has_framed_cluster):
            logger.debug(
                "has_chart_marks p%s: cluster area=%.0f — no data marks and no "
                "framed data cluster; skip",
                page_num_label,
                area,
            )
            continue

        ratio = area / max(page_area, 1.0)
        gate_name = "data-marks" if has_data_marks else "framed-cluster"
        logger.debug(
            "has_chart_marks p%s: CHART cluster found (%s) — area=%.0f ratio=%.3f %d drawings",
            page_num_label,
            gate_name,
            area,
            ratio,
            len(cluster_drawings),
        )
        return True

    logger.debug(
        "has_chart_marks p%s: no qualifying chart cluster in %d cluster(s)",
        page_num_label,
        len(clusters),
    )
    return False
