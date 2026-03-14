"""Figure extraction from PDF files.

Consolidates the ~400 lines duplicated between processor.py and
hpc_sequential_pipeline.py into a single shared module.

Three extraction strategies (applied per page, in order):
  0. Vector figure clustering (union-find on drawing bounding boxes)
  1. IMAGE blocks from page text dict
  2. Raw embedded images via xref
"""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFigure:
    """A figure image extracted from a PDF page."""

    figure_num: int
    page_num: int
    image: "Image.Image"  # PIL Image (lazy import)
    saved_path: str | None = None


# --- Defaults ---
RENDER_DPI = 150
MAX_DIM = 1024
MIN_AREA = 80 * 80
MIN_DRAWINGS_FOR_VECTOR = 5
MIN_VECTOR_AREA_RATIO = 0.05
MAX_VECTOR_AREA_RATIO = 0.85
HEADER_FOOTER_MARGIN = 0.1
CLUSTER_GAP = 30


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

    def extract(self, pdf_path: Path) -> list[ExtractedFigure]:
        """Extract all figures from a PDF. Returns list of ExtractedFigure."""
        import fitz
        from PIL import Image

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        figures: list[ExtractedFigure] = []
        counter = 1

        try:
            with fitz.open(pdf_path) as pdf:
                for page_index in range(len(pdf)):
                    if counter > self.max_total:
                        break

                    page = pdf[page_index]
                    page_num = page_index + 1
                    per_page = 0
                    processed: set[tuple[int, int, int, int]] = set()

                    page_width = page.rect.width
                    page_height = page.rect.height
                    page_area = page_width * page_height
                    is_landscape = page_width > page_height

                    # Landscape adjustments
                    min_area_ratio = MIN_VECTOR_AREA_RATIO * 0.5 if is_landscape else MIN_VECTOR_AREA_RATIO
                    max_area_ratio = 0.98 if is_landscape else MAX_VECTOR_AREA_RATIO
                    min_drawings = 3 if is_landscape else MIN_DRAWINGS_FOR_VECTOR

                    # --- Strategy 0: Vector figures ---
                    try:
                        drawings = page.get_drawings()
                        if len(drawings) >= min_drawings:
                            regions = _cluster_drawings(drawings, page_width, page_height, CLUSTER_GAP)

                            for region_drawings, bbox in regions:
                                if counter > self.max_total or per_page >= self.max_per_page:
                                    break

                                x0, y0, x1, y1 = bbox
                                w, h = x1 - x0, y1 - y0
                                area = w * h
                                ratio = area / page_area

                                if area < MIN_AREA or w < 50 or h < 50:
                                    continue
                                if ratio < min_area_ratio or ratio > max_area_ratio:
                                    continue
                                if len(region_drawings) < min_drawings:
                                    continue

                                # Skip header/footer
                                if not is_landscape:
                                    cy = (y0 + y1) / 2
                                    in_margin = cy < page_height * HEADER_FOOTER_MARGIN or cy > page_height * (1 - HEADER_FOOTER_MARGIN)
                                    if in_margin and len(region_drawings) < 20:
                                        continue

                                key = (int(x0), int(y0), int(x1), int(y1))
                                if key in processed:
                                    continue
                                processed.add(key)

                                img = _render_region(page, x0, y0, x1, y1, page_width, page_height)
                                if img is None:
                                    continue

                                fig = ExtractedFigure(figure_num=counter, page_num=page_num, image=img)
                                if self.save_dir:
                                    fig.saved_path = str(self._save(img, counter, page_num))
                                figures.append(fig)
                                counter += 1
                                per_page += 1

                            # Presentation fallback
                            if is_landscape and per_page == 0 and len(drawings) >= 10:
                                img = _render_region(
                                    page,
                                    page_width * 0.05, page_height * 0.15,
                                    page_width * 0.95, page_height * 0.90,
                                    page_width, page_height,
                                )
                                if img:
                                    fig = ExtractedFigure(figure_num=counter, page_num=page_num, image=img)
                                    if self.save_dir:
                                        fig.saved_path = str(self._save(img, counter, page_num))
                                    figures.append(fig)
                                    counter += 1
                                    per_page += 1
                    except Exception:
                        pass

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

                            key = (int(x0), int(y0), int(x1), int(y1))
                            if key in processed:
                                continue
                            processed.add(key)

                            img = _render_region(page, x0, y0, x1, y1, page_width, page_height, padding=0)
                            if img is None:
                                continue

                            fig = ExtractedFigure(figure_num=counter, page_num=page_num, image=img)
                            if self.save_dir:
                                fig.saved_path = str(self._save(img, counter, page_num))
                            figures.append(fig)
                            counter += 1
                            per_page += 1
                    except Exception:
                        pass

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

                        try:
                            raw = pdf.extract_image(xref)
                            if len(raw.get("image", b"")) < 5000:
                                continue
                        except Exception:
                            continue

                        img = _extract_xref_image(pdf, xref)
                        if img is None:
                            continue

                        fig = ExtractedFigure(figure_num=counter, page_num=page_num, image=img)
                        if self.save_dir:
                            fig.saved_path = str(self._save(img, counter, page_num))
                        figures.append(fig)
                        counter += 1
                        per_page += 1

        except Exception as e:
            logger.error(f"Figure extraction failed: {e}")

        logger.info(f"Extracted {len(figures)} figures from {pdf_path.name}")
        return figures

    def _save(self, img: "Image.Image", fig_num: int, page_num: int) -> Path:
        path = self.save_dir / f"figure_{fig_num}_page{page_num}.png"
        img.save(path)
        return path


# --- Helpers ---

def _render_region(
    page,
    x0: float, y0: float, x1: float, y1: float,
    page_width: float, page_height: float,
    padding: int = 10,
) -> "Image.Image | None":
    """Render a rectangular region of a PDF page to PIL Image."""
    import fitz
    from PIL import Image

    clip = fitz.Rect(
        max(0, x0 - padding), max(0, y0 - padding),
        min(page_width, x1 + padding), min(page_height, y1 + padding),
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


def _extract_xref_image(pdf, xref: int) -> "Image.Image | None":
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

            h_gap = max(0, x0_j - x1_i) if x1_i < x0_j else max(0, x0_i - x1_j) if x1_j < x0_i else 0
            v_gap = max(0, y0_j - y1_i) if y1_i < y0_j else max(0, y0_i - y1_j) if y1_j < y0_i else 0

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
