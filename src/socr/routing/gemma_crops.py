"""300-dpi half-page crops for the Gemma OCR rung."""

from __future__ import annotations

from pathlib import Path

from socr.core.born_digital import upright_rotation_for


def render_page_half_crops(
    pdf_path: Path,
    page_num: int,
    dest_dir: Path,
    *,
    dpi: int,
) -> list[tuple[str, Path]]:
    """Render left and right half crops; return ``(stem, path)`` pairs."""
    import fitz
    from PIL import Image

    from socr.core.pdf import open_pdf

    dest_dir.mkdir(parents=True, exist_ok=True)
    stems: list[tuple[str, Path]] = []
    with open_pdf(pdf_path) as doc:
        page = doc[page_num - 1]
        rect = page.rect
        mid_x = (rect.x0 + rect.x1) / 2.0
        rotation = upright_rotation_for(page)
        for label, clip in (
            ("L", fitz.Rect(rect.x0, rect.y0, mid_x, rect.y1)),
            ("R", fitz.Rect(mid_x, rect.y0, rect.x1, rect.y1)),
        ):
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            if rotation:
                mat = mat.prerotate(rotation)
            pix = page.get_pixmap(matrix=mat, clip=clip)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            stem = f"page_{page_num:04d}_{label}"
            path = dest_dir / f"{stem}.png"
            img.save(path)
            stems.append((stem, path))
    return stems
