"""Rasterize born-digital PDFs into synthetic scanned (image-only) PDFs.

Renders each page to a pixmap at a specified DPI, then reconstructs
the document as an image-only PDF for testing OCR engines on scanned input.
"""

from __future__ import annotations

from pathlib import Path

import fitz


class PaperRasterizer:
    """Convert born-digital PDFs to image-only PDFs."""

    def rasterize(self, pdf_path: Path, output_path: Path, dpi: int = 200) -> Path:
        """Rasterize a PDF to images and reconstruct as image-only PDF.

        Each page is rendered at the given DPI, then inserted into a new
        PDF as a full-page image. The result has no text layer.

        Args:
            pdf_path: Path to the source PDF.
            output_path: Path for the output image-only PDF.
            dpi: Resolution for rasterization (default 200).

        Returns:
            Path to the created image-only PDF.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        src_doc = fitz.open(pdf_path)
        out_doc = fitz.open()

        try:
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)

            for page_idx in range(len(src_doc)):
                src_page = src_doc[page_idx]
                pix = src_page.get_pixmap(matrix=mat)

                # Create a new page with the same dimensions as the source
                page_rect = src_page.rect
                new_page = out_doc.new_page(
                    width=page_rect.width,
                    height=page_rect.height,
                )

                # Insert the rasterized image as a full-page image
                img_bytes = pix.tobytes("png")
                new_page.insert_image(page_rect, stream=img_bytes)

            out_doc.save(str(output_path))
        finally:
            src_doc.close()
            out_doc.close()

        return output_path


# Papers to rasterize for the benchmark scanned category.
RASTERIZE_SPECS: list[dict] = [
    {
        "source_name": "guerrieri_lorenzoni_werning__global_price_shocks",
        "output_name": "guerrieri_lorenzoni_werning__scanned_200dpi",
        "dpi": 200,
        "notes": "Math challenge for scanned OCR",
    },
    {
        "source_name": "goldsmith_pinkham_hirtle_lucca__parsing_bank_supervision",
        "output_name": "goldsmith_pinkham_hirtle_lucca__scanned_200dpi",
        "dpi": 200,
        "notes": "Table challenge for scanned OCR",
    },
]
