"""Lazy document handle for OCR processing.

DocumentHandle holds only the PDF path and metadata — no page rendering.
Engines receive the PDF path directly and call their CLI on the whole document.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DocumentHandle:
    """Lazy handle to a PDF document. No PIL rendering, no page images in memory."""

    path: Path
    page_count: int = 0
    file_hash: str = ""
    _file_size_bytes: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.path, str):
            self.path = Path(self.path)
        if self.path.exists() and not self._file_size_bytes:
            self._file_size_bytes = self.path.stat().st_size
        if not self.page_count and self.path.exists():
            self.page_count = self._count_pages()
        if not self.file_hash and self.path.exists():
            self.file_hash = self._compute_hash()

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def size_mb(self) -> float:
        return self._file_size_bytes / (1024 * 1024)

    def _count_pages(self) -> int:
        """Count pages without rendering them."""
        import fitz

        with fitz.open(self.path) as pdf:
            return len(pdf)

    def _compute_hash(self) -> str:
        """SHA256 of file contents for change detection."""
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def render_page(self, page_num: int, dpi: int = 200) -> "Image.Image":
        """Render a single page to PIL Image on demand (1-indexed)."""
        import fitz
        from PIL import Image

        with fitz.open(self.path) as pdf:
            page = pdf[page_num - 1]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    def render_all_pages(self, dpi: int = 200) -> dict[int, "Image.Image"]:
        """Render all pages to PIL Images. Returns {page_num: Image}."""
        import fitz
        from PIL import Image

        images = {}
        with fitz.open(self.path) as pdf:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            for i in range(len(pdf)):
                pix = pdf[i].get_pixmap(matrix=mat)
                images[i + 1] = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return images

    @classmethod
    def from_path(cls, path: Path | str) -> "DocumentHandle":
        """Create a DocumentHandle from a file path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if not path.suffix.lower() == ".pdf":
            raise ValueError(f"Not a PDF: {path}")
        return cls(path=path)
