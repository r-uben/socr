"""Crop-pass table extractor (Pass B).

Crops each located table to a high-resolution PNG and re-reads it with a
table-specialised VLM pass. Reusing the judge's direct image -> Ollama path
(``/api/generate`` with a base64 image) rather than the CLI engines, because we
want to hand the model *just the table crop*, not a whole rendered page.

The VLM call is injected (``TableReader`` protocol) so the cropping/orchestration
logic is testable without a model. ``OllamaTableReader`` is the default backend,
mirroring ``OllamaVisionJudge``.
"""

from __future__ import annotations

import base64
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from socr.tables.locate import TableBox

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "table_extract.md"

# Crops cover a small fraction of a page, so they can afford a higher render DPI
# than a full page without large images: small table digits/parens become crisp.
# This is a rendering setting, not a model threshold.
DEFAULT_CROP_DPI = 400
# Padding (PDF points) around a located bbox so a rule or edge digit is never
# clipped by an off-by-a-pixel boundary.
_CROP_PADDING_PT = 6.0


def load_table_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


class TableReader(Protocol):
    """Anything that turns a table-crop image into Markdown."""

    def read(self, image_path: Path) -> str: ...


@dataclass
class CropTable:
    """One crop-pass result, in reading order."""

    markdown: str
    source: str  # locator tag from TableBox ("ruled" | "booktabs")
    bbox: tuple[float, float, float, float]


class OllamaTableReader:
    """Default crop reader: a local/cloud Ollama vision model via /api/generate."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._prompt = load_table_prompt()

    def read(self, image_path: Path) -> str:
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        resp = httpx.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": self._prompt,
                "images": [image_b64],
                "stream": False,
                "options": {"temperature": 0},  # transcription must be deterministic
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return _clean_markdown(resp.json().get("response", ""))


class TableCropExtractor:
    """Render table crops and read them with an injected ``TableReader``."""

    def __init__(self, reader: TableReader, crop_dpi: int = DEFAULT_CROP_DPI) -> None:
        self._reader = reader
        self._crop_dpi = crop_dpi

    def extract(self, pdf_path: Path, page_num: int, boxes: list[TableBox]) -> list[CropTable]:
        """Crop each box on ``page_num`` (1-indexed) and read it. Never raises.

        A failed crop/read drops that table (returns no entry for it) rather than
        aborting the page — the reconciler then sees a count mismatch and flags
        rather than patches, which is the safe outcome.
        """
        import fitz

        out: list[CropTable] = []
        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("dual-pass: cannot open %s (%s)", pdf_path, exc)
            return out
        try:
            page = doc[page_num - 1]
            page_rect = page.rect
            for box in boxes:
                img_path = self._render_crop(page, box, page_rect)
                if img_path is None:
                    continue
                try:
                    md = self._reader.read(img_path)
                except Exception as exc:
                    logger.warning("dual-pass: crop read failed p%d (%s)", page_num, exc)
                    continue
                finally:
                    img_path.unlink(missing_ok=True)
                if md.strip():
                    out.append(CropTable(markdown=md, source=box.source, bbox=box.bbox))
        finally:
            doc.close()
        return out

    def _render_crop(self, page, box: TableBox, page_rect) -> Path | None:
        import fitz
        from PIL import Image

        x0, y0, x1, y1 = box.bbox
        clip = fitz.Rect(
            max(page_rect.x0, x0 - _CROP_PADDING_PT),
            max(page_rect.y0, y0 - _CROP_PADDING_PT),
            min(page_rect.x1, x1 + _CROP_PADDING_PT),
            min(page_rect.y1, y1 + _CROP_PADDING_PT),
        )
        mat = fitz.Matrix(self._crop_dpi / 72, self._crop_dpi / 72)
        try:
            pix = page.get_pixmap(matrix=mat, clip=clip)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("dual-pass: crop render failed (%s)", exc)
            return None
        fd, name = tempfile.mkstemp(prefix="socr_tablecrop_", suffix=".png")
        path = Path(name)
        try:
            import os

            os.close(fd)
            img.save(path)
        except Exception:  # pragma: no cover - defensive
            path.unlink(missing_ok=True)
            return None
        return path


def _clean_markdown(text: str) -> str:
    """Strip code fences / stray prose, keep the markdown table lines.

    The prompt asks for a bare table, but small models sometimes wrap it in a
    ```` ```markdown ```` fence or add a lead-in line. Keep the contiguous run of
    pipe-bearing lines.
    """
    lines = text.strip().splitlines()
    cleaned = [ln for ln in lines if not ln.strip().startswith("```")]
    table_lines = [ln for ln in cleaned if "|" in ln]
    return "\n".join(table_lines).strip() if table_lines else ""
