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
import concurrent.futures
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from socr.tables.locate import TableBox

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "table_extract.md"

# Crops cover a small fraction of a page, so a moderate render DPI keeps small
# table digits/parens crisp while staying fast. 400 dpi was measured to push a
# single dense-table crop read on qwen3-vl:30b-a3b-instruct past the per-crop
# wall-clock deadline (GH-56), which forced crop timeouts and disabled the
# Tier-2 crop-repair fallback. 250 dpi is ~3x faster per read (a full page reads
# in ~120 s at 200 dpi) and remains legible for dense numerals, so crop reads fit
# the deadline and the fallback can actually fire. Rendering setting, not a model
# threshold.
DEFAULT_CROP_DPI = 250
# Padding (PDF points) around a located bbox so a rule or edge digit is never
# clipped by an off-by-a-pixel boundary.
_CROP_PADDING_PT = 6.0

# Wall-clock deadline (seconds) applied per-crop in addition to the httpx I/O
# timeout. Crops are small (fraction of a page), so a crop reread at
# qwen3-vl:30b-a3b-instruct should complete well under 120 s. The ThreadPoolExecutor
# guard (mirroring agentic.py route_page) guarantees the call cannot hold the
# pipeline hostage even when the httpx read-timeout does not fire (e.g. a wedged
# Ollama socket that never closes the response stream).
#
# Residual: closing the httpx client does NOT abort Ollama server-side generation
# (stream:false). The GPU continues until the model finishes, so strict-local mode
# stays serial even after a crop is abandoned. Ollama-side cancellation is out of
# scope; callers that need hard GPU preemption must restart the Ollama process.
#
# Basis: the httpx OllamaTableReader.timeout default is 120 s; we give the
# ThreadPoolExecutor a headroom multiplier of 2.0 so the I/O timeout can fire
# first in normal wedge scenarios and dense multi-table pages (GH-56) get modest
# extra room before the wall-clock guard trips. Callers may pass a different
# deadline.
_CROP_WALL_CLOCK_MULTIPLIER = 2.0
# Minimum deadline, regardless of multiplier outcome, to avoid rounding to 0.
_CROP_DEADLINE_FLOOR_S = 30.0


def crop_wall_clock_deadline(reader_timeout_s: float) -> float:
    """Return the per-crop ThreadPoolExecutor wall-clock deadline in seconds.

    Derived from the reader's httpx timeout: give the OS I/O timeout a chance to
    fire first (multiplier > 1), but never less than the floor. The result is not
    a magic constant — it tracks the configured reader timeout so both guards
    scale together when models or hardware change.
    """
    return max(_CROP_DEADLINE_FLOOR_S, reader_timeout_s * _CROP_WALL_CLOCK_MULTIPLIER)


def probe_ollama_idle(host: str = "http://localhost:11434", timeout: float = 5.0) -> bool:
    """Return True if the Ollama HTTP server is responding (idle/ready).

    Used as a cascade guard after a crop timeout: if the backend is unreachable,
    the pipeline should not fire additional VLM calls into the wedged GPU.

    This is a lightweight /api/tags ping — it does NOT check whether a generation
    is still running server-side (Ollama does not expose that). It only tells us
    whether the HTTP layer is healthy.
    """
    try:
        resp = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        resp.raise_for_status()
        return True
    except (httpx.HTTPError, OSError):
        return False


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


class VllmTableReader:
    """Crop reader for an OpenAI-compatible server (vLLM / SGLang).

    Used on HPC, where Ollama/llama.cpp are forbidden on server GPUs and the
    VLM is served by vLLM. Talks ``/v1/chat/completions`` with the crop image as
    a base64 data URL (OpenAI multimodal message). Mirrors ``OllamaTableReader``
    so it drops into the same ``TableReader`` slot; exposes ``.timeout`` so the
    crop wall-clock deadline scales the same way.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        timeout: float = 120.0,
        api_key: str = "EMPTY",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.host = base_url  # for cascade-probe parity with OllamaTableReader
        self._api_key = api_key
        self._prompt = load_table_prompt()

    def read(self, image_path: Path) -> str:
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self.model,
                "temperature": 0,  # transcription must be deterministic
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                            },
                        ],
                    }
                ],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        choices = resp.json().get("choices") or [{}]
        return _clean_markdown(choices[0].get("message", {}).get("content", ""))


def make_table_reader(
    *,
    backend: str,
    model: str,
    timeout: float = 120.0,
    ollama_host: str = "http://localhost:11434",
    vllm_url: str = "http://localhost:8000/v1",
) -> TableReader:
    """Build the crop reader for *backend* ("vllm"/"sglang" -> OpenAI server, else Ollama).

    Keeps Ollama as the default local path; selects the vLLM/OpenAI reader for
    server/HPC backends. Single place that maps backend -> reader so both crop
    construction sites stay consistent.
    """
    if backend in ("vllm", "sglang", "openai", "api"):
        return VllmTableReader(model=model, base_url=vllm_url, timeout=timeout)
    return OllamaTableReader(model=model, host=ollama_host, timeout=timeout)


class TableCropExtractor:
    """Render table crops and read them with an injected ``TableReader``."""

    def __init__(self, reader: TableReader, crop_dpi: int = DEFAULT_CROP_DPI) -> None:
        self._reader = reader
        self._crop_dpi = crop_dpi

    def extract(
        self,
        pdf_path: Path,
        page_num: int,
        boxes: list[TableBox],
        *,
        deadline: float | None = None,
        cascade_probe: bool = True,
    ) -> list[CropTable]:
        """Crop each box on ``page_num`` (1-indexed) and read it. Never raises.

        Each VLM call is wrapped in a ThreadPoolExecutor wall-clock guard (mirroring
        agentic.py ``route_page``). On ``TimeoutError`` the future is abandoned and a
        ``CropTimeout`` sentinel is injected into the output list so callers can
        record an audit event and apply the cascade guard without re-running
        detection. ``deadline`` defaults to ``crop_wall_clock_deadline(reader.timeout)``
        when the reader exposes a ``timeout`` attribute; otherwise 180 s.

        cascade_probe — when True (default), after any crop timeout this method
        pings the Ollama backend; if it is unreachable, all remaining crops are
        skipped and ``_backend_degraded`` is set on self. A future PP-2 document-
        level halt can test this attribute to abort the whole document.

        A failed crop/read (non-timeout) drops that table (returns no entry for it)
        rather than aborting the page — the reconciler then sees a count mismatch
        and flags rather than patches, which is the safe outcome.
        """
        import fitz

        # Resolve the per-crop wall-clock deadline.
        if deadline is None:
            reader_timeout = getattr(self._reader, "timeout", None)
            deadline = (
                crop_wall_clock_deadline(reader_timeout) if reader_timeout is not None else 180.0
            )

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
                if getattr(self, "_backend_degraded", False):
                    # Cascade guard: a prior timeout left the GPU in an unknown
                    # state — don't fire more VLM calls into the wedged backend.
                    logger.warning(
                        "dual-pass: backend degraded; skipping remaining crops on p%d",
                        page_num,
                    )
                    break
                img_path = self._render_crop(page, box, page_rect)
                if img_path is None:
                    continue
                try:
                    md = self._read_with_deadline(img_path, deadline, page_num)
                except _CropTimeoutError:
                    # Timeout: emit a sentinel so the orchestrator can log an audit
                    # event and apply the cascade guard.
                    out.append(
                        CropTable(
                            markdown="",
                            source=box.source,
                            bbox=box.bbox,
                        )
                    )
                    out[-1]._timed_out = True  # type: ignore[attr-defined]
                    # Mark backend degraded UNCONDITIONALLY after any crop timeout.
                    # /api/tags may answer while /api/generate is still running on
                    # the GPU, so using the probe as the degradation condition is
                    # unsound. We degrade first, then probe only to enrich the log.
                    self._backend_degraded = True
                    if cascade_probe:
                        host = getattr(self._reader, "host", "http://localhost:11434")
                        idle = probe_ollama_idle(host)
                        logger.warning(
                            "dual-pass: crop timeout on p%d — backend marked degraded "
                            "(probe idle=%s)",
                            page_num,
                            idle,
                        )
                    img_path.unlink(missing_ok=True)
                    continue
                except Exception as exc:
                    logger.warning("dual-pass: crop read failed p%d (%s)", page_num, exc)
                    img_path.unlink(missing_ok=True)
                    continue
                img_path.unlink(missing_ok=True)
                if md.strip():
                    out.append(CropTable(markdown=md, source=box.source, bbox=box.bbox))
        finally:
            doc.close()
        return out

    def _read_with_deadline(self, img_path: Path, deadline: float, page_num: int) -> str:
        """Submit ``reader.read`` to a single-worker executor; raise ``_CropTimeoutError``
        if the wall-clock deadline expires.

        Mirrors the ``ThreadPoolExecutor`` pattern in ``agentic.py route_page``
        (~lines 213-247): abandon the future with ``wait=False`` so the pipeline
        is not blocked by a stalled thread. The daemon thread is cleaned up when
        it eventually unblocks or the process exits.
        """
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = ex.submit(self._reader.read, img_path)
        try:
            return future.result(timeout=deadline)
        except concurrent.futures.TimeoutError:
            future.cancel()
            logger.warning(
                "dual-pass: crop VLM call timed out after %.1f s on p%d — releasing",
                deadline,
                page_num,
            )
            raise _CropTimeoutError(deadline, page_num)
        finally:
            # wait=False: release the executor without blocking. On the success
            # path this reclaims the idle worker immediately; on timeout the
            # stalled thread (blocked on httpx response) is NOT reaped — it keeps
            # running until it unblocks. ThreadPoolExecutor workers are NOT daemon
            # threads; they keep the process alive if it tries to exit while a
            # worker is blocked. In practice the orchestrator is long-lived and the
            # thread unblocks once Ollama finishes generating (or the process exits
            # naturally). This is the accepted trade-off in the agentic.py pattern.
            ex.shutdown(wait=False)

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


class _CropTimeoutError(Exception):
    """Internal sentinel: a single crop VLM call exceeded its wall-clock deadline."""

    def __init__(self, deadline: float, page_num: int) -> None:
        super().__init__(f"crop reread timed out after {deadline:.1f}s on p{page_num}")
        self.deadline = deadline
        self.page_num = page_num


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
