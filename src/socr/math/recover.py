"""Recover font-corrupted equation regions as LaTeX via a local vision model.

Pipeline for a born-digital page whose math is font-map corrupted:

  1. ``corrupt_math_line_rects(page)`` — find text lines containing math mojibake
     and merge vertically-adjacent ones into equation-region rectangles.
  2. ``latex_for_image(png, ...)`` — render a region and read it back as LaTeX
     with a local Ollama vision model (qwen3-vl by default; benchmarked faithful
     on this corruption class where olmocr2 hallucinated).
  3. ``splice_math(page, native_text, regions)`` — replace the corrupted lines in
     the native text with the recovered LaTeX, leaving prose byte-for-byte intact.

Only the OCR call touches the network (localhost Ollama); everything else is pure
geometry/text and is unit-tested without a model. ``urllib`` only — no new deps.
"""

from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request

from socr.core.born_digital import line_has_corrupt_math

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen3-vl:8b"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_DPI = 300

# Vertical gap (in points) below which two corrupt-math lines belong to the same
# equation region. A single display equation may wrap; consecutive numbered
# equations sit within ~a line height of each other.
_REGION_MERGE_GAP_PT = 6.0
# Horizontal padding added to a region crop so sub/superscripts and wide
# delimiters are not clipped at the edge.
_REGION_PAD_PT = 4.0

_PROMPT = (
    "Transcribe the mathematics in this image as LaTeX. "
    "Output ONLY the equations, one per line, with no prose, no markdown code "
    "fences, and no surrounding $ delimiters. Be faithful to every symbol: "
    "= + - parentheses, fractions, subscripts, superscripts, and Greek letters."
)


def corrupt_math_line_rects(page) -> list[object]:
    """Return merged ``fitz.Rect`` regions covering corrupt-math text lines.

    Lines are taken from ``get_text("dict")``; a line qualifies if its text trips
    ``line_has_corrupt_math``. Vertically-adjacent qualifying lines are merged so
    a multi-line display equation renders as one crop. Never raises.
    """
    import fitz

    try:
        data = page.get_text("dict")
    except Exception:  # pragma: no cover - defensive
        return []

    bad: list[object] = []
    for block in data.get("blocks", []):
        if block.get("type", 0) == 1:  # image block
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            if text.strip() and line_has_corrupt_math(text):
                bad.append(fitz.Rect(line["bbox"]))

    if not bad:
        return []

    bad.sort(key=lambda r: r.y0)
    merged: list[object] = [fitz.Rect(bad[0])]
    for rect in bad[1:]:
        last = merged[-1]
        if rect.y0 - last.y1 <= _REGION_MERGE_GAP_PT:
            last.include_rect(rect)
        else:
            merged.append(fitz.Rect(rect))
    return merged


def clean_latex(raw: str) -> str:
    """Strip code fences, stray ``$`` delimiters, and prose lead-ins from a model
    response, keeping the LaTeX lines."""
    text = raw.strip()
    if text.startswith("```"):
        # drop opening fence (optionally ```latex) and trailing fence
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("$$") and s.endswith("$$") and len(s) > 4:
            s = s[2:-2].strip()
        elif s.startswith("$") and s.endswith("$") and len(s) > 2:
            s = s[1:-1].strip()
        out.append(s)
    return "\n".join(out).strip()


def latex_for_image(
    png_bytes: bytes,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout: float = 240.0,
) -> str:
    """OCR a rendered equation crop to LaTeX via the local Ollama API.

    Returns cleaned LaTeX, or "" on any transport/parse failure (caller keeps the
    crop image as the faithful fallback). Never raises.
    """
    payload = json.dumps(
        {
            "model": model,
            "prompt": _PROMPT,
            "images": [base64.b64encode(png_bytes).decode()],
            "stream": False,
            # Cap the context window: qwen3-vl otherwise loads its full 262k
            # context (~48 GB, slow first token). A single equation crop needs
            # only a few k tokens, so a small num_ctx is dramatically faster with
            # no quality loss.
            "options": {"temperature": 0, "num_ctx": 8192},
        }
    ).encode()
    req = urllib.request.Request(
        f"{host}/api/generate", data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        logger.warning("math OCR call failed: %s", exc)
        return ""
    return clean_latex(body.get("response", ""))


def recover_math_regions(
    page,
    ocr=latex_for_image,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    dpi: int = DEFAULT_DPI,
) -> list[tuple[object, str]]:
    """Render each corrupt-math region and read it back as LaTeX.

    Returns ``(rect, latex)`` pairs (LaTeX may be "" if the model failed — the
    caller decides whether to keep the crop). ``ocr`` is injectable for testing.
    Never raises on the geometry/render path.
    """
    import fitz

    regions = corrupt_math_line_rects(page)
    if not regions:
        return []

    scale = dpi / 72.0
    out: list[tuple[object, str]] = []
    for rect in regions:
        crop = fitz.Rect(rect) + (-_REGION_PAD_PT, -_REGION_PAD_PT, _REGION_PAD_PT, _REGION_PAD_PT)
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=crop)
            png = pix.tobytes("png")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("failed to render math region %s: %s", rect, exc)
            continue
        latex = ocr(png, model=model, host=host)
        out.append((fitz.Rect(rect), latex))
    return out


def splice_math(page, native_text: str, regions: list[tuple[object, str]]) -> str:
    """Replace corrupt-math lines in ``native_text`` with recovered LaTeX.

    Walks the page's dict lines in reading order; a line covered by a recovered
    region is emitted as its LaTeX (once per region, as a ``$$``-delimited block);
    every other line is emitted as its native text, byte-for-byte. Lines whose
    region produced empty LaTeX are dropped in favour of a crop reference note so
    corrupted glyphs never reach the corpus.
    """
    import fitz

    if not regions:
        return native_text
    try:
        data = page.get_text("dict")
    except Exception:  # pragma: no cover - defensive
        return native_text

    emitted: set[int] = set()
    parts: list[str] = []
    for block in data.get("blocks", []):
        if block.get("type", 0) == 1:
            continue
        for line in block.get("lines", []):
            lrect = fitz.Rect(line["bbox"])
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            covering = next(
                (i for i, (r, _) in enumerate(regions) if fitz.Rect(r).intersects(lrect)),
                None,
            )
            if covering is None:
                if text.strip():
                    parts.append(text.strip())
                continue
            if covering in emitted:
                continue
            emitted.add(covering)
            latex = regions[covering][1].strip()
            if latex:
                parts.append(f"$$\n{latex}\n$$")
            else:
                parts.append("*(equation — see page image; OCR unavailable)*")
    return "\n".join(parts).strip()
