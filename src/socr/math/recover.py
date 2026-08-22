"""Recover font-corrupted equation regions without rewriting surrounding prose.

A born-digital page can have a trustworthy native prose layer but unusable equation
characters because an embedded font has no usable Unicode mapping.  This module:

1. locates complete numbered equation rows when source labels provide reliable
   anchors, otherwise falling back to positively corrupt native-text lines;
2. renders and retains each matching crop as visual ground truth;
3. asks an Ollama-compatible vision model for LaTeX and applies the deterministic
   structural syntax gate; and
4. replaces only an exact native-text slice, preserving the surrounding prose.

Structural validation proves syntax only.  A parseable candidate can still contain a
wrong sign, index, factor, or operator, so every candidate remains visibly marked as
non-authoritative and the crop remains the ground truth.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from socr.core.born_digital import clean_native_text, line_has_corrupt_math
from socr.math.validate_latex import validate_latex_structure

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen3-vl:30b-a3b-instruct"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_DPI = 300

# Geometry constants are named because they define which damaged lines share one
# visual crop.  They are layout units, not empirical quality thresholds.
_REGION_MERGE_GAP_PT = 6.0
_REGION_PAD_PT = 4.0

CORRUPT_MATH_PROMPT = (
    "Transcribe the mathematics in this image as LaTeX. "
    "Output ONLY the equations, one per line, with no prose, no markdown code "
    "fences, and no surrounding $ delimiters. Be faithful to every symbol: "
    "= + - parentheses, fractions, subscripts, superscripts, and Greek letters."
)

_CORRUPT_CANDIDATE_HEADER = (
    "<!-- socr-corrupt-equation: structurally-valid LaTeX candidate "
    "(syntax only, non-authoritative — crop is ground truth) -->"
)
_CORRUPT_UNRESOLVED_PREFIX = "[corrupt equation unresolved"


def _candidate_with_source_label(region: CorruptMathRegion) -> str:
    """Replace only this row's model-authored terminal label with the source label."""
    candidate = region.raw_latex.strip()
    label = region.equation_label.strip()
    if not label:
        return candidate
    tag = label.removeprefix("(").removesuffix(")").strip()
    terminal_label = re.compile(
        rf"(?:\\tag\{{\s*(?:\({re.escape(tag)}\)|{re.escape(tag)})\s*\}}|"
        rf"\\text\{{\s*{re.escape(label)}\s*\}}|"
        rf"(?:\\(?:quad|qquad)\s*)?{re.escape(label)})\s*$"
    )
    candidate = terminal_label.sub("", candidate).rstrip(" ,.;")
    return f"{candidate} \\tag{{{tag}}}"


@dataclass
class CorruptMathRegion:
    """One corrupt region's evidence, candidate, and deterministic disposition."""

    rect: object
    source_text: str
    crop_path: str | None
    raw_latex: str
    validation_ok: bool
    validation_reason: str
    model_id: str
    equation_label: str = ""
    attempts: int = 0
    source_aligned: bool = False

    @property
    def resolved(self) -> bool:
        return bool(
            self.crop_path and self.validation_ok and self.raw_latex.strip() and self.source_aligned
        )


@dataclass
class _CorruptLineGroup:
    rect: object
    lines: list[str]
    crop_rect: object | None = None
    equation_label: str = ""

    @property
    def source_text(self) -> str:
        return "\n".join(self.lines)


def _corrupt_math_line_groups(
    page,
    *,
    covered_groups: list[_CorruptLineGroup] | None = None,
) -> list[_CorruptLineGroup]:
    """Return corrupt-line groups not already owned by aligned numbered rows."""
    import fitz

    try:
        data = page.get_text("dict")
    except Exception:  # pragma: no cover - defensive
        return []
    try:
        native_text = clean_native_text(page.get_text("text"))[0]
    except Exception:  # pragma: no cover - defensive
        native_text = ""

    bad: list[_CorruptLineGroup] = []
    for block in data.get("blocks", []):
        if block.get("type", 0) == 1:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            text, _stripped, _spaces = clean_native_text(text)
            text = text.rstrip()
            if text.strip() and line_has_corrupt_math(text):
                rect = fitz.Rect(line["bbox"])
                already_covered = any(
                    text in group.lines and rect.intersects(group.rect)
                    for group in covered_groups or []
                )
                if not already_covered:
                    bad.append(_CorruptLineGroup(rect, [text]))

    if not bad:
        return []

    bad.sort(key=lambda group: group.rect.y0)
    merged = [_CorruptLineGroup(fitz.Rect(bad[0].rect), list(bad[0].lines))]
    for group in bad[1:]:
        last = merged[-1]
        combined_source = f"{last.source_text}\n{group.source_text}"
        if group.rect.y0 - last.rect.y1 <= _REGION_MERGE_GAP_PT and combined_source in native_text:
            last.rect.include_rect(group.rect)
            last.lines.extend(group.lines)
        else:
            merged.append(_CorruptLineGroup(fitz.Rect(group.rect), list(group.lines)))
    return merged


def _numbered_math_groups(page) -> list[_CorruptLineGroup]:
    """Return complete numbered equation rows with exact native-text slices."""
    import fitz

    from socr.math.detect_equations import detect_display_equations

    detected = detect_display_equations(page, page_num=1)
    return [
        _CorruptLineGroup(
            fitz.Rect(region.source_bbox),
            region.source_text.splitlines(),
            fitz.Rect(region.bbox),
            region.equation_label,
        )
        for region in detected.regions
        if (
            region.has_eq_number
            and region.source_text
            and line_has_corrupt_math(region.source_text)
        )
    ]


def _recovery_groups(page) -> list[_CorruptLineGroup]:
    """Return numbered rows plus corrupt lines outside every numbered row."""
    numbered = _numbered_math_groups(page)
    fallback = _corrupt_math_line_groups(page, covered_groups=numbered)
    groups = [*numbered, *fallback]
    groups.sort(key=lambda group: group.rect.y0)
    return groups


def corrupt_math_line_rects(page) -> list[object]:
    """Return rectangles selected for corrupt-math recovery."""
    return [group.rect for group in _recovery_groups(page)]


def clean_latex(raw: str) -> str:
    """Strip fences and surrounding math delimiters from a model response."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            stripped = stripped[2:-2].strip()
        elif stripped.startswith("$") and stripped.endswith("$") and len(stripped) > 2:
            stripped = stripped[1:-1].strip()
        out.append(stripped)
    return "\n".join(out).strip()


def latex_for_image(
    png_bytes: bytes,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout: float = 300.0,
    keep_alive: str = "30m",
) -> str:
    """Read an equation crop through an Ollama-compatible vision endpoint.

    Returns cleaned LaTeX or an empty string on transport/response failure.  Empty
    output is not success; callers retain the crop and record the failed disposition.
    """
    payload = json.dumps(
        {
            "model": model,
            "prompt": CORRUPT_MATH_PROMPT,
            "images": [base64.b64encode(png_bytes).decode()],
            "stream": False,
            "keep_alive": keep_alive,
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


def _save_crop(
    png: bytes,
    crop_dir: Path | None,
    crop_ref_dir: str,
    page_num: int,
    region_index: int,
) -> str | None:
    if crop_dir is None:
        return None
    filename = f"corrupt_math_p{page_num:05d}_r{region_index + 1:03d}.png"
    try:
        crop_dir.mkdir(parents=True, exist_ok=True)
        (crop_dir / filename).write_bytes(png)
    except OSError as exc:
        logger.warning("failed to retain corrupt-math crop %s: %s", filename, exc)
        return None
    return str(Path(crop_ref_dir) / filename)


def recover_math_regions(
    page,
    ocr=latex_for_image,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    dpi: int = DEFAULT_DPI,
    *,
    crop_dir: Path | None = None,
    crop_ref_dir: str = "equations",
    page_num: int = 1,
    model_disabled_reason: str = "",
) -> list[CorruptMathRegion]:
    """Render, retain, transcribe, and syntax-check every corrupt-math region.

    Render failures remain represented as failed results.  An empty model response
    is retried once and then fails closed.  The returned records never claim that
    syntax validation establishes mathematical fidelity.
    """
    import fitz

    groups = _recovery_groups(page)
    if not groups:
        return []

    scale = dpi / 72.0
    out: list[CorruptMathRegion] = []
    for region_index, group in enumerate(groups):
        crop = (
            fitz.Rect(group.crop_rect)
            if group.crop_rect is not None
            else fitz.Rect(group.rect)
            + (-_REGION_PAD_PT, -_REGION_PAD_PT, _REGION_PAD_PT, _REGION_PAD_PT)
        )
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=crop)
            png = pix.tobytes("png")
        except Exception as exc:  # pragma: no cover - defensive
            reason = f"crop render failed: {exc}"
            logger.warning("failed to render math region %s: %s", group.rect, exc)
            out.append(
                CorruptMathRegion(
                    rect=fitz.Rect(group.rect),
                    source_text=group.source_text,
                    crop_path=None,
                    raw_latex="",
                    validation_ok=False,
                    validation_reason=reason,
                    model_id=model,
                    equation_label=group.equation_label,
                )
            )
            continue

        crop_path = _save_crop(
            png,
            crop_dir,
            crop_ref_dir,
            page_num,
            region_index,
        )
        attempts = 0
        latex = ""
        if model_disabled_reason:
            validation_ok = False
            validation_reason = model_disabled_reason
        else:
            attempts = 1
            latex = ocr(png, model=model, host=host)
            if not latex:
                attempts += 1
                latex = ocr(png, model=model, host=host)
            latex = clean_latex(latex)
            validation_ok, validation_reason = validate_latex_structure(latex)
        if crop_path is None:
            validation_ok = False
            crop_reason = "crop could not be retained as visual ground truth"
            validation_reason = (
                f"{validation_reason}; {crop_reason}" if validation_reason else crop_reason
            )
        out.append(
            CorruptMathRegion(
                rect=fitz.Rect(group.rect),
                source_text=group.source_text,
                crop_path=crop_path,
                raw_latex=latex,
                validation_ok=validation_ok,
                validation_reason=validation_reason,
                model_id=model,
                equation_label=group.equation_label,
                attempts=attempts,
            )
        )
    return out


def _region_replacement(region: CorruptMathRegion) -> str:
    crop = (
        f"![Corrupt equation crop]({region.crop_path})"
        if region.crop_path
        else "<!-- socr-corrupt-equation: crop unavailable -->"
    )
    if region.resolved:
        candidate = _candidate_with_source_label(region)
        return f"{crop}\n{_CORRUPT_CANDIDATE_HEADER}\n$$\n{candidate}\n$$"
    evidence = "; see retained crop" if region.crop_path else "; no crop was retained"
    return f"{crop}\n{_CORRUPT_UNRESOLVED_PREFIX}: {region.validation_reason}{evidence}]"


def splice_math(page, native_text: str, regions: list[CorruptMathRegion]) -> str:
    """Patch exact corrupt substrings in ``native_text`` and preserve all other bytes.

    The function updates each record's ``source_aligned`` field.  An aligned
    region with a retained crop replaces its native slice even when LaTeX is
    unresolved; that replacement contains the crop plus an explicit unresolved
    marker rather than silently leaving corrupt glyphs in place.

    If the PDF extractor's region text cannot be found verbatim in the native
    snapshot, the snapshot is left untouched and the evidence plus an explicit
    unresolved marker is appended.  The function never rebuilds prose from page
    geometry and never silently drops a damaged equation.
    """
    del page  # retained in the public signature for existing callers
    if not regions:
        return native_text

    text = native_text
    unmatched: list[str] = []
    for region in regions:
        source = region.source_text
        source_in_native = bool(source and source in native_text)
        source_available = bool(source and source in text)
        region.source_aligned = source_in_native and source_available
        replacement = _region_replacement(region)
        if region.source_aligned and (region.resolved or region.crop_path):
            text = text.replace(source, replacement, 1)
        else:
            if source_in_native and not source_available:
                retention_reason = "source overlapped an earlier recovery"
                retention_note = "original bytes represented by that earlier recovery"
            elif not region.source_aligned:
                retention_reason = "source could not be aligned"
                retention_note = "native text retained"
            else:
                retention_reason = "crop could not be retained"
                retention_note = "native text retained"
            unmatched.append(
                f"{replacement}\n[corrupt equation {retention_reason}; {retention_note}]"
            )
    if unmatched:
        suffix = "\n\n".join(unmatched)
        text = f"{text}\n\n{suffix}" if text else suffix
    return text
