"""Shared cascade routing policy for live runs and offline manifest replay.

Single source for which rejections must not escalate to paid engines and which
pages are legitimately sparse for ``audit_min_words``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from socr.core.config import EngineType
from socr.core.providers import ProviderProfile

# Default agentic ladder when ``enabled_engines`` is the full enum set.
# Qwen (local) → marker (free layout OCR) → Gemma (Ollama Cloud) → Gemini (paid).
DEFAULT_AGENTIC_ENGINE_ORDER: tuple[EngineType, ...] = (
    EngineType.QWEN,
    EngineType.MARKER,
    EngineType.GEMMA,
    EngineType.GEMINI,
    EngineType.GLM,
    EngineType.DEEPSEEK,
    EngineType.MISTRAL,
    EngineType.NOUGAT,
)

SOURCE_EVIDENCE_MARKER = "source_evidence_table"
_WORD_COUNT_RE = re.compile(r"^Word count\b", re.IGNORECASE)

# ONNX / thread-pool noise that prefixes real Gemini CLI errors in batch logs.
_THREAD_NOISE_LINE_RE = re.compile(
    r"pthread_setaffinity_np|onnxruntime.*thread|OMP:|KMP_|libgomp",
    re.IGNORECASE,
)


def ordered_enabled_engines(enabled: set[EngineType]) -> list[EngineType]:
    """``enabled_engines`` in cascade probe order, then any extras."""
    seen: set[EngineType] = set()
    out: list[EngineType] = []
    for engine in DEFAULT_AGENTIC_ENGINE_ORDER:
        if engine in enabled and engine not in seen:
            out.append(engine)
            seen.add(engine)
    for engine in sorted(enabled, key=lambda e: e.value):
        if engine not in seen and engine not in (EngineType.AUTO,):
            out.append(engine)
            seen.add(engine)
    return out


def is_paid_profile(profile: ProviderProfile) -> bool:
    return profile.cost_per_page_usd > 0.0


def blocks_paid_escalation(judge_reason: str) -> bool:
    """True when the next paid rung must not run (fail-closed flag instead)."""
    reason = judge_reason or ""
    if SOURCE_EVIDENCE_MARKER in reason:
        return True
    return False


def is_word_count_rejection(judge_reason: str) -> bool:
    return bool(_WORD_COUNT_RE.match((judge_reason or "").strip()))


@dataclass(frozen=True)
class SparsePageSignals:
    """Optional page metadata for replay / sparse heuristics."""

    is_born_digital: bool = False
    native_word_count: int = 0
    has_figures: bool = False
    has_chart_asset: bool = False
    image_dominant: bool = False
    prior_ocr_layer: bool = False
    invisible_text_over_raster: bool = False


def sparse_page_ok(signals: SparsePageSignals, *, audit_min_words: int) -> bool:
    """Whether low word count is expected (do not escalate on Word count alone)."""
    if signals.has_chart_asset:
        return True
    if signals.image_dominant or signals.invisible_text_over_raster:
        return True
    if signals.is_born_digital and (signals.native_word_count or 0) < audit_min_words:
        return True
    if signals.has_figures and signals.image_dominant:
        return True
    return False


def sanitize_cli_stderr(stderr: str, stdout: str = "") -> str:
    """Drop known thread-pool noise; return the actionable tail for logs/manifest."""
    lines: list[str] = []
    for chunk in (stderr or "").splitlines():
        if _THREAD_NOISE_LINE_RE.search(chunk):
            continue
        stripped = chunk.strip()
        if stripped:
            lines.append(stripped)
    if not lines and stdout:
        for chunk in stdout.splitlines():
            if _THREAD_NOISE_LINE_RE.search(chunk):
                continue
            stripped = chunk.strip()
            if stripped:
                lines.append(stripped)
    text = " ".join(lines)
    return text[:500] + ("..." if len(text) > 500 else "")


_TOKEN_USAGE_RE = re.compile(
    r"(?:(?:prompt|input)[_\s-]*token(?:s)?|cached[_\s-]*content[_\s-]*token(?:s)?)"
    r"[\s:=]+(\d+)",
    re.IGNORECASE,
)
_OUTPUT_TOKEN_RE = re.compile(
    r"(?:(?:completion|output|candidates)[_\s-]*token(?:s)?|total[_\s-]*token(?:s)?)"
    r"[\s:=]+(\d+)",
    re.IGNORECASE,
)


def parse_token_usage(stderr: str, stdout: str = "") -> tuple[int | None, int | None]:
    """Best-effort input/output token counts from CLI logs (Gemini / generic)."""
    blob = f"{stderr or ''}\n{stdout or ''}"
    inp = out = None
    for pat in (_TOKEN_USAGE_RE,):
        m = pat.search(blob)
        if m:
            inp = int(m.group(1))
            break
    m = _OUTPUT_TOKEN_RE.search(blob)
    if m:
        out = int(m.group(1))
    return inp, out
