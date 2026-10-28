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
# Load-bearing prefix in ``SourceEvidenceTableJudge`` reject reasons (#658).
_NO_WITNESS_CAUSE = "no_witness_backend"
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
    """True when the next paid rung must not run (fail-closed flag instead).

    Only witness-backed disputes block: the pixels were read and the model's table
    tokens were contradicted. Unwitnessed pages, unparsed HTML, and other
    could-not-check endings may still escalate to paid OCR.
    """
    reason = judge_reason or ""
    if SOURCE_EVIDENCE_MARKER not in reason:
        return False
    if _NO_WITNESS_CAUSE in reason:
        return False
    lower = reason.lower()
    return (
        "numeric tokens unsupported" in lower or "content labels unsupported" in lower
    )


def is_word_count_rejection(judge_reason: str) -> bool:
    return bool(_WORD_COUNT_RE.match((judge_reason or "").strip()))


def classify_escalation_reason(judge_reason: str) -> str:
    """Bucket the last free-rung reject that preceded Gemini (for replay sampling).

    Returns one of: ``table_structure``, ``garbage_chars``, ``source_evidence_table``,
    ``min_words``, or ``other``.
    """
    reason = (judge_reason or "").strip()
    if SOURCE_EVIDENCE_MARKER in reason:
        return "source_evidence_table"
    if is_word_count_rejection(reason):
        return "min_words"
    lower = reason.lower()
    if "table_structure" in lower or "table emission" in lower:
        return "table_structure"
    if (
        "garbage" in lower
        or "gibberish" in lower
        or "hallucin" in lower
        or "encoding" in lower
        or "private-use" in lower
    ):
        return "garbage_chars"
    return "other"


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


_GEMINI_HTTP_STATUS_RE = re.compile(
    r"(?:HTTP\s*(?:Error|/|\s)|status(?:\s*code)?\s*[=:]?\s*|code\s*[=:]?\s*)(\d{3})",
    re.IGNORECASE,
)
_GEMINI_QUOTA_RE = re.compile(
    r"429|RESOURCE_EXHAUSTED|rate[\s_-]*limit|quota|too many requests",
    re.IGNORECASE,
)
_GEMINI_SAFETY_RE = re.compile(
    r"SAFETY|BLOCK(?:ED|ING)?|harm_category|candidate was blocked|finish_reason.*SAFETY",
    re.IGNORECASE,
)
_GEMINI_TIMEOUT_RE = re.compile(r"timed?\s*out|deadline exceeded|DEADLINE_EXCEEDED", re.IGNORECASE)


def extract_cli_actionable_error(
    stderr: str,
    stdout: str = "",
    *,
    returncode: int | None = None,
) -> str:
    """Preserve provider failure semantics (HTTP 429, safety block, timeout) for journals.

    Strips ONNX/thread noise via :func:`sanitize_cli_stderr`, then prefixes a short
    classification when the combined log matches known Gemini/API failure shapes.
    """
    body = sanitize_cli_stderr(stderr, stdout)
    combined = f"{stderr or ''}\n{stdout or ''}"
    prefix = ""
    if _GEMINI_QUOTA_RE.search(combined):
        prefix = "quota/rate-limit"
    elif _GEMINI_SAFETY_RE.search(combined):
        prefix = "safety-block"
    elif _GEMINI_TIMEOUT_RE.search(combined):
        prefix = "timeout"
    else:
        m = _GEMINI_HTTP_STATUS_RE.search(combined)
        if m:
            prefix = f"HTTP {m.group(1)}"
    if returncode not in (None, 0) and not prefix:
        prefix = f"exit {returncode}"
    if prefix and body:
        return f"{prefix}: {body}"
    if prefix:
        return prefix
    return body


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
