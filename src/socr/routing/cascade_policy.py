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
REJECT_REASON_SOURCE_EVIDENCE_PREFIX = f"{SOURCE_EVIDENCE_MARKER}:"
PAID_BLOCK_REASON_SOURCE_EVIDENCE = SOURCE_EVIDENCE_MARKER
PAID_BLOCK_REASON_SPARSE_WORD_COUNT = "sparse_word_count"
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
    return "numeric tokens unsupported" in lower or "content labels unsupported" in lower


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


def heuristic_sparse_page_ok(page_assessment, *, audit_min_words: int) -> bool:
    """Born-digital-only sparse hook for the heuristic page judge (issue #39).

    Scanned pages never earn leniency here — a short read of a scan must stay a
    failure. Deliberately narrow: ``has_figures`` alone does NOT qualify.
    """
    if page_assessment is None:
        return False
    if not getattr(page_assessment, "is_born_digital", False):
        return False
    return int(getattr(page_assessment, "word_count", 0) or 0) < audit_min_words


def sparse_paid_escalation_ok(signals: SparsePageSignals, *, audit_min_words: int) -> bool:
    """Whether a word-count reject may block paid Gemini (not heuristic acceptance).

    Born-digital sparse captions and explicit chart-asset pages only. Scanned pages
    do not qualify on raster / invisible-text / figure heuristics alone.
    """
    if signals.has_chart_asset:
        return True
    if signals.is_born_digital and (signals.native_word_count or 0) < audit_min_words:
        return True
    return False


def sparse_signals_from_page_assessment(page_assessment) -> SparsePageSignals:
    """Build :class:`SparsePageSignals` from a live ``PageAssessment`` (or ``None``)."""
    if page_assessment is None:
        return SparsePageSignals()
    word_count = int(getattr(page_assessment, "word_count", 0) or 0)
    is_bd = bool(getattr(page_assessment, "is_born_digital", False))
    has_figures = bool(
        getattr(page_assessment, "has_figures", False)
        or getattr(page_assessment, "has_images", False)
    )
    return SparsePageSignals(
        is_born_digital=is_bd,
        native_word_count=word_count,
        has_figures=has_figures,
    )


def paid_gemini_block_after_reject(
    judge_reason: str,
    signals: SparsePageSignals,
    *,
    audit_min_words: int,
    block_paid_on_source_evidence: bool = True,
    sparse_word_count_exempt: bool = True,
) -> str:
    """Return a block-reason tag when Gemini must not run, else ``""``.

    Shared by the live router (``route_page``) and offline ``cascade_replay``.
    """
    if block_paid_on_source_evidence and blocks_paid_escalation(judge_reason):
        return PAID_BLOCK_REASON_SOURCE_EVIDENCE
    if sparse_word_count_exempt and is_word_count_rejection(judge_reason):
        if sparse_paid_escalation_ok(signals, audit_min_words=audit_min_words):
            return PAID_BLOCK_REASON_SPARSE_WORD_COUNT
    return ""


def _word_count_from_journal(journal: list[dict]) -> int:
    for row in journal:
        reason = str(row.get("reason") or "")
        m = re.search(r"Word count\s+(\d+)", reason, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 0


def sparse_signals_from_manifest_entry(entry: dict, journal: list[dict]) -> SparsePageSignals:
    """Best-effort sparse signals from a recorded manifest page entry.

    Mirrors :func:`sparse_signals_from_page_assessment` when ``entry["routing_meta"]``
    is present (tests / future manifests); otherwise infers word count and chart
    routing from the journal and final engine fingerprint only.
    """
    fp = entry.get("fingerprint") or {}
    meta = entry.get("routing_meta") or {}
    engine = str(fp.get("engine") or "")
    has_chart = engine == "chart_asset" or bool(meta.get("has_chart_asset"))
    word_count = int(meta.get("word_count") or meta.get("native_word_count") or 0)
    if word_count <= 0:
        word_count = _word_count_from_journal(journal)
    is_bd = bool(meta.get("is_born_digital"))
    if not meta and engine == "native" and 0 < word_count < 40:
        is_bd = True
    has_figures = bool(meta.get("has_figures") or meta.get("has_images") or has_chart)
    return SparsePageSignals(
        is_born_digital=is_bd,
        native_word_count=word_count,
        has_figures=has_figures,
        has_chart_asset=has_chart,
    )


_GEMINI_HTTP_STATUS_RE = re.compile(
    r"(?:HTTP\s*(?:Error|/|\s)|status(?:\s*code)?\s*[=:]?\s*|code\s*[=:]?\s*)(\d{3})",
    re.IGNORECASE,
)
_GEMINI_QUOTA_RE = re.compile(
    r"\b429\b|RESOURCE_EXHAUSTED|rate[\s_-]*limit|quota|too many requests",
    re.IGNORECASE,
)
# Stdout may carry page OCR; only match provider-shaped error lines there.
_GEMINI_QUOTA_STDOUT_ANCHORED_RE = re.compile(
    r"(?m)^(?:"
    r"google\.api_core\.exceptions\.ResourceExhausted\b|"
    r"RESOURCE_EXHAUSTED\b|"
    r"Error:\s*(?:429\b|.*\bquota\b)"
    r")",
    re.IGNORECASE,
)
_GEMINI_SAFETY_RE = re.compile(
    r"\bSAFETY\b|finish_reason\s*[=:]\s*SAFETY|harm_category|candidate was blocked",
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
    classification when stderr matches known Gemini/API failure shapes.
    """
    body = sanitize_cli_stderr(stderr, stdout)
    err_text = stderr or ""
    prefix = ""
    if _GEMINI_QUOTA_RE.search(err_text):
        prefix = "quota/rate-limit"
    elif _GEMINI_SAFETY_RE.search(err_text):
        prefix = "safety-block"
    elif _GEMINI_TIMEOUT_RE.search(err_text):
        prefix = "timeout"
    elif _GEMINI_QUOTA_STDOUT_ANCHORED_RE.search(stdout or ""):
        prefix = "quota/rate-limit"
    else:
        m = _GEMINI_HTTP_STATUS_RE.search(err_text)
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
    r"(?:(?:completion|output|candidates)[_\s-]*token(?:s)?)"
    r"[\s:=]+(\d+)",
    re.IGNORECASE,
)


def parse_token_usage(stderr: str, stdout: str = "") -> tuple[int | None, int | None]:
    """Best-effort input/output token counts from CLI logs (Gemini / generic).

    ``total_tokens`` is never treated as output — only explicit completion/output
    token fields count toward ``llm_output_tokens``.
    """
    blob = f"{stderr or ''}\n{stdout or ''}"
    inp = out = None
    m = _TOKEN_USAGE_RE.search(blob)
    if m:
        inp = int(m.group(1))
    m = _OUTPUT_TOKEN_RE.search(blob)
    if m:
        out = int(m.group(1))
    return inp, out
