"""Gemini paid-OCR quota detection, run stats, and circuit-breaker helpers."""

from __future__ import annotations

import re
from collections import Counter

PAID_DISABLED_QUOTA_EXHAUSTED = "quota_exhausted"
PAID_DISABLED_SKIP_REASON = "paid_disabled: quota_exhausted"

# Spending-cap / billing 429s must not retry; transient rate limits may backoff once.
_GEMINI_SPENDING_CAP_RE = re.compile(
    r"spending\s+cap|monthly\s+spending|billing|budget\s+exhaust|payment\s+required",
    re.IGNORECASE,
)
_GEMINI_TRANSIENT_QUOTA_RE = re.compile(
    r"\b429\b|RESOURCE_EXHAUSTED|rate[\s_-]*limit|too many requests|quota/rate-limit",
    re.IGNORECASE,
)

GEMINI_TRANSIENT_BACKOFF_SEC = 3.0

GEMINI_EMPTY_EXHAUSTED_SKIP = "gemini_empty_exhausted: no further calls"

# HTTP 200 with no parts — gemini-ocr-cli logs this shape (cluster audit, job w/ 15 repeats).
_GEMINI_EMPTY_RESPONSE_RE = re.compile(
    r"empty\s+response.*finish_reason=(?:FinishReason\.)?STOP\b|"
    r"finish_reason=(?:FinishReason\.)?STOP,\s*len\(parts\)=0",
    re.IGNORECASE,
)
_GEMINI_SAFETY_RE = re.compile(
    r"\bSAFETY\b|finish_reason\s*[=:]\s*SAFETY|harm_category|candidate was blocked",
    re.IGNORECASE,
)


def gemini_empty_response_message(text: str) -> bool:
    """True when stderr/error text is Gemini's empty STOP answer (HTTP 200, zero parts)."""
    blob = (text or "").strip()
    return bool(blob and _GEMINI_EMPTY_RESPONSE_RE.search(blob))


def gemini_quota_failure_class(text: str) -> str:
    """Classify provider quota errors: ``exhausted``, ``transient``, or empty."""
    blob = (text or "").strip()
    if not blob:
        return ""
    if gemini_empty_response_message(blob):
        return ""
    lower = blob.lower()
    if _GEMINI_SPENDING_CAP_RE.search(blob):
        return "exhausted"
    if "quota exhausted" in lower and "rate" not in lower:
        return "exhausted"
    if "quota/rate-limit" in lower and any(
        token in lower for token in ("spending", "budget", "billing", "cap")
    ):
        return "exhausted"
    if _GEMINI_TRANSIENT_QUOTA_RE.search(blob):
        return "transient"
    return ""


def gemini_attempt_failure_kind(attempt) -> str:
    """Bucket one Gemini journal attempt for run summaries (not page-level)."""
    reason = str(getattr(attempt, "skip_reason", "") or "")
    err = str(getattr(attempt, "error", "") or "")
    blob = f"{reason} {err}".strip()
    if reason.startswith("paid_disabled:"):
        return "skipped_quota_circuit"
    if getattr(attempt, "audit_passed", False):
        return "accepted"
    raw_fm = getattr(attempt, "failure_mode", None)
    fm = getattr(raw_fm, "value", raw_fm) if raw_fm is not None else ""
    fm = str(fm or "")
    if fm in ("empty_response", "empty_output") or gemini_empty_response_message(blob):
        return "empty_response"
    if _GEMINI_SAFETY_RE.search(blob) or fm in ("recitation", "safety"):
        return "safety"
    if fm == "cli_error" and gemini_quota_failure_class(blob):
        return "quota"
    if fm == "cli_error" or fm == "model_unavailable":
        return "cli_error"
    if (attempt.text or "").strip():
        return "judge_rejected"
    return "other"


def _is_gemini_attempt(output) -> bool:
    engine = str(getattr(output, "engine", "") or "")
    return engine == "gemini" or engine.startswith("gemini")


def summarize_gemini_attempts(state) -> dict[str, int]:
    """Roll up Gemini journal attempts from ``state.pages`` (call-level, not page-level)."""
    stats: Counter = Counter()
    for page in state.pages.values():
        for attempt in page.attempts or []:
            if not _is_gemini_attempt(attempt):
                continue
            stats["gemini_call_attempts"] += 1
            kind = gemini_attempt_failure_kind(attempt)
            if kind == "skipped_quota_circuit":
                stats["gemini_skipped_quota_circuit"] += 1
            elif kind == "accepted":
                stats["gemini_pages_accepted"] += 1
            elif kind == "quota":
                stats["gemini_failures_quota"] += 1
            elif kind == "empty_response":
                stats["gemini_failures_empty_response"] += 1
            elif kind == "safety":
                stats["gemini_failures_safety"] += 1
            elif kind == "cli_error":
                stats["gemini_failures_cli_error"] += 1
            elif kind == "judge_rejected":
                stats["gemini_failures_judge_rejected"] += 1
            else:
                stats["gemini_failures_other"] += 1
    # Deprecated alias: attempts, not accepted pages — kept for older log parsers.
    stats["gemini_pages"] = stats["gemini_call_attempts"]
    return dict(stats)


def format_gemini_run_summary(stats: dict[str, int]) -> str:
    """One-line headline: accepted calls vs total attempts, with failure buckets."""
    if not stats.get("gemini_call_attempts") and not stats.get("gemini_skipped_quota_circuit"):
        return ""
    accepted = stats.get("gemini_pages_accepted", 0)
    attempts = stats.get("gemini_call_attempts", 0)
    headline = f"Gemini: {accepted} accepted / {attempts} calls"
    extras: list[str] = []
    for key, label in (
        ("gemini_failures_quota", "quota"),
        ("gemini_failures_empty_response", "empty_response"),
        ("gemini_failures_safety", "safety"),
        ("gemini_failures_cli_error", "cli_error"),
        ("gemini_failures_judge_rejected", "judge-rejected"),
        ("gemini_skipped_quota_circuit", "skipped-after-cap"),
    ):
        n = stats.get(key, 0)
        if n:
            extras.append(f"{label} {n}")
    if extras:
        return f"{headline} ({', '.join(extras)})"
    return headline
