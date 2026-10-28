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


def gemini_quota_failure_class(text: str) -> str:
    """Classify provider quota errors: ``exhausted``, ``transient``, or empty."""
    blob = (text or "").strip()
    if not blob:
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
            reason = str(getattr(attempt, "skip_reason", "") or "")
            err = str(getattr(attempt, "error", "") or "")
            blob = f"{reason} {err}"
            if reason.startswith("paid_disabled:"):
                stats["gemini_skipped_quota_circuit"] += 1
                continue
            if getattr(attempt, "audit_passed", False):
                stats["gemini_pages_accepted"] += 1
                continue
            raw_fm = getattr(attempt, "failure_mode", None)
            fm = getattr(raw_fm, "value", raw_fm) if raw_fm is not None else ""
            fm = str(fm or "")
            if fm == "cli_error" and gemini_quota_failure_class(blob):
                stats["gemini_failures_quota"] += 1
            elif fm in ("empty_response", "empty_output") or "empty/error" in reason.lower():
                stats["gemini_failures_empty"] += 1
            elif fm == "cli_error" or fm == "model_unavailable":
                stats["gemini_failures_cli_error"] += 1
            elif (attempt.text or "").strip():
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
        ("gemini_failures_empty", "empty"),
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
