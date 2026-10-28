"""Table-verification-only fail-closed floors: keep verified prose, withhold tables."""

from __future__ import annotations

from socr.core.result import FailureMode, PageOutput

#: Prefixes on ``PageOutput.judge_reason`` / attempt reasons that mean ONLY the table
#: chain refused the reading — not a prose or page-level check.
TABLE_VERIFICATION_REASON_PREFIXES: tuple[str, ...] = (
    "native_table_verifier:",
    "source_evidence_table:",
)

#: Audit kind written when a page ships prose recovered under this ruling.
TABLE_VERIFICATION_PROSE_RECOVERY_KIND = "table_verification_prose_recovery"


def _attempt_refusal_reason(att) -> str:
    if isinstance(att, PageOutput):
        return (
            getattr(att, "judge_reason", None) or getattr(att, "skip_reason", None) or ""
        ).strip()
    return (getattr(att, "judge_reason", None) or getattr(att, "reason", None) or "").strip()


def page_has_table_verifier_refusal(p) -> bool:
    """True when any attempt was refused by the native table verifier chain."""
    for att in getattr(p, "attempts", None) or []:
        if isinstance(att, PageOutput) and att.audit_passed:
            continue
        reason = _attempt_refusal_reason(att)
        if any(reason.startswith(prefix) for prefix in TABLE_VERIFICATION_REASON_PREFIXES):
            return True
    return False


def page_has_non_table_judge_refusal(p) -> bool:
    """True when a judge refused for a reason other than table verification."""
    for att in getattr(p, "attempts", None) or []:
        if isinstance(att, PageOutput) and att.audit_passed:
            continue
        reason = _attempt_refusal_reason(att)
        if reason and not any(
            reason.startswith(prefix) for prefix in TABLE_VERIFICATION_REASON_PREFIXES
        ):
            return True
    return False


def maybe_attach_withheld_table_manifest(p) -> None:
    """Record withheld tables when prose ships under a table-only verification failure."""
    if not page_failed_only_table_verification(p):
        return
    records = withheld_table_records_from_attempts(p)
    if not records:
        return
    try:
        p.withheld_table_manifest = records
    except Exception:  # pragma: no cover - read-only test stand-ins
        pass


def page_failed_only_table_verification(p) -> bool:
    """True when every refused attempt was refused for table verification alone.

    Prose-level refusals (page judge, coverage, reading agreement) must not
    trigger prose recovery — the table rule is fail-closed, not a prose rescue.
    """
    attempts = list(getattr(p, "attempts", None) or [])
    if not attempts:
        return False
    saw_table_refusal = False
    for att in attempts:
        if isinstance(att, PageOutput) and att.audit_passed:
            continue
        reason = _attempt_refusal_reason(att)
        combined = reason
        if combined and not any(
            combined.startswith(prefix) for prefix in TABLE_VERIFICATION_REASON_PREFIXES
        ):
            # A non-table refusal with a stated reason blocks recovery.
            if reason:
                return False
            fm = getattr(att, "failure_mode", None)
            if fm not in (
                FailureMode.NONE,
                FailureMode.HALLUCINATION,
                FailureMode.NO_WITNESS_BACKEND,
                None,
            ):
                return False
            continue
        if any(combined.startswith(prefix) for prefix in TABLE_VERIFICATION_REASON_PREFIXES):
            saw_table_refusal = True
    return saw_table_refusal


def withheld_table_records_from_attempts(p) -> list[dict[str, str]]:
    """One record per table-verification refusal on this page's attempts."""
    out: list[dict[str, str]] = []
    for att in getattr(p, "attempts", None) or []:
        reason = _attempt_refusal_reason(att)
        if not any(reason.startswith(pfx) for pfx in TABLE_VERIFICATION_REASON_PREFIXES):
            continue
        engine = getattr(att, "engine", "") or ""
        out.append(
            {
                "engine": engine,
                "reason": reason,
                "table_disputed": getattr(att, "table_disputed", "") or "",
                "table_unverifiable": getattr(att, "table_unverifiable", "") or "",
            }
        )
    return out
