"""Table verification metadata: typed judge outcomes and withheld-table records."""

from __future__ import annotations

from socr.core.result import (
    JUDGE_OUTCOME_COMPLETED,
    JUDGE_OUTCOME_NATIVE_TABLE_VERIFIER,
    JUDGE_OUTCOME_SOURCE_EVIDENCE,
    JUDGE_OUTCOME_TIMEOUT,
    JUDGE_OUTCOME_VERIFIER_ERROR,
    PageOutput,
    TABLE_VERIFICATION_JUDGE_OUTCOMES,
)

TABLE_VERIFICATION_PROSE_RECOVERY_KIND = "table_verification_prose_recovery"

#: Outcomes that mean the judge never completed a prose-level verdict.
_NON_PROSE_JUDGE_OUTCOMES = frozenset(
    {
        JUDGE_OUTCOME_TIMEOUT,
        JUDGE_OUTCOME_VERIFIER_ERROR,
    }
)


def page_has_table_verification_refusal(p) -> bool:
    """True when any attempt carries a table-verification typed outcome."""
    for att in getattr(p, "attempts", None) or []:
        if not isinstance(att, PageOutput):
            continue
        if att.audit_passed:
            continue
        if getattr(att, "judge_outcome", "") in TABLE_VERIFICATION_JUDGE_OUTCOMES:
            return True
    return False


def page_failed_only_table_verification(p) -> bool:
    """True when every completed refusal was table verification (typed outcomes)."""
    attempts = list(getattr(p, "attempts", None) or [])
    if not attempts:
        return False
    saw_table = False
    for att in attempts:
        if not isinstance(att, PageOutput):
            continue
        if att.audit_passed:
            continue
        outcome = getattr(att, "judge_outcome", "") or ""
        if outcome in _NON_PROSE_JUDGE_OUTCOMES:
            return False
        if outcome in TABLE_VERIFICATION_JUDGE_OUTCOMES:
            saw_table = True
            continue
        if outcome == JUDGE_OUTCOME_COMPLETED and not att.audit_passed:
            return False
        if outcome and outcome not in TABLE_VERIFICATION_JUDGE_OUTCOMES:
            return False
    return saw_table


def withheld_table_manifest_from_page(p) -> list[dict]:
    """Tables actually withheld on this page (detection geometry), not attempt reasons."""
    count = getattr(p, "detected_table_count", 0)
    boxes = list(getattr(p, "detected_table_bboxes", None) or [])
    if not isinstance(count, int) or count <= 0:
        return []
    out: list[dict] = []
    for idx, box in enumerate(boxes[:count]):
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            region = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
        else:
            region = list(box) if box is not None else []
        out.append(
            {
                "table_index": idx,
                "detected_table_count": count,
                "bbox": region,
                "reason": "withheld_unverified_table",
            }
        )
    if len(boxes) != count:
        out.append(
            {
                "table_index": None,
                "detected_table_count": count,
                "bbox_count": len(boxes),
                "reason": "withheld_inventory_incomplete",
            }
        )
    return out


def attach_withheld_table_manifest(p) -> list[dict]:
    records = withheld_table_manifest_from_page(p)
    if not records:
        return []
    try:
        p.withheld_table_manifest = records
    except Exception:  # pragma: no cover - read-only test stand-ins
        pass
    return records


def record_table_prose_recovery_audit(state, page_num: int, p, *, engine: str = "") -> None:
    """Append recovery audit to ``state.events`` and set the withheld-table manifest."""
    records = attach_withheld_table_manifest(p)
    if not records:
        return
    from socr.core.audit_log import AuditEvent

    state.events.append(
        AuditEvent(
            page_num=page_num,
            kind=TABLE_VERIFICATION_PROSE_RECOVERY_KIND,
            engine=engine,
            detail="prose kept; tables withheld at regional markers",
            data={"tables": list(records)},
        )
    )
