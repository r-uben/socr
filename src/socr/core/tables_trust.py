"""GH-95: consumer-visible table distrust.

Dual-pass and the native table verifier already *detect* untrusted tables and
record them in ``audit_log.json``. Flag-only-by-default is a defensible *write*
policy (a sloppy auto-edit on a citation corpus is worse than a flag), but it is
a bad *read* policy when the durable artifacts hide the flags: on the reference
run the assembled markdown looked like clean prose and tables while 19 pages
carried table flags, and ``metadata.json`` named exactly one page.

This module derives a compact ``tables_trust.json`` from the same events, so a
downstream consumer can gate on table trust by reading one small file instead of
parsing the full audit log.

Deliberately **read-only over the audit events**: it changes no OCR output and no
write policy. Enabling ``--auto-patch-tables`` remains a separate, opt-in
decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from socr.tables.reconcile import PATCH_ELIGIBLE_NOTE

# Audit kinds that mean "the digits in this page's table(s) may be wrong".
#
# Included are detections that were *surfaced but not resolved* — the page still
# ships a table a consumer might read as authoritative.
#
# Deliberately excluded:
#   - ``native_table_verifier_exact_pass``  — a pass, not a distrust signal.
#   - ``dualpass_patched`` / ``table_header_repair`` — the disagreement was
#     resolved, so the shipped table is the corrected one.
#   - ``page_failed`` — not table-specific, and already surfaced at document
#     level via ``LOST_CONTENT_NOTE``. Table pages that fail closed carry
#     ``table_region_unverifiable`` as well, so they are still captured here.
# GH-96: an accepted escalation replaced the page's table with one that measured
# strictly better, so every distrust event recorded against the OLD text refers to
# markdown that no longer exists. Without this the sidecar keeps steering consumers
# away from a page the pipeline just fixed — on the reference run, a page taken from
# 39.1% to 100.0% still listed as untrusted.
#
# Same rule as excluding ``dualpass_patched``: a resolved disagreement is not
# distrust. It only arrives later in the pipeline.
RESOLVING_KINDS: frozenset[str] = frozenset({"table_escalation_accepted"})

TABLE_DISTRUST_KINDS: frozenset[str] = frozenset(
    {
        "dualpass_flagged",
        "native_table_verifier_warn",
        "native_table_verifier_hard_fail",
        "value_guard_row_count_warning",
        "table_region_unverifiable",
        "source_evidence_table_reject",
        "table_row_repetition_truncated",
        # GH-96: an escalation that was refused or timed out leaves the SUSPECT
        # table shipping, so the page stays untrusted. An ACCEPTED escalation is
        # deliberately absent, by the same rule that excludes dualpass_patched:
        # the disagreement was resolved and the shipped table is the better one.
        "table_escalation_refused",
        "table_escalation_timeout",
        # #123 TICKET-C1: two more shapes of the same no-silent-content-loss rule.
        # "table_unexplained_lanes" — the native layer supports a column the
        # emitted table has no home for; a threshold-free fact once B2's lane
        # alignment exists (zero versus non-zero, not a tuned gap).
        # "table_not_scorable" — B1's grid gate (no usable ground truth) used to
        # make a page silently disappear from measurement; routed through the
        # same mechanism rather than a second one, per the B1 review finding.
        "table_unexplained_lanes",
        "table_not_scorable",
        # GH-151 TICKET-B1 review: a fourth shape of the same rule. A2/A2b's
        # structural gate demotes a page to WARNING/audit_passed=False when the
        # native grid genuinely splits a row's label from its values (ragged
        # widths or a detached-label pair) -- without this, a consumer reading
        # tables_trust.json would still mark that page trusted while the audit
        # log and the shipped page status both say its table structure failed.
        # No resolving kind is added alongside it (unlike
        # ``table_escalation_accepted``): this event fires at analyze time from
        # the native attempt alone, before OCR routing is decided, and no
        # MEASURED comparison exists at the point a later OCR read ships
        # instead -- so, in agentic mode, a page whose native attempt tripped
        # this gate but whose final shipped output is a passing OCR table
        # still surfaces here. That is over-flagging relative to the shipped
        # content, not under-flagging; consistent with this module's own
        # stated bias (a flag on a fine page costs a look, a missing flag on a
        # wrong one costs a wrong number in the corpus).
        "table_structure_failed",
    }
)


@dataclass
class PageTrust:
    """Trust record for one page's tables."""

    page_num: int
    reasons: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)
    patch_eligible: bool = False

    def to_dict(self) -> dict:
        return {
            "flagged": True,
            "reasons": sorted(set(self.reasons)),
            "patch_eligible": self.patch_eligible,
            "details": self.details,
        }


@dataclass
class TablesTrust:
    """Document-level table-trust index, derived from audit events."""

    pdf_filename: str
    pages: dict[int, PageTrust] = field(default_factory=dict)
    # Pages that WERE flagged but whose table was replaced by a better-measuring
    # read. Reported so the resolution is visible rather than merely absent.
    resolved_pages: list[int] = field(default_factory=list)

    @property
    def untrusted_pages(self) -> list[int]:
        return sorted(self.pages)

    @property
    def flag_count(self) -> int:
        """Total distrust events, not pages — a page can carry several."""
        return sum(len(p.reasons) for p in self.pages.values())

    def counts_by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for page in self.pages.values():
            for reason in page.reasons:
                out[reason] = out.get(reason, 0) + 1
        return out

    def to_dict(self) -> dict:
        return {
            "pdf_filename": self.pdf_filename,
            # Top-level summary first: a consumer can gate on these three keys
            # without walking ``pages``.
            "untrusted_page_count": len(self.pages),
            "untrusted_pages": self.untrusted_pages,
            "table_flags_n": self.flag_count,
            "counts_by_kind": self.counts_by_kind(),
            "resolved_by_escalation": self.resolved_pages,
            "patch_eligible_pages": sorted(
                num for num, page in self.pages.items() if page.patch_eligible
            ),
            "pages": {str(num): self.pages[num].to_dict() for num in self.untrusted_pages},
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def summary_line(self) -> str:
        """One-line console summary, or '' when every table is trusted."""
        if not self.pages:
            return ""
        eligible = sum(1 for p in self.pages.values() if p.patch_eligible)
        line = (
            f"{len(self.pages)} page(s) with untrusted tables "
            f"({self.flag_count} flag(s)): {', '.join(str(n) for n in self.untrusted_pages)}"
        )
        if eligible:
            line += f"; {eligible} patch-eligible (see --auto-patch-tables)"
        return line


def build_tables_trust(pdf_filename: str, events: list) -> TablesTrust:
    """Derive the trust index from a run's audit events.

    Only ``TABLE_DISTRUST_KINDS`` contribute. Pages with no distrust event do not
    appear at all, so a clean document yields an empty index and prose-only pages
    stay unmarked.
    """
    trust = TablesTrust(pdf_filename=pdf_filename)

    # Pages whose table was superseded by a measurably better read. Collected first
    # because the resolving event is emitted AFTER the distrust events it resolves.
    resolved = {
        getattr(e, "page_num", 0) for e in events if getattr(e, "kind", "") in RESOLVING_KINDS
    }
    trust.resolved_pages = sorted(resolved)

    for event in events:
        if getattr(event, "page_num", 0) in resolved:
            continue
        kind = getattr(event, "kind", "")
        if kind not in TABLE_DISTRUST_KINDS:
            continue

        page_num = getattr(event, "page_num", 0)
        page = trust.pages.setdefault(page_num, PageTrust(page_num=page_num))
        page.reasons.append(kind)

        detail = getattr(event, "detail", "") or ""
        if detail:
            page.details.append(f"{kind}: {detail}")

        data = getattr(event, "data", None) or {}
        if data.get("note") == PATCH_ELIGIBLE_NOTE:
            page.patch_eligible = True

    return trust


# ``DocMetadata`` (ocr-output-contract) has a fixed field set with no slot for a
# trust count, so the document-level signal rides in the one free-text field.
TRUST_NOTE_PREFIX = "untrusted tables"


def trust_note(trust: TablesTrust) -> str | None:
    """Document-level one-liner for ``metadata.json``'s ``error`` field.

    ``None`` when nothing is flagged, so a clean run's surface is untouched.
    Deliberately terse and prefixed with a stable token so a downstream CLI can
    gate on it by substring without parsing ``audit_log.json``.

    Reports a *count* rather than the page list: on the reference run 20 of 20
    table pages were flagged, and enumerating them made the field unbounded in
    the document's page count. ``tables_trust.json`` carries the page list and is
    the authoritative record; this is a pointer to it.
    """
    if not trust.pages:
        return None
    return (
        f"{TRUST_NOTE_PREFIX} on {len(trust.pages)} page(s), "
        f"{trust.flag_count} flag(s) (see tables_trust.json)"
    )
