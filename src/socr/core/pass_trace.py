"""#1183, #1222, #1249 (C2 part 1): the per-page pass trace.

A verdict lists the checks that did NOT pass. A page whose list is empty therefore says
nothing about which checks ran: on run ``fed_rerun_51270eb5_words`` all 8,017 native success
pages carried an empty verdict, and a reader could not tell "every check passed" from "no
check ran". The trace is the other half of the record: one entry per check outcome, whatever
the outcome, so the absence of an entry can never be read as a pass.

Three rules, all about not inventing assurance:

* **Each check reports its own outcome.** ``PASS`` means the check ran on this page and found
  nothing. ``NOT_APPLICABLE`` means it does not apply here, and says why. ``NOT_RUN`` means it
  could have applied and did not run (no provider, a failed scan, an earlier-OCR layer, a
  resumed page that recorded no trace). A trace is never derived from "no clause" (see
  ``core.verdict.CheckOutcome``, which every check function returns).
* **Scope is kept.** An entry names the region it is about exactly as a clause does
  (``page``, ``table:2``, ``figure:1``). A table-scope pass says nothing about the rest of the
  page (#1181), so a table pass never reads as a page pass.
* **A PASS names the text it was computed on.** A page-scope PASS carries ``text_sha`` (the
  digest of the shipped page text) and a ``row_binding`` ``table:N`` PASS carries ``table_sha``
  (the digest of that shipped table block). Wherever the trace is written, a PASS whose digest
  is not the digest of the text being shipped now, or that carries none, is NOT_RUN: a
  recorded negative is never kept for bytes it did not see (``core.verdict.verify_pass_digests``).
* **It is a record, not an input.** The trace never changes a status, a failure mode,
  ``audit_passed`` or one byte of the ``.md``, and it is not part of the run fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from socr.core.marks import SCOPE_PAGE


class TraceResult(str, Enum):
    """One check's outcome on one scope of one page."""

    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_RUN = "NOT_RUN"


@dataclass(frozen=True)
class TraceEntry:
    """``{check, result, scope, reason}``. ``check`` is a string and not a ``Check`` member
    because a check can be traced without being a verdict check (``row_binding`` is a table
    check whose failures surface as table marks, not as ``Check`` clauses)."""

    check: str
    result: TraceResult
    scope: str = SCOPE_PAGE
    reason: str = ""
    #: sha256 of the shipped page text a PASS was computed on (every PASS but a table's own).
    text_sha: str = ""
    #: sha256 of the shipped table block a ``row_binding`` ``table:N`` PASS was computed on.
    table_sha: str = ""

    def to_dict(self) -> dict[str, str]:
        d = {
            "check": self.check,
            "result": self.result.value,
            "scope": self.scope,
            "reason": self.reason,
        }
        # Written only when set, so an entry that is not a PASS keeps the four-key shape.
        if self.text_sha:
            d["text_sha"] = self.text_sha
        if self.table_sha:
            d["table_sha"] = self.table_sha
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TraceEntry:
        # Strict: an entry is read only when every field is the string the writer wrote. A
        # missing scope is not "page" (a table pass would then read as a page pass), and a
        # coerced non-string would report assurance for a check that never existed.
        check, scope, reason = d["check"], d["scope"], d["reason"]
        if not all(isinstance(v, str) for v in (check, scope, reason)) or not check or not scope:
            raise TypeError("a trace entry's check, scope and reason must be strings")
        digests = (d.get("text_sha", ""), d.get("table_sha", ""))
        if not all(isinstance(v, str) for v in digests):
            raise TypeError("a trace entry's digests must be strings")
        return cls(
            check=check,
            result=TraceResult(d["result"]),
            scope=scope,
            reason=reason,
            text_sha=digests[0],
            table_sha=digests[1],
        )


def trace_to_dicts(entries: tuple[TraceEntry, ...] | list[TraceEntry]) -> list[dict[str, str]]:
    return [e.to_dict() for e in entries]


def trace_from_dicts(raw: Any) -> tuple[TraceEntry, ...] | None:
    """A stored trace, or ``None`` when *raw* is not one (absent key, wrong type, or any
    unreadable entry: one malformed entry makes the whole trace unreadable, so a valid PASS beside
    it is never reported). ``None`` and ``()`` differ: ``()`` is a recorded empty trace."""
    if not isinstance(raw, list):
        return None
    try:
        return tuple(TraceEntry.from_dict(d) for d in raw)
    except (KeyError, ValueError, TypeError):
        return None
