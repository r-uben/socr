"""#1185: prove the text-layer check RAN on a page and CHECKED it, not that it was skipped.

A test whose control page is meant to be CHECKED must show the comparison happened with at
least one number compared: a clause that never fires is indistinguishable from a page the
check never looked at. The spy records every comparison and every clause decision and
returns the real results unchanged, so it observes the check and never alters it.

Round 2: agreement alone is no longer enough. Agreement with a layer the page analysis
refused is UNKNOWN (Argon, PR #1219 review), so a control on such a layer would still
"agree" while its page is no longer CHECKED; the clause decision is recorded and asserted.
"""

from __future__ import annotations

import pytest


def spy_agreement(monkeypatch: pytest.MonkeyPatch) -> list:
    """Record ``("compare", reading, record)`` for every call of the comparator and
    ``("clause", text, clause)`` for every clause decision, behaviour unchanged."""
    import socr.core.verdict as verdict
    import socr.tables.reading_agreement as reading_agreement

    calls: list = []
    real_compare = reading_agreement.compare_numbers
    real_clause = verdict.text_layer_numbers_clause

    def recording(reference, reading, *, provenance):
        record = real_compare(reference, reading, provenance=provenance)
        calls.append(("compare", reading, record))
        return record

    def deciding(output, p):
        clause = real_clause(output, p)
        calls.append(("clause", output.text or "", clause))
        return clause

    monkeypatch.setattr(reading_agreement, "compare_numbers", recording)
    monkeypatch.setattr(verdict, "text_layer_numbers_clause", deciding)
    return calls


def assert_agreed_on(calls: list, snippet: str) -> None:
    """The check ran on the reading containing ``snippet``, compared numbers, agreed, and
    raised no clause: the page is CHECKED by this check."""
    hits = [record for kind, reading, record in calls if kind == "compare" and snippet in reading]
    assert hits, f"the text-layer check never ran on the reading containing {snippet!r}"
    for record in hits:
        assert record.compared > 0, f"vacuous agreement on {snippet!r}: nothing was compared"
        assert record.agreed, f"the control reading disagrees with its layer: {record.disputed}"
    clauses = [c for kind, text, c in calls if kind == "clause" and snippet in text]
    assert clauses, f"no clause decision was recorded for {snippet!r}"
    assert all(c is None for c in clauses), f"the control is not CHECKED: {clauses}"
