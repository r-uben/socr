"""#1185: prove the text-layer agreement check RAN on a page and AGREED, not that it was skipped.

A test whose control page is meant to be CHECKED must show the comparison happened with at
least one number compared: a clause that never fires is indistinguishable from a page the
check never looked at. The spy records every comparison and returns the real result
unchanged, so it observes the check and never alters it.
"""

from __future__ import annotations

import pytest


def spy_agreement(monkeypatch: pytest.MonkeyPatch) -> list:
    """Record ``(reading, record)`` for every call of the comparator, behaviour unchanged."""
    import socr.tables.reading_agreement as reading_agreement

    calls: list = []
    real = reading_agreement.compare_numbers

    def recording(reference, reading, *, provenance):
        record = real(reference, reading, provenance=provenance)
        calls.append((reading, record))
        return record

    monkeypatch.setattr(reading_agreement, "compare_numbers", recording)
    return calls


def assert_agreed_on(calls: list, snippet: str) -> None:
    """The check ran on the reading containing ``snippet``, compared numbers, and agreed."""
    hits = [record for reading, record in calls if snippet in reading]
    assert hits, f"the text-layer check never ran on the reading containing {snippet!r}"
    for record in hits:
        assert record.compared > 0, f"vacuous agreement on {snippet!r}: nothing was compared"
        assert record.agreed, f"the control reading disagrees with its layer: {record.disputed}"
