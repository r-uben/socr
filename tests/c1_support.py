"""#1244 C1 helper for tests whose control document holds model readings nothing can check.

A document is ``completed`` only if every page has a final record that publishes as SUCCESS. A
model reading is CHECKED only against a born-digital layer of the page's own that shares a number
with it, so a hand-built or blank-PDF control document whose pages are model readings carries
``text_layer_numbers=UNKNOWN`` WARNING pages and is no longer SUCCESS. Those tests are about the
loop, the ladder or the resume gate, not about that label.

``assert_clean_or_only_unchecked`` keeps their intent without letting an unrelated demotion
pass: the document is either SUCCESS (a provider state whose pages are native or checked) or
AUDIT_FAILED by the C1 invariant alone. In the second case the pages the note names must be a
subset of ``pages`` (the pages the fixture is DESIGNED to leave unchecked), and every other
sentence of the document's ``error`` must be one of ``other_notes`` (substrings of notes the
fixture is designed to carry). A late ERROR, a page outside the fixture's design, or a
different defect bucket fails the assertion.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from socr.core.result import DocumentStatus

_C1_NOTE = re.compile(r"page\(s\) ([\d, ]+) did not finish with status success")


def assert_clean_or_only_unchecked(
    result, *, pages: Iterable[int], other_notes: Iterable[str] = ()
) -> None:
    if result.status == DocumentStatus.SUCCESS:
        return
    assert result.status == DocumentStatus.AUDIT_FAILED, result.status
    error = result.error or ""
    match = _C1_NOTE.search(error)
    assert match, f"AUDIT_FAILED without the C1 note: {error!r}"
    named = {int(n) for n in match.group(1).split(",")}
    assert named <= set(pages), f"C1 names pages {sorted(named)} outside the design {sorted(pages)}"
    allowed = tuple(other_notes)
    for sentence in (s.strip() for s in error.split(";")):
        if not sentence or _C1_NOTE.search(sentence) or "passing audit" in sentence:
            continue
        # a note of its own: it must be one the fixture is designed to carry
        assert any(a in sentence for a in allowed), f"unrelated note in error: {sentence!r}"
