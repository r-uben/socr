"""#1244 C1 helper for tests whose control document holds model readings nothing can check.

A document is ``completed`` only if every page has a final record with status SUCCESS. A model
reading is CHECKED only against a born-digital layer of the page's own that shares a number with
it, so a hand-built or blank-PDF control document whose pages are model readings carries
``text_layer_numbers=UNKNOWN`` WARNING pages and is no longer SUCCESS. Those tests are about the
loop, the ladder or the resume gate, not about that label.

``assert_clean_or_only_unchecked`` keeps their intent: the document is either SUCCESS (a provider
state whose pages are native or checked) or demoted by the C1 invariant, named in the
document's ``error``. A document that is neither SUCCESS nor carrying that
note (a failed page, a different demotion) fails the assertion. Other, informational notes in
``error`` are tolerated: they were present before C1 and did not move the status.
"""

from __future__ import annotations

import re

from socr.core.result import DocumentStatus

_ONLY_UNCHECKED = re.compile(r"page\(s\) [\d, ]+ did not finish with status success")


def assert_clean_or_only_unchecked(result) -> None:
    if result.status == DocumentStatus.SUCCESS:
        return
    assert result.status == DocumentStatus.AUDIT_FAILED, result.status
    assert result.error and _ONLY_UNCHECKED.search(result.error), result.error
