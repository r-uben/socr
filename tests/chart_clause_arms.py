"""#1145 slice 3 (chart): the ONE way another ticket's test runs both arms of this change.

Sixteen tests across six tickets assert that a clean chart page ships SUCCESS. #1145 makes
a chart shipped as an image with untranscribed values a page socr did not CHECK, so those
pages are demoted and marked. Their expectations are superseded, not wrong.

Rewriting each assertion to "now WARNING" would have been the cheap conversion and the wrong
one: every one of those tests exists for a property of ITS OWN ticket -- the #960 scan guard
refusing a stale cached page, #913 withholding a digit, #263 refusing shredded text -- and
an assertion rewritten around the new status no longer says whether that property holds. So
each test runs the SAME scenario twice in one process with this change on and off, and
asserts:

* the clause-OFF arm still satisfies its ticket's original assertion, unchanged and at full
  strength;
* the clause-ON arm differs from it by exactly the ``figure_text`` mark and the WARNING
  status, and by nothing else;
* its ticket's real property holds in BOTH arms.

A regression in #913/#960/#990/#994/#263's own behaviour therefore still fails its own test,
and #1145 is pinned as exactly the intended delta.

**What is patched, and why it is this and nothing else.** ``_chart_values_clause`` -- the one
clause builder. Not ``visual_values_not_transcribed``'s emitter, not a predicate production
shares. Round 2 of this ticket produced a "259 pages" figure that reached a decision log and
a PR before Fable found it was an artefact of patching ``native_math_damage_ships``, which
#1005's production guard also calls: forcing it True demoted ~256 pages in the control arm
alone. The real figure was 3. To measure what one function does, replace that function.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

from socr.core.marks import Check
from socr.core.result import FailureMode, PageStatus
from verdict_marks_helper import assert_differs_only_by_marks, canonical_marks


@contextmanager
def chart_clause_off():
    """Run the body with #1145's chart clause disabled, and nothing else changed."""
    with patch("socr.core.verdict._chart_values_clause", return_value=None):
        yield


def assert_only_the_chart_clause_differs(
    on_text: str,
    off_text: str,
    page_num: int,
    *,
    on_status=None,
    off_status=None,
    on_mode=None,
) -> None:
    """The clause-ON arm is the clause-OFF arm plus the mark, and plus the demotion.

    Four assertions, because any three can hold while the change is still wrong: the OFF arm
    carries no ``figure_text`` mark (if it did, this helper would be comparing two marked
    pages and the delta would be invisible); the ON arm carries exactly one; the bodies
    underneath are byte-identical; and the status moved from SUCCESS to WARNING with the
    mode that explains it, rather than the text changing while the status stayed clean.
    """
    assert not canonical_marks(off_text, page_num), (
        "the clause-off arm carries a canonical mark, so the clause is not actually off "
        "and the comparison below proves nothing"
    )
    marks = canonical_marks(on_text, page_num)
    assert len(marks) == 1, f"expected exactly one canonical line, got {marks!r}"
    assert f"{Check.FIGURE_TEXT.value}=UNKNOWN" in marks[0], marks[0]
    assert_differs_only_by_marks(on_text, off_text, page_num)

    if off_status is not None:
        assert off_status == PageStatus.SUCCESS.value or off_status is PageStatus.SUCCESS, (
            f"the clause-off arm is not SUCCESS ({off_status}), so this ticket's original "
            "assertion is not what is being preserved"
        )
    if on_status is not None:
        assert on_status == PageStatus.WARNING.value or on_status is PageStatus.WARNING, (
            f"the clause-on arm is not WARNING ({on_status}): the page is marked but every "
            "consumer that reads the status still calls it clean"
        )
    if on_mode is not None:
        expected = FailureMode.VISUAL_VALUES_UNTRANSCRIBED
        assert on_mode == expected.value or on_mode is expected, (
            f"the demotion does not explain itself ({on_mode}), so a later pass cannot tell "
            "it from a page demoted by a guard that names nothing"
        )
