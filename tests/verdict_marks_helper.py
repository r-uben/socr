"""#1145: the ONE way a test strips the canonical mark block before comparing a body.

Many tests predate #1145 and assert that a demoted page's body is unchanged, byte for
byte. #1145 deliberately changes that: a page socr did not CHECK now carries one canonical
line per uncovered clause as its first rendered block. Those expectations are superseded,
not wrong -- so each is converted through this single helper rather than loosened in place.

The contract, and why it is shaped this way:

* ``strip_verdict_marks`` removes EXACTLY the leading canonical block and nothing else, so
  the caller's original comparison runs on the original bytes at the original strictness.
  A body change anywhere still fails it -- pinned by ``test_gh1145_strip_helper.py`` and by
  the external ``strip-helper-hides-body-change`` mutant.
* It ASSERTS the block was there and parses under ``core/marks.py``. A helper that quietly
  returned the text unchanged when no mark was found would turn every converted test into
  a tautology: the mark could stop being emitted entirely and they would all still pass.
  That is the failure mode this file exists to avoid, so the assertion is not optional.
* It recognises only the #1145 canonical grammar. Legacy marks (``[page N failed: …]``,
  ``> [table unverified: …]``, #1129's ``> [page unaccepted: …]``) are left in place: they
  predate this change, the old expectations already account for them, and stripping them
  here would hide a real regression in their emitters.
"""

from __future__ import annotations

import re

from socr.core.marks import CANONICAL_MARK_RE, is_canonical_mark, leading_mark_blocks

#: ``assemble_pages`` writes ``## Page N\n\n<body>``, so a test holding a DOCUMENT string
#: (or an ``EngineResult.pages[i].text``, which carries the header) has the heading as its
#: FIRST block and the mark block second. Reading only the leading blocks there finds no
#: marks, which would make a converted assertion vacuous -- it is how the #165
#: byte-identity test first reported "unmarked" on a page that was plainly marked.
#: Production never sees this form (the seam works on bodies), so the tolerance lives here
#: rather than in ``core/marks.py``, where it could mask a genuinely mislocated mark.
_PAGE_HEADING_RE = re.compile(r"^##\s+Page\s+\d+\s*\n+")


def _body_of(text: str) -> str:
    return _PAGE_HEADING_RE.sub("", text, count=1)


def canonical_marks(text: str, page_num: int) -> list[str]:
    """The canonical #1145 lines in this page's leading mark blocks, in order.

    Accepts a page body or a one-page document string (the ``## Page N`` heading is
    skipped).
    """
    return [
        ln
        for ln in leading_mark_blocks(_body_of(text), page_num)
        if is_canonical_mark(ln, page_num)
    ]


def strip_verdict_marks(text: str, page_num: int, *, required: bool = True) -> str:
    """Return *text* without its leading canonical mark lines.

    With ``required`` (the default) it asserts at least one such line was present and
    well-formed. Pass ``required=False`` only where the test itself parametrises over
    marked and unmarked pages, and assert the presence separately.
    """
    found = canonical_marks(text, page_num)
    if required:
        assert found, (
            f"expected a canonical #1145 mark block on page {page_num}, found none. "
            f"Leading blocks were {leading_mark_blocks(text, page_num)!r}. If the mark is no "
            f"longer emitted here, that is a regression, not a stale expectation."
        )
    for line in found:
        assert is_canonical_mark(line, page_num), f"malformed canonical mark: {line!r}"

    # Keep a ``## Page N`` heading exactly where it was: the caller may be comparing a
    # whole document string, and the heading is part of what it expects back. Only the
    # mark lines are removed.
    heading = ""
    m = _PAGE_HEADING_RE.match(text)
    if m:
        heading, text = m.group(0), text[m.end() :]

    lines = text.splitlines(keepends=True)
    out: list[str] = []
    idx = 0
    removed = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if stripped and CANONICAL_MARK_RE.match(stripped) and stripped in found:
            idx += 1
            removed += 1
            # The seam joins the banner block and the body with one blank line; drop it
            # with the last mark so the remaining body starts exactly where it used to.
            if removed == len(found) and idx < len(lines) and not lines[idx].strip():
                idx += 1
            continue
        if not stripped and removed and removed < len(found):
            idx += 1
            continue
        break
    out = lines[idx:]
    assert removed == len(found), (
        f"stripped {removed} of {len(found)} canonical marks; the block is not leading"
    )
    return heading + "".join(out)


def assert_differs_only_by_marks(marked: str, clean: str, page_num: int) -> None:
    """Pin the ASYMMETRY two arms of a before/after test now have, not a value.

    Several tests predate #1145 by running the pipeline twice over one fixture, changing
    only the one signal under test, and asserting the two texts are equal. #1145 changes
    those arms asymmetrically on purpose: the arm that fires is DEMOTED, so it is a page
    socr did not confirm and must say so, while the other ships SUCCESS and carries
    nothing. Stripping both sides would be the wrong conversion twice over -- it would pass
    if the mark stopped being emitted, and it would pass if the mark appeared on the clean
    arm too.

    So all three halves are asserted here: the clean arm carries no canonical mark, the
    demoted arm carries at least one, and once those lines are removed the two texts are
    byte-identical. The last one is the original assertion, at the original strictness: a
    body change anywhere on either arm still fails it.
    """
    assert not canonical_marks(clean, page_num), (
        f"the arm that did not fire carries a canonical mark: "
        f"{canonical_marks(clean, page_num)!r}. A CHECKED page must carry nothing."
    )
    assert canonical_marks(marked, page_num), (
        "the arm that fired is demoted and carries no canonical mark; #1145 requires one. "
        "If the demotion itself is gone, that is the regression."
    )
    assert strip_verdict_marks(marked, page_num) == clean
