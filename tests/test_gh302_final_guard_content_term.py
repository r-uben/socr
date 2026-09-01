"""GH-302: the last shipping backstop must ask the content question too.

`_apply_table_emission_guard` is what stands between a whole-document CLI
attempt and the reader -- those paths never reach the agentic judge or the
post-route recheck. It ran `table_emission_defect` alone, and an empty but
well-formed table is not an EMISSION defect, so GH-190's own fixture shipped
SUCCESS through the final guard.

Scope, deliberately: the SHAPE term is not added here. Running the whole of
`table_output_defect` at this seam was tried and turns pages `--native-only`
ships FLAGGED into hard failures -- a routing change #302 rules out. That is
pinned below so the scope cannot be widened by accident.
"""

from __future__ import annotations

import pytest

from socr.core.manifest import _apply_table_emission_guard
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.tables.reconcile import table_emission_defect

# GH-190's fixture: well-formed, equal widths, and completely empty.
EMPTY_TABLE = "| A | B |\n| --- | --- |\n|  |  |\n"
# GH-301's two spellings: the delimiter is narrower than the header.
BLANK_HEADER = "|  |  |  |\n| --- | --- |\n|  |  |  |\n"
POPULATED_HEADER = "| A | B | C |\n| --- | --- |\n|  |  |\n"
# The control: same shape, carrying values.
POPULATED = "| A | B |\n| --- | --- |\n| 1 | 2 |\n"

EMPTIES = [
    ("gh190 equal width", EMPTY_TABLE),
    ("gh301 blank header", BLANK_HEADER),
    ("gh301 populated header", POPULATED_HEADER),
]


def _shipped(text: str) -> PageOutput:
    """Drive the guard exactly as `_winning_page_output` does."""
    return _apply_table_emission_guard(
        PageOutput(
            page_num=1,
            text=text,
            status=PageStatus.SUCCESS,
            engine="native",
            audit_passed=True,
        ),
        1,
    )


@pytest.mark.parametrize(("name", "markdown"), EMPTIES)
def test_an_empty_table_does_not_ship_success(name: str, markdown: str) -> None:
    """No silent content loss: the failure has to surface, not just be known."""
    out = _shipped(markdown)
    assert out.status is PageStatus.ERROR, f"{name}: shipped {out.status}"
    assert out.audit_passed is False, f"{name}: shipped with audit_passed True"
    assert out.failure_mode is FailureMode.TABLE_EMISSION_INVALID
    assert "table_content_empty" in (out.error or ""), (
        f"{name}: the reason does not name the defect: {out.error!r}"
    )
    assert any("table_content_empty" in n for n in out.audit_notes), (
        f"{name}: the defect never reached audit_notes: {out.audit_notes}"
    )


@pytest.mark.parametrize(("name", "markdown"), EMPTIES)
def test_the_emission_term_alone_would_have_missed_it(name: str, markdown: str) -> None:
    """Anchor: this is why the content term had to be added.

    Without it, the guard's own predicate returns nothing on all three, so the
    test above would be measuring a defect the guard never saw.
    """
    assert table_emission_defect(markdown) == "", (
        f"{name}: fixture is caught by the emission term, so it does not "
        "exercise the content term this ticket adds"
    )


def test_a_populated_table_still_ships_clean() -> None:
    """Control: the guard must not start failing tables that carry values."""
    out = _shipped(POPULATED)
    assert out.status is PageStatus.SUCCESS
    assert out.audit_passed is True
    assert out.failure_mode is FailureMode.NONE


def test_the_shape_term_is_deliberately_not_applied_here() -> None:
    """Scope pin: #302 must not become a routing change.

    A width mismatch WITH content is `grid_shape` at `table_output_defect`, and
    running that whole predicate here turns pages `--native-only` ships FLAGGED
    into hard failures. Shape keeps its existing owner; this seam asks only
    "is it empty".
    """
    from socr.tables.structure_check import table_output_defect

    ragged = "| A | B | C |\n| --- | --- |\n| 1 | 2 |\n"
    assert table_output_defect(ragged, None) == "grid_shape", (
        "fixture must be a shape defect, or this pin measures nothing"
    )

    out = _shipped(ragged)
    assert out.status is PageStatus.SUCCESS, (
        "the final guard hard-failed a shape defect; that is a routing change #302 rules out"
    )


MIXED_PAGE = (
    "## Section 4\n\n"
    "Real prose that must survive, carrying a finding worth 0.86.\n\n"
    "| A | B |\n| --- | --- |\n|  |  |\n\n"
    "More prose after the table, also worth keeping.\n"
)


def test_a_mixed_page_is_demoted_not_discarded() -> None:
    """#449 review: the fix must not become the loss it exists to prevent.

    `table_content_defect` fires on ONE table run, but the guard's failure
    branch replaces the WHOLE page with a marker. So a page carrying real prose
    beside an empty table had all of it swapped out -- a content loss
    introduced by a no-content-loss fix.

    The content term therefore DEMOTES: same status, audit flag, failure mode
    and notes, with the page's text intact. Compare #252: never destroy a page
    in order to flag it.
    """
    out = _shipped(MIXED_PAGE)

    assert out.status is PageStatus.ERROR, "the empty table must still surface"
    assert out.audit_passed is False
    assert "table_content_empty" in (out.error or "")

    assert "Real prose that must survive" in (out.text or ""), (
        f"the prose was discarded to flag the table: {out.text!r}"
    )
    assert "More prose after the table" in (out.text or "")
    assert "| A | B |" in (out.text or ""), "the table itself must not vanish either"
    assert not (out.text or "").startswith("[page "), "the page was replaced by a failure marker"


def test_an_emission_defect_still_replaces_the_page() -> None:
    """The emission branch keeps its existing behaviour, deliberately.

    An emission defect means the markdown itself is malformed, so there is
    nothing on the page that could be trusted to keep. Only the content term
    changed; widening the demote-don't-discard rule to emission would be a
    separate decision this ticket did not make.
    """
    bad = "| A | B | C |\n| --- | --- |\n| 1 | 2 | 3 |\n"
    assert table_emission_defect(bad), "fixture must be an emission defect"

    out = _shipped(bad)
    assert out.status is PageStatus.ERROR
    assert (out.text or "").startswith("[page 1 failed:"), (
        f"the emission branch stopped replacing the page: {out.text!r}"
    )
