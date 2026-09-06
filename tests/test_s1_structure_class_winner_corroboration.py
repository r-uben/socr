"""TICKET-A1b (#634): structure-class selection consults native row
corroboration before the fail-closed floor.

CONSILIUM-GATE ruling (2026-09-06, recorded in full in
``docs/log/2026-09-06_A1b-corroboration-selection.md``): the fallback fires
whenever the S1 strict grid-authored pool (``_strict_grid_authored_pool``) is
EMPTY, i.e. the floor would otherwise ship the fail-closed marker -- not
gated on ``native_table_header_unattributed``, which measures ``False`` on
every one of the ticket's own six motivating ECB fixtures.

This file builds ``PageState`` objects directly (no PDF, no provider) with a
synthetic three-row table and hand-built native word tuples shaped exactly
like ``page.get_text("words")`` -- ``(x0, y0, x1, y1, text)`` -- so the
on/off differential is deterministic and has nothing to do with any real
fixture cache.
"""

from __future__ import annotations

from socr.core.manifest import (
    _row_corroborated_grid_winner,
    _strict_grid_authored_pool,
    structure_class_floor_applies,
    structure_class_grid_winner,
)
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import PageState

# A ragged (non-uniform-body) table: has_strict_table_grid() is False (so it
# never enters S1 case (i)'s strict pool) but has_authored_table_grid() is
# True (so _grid_reading_attempt admits it to the corroboration fallback's
# own pool). See docs/log/2026-09-06_A1b-corroboration-selection.md for the
# has_strict_table_grid / has_authored_table_grid measurement this shape is
# based on.
GOOD_MD = (
    "| Year | A | B |\n"
    "|---|---|---|\n"
    "| units |\n"
    "| 2018 | 100.0 | 200.0 |\n"
    "| 2019 | 110.0 | 210.0 |\n"
    "| 2020 | 120.0 | 220.0 |\n"
)

# Same shape, values that do not appear anywhere in the native words below --
# corroborate_rows must bind zero rows against this one.
BAD_MD = (
    "| Year | A | B |\n"
    "|---|---|---|\n"
    "| units |\n"
    "| 2018 | 999.0 | 888.0 |\n"
    "| 2019 | 777.0 | 666.0 |\n"
    "| 2020 | 555.0 | 444.0 |\n"
)

NATIVE_PROSE = "Table 1 below reports quarterly balances for 2018-2020."
REGION = (0.0, 0.0, 200.0, 100.0)


def _row_words(y: float, tokens: list[str]) -> list[tuple]:
    words = []
    x = 0.0
    for tok in tokens:
        words.append((x, y, x + 8.0, y + 10.0, tok))
        x += 12.0
    return words


NATIVE_WORDS: list[tuple] = (
    _row_words(10.0, ["2018", "100.0", "200.0"])
    + _row_words(30.0, ["2019", "110.0", "210.0"])
    + _row_words(50.0, ["2020", "120.0", "220.0"])
)


def _grid_reading_output(engine: str, text: str) -> PageOutput:
    """A non-native attempt shaped so it reaches the S1 branch: no judge
    verdict either way (``audit_passed=False``, ``rejection_class=""``), so
    it is excluded from ``_strict_grid_authored_pool`` regardless of grid
    strictness -- the strict pool must be EMPTY for the corroboration
    fallback to be reachable at all.
    """
    return PageOutput(
        page_num=1,
        text=text,
        status=PageStatus.SUCCESS,
        engine=engine,
        audit_passed=False,
        confidence=0.5,
        failure_mode=FailureMode.NONE,
    )


def _floored_structure_class_page(
    *, with_native_words: bool, attempts: list[PageOutput]
) -> PageState:
    """A born-digital, structure-class page reaching the S1 branch with an
    empty strict grid pool -- exactly the shape S1's floor applies to before
    A1b.

    Deliberately does NOT set ``native_table_header_unattributed`` (or
    ``native_table_structure_failed``): the CONSILIUM-GATE ruling this ticket
    implements exists precisely because that flag measures ``False`` on the
    ticket's own six motivating ECB fixtures -- setting it here would route
    the page to the OLDER D3 floor (``_reaches_structure_class_branch``
    returns ``False`` whenever that flag is set alongside attempts, since D3
    owns the page instead of S1) rather than the S1 branch this test targets.
    """
    p = PageState(page_num=1)
    p.is_born_digital = True
    p.native_text = NATIVE_PROSE
    p.has_tables = True
    p.attempts = attempts
    p.best_output = attempts[-1] if attempts else None
    if with_native_words:
        p.native_words = NATIVE_WORDS
        p.detected_table_bboxes = [REGION]
    return p


def test_strict_pool_empty_precondition() -> None:
    """Sanity: the ragged candidates used below never enter the strict pool,
    so any winner ``structure_class_grid_winner`` picks must come from the
    corroboration fallback, not S1 case (i).
    """
    p = _floored_structure_class_page(
        with_native_words=True,
        attempts=[_grid_reading_output("qwen", GOOD_MD)],
    )
    assert _strict_grid_authored_pool(p) == []


def test_corroborating_candidate_wins_over_the_floor() -> None:
    """A ragged candidate whose rows reproduce the native page, scored
    against the OTHER (non-corroborating) candidate on the same page, wins
    the corroboration fallback and the floor no longer applies.
    """
    good = _grid_reading_output("qwen", GOOD_MD)
    bad = _grid_reading_output("gemini", BAD_MD)
    p = _floored_structure_class_page(with_native_words=True, attempts=[bad, good])

    assert structure_class_floor_applies(p) is False
    winner = structure_class_grid_winner(p)
    assert winner is not None
    assert winner.engine == "qwen"
    assert "100.0" in (winner.text or "")


def test_non_corroborating_candidate_alone_still_floors() -> None:
    """The floor must still apply when nothing on the page corroborates --
    the fallback is a rescue, not a guarantee of a winner.
    """
    bad = _grid_reading_output("gemini", BAD_MD)
    p = _floored_structure_class_page(with_native_words=True, attempts=[bad])

    assert _row_corroborated_grid_winner(p) is None
    assert structure_class_grid_winner(p) is None
    assert structure_class_floor_applies(p) is True


def test_no_native_words_is_identical_to_pre_a1b_floor() -> None:
    """A page with no cached ``native_words`` (or no detected table bbox) is
    unaffected by A1b: the fallback abstains and the floor applies exactly
    as it did before this ticket, even for the SAME corroborating candidate
    that wins when native words ARE present.
    """
    good = _grid_reading_output("qwen", GOOD_MD)
    p_without = _floored_structure_class_page(with_native_words=False, attempts=[good])
    p_with = _floored_structure_class_page(with_native_words=True, attempts=[good])

    assert _row_corroborated_grid_winner(p_without) is None
    assert structure_class_grid_winner(p_without) is None
    assert structure_class_floor_applies(p_without) is True

    assert structure_class_grid_winner(p_with) is not None
    assert structure_class_floor_applies(p_with) is False


def test_row_unverified_marker_spliced_for_unbound_row() -> None:
    """A row_corroboration result naming an unbound row must be spliced into
    the shipped markdown as a trailing ``<!-- row unverified -->`` comment on
    THAT row only, never on a bound row -- ``RowCorroboration`` itself is
    built directly here (rather than re-deriving a ``clears``-True fixture
    with exactly one wrong row, a share/extra-share balancing act belonging
    to ``test_row_corroboration.py``, not this selection-level file) since
    ``_splice_unverified_row_markers`` only consumes ``unbound_rows``.
    """
    from dataclasses import replace as dc_replace

    from socr.core.manifest import _apply_row_corroboration_disclosure
    from socr.tables.row_corroboration import corroborate_rows

    good = _grid_reading_output("qwen", GOOD_MD)
    rc = corroborate_rows(NATIVE_WORDS, GOOD_MD, REGION)
    assert rc.unbound_rows == ((),)  # sanity: all three rows really do bind
    # Force the middle row "unbound" to exercise splicing independent of the
    # real gate -- this test is about marker PLACEMENT, not the clears gate.
    forced = dc_replace(rc, unbound_rows=((1,),))

    disclosed = _apply_row_corroboration_disclosure(
        state=_FakeState(),
        page_num=1,
        grid_winner=good,
        corroboration=forced,
        region_kind="bbox_union",
        coverage_share=1.0,
    )
    lines = disclosed.text.splitlines()
    marked = [ln for ln in lines if "row unverified" in ln]
    assert len(marked) == 1
    assert "2019" in marked[0]  # the second (index-1) numeric body row
    assert "2018" not in marked[0]
    assert "2020" not in marked[0]


class _FakeState:
    def __init__(self) -> None:
        self.events: list = []
