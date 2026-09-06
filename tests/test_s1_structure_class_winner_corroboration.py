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


# TICKET-A1b (#634) round 3 (owner redesign, 2026-09-06): a bigger synthetic
# table -- 20 rows, not 3 -- so dropping rows crosses A1a's own
# ROW_CORROBORATION_MIN (36/39 ~= 0.9231) allowance the same way the
# reviewer's real repro did (bulletin p1 qwen candidate, 39 real rows, minus
# its last 10 -- see the review this round answers, and
# docs/log/2026-09-06_A1b-corroboration-selection.md's round-3 section for
# why a bbox- or distance-anchored region could not catch this and the
# shape-based reconciliation replaces both the row-count-allowance and the
# edge-row check). ceil(20 * 36/39) == 19, so dropping even one row from
# either end must fail to reconcile; the complete table must still win.
_LARGE_ROWS = [(2000 + i, float(100 + i), float(200 + i)) for i in range(20)]


def _large_md(rows: list[tuple[int, float, float]]) -> str:
    lines = ["| Year | A | B |", "|---|---|---|"]
    lines += [f"| {year} | {a} | {b} |" for year, a, b in rows]
    return "\n".join(lines) + "\n"


_LARGE_NATIVE_WORDS: list[tuple] = []
for _i, (_year, _a, _b) in enumerate(_LARGE_ROWS):
    _LARGE_NATIVE_WORDS += _row_words(10.0 + _i * 20.0, [str(_year), str(_a), str(_b)])

_LARGE_REGION = (0.0, 0.0, 200.0, 10.0 + len(_LARGE_ROWS) * 20.0)


def _large_page(attempts: list[PageOutput]) -> PageState:
    p = PageState(page_num=1)
    p.is_born_digital = True
    p.native_text = NATIVE_PROSE
    p.has_tables = True
    p.attempts = attempts
    p.best_output = attempts[-1] if attempts else None
    p.native_words = _LARGE_NATIVE_WORDS
    p.detected_table_bboxes = [_LARGE_REGION]
    return p


def test_candidate_missing_trailing_rows_does_not_win() -> None:
    """The reviewer's repro (round 3): a candidate whose LAST rows are
    silently dropped, but whose remaining rows all still bind cleanly, must
    NOT win the corroboration fallback -- the complete candidate must.

    ``ceil(20 * 36/39) == 19``: dropping ONE row (19/20) would still clear
    the allowance -- this drops TWO (18/20) to sit cleanly below it.
    """
    complete = _grid_reading_output("qwen", _large_md(_LARGE_ROWS))
    truncated = _grid_reading_output("qwen", _large_md(_LARGE_ROWS[:-2]))

    p_complete = _large_page([complete])
    p_truncated = _large_page([truncated])

    assert _strict_grid_authored_pool(p_complete) == []
    assert _strict_grid_authored_pool(p_truncated) == []

    winner = structure_class_grid_winner(p_complete)
    assert winner is not None
    assert winner.engine == "qwen"

    assert _row_corroborated_grid_winner(p_truncated) is None
    assert structure_class_grid_winner(p_truncated) is None
    assert structure_class_floor_applies(p_truncated) is True


def test_candidate_missing_leading_rows_does_not_win() -> None:
    """The same defect at the OTHER edge: dropping the FIRST two rows
    instead of the last must also fail to reconcile -- the shape-based
    check counts every unmatched table-shaped native band wherever it sits,
    not just past the bound range's trailing edge (unlike the round-2
    edge-row walk it replaces, which was scored anchored to a region and
    could not be trusted on real fixtures at all -- see the log).
    """
    truncated = _grid_reading_output("qwen", _large_md(_LARGE_ROWS[2:]))
    p_truncated = _large_page([truncated])

    assert _strict_grid_authored_pool(p_truncated) == []
    assert _row_corroborated_grid_winner(p_truncated) is None
    assert structure_class_grid_winner(p_truncated) is None
    assert structure_class_floor_applies(p_truncated) is True


def test_strict_pool_winner_ships_even_with_a_corroborating_candidate_present() -> None:
    """When S1 case (i)'s own strict grid-authored pool is NOT empty, the
    corroboration fallback must never fire, let alone override it -- even
    when a separate, ragged candidate on the SAME page would otherwise
    corroborate and win the fallback outright.

    ``p.best_output`` is deliberately the (not-yet-audit_passed)
    ``corroborating`` attempt, not ``strict`` directly: ``_reaches_
    structure_class_branch`` treats an already-``audit_passed`` non-native
    ``best_output`` as proof some EARLIER branch already shipped it (a
    precondition this test must not trip, since it is testing what S1
    itself picks from ``p.attempts``, mirroring how S1 selection actually
    assigns ``best_output`` only after choosing a winner).
    """
    strict_md = (
        "| Year | A | B |\n"
        "|---|---|---|\n"
        "| 2018 | 100.0 | 200.0 |\n"
        "| 2019 | 110.0 | 210.0 |\n"
        "| 2020 | 120.0 | 220.0 |\n"
    )
    strict = PageOutput(
        page_num=1,
        text=strict_md,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
        confidence=0.9,
        failure_mode=FailureMode.NONE,
    )
    corroborating = _grid_reading_output("gemini", GOOD_MD)

    # attempts=[strict, corroborating]: ``_floored_structure_class_page`` sets
    # ``best_output = attempts[-1]``, so ``corroborating`` (audit_passed=False)
    # ships as best_output -- keeping ``_reaches_structure_class_branch``'s
    # early-return guard (which fires whenever a non-native best_output is
    # already audit_passed) from short-circuiting before S1 even runs.
    # ``strict`` stays reachable to ``_strict_grid_authored_pool`` via
    # ``p.attempts`` regardless of position.
    p = _floored_structure_class_page(with_native_words=True, attempts=[strict, corroborating])

    assert _strict_grid_authored_pool(p) == [strict]

    winner = structure_class_grid_winner(p)
    assert winner is not None
    assert winner is strict
    assert winner.engine == "qwen"
