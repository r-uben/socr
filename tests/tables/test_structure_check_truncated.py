"""TICKET-A2 (#645): a truncated model candidate must never beat a complete one.

Two layers, mirroring the ticket's own split:

1. ``structure_check.table_truncated`` / ``table_output_defect`` unit tests --
   pure string (+ optional ``words``) predicates, no ``PageState``.
2. S1 winner-selection integration tests using the exact ``PageState`` /
   ``PageOutput`` fixture pattern established in
   ``tests/test_s1_structure_class_winner_corroboration.py`` -- including the
   two REAL fixtures named in the ticket (ECB economic bulletin p2 and p3,
   both cached live on 2026-09-07 / 2026-09-06), so the guard is proven
   against the actual defect, not only a synthetic shape.
"""

from __future__ import annotations

from socr.core.manifest import (
    _strict_grid_authored_pool,
    structure_class_grid_winner,
    structure_class_truncated_engines,
)
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import PageState
from socr.tables.structure_check import DEFECT_TABLE_TRUNCATED, table_output_defect, table_truncated

# ---------------------------------------------------------------------------
# Layer 1: table_truncated / table_output_defect unit tests
# ---------------------------------------------------------------------------

# Mixed style: a fully-bordered body row, then a final row missing its
# trailing pipe -- the shape term (a) targets (the census p3 fixture below
# reproduces this exact shape verbatim). Same column count on every row
# (``has_strict_table_grid`` needs the uniform width) -- only the RIGHT
# border of the last row is missing.
MIXED_STYLE_MD = (
    "| Year | A | B |\n| :--- | :--- | :--- |\n| 2018 | 100.0 | 200.0 |\n| 2019 | 110.0 | 21"
)

# Same rows, but NONE of them close the right border -- a candidate's own
# consistent (if unusual) formatting choice, not truncation.
ALL_UNTERMINATED_MD = (
    "| Year | A | B\n"
    "| :--- | :--- | :---\n"
    "| 2018 | 100.0 | 200.0\n"
    "| 2019 | 110.0 | 210.0\n"
    "| 2020 | 120.0 | 220.0"
)

# A single body row: nothing to establish a style against, so term (a) must
# abstain regardless of whether that one row is terminated.
SINGLE_ROW_MD = "| Year | A | B |\n| :--- | :--- | :--- |\n| 2018 | 100.0 | 200.0 |\n"

COMPLETE_MD = (
    "| Year | A | B |\n"
    "| :--- | :--- | :--- |\n"
    "| 2018 | 100.0 | 200.0 |\n"
    "| 2019 | 110.0 | 210.0 |\n"
    "| 2020 | 120.0 | 220.0 |\n"
)


def test_mixed_style_final_row_is_truncated() -> None:
    assert table_truncated(MIXED_STYLE_MD, None) is True


def test_all_unterminated_style_is_not_truncated() -> None:
    """A candidate's own consistent no-trailing-pipe convention must not be
    mistaken for a row cut off mid-emission.
    """
    assert table_truncated(ALL_UNTERMINATED_MD, None) is False


def test_single_body_row_abstains() -> None:
    """Nothing to compare the last row's style against -- must not flag."""
    assert table_truncated(SINGLE_ROW_MD, None) is False


def test_complete_table_is_not_truncated() -> None:
    assert table_truncated(COMPLETE_MD, None) is False


def test_table_output_defect_reports_truncation() -> None:
    assert table_output_defect(MIXED_STYLE_MD, None) == DEFECT_TABLE_TRUNCATED


def test_table_output_defect_silent_on_complete_table() -> None:
    assert table_output_defect(COMPLETE_MD, None) != DEFECT_TABLE_TRUNCATED


def _row_words(y: float, tokens: list[str]) -> list[tuple]:
    words = []
    x = 0.0
    for tok in tokens:
        words.append((x, y, x + 8.0, y + 10.0, tok))
        x += 12.0
    return words


def test_row_shortfall_past_allowance_is_truncated() -> None:
    """Term (b): candidate's numeric rows undercount the native table-shaped
    row count by more than ``ROW_CORROBORATION_MIN`` (36/39) permits, even
    though every row it DOES emit is fully bordered (term (a) alone would
    miss this -- no style break, whole rows are simply absent).
    """
    rows = [(2000 + i, float(100 + i), float(200 + i)) for i in range(20)]
    complete_md = "| Year | A | B |\n|---|---|---|\n"
    complete_md += "".join(f"| {y} | {a} | {b} |\n" for y, a, b in rows)
    truncated_md = "| Year | A | B |\n|---|---|---|\n"
    truncated_md += "".join(f"| {y} | {a} | {b} |\n" for y, a, b in rows[:-2])

    words: list[tuple] = []
    for i, (year, a, b) in enumerate(rows):
        words += _row_words(10.0 + i * 20.0, [str(year), str(a), str(b)])

    assert table_truncated(complete_md, words) is False
    assert table_truncated(truncated_md, words) is True


def test_row_shortfall_within_allowance_is_not_truncated() -> None:
    """Dropping a single row out of 20 (19/20, above ceil(20*36/39)==19)
    must clear the allowance and NOT be flagged.
    """
    rows = [(2000 + i, float(100 + i), float(200 + i)) for i in range(20)]
    almost_complete_md = "| Year | A | B |\n|---|---|---|\n"
    almost_complete_md += "".join(f"| {y} | {a} | {b} |\n" for y, a, b in rows[:-1])

    words: list[tuple] = []
    for i, (year, a, b) in enumerate(rows):
        words += _row_words(10.0 + i * 20.0, [str(year), str(a), str(b)])

    assert table_truncated(almost_complete_md, words) is False


def test_row_shortfall_abstains_without_words() -> None:
    rows = [(2000 + i, float(100 + i), float(200 + i)) for i in range(20)]
    truncated_md = "| Year | A | B |\n|---|---|---|\n"
    truncated_md += "".join(f"| {y} | {a} | {b} |\n" for y, a, b in rows[:-5])
    assert table_truncated(truncated_md, None) is False


# ---------------------------------------------------------------------------
# Layer 2: S1 winner-selection integration -- synthetic pair
# ---------------------------------------------------------------------------

NATIVE_PROSE = "Table 1 below reports quarterly balances for 2018-2020."
REGION = (0.0, 0.0, 200.0, 100.0)
NATIVE_WORDS: list[tuple] = (
    _row_words(10.0, ["2018", "100.0", "200.0"])
    + _row_words(30.0, ["2019", "110.0", "210.0"])
    + _row_words(50.0, ["2020", "120.0", "220.0"])
)


def _grid_reading_output(engine: str, text: str, *, page_num: int = 1) -> PageOutput:
    """A non-native attempt with NO judge verdict either way -- reaches only
    the ragged corroboration fallback's pool (``_grid_reading_attempt``), the
    same shape ``test_s1_structure_class_winner_corroboration.py`` uses.
    """
    return PageOutput(
        page_num=page_num,
        text=text,
        status=PageStatus.SUCCESS,
        engine=engine,
        audit_passed=False,
        confidence=0.5,
        failure_mode=FailureMode.NONE,
    )


def _strict_grid_output(engine: str, text: str, *, page_num: int = 1) -> PageOutput:
    """A judge-cleared (``audit_passed=True``) reading of a UNIFORM-width
    grid -- qualifies for S1 case (i)'s own strict pool
    (``_grid_authored_attempt``), so the truncation guard's effect on that
    pool specifically can be exercised directly, independent of the
    row-corroboration fallback's own matching requirements.
    """
    return PageOutput(
        page_num=page_num,
        text=text,
        status=PageStatus.SUCCESS,
        engine=engine,
        audit_passed=True,
        confidence=0.9,
        failure_mode=FailureMode.NONE,
    )


def _page(attempts: list[PageOutput], *, page_num: int = 1) -> PageState:
    p = PageState(page_num=page_num)
    p.is_born_digital = True
    p.native_text = NATIVE_PROSE
    p.has_tables = True
    p.attempts = attempts
    p.best_output = attempts[-1] if attempts else None
    p.native_words = NATIVE_WORDS
    p.detected_table_bboxes = [REGION]
    return p


def _native_filler(page_num: int = 1) -> PageOutput:
    """A native-engine placeholder ``best_output`` -- never a candidate
    itself (``_grid_shaped_attempt``/``_grid_reading_attempt`` both exclude
    any ``engine.startswith("native")``), and NOT ``audit_passed``, so
    ``_reaches_structure_class_branch``'s own early-return guard (fires only
    when ``best_output.audit_passed`` is already True) does not short-circuit
    before S1 runs. Lets a strict-pool test set real ``audit_passed=True``
    candidates in ``p.attempts`` -- the only way ``_grid_authored_attempt``
    admits them -- while still reaching the S1 branch at all, mirroring
    ``test_s1_structure_class_winner_corroboration.py``'s own
    ``corroborating``-as-``best_output`` pattern.
    """
    return PageOutput(
        page_num=page_num,
        text=NATIVE_PROSE,
        status=PageStatus.SUCCESS,
        engine="native",
        audit_passed=False,
        confidence=0.1,
        failure_mode=FailureMode.NONE,
    )


def _strict_page(strict_attempts: list[PageOutput], *, page_num: int = 1) -> PageState:
    """Like ``_page``, but with a native, non-qualifying filler as
    ``best_output`` so the *strict* candidates in ``strict_attempts`` are
    reachable to ``_strict_grid_authored_pool`` purely via ``p.attempts``.
    """
    return _page([*strict_attempts, _native_filler(page_num)], page_num=page_num)


def test_truncated_candidate_loses_to_complete_one() -> None:
    truncated = _strict_grid_output("qwen", MIXED_STYLE_MD)
    complete = _strict_grid_output("gemini", COMPLETE_MD)
    p = _strict_page([truncated, complete])

    assert truncated not in _strict_grid_authored_pool(p)
    winner = structure_class_grid_winner(p)
    assert winner is not None
    assert winner.engine == "gemini"
    assert structure_class_truncated_engines(p) == ("qwen",)


def test_truncated_candidate_alone_still_ships_flagged() -> None:
    """TICKET-A2's own clause: with nothing to compare against, a truncated
    reading is still the only evidence there is and must still ship.
    """
    truncated = _strict_grid_output("qwen", MIXED_STYLE_MD)
    p = _strict_page([truncated])

    assert truncated in _strict_grid_authored_pool(p)
    winner = structure_class_grid_winner(p)
    assert winner is not None
    assert winner.engine == "qwen"
    # nothing to compare against -- not reported as dropped
    assert structure_class_truncated_engines(p) == ()


def test_all_candidates_truncated_ships_the_available_one() -> None:
    """If every candidate for the page truncates, nothing is dropped by
    design -- there is no complete reading to prefer.
    """
    t1 = _strict_grid_output("qwen", MIXED_STYLE_MD)
    t2 = _strict_grid_output("gemini", MIXED_STYLE_MD.replace("110.0", "120.0"))
    p = _strict_page([t1, t2])

    assert structure_class_truncated_engines(p) == ()
    assert _strict_grid_authored_pool(p) == [t1, t2]
    winner = structure_class_grid_winner(p)
    assert winner is not None


def test_all_unterminated_style_candidate_not_dropped() -> None:
    """A candidate whose OWN consistent formatting never closes the right
    border must not be treated as truncated, even alongside a candidate
    that is genuinely cut off mid-row.
    """
    consistent = _strict_grid_output("qwen", ALL_UNTERMINATED_MD)
    truncated = _strict_grid_output("gemini", MIXED_STYLE_MD)
    p = _strict_page([truncated, consistent])

    assert structure_class_truncated_engines(p) == ("gemini",)
    assert consistent in _strict_grid_authored_pool(p)
    winner = structure_class_grid_winner(p)
    assert winner is not None
    assert winner.engine == "qwen"


def test_truncation_guard_pins_the_difference_strict_pool() -> None:
    """Pin the DIFFERENCE the guard makes to the strict pool's own
    membership, not an absolute outcome: with the guard's predicate true for
    one candidate and false for the other, the strict pool must differ from
    what it would be if truncation were ignored entirely.
    """
    truncated = _strict_grid_output("qwen", MIXED_STYLE_MD)
    complete = _strict_grid_output("gemini", COMPLETE_MD)
    p = _strict_page([truncated, complete])

    with_guard = _strict_grid_authored_pool(p)
    # Simulate "guard off": both candidates would be admitted to the pool by
    # the grid-authored predicate alone (module-private call, matching what
    # _strict_grid_authored_pool used to return pre-A2).
    from socr.core.manifest import _grid_authored_attempt

    without_guard = [out for out in [truncated, complete] if _grid_authored_attempt(out)]

    assert truncated not in with_guard
    assert truncated in without_guard
    assert with_guard != without_guard
    assert with_guard != without_guard


def test_cross_pool_truncated_strict_loses_to_complete_wide_pool_only() -> None:
    """TICKET-A2 cross-pool difference (#648, the #645 bulletin p2 shape):
    the truncated candidate is the ONLY strict-pool member (judge-cleared,
    ``audit_passed=True``), while the complete candidate only clears the
    WIDER ``_grid_reading_attempt`` pool (``audit_passed=False`` -- never
    admitted to the strict pool at all). A within-pool comparison (both
    candidates strict, as ``test_truncated_candidate_loses_to_complete_one``
    above exercises) would miss this: here the strict pool never contains
    the complete candidate to compare against in the first place, so only
    scoring truncation against the WIDER union (``_truncated_grid_reading_ids``)
    lets the strict pool see a complete alternative exists and empty itself,
    letting the row-corroboration fallback ship the complete wide-pool-only
    reading instead of the truncated strict one.
    """
    truncated = _strict_grid_output("qwen", MIXED_STYLE_MD)
    complete = _grid_reading_output("gemini", COMPLETE_MD)
    p = _strict_page([truncated, complete])

    assert _strict_grid_authored_pool(p) == []
    winner = structure_class_grid_winner(p)
    assert winner is not None
    assert winner.engine == "gemini"
    assert structure_class_truncated_engines(p) == ("qwen",)


# ---------------------------------------------------------------------------
# Layer 2: S1 winner-selection integration -- real fixtures
# ---------------------------------------------------------------------------

# ECB economic bulletin p2 (5.4 MFI loans table), cached live 2026-09-07 at
# ~/Data/socr/a1c-live-2026-09-07/.../cache/{08,e7}/*.json -- the ticket's
# own motivating fixture: the S1 strict pool held ONLY the truncated qwen
# candidate, so A1b's corroboration fallback never ran.
BULLETIN_P2_TRUNCATED = (
    "# 5 Money and credit\n\n"
    "## 5.4 MFI loans to euro area non-financial corporations and households 1)\n"
    "(EUR billions and annual growth rates; seasonally adjusted; outstanding amounts "
    "and growth rates at end of period; transactions during period)\n\n"
    "| | **Non-financial corporations 2)** | | | | | **Households 3)** | | | | |\n"
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    "| | **Total** | **Adjusted loans 4)** | **Up to 1 year** | **Over 1 and up to 5 years** "
    "| **Over 5 years** | **Total** | **Adjusted loans 4)** | **Loans for consumption** "
    "| **Loans for house purchase** | **Other loans** |\n"
    "| | **1** | **2** | **3** | **4** | **5** | **6** | **7** | **8** | **9** | **10** |\n\n"
    "**Outstanding amounts**\n\n"
    "| | | | | | | | | | | |\n"
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    "| 2018 | 4,404.9 | 4,489.0 | 991.4 | 844.2 | 2,569.4 | 5,741.9 | 6,024.9 | 682.6 | 4,356.4 | 702.9 |\n"
    "| 2019 | 4"
)

BULLETIN_P2_COMPLETE = (
    "# 5 Money and credit\n\n"
    "## 5.4 MFI loans to euro area non-financial corporations and households 1)\n"
    "(EUR billions and annual growth rates; seasonally adjusted; outstanding amounts "
    "and growth rates at end of period; transactions during period)\n\n"
    "| | **Non-financial corporations 2)** | | | | | **Households 3)** | | | | |\n"
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    "| | **Total** | **Adjusted loans 4)** | **Up to 1 year** | **Over 1 and up to 5 years** "
    "| **Over 5 years** | **Total** | **Adjusted loans 4)** | **Loans for consumption** "
    "| **Loans for house purchase** | **Other loans** |\n"
    "| | **1** | **2** | **3** | **4** | **5** | **6** | **7** | **8** | **9** | **10** |\n\n"
    "**Outstanding amounts**\n\n"
    "| | | | | | | | | | | |\n"
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    "| 2018 | 4,404.9 | 4,489.0 | 991.4 | 844.2 | 2,569.4 | 5,741.9 | 6,024.9 | 682.6 | 4,356.4 | 702.9 |\n"
    "| 2019 | 4,475.8 | 4,577.9 | 967.4 | 878.0 | 2,630.4 | 5,931.1 | 6,224.0 | 720.1 | 4,524.6 | 686.4 |\n"
    "| 2020 | 4,723.6 | 4,841.3 | 898.9 | 1,012.0 | 2,812.7 | 6,119.9 | 6,390.1 | 700.2 | 4,725.1 | 694.6 |\n"
)

# ECB economic bulletin p3 (5.5 counterparts to M3 table), the exact defect
# named in the ticket's problem statement: cached at
# ~/Data/socr/census-ecb-2026-09-06/out/.../cache/{1d,83}/*.json -- qwen ends
# mid-number at "| 2019 | 364.2 | 7,05" (34/389 words) vs gemini's complete
# reading (389/389 words).
BULLETIN_P3_TRUNCATED = (
    "# 5 Money and credit\n\n"
    "## 5.5 Counterparts to M3 other than credit to euro area residents 1)\n"
    "(EUR billions and annual growth rates; seasonally adjusted; outstanding amounts "
    "and growth rates at end of period; transactions during period)\n\n"
    "| | MFI liabilities | | | | | MFI assets | | | |\n"
    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    "| **Central government holdings 2)** | **Longer-term financial liabilities vis-à-vis "
    "other euro area residents** | | | | | **Net external assets** | **Other** | | |\n"
    "| | **Total** | **Deposits with an agreed maturity of over 2 years** "
    "| **Deposits redeemable at notice of over 3 months** | **Debt securities with a maturity "
    "of over 2 years** | **Capital and reserves** | | **Total** | **Repos with central "
    "counterparties 3)** | **Reverse repos to central counterparties 3)** |\n"
    "| **1** | **2** | **3** | **4** | **5** | **6** | **7** | **8** | **9** | **10** |\n\n"
    "### Outstanding amounts\n\n"
    "| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |\n"
    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    "| 2018 | 389.2 | 6,817.4 | 1,940.0 | 56.1 | 2,099.7 | 2,721.6 | 1,030.0 | 460.2 | 187.0 | 194.9 |\n"
    "| 2019 | 364.2 | 7,05"
)

BULLETIN_P3_COMPLETE = (
    "# 5 Money and credit\n\n"
    "### 5.5 Counterparts to M3 other than credit to euro area residents $^{1)}$\n"
    "(EUR billions and annual growth rates; seasonally adjusted; outstanding amounts "
    "and growth rates at end of period; transactions during period)\n\n"
    "| | MFI liabilities | | | | | | MFI assets | | | |\n"
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    "| | Central government holdings $^{2)}$ | Longer-term financial liabilities vis-à-vis "
    "other euro area residents | | | | | Net external assets | Other | | |\n"
    "| | | Total | Deposits with an agreed maturity of over 2 years "
    "| Deposits redeemable at notice of over 3 months | Debt securities with a maturity of "
    "over 2 years | Capital and reserves | | Total | | |\n"
    "| | | | | | | | | | Repos with central counter-parties $^{3)}$ "
    "| Reverse repos to central counter-parties $^{3)}$ |\n"
    "| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |\n"
    "| **Outstanding amounts** | | | | | | | | | | |\n"
    "| 2018 | 389.2 | 6,817.4 | 1,940.0 | 56.1 | 2,099.7 | 2,721.6 | 1,030.0 | 460.2 | 187.0 | 194.9 |\n"
    "| 2019 | 364.2 | 7,058.9 | 1,946.1 | 50.1 | 2,156.5 | 2,906.1 | 1,455.5 | 452.3 | 178.9 | 187.2 |\n"
    "| 2020 | 749.0 | 6,967.4 | 1,916.7 | 42.1 | 1,994.9 | 3,013.7 | 1,432.7 | 539.6 | 130.1 | 139.2 |\n"
)


def _fixture_words(rows: list[list[str]]) -> list[tuple]:
    """Hand-built native words shaped like ``page.get_text("words")``, one
    row of tokens per printed table row -- following
    ``test_s1_structure_class_winner_corroboration.py``'s own ``_row_words``
    pattern, faithful to the values the COMPLETE candidate's table actually
    contains (real geometry is not cached alongside these JSON fixtures).
    """
    words: list[tuple] = []
    for i, tokens in enumerate(rows):
        words += _row_words(10.0 + i * 20.0, tokens)
    return words


def test_real_fixture_bulletin_p2_picks_complete_candidate() -> None:
    truncated = _grid_reading_output("qwen", BULLETIN_P2_TRUNCATED, page_num=2)
    complete = _grid_reading_output("qwen", BULLETIN_P2_COMPLETE, page_num=2)

    words = _fixture_words(
        [
            [
                "2018",
                "4,404.9",
                "4,489.0",
                "991.4",
                "844.2",
                "2,569.4",
                "5,741.9",
                "6,024.9",
                "682.6",
                "4,356.4",
                "702.9",
            ],
            [
                "2019",
                "4,475.8",
                "4,577.9",
                "967.4",
                "878.0",
                "2,630.4",
                "5,931.1",
                "6,224.0",
                "720.1",
                "4,524.6",
                "686.4",
            ],
            [
                "2020",
                "4,723.6",
                "4,841.3",
                "898.9",
                "1,012.0",
                "2,812.7",
                "6,119.9",
                "6,390.1",
                "700.2",
                "4,725.1",
                "694.6",
            ],
        ]
    )
    region = (0.0, 0.0, 400.0, 90.0)

    p = PageState(page_num=2)
    p.is_born_digital = True
    p.native_text = "5.4 MFI loans to euro area non-financial corporations and households"
    p.has_tables = True
    p.attempts = [truncated, complete]
    p.best_output = complete
    p.native_words = words
    p.detected_table_bboxes = [region]

    assert table_truncated(BULLETIN_P2_TRUNCATED, words) is True
    assert table_truncated(BULLETIN_P2_COMPLETE, words) is False
    assert truncated not in _strict_grid_authored_pool(p)

    winner = structure_class_grid_winner(p)
    assert winner is not None
    assert "4,475.8" in (winner.text or "")  # the 2019 row's real value
    assert structure_class_truncated_engines(p) == ("qwen",)


def test_real_fixture_bulletin_p3_picks_complete_candidate() -> None:
    """The ticket's own headline defect: qwen ends mid-number
    (``| 2019 | 364.2 | 7,05``) while gemini's reading is complete.
    """
    truncated = _grid_reading_output("qwen", BULLETIN_P3_TRUNCATED, page_num=3)
    complete = _grid_reading_output("gemini", BULLETIN_P3_COMPLETE, page_num=3)

    words = _fixture_words(
        [
            [
                "2018",
                "389.2",
                "6,817.4",
                "1,940.0",
                "56.1",
                "2,099.7",
                "2,721.6",
                "1,030.0",
                "460.2",
                "187.0",
                "194.9",
            ],
            [
                "2019",
                "364.2",
                "7,058.9",
                "1,946.1",
                "50.1",
                "2,156.5",
                "2,906.1",
                "1,455.5",
                "452.3",
                "178.9",
                "187.2",
            ],
            [
                "2020",
                "749.0",
                "6,967.4",
                "1,916.7",
                "42.1",
                "1,994.9",
                "3,013.7",
                "1,432.7",
                "539.6",
                "130.1",
                "139.2",
            ],
        ]
    )
    region = (0.0, 0.0, 400.0, 90.0)

    p = PageState(page_num=3)
    p.is_born_digital = True
    p.native_text = "5.5 Counterparts to M3 other than credit to euro area residents"
    p.has_tables = True
    p.attempts = [truncated, complete]
    p.best_output = complete
    p.native_words = words
    p.detected_table_bboxes = [region]

    assert table_truncated(BULLETIN_P3_TRUNCATED, words) is True
    assert table_truncated(BULLETIN_P3_COMPLETE, words) is False
    assert truncated not in _strict_grid_authored_pool(p)

    winner = structure_class_grid_winner(p)
    assert winner is not None
    assert "7,058.9" in (winner.text or "")  # the real 2019 value; qwen never emits it
    assert structure_class_truncated_engines(p) == ("qwen",)
