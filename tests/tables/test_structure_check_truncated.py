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
# #643 (round 2, reviewed): table_shaped_native_row_count's footnote-band /
# right-aligned-lane fix, exercised through THIS caller too (structure_check
# is the second of the two consumers the shared helper serves -- see
# manifest.py's own reconciliation tests in test_row_corroboration.py for the
# identical fixtures scored through the other caller).
# ---------------------------------------------------------------------------


def _label_row_words(y: float, label: str, values: list[str]) -> list[tuple]:
    words = [(10.0, y, 10.0 + len(label) * 6.0, y + 10.0, label)]
    x = 100.0
    for value in values:
        words.append((x, y, x + len(value) * 6.0, y + 10.0, value))
        x += 60.0
    return words


def _bare_marker_footnote_words(n: int, y_start: float = 500.0) -> list[tuple]:
    """*n* BARE numeric footnote lines led by a marker (``1) 45 12``, no
    prose at all) -- the issue's own original synthetic repro. #643 round
    4 (owner ruling): deliberately KEPT, not excluded -- see
    ``tests/tables/test_row_corroboration.py::_footnote_bands`` for the
    identical construction and the ruling's full rationale."""
    words: list[tuple] = []
    for i in range(n):
        y = y_start + i * 20.0
        base = 500.0 + i * 200.0
        words.append((base, y, base + 20.0, y + 10.0, f"{i + 1})"))
        words.append((base + 20.0, y, base + 30.0, y + 10.0, "45"))
        words.append((base + 50.0, y, base + 60.0, y + 10.0, "12"))
    return words


def test_bare_marker_footnotes_stay_fail_closed_through_structure_check() -> None:
    """#643 round 4 (owner ruling): a COMPLETE 3-row narrow-table candidate
    sharing a page with 5 BARE marker-led numeric lines (no prose) IS
    flagged truncated -- accepted, documented collateral (see the mirrored
    test and its rationale in test_row_corroboration.py), not a bug."""
    rows = [("Revenue", "1,204", "980"), ("Costs", "500", "410"), ("Total", "704", "570")]
    words: list[tuple] = []
    md_rows = []
    for i, (label, a, b) in enumerate(rows):
        words += _label_row_words(10.0 + i * 20.0, label, [a, b])
        md_rows.append((label, a, b))
    words += _bare_marker_footnote_words(5)
    complete_md = "| Item | A | B |\n|---|---|---|\n"
    complete_md += "".join(f"| {label} | {a} | {b} |\n" for label, a, b in md_rows)
    assert table_truncated(complete_md, words) is True


def test_truncated_narrow_candidate_also_flagged_with_bare_footnotes_present() -> None:
    """The SAME native page as above, candidate now emitting only 1 of its 3
    real rows: also flagged -- the bare-footnote collateral does not make a
    genuinely truncated candidate look any more complete."""
    rows = [("Revenue", "1,204", "980"), ("Costs", "500", "410"), ("Total", "704", "570")]
    words: list[tuple] = []
    for i, (label, a, b) in enumerate(rows):
        words += _label_row_words(10.0 + i * 20.0, label, [a, b])
    words += _bare_marker_footnote_words(5)
    truncated_md = "| Item | A | B |\n|---|---|---|\n| Revenue | 1,204 | 980 |\n"
    assert table_truncated(truncated_md, words) is True


def test_right_aligned_column_truncation_still_detected() -> None:
    """A right-aligned numeric column (values of differing digit-width, so
    x0 differs even though x1 recurs) must still catch a truncated
    candidate. Originally written for round-2's x0/x1 lane check; round 4
    deleted lane geometry entirely, but these rows are not marker-led, so
    they are unconditionally counted regardless -- kept as a general
    truncation-detection regression test, since alignment must never
    matter to whether a row counts."""
    values = [str(10 + i) if i % 2 == 0 else str(100 + i) for i in range(10)]
    words: list[tuple] = []
    rows = []
    for i, value in enumerate(values):
        y = 10.0 + i * 20.0
        x1 = 130.0
        x0 = x1 - len(value) * 6.0
        words += [
            (10.0, y, 40.0, y + 10.0, f"Line{i}"),
            (x0, y, x1, y + 10.0, value),
        ]
        rows.append((f"Line{i}", value))
    complete_md = "| Item | Value |\n|---|---|\n"
    complete_md += "".join(f"| {label} | {value} |\n" for label, value in rows)
    truncated_md = "| Item | Value |\n|---|---|\n"
    truncated_md += "".join(f"| {label} | {value} |\n" for label, value in rows[:-3])
    assert table_truncated(complete_md, words) is False
    assert table_truncated(truncated_md, words) is True


def test_aligned_marker_footnotes_do_not_truncate_complete_table() -> None:
    """Reviewed round-2 (P2): two footnotes with IDENTICAL phrasing (so their
    numbers align at the same x as a genuine table column) must still be
    excluded -- via the marker + non-contiguous-numeric-tokens signal, not
    lane recurrence -- so a complete 3-row table is not flagged truncated."""
    rows = [("Alpha", "50", "60"), ("Beta", "70", "80"), ("Gamma", "90", "100")]
    words: list[tuple] = []
    md_rows = []
    for i, (label, a, b) in enumerate(rows):
        y = 20.0 + i * 20.0
        words += [
            (10.0, y, 40.0, y + 10.0, label),
            (100.0, y, 120.0, y + 10.0, a),
            (160.0, y, 180.0, y + 10.0, b),
        ]
        md_rows.append((label, a, b))
    for i in range(2):
        y = 200.0 + i * 20.0
        words += [
            (10.0, y, 25.0, y + 10.0, f"{i + 1})"),
            (30.0, y, 80.0, y + 10.0, "See pages"),
            (90.0, y, 110.0, y + 10.0, "45"),
            (120.0, y, 150.0, y + 10.0, "and"),
            (160.0, y, 180.0, y + 10.0, "12"),
        ]
    complete_md = "| Item | A | B |\n|---|---|---|\n"
    complete_md += "".join(f"| {label} | {a} | {b} |\n" for label, a, b in md_rows)
    assert table_truncated(complete_md, words) is False


# ---------------------------------------------------------------------------
# #643 round 3 (reviewed, Astra's second pass): the round-2 allowlist itself
# lost source evidence -- a marker-led row was dropped whenever nothing else
# on the page happened to share its lane, even with no prose signal at all.
# Round 3 is a denylist: every shape-eligible band counts unless positively
# excluded. Mirrors the round-3 probes in test_row_corroboration.py through
# THIS caller too, at a 10-row scale so structure_check's own
# ``_STRAY_HEADER_BAND_ALLOWANCE`` (== 1) cannot swallow the signal.
# ---------------------------------------------------------------------------


def _numbered_table_words(
    rows: list[tuple[str, str]], x_marker: float = 10.0, y_start: float = 10.0
) -> list[tuple]:
    """A marker-led, single-numeric-column table: ``1) Alpha | 80``."""
    words: list[tuple] = []
    for i, (label, value) in enumerate(rows):
        y = y_start + i * 20.0
        words += [
            (x_marker, y, x_marker + 20.0, y + 10.0, f"{i + 1})"),
            (x_marker + 20.0, y, x_marker + 200.0, y + 10.0, label),
            (x_marker + 200.0, y, x_marker + 220.0, y + 10.0, value),
        ]
    return words


def _numbered_table_md(rows: list[tuple[str, str]]) -> str:
    md = "| Item | Value |\n|---|---|\n"
    md += "".join(f"| {i + 1}) {label} | {value} |\n" for i, (label, value) in enumerate(rows))
    return md


def test_complete_and_truncated_numbered_table() -> None:
    """A COMPLETE 10-row numbered table must not be flagged truncated -- the
    printed row numbers must never be misread as footnote markers stripping
    every row. The SAME page with the candidate's last 3 rows dropped must
    still be flagged."""
    rows = [(f"Item{i}", str(80 + i)) for i in range(10)]
    words = _numbered_table_words(rows)
    complete_md = _numbered_table_md(rows)
    truncated_md = _numbered_table_md(rows[:-3])
    assert table_truncated(complete_md, words) is False
    assert table_truncated(truncated_md, words) is True


def test_genuine_second_numbered_table_still_flags_truncation() -> None:
    """Two REAL numbered tables at different x offsets. A candidate covering
    only the first must still be flagged truncated -- each row has exactly
    ONE remaining numeric token after its own marker, so being marker-led
    does not exempt either table (neither exclusion test fires, regardless
    of x position)."""
    rows1 = [(f"Item{i}", str(80 + i)) for i in range(5)]
    words1 = _numbered_table_words(rows1, x_marker=10.0, y_start=10.0)
    rows2 = [(f"Line{i}", str(500 + i)) for i in range(5)]
    words2 = _numbered_table_words(rows2, x_marker=600.0, y_start=300.0)
    words = words1 + words2
    md1 = _numbered_table_md(rows1)
    assert table_truncated(md1, words) is True


def test_one_row_second_numbered_table_beside_multirow_first_not_truncating() -> None:
    """A one-row second table (still marker-led, ``1) Solo | 999``) beside a
    10-row first table must not, by itself, cause the first table's own
    COMPLETE candidate to be flagged truncated -- its single remaining
    numeric token triggers neither exclusion test, so nothing in this
    module chokes on it. (The exact ``table_shaped_native_row_count`` this
    produces -- 11, not 10 -- is pinned directly in
    ``test_row_corroboration.py``'s mirror of this fixture;
    ``_STRAY_HEADER_BAND_ALLOWANCE`` makes 10 vs. 11 indistinguishable
    through THIS caller's boolean output alone.)"""
    rows1 = [(f"Item{i}", str(80 + i)) for i in range(10)]
    words1 = _numbered_table_words(rows1, x_marker=10.0, y_start=10.0)
    words2 = [
        (600.0, 500.0, 620.0, 510.0, "1)"),
        (620.0, 500.0, 700.0, 510.0, "Solo"),
        (800.0, 500.0, 820.0, 510.0, "999"),
    ]
    words = words1 + words2
    complete_md = _numbered_table_md(rows1)
    assert table_truncated(complete_md, words) is False


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
