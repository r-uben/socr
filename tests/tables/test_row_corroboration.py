"""Tests for row corroboration (``socr.tables.row_corroboration``).

Hermetic: synthetic ``page.get_text("words")``-shaped tuples and literal
markdown strings only. No PDFs, no corpus, no provider. Each test states
the behavioural claim it pins so a body-swapped or logic-stripped module
(imports intact) fails the assertion, not merely that the module is absent.
"""

from __future__ import annotations

from socr.core.manifest import _row_shape_reconciliation_ok
from socr.tables.row_corroboration import (
    EXTRA_NUMBERS_MAX_SHARE,
    ROW_CORROBORATION_MIN,
    corroborate_rows,
    table_shaped_native_row_count,
)

REGION = (0.0, 0.0, 400.0, 400.0)


def w(x0: float, y0: float, text: str, height: float = 10.0) -> tuple:
    """One ``page.get_text("words")`` tuple: (x0, y0, x1, y1, text, block, line, word)."""
    x1 = x0 + max(len(text) * 6.0, 6.0)
    y1 = y0 + height
    return (x0, y0, x1, y1, text, 0, 0, 0)


def native_row(y0: float, label: str, values: list[str]) -> list[tuple]:
    words = [w(10.0, y0, label)]
    x = 100.0
    for value in values:
        words.append(w(x, y0, value))
        x += 60.0
    return words


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def test_perfect_page_binds_every_row():
    """Every candidate row matches its native line -> bound == total, clears True."""
    words = []
    words += native_row(10.0, "Revenue", ["1,204", "980"])
    words += native_row(30.0, "Costs", ["500", "410"])
    words += native_row(50.0, "Total", ["704", "570"])
    markdown = md_table(
        ["Item", "2023", "2022"],
        [
            ["Revenue", "1,204", "980"],
            ["Costs", "500", "410"],
            ["Total", "704", "570"],
        ],
    )
    result = corroborate_rows(words, markdown, REGION)
    assert result.bound == result.total == 3
    assert result.native_numeric_rows == 3
    assert result.extra_numbers == ()
    assert result.share == 1.0
    assert result.clears is True


def test_wrapped_label_page_36_of_39_clears():
    """A candidate whose row-value share matches the census's measured 36/39
    floor is judged corroborated (inclusive boundary).

    Each native line carries FOUR genuine values (real ECB rows this shape
    was measured on typically carry several data columns; a wider row also
    dilutes the corrupted rows' duplicate-value extras below
    EXTRA_NUMBERS_MAX_SHARE, matching the real bulletin-p2 fixture's own
    measured extra_share of ~0.0075). A misattributed row can fail to bind
    (its values never sit together on one native line) without either value
    being individually fabricated -- exactly the wrapped-label defect: the
    row's own label split it away from its native line's OTHER value, not
    from a value that doesn't exist at all.
    """
    words = []
    rows = []
    for i in range(39):
        y = 10.0 + i * 20.0
        a, b, c, d = f"{100 + i}.5", f"{500 + i}.5", f"{900 + i}.5", f"{1300 + i}.5"
        words += native_row(y, f"Line {i}", [a, b, c, d])
        rows.append([f"Line {i}", a, b, c, d])
    # Corrupt 3 rows: pair this row's first value with the NEXT row's second
    # value. Both values are individually genuine (present natively on some
    # line) but never appear TOGETHER, in order, on one native line.
    for i in (5, 15, 25):
        rows[i][2] = rows[i + 1][2]
    markdown = md_table(["Item", "A", "B", "C", "D"], rows)
    tall_region = (0.0, 0.0, 400.0, 800.0)  # 39 rows at 20pt pitch span ~780pt
    result = corroborate_rows(words, markdown, tall_region)
    assert result.total == 39
    assert result.bound == 36
    assert result.share == ROW_CORROBORATION_MIN
    # Each swapped "B" value now occurs twice in the candidate (its wrong
    # placement and its own true row) but only once natively -- the second
    # occurrence is legitimately an "extra" (duplicated) value, not zero.
    assert len(result.extra_numbers) == 3
    assert result.extra_share < EXTRA_NUMBERS_MAX_SHARE
    assert result.clears is True


def test_row_value_swap_between_numeric_labels_does_not_clear():
    """Whole-row value misattribution: swapping two numeric-labeled rows'
    VALUE cells (labels stay in place) on a realistic single-table-block
    candidate (13 rows, same row shape as one section of the real ECB
    bulletin p1 qwen fixture) must not corroborate.

    Anchoring a numeric-labeled row (a bare year, here) to its own label
    ties its match to its OWN printed native line; without that anchor, the
    swapped-in value tuple is still a genuine contiguous run on the OTHER
    row's native band and clears wrongly (the exact defect this ticket's
    review found: bound stayed 39/39 on the real fixture before this fix).
    See docs/log/2026-09-06_A1a-row-corroboration.md for the confirmation
    against the real fixture itself (swapping all three repeated 2018/2019
    occurrences across the candidate's three sections: 39 total rows,
    bound drops 39 -> 33, clears False).
    """
    labels = [
        "2018",
        "2019",
        "2020",
        "2020 Q2",
        "Q3",
        "Q4",
        "2021 Q1",
        "2020 Nov.",
        "Dec.",
        "2021 Jan.",
        "Feb.",
        "Mar.",
        "Apr. (b)",
    ]
    words = []
    rows = []
    for i, label in enumerate(labels):
        y = 10.0 + i * 20.0
        a, b = f"{100 + i}.1", f"{500 + i}.2"
        words += native_row(y, label, [a, b])
        rows.append([label, a, b])
    # Swap 2018's and 2019's VALUE cells; labels stay in place.
    rows[0][1:], rows[1][1:] = rows[1][1:], rows[0][1:]
    markdown = md_table(["Item", "A", "B"], rows)
    result = corroborate_rows(words, markdown, REGION)
    assert result.total == 13
    assert result.bound == 11  # exactly the two swapped rows fail to bind
    assert result.clears is False


def test_dropped_row_does_not_clear():
    """A candidate row OMITTED entirely (not merely garbled) shrinks bound
    and total together, so ROW_CORROBORATION_MIN alone cannot see it: a
    39-row perfect candidate with one MIDDLE row deleted still measures
    bound == total (every remaining row still finds its own native band,
    just at a permanently shifted index) and extra_numbers == () (nothing
    fabricated) -- share == 1.0, clears would wrongly be True without the
    skipped_native_rows gate (round 3 review, Astra: real bulletin p1 qwen
    candidate minus its own 2018 row measured exactly this: bound=38,
    total=38, share=1.0, extras=0, clears=True before this fix).

    The dropped row leaves its own native band unmatched, strictly between
    the bound rows immediately before and after it -- a gap
    ``skipped_native_rows`` counts precisely because no unbound candidate
    row (a present-but-garbled row) explains it away. See
    docs/log/2026-09-06_A1a-row-corroboration.md, round 3, for why a
    dropped row at the very START or END of a table block is NOT caught by
    this mechanism (there is no bound-row pair to straddle it) -- an
    acknowledged, documented blind spot at whole-page-region scoping.
    """
    words = []
    rows = []
    for i in range(39):
        y = 10.0 + i * 20.0
        a, b, c, d = f"{100 + i}.5", f"{500 + i}.5", f"{900 + i}.5", f"{1300 + i}.5"
        words += native_row(y, f"Line {i}", [a, b, c, d])
        rows.append([f"Line {i}", a, b, c, d])
    del rows[20]  # a genuine middle row, not the block's first or last
    markdown = md_table(["Item", "A", "B", "C", "D"], rows)
    tall_region = (0.0, 0.0, 400.0, 800.0)
    result = corroborate_rows(words, markdown, tall_region)
    assert result.bound == result.total == 38
    assert result.share == 1.0
    assert result.extra_numbers == ()
    assert result.skipped_native_rows == 1
    assert result.clears is False


def test_duplicate_row_second_occurrence_unbound():
    """Strict (not non-decreasing) monotonicity: a candidate row DUPLICATED
    verbatim can only bind its native band ONCE -- the second occurrence's
    identical token run has nowhere left to go (the native page prints that
    line exactly once) and is left unbound, surfaced via ``bound < total``
    and the block's own ``unbound_rows`` index for the duplicate's own
    position.

    On its own, one duplicated row (out of 55) is a "known partial": it
    trips neither ROW_CORROBORATION_MIN (54/55 well above the 36/39 floor),
    EXTRA_NUMBERS_MAX_SHARE (the duplicate's own values are legitimate
    elsewhere in the region, so they register as ordinary "extra"
    duplicated-value occurrences, diluted below the 0.02 gate at this row
    count), nor ``skipped_native_rows`` (no band is skipped -- the
    duplicate simply fails to bind, it does not shift anyone else's
    index). ``clears`` stays True; A1b's own row-count reconciliation
    against the region's native effective row count is the intended second
    line of defense for a single duplicate, not this module's gates (see
    docs/log/2026-09-06_A1a-row-corroboration.md, round 3).
    """
    words = []
    rows = []
    for i in range(55):
        y = 10.0 + i * 20.0
        a, b, c, d = f"{100 + i}.5", f"{500 + i}.5", f"{900 + i}.5", f"{1300 + i}.5"
        words += native_row(y, f"Line {i}", [a, b, c, d])
        rows.append([f"Line {i}", a, b, c, d])
    rows.insert(21, list(rows[20]))  # duplicate a middle row verbatim
    markdown = md_table(["Item", "A", "B", "C", "D"], rows)
    tall_region = (0.0, 0.0, 400.0, 1200.0)
    result = corroborate_rows(words, markdown, tall_region)
    assert result.total == 56
    assert result.bound == 55  # exactly the duplicate's second occurrence is unbound
    assert result.skipped_native_rows == 0
    assert result.clears is True


def test_zero_numeric_native_page_abstains():
    """No native numeric evidence in the region -> abstain, not fail."""
    words = [w(10.0, 10.0, "Notes"), w(10.0, 30.0, "See appendix")]
    markdown = md_table(["Item", "Value"], [["Revenue", "1,204"]])
    result = corroborate_rows(words, markdown, REGION)
    assert result.native_numeric_rows == 0
    assert result.clears is None


def test_zero_candidate_rows_abstains():
    """A candidate with no numeric body rows has nothing to corroborate."""
    words = native_row(10.0, "Revenue", ["1,204"])
    markdown = md_table(["Item", "Notes"], [["Revenue", "see text"]])
    result = corroborate_rows(words, markdown, REGION)
    assert result.total == 0
    assert result.clears is None


def test_all_rows_bound_plus_fabricated_rows_fails_extra_numbers():
    """Every real row binds, but extra fabricated rows push extra_share over
    EXTRA_NUMBERS_MAX_SHARE -> clears False even though share == 1.0."""
    words = []
    rows = []
    for i in range(10):
        y = 10.0 + i * 20.0
        value = f"{100 + i}.5"
        words += native_row(y, f"Line {i}", [value])
        rows.append([f"Line {i}", value])
    # Fabricated extra rows: values with no native counterpart at all.
    fabricated_count = 3
    for j in range(fabricated_count):
        rows.append([f"Ghost {j}", f"{9000 + j}.1"])
    markdown = md_table(["Item", "Value"], rows)
    result = corroborate_rows(words, markdown, REGION)
    assert result.bound == 10  # every real row still binds
    assert result.total == 10 + fabricated_count
    assert result.extra_share is not None
    assert result.extra_share > EXTRA_NUMBERS_MAX_SHARE
    assert result.clears is False


def test_two_table_page_scores_each_region_independently():
    """Two tables on one page: scoping by region isolates each table's own
    native words, so a fabricated second table does not contaminate the
    first table's (perfect) score."""
    table1_region = (0.0, 0.0, 400.0, 100.0)
    table2_region = (0.0, 200.0, 400.0, 400.0)

    table1_words = native_row(10.0, "Revenue", ["1,204"]) + native_row(30.0, "Costs", ["500"])
    table1_markdown = md_table(["Item", "Value"], [["Revenue", "1,204"], ["Costs", "500"]])

    # Table 2's own native words support NEITHER of its candidate rows.
    table2_words = native_row(210.0, "Assets", ["1"]) + native_row(230.0, "Liabilities", ["2"])
    table2_markdown = md_table(["Item", "Value"], [["Assets", "9,999"], ["Liabilities", "8,888"]])

    all_words = table1_words + table2_words

    result1 = corroborate_rows(all_words, table1_markdown, table1_region)
    assert result1.bound == result1.total == 2
    assert result1.clears is True

    result2 = corroborate_rows(all_words, table2_markdown, table2_region)
    assert result2.bound == 0
    assert result2.total == 2
    assert result2.clears is False


def test_all_markdown_table_blocks_handled_not_just_first():
    """Unlike ``parse_grid`` (which returns only the first table block),
    ``corroborate_rows`` aggregates every block found in one markdown string."""
    words = native_row(10.0, "A", ["1"]) + native_row(60.0, "B", ["2"])
    markdown = (
        md_table(["Item", "Value"], [["A", "1"]])
        + "\n\nsome prose between the two tables\n\n"
        + md_table(["Item", "Value"], [["B", "2"]])
    )
    region = (0.0, 0.0, 400.0, 400.0)
    result = corroborate_rows(words, markdown, region)
    assert result.total == 2
    assert result.bound == 2


def test_column_index_legend_row_excluded_not_counted_as_data_row():
    """A printed column-index legend row (non-blank stub '1', values 2..K in
    order) must not be scored as a numeric body row -- see the real ECB
    bulletin p3 qwen fixture, whose candidate carries a bold
    ``| **1** | **2** | ... | **10** |`` row right below the leaf header.
    Its non-blank stub ("1") means the empty-stub exclusion does not catch
    it; the exclusion here is structural (values are exactly 1..K, in
    order), not lexical."""
    words = native_row(10.0, "Revenue", ["1,204", "980"])
    markdown = md_table(
        ["Item", "2023", "2022"],
        [
            ["1", "2", "3"],  # spurious column-index legend row
            ["Revenue", "1,204", "980"],
        ],
    )
    result = corroborate_rows(words, markdown, REGION)
    assert result.total == 1  # the index row does not count
    assert result.bound == 1
    assert result.clears is True


def _narrow_table(rows: list[tuple[str, str, str]]) -> tuple[list[tuple], str]:
    """A 2-numeric-column table (label + two values, lane x0 = 100.0/160.0),
    matching #643's repro shape ("for a 2-3 column table [ROW_SHAPE_MIN] is 2")."""
    words: list[tuple] = []
    md_rows: list[list[str]] = []
    for i, (label, a, b) in enumerate(rows):
        y = 10.0 + i * 20.0
        words += native_row(y, label, [a, b])
        md_rows.append([label, a, b])
    markdown = md_table(["Item", "A", "B"], md_rows)
    return words, markdown


def _footnote_bands(n: int, y_start: float = 500.0) -> list[tuple]:
    """*n* BARE numeric footnote-style lines, LED BY A MARKER (``1) 45 12``,
    no prose at all) -- the issue's own original synthetic repro. #643
    round 4 (owner ruling): a marker-led line with no prose has no property
    -- lexical or geometric -- that distinguishes it from a genuine
    numbered data row, and is deliberately KEPT (not excluded); a page
    combining bare lines like these with a real table stays fail-closed,
    exactly as it did before #643. Only a marker-led line that ALSO carries
    prose (a real cross-reference footnote, e.g. "1) See pages 45 and 12")
    is excluded -- see ``test_aligned_marker_footnotes_do_not_truncate_complete_table``
    in test_structure_check_truncated.py, and ``test_aligned_numeric_footnotes``
    in test_gh643_reviewed_probes.py, for that side of the line."""
    words: list[tuple] = []
    for i in range(n):
        y = y_start + i * 20.0
        base = 500.0 + i * 200.0
        words.append(w(base, y, f"{i + 1})"))
        words.append(w(base + 20.0, y, "45"))
        words.append(w(base + 50.0, y, "12"))
    return words


def test_bare_marker_footnotes_stay_fail_closed_by_design():
    """#643 round 4 (owner ruling): a COMPLETE 3-row narrow-table candidate
    sharing a page with 5 BARE marker-led numeric lines (no prose) is
    correctly REJECTED -- table_shaped_native_row_count counts all 8 bands
    (native_table_rows=8, threshold=ceil(8*36/39)=8, candidate=3 -> fails).
    This is accepted, documented collateral, not a bug: nothing at this
    call site can tell ``1) 45 12`` apart from a genuine numbered data row
    when it carries no prose at all. A real cross-reference footnote always
    carries prose and is excluded instead (see the aligned-prose-footnote
    tests elsewhere in this file and in test_structure_check_truncated.py)."""
    table_words, markdown = _narrow_table(
        [("Revenue", "1,204", "980"), ("Costs", "500", "410"), ("Total", "704", "570")]
    )
    words = table_words + _footnote_bands(5)
    assert table_shaped_native_row_count(words, row_shape_min=2) == 8
    assert _row_shape_reconciliation_ok(words, markdown) is False


def test_truncated_narrow_candidate_also_rejected_with_bare_footnotes_present():
    """The SAME native page as above (3 real rows + 5 bare marker-led
    lines), candidate now emitting only 1 of its 3 real rows: also
    rejected (1 < ceil(8*36/39)=8) -- the bare-footnote collateral does not
    somehow make a genuinely truncated candidate look MORE complete."""
    table_words, _ = _narrow_table(
        [("Revenue", "1,204", "980"), ("Costs", "500", "410"), ("Total", "704", "570")]
    )
    words = table_words + _footnote_bands(5)
    truncated_markdown = md_table(["Item", "A", "B"], [["Revenue", "1,204", "980"]])
    assert table_shaped_native_row_count(words, row_shape_min=2) == 8
    assert _row_shape_reconciliation_ok(words, truncated_markdown) is False


def test_second_table_at_a_different_x_lane_still_counts_when_candidate_covers_only_one():
    """Reviewer addendum: two REAL tables at different x offsets. A
    candidate covering only the first table must still be rejected --
    neither table's rows are marker-led, so #643 round 4's denylist never
    touches either one; both are unconditionally counted regardless of x
    position (no regression on the existing "two independent tables"
    reconciliation case, and no lane geometry involved at all)."""
    table1_words, table1_markdown = _narrow_table(
        [("Revenue", "1,204", "980"), ("Costs", "500", "410"), ("Total", "704", "570")]
    )
    table2_rows = [
        ("Assets", "3,001", "2,900"),
        ("Liab.", "1,500", "1,400"),
        ("Equity", "900", "850"),
    ]
    table2_words = []
    for i, (label, a, b) in enumerate(table2_rows):
        y = 300.0 + i * 20.0
        table2_words += [
            w(10.0, y, label),
            w(300.0, y, a),
            w(360.0, y, b),
        ]
    words = table1_words + table2_words
    assert table_shaped_native_row_count(words, row_shape_min=2) == 6
    assert _row_shape_reconciliation_ok(words, table1_markdown) is False


# ---------------------------------------------------------------------------
# #643 round 3 (reviewed, Astra's second pass): the round-2 allowlist was
# itself an evidence-loss bug -- a marker-led band that positively matched no
# footnote signal was still dropped whenever nothing ELSE on the page shared
# its lane. Round 3 flips to a denylist: every shape-eligible band counts
# UNLESS excluded by explicit prose/no-recurrence evidence. These tests pin
# the two reviewer regressions plus the explicitly requested "genuine second
# numbered table" and "multirow first table beside a one-row second table"
# variants, at a scale (10 rows) large enough that structure_check's own
# stray-header-band allowance cannot swallow the signal (see the mirrored
# tests in test_structure_check_truncated.py for that caller).
# ---------------------------------------------------------------------------


def _numbered_table(
    rows: list[tuple[str, str]], x_marker: float = 10.0, y_start: float = 10.0
) -> tuple[list[tuple], str]:
    """A marker-led, single-numeric-column table: ``1) Alpha | 80``. The
    marker is ordinary table structure (a printed row number), not a
    footnote -- round 3's whole point. *y_start* lets two independent
    tables sit on their own, non-overlapping y-bands (a shared y would
    merge their rows into one band -- an unrelated baseline_bands
    artefact, not the thing under test here)."""
    words: list[tuple] = []
    md_rows: list[list[str]] = []
    for i, (label, value) in enumerate(rows):
        y = y_start + i * 20.0
        words += [
            w(x_marker, y, f"{i + 1})"),
            w(x_marker + 20.0, y, label),
            w(x_marker + 200.0, y, value),
        ]
        md_rows.append([f"{i + 1}) {label}", value])
    markdown = md_table(["Item", "Value"], md_rows)
    return words, markdown


def test_complete_numbered_table_reconciles():
    """A COMPLETE 10-row numbered table must reconcile -- the printed row
    numbers must never be misread as footnote markers stripping every row."""
    rows = [(f"Item{i}", str(80 + i)) for i in range(10)]
    words, markdown = _numbered_table(rows)
    assert table_shaped_native_row_count(words, row_shape_min=1) == 10
    assert _row_shape_reconciliation_ok(words, markdown) is True


def test_truncated_numbered_table_still_rejected():
    """The SAME numbered page, candidate missing its last 3 rows, must still
    be rejected -- round 3's default-keep policy for marker-led rows must
    not also excuse a genuinely dropped row."""
    rows = [(f"Item{i}", str(80 + i)) for i in range(10)]
    words, _ = _numbered_table(rows)
    truncated_markdown = md_table(
        ["Item", "Value"], [[f"{i + 1}) Item{i}", str(80 + i)] for i in range(7)]
    )
    assert table_shaped_native_row_count(words, row_shape_min=1) == 10
    assert _row_shape_reconciliation_ok(words, truncated_markdown) is False


def test_genuine_second_numbered_table_still_counted():
    """Two REAL numbered tables at different x offsets. A candidate covering
    only the first must still be rejected -- each row has exactly ONE
    remaining numeric token after its own marker, so neither exclusion test
    (contiguity is vacuous below 2 tokens; 1 label word never outnumbers 1
    value) fires for either table, marker-led or not, regardless of x
    position."""
    rows1 = [(f"Item{i}", str(80 + i)) for i in range(5)]
    words1, markdown1 = _numbered_table(rows1, x_marker=10.0, y_start=10.0)
    rows2 = [(f"Line{i}", str(500 + i)) for i in range(5)]
    words2, _ = _numbered_table(rows2, x_marker=600.0, y_start=300.0)
    words = words1 + words2
    assert table_shaped_native_row_count(words, row_shape_min=1) == 10
    assert _row_shape_reconciliation_ok(words, markdown1) is False


def test_one_row_second_numbered_table_beside_multirow_first():
    """A one-row second table (still marker-led, ``1) Solo | 999``) sitting
    beside a 10-row first table. #643 round 4: neither exclusion test fires
    (contiguity is vacuous below 2 numeric tokens after the marker, and the
    marker + 1 label word never outnumbers its 2 numeric tokens) -- kept
    regardless of it being the only row of its own table."""
    rows1 = [(f"Item{i}", str(80 + i)) for i in range(10)]
    words1, markdown1 = _numbered_table(rows1, x_marker=10.0)
    words2 = [w(600.0, 500.0, "1)"), w(620.0, 500.0, "Solo"), w(820.0, 500.0, "999")]
    words = words1 + words2
    assert table_shaped_native_row_count(words, row_shape_min=1) == 11
    assert _row_shape_reconciliation_ok(words, markdown1) is False


# ---------------------------------------------------------------------------
# #643 round 4 (reviewed, Astra's third pass): round 3's marker-stripping
# disagreed with the CANDIDATE side's own parser (numeric_body_rows anchors
# a numeric stub like "3)" as a real row token, so a marker-led candidate
# row genuinely has one MORE numeric token than its own data columns) --
# every source row then looked "too narrow" and a truncated candidate
# passed in both callers. Round 3's lane-recurrence exclusion also erased a
# genuinely short (one- or two-row) numbered table, since a short table has
# no "other" band to share a lane with. Round 4 deletes lane recurrence
# entirely and compares candidate/source width LIKE WITH LIKE (marker
# never stripped for the width check). These pin the two additional
# scenarios requested beyond Astra's own probe file
# (test_gh643_reviewed_probes_round4.py): a one-row numbered table and a
# marker-only numeric stub row, both kept.
# ---------------------------------------------------------------------------


def test_one_row_numbered_table_with_two_numerics_kept():
    """A single-row numbered table (marker + label + 2 numeric values) is
    kept -- the absence of a second row is not evidence of anything."""
    words = [
        w(10.0, 20.0, "3)"),
        w(35.0, 20.0, "Alpha"),
        w(130.0, 20.0, "12"),
        w(210.0, 20.0, "45"),
    ]
    assert table_shaped_native_row_count(words, row_shape_min=2) == 1


def test_marker_only_numeric_stub_row_kept():
    """A band whose ONLY genuine numeric token is a row-number marker (no
    data value on that printed line at all -- e.g. a numbered index column
    beside a blank data cell) triggers neither exclusion test (only 1
    numeric token, so contiguity is vacuous; 1 label word does not
    outnumber it) and is kept."""
    words = [w(10.0, 20.0, "3)"), w(35.0, 20.0, "Alpha")]
    assert table_shaped_native_row_count(words, row_shape_min=1) == 1
