"""Tests for row corroboration (``socr.tables.row_corroboration``).

Hermetic: synthetic ``page.get_text("words")``-shaped tuples and literal
markdown strings only. No PDFs, no corpus, no provider. Each test states
the behavioural claim it pins so a body-swapped or logic-stripped module
(imports intact) fails the assertion, not merely that the module is absent.
"""

from __future__ import annotations

from socr.tables.row_corroboration import (
    EXTRA_NUMBERS_MAX_SHARE,
    ROW_CORROBORATION_MIN,
    corroborate_rows,
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
