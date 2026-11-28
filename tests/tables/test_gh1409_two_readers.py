"""#1409 step 3: the two-reader agreement test. Pure: no PDF, no model, no network.

Every condition is tested the same way: a pair of readings that agrees, and the same pair
with ONE thing changed, which must not.
"""

from __future__ import annotations

import pytest

from socr.tables import two_readers as rule
from socr.tables.two_readers import compare_two_readings, word_tokens

TABLE_A = """<table>
  <thead>
    <tr>
      <th></th>
      <th colspan="2">1997</th>
      <th>1998</th>
    </tr>
    <tr>
      <th>Item</th>
      <th>Q3</th>
      <th>Q4</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Exports</td>
      <td>0.8</td>
      <td>-1.3</td>
      <td>-0.7</td>
    </tr>
    <tr>
      <td>Imports</td>
      <td>0.2</td>
      <td>-4.9</td>
      <td>-1.9</td>
    </tr>
  </tbody>
</table>"""

TABLE_B = (
    '<table><tr><th></th><th colspan="2"><strong>1997</strong></th><th>1998</th></tr>'
    "<tr><th>Item</th><th>Q3</th><th>Q4</th><th></th></tr>"
    "<tr><td>Exports</td><td>0.8</td><td>−1.3</td><td>-0.7</td></tr>"
    "<tr><td>Imports</td><td>0.2</td><td>-4.9</td><td>-1.9</td></tr></table>"
)

#: Reader A writes page furniture as lines and tables as indented HTML.
A = f"""11/13/90

-2-

MR. KOHN. Inflation reaches 2.9 percent by the end of the year, com-
pared with 1 ½ percent in the baseline.

### Selected Trade Prices

{TABLE_A}

NOTE. Prices are on a chain-weighted basis.

Page 10 of 96
"""

#: Reader B tags furniture, fences its output and writes tables on one line.
B = f"""```markdown
<page_header>11/13/90 -2-</page_header>

MR. KOHN. Inflation reaches 2.9 percent by the end of the year,
compared with 1 $\\frac{{1}}{{2}}$ percent in the baseline.

**Selected Trade Prices**
{TABLE_B}

NOTE. Prices are on a chain-weighted basis.

<page_footer>Page <page_number>10</page_number> of 96</page_footer>
```"""


def _reasons(a: str, b: str, **kw) -> tuple[str, ...]:
    return compare_two_readings(a, b, **kw).reasons()


def test_two_readings_that_differ_only_in_markup_agree():
    verdict = compare_two_readings(A, B, finish_reason_a="stop", finish_reason_b="stop")
    assert verdict.agree, verdict.failures
    assert verdict.failures == ()
    assert (verdict.tables, verdict.furniture_blocks, verdict.same_grids) == (1, 2, True)
    assert verdict.number_rows > 0 and verdict.words > 0


def test_agreement_does_not_depend_on_which_reading_is_called_a():
    assert compare_two_readings(B, A).agree


# --- 1. usable --------------------------------------------------------------------------


@pytest.mark.parametrize("empty", ["", "  \n", None])
def test_an_empty_reading_does_not_agree(empty):
    assert _reasons(A, empty) == (rule.UNUSABLE_EMPTY,)
    assert _reasons(empty, B) == (rule.UNUSABLE_EMPTY,)


def test_a_cut_off_reading_does_not_agree():
    assert compare_two_readings(A, B, finish_reason_b="stop").agree
    assert _reasons(A, B, finish_reason_b="length") == (rule.UNUSABLE_CUT_OFF,)
    assert _reasons(A, B, finish_reason_a="length") == (rule.UNUSABLE_CUT_OFF,)


def test_an_open_code_fence_does_not_agree():
    assert _reasons(A, B.removesuffix("```")) == (rule.UNUSABLE_CUT_OFF,)


def test_an_unterminated_table_does_not_agree():
    cut = A[: A.index("</tbody>")]
    assert _reasons(cut, B) == (rule.UNUSABLE_UNCLOSED,)
    assert compare_two_readings(cut, B).failures[0].side == rule.SIDE_A


# --- 2. furniture -----------------------------------------------------------------------


def test_a_tagged_head_is_the_same_words_on_two_lines_of_the_other_reading():
    one_line = A.replace("11/13/90\n\n-2-", "11/13/90 -2-")
    assert compare_two_readings(one_line, B).agree


def test_furniture_one_reader_tagged_and_the_other_lacks_does_not_agree():
    """Nothing else differs: the bodies are equal once B's tagged head is cut. Only this
    rule keeps a page number one reader saw from being silently absent from what ships."""
    without = A.replace("11/13/90\n\n-2-\n\n", "")
    assert _reasons(without, B) == (rule.FURNITURE_UNMATCHED,)


def test_furniture_with_another_page_number_does_not_agree():
    assert rule.FURNITURE_UNMATCHED in _reasons(A.replace("-2-", "-3-"), B)


def test_a_line_is_furniture_only_when_a_reader_tagged_it():
    """The same lines untagged in both readings are body, and are compared as body."""
    untagged = B.replace("<page_header>11/13/90 -2-</page_header>", "11/13/90 -2-")
    assert compare_two_readings(A, untagged).agree
    assert rule.NUMBERS_DIFFER in _reasons(A.replace("-2-", "-3-"), untagged)


def test_a_body_line_equal_to_a_furniture_block_is_not_cut_from_the_middle():
    """Only lines at the top or bottom of the other reading can be a tagged block."""
    moved = A.replace("11/13/90\n\n-2-\n\n", "").replace(
        "### Selected", "11/13/90\n\n-2-\n\n### Selected"
    )
    assert rule.FURNITURE_UNMATCHED in _reasons(moved, B)


# --- 3. numbers -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "old, new",
    [
        ("<td>-1.3</td>", "<td>1.3</td>"),  # a lost minus
        ("<td>0.8</td>", "<td>8</td>"),  # a dropped point
        ("2.9 percent", "2.8 percent"),  # a digit in running text
        ("1 ½ percent", "1 1 2 percent"),  # a broken fraction
    ],
)
def test_one_different_number_does_not_agree(old, new):
    assert old in A
    assert rule.NUMBERS_DIFFER in _reasons(A.replace(old, new), B)


def test_a_lost_minus_is_seen_by_the_number_rule_alone():
    """The word rule folds dashes, so numbers and cell grid numbers tell ``-1.3`` from ``1.3``."""
    reasons = _reasons(A.replace("<td>-1.3</td>", "<td>1.3</td>"), B)
    assert rule.NUMBERS_DIFFER in reasons
    assert rule.TABLE_GRIDS_DIFFER in reasons


def test_cross_row_sign_swap_does_not_agree():
    """Swapping a minus sign between two rows keeps the row-key set and word grid identical,
    so per-cell signed numbers in ``_grid`` must catch it."""
    a = "<table><tr><td>Exports</td><td>-1.0</td></tr><tr><td>Imports</td><td>1.0</td></tr></table>"
    b = "<table><tr><td>Exports</td><td>1.0</td></tr><tr><td>Imports</td><td>-1.0</td></tr></table>"
    verdict = compare_two_readings(a, b)
    assert not verdict.agree
    assert rule.TABLE_GRIDS_DIFFER in verdict.reasons()


def test_cross_row_bracket_swap_does_not_agree():
    """Swapping accounting/SE brackets ``(0.4)`` vs ``0.4`` between two rows must not agree."""
    a = (
        "<table><tr><td>Exports</td><td>(0.4)</td></tr>"
        "<tr><td>Imports</td><td>0.4</td></tr></table>"
    )
    b = (
        "<table><tr><td>Exports</td><td>0.4</td></tr>"
        "<tr><td>Imports</td><td>(0.4)</td></tr></table>"
    )
    verdict = compare_two_readings(a, b)
    assert not verdict.agree
    assert rule.TABLE_GRIDS_DIFFER in verdict.reasons()


def test_inequalities_and_non_html_angle_tokens_are_not_deleted():
    """Allowlisted HTML stripping before ``html.unescape`` keeps inequalities and ``<PR>``."""
    assert word_tokens("p < 0.05 and z > 1.96") == ("p", "0.05", "and", "z", "1.96")
    assert word_tokens("p &lt; 0.05 and z &gt; 1.96") == ("p", "0.05", "and", "z", "1.96")
    assert word_tokens("Table <1> series <PR> grew") == ("table", "1", "series", "pr", "grew")
    assert compare_two_readings(
        "Threshold p < 0.05 and z > 1.96", "Threshold p &lt; 0.05 and z &gt; 1.96"
    ).agree
    assert not compare_two_readings(
        "Threshold p < 0.05 and z > 1.96", "Threshold p &lt; 0.50 and z &gt; 1.96"
    ).agree


def test_a_table_row_only_one_reader_wrote_does_not_agree():
    extra = "    <tr>\n      <td>Oil</td>\n      <td>18.00</td>\n    </tr>\n  </tbody>"
    reasons = _reasons(A.replace("  </tbody>", extra), B)
    assert rule.NUMBERS_DIFFER in reasons


def test_a_table_row_written_twice_is_counted():
    """Every row is found in the other reading; only the counts say one was written twice."""
    cells = "".join(f"      <td>{c}</td>\n" for c in ("Imports", "0.2", "-4.9", "-1.9"))
    row = f"    <tr>\n{cells}    </tr>\n"
    assert A.count(row) == 1
    failures = compare_two_readings(A.replace(row, row + row), B).failures
    counted = [f for f in failures if f.reason == rule.NUMBERS_DIFFER]
    assert [f.detail for f in counted] == [
        "-1.9 occurs 2 time(s) in a and 1 in b",
        "-4.9 occurs 2 time(s) in a and 1 in b",
        "0.2 occurs 2 time(s) in a and 1 in b",
    ]


def test_a_value_moved_to_the_next_table_row_does_not_agree():
    moved = TABLE_A.replace(
        "      <td>-0.7</td>\n    </tr>\n    <tr>\n      <td>Imports</td>",
        "    </tr>\n    <tr>\n      <td>Imports</td>\n      <td>-0.7</td>",
    )
    assert moved != TABLE_A
    assert rule.NUMBERS_DIFFER in _reasons(A.replace(TABLE_A, moved), B)


# --- 4. words ---------------------------------------------------------------------------


def test_one_different_word_does_not_agree():
    reasons = _reasons(A.replace("Inflation", "Deflation"), B)
    assert reasons == (rule.WORDS_DIFFER, rule.WORD_ORDER_DIFFERS)


def test_the_same_words_in_another_order_outside_tables_do_not_agree():
    swapped = A.replace("chain-weighted basis", "basis chain-weighted")
    assert _reasons(swapped, B) == (rule.WORD_ORDER_DIFFERS,)


def test_a_dropped_paragraph_does_not_agree():
    assert rule.WORDS_DIFFER in _reasons(
        A.replace("NOTE. Prices are on a chain-weighted basis.", ""), B
    )


@pytest.mark.parametrize(
    "written, tokens",
    [
        ("com-\npared &amp; Co.", ("compared", "co")),
        ("long-term—rates", ("long-term", "rates")),
        ("1 $\\frac{1}{2}$\\%", ("1", "1/2")),
        ("GDP<sup>1</sup> x^{2} y³", ("gdp", "1", "x", "2", "y", "3")),
        ("**Bold** _it_ # H | c > q [x]", ("bold", "it", "h", "c", "q", "x")),
        ("Don’t 18.50/b 1,234.5", ("don't", "18.50/b", "1,234.5")),
        ("Consumer attitudes . . . . . 1", ("consumer", "attitudes", "1")),
        ("![image](image_1.png) text", ("text",)),
        ("```markdown\nA\n```", ("a",)),
    ],
)
def test_word_normalisation(written, tokens):
    assert word_tokens(written) == tokens


# --- 5. what a reader is known to get wrong ---------------------------------------------


def test_a_tilde_before_a_digit_does_not_agree_even_when_both_readers_wrote_it():
    a, b = A.replace("-1.3", "~1.3"), B.replace("−1.3", "~1.3")
    reasons = _reasons(a, b)
    assert reasons == (rule.TILDE_BEFORE_DIGIT,)
    assert {f.side for f in compare_two_readings(a, b).failures} == {rule.SIDE_A, rule.SIDE_B}


@pytest.mark.parametrize("picture", ["![image](image_1.png)", '<img src="chart.png">'])
def test_an_image_placeholder_does_not_agree(picture):
    assert _reasons(A + "\n" + picture + "\n", B) == (rule.IMAGE_PLACEHOLDER,)


def test_a_table_only_one_reader_wrote_does_not_agree():
    """The reader that tabulates a chart: its table has no counterpart."""
    invented = B.replace("NOTE.", "<table><tr><td>1997</td><td>1.37</td></tr></table>\n\nNOTE.")
    assert rule.TABLE_WITHOUT_COUNTERPART in _reasons(A, invented)


def test_an_empty_table_only_one_reader_wrote_does_not_agree():
    """No word and no number differs: only the count of tables does."""
    empty = B.replace("NOTE.", "<table><tr><td></td></tr></table>\n\nNOTE.")
    assert _reasons(A, empty) == (rule.TABLE_WITHOUT_COUNTERPART,)


def test_a_table_written_as_text_by_the_other_reader_does_not_agree():
    flat = "\n\n".join(
        ["1997 1998", "Item Q3 Q4", "Exports 0.8 -1.3 -0.7", "Imports 0.2 -4.9 -1.9"]
    )
    assert rule.TABLE_WITHOUT_COUNTERPART in _reasons(A.replace(TABLE_A, flat), B)


# --- 6. tables cell for cell ------------------------------------------------------------


def test_a_header_one_column_off_for_want_of_a_colspan_does_not_agree():
    """Same rows, same numbers, same words: only the grid shows 1998 over the wrong column."""
    no_span = A.replace('<th colspan="2">1997</th>', "<th>1997</th>")
    assert _reasons(no_span, B) == (rule.TABLE_GRIDS_DIFFER,)


def test_two_rows_merged_into_one_cell_do_not_agree():
    merged = A.replace(
        "      <td>Exports</td>\n      <td>0.8</td>\n      <td>-1.3</td>\n      <td>-0.7</td>\n"
        "    </tr>\n    <tr>\n      <td>Imports</td>",
        "      <td>Exports 0.8 -1.3 -0.7 Imports</td>",
    )
    assert merged != A
    assert _reasons(merged, B) == (rule.TABLE_GRIDS_DIFFER,)


def test_trailing_empty_cells_are_not_a_difference():
    padded = B.replace("<td>-1.9</td></tr>", "<td>-1.9</td><td></td></tr>")
    assert compare_two_readings(A, padded).agree


# --- record -----------------------------------------------------------------------------


def test_the_verdict_names_every_failure_and_round_trips_to_a_dict():
    verdict = compare_two_readings(A.replace("-1.3", "1.3").replace("Inflation", "Deflation"), B)
    record = verdict.to_dict()
    assert not verdict.agree and record["agree"] is False
    assert (record["rule"], record["rule_version"]) == (rule.RULE, rule.RULE_VERSION)
    assert {f["reason"] for f in record["failures"]} == set(verdict.reasons())
    assert rule.NUMBERS_DIFFER in verdict.reasons() and rule.WORDS_DIFFER in verdict.reasons()
    assert all(f["detail"] for f in record["failures"])
