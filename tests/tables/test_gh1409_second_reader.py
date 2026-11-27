"""#1409: the second-reader comparator. Pure: no PDF, no model, no network."""

from __future__ import annotations

import pytest

from socr.tables.second_reader import (
    DIFF_COUNT,
    DIFF_ROW_NOT_FOUND,
    SIDE_READER,
    SIDE_SHIPPED,
    UNKNOWN_EMPTY,
    UNKNOWN_NO_NUMBERS,
    UNKNOWN_TRUNCATED,
    UNKNOWN_UNCLOSED,
    Agreement,
    compare_readings,
    describe_differences,
    row_numbers,
)


def _reader(body: str) -> str:
    return f"```markdown\n{body}\n```"


def _table(*rows: tuple[str, ...]) -> str:
    cells = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table>{cells}</table>"


# --- normalisation ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "written",
    ["-.5", "-0.5", "‐.5", "‑.5", "‒.5", "–.5", "—.5", "−.5"],
)
def test_minus_variants_before_a_number_are_one_sign(written):
    assert row_numbers(f"GDP {written} 2.5") == ("-0.5", "2.5")


def test_a_dash_between_numbers_or_after_a_word_is_a_separator():
    assert row_numbers("1.7–1.8 and 5-8 and Apr-09") == ("1.7", "1.8", "5", "8", "09")


def test_a_dash_set_apart_from_its_number_is_not_joined():
    assert row_numbers("GDP ‐ .5") == ("0.5",)


def test_tilde_before_a_number_is_never_a_minus():
    assert row_numbers("~.5") == ("~0.5",)
    assert row_numbers("~.5") != row_numbers("-.5")


def test_thousands_commas_and_leading_zero():
    assert row_numbers("1,234.5 and .5 and 0.5") == ("1234.5", "0.5", "0.5")


def test_stated_precision_is_kept():
    assert row_numbers("2.10") != row_numbers("2.1")


def test_a_dropped_point_stays_a_difference():
    assert row_numbers("-.7") != row_numbers("-7")


@pytest.mark.parametrize(
    "written",
    [
        "2 $\\frac{1}{2}$ percent",
        "2 \\tfrac{1}{2} percent",
        "2 <sup>1</sup>/<sub>2</sub> percent",
        "2 <sup>1</sup>/2 percent",
        "2 ½ percent",
        "2½ percent",
        "2 1/2 percent",
        "2 ¹⁄₂ percent",
    ],
)
def test_fraction_notations_are_one_token(written):
    assert row_numbers(written) == ("2", "1/2")


def test_a_date_and_a_ratio_of_words_are_not_fractions():
    assert row_numbers("6/10/2015 and Q4/Q4") == ("6", "10", "2015", "4", "4")


def test_own_brackets_are_kept_and_other_brackets_are_not():
    assert row_numbers("(0.4) and (line 2) and 0.4") == ("(0.4)", "2", "0.4")


def test_superscript_callout_is_its_own_number():
    assert row_numbers("2.5¹") == ("2.5", "1")
    assert row_numbers("2.5<sup>1</sup>") == ("2.5", "1")


def test_full_stop_run_into_a_callout_is_not_a_decimal_point():
    assert row_numbers("in the baseline.13 In response") == ("13",)
    assert row_numbers("rose 2 percent).4 and") == ("2", "4")
    assert row_numbers("GDP .5 2.5") == ("0.5", "2.5")


def test_numbers_glued_into_words_are_found():
    assert row_numbers("loweringthegrowthrateby1.2percent") == ("1.2",)


def test_image_paths_and_comments_hold_no_numbers():
    assert row_numbers("![chart region 1](figures/chart_region_p63_1.png) <!-- 7 -->") == ()


# --- agreement --------------------------------------------------------------------------

SHIPPED_TABLE = (
    "Estimated effects\n\n2015:Q1 2015:Q2\n\n"
    "1. Change in real GDP -.5 2.5 Contribution from special factors:\n\n"
    "2. Residual seasonality -.8 .6 3. Weather -.2 .2\n\nPage 10 of 96"
)
READER_TABLE = _reader(
    "<page_header>Authorized for Public Release</page_header>\n\nEstimated effects\n"
    + _table(
        ("", "2015:Q1", "2015:Q2"),
        ("1. Change in real GDP", "−.5", "2.5"),
        ("Contribution from special factors:", "", ""),
        ("2. Residual seasonality", "-.8", ".6"),
        ("3. Weather", "-.2", ".2"),
    )
    + "\n\n<page_footer>Page <page_number>10</page_number> of 96</page_footer>"
)


def test_flat_shipped_lines_agree_with_the_readers_table_rows():
    verdict = compare_readings(SHIPPED_TABLE, READER_TABLE)
    assert verdict.outcome is Agreement.AGREE, verdict.differences
    assert verdict.differences == ()


def test_lost_minus_sign_disagrees():
    """#1411: the layer set its minus signs on lines of their own (June 2015 Tealbook A p13)."""
    shipped = SHIPPED_TABLE.replace("-.5", ".5").replace("-.8", ".8") + "\n\n‐\n\n‐"
    verdict = compare_readings(shipped, READER_TABLE)
    assert verdict.outcome is Agreement.DISAGREE
    assert any(
        d.side == SIDE_SHIPPED and d.kind == DIFF_ROW_NOT_FOUND and d.numbers == ("1", "0.5", "2.5")
        for d in verdict.differences
    )
    assert any(d.side == SIDE_READER and "-0.5" in d.numbers for d in verdict.differences)


def test_hyphen_u2010_minus_in_shipped_text_agrees_with_ascii_minus():
    shipped = SHIPPED_TABLE.replace("-.", "‐.")
    assert compare_readings(shipped, READER_TABLE).outcome is Agreement.AGREE


def test_tilde_where_the_page_prints_a_minus_disagrees():
    reader = READER_TABLE.replace("<td>-.8</td>", "<td>~.8</td>")
    verdict = compare_readings(SHIPPED_TABLE, reader)
    assert verdict.outcome is Agreement.DISAGREE
    assert any("~0.8" in d.numbers for d in verdict.differences)


def test_broken_fraction_disagrees():
    """#1412: a stacked fraction shipped as lone digits."""
    shipped = "growth to slow to around 1 1 2% to 2% (annual rate)\n\n4 2"
    reader = _reader("growth to slow to around 1 $\\frac{1}{2}$% to 2% (annual rate)")
    verdict = compare_readings(shipped, reader)
    assert verdict.outcome is Agreement.DISAGREE
    assert any(d.numbers == ("1", "1", "2", "2") for d in verdict.differences)


def test_fraction_markup_alone_agrees():
    shipped = "growth of about 2 1/2 percent and ¾ point"
    reader = _reader("growth of about 2 <sup>1</sup>/2 percent and $\\frac{3}{4}$ point")
    assert compare_readings(shipped, reader).outcome is Agreement.AGREE


def test_doubled_fraction_disagrees():
    shipped = "Potential GDP growth is 2 ½ ½ percent."
    reader = _reader("Potential GDP growth is 2 $\\frac{1}{2}$ percent.")
    verdict = compare_readings(shipped, reader)
    assert verdict.outcome is Agreement.DISAGREE
    assert any(d.kind == DIFF_COUNT and d.text == "1/2" for d in verdict.differences)


def test_shipped_only_hidden_text_number_disagrees():
    """A timestamp the PDF hides: the image does not show it, the shipped text does."""
    shipped = "29 Oct 2009 3:56 PM\n\nApr-09 -7.7 May-09 10.2\n\nOctober 29, 2009 Page 64 of 77"
    reader = _reader(
        "<page_header>October 29, 2009 Page <page_number>64</page_number> of 77</page_header>\n"
        + _table(("Apr-09", "-7.7"), ("May-09", "10.2"))
    )
    verdict = compare_readings(shipped, reader)
    assert verdict.outcome is Agreement.DISAGREE
    rows = [d for d in verdict.differences if d.kind == DIFF_ROW_NOT_FOUND]
    assert [d.numbers for d in rows] == [("29", "2009", "3", "56")]
    assert rows[0].side == SIDE_SHIPPED


def test_reader_furniture_accounts_for_a_shipped_line_and_is_not_required():
    reader = _reader(
        "<page_header>June 10, 2015</page_header>\n\nGrowth was 2.5 percent.\n\n"
        "<page_footer>Page <page_number>10</page_number> of 96</page_footer>"
    )
    assert compare_readings("Growth was 2.5 percent.", reader).outcome is Agreement.AGREE
    with_furniture = "June 10, 2015\n\nGrowth was 2.5 percent.\n\nPage 10 of 96"
    assert compare_readings(with_furniture, reader).outcome is Agreement.AGREE


def test_value_moved_between_rows_disagrees():
    shipped = "| A | 1.1 | 3.3 |\n| B | 2.2 | 4.4 |"
    reader = _reader(_table(("A", "1.1", "2.2"), ("B", "3.3", "4.4")))
    verdict = compare_readings(shipped, reader)
    assert verdict.outcome is Agreement.DISAGREE
    assert {d.side for d in verdict.differences} == {SIDE_SHIPPED, SIDE_READER}
    assert all(d.kind == DIFF_ROW_NOT_FOUND for d in verdict.differences)


def test_value_moved_across_a_table_row_end_disagrees():
    """A flat shipped line may hold whole table rows, never a row and part of the next."""
    shipped = "A 1.1 2.2 3.3\n\nB 4.4"
    reader = _reader(_table(("A", "1.1", "2.2"), ("B", "3.3", "4.4")))
    verdict = compare_readings(shipped, reader)
    assert verdict.outcome is Agreement.DISAGREE
    assert any(
        d.side == SIDE_SHIPPED and d.numbers == ("1.1", "2.2", "3.3") for d in verdict.differences
    )


def test_running_text_wrapped_differently_agrees():
    shipped = "rose 2.9 percent in 2022 and\n1.8 percent in 2023. 3 See the\nMay 4 note."
    reader = _reader("rose 2.9 percent in 2022 and 1.8 percent in 2023.\n\n3 See the May 4 note.")
    assert compare_readings(shipped, reader).outcome is Agreement.AGREE


def test_paragraph_split_around_another_block_agrees():
    shipped = "would be 1.3 to 4.7 percent\n\nOther block 70 percent\n\nTable 2 shows 3 ranges of"
    reader = _reader(
        "Table 2 shows 3 ranges of would be 1.3 to 4.7 percent\n\nOther block 70 percent"
    )
    assert compare_readings(shipped, reader).outcome is Agreement.AGREE


def test_row_shipped_twice_disagrees_on_counts():
    shipped = "A 1.1 2.2\n\nA 1.1 2.2\n\nB 3.3 4.4"
    reader = _reader(_table(("A", "1.1", "2.2"), ("B", "3.3", "4.4")))
    verdict = compare_readings(shipped, reader)
    assert verdict.outcome is Agreement.DISAGREE
    assert {d.kind for d in verdict.differences} == {DIFF_COUNT}
    assert all(d.side == SIDE_SHIPPED and d.shipped_count == 2 for d in verdict.differences)


def test_reader_row_the_shipped_text_dropped_disagrees():
    reader = _reader(_table(("A", "1.1", "2.2"), ("B", "3.3", "4.4")))
    verdict = compare_readings("A 1.1 2.2", reader)
    assert verdict.outcome is Agreement.DISAGREE
    assert any(d.side == SIDE_READER and d.numbers == ("3.3", "4.4") for d in verdict.differences)


# --- charts -----------------------------------------------------------------------------

CHART_SHIPPED = (
    "Growth was 2.5 percent in 2019.\n\n![chart region 1](figures/chart_region_p5_1.png)\n\n"
    "- **Y-axis ticks:** 0, 1, 2, 3\n\nSource: staff, 2019."
)
CHART_READER = _reader(
    "Growth was 2.5 percent in 2019.\n\n### Growth — Percent, 2017 to 2019\n"
    + _table(("2017", "1.37"), ("2018", "2.91"), ("2019", "2.48"))
    + "\n\nSource: staff, 2019."
)


def test_chart_only_reader_table_does_not_mark_a_page_that_ships_the_picture():
    verdict = compare_readings(CHART_SHIPPED, CHART_READER, chart_label_text="0 1 2 3 2017")
    assert verdict.outcome is Agreement.AGREE, verdict.differences
    assert verdict.chart_fallback
    assert len(verdict.excluded_reader_rows) == 4  # three rows and the line above the table
    assert verdict.excluded_shipped_rows == ("- **Y-axis ticks:** 0, 1, 2, 3",)


def test_chart_only_reader_table_marks_a_page_that_ships_no_picture():
    shipped = CHART_SHIPPED.replace("![chart region 1](figures/chart_region_p5_1.png)", "")
    verdict = compare_readings(shipped, CHART_READER, chart_label_text="0 1 2 3 2017")
    assert verdict.outcome is Agreement.DISAGREE
    assert not verdict.chart_fallback


def test_masked_image_turns_the_fallback_off():
    verdict = compare_readings(
        CHART_SHIPPED, CHART_READER, chart_picture=False, chart_label_text="0 1 2 3 2017"
    )
    assert verdict.outcome is Agreement.DISAGREE
    assert verdict.excluded_reader_rows == ()


def test_shipped_tick_row_without_layer_labels_disagrees():
    verdict = compare_readings(CHART_SHIPPED, CHART_READER)
    assert verdict.outcome is Agreement.DISAGREE
    assert [d.numbers for d in verdict.differences if d.kind == DIFF_ROW_NOT_FOUND] == [
        ("0", "1", "2", "3")
    ]


def test_wrong_shipped_number_on_a_chart_page_still_disagrees():
    shipped = CHART_SHIPPED.replace("2.5 percent", "25 percent")
    verdict = compare_readings(shipped, CHART_READER, chart_label_text="0 1 2 3 2017")
    assert verdict.outcome is Agreement.DISAGREE
    assert any(d.side == SIDE_SHIPPED and "25" in d.numbers for d in verdict.differences)


def test_chart_labels_do_not_excuse_a_row_with_a_number_that_is_no_label():
    shipped = CHART_SHIPPED.replace("0, 1, 2, 3", "0, 1, 2, 3, 99")
    verdict = compare_readings(shipped, CHART_READER, chart_label_text="0 1 2 3 2017")
    assert verdict.outcome is Agreement.DISAGREE


# --- unknown ----------------------------------------------------------------------------


@pytest.mark.parametrize("reader", ["", "   \n", None])
def test_empty_reader_output_is_unknown(reader):
    verdict = compare_readings("Growth was 2.5 percent.", reader)
    assert verdict.outcome is Agreement.UNKNOWN
    assert verdict.unknown_reason == UNKNOWN_EMPTY


def test_reader_output_the_caller_knows_was_cut_off_is_unknown():
    verdict = compare_readings("Growth 2.5", _reader("Growth 2.5"), reader_truncated=True)
    assert (verdict.outcome, verdict.unknown_reason) == (Agreement.UNKNOWN, UNKNOWN_TRUNCATED)


def test_open_code_fence_is_unknown():
    verdict = compare_readings("Growth 2.5", "```markdown\nGrowth 2.5")
    assert (verdict.outcome, verdict.unknown_reason) == (Agreement.UNKNOWN, UNKNOWN_TRUNCATED)


@pytest.mark.parametrize(
    "reader",
    [
        "Growth\n<table><tr><td>A</td><td>1.1</td></tr><tr><td>B</td><td>3.",
        "<table><tr><td>A</td><td>1.1</td></tr>",
        "<page_header>June 10, 2015\n\nGrowth 2.5",
    ],
)
def test_unclosed_block_is_unknown(reader):
    verdict = compare_readings("A 1.1\n\nB 3.3", reader)
    assert (verdict.outcome, verdict.unknown_reason) == (Agreement.UNKNOWN, UNKNOWN_UNCLOSED)


def test_no_numbers_on_either_side_is_unknown():
    verdict = compare_readings("Only words.", _reader("Only words."))
    assert (verdict.outcome, verdict.unknown_reason) == (Agreement.UNKNOWN, UNKNOWN_NO_NUMBERS)


def test_reader_without_numbers_against_a_page_with_numbers_disagrees():
    assert compare_readings("Growth 2.5", _reader("Only words.")).outcome is Agreement.DISAGREE


# --- record -----------------------------------------------------------------------------


def test_socr_own_lines_are_not_compared():
    shipped = (
        "> [page 4 unchecked: table 2 could not be checked]\n\n"
        "Growth was 2.5 percent. [socr: removed 3 control bytes]"
    )
    assert compare_readings(shipped, _reader("Growth was 2.5 percent.")).outcome is Agreement.AGREE


def test_every_difference_is_named_in_the_reason():
    shipped = SHIPPED_TABLE.replace("-.5", ".5")
    verdict = compare_readings(shipped, READER_TABLE)
    reason = describe_differences(verdict)
    assert reason.count(" / ") == len(verdict.differences) - 1
    assert "shipped row with the numbers 1 0.5 2.5 is not in the second reading" in reason
    assert "[" not in reason and "]" not in reason
    assert ";" not in reason


def test_verdict_round_trips_to_a_dict():
    record = compare_readings(SHIPPED_TABLE.replace("-.5", ".5"), READER_TABLE).to_dict()
    assert record["outcome"] == "disagree"
    assert record["differences"][0]["kind"] in (DIFF_ROW_NOT_FOUND, DIFF_COUNT)
    assert record["normalisation"]
