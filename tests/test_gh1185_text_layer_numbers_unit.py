"""#1185: a model-read page is CHECKED only if its numbers occur in the page's own text layer.

Pins the comparator (``tables.reading_agreement``), the clause that turns a disagreement
into a mark (``verdict.text_layer_numbers_clause``) and the seam that demotes the page
(``manifest._apply_verdict_marks``), including the resume permanence of the demotion.
The motivating case is 2018 Tealbook p125, where a reading shipped ``-8`` for a printed
``-.8`` and the page judge called it faithful.
"""

from __future__ import annotations

from pathlib import Path

import socr
from socr.core.marks import Check, CheckResult
from socr.core.manifest import _apply_verdict_marks
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import PageState
from socr.core.verdict import (
    DAMAGED_LAYER_PREFIX,
    NO_NUMBER_REASON,
    NO_OWN_TEXT_REASON,
    text_layer_numbers_clause,
)
from socr.pipeline.orchestrator import _resume_verdict_is_permanently_unverifiable
from socr.tables.reading_agreement import (
    PROVENANCE_OWN_CHARACTERS,
    compare_numbers,
    describe_disputes,
    numeric_tokens,
)

import pytest

TEALBOOK_LAYER = "Korea\n4.0\n2.6\n5.7\n-.8\n4.0\n3.3"
TEALBOOK_READING = "| Korea | 4.0 | 2.6 | 5.7 | -8 | 4.0 | 3.3 |"


def test_loaded_source_is_this_checkout() -> None:
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def _page(layer: str = TEALBOOK_LAYER, *, born_digital: bool = True) -> PageState:
    p = PageState(page_num=1)
    p.own_layer_text = layer
    p.is_born_digital = born_digital
    return p


def _qwen(text: str) -> PageOutput:
    return PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, engine="qwen")


# --- comparator -------------------------------------------------------------------------


def test_tealbook_minus_dropped_is_disputed_and_named() -> None:
    """A reading that ships -8 for a layer's -.8 is disputed, with the counterpart named."""
    record = compare_numbers(TEALBOOK_LAYER, TEALBOOK_READING, provenance=PROVENANCE_OWN_CHARACTERS)
    assert record.disputed == (("-.8", "-8"),)
    assert record.agreed is False
    assert describe_disputes(record) == "text layer says -.8, reading says -8"


def test_faithful_minus_point_eight_agrees() -> None:
    """Shipping -.8 or -0.8 for a layer's -.8 is agreement: the two spellings are one value."""
    for reading in (
        "| Korea | 4.0 | 2.6 | 5.7 | -.8 | 4.0 | 3.3 |",
        "| Korea | 4.0 | 2.6 | 5.7 | -0.8 | 4.0 | 3.3 |",
    ):
        record = compare_numbers(TEALBOOK_LAYER, reading, provenance=PROVENANCE_OWN_CHARACTERS)
        assert record.disputed == ()
        assert record.agreed is True


def test_reading_order_does_not_create_disputes() -> None:
    """The same multiset in a different stream order is not a dispute."""
    record = compare_numbers(
        "a 1.5 b 2.5\nc 3.5", "c 3.5 a 1.5 b 2.5", provenance=PROVENANCE_OWN_CHARACTERS
    )
    assert record.disputed == ()


def test_one_inserted_number_is_the_only_dispute() -> None:
    """One extra number in the reading is disputed alone; the numbers after it do not cascade."""
    record = compare_numbers(
        "1.1 2.2 3.3 4.4 5.5",
        "1.1 2.2 9.9 3.3 4.4 5.5",
        provenance=PROVENANCE_OWN_CHARACTERS,
    )
    assert record.disputed == ((None, "9.9"),)


def test_shipping_a_number_more_often_than_the_layer_is_one_dispute() -> None:
    """A surplus copy of a number the layer prints once is disputed exactly once."""
    record = compare_numbers("2.1 3.0", "2.1 2.1 3.0", provenance=PROVENANCE_OWN_CHARACTERS)
    assert len(record.disputed) == 1


def test_no_numbers_shipped_is_not_agreement() -> None:
    """Agreement over nothing is vacuous: compared == 0 must not read as agreed."""
    record = compare_numbers("1.5", "no numbers here", provenance=PROVENANCE_OWN_CHARACTERS)
    assert record.agreed is False
    assert record.compared == 0


def test_table_bracket_is_part_of_the_token() -> None:
    """In a table row, (0.014) is a standard error and does not match a bare 0.014."""
    record = compare_numbers("0.014", "| x | (0.014) |", provenance=PROVENANCE_OWN_CHARACTERS)
    assert record.disputed


def test_running_text_bracket_only_difference_is_matched() -> None:
    """In running text, a citation (2017) matches a layer's bare 2017."""
    record = compare_numbers(
        "see Kilian 2017 here", "see Kilian (2017) here", provenance=PROVENANCE_OWN_CHARACTERS
    )
    assert record.disputed == ()


def test_running_text_year_with_run_on_callout_matches() -> None:
    """A year whose footnote callout the layer ran on (2009.9 for 2009.) is not a dispute."""
    record = compare_numbers(
        "half of 2009.9 We drop", "half of 2009. We drop", provenance=PROVENANCE_OWN_CHARACTERS
    )
    assert record.disputed == ()


def test_dropped_minus_is_caught_when_the_layer_detaches_it() -> None:
    """A minus the layer sets apart with a space is joined, so dropping it is disputed."""
    record = compare_numbers(
        "Constant 0.001 − 0.077",
        "| Constant | 0.001 | 0.077 |",
        provenance=PROVENANCE_OWN_CHARACTERS,
    )
    assert record.disputed == (("−0.077", "0.077"),)


def test_a_spaced_ascii_hyphen_mid_line_is_a_range_not_a_sign() -> None:
    """``5 - 8`` in the layer must not manufacture ``-8`` that would clear a reading's ``-8``."""
    record = compare_numbers(
        "rose 5 - 8 percent", "rose -8 percent", provenance=PROVENANCE_OWN_CHARACTERS
    )
    assert [read for _ref, read in record.disputed] == ["-8"]


def test_a_hyphenated_label_does_not_manufacture_a_negative() -> None:
    """``Table A-8`` and ``t-1`` in the layer are labels: they must not clear ``-8`` / ``-1``."""
    record = compare_numbers(
        "Table A-8 lag t-1\nKorea -.8", "| Korea | -8 |", provenance=PROVENANCE_OWN_CHARACTERS
    )
    assert [read for _ref, read in record.disputed] == ["-8"]


def test_a_footnoted_cell_keeps_its_number() -> None:
    """``142^a`` is the cell 142 with a footnote letter, not a power: 142 is still checked."""
    record = compare_numbers(
        "Korea 198a", "| Korea | $142^a$ |", provenance=PROVENANCE_OWN_CHARACTERS
    )
    assert [read for _ref, read in record.disputed] == ["142"]


def test_currency_does_not_open_a_maths_span() -> None:
    """``-$8 billion to $1.2 billion`` keeps both amounts; a dropped point is caught."""
    record = compare_numbers(
        "Losses went from -$.8 billion to $1.2 billion.",
        "Losses went from -$8 billion to $1.2 billion.",
        provenance=PROVENANCE_OWN_CHARACTERS,
    )
    assert [read for _ref, read in record.disputed] == ["-8"]


def test_a_model_transcribed_quote_is_checked_and_socr_notes_are_not() -> None:
    """A quote the model read is part of the reading; socr's own quoted notes are not."""
    from socr.core.verdict import reading_numbers_text

    text = (
        "> **Unresolved chart placement on page 3** region 1, 2\n"
        "> continuation 7\n\n"
        "## Page 3\n\n"
        "> Growth was -8 percent.\n"
    )
    assert numeric_tokens(reading_numbers_text(text, 3), reading=True) == (("-8", "-8"),)


def test_a_numberless_reading_names_how_many_numbers_its_layer_prints() -> None:
    """Group (a): the layer prints numbers the reading lacks; the mark says how many."""
    from socr.core.verdict import NO_NUMBER_REASON, text_layer_numbers_clause

    p = PageState(page_num=1)
    p.is_born_digital = True
    p.own_layer_text = "Growth was 2.1 and 3.4 in 2017."
    some = text_layer_numbers_clause(
        PageOutput(page_num=1, text="Growth was steady.", status=PageStatus.SUCCESS, engine="qwen"),
        p,
    )
    assert some.result is CheckResult.UNKNOWN
    assert "the page's text layer prints 3" in some.reason and "#1218" in some.reason
    p.own_layer_text = "Growth was steady."
    none = text_layer_numbers_clause(
        PageOutput(page_num=1, text="Growth was steady.", status=PageStatus.SUCCESS, engine="qwen"),
        p,
    )
    assert none.reason == NO_NUMBER_REASON


def test_reading_mathematics_outside_tables_is_dropped_but_table_cell_kept() -> None:
    """Prose mathematics is dropped from the reading; a math table cell is kept."""
    assert numeric_tokens(r"we have $1 - \alpha$ and $x_{t+1}$", reading=True) == ()
    assert numeric_tokens(r"| a | $-0.04^{***}$ |", reading=True) == (("-0.04", "-0.04"),)


def test_lone_number_in_prose_mathematics_is_kept() -> None:
    """A math expression that is a lone number in running text is still a number."""
    assert ("-0.80", "-0.80") in numeric_tokens(r"the slope is $-0.80$ here", reading=True)


def test_interval_is_not_a_number() -> None:
    """(0,1) and [0,1] are intervals, not numbers."""
    assert numeric_tokens("on (0,1) and [0,1]") == ()


def test_unknown_provenance_is_rejected() -> None:
    """compare_numbers refuses a provenance it does not know."""
    with pytest.raises(ValueError):
        compare_numbers("1.5", "1.5", provenance="hearsay")


# --- clause -----------------------------------------------------------------------------


def test_clause_is_none_when_numbers_agree() -> None:
    """A reading whose numbers occur in the layer raises no clause."""
    agreeing = "| Korea | 4.0 | 2.6 | 5.7 | -.8 | 4.0 | 3.3 |"
    assert text_layer_numbers_clause(_qwen(agreeing), _page()) is None


def test_clause_fails_and_names_the_dispute_on_a_born_digital_layer() -> None:
    """A disagreement on a born-digital layer is a FAIL that quotes both numbers."""
    clause = text_layer_numbers_clause(_qwen(TEALBOOK_READING), _page())
    assert clause is not None
    assert clause.check is Check.TEXT_LAYER_NUMBERS
    assert clause.result is CheckResult.FAIL
    assert "text layer says -.8, reading says -8" in clause.reason


def test_damaged_layer_turns_the_disagreement_into_unknown() -> None:
    """A layer the detector calls damaged may be at fault: UNKNOWN, still naming the dispute."""
    clause = text_layer_numbers_clause(_qwen(TEALBOOK_READING), _page(born_digital=False))
    assert clause is not None
    assert clause.result is CheckResult.UNKNOWN
    assert clause.reason.startswith(DAMAGED_LAYER_PREFIX)
    assert "reading says -8" in clause.reason


def test_minus_as_digit_hits_turns_the_disagreement_into_unknown() -> None:
    """A minus-as-digit finding on a born-digital layer also makes the layer suspect."""
    p = _page()
    p.minus_as_digit_hits = 1
    clause = text_layer_numbers_clause(_qwen(TEALBOOK_READING), p)
    assert clause is not None
    assert clause.result is CheckResult.UNKNOWN


def test_no_own_text_layer_is_unknown() -> None:
    """A page with no text layer of its own cannot confirm its numbers: UNKNOWN."""
    clause = text_layer_numbers_clause(_qwen(TEALBOOK_READING), _page(layer=""))
    assert clause is not None
    assert clause.result is CheckResult.UNKNOWN
    assert clause.reason == NO_OWN_TEXT_REASON


def test_reading_without_a_number_is_unknown() -> None:
    """A reading that ships no number gives vacuous agreement, so it is UNKNOWN; the page's
    layer prints numbers here, so the mark says how many (group (a), #1218)."""
    from socr.core.verdict import NO_NUMBER_LAYER_HAS_SOME_REASON

    clause = text_layer_numbers_clause(_qwen("A reading with no figures at all."), _page())
    assert clause is not None
    assert clause.result is CheckResult.UNKNOWN
    assert clause.reason == NO_NUMBER_LAYER_HAS_SOME_REASON.format(count=6)
    assert clause.reason != NO_NUMBER_REASON


def test_prior_ocr_page_is_left_to_the_prior_ocr_clause() -> None:
    """#1156: a prior-OCR layer is not this clause's to judge."""
    p = _page()
    p.prior_ocr_layer_suspect = True
    assert text_layer_numbers_clause(_qwen(TEALBOOK_READING), p) is None


@pytest.mark.parametrize("engine", ["native", "chart_asset"])
def test_non_model_engines_are_not_compared(engine: str) -> None:
    """A native lane or a chart picture is not a model reading of the page's characters."""
    output = PageOutput(page_num=1, text=TEALBOOK_READING, status=PageStatus.SUCCESS, engine=engine)
    assert text_layer_numbers_clause(output, _page()) is None


def test_socrs_own_quote_lines_are_not_the_reading() -> None:
    """socr's blockquote notes carry socr's numbers (99, 1), which must not be compared."""
    output = _qwen("> [page 1 unchecked: chart_region=FAIL; scope=page; reason=x 99]\n\nKorea -.8")
    assert text_layer_numbers_clause(output, _page(layer="Korea -.8")) is None


# --- seam -------------------------------------------------------------------------------


def test_disagreeing_success_page_is_demoted_and_marked() -> None:
    """A SUCCESS page whose reading disagrees with its layer is WARNING, marked and explained."""
    out, _verdict = _apply_verdict_marks(_qwen(TEALBOOK_READING), _page(), 1)
    assert out.status is PageStatus.WARNING
    assert out.failure_mode is FailureMode.NUMBERS_DISAGREE_WITH_TEXT_LAYER
    assert out.text.startswith(
        "> [page 1 unchecked: text_layer_numbers=FAIL; scope=page; "
        "reason=text layer says -.8, reading says -8]"
    )


def test_applying_the_seam_twice_is_idempotent() -> None:
    """Re-running the seam over its own output changes nothing and never falls back."""
    p = _page()
    first_out, first_verdict = _apply_verdict_marks(_qwen(TEALBOOK_READING), p, 1)
    second_out, second_verdict = _apply_verdict_marks(first_out, p, 1)
    assert second_out.text == first_out.text
    assert second_out.status is first_out.status
    assert second_out.failure_mode is first_out.failure_mode
    assert second_verdict.clauses == first_verdict.clauses
    assert all(c.check is not Check.UNKNOWN_CHECK for c in second_verdict.clauses)


def test_unknown_case_demotes_with_numbers_unchecked() -> None:
    """With no own text layer the page is demoted with the NUMBERS_UNCHECKED mode."""
    out, _verdict = _apply_verdict_marks(_qwen(TEALBOOK_READING), _page(layer=""), 1)
    assert out.status is PageStatus.WARNING
    assert out.failure_mode is FailureMode.NUMBERS_UNCHECKED


def test_disagreement_is_permanent_across_resume() -> None:
    """The demotion's verdict is one a resume may skip-and-keep rather than re-OCR forever."""
    out, verdict = _apply_verdict_marks(_qwen(TEALBOOK_READING), _page(), 1)
    assert _resume_verdict_is_permanently_unverifiable(verdict.to_dict(), out.failure_mode.value)


def test_unchecked_is_permanent_across_resume() -> None:
    """The UNKNOWN demotion is equally permanent across resume."""
    out, verdict = _apply_verdict_marks(_qwen(TEALBOOK_READING), _page(layer=""), 1)
    assert _resume_verdict_is_permanently_unverifiable(verdict.to_dict(), out.failure_mode.value)
