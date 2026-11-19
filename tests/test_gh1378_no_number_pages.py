"""#1378: a page that prints no number is not doubted by the number check.

Pins (a) the Roman-numeral page references the two sides of the comparison tokenised
differently (``VIII-1``), and (b) the exit for a page with nothing to check, which needs the
layer to hold no further number AND independent evidence (the #1050 ink coverage) that the
layer holds all of the page. Each guard is a DIFFERENCE between two runs of the same clause
that differ in one input, so no outcome measured on one machine is pinned.
"""

from __future__ import annotations

import pytest

from socr.core.marks import CheckResult
from socr.core.result import PageOutput, PageStatus
from socr.core.state import PageState
from socr.core.verdict import (
    NO_NUMBER_ON_PAGE_REASON,
    text_layer_numbers_clause,
    text_layer_numbers_outcome,
)
from socr.tables.reading_agreement import PROVENANCE_OWN_CHARACTERS, compare_numbers

CONTENTS = "\n".join(
    f"{name} ..... {ref}"
    for name, ref in [
        ("First", "I-1"),
        ("Second", "II-1"),
        ("Seventh", "VII-1"),
        ("Eighth", "VIII-1"),
        ("Thirteenth", "XIII-12"),
    ]
)


def _page(layer: str, *, coverage: str = "passed", born_digital: bool = False, shape: bool = True):
    p = PageState(page_num=1)
    p.own_layer_text = layer
    p.is_born_digital = born_digital
    p.layer_refused_for_shape_alone = shape
    p.text_coverage_verdict = coverage
    return p


def _read(text: str) -> PageOutput:
    return PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, engine="qwen")


def _result(output: PageOutput, p: PageState):
    clause = text_layer_numbers_clause(output, p)
    return None if clause is None else clause.result


# --- tokenisation --------------------------------------------------------------------


def test_roman_page_references_tokenise_the_same_on_both_sides() -> None:
    """The layer's ``VIII-1`` frees a ``1``; the reading must claim it, as it does ``VII-1``."""
    record = compare_numbers(CONTENTS, CONTENTS, provenance=PROVENANCE_OWN_CHARACTERS)
    assert record.compared == 0
    assert record.reference_only == 0
    assert record.disputed == ()


def test_roman_reference_claim_does_not_hide_a_dropped_value() -> None:
    """A label digit claims ONE layer number: a layer value the reading lacks stays counted."""
    layer = CONTENTS + "\nTotal 41"
    record = compare_numbers(layer, CONTENTS, provenance=PROVENANCE_OWN_CHARACTERS)
    assert record.reference_only == 1


def test_a_word_hyphen_digit_is_still_not_a_label() -> None:
    """Only a well-formed Roman numeral is widened: ``Korea-8`` keeps its old treatment."""
    base = compare_numbers("Korea 8", "Korea-8", provenance=PROVENANCE_OWN_CHARACTERS)
    widened = compare_numbers("Korea-8", "Korea-8", provenance=PROVENANCE_OWN_CHARACTERS)
    assert base.compared == widened.compared == 0
    assert widened.reference_only == 1  # the layer's 8 is not claimed by the reading's word


# --- the exit: only with the ink-coverage evidence -----------------------------------


@pytest.mark.parametrize(
    ("layer", "reading"),
    [
        ("International Developments", "# International Developments"),
        ("(This page intentionally blank.)", "(This page intentionally blank.)"),
        ("(Page I-58 is intentionally blank.)", "(Page I-58 is intentionally blank.)"),
        (CONTENTS, CONTENTS),
    ],
)
def test_a_page_with_no_number_is_not_doubted_only_when_coverage_passed(layer, reading) -> None:
    covered = text_layer_numbers_outcome(_read(reading), _page(layer))
    uncovered = text_layer_numbers_outcome(_read(reading), _page(layer, coverage=""))
    assert covered.clause is None
    assert covered.entry.reason == NO_NUMBER_ON_PAGE_REASON
    assert uncovered.clause is not None
    assert uncovered.clause.result is CheckResult.UNKNOWN


@pytest.mark.parametrize("verdict", ["flagged", "unknown", ""])
def test_coverage_that_did_not_pass_keeps_the_page_warned(verdict) -> None:
    p = _page("International Developments", coverage=verdict)
    assert _result(_read("# International Developments"), p) is CheckResult.UNKNOWN


def test_guard_layer_prints_numbers_and_reading_ships_none_is_still_warned() -> None:
    p = _page("Growth was 2.1 and 3.4 in 2017.", born_digital=True, shape=False)
    assert _result(_read("Growth was steady."), p) is CheckResult.UNKNOWN
    # the difference: the same reading on a layer with no number is not doubted
    p_none = _page("Growth was steady.", born_digital=True, shape=False)
    assert _result(_read("Growth was steady."), p_none) is None


def test_guard_scan_without_layer_is_still_warned_whatever_the_coverage() -> None:
    for coverage in ("passed", "flagged", ""):
        p = _page("", coverage=coverage)
        assert _result(_read("# International Developments"), p) is CheckResult.UNKNOWN


def test_guard_layer_not_shown_clean_keeps_the_page_warned() -> None:
    """A refused layer that is not shape-refused-clean (p011's dot leaders) confirms nothing."""
    refused = _page("Summary", shape=False)
    assert _result(_read("Summary"), refused) is CheckResult.UNKNOWN
    shape_ok = _page("Summary", shape=True)
    assert _result(_read("Summary"), shape_ok) is None
    damaged = _page("Summary", born_digital=True, shape=False)
    damaged.minus_as_digit_hits = True
    assert _result(_read("Summary"), damaged) is CheckResult.UNKNOWN


def test_guard_numbers_in_the_reading_are_still_compared() -> None:
    """The exit is for ``compared == 0`` only: a number the layer lacks is still a FAIL/UNKNOWN."""
    p = _page("Rate", born_digital=True, shape=False)
    assert _result(_read("Rate was 9.31"), p) is CheckResult.FAIL


# --- the words check confirms a clean short layer (second commit) -------------------------

from socr.core.marks import Check  # noqa: E402
from socr.core.verdict import (  # noqa: E402
    WORDS_LAYER_NOT_EVIDENCE_REASON,
    independent_page_evidence,
    text_layer_words_outcome,
)


def _with_words(p: PageState, words: str) -> PageState:
    p.own_layer_word_reference = words
    return p


def _words(text: str, p: PageState):
    return text_layer_words_outcome(_read(text), p).entry


def test_words_agree_on_a_clean_short_layer_is_a_pass() -> None:
    p = _with_words(_page("International Developments"), "international developments")
    entry = _words("# International Developments", p)
    assert entry.result.value == "PASS"
    p_open = _with_words(
        _page("International Developments", coverage=""), "international developments"
    )
    other = _words("# International Developments", p_open)
    assert other.result.value == "NOT_RUN"
    assert other.reason == WORDS_LAYER_NOT_EVIDENCE_REASON


def test_one_word_differs_keeps_the_page_warned() -> None:
    p = _with_words(_page("International Developments"), "international developments")
    outcome = text_layer_words_outcome(_read("# International Development"), p)
    assert outcome.clause is not None
    assert outcome.clause.result is CheckResult.UNKNOWN


@pytest.mark.parametrize("verdict", ["flagged", "unknown", ""])
def test_a_short_layer_with_coverage_not_passed_is_not_evidence(verdict) -> None:
    """A short running head over a scanned body: the body's ink is uncovered, so ``flagged``."""
    p = _with_words(_page("Running head", coverage=verdict), "running head")
    assert _words("Running head", p).result.value == "NOT_RUN"


def test_a_layer_not_shown_clean_is_not_evidence_even_with_coverage() -> None:
    p = _with_words(_page("Summary", shape=False), "summary")
    assert _words("Summary", p).result.value == "NOT_RUN"
    sym = _with_words(_page("Summary"), "summary")
    sym.has_unrecovered_symbol_glyphs = True
    assert _words("Summary", sym).result.value == "NOT_RUN"


def test_a_divider_page_is_independently_confirmed_only_with_a_words_pass() -> None:
    """Numbers N/A alone is no confirmation (the page stays uncorroborated); with a words PASS
    on the same text it is."""
    from socr.core.verdict import stamp_text_digest

    text = "# International Developments"
    for coverage, confirmed in (("passed", True), ("", False)):
        p = _with_words(
            _page("International Developments", coverage=coverage), "international developments"
        )
        out = _read(text)
        entries = [
            stamp_text_digest(text_layer_words_outcome(out, p).entry, text),
            stamp_text_digest(text_layer_numbers_outcome(out, p).entry, text),
        ]
        assert independent_page_evidence(entries, text) is confirmed
    assert Check.TEXT_LAYER_WORDS.value == "text_layer_words"
