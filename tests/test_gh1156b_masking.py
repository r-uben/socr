"""#1156 Part B: the masker and the shred rule.

Pure unit tests, ported from the abandoned first attempt (``fix/issue-1156``).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def test_a_fraction_written_as_one_glyph_is_withheld() -> None:
    """``\\d`` is Unicode ``Nd`` only, so "3½" shipped its half and "½ to ¾" shipped both."""
    from socr.tables.prior_ocr_mask import mask_numbers

    masked, withheld = mask_numbers("rates from 3 to 3½ per cent")
    assert "½" not in masked
    # Two: the "3", and "3½" as ONE number -- a fraction glyph closes the run it sits
    # on, and the entity form "3&#189;" reads the same way (orchestrator, round 3).
    assert withheld == 2

    masked_both, withheld_both = mask_numbers("½ to ¾")
    assert withheld_both == 2, "a page of vulgar fractions withheld nothing and retained nothing"
    assert "½" not in masked_both and "¾" not in masked_both


def test_a_confirmed_number_does_not_uncover_a_different_one() -> None:
    """Unmasking is by POSITION. By value, a confirmed 400 uncovers "(400)" as well --
    a standard error, not the coefficient that was checked."""
    from socr.tables.prior_ocr_mask import mask_numbers, numeric_tokens_in_order

    text = "the coefficient 400 and its standard error (400)"
    first = numeric_tokens_in_order(text).index("400")
    masked, _withheld = mask_numbers(text, confirmed=frozenset({first}))
    assert "and its standard error ([number withheld, see image])" in masked
    assert masked.count("400") == 1


def test_every_position_confirmed_masks_nothing() -> None:
    """One enumeration: confirming every position the tokeniser emits leaves the text
    byte-identical, so a confirmed position can never unmask the wrong number."""
    from socr.tables.prior_ocr_mask import mask_numbers, numeric_tokens_in_order

    text = "rates 3 to 3 Y, on July 12-13, 1983 with $400 million"
    tokens = numeric_tokens_in_order(text)
    masked, withheld = mask_numbers(text, confirmed=frozenset(range(len(tokens))))
    assert withheld == 0 and masked == text


def test_a_shred_is_a_line_holding_one_letter() -> None:
    """The #1027 page's own shape: a rotated axis label spelled down the page.

    A lone DIGIT is not a shred -- it can be a table value -- and neither is a short word.
    """
    from socr.tables.prior_ocr_mask import withhold_shreds

    layer = "Ave. 479 240 201\nP\nr\ni\nc\ne\n50\n5\nof\n"
    kept, dropped = withhold_shreds(layer)
    assert dropped == 5
    assert kept.splitlines() == ["Ave. 479 240 201", "50", "5", "of"]


def test_the_issues_own_page_keeps_its_prose_and_the_garbage_page_does_not() -> None:
    """Both populations, from the real PDFs, through the production helpers.

    1957-05-28 p2 is render-mode 3 AND readable: the old scoping deleted all 4,519
    characters of it. Forsythe-Lundholm p15 is #1027's own case: readable table rows
    beside a shredded label.
    """
    import pymupdf

    from socr.tables.prior_ocr_mask import mask_numbers, withhold_shreds

    corpus = Path(
        "/Users/rubenffuertes/repos/tools/socr/scratch/routing-baseline/corpus/pdfs/"
        "1990__Forsythe_Lundholm__Information_Aggregation_in_an_Experimental_Market__ECMA.pdf"
    )
    fed = Path(
        "/Users/rubenffuertes/repos/research/central-bank-network/data/ocr-staging/fed-01/pdf/"
        "fed-meetings-1957-1957-05-1957-05-28-record-of-policy-actions.pdf"
    )
    if not corpus.exists() or not fed.exists():
        pytest.skip("corpus pages not present on this machine")

    fed_kept, fed_shreds = withhold_shreds(pymupdf.open(fed)[1].get_text())
    fed_masked, fed_withheld = mask_numbers(fed_kept)
    assert fed_shreds == 0, "the readable Fed page had lines dropped as shreds"
    assert "ANNUAL REPORT OF BOARD OF GOVERNORS" in fed_masked
    assert "3 Y," not in fed_masked and fed_withheld > 0

    g_kept, g_shreds = withhold_shreds(pymupdf.open(corpus)[14].get_text())
    assert g_shreds >= 5, "the shredded axis label was kept as if it were text"
    assert "ROBERT FORSYTHE" in g_kept, "its readable rows were thrown away with the shreds"
    for letter in ("\nP\n", "\nr\n", "\ni\n"):
        assert letter not in f"\n{g_kept}\n"


@pytest.mark.parametrize(
    ("left", "right", "why"),
    [
        ("(0.023*)", "0.023*", "a significance star inside the bracket"),
        ("($400)", "$400", "a currency symbol inside the bracket"),
        ("¹-0.021", "¹+0.021", "a sign after a footnote marker"),
        ("-.021", "+.021", "a leading-dot decimal keeps its sign"),
        ("-.021", "021", "a leading-dot decimal keeps its point"),
        ("(.023)", ".023", "a leading-dot decimal keeps its brackets"),
        ("-½", "+½", "a sign before a fraction glyph"),
        ("-&#189;", "+&#189;", "a sign before an entity"),
    ],
)
def test_the_grammar_tells_these_apart(left: str, right: str, why: str) -> None:
    """Ported from the comparator's cases: the two sides tokenise differently, so a
    second reader (#1208) comparing positions would see a disagreement."""
    from socr.tables.prior_ocr_mask import numeric_tokens_in_order

    assert numeric_tokens_in_order(left) != numeric_tokens_in_order(right), why


def test_a_comma_joins_thousands_and_separates_everything_else() -> None:
    """ "June 18,1957" is two numbers. Read as one, it also shifted every confirmed
    position after it, so the wrong numbers were unmasked."""
    from socr.tables.prior_ocr_mask import numeric_tokens_in_order

    assert numeric_tokens_in_order("June 18,1957") == ("18", "1957")
    assert numeric_tokens_in_order("1,204 and 1,234,567") == ("1204", "1234567")


def test_a_fraction_reads_the_same_as_its_entity() -> None:
    from socr.tables.prior_ocr_mask import numeric_tokens_in_order

    assert numeric_tokens_in_order("3&#189;") == numeric_tokens_in_order("3½") != ()


def test_a_letter_number_is_withheld() -> None:
    """Unicode ``Nl`` (the precomposed Roman numerals) is the third numeric category the
    first attempt missed (Argon, Part B design review)."""
    from socr.tables.prior_ocr_mask import mask_numbers

    masked, withheld = mask_numbers("see Chapter Ⅻ and Part Ⅳ")
    assert withheld == 2
    assert "Ⅻ" not in masked and "Ⅳ" not in masked


def test_every_unicode_numeral_is_withheld() -> None:
    """Every character Unicode files under Nd, No or Nl -- not a hand list, which missed
    1,702 of them (Arabic-Indic and fullwidth digits, circled numbers; Argon)."""
    import sys
    import unicodedata

    from socr.tables.prior_ocr_mask import mask_numbers

    missed = [
        hex(cp)
        for cp in range(sys.maxunicode + 1)
        if unicodedata.category(chr(cp)) in ("Nd", "No", "Nl") and mask_numbers(chr(cp))[1] != 1
    ]
    assert missed == []
