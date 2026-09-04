"""GH-103: presentation stripping in the numeric tokenizer.

``collect_table_tokens`` fed markdown cells to the anchored ``_NUM_TOKEN_RE``
without removing markdown emphasis, so every bolded numeric cell was dropped from
the multiset. The source-evidence gate is built on that multiset, so on a table
whose numeric cells are bold — which is how VLMs emit section totals — the gate had
nothing left to check and passed vacuously.

The table recorded in ``docs/log/2026-07-30_obr-table-bakeoff.md`` (Result 4), a
vision model's entirely fabricated fiscal data returned at exit 0, is bold-celled.
Every invented number in it was invisible to the tokenizer.

Two related gaps fixed in the same pass: U+2212 minus (native fiscal PDFs carry it,
VLMs emit ASCII ``-``, so both sides were dropped and a value that actually agreed
looked absent from both) and currency prefixes.

GH-206 closed two further gaps in the same predicate: leading-decimal values
(``.034``, no leading zero — an ordinary way to print a coefficient below 1) and
the Unicode significance-star/dagger glyphs typeset econometrics tables use
(``∗`` U+2217 is what LaTeX ``\\ast`` emits, not ASCII ``*``).
"""

from __future__ import annotations

import pytest

from socr.tables.native_verifier import (
    _normalize_numeric_token,
    _numeric_multiset_from_tokens,
    is_numeric_token,
    strip_presentation,
)
from socr.tables.source_evidence import collect_table_tokens

# Verbatim from the fabrication in Result 4 of the bake-off note.
_FABRICATED = "\n".join(
    [
        "| | 2017 | 2018 | 2019 | 2020 |",
        "|---|---|---|---|---|",
        "| **Revenue** | **23,126** | **25,123** | **26,204** | **25,607** |",
        "| Taxes | 19,451 | 21,326 | 22,239 | 21,650 |",
        "| of which: indirect taxes | 10,230 | 11,450 | 12,010 | 11,500 |",
    ]
)


def test_the_fabricated_table_is_no_longer_invisible():
    """The regression that matters: invented bold numbers must be checkable."""
    tokens = collect_table_tokens(_FABRICATED)

    assert tokens is not None
    # Every bolded invented value is now present and can be tested for support.
    for invented in ("23126", "25123", "26204", "25607"):
        assert invented in tokens.numeric, f"{invented} invisible to the evidence gate"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("**23,126**", "23126"),
        ("__1,204__", "1204"),
        ("*45%*", "45"),
        ("−24.8", "-24.8"),  # U+2212
        ("–24.8", "-24.8"),  # en dash
        ("£43.2", "43.2"),
        ("$1,204", "1204"),
        ("€0.5", "0.5"),
        ("(0.014)", "0.014"),  # econ negative convention, preserved
        ("1,234,567", "1234567"),
        ("2.10", "2.10"),  # stated precision preserved
    ],
)
def test_normalization_of_decorated_tokens(raw, expected):
    assert _normalize_numeric_token(raw) == expected


@pytest.mark.parametrize("raw", ["**23,126**", "−24.8", "£43.2", "24.8", "(0.014)", "45%"])
def test_decorated_tokens_are_recognised_as_numeric(raw):
    assert is_numeric_token(raw)


@pytest.mark.parametrize("raw", ["Revenue", "**Revenue**", "of which:", "", "---", "—"])
def test_labels_are_still_not_numeric(raw):
    assert not is_numeric_token(raw)


def test_unicode_and_ascii_minus_agree():
    """The two sides of a comparison must not disagree over an encoding."""
    assert _normalize_numeric_token("−24.8") == _normalize_numeric_token("-24.8")


def test_a_currency_prefixed_value_matches_its_bare_form():
    native = _numeric_multiset_from_tokens(["£43.2", "£26.8"])
    emitted = _numeric_multiset_from_tokens(["43.2", "26.8"])

    assert native == emitted


def test_a_bold_table_and_its_plain_form_tokenize_identically():
    plain = "| A | 24.8 | 18.4 |\n| --- | --- | --- |\n| B | 1.0 | 2.0 |"
    bold = "| A | **24.8** | **18.4** |\n| --- | --- | --- |\n| B | 1.0 | 2.0 |"

    assert collect_table_tokens(plain).numeric == collect_table_tokens(bold).numeric


def test_strip_presentation_leaves_meaning_bearing_marks():
    """Parentheses, percent and commas are handled by the normalizer, not here."""
    assert strip_presentation("(0.014)") == "(0.014)"
    assert strip_presentation("45%") == "45%"
    assert strip_presentation("1,204") == "1,204"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (".034", "0.034"),  # leading-decimal, no leading zero
        ("(.034)", "(0.034)"),
        ("-.034", "-0.034"),
        ("0.67∗∗∗", "0.67"),  # U+2217 ASTERISK OPERATOR (LaTeX \ast)
        ("0.67†", "0.67"),  # U+2020 DAGGER
        ("0.67‡", "0.67"),  # U+2021 DOUBLE DAGGER
        ("0.67§", "0.67"),  # U+00A7 SECTION SIGN
    ],
)
def test_gh206_notation_gaps_are_stripped(raw, expected):
    assert strip_presentation(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [".034", "(.034)", "-.034", "0.67∗∗∗", "0.67†", "0.67‡", "0.67§"],
)
def test_gh206_notation_gaps_are_recognised_as_numeric(raw):
    assert is_numeric_token(raw)


def test_gh206_leading_decimal_normalizes_the_same_as_its_zero_prefixed_form():
    """The two sides of a comparison must not disagree over a missing leading zero."""
    assert _normalize_numeric_token(".034") == _normalize_numeric_token("0.034")


@pytest.mark.parametrize(
    "raw",
    [
        "p<.05",  # not a bare value: comparator prefix
        "e.g.",
        "N/A",
        "..034",  # malformed, not a value
        ".a",
    ],
)
def test_gh206_notation_fix_does_not_admit_prose_tokens(raw):
    assert not is_numeric_token(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "4$3",  # stray/embedded $, not a wrapper
        "43$",  # trailing-only $, not a balanced whole-token wrap
        "$10^2$",  # exponent changes the value: an expression, not "102"
        "1_2",  # bare subscript marker outside any math wrap
        "(/997)",  # documented codebook/corruption string, not a $-case
        "/53",  # ditto
        "LoIic6",  # ditto
        "$�43$",  # U+FFFD replacement char: encoding garbage
        "$43$",  # BMP private-use char: encoding garbage
        "$$43$",  # doubled leading $: interior contains the delimiter,
        # not a single balanced wrap -- must not unwrap to "$43" and then
        # fall into the currency-prefix strip as a false numeric match
        "$43$$",  # doubled trailing $, same reasoning
        "$$",  # only delimiters, no content at all
    ],
)
def test_gh582_numeric_path_leaves_non_wrapped_and_garbage_tokens_alone(raw):
    """GH-582: ``strip_math_presentation`` (numeric path, the one
    ``is_numeric_token``/``_normalize_numeric_token`` route through) only
    unwraps ONE balanced ``$...$``/``\\(...\\)`` pair around the WHOLE
    token. A stray or embedded delimiter is not a wrapper and is left
    alone; a numeric ``^``/``_`` script marker is never dropped in this
    path (an exponent/subscript changes the value, it is not
    typesetting); and encoding garbage inside a wrap still fails the
    numeric regexes on its own merits. Every case here must stay exactly
    as non-numeric as it was before GH-582."""
    assert not is_numeric_token(raw)


def test_gh582_numeric_path_unwraps_a_balanced_whole_token_wrap():
    """GH-582: the two shapes the issue actually reports -- ``$...$`` and
    ``\\(...\\)`` fully enclosing a numeric cell -- now compare on value.
    A clean, unsigned wrap (``$43$``) takes the identical balanced-wrapper
    path as the signed decimal the issue reports (``$-0.06$``); there is
    no principled reason to unwrap one and not the other, so both change."""
    assert is_numeric_token("$-0.06$")
    assert _normalize_numeric_token("$-0.06$") == "-0.06"
    assert is_numeric_token(r"\(0.5\)")
    assert _normalize_numeric_token(r"\(0.5\)") == "0.5"
    assert is_numeric_token("$43$")
    assert _normalize_numeric_token("$43$") == "43"


def test_content_labels_were_never_the_bug():
    """Documented for the record: the content tokenizer scans, so bold was fine.

    The GH-103 issue text claimed bold labels were dropped too. They were not —
    ``Revenue`` was absent from the reproduction only because it sat in the header
    row, which is skipped by design.
    """
    md = "| h1 | h2 |\n| --- | --- |\n| **Energy price guarantee** | 24.8 |"

    tokens = collect_table_tokens(md)

    assert "energy" in tokens.content
    assert "guarantee" in tokens.content
