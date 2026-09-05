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
    label_key,
    label_key_is_bare_symbolic,
    strip_math_presentation,
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


# ---------------------------------------------------------------------------
# GH-585: sibling LaTeX presentation classes the GH-582 wrap fix left open.
# Doc05/doc07 ladder-corpus held pairs (`` `/consilium` `` A1 replay).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("native", "model"),
    [
        ("∆Slope (3m)", r"$\Delta \text{ Slope (3m)}$"),
        ("∆log Comm. price (3m)", r"$\Delta \log \text{ Comm. price (3m)}$"),
    ],
)
def test_gh585_greek_command_and_word_command_labels_agree(native, model):
    """GH-585: a Greek-letter command (``\\Delta``) must key to the same
    ``("greek", ...)`` token as the native Unicode symbol, and a plain
    alphabetic word command (``\\log``) must lose only its backslash — on
    ``main`` (GH-582 fix only) these two pairs still contradict because
    ``\\Delta`` survives as the spelled-out ASCII word ``delta``, which
    ``normalize_label`` does NOT discard the way it discards the native
    ``∆`` symbol."""
    assert label_key(native) == label_key(model)


def test_gh585_a_real_text_difference_still_disagrees_after_the_map():
    """GH-585: the map is exact, not a widening — `` S&P `` vs
    `` S&P 500 (3m) `` is a genuine text difference (missing ``500 (3m)``)
    and must still fail to match after the Greek/escape/word-command map."""
    native = "∆log S&P"
    model = r"$\Delta \log \text{ S\&P 500 (3m)}$"
    assert label_key(native) != label_key(model)


@pytest.mark.parametrize(
    ("command", "tag"),
    [
        (r"\Delta", "capdelta"),
        (r"\beta", "beta"),
        (r"\Sigma", "capsigma"),
    ],
)
def test_gh585_greek_command_keys_to_its_tag(command, tag):
    """GH-585 review round 4: the Greek command's identity now lives in
    ``label_key()``'s structured tuple, not in a plain string
    ``strip_math_presentation`` returns -- a bare command with no other
    content is the "bare symbolic" shape (see the UNVERIFIABLE test
    below), so this checks the tag directly rather than equality with
    another label."""
    assert label_key(command) == (("greek", tag),)


def test_gh585_variant_greek_command_keys_to_its_own_distinct_tag():
    """GH-585 review round 2/3/4: ``\\varepsilon`` is a DIFFERENT glyph
    from ``\\epsilon`` (not established to be the same symbol in this
    corpus, unlike the U+0394/U+2206 Delta pair) — it must key to its own
    ``varepsilon`` tag, never to the base letter's tag."""
    assert label_key(r"$\varepsilon$ x") == (("greek", "varepsilon"), ("lit", "x"))
    assert label_key(r"$\epsilon$ x") == (("greek", "epsilon"), ("lit", "x"))


@pytest.mark.parametrize(
    ("escaped", "expected"),
    [
        (r"\&", "&"),
        (r"\%", "%"),
        (r"\_", "_"),
        (r"\$", "$"),
        (r"\#", "#"),
    ],
)
def test_gh585_escaped_punctuation_unescapes_in_the_cell_path(escaped, expected):
    assert strip_math_presentation(escaped, label=False) == expected


@pytest.mark.parametrize("command", [r"\log", r"\ln", r"\exp"])
def test_gh585_alphabetic_word_commands_key_as_the_bare_word(command):
    """GH-585 review round 4: a real, verified LaTeX operator is a textual
    equivalence, not a symbol -- it folds into the surrounding literal
    text (a ``("lit", ...)`` token), not a ``("greek", ...)``/``("cmd",
    ...)`` tag."""
    assert label_key(command) == (("lit", command[1:]),)


def test_gh585_unsupported_word_command_is_tagged_not_silently_dropped():
    """GH-585 review round 2/3/4: only the standard LaTeX math-operator
    names fold into their bare word. An unverified/unsupported command
    (``\\logx`` -- not a real LaTeX macro, but shaped like one) must not
    gain an agreement it never earned. Round 3 tried keeping the literal
    backslash, but ``normalize_label`` strips ANY leftover backslash
    unconditionally downstream, so that alone was not sufficient. Round 4:
    ``label_key`` tags it ``("cmd", "logx")``, a different tuple shape
    from both ``("lit", "log")`` and ``("lit", "logx")`` structurally, not
    merely a string that happens to look different."""
    assert label_key(r"\logx") == (("cmd", "logx"),)
    assert label_key(r"\log") == (("lit", "log"),)


# ---------------------------------------------------------------------------
# GH-585 review round 2: Greek-letter identity must survive the compare, and
# the escape-unmap class also belongs on the numeric (``label=False``) path.
# ---------------------------------------------------------------------------


def test_gh585_different_greek_letters_with_identical_trailing_text_disagree():
    """Before this fix, ``normalize_label``'s ASCII-only filter erased EVERY
    Greek letter identically (both to ""), so ``α Coefficient`` and
    ``$\\beta$ Coefficient`` folded to the same key and falsely agreed
    regardless of which letter either side named. Tagging each Greek
    letter with its own tag must make two different letters keep
    disagreeing."""
    alpha_key = label_key("α Coefficient")
    beta_key = label_key(r"$\beta$ Coefficient")
    assert alpha_key != beta_key
    # And the same letter on both sides must still agree.
    same_key = label_key(r"$\alpha$ Coefficient")
    assert alpha_key == same_key


def test_gh585_bare_greek_symbol_label_is_unverifiable_not_a_match():
    """The tagging must not defeat binding.py's bare-symbol-label
    fail-closed rule: a label that IS a Greek letter and nothing else
    (native ``β``, model ``$\\beta$``) keys to the single-element tuple
    ``label_key_is_bare_symbolic`` flags as unprovable -- same policy as
    before this fix, expressed structurally now -- because a Greek letter
    embedded inside a longer label is a different case (previous test)."""
    native_key = label_key("β")
    model_key = label_key(r"$\beta$")
    assert native_key == model_key == (("greek", "beta"),)
    assert label_key_is_bare_symbolic(native_key)
    assert label_key_is_bare_symbolic(model_key)


def test_gh585_real_greek_capital_delta_codepoint_aliases_the_increment_glyph():
    """U+0394 (the actual GREEK CAPITAL LETTER DELTA) and U+2206 (INCREMENT,
    what the corpus's native layer emits for ``\\Delta``) render identically
    and must key to the same tag."""
    assert label_key("ΔSlope") == label_key("∆Slope")


def test_gh585_typed_greek_token_does_not_collide_with_the_spelled_out_prose_word():
    """GH-585 review round 3/4: a plain-ASCII-word transliteration (``∆`` ->
    ``"Delta"``) is itself a widening -- a label that literally types the
    English word "Delta" would then falsely agree with the native symbol.
    A structured ``("greek", ...)`` tag can never equal a ``("lit", ...)``
    literal-text token regardless of what the literal text says, so
    ``∆Slope (3m)`` must NOT agree with a label that spells the word out
    -- even the sentinel word a PREVIOUS round of this fix used
    (``"greekcapdelta"``/``"greekalpha"``/``"unmappedlogx"``, round 3's
    string tokens)."""
    assert label_key("∆Slope (3m)") != label_key("Delta Slope (3m)")
    assert label_key("α Coefficient") != label_key("greekalpha Coefficient")
    assert label_key(r"\logx y") != label_key("unmappedlogx y")


def test_gh585_typed_greek_token_preserves_case_distinction():
    """GH-585 review round 3/4: ``normalize_label`` lowercases, so a plain
    lower-case name for both ``Δ`` (capital) and ``δ`` (lower-case, a
    DIFFERENT letter) would fold together. The ``cap``-prefixed tag keeps
    case distinct through the compare."""
    assert label_key("Δ x") != label_key("δ x")


@pytest.mark.parametrize(
    ("native", "model", "base_letter"),
    [
        ("ϑ Coefficient", r"$\vartheta$ Coefficient", "θ"),
        ("ϕ Coefficient", r"$\varphi$ Coefficient", "φ"),
        ("ϖ Coefficient", r"$\varpi$ Coefficient", "π"),
        ("ϱ Coefficient", r"$\varrho$ Coefficient", "ρ"),
        ("ς Coefficient", r"$\varsigma$ Coefficient", "σ"),
        ("ϵ Coefficient", r"$\varepsilon$ Coefficient", "ε"),
    ],
)
def test_gh585_variant_unicode_glyphs_agree_with_their_variant_command(native, model, base_letter):
    """GH-585 review round 4: a native PDF's text layer can emit the
    variant-glyph codepoint directly (``ϑ`` U+03D1, ``ϕ`` U+03D5 -- distinct
    from base ``φ`` U+03C6, ``ϖ`` U+03D6, ``ϱ`` U+03F1, ``ς`` U+03C2, ``ϵ``
    U+03F5) instead of the base letter; each must key to the same tag as
    its ``\\var...`` command, not the base letter's tag."""
    assert label_key(native) == label_key(model)
    base_form = native[0]  # the variant glyph itself, first char of the parametrized text
    assert label_key(native) != label_key(native.replace(base_form, base_letter))


def test_gh585_escape_unmap_also_applies_to_the_numeric_cell_path():
    """GH-585 review: a cell path escape (``12\\%``) must not be silently
    invisible to the numeric tokenizer just because it was never wrapped in
    ``$...$`` — ``label=False`` did not unescape at all before this fix."""
    assert strip_math_presentation(r"12\%", label=False) == "12%"
    assert is_numeric_token(r"12\%")
    assert _normalize_numeric_token(r"12\%") == _normalize_numeric_token("12%")
    # A plain, already-unescaped value is unaffected.
    assert strip_math_presentation("12%", label=False) == "12%"
    assert is_numeric_token("12%")


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
