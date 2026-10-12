"""#1097: a currency $ must not switch off paragraph reflow.

Fail-closed: real inline maths and an unbalanced maths $ still protect. The
classifier is ``_is_currency_dollar``. Round-3 mutants: allow 1-letter
words after the amount → P0-1 fails; drop the standalone-token suffix
check → P0-2 fails.
"""

from __future__ import annotations

from socr.core.native_paragraphs import (
    NativeVocabulary,
    _is_currency_dollar,
    native_vocabulary,
    reflow_native_prose,
)


def _vocab(*texts: str) -> NativeVocabulary:
    return native_vocabulary(list(texts))


def _para(*printed: str):
    return tuple((p,) for p in printed)


def test_currency_amounts_do_not_block_reflow() -> None:
    vocab = _vocab("ab")
    for lead in ("$29.99", "$1,200 million", "US$5", "$ 95.6", "billions of 1987$"):
        text = f"{lead} on the page\na-\nb"
        assert reflow_native_prose(text, (_para("a-", "b"),), vocab) == f"{lead} on the page\nab", (
            lead
        )


def test_inline_maths_stays_protected() -> None:
    """A maths line is inside the protected span and is not joined to a neighbour.

    Balanced ``$...$`` on one line does not protect the tail (that is the
    pre-existing last-marker rule). The pin is that the formula itself is not
    swallowed into the previous printed line.
    """
    vocab = _vocab("ab")
    for math in ("$x_t$", "$5 + x$", r"$\alpha$"):
        text = f"lead a-\nsee {math}"
        assert reflow_native_prose(text, (_para("lead a-", f"see {math}"),), vocab) == text, math


def test_unbalanced_maths_dollar_still_protects_to_the_end() -> None:
    text = "open $x\na-\nb"
    assert reflow_native_prose(text, (_para("a-", "b"),), _vocab("ab")) == text


def test_a_page_mixing_currency_and_maths() -> None:
    """Currency above and below a balanced inline formula; only the formula blocks."""
    vocab = _vocab("ab ef")
    text = "costs $29.99 apiece a-\nb\n$x_t$\nbottom e-\nf"
    out = reflow_native_prose(
        text,
        (_para("costs $29.99 apiece a-", "b"), _para("$x_t$"), _para("bottom e-", "f")),
        vocab,
    )
    assert out == "costs $29.99 apiece ab\n$x_t$\nbottom ef"


def test_numeric_maths_opener_is_not_currency() -> None:
    """P0-1: ``$5 x`` / ``$5.`` are maths. If the opener were hidden, ``a-``/``b`` join."""
    text = "$5 x\na-\nb\n$"
    assert reflow_native_prose(text, (_para("a-", "b"),), _vocab("ab")) == text
    assert not _is_currency_dollar(text, 0)
    assert not _is_currency_dollar("$5.", 0)
    assert not _is_currency_dollar("$5 x", 0)
    eol = "$5\na-\nb\n$"
    assert reflow_native_prose(eol, (_para("a-", "b"),), _vocab("ab")) == eol
    assert not _is_currency_dollar(eol, 0)


def test_numeric_closers_do_not_balance_separate_formulas() -> None:
    """P0-2: operator-glued closers ``2$`` / ``3$`` must not hide and pair across formulas."""
    text = "$x+\na-\nb+2$\n$y+\na-\nb+3$"
    assert (
        reflow_native_prose(
            text,
            (_para("a-", "b+2$"), _para("a-", "b+3$")),
            _vocab("ab"),
        )
        == text
    )
    assert not _is_currency_dollar(text, text.index("2$") + 1)
    assert not _is_currency_dollar(text, text.index("3$") + 1)
    earlier = "$1$\n$x+\na-\nb+2$"
    assert reflow_native_prose(earlier, (_para("a-", "b+2$"),), _vocab("ab")) == earlier


def test_balanced_numeric_maths_leaves_the_tail_to_reflow() -> None:
    """P2: hiding the closer of ``$2$`` left an odd count and blocked the tail."""
    text = "$2$\na-\nb"
    assert reflow_native_prose(text, (_para("a-", "b"),), _vocab("ab")) == "$2$\nab"
    assert not _is_currency_dollar("$2$", 0)
    assert not _is_currency_dollar("$2$", 2)


def test_two_dollars_on_one_line_stay_protected() -> None:
    """Two ``$`` on one line are not currency; the line stays a marker line.

    The pair is balanced, so the tail may reflow (last-marker rule). The pin is
    that the line itself is not joined into the previous printed line.
    """
    text = "lead a-\n$5 per share and $10"
    assert (
        reflow_native_prose(text, (_para("lead a-", "$5 per share and $10"),), _vocab("ab")) == text
    )
    assert not _is_currency_dollar("$5 per share and $10", 0)
    assert not _is_currency_dollar("$5 per share and $10", 17)


def test_multiline_formula_starting_with_dollar_digit_stays_protected() -> None:
    """``$5`` then ``+ x$`` is maths; the opener stays a marker even without a closer."""
    text = "$5\n+ x$"
    assert reflow_native_prose(text, (_para("$5", "+ x$"),), _vocab("ab")) == text
    assert not _is_currency_dollar("$5\n+ x$", 0)
    open_only = "$5\na-\nb\n$"
    assert reflow_native_prose(open_only, (_para("a-", "b"),), _vocab("ab")) == open_only
    assert not _is_currency_dollar(open_only, 0)


def test_currency_classifier_rejects_maths_and_accepts_amounts() -> None:
    assert _is_currency_dollar("$29.99 apiece", 0)
    assert not _is_currency_dollar("$29.99", 0)
    assert _is_currency_dollar("US$5 each", 2)
    assert not _is_currency_dollar("US$5", 2)
    assert _is_currency_dollar("$1,200 million", 0)
    assert _is_currency_dollar("1987$.", 4)
    assert _is_currency_dollar("1987$ ", 4)
    assert not _is_currency_dollar("$x_t$", 0)
    assert not _is_currency_dollar("$5 + x$", 0)
    assert not _is_currency_dollar(r"$\alpha$", 0)
    assert not _is_currency_dollar("$x", 0)
    assert not _is_currency_dollar("$5%", 0)
    assert not _is_currency_dollar("$5x", 0)
