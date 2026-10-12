"""#1098: a ``|`` inside LaTeX maths must never form a Markdown table.

Scoped to ``collect_table_tokens`` / ``verify_scanned_table``. The shared
``find_table_blocks`` is unchanged. Round-4 mutants: drop the letter/command
interior rule → compact ``$5|$10`` is excluded; drop odd-backslash parity →
``C:\\\\|99.9`` is excluded.
"""

from __future__ import annotations

from collections import Counter

from socr.tables.reconcile import find_table_blocks
from socr.tables.source_evidence import (
    SourceEvidenceBundle,
    collect_table_tokens,
    verify_scanned_table,
    verify_table_tokens,
)


def _empty_page():
    import fitz

    return fitz.open().new_page(width=500, height=700)


def _invented_number_is_rejected(text: str) -> None:
    tokens = collect_table_tokens(text)
    assert tokens is not None
    assert tokens.numeric["99.9"] == 1
    result = verify_table_tokens(
        SourceEvidenceBundle(numeric=Counter({"1.0": 1}), has_content_evidence=True),
        tokens,
    )
    assert not result.passed
    assert "unsupported" in result.reason


def test_loaded_source_is_this_checkout() -> None:
    import socr
    from pathlib import Path

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def test_conditional_probability_is_not_a_table() -> None:
    text = "$P(A|B)$\n$P(B|A) = P(A)$\n$P(A|B) = \\frac{P(B|A)}{P(B)}$"
    assert collect_table_tokens(text) is None
    assert find_table_blocks(text), "shared finder still sees the pipe run"


def test_left_right_abs_is_not_a_table() -> None:
    text = r"$\left|x\right|$" + "\n" + r"$\left|y\right|$"
    assert collect_table_tokens(text) is None


def test_norm_pipes_are_not_a_table() -> None:
    text = r"$\|v\|_2$" + "\n" + r"$\|w\|_2$"
    assert collect_table_tokens(text) is None


def test_maths_only_page_is_not_judged_by_the_table_check() -> None:
    page = _empty_page()
    result = verify_scanned_table(page, "$P(A|B)$\n$P(C|D)$", native_trusted=False)
    assert result.passed
    assert result.deferred
    assert "no markdown table" in result.reason


def test_currency_cell_is_still_judged() -> None:
    """P0-1: ``$29.99`` is not a maths opener; 99.9 must still be checked."""
    _invented_number_is_rejected("| apple | $29.99 | 99.9 |\n| pear | 1.0 | 2.0 |\n")


def test_two_currency_cells_on_one_row_are_still_judged() -> None:
    """Spaced ``$5 | $10`` is currency: interior has no letter or command."""
    _invented_number_is_rejected("$5 | $10\n$1 | $99.9\n")


def test_compact_currency_cells_are_still_judged() -> None:
    """``$5|$10`` has no `` | ``; interior is digits and a pipe, not maths."""
    _invented_number_is_rejected("$5|$10\n$1|$99.9\n")


def test_even_backslash_run_is_not_an_escaped_pipe() -> None:
    """Two literal backslashes before ``|`` are a path, not ``\\|``."""
    _invented_number_is_rejected("C:\\\\|99.9\nD:\\\\|1.0\n")


def test_unclosed_math_before_a_table_does_not_mask_it() -> None:
    """P0-2: an unclosed opener earlier on the page must not hide a later table."""
    table = "| var | val |\n| --- | --- |\n| gdp | 99.9 |\n"
    for prefix in ("\\( x\n\n", "$$\n\n", "\\[\n\n", "\\begin{align}\nx &= 1\n\n"):
        _invented_number_is_rejected(prefix + table)
    _invented_number_is_rejected("$$\n" + table + "$$\n")


def test_table_inside_center_environment_is_still_judged() -> None:
    """P0-3: ``\\begin{center}`` is not maths."""
    _invented_number_is_rejected(
        "\\begin{center}\n| var | val |\n| --- | --- |\n| gdp | 99.9 |\n\\end{center}\n"
    )


def test_separator_table_of_maths_pipes_is_still_judged() -> None:
    """A ``|---|`` between ``$P(A|B)$`` lines is a table. Mutant that drops (a) fails."""
    text = "$P(A|B)$\n|---|\n$P(C|D)$"
    assert collect_table_tokens(text) is not None


def test_real_pipe_table_next_to_maths_still_counts() -> None:
    text = (
        "$P(A|B)$ is the posterior.\n\n| var | val |\n| --- | --- |\n| gdp | 2.1 |\n| cpi | 1.9 |\n"
    )
    tokens = collect_table_tokens(text)
    assert tokens is not None
    assert tokens.numeric["2.1"] == 1
    assert tokens.numeric["1.9"] == 1


def test_separatorless_pipe_table_is_still_judged() -> None:
    text = "| gdp | 2.1 |\n| cpi | 99.9 |\n"
    tokens = collect_table_tokens(text)
    assert tokens is not None
    assert tokens.numeric["2.1"] == 1
    assert tokens.numeric["99.9"] == 1
    result = verify_table_tokens(
        SourceEvidenceBundle(numeric=Counter({"2.1": 1}), has_content_evidence=True),
        tokens,
    )
    assert not result.passed
    assert "unsupported" in result.reason


def test_currency_dollar_does_not_swallow_a_later_table() -> None:
    text = "The product is priced at $29.99 per month.\n\n| Feature | Price |\n| --- | --- |\n| Cloud | 10 |\n"
    tokens = collect_table_tokens(text)
    assert tokens is not None
    assert tokens.numeric["10"] == 1
