"""Source-evidence numeric normalization: thousands, negatives, percent, footnote marks."""

from __future__ import annotations

from collections import Counter

from socr.tables.source_evidence import (
    SourceEvidenceBundle,
    collect_table_tokens,
    verify_table_tokens,
)


def test_thousands_separators_match_evidence() -> None:
    md = "| x | y |\n|---|---|\n| a | 1,234 |\n"
    tokens = collect_table_tokens(md)
    assert tokens is not None
    bundle = SourceEvidenceBundle(
        numeric=Counter({"1234": 1}), content=frozenset(), has_content_evidence=True
    )
    result = verify_table_tokens(bundle, tokens)
    assert result.passed is True


def test_thin_space_thousands_match_evidence() -> None:
    md = "| x | y |\n|---|---|\n| a | 1\u2009234 |\n"
    tokens = collect_table_tokens(md)
    assert tokens is not None
    bundle = SourceEvidenceBundle(
        numeric=Counter({"1234": 1}), content=frozenset(), has_content_evidence=True
    )
    assert verify_table_tokens(bundle, tokens).passed is True


def test_parentheses_negative_and_percent() -> None:
    md = "| x | y |\n|---|---|\n| a | (12.5) |\n| b | 9% |\n"
    tokens = collect_table_tokens(md)
    assert tokens is not None
    bundle = SourceEvidenceBundle(
        numeric=Counter({"12.5": 1, "9": 1}),
        content=frozenset(),
        has_content_evidence=True,
    )
    assert verify_table_tokens(bundle, tokens).passed is True


def test_footnote_star_still_verifies_bare_number() -> None:
    md = "| x | y |\n|---|---|\n| a | 12* |\n"
    tokens = collect_table_tokens(md)
    assert tokens is not None
    bundle = SourceEvidenceBundle(
        numeric=Counter({"12": 1}), content=frozenset(), has_content_evidence=True
    )
    assert verify_table_tokens(bundle, tokens).passed is True


def test_wrong_number_still_rejects() -> None:
    md = "| x | y |\n|---|---|\n| a | 1,234 |\n"
    tokens = collect_table_tokens(md)
    assert tokens is not None
    bundle = SourceEvidenceBundle(
        numeric=Counter({"9999": 1}), content=frozenset(), has_content_evidence=True
    )
    result = verify_table_tokens(bundle, tokens)
    assert result.passed is False
    assert "1234" in result.reason or "unsupported" in result.reason.lower()
