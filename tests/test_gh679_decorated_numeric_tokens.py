"""#679: ``collect_table_tokens`` still skipped entity/dash-decorated NUMBERS
after #668 closed the equivalent gap for labels.

``is_numeric_token(cell)`` ran on the raw cell; ``html.unescape`` only
happened later, in the label branch, past the ``row_idx == 0`` continue. So
``&nbsp;62.5`` and ``62.5--`` never entered ``raw_numeric`` -- the numeric
multiset the fail-closed gate actually checks against evidence
(``verify_table_tokens`` / ``_multiset_supported``). That is fail-open on the
citable values: a decorated number ships unverified while the table can
still pass.

The fix (``_normalize_cell``) is applied identically on BOTH the candidate
side (``collect_table_tokens``) and the evidence side
(``_tokens_from_plain_text``) -- see ``test_symmetric_decoration_still_passes``
for why asymmetric normalization is itself a regression (measured
2026-09-16, docs/log/2026-09-16_679.md): normalizing only the candidate would
make a candidate that genuinely AGREES with an evidence text carrying the
same raw decoration look unsupported.

Hermetic: pure unit tests against ``source_evidence.py``'s public functions.
No ollama, no provider ladder, no real tesseract, no fitz page I/O.
"""

from __future__ import annotations

from collections import Counter

import pytest

from socr.tables.source_evidence import (
    SourceEvidenceBundle,
    collect_table_tokens,
    verify_table_tokens,
)

# --------------------------------------------------------------------------
# (a) &nbsp; and trailing "--" decorated NUMBERS land in the numeric
#     multiset the gate actually checks -- the ticket's core hole.
# --------------------------------------------------------------------------


def test_nbsp_decorated_number_enters_the_numeric_multiset() -> None:
    markdown = "| Item | Value |\n| --- | --- |\n| Settlements | &nbsp;62.5 |\n"
    tokens = collect_table_tokens(markdown)
    assert tokens is not None
    assert tokens.numeric == Counter({"62.5": 1})


def test_trailing_dash_decorated_number_enters_the_numeric_multiset() -> None:
    markdown = "| Item | Value |\n| --- | --- |\n| Reserves | 41.3-- |\n"
    tokens = collect_table_tokens(markdown)
    assert tokens is not None
    assert tokens.numeric == Counter({"41.3": 1})


def test_unicode_em_dash_entity_decoration_also_enters_the_multiset() -> None:
    """``&mdash;`` decodes (via ``html.unescape``) to U+2014 EM DASH, not
    ASCII ``-``. A strip that only recognised ASCII would leave the decoded
    glyph trailing and still unmatched -- half-closing this exact ticket.
    """
    markdown = "| Item | Value |\n| --- | --- |\n| Reserves | &nbsp;62.5&mdash; |\n"
    tokens = collect_table_tokens(markdown)
    assert tokens is not None
    assert tokens.numeric == Counter({"62.5": 1})


def test_decorated_number_now_passes_when_evidence_confirms_the_bare_value() -> None:
    markdown = "| Item | Value |\n| --- | --- |\n| Reserves | 41.3-- |\n"
    tokens = collect_table_tokens(markdown)
    assert tokens is not None

    bundle = SourceEvidenceBundle(
        numeric=Counter({"41.3": 1}), content=frozenset(), has_content_evidence=True
    )
    result = verify_table_tokens(bundle, tokens)
    assert result.verifiable is True
    assert result.passed is True


# --------------------------------------------------------------------------
# (b) a decorated number the evidence does NOT support still REJECTS -- the
#     fix must not turn the gate off, only stop it skipping the check.
# --------------------------------------------------------------------------


def test_unsupported_decorated_number_still_rejects() -> None:
    markdown = "| Item | Value |\n| --- | --- |\n| Fabricated | 999.9-- |\n"
    tokens = collect_table_tokens(markdown)
    assert tokens is not None
    assert tokens.numeric == Counter({"999.9": 1})

    bundle = SourceEvidenceBundle(
        numeric=Counter({"12.5": 1}), content=frozenset(), has_content_evidence=True
    )
    result = verify_table_tokens(bundle, tokens)
    assert result.verifiable is True
    assert result.passed is False
    assert "999.9" in result.reason


# --------------------------------------------------------------------------
# (c) the negative sign is a VALUE, not decoration -- must never be stripped.
# --------------------------------------------------------------------------


def test_leading_minus_sign_is_preserved_not_stripped_as_decoration() -> None:
    markdown = "| Item | Value |\n| --- | --- |\n| Delta | -5.2 |\n"
    tokens = collect_table_tokens(markdown)
    assert tokens is not None
    assert tokens.numeric == Counter({"-5.2": 1})


def test_leading_true_minus_sign_is_preserved_not_stripped_as_decoration() -> None:
    """U+2212 MINUS SIGN, the typeset minus a VLM emits for negative values
    (distinct from the presentation-dash glyphs this ticket strips). A LEADING
    occurrence is a sign, not decoration, and stripping it would silently flip
    the value's sign -- worse than the bug this ticket fixes.
    """
    markdown = "| Item | Value |\n| --- | --- |\n| Delta | −5.2 |\n"
    tokens = collect_table_tokens(markdown)
    assert tokens is not None
    assert tokens.numeric == Counter({"-5.2": 1})


# --------------------------------------------------------------------------
# (c2) the full Unicode dash-decoration class, not just ASCII "--". An HTML
#      entity (``&mdash;``, ``&ndash;``) decodes to a Unicode glyph via
#      ``html.unescape`` -- a strip that only recognised ASCII "-" leaves
#      that decoded glyph trailing and still unmatched, half-closing the
#      ticket. Each of these pins independently fails against the earlier
#      ASCII-only ``rstrip("-")`` (verified 2026-09-16 before widening the
#      class, see docs/log/2026-09-16_679.md).
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "decorated_cell",
    [
        "62.5‐‐",  # HYPHEN x2
        "62.5‑‑",  # NON-BREAKING HYPHEN x2
        "62.5‒‒",  # FIGURE DASH x2
        "62.5–",  # EN DASH (what &ndash; decodes to)
        "62.5—",  # EM DASH (what &mdash; decodes to)
        "62.5――",  # HORIZONTAL BAR x2
        "&nbsp;62.5&mdash;",  # entity form, end to end through html.unescape
        "62.5&ndash;",
    ],
)
def test_unicode_dash_decoration_variants_all_enter_the_numeric_multiset(
    decorated_cell: str,
) -> None:
    markdown = f"| Item | Value |\n| --- | --- |\n| Reserves | {decorated_cell} |\n"
    tokens = collect_table_tokens(markdown)
    assert tokens is not None
    assert tokens.numeric == Counter({"62.5": 1}), decorated_cell


# --------------------------------------------------------------------------
# (d) symmetric decoration on both sides: a candidate that genuinely AGREES
#     with a raw evidence text carrying the identical dash decoration must
#     still pass -- proves the fix was NOT applied candidate-only (which
#     regresses this exact case, see docs/log/2026-09-16_679.md).
# --------------------------------------------------------------------------


def test_symmetric_decoration_still_passes() -> None:
    """THE SOLE GUARD for the evidence-side half of ``_normalize_cell``
    (``_tokens_from_plain_text``). Reverting only that half (leaving
    ``collect_table_tokens`` fixed) makes exactly this one test fail out of
    the 16 in this file -- measured 2026-09-16, mutation guard round 2,
    docs/log/2026-09-16_679.md. It is the entire reason (b) -- normalizing
    BOTH sides -- was chosen over (a) -- candidate-only: (a) measurably
    turns a page whose raw evidence text genuinely carries the SAME
    decoration as the candidate into an active reject, because the
    candidate's now-bare token stops matching an evidence multiset still
    keyed on the raw, undecorated one. If this test is weakened or deleted
    during a later refactor, that (a) regression can walk back in with an
    otherwise-green suite -- read the log entry above before touching it.
    """
    from socr.tables.source_evidence import _tokens_from_plain_text

    markdown = "| Item | Value |\n| --- | --- |\n| Reserves | 41.3-- |\n"
    tokens = collect_table_tokens(markdown)
    assert tokens is not None

    # The page's own raw text carries the identical decoration a VLM copied
    # verbatim from the printed table.
    raw_evidence_text = "Reserves 41.3-- as printed on the page."
    numeric_evidence, content_evidence = _tokens_from_plain_text(raw_evidence_text)
    assert numeric_evidence == Counter({"41.3": 1})  # evidence-side gets the same strip

    bundle = SourceEvidenceBundle(
        numeric=numeric_evidence,
        content=frozenset(content_evidence),
        has_content_evidence=True,
    )
    result = verify_table_tokens(bundle, tokens)
    assert result.verifiable is True
    assert result.passed is True


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
