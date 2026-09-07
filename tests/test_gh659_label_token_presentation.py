"""#659: the source-evidence content check must not fail a table closed over
markup that only LOOKS like unsupported content.

A VLM emits a scanned table's row-label indentation as the literal HTML
entity ``&nbsp;`` (see #624) and a rule row's dashes glued onto the word
before them (``Settlements--``). ``collect_table_tokens`` /
``_tokens_from_plain_text`` tokenised both raw, so ``nbsp`` and
``settlements--`` became content-label tokens that no OCR witness --
reading the actual page pixels, which never contain the string "nbsp" or a
trailing double-dash -- could ever confirm. ``_content_supported`` then
rejected a table whose 62/62 numbers were fully corroborated, purely
because of presentation. Measured 2026-09-08 on the Fed 1977/1982/1990
swap-line minutes p3 (see the issue for the exact numbers).

Fix: decode HTML entities and strip leading/trailing dashes before adding a
match to the content-label set (``_content_token``), and treat a genuinely
missing label token as UNVERIFIED (flag, ship) rather than REJECTED when
every numeric token is supported -- a fabricated label paired with an
unsupported number, or unsupported numbers alone, must still reject.

Hermetic: pure unit tests against ``source_evidence.py``'s dataclasses and
functions, plus one test driving ``SourceEvidenceTableJudge`` (agentic.py)
with the classical-OCR witness patched, exactly like the #658 test file
patches it. No ollama, no provider ladder, no real tesseract.
"""

from __future__ import annotations

from collections import Counter
from unittest.mock import MagicMock, patch

import fitz
import pytest

from socr.core.audit_log import AuditEvent
from socr.core.result import PageOutput, PageStatus
from socr.pipeline.agentic import SourceEvidenceTableJudge
from socr.tables.source_evidence import (
    WITNESS_READING,
    SourceEvidenceBundle,
    collect_table_tokens,
    verify_table_tokens,
)

# --------------------------------------------------------------------------
# (a) &nbsp; + trailing-dash label, everything else corroborated -> PASS.
# --------------------------------------------------------------------------


def test_nbsp_and_trailing_dash_label_no_longer_rejects_a_supported_table() -> None:
    markdown = "| Counterparty | Amount |\n| --- | --- |\n| &nbsp;&nbsp;Settlements-- | 62.5 |\n"
    tokens = collect_table_tokens(markdown)
    assert tokens is not None
    # Precondition: without the fix this would be {"nbsp", "settlements--"},
    # neither of which any bare-word evidence could ever contain.
    assert tokens.content == frozenset({"settlements"})

    bundle = SourceEvidenceBundle(
        numeric=Counter({"62.5": 1}),
        content=frozenset({"settlements"}),
        has_content_evidence=True,
    )
    result = verify_table_tokens(bundle, tokens)
    assert result.verifiable is True
    assert result.passed is True
    assert result.content_unverified == ""


# --------------------------------------------------------------------------
# (b) fabricated label AND an unsupported number -> still REJECTED.
# --------------------------------------------------------------------------


def test_fabricated_label_with_unsupported_number_still_rejects() -> None:
    markdown = "| Counterparty | Amount |\n| --- | --- |\n| Fabricatedbank | 999.9 |\n"
    tokens = collect_table_tokens(markdown)
    assert tokens is not None

    bundle = SourceEvidenceBundle(
        numeric=Counter({"12.5": 1}),
        content=frozenset({"realbank"}),
        has_content_evidence=True,
    )
    result = verify_table_tokens(bundle, tokens)
    assert result.verifiable is True
    assert result.passed is False
    assert "numeric tokens unsupported" in result.reason


def test_unsupported_numbers_alone_still_rejects() -> None:
    """No content tokens at all -- the pre-#659 numeric fail-closed path must
    be completely untouched by the label-unverified carve-out.
    """
    markdown = "| Counterparty | Amount |\n| --- | --- |\n| 999.9 | 1.0 |\n"
    tokens = collect_table_tokens(markdown)
    assert tokens is not None
    assert not tokens.content

    bundle = SourceEvidenceBundle(
        numeric=Counter({"12.5": 1}), content=frozenset(), has_content_evidence=True
    )
    result = verify_table_tokens(bundle, tokens)
    assert result.passed is False
    assert result.content_unverified == ""


# --------------------------------------------------------------------------
# (c) numbers all supported, one REAL label word absent -> flagged, not
#     rejected, and the flag is visible on the result object.
# --------------------------------------------------------------------------


def test_one_missing_real_label_word_flags_unverified_instead_of_rejecting() -> None:
    markdown = "| Counterparty | Amount |\n| --- | --- |\n| Realbank Missingword | 62.5 |\n"
    tokens = collect_table_tokens(markdown)
    assert tokens is not None
    assert tokens.content == frozenset({"realbank", "missingword"})

    bundle = SourceEvidenceBundle(
        numeric=Counter({"62.5": 1}),
        content=frozenset({"realbank"}),  # "missingword" absent from evidence
        has_content_evidence=True,
    )
    result = verify_table_tokens(bundle, tokens)
    assert result.verifiable is True
    assert result.passed is True
    assert result.content_unverified != ""
    assert "missingword" in result.content_unverified


def test_pure_label_table_with_missing_word_still_rejects() -> None:
    """No numeric token exists to anchor the #659 trade-off, so a pure-label
    table keeps the pre-#659 reject behaviour (``is_alpha_only`` branch).
    """
    markdown = "| Counterparty |\n| --- |\n| Realbank Missingword |\n"
    tokens = collect_table_tokens(markdown)
    assert tokens is not None
    assert tokens.is_alpha_only is True

    bundle = SourceEvidenceBundle(
        numeric=Counter(), content=frozenset({"realbank"}), has_content_evidence=True
    )
    result = verify_table_tokens(bundle, tokens)
    assert result.passed is False
    assert result.content_unverified == ""


# --------------------------------------------------------------------------
# (d) the flag reaches the page audit events via agentic.py.
# --------------------------------------------------------------------------


def _scanned_page() -> fitz.Page:
    doc = fitz.open()
    return doc.new_page(width=500, height=700)


def test_label_unverified_flag_reaches_the_audit_events() -> None:
    candidate_table = "| Counterparty | Amount |\n| --- | --- |\n| Bundesbank Reserve | 62.5 |\n"
    # A real witness reading that confirms the number and one label word, but
    # not "reserve" -- e.g. it was clipped or misread by the classical OCR.
    witness_text = "Counterparty Amount Bundesbank 62.5"

    events: list[AuditEvent] = []
    inner = MagicMock()
    inner.assess.return_value = "inner-accept-sentinel"
    page = _scanned_page()
    judge = SourceEvidenceTableJudge(
        inner=inner,
        get_fitz_page=lambda pn: page,
        record_event=events.append,
        native_trusted=lambda pn: False,
    )
    output = PageOutput(
        page_num=5,
        text=candidate_table,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
        confidence=0.9,
    )
    with patch(
        "socr.tables.source_evidence.classical_ocr_with_state",
        return_value=(witness_text, WITNESS_READING, ""),
    ):
        decision = judge.assess(output, MagicMock())

    # The table shipped: the inner judge was consulted (not a reject path).
    inner.assess.assert_called_once()
    assert decision == "inner-accept-sentinel"

    label_events = [e for e in events if e.kind == "source_evidence_table_label_unverified"]
    assert len(label_events) == 1
    assert label_events[0].page_num == 5
    assert "reserve" in label_events[0].detail
    # The base reject kind must NOT also fire -- this page was accepted.
    assert not any(e.kind == "source_evidence_table_reject" for e in events)

    # Astra review round 1 (P1): the event alone is decoration with no
    # consumer. The output itself must carry a standing, candidate-associated
    # marker -- WARNING status, an audit note, and the field the finalization
    # note/CLI/tables_trust reporting reads -- without flipping the winner
    # selector (``audit_passed`` stays whatever it was).
    assert output.status is PageStatus.WARNING
    assert output.audit_passed is True
    assert output.table_label_unverified
    assert "reserve" in output.table_label_unverified
    assert any("unverified" in note for note in output.audit_notes)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
