"""Source-evidence reject reasons use shared cascade constants."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.pipeline.agentic import SourceEvidenceTableJudge
from socr.routing.cascade_policy import (
    PAID_BLOCK_REASON_SOURCE_EVIDENCE,
    REJECT_REASON_SOURCE_EVIDENCE_PREFIX,
    blocks_paid_escalation,
)


def test_witnessed_contradiction_uses_shared_prefix() -> None:
    inner = MagicMock()
    judge = SourceEvidenceTableJudge(inner, get_fitz_page=lambda _n: MagicMock())
    output = PageOutput(
        page_num=1,
        text="| a | b |\n|---|---|\n| 1 | 2 |",
        status=PageStatus.SUCCESS,
        engine="qwen",
    )

    from socr.tables.source_evidence import SourceEvidenceResult

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "socr.tables.source_evidence.verify_scanned_table",
            lambda *_a, **_k: SourceEvidenceResult(
                verifiable=True,
                passed=False,
                reason="numeric tokens unsupported by page evidence: ['9']",
                cause="",
            ),
        )
        decision = judge.assess(output, PROFILE_QWEN_LOCAL)

    assert decision.accept is False
    assert decision.reason.startswith(REJECT_REASON_SOURCE_EVIDENCE_PREFIX)
    assert blocks_paid_escalation(decision.reason)
    from socr.routing.cascade_policy import (
        SparsePageSignals,
        paid_gemini_block_after_reject,
    )

    assert (
        paid_gemini_block_after_reject(decision.reason, SparsePageSignals(), audit_min_words=40)
        == PAID_BLOCK_REASON_SOURCE_EVIDENCE
    )
