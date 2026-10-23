"""JCX-81-03 page 1: born-digital two-panel revenue table fail-closed regression.

The June 2026 qwen reading kept every number it wrote but dropped whole Senate
rows and spanned-text cells; the native verifier saw the gap yet the page still
shipped SUCCESS / audit_passed=True. These tests pin the fix on current main
using the public JCT PDF fixture and recorded model markdown (no live model).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import fitz
import pytest

from socr.core.result import PageOutput, PageStatus
from socr.pipeline.agentic import AcceptDecision, NativeTableVerifierJudge
from socr.tables.native_verifier import (
    COMPLETENESS_ROW_GAP_HARD_FAIL,
    verify_native_table,
)
from socr.tables.row_subtotal_check import advise_five_year_subtotals

FIXTURES = Path(__file__).parent / "fixtures" / "jct_jcx_81_03"
PDF = FIXTURES / "jcx_81_03_p1.pdf"
SIDECAR = FIXTURES / "qwen_p1_sidecar.json"

SENATE_ROW_VALUES = ("-16", "-34", "-290", "31", "46", "402")


@pytest.fixture(scope="module")
def qwen_markdown() -> str:
    data = json.loads(SIDECAR.read_text(encoding="utf-8"))
    return data["winning_output"]["text"]


@pytest.fixture(scope="module")
def jcx_page():
    doc = fitz.open(PDF)
    page = doc[0]
    yield page
    doc.close()


class TestJCX81NativeVerifier:
    def test_recorded_qwen_markdown_missing_senate_row_values(self, qwen_markdown: str) -> None:
        assert "-290" not in qwen_markdown
        assert "402" not in qwen_markdown
        ethanol_row = next(
            line for line in qwen_markdown.splitlines() if "small ethanol producer" in line.lower()
        )
        assert "-16" not in ethanol_row
        assert "-34" not in ethanol_row

    def test_verifier_hard_fails_completeness_gap(self, jcx_page, qwen_markdown: str) -> None:
        vr = verify_native_table(jcx_page, qwen_markdown)
        assert vr.hard_fail is True, vr.reason
        assert "value_guard_completeness" in vr.reason
        assert vr.row_count_warn is True
        gap = vr.drifted_rows[0].get("row_gap", 0)
        assert gap >= COMPLETENESS_ROW_GAP_HARD_FAIL

    @pytest.mark.parametrize("token", SENATE_ROW_VALUES)
    def test_native_pdf_contains_senate_tokens(self, jcx_page, token: str) -> None:
        words = jcx_page.get_text("words")
        assert any(w[4] == token for w in words), f"{token!r} not in text layer"


class TestJCX81AgenticJudge:
    def test_inner_judge_accept_still_rejects_when_completeness_fails(
        self, jcx_page, qwen_markdown: str
    ) -> None:
        inner = MagicMock()
        inner.assess.return_value = AcceptDecision(
            accept=True, reason="mock inner accept", confidence=0.95
        )
        out = PageOutput(
            page_num=1,
            text=qwen_markdown,
            status=PageStatus.SUCCESS,
            engine="qwen",
            confidence=0.9,
        )
        judge = NativeTableVerifierJudge(
            inner=inner,
            get_fitz_page=lambda _pn: jcx_page,
            is_table_page=lambda _pn: True,
            record_event=lambda _ev: None,
            native_layer_found=lambda _pn: True,
        )
        decision = judge.assess(out, MagicMock())
        assert decision.accept is False
        assert "value_guard_completeness" in decision.reason or "native_table_verifier" in (
            decision.reason
        )


class TestJCX81SubtotalAdvisory:
    def test_five_year_subtotals_mostly_match_on_recorded_table(self, qwen_markdown: str) -> None:
        """Advisory sum check: on this page most House rows reconcile within ±2."""
        advisories = advise_five_year_subtotals(qwen_markdown, tolerance=2.0)
        # Spanned / blank Senate rows and the malformed header produce some noise;
        # the check must not fire on every row.
        assert len(advisories) < 15
