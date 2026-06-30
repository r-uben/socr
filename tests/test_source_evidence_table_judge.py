"""GH-90: fail-closed source-evidence table judge for scanned pages.

Hermetic tests — no ollama, no GPU.  Covers:
- Scanned page + hallucinated markdown table → hard reject (not success).
- Born-digital page with matching native table → still passes via native verifier.
- Alpha-only tables: pass with label evidence, fail without support.
- Manifest scanned-table floor ships failure marker, not hallucinated text.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz

from socr.core.audit_log import AuditEvent
from socr.core.manifest import _winning_page_output, is_page_failed_marker
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.agentic import (
    AcceptDecision,
    HeuristicPageJudge,
    NativeTableVerifierJudge,
    SourceEvidenceTableJudge,
)
from socr.tables.source_evidence import (
    SourceEvidenceBundle,
    TableTokens,
    build_scanned_evidence,
    collect_table_tokens,
    page_has_native_words,
    verify_scanned_table,
    verify_table_tokens,
)

_PHYS_COL_GAP: float = 60.0


def _make_empty_page() -> fitz.Page:
    doc = fitz.open()
    return doc.new_page(width=500, height=700)


def _make_fitz_page_with_words(rows: list[list[tuple[float, str]]]) -> fitz.Page:
    doc = fitz.open()
    page = doc.new_page(width=700, height=900)
    for row_idx, cells in enumerate(rows):
        y = 100.0 + row_idx * 30
        for x, word in cells:
            page.insert_text((x, y), word, fontsize=9)
    return page


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    sep = "| " + " | ".join(["---"] * len(header)) + " |"
    lines = ["| " + " | ".join(header) + " |", sep]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# JCS-18-81 p6 style hallucination: fluent SaaS table on a tax-policy scan.
_HALLUCINATED_SAAS_TABLE = _md_table(
    ["Feature", "Cloud-Based", "Mobile App", "24/7 Support"],
    [
        ["Storage", "Unlimited", "Sync", "Yes"],
        ["Access", "Anywhere", "iOS/Android", "Live Chat"],
    ],
)

_ALPHA_ONLY_SUPPORTED = _md_table(
    ["Category", "Description"],
    [
        ["Revenue", "Taxable income"],
        ["Expense", "Deductible costs"],
    ],
)

_ALPHA_ONLY_UNSUPPORTED = _md_table(
    ["Feature", "Cloud-Based", "Mobile App"],
    [
        ["Storage", "Unlimited", "Sync"],
        ["Access", "Anywhere", "iOS/Android"],
    ],
)


def _make_output(page_num: int, text: str) -> PageOutput:
    return PageOutput(
        page_num=page_num,
        text=text,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )


def _make_handle(page_count: int = 1):
    from socr.core.document import DocumentHandle

    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        return DocumentHandle(path=Path("/tmp/fake.pdf"), page_count=page_count)


# --------------------------------------------------------------------------
# Unit tests — token collection and verification
# --------------------------------------------------------------------------


class TestCollectTableTokens:
    def test_numeric_tokens_from_table(self) -> None:
        text = _md_table(["var", "val"], [["gdp", "2.1"], ["cpi", "1.9"]])
        tokens = collect_table_tokens(text)
        assert tokens is not None
        assert tokens.has_numeric
        assert tokens.numeric["2.1"] == 1
        assert tokens.numeric["1.9"] == 1

    def test_alpha_only_detection(self) -> None:
        tokens = collect_table_tokens(_ALPHA_ONLY_SUPPORTED)
        assert tokens is not None
        assert tokens.is_alpha_only
        assert not tokens.has_numeric
        assert "revenue" in tokens.content
        assert "taxable" in tokens.content


class TestVerifyTableTokens:
    def test_alpha_only_passes_with_label_support(self) -> None:
        tokens = collect_table_tokens(_ALPHA_ONLY_SUPPORTED)
        bundle = SourceEvidenceBundle(
            content=frozenset({"revenue", "taxable", "income", "expense", "deductible", "costs"}),
            has_content_evidence=True,
        )
        result = verify_table_tokens(bundle, tokens)
        assert result.verifiable
        assert result.passed

    def test_alpha_only_fails_without_support(self) -> None:
        tokens = collect_table_tokens(_ALPHA_ONLY_UNSUPPORTED)
        bundle = SourceEvidenceBundle(
            content=frozenset({"revenue", "taxable", "income"}),
            has_content_evidence=True,
        )
        result = verify_table_tokens(bundle, tokens)
        assert result.verifiable
        assert not result.passed
        assert "unsupported" in result.reason

    def test_numeric_fails_when_unsupported(self) -> None:
        tokens = TableTokens(
            numeric=Counter({"2.1": 1, "99.9": 1}),
            content=frozenset(),
            has_numeric=True,
            is_alpha_only=False,
        )
        bundle = SourceEvidenceBundle(
            numeric=Counter({"2.1": 1}),
            has_content_evidence=True,
        )
        result = verify_table_tokens(bundle, tokens)
        assert not result.passed

    def test_unverifiable_when_no_evidence(self) -> None:
        tokens = collect_table_tokens(_HALLUCINATED_SAAS_TABLE)
        bundle = SourceEvidenceBundle(has_content_evidence=False)
        result = verify_table_tokens(bundle, tokens)
        assert not result.verifiable
        assert not result.passed


# --------------------------------------------------------------------------
# Judge integration
# --------------------------------------------------------------------------


class TestSourceEvidenceTableJudge:
    def _wrap_chain(
        self,
        fitz_page,
        *,
        is_table_page: bool = True,
        ocr_image_fn=None,
    ) -> tuple[SourceEvidenceTableJudge, MagicMock, list[AuditEvent]]:
        events: list[AuditEvent] = []
        inner = MagicMock()
        inner.assess.return_value = AcceptDecision(accept=True, reason="inner ok")

        native = NativeTableVerifierJudge(
            inner=inner,
            get_fitz_page=lambda pn: fitz_page,
            is_table_page=lambda pn: is_table_page,
            record_event=events.append,
        )
        judge = SourceEvidenceTableJudge(
            inner=native,
            get_fitz_page=lambda pn: fitz_page,
            record_event=events.append,
            ocr_image_fn=ocr_image_fn,
        )
        return judge, inner, events

    def test_scanned_hallucinated_table_rejected(self) -> None:
        """Scanned page with no native words: fluent fake table must fail closed."""
        fitz_page = _make_empty_page()
        assert not page_has_native_words(fitz_page)

        judge, inner, events = self._wrap_chain(fitz_page, ocr_image_fn=lambda pix: "")
        output = _make_output(1, _HALLUCINATED_SAAS_TABLE)
        decision = judge.assess(output, MagicMock())

        assert decision.accept is False
        assert "source_evidence_table" in decision.reason
        assert output.audit_passed is False
        assert output.status == PageStatus.ERROR
        assert output.failure_mode == FailureMode.HALLUCINATION
        inner.assess.assert_not_called()
        assert any(e.kind == "source_evidence_table_reject" for e in events)

    def test_born_digital_matching_table_passes(self) -> None:
        """Born-digital native table still passes through native verifier (no regression)."""
        native_rows = [
            [
                (200.0, "0.1"),
                (200.0 + _PHYS_COL_GAP, "0.2"),
                (200.0 + 2 * _PHYS_COL_GAP, "0.3"),
            ],
        ]
        fitz_page = _make_fitz_page_with_words(native_rows)
        output_text = _md_table(
            ["label", "c1", "c2", "c3"],
            [["row1", "0.1", "0.2", "0.3"]],
        )
        judge, inner, events = self._wrap_chain(fitz_page)
        output = _make_output(1, output_text)
        decision = judge.assess(output, MagicMock())

        assert decision.accept is True
        inner.assess.assert_not_called()
        assert any(e.kind == "native_table_verifier_exact_pass" for e in events)

    def test_alpha_only_passes_with_ocr_evidence(self) -> None:
        """Alpha-only scanned table passes when classical OCR supplies labels."""
        fitz_page = _make_empty_page()
        evidence_text = "Revenue from taxable income and Expense for deductible costs on this page."

        judge, inner, _events = self._wrap_chain(
            fitz_page,
            ocr_image_fn=lambda pix: evidence_text,
        )
        output = _make_output(1, _ALPHA_ONLY_SUPPORTED)
        decision = judge.assess(output, MagicMock())

        assert decision.accept is True
        inner.assess.assert_called_once()

    def test_alpha_only_fails_without_ocr_support(self) -> None:
        fitz_page = _make_empty_page()
        judge, inner, _events = self._wrap_chain(
            fitz_page,
            ocr_image_fn=lambda pix: "Unrelated prose about tax policy in 1981.",
        )
        output = _make_output(1, _ALPHA_ONLY_UNSUPPORTED)
        decision = judge.assess(output, MagicMock())

        assert decision.accept is False
        inner.assess.assert_not_called()

    def test_prose_without_tables_delegates_to_inner(self) -> None:
        fitz_page = _make_empty_page()
        judge, inner, _events = self._wrap_chain(fitz_page)
        output = _make_output(1, "Plain prose about monetary policy with no pipe tables.")
        decision = judge.assess(output, MagicMock())

        assert decision.accept is True
        inner.assess.assert_called_once()


class TestVerifyScannedTable:
    def test_defers_when_native_words_present(self) -> None:
        fitz_page = _make_fitz_page_with_words([[(100.0, "1.0"), (160.0, "2.0")]])
        result = verify_scanned_table(fitz_page, _HALLUCINATED_SAAS_TABLE)
        assert result.deferred

    def test_empty_page_hallucination_unverifiable(self) -> None:
        fitz_page = _make_empty_page()
        result = verify_scanned_table(
            fitz_page,
            _HALLUCINATED_SAAS_TABLE,
            ocr_image_fn=lambda pix: "",
        )
        assert not result.deferred
        assert not result.passed


# --------------------------------------------------------------------------
# Manifest scanned-table floor
# --------------------------------------------------------------------------


class TestScannedTableFloorManifest:
    def test_floor_ships_failure_marker_not_hallucination(self) -> None:
        state = DocumentState(handle=_make_handle(1))
        ps = state.pages[1]
        ps.is_born_digital = False
        ps.scanned_table_evidence_failed = True
        ps.attempts.append(_make_output(1, _HALLUCINATED_SAAS_TABLE))
        ps.best_output = PageOutput(
            page_num=1,
            text="[page 1 failed: unverifiable table — see image]\n\n",
            status=PageStatus.ERROR,
            engine="qwen",
            audit_passed=False,
            failure_mode=FailureMode.HALLUCINATION,
        )

        winner = _winning_page_output(state, 1, None)
        assert is_page_failed_marker(winner.text)
        assert "Cloud-Based" not in winner.text
        assert winner.audit_passed is False
        assert winner.status == PageStatus.ERROR

    def test_passing_best_output_still_wins(self) -> None:
        state = DocumentState(handle=_make_handle(1))
        ps = state.pages[1]
        ps.is_born_digital = True
        passing = _make_output(1, "Good OCR prose without tables.")
        ps.best_output = passing
        winner = _winning_page_output(state, 1, None)
        assert winner.text == passing.text
        assert winner.audit_passed is True


# --------------------------------------------------------------------------
# Orchestrator judge wiring
# --------------------------------------------------------------------------


class TestBuildPageJudgeWiring:
    def test_outermost_wrapper_is_source_evidence(self) -> None:
        from socr.core.config import PipelineConfig
        from socr.pipeline.orchestrator import UnifiedPipeline

        pipeline = UnifiedPipeline(
            PipelineConfig(quiet=True, tiered=False, judge_backend="heuristic")
        )
        state = DocumentState(handle=_make_handle(1))
        judge = pipeline._build_page_judge(state)

        assert isinstance(judge, SourceEvidenceTableJudge)
        assert isinstance(judge._inner, NativeTableVerifierJudge)
        assert isinstance(judge._inner._inner, HeuristicPageJudge)


# --------------------------------------------------------------------------
# Evidence builder smoke test (hermetic)
# --------------------------------------------------------------------------


def test_build_scanned_evidence_empty_scan_has_no_content() -> None:
    page = _make_empty_page()
    bundle = build_scanned_evidence(page, ocr_image_fn=lambda pix: "")
    assert not bundle.has_content_evidence
