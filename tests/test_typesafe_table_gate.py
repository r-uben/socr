"""Typesafe confirmation for vision-model tables on unsure native pages."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import fitz

from socr.core.born_digital import DocumentAssessment, PageAssessment
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.agentic import AcceptDecision, NativeTableVerifierJudge
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.typesafe import (
    TABLE_MATCH_QUESTION,
    TypesafeGate,
    needs_typesafe_confirmation,
    stored_words_match_table,
    typesafe_table_matches_page,
)

_TABLE = (
    "| Counterparty | Amount | Drawn |\n"
    "| --- | --- | --- |\n"
    "| Bundesbank | 62.5 | 12.5 |\n"
    "| Bank of Japan | 67.0 | 15.0 |\n"
)


def _mock_response(payload: dict):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    return resp


def _page_with_stored_table_words() -> fitz.Page:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Bundesbank 62.5 12.5", fontsize=10)
    page.insert_text((72, 120), "Bank of Japan 67.0 15.0", fontsize=10)
    return page


class TestTypesafeHelpers:
    def test_stored_words_match_skips_typesafe(self) -> None:
        page = _page_with_stored_table_words()
        assert stored_words_match_table(page, _TABLE) is True
        assert needs_typesafe_confirmation(page, _TABLE, vision_model_output=True) is False

    def test_typesafe_http_yes(self) -> None:
        page = _page_with_stored_table_words()
        calls: list[dict] = []

        def post_fn(url, *, json, headers, timeout):
            calls.append({"url": url, "json": json, "headers": headers})
            return _mock_response({"answers": ["yes"]})

        gate = TypesafeGate(api_key="test-key", post_fn=post_fn)
        assert gate.confirm(page, _TABLE) is True
        assert calls[0]["json"]["model"] == "jev"
        assert calls[0]["headers"]["Authorization"] == "Bearer test-key"
        assert TABLE_MATCH_QUESTION in calls[0]["json"]["questions"]

    def test_typesafe_http_no_fails_closed(self) -> None:
        page = _page_with_stored_table_words()

        def post_fn(url, *, json, headers, timeout):
            return _mock_response({"answers": ["no"]})

        assert TypesafeGate(api_key="test-key", post_fn=post_fn).confirm(page, _TABLE) is False

    def test_missing_api_key_fails_closed(self) -> None:
        page = _page_with_stored_table_words()
        with patch.dict("os.environ", {}, clear=True):
            assert typesafe_table_matches_page(page, _TABLE, api_key=None) is False


def _judge_with_typesafe(tmp_path, *, confirm: bool):
    pdf = tmp_path / "rotated.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Bundesbank 62.5 12.5", fontsize=10)
    page.insert_text((72, 120), "Bank of Japan 67.0 15.0", fontsize=10)
    doc.save(pdf)
    doc.close()

    gate = MagicMock()
    gate.confirm = MagicMock(return_value=confirm)

    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            judge_backend="heuristic",
        )
    )
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    state.pages[1].is_born_digital = True
    state.pages[1].has_tables = True
    pipeline._last_assessment = DocumentAssessment(
        path=pdf,
        pages=[
            PageAssessment(
                page_num=1,
                is_born_digital=True,
                native_text="",
                confidence=1.0,
                has_tables=True,
                native_table_lane_refused=True,
            )
        ],
    )

    def get_fitz_page(page_num: int):
        d = fitz.open(pdf)
        return d[page_num - 1]

    def record_event(event) -> None:
        state.events.append(event)

    inner = MagicMock()
    inner.assess = MagicMock(return_value=AcceptDecision(accept=True, reason="test", confidence=1.0))

    judge = NativeTableVerifierJudge(
        inner=inner,
        get_fitz_page=get_fitz_page,
        is_table_page=lambda _n: True,
        record_event=record_event,
        typesafe_gate=gate,
    )
    return judge, state


def test_stored_words_match_never_calls_typesafe(tmp_path) -> None:
    judge, _state = _judge_with_typesafe(tmp_path, confirm=True)
    output = PageOutput(
        page_num=1,
        text=_TABLE,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    decision = judge.assess(output, PROFILE_QWEN_LOCAL)
    assert decision.accept is True
    judge._typesafe_gate.confirm.assert_not_called()


def test_typesafe_reject_on_vision_table(tmp_path) -> None:
    from socr.tables.native_verifier import VerifierResult, VerifierState

    judge, state = _judge_with_typesafe(tmp_path, confirm=False)
    mismatched = (
        "| Counterparty | Amount | Drawn |\n"
        "| --- | --- | --- |\n"
        "| Bundesbank | 99.9 | 12.5 |\n"
        "| Bank of Japan | 67.0 | 15.0 |\n"
    )
    output = PageOutput(
        page_num=1,
        text=mismatched,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    warn_result = VerifierResult()
    warn_result.warn = True
    warn_result.state = VerifierState.AMBIGUOUS
    warn_result.reason = "lane gap (test)"
    before = len(state.events)
    with (
        patch(
            "socr.tables.native_verifier.verify_native_table",
            return_value=warn_result,
        ),
        patch.object(judge, "_maybe_repair_collapsed_headers", side_effect=lambda _p, _o, vr: vr),
    ):
        decision = judge.assess(output, PROFILE_QWEN_LOCAL)
    assert decision.accept is False
    kinds = {e.kind for e in state.events[before:]}
    assert "typesafe_table_reject" in kinds
    judge._typesafe_gate.confirm.assert_called_once()


def test_process_wires_typesafe_gate(tmp_path) -> None:
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "hello", fontsize=12)
    doc.save(pdf)
    doc.close()

    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN, enabled_engines=[EngineType.QWEN], quiet=True
        )
    )
    captured: list = []

    class _RecordingGate:
        def confirm(self, page, markdown: str) -> bool:
            captured.append(markdown)
            return True

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch(
            "socr.tables.typesafe.TypesafeGate",
            side_effect=lambda **kwargs: _RecordingGate(),
        ),
    ):
        pipeline.process(pdf, tmp_path / "out")
    # Empty ladder: no model table path exercised; gate must not run.
    assert captured == []
