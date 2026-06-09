"""P1 Stage-1 regressions (issue #39): the cascade's economics.

Pins: (1) paid rungs are reachable — escalation is bounded by the ladder and
the budget, never a retry count; (2) the budget is enforced BEFORE each paid
rung; (3) token-limit truncation is TRUNCATED, not success; (4) legitimately
sparse pages stop hard-failing the word-count gate.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from socr.audit.heuristics import HeuristicsChecker
from socr.audit.scorer import FailureModeScorer
from socr.core.config import EngineType, PipelineConfig
from socr.core.providers import provider_ladder
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.pipeline.agentic import AcceptDecision, route_page

# Five free locals + two paid cloud rungs — the configuration under which the
# old max_retries+1=3 cap made GEMINI/MISTRAL mathematically unreachable.
FULL_LADDER = provider_ladder(
    {
        EngineType.QWEN,
        EngineType.GLM,
        EngineType.NOUGAT,
        EngineType.DEEPSEEK,
        EngineType.MARKER,
        EngineType.GEMINI,
        EngineType.MISTRAL,
    }
)


def _run(text="some ocr text"):
    def run(engine: EngineType, page_num: int) -> PageOutput:
        return PageOutput(
            page_num=page_num,
            text=text,
            status=PageStatus.SUCCESS,
            engine=engine.value,
            audit_passed=True,
            confidence=0.5,
        )

    return run


class _AcceptOnly:
    def __init__(self, accept):
        self.accept = set(accept)

    def assess(self, output, provider):
        return AcceptDecision(accept=provider.engine in self.accept, reason="stub")


class TestPaidRungsReachable:
    def test_cloud_reached_when_all_locals_rejected(self) -> None:
        """The issue #39 cap bug: with 5 free locals installed, the judge
        rejecting all of them must still escalate into the paid rungs."""
        d = route_page(1, FULL_LADDER, _run(), _AcceptOnly({EngineType.GEMINI}))
        assert d.accepted
        assert d.winning_engine == "gemini"
        # All five free rungs were genuinely tried first
        assert len(d.attempts) == 6

    def test_explicit_max_attempts_still_caps(self) -> None:
        d = route_page(
            1, FULL_LADDER, _run(), _AcceptOnly({EngineType.GEMINI}), max_attempts=2
        )
        assert not d.accepted
        assert len(d.attempts) == 2


class TestBudgetPreCheck:
    def test_unaffordable_paid_rungs_skipped_free_kept(self) -> None:
        """Budget too small for any paid rung: paid rungs are skipped BEFORE
        spending, free rungs still run, page ships best-effort."""
        d = route_page(
            1,
            FULL_LADDER,
            _run(),
            _AcceptOnly({EngineType.GEMINI}),  # only gemini would pass
            remaining_budget=0.0001,  # < gemini's 0.0002
        )
        assert not d.accepted
        tried = {a.engine for a in d.attempts}
        assert EngineType.GEMINI not in tried
        assert EngineType.MISTRAL not in tried
        assert EngineType.QWEN in tried  # free rungs always fit
        assert d.total_cost_usd == 0.0

    def test_budget_covers_first_paid_rung_only(self) -> None:
        """Within one page the budget decrements: gemini fits, mistral no
        longer does afterwards."""
        d = route_page(
            1,
            FULL_LADDER,
            _run(),
            _AcceptOnly(set()),  # nothing accepted -> walk the whole ladder
            remaining_budget=0.0005,  # gemini (0.0002) fits; mistral (0.001) never
        )
        tried = {a.engine for a in d.attempts}
        assert EngineType.GEMINI in tried
        assert EngineType.MISTRAL not in tried

    def test_no_budget_means_whole_ladder(self) -> None:
        d = route_page(1, FULL_LADDER, _run(), _AcceptOnly(set()))
        assert len(d.attempts) == len(FULL_LADDER)


class TestTruncationDetected:
    def _response(self, finish_reason):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [
                {
                    "message": {"content": "partial page text " * 10},
                    "finish_reason": finish_reason,
                }
            ]
        }
        return resp

    def _engine(self, finish_reason):
        from socr.engines.deepseek_vllm import DeepSeekVLLMEngine

        engine = DeepSeekVLLMEngine()
        engine._initialized = True
        client = MagicMock()
        client.post.return_value = self._response(finish_reason)
        engine._client = client
        return engine

    def test_length_stop_is_truncated_not_success(self) -> None:
        from PIL import Image

        out = self._engine("length").process_image(Image.new("RGB", (10, 10)), page_num=3)
        assert out.failure_mode == FailureMode.TRUNCATED
        assert out.status != PageStatus.SUCCESS
        # Partial text is preserved for best-effort fallback, never shipped as clean
        assert "partial page text" in out.text

    def test_normal_stop_is_success(self) -> None:
        from PIL import Image

        out = self._engine("stop").process_image(Image.new("RGB", (10, 10)), page_num=3)
        assert out.status == PageStatus.SUCCESS
        assert out.failure_mode == FailureMode.NONE


class TestSparsePageGate:
    CAPTION = "Figure 2: Impulse responses of the policy rate to a one basis point surprise."

    def test_sparse_page_low_word_count_is_warning(self) -> None:
        checker = HeuristicsChecker(min_word_count=50)
        result = checker.check(self.CAPTION, sparse_ok=True)
        assert result.passed  # no hard failure
        assert any("Word count" in w for w in result.warnings)

    def test_dense_page_low_word_count_still_fails(self) -> None:
        checker = HeuristicsChecker(min_word_count=50)
        result = checker.check(self.CAPTION, sparse_ok=False)
        assert not result.passed

    def test_scorer_passthrough(self) -> None:
        scorer = FailureModeScorer()
        assert scorer.score(self.CAPTION, sparse_ok=True).passed
        assert not scorer.score(self.CAPTION, sparse_ok=False).passed


class TestSparsePageDetection:
    def _pipeline_with_assessment(self, pa_kwargs):
        from socr.core.born_digital import DocumentAssessment, PageAssessment
        from socr.core.document import DocumentHandle
        from socr.pipeline.orchestrator import UnifiedPipeline

        config = PipelineConfig(quiet=True, tiered=False)
        pipeline = UnifiedPipeline(config)
        defaults = dict(page_num=1, is_born_digital=True, native_text="x", confidence=1.0)
        defaults.update(pa_kwargs)
        with patch.object(DocumentHandle, "__post_init__", lambda self: None):
            pipeline._last_assessment = DocumentAssessment(
                path=Path("/tmp/fake.pdf"), pages=[PageAssessment(**defaults)]
            )
        return pipeline

    def test_figure_page_is_sparse_ok(self) -> None:
        p = self._pipeline_with_assessment({"has_figures": True})
        assert p._sparse_page_ok(1) is True

    def test_source_page_with_few_words_is_sparse_ok(self) -> None:
        p = self._pipeline_with_assessment({"word_count": 24})
        assert p._sparse_page_ok(1) is True

    def test_word_rich_page_is_not_sparse(self) -> None:
        p = self._pipeline_with_assessment({"word_count": 400})
        assert p._sparse_page_ok(1) is False

    def test_no_assessment_is_not_sparse(self) -> None:
        from socr.pipeline.orchestrator import UnifiedPipeline

        p = UnifiedPipeline(PipelineConfig(quiet=True, tiered=False))
        p._last_assessment = None
        assert p._sparse_page_ok(1) is False
