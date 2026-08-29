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
# include_ineligible=True so DeepSeek and Mistral appear (they are auto_eligible=False
# by default and excluded from the production ladder; the cascade tests need them to
# validate reachability of all rungs).
FULL_LADDER = provider_ladder(
    {
        EngineType.QWEN,
        EngineType.GLM,
        EngineType.NOUGAT,
        EngineType.DEEPSEEK,
        EngineType.MARKER,
        EngineType.GEMINI,
        EngineType.MISTRAL,
    },
    include_ineligible=True,
)


def _run(text="some ocr text"):
    def run(profile, page_num: int) -> PageOutput:
        # GH-159: the router passes the whole ProviderProfile, not a bare EngineType.
        return PageOutput(
            page_num=page_num,
            text=text,
            status=PageStatus.SUCCESS,
            engine=profile.engine.value,
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
        d = route_page(1, FULL_LADDER, _run(), _AcceptOnly({EngineType.GEMINI}), max_attempts=2)
        assert not d.accepted
        assert len(d.attempts) == 2


class TestBudgetPreCheck:
    def test_unaffordable_paid_rungs_skipped_free_kept(self) -> None:
        """Budget too small for any paid rung: paid rungs are skipped BEFORE
        spending (stub attempts recorded for journal), free rungs still run,
        page ships best-effort."""
        d = route_page(
            1,
            FULL_LADDER,
            _run(),
            _AcceptOnly({EngineType.GEMINI}),  # only gemini would pass
            remaining_budget=0.0001,  # < gemini's 0.0002
        )
        assert not d.accepted
        # Skipped rungs now appear as stub attempts with reason="budget exceeded".
        # Distinguish "actually ran OCR" from "budget-skip stub" by cost_usd > 0
        # OR by looking at reason.  Paid engines that ran would have cost_usd > 0;
        # skipped stubs have cost_usd == 0.0 and reason == "budget exceeded".
        skipped = {a.engine for a in d.attempts if a.reason == "budget exceeded"}
        ran = {a.engine for a in d.attempts if a.reason != "budget exceeded"}
        assert EngineType.GEMINI in skipped  # skipped, not run
        assert EngineType.MISTRAL in skipped  # skipped, not run
        assert EngineType.QWEN in ran  # free rungs always fit
        assert d.total_cost_usd == 0.0

    def test_budget_covers_first_paid_rung_only(self) -> None:
        """Within one page the budget decrements: with $0.0011, mistral
        ($0.001) would fit BEFORE gemini spends $0.0002 but must not fit
        after — only the decrement makes it skipped."""
        d = route_page(
            1,
            FULL_LADDER,
            _run(),
            _AcceptOnly(set()),  # nothing accepted -> walk the whole ladder
            remaining_budget=0.0011,
        )
        skipped = {a.engine for a in d.attempts if a.reason == "budget exceeded"}
        ran = {a.engine for a in d.attempts if a.reason != "budget exceeded"}
        assert EngineType.GEMINI in ran  # gemini ran (fits budget)
        assert EngineType.MISTRAL in skipped  # mistral skipped after gemini decrements budget

    def test_budget_exactly_covering_a_rung_runs_it(self) -> None:
        """Boundary: a rung that exactly fits the remaining budget runs
        (a '>' vs '>=' slip in the pre-check would strand it)."""
        d = route_page(
            1,
            FULL_LADDER,
            _run(),
            _AcceptOnly(set()),
            remaining_budget=0.0002,  # == gemini's price
        )
        tried = {a.engine for a in d.attempts}
        assert EngineType.GEMINI in tried

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


class TestSparseRoutingEndToEnd:
    """The issue #39 review critical: sparse_ok must act at the ESCALATION
    decision point, not only in post-hoc scoring — otherwise a correct
    sparse caption page walks the whole uncapped ladder, paying the paid
    rungs for deterministic rejections."""

    CAPTION = (
        "Figure 2: Impulse responses of the federal funds rate to a "
        "one basis point monetary policy surprise, 1989 to 2002."
    )

    def test_sparse_caption_accepted_at_first_free_rung(self) -> None:
        from socr.audit.heuristics import HeuristicsChecker
        from socr.pipeline.agentic import HeuristicPageJudge

        judge = HeuristicPageJudge(
            HeuristicsChecker(min_word_count=50), sparse_ok=lambda page_num: True
        )
        d = route_page(1, FULL_LADDER, _run(text=self.CAPTION), judge)
        assert d.accepted
        assert len(d.attempts) == 1
        assert d.total_cost_usd == 0.0

    def test_dense_page_garbage_still_walks_ladder(self) -> None:
        from socr.audit.heuristics import HeuristicsChecker
        from socr.pipeline.agentic import HeuristicPageJudge

        judge = HeuristicPageJudge(
            HeuristicsChecker(min_word_count=50), sparse_ok=lambda page_num: False
        )
        d = route_page(1, FULL_LADDER, _run(text=self.CAPTION), judge)
        assert not d.accepted
        assert len(d.attempts) == len(FULL_LADDER)

    def test_build_page_judge_wires_sparse_predicate(self) -> None:
        from socr.core.document import DocumentHandle
        from socr.core.state import DocumentState
        from socr.pipeline.agentic import (
            HeuristicPageJudge,
            NativeTableVerifierJudge,
            SourceEvidenceTableJudge,
        )
        from socr.pipeline.orchestrator import UnifiedPipeline

        pipeline = UnifiedPipeline(
            PipelineConfig(quiet=True, tiered=False, judge_backend="heuristic")
        )
        with patch.object(DocumentHandle, "__post_init__", lambda self: None):
            handle = DocumentHandle(path=Path("/tmp/fake.pdf"), page_count=1)
        judge = pipeline._build_page_judge(DocumentState(handle=handle))
        # GH-90: SourceEvidenceTableJudge wraps NativeTableVerifierJudge wraps inner.
        assert isinstance(judge, SourceEvidenceTableJudge)
        assert isinstance(judge._inner, NativeTableVerifierJudge)
        inner = judge._inner._inner
        assert isinstance(inner, HeuristicPageJudge)
        assert inner._sparse_ok == pipeline._sparse_page_ok

    def test_agentic_call_site_passes_no_cap(self) -> None:
        """Pin the production call site: reintroducing a max_attempts cap in
        _phase_agentic must fail a test, not just route_page's default."""
        import inspect

        from socr.pipeline.orchestrator import UnifiedPipeline

        src = inspect.getsource(UnifiedPipeline._phase_agentic)
        assert "max_attempts" not in src
        assert "remaining_budget=remaining" in src


class TestTruncatedRoutsToCapable:
    def test_truncated_output_not_promoted_to_best(self) -> None:
        """_create_error_result must set audit_passed=False so apply_result
        never promotes a truncated/error page as clean."""
        from socr.engines.base import BaseHTTPEngine

        out = BaseHTTPEngine._create_error_result(1, "truncated", FailureMode.TRUNCATED)
        assert out.audit_passed is False


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

    def test_source_page_with_few_words_is_sparse_ok(self) -> None:
        p = self._pipeline_with_assessment({"word_count": 24})
        assert p._sparse_page_ok(1) is True

    def test_blank_source_page_is_sparse_ok(self) -> None:
        p = self._pipeline_with_assessment({"word_count": 0})
        assert p._sparse_page_ok(1) is True

    def test_dense_page_with_figure_is_not_sparse(self) -> None:
        """has_figures alone must NOT unlock the relaxed gate — garbage on a
        dense page that merely contains an image must still fail (review)."""
        p = self._pipeline_with_assessment({"has_figures": True, "word_count": 400})
        assert p._sparse_page_ok(1) is False

    def test_scanned_page_never_sparse(self) -> None:
        """Scanned text-layer word counts are junk; they earn no leniency."""
        p = self._pipeline_with_assessment({"is_born_digital": False, "word_count": 3})
        assert p._sparse_page_ok(1) is False

    def test_word_rich_page_is_not_sparse(self) -> None:
        p = self._pipeline_with_assessment({"word_count": 400})
        assert p._sparse_page_ok(1) is False

    def test_no_assessment_is_not_sparse(self) -> None:
        from socr.pipeline.orchestrator import UnifiedPipeline

        p = UnifiedPipeline(PipelineConfig(quiet=True, tiered=False))
        p._last_assessment = None
        assert p._sparse_page_ok(1) is False
