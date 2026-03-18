"""Tests for RepairRouter — selective page repair routing."""

from pathlib import Path
from unittest.mock import patch

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState, PageState
from socr.pipeline.repair import PageRepair, RepairPlan, RepairRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_handle(page_count: int = 5) -> DocumentHandle:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        h = DocumentHandle(path=Path("/tmp/fake.pdf"), page_count=page_count)
    return h


def _make_page_output(
    page_num: int,
    text: str = "some text",
    audit_passed: bool = True,
    engine: str = "deepseek",
    failure_mode: FailureMode = FailureMode.NONE,
) -> PageOutput:
    status = PageStatus.SUCCESS if audit_passed else PageStatus.ERROR
    return PageOutput(
        page_num=page_num,
        text=text,
        status=status,
        audit_passed=audit_passed,
        engine=engine,
        failure_mode=failure_mode,
    )


def _make_config(**overrides) -> PipelineConfig:
    defaults = dict(
        primary_engine=EngineType.DEEPSEEK,
        fallback_chain=[EngineType.GEMINI, EngineType.MISTRAL],
        enabled_engines=list(EngineType),
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_state(
    page_count: int = 5,
    born_digital: dict[int, str] | None = None,
    attempts: dict[int, list[PageOutput]] | None = None,
    best_outputs: dict[int, PageOutput] | None = None,
) -> DocumentState:
    """Build a DocumentState with optional pre-populated page data."""
    state = DocumentState(handle=_make_handle(page_count))

    if born_digital:
        for pg, text in born_digital.items():
            if pg in state.pages:
                state.pages[pg].is_born_digital = True
                state.pages[pg].native_text = text

    if attempts:
        for pg, att_list in attempts.items():
            if pg in state.pages:
                state.pages[pg].attempts = att_list

    if best_outputs:
        for pg, best in best_outputs.items():
            if pg in state.pages:
                state.pages[pg].best_output = best

    return state


# ---------------------------------------------------------------------------
# Page selection
# ---------------------------------------------------------------------------

class TestPagesNeedingRepair:
    def test_all_pages_need_repair_initially(self) -> None:
        state = _make_state(page_count=3)
        router = RepairRouter(_make_config())
        result = router.pages_needing_repair(state)

        assert [pn for pn, _ in result] == [1, 2, 3]

    def test_pages_with_passing_best_output_excluded(self) -> None:
        good = _make_page_output(1, "ok", audit_passed=True)
        state = _make_state(
            page_count=3,
            best_outputs={1: good},
        )
        router = RepairRouter(_make_config())
        result = router.pages_needing_repair(state)

        assert [pn for pn, _ in result] == [2, 3]

    def test_born_digital_with_text_excluded(self) -> None:
        state = _make_state(
            page_count=3,
            born_digital={2: "native text"},
        )
        router = RepairRouter(_make_config())
        result = router.pages_needing_repair(state)

        assert [pn for pn, _ in result] == [1, 3]

    def test_failed_audit_best_output_still_needs_repair(self) -> None:
        bad = _make_page_output(1, "garbage", audit_passed=False)
        state = _make_state(
            page_count=1,
            best_outputs={1: bad},
        )
        router = RepairRouter(_make_config())
        result = router.pages_needing_repair(state)

        assert len(result) == 1
        assert result[0][0] == 1

    def test_empty_document(self) -> None:
        state = _make_state(page_count=0)
        router = RepairRouter(_make_config())
        assert router.pages_needing_repair(state) == []

    def test_all_pages_done(self) -> None:
        state = _make_state(
            page_count=2,
            best_outputs={
                1: _make_page_output(1, "ok"),
                2: _make_page_output(2, "ok"),
            },
        )
        router = RepairRouter(_make_config())
        assert router.pages_needing_repair(state) == []

    def test_returns_page_state_objects(self) -> None:
        state = _make_state(page_count=2)
        router = RepairRouter(_make_config())
        result = router.pages_needing_repair(state)

        for page_num, page_state in result:
            assert isinstance(page_state, PageState)
            assert page_state.page_num == page_num


# ---------------------------------------------------------------------------
# Engine routing by failure mode
# ---------------------------------------------------------------------------

class TestSelectRepairEngine:
    def test_hallucination_picks_different_family(self) -> None:
        router = RepairRouter(_make_config())
        engine = router.select_repair_engine(
            FailureMode.HALLUCINATION,
            tried_engines={EngineType.DEEPSEEK},
        )
        # DeepSeek is "deepseek" family; should pick from another family
        assert engine is not None
        assert engine not in {EngineType.DEEPSEEK, EngineType.DEEPSEEK_VLLM}

    def test_hallucination_different_family_from_gemini(self) -> None:
        router = RepairRouter(_make_config(
            fallback_chain=[EngineType.DEEPSEEK, EngineType.MISTRAL],
        ))
        engine = router.select_repair_engine(
            FailureMode.HALLUCINATION,
            tried_engines={EngineType.GEMINI},
        )
        assert engine is not None
        # Gemini is "google" family; should pick from another
        assert engine != EngineType.GEMINI

    def test_refusal_picks_cloud_engine(self) -> None:
        router = RepairRouter(_make_config())
        engine = router.select_repair_engine(
            FailureMode.REFUSAL,
            tried_engines={EngineType.GLM},  # local engine refused
        )
        assert engine is not None
        # Should prefer cloud engines
        assert engine in {EngineType.GEMINI, EngineType.MISTRAL, EngineType.DEEPSEEK}

    def test_garbage_picks_capable_engine(self) -> None:
        router = RepairRouter(_make_config())
        engine = router.select_repair_engine(
            FailureMode.GARBAGE,
            tried_engines={EngineType.NOUGAT},
        )
        assert engine is not None
        assert engine in {
            EngineType.GEMINI, EngineType.MISTRAL,
            EngineType.DEEPSEEK, EngineType.DEEPSEEK_VLLM,
        }

    def test_low_word_count_picks_capable_engine(self) -> None:
        router = RepairRouter(_make_config())
        engine = router.select_repair_engine(
            FailureMode.LOW_WORD_COUNT,
            tried_engines={EngineType.GLM},
        )
        assert engine is not None
        assert engine in {
            EngineType.GEMINI, EngineType.MISTRAL,
            EngineType.DEEPSEEK, EngineType.DEEPSEEK_VLLM,
        }

    def test_timeout_picks_lighter_engine(self) -> None:
        router = RepairRouter(_make_config(
            fallback_chain=[EngineType.GLM, EngineType.NOUGAT, EngineType.GEMINI],
        ))
        engine = router.select_repair_engine(
            FailureMode.TIMEOUT,
            tried_engines={EngineType.DEEPSEEK},
        )
        assert engine is not None
        assert engine in {EngineType.GLM, EngineType.NOUGAT, EngineType.MARKER}

    def test_empty_output_picks_first_untried(self) -> None:
        config = _make_config(
            fallback_chain=[EngineType.GEMINI, EngineType.MISTRAL],
        )
        router = RepairRouter(config)
        engine = router.select_repair_engine(
            FailureMode.EMPTY_OUTPUT,
            tried_engines={EngineType.DEEPSEEK},
        )
        # First in fallback chain that wasn't tried
        assert engine == EngineType.GEMINI

    def test_api_error_picks_first_untried(self) -> None:
        config = _make_config(
            fallback_chain=[EngineType.GEMINI, EngineType.MISTRAL],
        )
        router = RepairRouter(config)
        engine = router.select_repair_engine(
            FailureMode.API_ERROR,
            tried_engines={EngineType.GEMINI},
        )
        assert engine == EngineType.MISTRAL


# ---------------------------------------------------------------------------
# Skipping already-tried engines
# ---------------------------------------------------------------------------

class TestSkipTriedEngines:
    def test_skips_tried_engine(self) -> None:
        config = _make_config(
            fallback_chain=[EngineType.GEMINI, EngineType.MISTRAL],
        )
        router = RepairRouter(config)
        engine = router.select_repair_engine(
            FailureMode.EMPTY_OUTPUT,
            tried_engines={EngineType.GEMINI, EngineType.DEEPSEEK},
        )
        assert engine == EngineType.MISTRAL

    def test_returns_none_when_all_tried(self) -> None:
        config = _make_config(
            fallback_chain=[EngineType.GEMINI],
            enabled_engines=[EngineType.DEEPSEEK, EngineType.GEMINI],
        )
        router = RepairRouter(config)
        engine = router.select_repair_engine(
            FailureMode.EMPTY_OUTPUT,
            tried_engines={EngineType.DEEPSEEK, EngineType.GEMINI},
        )
        assert engine is None

    def test_hallucination_falls_back_to_same_family_if_all_other_families_tried(self) -> None:
        config = _make_config(
            fallback_chain=[EngineType.DEEPSEEK_VLLM, EngineType.GEMINI, EngineType.MISTRAL],
            enabled_engines=[EngineType.DEEPSEEK, EngineType.DEEPSEEK_VLLM, EngineType.GEMINI, EngineType.MISTRAL],
        )
        router = RepairRouter(config)
        # Tried DeepSeek + Gemini + Mistral — all families covered
        engine = router.select_repair_engine(
            FailureMode.HALLUCINATION,
            tried_engines={EngineType.DEEPSEEK, EngineType.GEMINI, EngineType.MISTRAL},
        )
        # Only DEEPSEEK_VLLM is left (same family as deepseek), should still return it
        assert engine == EngineType.DEEPSEEK_VLLM

    def test_primary_engine_included_in_candidates(self) -> None:
        """Primary engine can be used for repair if not already tried on that page."""
        config = _make_config(
            primary_engine=EngineType.DEEPSEEK,
            fallback_chain=[EngineType.GEMINI],
            enabled_engines=[EngineType.DEEPSEEK, EngineType.GEMINI],
        )
        router = RepairRouter(config)
        engine = router.select_repair_engine(
            FailureMode.EMPTY_OUTPUT,
            tried_engines={EngineType.GEMINI},
        )
        assert engine == EngineType.DEEPSEEK


# ---------------------------------------------------------------------------
# Repair plan generation
# ---------------------------------------------------------------------------

class TestPlanRepairs:
    def test_plan_for_fresh_document(self) -> None:
        """All pages need repair, each gets the first fallback engine."""
        config = _make_config(
            fallback_chain=[EngineType.GEMINI, EngineType.MISTRAL],
        )
        state = _make_state(page_count=3)
        router = RepairRouter(config)
        plan = router.plan_repairs(state)

        assert len(plan.repairs) == 3
        assert plan.pages_skipped == []
        for repair in plan.repairs:
            assert isinstance(repair, PageRepair)

    def test_plan_skips_done_pages(self) -> None:
        config = _make_config()
        state = _make_state(
            page_count=3,
            best_outputs={1: _make_page_output(1, "ok")},
            born_digital={2: "native"},
        )
        router = RepairRouter(config)
        plan = router.plan_repairs(state)

        assert len(plan.repairs) == 1
        assert plan.repairs[0].page_num == 3

    def test_plan_uses_failure_mode_from_attempts(self) -> None:
        config = _make_config(
            fallback_chain=[EngineType.GEMINI, EngineType.MISTRAL],
        )
        hallu_attempt = _make_page_output(
            1, "repeat repeat repeat",
            audit_passed=False,
            engine="deepseek",
            failure_mode=FailureMode.HALLUCINATION,
        )
        state = _make_state(
            page_count=1,
            attempts={1: [hallu_attempt]},
        )
        router = RepairRouter(config)
        plan = router.plan_repairs(state)

        assert len(plan.repairs) == 1
        # Should pick a different family than deepseek
        assert plan.repairs[0].engine not in {EngineType.DEEPSEEK, EngineType.DEEPSEEK_VLLM}
        assert "hallucination" in plan.repairs[0].reason

    def test_plan_exhausted_engines_go_to_skipped(self) -> None:
        config = _make_config(
            fallback_chain=[EngineType.GEMINI],
            enabled_engines=[EngineType.DEEPSEEK, EngineType.GEMINI],
        )
        # Page 1 has tried both available engines
        att1 = _make_page_output(1, "bad", audit_passed=False, engine="deepseek")
        att2 = _make_page_output(1, "bad", audit_passed=False, engine="gemini")
        state = _make_state(
            page_count=1,
            attempts={1: [att1, att2]},
        )
        router = RepairRouter(config)
        plan = router.plan_repairs(state)

        assert plan.repairs == []
        assert plan.pages_skipped == [1]

    def test_plan_reason_includes_failure_and_engine(self) -> None:
        config = _make_config(fallback_chain=[EngineType.GEMINI])
        state = _make_state(page_count=1)
        router = RepairRouter(config)
        plan = router.plan_repairs(state)

        assert len(plan.repairs) == 1
        reason = plan.repairs[0].reason
        assert "selected=" in reason
        assert "failure=" in reason

    def test_plan_is_empty_when_nothing_to_repair(self) -> None:
        config = _make_config()
        state = _make_state(
            page_count=2,
            best_outputs={
                1: _make_page_output(1, "ok"),
                2: _make_page_output(2, "ok"),
            },
        )
        router = RepairRouter(config)
        plan = router.plan_repairs(state)

        assert plan.is_empty
        assert plan.pages_skipped == []


# ---------------------------------------------------------------------------
# Grouping by engine
# ---------------------------------------------------------------------------

class TestGroupByEngine:
    def test_groups_repairs_by_engine(self) -> None:
        plan = RepairPlan(repairs=[
            PageRepair(page_num=1, engine=EngineType.GEMINI, reason="r1"),
            PageRepair(page_num=2, engine=EngineType.MISTRAL, reason="r2"),
            PageRepair(page_num=3, engine=EngineType.GEMINI, reason="r3"),
            PageRepair(page_num=4, engine=EngineType.GEMINI, reason="r4"),
        ])

        groups = plan.by_engine
        assert set(groups.keys()) == {EngineType.GEMINI, EngineType.MISTRAL}
        assert len(groups[EngineType.GEMINI]) == 3
        assert len(groups[EngineType.MISTRAL]) == 1

    def test_groups_page_nums_correct(self) -> None:
        plan = RepairPlan(repairs=[
            PageRepair(page_num=3, engine=EngineType.GEMINI, reason="r"),
            PageRepair(page_num=7, engine=EngineType.GEMINI, reason="r"),
            PageRepair(page_num=12, engine=EngineType.GEMINI, reason="r"),
        ])

        gemini_pages = [r.page_num for r in plan.by_engine[EngineType.GEMINI]]
        assert gemini_pages == [3, 7, 12]

    def test_empty_plan_groups(self) -> None:
        plan = RepairPlan()
        assert plan.by_engine == {}

    def test_single_engine_group(self) -> None:
        plan = RepairPlan(repairs=[
            PageRepair(page_num=1, engine=EngineType.MISTRAL, reason="r"),
            PageRepair(page_num=2, engine=EngineType.MISTRAL, reason="r"),
        ])

        groups = plan.by_engine
        assert len(groups) == 1
        assert EngineType.MISTRAL in groups


# ---------------------------------------------------------------------------
# Exhausted engines
# ---------------------------------------------------------------------------

class TestExhaustedEngines:
    def test_all_fallback_chain_exhausted(self) -> None:
        config = _make_config(
            primary_engine=EngineType.DEEPSEEK,
            fallback_chain=[EngineType.GEMINI, EngineType.MISTRAL],
            enabled_engines=[EngineType.DEEPSEEK, EngineType.GEMINI, EngineType.MISTRAL],
        )
        att_ds = _make_page_output(1, "bad", audit_passed=False, engine="deepseek")
        att_gm = _make_page_output(1, "bad", audit_passed=False, engine="gemini")
        att_ms = _make_page_output(1, "bad", audit_passed=False, engine="mistral")
        state = _make_state(
            page_count=1,
            attempts={1: [att_ds, att_gm, att_ms]},
        )
        router = RepairRouter(config)
        plan = router.plan_repairs(state)

        assert plan.is_empty
        assert plan.pages_skipped == [1]

    def test_some_pages_exhausted_some_not(self) -> None:
        config = _make_config(
            primary_engine=EngineType.DEEPSEEK,
            fallback_chain=[EngineType.GEMINI],
            enabled_engines=[EngineType.DEEPSEEK, EngineType.GEMINI],
        )
        # Page 1: both engines tried
        att1a = _make_page_output(1, "bad", audit_passed=False, engine="deepseek")
        att1b = _make_page_output(1, "bad", audit_passed=False, engine="gemini")
        # Page 2: only deepseek tried
        att2 = _make_page_output(2, "bad", audit_passed=False, engine="deepseek")

        state = _make_state(
            page_count=2,
            attempts={1: [att1a, att1b], 2: [att2]},
        )
        router = RepairRouter(config)
        plan = router.plan_repairs(state)

        assert len(plan.repairs) == 1
        assert plan.repairs[0].page_num == 2
        assert plan.repairs[0].engine == EngineType.GEMINI
        assert plan.pages_skipped == [1]

    def test_returns_none_directly(self) -> None:
        """select_repair_engine returns None when no candidates left."""
        config = _make_config(
            fallback_chain=[],
            enabled_engines=[EngineType.DEEPSEEK],
        )
        router = RepairRouter(config)
        result = router.select_repair_engine(
            FailureMode.GARBAGE,
            tried_engines={EngineType.DEEPSEEK},
        )
        assert result is None


# ---------------------------------------------------------------------------
# RepairPlan dataclass
# ---------------------------------------------------------------------------

class TestRepairPlanDataclass:
    def test_is_empty_true_for_no_repairs(self) -> None:
        plan = RepairPlan()
        assert plan.is_empty

    def test_is_empty_false_with_repairs(self) -> None:
        plan = RepairPlan(repairs=[
            PageRepair(page_num=1, engine=EngineType.GEMINI, reason="r"),
        ])
        assert not plan.is_empty

    def test_is_empty_true_even_with_skipped(self) -> None:
        plan = RepairPlan(pages_skipped=[1, 2])
        assert plan.is_empty


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_page_with_no_attempts_uses_empty_output_routing(self) -> None:
        """Fresh pages (no attempts) should route as EMPTY_OUTPUT."""
        config = _make_config(
            fallback_chain=[EngineType.GEMINI, EngineType.MISTRAL],
        )
        state = _make_state(page_count=1)  # no attempts
        router = RepairRouter(config)
        plan = router.plan_repairs(state)

        assert len(plan.repairs) == 1
        # EMPTY_OUTPUT routing picks first untried from chain
        assert plan.repairs[0].engine == EngineType.GEMINI

    def test_most_recent_failure_mode_used(self) -> None:
        """When a page has multiple attempts, the most recent failure wins."""
        config = _make_config(
            fallback_chain=[EngineType.GLM, EngineType.NOUGAT, EngineType.GEMINI],
        )
        old = _make_page_output(
            1, "bad", audit_passed=False, engine="deepseek",
            failure_mode=FailureMode.GARBAGE,
        )
        recent = _make_page_output(
            1, "bad", audit_passed=False, engine="gemini",
            failure_mode=FailureMode.TIMEOUT,
        )
        state = _make_state(
            page_count=1,
            attempts={1: [old, recent]},
        )
        router = RepairRouter(config)
        plan = router.plan_repairs(state)

        # TIMEOUT should pick a lighter engine
        assert len(plan.repairs) == 1
        assert plan.repairs[0].engine in {EngineType.GLM, EngineType.NOUGAT, EngineType.MARKER}

    def test_mixed_failure_modes_across_pages(self) -> None:
        """Different pages can get different engines based on their failure."""
        config = _make_config(
            fallback_chain=[EngineType.GEMINI, EngineType.GLM, EngineType.NOUGAT, EngineType.MISTRAL],
        )
        hallu = _make_page_output(
            1, "bad", audit_passed=False, engine="deepseek",
            failure_mode=FailureMode.HALLUCINATION,
        )
        timeout = _make_page_output(
            2, "bad", audit_passed=False, engine="deepseek",
            failure_mode=FailureMode.TIMEOUT,
        )
        state = _make_state(
            page_count=2,
            attempts={1: [hallu], 2: [timeout]},
        )
        router = RepairRouter(config)
        plan = router.plan_repairs(state)

        assert len(plan.repairs) == 2
        engines = {r.page_num: r.engine for r in plan.repairs}
        # Page 1 (hallucination) should get different family from deepseek
        assert engines[1] not in {EngineType.DEEPSEEK, EngineType.DEEPSEEK_VLLM}
        # Page 2 (timeout) should get a lighter engine
        assert engines[2] in {EngineType.GLM, EngineType.NOUGAT, EngineType.MARKER}
