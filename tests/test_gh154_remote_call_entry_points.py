"""GH-154 round 3 (Astra/Codex review): an EXPLICIT ``--max-cost-per-page 0``
must forbid a cloud/remote call at EVERY entry point, not just the main
routing ladder and the clean-equation lane.

Round 1/2 wired ``max_cost_per_page_pinned`` into ``provider_ladder`` and the
two ladder-building call sites. The review found four more direct entry
points that skip the ladder entirely and read the config flags themselves --
each is covered here, all pinned to the shared
``socr.core.providers.zero_cap_pinned_forbids_cloud`` predicate. The blind-cell
adjudicator's own cap gate (``judge/table_cell_guard.py``) is covered in
``tests/test_table_cell_guard.py::test_gh154_pinned_zero_forbids_the_adjudicator_call``.

Hermetic: no provider, no network, no live model.
"""

from __future__ import annotations

from socr.core.config import PipelineConfig
from socr.core.providers import zero_cap_pinned_forbids_cloud
from socr.pipeline.orchestrator import UnifiedPipeline


def _pipeline(**overrides) -> UnifiedPipeline:
    cfg = PipelineConfig(quiet=True, **overrides)
    pipe = object.__new__(UnifiedPipeline)
    pipe.config = cfg
    return pipe


# ---------------------------------------------------------------------------
# The shared predicate itself
# ---------------------------------------------------------------------------


def test_predicate_false_for_unset_default():
    cfg = PipelineConfig()
    assert cfg.max_cost_per_page == 0.0
    assert cfg.max_cost_per_page_pinned is False
    assert zero_cap_pinned_forbids_cloud(cfg) is False


def test_predicate_true_only_when_pinned_and_zero():
    assert zero_cap_pinned_forbids_cloud(
        PipelineConfig(max_cost_per_page=0.0, max_cost_per_page_pinned=True)
    )
    # Pinned but a real positive cap: the predicate is about the ZERO case only.
    assert not zero_cap_pinned_forbids_cloud(
        PipelineConfig(max_cost_per_page=0.05, max_cost_per_page_pinned=True)
    )
    # Zero but not pinned (the ordinary "no cap" default): untouched.
    assert not zero_cap_pinned_forbids_cloud(
        PipelineConfig(max_cost_per_page=0.0, max_cost_per_page_pinned=False)
    )


# ---------------------------------------------------------------------------
# 1 — direct corrupt-math model call (Astra's original probe)
# ---------------------------------------------------------------------------


def test_pinned_zero_blocks_direct_cloud_math():
    pipe = _pipeline(max_cost_per_page=0, max_cost_per_page_pinned=True, math_model="qwen3.5:cloud")
    assert pipe._corrupt_math_model_disabled_reason()


def test_unpinned_zero_still_allows_direct_cloud_math_by_default():
    # Control: the ordinary omitted-flag default must be untouched.
    pipe = _pipeline(max_cost_per_page=0.0, math_model="qwen3.5:cloud")
    assert pipe._corrupt_math_model_disabled_reason() == ""


# ---------------------------------------------------------------------------
# 2 — clean-equation-region lane's early cloud guard
# ---------------------------------------------------------------------------


def test_pinned_zero_blocks_equation_lane_cloud_model():
    pipe = _pipeline(
        max_cost_per_page=0,
        max_cost_per_page_pinned=True,
        clean_equation_model="qwen3.5:cloud",
    )
    profile, reason = pipe._equation_lane_provider(available_profiles=[])
    assert profile is None
    assert "max-cost-per-page 0" in reason


# ---------------------------------------------------------------------------
# 3 — cloud table-judge ladder construction + reachability
# ---------------------------------------------------------------------------


def test_pinned_zero_empties_the_table_judge_ladder():
    pipe = _pipeline(
        table_judge_ladder=True,
        max_cost_per_page=0,
        max_cost_per_page_pinned=True,
    )
    assert pipe._build_table_judge_rungs() == []


def test_unpinned_zero_still_builds_the_table_judge_ladder_by_default():
    pipe = _pipeline(table_judge_ladder=True, max_cost_per_page=0.0)
    assert pipe._build_table_judge_rungs() != []


def test_pinned_zero_makes_no_table_judge_rung_available():
    pipe = _pipeline(
        table_judge_ladder=True,
        max_cost_per_page=0,
        max_cost_per_page_pinned=True,
    )
    assert pipe._table_judge_rung_available_now() is False


# ---------------------------------------------------------------------------
# 4 — blind-cell adjudicator construction
# ---------------------------------------------------------------------------


def test_pinned_zero_refuses_to_build_the_adjudicator():
    pipe = _pipeline(
        table_judge_ladder=True,
        max_cost_per_page=0,
        max_cost_per_page_pinned=True,
    )
    assert pipe._build_table_cell_adjudicator() is None


def test_unpinned_zero_still_builds_the_adjudicator_by_default():
    pipe = _pipeline(table_judge_ladder=True, max_cost_per_page=0.0)
    assert pipe._build_table_cell_adjudicator() is not None
