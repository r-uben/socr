"""GH-940: the resume skip decision probes engines WITHOUT side effects, and once.

``_escalation_retry_blocks_resume`` used to call ``_available_engines_for_agentic``,
which resets ``_qwen_cloud_pin_unavailable``, emits the once-per-pipeline unservable
report and runs per-engine CLI checks; when the gate then refused the skip,
``_phase_agentic`` probed again. Now the gate holds a pure probe and the next
``_available_engines_for_agentic`` consumes it.

Hermetic: the probe beneath the wrapper is stubbed, so nothing needs ollama.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from socr.core.providers import PROFILE_GEMINI, PROFILE_QWEN_LOCAL
from socr.core.config import EngineType
from socr.pipeline.orchestrator import UnifiedPipeline
from test_gh851_escalation_latch_evidence import _run
from test_gh855_scoring_independent_of_lane_health import _config

_LADDERS = {
    "local+cloud": [PROFILE_QWEN_LOCAL, PROFILE_GEMINI],
    "local-only": [PROFILE_QWEN_LOCAL],
    "cloud-only": [PROFILE_GEMINI],
    "empty": [],
}


def _pipeline(**cfg_kw) -> UnifiedPipeline:
    cfg = _config()
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    return UnifiedPipeline(cfg)


def _stub_probe(pipeline, available, *, unservable=(), pin=""):
    return patch.object(
        pipeline,
        "_probe_engines_for_agentic",
        return_value=(list(available), list(unservable), pin),
    )


def test_gate_does_not_report_or_reset_the_pin_flag() -> None:
    p = _pipeline()
    p._qwen_cloud_pin_unavailable = "earlier failure"
    with (
        _stub_probe(p, _LADDERS["local+cloud"], unservable=[EngineType.AUTO], pin="new"),
        patch.object(p, "_report_unservable_engines") as report,
    ):
        assert p._escalation_retry_blocks_resume() is True
    report.assert_not_called()
    assert p._qwen_cloud_pin_unavailable == "earlier failure"


def test_a_refused_skip_probes_once_and_the_phase_gets_the_effects() -> None:
    p = _pipeline()
    with (
        _stub_probe(p, _LADDERS["local+cloud"], unservable=[EngineType.AUTO], pin="why") as probe,
        patch.object(p, "_report_unservable_engines") as report,
    ):
        assert p._escalation_retry_blocks_resume() is True
        available = p._available_engines_for_agentic()
        assert probe.call_count == 1
        report.assert_called_once_with([EngineType.AUTO])
        assert p._qwen_cloud_pin_unavailable == "why"
        assert available == _LADDERS["local+cloud"]
        # the held probe is consumed: a later call samples afresh, not stale
        p._available_engines_for_agentic()
        assert probe.call_count == 2


def test_a_gate_that_lets_the_skip_through_holds_nothing() -> None:
    p = _pipeline()
    with _stub_probe(p, _LADDERS["empty"]) as probe:
        assert p._escalation_retry_blocks_resume() is False
        p._available_engines_for_agentic()
    assert probe.call_count == 2


@pytest.mark.parametrize("strict_local", [False, True])
@pytest.mark.parametrize("escalate", [False, True])
@pytest.mark.parametrize("agentic", [False, True])
@pytest.mark.parametrize("ladder", sorted(_LADDERS))
def test_gate_outcome_is_identical_to_the_side_effecting_reference(
    ladder, agentic, escalate, strict_local
) -> None:
    """Difference test: old gate body (full wrapper) vs the new one, same inputs."""

    def reference(p: UnifiedPipeline) -> bool:
        if not (p.config.agentic and getattr(p.config, "escalate_ambiguous_tables", False)):
            return False
        available = p._available_engines_for_agentic()
        if p.config.strict_local:
            from socr.core.providers import TIER_LOCAL

            available = [x for x in available if x.tier == TIER_LOCAL]
        _, profile = p._build_ladder_and_escalation_profile(available)
        return profile is not None

    p = _pipeline(agentic=agentic, escalate_ambiguous_tables=escalate, strict_local=strict_local)
    with _stub_probe(p, _LADDERS[ladder]):
        assert p._escalation_retry_blocks_resume() == reference(p)


def test_end_to_end_a_refused_resume_probes_exactly_once(tmp_path: Path) -> None:
    _run(tmp_path, mode="wedge")  # leaves table_escalation_retry_pending
    ladder = (list(_LADDERS["local+cloud"]), [], "")
    with patch.object(UnifiedPipeline, "_probe_engines_for_agentic", return_value=ladder) as probe:
        calls, _, _ = _run(tmp_path, mode="healthy", stub_available=False)
    assert 2 in calls, "the gate must have refused the skip"
    assert probe.call_count == 1
