"""GH-940: the resume skip decision probes engines WITHOUT side effects.

``_escalation_retry_blocks_resume`` used to call ``_available_engines_for_agentic``,
which resets ``_qwen_cloud_pin_unavailable`` and emits the once-per-pipeline
unservable report. A skip DECISION must do neither, so the gate now asks the pure
``_probe_engines_for_agentic``. A refused skip still means ``_phase_agentic`` probes
again; that is accepted (the probes are cheap), the side effects are what mattered.

Hermetic: the probe is stubbed, so nothing needs ollama.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from socr.core.config import EngineType
from socr.core.providers import PROFILE_GEMINI, PROFILE_QWEN_LOCAL
from socr.pipeline.orchestrator import UnifiedPipeline
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


def test_gate_adds_exactly_one_side_effect_free_probe() -> None:
    p = _pipeline()
    with (
        _stub_probe(p, _LADDERS["local+cloud"], unservable=[EngineType.AUTO], pin="why") as probe,
        patch.object(p, "_report_unservable_engines") as report,
    ):
        p._escalation_retry_blocks_resume()
        assert probe.call_count == 1
        report.assert_not_called()
        # the phase then probes for itself and is the one that records effects
        p._available_engines_for_agentic()
        assert probe.call_count == 2
        report.assert_called_once_with([EngineType.AUTO])
        assert p._qwen_cloud_pin_unavailable == "why"


def _old_gate(p: UnifiedPipeline) -> bool:
    """The gate body as on origin/main before GH-940, copied verbatim."""
    if not (p.config.agentic and getattr(p.config, "escalate_ambiguous_tables", False)):
        return False
    available = p._available_engines_for_agentic()
    if p.config.strict_local:
        from socr.core.providers import TIER_LOCAL

        available = [p for p in available if p.tier == TIER_LOCAL]
    _, profile = p._build_ladder_and_escalation_profile(available)
    return profile is not None


@pytest.mark.parametrize("strict_local", [False, True])
@pytest.mark.parametrize("escalate", [False, True])
@pytest.mark.parametrize("agentic", [False, True])
@pytest.mark.parametrize("ladder", sorted(_LADDERS))
def test_gate_outcome_is_identical_to_the_old_gate(ladder, agentic, escalate, strict_local) -> None:
    """Difference test on two INDEPENDENT pipelines: new gate vs the old body."""
    kw = dict(agentic=agentic, escalate_ambiguous_tables=escalate, strict_local=strict_local)
    new, old = _pipeline(**kw), _pipeline(**kw)
    with (
        _stub_probe(new, _LADDERS[ladder]),
        patch.object(old, "_available_engines_for_agentic", return_value=list(_LADDERS[ladder])),
    ):
        assert new._escalation_retry_blocks_resume() == _old_gate(old)
