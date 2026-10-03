"""#987: a page-JUDGE timeout must not halt the document via the VLM canary.

Measured (11 re-OCR'd papers): 8 halts, each started by a page-judge timeout
(a different, large model on the same GPU), never an OCR-rung timeout. The
canary then hit the OCR model cold (reload ~37.5s > the 30s floor) and armed
PARTIAL_SAVE_VLM_TIMEOUT, so no model ran on any later page.

Hermetic: ``_available_engines_for_agentic`` and ``probe_ollama_idle`` are
patched. Outcomes are pinned as DIFFERENCES between two runs in one process
that change only the kind of timeout (CLAUDE.md: never pin a locally measured
absolute).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from test_pp2_agentic_fuse import (
    _make_bd_assessment,
    _make_config,
    _make_pipeline,
    _real_pdf,
)

from socr.core.config import EngineType
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import JUDGE_OUTCOME_TIMEOUT, PageOutput, PageStatus
from socr.pipeline.agentic import (
    REASON_PROVIDER_TIMEOUT,
    PageDecision,
    ProviderAttempt,
)
from socr.tables import extract as extract_mod


def _decision(page_num: int, ladder, kind: str) -> PageDecision:
    prof = ladder[0]
    out = PageOutput(
        page_num=page_num,
        text="body text" if kind == "ok" else "",
        status=PageStatus.SUCCESS if kind != "provider" else PageStatus.ERROR,
        engine="qwen",
        audit_passed=kind == "ok",
    )
    reason = {
        "ok": "accepted",
        "judge": "judge raised: page judge timeout after 120.00s",
        "provider": REASON_PROVIDER_TIMEOUT,
    }[kind]
    if kind == "judge":
        out.judge_outcome = JUDGE_OUTCOME_TIMEOUT
    att = ProviderAttempt(
        engine=prof.engine,
        output=out,
        cost_usd=0.0,
        accepted=kind == "ok",
        reason=reason,
        provider_id=prof.id,
        model=prof.model,
        backend=prof.backend,
    )
    return PageDecision(page_num=page_num, final_output=out, attempts=[att])


def _run(tmp_path: Path, monkeypatch, kind_on_page2: str, probe_idle: bool):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = _real_pdf(tmp_path, page_count=4)
    pipeline = _make_pipeline(_make_config(agentic=True, enabled_engines=[EngineType.QWEN]))
    pipeline.bd_detector = MagicMock()
    pipeline.bd_detector.detect.return_value = _make_bd_assessment(4, born_digital_pages=set())
    routed: list[int] = []
    probes: list[object] = []

    def _fake_route(page_num, ladder, run_provider, judge, **kwargs):
        routed.append(page_num)
        return _decision(page_num, ladder, kind_on_page2 if page_num == 2 else "ok")

    def _probe(*a, **k):
        probes.append(a)
        return probe_idle

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch("socr.pipeline.orchestrator.route_page", side_effect=_fake_route),
        patch("socr.pipeline.orchestrator.probe_ollama_idle", side_effect=_probe),
    ):
        result = pipeline.process(pdf, tmp_path)
    halted = "PARTIAL_SAVE_VLM_TIMEOUT" in (result.error or "")
    return routed, halted, probes


def test_judge_timeout_on_page_2_does_not_halt_but_ocr_wedge_does(tmp_path, monkeypatch) -> None:
    """DIFFERENCE: only the kind of timeout on page 2 changes; the probe says 'dead'."""
    j_routed, j_halted, j_probes = _run(tmp_path / "j", monkeypatch, "judge", probe_idle=False)
    p_routed, p_halted, _ = _run(tmp_path / "p", monkeypatch, "provider", probe_idle=False)

    # Judge timeout: pages 3 and 4 still run, nothing halts, the canary is not asked.
    assert j_routed == [1, 2, 3, 4], j_routed
    assert j_halted is False
    assert j_probes == []
    # Control: an OCR-rung wedge still halts the document after page 2.
    assert p_routed == [1, 2], p_routed
    assert p_halted is True


def test_ocr_timeout_alongside_a_judge_timeout_still_arms_the_halt() -> None:
    """A judge timeout must not MASK a real provider timeout on the same page."""
    from socr.pipeline.orchestrator import UnifiedPipeline

    class _Prof:
        engine = EngineType.QWEN
        id = "p"
        model = "m"
        backend = "b"

    judge_att = _decision(1, [_Prof()], "judge").attempts[0]
    prov_att = _decision(1, [_Prof()], "provider").attempts[0]
    assert UnifiedPipeline._attempts_show_timeout([judge_att]) is False
    assert UnifiedPipeline._attempts_show_timeout([judge_att, prov_att]) is True


def test_canary_default_covers_a_cold_load() -> None:
    """The default canary budget exceeds the measured 37.5s cold load."""
    assert extract_mod.canary_deadline() > 37.5
    assert extract_mod.canary_deadline() > extract_mod._CROP_DEADLINE_FLOOR_S


@pytest.mark.parametrize(
    ("allowance", "alive"),
    [(1.0, True), (0.0, False)],
)
def test_slow_but_alive_first_response_is_not_a_wedge(monkeypatch, allowance, alive) -> None:
    """A response that arrives after the floor but inside floor+allowance is alive.

    Constants are scaled down (floor 0.1s, response at 0.4s) so the test is fast
    and hermetic. With no allowance the same slow response reads as a wedge, so
    the allowance, not the fake, decides the outcome.
    """
    import time

    class _Resp:
        def raise_for_status(self) -> None:
            return None

    def _slow_post(*a, **k):
        time.sleep(0.4)
        return _Resp()

    monkeypatch.setattr(extract_mod, "_CROP_DEADLINE_FLOOR_S", 0.1)
    monkeypatch.setattr(extract_mod, "CANARY_LOAD_ALLOWANCE_S", allowance)
    monkeypatch.setattr(extract_mod.httpx, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(extract_mod.httpx, "post", _slow_post)

    assert extract_mod.probe_ollama_idle("http://gpu-node:11434", model="m") is alive
