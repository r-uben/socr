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


# ---------------------------------------------------------------------------
# Judge circuit breaker: a WEDGED judge (not merely slow) stops costing a full
# judge deadline per page. Evidence is one 1-token generation probe.
# ---------------------------------------------------------------------------


def _run_breaker(tmp_path: Path, monkeypatch, *, mode: str, judge_probe_alive: bool):
    """Drive process() with the REAL judge chain; only route_page's guard is faked.

    ``mode`` "judge": every page's judge call exceeds the (scaled) deadline.
    ``mode`` "provider": page 2's OCR rung times out (an OCR wedge).
    """
    import json
    import time

    from socr.core.config import EngineType as _E
    from socr.judge.judge import PageJudgeTimeoutError
    from socr.judge.ollama_judge import OllamaVisionJudge

    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = _real_pdf(tmp_path, page_count=4)
    pipeline = _make_pipeline(_make_config(agentic=True, enabled_engines=[_E.QWEN]))
    pipeline.config.judge_backend = "vlm"
    pipeline.bd_detector = MagicMock()
    pipeline.bd_detector.detect.return_value = _make_bd_assessment(4, born_digital_pages=set())
    judge_calls: list[int] = []
    probe_calls: list[tuple] = []

    def _slow_judge(self, image_path, text, *a, **k):
        judge_calls.append(1)
        time.sleep(0.6)
        raise AssertionError("deadline should have fired first")

    def _fake_route(page_num, ladder, run_provider, judge, **kwargs):
        prof = ladder[0]
        if mode == "provider" and page_num == 2:
            return _decision(page_num, ladder, "provider")
        out = PageOutput(
            page_num=page_num,
            text="body text " * 40,
            status=PageStatus.SUCCESS,
            engine="qwen",
        )
        reason, outcome = "accepted", ""
        try:
            dec = judge.assess(out, prof)
            reason = dec.reason
        except PageJudgeTimeoutError as exc:
            reason, outcome = f"judge raised: {exc}", JUDGE_OUTCOME_TIMEOUT
        out.judge_outcome = outcome
        att = ProviderAttempt(
            engine=prof.engine,
            output=out,
            cost_usd=0.0,
            accepted=False,
            reason=reason,
            provider_id=prof.id,
            model=prof.model,
            backend=prof.backend,
        )
        return PageDecision(page_num=page_num, final_output=out, attempts=[att])

    def _probe_model(host, model, timeout, **kw):
        probe_calls.append((host, model, timeout))
        return (judge_probe_alive, "" if judge_probe_alive else "timed out")

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value="judge-model"),
        patch.object(pipeline, "_make_page_renderer", return_value=lambda p: "img"),
        patch("socr.pipeline.agentic.DEFAULT_PROVIDER_TIMEOUTS", {_E.QWEN: 0.05}),
        patch.object(OllamaVisionJudge, "judge", _slow_judge),
        patch("socr.pipeline.orchestrator.route_page", side_effect=_fake_route),
        patch("socr.pipeline.orchestrator.probe_model_generation", side_effect=_probe_model),
        patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=False),
    ):
        result = pipeline.process(pdf, tmp_path)
    log = tmp_path / "doc" / "audit_log.json"  # absent when no event was recorded
    audit = json.loads(log.read_text()) if log.exists() else []
    events = audit["events"] if isinstance(audit, dict) else audit
    kinds = [e.get("kind") for e in events]
    halted = "PARTIAL_SAVE_VLM_TIMEOUT" in (result.error or "")
    return len(judge_calls), kinds, probe_calls, halted


def test_wedged_judge_is_cut_off_after_one_probe_slow_judge_is_not(tmp_path, monkeypatch) -> None:
    """DIFFERENCE: the same four judge timeouts; only the liveness probe differs."""
    n_w, kinds_w, probes_w, halted_w = _run_breaker(
        tmp_path / "w", monkeypatch, mode="judge", judge_probe_alive=False
    )
    n_a, kinds_a, probes_a, halted_a = _run_breaker(
        tmp_path / "a", monkeypatch, mode="judge", judge_probe_alive=True
    )
    # Wedged: the judge is called on page 1 only, later pages never reach it.
    assert n_w == 1, n_w
    assert kinds_w.count("judge_wedged_degraded_to_heuristic") == 1
    assert len(probes_w) == 1 and probes_w[0][1] == "judge-model"
    # Slow but alive: judge stays active on every page, nothing surfaced.
    assert n_a == 4, n_a
    assert "judge_wedged_degraded_to_heuristic" not in kinds_a
    assert len(probes_a) == 4
    # A judge timeout never halts the document either way.
    assert halted_w is False and halted_a is False


def test_ocr_halt_is_unaffected_by_the_judge_breaker(tmp_path, monkeypatch) -> None:
    n, kinds, probes, halted = _run_breaker(
        tmp_path / "o", monkeypatch, mode="provider", judge_probe_alive=True
    )
    assert halted is True
    assert "partial_save_vlm_timeout" in kinds
