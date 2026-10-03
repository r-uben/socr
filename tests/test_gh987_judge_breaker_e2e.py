"""#987: the judge circuit breaker, observed through ``process()``.

A wedged page judge must not cost a full judge deadline on every remaining page,
and must not make later pages weaker: a short-circuited page fails closed exactly
as a page whose judge really timed out. Real ``route_page``, real judge chain;
only the OCR engine call and the Ollama judge call are replaced. Hermetic: the
provider ladder, the judge model resolution, the judge liveness probe and the
OCR-backend probe are all patched, so nothing reaches an ambient Ollama.

Outcomes are pinned as DIFFERENCES between runs in one process (CLAUDE.md):
provider-dependent machinery may differ in CI, so no absolute tuple measured
locally is asserted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import DocumentStatus, PageOutput, PageStatus
from socr.pipeline import agentic as agentic_mod
from socr.pipeline.orchestrator import UnifiedPipeline

fitz = pytest.importorskip("fitz")

#: Scaled judge deadline: the stand-in judge outlives it, so the timeout is real.
_DEADLINE = 0.2
_JUDGE_HANG = 1.0
_PAGES = 4
BREAKER_EVENT = "judge_wedged_circuit_open"
_VOLATILE = {"input_checksum", "timings_s"}


def _pdf(tmp_path: Path) -> Path:
    doc = fitz.open()
    for n in range(_PAGES):
        page = doc.new_page()
        y = 80
        for _ in range(14):
            page.insert_text((60, y), f"Estimated coefficient 0.08{n} significant", fontsize=9)
            y += 16
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "doc.pdf"
    doc.save(str(path))
    doc.close()
    return path


def _make_pipeline(monkeypatch, *, probe, judge_calls, judge="hang", vllm=False, reprocess=False):
    """Real route_page + real judge chain; the Ollama/vLLM judge call is replaced.

    ``judge``: "hang" outlives the (scaled) deadline; "accept" returns faithful.
    ``probe``: the judge liveness answer, a ``(alive, reason)`` pair or an exception.
    """
    from socr.judge.judge import JudgeVerdict
    from socr.judge.ollama_judge import OllamaVisionJudge
    from socr.judge.vllm_judge import VLLMVisionJudge

    monkeypatch.setattr(
        agentic_mod, "DEFAULT_PROVIDER_TIMEOUTS", {e: _DEADLINE for e in EngineType}
    )

    def _judge(self, image_path, text, *a, **k):
        judge_calls.append(1)
        if judge == "accept":
            return JudgeVerdict(faithful=True, confidence=1.0)
        time.sleep(_JUDGE_HANG)
        raise AssertionError("the deadline should have fired first")

    monkeypatch.setattr(VLLMVisionJudge if vllm else OllamaVisionJudge, "judge", _judge)

    extra = {"judge_vllm_url": "http://127.0.0.1:1/v1", "judge_vllm_model": "vj"} if vllm else {}
    pipe = UnifiedPipeline(
        PipelineConfig(
            agentic=True,
            quiet=True,
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            write_manifest=False,
            judge_backend="vlm",
            reprocess=reprocess,
            **extra,
        )
    )
    detect = pipe.bd_detector.detect

    def _needs_ocr(path):
        assessment = detect(path)
        for p in assessment.pages:
            p.needs_ocr_enhancement = True
        return assessment

    pipe.bd_detector.detect = _needs_ocr
    pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
    pipe._resolve_crop_vlm_model = lambda: None
    pipe._resolve_judge_model = lambda *a, **k: "vj" if vllm else "judge-model"
    pipe._make_page_renderer = lambda state: lambda page_num: "img"
    pipe._probe_backend_idle = lambda: True  # the OCR backend is healthy

    def _answer():
        if isinstance(probe, Exception):
            raise probe
        return probe

    monkeypatch.setattr(
        "socr.pipeline.orchestrator.probe_model_generation", lambda *a, **k: _answer()
    )
    pipe_openai_kwargs: list[dict] = []

    def _openai(*a, **k):
        pipe_openai_kwargs.append(k)
        return _answer()[0]

    monkeypatch.setattr("socr.pipeline.orchestrator.probe_openai_server_idle", _openai)

    pipe.ocr_calls = []
    pipe.openai_probe_kwargs = pipe_openai_kwargs

    def ocr(state, nums, nat, eng, phase, profile=None, **_kwargs):
        pipe.ocr_calls.append(list(nums))
        return [
            PageOutput(
                page_num=p,
                text=f"Estimated coefficient text for page {p} " * 8,
                status=PageStatus.SUCCESS,
                engine="qwen",
            )
            for p in nums
        ]

    pipe._run_engine_on_pages = ocr
    return pipe


def _sidecars(out_dir: Path) -> dict[int, dict]:
    return {int(path.stem): json.loads(path.read_text()) for path in out_dir.rglob("pages/*.json")}


def _event_kinds(out_dir: Path) -> list[str]:
    logs = list(out_dir.rglob("audit_log.json"))  # absent when nothing was recorded
    if not logs:
        return []
    data = json.loads(logs[0].read_text())
    return [e.get("kind") for e in (data["events"] if isinstance(data, dict) else data)]


def _run(tmp_path: Path, monkeypatch, probe, **kw):
    judge_calls: list[int] = []
    pipe = _make_pipeline(monkeypatch, probe=probe, judge_calls=judge_calls, **kw)
    out = tmp_path / "out"
    result = pipe.process(_pdf(tmp_path), output_dir=out)
    return {
        "result": result,
        "judge_calls": len(judge_calls),
        "sidecars": _sidecars(out),
        "events": _event_kinds(out),
        "pipe": pipe,
        "out": out,
        "pdf": tmp_path / "doc.pdf",
    }


def _shape(sidecar: dict) -> dict:
    shape = {k: v for k, v in sidecar.items() if k not in _VOLATILE}
    # The breaker's own event rides the trigger page's sidecar (so a resume replays it);
    # it is the one intended difference from the slow-judge run.
    shape["audit_events"] = [
        e for e in sidecar.get("audit_events", []) if e.get("kind") != BREAKER_EVENT
    ]
    return shape


def _passed(sidecar: dict) -> bool:
    return sidecar["status"] == "success" and sidecar["audit_passed"] is True


WEDGED = (False, "timed out")
ALIVE = (True, "")


def test_wedged_judge_fails_pages_closed_exactly_like_a_slow_one_but_instantly(
    tmp_path, monkeypatch
) -> None:
    """DIFFERENCE: the same four judge timeouts; only the liveness probe differs.

    The breaker may remove the waits. It may not change what a page becomes: every
    sidecar, and the document status, must equal the run where the judge really
    timed out on every page.
    """
    wedged = _run(tmp_path / "w", monkeypatch, WEDGED)
    alive = _run(tmp_path / "a", monkeypatch, ALIVE)

    assert wedged["judge_calls"] == 1  # only the page that tripped it paid a deadline
    assert alive["judge_calls"] == _PAGES
    assert sorted(wedged["sidecars"]) == sorted(alive["sidecars"]) == list(range(1, _PAGES + 1))
    for n in wedged["sidecars"]:
        assert _shape(wedged["sidecars"][n]) == _shape(alive["sidecars"][n]), n
        assert not _passed(wedged["sidecars"][n]), f"p{n} shipped as accepted"
    assert wedged["result"].status == alive["result"].status
    assert wedged["result"].status is not DocumentStatus.SUCCESS


def test_the_breaker_is_surfaced_once_at_document_level(tmp_path, monkeypatch) -> None:
    wedged = _run(tmp_path / "w", monkeypatch, WEDGED)
    alive = _run(tmp_path / "a", monkeypatch, ALIVE)
    assert wedged["events"].count(BREAKER_EVENT) == 1
    assert BREAKER_EVENT not in alive["events"]


def test_a_probe_that_raises_counts_as_wedged(tmp_path, monkeypatch) -> None:
    """Fail closed: an unanswerable probe is not evidence of life."""
    r = _run(tmp_path, monkeypatch, RuntimeError("probe blew up"))
    assert r["judge_calls"] == 1
    assert r["events"].count(BREAKER_EVENT) == 1
    assert not any(_passed(sc) for sc in r["sidecars"].values())


def test_a_vllm_judge_is_probed_with_a_generation_not_its_listing(tmp_path, monkeypatch) -> None:
    wedged = _run(tmp_path / "w", monkeypatch, WEDGED, vllm=True)
    alive = _run(tmp_path / "a", monkeypatch, ALIVE, vllm=True)
    assert wedged["judge_calls"] == 1 and alive["judge_calls"] == _PAGES
    assert wedged["events"].count(BREAKER_EVENT) == 1
    # No cold-load allowance for an openai-compatible judge (no eviction).
    assert wedged["pipe"].openai_probe_kwargs
    assert all("generation_timeout" not in k for k in wedged["pipe"].openai_probe_kwargs)
    for n in wedged["sidecars"]:
        assert _shape(wedged["sidecars"][n]) == _shape(alive["sidecars"][n]), n


def test_short_circuited_pages_are_reprocessed_on_resume_when_the_judge_is_healthy(
    tmp_path, monkeypatch
) -> None:
    """Resume: pages failed closed by the breaker are not terminal-SUCCESS.

    CONTROL in the same process: a run whose judge accepts leaves SUCCESS pages,
    and the identical second run skips every one (0 OCR calls). The wedged run's
    second run, with a healthy judge, re-reads all four pages.
    """
    control = _run(tmp_path / "c", monkeypatch, ALIVE, judge="accept")
    assert control["judge_calls"] == _PAGES
    assert all(_passed(sc) for sc in control["sidecars"].values()), "control must pass"
    control["pipe"].config.reprocess = True  # past the document-level skip only
    control["pipe"].ocr_calls.clear()
    control["pipe"].process(control["pdf"], output_dir=control["out"])
    assert control["pipe"].ocr_calls == [], "a terminal-SUCCESS page must be skipped"

    wedged = _run(tmp_path / "w", monkeypatch, WEDGED)
    assert wedged["events"].count(BREAKER_EVENT) == 1
    wedged["pipe"].ocr_calls.clear()
    resumed = _make_pipeline(
        monkeypatch, probe=ALIVE, judge_calls=[], judge="accept", reprocess=True
    )
    resumed.process(wedged["pdf"], output_dir=wedged["out"])
    assert sorted(n for call in resumed.ocr_calls for n in call) == list(range(1, _PAGES + 1))
    assert all(_passed(sc) for sc in _sidecars(wedged["out"]).values())


def test_the_breaker_event_rides_a_page_sidecar_and_is_replayed_on_resume(
    tmp_path, monkeypatch
) -> None:
    """cubic P2: only a page's own events reach its sidecar, and the sidecar is what
    ``resume_restore_kinds`` replays. The event must sit on the trigger page AND the kind
    must be in the allowlist, or the audit log loses it after a resume."""
    wedged = _run(tmp_path, monkeypatch, WEDGED)
    carrying = [
        n
        for n, sc in wedged["sidecars"].items()
        if any(e.get("kind") == BREAKER_EVENT for e in sc.get("audit_events", []))
    ]
    assert carrying == [1], carrying  # the page whose judge timed out and tripped it
    assert BREAKER_EVENT in UnifiedPipeline.resume_restore_kinds()
