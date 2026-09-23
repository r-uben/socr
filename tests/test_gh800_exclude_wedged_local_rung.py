"""GH-800: a local rung that timed out on a rescued page, and is still wedged,
is dropped for the rest of the document instead of being retried on every page.

#799 stopped a rescued page from halting the document -- correct, the cloud rung
recovered the text -- but the ladder is built once per document, so every later
page walked into the same wedged local backend, paid its full timeout, and only
then reached the rung that works.

Two DISTINCT provider identities, as the issue requires (#799's tests used one
for both attempts and could not tell them apart): a local rung that times out
and a cloud rung that accepts. The only thing varied between runs is whether the
backend probe says the local backend is still unresponsive. Hermetic: engines
pinned (#841), the provider ladder, judge and crop-VLM probe patched, the
per-provider deadline shrunk so the timeout is real but fast.
"""

from __future__ import annotations

import time

import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.providers import PROFILE_QWEN_CLOUD, PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.pipeline import agentic as agentic_mod
from socr.pipeline.agentic import AcceptDecision
from socr.pipeline.orchestrator import UnifiedPipeline

fitz = pytest.importorskip("fitz")

#: Per-provider deadline for the run. The cloud stand-in returns immediately, so
#: this only has to be comfortably above scheduling jitter on a slow CI runner --
#: if it were not, the RESCUING rung could time out too and the test would
#: exercise the halt path instead of the exclusion path.
_DEADLINE = 0.5
#: How long the local stand-in blocks: clearly past the deadline, so its attempt is
#: recorded as a genuine provider timeout rather than a near-miss.
_LOCAL_HANG = 1.2
_PAGES = 3


class _AcceptingJudge:
    def assess(self, output, provider):
        ok = output.status == PageStatus.SUCCESS and bool(output.text.strip())
        return AcceptDecision(accept=ok, reason="accepted" if ok else "empty", confidence=1.0)


def _pdf(tmp_path):
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


def _run(tmp_path, monkeypatch, *, backend_idle: bool):
    # ``_phase_agentic`` imports this table from ``socr.pipeline.agentic`` at call
    # time, so that module is where the deadline has to be shrunk.
    monkeypatch.setattr(
        agentic_mod, "DEFAULT_PROVIDER_TIMEOUTS", {e: _DEADLINE for e in EngineType}
    )
    pipe = UnifiedPipeline(
        PipelineConfig(
            agentic=True,
            quiet=True,
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            write_manifest=False,
        )
    )
    detect = pipe.bd_detector.detect

    def _needs_ocr(path):
        assessment = detect(path)
        for p in assessment.pages:
            p.needs_ocr_enhancement = True
        return assessment

    pipe.bd_detector.detect = _needs_ocr
    pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL, PROFILE_QWEN_CLOUD]
    pipe._build_page_judge = lambda state: _AcceptingJudge()
    pipe._resolve_crop_vlm_model = lambda: None
    pipe._resolve_judge_model = lambda *a, **k: ""
    pipe._probe_backend_idle = lambda: backend_idle

    calls: list[tuple[str, int]] = []

    def spy(state, nums, nat, eng, phase, profile=None, **_kwargs):
        for n in nums:
            calls.append((profile.id if profile else "?", n))
        if profile is not None and profile.id == PROFILE_QWEN_LOCAL.id:
            time.sleep(_LOCAL_HANG)  # outlives the deadline: a real provider timeout
        return [
            PageOutput(page_num=p, text=f"text {p}", status=PageStatus.SUCCESS, engine="qwen")
            for p in nums
        ]

    pipe._run_engine_on_pages = spy
    pipe.process(_pdf(tmp_path), output_dir=tmp_path / "out")
    return calls


def _local_pages(calls):
    return sorted({n for pid, n in calls if pid == PROFILE_QWEN_LOCAL.id})


def _cloud_pages(calls):
    return sorted({n for pid, n in calls if pid == PROFILE_QWEN_CLOUD.id})


def test_the_harness_really_times_out_the_local_rung(tmp_path, monkeypatch):
    """Guard against a vacuous pass: page 1 must reach BOTH rungs -- local timed
    out, cloud rescued -- or the exclusion path is never armed."""
    calls = _run(tmp_path, monkeypatch, backend_idle=False)
    assert 1 in _local_pages(calls)
    assert 1 in _cloud_pages(calls)


def test_a_wedged_local_rung_is_dropped_after_a_rescued_timeout(tmp_path, monkeypatch):
    """The difference pin: same document, same providers; only the probe differs."""
    wedged = _run(tmp_path / "w", monkeypatch, backend_idle=False)
    healthy = _run(tmp_path / "h", monkeypatch, backend_idle=True)

    assert _local_pages(wedged) == [1], "a still-wedged local rung must not be retried"
    assert _local_pages(healthy) == list(range(1, _PAGES + 1)), (
        "a backend that answers the probe keeps its rung -- one slow page is not a wedge"
    )


def test_the_document_is_never_truncated(tmp_path, monkeypatch):
    """#227 must not regress: with the local rung dropped, every page is still read."""
    calls = _run(tmp_path, monkeypatch, backend_idle=False)
    assert _cloud_pages(calls) == list(range(1, _PAGES + 1))


def test_the_exclusion_is_recorded(tmp_path, monkeypatch):
    """Surfaced, not silent: the run's events carry the exclusion."""
    captured = {}
    real = UnifiedPipeline._exclude_wedged_local_rungs

    def _spy(self, state, page_num, decision, ladder):
        out = real(self, state, page_num, decision, ladder)
        captured["events"] = [e.kind for e in state.events]
        return out

    monkeypatch.setattr(UnifiedPipeline, "_exclude_wedged_local_rungs", _spy)
    _run(tmp_path, monkeypatch, backend_idle=False)
    assert "local_rung_excluded_after_rescue" in captured.get("events", [])
