"""GH-221: the liveness probe is blind to a wedged GPU.

``probe_ollama_idle``/``probe_openai_server_idle`` used to be a lightweight
``/api/tags``-or-``/models`` ping. The issue measured that ping returning
200 OK in 0.05-0.14s while ``qwen3-vl:30b-a3b-instruct`` was mid-generation at
100% GPU — the HTTP layer answers regardless of what the model is doing, so
the cascade-halt guard (``_had_timeout and not probe_..._idle()``) never
armed for the exact failure it exists to catch.

Scope: GH-222 (host resolution) is untouched and covered by
``test_gh222_probe_host.py``; the ``recorded_urls`` fixture there now stubs
``httpx.post`` too so this ticket's change does not silently break it.

Hermetic: every test here stubs ``httpx.get``/``httpx.post`` on
``socr.tables.extract``; nothing touches the network. CI has no ollama and no
provider — a test that passes here because a real Ollama daemon happens to be
running and fails in CI is worse than no test.
"""

from __future__ import annotations

import httpx
import pytest

from socr.tables import extract as extract_mod
from socr.tables.extract import (
    TableCropExtractor,
    _CropTimeoutError,
    probe_ollama_idle,
    probe_openai_server_idle,
)


# ---------------------------------------------------------------------------
# AC1 / AC2 — the canary must tell a wedged GPU from a healthy one, and must
# not false-positive on a healthy one (a false halt stops a legitimate run).
# ---------------------------------------------------------------------------


def test_wedged_gpu_with_healthy_http_layer_is_detected_as_not_idle(monkeypatch) -> None:
    """AC1 — the defect this ticket fixes.

    ``/api/tags`` answers 200 OK instantly (the HTTP layer IS healthy); the
    generation call blocks until it times out, exactly like a request queued
    behind an in-flight generation on a wedged GPU.
    """

    class _TagsResp:
        def raise_for_status(self) -> None:
            return None

    def _fake_get(url, *args, **kwargs):
        assert "/api/tags" in url
        return _TagsResp()

    def _fake_post(url, *args, **kwargs):
        assert "/api/generate" in url
        raise httpx.TimeoutException("wedged: queued behind an in-flight generation")

    monkeypatch.setattr(extract_mod.httpx, "get", _fake_get)
    monkeypatch.setattr(extract_mod.httpx, "post", _fake_post)

    assert probe_ollama_idle("http://gpu-node:11434") is False


def test_healthy_backend_is_still_detected_as_idle(monkeypatch) -> None:
    """AC2 — the reverse case matters as much as AC1: no false halt."""

    class _Resp:
        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(extract_mod.httpx, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(extract_mod.httpx, "post", lambda *a, **k: _Resp())

    assert probe_ollama_idle("http://gpu-node:11434") is True


def test_the_same_process_reports_both_outcomes_from_one_toggle(monkeypatch) -> None:
    """Pin a DIFFERENCE, not an absolute: only the canary's simulated response
    changes between the two calls, in the same process, and the decision
    differs exactly as intended."""

    class _TagsResp:
        def raise_for_status(self) -> None:
            return None

    wedged = {"value": True}

    def _fake_get(url, *args, **kwargs):
        return _TagsResp()

    def _fake_post(url, *args, **kwargs):
        if wedged["value"]:
            raise httpx.TimeoutException("wedged")
        return _TagsResp()

    monkeypatch.setattr(extract_mod.httpx, "get", _fake_get)
    monkeypatch.setattr(extract_mod.httpx, "post", _fake_post)

    wedged_result = probe_ollama_idle("http://gpu-node:11434")
    wedged["value"] = False
    healthy_result = probe_ollama_idle("http://gpu-node:11434")

    assert wedged_result is False
    assert healthy_result is True
    assert wedged_result != healthy_result


def test_openai_compatible_backend_same_distinction(monkeypatch) -> None:
    """The vLLM/SGLang sibling must draw the same line."""

    class _Resp:
        def raise_for_status(self) -> None:
            return None

    def _fake_get(url, *args, **kwargs):
        assert "/models" in url
        return _Resp()

    calls = {"n": 0}

    def _fake_post(url, *args, **kwargs):
        calls["n"] += 1
        assert "/chat/completions" in url
        if calls["n"] == 1:
            raise httpx.TimeoutException("wedged")
        return _Resp()

    monkeypatch.setattr(extract_mod.httpx, "get", _fake_get)
    monkeypatch.setattr(extract_mod.httpx, "post", _fake_post)

    assert probe_openai_server_idle("http://gpu-node:8000/v1") is False
    assert probe_openai_server_idle("http://gpu-node:8000/v1") is True


def test_ollama_canary_exercises_the_vision_path_not_a_text_one(monkeypatch) -> None:
    """Review gap: the workload this guards (``TableCropExtractor``) reads
    IMAGES, not plain text. A probe that sends no image exercises a different
    code path than the one that wedges — a hang localised to image handling
    could pass a text-only canary while the vision path stays jammed. The
    canary's request must carry ``images``, matching every other vision call
    in this codebase (judge/ollama_judge.py, judge/table_rung_ollama.py,
    math/equation_latex.py, engines/gemini_api.py)."""

    class _Resp:
        def raise_for_status(self) -> None:
            return None

    captured: dict = {}

    def _fake_post(url, *args, **kwargs):
        captured.update(kwargs.get("json", {}))
        return _Resp()

    monkeypatch.setattr(extract_mod.httpx, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(extract_mod.httpx, "post", _fake_post)

    probe_ollama_idle("http://gpu-node:11434")

    assert captured.get("images"), "the Ollama canary sent no image payload"
    assert isinstance(captured["images"][0], str) and captured["images"][0], (
        "the image payload must be a non-empty base64 string"
    )


def test_openai_canary_exercises_the_vision_path_not_a_text_one(monkeypatch) -> None:
    """Same gap, OpenAI-compatible sibling: the message must carry an
    ``image_url`` part, not text only."""

    class _Resp:
        def raise_for_status(self) -> None:
            return None

    captured: dict = {}

    def _fake_post(url, *args, **kwargs):
        captured.update(kwargs.get("json", {}))
        return _Resp()

    monkeypatch.setattr(extract_mod.httpx, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(extract_mod.httpx, "post", _fake_post)

    probe_openai_server_idle("http://gpu-node:8000/v1")

    content = captured["messages"][0]["content"]
    image_parts = [part for part in content if part.get("type") == "image_url"]
    assert image_parts, "the OpenAI-compatible canary sent no image_url part"
    assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# AC3 — the canary must not run on the happy path (no timeout observed).
# ---------------------------------------------------------------------------


def test_canary_never_fires_on_a_successful_crop_read(tmp_path, monkeypatch) -> None:
    """AC3 — a page with no timeout must not gain a generation call.

    ``TableCropExtractor.extract`` is the real call site that reaches
    ``_probe_reader_idle`` -> the canary, gated behind ``except
    _CropTimeoutError``. Driving it with a reader that never times out must
    produce zero POST calls: the canary piggybacks on the crop reader's own
    post-timeout call site, it does not add a call of its own to the per-page
    happy path.
    """
    fitz = pytest.importorskip("fitz")
    from socr.tables.locate import TableBox

    post_calls: list[str] = []
    monkeypatch.setattr(extract_mod.httpx, "post", lambda url, *a, **k: post_calls.append(url))

    pdf = tmp_path / "t.pdf"
    doc = fitz.open()
    doc.new_page(width=500, height=600)
    doc.save(str(pdf))
    doc.close()

    class _HealthyReader:
        host = "http://localhost:11434"
        model = "qwen3-vl:30b-a3b-instruct"
        timeout = 120.0

        def read(self, image_path) -> str:
            return "| a | b |\n| - | - |\n| 1 | 2 |\n"

    extractor = TableCropExtractor(reader=_HealthyReader())
    box = TableBox(bbox=(100.0, 100.0, 400.0, 400.0), source="ruled")

    crops = extractor.extract(pdf, 1, [box])

    assert crops and crops[0].markdown.strip(), "the happy-path crop read did not run at all"
    assert post_calls == [], (
        f"a generation canary POST fired with no timeout observed: {post_calls}"
    )


# ---------------------------------------------------------------------------
# AC4 — no new magic threshold: the canary's timeout is derived, not invented.
# ---------------------------------------------------------------------------


def test_generation_canary_timeout_is_derived_not_invented(monkeypatch) -> None:
    """The default ``generation_timeout`` must equal the existing
    ``_CROP_DEADLINE_FLOOR_S`` constant this file already uses to budget a
    normal crop read — not a second, independently-chosen number."""

    class _TagsResp:
        def raise_for_status(self) -> None:
            return None

    seen_timeouts: list[float] = []

    def _fake_post(url, *args, **kwargs):
        seen_timeouts.append(kwargs["timeout"])
        return _TagsResp()

    monkeypatch.setattr(extract_mod.httpx, "get", lambda *a, **k: _TagsResp())
    monkeypatch.setattr(extract_mod.httpx, "post", _fake_post)

    probe_ollama_idle("http://gpu-node:11434")

    assert seen_timeouts == [extract_mod._CROP_DEADLINE_FLOOR_S], seen_timeouts


def test_generation_timeout_override_is_still_honoured(monkeypatch) -> None:
    """A caller that DOES have a basis (e.g. the observed timeout) may still
    override the default rather than being forced onto the floor."""

    class _TagsResp:
        def raise_for_status(self) -> None:
            return None

    seen_timeouts: list[float] = []

    def _fake_post(url, *args, **kwargs):
        seen_timeouts.append(kwargs["timeout"])
        return _TagsResp()

    monkeypatch.setattr(extract_mod.httpx, "get", lambda *a, **k: _TagsResp())
    monkeypatch.setattr(extract_mod.httpx, "post", _fake_post)

    probe_ollama_idle("http://gpu-node:11434", generation_timeout=99.0)

    assert seen_timeouts == [99.0], seen_timeouts


# ---------------------------------------------------------------------------
# AC5 — unreachable backend stays "not idle", never raises.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "probe_fn_name,url_fragment",
    [
        ("probe_ollama_idle", "http://gpu-node:11434"),
        ("probe_openai_server_idle", "http://gpu-node:8000/v1"),
    ],
)
def test_connection_refused_on_the_generation_call_fails_closed(
    monkeypatch, probe_fn_name, url_fragment
) -> None:
    """The precondition passes, but the generation call itself hits a dead
    socket — this must read as "not idle", never escape as an exception into
    the pipeline (a lost document is worse than a conservative halt)."""

    class _Resp:
        def raise_for_status(self) -> None:
            return None

    def _fake_post(url, *args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(extract_mod.httpx, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(extract_mod.httpx, "post", _fake_post)

    probe_fn = getattr(extract_mod, probe_fn_name)
    assert probe_fn(url_fragment) is False


def test_precondition_failure_short_circuits_before_any_generation_call(monkeypatch) -> None:
    """An unreachable host must fail at the cheap ``/api/tags`` precondition
    and never reach the generation canary at all."""

    def _fake_get(url, *args, **kwargs):
        raise httpx.ConnectError("connection refused")

    post_calls: list[str] = []
    monkeypatch.setattr(extract_mod.httpx, "get", _fake_get)
    monkeypatch.setattr(extract_mod.httpx, "post", lambda url, *a, **k: post_calls.append(url))

    assert probe_ollama_idle("http://gpu-node:11434") is False
    assert post_calls == [], "generation canary ran despite the precondition failing"


# ---------------------------------------------------------------------------
# Integration: the real cascade-halt guard actually reads the canary's verdict.
# ---------------------------------------------------------------------------


def test_cascade_halt_arms_only_when_the_functional_canary_reports_wedged(
    tmp_path, monkeypatch
) -> None:
    """End-to-end through ``UnifiedPipeline.process()``: a timeout plus a
    genuinely wedged GPU halts; the same timeout plus a genuinely idle GPU
    does not. Only the canary's simulated response changes between the two
    ``probe_ollama_idle`` calls made inside the same run.

    Hermetic: ``_available_engines_for_agentic`` is patched so the ladder is
    non-empty regardless of whether ollama/qwen is installed locally; the
    real ``probe_ollama_idle`` runs, but its own httpx calls are stubbed.
    """
    from unittest.mock import MagicMock, patch

    from socr.core.config import EngineType
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.core.result import PageOutput, PageStatus
    from test_pp2_agentic_fuse import _make_bd_assessment, _make_config, _make_pipeline, _real_pdf

    class _TagsResp:
        def raise_for_status(self) -> None:
            return None

    def _run(wedged: bool) -> str:
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        monkeypatch.setattr(extract_mod.httpx, "get", lambda *a, **k: _TagsResp())

        def _fake_post(url, *args, **kwargs):
            if wedged:
                raise httpx.TimeoutException("wedged")
            return _TagsResp()

        monkeypatch.setattr(extract_mod.httpx, "post", _fake_post)

        pdf_path = _real_pdf(tmp_path, page_count=2)
        config = _make_config(agentic=True, enabled_engines=[EngineType.QWEN])
        pipeline = _make_pipeline(config)
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(2, born_digital_pages=set())

        def _fake_route(page_num, ladder, run_provider, judge, **kwargs):
            prof = ladder[0]
            out = PageOutput(
                page_num=page_num,
                text="",
                status=PageStatus.ERROR,
                engine="qwen",
                audit_passed=False,
            )
            from socr.pipeline.agentic import PageDecision, ProviderAttempt

            att = ProviderAttempt(
                engine=prof.engine,
                output=out,
                cost_usd=0.0,
                accepted=False,
                reason="provider timeout",
                provider_id=prof.id,
                model=prof.model,
                backend=prof.backend,
            )
            return PageDecision(page_num=page_num, final_output=out, attempts=[att])

        with (
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch("socr.pipeline.orchestrator.route_page", side_effect=_fake_route),
        ):
            result = pipeline.process(pdf_path, tmp_path)
        return result.error or ""

    wedged_error = _run(wedged=True)
    healthy_error = _run(wedged=False)

    assert "PARTIAL_SAVE_VLM_TIMEOUT" in wedged_error, wedged_error
    assert "PARTIAL_SAVE_VLM_TIMEOUT" not in healthy_error, healthy_error
    assert wedged_error != healthy_error
