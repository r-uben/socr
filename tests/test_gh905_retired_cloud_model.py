"""GH-905: the retired ``qwen3.5:cloud`` must not be a default rung or the math model.

Ollama Cloud retired ``qwen3.5:cloud`` on 2026-09-25 (every call 410 Gone) while
``ollama list`` / ``/api/tags`` kept listing it. Pinned here:

1. ``PipelineConfig.math_model`` defaults to the local instruct model (the
   default-ladder half is in ``test_b2_routing.py::TestCloudRungReachable``);
2. ``cloud_model_available`` (still used if a caller names the cloud profile)
   probes by a real generation, so a 410 reads as unavailable; and
3. a rung whose CLI call fails leaves the error text in the manifest journal,
   not only on the console.

Hermetic: no daemon, no CLI, no provider. Tests that would otherwise touch the
network stub ``httpx.post`` and the reachability pre-check.
"""

from __future__ import annotations

import importlib
import json

import httpx
import pytest

import socr.core.ollama_utils as ollama_utils
from socr.core.config import DEFAULT_MATH_MODEL, EngineType, PipelineConfig
from socr.core.killable import KillableTimeoutError
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.engines import qwen as qwen_engine
from socr.pipeline.agentic import AcceptDecision
from socr.pipeline.orchestrator import UnifiedPipeline

# ---------------------------------------------------------------------------
# 1. math_model default
# ---------------------------------------------------------------------------


def test_math_model_defaults_to_the_local_instruct_model() -> None:
    cfg = PipelineConfig()
    assert cfg.math_model == DEFAULT_MATH_MODEL == "qwen3-vl:30b-a3b-instruct"
    assert cfg.math_model == PROFILE_QWEN_LOCAL.model
    assert "cloud" not in cfg.math_model


def test_default_math_model_is_not_gated_by_strict_local_or_zero_cap() -> None:
    """A local model must run under the strictest policy; that is the point."""
    pipe = UnifiedPipeline(
        PipelineConfig(strict_local=True, max_cost_per_page=0, max_cost_per_page_pinned=True)
    )
    assert pipe._corrupt_math_model_disabled_reason() == ""


def test_explicit_cloud_math_model_override_still_works_and_is_still_gated() -> None:
    """``--math-model <x>:cloud`` stays an explicit opt-in, refused by policy."""
    cfg = PipelineConfig(math_model="some-model:cloud")
    assert cfg.math_model == "some-model:cloud"
    assert UnifiedPipeline(cfg)._corrupt_math_model_disabled_reason() == ""
    strict = UnifiedPipeline(PipelineConfig(math_model="some-model:cloud", strict_local=True))
    assert "strict-local" in strict._corrupt_math_model_disabled_reason()


def test_cli_help_no_longer_names_the_retired_model_as_a_default() -> None:
    from click.testing import CliRunner

    from socr.cli import cli

    result = CliRunner().invoke(cli, ["process", "--help"])
    assert "--math-model" in result.output
    assert "default: qwen3.5:cloud" not in result.output


# ---------------------------------------------------------------------------
# 2. cloud_model_available probes by generation, not by listing
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, status: int, body: dict | None = None):
        self.status_code = status
        self._body = body or {}
        self.text = json.dumps(self._body)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status_code}",
                request=httpx.Request("POST", "http://x/api/generate"),
                response=self,  # type: ignore[arg-type]
            )

    def json(self) -> dict:
        return self._body


def _run_killable_inprocess(spec, timeout):
    module_name, _, qualname = spec.func.partition(":")
    fn = getattr(importlib.import_module(module_name), qualname)
    try:
        return fn(*spec.args, **(spec.kwargs or {}))
    except httpx.TimeoutException as exc:
        raise KillableTimeoutError(spec.func, timeout, killed=False) from exc


@pytest.fixture
def cloud_probe_env(monkeypatch):
    """Reachable host, in-process ``run_killable``; the LISTING says the model exists."""
    monkeypatch.setattr(ollama_utils, "host_reachable", lambda *a, **k: True)
    monkeypatch.setattr("socr.core.killable.run_killable", _run_killable_inprocess)
    # The old (broken) check. If cloud_model_available ever consults it again the
    # listing says "present" -- exactly what Ollama did for the retired model.
    monkeypatch.setattr(qwen_engine, "_check_ollama_model", lambda name: None)
    monkeypatch.setattr(ollama_utils, "check_ollama_model", lambda name: None)


def test_a_410_on_the_cloud_probe_is_unavailable_even_though_it_is_listed(
    cloud_probe_env, monkeypatch
):
    seen: list[dict] = []

    def _post(url, json=None, **kw):
        seen.append(json)
        return _Resp(410, {"error": "qwen3.5:397b was retired"})

    monkeypatch.setattr(httpx, "post", _post)

    assert qwen_engine.cloud_model_available() is False
    assert seen, "the probe must issue a real generation call"
    assert seen[0]["think"] is False
    assert seen[0]["options"] == {"num_predict": 1}
    assert seen[0]["model"] == "qwen3.5:cloud"


def test_a_successful_cloud_generation_is_available(cloud_probe_env, monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp(200, {"response": "h"}))
    assert qwen_engine.cloud_model_available() is True


def test_cloud_probe_difference_pin_only_the_generation_status_varies(cloud_probe_env, monkeypatch):
    """The listing is held constant ("present"); only the generation status changes."""
    outcomes = {}
    for status in (200, 410):
        monkeypatch.setattr(httpx, "post", lambda *a, _s=status, **k: _Resp(_s, {"response": "h"}))
        outcomes[status] = qwen_engine.cloud_model_available()
    assert outcomes == {200: True, 410: False}


def test_unreachable_host_is_unavailable_without_a_generation(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")

    def _boom(*a, **k):  # pragma: no cover - must not run
        raise AssertionError("no generation may be attempted against an unreachable host")

    monkeypatch.setattr(httpx, "post", _boom)
    monkeypatch.setattr("socr.core.killable.run_killable", _boom)
    assert qwen_engine.cloud_model_available() is False


def test_shared_probe_reports_the_reason(cloud_probe_env, monkeypatch):
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _Resp(410, {"error": "qwen3.5:397b was retired"})
    )
    ok, reason = ollama_utils.probe_model_generation("http://h:1", "m:cloud", 5.0)
    assert ok is False
    assert reason == "HTTP 410: qwen3.5:397b was retired"


def test_profile_qwen_cloud_stays_registered_for_historical_manifests() -> None:
    from socr.core.providers import PROFILE_QWEN_CLOUD, profile_by_id

    assert profile_by_id("qwen-cloud") is PROFILE_QWEN_CLOUD


# ---------------------------------------------------------------------------
# 3. a failing rung leaves its error in the manifest journal
# ---------------------------------------------------------------------------

_CLI_ERROR = (
    "CLI exited 1: httpx.HTTPStatusError: Client error '410 Gone' for url "
    "'http://localhost:11434/v1/chat/completions'"
)


class _RejectEmptyOrError:
    """Same verdict text the real heuristic judge gives an ERROR output."""

    def assess(self, output, provider):
        ok = output.status == PageStatus.SUCCESS and bool(output.text.strip())
        return AcceptDecision(accept=ok, reason="accepted" if ok else "empty/error output")


def _pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    y = 80
    for _ in range(14):
        page.insert_text((60, y), "Estimated coefficient 0.081 significant", fontsize=9)
        y += 16
    path = tmp_path / "doc.pdf"
    doc.save(str(path))
    doc.close()
    return path


def _journal_reasons(tmp_path, error: str) -> list[str]:
    pipe = UnifiedPipeline(
        PipelineConfig(
            agentic=True,
            quiet=True,
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
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
    pipe._build_page_judge = lambda state: _RejectEmptyOrError()
    pipe._resolve_crop_vlm_model = lambda: None
    pipe._resolve_judge_model = lambda *a, **k: ""

    def failing_rung(state, nums, nat, eng, phase, profile=None, **_kw):
        return [
            PageOutput(
                page_num=p,
                text="",
                status=PageStatus.ERROR,
                engine="qwen",
                failure_mode=FailureMode.CLI_ERROR,
                error=error,
            )
            for p in nums
        ]

    pipe._run_engine_on_pages = failing_rung
    out = tmp_path / "out"
    pipe.process(_pdf(tmp_path), output_dir=out)

    reasons: list[str] = []
    for manifest in out.rglob("manifest.json"):
        data = json.loads(manifest.read_text())
        for entry in data.get("entries", {}).values():
            reasons.extend(str(j.get("reason", "")) for j in entry.get("journal", []))
    return reasons


def test_a_rung_cli_failure_is_recorded_in_the_manifest_journal(tmp_path):
    reasons = _journal_reasons(tmp_path, _CLI_ERROR)
    assert reasons, "no journal entries were written; the fixture never reached the manifest"
    assert any("410 Gone" in r for r in reasons), reasons


def test_journal_reason_differs_exactly_by_the_provider_error(tmp_path):
    """Difference pin: same run, only the rung's error text changes."""
    a = _journal_reasons(tmp_path / "a", "CLI exited 1: alpha")
    b = _journal_reasons(tmp_path / "b", "CLI exited 1: beta")
    assert a and b
    assert a != b
    assert any("alpha" in r for r in a) and not any("beta" in r for r in a)
    assert any("beta" in r for r in b) and not any("alpha" in r for r in b)


# ---------------------------------------------------------------------------
# 4. review round 2: a pinned cloud tag on the qwen rung is policy-gated
# ---------------------------------------------------------------------------


def _qwen_cfg(**kw) -> PipelineConfig:
    return PipelineConfig(
        primary_engine=EngineType.QWEN,
        local_engine=EngineType.QWEN,
        enabled_engines=[EngineType.QWEN],
        quiet=True,
        **kw,
    )


def test_pinned_cloud_qwen_model_is_refused_under_strict_local_but_not_without_it():
    from socr.core.providers import cloud_pinned_qwen_refusal

    pin = dict(qwen_model="foo:cloud", qwen_model_pinned=True)
    on = cloud_pinned_qwen_refusal(_qwen_cfg(strict_local=True, **pin))
    off = cloud_pinned_qwen_refusal(_qwen_cfg(strict_local=False, **pin))
    assert "strict-local" in on and "foo:cloud" in on
    assert off == ""


def test_pinned_cloud_qwen_model_is_refused_under_a_typed_zero_cap():
    from socr.core.providers import cloud_pinned_qwen_refusal

    pin = dict(qwen_model="foo:cloud", qwen_model_pinned=True)
    typed_zero = cloud_pinned_qwen_refusal(
        _qwen_cfg(max_cost_per_page=0.0, max_cost_per_page_pinned=True, **pin)
    )
    defaulted_zero = cloud_pinned_qwen_refusal(_qwen_cfg(**pin))
    assert "--max-cost-per-page 0" in typed_zero
    assert defaulted_zero == ""


def test_a_local_or_unpinned_qwen_model_is_never_refused():
    from socr.core.providers import cloud_pinned_qwen_refusal

    strict = dict(strict_local=True, max_cost_per_page=0.0, max_cost_per_page_pinned=True)
    assert cloud_pinned_qwen_refusal(_qwen_cfg(**strict)) == ""
    local_pin = _qwen_cfg(qwen_model="qwen3.5:27b", qwen_model_pinned=True, **strict)
    assert cloud_pinned_qwen_refusal(local_pin) == ""


class _State:
    def __init__(self):
        self.events = []


def test_refusal_drops_only_the_qwen_rung_and_leaves_a_surfaced_audit_event():
    from socr.core.providers import PROFILE_GEMINI

    pipe = UnifiedPipeline(
        _qwen_cfg(strict_local=True, qwen_model="foo:cloud", qwen_model_pinned=True)
    )
    state = _State()
    kept = pipe._refuse_cloud_pinned_qwen_rung(state, [PROFILE_QWEN_LOCAL, PROFILE_GEMINI])
    assert kept == [PROFILE_GEMINI]
    [event] = state.events
    assert event.kind == "qwen_cloud_pin_refused"
    assert "foo:cloud" in event.detail

    # Same pin, policy off: the rung stays and nothing is recorded.
    pipe = UnifiedPipeline(
        _qwen_cfg(strict_local=False, qwen_model="foo:cloud", qwen_model_pinned=True)
    )
    state = _State()
    kept = pipe._refuse_cloud_pinned_qwen_rung(state, [PROFILE_QWEN_LOCAL, PROFILE_GEMINI])
    assert kept == [PROFILE_QWEN_LOCAL, PROFILE_GEMINI]
    assert state.events == []


def _rung_calls(tmp_path, **cfg_kw) -> int:
    pipe = UnifiedPipeline(_qwen_cfg(qwen_model="foo:cloud", qwen_model_pinned=True, **cfg_kw))
    detect = pipe.bd_detector.detect

    def _needs_ocr(path):
        assessment = detect(path)
        for p in assessment.pages:
            p.needs_ocr_enhancement = True
        return assessment

    pipe.bd_detector.detect = _needs_ocr
    pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
    pipe._build_page_judge = lambda state: _RejectEmptyOrError()
    pipe._resolve_crop_vlm_model = lambda: None
    pipe._resolve_judge_model = lambda *a, **k: ""
    calls: list[str] = []

    def spy(state, nums, nat, eng, phase, profile=None, **_kw):
        calls.append(profile.id if profile else "?")
        return [
            PageOutput(page_num=p, text=f"t {p}", status=PageStatus.SUCCESS, engine="qwen")
            for p in nums
        ]

    pipe._run_engine_on_pages = spy
    pipe.process(_pdf(tmp_path), output_dir=tmp_path / "out")
    return len(calls)


def test_strict_local_stops_pages_reaching_a_pinned_cloud_qwen_model(tmp_path):
    """Difference pin through the real ``_phase_agentic``: only strict_local varies."""
    allowed = _rung_calls(tmp_path / "off", strict_local=False)
    refused = _rung_calls(tmp_path / "on", strict_local=True)
    assert allowed > 0
    assert refused == 0


# ---------------------------------------------------------------------------
# 5. review round 2: the folded-in provider error is capped and not duplicated
# ---------------------------------------------------------------------------


def _err_output(reason: str, error: str) -> PageOutput:
    return PageOutput(
        page_num=1,
        text="",
        status=PageStatus.ERROR,
        engine="qwen",
        error=error,
        skip_reason=reason,
    )


def test_a_long_multiline_provider_error_is_flattened_and_capped():
    cap = UnifiedPipeline._SKIP_REASON_ERROR_MAX_CHARS
    error = "CLI exited 1: " + "\n".join(["Traceback line " + "x" * 80] * 200)
    out = UnifiedPipeline._skip_reason_with_provider_error(_err_output("empty/error output", error))
    assert out.startswith("empty/error output: CLI exited 1: ")
    assert "\n" not in out
    assert len(out) <= len("empty/error output: ") + cap + len("...")


def test_a_timeout_reason_is_not_duplicated_with_its_own_error():
    from socr.pipeline.agentic import REASON_PROVIDER_TIMEOUT

    out = UnifiedPipeline._skip_reason_with_provider_error(
        _err_output(REASON_PROVIDER_TIMEOUT, "qwen: timed out after 300s")
    )
    assert out == REASON_PROVIDER_TIMEOUT
