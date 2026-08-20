"""GH-222: the cascade-halt liveness probe must not be hardcoded to localhost.

``probe_ollama_idle(host="http://localhost:11434")`` with a call site that
passed no ``host`` meant that on any deployment without a local Ollama daemon —
vLLM, HPC, a remote Ollama host — the probe returned False unconditionally,
forever, on a perfectly healthy machine. A single timeout anywhere in the ladder
then truncated the document with ``PARTIAL_SAVE_VLM_TIMEOUT``, naming a hardware
failure that never happened. ``--strict-local`` on vLLM is the worst case: the
ladder is local-only, so one slow page ends the run.

Two things have to be right, and the second is the one #222 is actually about:
the host must be resolved rather than assumed, AND the resolution must reach the
real CLI path. A configured vLLM backend was still asking an Ollama daemon at
localhost, because the deployment path constructs ``UnifiedPipeline(config)`` and
nothing assigns a probe seam.

Scope is the PROBE only. Which attempts arm the guard (``_had_timeout`` scanning
every attempt, and matching the bare substring ``"timeout"``) is #221/#227's
territory and is deliberately untouched — see ``test_trigger_predicate_unchanged``.
This PR does NOT close #222: it fixes which machine is asked, not what the probe
can detect.

Hermetic: ``httpx.get`` is stubbed, so no network call is made.
"""

from __future__ import annotations

import pytest

from socr.core.config import PipelineConfig
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables import extract as extract_mod
from socr.tables.extract import probe_ollama_idle, resolve_ollama_host


@pytest.fixture
def recorded_urls(monkeypatch) -> list[str]:
    """Capture the URL the probe asks for; never touch the network."""
    urls: list[str] = []

    class _Resp:
        def raise_for_status(self) -> None:
            return None

    def _fake_get(url, *args, **kwargs):
        urls.append(url)
        return _Resp()

    monkeypatch.setattr(extract_mod.httpx, "get", _fake_get)
    return urls


def test_probe_targets_the_configured_host_not_localhost(recorded_urls, monkeypatch) -> None:
    """The defect: with OLLAMA_HOST pointing elsewhere, the probe still asked
    localhost — and so reported a healthy remote backend dead."""
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-node.cluster:11434")

    assert probe_ollama_idle() is True
    assert recorded_urls == ["http://gpu-node.cluster:11434/api/tags"], (
        f"the probe asked the wrong machine: {recorded_urls}"
    )


def test_bare_host_env_keeps_the_daemon_default_port(recorded_urls, monkeypatch) -> None:
    """``OLLAMA_HOST`` is very commonly spelled bare. Reading it literally would
    resolve to port 80 and merely RELOCATE the wrong-machine defect."""
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1")

    probe_ollama_idle()

    assert recorded_urls == ["http://127.0.0.1:11434/api/tags"], recorded_urls


def test_localhost_remains_the_default_when_nothing_is_configured(
    recorded_urls, monkeypatch
) -> None:
    """Reverse regression: an existing local deployment must be unaffected.

    A fix that changed the default would break every machine that legitimately
    runs Ollama on localhost and never set the variable.
    """
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    probe_ollama_idle()

    assert recorded_urls == ["http://localhost:11434/api/tags"], recorded_urls


def test_explicit_host_argument_still_wins(recorded_urls, monkeypatch) -> None:
    """Reverse regression: the existing positional call in ``extract.py``'s
    crop-timeout path passes ``self._reader.host`` and must keep winning."""
    monkeypatch.setenv("OLLAMA_HOST", "http://ignored:11434")

    probe_ollama_idle("http://explicit:9999")

    assert recorded_urls == ["http://explicit:9999/api/tags"], recorded_urls


def test_trigger_predicate_unchanged() -> None:
    """Scope guard: this ticket must not touch WHICH attempts arm the halt.

    Narrowing ``_had_timeout`` needs the timed-out attempt to carry its backend
    identity (#159), and #227 warns that fixing #221's probe alone makes
    behaviour worse. The predicate stays exactly as it was; only the machine the
    probe asks about changes.

    This pins the SPELLING. On its own that is weak — it would still pass if a
    second condition bypassed the predicate while the line survived — so
    ``test_probe_is_consulted_only_after_a_timeout`` below pins the BEHAVIOUR.
    """
    import inspect

    from socr.pipeline import orchestrator

    src = inspect.getsource(orchestrator.UnifiedPipeline._phase_agentic)
    assert '_had_timeout = any("timeout" in (att.reason or "") for att in decision.attempts)' in src


@pytest.mark.parametrize(
    ("attempt_reason", "expect_probe"),
    [("provider timeout", True), ("judge reject", False)],
)
def test_probe_is_consulted_only_after_a_timeout(
    tmp_path, monkeypatch, attempt_reason, expect_probe
) -> None:
    """Behavioural half of the scope guard: the trigger, not its spelling.

    Drives the real ``process()`` cascade-halt path and asserts the probe is
    consulted iff an attempt's reason contains ``"timeout"``. A second condition
    that armed the guard some other way — the exact creep the source-text
    assertion above cannot see — fails here.

    ``_available_engines_for_agentic`` is patched: CI has no ollama and no
    provider, so without it the ladder is empty and the loop bails before the
    guard is reached.
    """
    from unittest.mock import MagicMock, patch

    from socr.core.config import EngineType
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.core.result import PageOutput, PageStatus

    # Reuses the cascade-halt fixture that already exists for PP-2 rather than
    # building a second one that could drift from it.
    from test_pp2_agentic_fuse import (
        _make_bd_assessment,
        _make_config,
        _make_pipeline,
        _real_pdf,
    )

    pdf_path = _real_pdf(tmp_path, page_count=1)
    pipeline = _make_pipeline(_make_config(agentic=True, enabled_engines=[EngineType.QWEN]))
    pipeline.bd_detector = MagicMock()
    pipeline.bd_detector.detect.return_value = _make_bd_assessment(1, born_digital_pages=set())

    def _fake_route(page_num, ladder, run_provider, judge, **kwargs):
        from socr.pipeline.agentic import PageDecision, ProviderAttempt

        prof = ladder[0]
        out = PageOutput(
            page_num=page_num,
            text="",
            status=PageStatus.ERROR,
            engine="qwen",
            audit_passed=False,
        )
        att = ProviderAttempt(
            engine=prof.engine,
            output=out,
            cost_usd=0.0,
            accepted=False,
            reason=attempt_reason,
            provider_id=prof.id,
            model=prof.model,
            backend=prof.backend,
        )
        return PageDecision(page_num=page_num, final_output=out, attempts=[att])

    probe_calls: list[object] = []

    def _recording_probe(*args, **kwargs):
        probe_calls.append((args, kwargs))
        return False

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch("socr.pipeline.orchestrator.route_page", side_effect=_fake_route),
        patch("socr.pipeline.orchestrator.probe_ollama_idle", side_effect=_recording_probe),
    ):
        result = pipeline.process(pdf_path, tmp_path)

    assert bool(probe_calls) is expect_probe, (
        f"reason={attempt_reason!r}: probe consulted={bool(probe_calls)}, expected {expect_probe}"
    )
    halted = "PARTIAL_SAVE_VLM_TIMEOUT" in (result.error or "")
    assert halted is expect_probe, (
        f"reason={attempt_reason!r}: halted={halted}, expected {expect_probe}"
    )


def test_existing_module_level_patches_still_reach_the_probe(monkeypatch) -> None:
    """The module-level ``probe_ollama_idle`` name is load-bearing.

    Five test files patch ``socr.pipeline.orchestrator.probe_ollama_idle``.
    Routing around that name would leave every one of those patches silently
    ineffective — the tests would keep passing while testing nothing. This
    fails loudly if the indirection is ever "cleaned up".
    """
    from unittest.mock import patch

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    pipeline = _pipeline()  # default backend: Ollama

    with patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True) as patched:
        assert pipeline._probe_backend_idle() is True

    assert patched.call_count == 1, "the module-level name was bypassed"


# ----------------------------------------------------------------------
# The blocking defect: the resolution must reach the REAL CLI path.
#
# The CLI only ever does ``UnifiedPipeline(config)``. Nothing assigns a probe
# seam, so a correct resolver that only a hand-injected seam can reach fixes
# nothing for an actual deployment.
# ----------------------------------------------------------------------


def _pipeline(**overrides) -> UnifiedPipeline:
    """A pipeline built the way the CLI builds one: config in, nothing injected."""
    return UnifiedPipeline(PipelineConfig(**overrides))


def test_configured_vllm_backend_is_not_probed_at_ollama_localhost(
    recorded_urls, monkeypatch
) -> None:
    """The blocking defect, on the path the CLI actually takes.

    An HPC run serves the VLM with vLLM on a remote node. Asking an Ollama daemon
    at localhost about it answers "dead" forever on a healthy machine, and one
    timeout anywhere in the ladder then truncates the document and blames
    hardware. Nobody assigns ``backend_probe``, so the default path has to be
    right by itself.
    """
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    pipeline = _pipeline(
        qwen_backend="vllm",
        qwen_vllm_url="http://gpu-node.cluster:8000/v1",
    )

    assert pipeline.backend_probe is None, "the CLI injects nothing; the default must be right"
    assert pipeline._probe_backend_idle() is True
    assert recorded_urls == ["http://gpu-node.cluster:8000/v1/models"], (
        f"the probe asked the wrong machine (or the wrong endpoint): {recorded_urls}"
    )


def test_vllm_probe_uses_an_endpoint_a_vllm_server_actually_serves(
    recorded_urls, monkeypatch
) -> None:
    """/api/tags is Ollama's. A vLLM server 404s it, which reads as "dead"."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    pipeline = _pipeline(qwen_backend="vllm", qwen_vllm_url="http://gpu-node:8000/v1")

    pipeline._probe_backend_idle()

    assert recorded_urls
    assert not any("/api/tags" in url for url in recorded_urls), recorded_urls


def test_ollama_backend_on_a_remote_host_is_probed_there(recorded_urls, monkeypatch) -> None:
    """The other half of the same defect: a remote Ollama daemon."""
    monkeypatch.setenv("OLLAMA_HOST", "gpu-node")
    pipeline = _pipeline(qwen_backend="ollama")

    assert pipeline._probe_backend_idle() is True
    assert recorded_urls == ["http://gpu-node:11434/api/tags"], recorded_urls


def test_explicit_config_host_outranks_the_environment(recorded_urls, monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://ignored:11434")
    pipeline = _pipeline(ollama_host="http://pinned:11434")

    pipeline._probe_backend_idle()

    assert recorded_urls == ["http://pinned:11434/api/tags"], recorded_urls


def test_default_local_deployment_is_unchanged(recorded_urls, monkeypatch) -> None:
    """Reverse regression: a plain local run must still probe localhost Ollama."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    pipeline = _pipeline()  # qwen_backend defaults to "auto"

    pipeline._probe_backend_idle()

    assert recorded_urls == ["http://localhost:11434/api/tags"], recorded_urls


def test_injected_backend_probe_still_wins(monkeypatch) -> None:
    """Reverse regression: the override seam keeps overriding."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    pipeline = _pipeline(qwen_backend="vllm")
    pipeline.backend_probe = lambda: True

    def _no_network(*args, **kwargs):
        raise AssertionError("an injected probe must short-circuit the built-in one")

    monkeypatch.setattr(extract_mod.httpx, "get", _no_network)

    assert pipeline._probe_backend_idle() is True


def test_probe_and_crop_reader_agree_on_which_server_to_ask() -> None:
    """They must not drift: one predicate decides both.

    The reader and the probe reading different config is how a run ends up
    sending crops to one machine and asking a different one whether it is alive.
    """
    from socr.tables.extract import (
        OPENAI_COMPATIBLE_BACKENDS,
        OllamaTableReader,
        VllmTableReader,
        make_table_reader,
    )

    for backend in ("auto", "ollama", "vllm", "sglang", "openai", "api"):
        reader = make_table_reader(backend=backend, model="m")
        pipeline = _pipeline(qwen_backend=backend)
        reader_is_openai = isinstance(reader, VllmTableReader)
        assert reader_is_openai is (backend in OPENAI_COMPATIBLE_BACKENDS), backend
        assert pipeline._local_backend_is_openai_compatible() is reader_is_openai, backend
        if not reader_is_openai:
            assert isinstance(reader, OllamaTableReader)


# ----------------------------------------------------------------------
# IPv6 and the other host spellings
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # A bare IPv6 literal is a spelling shells and users really produce.
        # Unbracketed it is not a URL at all: httpx cannot connect to
        # ``http://::1`` and urlsplit cannot find a port in it.
        ("::1", "http://[::1]:11434"),
        ("[::1]", "http://[::1]:11434"),
        ("[::1]:11434", "http://[::1]:11434"),
        ("[::1]:9999", "http://[::1]:9999"),
        ("http://::1", "http://[::1]:11434"),
        ("2001:db8::1", "http://[2001:db8::1]:11434"),
        ("fe80::1%eth0", "http://[fe80::1%eth0]:11434"),
        # One colon is a port, not an address.
        ("gpu-node", "http://gpu-node:11434"),
        ("gpu-node:9999", "http://gpu-node:9999"),
        ("127.0.0.1", "http://127.0.0.1:11434"),
        # Scheme already present, and a trailing slash.
        ("http://gpu-node:11434", "http://gpu-node:11434"),
        ("https://gpu-node:8443", "https://gpu-node:8443"),
        ("http://gpu-node/", "http://gpu-node:11434/"),
        # Blank means unset, not "an empty host".
        ("", "http://localhost:11434"),
        ("   ", "http://localhost:11434"),
    ],
)
def test_host_shapes_resolve_to_a_usable_url(raw, expected, monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert resolve_ollama_host(raw) == expected


def test_bare_ipv6_actually_reaches_the_probe_as_a_valid_url(recorded_urls, monkeypatch) -> None:
    """End of the same defect: the resolved value has to be a URL httpx accepts."""
    monkeypatch.setenv("OLLAMA_HOST", "::1")

    assert probe_ollama_idle() is True
    assert recorded_urls == ["http://[::1]:11434/api/tags"], recorded_urls


def test_an_unparseable_host_fails_the_probe_instead_of_raising(monkeypatch) -> None:
    """A malformed host must produce a failed probe, never an exception.

    The caller is a liveness check on a failure path; raising there would lose
    the document over a typo in an environment variable.
    """
    monkeypatch.setenv("OLLAMA_HOST", "::1:11434")  # no unambiguous reading

    assert resolve_ollama_host() == "http://[::1:11434]"  # bracketed, still unparseable
    assert probe_ollama_idle() is False
