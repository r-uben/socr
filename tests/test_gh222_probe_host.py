"""GH-222: the cascade-halt liveness probe must not be hardcoded to localhost.

``probe_ollama_idle(host="http://localhost:11434")`` with a call site that
passed no ``host`` meant that on any deployment without a local Ollama daemon —
vLLM, HPC, a remote Ollama host — the probe returned False unconditionally,
forever, on a perfectly healthy machine. A single timeout anywhere in the ladder
then truncated the document with ``PARTIAL_SAVE_VLM_TIMEOUT``, naming a hardware
failure that never happened. ``--strict-local`` on vLLM is the worst case: the
ladder is local-only, so one slow page ends the run.

Scope is the PROBE only. Which attempts arm the guard (``_had_timeout`` scanning
every attempt, and matching the bare substring ``"timeout"``) is #221/#227's
territory and is deliberately untouched — see ``test_trigger_predicate_unchanged``.

Hermetic: ``httpx.get`` is stubbed, so no network call is made.
"""

from __future__ import annotations

import pytest

from socr.tables import extract as extract_mod
from socr.tables.extract import probe_ollama_idle


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
    """
    import inspect

    from socr.pipeline import orchestrator

    src = inspect.getsource(orchestrator.UnifiedPipeline._phase_agentic)
    assert '_had_timeout = any("timeout" in (att.reason or "") for att in decision.attempts)' in src
