"""GH-238: the fingerprint must record the caption engine actually selected,
not ``cfg.gemini_model`` (pure config).

``_get_vision_engine`` returns three observably different caption producers
depending on what is reachable at call time:

1. ``LocalFirstFigureEngine(OllamaFigureEngine, gemini_fallback)`` -- Ollama up
2. ``GeminiAPIEngine`` alone -- Ollama down, an API key present
3. ``None`` -- neither reachable; figures saved with no descriptions

Caption bytes differ across all three, so a document OCR'd under one and
resumed under another must not share a fingerprint. The converse matters
just as much: with no engine reachable, changing ``gemini_model`` must NOT
move the fingerprint, or every unreachable-engine run pays a needless
reprocess whenever an unused default changes.

Hermetic: no real ollama, no real Gemini call. ``OllamaFigureEngine.is_available``
is patched directly (its own probe is a bare ``httpx.get`` with a 3s timeout --
patching one level up would still let the timeout fire on a machine with no
Ollama listening). ``GEMINI_API_KEY`` is set/unset via ``monkeypatch.setenv`` /
``delenv``; the resolver checks presence only, it never calls
``GeminiAPIEngine.initialize()`` (see ``_resolve_caption_engine_identity``'s
docstring: that HTTP round trip is why it isn't imitated here either).
"""

from __future__ import annotations

import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.pipeline import orchestrator as orch

# Pins every OTHER route to ``gemini_model`` away from GEMINI (mirrors
# ``test_fingerprint_flag_coverage.test_caption_fallback_model_invalidates_
# outside_enabled_engines``'s own pin): ``fallback_chain`` defaults to
# ``[GEMINI]`` and ``primary_engine``/``local_engine`` default to AUTO, which
# probes reachability and can resolve to GEMINI -- either route would move
# the fingerprint via ``enabled_engine_determinants`` for a reason that has
# nothing to do with the caption-engine identity this file is isolating.
_ISOLATE_FROM_GEMINI_ROUTE = {
    "enabled_engines": [EngineType.QWEN],
    "fallback_chain": [],
    "primary_engine": EngineType.QWEN,
    "local_engine": EngineType.QWEN,
}


@pytest.fixture(autouse=True)
def _pin_source_digest():
    """Hold socr's own source identity constant so only the knob under test varies."""
    orch._SOURCE_DIGEST_CACHE = None
    orig = orch._socr_source_digest
    orch._socr_source_digest = lambda: "pinned-digest"
    yield
    orch._socr_source_digest = orig
    orch._SOURCE_DIGEST_CACHE = None


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def _pipeline(**overrides: object) -> orch.UnifiedPipeline:
    # #885: pin the primary/local engine by default -- this file isolates the
    # CAPTION engine's identity, and an unpinned AUTO default both shells out
    # to `ollama` and (per the module docstring above) can itself resolve to
    # GEMINI, confounding the very route this file is isolating. Callers that
    # pass their own engine fields (none currently do) still win via override.
    config = PipelineConfig(**_ISOLATE_FROM_GEMINI_ROUTE)
    for key, value in overrides.items():
        assert hasattr(config, key), f"PipelineConfig has no field {key!r}"
        setattr(config, key, value)
    return orch.UnifiedPipeline(config)


def _fingerprint(**overrides: object) -> str:
    return _pipeline(**overrides)._run_fingerprint()


def _ollama_up(monkeypatch) -> None:
    monkeypatch.setattr(
        "socr.engines.gemini_api.OllamaFigureEngine.is_available", lambda self: True
    )


def _ollama_down(monkeypatch) -> None:
    monkeypatch.setattr(
        "socr.engines.gemini_api.OllamaFigureEngine.is_available", lambda self: False
    )


# -- 1. two runs whose only change is which engine is reachable -> DIFFER ---


def test_ollama_reachable_vs_gemini_reachable_differ(monkeypatch) -> None:
    """LocalFirst+Ollama and GeminiAPIEngine-alone are different caption producers."""
    _ollama_up(monkeypatch)
    ollama_fp = _fingerprint(describe_figures=True)

    _ollama_down(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    gemini_fp = _fingerprint(describe_figures=True)

    assert ollama_fp != gemini_fp


def test_ollama_reachable_vs_nothing_reachable_differ(monkeypatch) -> None:
    """LocalFirst+Ollama vs. no engine at all -- captions vs. none."""
    _ollama_up(monkeypatch)
    ollama_fp = _fingerprint(describe_figures=True)

    _ollama_down(monkeypatch)
    nothing_fp = _fingerprint(describe_figures=True)

    assert ollama_fp != nothing_fp


def test_gemini_reachable_vs_nothing_reachable_differ(monkeypatch) -> None:
    """GeminiAPIEngine-alone vs. no engine at all -- captions vs. none."""
    _ollama_down(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    gemini_fp = _fingerprint(describe_figures=True)

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    nothing_fp = _fingerprint(describe_figures=True)

    assert gemini_fp != nothing_fp


# -- 2. no engine reachable, gemini_model changed -> AGREE (the converse) ---


def test_no_engine_reachable_ignores_gemini_model(monkeypatch) -> None:
    """Converse (the ticket's half-a-fix guard): with no engine reachable, a
    changed ``gemini_model`` must not move the fingerprint -- it names a model
    that never ran and never will while nothing is reachable.
    """
    _ollama_down(monkeypatch)
    assert _fingerprint(
        describe_figures=True, gemini_model="gemini-a", **_ISOLATE_FROM_GEMINI_ROUTE
    ) == _fingerprint(describe_figures=True, gemini_model="gemini-b", **_ISOLATE_FROM_GEMINI_ROUTE)


# -- 3. describe_figures=False -> no probe at all, field stays absent -------


def test_describe_figures_false_never_probes(monkeypatch) -> None:
    """No probe when captions cannot be produced -- mirrors the judge's
    ``--judge-backend heuristic`` short-circuit exactly.
    """
    probed = {"called": False}

    def _boom(self):
        probed["called"] = True
        raise AssertionError("OllamaFigureEngine.is_available must not be called")

    monkeypatch.setattr("socr.engines.gemini_api.OllamaFigureEngine.is_available", _boom)
    pipeline = _pipeline(describe_figures=False)
    pipeline._run_fingerprint()
    assert not probed["called"]


def test_describe_figures_false_field_is_none() -> None:
    """The field itself stays the pre-existing ``None`` sentinel: ``describe_figures=False``
    means "captions never attempted", which must not collide with
    ``CAPTION_IDENTITY_NONE`` ("captions attempted, produced none").
    """
    from unittest.mock import patch

    captured: dict[str, object] = {}

    def _capture(*_args, extra=None, **_kwargs):
        captured.update(extra or {})
        return "unused"

    with patch("ocr_output_contract.run_fingerprint", side_effect=_capture):
        _pipeline(describe_figures=False)._run_fingerprint()

    assert captured["figure_caption_fallback_model"] is None


def test_no_engine_reachable_field_is_the_sentinel_not_none(monkeypatch) -> None:
    """``describe_figures=True`` with nothing reachable records
    ``CAPTION_IDENTITY_NONE``, not the bare ``None`` the off-switch uses --
    the two "no caption" causes stay distinguishable in the fingerprint.
    """
    from unittest.mock import patch

    _ollama_down(monkeypatch)
    captured: dict[str, object] = {}

    def _capture(*_args, extra=None, **_kwargs):
        captured.update(extra or {})
        return "unused"

    with patch("ocr_output_contract.run_fingerprint", side_effect=_capture):
        _pipeline(describe_figures=True)._run_fingerprint()

    assert captured["figure_caption_fallback_model"] == orch.CAPTION_IDENTITY_NONE
    assert captured["figure_caption_fallback_model"] is not None


# -- memoization: one probe per pipeline lifetime, not one per fingerprint call --


def test_resolution_is_memoized_across_fingerprint_calls(monkeypatch) -> None:
    """``_run_fingerprint`` runs once per page; the reachability probe must not.

    #238's required evidence item 5: say how many HTTP probes the change adds
    and when. Answer: at most one ``OllamaFigureEngine.is_available()`` call
    per pipeline instance, on the first ``describe_figures=True`` fingerprint
    -- every subsequent call within the same run (i.e. every other page)
    reuses the cached identity.
    """
    calls = {"n": 0}

    def _counted(self):
        calls["n"] += 1
        return True

    monkeypatch.setattr("socr.engines.gemini_api.OllamaFigureEngine.is_available", _counted)
    pipeline = _pipeline(describe_figures=True)
    pipeline._run_fingerprint()
    pipeline._run_fingerprint()
    pipeline._run_fingerprint()
    assert calls["n"] == 1


def test_pipeline_built_via_new_without_init_does_not_explode() -> None:
    """Class-level cache default: tests (and this repo's own test suite, see
    ``_judge_model_cache``'s docstring) build pipelines via ``object.__new__``
    to skip constructor side effects. A fingerprint call on such an instance
    must not raise ``AttributeError`` on the new cache attribute.
    """
    pipeline = object.__new__(orch.UnifiedPipeline)
    pipeline.config = PipelineConfig()
    pipeline.config.describe_figures = False
    assert pipeline._caption_engine_identity_cache is False
