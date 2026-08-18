"""Issues #229/#230/#232/#233: output-changing config knobs missing from the fingerprint.

``_run_fingerprint`` decides whether an already-finished page may be reused on a
resumed run. Four knobs changed the saved ``.md`` without moving the fingerprint, so
flipping one silently reused output produced under the *other* setting:

* ``auto_patch_tables``      (#229) rewrites table cells
* ``clean_equation_model``   (#230) produces the LaTeX sidecars
* ``figures_engine`` model   (#232) produces figure captions
* ``recover_corrupt_math`` / ``math_model`` (#233) rewrite equation text

Hermetic: no engine, no provider, no ollama. The source digest is pinned so it cannot
mask a missing knob by changing for an unrelated reason.
"""

from __future__ import annotations

import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.pipeline import orchestrator as orch


@pytest.fixture(autouse=True)
def _pin_source_digest(monkeypatch: pytest.MonkeyPatch):
    """Hold socr's own source identity constant so only the knob under test varies."""
    orch._SOURCE_DIGEST_CACHE = None
    monkeypatch.setattr(orch, "_socr_source_digest", lambda: "pinned-digest")
    yield
    orch._SOURCE_DIGEST_CACHE = None


def _fingerprint(**overrides: object) -> str:
    config = PipelineConfig()
    for key, value in overrides.items():
        assert hasattr(config, key), f"PipelineConfig has no field {key!r}"
        setattr(config, key, value)
    return orch.UnifiedPipeline(config)._run_fingerprint()


def test_auto_patch_tables_invalidates() -> None:
    """#229: table patching rewrites cells, so the two states are different runs."""
    assert _fingerprint(auto_patch_tables=False) != _fingerprint(auto_patch_tables=True)


def test_recover_corrupt_math_invalidates() -> None:
    """#233: the corrupt-font math lane replaces page text."""
    assert _fingerprint(recover_corrupt_math=False) != _fingerprint(recover_corrupt_math=True)


def test_math_model_invalidates_while_the_lane_is_on() -> None:
    """#233: a different VLM produces different equation text."""
    assert _fingerprint(recover_corrupt_math=True, math_model="model-a") != _fingerprint(
        recover_corrupt_math=True, math_model="model-b"
    )


def test_math_model_is_ignored_while_the_lane_is_off() -> None:
    """Converse: an unused model default must not force a needless reprocess."""
    assert _fingerprint(recover_corrupt_math=False, math_model="model-a") == _fingerprint(
        recover_corrupt_math=False, math_model="model-b"
    )


def test_clean_equation_model_invalidates_while_the_lane_is_on() -> None:
    """#230: the model that writes the LaTeX sidecars changes the saved bytes."""
    assert _fingerprint(
        recover_clean_equations=True, clean_equation_model="model-a"
    ) != _fingerprint(recover_clean_equations=True, clean_equation_model="model-b")


def test_clean_equation_model_is_ignored_while_the_lane_is_off() -> None:
    """Converse: no sidecars are produced, so the model identity is irrelevant."""
    assert _fingerprint(
        recover_clean_equations=False, clean_equation_model="model-a"
    ) == _fingerprint(recover_clean_equations=False, clean_equation_model="model-b")


def test_figures_engine_model_invalidates_outside_enabled_engines() -> None:
    """#232: the caption engine need not appear in ``enabled_engines``.

    Only the engine NAME was fingerprinted, so under a custom ``enabled_engines``
    that excludes GEMINI a ``gemini_model`` swap moved nothing and stale captions
    survived resume.
    """
    common = {
        "describe_figures": True,
        "figures_engine": EngineType.GEMINI,
        "enabled_engines": [EngineType.QWEN],
        # GEMINI must reach the fingerprint ONLY as the caption engine: it is the
        # default ``fallback_chain`` member, which would otherwise cover the model
        # through ``fallback_determinants`` and make this test pass vacuously.
        "fallback_chain": [],
    }
    assert _fingerprint(**common, gemini_model="gemini-a") != _fingerprint(
        **common, gemini_model="gemini-b"
    )


def test_figures_engine_model_is_ignored_without_descriptions() -> None:
    """Converse: no captions are produced, so the caption model must not invalidate."""
    common = {
        "describe_figures": False,
        "figures_engine": EngineType.GEMINI,
        "enabled_engines": [EngineType.QWEN],
        "fallback_chain": [],
    }
    assert _fingerprint(**common, gemini_model="gemini-a") == _fingerprint(
        **common, gemini_model="gemini-b"
    )
