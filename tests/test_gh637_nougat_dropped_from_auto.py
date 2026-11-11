"""GH-637: nougat must not be auto-selected when other engines are unavailable.

D1 (#628/#632) set ``PROFILE_NOUGAT.auto_eligible=False`` so the agentic ladder
skips it, but ``AUTO_ENGINE_ORDER`` / ``_LOCAL_ENGINE_ORDER`` (the non-agentic
auto-select paths) still listed ``EngineType.NOUGAT`` and would pick it when
every earlier engine in the order is unavailable. This falsifies main's
pre-fix behaviour: with QWEN/GEMINI/MARKER/GLM (and DeepSeek/Mistral) mocked
unavailable, ``resolve_auto_engine`` / ``resolve_local_engine`` used to fall
through to nougat instead of the documented "no engine" outcome.
"""

from __future__ import annotations

from unittest.mock import patch

from socr.core.config import AUTO_ENGINE_ORDER, EngineType
from socr.engines.registry import _LOCAL_ENGINE_ORDER, resolve_auto_engine, resolve_local_engine


def _unavailable_engine():
    class _Unavailable:
        def is_available(self):
            return False

    return _Unavailable


def test_nougat_absent_from_auto_engine_order():
    assert EngineType.NOUGAT not in AUTO_ENGINE_ORDER


def test_nougat_absent_from_local_engine_order():
    assert EngineType.NOUGAT not in _LOCAL_ENGINE_ORDER


def test_resolve_auto_engine_never_returns_nougat_when_others_unavailable():
    # Every real engine mocked unavailable; nougat is deliberately NOT mocked
    # available either -- if it were still in AUTO_ENGINE_ORDER it would be
    # tried and, since NougatEngine().is_available() is not patched here, the
    # fallback would be governed by nougat rather than the documented
    # GEMINI-fallback contract.
    # CH-12: derive the set from AUTO_ENGINE_ORDER so a newly added engine cannot
    # leak through (GEMMA's is_available is CLI presence: qwen-ocr on PATH made it
    # win locally, not on CI). GEMINI is mocked unavailable too, as before: it is
    # returned as the documented no-engine fallback, not because it probed up.
    # NOUGAT is left unmocked on purpose: if it ever re-enters the order, it must be
    # free to win so this test catches it (CH-12 mutant M8).
    unavailable = {e: _unavailable_engine() for e in AUTO_ENGINE_ORDER if e != EngineType.NOUGAT}
    with patch.dict("socr.engines.registry._ENGINES", unavailable):
        result = resolve_auto_engine()
    assert result != EngineType.NOUGAT
    assert result == EngineType.GEMINI  # documented fallback


def test_resolve_local_engine_never_returns_nougat_when_others_unavailable():
    with patch.dict(
        "socr.engines.registry._ENGINES",
        {
            EngineType.QWEN: _unavailable_engine(),
            EngineType.GLM: _unavailable_engine(),
            EngineType.MARKER: _unavailable_engine(),
        },
    ):
        result = resolve_local_engine()
    assert result != EngineType.NOUGAT
    assert result is None  # documented "no local engine" outcome
