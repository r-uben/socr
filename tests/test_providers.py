"""Provider cost registry tests: the cost-ordered escalation ladder."""

from __future__ import annotations

from socr.core.config import EngineType
from socr.core.providers import (
    DEFAULT_PROVIDERS,
    ProviderProfile,
    cost_of,
    provider_ladder,
)


def test_ladder_is_cheapest_first():
    ladder = provider_ladder({EngineType.GEMINI, EngineType.MISTRAL, EngineType.GLM})
    engines = [p.engine for p in ladder]
    # GLM (free) before GEMINI ($0.0002) before MISTRAL ($0.001)
    assert engines == [EngineType.GLM, EngineType.GEMINI, EngineType.MISTRAL]
    assert [p.cost_per_page_usd for p in ladder] == [0.0, 0.0002, 0.001]


def test_free_engines_tie_broken_by_priority():
    # All local/free -> order falls back to ENGINE_PRIORITY (GLM<DeepSeek<Marker)
    ladder = provider_ladder({EngineType.MARKER, EngineType.DEEPSEEK, EngineType.GLM})
    assert [p.engine for p in ladder] == [
        EngineType.GLM,
        EngineType.DEEPSEEK,
        EngineType.MARKER,
    ]


def test_availability_filters_ladder():
    ladder = provider_ladder({EngineType.GEMINI})
    assert [p.engine for p in ladder] == [EngineType.GEMINI]


def test_none_available_means_all_known():
    ladder = provider_ladder(None)
    assert len(ladder) == len(DEFAULT_PROVIDERS)
    # still cheapest-first overall
    assert ladder[0].cost_per_page_usd <= ladder[-1].cost_per_page_usd


def test_max_cost_per_page_cap():
    # Cap below Mistral's price drops it.
    ladder = provider_ladder(
        {EngineType.GLM, EngineType.GEMINI, EngineType.MISTRAL},
        max_cost_per_page=0.0005,
    )
    assert EngineType.MISTRAL not in [p.engine for p in ladder]
    assert EngineType.GEMINI in [p.engine for p in ladder]


def test_per_page_only_filter():
    reg = {
        EngineType.GLM: ProviderProfile(EngineType.GLM, "local", 0.0, supports_per_page=True),
        EngineType.NOUGAT: ProviderProfile(
            EngineType.NOUGAT, "local", 0.0, supports_per_page=False
        ),
    }
    ladder = provider_ladder(None, registry=reg, per_page_only=True)
    assert [p.engine for p in ladder] == [EngineType.GLM]


def test_cost_of():
    assert cost_of(EngineType.GLM, 10) == 0.0
    assert cost_of(EngineType.GEMINI, 5) == 0.0002 * 5
    assert cost_of(EngineType.MISTRAL, 3) == 0.001 * 3
    assert cost_of(EngineType.AUTO, 10) == 0.0  # unknown -> 0
