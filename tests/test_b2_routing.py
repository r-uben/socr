"""Tests for TICKET-B2: agentic-as-default + --legacy-routing + --strict-local.

Coverage:
- PipelineConfig().agentic is True (new default)
- --legacy-routing flag disables agentic (config.agentic = False)
- --strict-local filters cloud rungs from the agentic ladder
- strict_local is included in the run fingerprint
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from socr.core.config import EngineType, PipelineConfig
from socr.core.providers import PROFILE_GEMINI, PROFILE_GLM, ProviderProfile
from socr.pipeline.orchestrator import UnifiedPipeline

# ---------------------------------------------------------------------------
# 1. Default config tests
# ---------------------------------------------------------------------------


def test_agentic_is_default() -> None:
    """PipelineConfig() should have agentic=True out of the box."""
    assert PipelineConfig().agentic is True


def test_strict_local_default_false() -> None:
    """PipelineConfig() should have strict_local=False out of the box."""
    assert PipelineConfig().strict_local is False


# ---------------------------------------------------------------------------
# 2. CLI flag wiring — test via build_config()
# ---------------------------------------------------------------------------


def test_legacy_routing_disables_agentic() -> None:
    """--legacy-routing should set config.agentic = False."""
    from socr.cli import build_config

    config = build_config(legacy_routing=True)
    assert config.agentic is False


def test_legacy_routing_overrides_explicit_agentic() -> None:
    """--legacy-routing wins even when --agentic is also passed."""
    from socr.cli import build_config

    config = build_config(agentic=True, legacy_routing=True)
    assert config.agentic is False


def test_agentic_flag_keeps_agentic_true() -> None:
    """--agentic (backward-compat flag) still works; config.agentic stays True."""
    from socr.cli import build_config

    config = build_config(agentic=True)
    assert config.agentic is True


def test_strict_local_flag_sets_strict_local() -> None:
    """--strict-local should set config.strict_local = True."""
    from socr.cli import build_config

    config = build_config(strict_local=True)
    assert config.strict_local is True


def test_strict_local_default_in_build_config() -> None:
    """Without --strict-local, strict_local stays False."""
    from socr.cli import build_config

    config = build_config()
    assert config.strict_local is False


# ---------------------------------------------------------------------------
# 3. --strict-local filters cloud rungs in _phase_agentic
# ---------------------------------------------------------------------------


def _make_provider_profile(engine: EngineType, cost: float) -> ProviderProfile:
    return ProviderProfile(
        engine=engine,
        tier="local" if cost == 0.0 else "cloud",
        cost_per_page_usd=cost,
        id=engine.value,
        backend="test",
        model="test",
    )


class TestStrictLocalFiltersCloudRungs:
    """strict_local=True must remove paid providers from the agentic ladder."""

    def _make_pipeline(self, strict_local: bool) -> UnifiedPipeline:
        config = PipelineConfig(
            agentic=True,
            strict_local=strict_local,
            judge_backend="heuristic",
            enabled_engines=[EngineType.GLM, EngineType.GEMINI],
            quiet=True,
        )
        return UnifiedPipeline(config)

    def test_strict_local_drops_paid_from_available(self) -> None:
        """_phase_agentic should filter to is_free profiles when strict_local=True."""
        pipeline = self._make_pipeline(strict_local=True)

        # Mock _available_engines_for_agentic to return one free + one paid profile.
        mixed_profiles = [PROFILE_GLM, PROFILE_GEMINI]  # GLM free, Gemini paid

        collected_ladders: list[list] = []

        def mock_provider_ladder(available, **kwargs):
            collected_ladders.append(list(available))
            return []  # short-circuit; we only care about what was passed

        with (
            patch.object(pipeline, "_available_engines_for_agentic", return_value=mixed_profiles),
            # provider_ladder is a local import inside _phase_agentic; patch at source.
            patch(
                "socr.core.providers.provider_ladder",
                side_effect=mock_provider_ladder,
            ),
        ):
            # We need a state object to call _phase_agentic; build a minimal one.
            from socr.core.document import DocumentHandle
            from socr.core.state import DocumentState

            with patch.object(DocumentHandle, "__post_init__", lambda self: None):
                handle = DocumentHandle(path=Path("/tmp/fake.pdf"), page_count=1)
            state = DocumentState(handle=handle)
            # Mark the single page as scanned (not born-digital) so it goes to OCR.
            state.pages[1].is_born_digital = False

            pipeline._phase_agentic(state, Path("/tmp/out"))

        # provider_ladder should have been called with only the free profile
        assert collected_ladders, "provider_ladder was never called"
        passed = collected_ladders[0]
        assert all(p.is_free for p in passed), (
            f"strict_local=True should only pass free profiles; got: {passed}"
        )
        # Gemini (paid) must NOT be in the list
        assert PROFILE_GEMINI not in passed
        # GLM (free) must be in the list
        assert PROFILE_GLM in passed

    def test_no_strict_local_keeps_paid_rungs(self) -> None:
        """Without strict_local, paid profiles are passed to provider_ladder unchanged."""
        pipeline = self._make_pipeline(strict_local=False)

        mixed_profiles = [PROFILE_GLM, PROFILE_GEMINI]
        collected_ladders: list[list] = []

        def mock_provider_ladder(available, **kwargs):
            collected_ladders.append(list(available))
            return []

        with (
            patch.object(pipeline, "_available_engines_for_agentic", return_value=mixed_profiles),
            patch(
                "socr.core.providers.provider_ladder",
                side_effect=mock_provider_ladder,
            ),
        ):
            from socr.core.document import DocumentHandle
            from socr.core.state import DocumentState

            with patch.object(DocumentHandle, "__post_init__", lambda self: None):
                handle = DocumentHandle(path=Path("/tmp/fake.pdf"), page_count=1)
            state = DocumentState(handle=handle)
            state.pages[1].is_born_digital = False

            pipeline._phase_agentic(state, Path("/tmp/out"))

        assert collected_ladders
        passed = collected_ladders[0]
        # Both profiles should be present when strict_local is False
        assert PROFILE_GEMINI in passed
        assert PROFILE_GLM in passed


# ---------------------------------------------------------------------------
# 4. Run fingerprint includes strict_local
# ---------------------------------------------------------------------------


class TestRunFingerprintIncludesStrictLocal:
    """Changing strict_local must produce a different fingerprint."""

    def _pipeline(self, strict_local: bool) -> UnifiedPipeline:
        cfg = PipelineConfig(
            agentic=True,
            strict_local=strict_local,
            primary_engine=EngineType.GEMINI,
            quiet=True,
        )
        return UnifiedPipeline(cfg)

    def test_strict_local_changes_fingerprint(self) -> None:
        """Two configs differing only in strict_local must have different fingerprints."""
        fp_off = self._pipeline(strict_local=False)._run_fingerprint(EngineType.GEMINI)
        fp_on = self._pipeline(strict_local=True)._run_fingerprint(EngineType.GEMINI)
        assert fp_off != fp_on, "strict_local flag must invalidate the resume cache"


# ---------------------------------------------------------------------------
# 5. Click CLI flags are registered and parseable
# ---------------------------------------------------------------------------


class TestCLIFlags:
    """Smoke test that the new flags are wired into the click commands."""

    def test_legacy_routing_flag_recognized(self) -> None:
        """socr process --legacy-routing should not error on flag parsing."""
        from click.testing import CliRunner

        from socr.cli import process

        runner = CliRunner()
        # Invoke with --help to see if the flag is listed; no real file needed.
        result = runner.invoke(process, ["--help"])
        assert result.exit_code == 0
        assert "--legacy-routing" in result.output

    def test_strict_local_flag_recognized(self) -> None:
        """socr process --strict-local should appear in --help."""
        from click.testing import CliRunner

        from socr.cli import process

        runner = CliRunner()
        result = runner.invoke(process, ["--help"])
        assert result.exit_code == 0
        assert "--strict-local" in result.output
