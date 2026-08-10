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
from socr.core.providers import (
    PROFILE_GEMINI,
    PROFILE_GLM,
    PROFILE_QWEN_CLOUD,
    TIER_LOCAL,
    ProviderProfile,
)
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


class TestStrictLocalFiltersCloudRungs:
    """strict_local=True must remove ALL cloud-tier providers from the agentic ladder,
    including free-cloud providers like PROFILE_QWEN_CLOUD (tier=cloud, cost=0).
    The filter is by tier, not by cost."""

    def _make_pipeline(self, strict_local: bool) -> UnifiedPipeline:
        config = PipelineConfig(
            agentic=True,
            strict_local=strict_local,
            judge_backend="heuristic",
            enabled_engines=[EngineType.GLM, EngineType.QWEN, EngineType.GEMINI],
            quiet=True,
        )
        return UnifiedPipeline(config)

    def _run_phase_agentic_collecting_ladder(
        self, pipeline: UnifiedPipeline, profiles: list[ProviderProfile]
    ) -> list[list]:
        """Run _phase_agentic with a mocked available set; collect what reaches provider_ladder."""
        collected_ladders: list[list] = []

        def mock_provider_ladder(available, **kwargs):
            collected_ladders.append(list(available))
            return []  # short-circuit; we only care about what was passed

        with (
            patch.object(pipeline, "_available_engines_for_agentic", return_value=profiles),
            # provider_ladder is a local import inside _phase_agentic; patch at source.
            patch("socr.core.providers.provider_ladder", side_effect=mock_provider_ladder),
        ):
            from socr.core.document import DocumentHandle
            from socr.core.state import DocumentState

            with patch.object(DocumentHandle, "__post_init__", lambda self: None):
                handle = DocumentHandle(path=Path("/tmp/fake.pdf"), page_count=1)
            state = DocumentState(handle=handle)
            # Mark page as scanned (not born-digital) so it goes to OCR.
            state.pages[1].is_born_digital = False
            pipeline._phase_agentic(state, Path("/tmp/out"))

        return collected_ladders

    def test_strict_local_drops_cloud_rungs_by_tier(self) -> None:
        """_phase_agentic with strict_local=True must exclude ALL cloud-tier profiles.

        This includes both paid cloud (Gemini) and FREE cloud (QWEN Cloud / Ollama Cloud):
        the predicate is tier==TIER_LOCAL, not is_free, so a zero-cost cloud provider
        such as PROFILE_QWEN_CLOUD is still excluded.
        """
        pipeline = self._make_pipeline(strict_local=True)

        # Three profiles: local-free (GLM), cloud-free (QWEN Cloud), cloud-paid (Gemini).
        mixed_profiles = [PROFILE_GLM, PROFILE_QWEN_CLOUD, PROFILE_GEMINI]
        collected = self._run_phase_agentic_collecting_ladder(pipeline, mixed_profiles)

        assert collected, "provider_ladder was never called"
        passed = collected[0]

        # Only local-tier profiles must survive.
        assert all(p.tier == TIER_LOCAL for p in passed), (
            f"strict_local=True should only pass TIER_LOCAL profiles; got: {passed}"
        )
        # GLM (local) is kept.
        assert PROFILE_GLM in passed
        # QWEN Cloud (cloud-tier, cost=0) is excluded — this is the key regression test.
        assert PROFILE_QWEN_CLOUD not in passed, (
            "PROFILE_QWEN_CLOUD is free but cloud-tier and must be excluded by --strict-local"
        )
        # Gemini (cloud-tier, paid) is also excluded.
        assert PROFILE_GEMINI not in passed

    def test_no_strict_local_keeps_all_rungs(self) -> None:
        """Without strict_local, all profiles (local and cloud) reach provider_ladder."""
        pipeline = self._make_pipeline(strict_local=False)

        mixed_profiles = [PROFILE_GLM, PROFILE_QWEN_CLOUD, PROFILE_GEMINI]
        collected = self._run_phase_agentic_collecting_ladder(pipeline, mixed_profiles)

        assert collected
        passed = collected[0]
        # All three profiles should reach provider_ladder unchanged.
        assert PROFILE_GLM in passed
        assert PROFILE_QWEN_CLOUD in passed
        assert PROFILE_GEMINI in passed


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


# ---------------------------------------------------------------------------
# 4. GH-46-E2: the Ollama-Cloud rung must be reachable
# ---------------------------------------------------------------------------


class TestCloudRungReachable:
    """The declared local -> Ollama-Cloud -> Gemini ladder must have its middle rung.

    These tests call the REAL ``_available_engines_for_agentic``. The class above
    hand-builds its profile list, which is exactly how the missing rung went
    unnoticed: every existing assertion was made against a list no production
    code path could produce.

    CI hermeticity: CI has no ollama and no ``qwen-ocr`` CLI, so BOTH probes must
    be patched by name or these pass locally and fail there. ``get_engine`` is
    patched in the orchestrator namespace (module-level import) and the cloud
    probe at its definition site (function-level import, resolved per call).
    """

    @staticmethod
    def _pipeline() -> UnifiedPipeline:
        return UnifiedPipeline(
            PipelineConfig(agentic=True, enabled_engines=[EngineType.QWEN], quiet=True)
        )

    @staticmethod
    def _engine(available: bool):
        from unittest.mock import MagicMock

        engine = MagicMock()
        engine.is_available.return_value = available
        return engine

    def test_both_qwen_rungs_appear_as_distinct_profiles(self) -> None:
        """The regression this ticket exists for: cloud was unreachable by construction."""
        pipeline = self._pipeline()
        with (
            patch("socr.pipeline.orchestrator.get_engine", return_value=self._engine(True)),
            patch("socr.engines.qwen.cloud_model_available", return_value=True),
        ):
            profiles = pipeline._available_engines_for_agentic()

        ids = [p.id for p in profiles]
        assert "qwen-local-instruct" in ids
        assert "qwen-cloud" in ids
        # Distinct rungs, not one profile counted twice.
        assert len(set(ids)) == len(ids)
        assert len({id(p) for p in profiles}) == 2

    def test_cloud_rung_alone_when_local_model_absent(self) -> None:
        """A machine that can reach cloud but has no local pull still gets a rung."""
        pipeline = self._pipeline()
        with (
            patch("socr.pipeline.orchestrator.get_engine", return_value=self._engine(False)),
            patch("socr.engines.qwen.cloud_model_available", return_value=True),
        ):
            ids = [p.id for p in pipeline._available_engines_for_agentic()]

        assert ids == ["qwen-cloud"]

    def test_local_rung_alone_when_cloud_unreachable(self) -> None:
        """The pre-GH-46-E2 world stays intact when the cloud model is not pulled."""
        pipeline = self._pipeline()
        with (
            patch("socr.pipeline.orchestrator.get_engine", return_value=self._engine(True)),
            patch("socr.engines.qwen.cloud_model_available", return_value=False),
        ):
            ids = [p.id for p in pipeline._available_engines_for_agentic()]

        assert ids == ["qwen-local-instruct"]

    def test_local_probe_crash_does_not_suppress_cloud_rung(self) -> None:
        """The two probes are independent; neither may gate the other."""
        pipeline = self._pipeline()
        engine = self._engine(True)
        engine.is_available.side_effect = RuntimeError("ollama socket exploded")
        with (
            patch("socr.pipeline.orchestrator.get_engine", return_value=engine),
            patch("socr.engines.qwen.cloud_model_available", return_value=True),
        ):
            ids = [p.id for p in pipeline._available_engines_for_agentic()]

        assert ids == ["qwen-cloud"]

    def test_cloud_probe_crash_never_crashes_routing(self) -> None:
        pipeline = self._pipeline()
        with (
            patch("socr.pipeline.orchestrator.get_engine", return_value=self._engine(True)),
            patch(
                "socr.engines.qwen.cloud_model_available",
                side_effect=RuntimeError("network on fire"),
            ),
        ):
            ids = [p.id for p in pipeline._available_engines_for_agentic()]

        assert ids == ["qwen-local-instruct"]

    def test_escalation_lane_prefers_cloud_rung_over_gemini(self) -> None:
        """Real behaviour change to the GH-96 lane: $0.00 beats $0.0002.

        ``_resolve_table_escalation_provider`` takes the cheapest non-local
        per-page provider. Before this ticket the cloud rung could never be in
        ``available``, so Gemini won by default.
        """
        pipeline = self._pipeline()
        chosen = pipeline._resolve_table_escalation_provider([PROFILE_QWEN_CLOUD, PROFILE_GEMINI])

        assert chosen is not None
        assert chosen.id == "qwen-cloud"
        assert chosen.cost_per_page_usd < PROFILE_GEMINI.cost_per_page_usd

    def test_escalation_lane_still_falls_back_to_gemini_without_cloud(self) -> None:
        pipeline = self._pipeline()
        chosen = pipeline._resolve_table_escalation_provider([PROFILE_GEMINI])

        assert chosen is not None
        assert chosen.id == "gemini"
