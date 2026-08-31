"""GH-370: the manifest must name the backend that actually ran.

``ProviderProfile.backend``/``.model`` describe what a rung IS in the registry.
For QWEN that is not what executes: ``PROFILE_QWEN_LOCAL`` declares
``backend="ollama"`` and the Ollama tag, while ``execution_overrides``
deliberately returns ``{}`` for it so the operator's ``--qwen-backend`` wins.
A vLLM/HPC run therefore executed on vLLM and recorded ``ollama`` -- on hosts
where Ollama is not installed at all.

Every assertion below is a DIFFERENCE (#253/#257): the same profile, resolved
under two configs that differ in exactly one field. Nothing pins an absolute
outcome measured on one machine, and none of it needs a provider to be
reachable.
"""

from __future__ import annotations

from socr.core.config import PipelineConfig
from socr.core.providers import (
    PROFILE_GEMINI,
    PROFILE_QWEN_CLOUD,
    PROFILE_QWEN_LOCAL,
    execution_overrides,
    profile_by_id,
    resolved_provenance,
)

_HF_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


def _cfg(**overrides) -> PipelineConfig:
    return PipelineConfig(**overrides)


class TestQwenBackendIsRecordedAsExecuted:
    def test_backend_choice_is_the_only_difference_and_it_reaches_provenance(self) -> None:
        """The reported bug: identical rung, identical everything but
        ``qwen_backend``, and the recorded backend must follow the operator's
        choice rather than the registry's label."""
        ollama_backend, ollama_model = resolved_provenance(
            PROFILE_QWEN_LOCAL, _cfg(qwen_backend="ollama")
        )
        vllm_backend, vllm_model = resolved_provenance(
            PROFILE_QWEN_LOCAL, _cfg(qwen_backend="vllm", qwen_vllm_model=_HF_MODEL)
        )

        assert (ollama_backend, ollama_model) != (vllm_backend, vllm_model), (
            "a vLLM run and an Ollama run of the same rung must not be "
            "indistinguishable in the record -- that is the whole defect"
        )
        assert vllm_backend != PROFILE_QWEN_LOCAL.backend
        assert vllm_backend == "vllm"
        assert ollama_backend == "ollama"

    def test_vllm_run_records_the_served_model_not_the_ollama_tag(self) -> None:
        """The model string was wrong too: the Ollama tag was recorded for a
        server that was serving the HF id."""
        _backend, model = resolved_provenance(
            PROFILE_QWEN_LOCAL, _cfg(qwen_backend="vllm", qwen_vllm_model=_HF_MODEL)
        )
        assert model == _HF_MODEL
        assert model != PROFILE_QWEN_LOCAL.model

    def test_ollama_run_is_unchanged_by_the_fix(self) -> None:
        """Control: the Mac/Ollama path must still report exactly what it
        always did, or the fix would have traded one wrong label for another."""
        assert resolved_provenance(PROFILE_QWEN_LOCAL, _cfg(qwen_backend="ollama")) == (
            PROFILE_QWEN_LOCAL.backend,
            PROFILE_QWEN_LOCAL.model,
        )


class TestNonQwenProfilesAreUntouched:
    def test_qwen_backend_does_not_leak_into_an_unrelated_rung(self) -> None:
        """A non-QWEN rung's ``EngineType`` maps 1:1 to a deployment, so there
        is nothing to resolve -- and ``--qwen-backend`` must not perturb it."""
        under_ollama = resolved_provenance(PROFILE_GEMINI, _cfg(qwen_backend="ollama"))
        under_vllm = resolved_provenance(
            PROFILE_GEMINI, _cfg(qwen_backend="vllm", qwen_vllm_model=_HF_MODEL)
        )
        assert under_ollama == under_vllm == (PROFILE_GEMINI.backend, PROFILE_GEMINI.model)


class TestRecordingAgreesWithExecution:
    def test_cloud_qwen_provenance_matches_what_overrides_force(self) -> None:
        """``resolved_provenance`` is the recording counterpart of
        ``execution_overrides``. If they disagree for the one profile that
        pins its execution, the manifest is lying again by a different route.
        """
        overrides = execution_overrides(PROFILE_QWEN_CLOUD)
        backend, model = resolved_provenance(PROFILE_QWEN_CLOUD, _cfg())

        assert overrides["qwen_backend"] == backend
        assert overrides["qwen_model"] == model
        assert backend != PROFILE_QWEN_CLOUD.backend, (
            "the descriptive 'ollama-cloud' label is not a transport; Ollama "
            "Cloud is served by the local Ollama runtime"
        )


class TestProfileLookup:
    def test_every_named_profile_resolves_by_its_own_id(self) -> None:
        """The agentic recording site resolves an attempt back to its profile
        by ``provider_id``. A profile missing from the lookup would silently
        fall back to the registry label -- the bug, restored."""
        for prof in (PROFILE_QWEN_LOCAL, PROFILE_QWEN_CLOUD, PROFILE_GEMINI):
            assert profile_by_id(prof.id) is prof

    def test_unknown_id_is_none_not_a_wrong_profile(self) -> None:
        assert profile_by_id("no-such-provider") is None


class TestAutoBackendOnAnHpcHost:
    """cubic P1 on PR #382. ``auto`` is not a transport.

    ``PipelineConfig`` adopts ``VLLM_BASE_URL`` into ``qwen_vllm_url`` while
    leaving ``qwen_backend`` at its ``"auto"`` default, so exporting ONE
    environment variable IS the HPC deployment (see
    ``UnifiedPipeline._local_backend_is_openai_compatible``). Resolving ``auto``
    with the local backends would hand back the Ollama tag and reproduce the
    reported defect by a shorter route.
    """

    def test_env_var_is_the_only_difference_and_it_reaches_provenance(self, monkeypatch) -> None:
        cfg = _cfg(qwen_backend="auto", qwen_vllm_model=_HF_MODEL)

        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        without_env = resolved_provenance(PROFILE_QWEN_LOCAL, cfg)

        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        with_env = resolved_provenance(PROFILE_QWEN_LOCAL, cfg)

        assert without_env != with_env
        assert with_env == ("vllm", _HF_MODEL)
        assert with_env[1] != PROFILE_QWEN_LOCAL.model, (
            "an auto/HPC run must not record the Ollama tag -- that is the "
            "original bug reached without touching a flag"
        )

    def test_an_explicit_backend_outranks_the_environment(self, monkeypatch) -> None:
        """A value the user typed beats one the environment happens to carry --
        in both directions, matching the existing predicate's contract."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        assert resolved_provenance(PROFILE_QWEN_LOCAL, _cfg(qwen_backend="ollama")) == (
            "ollama",
            PROFILE_QWEN_LOCAL.model,
        )

    def test_auto_without_the_env_var_is_not_claimed_as_vllm(self, monkeypatch) -> None:
        """Control: no environment, no vLLM claim. Guards against the fix
        over-reaching into every auto run on a laptop."""
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        backend, _model = resolved_provenance(PROFILE_QWEN_LOCAL, _cfg(qwen_backend="auto"))
        assert backend != "vllm"


class TestOneCopyOfTheAutoRule:
    """cubic P2 on PR #382. Execution and provenance must ask the SAME question.

    ``_local_backend_is_openai_compatible`` decides which server execution talks
    to; ``resolved_provenance`` decides what the manifest says ran. Two copies of
    the auto+VLLM_BASE_URL rule is precisely the drift this ticket removes, so
    the orchestrator delegates its case 2 to the shared helper.
    """

    def _pipeline(self, **overrides):
        from socr.core.config import EngineType, PipelineConfig
        from socr.pipeline.orchestrator import UnifiedPipeline

        return UnifiedPipeline(
            PipelineConfig(
                primary_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                quiet=True,
                **overrides,
            )
        )

    def test_execution_and_provenance_agree_across_every_backend_state(self, monkeypatch) -> None:
        """Whenever execution says "this run is on an OpenAI-compatible server",
        provenance must record vllm -- and when it does not, it must not."""
        for backend, env, expect_openai in (
            ("auto", None, False),
            ("auto", "http://localhost:8000/v1", True),
            ("ollama", "http://localhost:8000/v1", False),
            ("vllm", None, True),
        ):
            if env is None:
                monkeypatch.delenv("VLLM_BASE_URL", raising=False)
            else:
                monkeypatch.setenv("VLLM_BASE_URL", env)

            pipeline = self._pipeline(qwen_backend=backend, qwen_vllm_model=_HF_MODEL)
            executes_openai = pipeline._local_backend_is_openai_compatible()
            recorded_backend, _model = resolved_provenance(PROFILE_QWEN_LOCAL, pipeline.config)

            assert executes_openai is expect_openai, (backend, env)
            assert (recorded_backend == "vllm") is executes_openai, (
                f"backend={backend!r} env={env!r}: execution says "
                f"openai_compatible={executes_openai} but provenance recorded "
                f"{recorded_backend!r} -- the two copies have drifted"
            )
