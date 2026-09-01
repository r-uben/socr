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


class TestTheInvocationAndTheRecordTellOneStory:
    """GH-384. GH-370 rewrote ``auto`` + VLLM_BASE_URL to vllm at the RECORDING
    site only. ``resolve_qwen_intent`` still returned ``("auto", OLLAMA_MODEL)``
    and ``_build_command`` still sent ``--backend auto --model
    qwen3-vl:30b-a3b-instruct`` beside a sidecar saying vllm -- the drift GH-370
    existed to remove, inverted: the manifest naming a backend the invocation
    never asked for.

    The rewrite now lives in ``resolve_qwen_intent``, which ``_build_command``
    already reads, so agreement is structural rather than maintained by hand.
    """

    def _command(self, config) -> list[str]:
        from pathlib import Path

        from socr.engines.qwen import QwenEngine

        return QwenEngine()._build_command(Path("in.pdf"), Path("out"), config)

    def _flag(self, cmd: list[str], flag: str) -> str:
        return cmd[cmd.index(flag) + 1] if flag in cmd else ""

    def test_command_and_record_agree_under_auto_plus_env(self, monkeypatch) -> None:
        from socr.engines.qwen import resolve_qwen_intent

        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        cfg = _cfg(qwen_backend="auto", qwen_vllm_model=_HF_MODEL)

        intent_backend, intent_model = resolve_qwen_intent(cfg)
        record_backend, record_model = resolved_provenance(PROFILE_QWEN_LOCAL, cfg)
        cmd = self._command(cfg)

        assert (intent_backend, intent_model) == (record_backend, record_model)
        assert self._flag(cmd, "--backend") == record_backend == "vllm"
        assert self._flag(cmd, "--model") == record_model == _HF_MODEL
        assert self._flag(cmd, "--backend") != "auto", (
            "the invocation must not say auto while the sidecar says vllm"
        )

    def test_no_env_leaves_command_and_record_on_the_local_path(self, monkeypatch) -> None:
        """Control: without the environment variable nothing is rewritten, and
        the two still agree."""
        from socr.engines.qwen import resolve_qwen_intent

        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        cfg = _cfg(qwen_backend="auto")

        assert resolve_qwen_intent(cfg) == resolved_provenance(PROFILE_QWEN_LOCAL, cfg)
        assert self._flag(self._command(cfg), "--backend") == "auto"

    def test_explicit_ollama_still_outranks_the_environment(self, monkeypatch) -> None:
        from socr.engines.qwen import resolve_qwen_intent

        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        cfg = _cfg(qwen_backend="ollama")

        assert resolve_qwen_intent(cfg) == resolved_provenance(PROFILE_QWEN_LOCAL, cfg)
        assert self._flag(self._command(cfg), "--backend") == "ollama"


class TestProcessDocumentGatesOnTheResolvedBackend:
    """GH-391. ``process_document`` pre-checked the Ollama model whenever the RAW
    config said ``auto``. On an HPC host -- no Ollama, by policy -- an
    ``auto`` + VLLM_BASE_URL run was refused as MODEL_UNAVAILABLE, even though
    ``is_available`` already allowed it and ``resolve_qwen_intent`` already
    rewrote it. Third copy of the rule, disagreeing with the other two.

    Hermetic: the Ollama probe is stubbed to report the model missing, which is
    exactly the HPC condition, so nothing here needs a daemon.
    """

    def _engine_and_result(self, monkeypatch, backend: str, tmp_path):
        from socr.core.result import DocumentStatus
        from socr.engines import qwen as qwen_mod

        monkeypatch.setattr(qwen_mod, "_check_ollama_model", lambda _m: "ollama model not found")
        # Stop the real run: we only care whether the gate refused first.
        sentinel = object()
        monkeypatch.setattr(qwen_mod.BaseEngine, "process_document", lambda *a, **k: sentinel)
        engine = qwen_mod.QwenEngine()
        cfg = _cfg(qwen_backend=backend, qwen_vllm_model=_HF_MODEL)
        out = engine.process_document(tmp_path / "in.pdf", tmp_path / "out", cfg)
        refused = out is not sentinel and getattr(out, "status", None) == DocumentStatus.ERROR
        return refused

    def test_auto_plus_vllm_is_not_refused_when_ollama_is_absent(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        assert not self._engine_and_result(monkeypatch, "auto", tmp_path), (
            "auto + VLLM_BASE_URL is the HPC deployment; a missing Ollama model "
            "must not refuse the run"
        )

    def test_explicit_ollama_is_still_refused_when_the_model_is_missing(
        self, monkeypatch, tmp_path
    ) -> None:
        """Control: the gate must still protect the path it was written for."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        assert self._engine_and_result(monkeypatch, "ollama", tmp_path)

    def test_auto_without_the_env_var_is_still_refused(self, monkeypatch, tmp_path) -> None:
        """Control: plain local auto still needs the local model."""
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        assert self._engine_and_result(monkeypatch, "auto", tmp_path)


class TestTheOrchestratorWritersAreWhatRecordProvenance:
    """GH-385. This pins the agentic B3 writer -- the site that stamps the
    resolved pair onto ``PageOutput`` for a normally-routed page, which is what
    the manifest and sidecar read. Every other test in this file calls
    ``resolved_provenance`` directly, so reverting that call site to
    ``att.backend`` leaves the suite green while a ``--qwen-backend vllm`` run
    records ``ollama`` again.

    SCOPE, deliberately stated: the OTHER writer -- ``_escalate_table_page``
    (~2327) -- is NOT pinned here. ``_fake_route`` returns an accepted decision
    on a plain-text page, so the escalation lane never runs and reverting that
    site would still pass. Reaching it needs a non-local provider configured and
    an accepted escalation candidate. Filed separately rather than claimed
    falsely (cubic P2 on #396).

    Same shape as an unpinned assemble writer (#381): a green helper suite is
    not a gate on the value that ships.

    Hermetic: the provider ladder is patched, ``route_page`` is faked, and the
    judge model is stubbed empty, so nothing here needs ollama or a vLLM server.
    """

    def _run(self, tmp_path, monkeypatch, *, backend: str):
        from contextlib import ExitStack
        from unittest.mock import patch

        import fitz

        from socr.core.config import EngineType, PipelineConfig
        from socr.core.providers import PROFILE_QWEN_LOCAL
        from socr.core.result import PageOutput, PageStatus
        from socr.pipeline.orchestrator import UnifiedPipeline

        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        out_dir = tmp_path / f"out_{backend}"
        pdf_dir = tmp_path / f"src_{backend}"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / "doc.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 100), "A paragraph of ordinary prose on the page.", fontsize=11)
        doc.save(pdf_path)
        doc.close()

        pipeline = UnifiedPipeline(
            PipelineConfig(
                primary_engine=EngineType.QWEN,
                agentic=True,
                judge_backend="heuristic",
                enabled_engines=[EngineType.QWEN],
                quiet=True,
                save_figures=False,
                write_manifest=False,
                qwen_backend=backend,
                qwen_vllm_model=_HF_MODEL,
            )
        )

        def _fake_route(page_num, ladder, run_provider, judge, **kwargs):
            from socr.pipeline.agentic import PageDecision, ProviderAttempt

            out = PageOutput(
                page_num=page_num,
                text="A paragraph of ordinary prose on the page.",
                status=PageStatus.SUCCESS,
                engine="qwen",
                audit_passed=True,
            )
            prof = ladder[0]
            att = ProviderAttempt(
                engine=prof.engine,
                output=out,
                cost_usd=prof.cost_per_page_usd,
                accepted=True,
                reason="ok",
                provider_id=prof.id,
                model=prof.model,
                backend=prof.backend,
            )
            return PageDecision(page_num=page_num, final_output=out, attempts=[att], accepted=True)

        with ExitStack() as stack:
            stack.enter_context(
                patch("socr.pipeline.orchestrator.route_page", side_effect=_fake_route)
            )
            stack.enter_context(
                patch.object(
                    pipeline,
                    "_available_engines_for_agentic",
                    return_value=[PROFILE_QWEN_LOCAL],
                )
            )
            stack.enter_context(patch.object(pipeline, "_resolve_judge_model", return_value=""))
            pipeline.process(pdf_path, out_dir)

        found = list(out_dir.rglob("00001.json")) or list(out_dir.rglob("001.json"))
        assert found, f"no page sidecar under {out_dir}"
        import json

        return json.loads(found[0].read_text(encoding="utf-8")).get("winning_output", {})

    def test_the_recorded_backend_follows_the_config_not_the_registry_label(
        self, tmp_path, monkeypatch
    ) -> None:
        """The reported bug, at the value that actually ships. The fake attempt
        carries the REGISTRY label in ``att.backend`` either way, so anything
        the sidecar shows beyond that came from the orchestrator writer."""
        from socr.core.providers import PROFILE_QWEN_LOCAL

        ollama_run = self._run(tmp_path, monkeypatch, backend="ollama")
        vllm_run = self._run(tmp_path, monkeypatch, backend="vllm")

        assert ollama_run.get("provider_backend") != vllm_run.get("provider_backend"), (
            "a vLLM run and an Ollama run must not be indistinguishable in the "
            "sidecar -- that is the defect GH-370 fixed and this pins"
        )
        assert vllm_run.get("provider_backend") == "vllm"
        assert vllm_run.get("provider_backend") != PROFILE_QWEN_LOCAL.backend
        assert vllm_run.get("provider_model") == _HF_MODEL

    def test_the_ollama_run_still_records_the_registry_values(self, tmp_path, monkeypatch) -> None:
        """Control: the local path is unchanged, so the pin cannot be satisfied
        by a writer that simply stamps something different every time."""
        from socr.core.providers import PROFILE_QWEN_LOCAL

        ollama_run = self._run(tmp_path, monkeypatch, backend="ollama")
        assert ollama_run.get("provider_backend") == PROFILE_QWEN_LOCAL.backend
        assert ollama_run.get("provider_model") == PROFILE_QWEN_LOCAL.model
