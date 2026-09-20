"""``PipelineConfig.from_file`` must not silently drop settings (#240).

The old implementation restored scalars from a hand-maintained list of names, so
14 of the dataclass's fields -- including both cost caps and ``agentic`` -- were
ignored when set in a YAML config file. These tests pin the invariant that made
that possible: the set of fields a config file can restore is the dataclass's own
field set, minus an explicitly named exception list.
"""

import dataclasses
from pathlib import Path

import pytest
import yaml

from socr.core.config import EngineType, HPCConfig, PipelineConfig
from socr.core.providers import zero_cap_pinned_forbids_cloud

# Fields that cannot be probed by the generic scalar round-trip below, each for a
# stated structural reason -- NOT because they are allowed to be unrestorable.
# Both are covered by dedicated tests further down.
UNPROBED_FIELDS = {
    "hpc",  # nested dataclass, not a scalar -- see test_hpc_block_round_trips
    "enabled_engines",  # list[EngineType] -- see test_engine_fields_round_trip
    "fallback_chain",  # list[EngineType] -- see test_engine_fields_round_trip
}

# The 14 fields the hand-maintained list forgot. Listed explicitly so this file
# fails loudly if the fix is reverted, independent of the generic sweep.
PREVIOUSLY_DROPPED = [
    "judge_hard_pages",
    "escalate_ambiguous_tables",
    "escalation_timeout_sec",
    "dual_pass_tables",
    "auto_patch_tables",
    "agentic",
    "strict_local",
    "judge_backend",
    "judge_model",
    "max_cost_per_page",
    "cost_budget",
    "write_manifest",
    "qwen_vllm_model",
    "qwen_vllm_url",
]


def _probe_value(name: str, default: object) -> object:
    """A YAML-representable value that differs from ``default``."""
    if isinstance(default, bool):
        return not default
    if isinstance(default, int):
        return default + 7
    if isinstance(default, float):
        return default + 1.5
    if default is None:
        # An optional scalar (``str | None``). ``None`` is a real default meaning
        # "unset / resolve it", so the field still has to round-trip: a config
        # file that names it must not be silently dropped, which is this whole
        # module's invariant. A string probe is representable in YAML and
        # differs from the default by construction.
        return f"probe-{name}"
    if isinstance(default, EngineType):
        return EngineType.NOUGAT.value if default != EngineType.NOUGAT else EngineType.GLM.value
    if isinstance(default, Path):
        return "/tmp/socr-probe-output"
    if isinstance(default, str):
        return f"probe-{name}"
    raise AssertionError(f"no probe value for field {name!r} of type {type(default).__name__}")


def _probeable_fields() -> list[dataclasses.Field]:
    return [f for f in dataclasses.fields(PipelineConfig) if f.name not in UNPROBED_FIELDS]


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "probe.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


class TestFromFileCoverage:
    def test_every_field_is_restorable(self, tmp_path):
        """A config file setting every field must change every field.

        This is the regression guard: a newly added PipelineConfig field fails
        here unless it round-trips or is added to UNPROBED_FIELDS deliberately.
        """
        defaults = PipelineConfig()
        payload = {
            f.name: _probe_value(f.name, getattr(defaults, f.name)) for f in _probeable_fields()
        }

        config = PipelineConfig.from_file(_write(tmp_path, payload))

        unrestored = []
        for name, written in payload.items():
            got = getattr(config, name)
            expected = Path(written) if isinstance(getattr(defaults, name), Path) else written
            if isinstance(getattr(defaults, name), EngineType):
                expected = EngineType(written)
            if got != expected:
                unrestored.append(f"{name}: file said {written!r}, config has {got!r}")
        assert not unrestored, "fields silently dropped by from_file:\n" + "\n".join(unrestored)

    def test_unprobed_exception_list_is_exhaustive(self):
        """Both directions: the exception list names real fields, and nothing escapes it.

        The subset check alone would let a newly added list-valued field slip past
        the generic round-trip guard unnoticed. A field is unprobable here iff its
        default is a container the generic probe cannot synthesise a value for --
        a list, or the nested HPCConfig dataclass. Scalars, Paths and EngineTypes
        are all probed generically.
        """
        names = {f.name for f in dataclasses.fields(PipelineConfig)}
        assert UNPROBED_FIELDS <= names

        defaults = PipelineConfig()
        unprobable = {
            f.name
            for f in dataclasses.fields(PipelineConfig)
            if isinstance(getattr(defaults, f.name), (list, HPCConfig))
        }
        assert unprobable == UNPROBED_FIELDS, (
            "fields the generic probe cannot cover but which are not declared in "
            f"UNPROBED_FIELDS (or vice versa): {unprobable ^ UNPROBED_FIELDS}"
        )

    @pytest.mark.parametrize("name", PREVIOUSLY_DROPPED)
    def test_previously_dropped_field_round_trips(self, name, tmp_path):
        default = getattr(PipelineConfig(), name)
        written = _probe_value(name, default)

        config = PipelineConfig.from_file(_write(tmp_path, {name: written}))

        assert getattr(config, name) == written

    def test_cost_caps_from_issue_reproduction(self, tmp_path):
        """The exact reproduction in #240."""
        config = PipelineConfig.from_file(
            _write(
                tmp_path,
                {
                    "qwen_vllm_model": "Org/DIFFERENT-MODEL",
                    "cost_budget": 0.01,
                    "max_cost_per_page": 0.005,
                    "strict_local": True,
                    "agentic": False,
                },
            )
        )

        assert config.qwen_vllm_model == "Org/DIFFERENT-MODEL"
        assert config.cost_budget == 0.01
        assert config.max_cost_per_page == 0.005
        assert config.strict_local is True
        assert config.agentic is False


class TestExplicitlyHandledFields:
    def test_engine_fields_round_trip(self, tmp_path):
        config = PipelineConfig.from_file(
            _write(
                tmp_path,
                {
                    "primary_engine": "qwen",
                    "local_engine": "glm",
                    "figures_engine": "gemini",
                    "fallback_chain": ["nougat", "marker"],
                    "enabled_engines": ["qwen", "glm"],
                },
            )
        )

        assert config.primary_engine == EngineType.QWEN
        assert config.local_engine == EngineType.GLM
        assert config.figures_engine == EngineType.GEMINI
        assert config.fallback_chain == [EngineType.NOUGAT, EngineType.MARKER]
        assert config.enabled_engines == [EngineType.QWEN, EngineType.GLM]

    def test_legacy_fallback_engine_alias_still_accepted(self, tmp_path):
        config = PipelineConfig.from_file(_write(tmp_path, {"fallback_engine": "mistral"}))

        assert config.fallback_chain == [EngineType.MISTRAL]

    def test_output_dir_becomes_a_path(self, tmp_path):
        config = PipelineConfig.from_file(_write(tmp_path, {"output_dir": "/tmp/socr-out"}))

        assert config.output_dir == Path("/tmp/socr-out")

    def test_hpc_block_round_trips(self, tmp_path):
        from socr.core.config import HPCConfig

        defaults = HPCConfig()
        payload = {
            f.name: _probe_value(f.name, getattr(defaults, f.name))
            for f in dataclasses.fields(HPCConfig)
        }

        config = PipelineConfig.from_file(_write(tmp_path, {"hpc": payload}))

        for name, written in payload.items():
            assert getattr(config.hpc, name) == written, name


class TestUnknownKeys:
    def test_unknown_top_level_key_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="cost_budgets"):
            PipelineConfig.from_file(_write(tmp_path, {"cost_budgets": 0.5}))

    def test_unknown_hpc_key_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match=r"hpc\.gpu_typo"):
            PipelineConfig.from_file(_write(tmp_path, {"hpc": {"gpu_typo": "a100"}}))

    def test_error_message_points_at_the_valid_names(self, tmp_path):
        """The message must be actionable without reading the source."""
        with pytest.raises(ValueError) as exc:
            PipelineConfig.from_file(_write(tmp_path, {"cost_budgets": 0.5}))

        message = str(exc.value)
        assert "PipelineConfig" in message
        assert "HPCConfig" in message
        assert "config.py" in message

    def test_all_unknown_keys_are_reported_together(self, tmp_path):
        with pytest.raises(ValueError) as exc:
            PipelineConfig.from_file(_write(tmp_path, {"foo": 1, "bar": 2}))

        assert "bar" in str(exc.value)
        assert "foo" in str(exc.value)

    def test_non_string_unknown_key_is_reported_not_crashed(self, tmp_path):
        """YAML allows non-string keys; a bare ``1:`` must not crash the reporter.

        Sorting and joining a mix of int and str raises TypeError while building
        the very message meant to explain the problem, turning a clear config
        error into an opaque stack trace.
        """
        with pytest.raises(ValueError, match="1"):
            PipelineConfig.from_file(_write(tmp_path, {1: "oops"}))

    def test_non_mapping_hpc_block_is_rejected(self, tmp_path):
        """``hpc:`` given a scalar silently yielded the default HPCConfig."""
        with pytest.raises(ValueError, match="hpc"):
            PipelineConfig.from_file(_write(tmp_path, {"hpc": "not-a-mapping"}))

    def test_empty_file_loads_defaults(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")

        assert PipelineConfig.from_file(path).agentic is PipelineConfig().agentic


class TestGH678YamlCapPinsLikeCli:
    """GH-678: a YAML ``max_cost_per_page`` must pin the same way the CLI flag
    does, so the two channels agree on the resulting cloud-egress policy.

    The CLI side is driven through the REAL Click command (``test_gh168_config_
    precedence.py``'s ``_run_with`` helper -- monkeypatch ``UnifiedPipeline``
    with a config-capturing stub, invoke ``socr process``, read back the config
    it built), not a hand-written model of ``cli.py``'s
    ``_explicitly_given("max_cost_per_page")`` block. A hand-copied mirror of
    that block would keep passing if the block itself changed -- the two sides
    drifting apart unnoticed is this ticket's own failure mode one layer up.
    """

    @staticmethod
    def _cli_config(tmp_path, monkeypatch, value):
        from test_gh168_config_precedence import _run_with

        return _run_with(tmp_path, "", ["--max-cost-per-page", str(value)], monkeypatch)

    def test_zero_cap_parity_forbids_cloud_on_both_channels(self, tmp_path, monkeypatch):
        yaml_dir = tmp_path / "yaml"
        yaml_dir.mkdir()
        yaml_config = PipelineConfig.from_file(_write(yaml_dir, {"max_cost_per_page": 0}))
        cli_config = self._cli_config(tmp_path / "cli", monkeypatch, 0)

        assert yaml_config.max_cost_per_page_pinned is True
        assert cli_config.max_cost_per_page_pinned is True
        assert zero_cap_pinned_forbids_cloud(yaml_config) is zero_cap_pinned_forbids_cloud(
            cli_config
        )
        assert zero_cap_pinned_forbids_cloud(yaml_config) is True

    def test_nonzero_cap_parity_does_not_forbid_cloud_on_either_channel(
        self, tmp_path, monkeypatch
    ):
        yaml_dir = tmp_path / "yaml"
        yaml_dir.mkdir()
        yaml_config = PipelineConfig.from_file(_write(yaml_dir, {"max_cost_per_page": 5}))
        cli_config = self._cli_config(tmp_path / "cli", monkeypatch, 5)

        assert yaml_config.max_cost_per_page_pinned is True
        assert cli_config.max_cost_per_page_pinned is True
        assert zero_cap_pinned_forbids_cloud(yaml_config) is zero_cap_pinned_forbids_cloud(
            cli_config
        )
        assert zero_cap_pinned_forbids_cloud(yaml_config) is False

    def test_absent_key_leaves_unpinned(self, tmp_path):
        """A fix that pins unconditionally would forbid cloud for every
        config-file user -- far worse than the bug this ticket fixes."""
        config = PipelineConfig.from_file(_write(tmp_path, {"agentic": False}))

        assert config.max_cost_per_page_pinned is False
        assert zero_cap_pinned_forbids_cloud(config) is False


class TestGH678MalformedCapFailsAtLoad:
    """GH-678 round 2: pinning on key presence made a malformed cap reachable.

    ``zero_cap_pinned_forbids_cloud`` is ``bool(pinned) and (value <= 0.0)``.
    Before this ticket a YAML config never set the pin, so the first operand
    short-circuited and the comparison never ran — a null or quoted cap was
    inert. Pinning on presence removes the short-circuit, and that function is
    called on essentially every run (orchestrator, hpc_pipeline,
    table_cell_guard), so a malformed value would surface as a TypeError three
    frames deep in provider routing rather than as a config error.

    These pin that the failure is a ``ValueError`` naming the key, raised while
    the config is being loaded.
    """

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(None, id="yaml-null"),
            pytest.param("abc", id="non-numeric-string"),
            pytest.param(float("nan"), id="nan-disables-the-cap-silently"),
        ],
    )
    def test_malformed_cap_raises_at_load(self, tmp_path, value):
        path = _write(tmp_path, {"max_cost_per_page": value})

        with pytest.raises(ValueError, match="max_cost_per_page"):
            PipelineConfig.from_file(path)

    def test_boolean_cap_is_refused_rather_than_read_as_zero(self, tmp_path):
        """``false`` compares equal to 0 and would forbid ALL cloud egress.

        Silently inferring a no-cloud policy from a boolean is worse than
        refusing it: the user never wrote that policy.
        """
        path = _write(tmp_path, {"max_cost_per_page": False})

        with pytest.raises(ValueError, match="boolean"):
            PipelineConfig.from_file(path)

    @pytest.mark.parametrize("value", [0, 5, -1, 0.25, "0", "5.5"])
    def test_wellformed_caps_still_load_and_pin(self, tmp_path, value):
        """The validation must not reject the values the fix exists to support.

        A negative cap is deliberately allowed: ``zero_cap_pinned_forbids_cloud``
        treats anything ``<= 0.0`` as "no paid calls", so ``-1`` is a stated
        policy, not a malformed one. A quoted number (``max_cost_per_page: "0"``,
        which YAML reads as a string) is accepted and coerced rather than
        refused — the user wrote a number, and the resulting policy is the one
        they wrote. It crashed before this validation existed, which is what
        made the quoting matter at all.
        """
        config = PipelineConfig.from_file(_write(tmp_path, {"max_cost_per_page": value}))

        assert config.max_cost_per_page == float(value)
        assert config.max_cost_per_page_pinned is True
        assert zero_cap_pinned_forbids_cloud(config) is (float(value) <= 0.0)


class TestGH825YamlQwenModelPinsLikeCli:
    """GH-825: the GH-678 shape, one field over.

    ``cli.py``'s ``--qwen-model`` block sets ``qwen_model`` AND
    ``qwen_model_pinned`` together. ``from_file``'s generic restore loop set the
    value and never touched the pin, so rule 1 of ``resolve_qwen_intent``
    ("explicit pin -> pass the model through unchanged") never fired for a
    YAML-set model: the same model was honoured from the flag and silently
    replaced by the local instruct model from the file.

    The assertions pin a DIFFERENCE between the two channels rather than an
    absolute resolved model, so they do not encode whatever ``OLLAMA_MODEL``
    happens to be.
    """

    CLOUD_MODEL = "qwen3.5:cloud"

    @staticmethod
    def _cli_config(tmp_path, monkeypatch, value):
        from test_gh168_config_precedence import _run_with

        return _run_with(tmp_path, "", ["--qwen-model", value], monkeypatch)

    @pytest.fixture(autouse=True)
    def _no_vllm_env(self, monkeypatch):
        # ``auto`` + VLLM_BASE_URL resolves to the vllm rung, which takes a
        # different branch of ``resolve_qwen_intent``. Unset it so the test
        # measures the local/auto branch this ticket is about.
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)

    def test_yaml_model_pins_and_resolves_exactly_as_the_flag_does(self, tmp_path, monkeypatch):
        from socr.engines.qwen import resolve_qwen_intent

        yaml_dir = tmp_path / "yaml"
        yaml_dir.mkdir()
        yaml_config = PipelineConfig.from_file(
            _write(yaml_dir, {"qwen_model": self.CLOUD_MODEL, "qwen_backend": "auto"})
        )
        cli_config = self._cli_config(tmp_path / "cli", monkeypatch, self.CLOUD_MODEL)

        assert yaml_config.qwen_model_pinned is True
        assert cli_config.qwen_model_pinned is True
        assert resolve_qwen_intent(yaml_config) == resolve_qwen_intent(cli_config)
        assert resolve_qwen_intent(yaml_config)[1] == self.CLOUD_MODEL

    def test_absent_key_leaves_unpinned_and_resolves_to_the_local_default(self, tmp_path):
        """The negative control.

        A fix that pinned unconditionally would let a stale ``qwen_model``
        default reach a local backend for every config-file user -- the exact
        accident rule 3 exists to prevent. With no ``qwen_model`` key the pin
        must stay False and the resolved model must be the local instruct MoE,
        NOT whatever the dataclass default string happens to be.
        """
        from socr.engines.qwen import OLLAMA_MODEL, resolve_qwen_intent

        config = PipelineConfig.from_file(
            _write(tmp_path, {"agentic": False, "qwen_backend": "auto"})
        )

        assert config.qwen_model_pinned is False
        assert resolve_qwen_intent(config) == ("auto", OLLAMA_MODEL)

    def test_the_key_is_what_makes_the_difference(self, tmp_path):
        """Pin the DIFFERENCE, not the value: the same loader, the same backend,
        one key added, and only the resolved model changes."""
        from socr.engines.qwen import resolve_qwen_intent

        without_dir = tmp_path / "without"
        without_dir.mkdir()
        with_dir = tmp_path / "with"
        with_dir.mkdir()

        without = PipelineConfig.from_file(_write(without_dir, {"qwen_backend": "auto"}))
        with_key = PipelineConfig.from_file(
            _write(with_dir, {"qwen_backend": "auto", "qwen_model": self.CLOUD_MODEL})
        )

        assert resolve_qwen_intent(without)[0] == resolve_qwen_intent(with_key)[0]
        assert resolve_qwen_intent(without)[1] != resolve_qwen_intent(with_key)[1]
        assert resolve_qwen_intent(with_key)[1] == self.CLOUD_MODEL
