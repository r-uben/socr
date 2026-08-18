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

from socr.core.config import EngineType, PipelineConfig

# Fields that cannot be probed by the generic scalar round-trip below, each for a
# stated structural reason -- NOT because they are allowed to be unrestorable.
# Both are covered by dedicated tests further down.
UNPROBED_FIELDS = {
    "hpc",  # nested dataclass, not a scalar -- see test_hpc_block_round_trips
    "enabled_engines",  # list[EngineType] -- see test_engine_fields_round_trip
    "fallback_chain",  # list[EngineType] -- see test_engine_fields_round_trip
    "multi_engine",  # list[EngineType] -- see test_engine_fields_round_trip
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

    def test_unproved_exception_list_is_exhaustive(self):
        """UNPROBED_FIELDS must name only real PipelineConfig fields."""
        names = {f.name for f in dataclasses.fields(PipelineConfig)}
        assert UNPROBED_FIELDS <= names

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
                    "multi_engine": ["qwen", "gemini"],
                },
            )
        )

        assert config.primary_engine == EngineType.QWEN
        assert config.local_engine == EngineType.GLM
        assert config.figures_engine == EngineType.GEMINI
        assert config.fallback_chain == [EngineType.NOUGAT, EngineType.MARKER]
        assert config.enabled_engines == [EngineType.QWEN, EngineType.GLM]
        assert config.multi_engine == [EngineType.QWEN, EngineType.GEMINI]

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

    def test_empty_file_loads_defaults(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")

        assert PipelineConfig.from_file(path).agentic is PipelineConfig().agentic
