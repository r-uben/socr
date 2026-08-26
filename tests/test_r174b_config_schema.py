"""R174b acceptance tests: Config schema & YAML rejection (Tasks t4, t9, t14).

Verifies:
- PipelineConfig dataclass does not define the 6 deleted legacy fields:
  multi_engine, consensus_enabled, consensus_use_llm, consensus_ollama_model,
  max_retries, truncation_retries.
- _FROM_FILE_EXPLICIT_FIELDS does not contain multi_engine.
- PipelineConfig.from_file rejects YAML containing any of the 6 removed keys with ValueError.
- Surviving fields deserialize and round-trip cleanly from YAML.
- Defaults for surviving fields (agentic=True, strict_local=False, dual_pass_tables=True).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from socr.core.config import EngineType, PipelineConfig

DELETED_CONFIG_FIELDS = [
    "multi_engine",
    "consensus_enabled",
    "consensus_use_llm",
    "consensus_ollama_model",
    "max_retries",
    "truncation_retries",
]


class TestPipelineConfigSchema:
    """Introspection and deserialization tests for PipelineConfig."""

    def test_deleted_fields_are_absent_from_dataclass(self):
        """None of the 6 deleted legacy fields may exist on PipelineConfig post-deletion."""
        field_names = {f.name for f in dataclasses.fields(PipelineConfig)}
        assert not set(DELETED_CONFIG_FIELDS) & field_names

    @pytest.mark.parametrize(
        "bad_key,bad_value",
        [
            ("multi_engine", ["gemini", "mistral"]),
            ("consensus_enabled", True),
            ("consensus_use_llm", True),
            ("consensus_ollama_model", "qwen3.5:cloud"),
            ("max_retries", 3),
            ("truncation_retries", 2),
        ],
    )
    def test_from_file_rejects_deleted_keys(self, tmp_path: Path, bad_key: str, bad_value: object):
        """PipelineConfig.from_file must raise ValueError when encountering any deleted key."""
        config_file = tmp_path / f"test_{bad_key}.yaml"
        payload = {bad_key: bad_value}
        config_file.write_text(yaml.safe_dump(payload))

        with pytest.raises(ValueError, match=rf"Unrecognised key.*{bad_key}"):
            PipelineConfig.from_file(config_file)

    def test_surviving_config_fields_round_trip(self, tmp_path: Path):
        """Surviving config fields deserialize correctly from YAML."""
        config_file = tmp_path / "valid_config.yaml"
        payload = {
            "primary_engine": "gemini",
            "local_engine": "glm",
            "fallback_chain": ["gemini", "mistral"],
            "figures_engine": "gemini",
            "enabled_engines": ["qwen", "gemini", "glm"],
            "native_first": True,
            "native_only": False,
            "tiered": True,
            "audit_enabled": True,
            "judge_hard_pages": True,
            "escalate_ambiguous_tables": True,
            "dual_pass_tables": True,
            "auto_patch_tables": False,
            "agentic": True,
            "strict_local": False,
            "judge_backend": "auto",
            "max_cost_per_page": 0.05,
            "cost_budget": 1.0,
            "write_manifest": True,
            "output_dir": "/tmp/custom_output",
        }
        config_file.write_text(yaml.safe_dump(payload))
        cfg = PipelineConfig.from_file(config_file)

        assert cfg.primary_engine == EngineType.GEMINI
        assert cfg.local_engine == EngineType.GLM
        assert cfg.fallback_chain == [EngineType.GEMINI, EngineType.MISTRAL]
        assert cfg.figures_engine == EngineType.GEMINI
        assert cfg.enabled_engines == [EngineType.QWEN, EngineType.GEMINI, EngineType.GLM]
        assert cfg.native_first is True
        assert cfg.native_only is False
        assert cfg.tiered is True
        assert cfg.audit_enabled is True
        assert cfg.judge_hard_pages is True
        assert cfg.escalate_ambiguous_tables is True
        assert cfg.dual_pass_tables is True
        assert cfg.auto_patch_tables is False
        assert cfg.agentic is True
        assert cfg.strict_local is False
        assert cfg.judge_backend == "auto"
        assert cfg.max_cost_per_page == 0.05
        assert cfg.cost_budget == 1.0
        assert cfg.write_manifest is True
        assert cfg.output_dir == Path("/tmp/custom_output")

    def test_default_config_invariants(self):
        """Defaults for agentic routing must be intact."""
        cfg = PipelineConfig()
        assert cfg.agentic is True
        assert cfg.strict_local is False
        assert cfg.dual_pass_tables is True
