"""TICKET-G1: table judge ladder config, CLI flag, and YAML round-trip (GH-353).

The ladder (docs/log/2026-08-30_table-judge-ladder.md) needs a default-off switch
and every knob config-visible so tests and later tickets (A2/A3/B1/H1) can inject
dummy rung identities instead of reading bare constants. This file pins:

- ``PipelineConfig().table_judge_ladder is False`` (byte-identity/golden tests must
  stay unaffected until the flag flips).
- ``socr process --help`` lists ``--table-judge-ladder``.
- The CLI flag does not clobber a YAML-config value when the flag is not passed
  (the ``cli.py:371``-area unconditional-override trap this ticket calls out).
- All five new fields round-trip through ``PipelineConfig.from_file`` (the
  dataclass-fields generic sweep in ``test_config_from_file.py`` already covers
  this mechanically; this file adds an explicit, readable pin).
- ``table_judge_timeout_sec`` defaults to 600 (>= the GH-356 bake-off's measured
  worst case: one 300s timeout that was correct on retry at a 590s cap).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from socr.cli import build_config, cli
from socr.core.config import TABLE_JUDGE_TIMEOUT_SEC_DEFAULT, PipelineConfig
from socr.core.result import DocumentStatus, EngineResult


@pytest.fixture
def dummy_pdf(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "dummy.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Sample text for CLI option parsing guard.")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


class _StubPipeline:
    """Stand-in for `UnifiedPipeline`: never touches an engine, a judge, or the
    network. `process()` returns a canned SUCCESS.

    Mirrors the `_PipelineStub` seam in `test_gh190_empty_table_surfacing.py`:
    patching `socr.pipeline.orchestrator.UnifiedPipeline` itself is the only
    way to make a `process()` CLI invocation hermetic here, because
    `--dry-run` is a `batch`-only guard (`process_batch` in orchestrator.py)
    and is silently INERT for the single-file `process` command — the bug
    that made the flag-off exit-0 test above look hermetic locally (ollama
    present) and fail in CI (no provider -> real routing was attempted).
    """

    def __init__(self, config: PipelineConfig) -> None:
        pass

    def process(self, pdf_path: Path, output_dir: Path | None = None) -> EngineResult:
        return EngineResult(document_path=pdf_path, engine="stub", status=DocumentStatus.SUCCESS)


class TestDefaults:
    def test_flag_defaults_off(self):
        assert PipelineConfig().table_judge_ladder is False

    def test_rung_identities_have_sane_defaults(self):
        config = PipelineConfig()
        assert config.table_judge_rung1_model == "glm-5.3-flash:cloud"
        assert config.table_judge_rung1_host is None
        # "agy" (Antigravity CLI), not the bare "gemini" CLI: the pre-merge B1
        # live smoke (2026-08-30) found the free-tier "gemini" headless auth
        # is dead (docs/log/2026-08-30_gh353-ticket-a3.md); agy reaches the
        # same model family through a live, working headless surface.
        assert config.table_judge_rung2_binary == "agy"

    def test_timeout_defaults_to_the_named_bakeoff_constant(self):
        config = PipelineConfig()
        assert config.table_judge_timeout_sec == TABLE_JUDGE_TIMEOUT_SEC_DEFAULT
        # The bake-off (docs/log/2026-08-30_gh356-bakeoff.md) measured a correct
        # retry at a 590s cap after a 300s timeout; the follow-up explicitly says
        # "timeout >= 600 s". Pin the floor, not a bare literal.
        assert TABLE_JUDGE_TIMEOUT_SEC_DEFAULT >= 600.0


class TestCLIFlag:
    def test_process_help_lists_the_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["process", "--help"])
        assert result.exit_code == 0
        assert "--table-judge-ladder" in result.output

    def test_batch_help_lists_the_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["batch", "--help"])
        assert result.exit_code == 0
        assert "--table-judge-ladder" in result.output

    def test_flag_absent_leaves_ladder_off_end_to_end(self, dummy_pdf: Path, tmp_path: Path):
        """The `process` CLI path, with the pipeline stubbed at the same seam
        `test_gh190_empty_table_surfacing.py` uses, actually builds a config
        with `table_judge_ladder is False` when the flag is never passed.

        `--primary qwen` avoids `EngineType.AUTO` -> `resolve_auto_engine()`,
        the other real-provider probe on this path; between that and the
        stubbed pipeline, nothing here can touch an engine, a judge, or the
        network.
        """
        runner = CliRunner()
        with patch(
            "socr.pipeline.orchestrator.UnifiedPipeline", side_effect=_StubPipeline
        ) as mock_pipeline_cls:
            result = runner.invoke(
                cli,
                [
                    "process",
                    str(dummy_pdf),
                    "--primary",
                    "qwen",
                    "-o",
                    str(tmp_path / "out"),
                    "-q",
                ],
            )

        assert result.exit_code == 0, result.output
        mock_pipeline_cls.assert_called_once()
        (built_config,) = mock_pipeline_cls.call_args.args
        assert built_config.table_judge_ladder is False

    def test_flag_absent_leaves_ladder_off_with_ollama_unreachable(
        self, dummy_pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Proves the above hermeticity claim: point `OLLAMA_HOST` at a dead
        port before invoking. If anything on this path made a live call, it
        would time out or raise a connection error instead of returning
        exit 0 — this must pass exactly like the reachable case.
        """
        monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")
        runner = CliRunner()
        with patch(
            "socr.pipeline.orchestrator.UnifiedPipeline", side_effect=_StubPipeline
        ) as mock_pipeline_cls:
            result = runner.invoke(
                cli,
                [
                    "process",
                    str(dummy_pdf),
                    "--primary",
                    "qwen",
                    "-o",
                    str(tmp_path / "out"),
                    "-q",
                ],
            )

        assert result.exit_code == 0, result.output
        (built_config,) = mock_pipeline_cls.call_args.args
        assert built_config.table_judge_ladder is False

    def test_build_config_flag_on(self):
        config = build_config(table_judge_ladder=True)
        assert config.table_judge_ladder is True

    def test_build_config_flag_off_by_default(self):
        config = build_config()
        assert config.table_judge_ladder is False

    def test_cli_flag_does_not_clobber_a_yaml_true_when_absent(self, tmp_path: Path):
        """The `cli.py:371` unconditional-override trap.

        `--strict-local` is a positive-only flag: `build_config` only ever flips
        it ON, never back OFF, so a YAML-config `True` survives an unset CLI
        flag. `--table-judge-ladder` must follow the same shape, unlike
        `judge_backend`/`max_cost_per_page`/etc., which DO clobber the loaded
        config because their CLI options carry non-None defaults of their own.
        """
        config_path = tmp_path / "ladder.yaml"
        config_path.write_text(yaml.dump({"table_judge_ladder": True}))

        config = build_config(config_path=config_path, table_judge_ladder=False)

        assert config.table_judge_ladder is True

    def test_cli_flag_turns_it_on_even_over_a_yaml_false(self, tmp_path: Path):
        config_path = tmp_path / "ladder.yaml"
        config_path.write_text(yaml.dump({"table_judge_ladder": False}))

        config = build_config(config_path=config_path, table_judge_ladder=True)

        assert config.table_judge_ladder is True


class TestYAMLRoundTrip:
    """Explicit pin alongside the generic sweep in test_config_from_file.py."""

    def test_all_five_fields_round_trip(self, tmp_path: Path):
        data = {
            "table_judge_ladder": True,
            "table_judge_rung1_model": "probe-model:cloud",
            "table_judge_rung1_host": "http://probe-host:11434",
            "table_judge_rung2_binary": "probe-gemini",
            "table_judge_timeout_sec": 900.0,
        }
        config_path = tmp_path / "ladder.yaml"
        config_path.write_text(yaml.dump(data))

        config = PipelineConfig.from_file(config_path)

        assert config.table_judge_ladder is True
        assert config.table_judge_rung1_model == "probe-model:cloud"
        assert config.table_judge_rung1_host == "http://probe-host:11434"
        assert config.table_judge_rung2_binary == "probe-gemini"
        assert config.table_judge_timeout_sec == 900.0

    def test_rung1_host_none_round_trips_as_unset(self, tmp_path: Path):
        """A config file that never mentions the host keeps the resolve-it sentinel."""
        config_path = tmp_path / "ladder.yaml"
        config_path.write_text(yaml.dump({"table_judge_ladder": True}))

        config = PipelineConfig.from_file(config_path)

        assert config.table_judge_rung1_host is None
