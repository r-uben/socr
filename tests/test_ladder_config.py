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
from socr.core.config import (
    TABLE_JUDGE_ADJUDICATOR_MODEL_DEFAULT,
    TABLE_JUDGE_TIMEOUT_SEC_DEFAULT,
    PipelineConfig,
)
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
    # P1 (owner ruling Q3, 2026-09-03): the default flipped to True. The
    # default-off pin that used to live here is SUPERSEDED, not deleted --
    # it moved to ``TestP1DefaultFlip`` below, which pins the new default and
    # the tri-state CLI pair that keeps a YAML value alive.

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

    def test_flag_absent_leaves_the_configured_default_end_to_end(
        self, dummy_pdf: Path, tmp_path: Path
    ):
        """The `process` CLI path, with the pipeline stubbed at the same seam
        `test_gh190_empty_table_surfacing.py` uses, actually builds a config
        carrying the CONFIG default when the flag is never passed -- which the
        Q3 ruling moved from False to True.

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
        assert built_config.table_judge_ladder is PipelineConfig().table_judge_ladder

    def test_flag_absent_leaves_the_default_with_ollama_unreachable(
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
        assert built_config.table_judge_ladder is PipelineConfig().table_judge_ladder

    def test_build_config_flag_on(self):
        config = build_config(table_judge_ladder=True)
        assert config.table_judge_ladder is True

    def test_build_config_takes_the_config_default_when_no_flag_is_passed(self):
        # P1 (owner ruling Q3): the assertion is now against the config's own
        # default rather than a literal False, so this test says what it means
        # -- an omitted CLI flag changes nothing -- and stops re-pinning a
        # value another test already owns.
        config = build_config()
        assert config.table_judge_ladder is PipelineConfig().table_judge_ladder

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

        # P1 (owner ruling Q3): "the user typed nothing" is now ``None``, not
        # ``False`` -- ``False`` is an explicit --no-table-judge-ladder and
        # MUST win over a YAML True. Passing None here keeps this test about
        # what it was always about: an unset flag not clobbering the config.
        config = build_config(config_path=config_path, table_judge_ladder=None)

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


# ---------------------------------------------------------------------------
# P1 (task t3): the adjudicator's own config knobs. Deliberately only the
# ruled ones -- binary, model, and cost -- per the plan's explicit
# constraint: no table_judge_tiebreak/table_judge_withhold_rejected feature
# switches, which would create unruled product states merely to serve tests.
# ---------------------------------------------------------------------------


class TestKimiAdjudicatorConfig:
    def test_adjudicator_config_fields_have_sane_defaults(self):
        config = PipelineConfig()
        # The model is a named constant, not an inline literal: it records a
        # measured transport choice (cold review round 1, finding 4).
        assert config.table_judge_adjudicator_model == TABLE_JUDGE_ADJUDICATOR_MODEL_DEFAULT
        assert config.table_judge_adjudicator_model == "kimi-k2.6:cloud"
        # A different VENDOR from either reader rung -- that is where the
        # independence lives, since the transport is deliberately shared.
        assert config.table_judge_adjudicator_model != config.table_judge_rung1_model
        assert config.table_judge_adjudicator_host is None
        # A subscription-backed rung defaults to a KNOWN zero, not None
        # (unmetered) and not a guessed nonzero price.
        assert config.table_judge_adjudicator_cost_per_call_usd == 0.0

    def test_adjudicator_fields_round_trip_through_yaml(self, tmp_path: Path):
        data = {
            "table_judge_ladder": True,
            "table_judge_adjudicator_model": "probe-adjudicator:cloud",
            "table_judge_adjudicator_host": "http://probe-host:11434",
            "table_judge_adjudicator_cost_per_call_usd": 0.03,
        }
        config_path = tmp_path / "adjudicator.yaml"
        config_path.write_text(yaml.dump(data))

        config = PipelineConfig.from_file(config_path)

        assert config.table_judge_adjudicator_model == "probe-adjudicator:cloud"
        assert config.table_judge_adjudicator_host == "http://probe-host:11434"
        assert config.table_judge_adjudicator_cost_per_call_usd == 0.03

    def test_no_unruled_tiebreak_or_withhold_feature_flags_exist(self):
        """The plan's explicit constraint: do not add
        table_judge_tiebreak/table_judge_withhold_rejected switches -- the
        chain is always-on once the ladder is enabled."""
        config = PipelineConfig()
        assert not hasattr(config, "table_judge_tiebreak")
        assert not hasattr(config, "table_judge_withhold_rejected")


class TestRunFingerprintBindsKimiOnlyWhenLadderEnabled:
    def test_kimi_binary_changes_the_enabled_fingerprint(self):
        base = _run_fingerprint_for(
            PipelineConfig(table_judge_ladder=True, table_judge_adjudicator_host=None)
        )
        changed = _run_fingerprint_for(
            PipelineConfig(
                table_judge_ladder=True, table_judge_adjudicator_host="http://elsewhere:11434"
            )
        )
        assert base != changed

    def test_kimi_model_changes_the_enabled_fingerprint(self):
        base = _run_fingerprint_for(
            PipelineConfig(table_judge_ladder=True, table_judge_adjudicator_model="kimi-k3-max")
        )
        changed = _run_fingerprint_for(
            PipelineConfig(
                table_judge_ladder=True, table_judge_adjudicator_model="other-adjudicator:cloud"
            )
        )
        assert base != changed

    def test_kimi_cost_changes_the_enabled_fingerprint(self):
        base = _run_fingerprint_for(
            PipelineConfig(table_judge_ladder=True, table_judge_adjudicator_cost_per_call_usd=0.0)
        )
        changed = _run_fingerprint_for(
            PipelineConfig(table_judge_ladder=True, table_judge_adjudicator_cost_per_call_usd=0.05)
        )
        assert base != changed

    def test_none_of_the_kimi_fields_change_the_disabled_fingerprint(self):
        base = _run_fingerprint_for(
            PipelineConfig(
                table_judge_ladder=False,
                table_judge_adjudicator_host=None,
                table_judge_adjudicator_cost_per_call_usd=0.0,
            )
        )
        changed = _run_fingerprint_for(
            PipelineConfig(
                table_judge_ladder=False,
                table_judge_adjudicator_host="http://elsewhere:11434",
                table_judge_adjudicator_cost_per_call_usd=0.09,
            )
        )
        assert base == changed


def _run_fingerprint_for(config: PipelineConfig) -> str:
    """Resolve the run fingerprint the same way the resume gate does."""
    from socr.pipeline.orchestrator import UnifiedPipeline

    pipeline = UnifiedPipeline(config)
    return pipeline._run_fingerprint()


# ---------------------------------------------------------------------------
# P1 (task t3): reader vs adjudicator role separation. Kimi must never widen
# the "any reader reachable" predicate the no-reader startup diagnostic and
# the ordinary two-rung ladder both use.
# ---------------------------------------------------------------------------


class TestReaderAdjudicatorRoleSeparation:
    def test_kimi_only_reachable_still_reports_no_reader_available(self):
        from socr.pipeline.orchestrator import UnifiedPipeline

        pipeline = UnifiedPipeline(_config_ladder_on())
        with (
            patch(
                "socr.pipeline.orchestrator.table_judge_ollama_rung_reachable", return_value=False
            ),
            # The orchestrator imports the gemini probe under this alias; the
            # tests-1 draft patched a name that does not exist there, which
            # would have silently probed the real binary.
            patch(
                "socr.pipeline.orchestrator.table_judge_gemini_rung_reachable", return_value=False
            ),
            patch(
                "socr.pipeline.orchestrator.table_judge_adjudicator_reachable", return_value=True
            ),
        ):
            assert pipeline._any_table_judge_reader_reachable() is False

    def test_the_adjudicator_is_never_counted_as_a_reader_rung_kind(self):
        from socr.judge.table_verdict import (
            RUNG_KIND_CELL_ADJUDICATOR,
            RUNG_KIND_GEMINI,
            RUNG_KIND_OLLAMA,
        )
        from socr.pipeline.orchestrator import UnifiedPipeline

        assert RUNG_KIND_CELL_ADJUDICATOR not in UnifiedPipeline.TABLE_JUDGE_RUNG_KINDS
        assert {RUNG_KIND_OLLAMA, RUNG_KIND_GEMINI} == set(UnifiedPipeline.TABLE_JUDGE_RUNG_KINDS)
        # But it IS probeable in its own right (cold review round 1, finding 2):
        # latchable without being askable is how a latch becomes permanent.
        assert RUNG_KIND_CELL_ADJUDICATOR in UnifiedPipeline.TABLE_JUDGE_PROBEABLE_KINDS


def _config_ladder_on(**overrides) -> PipelineConfig:
    kwargs = dict(table_judge_ladder=True)
    kwargs.update(overrides)
    return PipelineConfig(**kwargs)


# ---------------------------------------------------------------------------
# P1 (task t13): the default flip -- table_judge_ladder=True -- and the
# tri-state --table-judge-ladder/--no-table-judge-ladder CLI pair that keeps
# a YAML value alive when neither spelling is passed on the command line.
#
# NOTE: this class intentionally SUPERSEDES the pre-P1 ``TestDefaults`` and
# some ``TestCLIFlag`` assertions above, which pinned the OLD default-off
# behaviour (docstring: "byte-identity/golden tests must stay unaffected
# until the flag flips" -- P1 IS that flip). Both classes are kept: the old
# ones document what changed, and CI failing on both at once until the
# implementation lands is the expected RED state for this ticket.
# ---------------------------------------------------------------------------


class TestP1DefaultFlip:
    def test_flag_defaults_on(self):
        assert PipelineConfig().table_judge_ladder is True

    def test_build_config_defaults_on_with_no_flag_passed(self):
        config = build_config()
        assert config.table_judge_ladder is True

    def test_cli_no_flag_form_turns_it_off(self):
        config = build_config(table_judge_ladder=False)
        assert config.table_judge_ladder is False

    def test_yaml_false_survives_when_neither_cli_spelling_is_passed(self, tmp_path: Path):
        """The tri-state contract: build_config's CLI param must default to
        None (not False) so an omitted flag does not clobber a YAML False
        back to the new True default."""
        config_path = tmp_path / "ladder.yaml"
        config_path.write_text(yaml.dump({"table_judge_ladder": False}))

        config = build_config(config_path=config_path, table_judge_ladder=None)

        assert config.table_judge_ladder is False

    def test_yaml_true_survives_when_neither_cli_spelling_is_passed(self, tmp_path: Path):
        config_path = tmp_path / "ladder.yaml"
        config_path.write_text(yaml.dump({"table_judge_ladder": True}))

        config = build_config(config_path=config_path, table_judge_ladder=None)

        assert config.table_judge_ladder is True

    def test_explicit_cli_off_wins_over_yaml_true(self, tmp_path: Path):
        config_path = tmp_path / "ladder.yaml"
        config_path.write_text(yaml.dump({"table_judge_ladder": True}))

        config = build_config(config_path=config_path, table_judge_ladder=False)

        assert config.table_judge_ladder is False

    def test_process_help_lists_both_spellings(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["process", "--help"])
        assert result.exit_code == 0
        assert "--table-judge-ladder" in result.output
        assert "--no-table-judge-ladder" in result.output

    def test_help_states_fail_closed_and_distinguishes_withheld(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["process", "--help"])
        # Narrowed from the tests-1 draft, which searched the WHOLE help
        # output for "default off" -- unrelated flags (the equation lane) are
        # legitimately default-off and made the assertion fail on their text.
        # Read the ladder flag's own help block instead.
        block = _help_block_for(result.output, "--table-judge-ladder")
        lowered = " ".join(block.lower().split())
        assert "on by default" in lowered
        assert "default off" not in lowered
        assert "unverified" in lowered
        assert "withheld" in lowered

    def test_flag_absent_leaves_ladder_on_end_to_end(self, dummy_pdf: Path, tmp_path: Path):
        runner = CliRunner()
        with patch(
            "socr.pipeline.orchestrator.UnifiedPipeline", side_effect=_StubPipeline
        ) as mock_pipeline_cls:
            result = runner.invoke(
                cli,
                ["process", str(dummy_pdf), "--primary", "qwen", "-o", str(tmp_path / "out"), "-q"],
            )

        assert result.exit_code == 0, result.output
        (built_config,) = mock_pipeline_cls.call_args.args
        assert built_config.table_judge_ladder is True

    def test_explicit_no_table_judge_ladder_flag_turns_it_off_end_to_end(
        self, dummy_pdf: Path, tmp_path: Path
    ):
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
                    "--no-table-judge-ladder",
                    "-o",
                    str(tmp_path / "out"),
                    "-q",
                ],
            )

        assert result.exit_code == 0, result.output
        (built_config,) = mock_pipeline_cls.call_args.args
        assert built_config.table_judge_ladder is False


def _help_block_for(output: str, option: str) -> str:
    """The help text belonging to one option, up to the next option line."""
    lines = output.splitlines()
    start = next(i for i, line in enumerate(lines) if option in line)
    block = [lines[start]]
    for line in lines[start + 1 :]:
        if line.strip().startswith("-") and not line.strip().startswith("--strict"):
            if line.lstrip().startswith("-") and "  " in line.strip():
                break
        block.append(line)
        if len(block) > 12:
            break
    return "\n".join(block)


class TestP1NoReaderStartupDiagnostic:
    """The generic diagnostic missing at prep time (found absent in this
    checkout, not merely "not yet wired" as #553's text implied): when the
    ladder is enabled and NEITHER reader rung is reachable, name the fallback
    outcome and the opt-out. Kimi reachability must never suppress it."""

    def test_no_reader_reachable_prints_the_fallback_and_opt_out(
        self, dummy_pdf: Path, tmp_path: Path
    ):
        runner = CliRunner()
        with (
            patch("socr.pipeline.orchestrator.UnifiedPipeline", side_effect=_StubPipeline),
            patch("socr.cli.table_judge_ollama_rung_reachable", return_value=False),
            patch("socr.cli.gemini_rung_reachable", return_value=False),
        ):
            result = runner.invoke(
                cli,
                ["process", str(dummy_pdf), "--primary", "qwen", "-o", str(tmp_path / "out")],
            )

        assert result.exit_code == 0, result.output
        assert "UNVERIFIED" in result.output
        assert "--no-table-judge-ladder" in result.output

    def test_kimi_reachable_does_not_suppress_the_no_reader_diagnostic(
        self, dummy_pdf: Path, tmp_path: Path
    ):
        runner = CliRunner()
        with (
            patch("socr.pipeline.orchestrator.UnifiedPipeline", side_effect=_StubPipeline),
            patch("socr.cli.table_judge_ollama_rung_reachable", return_value=False),
            patch("socr.cli.gemini_rung_reachable", return_value=False),
            patch(
                "socr.pipeline.orchestrator.table_judge_adjudicator_reachable", return_value=True
            ),
        ):
            result = runner.invoke(
                cli,
                ["process", str(dummy_pdf), "--primary", "qwen", "-o", str(tmp_path / "out")],
            )

        assert "UNVERIFIED" in result.output

    def test_one_reader_reachable_suppresses_the_diagnostic(self, dummy_pdf: Path, tmp_path: Path):
        runner = CliRunner()
        with (
            patch("socr.pipeline.orchestrator.UnifiedPipeline", side_effect=_StubPipeline),
            patch("socr.cli.table_judge_ollama_rung_reachable", return_value=True),
            patch("socr.cli.gemini_rung_reachable", return_value=False),
        ):
            result = runner.invoke(
                cli,
                ["process", str(dummy_pdf), "--primary", "qwen", "-o", str(tmp_path / "out")],
            )

        assert "every table page will ship UNVERIFIED" not in result.output

    def test_diagnostic_is_suppressed_under_quiet(self, dummy_pdf: Path, tmp_path: Path):
        runner = CliRunner()
        with (
            patch("socr.pipeline.orchestrator.UnifiedPipeline", side_effect=_StubPipeline),
            patch("socr.cli.table_judge_ollama_rung_reachable", return_value=False),
            patch("socr.cli.gemini_rung_reachable", return_value=False),
        ):
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

        assert "every table page will ship UNVERIFIED" not in result.output

    def test_diagnostic_is_suppressed_when_ladder_explicitly_disabled(
        self, dummy_pdf: Path, tmp_path: Path
    ):
        runner = CliRunner()
        with (
            patch("socr.pipeline.orchestrator.UnifiedPipeline", side_effect=_StubPipeline),
            patch("socr.cli.table_judge_ollama_rung_reachable", return_value=False),
            patch("socr.cli.gemini_rung_reachable", return_value=False),
        ):
            result = runner.invoke(
                cli,
                [
                    "process",
                    str(dummy_pdf),
                    "--primary",
                    "qwen",
                    "--no-table-judge-ladder",
                    "-o",
                    str(tmp_path / "out"),
                ],
            )

        assert "every table page will ship UNVERIFIED" not in result.output

    def test_each_reader_kind_is_probed_at_most_once(self, dummy_pdf: Path, tmp_path: Path):
        calls: list[str] = []

        def _spy_ollama(*a, **kw):
            calls.append("ollama")
            return False

        def _spy_gemini(*a, **kw):
            calls.append("gemini")
            return False

        runner = CliRunner()
        with (
            patch("socr.pipeline.orchestrator.UnifiedPipeline", side_effect=_StubPipeline),
            patch("socr.cli.table_judge_ollama_rung_reachable", side_effect=_spy_ollama),
            patch("socr.cli.gemini_rung_reachable", side_effect=_spy_gemini),
        ):
            runner.invoke(
                cli,
                ["process", str(dummy_pdf), "--primary", "qwen", "-o", str(tmp_path / "out")],
            )

        assert calls.count("ollama") <= 1
        assert calls.count("gemini") <= 1


class TestAdjudicatorRateIsValidatedAtLoad:
    """Cold review round 1, finding 9.

    The per-call rate is compared against the per-page cap and the remaining
    document budget BEFORE a call and then recorded as that call's cost. A
    negative rate passes every pre-call check and then *reduces* the
    document's total after each call, manufacturing budget for later paid
    calls the user's cap was set to forbid. A non-finite one is worse: every
    comparison against NaN is False, so the cap silently stops existing.
    """

    @pytest.mark.parametrize("bad", [-1, -0.01, float("nan"), float("inf"), float("-inf")])
    def test_a_rate_that_cannot_mean_a_rate_is_rejected_in_the_constructor(self, bad):
        with pytest.raises(ValueError, match="table_judge_adjudicator_cost_per_call_usd"):
            PipelineConfig(table_judge_adjudicator_cost_per_call_usd=bad)

    @pytest.mark.parametrize("bad", [-1, ".nan", ".inf"])
    def test_a_rate_that_cannot_mean_a_rate_is_rejected_at_yaml_load(self, tmp_path: Path, bad):
        """``from_file`` assigns onto an already-constructed object, so it
        never re-enters ``__post_init__`` -- the check has to run there too."""
        config_path = tmp_path / "bad.yaml"
        config_path.write_text(f"table_judge_adjudicator_cost_per_call_usd: {bad}\n")
        with pytest.raises(ValueError, match="table_judge_adjudicator_cost_per_call_usd"):
            PipelineConfig.from_file(config_path)

    @pytest.mark.parametrize("ok", [0, 0.0, 0.05, 12])
    def test_a_known_non_negative_rate_is_accepted_and_kept_as_a_float(self, ok):
        config = PipelineConfig(table_judge_adjudicator_cost_per_call_usd=ok)
        assert config.table_judge_adjudicator_cost_per_call_usd == float(ok)
        assert isinstance(config.table_judge_adjudicator_cost_per_call_usd, float)
