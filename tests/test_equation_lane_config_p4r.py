"""P4-R t4: `equation_region_lane` config field, CLI kill switch, YAML
round-trip, and `_run_fingerprint` coverage.

`equation_region_lane` is the default-on P4-R widening (trigger `has_equations`
as detected, native-prose floor). It is a SEPARATE field from the existing
`detect_equations` / `recover_clean_equations` (both default False, and both
continue to describe the legacy GH-36 opt-in path unchanged by this ticket).

`--no-equation-region-lane` is the kill switch through `common_options` /
`build_config`. `clean_equation_model` must be fingerprinted when EITHER P4-R
or the legacy `detect_equations and recover_clean_equations` lane can run, and
ignored only when neither consumer can run (this widens, not replaces, the
existing GH-36 fingerprint contract at orchestrator.py:~606).

These tests target a field/flag that does not exist on this branch yet (t4 is
unimplemented) and are expected to fail until it lands.
"""

from __future__ import annotations

import dataclasses

import pytest

from socr.core.config import PipelineConfig


def test_the_field_exists_at_all():
    """Cold review round 1, finding 6: this file used to skip itself when the
    field was missing, so deleting the feature turned every acceptance test in
    it green. The absence is now a failure, once, here."""
    assert any(f.name == "equation_region_lane" for f in dataclasses.fields(PipelineConfig))


class TestConfigField:
    def test_default_is_true(self):
        cfg = PipelineConfig()
        assert cfg.equation_region_lane is True

    def test_legacy_defaults_unchanged(self):
        cfg = PipelineConfig()
        assert cfg.detect_equations is False
        assert cfg.recover_clean_equations is False

    def test_field_round_trips_through_from_file(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("equation_region_lane: false\n")
        cfg = PipelineConfig.from_file(yaml_path)
        assert cfg.equation_region_lane is False

    def test_field_round_trips_true_through_from_file(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("equation_region_lane: true\n")
        cfg = PipelineConfig.from_file(yaml_path)
        assert cfg.equation_region_lane is True


class TestCLIKillSwitch:
    def test_process_help_shows_kill_switch(self):
        from click.testing import CliRunner

        from socr.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["process", "--help"])
        assert result.exit_code == 0
        assert "--no-equation-region-lane" in result.output

    def test_batch_help_shows_kill_switch(self):
        from click.testing import CliRunner

        from socr.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["batch", "--help"])
        assert result.exit_code == 0
        assert "--no-equation-region-lane" in result.output

    def test_explicit_flag_sets_field_false(self):
        from socr.cli import build_config

        cfg = build_config(equation_region_lane=False)
        assert cfg.equation_region_lane is False

    def test_omitted_flag_does_not_override_yaml_value(self, tmp_path):
        # NOTE: the parameter is ``config_path`` (a Path), not ``config_file``;
        # the tests stage guessed the name. Corrected against the real
        # ``build_config`` signature -- no ruling is involved.
        from socr.cli import build_config

        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("equation_region_lane: false\n")
        cfg = build_config(config_path=yaml_path)
        assert cfg.equation_region_lane is False

    def test_kill_switch_flag_sets_field_false(self):
        from socr.cli import build_config

        cfg = build_config(no_equation_region_lane=True)
        assert cfg.equation_region_lane is False

    def test_kill_switch_omitted_leaves_default_on(self):
        from socr.cli import build_config

        cfg = build_config()
        assert cfg.equation_region_lane is True


class TestFingerprintCoverage:
    def _pipeline(self, **overrides):
        from socr.pipeline.orchestrator import UnifiedPipeline

        cfg = PipelineConfig(**overrides)
        return UnifiedPipeline(cfg)

    def test_fingerprint_records_flag_on_vs_off(self):
        fp_on = self._pipeline(equation_region_lane=True)._run_fingerprint()
        fp_off = self._pipeline(equation_region_lane=False)._run_fingerprint()
        assert fp_on != fp_off

    def test_model_changes_fingerprint_with_p4r_on(self):
        fp_a = self._pipeline(
            equation_region_lane=True, clean_equation_model="qwen3-vl:30b-a3b-instruct"
        )._run_fingerprint()
        fp_b = self._pipeline(
            equation_region_lane=True, clean_equation_model="some-other-model"
        )._run_fingerprint()
        assert fp_a != fp_b

    def test_model_changes_fingerprint_with_p4r_off_but_legacy_gh36b_on(self):
        fp_a = self._pipeline(
            equation_region_lane=False,
            detect_equations=True,
            recover_clean_equations=True,
            clean_equation_model="qwen3-vl:30b-a3b-instruct",
        )._run_fingerprint()
        fp_b = self._pipeline(
            equation_region_lane=False,
            detect_equations=True,
            recover_clean_equations=True,
            clean_equation_model="some-other-model",
        )._run_fingerprint()
        assert fp_a != fp_b

    def test_model_does_not_change_fingerprint_with_both_consumers_off(self):
        fp_a = self._pipeline(
            equation_region_lane=False,
            detect_equations=False,
            recover_clean_equations=False,
            clean_equation_model="qwen3-vl:30b-a3b-instruct",
        )._run_fingerprint()
        fp_b = self._pipeline(
            equation_region_lane=False,
            detect_equations=False,
            recover_clean_equations=False,
            clean_equation_model="some-other-model",
        )._run_fingerprint()
        assert fp_a == fp_b
