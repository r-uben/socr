"""P5 (GH-513 follow-up): the CLI surface for ``dual_pass_tables``.

Before this ticket the CLI exposed only a negative-only ``--no-dual-pass-
tables`` flag paired with a config default of True. Now the default is
False (see tests/test_dual_pass_tables.py::test_phase_disabled_flag_default),
so a negative-only flag can no longer turn the crop-reread escalation tool
ON from the CLI at all -- there must be a paired ``--dual-pass-tables`` /
``--no-dual-pass-tables`` option, and its ABSENCE must not clobber a value
loaded from ``--config``/``--profile`` (the GH-168 "flag that lies" failure
mode: test_gh168_config_precedence.py).

Driven through the real Click command for the same reason GH-168's tests
are: only Click can distinguish "the user typed nothing" from "the user
typed the default value" (``ParameterSource``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from socr.cli import cli


def _make_pdf(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    pdf = tmp_path / "d.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "born digital text long enough to be a text layer.")
    doc.save(str(pdf))
    doc.close()
    return pdf


def _run_capturing_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str],
    config_text: str | None = None,
):
    pdf = _make_pdf(tmp_path)
    seen: dict = {}

    class _Stub:
        def __init__(self, config):
            seen["config"] = config

        def process(self, *_a, **_k):
            raise SystemExit(0)

        def _resolve_output_root(self, *_a, **_k):
            return tmp_path / "out"

    monkeypatch.setattr("socr.pipeline.orchestrator.UnifiedPipeline", _Stub)

    cmd = ["process", str(pdf)]
    if config_text is not None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text(config_text)
        cmd += ["--config", str(cfg)]
    cmd += extra

    result = CliRunner().invoke(cli, cmd)
    assert "config" in seen, (
        f"the pipeline was never constructed, so nothing was measured: "
        f"exit_code={result.exit_code}, output={result.output}"
    )
    return seen["config"]


class TestDualPassTablesCLIDefaultAndOverrides:
    def test_default_is_off(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _run_capturing_config(tmp_path, monkeypatch, [])
        assert config.dual_pass_tables is False

    def test_explicit_on_flag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _run_capturing_config(tmp_path, monkeypatch, ["--dual-pass-tables"])
        assert config.dual_pass_tables is True

    def test_explicit_off_flag_still_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _run_capturing_config(tmp_path, monkeypatch, ["--no-dual-pass-tables"])
        assert config.dual_pass_tables is False

    def test_yaml_true_with_no_cli_override_is_preserved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GH-168 shape: an absent CLI flag must not silently override a value
        loaded from --config back to the new False default."""
        config = _run_capturing_config(
            tmp_path, monkeypatch, [], config_text="dual_pass_tables: true\n"
        )
        assert config.dual_pass_tables is True

    def test_yaml_true_overridden_by_explicit_no_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _run_capturing_config(
            tmp_path,
            monkeypatch,
            ["--no-dual-pass-tables"],
            config_text="dual_pass_tables: true\n",
        )
        assert config.dual_pass_tables is False

    def test_yaml_false_overridden_by_explicit_on_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _run_capturing_config(
            tmp_path,
            monkeypatch,
            ["--dual-pass-tables"],
            config_text="dual_pass_tables: false\n",
        )
        assert config.dual_pass_tables is True


class TestDualPassTablesCLIHelp:
    @pytest.mark.parametrize("command", ["process", "batch"])
    def test_help_mentions_the_paired_flag_and_default_off(self, command: str) -> None:
        result = CliRunner().invoke(cli, [command, "--help"])
        assert result.exit_code == 0
        help_text = result.output
        assert "--dual-pass-tables" in help_text
        assert "--no-dual-pass-tables" in help_text

    @pytest.mark.parametrize("command", ["process", "batch"])
    def test_auto_patch_tables_help_describes_the_signal_gate(self, command: str) -> None:
        """--auto-patch-tables can only ever fire inside an enabled,
        signal-triggered dual-pass reread now -- the help text must not
        describe it as acting on every table page."""
        result = CliRunner().invoke(cli, [command, "--help"])
        assert result.exit_code == 0
        help_text = result.output
        assert "--auto-patch-tables" in help_text
