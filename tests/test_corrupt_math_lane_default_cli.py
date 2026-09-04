"""Owner ruling 2026-09-04 (docs/log/2026-09-04_corrupt-math-lane-default-on.md):
the CLI surface for ``recover_corrupt_math`` after its default flipped to True.

Before this ticket ``--recover-corrupt-math`` was a one-way ``is_flag`` option
(``build_config(recover_corrupt_math: bool = False)``, unconditional
``if recover_corrupt_math: config.recover_corrupt_math = True``), so it could
only ever turn the lane ON and could never restore the pre-flip behaviour from
the CLI. Now the default is True, so there must be a paired
``--recover-corrupt-math`` / ``--no-recover-corrupt-math`` option, and its
ABSENCE must not clobber a value loaded from ``--config``/``--profile`` (the
GH-168 "flag that lies" failure mode: test_gh168_config_precedence.py) --
mirroring test_p5_dual_pass_cli.py for ``dual_pass_tables``.

Driven through the real Click command for the same reason GH-168's tests are:
only Click can distinguish "the user typed nothing" from "the user typed the
default value" (``ParameterSource``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from socr.cli import cli
from socr.core.config import PipelineConfig


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
    command: str = "process",
):
    pdf = _make_pdf(tmp_path)
    seen: dict = {}

    class _Stub:
        def __init__(self, config):
            seen["config"] = config

        def process(self, *_a, **_k):
            raise SystemExit(0)

        def process_batch(self, *_a, **_k):
            raise SystemExit(0)

        def _resolve_output_root(self, *_a, **_k):
            return tmp_path / "out"

    monkeypatch.setattr("socr.pipeline.orchestrator.UnifiedPipeline", _Stub)

    if command == "batch":
        cmd = ["batch", str(tmp_path)]
    else:
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


def _run_capturing_config_via_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str],
    profile_config_text: str,
    profile_name: str = "testprofile",
):
    """Same as ``_run_capturing_config``, but load via ``--profile`` instead of
    ``--config``.  ``PipelineConfig.load`` resolves a profile from
    ``~/.config/socr/{profile}.yaml`` (``src/socr/core/config.py``), so
    ``Path.home`` is redirected to ``tmp_path`` for the duration of the test."""
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
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profile_dir = tmp_path / ".config" / "socr"
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / f"{profile_name}.yaml").write_text(profile_config_text)

    cmd = ["process", str(pdf), "--profile", profile_name, *extra]
    result = CliRunner().invoke(cli, cmd)
    assert "config" in seen, (
        f"the pipeline was never constructed, so nothing was measured: "
        f"exit_code={result.exit_code}, output={result.output}"
    )
    return seen["config"]


class TestPipelineConfigDefault:
    def test_recover_corrupt_math_defaults_to_on(self) -> None:
        """Pins the flip itself: a bare PipelineConfig() must have the lane on.

        If a future change restores the pre-2026-09-04 default, this test --
        not just the CLI tests below -- must fail.
        """
        assert PipelineConfig().recover_corrupt_math is True


class TestRecoverCorruptMathCLIDefaultAndOverrides:
    def test_unconfigured_cli_run_uses_the_default_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _run_capturing_config(tmp_path, monkeypatch, [])
        assert config.recover_corrupt_math is True

    def test_explicit_on_flag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _run_capturing_config(tmp_path, monkeypatch, ["--recover-corrupt-math"])
        assert config.recover_corrupt_math is True

    def test_explicit_off_flag_is_the_kill_switch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _run_capturing_config(tmp_path, monkeypatch, ["--no-recover-corrupt-math"])
        assert config.recover_corrupt_math is False

    def test_yaml_false_with_no_cli_override_is_preserved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GH-168 shape: an absent CLI flag must not silently override a value
        loaded from --config back to the new True default."""
        config = _run_capturing_config(
            tmp_path, monkeypatch, [], config_text="recover_corrupt_math: false\n"
        )
        assert config.recover_corrupt_math is False

    def test_yaml_true_with_no_cli_override_is_preserved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _run_capturing_config(
            tmp_path, monkeypatch, [], config_text="recover_corrupt_math: true\n"
        )
        assert config.recover_corrupt_math is True

    def test_yaml_true_overridden_by_explicit_no_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _run_capturing_config(
            tmp_path,
            monkeypatch,
            ["--no-recover-corrupt-math"],
            config_text="recover_corrupt_math: true\n",
        )
        assert config.recover_corrupt_math is False

    def test_yaml_false_overridden_by_explicit_on_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _run_capturing_config(
            tmp_path,
            monkeypatch,
            ["--recover-corrupt-math"],
            config_text="recover_corrupt_math: false\n",
        )
        assert config.recover_corrupt_math is True


class TestRecoverCorruptMathBatchCLIDefaultAndOverrides:
    """Same precedence rules, driven through ``batch`` instead of ``process``."""

    def test_unconfigured_batch_run_uses_the_default_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _run_capturing_config(tmp_path, monkeypatch, [], command="batch")
        assert config.recover_corrupt_math is True

    def test_batch_explicit_off_flag_is_the_kill_switch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _run_capturing_config(
            tmp_path, monkeypatch, ["--no-recover-corrupt-math"], command="batch"
        )
        assert config.recover_corrupt_math is False

    def test_batch_explicit_on_flag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _run_capturing_config(
            tmp_path, monkeypatch, ["--recover-corrupt-math"], command="batch"
        )
        assert config.recover_corrupt_math is True

    def test_batch_yaml_false_with_no_cli_override_is_preserved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _run_capturing_config(
            tmp_path,
            monkeypatch,
            [],
            config_text="recover_corrupt_math: false\n",
            command="batch",
        )
        assert config.recover_corrupt_math is False

    def test_batch_yaml_true_with_no_cli_override_is_preserved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _run_capturing_config(
            tmp_path,
            monkeypatch,
            [],
            config_text="recover_corrupt_math: true\n",
            command="batch",
        )
        assert config.recover_corrupt_math is True

    def test_batch_yaml_true_overridden_by_explicit_no_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _run_capturing_config(
            tmp_path,
            monkeypatch,
            ["--no-recover-corrupt-math"],
            config_text="recover_corrupt_math: true\n",
            command="batch",
        )
        assert config.recover_corrupt_math is False

    def test_batch_yaml_false_overridden_by_explicit_on_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _run_capturing_config(
            tmp_path,
            monkeypatch,
            ["--recover-corrupt-math"],
            config_text="recover_corrupt_math: false\n",
            command="batch",
        )
        assert config.recover_corrupt_math is True


class TestRecoverCorruptMathProfilePrecedence:
    """Same GH-168 precedence rules, driven through ``--profile`` instead of
    ``--config`` -- the docstring above claims both, so both are covered."""

    def test_profile_false_with_no_cli_override_is_preserved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _run_capturing_config_via_profile(
            tmp_path, monkeypatch, [], "recover_corrupt_math: false\n"
        )
        assert config.recover_corrupt_math is False

    def test_profile_true_with_no_cli_override_is_preserved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _run_capturing_config_via_profile(
            tmp_path, monkeypatch, [], "recover_corrupt_math: true\n"
        )
        assert config.recover_corrupt_math is True

    def test_profile_true_overridden_by_explicit_no_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _run_capturing_config_via_profile(
            tmp_path,
            monkeypatch,
            ["--no-recover-corrupt-math"],
            "recover_corrupt_math: true\n",
        )
        assert config.recover_corrupt_math is False

    def test_profile_false_overridden_by_explicit_on_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _run_capturing_config_via_profile(
            tmp_path,
            monkeypatch,
            ["--recover-corrupt-math"],
            "recover_corrupt_math: false\n",
        )
        assert config.recover_corrupt_math is True


class TestRecoverCorruptMathCLIHelp:
    @pytest.mark.parametrize("command", ["process", "batch"])
    def test_help_mentions_the_paired_flag_and_default_on(self, command: str) -> None:
        result = CliRunner().invoke(cli, [command, "--help"])
        assert result.exit_code == 0
        help_text = result.output
        assert "--recover-corrupt-math" in help_text
        assert "--no-recover-corrupt-math" in help_text
        assert "Default: on" in help_text
