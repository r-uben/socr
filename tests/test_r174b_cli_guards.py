"""R174b acceptance tests: CLI option deletion guards (Tasks t3, t6, t14).

Verifies via Click's CliRunner:
- --legacy-routing is absent on process or batch (fails at option parsing).
- --multi-engine is absent on process or batch.
- --consensus-llm is absent on process.
- --help for process and batch omits all three flags.
- --agentic survives as a backward-compatibility flag.
- Negative tests do not patch MagicMock or reach provider code.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from socr.cli import cli


@pytest.fixture
def dummy_pdf(tmp_path: Path) -> Path:
    """Create a minimal 1-page PDF for CLI option parsing tests."""
    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "dummy.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Sample text for CLI option parsing guard.")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


class TestCLINonexistenceGuards:
    """Option parsing rejection guards for deleted CLI flags."""

    @pytest.mark.parametrize(
        "flag_args",
        [
            ["--legacy-routing"],
            ["--multi-engine", "gemini,mistral"],
            ["--consensus-llm", "qwen3.5:cloud"],
        ],
    )
    def test_process_rejects_deleted_flags(self, dummy_pdf: Path, flag_args: list[str]):
        """socr process must reject deleted flags with Click exit_code=2 (unknown option)."""
        runner = CliRunner()
        cmd = ["process", str(dummy_pdf)] + flag_args
        result = runner.invoke(cli, cmd)
        msg = f"Expected exit_code=2 for {flag_args}, got {result.exit_code}: {result.output}"
        assert result.exit_code == 2, msg
        out_lower = result.output.lower()
        assert "no such option" in out_lower or "unrecognized option" in out_lower, (
            f"Output did not indicate unknown option: {result.output}"
        )

    @pytest.mark.parametrize(
        "flag_args",
        [
            ["--legacy-routing"],
            ["--multi-engine", "gemini,mistral"],
        ],
    )
    def test_batch_rejects_deleted_flags(self, tmp_path: Path, flag_args: list[str]):
        """socr batch must reject deleted flags with Click exit_code=2 (unknown option)."""
        runner = CliRunner()
        input_dir = tmp_path / "batch_in"
        input_dir.mkdir()
        cmd = ["batch", str(input_dir)] + flag_args
        result = runner.invoke(cli, cmd)
        msg = f"Expected option parse failure for batch {flag_args}: {result.output}"
        assert result.exit_code == 2, msg
        out_lower = result.output.lower()
        assert "no such option" in out_lower or "unrecognized option" in out_lower, (
            f"Output did not indicate unknown option: {result.output}"
        )

    def test_process_help_omits_deleted_flags(self):
        """socr process --help must not mention deleted options or multi-engine examples."""
        runner = CliRunner()
        result = runner.invoke(cli, ["process", "--help"])
        assert result.exit_code == 0
        help_text = result.output
        assert "--legacy-routing" not in help_text
        assert "--multi-engine" not in help_text
        assert "--consensus-llm" not in help_text

    def test_batch_help_omits_deleted_flags(self):
        """socr batch --help must not mention deleted options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["batch", "--help"])
        assert result.exit_code == 0
        help_text = result.output
        assert "--legacy-routing" not in help_text
        assert "--multi-engine" not in help_text
        assert "--consensus-llm" not in help_text

    def test_agentic_flag_accepted_as_compat_noop(self, dummy_pdf: Path):
        """--agentic must not fail option parsing."""
        runner = CliRunner()
        result = runner.invoke(cli, ["process", str(dummy_pdf), "--agentic", "--dry-run"])
        assert result.exit_code != 2, f"--agentic failed option parsing: {result.output}"
