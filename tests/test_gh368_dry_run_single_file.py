"""GH-368: ``--dry-run`` must be honoured on the single-file process path.

``--dry-run`` was consulted only inside ``process_batch``. The single-file path
never checked it and ran the full real pipeline, which is how a supposedly-dry
test OCR'd a PDF for ~56 s locally and then failed in CI.

Silently ignoring a flag the user typed is the no-silent-failure rule this repo
holds everywhere else: the run either does what was asked or says it cannot.

Pinned as a DIFFERENCE: the same invocation, the same file, flipping only
``--dry-run``, asserting whether the pipeline was entered at all. Nothing here
touches a model or a provider.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")


def _pdf(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "One page of prose, enough to be a real file.", fontsize=11)
    path = tmp_path / "doc.pdf"
    doc.save(path)
    doc.close()
    return path


def _invoke(pdf_path: Path, out_dir: Path, *, dry_run: bool, quiet: bool = True):
    """Drive the REAL CLI, recording whether the pipeline was ever entered.

    Invoked through ``CliRunner`` rather than by calling ``cli.process``: it is a
    Typer command, so a direct call goes to Typer's ``Context``, not the body --
    a test that did that would exercise nothing.
    """
    from click.testing import CliRunner

    from socr.cli import cli

    calls: list[tuple] = []

    class _FakePipeline:
        def __init__(self, config) -> None:
            self.config = config

        def process(self, pdf, output_dir, **kwargs):
            calls.append((pdf, output_dir))
            raise RuntimeError("stop here: reaching the pipeline is what we measure")

    args = ["process", str(pdf_path), "-o", str(out_dir), "--primary", "qwen"]
    if quiet:
        args.append("--quiet")
    if dry_run:
        args.append("--dry-run")

    with patch("socr.pipeline.orchestrator.UnifiedPipeline", _FakePipeline):
        result = CliRunner().invoke(cli, args)
    return calls, result


class TestDryRunIsHonouredOnTheSingleFilePath:
    def test_dry_run_does_not_enter_the_pipeline(self, tmp_path: Path) -> None:
        pdf = _pdf(tmp_path / "src")
        calls, _result = _invoke(pdf, tmp_path / "out", dry_run=True)
        assert calls == [], "--dry-run must not run the pipeline"

    def test_without_dry_run_the_pipeline_is_entered(self, tmp_path: Path) -> None:
        """Control. Without this, a change that broke the path entirely would
        satisfy the test above while doing nothing the user asked for."""
        pdf = _pdf(tmp_path / "src")
        calls, _result = _invoke(pdf, tmp_path / "out", dry_run=False)
        assert calls, "without --dry-run the pipeline must actually run"

    def test_dry_run_writes_no_output_directory_contents(self, tmp_path: Path) -> None:
        pdf = _pdf(tmp_path / "src")
        out = tmp_path / "out"
        _invoke(pdf, out, dry_run=True)
        assert not out.exists() or not any(out.rglob("*.md"))

    def test_dry_run_says_what_it_would_have_done(self, tmp_path: Path) -> None:
        """Honouring the flag silently would be its own small failure -- the
        batch path prints what it would process, and so must this one."""
        pdf = _pdf(tmp_path / "src")
        _calls, result = _invoke(pdf, tmp_path / "out", dry_run=True, quiet=False)
        assert "Would process" in result.output
        assert pdf.name in result.output

    def test_dry_run_without_o_names_the_real_destination(self, tmp_path: Path) -> None:
        """GH-401 review: omitting -o must not print "Output: None".

        A real run writes to the configured default, so a preview that names
        None describes a run that never happens.
        """
        from click.testing import CliRunner

        from socr.cli import cli

        pdf = _pdf(tmp_path / "src")
        result = CliRunner().invoke(cli, ["process", str(pdf), "--dry-run", "--primary", "qwen"])

        # Assert against the REAL destination, not merely the absence of
        # "None". The first version of this test only checked that "None" was
        # gone, so it passed while the message still named the wrong directory.
        from socr.cli import build_config
        from socr.pipeline.orchestrator import UnifiedPipeline

        expected = UnifiedPipeline(build_config(output_dir=None))._resolve_output_root(pdf, None)
        assert f"Output: {expected}" in result.output, (
            f"preview named the wrong destination; expected {expected}"
        )
        assert "Output: None" not in result.output
        assert "Would process" in result.output

    def test_a_directory_input_previews_the_same_root_the_run_would_use(
        self, tmp_path: Path
    ) -> None:
        """GH-401 review, P2. A FILE and its parent resolve to the same root, so
        the test above cannot tell ``pdf_path`` from ``pdf_path.parent``. A
        DIRECTORY can: the real call passes the path itself (orchestrator.py:643)
        and resolves ``<dir>/ocr``, while ``.parent`` would resolve
        ``<parent>/ocr``. This is the case that distinguishes them.
        """
        from click.testing import CliRunner

        from socr.cli import build_config, cli
        from socr.pipeline.orchestrator import UnifiedPipeline

        target = tmp_path / "corpus"
        target.mkdir()

        result = CliRunner().invoke(cli, ["process", str(target), "--dry-run", "--primary", "qwen"])
        expected = UnifiedPipeline(build_config(output_dir=None))._resolve_output_root(target, None)
        assert f"Output: {expected}" in result.output
        assert str(tmp_path / "ocr") not in result.output, (
            "resolved from the parent, not the path the real run passes"
        )
