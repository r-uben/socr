"""Round-3 canon remediation tests (ocr-output-contract v0.1.3 adoption).

Covers the round-3 blockers the socr orchestrator PR must close:

* HIGH (data loss) — ``socr batch DIR --limit N`` WITHOUT ``-o`` used to resolve
  the output root into the ephemeral ``TemporaryDirectory`` the limited PDFs were
  symlinked into, which is deleted on block exit (silently producing then
  destroying all OCR output). The fix resolves the REAL persistent output root
  from the ORIGINAL ``pdf_dir`` before entering the tmpdir, so output PERSISTS.

* HIGH — the resume-gate run fingerprint omitted output-affecting settings (e.g.
  ``save_figures``), so re-running a completed doc under a changed
  output-affecting flag was SKIPPED and stale output reused. The fingerprint now
  includes them, so a changed setting reprocesses.

* CRITICAL/HIGH (qwen + deepseek) — socr's ``process_pages`` read-back must
  consume a canon engine that aggregates a dir-of-page-images into ONE
  ``resolve_output_root(<images_dir>, -o)/<images_dir_stem>/<images_dir_stem>.md``
  with ``## Page N`` headers. socr locates it via the CONTRACT helpers and
  ``split_native_pages`` it, mapping section k -> the k-th filename-sorted input
  image, recovering ALL N pages (no EMPTY_OUTPUT) in input order.

Engine subprocess / model boundaries are mocked: no live engine is invoked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

fitz = pytest.importorskip("fitz")  # PyMuPDF; skip whole module if unavailable

from click.testing import CliRunner  # noqa: E402
from ocr_output_contract import (  # noqa: E402
    assemble_pages,
    doc_dir_for,
    markdown_path_for,
    relative_key,
    resolve_output_root,
)

from socr import cli as socr_cli  # noqa: E402
from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.result import (  # noqa: E402
    DocumentStatus,
    EngineResult,
    FailureMode,
    PageOutput,
    PageStatus,
)
from socr.engines.base import BaseEngine  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _config(**overrides) -> PipelineConfig:
    defaults = dict(
        primary_engine=EngineType.DEEPSEEK,  # configurable model (deepseek_*)
        quiet=True,
        audit_enabled=False,
        native_first=False,
        dual_pass_tables=False,
        judge_hard_pages=False,
        save_figures=False,
        write_manifest=False,
        # These tests pre-date the agentic default change and test the deterministic
        # backbone/audit/repair pipeline rather than agentic routing.
        agentic=False,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _real_pdf(path: Path, n_pages: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for i in range(n_pages):
        doc.new_page().insert_text((72, 72), f"page {i + 1}")
    doc.save(str(path))
    doc.close()
    return path


def _backbone_writing_pages(text_by_page):
    """Return a ``_phase_agentic`` replacement that populates per-page best_output."""

    def _fake(self, state, output_dir):
        for page_num, text in text_by_page.items():
            state.pages[page_num].best_output = PageOutput(
                page_num=page_num,
                text=text,
                status=PageStatus.SUCCESS,
                engine="deepseek",
                audit_passed=True,
            )
        # R174b: _phase_agentic returns None; the state IS the result.
        return None

    return _fake


class _StubEngine(BaseEngine):
    """A minimal CLI engine whose only purpose is exercising the read-back."""

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def cli_command(self) -> str:
        return "stub-ocr"

    def _build_command(self, pdf_path, output_dir, config):
        return ["stub-ocr", str(pdf_path), "-o", str(output_dir)]


# ---------------------------------------------------------------------------
# HIGH (data loss): --limit without -o writes to the REAL persistent root
# ---------------------------------------------------------------------------


class TestLimitOutputPersists:
    def test_limit_batch_without_o_persists_output(self, tmp_path, monkeypatch):
        """``socr batch DIR --limit N`` with NO ``-o`` writes the canonical output
        tree to the REAL ``<DIR>/ocr/`` root and it PERSISTS after the command
        returns — never into the ephemeral TemporaryDirectory the limited PDFs are
        symlinked into (the round-3 data-loss regression)."""
        in_dir = tmp_path / "papers"
        _real_pdf(in_dir / "a.pdf", n_pages=2)
        _real_pdf(in_dir / "b.pdf", n_pages=2)
        _real_pdf(in_dir / "c.pdf", n_pages=2)

        # AUTO must resolve to a configurable-model engine without probing CLIs.
        monkeypatch.setattr(
            "socr.engines.registry.resolve_auto_engine", lambda: EngineType.DEEPSEEK
        )

        with patch.object(
            UnifiedPipeline,
            "_phase_agentic",
            _backbone_writing_pages({1: "limited one", 2: "limited two"}),
        ):
            result = CliRunner().invoke(
                socr_cli.cli,
                # R174b: agentic is the only lane; _phase_agentic is patched above
                # for predictable output.
                ["batch", str(in_dir), "--limit", "2", "--primary", "deepseek"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output

        # The persistent canon root is <in_dir>/ocr/ (resolve_output_root for a dir
        # input). The tmpdir the limited PDFs were symlinked into is gone, but the
        # real output must remain on disk.
        out_root = resolve_output_root(in_dir)
        assert out_root == in_dir / "ocr"
        assert out_root.is_dir(), "persistent output root was destroyed (data loss)"

        # --limit 2 -> exactly the first two PDFs (filename-sorted) are processed,
        # each with a populated canonical .md that SURVIVES the command.
        for stem in ("a", "b"):
            md = markdown_path_for(doc_dir_for(out_root, f"{stem}.pdf"), f"{stem}.pdf")
            assert md.exists(), f"{md} missing — output did not persist"
            body = md.read_text(encoding="utf-8")
            assert "## Page 1" in body and "## Page 2" in body
            assert "limited one" in body and "limited two" in body

        # The third PDF was excluded by --limit, so no output for it.
        assert not markdown_path_for(doc_dir_for(out_root, "c.pdf"), "c.pdf").exists()

        # Root index sidecar persisted too.
        assert (out_root / "metadata.json").exists()


# ---------------------------------------------------------------------------
# HIGH: resume fingerprint includes output-affecting settings
# ---------------------------------------------------------------------------


class TestFingerprintCoversOutputAffectingFlags:
    def _run_once(self, pipeline, pdf, out):
        with patch.object(
            UnifiedPipeline,
            "_phase_agentic",
            _backbone_writing_pages({1: "content one", 2: "content two"}),
        ):
            return pipeline.process(pdf, out)

    def test_save_figures_toggle_reprocesses_not_skipped(self, tmp_path):
        """Re-running a COMPLETED doc with ``save_figures`` flipped changes the
        saved markdown (figure embedding), so the resume gate must NOT skip it —
        the fingerprint includes ``save_figures``."""
        pdf = _real_pdf(tmp_path / "src" / "paper.pdf", n_pages=2)
        out = tmp_path / "out"

        first = self._run_once(UnifiedPipeline(_config(save_figures=False)), pdf, out)
        assert first.status == DocumentStatus.SUCCESS

        # Same config -> resume gate SKIPS (sanity: the gate works at all).
        with patch.object(UnifiedPipeline, "_phase_agentic") as backbone:
            skipped = UnifiedPipeline(_config(save_figures=False)).process(pdf, out)
        assert skipped.status == DocumentStatus.SKIPPED
        backbone.assert_not_called()

        # save_figures flipped -> fingerprint changes -> reprocess (backbone runs).
        ran = self._run_once(UnifiedPipeline(_config(save_figures=True)), pdf, out)
        assert ran.status == DocumentStatus.SUCCESS

    def test_fingerprint_differs_across_output_affecting_flags(self, tmp_path):
        """Each output-affecting flag the round-3 fix added must change the run
        fingerprint when toggled (so a changed setting can never silently reuse a
        cached result)."""
        base = UnifiedPipeline(_config())
        base_fp = base._run_fingerprint()

        toggles = [
            dict(save_figures=True),
            dict(figures_max_total=99),
            dict(figures_max_per_page=9),
            dict(local_engine=EngineType.GEMINI),
            dict(fallback_chain=[EngineType.MISTRAL]),
            dict(tiered=False),
            dict(judge_backend="vlm"),
            dict(judge_model="qwen2-vl:7b"),
        ]
        for override in toggles:
            fp = UnifiedPipeline(_config(**override))._run_fingerprint()
            assert fp != base_fp, f"fingerprint did not change for {override}"


# ---------------------------------------------------------------------------
# CRITICAL/HIGH: aggregated dir-of-images read-back recovers ALL pages
# ---------------------------------------------------------------------------


def _aggregating_engine_subprocess(page_texts):
    """A fake subprocess that aggregates the rendered page-image dir into ONE
    canonical ``## Page N`` doc — exactly the shape qwen/deepseek emit for a pure
    image-directory input.

    The aggregate is written at the canon path the engines actually compute:
    ``resolve_output_root(images_dir, -o)/<images_dir_stem>/<images_dir_stem>.md``,
    where the engine's scan_root for a single image-dir document is
    ``images_dir.parent`` (so its rel_key is the dir name). ``page_texts`` is the
    ordered list of per-page bodies the engine produces (one per rendered image,
    filename-sorted).
    """

    def _run(cmd, *args, **kwargs):
        images_dir = Path(cmd[1])
        cli_out = Path(cmd[cmd.index("-o") + 1])
        # Mirror the engine's own resolution exactly.
        output_root = resolve_output_root(images_dir, cli_out)
        rel_key = relative_key(images_dir, images_dir.parent)
        doc_dir = doc_dir_for(output_root, rel_key)
        doc_dir.mkdir(parents=True, exist_ok=True)
        md_path = markdown_path_for(doc_dir, rel_key)
        md_path.write_text(assemble_pages(page_texts), encoding="utf-8")
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    return _run


class TestFailureRecordConformance:
    def test_failed_doc_uses_contract_failure_checksum(self, tmp_path):
        """A FAILED doc records a contract-VALID ``sha256:`` checksum (via the
        round-3 ``failure_checksum`` adoption), not the old non-conformant ``""``,
        so the contract's own ``assert_conforms`` harness passes on the failure
        path — not just the happy path (round-3 conformance-theater fix)."""
        from ocr_output_contract.conformance import ExpectedDoc, assert_conforms

        in_dir = tmp_path / "in"
        _real_pdf(in_dir / "bad.pdf", n_pages=2)
        _real_pdf(in_dir / "ok.pdf", n_pages=2)
        out = tmp_path / "out"

        def _selective_backbone(self, state, output_dir):
            # "bad.pdf" produces no text (failure); "ok.pdf" succeeds.
            if state.handle.path.stem == "ok":
                for n, t in ({1: "ok one", 2: "ok two"}).items():
                    state.pages[n].best_output = PageOutput(
                        page_num=n,
                        text=t,
                        status=PageStatus.SUCCESS,
                        engine="deepseek",
                        audit_passed=True,
                    )
                return None
            # R174b: _phase_agentic returns None; leaving every page without a
            # best_output is what makes the document fail.
            return None

        pipeline = UnifiedPipeline(_config())
        with patch.object(UnifiedPipeline, "_phase_agentic", _selective_backbone):
            pipeline.process_batch(in_dir, out)

        # The FAILED doc's recorded checksum is a valid sha256: sentinel/digest,
        # and the harness accepts the failure record (it would reject "" or null).
        import json

        entry = json.loads((out / "metadata.json").read_text(encoding="utf-8"))["files"]["bad.pdf"]
        assert entry["status"] == "failed"
        assert entry["checksum"].startswith("sha256:")

        assert_conforms(
            out,
            [
                ExpectedDoc(rel_key="bad.pdf", pages=2, status="failed"),
                ExpectedDoc(rel_key="ok.pdf", pages=2),
            ],
            require_failures_nonzero_exit=pipeline.last_outcome.exit_code != 0,
        )


class TestAggregatedReadbackRecoversAllPages:
    def test_realistic_aggregate_recovers_all_n_pages_in_order(self, tmp_path):
        """A realistic aggregated engine output (one '## Page N' doc named after
        the images dir) is split back into per-page text, recovering ALL N pages
        with no EMPTY_OUTPUT, in input page order."""
        pdf = _real_pdf(tmp_path / "doc.pdf", n_pages=4)
        engine = _StubEngine()
        config = _config(timeout=30)

        page_texts = [
            "aggregate body for page 1",
            "aggregate body for page 2",
            "aggregate body for page 3",
            "aggregate body for page 4",
        ]
        fake = _aggregating_engine_subprocess(page_texts)
        with patch("socr.engines.base.subprocess.run", side_effect=fake):
            outputs = engine.process_pages(pdf, [1, 2, 3, 4], config, dpi=72)

        assert len(outputs) == 4
        for po in outputs:
            assert po.status == PageStatus.SUCCESS, (po.page_num, po.failure_mode, po.error)
            assert po.failure_mode != FailureMode.EMPTY_OUTPUT
            assert po.text.strip()

        by_page = {po.page_num: po.text for po in outputs}
        for n in (1, 2, 3, 4):
            assert f"aggregate body for page {n}" in by_page[n]

    def test_aggregate_located_at_images_dir_stem_path(self, tmp_path):
        """The aggregate is found at the path keyed on the IMAGES DIR stem
        (``page_imgs/page_imgs.md``) — pinning the canon location the engine
        writes, so a future refactor that mislocates it (e.g. back to a per-page
        ``page_NNNN`` path) fails this test."""
        pdf = _real_pdf(tmp_path / "doc.pdf", n_pages=2)
        engine = _StubEngine()
        config = _config(timeout=30)

        captured: dict[str, Path] = {}

        def _run(cmd, *args, **kwargs):
            images_dir = Path(cmd[1])
            cli_out = Path(cmd[cmd.index("-o") + 1])
            output_root = resolve_output_root(images_dir, cli_out)
            rel_key = relative_key(images_dir, images_dir.parent)
            md_path = markdown_path_for(doc_dir_for(output_root, rel_key), rel_key)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(assemble_pages(["one", "two"]), encoding="utf-8")
            # The images dir socr renders into is named "images"; its aggregate
            # therefore lands at <root>/images/images.md.
            captured["md"] = md_path
            captured["stem"] = images_dir.name
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("socr.engines.base.subprocess.run", side_effect=_run):
            outputs = engine.process_pages(pdf, [1, 2], config, dpi=72)

        # The aggregate filename matches the images-dir stem (NOT a page_NNNN stem).
        assert captured["md"].name == f"{captured['stem']}.md"
        assert captured["md"].parent.name == captured["stem"]
        assert all(po.status == PageStatus.SUCCESS for po in outputs)
