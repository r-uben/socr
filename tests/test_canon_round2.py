"""Round-2 canon remediation tests (ocr-output-contract v0.1.2 adoption).

Covers the four round-2 blockers the socr orchestrator PR must close:

* HIGH — batch metadata clobber: the canonical contract ``RootIndex`` (+ per-doc
  ``write_doc_metadata``) is the SINGLE authoritative writer of
  ``<root>/metadata.json`` in BATCH mode; the legacy basename-keyed
  ``MetadataManager`` no longer overwrites it.
* HIGH — fingerprint absent: ``run_fingerprint(...)`` is written into the
  per-doc + root ``DocMetadata`` and consulted by ``RootIndex.is_completed`` so a
  re-run under a different model / task / output flag reprocesses (single-file
  AND batch).
* HIGH — status fabrication: a failed/truncated whole-doc attempt is NOT frozen
  as ``status=completed`` in the manifest/root index; the real status is carried.
* CRITICAL — qwen<->socr: socr's ``process_pages`` read-back consumes a canon
  engine that emits an AGGREGATED ``<stem>/<stem>.md`` with ``## Page N`` headers
  (split back to per-page via ``split_native_pages``), so qwen-style output no
  longer degrades every page to ``EMPTY_OUTPUT``.

Plus a >=2-page BATCH ``assert_conforms`` conformance test.

Engine subprocess / model boundaries are mocked: no live engine is invoked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

fitz = pytest.importorskip("fitz")  # PyMuPDF; skip whole module if unavailable

from ocr_output_contract import (  # noqa: E402
    assemble_pages,
    doc_dir_for,
    markdown_path_for,
    relative_key,
)

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.result import (  # noqa: E402
    DocumentStatus,
    EngineResult,
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
    """Return a ``_phase_backbone`` replacement that populates per-page best_output.

    Mirrors what a real CLI backbone run leaves on the state: each page gets a
    passing ``best_output``. ``text_by_page`` maps 1-based page_num -> text.
    """

    def _fake(self, state, output_dir):
        for page_num, text in text_by_page.items():
            state.pages[page_num].best_output = PageOutput(
                page_num=page_num,
                text=text,
                status=PageStatus.SUCCESS,
                engine="deepseek",
                audit_passed=True,
            )
        return EngineResult(
            document_path=state.handle.path,
            engine="deepseek",
            status=DocumentStatus.SUCCESS,
            pages_processed=state.handle.page_count,
        )

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


def _root_index(out_root: Path) -> dict:
    return json.loads((out_root / "metadata.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# HIGH 1: batch metadata is canonical (no legacy clobber)
# ---------------------------------------------------------------------------


class TestBatchMetadataAuthoritative:
    def test_batch_root_index_is_contract_schema_not_legacy(self, tmp_path):
        """After a 2-file batch the root metadata.json is the CONTRACT schema
        (model/backend/fingerprint, input-relative keys), never the legacy
        basename-keyed/TZ-naive schema the old MetadataManager wrote."""
        in_dir = tmp_path / "in"
        _real_pdf(in_dir / "a.pdf", n_pages=2)
        _real_pdf(in_dir / "b.pdf", n_pages=2)
        out = tmp_path / "out"

        pipeline = UnifiedPipeline(_config())
        with patch.object(
            UnifiedPipeline,
            "_phase_backbone",
            _backbone_writing_pages({1: "alpha one", 2: "alpha two"}),
        ):
            pipeline.process_batch(in_dir, out)

        index = _root_index(out)
        # Keyed by input-relative path (basenames here, but contract-keyed).
        assert set(index["files"]) == {"a.pdf", "b.pdf"}
        for key, entry in index["files"].items():
            # Contract schema fields the legacy writer NEVER produced.
            assert entry["status"] == "completed"
            assert entry["backend"] == "socr"
            assert entry["model"]  # engine recorded, not absent
            assert entry["fingerprint"].startswith("fp:")
            # UTC-aware timestamp (contract uses utc_timestamp -> +00:00 offset),
            # never the legacy TZ-naive datetime.now().isoformat().
            assert entry["timestamp"].endswith("+00:00")
            # output_path points at the canonical per-doc .md.
            assert entry["output_path"].endswith(f"{Path(key).stem}.md")

    def test_per_doc_sidecar_written_for_each_batch_member(self, tmp_path):
        in_dir = tmp_path / "in"
        _real_pdf(in_dir / "a.pdf", n_pages=2)
        _real_pdf(in_dir / "b.pdf", n_pages=2)
        out = tmp_path / "out"

        pipeline = UnifiedPipeline(_config())
        with patch.object(
            UnifiedPipeline,
            "_phase_backbone",
            _backbone_writing_pages({1: "x", 2: "y"}),
        ):
            pipeline.process_batch(in_dir, out)

        for name in ("a", "b"):
            rel_key = f"{name}.pdf"
            doc_dir = doc_dir_for(out, rel_key)
            sidecar = json.loads((doc_dir / "metadata.json").read_text(encoding="utf-8"))
            assert sidecar["key"] == rel_key
            assert sidecar["fingerprint"].startswith("fp:")
            assert sidecar["backend"] == "socr"


# ---------------------------------------------------------------------------
# HIGH 2: fingerprint written + consulted (single-file AND batch)
# ---------------------------------------------------------------------------


class TestFingerprintWrittenAndConsulted:
    def _run_once(self, pipeline, pdf, out):
        with patch.object(
            UnifiedPipeline,
            "_phase_backbone",
            _backbone_writing_pages({1: "content one", 2: "content two"}),
        ):
            return pipeline.process(pdf, out)

    def test_single_file_writes_fingerprint(self, tmp_path):
        pdf = _real_pdf(tmp_path / "src" / "paper.pdf", n_pages=2)
        out = tmp_path / "out"
        pipeline = UnifiedPipeline(_config())
        self._run_once(pipeline, pdf, out)

        entry = _root_index(out)["files"]["paper.pdf"]
        assert entry["fingerprint"].startswith("fp:")

    def test_resume_skips_only_when_fingerprint_matches(self, tmp_path):
        """A second run with the SAME config is skipped; a re-run after a model
        swap reprocesses (fingerprint mismatch), not silently reused."""
        pdf = _real_pdf(tmp_path / "src" / "paper.pdf", n_pages=2)
        out = tmp_path / "out"

        pipeline = UnifiedPipeline(_config())
        first = self._run_once(pipeline, pdf, out)
        assert first.status == DocumentStatus.SUCCESS

        # Same config -> resume gate skips (no backbone call needed).
        pipeline2 = UnifiedPipeline(_config())
        with patch.object(UnifiedPipeline, "_phase_backbone") as backbone:
            skipped = pipeline2.process(pdf, out)
        assert skipped.status == DocumentStatus.SKIPPED
        backbone.assert_not_called()

        # Different task -> fingerprint changes -> reprocess (backbone runs).
        pipeline3 = UnifiedPipeline(_config(deepseek_task="layout"))
        ran = self._run_once(pipeline3, pdf, out)
        assert ran.status == DocumentStatus.SUCCESS

    def test_batch_gate_reprocesses_on_model_swap(self, tmp_path):
        """The BATCH resume gate is fingerprint-aware: a task swap forces every
        file to reprocess instead of being skipped on input-checksum alone."""
        in_dir = tmp_path / "in"
        _real_pdf(in_dir / "a.pdf", n_pages=1)
        out = tmp_path / "out"

        with patch.object(
            UnifiedPipeline,
            "_phase_backbone",
            _backbone_writing_pages({1: "first model text"}),
        ):
            UnifiedPipeline(_config()).process_batch(in_dir, out)

        # Re-run with a different task: the gate must NOT skip -> process() runs.
        ran = {"count": 0}

        real_process = UnifiedPipeline.process

        def _counting_process(self, pdf, output_dir=None, scan_root=None):
            ran["count"] += 1
            return real_process(self, pdf, output_dir, scan_root)

        with (
            patch.object(
                UnifiedPipeline,
                "_phase_backbone",
                _backbone_writing_pages({1: "second model text"}),
            ),
            patch.object(UnifiedPipeline, "process", _counting_process),
        ):
            UnifiedPipeline(_config(deepseek_task="layout")).process_batch(in_dir, out)
        assert ran["count"] == 1  # reprocessed, not skipped

    def test_batch_gate_skips_when_unchanged(self, tmp_path):
        in_dir = tmp_path / "in"
        _real_pdf(in_dir / "a.pdf", n_pages=1)
        out = tmp_path / "out"

        with patch.object(
            UnifiedPipeline,
            "_phase_backbone",
            _backbone_writing_pages({1: "text"}),
        ):
            UnifiedPipeline(_config()).process_batch(in_dir, out)

        # Identical config -> skipped at the gate -> process() never invoked.
        with patch.object(UnifiedPipeline, "process") as proc:
            res = UnifiedPipeline(_config()).process_batch(in_dir, out)
        proc.assert_not_called()
        assert res == []


# ---------------------------------------------------------------------------
# HIGH 3: failed whole-doc attempt is NOT frozen as success in the batch index
# ---------------------------------------------------------------------------


class TestStatusNotFabricated:
    def test_failed_doc_records_failed_not_completed(self, tmp_path):
        """A doc whose backbone produced NO usable text records status=failed in
        the root index (and is never skipped on a later run), not completed."""
        in_dir = tmp_path / "in"
        _real_pdf(in_dir / "bad.pdf", n_pages=2)
        out = tmp_path / "out"

        def _empty_backbone(self, state, output_dir):
            # No best_output set on any page -> document has no text.
            return EngineResult(
                document_path=state.handle.path,
                engine="deepseek",
                status=DocumentStatus.ERROR,
                error="all pages failed",
            )

        pipeline = UnifiedPipeline(_config())
        with patch.object(UnifiedPipeline, "_phase_backbone", _empty_backbone):
            pipeline.process_batch(in_dir, out)

        entry = _root_index(out)["files"]["bad.pdf"]
        assert entry["status"] == "failed"
        assert entry["status"] != "completed"


# ---------------------------------------------------------------------------
# CRITICAL: qwen<->socr aggregated process_pages read-back
# ---------------------------------------------------------------------------


def _qwen_like_subprocess(page_text_by_index, *, count_override=None):
    """A fake subprocess that aggregates a dir-of-images into ONE canonical
    '## Page N' doc (qwen's shape), instead of one .md per input image.

    ``page_text_by_index`` maps the 1-based section index to its text. The
    aggregate is written to the contract's canonical path for the IMAGES dir as a
    single document: <out>/images/images.md. ``count_override`` lets a test write
    a deliberately wrong number of sections to exercise the mismatch guard.
    """

    def _run(cmd, *args, **kwargs):
        images_dir = Path(cmd[1])
        out_dir = Path(cmd[cmd.index("-o") + 1])
        n_imgs = len(list(images_dir.glob("*.png")))
        n = count_override if count_override is not None else n_imgs
        pages = [page_text_by_index.get(i + 1, f"section {i + 1}") for i in range(n)]
        rel_key = relative_key(images_dir, images_dir.parent)
        doc_dir = doc_dir_for(out_dir, rel_key)
        doc_dir.mkdir(parents=True, exist_ok=True)
        md_path = markdown_path_for(doc_dir, rel_key)
        md_path.write_text(assemble_pages(pages), encoding="utf-8")
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    return _run


class TestQwenAggregatedReadback:
    def test_aggregated_output_recovers_all_pages(self, tmp_path):
        """An engine that aggregates the image dir into one '## Page N' doc still
        yields one populated PageOutput per requested page (no EMPTY_OUTPUT)."""
        pdf = _real_pdf(tmp_path / "doc.pdf", n_pages=3)
        engine = _StubEngine()
        config = _config(timeout=30)

        fake = _qwen_like_subprocess(
            {1: "aggregated page one", 2: "aggregated page two", 3: "aggregated page three"}
        )
        with patch("socr.engines.base.subprocess.run", side_effect=fake):
            outputs = engine.process_pages(pdf, [1, 2, 3], config, dpi=72)

        assert len(outputs) == 3
        for po in outputs:
            assert po.status == PageStatus.SUCCESS
            assert po.failure_mode != po.failure_mode.EMPTY_OUTPUT
            assert po.text.strip()
        by_page = {po.page_num: po.text for po in outputs}
        assert "aggregated page one" in by_page[1]
        assert "aggregated page two" in by_page[2]
        assert "aggregated page three" in by_page[3]

    def test_aggregated_mapping_is_filename_sorted_not_request_order(self, tmp_path):
        """Non-ascending page_nums still map correctly: aggregated section k maps
        to the page whose rendered image is k-th in filename sort order, NOT to
        page_nums[k-1]."""
        pdf = _real_pdf(tmp_path / "doc.pdf", n_pages=9)
        engine = _StubEngine()
        config = _config(timeout=30)

        # Request a non-ascending, non-contiguous subset.
        page_nums = [7, 3, 9]
        # Rendered stems: page_0003, page_0007, page_0009 -> sorted order [3,7,9].
        # Section 1 -> page 3, section 2 -> page 7, section 3 -> page 9.
        fake = _qwen_like_subprocess(
            {1: "text for physical 3", 2: "text for physical 7", 3: "text for physical 9"}
        )
        with patch("socr.engines.base.subprocess.run", side_effect=fake):
            outputs = engine.process_pages(pdf, page_nums, config, dpi=72)

        by_page = {po.page_num: po.text for po in outputs}
        assert "text for physical 3" in by_page[3]
        assert "text for physical 7" in by_page[7]
        assert "text for physical 9" in by_page[9]

    def test_aggregated_count_mismatch_fails_all_pages(self, tmp_path):
        """A corrupt aggregate (wrong section count) fails ALL requested pages
        with a clear error, rather than silently mis-aligning the mapping."""
        pdf = _real_pdf(tmp_path / "doc.pdf", n_pages=3)
        engine = _StubEngine()
        config = _config(timeout=30)

        # Engine wrote only 2 sections for 3 images.
        fake = _qwen_like_subprocess({1: "only", 2: "two"}, count_override=2)
        with patch("socr.engines.base.subprocess.run", side_effect=fake):
            outputs = engine.process_pages(pdf, [1, 2, 3], config, dpi=72)

        assert len(outputs) == 3
        for po in outputs:
            assert po.status == PageStatus.ERROR
            assert po.failure_mode == po.failure_mode.EMPTY_OUTPUT
            assert "mismatch" in (po.error or "")

    def test_per_page_layout_still_preferred(self, tmp_path):
        """When the engine DOES emit per-page files, those win over any aggregate
        path (the aggregate fallback only fills genuinely missing pages)."""
        pdf = _real_pdf(tmp_path / "doc.pdf", n_pages=2)
        engine = _StubEngine()
        config = _config(timeout=30)

        def fake(cmd, *args, **kwargs):
            images_dir = Path(cmd[1])
            out_dir = Path(cmd[cmd.index("-o") + 1])
            for img in sorted(images_dir.glob("*.png")):
                rel_key = relative_key(img, images_dir)
                doc_dir = doc_dir_for(out_dir, rel_key)
                doc_dir.mkdir(parents=True, exist_ok=True)
                idx = int(img.stem.split("_")[-1])
                markdown_path_for(doc_dir, rel_key).write_text(
                    assemble_pages([f"per-page {idx}"]), encoding="utf-8"
                )
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("socr.engines.base.subprocess.run", side_effect=fake):
            outputs = engine.process_pages(pdf, [1, 2], config, dpi=72)
        by_page = {po.page_num: po.text for po in outputs}
        assert "per-page 1" in by_page[1]
        assert "per-page 2" in by_page[2]


# ---------------------------------------------------------------------------
# Conformance: a >=2-page BATCH run satisfies the contract harness
# ---------------------------------------------------------------------------


class TestBatchConformance:
    def test_two_file_batch_conforms(self, tmp_path):
        """A real batch of two >=2-page docs drives the contract's assert_conforms
        end-to-end (root index + per-doc sidecar + '## Page N' body)."""
        from ocr_output_contract.conformance import ExpectedDoc, assert_conforms

        in_dir = tmp_path / "in"
        _real_pdf(in_dir / "a.pdf", n_pages=2)
        _real_pdf(in_dir / "b.pdf", n_pages=2)
        out = tmp_path / "out"

        pipeline = UnifiedPipeline(_config())
        with patch.object(
            UnifiedPipeline,
            "_phase_backbone",
            _backbone_writing_pages({1: "conformant one", 2: "conformant two"}),
        ):
            pipeline.process_batch(in_dir, out)

        assert_conforms(
            out,
            [
                ExpectedDoc(rel_key="a.pdf", pages=2),
                ExpectedDoc(rel_key="b.pdf", pages=2),
            ],
        )
