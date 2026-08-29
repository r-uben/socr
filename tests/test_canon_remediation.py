"""Canon remediation tests (ocr-output-contract v0.1.1 adoption).

Covers the blockers fixed in the socr canon PR:

* the GUARDED canonical-first-then-legacy per-page/whole-doc read-back fallback
  (a legacy-layout engine output is still read, not silently dropped to EMPTY);
* the ``## Page N`` body conversion + manifest replay round-trip
  (saved .md splits back to the same per-page texts; replay == saved body);
* batch nonzero exit code on any failure (the canon uniform exit policy);
* the real run fingerprint (resolved model + populated prompt_hash, consensus
  label resolved) so model/task drift invalidates the cache;
* a >=2-page conformance test via the contract's ``assert_conforms`` (now
  requires ``## Page N`` body + resolves inline image links).

Engine subprocess / model boundaries are mocked: no live engines are invoked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")  # PyMuPDF; skip whole module if unavailable

from ocr_output_contract import (  # noqa: E402
    assemble_pages,
    run_fingerprint,
    split_native_pages,
)

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.result import (  # noqa: E402
    DocumentStatus,
    EngineResult,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState  # noqa: E402
from socr.engines.base import BaseEngine  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _config(**overrides) -> PipelineConfig:
    defaults = dict(
        primary_engine=EngineType.DEEPSEEK,
        enabled_engines=list(EngineType),
        quiet=True,
        tiered=False,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _real_pdf(path: Path, n_pages: int = 2) -> Path:
    doc = fitz.open()
    for i in range(n_pages):
        doc.new_page().insert_text((72, 72), f"page {i + 1}")
    doc.save(str(path))
    doc.close()
    return path


class _StubEngine(BaseEngine):
    """A minimal CLI engine whose only purpose is exercising the read-back."""

    @property
    def name(self) -> str:
        return "deepseek"  # has a configurable model in config (deepseek_*)

    @property
    def cli_command(self) -> str:
        return "stub-ocr"

    def _build_command(self, pdf_path, output_dir, config):
        return ["stub-ocr", str(pdf_path), "-o", str(output_dir)]


# ---------------------------------------------------------------------------
# 1. Guarded read-back fallback
# ---------------------------------------------------------------------------


class TestGuardedReadback:
    def test_per_page_canonical_layout_read(self, tmp_path):
        """The canonical <out>/<stem>_png/<stem>.md is read first."""
        eng = _StubEngine()
        images = tmp_path / "images"
        images.mkdir()
        out = tmp_path / "out"
        # v0.1.1 canonical layout for a .png input: <stem>_png/<stem>.md
        (out / "page_0001_png").mkdir(parents=True)
        (out / "page_0001_png" / "page_0001.md").write_text("CANON page text")

        text = eng._read_page_output("page_0001", out, scan_root=images)
        assert text == "CANON page text"

    def test_per_page_legacy_subdir_layout_read(self, tmp_path):
        """A legacy (v0.1.0) <out>/<stem>/<stem>.md layout is still read via the
        guarded fallback instead of yielding EMPTY for the page."""
        eng = _StubEngine()
        images = tmp_path / "images"
        images.mkdir()
        out = tmp_path / "out"
        # legacy/non-converged layout: no '_png' disambiguation
        (out / "page_0003").mkdir(parents=True)
        (out / "page_0003" / "page_0003.md").write_text("LEGACY page text")

        text = eng._read_page_output("page_0003", out, scan_root=images)
        assert text == "LEGACY page text"

    def test_per_page_flat_layout_read(self, tmp_path):
        """A flat <out>/<stem>.md layout is still read via the guarded fallback."""
        eng = _StubEngine()
        images = tmp_path / "images"
        images.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        (out / "page_0002.md").write_text("FLAT page text")

        text = eng._read_page_output("page_0002", out, scan_root=images)
        assert text == "FLAT page text"

    def test_per_page_rglob_is_stem_filtered(self, tmp_path):
        """The rglob net only returns a .md whose stem matches the page stem; a
        stray aggregate (qwen's pre-canon images.md) is NOT mistaken for it."""
        eng = _StubEngine()
        images = tmp_path / "images"
        images.mkdir()
        out = tmp_path / "out"
        (out / "weird").mkdir(parents=True)
        # A stray aggregate with a different stem must be ignored.
        (out / "weird" / "images.md").write_text("WRONG aggregate")
        # The real page under an unexpected dir, stem matches -> found.
        (out / "nested" / "deep").mkdir(parents=True)
        (out / "nested" / "deep" / "page_0007.md").write_text("RIGHT page text")

        text = eng._read_page_output("page_0007", out, scan_root=images)
        assert text == "RIGHT page text"

    def test_per_page_missing_returns_none(self, tmp_path):
        eng = _StubEngine()
        images = tmp_path / "images"
        images.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        assert eng._read_page_output("page_0001", out, scan_root=images) is None

    def test_whole_doc_legacy_layout_read(self, tmp_path):
        """_read_output reads a legacy <out>/<stem>/<stem>.md for a PDF input via
        the guarded fallback (PDF stem has no '_pdf' disambiguation, so canonical
        and legacy coincide; the flat fallback is the divergent case)."""
        eng = _StubEngine()
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        out = tmp_path / "out"
        # Flat legacy layout instead of <stem>/<stem>.md
        out.mkdir()
        (out / "paper.md").write_text("FLAT whole-doc text")
        assert eng._read_output(pdf, out) == "FLAT whole-doc text"


# ---------------------------------------------------------------------------
# 2. '## Page N' body + replay round-trip
# ---------------------------------------------------------------------------


class TestCanonicalBodyAndReplay:
    def test_saved_body_has_page_markers_and_splits_back(self, tmp_path):
        from socr.core.manifest import canonical_page_texts

        pdf = _real_pdf(tmp_path / "paper.pdf", n_pages=2)
        handle = DocumentHandle.from_path(pdf)
        state = DocumentState(handle=handle)
        for i in (1, 2):
            state.pages[i].best_output = PageOutput(
                page_num=i,
                text=f"body of page {i}",
                status=PageStatus.SUCCESS,
                engine="deepseek",
                audit_passed=True,
            )

        texts = canonical_page_texts(state)
        body = assemble_pages(texts)
        # One '## Page N' marker per page, round-trips bit-for-bit.
        assert body.count("## Page ") == 2
        assert split_native_pages(body) == ["body of page 1", "body of page 2"]

    def test_whole_doc_blob_does_not_double_header(self, tmp_path):
        """A whole-doc blob that already carries '## Page N' is split then
        re-assembled with exactly one header per page (no double headers)."""
        from socr.core.manifest import canonical_page_texts

        pdf = _real_pdf(tmp_path / "paper.pdf", n_pages=2)
        handle = DocumentHandle.from_path(pdf)
        state = DocumentState(handle=handle)
        blob = assemble_pages(["alpha", "beta"])  # '## Page 1..2'
        state.apply_result(
            EngineResult(
                document_path=pdf,
                engine="deepseek",
                status=DocumentStatus.SUCCESS,
                pages=[
                    PageOutput(
                        page_num=0,
                        text=blob,
                        status=PageStatus.SUCCESS,
                        engine="deepseek",
                        audit_passed=True,
                    )
                ],
            )
        )
        state.whole_doc_attempts[-1].audit_passed = True

        body = assemble_pages(canonical_page_texts(state))
        assert body.count("## Page ") == 2
        assert split_native_pages(body) == ["alpha", "beta"]

    def test_replay_is_bit_identical_to_saved_body(self, tmp_path):
        from socr.core.cache import BlobStore
        from socr.core.manifest import build_manifest, replay

        pdf = _real_pdf(tmp_path / "paper.pdf", n_pages=3)
        handle = DocumentHandle.from_path(pdf)
        state = DocumentState(handle=handle)
        for i in (1, 2, 3):
            state.pages[i].best_output = PageOutput(
                page_num=i,
                text=f"text {i}",
                status=PageStatus.SUCCESS,
                engine="deepseek",
                audit_passed=True,
            )
        saved_body = assemble_pages([f"text {i}" for i in (1, 2, 3)])

        store = BlobStore(tmp_path / "cache")
        manifest = build_manifest(state, store, dpi=120, saved_body=saved_body)
        assert replay(manifest, store) == saved_body

    def test_phase_assemble_split_count_matches_page_count(self, tmp_path):
        pdf = _real_pdf(tmp_path / "paper.pdf", n_pages=2)
        handle = DocumentHandle.from_path(pdf)
        state = DocumentState(handle=handle)
        for i in (1, 2):
            state.pages[i].best_output = PageOutput(
                page_num=i,
                text=f"page {i} content",
                status=PageStatus.SUCCESS,
                engine="deepseek",
                audit_passed=True,
            )
        pipeline = UnifiedPipeline(_config())
        out = tmp_path / "out"
        pipeline._phase_assemble(state, out)
        md = (out / "paper" / "paper.md").read_text()
        assert len(split_native_pages(md)) == handle.page_count

    def test_failed_audit_whole_doc_not_frozen_as_success(self, tmp_path):
        """A whole-doc attempt that FAILED audit is frozen with audit_passed=False
        (no status fabrication)."""
        from socr.core.cache import BlobStore
        from socr.core.manifest import build_manifest

        pdf = _real_pdf(tmp_path / "paper.pdf", n_pages=2)
        handle = DocumentHandle.from_path(pdf)
        state = DocumentState(handle=handle)
        blob = assemble_pages(["bad page 1", "bad page 2"])
        state.apply_result(
            EngineResult(
                document_path=pdf,
                engine="deepseek",
                status=DocumentStatus.AUDIT_FAILED,
                pages=[
                    PageOutput(
                        page_num=0,
                        text=blob,
                        status=PageStatus.WARNING,
                        engine="deepseek",
                        audit_passed=False,
                    )
                ],
            )
        )
        # The only whole-doc attempt failed audit.
        state.whole_doc_attempts[-1].audit_passed = False

        store = BlobStore(tmp_path / "cache")
        manifest = build_manifest(state, store, dpi=120)
        for pn in (1, 2):
            page = store.get_page(manifest.entries[pn].blob_ref)
            assert page.audit_passed is False
            assert page.status != PageStatus.SUCCESS


# ---------------------------------------------------------------------------
# 3. Batch nonzero exit on failure
# ---------------------------------------------------------------------------


class TestBatchExitCode:
    def test_batch_outcome_nonzero_on_failure(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _real_pdf(in_dir / "ok.pdf", n_pages=1)
        _real_pdf(in_dir / "bad.pdf", n_pages=1)
        out = tmp_path / "out"

        pipeline = UnifiedPipeline(_config())

        def _fake_process(pdf, output_dir, scan_root=None):
            ok = "ok" in pdf.name
            return EngineResult(
                document_path=pdf,
                engine="deepseek",
                status=DocumentStatus.SUCCESS if ok else DocumentStatus.ERROR,
                error=None if ok else "boom",
            )

        with patch.object(pipeline, "process", side_effect=_fake_process):
            pipeline.process_batch(in_dir, out)

        assert pipeline.last_outcome.exit_code != 0
        assert pipeline.last_outcome.failed >= 1

    def test_batch_outcome_zero_on_all_success(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _real_pdf(in_dir / "a.pdf", n_pages=1)
        _real_pdf(in_dir / "b.pdf", n_pages=1)
        out = tmp_path / "out"
        pipeline = UnifiedPipeline(_config())

        def _fake_process(pdf, output_dir, scan_root=None):
            return EngineResult(document_path=pdf, engine="deepseek", status=DocumentStatus.SUCCESS)

        with patch.object(pipeline, "process", side_effect=_fake_process):
            pipeline.process_batch(in_dir, out)
        assert pipeline.last_outcome.exit_code == 0

    def test_batch_scan_root_keys_relative_subtree(self, tmp_path):
        """Batch threads input_dir as scan_root, so the per-doc key mirrors the
        subtree, not the basename. (Verified via the resolved output layout.)"""
        captured = {}
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _real_pdf(in_dir / "paper.pdf", n_pages=1)
        out = tmp_path / "out"
        pipeline = UnifiedPipeline(_config())

        def _fake_process(pdf, output_dir, scan_root=None):
            captured["scan_root"] = scan_root
            return EngineResult(document_path=pdf, engine="deepseek", status=DocumentStatus.SUCCESS)

        with patch.object(pipeline, "process", side_effect=_fake_process):
            pipeline.process_batch(in_dir, out)
        assert captured["scan_root"] == in_dir


# ---------------------------------------------------------------------------
# 4. Real fingerprint
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_resolved_model_version_tracks_config(self):
        eng = _StubEngine()  # name == 'deepseek'
        cfg = _config()
        cfg.deepseek_model = "deepseek-ocr-v9"  # set an override the CLI would use
        assert eng.resolved_model_version(cfg) == "deepseek-ocr-v9"

    def test_resolved_model_version_falls_back_to_static(self):
        eng = _StubEngine()
        cfg = _config()
        # deepseek has no _model attr by default; should fall back to static "".
        assert eng.resolved_model_version(cfg) == eng.model_version

    def test_manifest_fingerprint_uses_resolved_model_and_prompt_hash(self, tmp_path):
        from socr.core.cache import BlobStore
        from socr.core.manifest import build_manifest

        pdf = _real_pdf(tmp_path / "paper.pdf", n_pages=1)
        handle = DocumentHandle.from_path(pdf)
        state = DocumentState(handle=handle)
        state.apply_result(
            EngineResult(
                document_path=pdf,
                engine="gemini",
                status=DocumentStatus.SUCCESS,
                model_version="gemini-3-flash-preview",  # the static literal
                pages=[
                    PageOutput(
                        page_num=1,
                        text="real text",
                        status=PageStatus.SUCCESS,
                        engine="gemini",
                        audit_passed=True,
                    )
                ],
            )
        )
        # Orchestrator resolves the ACTUAL model (config.gemini_model) + determinants.
        fp_inputs = {"gemini": ("gemini-3-pro", "socr", "convert", None)}
        store = BlobStore(tmp_path / "cache")
        manifest = build_manifest(state, store, dpi=120, fingerprint_inputs=fp_inputs)
        fp = manifest.entries[1].fingerprint
        # model_version is the RESOLVED model, not the static literal.
        assert fp.model_version == "gemini-3-pro"
        # prompt_hash is populated from run_fingerprint of the resolved determinants.
        assert fp.prompt_hash == run_fingerprint("gemini-3-pro", "socr", "convert", None)
        assert fp.prompt_hash.startswith("fp:")

    def test_consensus_label_resolves_to_underlying_engine(self, tmp_path):
        from socr.core.cache import BlobStore
        from socr.core.manifest import build_manifest

        pdf = _real_pdf(tmp_path / "paper.pdf", n_pages=1)
        handle = DocumentHandle.from_path(pdf)
        state = DocumentState(handle=handle)
        state.apply_result(
            EngineResult(
                document_path=pdf,
                engine="consensus(qwen)",
                status=DocumentStatus.SUCCESS,
                pages=[
                    PageOutput(
                        page_num=1,
                        text="merged text",
                        status=PageStatus.SUCCESS,
                        engine="consensus(qwen)",
                        audit_passed=True,
                    )
                ],
            )
        )
        fp_inputs = {"qwen": ("qwen3-vl:8b", "ollama", None, None)}
        store = BlobStore(tmp_path / "cache")
        manifest = build_manifest(state, store, dpi=120, fingerprint_inputs=fp_inputs)
        fp = manifest.entries[1].fingerprint
        # The consensus(<engine>) label resolved to the underlying engine's model.
        assert fp.model_version == "qwen3-vl:8b"
        assert fp.prompt_hash == run_fingerprint("qwen3-vl:8b", "ollama", None, None)


# ---------------------------------------------------------------------------
# 5. >=2-page conformance via the contract harness
# ---------------------------------------------------------------------------


class TestConformance:
    def test_two_page_output_conforms(self, tmp_path):
        """A real >=2-page assemble drives the contract's assert_conforms, which
        now requires the '## Page N' body and resolves inline image links."""
        from ocr_output_contract.conformance import ExpectedDoc, assert_conforms

        pdf = tmp_path / "src" / "paper.pdf"
        pdf.parent.mkdir(parents=True)
        _real_pdf(pdf, n_pages=2)

        handle = DocumentHandle.from_path(pdf)
        state = DocumentState(handle=handle)
        for i in (1, 2):
            state.pages[i].best_output = PageOutput(
                page_num=i,
                text=f"Conformant content for page {i}.",
                status=PageStatus.SUCCESS,
                engine="deepseek",
                audit_passed=True,
            )

        pipeline = UnifiedPipeline(_config())
        out_root = tmp_path / "out"
        # scan_root = pdf.parent -> rel_key 'paper.pdf'
        pipeline._scan_root = pdf.parent
        pipeline._phase_assemble(state, out_root)

        assert_conforms(out_root, [ExpectedDoc(rel_key="paper.pdf", pages=2)])
