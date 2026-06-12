"""Integration tests: socr consumes the canonical engine output structure.

Every engine CLI now emits the family-wide canonical layout via the shared
``ocr-output-contract`` package:

    <output_dir>/<rel/dir>/<stem>/<stem>.md   (all pages, '## Page N' headers)
    <output_dir>/<rel/dir>/<stem>/metadata.json
    <output_dir>/metadata.json                (root index, input-relative keys)

These tests mock the engine *subprocess boundary* so it writes a canonical tree,
then assert socr reads it back FULLY — never the empty-replay / empty-read-back
the orchestrator used to produce. No live engine or model is invoked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ocr_output_contract import (
    assemble_pages,
    doc_dir_for,
    markdown_path_for,
    relative_key,
)

from socr.core.config import PipelineConfig
from socr.core.result import DocumentStatus, FailureMode, PageStatus
from socr.engines.base import BaseEngine


class _CanonicalEngine(BaseEngine):
    """A concrete BaseEngine whose CLI subprocess writes the canonical tree."""

    @property
    def name(self) -> str:
        return "canon"

    @property
    def cli_command(self) -> str:
        return "canon-ocr"

    def _build_command(self, pdf_path, output_dir, config):
        # The actual command is irrelevant — subprocess.run is mocked. We stash
        # the (input, output_dir) so the fake subprocess can write canonical out.
        return ["canon-ocr", str(pdf_path), "-o", str(output_dir)]


def _fake_subprocess_writing_canon(page_texts_by_input):
    """Return a fake subprocess.run that writes a canonical tree for the input.

    ``page_texts_by_input`` maps an input *stem* to its list of per-page texts.
    The fake parses the mocked command ('<cli> <input> -o <dir>'), resolves the
    canonical doc dir for that input under <dir>, and writes <stem>.md with
    '## Page N' headers — exactly what a conformant engine CLI produces.
    """

    def _run(cmd, *args, **kwargs):
        input_path = Path(cmd[1])
        out_dir = Path(cmd[cmd.index("-o") + 1])
        stem = input_path.stem
        pages = page_texts_by_input.get(stem, [f"text for {stem}"])
        rel_key = relative_key(input_path, input_path.parent)
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


def _make_pdf(path: Path, n_pages: int = 2):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"native page {i + 1}")
    doc.save(str(path))
    doc.close()
    return path


def test_process_document_reads_canonical_md_fully(tmp_path):
    """A whole-PDF call reads back the full aggregated '## Page N' markdown."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    engine = _CanonicalEngine()
    config = PipelineConfig(timeout=30)

    fake = _fake_subprocess_writing_canon(
        {"paper": ["alpha content", "beta content", "gamma content"]}
    )
    with patch("socr.engines.base.subprocess.run", side_effect=fake):
        result = engine.process_document(pdf, tmp_path / "out", config)

    assert result.status == DocumentStatus.SUCCESS
    assert len(result.pages) == 1
    text = result.pages[0].text
    # All page sections present — no empty / truncated read-back.
    for token in ("alpha content", "beta content", "gamma content"):
        assert token in text
    assert "## Page 1" in text  # canonical headers preserved through read-back


def test_process_pages_reads_each_canonical_page(tmp_path):
    """Per-page-image calls each read back their own canonical <stem>/<stem>.md."""
    pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=3)
    engine = _CanonicalEngine()
    config = PipelineConfig(timeout=30)

    # Each rendered page image is named page_0001.png etc; the fake writes a
    # one-page canonical .md per image keyed by that stem.
    def fake(cmd, *args, **kwargs):
        images_dir = Path(cmd[1])
        out_dir = Path(cmd[cmd.index("-o") + 1])
        for img in sorted(images_dir.glob("*.png")):
            stem = img.stem
            rel_key = relative_key(img, images_dir)
            doc_dir = doc_dir_for(out_dir, rel_key)
            doc_dir.mkdir(parents=True, exist_ok=True)
            md_path = markdown_path_for(doc_dir, rel_key)
            # Embed the page index so we can verify per-page mapping.
            idx = int(stem.split("_")[-1])
            md_path.write_text(assemble_pages([f"page-image OCR {idx}"]), encoding="utf-8")
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    with patch("socr.engines.base.subprocess.run", side_effect=fake):
        outputs = engine.process_pages(pdf, [1, 2, 3], config, dpi=72)

    assert len(outputs) == 3
    for po in outputs:
        assert po.status == PageStatus.SUCCESS
        assert po.text.strip() != ""  # no empty read-back
    # Page n maps to its own canonical file (page_000n -> 'page-image OCR n').
    by_page = {po.page_num: po.text for po in outputs}
    assert "page-image OCR 1" in by_page[1]
    assert "page-image OCR 2" in by_page[2]
    assert "page-image OCR 3" in by_page[3]


def test_process_pages_salvages_partial_results_on_nonzero_exit(tmp_path):
    """Issue #40: one RECITATION-blocked page must not nuke the whole batch.

    Engine CLIs follow a uniform exit policy (nonzero if ANY page failed)
    while still writing the successful pages' files. socr must read back the
    pages that succeeded and classify the missing one as RECITATION from
    stderr — not discard all pages as CLI_ERROR.
    """
    pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=3)
    engine = _CanonicalEngine()
    config = PipelineConfig(timeout=30)

    def fake(cmd, *args, **kwargs):
        images_dir = Path(cmd[1])
        out_dir = Path(cmd[cmd.index("-o") + 1])
        blocked = images_dir / "page_0003.png"
        for img in sorted(images_dir.glob("*.png")):
            if img == blocked:
                continue  # RECITATION-blocked: no output file for this page
            rel_key = relative_key(img, images_dir)
            doc_dir = doc_dir_for(out_dir, rel_key)
            doc_dir.mkdir(parents=True, exist_ok=True)
            md_path = markdown_path_for(doc_dir, rel_key)
            idx = int(img.stem.split("_")[-1])
            md_path.write_text(assemble_pages([f"page-image OCR {idx}"]), encoding="utf-8")
        result = MagicMock()
        result.returncode = 1  # uniform exit policy: any page failed -> nonzero
        result.stdout = ""
        result.stderr = (
            f"ERROR:gemini_ocr.processor:Error processing {blocked}: "
            "Empty response: finish_reason=FinishReason.RECITATION, "
            "len(parts)=0, part_types=[], safety_ratings=None"
        )
        return result

    with patch("socr.engines.base.subprocess.run", side_effect=fake):
        outputs = engine.process_pages(pdf, [1, 2, 3], config, dpi=72)

    by_page = {po.page_num: po for po in outputs}
    assert by_page[1].status == PageStatus.SUCCESS
    assert "page-image OCR 1" in by_page[1].text
    assert by_page[2].status == PageStatus.SUCCESS
    assert "page-image OCR 2" in by_page[2].text
    # The blocked page fails alone, classified for the repair router.
    assert by_page[3].status == PageStatus.ERROR
    assert by_page[3].failure_mode == FailureMode.RECITATION


def test_process_pages_total_cli_failure_marks_all_cli_error(tmp_path):
    """A CLI that exits nonzero and writes nothing still fails every page."""
    pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=2)
    engine = _CanonicalEngine()
    config = PipelineConfig(timeout=30)

    def fake(cmd, *args, **kwargs):  # writes no files
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        result.stderr = "Error: GEMINI_API_KEY not set"
        return result

    with patch("socr.engines.base.subprocess.run", side_effect=fake):
        outputs = engine.process_pages(pdf, [1, 2], config, dpi=72)

    assert len(outputs) == 2
    for po in outputs:
        assert po.status == PageStatus.ERROR
        assert po.failure_mode == FailureMode.CLI_ERROR
        assert "CLI exited 1" in po.error


def test_missing_canonical_output_is_empty_not_crash(tmp_path):
    """If the engine writes nothing, read-back returns EMPTY_OUTPUT, not a crash."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    engine = _CanonicalEngine()
    config = PipelineConfig(timeout=30)

    def fake(cmd, *args, **kwargs):  # writes no files
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    with patch("socr.engines.base.subprocess.run", side_effect=fake):
        result = engine.process_document(pdf, tmp_path / "out", config)

    assert result.status == DocumentStatus.ERROR
