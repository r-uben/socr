"""GH-871: a PDF none of whose pages load is refused and RECORDED, not crashed on.

Measured on a real file (MuPDF "format error: non-page object in page tree"):
the document declared 64 pages, none loaded, and ``socr process`` died with a raw
traceback out of ``_phase_analyze`` -- no ``metadata.json``, no root-index entry,
no document status. ``open_pdf``'s default glyph recovery made it worse in a
different way: touching the pages triggered MuPDF's own repair and the page count
silently became 0.

The real file is copyrighted and cannot be a fixture. MuPDF also repairs the
simple page-tree breakages one can author, and does so differently across
versions, so a crafted "broken" file would be a flaky stand-in for the real
failure. These tests therefore make page loads raise the exact exception type the
real file produced, which is what socr's handling has to cope with.

Pins are differences -- probe readable vs unreadable, same pipeline -- not an
absolute outcome, and nothing here needs a provider.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from socr.core import pdf as pdf_mod
from socr.core.config import EngineType, PipelineConfig
from socr.core.pdf import PageLoadProbe, probe_page_loads
from socr.core.result import DocumentStatus, FailureMode
from socr.pipeline.orchestrator import UnifiedPipeline


def _real_pdf(path: Path, pages: int = 2) -> Path:
    doc = fitz.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"page {i + 1}")
    doc.save(path)
    doc.close()
    return path


class _DamagedDoc:
    """A document that declares pages but cannot load them, like the real file."""

    def __init__(self, declared: int, loadable: set[int] = frozenset()):
        self.page_count = declared
        self._loadable = loadable
        self.closed = False

    def load_page(self, index):
        if index in self._loadable:
            return object()
        raise fitz.mupdf.FzErrorFormat("malformed page tree")

    def close(self):
        self.closed = True


# --------------------------------------------------------------------------
# the probe
# --------------------------------------------------------------------------


def test_a_sound_pdf_probes_fully_loadable(tmp_path):
    probe = probe_page_loads(_real_pdf(tmp_path / "ok.pdf", pages=3))
    assert probe == PageLoadProbe(declared=3, loadable=3, first_error=None)
    assert not probe.unreadable


def test_a_file_whose_pages_all_fail_is_unreadable_and_names_the_error(monkeypatch, tmp_path):
    fake = _DamagedDoc(declared=64)
    monkeypatch.setattr(pdf_mod.fitz, "open", lambda p: fake)
    probe = probe_page_loads(tmp_path / "any.pdf")
    assert probe.declared == 64
    assert probe.loadable == 0
    assert probe.unreadable
    assert "malformed page tree" in (probe.first_error or "")
    assert fake.closed, "the probe must not leak the document handle"


def test_a_partially_damaged_file_is_NOT_unreadable(monkeypatch, tmp_path):
    """One loadable page means the whole-document refusal must not fire."""
    monkeypatch.setattr(pdf_mod.fitz, "open", lambda p: _DamagedDoc(declared=5, loadable={2}))
    probe = probe_page_loads(tmp_path / "any.pdf")
    assert (probe.declared, probe.loadable) == (5, 1)
    assert not probe.unreadable


def test_an_unopenable_file_is_reported_not_raised(tmp_path):
    bogus = tmp_path / "not-a-pdf.pdf"
    bogus.write_bytes(b"this is not a pdf at all")
    probe = probe_page_loads(bogus)
    assert probe.unreadable
    assert probe.first_error


def test_the_probe_does_not_go_through_glyph_recovery(monkeypatch, tmp_path):
    """The whole point of a separate probe: ``open_pdf(repair=True)`` touches pages,
    which let MuPDF's repair collapse the count to 0 and erase the evidence."""

    def _forbidden(*a, **k):
        raise AssertionError("probe_page_loads must not use open_pdf / glyph recovery")

    monkeypatch.setattr(pdf_mod, "open_pdf", _forbidden)
    monkeypatch.setattr(pdf_mod, "apply_glyph_recovery", _forbidden)
    assert probe_page_loads(_real_pdf(tmp_path / "ok.pdf")).loadable == 2


# --------------------------------------------------------------------------
# the refusal -- a difference over probe outcome, same pipeline
# --------------------------------------------------------------------------


@pytest.fixture
def pipeline(tmp_path):
    # Engines pinned (#841): the default ``AUTO`` makes ``process()`` shell out to
    # ``ollama`` to resolve an engine -- a live probe these tests never need.
    # Judge pinned (#886): ``_write_metadata``/``_resume_skip`` call
    # ``_run_fingerprint``, which resolves the page judge via
    # ``_resolve_judge_model`` -- another Ollama probe -- whenever
    # ``judge_backend`` is not "heuristic". None of these tests are about the
    # judge, so it stays off.
    return UnifiedPipeline(
        PipelineConfig(
            output_dir=tmp_path / "out",
            quiet=True,
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            judge_backend="heuristic",
        )
    )


def _set_probe(monkeypatch, probe: PageLoadProbe):
    monkeypatch.setattr(pdf_mod, "probe_page_loads", lambda path: probe)


def test_refusal_fires_only_when_nothing_loads(monkeypatch, pipeline, tmp_path):
    pdf = _real_pdf(tmp_path / "doc.pdf")
    out = tmp_path / "out"
    pipeline._scan_root = pdf.parent

    _set_probe(monkeypatch, PageLoadProbe(declared=2, loadable=2, first_error=None))
    readable = pipeline._refuse_unreadable_input(pdf, out)

    _set_probe(monkeypatch, PageLoadProbe(declared=64, loadable=0, first_error="FzErrorFormat: x"))
    refused = pipeline._refuse_unreadable_input(pdf, out)

    assert readable is None
    assert refused is not None
    assert refused.status is DocumentStatus.ERROR
    assert refused.failure_mode is FailureMode.UNREADABLE_INPUT
    assert "0 of 64" in (refused.error or "")


def test_the_refusal_is_recorded_in_metadata_and_the_root_index(monkeypatch, pipeline, tmp_path):
    """Surfacing at every level: the per-document record AND the root index."""
    pdf = _real_pdf(tmp_path / "doc.pdf")
    out = tmp_path / "out"
    pipeline._scan_root = pdf.parent
    _set_probe(monkeypatch, PageLoadProbe(declared=64, loadable=0, first_error="FzErrorFormat: x"))

    pipeline._refuse_unreadable_input(pdf, out)

    doc_meta = json.loads((out / "doc" / "metadata.json").read_text())
    assert doc_meta["status"] == "failed"
    assert doc_meta["pages"] == 64, "the declared count, not open_pdf's collapsed 0"
    assert FailureMode.UNREADABLE_INPUT.value in doc_meta["error"]

    root = (out / "metadata.json").read_text()
    assert '"failed"' in root
    assert FailureMode.UNREADABLE_INPUT.value in root


def test_process_refuses_before_analysis_and_does_not_crash(monkeypatch, pipeline, tmp_path):
    """End to end through ``process()``: the analysis phase must never be reached,
    because on the real file that is where the traceback came from."""
    pdf = _real_pdf(tmp_path / "doc.pdf")
    _set_probe(monkeypatch, PageLoadProbe(declared=64, loadable=0, first_error="FzErrorFormat: x"))

    def _explode(*a, **k):
        raise AssertionError("_phase_analyze must not run on an unreadable document")

    monkeypatch.setattr(pipeline, "_phase_analyze", _explode)
    result = pipeline.process(pdf, tmp_path / "out")

    assert result.status is DocumentStatus.ERROR
    assert result.failure_mode is FailureMode.UNREADABLE_INPUT


def test_a_refused_document_is_retried_not_skipped_on_the_next_run(monkeypatch, pipeline, tmp_path):
    """A FAILED record must not satisfy the resume gate, or a file fixed and
    re-downloaded under the same name would never be read."""
    pdf = _real_pdf(tmp_path / "doc.pdf")
    out = tmp_path / "out"
    pipeline._scan_root = pdf.parent
    _set_probe(monkeypatch, PageLoadProbe(declared=64, loadable=0, first_error="FzErrorFormat: x"))
    pipeline._refuse_unreadable_input(pdf, out)

    assert pipeline._resume_skip(pdf, out) is None

    # Control, so the line above cannot pass vacuously: the SAME record flipped to
    # ``completed`` with its markdown present must be skipped. If the gate could
    # never skip in this fixture, "not skipped" above would prove nothing.
    root_path = out / "metadata.json"
    root = json.loads(root_path.read_text())
    flipped = json.dumps(root).replace('"failed"', '"completed"')
    root_path.write_text(flipped)
    doc_dir = out / "doc"
    (doc_dir / "doc.md").write_text("content")
    entry_meta = json.loads((doc_dir / "metadata.json").read_text())
    entry_meta["output_path"] = str(doc_dir / "doc.md")
    (doc_dir / "metadata.json").write_text(json.dumps(entry_meta))
    root = json.loads(root_path.read_text())
    root_text = json.dumps(root).replace(
        '"output_path": ""', f'"output_path": "{doc_dir / "doc.md"}"'
    )
    root_path.write_text(root_text)

    assert pipeline._resume_skip(pdf, out) is not None, (
        "control failed: the gate never skips in this fixture, so the FAILED case is untested"
    )


def test_a_file_that_will_not_even_open_is_refused_and_recorded(pipeline, tmp_path):
    """PR #878 review: the unopenable case, through ``process()``, with NO mocks.

    ``probe_page_loads`` returns ``declared=0`` for bytes ``fitz`` rejects, and 0
    looked like "unset" to ``DocumentHandle``, which re-counted through
    ``open_pdf`` and raised -- the same raw traceback, one frame later. Every
    earlier test used ``declared=64``, so none reached it.
    """
    bogus = tmp_path / "bogus.pdf"
    bogus.write_bytes(b"not a pdf at all")
    out = tmp_path / "out"

    result = pipeline.process(bogus, out)

    assert result.status is DocumentStatus.ERROR
    assert result.failure_mode is FailureMode.UNREADABLE_INPUT
    doc_meta = json.loads((out / "bogus" / "metadata.json").read_text())
    assert doc_meta["status"] == "failed"
    assert FailureMode.UNREADABLE_INPUT.value in doc_meta["error"]


def test_a_known_zero_page_count_is_not_recounted(tmp_path):
    """The handle must keep a measured 0 rather than re-derive it."""
    from socr.core.document import DocumentHandle

    bogus = tmp_path / "bogus.pdf"
    bogus.write_bytes(b"not a pdf at all")
    handle = DocumentHandle(path=bogus, page_count=0, page_count_known=True)
    assert handle.page_count == 0


class _CountRaises:
    """A document whose declared page count itself cannot be read (#882)."""

    closed = False

    @property
    def page_count(self):
        raise fitz.mupdf.FzErrorFormat("cannot read page tree")

    def close(self):
        self.closed = True


def test_a_page_count_that_cannot_be_read_is_reported_not_raised(monkeypatch, tmp_path):
    """#882: ``probe_page_loads`` promises never to raise. ``page_count`` sat
    outside both guards, so a page tree that fails on the count escaped as a raw
    traceback -- the #871 failure one line earlier."""
    fake = _CountRaises()
    monkeypatch.setattr(pdf_mod.fitz, "open", lambda p: fake)
    probe = probe_page_loads(tmp_path / "any.pdf")
    assert probe.unreadable
    assert "cannot read page tree" in (probe.first_error or "")
    assert fake.closed, "the handle must still be closed on this path"


def test_process_refuses_a_file_whose_page_count_raises(monkeypatch, pipeline, tmp_path):
    """End to end: the refusal record, not a traceback."""
    pdf = _real_pdf(tmp_path / "doc.pdf")
    real_open = pdf_mod.fitz.open
    monkeypatch.setattr(
        pdf_mod.fitz, "open", lambda p: _CountRaises() if str(p) == str(pdf) else real_open(p)
    )
    # Hermetic (cubic on #894): no judge is under test, and the fingerprint the
    # refusal record carries would otherwise probe ollama for one (#886).
    pipeline._resolve_judge_model = lambda *a, **k: ""
    out = tmp_path / "out"
    result = pipeline.process(pdf, out)
    assert result.status is DocumentStatus.ERROR
    assert result.failure_mode is FailureMode.UNREADABLE_INPUT
    # The fix promises a RECORD, not only a return value (cubic on #894): a
    # regression that returned ERROR but dropped the metadata would otherwise pass.
    doc_meta = json.loads((out / "doc" / "metadata.json").read_text())
    assert doc_meta["status"] == "failed"
    assert FailureMode.UNREADABLE_INPUT.value in doc_meta["error"]
