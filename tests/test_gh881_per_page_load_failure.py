"""GH-881: one page MuPDF cannot load costs that page, not the document.

#871 refuses a PDF none of whose pages load. A PARTIALLY damaged one -- some pages
load, some raise -- still died with a raw MuPDF traceback out of
``BornDigitalDetector.detect`` and recorded nothing: every readable page was lost
along with the bad one.

No real damaged file is available (the one measured corpus file loads no pages and
is copyrighted), and MuPDF repairs hand-authored page-tree damage differently
across versions. So page loads are made to raise the exact exception the real file
produced, for one chosen index, on a real PDF. Everything else is the real
pipeline.

Pins are DIFFERENCES: the same document run damaged and undamaged, so no
provider-dependent outcome (the status of a page that needs OCR, the document
status of a no-provider run) is pinned as an absolute.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

from socr.core import providers
from socr.core.born_digital import BornDigitalDetector
from socr.core.config import EngineType, PipelineConfig
from socr.core.result import DocumentStatus, FailureMode, PageOutput, PageStatus
from socr.pipeline.agentic import AcceptDecision
from socr.pipeline.orchestrator import UnifiedPipeline

PAGES = 4
BAD_INDEX = 1  # zero-based: page 2
BAD_PAGE = BAD_INDEX + 1


def _pdf(path: Path) -> Path:
    doc = fitz.open()
    for p in range(PAGES):
        page = doc.new_page()
        y = 80
        for _ in range(14):
            page.insert_text(
                (60, y), f"Page {p + 1} coefficient 0.082 significant at 1 percent", fontsize=9
            )
            y += 16
    doc.save(str(path))
    doc.close()
    return path


@contextmanager
def _page_load_raises(index: int | None):
    """Make loading page ``index`` raise MuPDF's format error; ``None`` is a no-op."""
    if index is None:
        yield
        return
    real_load = fitz.Document.load_page
    real_get = fitz.Document.__getitem__

    def load_page(self, page_id=0, *a, **kw):
        if page_id == index:
            raise fitz.mupdf.FzErrorFormat("format error: non-page object in page tree")
        return real_load(self, page_id, *a, **kw)

    def getitem(self, i):
        if i == index:
            raise fitz.mupdf.FzErrorFormat("format error: non-page object in page tree")
        return real_get(self, i)

    with (
        patch.object(fitz.Document, "load_page", load_page),
        patch.object(fitz.Document, "__getitem__", getitem),
    ):
        yield


class _YesJudge:
    def assess(self, output, provider):
        return AcceptDecision(accept=True, reason="stub")


def _pipeline(**overrides) -> UnifiedPipeline:
    cfg = PipelineConfig(
        agentic=True,
        quiet=True,
        **overrides,
        primary_engine=EngineType.QWEN,
        local_engine=EngineType.QWEN,
        enabled_engines=[EngineType.QWEN],
    )
    pipe = UnifiedPipeline(cfg)
    # CI has no ollama and no provider: pin every ambient dependency.
    pipe._available_engines_for_agentic = lambda: [providers.PROFILE_QWEN_LOCAL]
    pipe._build_page_judge = lambda state: _YesJudge()
    pipe._resolve_judge_model = lambda: ""
    pipe._resolve_crop_vlm_model = lambda: None
    pipe._run_engine_on_pages = lambda state, nums, nat, eng, phase, profile=None, **kw: [
        PageOutput(page_num=p, text=f"ocr {p}", status=PageStatus.SUCCESS, engine="qwen")
        for p in nums
    ]
    return pipe


def _run(tmp_path: Path, damaged: int | None, name: str):
    pdf = _pdf(tmp_path / f"{name}.pdf")
    out = tmp_path / f"out_{name}"
    pipe = _pipeline()
    with _page_load_raises(damaged):
        result = pipe.process(pdf, output_dir=out)
    return pipe, pdf, out, result


def _doc_dir(out: Path) -> Path:
    cands = [p for p in out.rglob("pages") if p.is_dir()]
    assert len(cands) == 1, cands
    return cands[0].parent


def _sidecar(doc_dir: Path, n: int) -> dict:
    return json.loads((doc_dir / "pages" / f"{n:05d}.json").read_text())


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


def test_detect_assesses_every_other_page_and_marks_the_bad_one(tmp_path):
    pdf = _pdf(tmp_path / "d.pdf")
    with _page_load_raises(BAD_INDEX):
        assessment = BornDigitalDetector().detect(pdf)
    assert [p.page_num for p in assessment.pages] == list(range(1, PAGES + 1))
    bad = assessment.pages[BAD_INDEX]
    assert "non-page object in page tree" in bad.load_error
    others = [p for i, p in enumerate(assessment.pages) if i != BAD_INDEX]
    assert all(p.load_error == "" for p in others)
    assert all(p.is_born_digital and p.native_text for p in others)
    # The placeholder is inert: it must not read as a real (empty scanned) page.
    assert not bad.is_born_digital and bad.native_text == ""


def test_detect_on_an_undamaged_document_sets_no_load_error(tmp_path):
    assessment = BornDigitalDetector().detect(_pdf(tmp_path / "ok.pdf"))
    assert all(p.load_error == "" for p in assessment.pages)


# --------------------------------------------------------------------------
# the document survives, the page fails, and it says so at every level
# --------------------------------------------------------------------------


def test_the_other_pages_are_processed_and_the_bad_page_is_failed(tmp_path):
    _, _, out_bad, bad = _run(tmp_path, BAD_INDEX, "bad")
    _, _, out_ok, ok = _run(tmp_path, None, "ok")
    dd_bad, dd_ok = _doc_dir(out_bad), _doc_dir(out_ok)

    # The difference: the same document, damaged or not. Without the per-page
    # catch the damaged run raised out of _phase_analyze and wrote nothing.
    assert bad.status is not DocumentStatus.SUCCESS
    assert bad.status is not ok.status

    # Every undamaged page is byte-for-byte what the undamaged run produced.
    for n in range(1, PAGES + 1):
        if n == BAD_PAGE:
            continue
        assert (dd_bad / "pages" / f"{n:05d}.md").read_bytes() == (
            dd_ok / "pages" / f"{n:05d}.md"
        ).read_bytes()

    # The bad page: FAILED, with the specific mode, on its own sidecar.
    side = _sidecar(dd_bad, BAD_PAGE)
    assert side["status"] == PageStatus.ERROR.value
    assert side["failure_mode"] == FailureMode.UNREADABLE_INPUT.value
    assert "non-page object in page tree" in json.dumps(side)
    assert f"[page {BAD_PAGE} failed:" in (dd_bad / "pages" / f"{BAD_PAGE:05d}.md").read_text()

    # Control: the same page, undamaged, is not failed.
    assert _sidecar(dd_ok, BAD_PAGE)["failure_mode"] != FailureMode.UNREADABLE_INPUT.value


def test_the_failure_reaches_document_status_metadata_and_the_final_markdown(tmp_path):
    _, pdf, out, result = _run(tmp_path, BAD_INDEX, "bad")
    dd = _doc_dir(out)

    assert result.status is not DocumentStatus.SUCCESS

    meta = json.loads((dd / "metadata.json").read_text())
    assert meta["status"] != "completed"

    md = (dd / f"{pdf.stem}.md").read_text()
    assert md.count("## Page ") == PAGES  # marker balance: no page header lost
    assert f"[page {BAD_PAGE} failed:" in md

    # document-level error and audit trail name the CAUSE, not just "no output"
    assert FailureMode.UNREADABLE_INPUT.value in (result.error or "")
    assert FailureMode.UNREADABLE_INPUT.value in (meta.get("error") or "")
    audit = (dd / "audit_log.json").read_text()
    assert "page_unloadable" in audit and "page_failed" in audit


# --------------------------------------------------------------------------
# resume: a FAILED page is never terminal-skipped
# --------------------------------------------------------------------------


def test_a_failed_page_is_never_restored_from_the_ledger(tmp_path):
    """The per-page ledger accepts only SUCCESS. Control: a good page of the same
    run IS restored, so "not restored" cannot pass because the ledger is inert."""
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState

    pipe, pdf, out, _ = _run(tmp_path, BAD_INDEX, "bad")
    state = DocumentState(handle=DocumentHandle(path=pdf, page_count=PAGES, page_count_known=True))
    # Same pipeline instance: the ledger compares the run fingerprint it wrote.
    assert pipe._load_terminal_page(state, BAD_PAGE, out) is None
    restored = [n for n in range(1, PAGES + 1) if pipe._load_terminal_page(state, n, out)]
    assert restored, "control: no page at all was restorable, so the check above is vacuous"
    assert BAD_PAGE not in restored


def test_a_repaired_page_is_re_read_on_a_forced_rerun(tmp_path):
    _, pdf, out, first = _run(tmp_path, BAD_INDEX, "bad")
    dd = _doc_dir(out)
    assert first.status is not DocumentStatus.SUCCESS

    # The same bytes fail identically, so the document gate treats a PARTIAL result
    # as final (``_resume_skippable``); a forced rerun with the page now loading must
    # re-read it rather than restore its FAILED sidecar.
    _pipeline(reprocess=True).process(pdf, output_dir=out)
    assert _sidecar(dd, BAD_PAGE)["failure_mode"] != FailureMode.UNREADABLE_INPUT.value
    assert "failed:" not in (dd / f"{pdf.stem}.md").read_text()


def test_the_native_word_cache_does_not_trip_over_the_bad_page(tmp_path, caplog):
    """The words cache opens every non-born-digital page. The placeholder is one,
    and reading it raised -- costing every later page its words, logged only."""
    import logging

    with caplog.at_level(logging.WARNING):
        _run(tmp_path, BAD_INDEX, "bad")
    assert not [r for r in caplog.records if "failed to cache native words" in r.getMessage()]


def test_partial_damage_keeps_the_declared_page_count(tmp_path, monkeypatch):
    """``DocumentHandle`` counts through a repairing open that can report FEWER pages
    than declared on a damaged tree, which would drop the bad pages from the state
    so they could never be recorded as failed. Control: an undamaged document keeps
    the ordinary path."""
    from socr.core.document import DocumentHandle

    pdf = _pdf(tmp_path / "p.pdf")
    monkeypatch.setattr(
        DocumentHandle,
        "from_path",
        classmethod(lambda cls, path: cls(path=path, page_count=1, page_count_known=True)),
    )
    pipe = _pipeline()
    with _page_load_raises(BAD_INDEX):
        assert pipe._document_handle_for(pdf).page_count == PAGES
    assert pipe._document_handle_for(pdf).page_count == 1


# --------------------------------------------------------------------------
# review round 1 (Astra, PR #947)
# --------------------------------------------------------------------------


def _figure_pdf(path: Path) -> Path:
    """Four pages; the LAST carries a large embedded image (a figure)."""
    import io

    from PIL import Image

    doc = fitz.open()
    for p in range(PAGES - 1):
        doc.new_page().insert_text((72, 72), f"text page {p + 1}")
    page = doc.new_page()
    page.insert_text((72, 72), "Figure 1")
    buf = io.BytesIO()
    Image.new("RGB", (400, 500), color=(200, 200, 255)).save(buf, format="PNG")
    page.insert_image(fitz.Rect(72, 110, 472, 610), stream=buf.getvalue())
    doc.save(str(path))
    doc.close()
    return path


def test_a_bad_middle_page_does_not_cost_later_pages_their_figures(tmp_path):
    """The extractor used to load the page BEFORE its skip check, and its
    document-wide catch ended the loop: every later page lost its figures."""
    from socr.figures.extractor import FigureExtractor

    pdf = _figure_pdf(tmp_path / "f.pdf")
    clean = FigureExtractor().extract(pdf)
    assert [f.page_num for f in clean.figures] == [PAGES], "control: the figure page yields"
    with _page_load_raises(BAD_INDEX):
        damaged = FigureExtractor().extract(pdf)
    assert [f.page_num for f in damaged.figures] == [f.page_num for f in clean.figures]
    # A skipped bad page is never even REQUESTED. Recording accesses is what
    # distinguishes "skip, then load" from "load (guarded), then skip", which
    # return the same figures.
    requested: list[int] = []
    real_get, real_load = fitz.Document.__getitem__, fitz.Document.load_page

    def get_spy(self, i):
        requested.append(i)
        return real_get(self, i)

    def load_spy(self, page_id=0, *a, **kw):
        requested.append(page_id)
        return real_load(self, page_id, *a, **kw)

    with (
        patch.object(fitz.Document, "__getitem__", get_spy),
        patch.object(fitz.Document, "load_page", load_spy),
    ):
        FigureExtractor().extract(pdf, skip_pages={BAD_PAGE})
        assert BAD_INDEX not in requested
        requested.clear()
        FigureExtractor().extract(pdf)  # control: unskipped, the spy sees the page
        assert BAD_INDEX in requested


@pytest.mark.parametrize("removed", [PAGES - 1, BAD_INDEX], ids=["last", "middle"])
def test_a_repair_that_shrinks_the_page_count_fails_the_missing_page(
    tmp_path, monkeypatch, removed
):
    """Glyph recovery touches pages and MuPDF may repair the tree, shrinking
    ``len(doc)``. The page that vanished must be FAILED -- and it must be THAT page:
    when a middle page goes, the survivors shift down, and reading by index would
    put page 3's text under page 2's number."""
    pdf = _pdf(tmp_path / "s.pdf")
    det = BornDigitalDetector()
    monkeypatch.setattr(det, "_recover_symbol_fonts", lambda doc, path: doc.delete_page(removed))
    assessment = det.detect(pdf)
    assert [p.page_num for p in assessment.pages] == list(range(1, PAGES + 1))
    for p in assessment.pages:
        if p.page_num == removed + 1:
            assert "missing after repair" in p.load_error
            assert p.native_text == ""
        else:
            assert p.load_error == ""
            # identity: the text on page N came from source page N. The native layer
            # is tabularised, so compare the page-number tokens, not the raw string.
            text = " ".join(p.native_text.replace("|", " ").split())
            assert f"Page {p.page_num} coefficient" in text, text[:80]
            assert all(
                f"Page {q} coefficient" not in text for q in range(1, PAGES + 1) if q != p.page_num
            )

    # control: a recovery that keeps the count leaves every page loadable
    det2 = BornDigitalDetector()
    monkeypatch.setattr(det2, "_recover_symbol_fonts", lambda doc, path: None)
    assert all(p.load_error == "" for p in det2.detect(pdf).pages)


def test_with_an_empty_provider_ladder_the_bad_page_is_still_failed(tmp_path):
    """CI has no provider. Parametrised by difference: the bad page's outcome and
    the document verdict must not depend on the ladder."""
    results = {}
    for ladder in ([providers.PROFILE_QWEN_LOCAL], []):
        pdf = _pdf(tmp_path / f"l{len(ladder)}.pdf")
        out = tmp_path / f"out_l{len(ladder)}"
        pipe = _pipeline()
        pipe._available_engines_for_agentic = lambda ladder=ladder: ladder
        with _page_load_raises(BAD_INDEX):
            res = pipe.process(pdf, output_dir=out)
        side = _sidecar(_doc_dir(out), BAD_PAGE)
        meta = json.loads((_doc_dir(out) / "metadata.json").read_text())
        results[len(ladder)] = (side["status"], side["failure_mode"], meta["status"])
    assert results[0] == results[1]
    assert results[0][0] == PageStatus.ERROR.value
    assert results[0][1] == FailureMode.UNREADABLE_INPUT.value
    assert results[0][2] == "partial"


def _detect_without_the_catch(self, pdf_path):
    """``BornDigitalDetector.detect`` exactly as it was before #881: no per-page
    guard, no declared-count reconciliation."""
    from socr.core.born_digital import DocumentAssessment
    from socr.core.pdf import open_pdf

    pdf_path = Path(pdf_path)
    pages = []
    with open_pdf(pdf_path, repair=False) as doc:
        self._recover_symbol_fonts(doc, pdf_path)
        for page_idx in range(len(doc)):
            pages.append(self._assess_page(doc[page_idx], page_idx + 1))
        self._mark_unrecovered_glyphs(pages)
    return DocumentAssessment(path=pdf_path, pages=pages)


def test_an_undamaged_document_is_byte_identical_with_the_catch_inert(tmp_path, monkeypatch):
    """Same process, same inputs, one variable: the new per-page catch. Frozen
    hashes would pin formatting that legitimately changes; a difference does not."""
    _, _, out_new, new = _run(tmp_path, None, "ok")
    monkeypatch.setattr(BornDigitalDetector, "detect", _detect_without_the_catch)
    _, _, out_old, old = _run(tmp_path, None, "old")
    assert new.status is old.status
    dd_new, dd_old = _doc_dir(out_new), _doc_dir(out_old)
    assert (dd_new / "ok.md").read_bytes() == (dd_old / "old.md").read_bytes()
    for n in range(1, PAGES + 1):
        assert (dd_new / "pages" / f"{n:05d}.md").read_bytes() == (
            dd_old / "pages" / f"{n:05d}.md"
        ).read_bytes()
    # the pin is live: a different document body would show up
    assert "Page 1" in (dd_new / "ok.md").read_text() or "ocr" in (dd_new / "ok.md").read_text()


def test_whole_document_text_never_wins_for_an_unloadable_page():
    """A whole-doc engine's split section for the bad page is of untrustworthy
    alignment. Difference: the SAME state with and without ``load_error``."""
    from socr.core.document import DocumentHandle
    from socr.core.manifest import _whole_doc_page_texts, finalized_page_record
    from socr.core.state import DocumentState

    def record(load_error: str):
        state = DocumentState(
            handle=DocumentHandle(path=Path("x.pdf"), page_count=2, page_count_known=True)
        )
        state.whole_doc_attempts.append(
            PageOutput(
                page_num=0,
                text="## Page 1\n\nfirst words\n\n## Page 2\n\nsecond words\n",
                status=PageStatus.SUCCESS,
                engine="cli",
                audit_passed=True,
            )
        )
        state.pages[2].load_error = load_error
        return finalized_page_record(state, 2, _whole_doc_page_texts(state)).output

    healthy = record("")
    assert "second words" in healthy.text and healthy.status is PageStatus.SUCCESS  # control
    bad = record("FzErrorFormat: broken")
    assert "second words" not in bad.text
    assert bad.status is PageStatus.ERROR
    assert bad.failure_mode is FailureMode.UNREADABLE_INPUT
