"""#1145 slice 2: a document socr could not read says so where a reader looks.

Design §4 and §7, test T10. ``_refuse_unreadable_input`` (#871) records the failure in
``metadata.json`` and the root index, which is what that ticket fixed -- but it wrote no
``.md`` at all and left ``output_path`` empty, so a reader or a downstream tool that opens
the document's markdown found nothing. "No file" is indistinguishable from "not processed
yet". This is the document-level analogue of the page rule: the whole artifact is the mark.

Hermeticity is NOT by construction here, and the first draft of this file claimed it was.
``process`` reaches ``_resolve_judge_model`` through ``_run_fingerprint`` before it ever
looks at the input, so it probes the configured Ollama host even for a file ``fitz`` cannot
open -- the exact trap CLAUDE.md names, caught by the repo's own GH-984 connection guard
rather than by review. So ``_resolve_judge_model`` returns "" and the engines are pinned,
and no page loop runs because the refusal returns first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fitz")

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.marks import DOCUMENT_FAILED_MARK_RE  # noqa: E402
from socr.core.result import DocumentStatus, FailureMode  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402


def _pipeline() -> UnifiedPipeline:
    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            agentic=True,
            quiet=True,
            write_manifest=False,
        )
    )
    # ``_run_fingerprint`` resolves the judge model, which probes the host. Pinned to "" so
    # the run cannot depend on whether this machine has a judge model pulled -- and so it
    # cannot reach a daemon at all, which CI does not have.
    pipeline._resolve_judge_model = lambda *a, **k: ""
    return pipeline


def _unreadable_pdf(path: Path) -> Path:
    """A file with a PDF name and no loadable page. The refusal's own precondition."""
    path.write_bytes(b"%PDF-1.4\nnot a page tree at all\n%%EOF\n")
    return path


def _doc_metadata(out_dir: Path) -> dict:
    found = [p for p in out_dir.rglob("metadata.json") if p.parent != out_dir]
    assert len(found) == 1, found
    return json.loads(found[0].read_text())


def test_an_unreadable_document_ships_a_document_failed_mark(tmp_path: Path) -> None:
    """T10. The file exists, it is the mark, and ``output_path`` names it.

    Three assertions, because any two can hold while the thing is still broken: a file
    that exists but says nothing, a mark that is there but unreachable because
    ``output_path`` is empty, or a path recorded for a file that was never written.
    """
    pdf = _unreadable_pdf(tmp_path / "broken.pdf")
    out_dir = tmp_path / "out"
    pipeline = _pipeline()
    pipeline._scan_root = pdf.parent

    result = pipeline.process(pdf, output_dir=out_dir)

    assert result.status is DocumentStatus.ERROR
    assert result.failure_mode is FailureMode.UNREADABLE_INPUT

    meta = _doc_metadata(out_dir)
    recorded = meta.get("output_path") or ""
    assert recorded, "the document's markdown path is still empty, so the mark is unreachable"

    md = Path(recorded)
    assert md.is_file(), f"output_path names a file that was not written: {recorded}"
    body = md.read_text()
    assert DOCUMENT_FAILED_MARK_RE.search(body), f"the document mark is not in {body!r}"
    assert FailureMode.UNREADABLE_INPUT.value in body, (
        "the mark does not say what went wrong, so the reader learns nothing from it"
    )


def test_the_refusal_still_records_a_FAILED_document(tmp_path: Path) -> None:
    """The mark must not buy itself a softer status.

    Writing a markdown file is exactly the kind of change that quietly turns FAILED into
    PARTIAL -- ``_write_metadata`` derives both the status and ``output_path`` from
    ``has_text``, and #1145 had to separate them. If the status softened, the resume gate
    would stop refusing the entry and the next run would never retry the file: a document
    socr cannot read would be recorded as done, for ever.
    """
    pdf = _unreadable_pdf(tmp_path / "broken.pdf")
    out_dir = tmp_path / "out"
    pipeline = _pipeline()
    pipeline._scan_root = pdf.parent
    pipeline.process(pdf, output_dir=out_dir)

    meta = _doc_metadata(out_dir)
    assert meta["status"] == "failed", meta["status"]
    assert meta.get("pages") == 0 or isinstance(meta.get("pages"), int)
    assert meta.get("error"), "the recorded error went missing"


def test_the_mark_is_one_line_and_cannot_be_ended_early_by_the_error_text(
    tmp_path: Path,
) -> None:
    """The grammar's ``]`` terminator, on text socr does not author.

    The detail interpolates a PyMuPDF error string. A ``]`` in it would end the mark early
    and the remainder would read as prose, so a reader -- and the recogniser -- would see a
    truncated mark followed by noise. Driven through the real writer with a detail that
    contains one, rather than asserting on the regex alone.
    """
    pdf = _unreadable_pdf(tmp_path / "broken.pdf")
    out_dir = tmp_path / "out"
    pipeline = _pipeline()
    pipeline._scan_root = pdf.parent

    nasty = "boom ] and [ more"
    assert pipeline._write_document_failed_mark(pdf, out_dir, nasty) is True

    meta_md = [p for p in out_dir.rglob("*.md")]
    assert len(meta_md) == 1, meta_md
    body = meta_md[0].read_text()
    assert body.count("\n") == 1, f"the mark is not one line: {body!r}"
    found = DOCUMENT_FAILED_MARK_RE.findall(body)
    assert len(found) == 1, f"expected exactly one mark, got {found!r}"
    assert found[0].endswith("more]"), f"the mark was ended early: {found[0]!r}"
    assert "]" not in found[0][:-1], f"an unescaped terminator survived: {found[0]!r}"


def test_a_write_failure_leaves_output_path_empty_rather_than_lying(tmp_path: Path) -> None:
    """Fail closed: name the file only when the file is there.

    The refusal must survive a disk it cannot write to -- a document that cannot be written
    must still be REFUSED and recorded rather than crashing the batch -- but it must not
    then record a path to a file that does not exist, which would send every downstream
    reader to a missing file and look like corruption rather than a failure.
    """
    pdf = _unreadable_pdf(tmp_path / "broken.pdf")
    out_dir = tmp_path / "out"
    pipeline = _pipeline()
    pipeline._scan_root = pdf.parent

    import socr.pipeline.orchestrator as orch

    original = orch.UnifiedPipeline._write_document_failed_mark
    try:
        orch.UnifiedPipeline._write_document_failed_mark = lambda self, *a, **k: False
        result = pipeline.process(pdf, output_dir=out_dir)
    finally:
        orch.UnifiedPipeline._write_document_failed_mark = original

    assert result.status is DocumentStatus.ERROR, "the refusal itself stopped working"
    meta = _doc_metadata(out_dir)
    assert (meta.get("output_path") or "") == "", (
        "output_path names a markdown file that was never written"
    )


def test_every_other_caller_of_write_metadata_is_unchanged(tmp_path: Path) -> None:
    """``markdown_written`` defaults to ``has_text``, so nothing else moved.

    The parameter exists for one caller. A default that changed any other call site's
    ``output_path`` would be a silent behaviour change across every document socr writes,
    which is exactly the kind of thing a new keyword argument does by accident.
    """
    import inspect

    sig = inspect.signature(UnifiedPipeline._write_metadata)
    assert sig.parameters["markdown_written"].default is None, sig

    pdf = _unreadable_pdf(tmp_path / "broken.pdf")
    pipeline = _pipeline()
    pipeline._scan_root = pdf.parent

    # ``has_text`` alone still decides, both ways, when the new argument is not passed.
    from socr.core.document import DocumentHandle
    from socr.core.result import EngineResult
    from socr.core.state import DocumentState

    for has_text in (True, False):
        out_dir = tmp_path / f"o-{has_text}"
        state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1, page_count_known=True))
        pipeline._write_metadata(
            state,
            EngineResult(
                document_path=pdf,
                engine="qwen",
                status=DocumentStatus.SUCCESS if has_text else DocumentStatus.ERROR,
            ),
            out_dir,
            has_text=has_text,
        )
        recorded = (_doc_metadata(out_dir).get("output_path") or "") != ""
        assert recorded is has_text, (has_text, recorded)


# ---------------------------------------------------------------------------
# Review round 1 on PR #1168 (Fable): three ways the stub reached further than
# it should have
# ---------------------------------------------------------------------------


def test_an_unreadable_document_is_not_installed_into_the_library(tmp_path: Path) -> None:
    """The regression this stub introduced, driven through ``process_new``.

    ``install_staged`` gated on "a markdown file is here", which until slice 2 doubled as
    "socr got something out of this document": an unreadable PDF produced no markdown, so
    install raised, the stem stayed in ``missing_text``, and the next run re-queued it. The
    stub satisfied that file check, so the document INSTALLED, was reported as Installed,
    left ``pending_pdfs`` for ever, and landed in ``unverified`` beside documents that do
    have text. The status is now read from what the pipeline RECORDED.

    The positive control is the second half: a ``partial`` run that produced real text must
    still install, because that is the library's own rule (best available text, listed
    unverified). Without it this test would pass on an install that refuses everything.
    """
    import json as _json

    import yaml
    from test_gh964_library import _cfg_dict, _pdf

    from socr import library as lib

    root = tmp_path / "lib"
    root.mkdir()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg_dict(root)))
    cfg = lib.load_library_config(cfg_path)

    def failed_process(pdf: Path, out_root: Path):
        """What slice 2 now leaves behind for an unreadable PDF: a stub and a FAILED record."""
        d = out_root / pdf.stem
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{pdf.stem}.md").write_text("[document failed: unreadable_input: 0 of 64]\n")
        (d / "metadata.json").write_text(
            _json.dumps({"status": "failed", "error": "unreadable_input: 0 of 64", "pages": 64})
        )

        class R:
            success = False
            status = None

        return R()

    pdf = _pdf(cfg.pdf_dir / "broken.pdf")
    todo, _ = lib.pending_pdfs(cfg)
    assert [p.stem for p in todo] == ["broken"]

    out = lib.process_new(cfg, failed_process, todo)
    states = [(stem, state) for stem, state, _ in out]
    assert states == [("broken", lib.FAILED)], states
    assert not (cfg.text_dir / "broken").exists(), (
        "an unreadable document was installed as library text"
    )

    # And it is still queued, which is the behaviour the stub must not change.
    again, _ = lib.pending_pdfs(cfg)
    assert [p.stem for p in again] == ["broken"], "the document was dropped from the work list"

    # The control: real text with a non-clean status still installs.
    from test_gh964_library import _fake_process

    ok_pdf = _pdf(cfg.pdf_dir / "partial.pdf")
    out2 = lib.process_new(cfg, _fake_process(None, status="partial"), [ok_pdf])
    assert [(s, st) for s, st, _ in out2] == [("partial", lib.INSTALLED)], out2
    assert (cfg.text_dir / "partial" / "partial.md").is_file()


def test_a_newline_in_the_error_text_cannot_un_mark_the_document(tmp_path: Path) -> None:
    """``DOCUMENT_FAILED_MARK_RE`` rejects ``\\r`` and ``\\n``, so a multi-line detail
    produces a file with NO recognised mark in it -- the document counts
    ``document_failed_unmarked`` exactly as it did before slice 2, which is the whole thing
    slice 2 removes. Escaping only ``]`` is how this was missed: the detail interpolates a
    PyMuPDF error string socr does not author, and it is free to contain either.
    """
    pdf = _unreadable_pdf(tmp_path / "broken.pdf")
    out_dir = tmp_path / "out"
    pipeline = _pipeline()
    pipeline._scan_root = pdf.parent

    multiline = "cannot open:\n  broken xref\r\n  at object 12"
    assert pipeline._write_document_failed_mark(pdf, out_dir, multiline) is True

    mds = list(out_dir.rglob("*.md"))
    assert len(mds) == 1, mds
    body = mds[0].read_text()
    assert body.count("\n") == 1, f"the mark spans lines: {body!r}"
    found = DOCUMENT_FAILED_MARK_RE.findall(body)
    assert len(found) == 1, f"a multi-line detail left no recognised mark: {body!r}"
    for word in ("broken", "xref", "object"):
        assert word in found[0], f"{word} was lost from the flattened detail: {found[0]!r}"


def test_the_stub_never_overwrites_an_earlier_runs_text(tmp_path: Path) -> None:
    """Never silently destroy prior content, and never leave a stale mix.

    ``probe_page_loads`` can fail transiently, and this path runs before anything else
    looks at the input. Overwriting replaced an earlier successful run's real text with a
    one-line failure mark while its ``pages/`` stayed on disk, so the document became a
    stale MIX the meter reports as ``document_not_assembly`` -- restored next run, so a
    window rather than a loss, but silent, and of the only copy in that directory.
    """
    from ocr_output_contract import doc_dir_for, markdown_path_for, relative_key

    pdf = _unreadable_pdf(tmp_path / "broken.pdf")
    out_dir = tmp_path / "out"
    pipeline = _pipeline()
    pipeline._scan_root = pdf.parent

    rel = relative_key(pdf, pdf.parent)
    doc_dir = doc_dir_for(out_dir, rel)
    doc_dir.mkdir(parents=True, exist_ok=True)
    md = markdown_path_for(doc_dir, rel)
    precious = "## Page 1\n\nReal text from a run that worked.\n"
    md.write_text(precious)
    (doc_dir / "pages").mkdir(exist_ok=True)
    (doc_dir / "pages" / "00001.md").write_text("Real text from a run that worked.")

    assert pipeline._write_document_failed_mark(pdf, out_dir, "transient open failure") is False, (
        "the writer reported success while refusing to write, so output_path would lie"
    )
    assert md.read_text() == precious, "an earlier run's text was overwritten by the stub"

    # And the refusal propagates: the record must not name a stub that was not written.
    result = pipeline.process(pdf, output_dir=out_dir)
    assert result.failure_mode is FailureMode.UNREADABLE_INPUT
    assert md.read_text() == precious, "process() overwrote the text on the refusal path"
    assert (_doc_metadata(out_dir).get("output_path") or "") == "", (
        "output_path names a stub that was never written"
    )
