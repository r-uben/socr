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


# ---------------------------------------------------------------------------
# Review round 2 on PR #1168 (Fable): the gate had three sites and I fixed one
# ---------------------------------------------------------------------------


def _library(tmp_path: Path):
    """A real library config, and the stub-producing process the stub lane now leaves."""
    import json as _json

    import yaml
    from test_gh964_library import _cfg_dict

    from socr import library as lib

    root = tmp_path / "lib"
    root.mkdir(exist_ok=True)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg_dict(root)))
    cfg = lib.load_library_config(cfg_path)

    def failed_process(pdf: Path, out_root: Path):
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

    return lib, cfg, failed_process


def test_promote_never_archives_real_text_for_a_failure_stub(tmp_path: Path) -> None:
    """The route my first fix left open, end to end: rerun then promote.

    I gated ``install_staged`` and stopped there. ``staged_stems`` and ``check_promote`` ask
    the SAME question in their own words, so ``--rerun`` on a now-unreadable PDF staged the
    stub, the CLI announced "awaiting approval: socr library --promote <stem>", and promote
    ARCHIVED the real ``text/<stem>/`` and installed the stub over it. The approval was
    invited by socr and the destruction was the approved action.

    The assertion that matters is the last one: the real text is still readable at the end.
    """
    from test_gh964_library import _fake_doc, _pdf

    lib, cfg, failed_process = _library(tmp_path)
    pdf = _pdf(cfg.pdf_dir / "a.pdf")
    _fake_doc(cfg.text_dir, "a", body="## Page 1\n\nReal text worth keeping.\n")
    precious = (cfg.text_dir / "a" / "a.md").read_text()

    # The rerun stages a failure, exactly as the lane now does.
    failed_process(pdf, cfg.staging_dir)

    assert lib.staged_stems(cfg) == [], (
        "a failure stub is offered for approval; the CLI would invite a destructive promote"
    )
    with pytest.raises(lib.LibraryError, match="nothing staged"):
        lib.check_promote(cfg, "a")
    with pytest.raises(lib.LibraryError, match="nothing staged"):
        lib.promote(cfg, "a")

    assert (cfg.text_dir / "a" / "a.md").read_text() == precious, "the real text was replaced"
    assert not list(cfg.text_dir.glob("a.archived*")), "the real text was archived away"

    # Positive control: a staged run WITH real text is still offered and still promotes.
    _fake_doc(cfg.staging_dir, "b", body="## Page 1\n\nNew text.\n", status="partial")
    assert lib.staged_stems(cfg) == ["b"], lib.staged_stems(cfg)
    assert lib.check_promote(cfg, "b").name == "b"


def test_has_text_does_not_count_a_stub_as_text(tmp_path: Path) -> None:
    """The fourth site, found by auditing every ``markdown_path(...).is_file()`` here.

    ``has_text`` decides whether ``pending_pdfs`` still offers the PDF. A stub installed by
    a run from before the gate would read as "done" and the document would never be looked
    at again. With the gate such a stem has a text dir and no installable text, which
    surfaces as BLOCKED -- the right end state for a directory nobody can explain.
    """
    from test_gh964_library import _fake_doc, _pdf

    lib, cfg, _ = _library(tmp_path)
    _pdf(cfg.pdf_dir / "a.pdf")
    _fake_doc(
        cfg.text_dir, "a", status="failed", body="[document failed: unreadable_input: 0 of 64]\n"
    )

    assert lib.has_text(cfg, "a") is False, "a failure stub counts as the library's text"
    todo, blocked = lib.pending_pdfs(cfg)
    assert [p.stem for p in blocked] == ["a"], (todo, blocked)

    # Control: real text is text, whatever the status says.
    _fake_doc(cfg.text_dir, "b", status="partial", body="## Page 1\n\nReal text.\n")
    _pdf(cfg.pdf_dir / "b.pdf")
    assert lib.has_text(cfg, "b") is True


def test_a_rerun_rewrites_its_own_previous_stub(tmp_path: Path) -> None:
    """A FAILED record is retried every batch, so run 2 meets run 1's stub.

    The overwrite refusal was "a markdown file already exists", which is right for real text
    and wrong for this code's own output: the writer returned False, ``output_path`` went
    empty, and the WARNING claimed existing text was kept when there was none -- a log line
    that sends someone looking for text that does not exist.
    """
    pdf = _unreadable_pdf(tmp_path / "broken.pdf")
    out_dir = tmp_path / "out"
    pipeline = _pipeline()
    pipeline._scan_root = pdf.parent

    pipeline.process(pdf, output_dir=out_dir)
    first = _doc_metadata(out_dir).get("output_path") or ""
    assert first, "run 1 did not record the stub"

    pipeline.process(pdf, output_dir=out_dir)
    second = _doc_metadata(out_dir).get("output_path") or ""
    assert second == first, (
        "run 2 refused to rewrite its own stub, so output_path went empty and the record "
        "stopped naming the mark that is sitting right there"
    )
    body = Path(second).read_text()
    assert DOCUMENT_FAILED_MARK_RE.fullmatch(body.strip()), body
    assert body.count("\n") == 1, f"the stub was appended to rather than rewritten: {body!r}"


def test_real_text_that_merely_quotes_the_grammar_is_still_protected(tmp_path: Path) -> None:
    """``fullmatch`` on the whole file, not a search.

    A paper that quotes a failure mark, or a hand-written note beside one, is real content.
    Recognising the stub by substring would let the writer destroy it -- which is the exact
    failure the refusal exists to prevent, reintroduced by the fix to the refusal.
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
    quoting = (
        "## Page 1\n\nThe tool emits `[document failed: unreadable_input: 0 of 2]` in that case.\n"
    )
    md.write_text(quoting)

    assert pipeline._write_document_failed_mark(pdf, out_dir, "transient") is False
    assert md.read_text() == quoting, "real text quoting the grammar was destroyed"


# ---------------------------------------------------------------------------
# Review round 3 on PR #1168 (Fable)
# ---------------------------------------------------------------------------


def test_a_lost_metadata_write_still_cannot_install_the_stub(tmp_path: Path) -> None:
    """The ordering gap: the stub's refusal evidence lived in a different file.

    ``_write_metadata`` swallows every exception on purpose -- a metadata write must never
    lose OCR output -- so a lost write, or a process killed between the two writes, left the
    stub on disk with no ``failed`` status beside it. ``has_installable_text`` then read a
    markdown file with no refusal evidence and installed it.

    Two independent closures, and this test drives the second with the first disabled:
    metadata is now written FIRST, and the gate ALSO refuses content that is nothing but the
    mark. Either alone would do; both means neither a write order nor a file is load-bearing
    on its own.
    """
    import json as _json

    from socr import library as lib

    lib_mod, cfg, _ = _library(tmp_path)
    staged = cfg.staging_dir / "a"
    staged.mkdir(parents=True)
    (staged / "a.md").write_text("[document failed: unreadable_input: 0 of 64]\n")
    # The lost write: no metadata at all, which is exactly what the swallowed exception
    # leaves behind.
    assert not (staged / "metadata.json").exists()

    assert lib.has_installable_text(cfg, staged, "a") is False, (
        "the stub is installable when its metadata is missing; the content gate is not firing"
    )
    with pytest.raises(lib.LibraryError):
        lib.install_staged(cfg, "a")
    assert lib.staged_stems(cfg) == []

    # Control: the same directory with real text and no metadata IS installable, so the
    # content gate is not simply refusing everything that lacks metadata.
    (staged / "a.md").write_text("## Page 1\n\nReal text.\n")
    assert lib.has_installable_text(cfg, staged, "a") is True
    (staged / "metadata.json").write_text(_json.dumps({"status": "completed"}))
    assert lib.staged_stems(cfg) == ["a"]


def test_the_failure_record_is_written_before_the_stub(tmp_path: Path) -> None:
    """Order, asserted directly, because it is the first of the two closures.

    If the stub is written first, there is a window in which it exists with no ``failed``
    status beside it. Observed through the write order rather than argued from the source.
    """
    pdf = _unreadable_pdf(tmp_path / "broken.pdf")
    out_dir = tmp_path / "out"
    pipeline = _pipeline()
    pipeline._scan_root = pdf.parent

    order: list[str] = []
    real_meta = type(pipeline)._write_metadata
    real_mark = type(pipeline)._write_document_failed_mark

    def spy_meta(self, *a, **k):
        order.append("metadata")
        return real_meta(self, *a, **k)

    def spy_mark(self, *a, **k):
        order.append("mark")
        return real_mark(self, *a, **k)

    import socr.pipeline.orchestrator as orch

    try:
        orch.UnifiedPipeline._write_metadata = spy_meta
        orch.UnifiedPipeline._write_document_failed_mark = spy_mark
        pipeline.process(pdf, output_dir=out_dir)
    finally:
        orch.UnifiedPipeline._write_metadata = real_meta
        orch.UnifiedPipeline._write_document_failed_mark = real_mark

    assert order[:2] == ["metadata", "mark"], (
        f"the stub is written before its failure record: {order}"
    )
    # And the record still ends up naming the stub, so ordering did not cost the pointer.
    assert _doc_metadata(out_dir).get("output_path") or "", order


def test_a_long_error_detail_does_not_break_the_rewrite(tmp_path: Path) -> None:
    """The dropped ``4096`` cap, which was a magic number that did real harm.

    A detail longer than it made the next run fail to recognise its own stub, refuse to
    rewrite it, and log that existing text had been kept when there was none. Measured on
    the corpus: every ``unreadable_input`` detail is 103 characters, and the longest
    document error socr has ever recorded is 2,906 -- so the cap sat below real data.
    """
    pdf = _unreadable_pdf(tmp_path / "broken.pdf")
    out_dir = tmp_path / "out"
    pipeline = _pipeline()
    pipeline._scan_root = pdf.parent

    long_detail = "unreadable_input: " + ("malformed page tree " * 400)
    assert len(long_detail) > 4096

    assert pipeline._write_document_failed_mark(pdf, out_dir, long_detail) is True
    # The second run must recognise its own stub and rewrite it, cap or no cap.
    assert pipeline._write_document_failed_mark(pdf, out_dir, long_detail) is True

    mds = list(out_dir.rglob("*.md"))
    assert len(mds) == 1, mds
    body = mds[0].read_text()
    assert body.count("\n") == 1, "the stub was appended to rather than rewritten"
    assert DOCUMENT_FAILED_MARK_RE.fullmatch(body.strip()), body[:120]


def test_the_blocked_reason_says_what_is_actually_wrong(tmp_path: Path) -> None:
    """The CLI said "text directory exists without <stem>.md" when it was sitting there.

    True while the only way to be blocked was a missing file; false once the gate also
    refuses a document whose markdown is a failure mark -- and it sent the reader looking
    for the wrong problem. One row per way a stem can be blocked.
    """
    from test_gh964_library import _fake_doc, _pdf

    lib, cfg, _ = _library(tmp_path)

    # (a) no markdown at all: the original message, still correct.
    _pdf(cfg.pdf_dir / "a.pdf")
    (cfg.text_dir / "a").mkdir(parents=True)
    (cfg.text_dir / "a" / "notes.txt").write_text("hand edited")
    assert "without" in lib.blocked_reason(cfg, "a")

    # (b) markdown that IS a failure mark: must not claim the file is absent.
    _pdf(cfg.pdf_dir / "b.pdf")
    _fake_doc(
        cfg.text_dir, "b", status="failed", body="[document failed: unreadable_input: 0 of 2]\n"
    )
    reason_b = lib.blocked_reason(cfg, "b")
    assert "document-failure mark" in reason_b, reason_b
    assert "without" not in reason_b, f"the message claims the file is absent: {reason_b}"

    # (c) real text with a failed record: names the record, not the file.
    _pdf(cfg.pdf_dir / "c.pdf")
    _fake_doc(cfg.text_dir, "c", status="failed", body="## Page 1\n\nSome text.\n")
    reason_c = lib.blocked_reason(cfg, "c")
    assert "status=failed" in reason_c, reason_c


def test_both_layers_of_the_gate_refuse_on_their_own(tmp_path: Path) -> None:
    """Defence in depth is only depth if each layer is load-bearing by itself.

    Adding the content check made the STATUS check individually untestable by every existing
    test: a stub is caught by either, so deleting the status check left them all green (the
    ``shared-gate-ignores-the-status`` mutant SURVIVED, which is how this gap was found). An
    untested layer is not a second line of defence, it is unexamined code.

    So one case per layer, each of which only that layer can refuse:

    * real TEXT with a ``failed`` record -- the content check sees ordinary markdown and
      passes it; only the status refuses. This is a real shape, not a contrivance: a run
      that produced text and then failed records exactly it.
    * a stub with NO record at all -- the status check has nothing to read; only the content
      refuses. This is the lost-metadata window.
    """
    import json as _json

    from socr import library as lib

    _, cfg, _ = _library(tmp_path)

    # Layer 1 alone: text, but the run recorded failure.
    a = cfg.staging_dir / "a"
    a.mkdir(parents=True)
    (a / "a.md").write_text("## Page 1\n\nOrdinary text with no mark in it.\n")
    (a / "metadata.json").write_text(_json.dumps({"status": "failed", "error": "boom"}))
    from socr.core.marks import DOCUMENT_FAILED_MARK_RE as _re

    assert _re.search((a / "a.md").read_text()) is None, "the fixture is stub-shaped; retry"
    assert lib.has_installable_text(cfg, a, "a") is False, (
        "only the STATUS check can refuse this one, and it did not"
    )

    # Layer 2 alone: a stub, with no record to read.
    b = cfg.staging_dir / "b"
    b.mkdir(parents=True)
    (b / "b.md").write_text("[document failed: unreadable_input: 0 of 2]\n")
    assert not (b / "metadata.json").exists()
    assert lib.has_installable_text(cfg, b, "b") is False, (
        "only the CONTENT check can refuse this one, and it did not"
    )

    # And the control, so neither layer is simply refusing everything.
    c = cfg.staging_dir / "c"
    c.mkdir(parents=True)
    (c / "c.md").write_text("## Page 1\n\nOrdinary text.\n")
    (c / "metadata.json").write_text(_json.dumps({"status": "partial"}))
    assert lib.has_installable_text(cfg, c, "c") is True
