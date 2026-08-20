"""GH-225: an invented image URL must not ship as page content under SUCCESS.

Two pages of the OBR Nov-2022 EFO run carried
``![](https://i.imgur.com/1234567.png)`` — refs the model produced to fill a
figure slot.  The source is a UK government fiscal document with no external
image links at all.  They shipped with ``status: success``,
``judge_rejected: false``, and page 26 recorded zero audit events.

They survive because a hyperlink is text.  Every gate the pipeline has checks
table geometry, numeric-token preservation, word recall or structural shape, and
``OutputNormalizer.strip_phantom_images`` returns early on absolute URLs by
design — a pure text normalizer has no view of the source and so cannot tell a
real external image from an invented one.

Hermetic: drives ``_sanitize_agentic_page_image_refs`` directly, so no provider
ladder, no ``_phase_agentic``, no ``_available_engines_for_agentic`` patch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.manifest import _winning_page_output, canonical_page_texts
from socr.core.result import DocumentStatus, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline

# The exact ref observed on page 26 of the reference run.
FABRICATED_REF = "![](https://i.imgur.com/1234567.png)"


def _make_pipeline() -> UnifiedPipeline:
    return UnifiedPipeline(
        PipelineConfig(
            agentic=True,
            judge_backend="heuristic",
            enabled_engines=[EngineType.GEMINI],
            primary_engine=EngineType.DEEPSEEK,
            save_figures=False,
            dual_pass_tables=False,
            detect_equations=False,
            recover_clean_equations=False,
            quiet=True,
            audit_enabled=False,
            write_manifest=False,
        )
    )


def _pdf_without_links(tmp_path: Path) -> Path:
    """A PDF like the OBR source: real prose, no link annotations, no URLs."""
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "no_links.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Public sector net borrowing forecast " * 6)
    doc.save(str(path))
    doc.close()
    return path


def _pdf_with_link(tmp_path: Path, url: str) -> Path:
    """A PDF that really does carry *url* as a clickable link annotation."""
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "with_link.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "See the chart online " * 6)
    page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(72, 60, 300, 80), "uri": url})
    doc.save(str(path))
    doc.close()
    return path


def _shipped_body(state: DocumentState) -> str:
    """What actually reaches the saved ``.md`` and ``pages/NNN.md``.

    ``canonical_page_texts`` is the SINGLE source of truth for both (see its
    docstring); it runs ``_winning_page_output`` per page. Asserting here rather
    than on the ``PageOutput`` the gate mutated is the difference between
    testing the predicate and testing the corpus — a fix that strips too much,
    or that loses the winner to the native fallback, passes the former and fails
    the latter.
    """
    return "\n\n".join(canonical_page_texts(state))


def _state_for(
    pdf_path: Path, text: str, *, born_digital: bool = False, native_text: str = ""
) -> tuple[DocumentState, PageOutput]:
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    out = PageOutput(
        page_num=1,
        text=text,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    ps = state.pages[1]
    ps.is_born_digital = born_digital
    ps.native_text = native_text
    ps.attempts.append(out)
    ps.best_output = out
    return state, out


def test_fabricated_image_url_is_removed_demoted_and_surfaced(tmp_path: Path) -> None:
    pdf_path = _pdf_without_links(tmp_path)
    doc_dir = tmp_path / "out"
    doc_dir.mkdir()

    body = f"## Public finances\n\nBorrowing falls over the forecast.\n\n{FABRICATED_REF}\n"
    state, out = _state_for(pdf_path, body)

    pipeline = _make_pipeline()
    pipeline._sanitize_agentic_page_image_refs(state, 1, out, doc_dir)

    # 1. The invented reference does not ship.
    assert "i.imgur.com" not in out.text, out.text
    # 2. Real page content is untouched — this is a fabrication fix, not a
    #    content-stripping one.
    assert "Borrowing falls over the forecast." in out.text
    assert "## Public finances" in out.text
    # 3. Its removal is marked in the page body, not silent.
    assert "fabricated image reference removed" in out.text, out.text
    # 4. The page no longer ships under SUCCESS.
    assert out.status != PageStatus.SUCCESS, out.status
    # audit_passed MUST stay True. In this codebase it is the winner-selection
    # flag, not a "flag this page" flag: _winning_page_output returns
    # best_output only while it is True. Flipping it made assemble discard the
    # cleaned OCR page and ship flattened native text under a fresh SUCCESS —
    # a worse content loss than the fabrication. See
    # test_born_digital_page_keeps_its_ocr_table_after_redaction.
    assert out.audit_passed is True
    # 5. An audit event carries the invented target for forensics.
    events = [e for e in state.events if e.kind == "fabricated_image_ref"]
    assert len(events) == 1, state.events
    assert events[0].page_num == 1
    assert "https://i.imgur.com/1234567.png" in events[0].data["targets"]
    # 6. The document level names the page, and the page is counted so the
    #    document cannot report a clean SUCCESS.
    note = pipeline._fabricated_url_note(state.events)
    assert note is not None and "1" in note, note
    assert state.pages[1].fabricated_image_refs == 1

    # 7. END-TO-END: the same is true of what actually reaches the .md.
    body = _shipped_body(state)
    assert "i.imgur.com" not in body, body
    assert "fabricated image reference removed" in body, body
    assert "Borrowing falls over the forecast." in body, body


def test_url_present_in_the_source_pdf_still_ships(tmp_path: Path) -> None:
    """Reverse regression A: a transcribed REAL link must survive.

    Provenance, not reachability, and not a host allowlist — the same imgur host
    that is fabricated in the test above is legitimate here purely because this
    PDF genuinely links to it.
    """
    url = "https://i.imgur.com/1234567.png"
    pdf_path = _pdf_with_link(tmp_path, url)
    doc_dir = tmp_path / "out"
    doc_dir.mkdir()

    state, out = _state_for(pdf_path, f"Chart 2.3 below.\n\n![Chart 2.3]({url})\n")

    pipeline = _make_pipeline()
    pipeline._sanitize_agentic_page_image_refs(state, 1, out, doc_dir)

    assert f"![Chart 2.3]({url})" in out.text, out.text
    assert out.status == PageStatus.SUCCESS
    assert out.audit_passed is True
    assert not [e for e in state.events if e.kind == "fabricated_image_ref"]
    assert getattr(state.pages[1], "fabricated_image_refs", 0) == 0
    # END-TO-END: it must still be in what reaches the .md.
    assert f"![Chart 2.3]({url})" in _shipped_body(state), _shipped_body(state)


def test_local_asset_socr_wrote_still_ships(tmp_path: Path) -> None:
    """Reverse regression B: a real extracted figure must survive.

    A fix that stripped every image ref would pass the first test and cause
    exactly the content loss this ticket exists to prevent — every extracted
    figure would vanish from the corpus.
    """
    pdf_path = _pdf_without_links(tmp_path)
    doc_dir = tmp_path / "out"
    (doc_dir / "figures").mkdir(parents=True)
    (doc_dir / "figures" / "figure_1_page1.png").write_bytes(b"\x89PNG")

    ref = "![Figure 1](figures/figure_1_page1.png)"
    state, out = _state_for(pdf_path, f"Text above.\n\n{ref}\n")

    pipeline = _make_pipeline()
    pipeline._sanitize_agentic_page_image_refs(state, 1, out, doc_dir)

    assert ref in out.text, out.text
    assert out.status == PageStatus.SUCCESS
    assert out.audit_passed is True
    assert not [e for e in state.events if e.kind == "fabricated_image_ref"]
    # END-TO-END: a real extracted figure must still be in the shipped body.
    assert ref in _shipped_body(state), _shipped_body(state)


def test_inline_links_are_not_touched(tmp_path: Path) -> None:
    """Reverse regression C: a markdown LINK is prose, not a pointer.

    Removing ``[label](url)`` would delete the label — real text. The gate is
    deliberately scoped to image refs, which carry no content of their own.
    """
    pdf_path = _pdf_without_links(tmp_path)
    doc_dir = tmp_path / "out"
    doc_dir.mkdir()

    link = "[Office for Budget Responsibility](https://obr.uk/efo/november-2022/)"
    state, out = _state_for(pdf_path, f"Source: {link}.\n")

    pipeline = _make_pipeline()
    pipeline._sanitize_agentic_page_image_refs(state, 1, out, doc_dir)

    assert link in out.text, out.text
    assert out.status == PageStatus.SUCCESS
    assert link in _shipped_body(state), _shipped_body(state)


def test_data_uri_image_is_fabricated(tmp_path: Path) -> None:
    """A ``data:`` image ref can never have source provenance.

    A PDF cannot carry one in a link annotation, socr never emits one into page
    markdown (they are only ever sent TO a model), and a model cannot produce
    real image bytes.
    """
    pdf_path = _pdf_without_links(tmp_path)
    doc_dir = tmp_path / "out"
    doc_dir.mkdir()

    state, out = _state_for(pdf_path, "Before\n\n![x](data:image/png;base64,abc)\n\nAfter\n")

    pipeline = _make_pipeline()
    pipeline._sanitize_agentic_page_image_refs(state, 1, out, doc_dir)

    assert "data:image/png" not in out.text, out.text
    assert "Before" in out.text and "After" in out.text
    assert out.status != PageStatus.SUCCESS
    assert state.pages[1].fabricated_image_refs == 1
    body = _shipped_body(state)
    assert "data:image/png" not in body, body
    assert "Before" in body and "After" in body


def test_born_digital_page_keeps_its_ocr_table_after_redaction(tmp_path: Path) -> None:
    """The regression the first version of this fix CAUSED (PR #252 review, blocking).

    #225's own document class is born-digital: the OBR EFO pages have a native
    text layer AND a VLM OCR read that recovered the real table. Demoting the
    page by flipping ``best_output.audit_passed`` looked correct in isolation,
    but ``_winning_page_output`` returns ``best_output`` only while that flag is
    True — so assemble fell through to the born-digital native branch, shipped
    flattened native prose under a FRESH SUCCESS, and the extracted table went
    with the invented URL. Fabrication removed, corpus damaged, page restamped
    clean: strictly worse than the defect being fixed.

    So this asserts the whole point at the output: the URL is gone AND the table
    survives AND the page is not restamped clean.
    """
    pdf_path = _pdf_without_links(tmp_path)
    doc_dir = tmp_path / "out"
    doc_dir.mkdir()

    native = "Table 2.3 Public finances\n\nborrowing forecast 2022 2023 2024\n"
    ocr = (
        "## Table 2.3 Public finances\n\n"
        "| Year | Borrowing |\n|---|---|\n| 2022 | 177.0 |\n| 2023 | 139.2 |\n\n"
        f"{FABRICATED_REF}\n"
    )
    state, out = _state_for(pdf_path, ocr, born_digital=True, native_text=native)

    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent
    pipeline._sanitize_agentic_page_image_refs(state, 1, out, doc_dir)

    body = _shipped_body(state)

    # The fabrication is gone.
    assert "i.imgur.com" not in body, body
    assert "fabricated image reference removed" in body, body
    # The extracted table SURVIVES. This is the content-loss assertion.
    assert "177.0" in body and "139.2" in body, (
        "the OCR table was thrown away and native text shipped instead — the fix "
        f"caused exactly the content loss it exists to prevent:\n{body}"
    )
    # The winner is still the OCR page, not a synthetic native fallback.
    winner = _winning_page_output(state, 1, None)
    assert winner.engine == "qwen", winner
    assert winner.status != PageStatus.SUCCESS, winner
    # And the document cannot report clean.
    assert state.pages[1].fabricated_image_refs == 1


def test_provenanced_url_with_a_commonmark_title_still_ships(tmp_path: Path) -> None:
    """PR #252 review, major: a title made a REAL link look fabricated.

    CommonMark allows ``![alt](url "title")``. Taking the parenthetical verbatim
    swallowed the title into the URL, so it no longer matched the PDF's own link
    annotation and a genuine image was redacted.
    """
    url = "https://example.com/chart.png"
    pdf_path = _pdf_with_link(tmp_path, url)
    doc_dir = tmp_path / "out"
    doc_dir.mkdir()

    ref = f'![Chart]({url} "Official chart")'
    state, out = _state_for(pdf_path, f"See below.\n\n{ref}\n")

    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent
    pipeline._sanitize_agentic_page_image_refs(state, 1, out, doc_dir)

    assert ref in _shipped_body(state), _shipped_body(state)
    assert not [e for e in state.events if e.kind == "fabricated_image_ref"]
    assert getattr(state.pages[1], "fabricated_image_refs", 0) == 0


def test_fabricated_url_with_a_title_is_still_caught(tmp_path: Path) -> None:
    """Reverse of the above: parsing the title off must not open an escape hatch."""
    pdf_path = _pdf_without_links(tmp_path)
    doc_dir = tmp_path / "out"
    doc_dir.mkdir()

    state, out = _state_for(
        pdf_path, 'Before\n\n![c](https://i.imgur.com/1234567.png "Chart 2")\n\nAfter\n'
    )

    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent
    pipeline._sanitize_agentic_page_image_refs(state, 1, out, doc_dir)

    body = _shipped_body(state)
    assert "i.imgur.com" not in body, body
    assert "Before" in body and "After" in body
    assert state.pages[1].fabricated_image_refs == 1


def test_phase_major_document_status_is_demoted_by_the_sweep(tmp_path: Path) -> None:
    """PR #252 round-2 review, blocking: the sweep ran AFTER status was frozen.

    The phase-major lanes (single-engine, multi-engine, consensus, repair) never
    reach the agentic per-page seam, so they never increment
    ``PageState.fabricated_image_refs``. Their fabricated refs are caught by the
    document sweep — which used to run *after* ``pages_ok`` and ``status`` were
    already computed. Result: the ref was removed and an error note appended,
    while the document still finished ``SUCCESS`` with ``audit_passed=True``.
    A document-status surface that the issue explicitly requires, silently
    absent on exactly the lanes with no per-page seam.

    Drives the real ``_phase_assemble``, so it asserts the ORDERING rather than
    the sweep in isolation — the sweep worked fine before; it just ran too late.
    """
    pdf_path = _pdf_without_links(tmp_path)
    out_dir = tmp_path / "out"

    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent

    # A page whose winner carries a fabricated ref but which the per-page seam
    # never touched — the phase-major shape.
    state, _out = _state_for(pdf_path, f"Real prose that must survive.\n\n{FABRICATED_REF}\n")
    # ``getattr`` with a default on purpose: ``fabricated_image_refs`` is an
    # attribute THIS PR adds, so naming it directly makes this setup assert raise
    # AttributeError at the baseline, before ``_phase_assemble`` is ever called —
    # which converts the baseline proof for this test from behavioural into
    # vacuous. Same defence as the round-2 guards.
    assert getattr(state.pages[1], "fabricated_image_refs", 0) == 0, (
        "setup: per-page seam not involved"
    )

    result = pipeline._phase_assemble(state, out_dir)

    assert result.status != DocumentStatus.SUCCESS, (
        "a document whose only defect is a fabricated image ref still finished "
        f"SUCCESS: {result.status}, error={result.error!r}"
    )
    assert result.audit_passed is False, result
    # And the redaction still happened, with the real prose intact.
    saved = "\n".join(p.read_text(encoding="utf-8") for p in out_dir.rglob("*.md") if p.is_file())
    assert "i.imgur.com" not in saved, saved
    assert "Real prose that must survive." in saved, saved


def test_clean_phase_major_document_still_succeeds(tmp_path: Path) -> None:
    """Reverse regression: the new document term must not demote a clean run."""
    pdf_path = _pdf_without_links(tmp_path)
    out_dir = tmp_path / "out"

    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent
    state, _out = _state_for(pdf_path, "Real prose with no image references at all.\n")

    result = pipeline._phase_assemble(state, out_dir)

    assert result.status == DocumentStatus.SUCCESS, (
        f"a clean document was demoted: {result.status}, error={result.error!r}"
    )
