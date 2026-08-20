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
from socr.core.result import PageOutput, PageStatus
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


def _state_for(pdf_path: Path, text: str) -> tuple[DocumentState, PageOutput]:
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    out = PageOutput(
        page_num=1,
        text=text,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    ps = state.pages[1]
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
    # 4. The page no longer ships under SUCCESS with a passing audit.
    assert out.status != PageStatus.SUCCESS, out.status
    assert out.audit_passed is False
    # 5. An audit event carries the invented target for forensics.
    events = [e for e in state.events if e.kind == "fabricated_image_ref"]
    assert len(events) == 1, state.events
    assert events[0].page_num == 1
    assert "https://i.imgur.com/1234567.png" in events[0].data["targets"]
    # 6. The document level names the page.
    note = pipeline._fabricated_url_note(state.events)
    assert note is not None and "1" in note, note


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
    assert out.audit_passed is False
