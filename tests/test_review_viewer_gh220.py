"""GH-220: the review instrument must never imply a page is correct.

These tests are hermetic: they build a tiny PDF with PyMuPDF and plant sidecars, so no
engine, provider, or corpus file is touched. Each one is written to fail if the specific
production line it guards is reverted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz
import pytest

from socr.review.html import (
    ARTIFACT_BYTE_CAP,
    WRITE_REFUSAL_FLOOR,
    build_review_html,
    collect_pages,
)


def _make_pdf(path: Path, pages: int = 2) -> Path:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 100), f"page {i + 1} body text 12.34")
    doc.save(str(path))
    doc.close()
    return path


def _plant(doc_dir: Path, page_num: int, *, md: str | None, sidecar: dict) -> None:
    pages = doc_dir / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    if md is not None:
        (pages / f"{page_num:05d}.md").write_text(md, encoding="utf-8")
    (pages / f"{page_num:05d}.json").write_text(json.dumps(sidecar), encoding="utf-8")


def _payload(rendered: str) -> list[dict]:
    match = re.search(r"const PAGES = (\[.*?\]);\n", rendered, re.S)
    assert match, "PAGES payload not found in rendered output"
    return json.loads(match.group(1))


@pytest.fixture
def doc(tmp_path: Path) -> tuple[Path, Path]:
    pdf = _make_pdf(tmp_path / "src.pdf", pages=2)
    doc_dir = tmp_path / "out"
    doc_dir.mkdir()
    (doc_dir / "metadata.json").write_text(json.dumps({"status": "partial"}))
    return doc_dir, pdf


def test_page_universe_comes_from_the_pdf_not_the_pages_dir(doc: tuple[Path, Path]) -> None:
    """A fragment that was never written must surface as a gap, not shorten the document."""
    doc_dir, pdf = doc
    _plant(doc_dir, 1, md="# one", sidecar={"status": "success", "terminal": True})
    # page 2 deliberately not planted at all

    report = collect_pages(doc_dir, pdf)

    assert len(report.pages) == 2, "page count must follow the PDF, not the fragments on disk"
    missing = report.pages[1]
    assert missing.md_path_missing is True
    assert missing.json_path_missing is True
    assert "markdown fragment missing" in missing.signals
    assert "MISSING" in build_review_html(report)


def test_audit_passed_false_without_audit_events_is_still_flagged(
    doc: tuple[Path, Path],
) -> None:
    """The observed p58 case: audit failed, no audit_events recorded, status success.

    Deriving suspicion from audit_events alone makes this page invisible.
    """
    doc_dir, pdf = doc
    _plant(
        doc_dir,
        1,
        md="body",
        sidecar={
            "status": "success",
            "terminal": True,
            "audit_events": [],
            "winning_output": {"audit_passed": False},
        },
    )
    _plant(doc_dir, 2, md="body", sidecar={"status": "success", "terminal": True})

    report = collect_pages(doc_dir, pdf)

    page1 = report.pages[0]
    assert "audit_passed=false" in page1.signals
    assert page1.suspect is True
    assert page1.contradicts_itself is True, "success + a signal is a self-contradiction"
    assert report.pages[1].suspect is False


def test_empty_extract_is_not_rendered_as_a_blank_pane(doc: tuple[Path, Path]) -> None:
    doc_dir, pdf = doc
    _plant(doc_dir, 1, md="   \n\n", sidecar={"status": "success", "terminal": True})
    _plant(doc_dir, 2, md="ok", sidecar={"status": "success", "terminal": True})

    report = collect_pages(doc_dir, pdf)

    assert "extract is empty" in report.pages[0].signals
    assert "EMPTY EXTRACT" in build_review_html(report)


def test_untrusted_tables_flag_pages_even_with_no_audit_events(
    doc: tuple[Path, Path],
) -> None:
    doc_dir, pdf = doc
    (doc_dir / "tables_trust.json").write_text(
        json.dumps({"untrusted_pages": [2], "table_flags_n": 3})
    )
    _plant(doc_dir, 1, md="a", sidecar={"status": "success", "terminal": True})
    _plant(doc_dir, 2, md="b", sidecar={"status": "success", "terminal": True})

    report = collect_pages(doc_dir, pdf)

    assert "tables untrusted" in report.pages[1].signals
    assert report.untrusted_pages == [2]
    assert report.table_flag_count == 3


def test_non_terminal_page_is_flagged(doc: tuple[Path, Path]) -> None:
    doc_dir, pdf = doc
    _plant(doc_dir, 1, md="a", sidecar={"status": "success", "terminal": False})
    _plant(doc_dir, 2, md="b", sidecar={"status": "success", "terminal": True})

    report = collect_pages(doc_dir, pdf)

    assert "page not terminal" in report.pages[0].signals


def test_output_never_calls_an_unflagged_page_verified(doc: tuple[Path, Path]) -> None:
    """A page with no recorded signal must not be presented as correct."""
    doc_dir, pdf = doc
    _plant(doc_dir, 1, md="a", sidecar={"status": "success", "terminal": True})
    _plant(doc_dir, 2, md="b", sidecar={"status": "success", "terminal": True})

    rendered = build_review_html(collect_pages(doc_dir, pdf))

    assert "not verified" in rendered
    assert "nothing was recorded" in rendered


def test_output_is_self_contained(doc: tuple[Path, Path]) -> None:
    """A strict CSP blocks every external host, so any remote reference is a dead page."""
    doc_dir, pdf = doc
    _plant(doc_dir, 1, md="| a | b |\n| --- | --- |\n| 1 | 2 |", sidecar={"status": "success"})
    _plant(doc_dir, 2, md="text", sidecar={"status": "success"})

    rendered = build_review_html(collect_pages(doc_dir, pdf))

    for forbidden in ('src="http', 'href="http', "@import", "cdn.", "fonts.googleapis"):
        assert forbidden not in rendered, f"external reference {forbidden!r} in output"
    # Content is injected into a host-provided skeleton; it must not carry its own.
    for tag in ("<!doctype", "<html", "<body"):
        assert tag not in rendered.lower()


def test_encoding_is_declared_in_the_first_bytes(doc: tuple[Path, Path]) -> None:
    """Without a charset declaration a file:// load guesses windows-1252.

    That renders every UTF-8 pound sign as "A£" and every bullet as "a€¢", which reads as
    an OCR defect when the extracted markdown on disk is in fact correct. The declaration
    must land inside the first 1024 bytes or the browser has already guessed.
    """
    doc_dir, pdf = doc
    _plant(doc_dir, 1, md="cost £64.2 billion – see • note", sidecar={"status": "success"})
    _plant(doc_dir, 2, md="x", sidecar={"status": "success"})

    rendered = build_review_html(collect_pages(doc_dir, pdf))

    assert 'charset="utf-8"' in rendered[:1024].lower()
    # The chrome itself stays ASCII so it survives even a mis-sniffed load.
    chrome = rendered.split("const PAGES")[0]
    non_ascii = {c for c in chrome if ord(c) > 127}
    assert not non_ascii, f"non-ASCII in page chrome: {non_ascii}"


def test_pound_signs_in_the_extract_survive_into_the_payload(doc: tuple[Path, Path]) -> None:
    """Guard the user-reported symptom directly: the payload must carry a real U+00A3."""
    doc_dir, pdf = doc
    _plant(doc_dir, 1, md="costs £64.2 billion", sidecar={"status": "success"})
    _plant(doc_dir, 2, md="x", sidecar={"status": "success"})

    payload = _payload(build_review_html(collect_pages(doc_dir, pdf)))

    assert "£64.2" in payload[0]["md"]
    assert "Â£" not in payload[0]["md"], "mojibake introduced into the payload"


def test_document_status_is_carried_into_the_page(doc: tuple[Path, Path]) -> None:
    """The document verdict outranks the page verdict and must be visible."""
    doc_dir, pdf = doc
    _plant(doc_dir, 1, md="a", sidecar={"status": "success"})
    _plant(doc_dir, 2, md="b", sidecar={"status": "success"})

    report = collect_pages(doc_dir, pdf)
    assert report.doc_status == "partial"
    assert "PARTIAL" in build_review_html(report).upper()


def test_image_render_failure_is_surfaced_not_swallowed(doc: tuple[Path, Path]) -> None:
    doc_dir, pdf = doc
    _plant(doc_dir, 1, md="a", sidecar={"status": "success"})
    _plant(doc_dir, 2, md="b", sidecar={"status": "success"})

    report = collect_pages(doc_dir, pdf)
    report.pages[0].image_b64 = ""
    report.pages[0].image_error = "RuntimeError"

    assert "NO IMAGE" in build_review_html(report)


def test_every_page_is_present_in_the_payload(doc: tuple[Path, Path]) -> None:
    doc_dir, pdf = doc
    _plant(doc_dir, 1, md="a", sidecar={"status": "success"})
    _plant(doc_dir, 2, md="b", sidecar={"status": "success"})

    payload = _payload(build_review_html(collect_pages(doc_dir, pdf)))

    assert [p["n"] for p in payload] == [1, 2]
    assert all(p["img"] for p in payload), "every page must carry image bytes"


def test_rail_state_is_not_colour_only(doc: tuple[Path, Path]) -> None:
    """Signal state must be conveyed by glyph and aria-label, not colour alone."""
    doc_dir, pdf = doc
    _plant(doc_dir, 1, md="a", sidecar={"status": "success", "terminal": True})
    # page 2 never planted -> missing evidence

    rendered = build_review_html(collect_pages(doc_dir, pdf))

    assert "aria-label=" in rendered
    assert "aria-current=" in rendered
    assert "evidence missing" in rendered
    assert "no recorded signal, not verified" in rendered


def test_missing_evidence_is_visible_even_while_judging_cold(doc: tuple[Path, Path]) -> None:
    """The 'w' toggle hides recorded warnings. It must never hide absent evidence.

    A page whose markdown or image is absent cannot be judged at all; concealing that
    would turn the instrument into the thing it exists to catch.
    """
    doc_dir, pdf = doc
    _plant(doc_dir, 1, md="a", sidecar={"status": "success", "terminal": True})

    rendered = build_review_html(collect_pages(doc_dir, pdf))

    # The warning count is gated behind .reveal; the missing-evidence cross is not.
    assert "body:not(.reveal) #rail .sigct{display:none}" in rendered
    assert ".gone{" in rendered
    assert "#rail button.gone" not in rendered.split("body:not(.reveal)")[1][:200]


def test_refusal_floor_leaves_headroom_under_the_host_cap() -> None:
    assert WRITE_REFUSAL_FLOOR < ARTIFACT_BYTE_CAP
