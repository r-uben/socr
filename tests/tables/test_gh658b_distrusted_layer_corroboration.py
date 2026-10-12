"""GH-658b (rescue REMOVED in #1087): a distrusted text layer is never a witness.

``build_scanned_evidence`` excludes a distrusted (GH-163) layer, so a page whose
only reading is that layer has no evidence. #658b once rescued such a page by row
corroboration; #1087 removed the rescue (owner, 2026-10-12): reviewers beat every
number grammar, it rescued 0 of 181 stored pages, and a missing Tesseract now stops
the run (#1080). Such a page fails closed: not verifiable, no flag, no pass.

Every test goes through ``verify_scanned_table`` with a real tiny fitz page and an
empty pixel witness, so the ONLY evidence in play is the page's text layer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from socr.tables.source_evidence import verify_scanned_table  # noqa: E402


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# Swap-line-shaped candidate: the exact shape the Fed pages emit (a
# counterparty label plus two numeric columns).
CANDIDATE_ROWS = [
    ["Bundesbank", "62.5", "12.5"],
    ["Bank of Japan", "67.0", "15.0"],
    ["Bank of England", "45.0", "9.0"],
]
CANDIDATE_TABLE = _md_table(["Counterparty", "Amount", "Drawn"], CANDIDATE_ROWS)


def _stamp_full_page_raster(page) -> None:
    import io

    from PIL import Image

    w, h = int(page.rect.width * 3), int(page.rect.height * 3)
    img = Image.new("L", (w, h), color=245)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(page.rect, stream=buf.getvalue())


def _empty_pixel_witness(_pix) -> str:
    """Stands in for a host with no working classical-OCR reading at all --
    the pixels are looked at and nothing comes back. Deterministic, and
    independent of whether THIS machine happens to have tesseract."""
    return ""


def _page_with_matching_layer(tmp_path: Path):
    """A distrusted text layer that reproduces the candidate's rows exactly."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    _stamp_full_page_raster(page)
    for i, row in enumerate(CANDIDATE_ROWS):
        page.insert_text((72, 100 + i * 20), " ".join(row), fontsize=10)
    pdf = tmp_path / "scan.pdf"
    doc.save(pdf)
    doc.close()
    return fitz.open(pdf)


def _page_with_disagreeing_layer(tmp_path: Path):
    """A distrusted text layer present but carrying NONE of the candidate's
    numbers -- corroboration must find nothing to bind."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    _stamp_full_page_raster(page)
    for i in range(3):
        page.insert_text((72, 100 + i * 20), f"Some Other Line {i} 999.9 888.8", fontsize=10)
    pdf = tmp_path / "scan.pdf"
    doc.save(pdf)
    doc.close()
    return fitz.open(pdf)


def _blank_page(tmp_path: Path):
    """No text layer at all -- the scanned lane's ordinary precondition."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    pdf = tmp_path / "scan.pdf"
    doc.save(pdf)
    doc.close()
    return fitz.open(pdf)


# --------------------------------------------------------------------------
# Precondition: without a working pixel witness there is genuinely nothing
# to verify by -- confirms the fixtures actually exercise the "no pixel
# evidence at all" case the ticket is about, not some other reject path.
# --------------------------------------------------------------------------


def test_precondition_no_pixel_evidence_reaches_the_reject_before_the_fix_applies(
    tmp_path: Path,
) -> None:
    doc = _page_with_matching_layer(tmp_path / "precondition")
    try:
        # native_trusted=None keeps the pre-GH-163 word-presence deferral, so
        # this run never reaches the distrusted-layer gate at all -- it just
        # proves the page really carries words (a native_trusted=True/None
        # caller would defer to the native verifier on this same page).
        from socr.tables.source_evidence import page_has_native_words

        assert page_has_native_words(doc[0]), (
            "fixture has no words at all, so it cannot exercise the "
            "distrusted-layer corroboration path"
        )
    finally:
        doc.close()


NOT_VERIFIABLE = "no local content evidence available for scanned table"


def _verify(doc):
    return verify_scanned_table(
        doc[0], CANDIDATE_TABLE, ocr_image_fn=_empty_pixel_witness, native_trusted=False
    )


def _assert_fails_closed(result) -> None:
    assert result.passed is False
    assert result.verifiable is False
    assert result.reason == NOT_VERIFIABLE
    assert not result.content_unverified


def test_a_layer_that_reproduces_the_candidate_does_not_rescue_the_page(tmp_path: Path) -> None:
    doc = _page_with_matching_layer(tmp_path / "match")
    try:
        _assert_fails_closed(_verify(doc))
    finally:
        doc.close()


def test_a_disagreeing_layer_fails_closed_identically(tmp_path: Path) -> None:
    doc = _page_with_disagreeing_layer(tmp_path / "disagree")
    try:
        _assert_fails_closed(_verify(doc))
    finally:
        doc.close()


def test_a_page_with_no_text_layer_fails_closed_identically(tmp_path: Path) -> None:
    doc = _blank_page(tmp_path / "no_layer")
    try:
        _assert_fails_closed(_verify(doc))
    finally:
        doc.close()


def test_matching_and_disagreeing_layers_are_indistinguishable(tmp_path: Path) -> None:
    # The pinned DIFFERENCE: the layer's content changes nothing.
    matching = _page_with_matching_layer(tmp_path / "m")
    disagreeing = _page_with_disagreeing_layer(tmp_path / "d")
    try:
        assert _verify(matching) == _verify(disagreeing)
    finally:
        matching.close()
        disagreeing.close()
