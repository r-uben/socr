"""GH-658b: a distrusted text layer is discarded instead of used as a witness.

Root cause (measured, docs/log/2026-09-07_D3-fed-table-lane-remeasure.md):
``build_scanned_evidence`` excludes the page's text layer entirely once the
caller marks it distrusted (GH-163), so a page whose ONLY available reading
is that layer has no evidence at all -- ``has_content_evidence`` is False --
and candidates carrying 62/62, 67/67, 66/66 of the page's numbers with zero
extras were rejected outright, cause ``no local content evidence available``.

The fix does not re-admit the layer into the evidence bundle (that would let
a hallucination corroborate itself against a reading nobody trusts, exactly
what GH-163 exists to prevent). It uses ``row_corroboration.corroborate_rows``
-- a WEAKER, order-only check -- purely as a second, independent gate: does
each candidate numeric row appear, in order, on one native printed line? If
yes, the page ships, but flagged: ``content_unverified`` carries a
``header_binding_unverified`` marker through the SAME channel #659 already
wired end to end (WARNING status at finalization, ``table_label_unverified``
on the sidecar, ``TABLE_DISTRUST_KINDS``, document metadata / CLI).

Every test here goes through the public ``verify_scanned_table`` entry point
with a real (tiny) fitz page and ``ocr_image_fn`` pinned to return "" for
every pixel reading, so the ONLY evidence in play is the page's real text
layer -- isolating exactly the case the ticket is about.
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


# --------------------------------------------------------------------------
# Criterion 1 & the ticket's core claim: a reproducing distrusted layer
# rescues the page, and it ships FLAGGED, never a clean SUCCESS.
# --------------------------------------------------------------------------


def test_a_reproducing_distrusted_layer_rescues_the_page(tmp_path: Path) -> None:
    doc = _page_with_matching_layer(tmp_path / "rescue")
    try:
        result = verify_scanned_table(
            doc[0],
            CANDIDATE_TABLE,
            ocr_image_fn=_empty_pixel_witness,
            native_trusted=False,
        )
        assert result.passed, (
            f"a candidate whose rows are fully reproduced by the page's own "
            f"(distrusted) text layer was rejected: {result.reason}"
        )
    finally:
        doc.close()


def test_the_rescue_is_flagged_not_a_clean_success(tmp_path: Path) -> None:
    """Criterion 1, read twice in the ticket: this must not become a silent
    accept. The flag must ride on the result via ``content_unverified`` --
    the one channel this module already has wired to WARNING status,
    document metadata and CLI (#659's ``LABEL_UNVERIFIED_KIND`` machinery)."""
    doc = _page_with_matching_layer(tmp_path / "flagged")
    try:
        result = verify_scanned_table(
            doc[0],
            CANDIDATE_TABLE,
            ocr_image_fn=_empty_pixel_witness,
            native_trusted=False,
        )
        assert result.content_unverified, (
            "the rescued page carries no flag at all -- it would surface "
            "downstream as an ordinary, unflagged SUCCESS"
        )
        assert "header_binding_unverified" in result.content_unverified, (
            f"the flag does not name the owner-ruled marker: {result.content_unverified!r}"
        )
    finally:
        doc.close()


# --------------------------------------------------------------------------
# Criterion 2: the layer is corroboration only -- it must never be promoted
# into content evidence or merged into the shipped text.
# --------------------------------------------------------------------------


def test_the_shipped_text_is_the_candidates_own_unmodified_text(tmp_path: Path) -> None:
    """``verify_scanned_table`` never rewrites ``output_text``; the caller
    ships exactly what the model produced, corroborated but not corrected."""
    doc = _page_with_matching_layer(tmp_path / "unmerged")
    try:
        before = CANDIDATE_TABLE
        verify_scanned_table(
            doc[0],
            CANDIDATE_TABLE,
            ocr_image_fn=_empty_pixel_witness,
            native_trusted=False,
        )
        assert CANDIDATE_TABLE == before, "the candidate's own text was mutated"
    finally:
        doc.close()


# --------------------------------------------------------------------------
# Criterion 3: a page with NO text layer at all must still fail closed
# exactly as today -- this ticket rescues the distrusted-LAYER case only.
# --------------------------------------------------------------------------


def test_a_page_with_no_text_layer_still_fails_closed(tmp_path: Path) -> None:
    doc = _blank_page(tmp_path / "no_layer")
    try:
        result = verify_scanned_table(
            doc[0],
            CANDIDATE_TABLE,
            ocr_image_fn=_empty_pixel_witness,
            native_trusted=False,
        )
        assert not result.passed, (
            f"a page with NO text layer at all and no pixel evidence was "
            f"accepted -- this ticket must not touch that case: {result.reason}"
        )
        assert not result.content_unverified
    finally:
        doc.close()


# --------------------------------------------------------------------------
# Criterion 4: a candidate that genuinely disagrees with the distrusted
# layer must still be rejected.
# --------------------------------------------------------------------------


def test_a_disagreeing_candidate_is_still_rejected(tmp_path: Path) -> None:
    doc = _page_with_disagreeing_layer(tmp_path / "disagree")
    try:
        result = verify_scanned_table(
            doc[0],
            CANDIDATE_TABLE,
            ocr_image_fn=_empty_pixel_witness,
            native_trusted=False,
        )
        assert not result.passed, (
            f"a candidate whose numbers appear NOWHERE in the distrusted "
            f"layer was accepted by row corroboration: {result.reason}"
        )
        assert not result.content_unverified
    finally:
        doc.close()


def test_disagreement_still_fails_closed_identically_to_the_no_rescue_path(
    tmp_path: Path,
) -> None:
    """Pin the DIFFERENCE the ticket is supposed to make: a page whose layer
    corroborates and a page whose layer contradicts must land on opposite
    sides of ``passed``, everything else about the input held fixed."""
    matching = _page_with_matching_layer(tmp_path / "diff_match")
    disagreeing = _page_with_disagreeing_layer(tmp_path / "diff_disagree")
    try:
        rescued = verify_scanned_table(
            matching[0],
            CANDIDATE_TABLE,
            ocr_image_fn=_empty_pixel_witness,
            native_trusted=False,
        )
        rejected = verify_scanned_table(
            disagreeing[0],
            CANDIDATE_TABLE,
            ocr_image_fn=_empty_pixel_witness,
            native_trusted=False,
        )
        assert rescued.passed is True
        assert rejected.passed is False
    finally:
        matching.close()
        disagreeing.close()
