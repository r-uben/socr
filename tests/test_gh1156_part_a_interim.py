"""#1156 Part A: the interim behaviour of a page whose text layer is an earlier OCR run.

The owner split #1156 (2026-10-17). Part A, here: the trust fix, the drawing-order
detector, routing a flagged page to a model reading, retaining the original layer in the
sidecar unshipped, and this rule --

  * reading ACCEPTED   -> it ships, and the page is MARKED (never CHECKED);
  * reading REJECTED, or NEVER RAN (no provider, ``--native-only``) -> the page ships the
    page image plus the "earlier OCR, not re-read" mark, with no prior-layer text and no
    numbers;
  * either way the document cannot report a clean SUCCESS over the page.

Part B adds the comparator and the masked-prose ruling by re-finalizing from the retained
text. Nothing here masks, compares, or withholds shreds.

Synthetic PDFs driven through the real detector and the real selection cascade. No model,
no provider, no Ollama: hermetic for CI (CLAUDE.md). Routing pins are DIFFERENCES -- the
same input with the route live and neutralised -- never an absolute outcome measured on
one machine (#257).
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.born_digital import BornDigitalDetector  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import (  # noqa: E402
    SelectionProvenance,
    _select_page_output_tagged,
)
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState, PageState  # noqa: E402


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def _png(w: int, h: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (w, h), 245).save(buf, format="PNG")
    return buf.getvalue()


def _prior_ocr_pdf(path: Path, *, pages: int = 2) -> Path:
    """A scan whose text was painted over by the page raster: the #1156 family."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=612, height=792)
        nonce = "w" * (i + 1)
        # Prose, deliberately not column-shaped: a grid in the fixture would give the
        # control run its own reason to be unclean and the pin below would show nothing.
        sentences = [
            f"{nonce} The Committee met and reviewed conditions in the economy.",
            f"{nonce} Discount rates were raised from 3 to 3 Y, per cent at the Banks.",
            f"{nonce} Members discussed the outlook for prices and employment.",
            f"{nonce} The Secretary read the minutes of the previous meeting.",
        ]
        for row, sentence in enumerate(sentences):
            page.insert_text((72, 120 + row * 24), sentence, fontsize=10)
        page.insert_image(page.rect, stream=_png(300, 400))
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# P0: the page must reach a model, and must never ship as a clean pass
# ---------------------------------------------------------------------------


def test_a_flagged_page_is_routed_for_a_reading(tmp_path: Path) -> None:
    """The ruling's premise is "re-read, and withhold only if the reading is rejected".

    Without this the withholding branch is unreachable the other way round: no model
    attempt ran on any of the 63 flagged Fed pilot pages, so the page shipped its
    prior-OCR text untouched as a clean SUCCESS.
    """
    assessment = BornDigitalDetector().detect(_prior_ocr_pdf(tmp_path / "scan.pdf"))
    page = assessment.pages[0]
    assert page.prior_ocr_layer is True
    assert page.needs_ocr_enhancement is True, (
        "a prior-OCR page was not routed for a reading, so nothing can ever accept one "
        "and the page can only ever reach the floor"
    )


def _state_with_passing_native(*, prior_ocr: bool, scan_failed: bool = False) -> DocumentState:
    state = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=1))
    p = PageState(page_num=1)
    p.is_born_digital = True
    p.native_text = "discount rates from 3 to 3 Y, per cent and $400 million"
    # Round 2: the short-circuit guard reads the POSITIVE finding, because skipping it is
    # what costs the chart lane its figure link on a failed scan. The scan-failed arm
    # below checks that the page is still never a clean pass, by the clause instead.
    p.prior_ocr_layer = prior_ocr and not scan_failed
    p.prior_ocr_layer_scan_failed = scan_failed
    p.prior_ocr_layer_suspect = prior_ocr or scan_failed
    winner = PageOutput(
        page_num=1,
        text=p.native_text,
        status=PageStatus.SUCCESS,
        engine="native",
        audit_passed=True,
        failure_mode=FailureMode.NONE,
    )
    p.attempts = [winner]
    p.best_output = winner
    state.pages[1] = p
    return state


def test_a_passing_native_winner_does_not_short_circuit_to_a_clean_pass() -> None:
    """Difference pin: the same passing native page with and without the flag.

    ``PASSING_BEST_OUTPUT`` returns before every later branch, so a flagged page took it
    and shipped "3 to 3 Y, per cent" and "$400 million" verbatim, audit_passed true.
    """
    clean, clean_prov = _select_page_output_tagged(_state_with_passing_native(prior_ocr=False), 1)
    flagged, flagged_prov = _select_page_output_tagged(
        _state_with_passing_native(prior_ocr=True), 1
    )

    assert clean_prov is SelectionProvenance.PASSING_BEST_OUTPUT
    assert "400" in clean.text
    assert flagged_prov is not SelectionProvenance.PASSING_BEST_OUTPUT, (
        "a page whose text layer is an earlier OCR run still shipped as a clean pass"
    )

    # Round 2: a FAILED scan keeps its text (principle 2), so it DOES take this
    # short-circuit -- and must still not read as a clean pass. The demotion comes from
    # the clause instead of from this guard, which is the part that matters.
    from socr.core.manifest import _select_and_finalize_page

    unknown = _select_and_finalize_page(
        _state_with_passing_native(prior_ocr=True, scan_failed=True), 1
    )
    assert "400" in unknown.output.text, "a failed scan deleted the page's own text"
    assert unknown.output.status is not PageStatus.SUCCESS, "a failed scan shipped as a clean pass"


# ---------------------------------------------------------------------------
# P1: the witness must be a reading of text that lies UNDER the raster
# ---------------------------------------------------------------------------


def test_an_opaque_background_under_a_born_digital_table_is_not_a_witness(
    tmp_path: Path,
) -> None:
    """The required swapped-value probe, with the flag FORCED on a born-digital table.

    Four separate review rounds found a flag that means "trust this less" removing a
    check instead of adding one. The last shape: a born-digital table drawn ON TOP of an
    opaque background satisfies "the page is covered by opaque images", but the pixels are
    a reading of the background, not of the table. Forcing suspicion on such a page
    skipped row verification, and a table with two values swapped was ACCEPTED.
    """
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline.agentic import NativeTableVerifierJudge

    rows = (("Alpha", "2.1", "0.4"), ("Beta", "3.1", "0.5"), ("Gamma", "4.2", "0.6"))
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=_png(300, 400))  # background FIRST
    for x, label in ((72, "Series"), (260, "Value"), (400, "SE")):
        page.insert_text((x, 100), label, fontsize=10)
    page.draw_line((60, 108), (470, 108))
    y = 130
    for row in rows:
        for x, cell in zip((72, 260, 400), row):
            page.insert_text((x, y), cell, fontsize=10)
        y += 22
    page.draw_line((60, y), (470, y))
    pdf = tmp_path / "ontop.pdf"
    doc.save(pdf)
    doc.close()
    doc = fitz.open(pdf)

    swapped = "\n".join(
        [
            "| Series | Value | SE |",
            "| --- | --- | --- |",
            "| Alpha | 3.1 | 0.4 |",
            "| Beta | 2.1 | 0.5 |",
            "| Gamma | 4.2 | 0.6 |",
        ]
    )

    class _Accepting:
        def assess(self, output, provider):
            from socr.pipeline.agentic import AcceptDecision

            return AcceptDecision(accept=True, reason="inner accepted")

    try:
        decision = NativeTableVerifierJudge(
            inner=_Accepting(),
            get_fitz_page=lambda _n: doc[0],
            is_table_page=lambda _n: True,
            native_evidence_suspect=lambda _n: True,
        ).assess(
            PageOutput(
                page_num=1,
                text=swapped,
                status=PageStatus.SUCCESS,
                engine="qwen",
                audit_passed=True,
            ),
            PROFILE_QWEN_LOCAL,
        )
        assert decision.accept is False, (
            "the page's own table was graded as if the background image had read it, so "
            "two swapped values were accepted"
        )
    finally:
        doc.close()


def test_transparency_is_read_strictly_for_the_witness_and_loosely_for_the_detector() -> None:
    """The two consumers of opacity need OPPOSITE failure directions.

    The page-level graphics-state reader finds partial alpha on 0 of 598 corpus pages,
    because generators keep ``/ca`` inside indirect objects; following the references
    finds it on 76. Switching that on everywhere costs 2 detections of 594 pages, and both
    are genuine JSTOR scans -- shipping a prior-OCR layer as trusted native, the harm this
    issue exists to prevent. Leaving it off everywhere leaves the witness free to claim an
    independent reading it has not got, which is the weakening found four times.

    So the witness reads strictly and the detector does not. Pinned as a DIFFERENCE on one
    synthetic page that declares alpha only through an indirect state, which is the shape
    the corpus pages have: the two readers must disagree on it, and the predicate must
    follow whichever one its caller asked for.
    """
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.insert_text((20, 40), "a word", fontsize=10)
    try:
        # /ExtGState << /GS0 N 0 R >> with /ca 0.5 inside object N -- indirect, as real
        # generators write it, so only the dereferencing reader can see it.
        state_xref = doc.get_new_xref()
        doc.update_object(state_xref, "<< /Type /ExtGState /ca 0.5 >>")
        resources = doc.xref_get_key(page.xref, "Resources")
        # The page's Resources may itself be indirect, which is the common shape and the
        # one PyMuPDF refuses to write through; set the key on whichever object holds it.
        holder = int(resources[1].split()[0]) if resources[0] == "xref" else page.xref
        doc.xref_set_key(holder, "ExtGState", f"<< /GS0 {state_xref} 0 R >>")

        detector = BornDigitalDetector()
        loose = detector._page_declares_transparency(page)
        strict = detector._declares_transparency_following_references(page)
        assert loose is False, "fixture drifted: the page-level reader was supposed to miss this"
        assert strict is True, (
            "the dereferencing reader did not find alpha held in an indirect graphics "
            "state, which is how every corpus page that has one writes it"
        )
    finally:
        doc.close()


def test_an_indirect_alpha_state_denies_the_witness_but_not_the_detector(
    tmp_path: Path,
) -> None:
    """The consequence of the split, on one page, in both directions at once.

    A prior-OCR page that also declares alpha through an indirect graphics state: the
    DETECTOR must still see it (otherwise a genuine scan ships as trusted native -- the
    two JSTOR pages this was measured on), and the WITNESS must refuse it (otherwise a
    table skips row verification on the strength of pixels that may be transparent).

    Without the split one of those two is always wrong, which is what makes this a pin on
    the split rather than on either reader.
    """
    doc = _page_with_prior_layer(tmp_path, "indirect-alpha.pdf")
    try:
        page = doc[0]
        state_xref = doc.get_new_xref()
        doc.update_object(state_xref, "<< /Type /ExtGState /ca 0.5 >>")
        resources = doc.xref_get_key(page.xref, "Resources")
        holder = int(resources[1].split()[0]) if resources[0] == "xref" else page.xref
        doc.xref_set_key(holder, "ExtGState", f"<< /GS0 {state_xref} 0 R >>")

        from socr.core.born_digital import (
            _FURNITURE_CACHE,
            page_has_opaque_page_covering_raster,
        )

        detector = BornDigitalDetector()
        detected = detector._has_prior_ocr_layer(page, frozenset(), frozenset())
        # Through the PUBLIC entry point, which is what the table judge calls: asking the
        # predicate directly with the keyword would pass even if that caller stopped
        # passing it, which is exactly the edit this test has to catch.
        _FURNITURE_CACHE.clear()
        witness = page_has_opaque_page_covering_raster(page)

        assert detected is True, (
            "the detector stopped seeing a prior-OCR scan because the page declares alpha "
            "somewhere, so that scan would ship as trusted native"
        )
        assert witness is False, (
            "the witness claimed an independent reading of pixels that may be "
            "transparent, which is what lets a table skip row verification"
        )
    finally:
        doc.close()


def test_the_witness_test_uses_the_documents_own_furniture(tmp_path: Path) -> None:
    """A scan with a running header is still a scan.

    With empty slots the witness test asked "does this page print a word of its own?" and
    answered yes for the header, so the page counted as having no independent reading and
    the native check ran against the very layer under suspicion.
    """
    from socr.core.born_digital import page_has_opaque_page_covering_raster

    doc = fitz.open()
    for i in range(3):
        page = doc.new_page(width=612, height=792)
        for row in range(6):
            page.insert_text((72, 140 + row * 20), f"{'w' * (i + 1)} body line {row}", fontsize=10)
        page.insert_image(page.rect, stream=_png(300, 400))
        page.insert_text((72, 20), "November 10, 2004", fontsize=8)
        page.insert_text((300, 20), str(101 + i), fontsize=8)
        page.insert_text((330, 20), "of 107", fontsize=8)
    path = tmp_path / "headered.pdf"
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()

    opened = fitz.open(path)
    try:
        assert page_has_opaque_page_covering_raster(opened[0]) is True, (
            "a scan carrying a running header read as having no witness, which puts the "
            "prior layer back in charge of checking tables"
        )
    finally:
        opened.close()


# ---------------------------------------------------------------------------
# The opacity test must read the metadata of the image it is asking about
# ---------------------------------------------------------------------------


def _page_with_prior_layer(tmp_path: Path, name: str):
    """Text laid down first, then an opaque full-page raster over it: the #1156 family."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for row in range(6):
        page.insert_text((72, 140 + row * 20), f"body line {row}", fontsize=10)
    page.insert_image(page.rect, stream=_png(300, 400))
    path = tmp_path / name
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return fitz.open(path)


def test_an_unmatched_image_box_reads_as_not_opaque() -> None:
    """``_image_info_for_box`` fails closed, which is what both callers need.

    No metadata is not evidence of opacity. Returning ``{}`` makes ``_image_is_opaque``
    answer False, so the page keeps its native check and the witness test says "no
    independent reading" -- the direction that adds checks rather than removing them.
    """
    from socr.core.born_digital import _image_info_for_box

    page_rect = fitz.Rect(0, 0, 612, 792)
    infos = [{"bbox": (0, 0, 100, 100), "has-mask": False, "xref": 7}]

    hit = _image_info_for_box(infos, fitz.Rect(0, 0, 100, 100), page_rect)
    miss = _image_info_for_box(infos, fitz.Rect(300, 300, 400, 400), page_rect)

    assert hit is infos[0]
    assert miss == {}, "an image whose metadata was not found was treated as described"


def test_the_opacity_test_is_not_fooled_by_an_extra_metadata_entry(tmp_path: Path) -> None:
    """The measured desync, made load-bearing on an input.

    The old code paired the Nth image in the DRAWING ORDER with ``get_image_info()[N]``.
    Over 598 pages of 150 corpus PDFs, 25 disagree on the image count -- every one of them
    with FEWER images drawn than listed -- so from the first missing one onwards each
    drawn image was read against another image's metadata, and this test's page is that
    shape: one image drawn, two listed, the extra one first and masked.

    By ORDER the page's opaque raster is judged by the decoy's ``has-mask`` and the page
    stops being a detected scan. By GEOMETRY it is judged by its own entry.
    """
    doc = _page_with_prior_layer(tmp_path, "desync.pdf")
    try:
        page = doc[0]
        real = page.get_image_info(xrefs=True)
        assert len(real) == 1, "fixture drifted: the page should carry exactly one image"
        decoy = {
            "bbox": (0.0, 0.0, 10.0, 10.0),  # nowhere near the full-page raster
            "has-mask": True,  # ... and transparent, so order-pairing would say so
            "xref": 0,
        }
        page.get_image_info = lambda *a, **k: [decoy, *real]

        assert BornDigitalDetector()._has_prior_ocr_layer(page, frozenset(), frozenset()) is True, (
            "the page's own raster was judged by another image's metadata, so a scan "
            "with a prior OCR layer stopped being detected"
        )
    finally:
        doc.close()


def test_the_witness_cache_cannot_hand_over_another_documents_furniture(
    tmp_path: Path,
) -> None:
    """The cache is keyed on ``id(doc)``, which is unique only among LIVE objects.

    A ``fitz.Document`` freed between two calls lets CPython hand its address to the next
    document, and the cache would answer with another document's running headers. The
    entry therefore holds the document itself, which makes the key valid by construction;
    and two live documents get their own evidence.
    """
    from socr.core.born_digital import (
        _FURNITURE_CACHE,
        page_has_opaque_page_covering_raster,
    )

    first = _page_with_prior_layer(tmp_path, "cache-one.pdf")
    second = _page_with_prior_layer(tmp_path, "cache-two.pdf")
    try:
        _FURNITURE_CACHE.clear()
        assert page_has_opaque_page_covering_raster(first[0]) is True
        assert page_has_opaque_page_covering_raster(second[0]) is True

        for doc in (first, second):
            entry = _FURNITURE_CACHE.get(id(doc))
            assert entry is not None, "the document's furniture was not cached"
            assert entry[0] is doc, (
                "the cache does not hold the document its key is the id of, so that id "
                "can be recycled and the next document inherits this one's furniture"
            )
    finally:
        _FURNITURE_CACHE.clear()
        first.close()
        second.close()


# ---------------------------------------------------------------------------
# P2: the demoted page names its own reason once
# ---------------------------------------------------------------------------


def test_the_withheld_page_does_not_also_claim_an_unknown_check() -> None:
    """Without a row in the failure-mode table the page carried a second, bogus banner
    saying ``unknown=UNKNOWN`` beside the clause that already named the reason."""
    from socr.core.marks import Check
    from socr.core.verdict import build_page_verdict

    p = PageState(page_num=1)
    p.is_born_digital = True
    p.native_text = "text"
    p.prior_ocr_layer_suspect = True
    withheld = PageOutput(
        page_num=1,
        text="[page 1: earlier OCR, not re-read — see image]",
        status=PageStatus.WARNING,
        engine="native",
        audit_passed=False,
        failure_mode=FailureMode.PRIOR_OCR_LAYER_UNVERIFIED,
    )
    verdict = build_page_verdict(1, withheld, p)
    checks = [c.check for c in verdict.clauses]
    assert Check.PRIOR_OCR_LAYER in checks
    assert Check.UNKNOWN_CHECK not in checks, (
        "the page claims an unknown check beside the one that names its reason"
    )


def test_an_accepted_reading_is_demoted_with_its_own_reason_named() -> None:
    """The demotion must explain itself in the STATUS, not only in the mark.

    An accepted model reading arrives with ``failure_mode = NONE``, so the mode comes from
    the clause via ``_verdict_demotion_mode``. Without a row there the page is demoted with
    no mode, the fail-closed fallback then fires on a page whose reason WAS named, and the
    page carries a second, bogus ``unknown=UNKNOWN`` banner beside the real one. This
    drives the demotion instead of handing the mode in, which is what the hand-built
    verdict test above cannot do.
    """
    from socr.core.manifest import _select_and_finalize_page

    state = _suspect_state(attempts=True, detected=False, invisible=True)
    accepted = PageOutput(
        page_num=1,
        text="A MODEL READING THE JUDGE ACCEPTED",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
        failure_mode=FailureMode.NONE,
    )
    state.pages[1].attempts = [accepted]
    state.pages[1].best_output = accepted

    record = _select_and_finalize_page(state, 1)
    assert record.output.status is PageStatus.WARNING
    assert record.output.failure_mode is FailureMode.PRIOR_OCR_LAYER_UNVERIFIED, (
        "the demotion named no reason, so the status cannot say why the page is not clean"
    )
    assert "unknown=UNKNOWN" not in record.output.text, (
        "the page carries the fail-closed banner beside the clause that names its reason"
    )
    assert "earlier OCR, not re-read" in record.output.text
    # Idempotence, the property the mode exists to protect: assemble re-runs this seam
    # over every page.
    again = _select_and_finalize_page(state, 1, saved_text=record.output.text)
    assert again.output.text == record.output.text


def test_the_clause_fires_on_an_accepted_model_reading_too() -> None:
    """The rule is MARKED-never-CHECKED whatever won.

    Restricting the clause to native winners left a qwen reading of such a scan shipping
    SUCCESS with no mark at all, while the meter's own rule flagged the same page -- the
    page and the meter disagreeing about whether anything had checked it. Difference pin:
    the same accepted model output with and without the flag.
    """
    from socr.core.marks import Check
    from socr.core.verdict import build_page_verdict

    def verdict_for(*, suspect: bool):
        p = PageState(page_num=1)
        p.is_born_digital = True
        p.native_text = "the earlier OCR layer"
        p.prior_ocr_layer_suspect = suspect
        accepted = PageOutput(
            page_num=1,
            text="A MODEL READING THE JUDGE ACCEPTED",
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
            failure_mode=FailureMode.NONE,
        )
        return build_page_verdict(1, accepted, p)

    clean = [c.check for c in verdict_for(suspect=False).clauses]
    flagged = [c.check for c in verdict_for(suspect=True).clauses]
    assert Check.PRIOR_OCR_LAYER not in clean
    assert Check.PRIOR_OCR_LAYER in flagged, (
        "an accepted model reading of a prior-OCR scan shipped with no mark, so it counts "
        "CHECKED on one reading of a scan"
    )


# ---------------------------------------------------------------------------
# Every shipping path for a flagged page
# ---------------------------------------------------------------------------


def _suspect_state(
    *, attempts: bool, detected: bool = True, invisible: bool = False
) -> DocumentState:
    """A prior-OCR page whose layer is about to ship.

    ``attempts`` says whether a model reading was ever made -- with no provider, under
    ``--native-only``, or on the no-provider fallback it never is, and that is the hole
    both review seats found: the no-attempt page fell through to the native fallback,
    which shipped the layer verbatim.
    """
    state = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=1))
    p = PageState(page_num=1)
    p.is_born_digital = True
    p.native_text = "Raised discount rates from 3 to 3 Y, per cent and $400 million"
    p.prior_ocr_layer = detected
    p.prior_ocr_layer_suspect = detected or invisible
    p.invisible_text_over_raster = invisible
    p.needs_ocr_enhancement = True
    if attempts:
        rejected = PageOutput(
            page_num=1,
            text="A MODEL READING THE JUDGE REJECTED",
            status=PageStatus.WARNING,
            engine="qwen",
            audit_passed=False,
            failure_mode=FailureMode.AUDIT_FAILED,
        )
        p.attempts = [rejected]
        p.best_output = rejected
    state.pages[1] = p
    return state


@pytest.mark.parametrize("attempts", [True, False])
def test_the_layer_is_withheld_whether_or_not_a_reading_was_ever_made(attempts: bool) -> None:
    """The rule is about the TEXT, not about which branch shipped it.

    Enumerating the shipping paths is what this parametrisation is: with a rejected
    attempt the page would reach the flagged-model branch, and with none it falls to the
    native fallback, which shipped the layer 100% intact -- "3 to 3 Y, per cent" and
    "$400 million" straight onto the page. Unlike #1027's floor this branch is therefore
    NOT gated on a model attempt having run.
    """
    from socr.core.manifest import _select_and_finalize_page

    record = _select_and_finalize_page(_suspect_state(attempts=attempts), 1)
    text = record.output.text
    assert "3 Y," not in text, "the earlier OCR layer's wrong rate shipped"
    assert "400" not in text, "the earlier OCR layer's numbers shipped"
    assert "A MODEL READING THE JUDGE REJECTED" not in text, "the rejected reading shipped"
    assert "earlier OCR, not re-read" in text
    assert record.output.failure_mode is FailureMode.PRIOR_OCR_LAYER_UNVERIFIED
    assert record.output.status is not PageStatus.SUCCESS


def test_the_floored_page_is_visible_to_the_shared_marker_recogniser() -> None:
    """The floor's body must be readable by the ONE recogniser every consumer uses.

    ``is_page_failed_marker`` anchors on the ``[page N failed: ...]`` prefix, and the
    repo's rule is that a page's ending is read from the SHIPPED BYTES, never from its
    provenance alone. A marker without the prefix (Argon, Part A review) was invisible to
    `_phase_assemble`'s ``failed_pages``, so a floored page with a REJECTED model attempt
    fell into ``native_fallback_pages`` -- a list meaning "the page shipped its native
    text because OCR never passed" -- on the one page whose native text was withheld. It
    also belonged to no marker family, so the reason could not be named from the bytes,
    and it was absent from the kept-model-reading exclusion set.

    Pinned against #1027's floor, which is the known-correct sibling, so a future edit
    cannot drift one without the other.
    """
    from socr.core.manifest import (
        _MARKER_FAMILY_REASONS,
        _UNACCEPTED_OUTPUT_MARK_EXCLUDED_PROVENANCES,
        PagePrimaryReason,
        _select_and_finalize_page,
        _shipped_marker_reason,
        is_page_failed_marker,
    )

    for attempts in (True, False):
        record = _select_and_finalize_page(_suspect_state(attempts=attempts), 1)
        body = next(
            line
            for line in record.output.text.splitlines()
            if line.strip() and not line.startswith(">")
        )
        assert is_page_failed_marker(body), (
            f"the floored body is invisible to the shared recogniser: {body!r}"
        )
        assert _shipped_marker_reason(body) is PagePrimaryReason.PRIOR_OCR_LAYER_UNREAD, (
            "the shipped bytes do not name which floor they came from"
        )
        assert "socr kept this model reading" not in record.output.text, (
            "the page claims a model reading shipped, which the floor deleted"
        )

    assert ("earlier OCR, not re-read", PagePrimaryReason.PRIOR_OCR_LAYER_UNREAD) in (
        _MARKER_FAMILY_REASONS
    )
    assert (
        SelectionProvenance.PRIOR_OCR_LAYER_UNREAD in _UNACCEPTED_OUTPUT_MARK_EXCLUDED_PROVENANCES
    )


def test_a_clean_page_is_untouched() -> None:
    """Control: the floor fires on a positive detection, nothing else."""
    from socr.core.manifest import _select_and_finalize_page

    state = _suspect_state(attempts=True, detected=False)
    record = _select_and_finalize_page(state, 1)
    assert "3 Y," in record.output.text, "a page with no detection lost its native text"


def test_1027s_own_pages_keep_1027s_ending() -> None:
    """Scoping pin: #961's key is not widened, so #1027's pages are byte-for-byte as they
    were. Part A's floor is placed AFTER that branch for exactly this reason."""
    from socr.core.manifest import _select_and_finalize_page

    record = _select_and_finalize_page(
        _suspect_state(attempts=True, detected=False, invisible=True), 1
    )
    assert record.output.failure_mode is FailureMode.INVISIBLE_SCAN_UNREAD
    assert "invisible OCR layer unread" in record.output.text


# ---------------------------------------------------------------------------
# P0: the DOCUMENT cannot report a clean run over such a page
# ---------------------------------------------------------------------------


def test_the_document_does_not_report_success_over_a_prior_ocr_page(tmp_path: Path) -> None:
    """Argon: the run finished ``status="completed"`` while pages carried an unverified
    earlier OCR layer. A difference pin: the same document with the route neutralised.
    """
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.result import DocumentStatus
    from socr.pipeline.orchestrator import UnifiedPipeline

    pdf = _prior_ocr_pdf(tmp_path / "doc.pdf")

    def run(tag: str, *, detect: bool):
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                native_first=True,
                native_only=True,
                write_manifest=False,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                save_figures=False,
            )
        )
        pipe._available_engines_for_agentic = lambda: []
        pipe._resolve_judge_model = lambda *a, **k: ""
        pipe._resolve_crop_vlm_model = lambda: None
        if not detect:
            original = BornDigitalDetector._has_prior_ocr_layer
            BornDigitalDetector._has_prior_ocr_layer = lambda s, p, r, f=None: False
            try:
                return pipe.process(pdf, output_dir=tmp_path / f"out-{tag}")
            finally:
                BornDigitalDetector._has_prior_ocr_layer = original
        return pipe.process(pdf, output_dir=tmp_path / f"out-{tag}")

    quiet = run("quiet", detect=False)
    flagged = run("flagged", detect=True)

    assert quiet.status is DocumentStatus.SUCCESS, (
        "the control document is not clean, so the pin below shows nothing"
    )
    assert flagged.status is not DocumentStatus.SUCCESS, (
        "the document reported a clean run over pages whose text is an unverified earlier OCR layer"
    )


def test_the_document_is_not_clean_when_a_model_wins_on_a_suspect_scan(tmp_path: Path) -> None:
    """The accepted-reading arm of the same rule, through the real assemble.

    Deliberately NOT a re-implementation of the bucket's comprehension in the test: this
    drives ``_phase_assemble`` with a model output that is SUCCESS and ``audit_passed``
    true, which is the shape that escaped the mark when the bucket was narrowed to native
    winners. Difference pin on the flag alone.
    """
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.result import DocumentStatus
    from socr.pipeline.orchestrator import UnifiedPipeline

    def assemble(*, suspect: bool, tag: str):
        pipe = UnifiedPipeline(
            PipelineConfig(
                primary_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                quiet=True,
                judge_backend="heuristic",
                save_figures=False,
                write_manifest=False,
            )
        )
        state = DocumentState(
            handle=DocumentHandle(path=_prior_ocr_pdf(tmp_path / "m.pdf"), page_count=1)
        )
        p = PageState(page_num=1)
        p.is_born_digital = True
        p.native_text = "the earlier OCR layer, numbers included: 3 Y, per cent"
        p.prior_ocr_layer_suspect = suspect
        winner = PageOutput(
            page_num=1,
            text="A MODEL READING THE JUDGE ACCEPTED",
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
            failure_mode=FailureMode.NONE,
        )
        p.attempts = [winner]
        p.best_output = winner
        state.pages[1] = p
        out = tmp_path / f"asm-{tag}"
        out.mkdir(parents=True, exist_ok=True)
        return pipe._phase_assemble(state, out)

    clean = assemble(suspect=False, tag="clean")
    flagged = assemble(suspect=True, tag="flagged")

    assert clean.status is DocumentStatus.SUCCESS, (
        "the control document is not clean, so the pin below shows nothing"
    )
    assert flagged.status is not DocumentStatus.SUCCESS, (
        "the document reported a clean run over a page whose only reading was one model's "
        "pass over an earlier OCR layer"
    )


def test_the_detector_still_sees_folio_pages_on_a_later_document(tmp_path: Path) -> None:
    """One detector, several documents -- which is how a run uses it.

    The folio slots were stored on the attribute that held the METHOD, so document two
    raised "'frozenset' object is not callable", the handler swallowed it and emptied the
    slots, and every prior-OCR scan with a running header after the first document went
    undetected and shipped as a clean SUCCESS (Argon, re-review).
    """

    def stamped(path: Path, first_folio: int) -> Path:
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page(width=612, height=792)
            for row in range(4):
                page.insert_text(
                    (72, 140 + row * 24),
                    f"{'w' * (i + 1)} the Committee reviewed conditions in the economy",
                    fontsize=10,
                )
            page.insert_image(page.rect, stream=_png(300, 400))
            page.insert_text((72, 20), "November 10, 2004", fontsize=8)
            page.insert_text((300, 20), str(first_folio + i), fontsize=8)
            page.insert_text((330, 20), "of 107", fontsize=8)
        doc.save(path)
        doc.close()
        return path

    first = stamped(tmp_path / "one.pdf", 101)
    second = stamped(tmp_path / "two.pdf", 201)

    detector = BornDigitalDetector()  # ONE detector, as a run reuses it
    first_pages = detector.detect(first).pages
    second_pages = detector.detect(second).pages

    assert first_pages[0].prior_ocr_layer is True
    assert second_pages[0].prior_ocr_layer is True, (
        "the detector stopped seeing folio pages after the first document"
    )


# ---------------------------------------------------------------------------
# One transform, one set of bytes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attempts", [True, False])
def test_a_refinalised_body_is_byte_identical(attempts: bool) -> None:
    """Finalise, then re-finalise with the saved body: one set of bytes.

    ``_phase_assemble`` re-finalises every page with its saved text, and a resume does
    the same from the sidecar, so any transform applied to the page's text runs more than
    once unless it is derived from the source. Part A's floor replaces the body outright
    rather than transforming it, which is what makes it idempotent -- this pins that,
    because Part B will replace this branch with a transform and must not lose it.

    The withheld marker is itself a page-failed marker, so a second pass over it is
    exactly where "[page [number withheld] unchecked" came from in the Part B attempt.
    """
    from socr.core.manifest import _select_and_finalize_page

    state = _suspect_state(attempts=attempts)
    first = _select_and_finalize_page(state, 1)
    again = _select_and_finalize_page(state, 1, saved_text=first.output.text)

    assert again.output.text == first.output.text
    assert again.output.failure_mode is first.output.failure_mode
    # The re-finalised body must not have re-wrapped socr's own lines. On a FLOORED page
    # the marker IS the mark, so the verdict banner is suppressed exactly as it is for
    # #1027's and #263's floors -- that suppression depends on the marker being visible to
    # ``is_page_failed_marker``, which its own test pins. What must not happen is the
    # marker being wrapped, doubled, or re-marked by a second pass.
    assert again.output.text.count("earlier OCR, not re-read") == 1
    assert "[page 1 unchecked:" not in again.output.text
    assert "3 Y," not in again.output.text and "400" not in again.output.text


# ---------------------------------------------------------------------------
# Retaining the original, and resume
# ---------------------------------------------------------------------------


def _pipeline():
    from socr.core.config import EngineType, PipelineConfig
    from socr.pipeline.orchestrator import UnifiedPipeline

    return UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            judge_backend="heuristic",
            save_figures=False,
            write_manifest=False,
        )
    )


def _flushed(tmp_path: Path, *, suspect: bool) -> dict:
    import json

    out_dir = tmp_path / ("sus" if suspect else "clean")
    out_dir.mkdir(parents=True, exist_ok=True)
    state = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=1))
    p = PageState(page_num=1)
    p.is_born_digital = True
    p.native_text = "Raised discount rates from 3 to 3 Y, per cent and $400 million"
    p.prior_ocr_layer = suspect
    p.prior_ocr_layer_suspect = suspect
    p.best_output = PageOutput(
        page_num=1,
        text="[page 1: earlier OCR, not re-read — see image]",
        status=PageStatus.WARNING,
        engine="native",
        audit_passed=False,
        failure_mode=FailureMode.PRIOR_OCR_LAYER_UNVERIFIED,
    )
    state.pages[1] = p
    path = Path(_pipeline()._flush_page_sidecar(state, 1, out_dir, terminal=True))
    return json.loads(path.read_text())


def test_the_original_layer_is_retained_in_the_sidecar_and_never_shipped(tmp_path: Path) -> None:
    """The owner's "so we don't lose job done": the layer is kept where Part B can reach
    it, numbers included, and the page itself ships none of it.

    Difference pin on the flag: a clean page's sidecar does not grow the key.
    """
    suspect = _flushed(tmp_path, suspect=True)
    clean = _flushed(tmp_path, suspect=False)

    assert "prior_ocr_layer_text" not in clean
    assert clean.get("prior_ocr_layer_suspect") is None
    assert suspect["prior_ocr_layer_suspect"] is True
    assert "3 Y," in suspect["prior_ocr_layer_text"], (
        "the earlier layer was not retained, so Part B cannot recover its prose without "
        "re-OCRing the page"
    )
    assert "$400 million" in suspect["prior_ocr_layer_text"]
    # ... and none of it reaches the page the reader sees.
    shipped = suspect["winning_output"]["text"]
    assert "3 Y," not in shipped and "400" not in shipped


def test_a_resumed_page_keeps_its_suspicion(tmp_path: Path) -> None:
    """``_restore_terminal_page_state`` rebuilt the page from its sidecar without the
    flag, so the assemble re-flush dropped it: the page came back from a resume looking
    checked on evidence nothing could see.
    """
    sidecar = _flushed(tmp_path, suspect=True)
    resumed = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=1))
    resumed.pages[1] = PageState(page_num=1)
    restored = PageOutput.from_dict(sidecar["winning_output"])
    out_dir = tmp_path / "sus"
    _pipeline()._restore_terminal_page_state(resumed, 1, restored, out_dir)

    assert resumed.pages[1].prior_ocr_layer_suspect is True, (
        "a resumed page lost its suspicion, so the re-flush drops the mark and the page "
        "comes back cleaner than it left"
    )


# ---------------------------------------------------------------------------
# The acceptance meter: both lanes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lane", ["native", "qwen"])
def test_neither_lane_counts_a_flagged_page_as_checked(lane: str) -> None:
    """Coverage proves the layer is PRESENT, never that it is right; and a page judge's
    verdict on one model reading is not a comparison either. The fault is the page's
    evidence, so it is raised in BOTH lanes."""
    from socr.benchmark.acceptance import PRIOR_OCR_UNCHECKED, Page, rule_native, rule_qwen
    from socr.core.text_coverage import VERDICT_PASSED

    def flags(*, suspect: bool) -> list[str]:
        # The sidecar shape each lane's rule reads when it would otherwise say CHECKED.
        sidecar = {"text_coverage": VERDICT_PASSED}
        if suspect:
            sidecar["prior_ocr_layer_suspect"] = True
        sidecar["winning_output"] = {
            "status": "success",
            "engine": "native" if lane == "native" else "qwen",
            "audit_passed": True,
            "judge_outcome": "page_judge_completed",
        }
        page = Page(doc="d", num=1, sidecar=sidecar, md="body", manifest_entry=None)
        return rule_native(page) if lane == "native" else rule_qwen(page)

    assert PRIOR_OCR_UNCHECKED not in flags(suspect=False)
    assert PRIOR_OCR_UNCHECKED in flags(suspect=True), (
        f"the {lane} lane counted a prior-OCR page CHECKED"
    )


# ---------------------------------------------------------------------------
# Round 2: the cross-product, on fixtures that actually reach each lane
# ---------------------------------------------------------------------------
#
# Round 1's grid had 45 green cells and 18 of them were vacuous: the s1_prose and
# tr3_splice families used synthetic bboxes over text with no parseable table, so neither
# lane was ever entered and reverting their fix left every cell green (Argon 2.4, Grok P2).
# Each family below is built on the fixture that lane really needs, and
# ``scratch/1156a/mutants.py`` reverts each fix in turn and requires THAT family to go red.
#
# The two flag families are separate on purpose, because they carry different rules:
#   POSITIVE  (prior_ocr_layer / invisible_text_over_raster) -> the layer is withheld;
#   SCANFAILED (prior_ocr_layer_scan_failed alone)           -> "unknown" is not a finding,
#              so the layer is KEPT under the mark and each lane keeps its own artefacts.

_LANES = ("native", "chart_asset", "math_hybrid", "s1_prose", "tr3_splice", "grid_corroborated")
_ATTEMPTS = ("none", "rejected", "accepted")
# "invisible" is #961's POSITIVE route. Round 2: without it, reverting the floor's
# either-route condition left every cell green, because every positive cell used #1156's
# own flag (the r1/1 mutant). The rule is about a positive FINDING, not about which
# detector made it.
_FLAGS = ("positive", "invisible", "scanfailed")
_LAYER_TOKENS = ("3 Y,", "$400 million")
_ARTEFACTS = {
    "chart_asset": "figures/chart_p1.png",
    "math_hybrid": "figures/math_crop_p1.png",
}


def _grid_layer_text() -> str:
    return (
        "Opening paragraph of the section.\n"
        "Discount rates were raised from 3 to 3 Y, per cent.\n"
        "A total of $400 million was involved.\nClosing paragraph."
    )


def _grid_state(tmp_path: Path, *, flags: str, attempt: str, lane: str):
    """One cell, built to REACH the named lane rather than merely to carry its name."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from socr.core.manifest import ENGINE_NATIVE_MATH, REJECTION_AMBIGUOUS_DEFERRED

    layer = _grid_layer_text()

    if lane in ("s1_prose", "tr3_splice"):
        # gh520's own helper and fixture: the one shape where the GH-520 guard passes and
        # the S1 floor actually recovers prose. Synthetic bboxes cannot, which is what
        # made round 1's cells vacuous.
        from test_gh520_regional_floor_splice import TWO_PARSEABLE_TABLES, _floored_page

        sub = tmp_path / f"{lane}-{flags}-{attempt}"
        sub.mkdir(parents=True, exist_ok=True)
        state, p = _floored_page(sub, f"{layer}\n{TWO_PARSEABLE_TABLES}", detected=2)
        if lane == "tr3_splice":
            p.native_table_structure_failed = True
            p.native_table_unverifiable = True
            # the branch that splices UNCONDITIONALLY; the ordinal branch refuses unless
            # region identities match 1:1, which a synthetic fixture cannot satisfy.
            p.native_table_header_unattributed = True
    else:
        real_pdf = _prior_ocr_pdf(tmp_path / f"grid-{lane}-{flags}-{attempt}.pdf")
        state = DocumentState(handle=DocumentHandle(path=real_pdf, page_count=2))
        p = PageState(page_num=1)
        p.is_born_digital = True
        p.native_text = layer
        p.d3_floor_png_ref = "![Failed table page 1](figures/failed_table_p1.png)"
        state.pages[1] = p

    p.needs_ocr_enhancement = True
    if flags == "positive":
        p.prior_ocr_layer = True
    elif flags == "invisible":
        p.invisible_text_over_raster = True
    else:
        p.prior_ocr_layer_scan_failed = True
    p.prior_ocr_layer_suspect = True
    p.prior_ocr_page_png_ref = "![Scanned page 1](figures/prior_ocr_layer_page_p1.png)"

    if lane == "grid_corroborated":
        # The route that scores a model grid against ``p.native_words`` -- the suspect
        # layer itself. Principle 1: that is not evidence, whichever flag is set.
        p.has_tables = True
        p.detected_table_count = 1
        p.detected_table_bboxes = [(50.0, 50.0, 550.0, 750.0)]
        p.native_words = [
            (72.0, 200.0, 110.0, 212.0, "Rate", 0, 0, 0),
            (200.0, 200.0, 230.0, 212.0, "350", 0, 0, 1),
            (300.0, 200.0, 330.0, 212.0, "400", 0, 0, 2),
            (72.0, 230.0, 110.0, 242.0, "Total", 0, 1, 0),
            (200.0, 230.0, 230.0, 242.0, "500", 0, 1, 1),
            (300.0, 230.0, 330.0, 242.0, "600", 0, 1, 2),
        ]

    model = None
    if attempt != "none":
        accepted = attempt == "accepted"
        text = "A MODEL READING OF THE PAGE"
        if lane == "grid_corroborated":
            text = (
                "| Item | Col1 | Col2 |\n| --- | --- | --- |\n"
                "| Rate | 350 | 400 |\n| Total | 500 | 600 |\n"
            )
        model = PageOutput(
            page_num=1,
            text=text,
            status=PageStatus.SUCCESS if accepted else PageStatus.WARNING,
            engine="qwen",
            audit_passed=accepted,
            failure_mode=FailureMode.NONE if accepted else FailureMode.AUDIT_FAILED,
            rejection_class="" if lane == "grid_corroborated" else REJECTION_AMBIGUOUS_DEFERRED,
        )
        p.attempts = [model]
        p.best_output = model

    # Round 1 overwrote ``best_output`` with the lane's own output even on the accepted
    # arm, so the accepted cells never tested an accepted MODEL reading (Argon 2.4).
    if lane == "chart_asset" and attempt != "accepted":
        win = PageOutput(
            page_num=1,
            text=p.native_text + f"\n\n![chart]({_ARTEFACTS['chart_asset']})",
            status=PageStatus.SUCCESS,
            engine="chart_asset",
            audit_passed=True,
            failure_mode=FailureMode.NONE,
        )
        p.attempts = [*p.attempts, win]
        p.best_output = win
    elif lane == "math_hybrid" and attempt != "accepted":
        p.has_corrupt_math = True
        hybrid = PageOutput(
            page_num=1,
            text=p.native_text + f"\n\n![crop]({_ARTEFACTS['math_hybrid']})",
            status=PageStatus.WARNING,
            engine=ENGINE_NATIVE_MATH,
            audit_passed=False,
            failure_mode=FailureMode.AUDIT_FAILED,
        )
        p.attempts = [*p.attempts, hybrid]
        p.best_output = hybrid
        p.corrupt_math_hybrid = hybrid

    return state


@pytest.mark.parametrize("lane", _LANES)
@pytest.mark.parametrize("attempt", _ATTEMPTS)
@pytest.mark.parametrize("flags", _FLAGS)
def test_the_whole_flag_lane_attempt_grid(
    tmp_path: Path, flags: str, attempt: str, lane: str
) -> None:
    """Every cell, under the rule its flag family carries.

    POSITIVE: the layer's numbers never ship. SCANFAILED: "unknown" is not a finding, so
    the layer is kept under the mark and the lane keeps its own artefacts. Both: the page
    is MARKED and the document can never be clean over it.
    """
    from socr.core.manifest import _select_and_finalize_page
    from socr.core.marks import Check
    from socr.core.verdict import build_page_verdict

    state = _grid_state(tmp_path, flags=flags, attempt=attempt, lane=lane)
    record = _select_and_finalize_page(state, 1)
    body = record.output.text
    cell = f"flags={flags} attempt={attempt} lane={lane}"

    if flags in ("positive", "invisible"):
        leaked = [tok for tok in _LAYER_TOKENS if tok in body]
        assert leaked == [], f"{cell}: the earlier OCR layer shipped {leaked}"
        # Only the REJECTED arm: an ACCEPTED reading ships its own grid, numbers and
        # all, which is the point of accepting it. What principle 1 forbids is a
        # rejected one being rescued by the suspect layer's words.
        if lane == "grid_corroborated" and attempt == "rejected":
            assert "350" not in body, (
                f"{cell}: a rejected grid shipped, corroborated against the suspect layer"
            )
    else:
        # "unknown" must not delete anything: the layer and the lane's artefact survive.
        if lane in ("native", "chart_asset", "math_hybrid") and attempt != "accepted":
            assert "3 Y," in body, f"{cell}: a FAILED scan deleted the page's own text"
            artefact = _ARTEFACTS.get(lane)
            if artefact:
                assert artefact in body, f"{cell}: a FAILED scan dropped the lane's artefact"
        if lane in ("s1_prose", "tr3_splice") and attempt != "accepted":
            assert "3 Y," in body, f"{cell}: a FAILED scan deleted the recovered prose"

    clauses = [c.check for c in build_page_verdict(1, record.output, state.pages[1]).clauses]
    assert Check.PRIOR_OCR_LAYER in clauses, f"{cell}: the page is not marked"
    assert record.output.audit_passed is False or record.output.status is not PageStatus.SUCCESS, (
        f"{cell}: the page reads as a clean pass"
    )

    again = _select_and_finalize_page(state, 1, saved_text=body)
    assert again.output.text == body, f"{cell}: re-finalizing changed the body"


# ---------------------------------------------------------------------------
# Round 1: the PROVIDER path, which is the path the Fed run takes
# ---------------------------------------------------------------------------


class _StubReadingEngine:
    """A provider that returns one reading per page, so a judge can reject it."""

    name = "qwen"

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        return [
            PageOutput(
                page_num=n,
                text="A MODEL READING OF THE PAGE",
                status=PageStatus.SUCCESS,
                engine="qwen",
            )
            for n in page_nums
        ]


class _FixedJudge:
    def __init__(self, accept: bool) -> None:
        self._accept = accept

    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        return AcceptDecision(accept=self._accept, reason="fixed judge")


def _provider_run(tmp_path: Path, monkeypatch, tag: str, *, accept: bool):
    """gh1027's ``_run`` shape, on the painted-under fixture, with figures saved."""
    import json

    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch
    from socr.pipeline.orchestrator import UnifiedPipeline

    pdf = _prior_ocr_pdf(tmp_path / f"{tag}.pdf")
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: _StubReadingEngine())
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                native_first=True,
                write_manifest=False,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                save_figures=True,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
        pipe._build_page_judge = lambda state: _FixedJudge(accept)
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        result = pipe.process(pdf, output_dir=tmp_path / f"out-{tag}")
    out = tmp_path / f"out-{tag}"
    side = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())
    text = next(iter(out.rglob("pages/00001.md"))).read_text()
    audit = json.loads(next(iter(out.rglob("audit_log.json"))).read_text())
    return result, side, text, audit["events"]


def test_e2e_the_provider_path_floor_ships_the_marker_and_the_page_image(
    tmp_path: Path, monkeypatch
) -> None:
    """Round 1 (r-uben blocker 2): no test drove this path, and it is the Fed run's path.

    Every other floor test here is PNG-less or goes through ``--native-only``, so two
    mutants survived the full suite: dropping ``failed:`` from the SEE-IMAGE marker
    variant, and switching off the page-image render. Both are killed from here.
    """
    from socr.core.manifest import is_page_failed_marker
    from socr.core.result import DocumentStatus

    result, side, text, events = _provider_run(tmp_path, monkeypatch, "rej", accept=False)

    body = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith(">")]
    assert is_page_failed_marker(body[0]), f"the floor marker is not recognisable: {body[:2]}"
    assert "earlier OCR, not re-read" in text
    # The page IMAGE, which only this path renders.
    assert "prior_ocr_layer_page_p1" in text, f"the page image is missing: {text[:300]}"
    assert "see image" in text, "the marker does not point at the image it shipped with"

    # Neither the layer nor the rejected reading.
    assert "3 Y," not in text and "$400 million" not in text
    assert "A MODEL READING OF THE PAGE" not in text

    # A floored page is NOT a native fallback: that event claims native text shipped.
    kinds = [e.get("kind") for e in events if e.get("page_num") == 1]
    assert "native_fallback" not in kinds, f"the floored page was called a native fallback: {kinds}"

    assert side["prior_ocr_layer_suspect"] is True
    assert side["winning_output"]["status"] != "success"
    assert result.status is not DocumentStatus.SUCCESS


def test_e2e_the_provider_path_accepted_reading_ships_marked(tmp_path: Path, monkeypatch) -> None:
    """The other arm of the same path: an accepted reading ships, under its mark."""
    from socr.core.result import DocumentStatus

    result, side, text, _events = _provider_run(tmp_path, monkeypatch, "acc", accept=True)

    assert "A MODEL READING OF THE PAGE" in text, "the accepted reading did not ship"
    assert "3 Y," not in text and "$400 million" not in text, "the layer shipped beside it"
    assert "prior_ocr_layer=UNKNOWN" in text, "an accepted reading of a scan shipped unmarked"
    assert side["winning_output"]["status"] != "success", "the page reads as a clean pass"
    assert result.status is not DocumentStatus.SUCCESS


def test_a_restored_page_keeps_the_retained_layer(tmp_path: Path) -> None:
    """#1156 round 1 (Argon C): a re-flush from restored state used to WIPE the retention.

    Two halves, both pinned, because the second is why this was latent rather than live:

    1. ``_restore_terminal_page_state`` brought back the suspicion flag but not the layer,
       so the next sidecar write saw ``native_text == ""`` and omitted
       ``prior_ocr_layer_text`` -- the only thing Part A stores for Part B to re-finalize
       from, gone without a word.
    2. The page ledger admits only a TERMINAL SUCCESS page, and a flagged page is always
       WARNING, so a normal resume never reaches that restore on a flagged page. That is
       what made it latent; it is pinned here so a later change to either the ledger gate
       or the page's status cannot make it live unnoticed.
    """
    import json

    sidecar = _flushed(tmp_path, suspect=True)
    assert "3 Y," in sidecar["prior_ocr_layer_text"], "fixture premise: the layer is retained"

    resumed = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=1))
    resumed.pages[1] = PageState(page_num=1)
    restored = PageOutput.from_dict(sidecar["winning_output"])
    out_dir = tmp_path / "sus"
    _pipeline()._restore_terminal_page_state(resumed, 1, restored, out_dir)

    assert resumed.pages[1].prior_ocr_layer_suspect is True
    assert "3 Y," in (resumed.pages[1].native_text or ""), (
        "the restored page lost the retained layer, so the next flush drops it"
    )

    # ... and the re-flush keeps it, which is the consequence that matters.
    again_path = Path(_pipeline()._flush_page_sidecar(resumed, 1, out_dir, terminal=True))
    again = json.loads(again_path.read_text())
    assert "3 Y," in again.get("prior_ocr_layer_text", ""), "the re-flush wiped the retention"

    # Half 2: the status that keeps this off the normal resume path.
    assert sidecar["winning_output"]["status"] != "success", (
        "a flagged page became terminal SUCCESS, which puts it on the ledger's skip path"
    )


def test_a_detector_failure_is_suspicion_not_trust(tmp_path: Path, monkeypatch) -> None:
    """#1156 round 1 (r-uben blocker 3): a PyMuPDF failure must mean "unknown", not "clean".

    The two opacity reads used to swallow exceptions, so every image read not-opaque and
    ``_has_prior_ocr_layer`` RETURNED False instead of raising: the caller's ``except``
    never ran, ``prior_ocr_layer_scan_failed`` was unreachable from a PyMuPDF failure, and
    a detected scan became a clean pass. Measured at ``process()`` on this fixture: the
    document went ERROR -> SUCCESS, the page WARNING -> SUCCESS, the mark vanished, and the
    layer's numbers shipped unflagged.

    The witness keeps the opposite direction, which is the whole reason the swallow looked
    right: ``page_has_opaque_page_covering_raster`` catches and answers False, so a failure
    there means "nothing read these pixels" and the native check stays. Both directions are
    asserted, because fixing one by breaking the other is the trap.
    """
    import json

    from socr.core.born_digital import _FURNITURE_CACHE, page_has_opaque_page_covering_raster
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.result import DocumentStatus
    from socr.pipeline.orchestrator import UnifiedPipeline

    pdf = _prior_ocr_pdf(tmp_path / "failing.pdf")
    real = fitz.Document.xref_get_key
    # The two reads a PyMuPDF failure can hit are tested SEPARATELY, because the first
    # guards the second: ``_page_declares_transparency`` reads ``Resources`` and runs
    # before ``_image_is_opaque`` reads ``SMask``, so breaking both at once leaves the
    # second helper's own swallow unreachable and therefore unpinned.
    break_keys = {"Resources"}

    raised: list = [RuntimeError("simulated PyMuPDF failure")]

    def boom(self, xref, key):
        # BOTH reads a PyMuPDF failure can hit: the page's resource dictionary
        # (``_page_declares_transparency``) and the image's soft mask
        # (``_image_is_opaque``). Each used to swallow on its own, so breaking only one
        # leaves the other's swallow unpinned.
        if key in break_keys:
            raise raised[0]
        return real(self, xref, key)

    # Direction 1: the DETECTOR must raise, so its caller can record suspicion.
    detector = BornDigitalDetector()
    with fitz.open(pdf) as doc:
        slots = detector._repeated_text_slots(doc)
        folios = detector._folio_slots(doc, slots)
        monkeypatch.setattr(fitz.Document, "xref_get_key", boom)
        for keys in ({"Resources"}, {"SMask"}, {"ImageMask"}):
            break_keys.clear()
            break_keys.update(keys)
            with pytest.raises(Exception):
                detector._has_prior_ocr_layer(doc[0], slots, folios)
        break_keys.clear()
        break_keys.add("Resources")
        # Direction 2: the WITNESS must still answer, and answer "no witness".
        _FURNITURE_CACHE.clear()
        assert page_has_opaque_page_covering_raster(doc[0]) is False, (
            "the witness raised or claimed a reading; a failure there must keep the "
            "native check, not skip it"
        )
        monkeypatch.undo()

    # And the consequence at process(): suspicion, a mark, and an unclean document.
    # Round 2 (r-uben blocker 5): PyMuPDF's own ``xref_get_key`` raises
    # ``ValueError("bad xref")``, so the round-1 catch of ``(IndexError, TypeError,
    # ValueError)`` swallowed exactly the failure it was meant to let through. Both
    # exception kinds are driven here; a malformed VALUE stays a quiet False, which its
    # own assertion below pins.
    def run(tag: str, *, breaking: bool, exc=None):
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                native_first=True,
                native_only=True,
                write_manifest=False,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                save_figures=False,
            )
        )
        pipe._available_engines_for_agentic = lambda: []
        pipe._resolve_judge_model = lambda *a, **k: ""
        pipe._resolve_crop_vlm_model = lambda: None
        out = tmp_path / f"out-{tag}"
        if breaking:
            if exc is not None:
                raised.clear()
                raised.append(exc)
            monkeypatch.setattr(fitz.Document, "xref_get_key", boom)
        try:
            result = pipe.process(pdf, output_dir=out)
        finally:
            if breaking:
                monkeypatch.undo()
        side = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())
        return result, side

    intact, intact_side = run("intact", breaking=False)
    broken, broken_side = run("broken", breaking=True)
    # The kind PyMuPDF actually raises.
    value_doc, value_side = run("valueerr", breaking=True, exc=ValueError("bad xref"))
    assert value_side["prior_ocr_layer_suspect"] is True, (
        "a PyMuPDF ValueError was swallowed, so the page ships clean with its layer"
    )
    assert value_doc.status is not DocumentStatus.SUCCESS

    # ... while a malformed VALUE stays a quiet False, which is why the two are separated
    # by WHERE they happen rather than by exception type.
    from socr.core.born_digital import _any_alpha_below_one

    assert _any_alpha_below_one("/ca notanumber") is False
    assert _any_alpha_below_one("/ca 0.5") is True

    assert intact_side["prior_ocr_layer_suspect"] is True
    assert broken_side["prior_ocr_layer_suspect"] is True, (
        "a PyMuPDF failure left the page unflagged, so it ships as a clean pass"
    )
    assert intact.status is not DocumentStatus.SUCCESS
    assert broken.status is not DocumentStatus.SUCCESS, (
        "the document reported a clean run over a page nothing could read"
    )


def test_a_flagged_page_never_enters_the_math_recovery_lane() -> None:
    """#1156 round 1 (Argon): the math lane took the page OUT of whole-page OCR.

    ``_is_corrupt_math_recovery_page`` owns a page instead of the model, so a flagged page
    with corrupt maths was never read at all -- on the Fed run, GPU work skipped and never
    redone, with the hybrid shipping the earlier layer's prose. The same function already
    excludes #961's two keys for exactly this reason, so the flag joins them there rather
    than being withheld inside the lane: the page goes to OCR and reaches either an
    accepted reading or the floor.

    Difference pin on the flag alone; the failed-scan key is included because an unreadable
    page is not prose to repair around equations either.
    """
    from socr.core.config import EngineType, PipelineConfig
    from socr.pipeline.orchestrator import UnifiedPipeline

    pipe = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            recover_corrupt_math=True,
        )
    )

    def owns(**flags) -> bool:
        ps = PageState(page_num=1)
        ps.is_born_digital = True
        ps.native_text = "rates of 6-1/2 to 9-1/2 per cent"
        ps.has_corrupt_math = True
        for k, v in flags.items():
            setattr(ps, k, v)
        ps.prior_ocr_layer_suspect = bool(
            flags.get("prior_ocr_layer") or flags.get("prior_ocr_layer_scan_failed")
        )
        return pipe._is_corrupt_math_recovery_page(1, ps)

    assert owns() is True, "control: an unflagged corrupt-math page is the lane's own"
    assert owns(prior_ocr_layer=True) is False, (
        "a flagged page is still the math lane's, so the model never reads it"
    )
    assert owns(prior_ocr_layer_scan_failed=True) is False, (
        "an unreadable page is not prose to repair around equations"
    )


def test_the_s1_floor_and_the_tr3_splice_withhold_the_layers_prose(tmp_path: Path) -> None:
    """#1156 round 1 (r-uben 1b; Grok measured the other cell of the same branch).

    Both endings splice the page's own ``native_text`` around their table markers, and on
    a flagged page that text is an EARLIER OCR run. Reproduced on gh520's own
    ``_floored_page`` + ``TWO_PARSEABLE_TABLES`` fixture, which is the one shape where the
    GH-520 guard PASSES and prose recovery actually runs: flagged and unflagged were
    byte-identical apart from the banner, both carrying "3 to 3 Y, per cent".

    Why the two reviewers disagreed, pinned here so it is not rediscovered: recovery runs
    only when the guard passes (count > 0, boxes == count, a sane bbox, a parsed block per
    table). Guard fails -> whole-page marker, which is what Grok saw; guard passes ->
    prose, which is what r-uben saw, and what a real minutes page with parseable tables
    does. The unflagged arm is asserted too, so the fix cannot be "withhold from everyone".
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_gh520_regional_floor_splice import (
        PROSE_BEFORE,
        TWO_PARSEABLE_TABLES,
        _floored_page,
    )

    from socr.core.manifest import _winning_page_output

    rate = "Discount rates were raised from 3 to 3 Y, per cent."
    money = "A total of $400 million was involved."
    text = f"{rate}\n{money}\n{TWO_PARSEABLE_TABLES}"

    for tr3 in (False, True):
        bodies = {}
        for flagged in (False, True):
            sub = tmp_path / f"s1-{tr3}-{flagged}"
            sub.mkdir(parents=True, exist_ok=True)
            state, ps = _floored_page(sub, text, detected=2)
            ps.prior_ocr_layer = flagged
            ps.prior_ocr_layer_suspect = flagged
            if tr3:
                # The TR-3 ending, which splices the same text by a different route.
                # ``native_table_header_unattributed`` is the branch that splices
                # UNCONDITIONALLY (GH-375: no per-table header identity exists, so every
                # table is replaced and the surrounding prose kept). The ordinal branch
                # beside it refuses unless region identities match 1:1, which this
                # synthetic fixture cannot satisfy -- so pointing the test at that one
                # made the guard inert and its mutant survived.
                ps.native_table_structure_failed = True
                ps.native_table_unverifiable = True
                ps.native_table_header_unattributed = True
            bodies[flagged] = _winning_page_output(state, 1).text

        lane = "TR-3 splice" if tr3 else "S1 prose recovery"
        # The control must actually recover prose, or the pin below shows nothing.
        if not tr3:
            assert PROSE_BEFORE in bodies[False], (
                f"{lane}: the control did not recover prose, so this test proves nothing"
            )
        assert rate not in bodies[True], f"{lane}: the earlier layer's rate shipped"
        assert money not in bodies[True], f"{lane}: the earlier layer's figure shipped"
        assert "failed:" in bodies[True], f"{lane}: no marker shipped in its place"
        # Round 2 (r-uben 6): on a Fed minutes page the image is ALL that is left of the
        # table once the prose is withheld, so losing it loses the page.
        assert "failed_table_p1.png" in bodies[True], (
            f"{lane}: the floor withheld the prose and shipped no page image either"
        )


def test_a_disputed_grid_on_a_flagged_page_carries_its_page_image(tmp_path: Path) -> None:
    """#1156 round 2 (Argon 1, third S1 branch): the ruling's other half.

    ``STRUCTURE_CLASS_GRID_FLAGGED`` ships the MODEL's disputed grid, not the layer, so
    principle 2 is already satisfied -- no prior-layer number reaches the page, and that
    is asserted here too. What the owner's 2026-10-12 ruling ALSO requires of a disputed
    scanned table is a page-image link beside the mark, and this branch carried the marks
    without one. On a page whose text layer is an earlier OCR run that link is the only
    way a reader can check a disputed number: the layer cannot be the check.

    Difference pin on the flag, so every other disputed grid still ships byte-for-byte.
    """
    from socr.core.manifest import REJECTION_AMBIGUOUS_DEFERRED, _select_and_finalize_page

    png = "![Scanned page 1](figures/prior_ocr_layer_page_p1.png)"

    def ship(*, flagged: bool) -> str:
        state = DocumentState(
            handle=DocumentHandle(path=_prior_ocr_pdf(tmp_path / "disputed.pdf"), page_count=2)
        )
        p = PageState(page_num=1)
        p.is_born_digital = True
        p.native_text = _grid_layer_text()
        p.needs_ocr_enhancement = True
        p.has_tables = True
        p.prior_ocr_layer = flagged
        p.prior_ocr_layer_suspect = flagged
        p.prior_ocr_page_png_ref = png
        rejected = PageOutput(
            page_num=1,
            text="| Metric | 1956 | 1957 |\n| --- | --- | --- |\n| Rate | 99.9 | 88.8 |\n",
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=False,
            judge_outcome="page_judge_completed",
            judge_accepted=False,
            rejection_class=REJECTION_AMBIGUOUS_DEFERRED,
        )
        p.attempts = [rejected]
        p.best_output = rejected
        state.pages[1] = p
        return _select_and_finalize_page(state, 1).output.text

    clean, flagged_body = ship(flagged=False), ship(flagged=True)

    assert "99.9" in flagged_body, "the disputed grid itself must still ship"
    assert "3 Y," not in flagged_body and "$400 million" not in flagged_body, (
        "a prior-layer number reached the disputed grid's page"
    )
    assert "unaccepted" in flagged_body, "the disputed mark is gone"
    assert png in flagged_body, (
        "a disputed grid on a flagged page shipped with no page image, so a reader has "
        "nothing to check its numbers against"
    )
    assert png not in clean, "the link was added to every disputed grid, not just flagged ones"


def test_the_meter_reports_the_prior_ocr_class_in_every_lane() -> None:
    """#1156 round 2 (Argon P1): the meter is the gate, so a lane it does not read passes.

    Two separate holes. ``PRIOR_OCR_UNCHECKED`` was missing from ``VIOLATION_CLASSES``, so
    even where a rule raised it the class never reached ``to_json()["violations"]`` or
    ``format_report()`` -- the meter counted the page unclean and then did not say so. And
    ``rule_chart`` never raised it at all, so a ``chart_asset`` SUCCESS on a flagged page
    counted CHECKED, which is the lane that ships the layer.
    """
    from socr.benchmark.acceptance import (
        LANE_RULES,
        PRIOR_OCR_UNCHECKED,
        VIOLATION_CLASSES,
        Page,
    )
    from socr.core.text_coverage import VERDICT_PASSED

    assert PRIOR_OCR_UNCHECKED in VIOLATION_CLASSES, (
        "the class cannot be reported, so the meter cannot say why a page is unclean"
    )

    def flags(rule, *, suspect: bool) -> list[str]:
        sidecar = {"text_coverage": VERDICT_PASSED}
        if suspect:
            sidecar["prior_ocr_layer_suspect"] = True
        sidecar["winning_output"] = {
            "status": "success",
            "audit_passed": True,
            "judge_outcome": "page_judge_completed",
        }
        return rule(Page(doc="d", num=1, sidecar=sidecar, md="body", manifest_entry=None))

    # Every lane the meter can route a page to, not just the two #1156 started with.
    for engine, rule in sorted(LANE_RULES.items()):
        clean, suspect = flags(rule, suspect=False), flags(rule, suspect=True)
        # A lane that is NEVER checked (it must be marked) cannot count a flagged page
        # clean, so it is allowed not to name this class.
        if clean and not suspect:
            continue
        assert suspect, f"lane {engine} returned no violation for a flagged page"
        # No ``or clean == suspect`` escape: that let a lane whose OTHER violation
        # happened to fire pass without ever naming this one, which is how the chart
        # lane's mutant survived.
        assert PRIOR_OCR_UNCHECKED in suspect, (
            f"lane {engine} does not name the prior-OCR class on a flagged page"
        )


def test_the_structure_class_mirror_clause_stays_in_step(tmp_path: Path) -> None:
    """#1156 round 2 (r-uben 7): the document buckets must agree with what SHIPS.

    ``_reaches_structure_class_branch`` mirrors every precondition the S1 branch sits
    behind, and ``manifest.py``'s clean-pass guard carries the same #1156 clause a second
    time so the two cannot drift. A page that reaches S1 is one the #1156 floor yields to,
    so if the mirror and the branch disagree the document's bucket describes an ending the
    page never took -- the exact misclassification that predicate's own docstring records.
    """
    from socr.core.manifest import _reaches_structure_class_branch, _select_and_finalize_page

    for flags in ("positive", "scanfailed"):
        state = _grid_state(tmp_path, flags=flags, attempt="rejected", lane="s1_prose")
        p = state.pages[1]
        mirrors = _reaches_structure_class_branch(p)
        reason = _select_and_finalize_page(state, 1).disposition.primary_reason.value
        took_s1 = reason == "structure_class"
        assert mirrors is took_s1, (
            f"flags={flags}: the mirror says reaches_S1={mirrors} while the page ended in "
            f"{reason!r}, so the document bucket describes an ending it never took"
        )


def test_a_failed_scan_keeps_each_lanes_own_ending(tmp_path: Path) -> None:
    """#1156 round 2: "unknown" must not move a page to a DIFFERENT lane's ending.

    The grid asserts the layer and the artefacts survive a failed scan, which the math
    lane satisfied even with the destructive gate reading the union -- its crop came back
    through the native-recovery ending instead, so the artefact assertion could not tell
    the two apart and the mutant survived. The ENDING is what distinguishes them.
    """
    from socr.core.manifest import _select_and_finalize_page

    endings = {}
    for flags in ("positive", "scanfailed"):
        state = _grid_state(tmp_path, flags=flags, attempt="rejected", lane="math_hybrid")
        endings[flags] = _select_and_finalize_page(state, 1).disposition.primary_reason.value

    assert endings["scanfailed"] == "corrupt_math_hybrid", (
        "a failed scan pushed the page out of the math lane, so its hybrid was suppressed "
        f"on 'unknown' alone (ended in {endings['scanfailed']!r})"
    )
    assert endings["positive"] == "prior_ocr_layer_unread", (
        f"a positively flagged page stayed in the math lane (ended in {endings['positive']!r})"
    )


@pytest.mark.parametrize("flags", ["positive", "invisible", "scanfailed"])
def test_the_row_corroboration_abstains_on_a_suspect_page(tmp_path: Path, flags: str) -> None:
    """#1156 round 2 (Argon 1, P0): the suspect layer is never EVIDENCE.

    ``_row_corroboration_scan`` scores a model grid against ``p.native_words``. On a page
    whose text layer is an earlier OCR run those words ARE that layer, so the route
    corroborated a reading against the thing under suspicion, and a HARD-rejected grid
    shipped its numbers on that strength (Argon's probe, on a real 1957 Fed page).

    Pinned at the helper rather than end-to-end on purpose: the S1 route needs a real
    page's geometry to reach the corroboration pool, and a synthetic fixture declines for
    its own reasons, so an end-to-end cell here would pass whether the fix were present or
    not. The control arm below is the guard against that -- the SAME page with the flag
    cleared must still produce a scan, or this test proves nothing.

    The abstention takes the UNION flag, scan-failed included: "unknown" is not a reason
    to trust the words either, and abstaining removes trust without deleting anything --
    the page falls through to its own floor, marker and image intact.
    """
    from socr.core.manifest import _row_corroboration_scan

    state = _grid_state(tmp_path, flags=flags, attempt="rejected", lane="grid_corroborated")
    p = state.pages[1]

    assert _row_corroboration_scan(p) is None, (
        f"flags={flags}: the route scored a reading against the suspect layer's own words"
    )

    p.prior_ocr_layer = False
    p.invisible_text_over_raster = False
    p.prior_ocr_layer_scan_failed = False
    p.prior_ocr_layer_suspect = False
    assert _row_corroboration_scan(p) is not None, (
        "control: this page does not reach the corroboration pool at all, so the "
        "abstention above is vacuous"
    )
