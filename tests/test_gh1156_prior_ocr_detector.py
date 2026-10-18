"""#1156 PR 2: the prior-OCR layer read from the page's DRAWING ORDER.

#961 asks whether the page's text is painted in render mode 3 over a page-sized raster.
Every page the Fed pilot counted CHECKED hides its layer the other way -- ordinary fill
text painted OVER by the raster -- so #961 cannot see it. This route reads
``get_bboxlog`` (the drawing operations in order) instead, and never ``get_texttrace``,
which aborts the process on some documents and cost 610 pages of one Fed pass.

Both routes are kept, and each is pinned by a page only IT catches:
``test_only_the_drawing_order_route_sees_text_under_the_raster`` and
``test_only_the_render_mode_route_sees_a_masked_raster``. A route no test misses alone is
a route not shown to guard anything.

Positive evidence only. "No visible text of its own" is decided by repetition across the
document's pages (a stamp, header or page number), never by a guessed margin: the two
stamp tests below differ in nothing but whether the stamp repeats.

Hermetic: synthetic PDFs, no model, no provider.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.born_digital import BornDigitalDetector  # noqa: E402


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def _png(w: int, h: int, *, alpha: bool = False) -> bytes:
    from PIL import Image

    img = Image.new("LA" if alpha else "L", (w, h), (245, 90) if alpha else 245)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _doc(pages: int = 2):
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page(width=612, height=792)
    return doc


def _body(page, *, render_mode: int = 0, nonce: str = "") -> None:
    """The page's own reading. ``nonce`` prefixes each line with a string whose LENGTH
    differs per page, so every word after it lands at a different x: identical pages
    would make the BODY repeat at the same slots, and repetition is the evidence these
    tests rest on."""
    for i in range(12):
        page.insert_text(
            (72, 120 + i * 18),
            f"{nonce}body line {i} of the page's own reading",
            fontsize=10,
            render_mode=render_mode,
        )


def _verdict(doc, page_index: int = 0) -> bool:
    d = BornDigitalDetector()
    slots = d._repeated_text_slots(doc)
    return d._has_prior_ocr_layer(doc[page_index], slots, d._folio_slots(doc, slots))


def _render_mode_verdict(doc, page_index: int = 0) -> bool:
    return BornDigitalDetector()._has_invisible_text_over_raster(doc[page_index])


# ---------------------------------------------------------------------------
# Each route catches a page the other misses
# ---------------------------------------------------------------------------


def test_only_the_drawing_order_route_sees_text_under_the_raster() -> None:
    """The shape every CHECKED pilot page had: fill text, then the scan painted over it."""
    doc = _doc()
    for i, page in enumerate(doc):
        _body(page, nonce="w" * (i + 1) + " ")
        page.insert_image(page.rect, stream=_png(300, 400))
    try:
        assert _verdict(doc) is True
        assert _render_mode_verdict(doc) is False, (
            "fixture is inert: #961 already saw this page, so the new route is not "
            "shown to add anything"
        )
    finally:
        doc.close()


def test_only_the_render_mode_route_sees_a_masked_raster() -> None:
    """A render-mode-3 layer under a raster that is NOT opaque.

    The new route refuses it (no opaque raster is positive evidence of a scan), #961
    takes it on render mode alone. Deleting #961's branch would lose this page, which is
    why both routes are kept.
    """
    doc = _doc()
    for i, page in enumerate(doc):
        _body(page, render_mode=3, nonce="w" * (i + 1) + " ")
        page.insert_image(page.rect, stream=_png(300, 400, alpha=True))
    try:
        assert _render_mode_verdict(doc) is True
        assert _verdict(doc) is False
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Positive evidence: what must NOT fire
# ---------------------------------------------------------------------------


def test_a_page_that_prints_its_own_words_is_not_a_scan() -> None:
    """Gentzkow p30's shape, and it must carry HIDDEN text too.

    p30 has both: an invisible layer (961 of its 2,270 characters) AND 265 words of its
    own over the raster. A fixture with no hidden text would be refused one step earlier
    and would never reach the rule this pins.
    """
    doc = _doc()
    for i, page in enumerate(doc):
        page.insert_text((72, 90), "a caption the raster hides", fontsize=9, render_mode=3)
        page.insert_image(page.rect, stream=_png(300, 400))
        _body(page, nonce="w" * (i + 1) + " ")
    try:
        assert _verdict(doc) is False
    finally:
        doc.close()


def test_the_detector_records_the_verdict_on_the_assessment(tmp_path: Path) -> None:
    """Through ``detect()``, not the predicate: a route nothing calls guards nothing.

    Difference between two documents that differ only in whether the raster covers the
    text, so the assessment field cannot pass by being always-on.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)

    def build(name: str, *, cover: bool) -> Path:
        doc = _doc()
        for i, page in enumerate(doc):
            _body(page, nonce="w" * (i + 1) + " ")
            if cover:
                page.insert_image(page.rect, stream=_png(300, 400))
        path = tmp_path / name
        doc.save(path)
        doc.close()
        return path

    detector = BornDigitalDetector()
    scanned = detector.detect(build("scan.pdf", cover=True)).pages[0]
    born = detector.detect(build("born.pdf", cover=False)).pages[0]

    assert scanned.prior_ocr_layer is True, (
        "the drawing-order route never ran in the assessment, so nothing downstream "
        "ever sees its verdict"
    )
    assert born.prior_ocr_layer is False
    assert scanned.prior_ocr_layer_scan_failed is False


def test_a_transparent_overlay_over_text_is_not_a_scan() -> None:
    """Astra's misfire: a see-through full-page image leaves the text readable."""
    doc = _doc()
    for i, page in enumerate(doc):
        _body(page, nonce="w" * (i + 1) + " ")
        page.insert_image(page.rect, stream=_png(300, 400, alpha=True))
    try:
        assert _verdict(doc) is False
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Strip-tiled scans, and the stamp rule
# ---------------------------------------------------------------------------


def test_a_scan_stored_as_strips_is_one_raster() -> None:
    """Kim-Verrecchia p1-36: 29 tiles, the largest 3.5% of the page. Only the UNION
    of the tiles reaches the dominance ratio; the largest one never would."""
    doc = _doc()
    for i, page in enumerate(doc):
        _body(page, nonce="w" * (i + 1) + " ")
        h = page.rect.height / 29
        for strip in range(29):
            band = fitz.Rect(
                page.rect.x0,
                page.rect.y0 + strip * h,
                page.rect.x1,
                page.rect.y0 + (strip + 1) * h,
            )
            # keep_proportion would inset each strip and leave gutters no real scan has.
            page.insert_image(band, stream=_png(300, 16), keep_proportion=False)
    try:
        assert _verdict(doc) is True
    finally:
        doc.close()


def _stamped(pages: int, *, stamp_on: tuple[int, ...]):
    doc = _doc(pages)
    for i, page in enumerate(doc):
        _body(page, nonce="w" * (i + 1) + " ")
        page.insert_image(page.rect, stream=_png(300, 400))
        if i in stamp_on:
            page.insert_text((72, 760), "Content last modified 01/11/2011.", fontsize=8)
    return doc


def test_a_stamp_the_document_repeats_does_not_block_detection() -> None:
    """Fed 2004-11-10 p9/p11: a running header added after scanning, on every page."""
    doc = _stamped(3, stamp_on=(0, 1, 2))
    try:
        assert _verdict(doc) is True
    finally:
        doc.close()


def test_a_line_the_document_prints_once_is_not_shown_to_be_a_stamp() -> None:
    """The accepted miss (Fed 1977-10-18 p1, 1 page in 4,247).

    Same page as above but for the stamp appearing once in the document, so nothing
    distinguishes it from the page's own text. The page stays unknown rather than being
    called a scan on a guessed margin.
    """
    doc = _stamped(3, stamp_on=(0,))
    try:
        assert _verdict(doc) is False
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# The verdict reaches PageState, and failure is suspicion
# ---------------------------------------------------------------------------


def test_the_verdict_reaches_the_page_state_and_failure_is_suspicion(tmp_path: Path) -> None:
    """Difference pin on the two new assessment fields, through ``apply_born_digital``."""
    from socr.core.born_digital import DocumentAssessment, PageAssessment
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState, PageState

    def suspicion(**flags) -> bool:
        state = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=1))
        state.pages[1] = PageState(page_num=1)
        state.apply_born_digital(
            DocumentAssessment(
                path=Path("x.pdf"),
                pages=[
                    PageAssessment(
                        page_num=1,
                        is_born_digital=True,
                        native_text="some native text",
                        confidence=1.0,
                        **flags,
                    )
                ],
            )
        )
        return state.pages[1].prior_ocr_layer_suspect

    assert suspicion() is False
    assert suspicion(prior_ocr_layer=True) is True
    assert suspicion(prior_ocr_layer_scan_failed=True) is True, (
        "a detection failure was read as trust; an unreadable page is unknown, not clean"
    )


def test_a_visible_line_in_a_gutter_is_not_swallowed_by_the_seam_tolerance(
    tmp_path: Path,
) -> None:
    """``_SEAM_PT`` closes sub-point gaps between tiles, nothing wider.

    Two rasters leave a 3 pt gutter with a line of the page's own text in it. A
    tolerance big enough to swallow that line would call this page a bare scan.
    """
    doc = _doc()
    for i, page in enumerate(doc):
        gutter_top = 400.0
        page.insert_text((72, gutter_top + 2), f"a line {'w' * (i + 1)} nothing covers", fontsize=8)
        _body(page, nonce="w" * (i + 1) + " ")
        top = fitz.Rect(page.rect.x0, page.rect.y0, page.rect.x1, gutter_top)
        bottom = fitz.Rect(page.rect.x0, gutter_top + 3, page.rect.x1, page.rect.y1)
        page.insert_image(top, stream=_png(300, 200), keep_proportion=False)
        page.insert_image(bottom, stream=_png(300, 200), keep_proportion=False)
    try:
        assert _verdict(doc) is False
    finally:
        doc.close()


def test_the_render_mode_route_still_runs_in_the_assessment(tmp_path: Path) -> None:
    """#961's own branch, through ``detect()``: deleting it must break something."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = _doc()
    for i, page in enumerate(doc):
        _body(page, render_mode=3, nonce="w" * (i + 1) + " ")
        page.insert_image(page.rect, stream=_png(300, 400, alpha=True))
    path = tmp_path / "masked.pdf"
    doc.save(path)
    doc.close()

    page_assessment = BornDigitalDetector().detect(path).pages[0]
    assert page_assessment.invisible_text_over_raster is True, (
        "#961's route no longer reaches the assessment, so the masked-raster family "
        "it alone catches is lost"
    )
    assert page_assessment.prior_ocr_layer is False
    assert page_assessment.needs_ocr_enhancement is True


# ---------------------------------------------------------------------------
# Round 2 (Astra): opacity, furniture, and the crashing call
# ---------------------------------------------------------------------------


def test_an_invisible_image_does_not_cover_anything(tmp_path: Path) -> None:
    """An image drawn with alpha 0 leaves the rendered pixels unchanged.

    Reading missing or partial opacity as "opaque" let such a page flag its native text,
    which then dropped a check elsewhere. Opacity must be positive evidence.
    """
    doc = _doc()
    for i, page in enumerate(doc):
        _body(page, nonce="w" * (i + 1) + " ")
        page.draw_rect(page.rect, color=None, fill=(1, 1, 1), fill_opacity=0.0)
        page.insert_image(page.rect, stream=_png(300, 400))
    try:
        assert _verdict(doc) is False
    finally:
        doc.close()


def test_aligned_body_columns_are_not_furniture() -> None:
    """Astra's slide-like pages: different words, same positions.

    Counting positions alone made the page's own body "furniture", so a page that
    plainly prints its own text read as a bare scan. The word has to match too.
    """
    doc = _doc()
    for i, page in enumerate(doc):
        page.insert_text((72, 90), "a caption the raster hides", fontsize=9, render_mode=3)
        page.insert_image(page.rect, stream=_png(300, 400))
        for row in range(10):
            # same slots on both pages, different words
            page.insert_text((90, 150 + row * 20), f"alpha{i}{row}", fontsize=10)
            page.insert_text((300, 150 + row * 20), f"beta{i}{row}", fontsize=10)
    try:
        assert _verdict(doc) is False, (
            "body columns that merely align between pages were taken for page furniture"
        )
    finally:
        doc.close()


def test_the_aborting_trace_is_skipped_when_its_answer_cannot_be_yes() -> None:
    """#961 may only call ``get_texttrace`` when a render-mode-3 run exists.

    The call can abort the process (610 pages of one Fed pass). A page with no
    ``ignore-text`` entry has zero invisible characters, so #961's majority test cannot
    hold; skipping the call there is verdict-preserving, and it is exactly the Fed
    under-raster shape that crashed.
    """
    doc = _doc()
    for i, page in enumerate(doc):
        _body(page, nonce="w" * (i + 1) + " ")
        page.insert_image(page.rect, stream=_png(300, 400))
    try:
        page = doc[0]
        assert not any(kind == "ignore-text" for kind, _b in page.get_bboxlog())

        calls: list[int] = []
        real = fitz.Page.get_texttrace

        def counting(self, *a, **kw):
            calls.append(1)
            return real(self, *a, **kw)

        fitz.Page.get_texttrace = counting
        try:
            assert _render_mode_verdict(doc) is False
        finally:
            fitz.Page.get_texttrace = real
        assert calls == [], "the aborting call still runs on a page whose answer it cannot change"
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# The folio rule: counting is the evidence
# ---------------------------------------------------------------------------


def _scan_with_running_header(pages: int, *, numbers) -> "fitz.Document":
    """A scan whose only visible text is "November 10, 2004  <n> of 107"."""
    doc = _doc(pages)
    for i, page in enumerate(doc):
        _body(page, nonce="w" * (i + 1) + " ")
        page.insert_image(page.rect, stream=_png(300, 400))
        page.insert_text((72, 20), "November 10, 2004", fontsize=8)
        page.insert_text((300, 20), str(numbers[i]), fontsize=8)
        page.insert_text((330, 20), "of 107", fontsize=8)
    return doc


def test_a_page_number_that_counts_up_is_page_furniture() -> None:
    """Fed 2004-11-10 p9/p11: the visible text is a running header whose folio changes.

    Requiring the same WORD at the same place (round 2) cost these pages. The folio is
    recognised by the counting itself: same slot, repeated neighbours, and a value whose
    difference from the page index never changes.
    """
    doc = _scan_with_running_header(4, numbers=[101, 102, 103, 104])
    try:
        assert _verdict(doc) is True
    finally:
        doc.close()


def test_numbers_that_do_not_count_up_are_not_a_folio() -> None:
    """The same header with the numbers shuffled: no counting, no folio, no detection.

    This is what stops aligned body text from passing as page furniture -- it would have
    to count upwards in step with the page index to qualify.
    """
    doc = _scan_with_running_header(4, numbers=[101, 97, 240, 12])
    try:
        assert _verdict(doc) is False
    finally:
        doc.close()


def test_the_folio_rule_leaves_aligned_body_columns_quiet() -> None:
    """Astra's slide pages again, now with the folio rule in place."""
    doc = _doc()
    for i, page in enumerate(doc):
        page.insert_text((72, 90), "a caption the raster hides", fontsize=9, render_mode=3)
        page.insert_image(page.rect, stream=_png(300, 400))
        for row in range(10):
            page.insert_text((90, 150 + row * 20), f"alpha{i}{row}", fontsize=10)
            page.insert_text((300, 150 + row * 20), f"beta{i}{row}", fontsize=10)
    try:
        assert _verdict(doc) is False
    finally:
        doc.close()


def test_a_lone_counting_number_with_no_repeated_neighbour_is_not_a_folio() -> None:
    """A body cell that happens to count up is not page furniture.

    A folio sits beside text the document repeats ("November 10, 2004 … of 107"). A bare
    number that counts, with nothing repeated beside it, is just a number on the page --
    and treating it as furniture would let a page of counting body text read as a bare
    scan.
    """
    doc = _doc(4)
    for i, page in enumerate(doc):
        page.insert_text((72, 90), "a caption the raster hides", fontsize=9, render_mode=3)
        page.insert_image(page.rect, stream=_png(300, 400))
        page.insert_text((300, 400), str(101 + i), fontsize=10)
    try:
        assert _verdict(doc) is False
    finally:
        doc.close()
