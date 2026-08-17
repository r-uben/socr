"""GH-127: structure the flat native dump cannot represent must be countable.

The native lane ships ``page.get_text("text")``, a linear dump. Bullet glyphs become
literal characters inside a paragraph, headings become ordinary lines, and a paragraph
number set in the margin lands beside the sentence it annotates. Measured on the
EFO-Nov-2022 corpus: the native lane lost structure on 14 of 14 pages it handled.

These tests build tiny PDFs with PyMuPDF, so they are hermetic -- no engine, no
provider, no fixture binary. Every geometric decision under test is derived from the
page's own font sizes and margins, never a constant, so each test also renders a
scaled variant to prove no absolute threshold crept in.
"""

from __future__ import annotations

import fitz
import pytest

from socr.core.born_digital import detect_native_structure_loss

# PyMuPDF's base-14 fonts silently substitute U+2022 with U+00B7, so a bullet
# inserted with insert_text never reaches the text layer. The bundled "cjk" font
# carries the glyph and ships with PyMuPDF, keeping these tests hermetic.
_BULLET_FONT = fitz.Font("cjk")


def _page(build, scale: float = 1.0) -> fitz.Page:
    doc = fitz.open()
    page = doc.new_page(width=595 * scale, height=842 * scale)
    build(page, scale)
    return page


def _write_bullets(page: fitz.Page, s: float, count: int) -> None:
    writer = fitz.TextWriter(page.rect)
    for i in range(count):
        y = (230 + 20 * i) * s
        writer.append((96 * s, y), "\u2022", font=_BULLET_FONT, fontsize=11 * s)
        writer.append((122 * s, y), f"list item {i}", font=_BULLET_FONT, fontsize=11 * s)
    writer.write_text(page)


def _plain_body(page: fitz.Page, scale: float) -> None:
    for i in range(6):
        page.insert_text(
            (96 * scale, (120 + 16 * i) * scale), f"ordinary body line {i}", fontsize=11 * scale
        )


@pytest.mark.parametrize("scale", [1.0, 1.6])
def test_bullet_glyphs_with_no_markdown_list_are_counted(scale: float) -> None:
    """The page 12 shape: bullet glyphs present, emitted text has no list."""

    def build(page: fitz.Page, s: float) -> None:
        _plain_body(page, s)
        _write_bullets(page, s, 3)

    flat = "ordinary body line 0 • list item 0 • list item 1 • list item 2"
    cues = detect_native_structure_loss(_page(build, scale), flat)

    assert cues["unrepresented_lists"] == 3


@pytest.mark.parametrize("scale", [1.0, 1.6])
def test_bullets_are_not_counted_when_the_output_has_a_markdown_list(scale: float) -> None:
    """No loss to report if the emitted text already carries the list."""

    def build(page: fitz.Page, s: float) -> None:
        _plain_body(page, s)
        _write_bullets(page, s, 3)

    structured = "body\n\n- list item 0\n- list item 1\n- list item 2\n"
    cues = detect_native_structure_loss(_page(build, scale), structured)

    assert cues["unrepresented_lists"] == 0


@pytest.mark.parametrize("scale", [1.0, 1.6])
def test_text_larger_than_body_with_no_heading_in_output_is_counted(scale: float) -> None:
    """Heading detection is relative to the page's own modal body size."""

    def build(page: fitz.Page, s: float) -> None:
        page.insert_text((96 * s, 90 * s), "A Section Title", fontsize=16 * s)
        _plain_body(page, s)

    cues = detect_native_structure_loss(_page(build, scale), "A Section Title ordinary body line 0")

    assert cues["unrepresented_headings"] >= 1


@pytest.mark.parametrize("scale", [1.0, 1.6])
def test_heading_not_counted_when_output_has_an_atx_heading(scale: float) -> None:
    def build(page: fitz.Page, s: float) -> None:
        page.insert_text((96 * s, 90 * s), "A Section Title", fontsize=16 * s)
        _plain_body(page, s)

    cues = detect_native_structure_loss(_page(build, scale), "# A Section Title\n\nbody\n")

    assert cues["unrepresented_headings"] == 0


@pytest.mark.parametrize("scale", [1.0, 1.6])
def test_digits_left_of_the_body_column_are_flagged(scale: float) -> None:
    """A paragraph number set in the margin. Flat text drops it into the prose."""

    def build(page: fitz.Page, s: float) -> None:
        _plain_body(page, s)
        page.insert_text((57 * s, 120 * s), "9", fontsize=11 * s)

    cues = detect_native_structure_loss(_page(build, scale), "9 ordinary body line 0")

    assert cues["marginal_number_cues"] == 1


@pytest.mark.parametrize("scale", [1.0, 1.6])
def test_digits_inside_the_body_column_are_not_flagged(scale: float) -> None:
    """A number in running prose is not marginalia; flagging it would be noise."""

    def build(page: fitz.Page, s: float) -> None:
        _plain_body(page, s)
        page.insert_text((300 * s, 120 * s), "1998", fontsize=11 * s)

    cues = detect_native_structure_loss(_page(build, scale), "body 1998")

    assert cues["marginal_number_cues"] == 0


def test_a_plain_prose_page_reports_no_loss() -> None:
    """The negative control. Over-firing here would make the signal worthless."""
    cues = detect_native_structure_loss(
        _page(_plain_body), "ordinary body line 0 ordinary body line 1"
    )

    assert cues == {
        "unrepresented_lists": 0,
        "unrepresented_headings": 0,
        "marginal_number_cues": 0,
    }


def test_an_empty_page_reports_no_loss() -> None:
    doc = fitz.open()
    cues = detect_native_structure_loss(doc.new_page(), "")

    assert not any(cues.values())


def test_detector_never_raises_on_an_unreadable_page() -> None:
    """A detector that throws would take extraction down with it."""

    class Exploding:
        def get_text(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    cues = detect_native_structure_loss(Exploding(), "text")

    assert cues == {
        "unrepresented_lists": 0,
        "unrepresented_headings": 0,
        "marginal_number_cues": 0,
    }


@pytest.mark.parametrize("scale", [1.0, 1.6])
def test_partially_represented_list_reports_the_shortfall(scale: float) -> None:
    """Three bullets in the source, one in the output, is a loss of two.

    Treating any markdown list as full representation hid the partial case, which is
    the common one -- a lane that recovers some structure looked identical to one that
    recovered all of it.
    """

    def build(page: fitz.Page, s: float) -> None:
        _plain_body(page, s)
        _write_bullets(page, s, 3)

    partial = "body\n\n- list item 0\n"
    cues = detect_native_structure_loss(_page(build, scale), partial)

    assert cues["unrepresented_lists"] == 2


@pytest.mark.parametrize("scale", [1.0, 1.6])
def test_partially_represented_headings_report_the_shortfall(scale: float) -> None:
    def build(page: fitz.Page, s: float) -> None:
        page.insert_text((96 * s, 70 * s), "First Title", fontsize=16 * s)
        page.insert_text((96 * s, 95 * s), "Second Title", fontsize=16 * s)
        _plain_body(page, s)

    cues = detect_native_structure_loss(_page(build, scale), "# First Title\n\nbody\n")

    assert cues["unrepresented_headings"] == 1


def test_more_markdown_than_source_never_reports_negative_loss() -> None:
    """A lane that adds structure is not a loss; the count must floor at zero."""
    cues = detect_native_structure_loss(
        _page(_plain_body), "- a\n- b\n- c\n\n# h1\n\n## h2\n\nbody\n"
    )

    assert cues["unrepresented_lists"] == 0
    assert cues["unrepresented_headings"] == 0
