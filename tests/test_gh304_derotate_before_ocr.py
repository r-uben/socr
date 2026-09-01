"""GH-304: a page whose text runs sideways must be rendered upright for OCR.

socr already detects rotated text (``text_direction_is_rotated``) and refuses
native table reconstruction on such a page — then rendered it sideways anyway and
handed the model an image it predictably misreads. On the reference document
qwen returned the table TRANSPOSED, using row-group labels as column headers.

These tests assert the DIFFERENCE the fix makes, never an absolute outcome: a
rotated page's render changes orientation, a horizontal page's does not.
"""

from __future__ import annotations

import fitz
import pytest

from socr.core.born_digital import (
    dominant_text_direction,
    upright_rotation_degrees,
)
from socr.engines.base import _upright_rotation_for


class TestUprightRotationDegrees:
    """The angle is derived from the writing direction, not guessed."""

    @pytest.mark.parametrize(
        "direction,expected",
        [
            ((1.0, 0.0), 0),  # already horizontal
            ((0.0, -1.0), 90),  # reads upward — the reference document
            ((-1.0, 0.0), 180),  # upside down
            ((0.0, 1.0), 270),  # reads downward
            ((0.0, 0.0), 0),  # no directional evidence — must not rotate (#145)
            ((0.99, 0.05), 0),  # slight skew stays in the horizontal quadrant
        ],
    )
    def test_angle_is_derived_from_direction(
        self, direction: tuple[float, float], expected: int
    ) -> None:
        assert upright_rotation_degrees(direction) == expected

    def test_a_wrong_guess_would_be_upside_down(self) -> None:
        """270 is not an acceptable substitute for 90.

        Guessing the quadrant yields an upside-down page, which is no more
        readable than the sideways original — hence 'derived, not guessed'.
        """
        assert upright_rotation_degrees((0.0, -1.0)) != 270


def _sideways_pdf(tmp_path, text="Coefficient p-value 0.86"):
    """A one-page PDF whose only text is rotated 90 degrees."""
    doc = fitz.open()
    page = doc.new_page(width=432, height=648)
    page.insert_text((300, 400), text, fontsize=11, rotate=90)
    path = tmp_path / "sideways.pdf"
    doc.save(path)
    doc.close()
    return path


def _horizontal_pdf(tmp_path, text="Coefficient p-value 0.86"):
    doc = fitz.open()
    page = doc.new_page(width=432, height=648)
    page.insert_text((100, 400), text, fontsize=11)
    path = tmp_path / "horizontal.pdf"
    doc.save(path)
    doc.close()
    return path


class TestRenderOrientation:
    def test_sideways_page_is_detected_as_needing_rotation(self, tmp_path) -> None:
        with fitz.open(_sideways_pdf(tmp_path)) as doc:
            page = doc[0]
            direction = dominant_text_direction(page.get_text("dict")["blocks"])
            assert direction != (1.0, 0.0), "fixture is not actually rotated"
            assert _upright_rotation_for(page) != 0

    def test_horizontal_page_is_left_alone(self, tmp_path) -> None:
        with fitz.open(_horizontal_pdf(tmp_path)) as doc:
            assert _upright_rotation_for(doc[0]) == 0

    def test_rotation_changes_the_rendered_aspect(self, tmp_path) -> None:
        """The DIFFERENCE: a portrait page carrying sideways text renders landscape.

        Asserts the two renders differ in orientation rather than pinning pixel
        dimensions, which depend on DPI and page size.
        """
        with fitz.open(_sideways_pdf(tmp_path)) as doc:
            page = doc[0]
            plain = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            rotation = _upright_rotation_for(page)
            rotated = page.get_pixmap(matrix=fitz.Matrix(2, 2).prerotate(rotation))

        assert plain.width < plain.height, "fixture page is not portrait"
        assert rotated.width > rotated.height, (
            "de-rotated render should be landscape; got "
            f"{rotated.width}x{rotated.height} from rotation={rotation}"
        )

    def test_unreadable_page_fails_open_to_no_rotation(self) -> None:
        """A page that cannot be inspected must render exactly as before.

        The caller has no fallback if this raises mid-render, so the helper
        swallows and returns 0 — never worse than not having tried.
        """

        class _Exploding:
            def get_text(self, _kind):
                raise RuntimeError("cannot read page")

        assert _upright_rotation_for(_Exploding()) == 0

    def test_empty_page_is_not_rotated(self, tmp_path) -> None:
        doc = fitz.open()
        doc.new_page(width=432, height=648)
        path = tmp_path / "blank.pdf"
        doc.save(path)
        doc.close()
        with fitz.open(path) as d:
            assert _upright_rotation_for(d[0]) == 0


def test_prerotate_returns_the_matrix_and_mutates_it() -> None:
    """GH-311: the 304b ADR said ``prerotate`` returns None.

    It does not. On PyMuPDF 1.28.2 it mutates in place AND returns the matrix.

    GH-426: the wording here used to add that this is "why the call sites'
    ``mat = mat.prerotate(...)`` assignment is meaningful", and that both call
    sites rely on it. Neither is true. The tree carries BOTH forms -- the 304b
    crop lanes (``tables/extract.py``, ``source_evidence.py``, ``witness.py``)
    discard the return and rely on the mutation, while the page-level lanes
    (``engines/base.py``, ``core/document.py``, ``review/html.py``) assign it.
    They are equivalent precisely because both halves hold.

    Pinned because a PyMuPDF upgrade that changed EITHER half would silently
    break one family of call sites while leaving the other working, which is the
    hardest version of this bug to find.
    """
    import fitz

    mat = fitz.Matrix(1, 1)
    returned = mat.prerotate(90)

    assert returned is not None, "the ADR's 'returns None' claim would be true here"
    assert isinstance(returned, fitz.Matrix)
    assert mat != fitz.Matrix(1, 1), "prerotate must still mutate in place"
