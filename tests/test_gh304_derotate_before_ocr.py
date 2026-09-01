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
    sites rely on it. Neither is true. The tree carries BOTH forms:

    - mutate-only, discarding the return: ``tables/extract.py``,
      ``tables/source_evidence.py`` (x2), ``tables/witness.py``,
      ``pipeline/orchestrator.py`` (``_render_adjudication_crop``)
    - assignment: ``engines/base.py``, ``core/document.py`` (x2),
      ``review/html.py``, ``pipeline/orchestrator.py``
      (``_render_chart_page_png``, ``_render_chart_region_pngs``)

    GH-440: the mutate-only entry first named the D3 floor render, which does
    not call ``prerotate`` at all -- ``_render_d3_floor_png`` delegates to
    ``_render_chart_page_png``. Each site above is now the enclosing ``def`` of
    an actual ``prerotate`` occurrence.

    Not a clean split by lane, either -- ``orchestrator.py`` appears on both
    sides. They are equivalent precisely because both halves of the PyMuPDF
    behaviour hold.

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


# GH-440, and the #442 review. The ADR's call-site table, enumerated
# mechanically. Asserted as EXACT SETS: a new, renamed, moved or deleted
# `prerotate` call fails here, whichever file it is in. Anything less relocates
# the hand-drift this exists to stop instead of ending it.
_MUTATE_ONLY_SITES = {
    ("pipeline/orchestrator.py", "_render_adjudication_crop"),
    ("tables/extract.py", "_render_crop"),
    ("tables/source_evidence.py", "_render_crop_pixmap"),
    ("tables/source_evidence.py", "build_scanned_evidence"),
    ("tables/witness.py", "_render_crop_safe"),
}
_ASSIGNMENT_SITES = {
    ("core/document.py", "render_all_pages"),
    ("core/document.py", "render_page"),
    ("engines/base.py", "process_pages"),
    ("pipeline/orchestrator.py", "_render_chart_page_png"),
    ("pipeline/orchestrator.py", "_render_chart_region_pngs"),
    ("review/html.py", "_render_page_image"),
}


def _prerotate_sites() -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Every ``prerotate`` call in ``src/socr``, by (file, enclosing def).

    Split by whether the call is a bare statement (``ast.Expr`` -- mutate-only,
    relying on the in-place mutation) or not (its value is used, i.e. assigned).
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "socr"
    mutate: set[tuple[str, str]] = set()
    assign: set[tuple[str, str]] = set()

    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "prerotate"
            ):
                continue
            fn: ast.AST = node
            while fn in parents and not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = parents[fn]
            site = (str(path.relative_to(root)), getattr(fn, "name", "<module>"))
            (mutate if isinstance(parents.get(node), ast.Expr) else assign).add(site)

    return mutate, assign


def test_the_recorded_prerotate_call_sites_match_the_code() -> None:
    """GH-440: the ADR's call-site table must be checkable, not just careful.

    That table has now been wrong three times -- #423 claimed ``prerotate``
    returns None, #426 claimed production always assigns the result, and #437
    named the D3 floor render as a mutate-only caller when
    ``_render_d3_floor_png`` does not call ``prerotate`` at all. Every fix was
    hand-enumerated, which is exactly why the next one drifted.

    So enumerate it mechanically and compare EXACT sets. Asserting only the
    sites someone happened to think of would relocate the drift rather than end
    it (#442 review): a rename in ``tables/extract.py``, or ``document.py``
    switching from assignment to mutate-only, has to fail here too.

    When this fails, the code moved and the ADR table
    (``docs/log/2026-08-27_304b-crop-derotation.md``) needs the same edit.
    """
    mutate, assign = _prerotate_sites()

    assert mutate == _MUTATE_ONLY_SITES, (
        "mutate-only prerotate sites drifted from the ADR table.\n"
        f"  added:   {sorted(mutate - _MUTATE_ONLY_SITES)}\n"
        f"  missing: {sorted(_MUTATE_ONLY_SITES - mutate)}"
    )
    assert assign == _ASSIGNMENT_SITES, (
        "assignment prerotate sites drifted from the ADR table.\n"
        f"  added:   {sorted(assign - _ASSIGNMENT_SITES)}\n"
        f"  missing: {sorted(_ASSIGNMENT_SITES - assign)}"
    )
    assert not (mutate & assign), f"a site is recorded as both forms: {sorted(mutate & assign)}"
