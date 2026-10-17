"""#1189: the text-layer visibility checks must not call ``page.get_texttrace()``.

In PyMuPDF 1.27.2.2, 1.27.2.3 and 1.28.0 that call decrements ``None``'s reference count
once per span returned. The count falls monotonically to zero and the interpreter dies with
``Fatal Python error: none_dealloc`` -- an ABORT, so no ``except``, no ``finally`` and none
of socr's fail-closed handling runs. Measured on the staged Fed corpus: 45 of a seeded
200-document sample aborted inside ``BornDigitalDetector.detect()``. Upstream fixed it in
1.28.2 (PyMuPDF #5042), but the Alliance cluster builds its own wheels and ran the affected
``1.27.2.2+computecanada``, so socr reads the text layer's own ``alpha`` instead and does not
depend on which PyMuPDF is installed.

The pins here are DIFFERENCES (CLAUDE.md, #257): each page is read BOTH ways in the same
process and the two verdicts must agree. Deliberately NOT pinned: the two routes' raw
character counts. They agree exactly on these synthetic pages but differ by a few characters
on real scans (69 of 506 corpus pages, e.g. ``trace=(1958, 1958)`` vs ``alpha=(1959, 1959)``
-- glyph runs vs span text), while the verdict differed on none of the 506. Pinning the
counts would assert an invariant that only synthetic fixtures satisfy.

Hermetic: synthetic fitz pages only, no provider, no judge, no model.
"""

from __future__ import annotations

import ast
from pathlib import Path

import fitz
import pytest

from socr.core.born_digital import BornDigitalDetector
from socr.tables import native_contradiction as nc
from socr.tables.source_evidence import text_layer_invisible_char_counts

_PROSE = "The policy rate was raised to 5.25 percent in a unanimous vote. "

#: Invisible vs visible LINES per page. Includes both sides of the half-of-the-characters
#: boundary the #961 verdict turns on (5/3 is over it, 4/4 is exactly at it and so under),
#: because a route that agreed only on all-invisible and all-visible pages would prove
#: nothing about the comparison itself.
_SHAPES = {
    "all_invisible": (8, 0),
    "all_visible": (0, 8),
    "majority_invisible": (5, 3),
    "minority_invisible": (3, 5),
    "exactly_half_invisible": (4, 4),
    "no_text_at_all": (0, 0),
    "one_invisible_line_only": (1, 0),
}


def _pixmap(w: int = 200, h: int = 200) -> fitz.Pixmap:
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, w, h), False)
    pix.set_rect(pix.irect, (235, 235, 235))
    return pix


def _page(doc: fitz.Document, invisible_lines: int, visible_lines: int, *, raster: bool):
    page = doc.new_page()
    if raster:
        page.insert_image(page.rect, pixmap=_pixmap())
    y = 72
    for render_mode in [3] * invisible_lines + [0] * visible_lines:
        page.insert_text((72, y), _PROSE, fontname="helv", fontsize=10, render_mode=render_mode)
        y += 14
    # Reload so the inserted content is read back from the saved page stream rather than
    # from the in-memory builder, which is what a real document gives the detector.
    return doc.reload_page(page)


def _texttrace_counts(page) -> tuple[int, int]:
    """The pre-#1189 reading, kept HERE so no production path can reach it.

    Safe in a test: the leak is proportional to spans read, and these pages carry a few
    hundred, against a budget of roughly twelve thousand per process.
    """
    invisible = total = 0
    for span in page.get_texttrace():
        count = len(span.get("chars") or ())
        total += count
        if span.get("type") == 3:
            invisible += count
    return invisible, total


# -- the two production call sites, each against its own pre-#1189 formula ---------------


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_invisible_over_raster_verdict_is_unchanged(shape):
    """#961's verdict via ``alpha`` equals its verdict via ``get_texttrace``."""
    invisible_lines, visible_lines = _SHAPES[shape]
    with fitz.open() as doc:
        page = _page(doc, invisible_lines, visible_lines, raster=True)

        trace_invisible, trace_total = _texttrace_counts(page)
        expected = trace_invisible * 2 > trace_total
        actual = BornDigitalDetector()._has_invisible_text_over_raster(page)

    assert actual is expected, (
        f"{shape}: alpha route said {actual}, get_texttrace route said {expected}"
    )


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_text_layer_is_visible_verdict_is_unchanged(shape):
    """``native_contradiction``'s verdict is the complement, over the same two counts."""
    invisible_lines, visible_lines = _SHAPES[shape]
    with fitz.open() as doc:
        page = _page(doc, invisible_lines, visible_lines, raster=False)

        trace_invisible, trace_total = _texttrace_counts(page)
        expected = trace_total > 0 and trace_invisible * 2 <= trace_total
        actual = nc.text_layer_is_visible(page)

    assert actual is expected, (
        f"{shape}: alpha route said {actual}, get_texttrace route said {expected}"
    )


def test_the_shape_matrix_exercises_both_verdicts():
    """Guard against a vacuous suite: the matrix must produce both answers, both sides.

    If every shape landed on one verdict, the two tests above would pass against a
    constant and prove nothing.
    """
    invisible_over_raster = set()
    layer_visible = set()
    with fitz.open() as doc:
        for invisible_lines, visible_lines in _SHAPES.values():
            page = _page(doc, invisible_lines, visible_lines, raster=True)
            invisible_over_raster.add(BornDigitalDetector()._has_invisible_text_over_raster(page))
            layer_visible.add(nc.text_layer_is_visible(page))

    assert invisible_over_raster == {True, False}
    assert layer_visible == {True, False}


# -- the two known, intended divergences ------------------------------------------------
#
# The routes are not equivalent by construction. Both of these shapes diverge, both times
# with ``alpha`` giving the more faithful answer, and both times toward distrusting the text
# layer -- the safe direction when a wrong number is worse than a missing one. Neither shape
# occurred on any of the 506 corpus pages measured, so the change is inert on real
# documents; these pin the difference as intended instead of leaving it to be rediscovered.


def test_clip_only_text_is_invisible_and_the_old_route_missed_it():
    """Render mode 7 draws no ink. ``get_texttrace`` returns NO spans for it.

    The old route therefore scored such a page 0/0 and called it clean -- an invisible
    OCR layer that #961 exists to catch, waved through. ``alpha`` reports it.
    """
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_image(page.rect, pixmap=_pixmap())
        y = 72
        for _ in range(8):
            page.insert_text((72, y), _PROSE, fontname="helv", fontsize=10, render_mode=7)
            y += 14
        page = doc.reload_page(page)

        trace_invisible, trace_total = _texttrace_counts(page)
        assert (trace_invisible, trace_total) == (0, 0), (
            "premise of this test: get_texttrace reports nothing for clip-only text"
        )
        old_verdict = trace_invisible * 2 > trace_total
        assert old_verdict is False, "the old route called this invisible layer clean"

        assert BornDigitalDetector()._has_invisible_text_over_raster(page) is True


def test_transparent_fill_is_invisible_ink_in_a_visible_mode():
    """Render mode 0 with ``fill_opacity=0``: the old route saw type 0 and trusted it."""
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_image(page.rect, pixmap=_pixmap())
        y = 72
        for _ in range(8):
            page.insert_text(
                (72, y), _PROSE, fontname="helv", fontsize=10, render_mode=0, fill_opacity=0
            )
            y += 14
        page = doc.reload_page(page)

        trace_invisible, trace_total = _texttrace_counts(page)
        assert trace_total > 0, "premise: get_texttrace does see these spans"
        assert trace_invisible == 0, "premise: it classifies them as visible (type 0)"
        assert (trace_invisible * 2 > trace_total) is False

        assert BornDigitalDetector()._has_invisible_text_over_raster(page) is True


def test_neither_divergence_makes_an_unreadable_layer_look_readable():
    """Both shapes must also abstain in ``native_contradiction``, not invite a comparison.

    The complement verdict is where a wrong answer would do damage: treating an OCR
    layer as readable evidence is exactly what #961 forbids.
    """
    with fitz.open() as doc:
        for render_mode, opacity in ((7, 1), (0, 0)):
            page = doc.new_page()
            y = 72
            for _ in range(8):
                page.insert_text(
                    (72, y),
                    _PROSE,
                    fontname="helv",
                    fontsize=10,
                    render_mode=render_mode,
                    fill_opacity=opacity,
                )
                y += 14
            page = doc.reload_page(page)
            assert nc.text_layer_is_visible(page) is False, (
                f"render_mode={render_mode} fill_opacity={opacity} was read as visible"
            )


# -- the raster pre-gate still gates ----------------------------------------------------


def test_invisible_text_without_a_raster_is_not_a_scan():
    """#961 is invisible text OVER a raster; the pre-gate is not weakened by #1189."""
    with fitz.open() as doc:
        page = _page(doc, 8, 0, raster=False)
        assert BornDigitalDetector()._has_invisible_text_over_raster(page) is False

        over_raster = _page(doc, 8, 0, raster=True)
        assert BornDigitalDetector()._has_invisible_text_over_raster(over_raster) is True


# -- fail-closed is preserved -----------------------------------------------------------


class _RaisingTextPage:
    """A page whose text layer cannot be read."""

    def __init__(self, page):
        self._page = page

    def __getattr__(self, name):
        return getattr(self._page, name)

    def get_text(self, *args, **kwargs):
        raise RuntimeError("mupdf: cannot read text layer")


def test_an_unreadable_text_layer_raises_rather_than_reporting_clean():
    """An unreadable page is "unknown", never "clean" (#961's caller fails closed).

    This is the regression the obvious implementation would have introduced: the existing
    span iterator in ``source_evidence`` swallows exceptions, and reusing it would have
    turned an unreadable page into zero spans, i.e. a confident "no invisible text".
    """
    with fitz.open() as doc:
        page = _RaisingTextPage(_page(doc, 8, 0, raster=True))

        with pytest.raises(RuntimeError):
            text_layer_invisible_char_counts(page)
        with pytest.raises(RuntimeError):
            BornDigitalDetector()._has_invisible_text_over_raster(page)


class _RaisingDictPage:
    """A page whose ``get_text("dict")`` fails while its other reads work.

    Narrower than :class:`_RaisingTextPage` on purpose. ``_assess_page_signals`` calls
    ``get_text("text")`` long before the #961 branch, so a page that failed EVERY read
    never reaches the code under test -- it dies at the first line of the assessment and
    the test would be measuring the wrong failure.
    """

    def __init__(self, page):
        self._page = page

    def __getattr__(self, name):
        return getattr(self._page, name)

    def get_text(self, *args, **kwargs):
        if args and args[0] == "dict":
            raise RuntimeError("mupdf: cannot read text layer")
        return self._page.get_text(*args, **kwargs)


class _NoneDictPage:
    """A page whose ``get_text("dict")`` returns ``None`` instead of raising."""

    def __init__(self, page):
        self._page = page

    def __getattr__(self, name):
        return getattr(self._page, name)

    def get_text(self, *args, **kwargs):
        if args and args[0] == "dict":
            return None
        return self._page.get_text(*args, **kwargs)


def test_a_none_return_is_a_failure_not_an_empty_page():
    """``(page.get_text("dict") or {})`` would have called this page clean.

    A failure that arrives as a return value rather than an exception is the one shape
    a ``try/except`` cannot see. Zero spans from an unreadable page and zero spans from
    a page with no text are the same number and opposite facts, so the unreadable case
    has to be rejected before the counting starts. Argon flagged this as fail-open.
    """
    with fitz.open() as doc:
        page = _NoneDictPage(_page(doc, 8, 0, raster=True))

        with pytest.raises(TypeError, match="not a dict"):
            text_layer_invisible_char_counts(page)
        with pytest.raises(TypeError, match="not a dict"):
            BornDigitalDetector()._has_invisible_text_over_raster(page)


def test_a_dict_with_no_blocks_is_an_empty_page_not_a_failure():
    """The complement: a real page with no text layer must count (0, 0), not raise."""
    with fitz.open() as doc:
        page = _page(doc, 0, 0, raster=True)
        assert text_layer_invisible_char_counts(page) == (0, 0)
        assert BornDigitalDetector()._has_invisible_text_over_raster(page) is False


def test_assess_page_turns_an_unreadable_text_layer_into_needs_ocr():
    """The real consumer, driven for real: ``_assess_page`` must route to OCR.

    A raise that no caller acts on would be inert. This drives ``_assess_page`` itself
    and asserts the flag the pipeline branches on. An earlier version of this test
    asserted the raise twice and never reached ``_assess_page``; Argon caught that it
    proved nothing.
    """
    detector = BornDigitalDetector()
    with fitz.open() as doc:
        # Full-page raster so the #961 raster pre-gate is passed and the dict read is
        # actually reached, but VISIBLE text so the page is otherwise clean. If this
        # page already needed OCR the comparison below would be vacuous, so the premise
        # is asserted rather than assumed.
        page = _page(doc, 0, 8, raster=True)

        clean = detector._assess_page(page, 0)
        assert clean.needs_ocr_enhancement is False, (
            "premise: this page does not otherwise need OCR, so the flag below is "
            f"attributable to the failure (notes: {getattr(clean, 'notes', None)})"
        )

        failed = detector._assess_page(_RaisingDictPage(page), 0)

    assert failed.needs_ocr_enhancement is True, (
        "an unreadable text layer was trusted as native instead of routed to OCR"
    )


# -- the call must not come back --------------------------------------------------------


_SRC = Path(__file__).resolve().parents[1] / "src" / "socr"
_NAME = "get_texttrace"


def _texttrace_references(text: str) -> list[int]:
    """Line numbers of every reference to ``get_texttrace`` in parsed code.

    Parsed, not grepped. A regex over source lines has holes in both directions, and
    Argon found both: it misses ``getattr(page, "get_texttrace")()`` -- and
    ``getattr(page, "...")`` is a live pattern in ``born_digital.py``, not a
    hypothetical -- it misses a bound method saved and called later, and it misses a
    call whose parenthesis is on the next line. It also FIRES on a docstring that merely
    names ``page.get_texttrace()``, which this very file's module docstring would do if
    it lived in ``src``. The AST sees code and not prose, so both directions close at
    once.
    """
    hits: list[int] = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Attribute) and node.attr == _NAME:
            hits.append(node.lineno)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == _NAME
        ):
            hits.append(node.lineno)
    return hits


def test_no_production_code_references_get_texttrace():
    """The abort returns the moment any production path reaches it again.

    Scanned over the real source tree, so a new reference ANYWHERE in ``src/socr``
    fails this test, not just the two sites #1189 fixed.
    """
    assert _SRC.is_dir(), f"source tree not found at {_SRC}"
    scanned = 0
    offenders = []
    for path in sorted(_SRC.rglob("*.py")):
        scanned += 1
        for line in _texttrace_references(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(_SRC.parent.parent)}:{line}")

    # Not vacuous: assert the scan actually read the tree before trusting its silence.
    assert scanned > 100, f"only {scanned} source files scanned; the glob is wrong"
    assert not offenders, "production code references get_texttrace:\n" + "\n".join(offenders)


def test_the_guard_catches_every_way_of_reaching_it():
    """A guard never seen to fail has not been shown to guard anything.

    Each of these is a way the regex this replaced could have been walked past.
    """
    assert _texttrace_references("for span in page.get_texttrace():\n    pass\n")
    assert _texttrace_references('x = getattr(page, "get_texttrace")()\n')
    assert _texttrace_references("fn = page.get_texttrace\n")
    assert _texttrace_references("x = page.get_texttrace(\n)\n")


def test_the_guard_does_not_fire_on_prose():
    """It must survive the explanation of itself living next to the code it guards."""
    assert not _texttrace_references('"""``page.get_texttrace()`` is avoided; see #1189."""\n')
    assert not _texttrace_references("# page.get_texttrace() aborts the interpreter\n")


def test_outlined_text_diverges_the_other_way_but_fails_safe():
    """Render mode 2, transparent fill, opaque stroke: visible outlines, ``alpha`` 0.

    The one divergence where ``alpha`` is WRONG about the ink. Pinned deliberately: the
    page is routed to OCR and withheld from contradiction evidence, so the cost is a
    re-read, not a lost number. If someone later "fixes" this to trust such a page, this
    test should fail and the trade-off be re-argued rather than quietly inverted.
    """
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_image(page.rect, pixmap=_pixmap())
        y = 72
        for _ in range(8):
            page.insert_text(
                (72, y),
                _PROSE,
                fontname="helv",
                fontsize=10,
                render_mode=2,
                fill_opacity=0,
                stroke_opacity=1,
            )
            y += 14
        page = doc.reload_page(page)

        trace_invisible, trace_total = _texttrace_counts(page)
        assert trace_total > 0 and trace_invisible == 0, (
            "premise: get_texttrace calls these spans visible (no type 3)"
        )

        # Accepted, not equivalent: routed to OCR, and refused as evidence.
        assert BornDigitalDetector()._has_invisible_text_over_raster(page) is True
        assert nc.text_layer_is_visible(page) is False
