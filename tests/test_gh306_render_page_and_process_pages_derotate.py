"""GH-306: pin de-rotation at the two render sites `process()` actually uses.

#305 shipped the production half of GH-304 and left both call sites unguarded.
#440 later added a set-membership test that fails if a `prerotate` CALL
disappears -- but that is existence, not behaviour. Measured on main before this
file: changing either site to `prerotate(0)` leaves the call in place, disables
de-rotation completely, and the whole suite stays green.

So these are behaviour pins, driven through the real render paths:

- `DocumentHandle.render_page` feeds the VLM judge. A sideways image there is
  compared against a correctly transcribed table, so the judge cannot agree --
  it rejects or times out.
- `BaseEngine.process_pages` feeds the OCR engine itself. GH-304 measured qwen
  returning the reference table TRANSPOSED from a sideways render.

Each is a DIFFERENCE against the same page printed horizontally, so a renderer
that rotated everything would fail the control rather than pass the pin.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")


def _pdf(tmp_path: Path, rotate: int, name: str = "page.pdf") -> Path:
    """A portrait page whose text reads sideways (90) or horizontally (0)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i in range(8):
        if rotate:
            page.insert_text(
                (200 + i * 18, 500), "Coefficient 0.86 estimate", fontsize=11, rotate=90
            )
        else:
            page.insert_text((72, 100 + i * 18), "Coefficient 0.86 estimate", fontsize=11)
    path = tmp_path / name
    doc.save(path)
    doc.close()
    return path


def _measured_rotation(pdf: Path) -> int:
    """The rotation production will derive, asserted as != 0 rather than == 90.

    GH-306 review: PyMuPDF's line direction for a ``rotate=90`` insert can come
    back as ``(0, -1)`` (-> 90) or ``(0, 1)`` (-> 270) depending on version, and
    BOTH turn a portrait page landscape -- which is the behaviour the aspect
    pins below actually measure. Pinning ``== 90`` is stricter than the thing
    being pinned and would redden on a PyMuPDF where the fixture reports 270,
    with de-rotation working correctly. Matches the GH-304 precedent, and the
    house rule: pin a difference, not a locally measured value.
    """
    from socr.core.born_digital import upright_rotation_for

    doc = fitz.open(pdf)
    try:
        return upright_rotation_for(doc[0])
    finally:
        doc.close()


def _handle(pdf: Path):
    from socr.core.document import DocumentHandle

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(DocumentHandle, "__post_init__", lambda self: None)
        return DocumentHandle(path=pdf, page_count=1)


class TestRenderPage:
    """The image the VLM judge reads."""

    def test_a_sideways_page_comes_back_upright(self, tmp_path: Path) -> None:
        pdf = _pdf(tmp_path / "s", rotate=90)
        assert _measured_rotation(pdf) != 0, "fixture must actually be sideways"

        img = _handle(pdf).render_page(1, dpi=72)
        assert img.width > img.height, (
            f"render_page returned {img.width}x{img.height}; a derotated sideways "
            "portrait page must come back landscape, or the judge is handed a "
            "picture it cannot agree with"
        )

    def test_a_horizontal_page_is_left_alone(self, tmp_path: Path) -> None:
        """Control: a renderer that rotated everything would pass the test above."""
        pdf = _pdf(tmp_path / "h", rotate=0)
        assert _measured_rotation(pdf) == 0

        img = _handle(pdf).render_page(1, dpi=72)
        assert img.height > img.width, "an upright portrait page must stay portrait"


class _StubEngine:
    """Minimal CLI engine: the CLI is a no-op, so only the render loop runs."""

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def cli_command(self) -> str:
        return "true"

    def _build_command(self, pdf_path, output_dir, config):
        return ["true"]


def _rendered_size(pdf: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[int, int]:
    """Drive `process_pages` and capture the image it hands the engine."""
    from PIL import Image

    from socr.core.config import PipelineConfig
    from socr.engines.base import BaseEngine

    engine = type("_E", (_StubEngine, BaseEngine), {})()
    saved: list[tuple[int, int]] = []
    original = Image.Image.save

    def _save(self, fp, *args, **kwargs):
        saved.append((self.width, self.height))
        return original(self, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", _save)
    engine.process_pages(pdf, [1], PipelineConfig(quiet=True), dpi=72)
    assert saved, "process_pages never rendered a page image"
    return saved[0]


class TestProcessPages:
    """The image the OCR engine itself reads."""

    def test_a_sideways_page_is_rendered_upright(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pdf = _pdf(tmp_path / "ps", rotate=90)
        assert _measured_rotation(pdf) != 0, "fixture must actually be sideways"

        width, height = _rendered_size(pdf, monkeypatch)
        assert width > height, (
            f"process_pages handed the engine {width}x{height}; GH-304 measured qwen "
            "returning the reference table TRANSPOSED from exactly this input"
        )

    def test_a_horizontal_page_is_left_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control, same reason as above."""
        pdf = _pdf(tmp_path / "ph", rotate=0)
        assert _measured_rotation(pdf) == 0

        width, height = _rendered_size(pdf, monkeypatch)
        assert height > width, "an upright portrait page must stay portrait"
