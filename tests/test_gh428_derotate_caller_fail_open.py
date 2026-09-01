"""GH-428: the fail-open guard must hold AT THE CALLERS, not just in the helper.

#424 wrapped ``upright_rotation_for``'s direction inspection and pinned the
helper (malformed -> 0, well-formed -> 90). It did not pin the two callers the
ticket named, and those are where the failure actually costs something:

- ``_render_crop`` calls the helper BEFORE its own ``try`` around
  ``get_pixmap``, so a raise escapes ``extract()`` -- whose never-raises
  contract is what the dual-pass table path depends on.
- ``process_pages`` calls it inside the render loop with no boundary at all, so
  a raise abandons every page in the batch, not just the offending one.

Both are pinned as CALLER behaviour: the helper is made to fail the way a
malformed ``dir`` makes it fail, and the caller must still deliver. Reverting
the guard reddens both.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")


# A line whose ``dir`` is not a pair of numbers. ``dominant_text_direction``
# does ``float(d[0])`` on it, so this is what a malformed direction actually
# does to the real helper -- no patching of socr's own functions, which is what
# made the first version of the anchor below tautological (GH-428 cubic P2).
_MALFORMED_BLOCKS = {
    "blocks": [
        {
            "lines": [
                {"dir": "not-a-vector", "spans": [{"text": "Estimate 0.86"}]},
                {"dir": "not-a-vector", "spans": [{"text": "1.24 (0.02)"}]},
            ]
        }
    ]
}


def _patch_malformed_dirs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every ``get_text("dict")`` on any page return a malformed ``dir``.

    Patched at the PyMuPDF boundary, so socr's own direction helpers run for
    real on input that genuinely breaks them. Any other ``get_text`` mode is
    delegated untouched, so the callers still see real page content.
    """
    original = fitz.Page.get_text

    def _get_text(self, option="text", **kwargs):
        if option == "dict":
            return _MALFORMED_BLOCKS
        return original(self, option, **kwargs)

    monkeypatch.setattr(fitz.Page, "get_text", _get_text)


def _sideways_pdf(tmp_path: Path, name: str = "sideways.pdf") -> Path:
    """A page whose dominant text direction genuinely reads upward (90)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i in range(8):
        page.insert_text((200 + i * 18, 500), "Estimate 0.86 (0.02) 1.24", fontsize=11, rotate=90)
    path = tmp_path / name
    doc.save(path)
    doc.close()
    return path


def _page_pdf(tmp_path: Path, name: str = "page.pdf") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i in range(8):
        page.insert_text((72, 100 + i * 18), "Estimate 0.86 (0.02) 1.24", fontsize=11)
    path = tmp_path / name
    doc.save(path)
    doc.close()
    return path


def test_a_malformed_dir_really_does_break_the_real_helper() -> None:
    """Anchor 1: the fixture input is genuinely hostile, not a stubbed raise."""
    from socr.core.born_digital import dominant_text_direction

    with pytest.raises(Exception):
        dominant_text_direction(_MALFORMED_BLOCKS["blocks"])


def test_the_direction_helpers_are_actually_consulted(tmp_path: Path) -> None:
    """Anchor 2: the guard is not passing because the helpers went unused.

    A regression that stopped calling the direction helpers inside
    ``upright_rotation_for`` would leave every caller test below green while
    pinning nothing. This fails in that case: a genuinely sideways page must
    still come back as 90.
    """
    from socr.core.born_digital import upright_rotation_for

    doc = fitz.open(_sideways_pdf(tmp_path / "anchor"))
    try:
        assert upright_rotation_for(doc[0]) == 90
    finally:
        doc.close()


def test_the_guard_is_what_swallows_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anchor 3: with malformed dirs the SAME page falls back to 0, not 90."""
    from socr.core.born_digital import upright_rotation_for

    _patch_malformed_dirs(monkeypatch)
    doc = fitz.open(_sideways_pdf(tmp_path / "anchor2"))
    try:
        assert upright_rotation_for(doc[0]) == 0
    finally:
        doc.close()


class TestCropExtract:
    def test_extract_does_not_raise_on_a_malformed_direction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``TableCropExtractor.extract`` is documented "Never raises".

        The derotation call sits before ``_render_crop``'s own ``try``, so a
        raise there escapes that boundary and breaks the contract the dual-pass
        table path is built on. Driven through the public method, not the
        private renderer, because the contract is on the public one.
        """
        from socr.tables.extract import TableCropExtractor
        from socr.tables.locate import TableBox

        class _StubReader:
            timeout = 5.0

            def read(self, *_args, **_kwargs):
                return "| a | b |\n| --- | --- |\n| 1 | 2 |"

        _patch_malformed_dirs(monkeypatch)

        pdf = _page_pdf(tmp_path / "crop")
        box = TableBox(bbox=(60.0, 90.0, 400.0, 260.0), source="ruled")
        out = TableCropExtractor(_StubReader()).extract(pdf, 1, [box], cascade_probe=False)

        assert out, "extract returned nothing, so the crop never rendered"


class TestProcessPages:
    def test_the_ocr_render_loop_still_emits_a_page(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``process_pages`` must still return a PageOutput for the page.

        The CLI is stubbed to a no-op, so what is measured is whether the render
        loop completed -- a raise there abandons the whole batch before the CLI
        is ever built.
        """
        from socr.core.config import PipelineConfig
        from socr.engines.base import BaseEngine

        class _StubEngine(BaseEngine):
            @property
            def name(self) -> str:
                return "deepseek"

            @property
            def cli_command(self) -> str:
                return "true"

            def _build_command(self, pdf_path, output_dir, config):
                return ["true"]

        _patch_malformed_dirs(monkeypatch)

        pdf = _page_pdf(tmp_path / "ocr")
        pages = _StubEngine().process_pages(pdf, [1], PipelineConfig(quiet=True), dpi=72)

        assert len(pages) == 1, f"the render loop dropped the batch: {pages}"
        assert pages[0].page_num == 1
