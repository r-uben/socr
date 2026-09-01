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


def _boom(*_args, **_kwargs):
    """Stand-in for a malformed ``dir`` tuple reaching the direction helpers."""
    raise ValueError("malformed dir vector")


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


def test_the_helper_is_the_thing_that_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anchor: without the guard the patched helper really does raise.

    Without this, a guard that silently stopped calling the direction helpers
    at all would satisfy both caller tests below while pinning nothing.
    """
    import socr.core.born_digital as bd

    monkeypatch.setattr(bd, "dominant_text_direction", _boom)
    with pytest.raises(ValueError):
        bd.dominant_text_direction([{"lines": []}])


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
        import socr.core.born_digital as bd
        from socr.tables.extract import TableCropExtractor
        from socr.tables.locate import TableBox

        class _StubReader:
            timeout = 5.0

            def read(self, *_args, **_kwargs):
                return "| a | b |\n| --- | --- |\n| 1 | 2 |"

        monkeypatch.setattr(bd, "dominant_text_direction", _boom)

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
        import socr.core.born_digital as bd
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

        monkeypatch.setattr(bd, "dominant_text_direction", _boom)

        pdf = _page_pdf(tmp_path / "ocr")
        pages = _StubEngine().process_pages(pdf, [1], PipelineConfig(quiet=True), dpi=72)

        assert len(pages) == 1, f"the render loop dropped the batch: {pages}"
        assert pages[0].page_num == 1
