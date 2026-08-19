"""Every reader must open a PDF the same way, or the lanes disagree (#244).

#217 repairs the missing ToUnicode map on the ``fitz.Document`` object it is
handed. socr opens the same file in ~30 places, so wiring the repair into only
some of them leaves the native lane reading ``−0.12`` while table extraction,
opening its own handle, still reads ``20.12`` from the same page.

Hermetic: no corpus (copyrighted, public repo), no network, no provider.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import socr.core.pdf as pdf_module
from socr.core.glyph_recovery import GlyphRepairReport
from socr.core.pdf import apply_glyph_recovery, open_pdf, reset_repair_cache

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "socr"

# Modules that legitimately still call fitz.open directly, with the reason.
# ``fitz.open()`` with no argument creates a NEW EMPTY document and is never a
# read of an existing file, so it is excluded structurally rather than by name.
RAW_OPEN_ALLOWED = {
    # Rasterises pages to images. Never reads the text layer, so the font scan
    # would cost something and change nothing.
    "benchmark/rasterize.py",
    # Splits a PDF into page ranges by copying pages. Byte-level, no text read.
    "core/chunker.py",
    # The helper itself.
    "core/pdf.py",
}


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_repair_cache()
    yield
    reset_repair_cache()


class TestSingleEntryPoint:
    """The boundary, enforced structurally rather than by convention."""

    def test_no_module_opens_a_pdf_file_outside_the_helper(self):
        offenders = []
        for path in SRC.rglob("*.py"):
            rel = path.relative_to(SRC).as_posix()
            if rel in RAW_OPEN_ALLOWED:
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (
                    isinstance(func, ast.Attribute)
                    and func.attr == "open"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "fitz"
                ):
                    continue
                if not node.args:
                    continue  # fitz.open() -> a new empty document, not a read
                offenders.append(f"{rel}:{node.lineno}")
        assert not offenders, (
            "these open a PDF file outside socr.core.pdf.open_pdf, so they read "
            "unrepaired text and will disagree with the native lane:\n  " + "\n  ".join(offenders)
        )

    def test_the_allowlist_names_files_that_exist(self):
        """A stale allowlist entry would silently widen the boundary."""
        missing = [rel for rel in RAW_OPEN_ALLOWED if not (SRC / rel).exists()]
        assert not missing


class TestRepairCache:
    def test_clean_file_is_scanned_once(self, tmp_path, monkeypatch):
        calls = []

        def fake_repair(doc):
            calls.append(doc)
            return GlyphRepairReport()

        monkeypatch.setattr(pdf_module, "repair_symbol_font_text", fake_repair)
        target = tmp_path / "clean.pdf"
        target.write_bytes(b"%PDF-1.4\n")

        apply_glyph_recovery(object(), target)
        apply_glyph_recovery(object(), target)
        apply_glyph_recovery(object(), target)

        assert len(calls) == 1, "a clean file was rescanned on every open"

    def test_affected_file_is_repaired_every_time(self, tmp_path, monkeypatch):
        """The repair applies to a fresh Document, so it cannot be cached away."""
        calls = []

        def fake_repair(doc):
            calls.append(doc)
            return GlyphRepairReport(repaired_fonts=["F"], mapped_glyph_count=3)

        monkeypatch.setattr(pdf_module, "repair_symbol_font_text", fake_repair)
        target = tmp_path / "affected.pdf"
        target.write_bytes(b"%PDF-1.4\n")

        apply_glyph_recovery(object(), target)
        apply_glyph_recovery(object(), target)

        assert len(calls) == 2

    def test_a_file_needing_attention_is_not_cached_as_clean(self, tmp_path, monkeypatch):
        """Nothing repaired but glyphs unmapped is the worst case, not a clean one."""
        calls = []

        def fake_repair(doc):
            calls.append(doc)
            return GlyphRepairReport(unmapped_glyphs=["H99999"])

        monkeypatch.setattr(pdf_module, "repair_symbol_font_text", fake_repair)
        target = tmp_path / "unmapped.pdf"
        target.write_bytes(b"%PDF-1.4\n")

        apply_glyph_recovery(object(), target)
        apply_glyph_recovery(object(), target)

        assert len(calls) == 2

    def test_a_modified_file_is_rescanned(self, tmp_path, monkeypatch):
        """Cache identity includes size, so an edited file is not stale-clean."""
        calls = []
        monkeypatch.setattr(
            pdf_module,
            "repair_symbol_font_text",
            lambda doc: (calls.append(doc), GlyphRepairReport())[1],
        )
        target = tmp_path / "edited.pdf"
        target.write_bytes(b"%PDF-1.4\n")
        apply_glyph_recovery(object(), target)

        target.write_bytes(b"%PDF-1.4\n% now longer\n")
        apply_glyph_recovery(object(), target)

        assert len(calls) == 2

    def test_a_failing_repair_does_not_raise(self, tmp_path, monkeypatch):
        """A document that cannot be repaired must still be readable."""
        monkeypatch.setattr(
            pdf_module,
            "repair_symbol_font_text",
            lambda doc: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        target = tmp_path / "bad.pdf"
        target.write_bytes(b"%PDF-1.4\n")

        report = apply_glyph_recovery(object(), target)
        assert not report.repaired

    def test_a_failing_repair_is_not_cached_as_clean(self, tmp_path, monkeypatch):
        """A failure is not evidence the file is fine."""
        calls = []

        def boom(doc):
            calls.append(doc)
            raise RuntimeError("boom")

        monkeypatch.setattr(pdf_module, "repair_symbol_font_text", boom)
        target = tmp_path / "bad.pdf"
        target.write_bytes(b"%PDF-1.4\n")

        apply_glyph_recovery(object(), target)
        apply_glyph_recovery(object(), target)

        assert len(calls) == 2

    def test_a_missing_file_is_not_cached(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(
            pdf_module,
            "repair_symbol_font_text",
            lambda doc: (calls.append(doc), GlyphRepairReport())[1],
        )
        target = tmp_path / "gone.pdf"

        apply_glyph_recovery(object(), target)
        apply_glyph_recovery(object(), target)

        assert len(calls) == 2


class TestOpenPdf:
    def test_repair_can_be_skipped(self, tmp_path, monkeypatch):
        """Image-only and page-split paths should not pay for a font scan."""
        calls = []
        monkeypatch.setattr(
            pdf_module,
            "repair_symbol_font_text",
            lambda doc: (calls.append(doc), GlyphRepairReport())[1],
        )
        monkeypatch.setattr(pdf_module.fitz, "open", lambda path: object())

        open_pdf(tmp_path / "x.pdf", repair=False)
        assert calls == []

    def test_repair_is_the_default(self, tmp_path, monkeypatch):
        """A caller who does not think about it must get the safe behaviour."""
        calls = []
        monkeypatch.setattr(
            pdf_module,
            "repair_symbol_font_text",
            lambda doc: (calls.append(doc), GlyphRepairReport())[1],
        )
        monkeypatch.setattr(pdf_module.fitz, "open", lambda path: object())

        open_pdf(tmp_path / "x.pdf")
        assert len(calls) == 1
