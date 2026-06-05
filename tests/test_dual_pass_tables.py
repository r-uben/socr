"""Phase 4c — dual-pass table extraction (locate, crop-read, reconcile, patch)."""

from __future__ import annotations

from pathlib import Path

import fitz

from socr.core.born_digital import DocumentAssessment, PageAssessment
from socr.core.config import PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.extract import TableCropExtractor, _clean_markdown
from socr.tables.locate import locate_tables
from socr.tables.reconcile import (
    diff_grids,
    find_table_blocks,
    reconcile_page_tables,
)

# --------------------------------------------------------------------------
# Synthetic-page builders (mirror the styles validated during design)
# --------------------------------------------------------------------------

_DATA = [
    ["", "(1)", "(2)", "(3)"],
    ["log wage", "0.043", "0.051", "0.039"],
    ["", "(0.014)", "(0.016)", "(0.013)"],
    ["educ", "0.082", "0.077", "0.080"],
    ["N", "1,204", "1,204", "1,204"],
    ["R2", "0.31", "0.34", "0.36"],
]


def _draw_table(page, cols, rows):
    for r, row in enumerate(_DATA):
        for c, cell in enumerate(row):
            page.insert_text((cols[c] + 4, rows[r] + 12), cell, fontsize=9)


def _build_page(style: str):
    doc = fitz.open()
    page = doc.new_page()
    cols = [100, 220, 300, 380]
    rows = [100 + i * 22 for i in range(len(_DATA))]
    _draw_table(page, cols, rows)
    if style in ("ruled", "booktabs"):
        rule_ys = rows if style == "ruled" else [rows[0] - 4, rows[1] - 2, rows[-1] + 12]
        for yy in rule_ys:
            page.draw_line((100, yy), (460, yy))
    if style == "ruled":
        for xx in cols + [460]:
            page.draw_line((xx, rows[0]), (xx, rows[-1]))
    return doc, page


# --------------------------------------------------------------------------
# Localization
# --------------------------------------------------------------------------


def test_locate_ruled_table():
    _doc, page = _build_page("ruled")
    boxes = locate_tables(page)
    assert len(boxes) == 1
    assert boxes[0].source == "ruled"


def test_locate_booktabs_table():
    _doc, page = _build_page("booktabs")
    boxes = locate_tables(page)
    assert len(boxes) == 1
    assert boxes[0].source == "booktabs"
    # Band tightly bounds the table region, not the whole page.
    _x0, y0, _x1, y1 = boxes[0].bbox
    assert y1 - y0 < 200  # the table is ~150pt tall; not the full 792pt page


def test_locate_borderless_returns_nothing():
    # No rules, no cell borders -> no precise bbox -> out of v1 scope (the
    # whole-page judge still covers these).
    _doc, page = _build_page("borderless")
    assert locate_tables(page) == []


def test_locate_ignores_prose_on_mixed_page():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 80), "4. Results", fontsize=10)
    page.insert_text((72, 100), "We estimate returns to schooling.", fontsize=10)
    top = 140
    cols = [120, 300, 380, 460]
    rows = [top + i * 20 for i in range(len(_DATA))]
    _draw_table(page, cols, rows)
    for yy in [rows[0] - 4, rows[1] - 2, rows[-1] + 12]:
        page.draw_line((110, yy), (490, yy))
    page.insert_text((72, rows[-1] + 40), "Estimates are stable.", fontsize=10)

    boxes = locate_tables(page)
    assert len(boxes) == 1
    _x0, y0, _x1, y1 = boxes[0].bbox
    # The band starts at the first rule, not at the "4. Results" title (y~80).
    assert y0 > 120


# --------------------------------------------------------------------------
# Markdown parsing / diff
# --------------------------------------------------------------------------

_PAGE_MD = """## Results

Prose before.

| var | (1) | (2) |
| --- | --- | --- |
| educ | 0.082 | 0.077 |
| se | (0.009) | (0.0l0) |

Prose after.
"""

_CROP_MD = (
    "| var | (1) | (2) |\n| --- | --- | --- |\n"
    "| educ | 0.082 | 0.077 |\n| se | (0.009) | (0.010) |"
)


def test_find_table_blocks_locates_single_block():
    blocks = find_table_blocks(_PAGE_MD)
    assert len(blocks) == 1
    assert blocks[0].grid[0] == ["var", "(1)", "(2)"]  # separator row dropped


def test_diff_grids_names_the_changed_cell():
    a = [["se", "(0.0l0)"]]
    b = [["se", "(0.010)"]]
    diffs = diff_grids(a, b)
    assert len(diffs) == 1
    assert diffs[0].page_value == "(0.0l0)" and diffs[0].crop_value == "(0.010)"


def test_diff_grids_ignores_formatting_only_differences():
    a = [["x", "−0.5"]]   # unicode minus, extra spacing
    b = [["x", "  -0.5 "]]
    assert diff_grids(a, b) == []


# --------------------------------------------------------------------------
# Reconcile
# --------------------------------------------------------------------------


def test_reconcile_patches_misread_and_preserves_prose():
    r = reconcile_page_tables(_PAGE_MD, [(_CROP_MD, "booktabs")])
    assert r.patched and r.flagged
    assert "(0.010)" in r.text and "(0.0l0)" not in r.text
    assert "Prose before." in r.text and "Prose after." in r.text
    assert r.disagreements[0].action == "patched"


def test_reconcile_agreement_is_noop():
    clean = _PAGE_MD.replace("(0.0l0)", "(0.010)")
    r = reconcile_page_tables(clean, [(_CROP_MD, "booktabs")])
    assert not r.patched and not r.flagged


def test_reconcile_count_mismatch_flags_without_editing():
    r = reconcile_page_tables(_PAGE_MD, [(_CROP_MD, "booktabs"), (_CROP_MD, "ruled")])
    assert not r.patched
    assert all(d.action == "flagged" for d in r.disagreements)
    assert r.text == _PAGE_MD  # untouched


def test_reconcile_malformed_crop_flags_without_editing():
    r = reconcile_page_tables(_PAGE_MD, [("| only one row |", "booktabs")])
    assert not r.patched
    assert r.disagreements[0].action == "flagged"
    assert r.text == _PAGE_MD


def test_reconcile_no_crops_is_noop():
    r = reconcile_page_tables(_PAGE_MD, [])
    assert not r.patched and not r.flagged and r.text == _PAGE_MD


# --------------------------------------------------------------------------
# Crop extractor (injected reader, fail-open)
# --------------------------------------------------------------------------


class _StubReader:
    def __init__(self, value, raises=False):
        self.value = value
        self.raises = raises
        self.calls = 0

    def read(self, _image_path: Path) -> str:
        self.calls += 1
        if self.raises:
            raise RuntimeError("model down")
        return self.value


def test_extractor_reads_located_crops(tmp_path):
    doc, _ = _build_page("booktabs")
    pdf = tmp_path / "t.pdf"
    doc.save(pdf)
    boxes = locate_tables(fitz.open(pdf)[0])
    reader = _StubReader(_CROP_MD)
    crops = TableCropExtractor(reader).extract(pdf, 1, boxes)
    assert reader.calls == 1
    assert len(crops) == 1 and crops[0].markdown == _CROP_MD


def test_extractor_fail_open_on_reader_error(tmp_path):
    doc, _ = _build_page("booktabs")
    pdf = tmp_path / "t.pdf"
    doc.save(pdf)
    boxes = locate_tables(fitz.open(pdf)[0])
    crops = TableCropExtractor(_StubReader("", raises=True)).extract(pdf, 1, boxes)
    assert crops == []  # failed read drops the table; never raises


def test_clean_markdown_strips_fences_and_prose():
    raw = "Here is the table:\n```markdown\n| a | b |\n| - | - |\n| 1 | 2 |\n```\nDone."
    assert _clean_markdown(raw) == "| a | b |\n| - | - |\n| 1 | 2 |"


def test_clean_markdown_empty_when_no_table():
    assert _clean_markdown("The image contains no table.") == ""


# --------------------------------------------------------------------------
# Orchestrator phase (real PDF + locate + crop render; only the VLM is stubbed)
# --------------------------------------------------------------------------


def _assessment(has_tables=True):
    return DocumentAssessment(
        path=Path("/tmp/fake.pdf"),
        pages=[
            PageAssessment(
                page_num=1, is_born_digital=True, native_text="",
                confidence=1.0, has_tables=has_tables, has_equations=False,
            )
        ],
    )


def _state_with_table_page(pdf: Path, page_text: str, engine="gemini"):
    state = DocumentState(handle=DocumentHandle(path=pdf))
    bo = PageOutput(page_num=1, text=page_text, status=PageStatus.SUCCESS,
                    engine=engine, audit_passed=True)
    state.pages[1].attempts.append(bo)
    state.pages[1].best_output = bo
    return state, bo


def _wire_reader(monkeypatch, reader):
    import socr.tables.extract as extract_mod
    monkeypatch.setattr(extract_mod, "OllamaTableReader", lambda *a, **k: reader)


def test_phase_patches_corrupted_table_on_real_pdf(tmp_path, monkeypatch):
    doc, _ = _build_page("booktabs")
    pdf = tmp_path / "doc.pdf"
    doc.save(pdf)

    pipe = UnifiedPipeline(PipelineConfig(quiet=True))
    pipe._last_assessment = _assessment(has_tables=True)
    state, bo = _state_with_table_page(pdf, _PAGE_MD)  # contains the (0.0l0) misread
    monkeypatch.setattr(pipe, "_resolve_judge_model", lambda: "mock")
    _wire_reader(monkeypatch, _StubReader(_CROP_MD))

    pipe._phase_dual_pass_tables(state)

    assert "(0.010)" in bo.text and "(0.0l0)" not in bo.text
    assert any("dual-pass patched" in n for n in bo.audit_notes)


def test_phase_noop_without_vision_model(tmp_path, monkeypatch):
    doc, _ = _build_page("booktabs")
    pdf = tmp_path / "doc.pdf"
    doc.save(pdf)

    pipe = UnifiedPipeline(PipelineConfig(quiet=True))
    pipe._last_assessment = _assessment(has_tables=True)
    state, bo = _state_with_table_page(pdf, _PAGE_MD)
    monkeypatch.setattr(pipe, "_resolve_judge_model", lambda: None)

    pipe._phase_dual_pass_tables(state)
    assert bo.text == _PAGE_MD  # untouched


def test_phase_skips_native_pages(tmp_path, monkeypatch):
    doc, _ = _build_page("booktabs")
    pdf = tmp_path / "doc.pdf"
    doc.save(pdf)

    pipe = UnifiedPipeline(PipelineConfig(quiet=True))
    pipe._last_assessment = _assessment(has_tables=True)
    state, bo = _state_with_table_page(pdf, _PAGE_MD, engine="native")
    monkeypatch.setattr(pipe, "_resolve_judge_model", lambda: "mock")
    reader = _StubReader(_CROP_MD)
    _wire_reader(monkeypatch, reader)

    pipe._phase_dual_pass_tables(state)
    assert reader.calls == 0  # native text is char-exact; not re-read
    assert bo.text == _PAGE_MD


def test_phase_disabled_flag_default():
    assert PipelineConfig().dual_pass_tables is True
    assert PipelineConfig(dual_pass_tables=False).dual_pass_tables is False
