"""#1053 first slice: a single-column page's raster figure is cropped and placed inline.

Hermetic: synthetic fitz PDFs, stub provider, no judge. The process() pins are DIFFERENCES (the
crop path neutralised vs live in the same process), never an absolute outcome (CLAUDE.md, #257).
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.result import DocumentStatus, FailureMode
from socr.figures import figure_crops
from socr.figures.extractor import figure_boxes
from socr.pipeline.orchestrator import UnifiedPipeline

ABOVE = [f"Above paragraph line {w} discusses the estimated effect." for w in "ABCD"]
BELOW = [f"Below paragraph line {w} reports the robust standard errors." for w in "EFGH"]
CAPTION = "Figure 1: Estimated effect by region"
FIG = fitz.Rect(72, 200, 372, 400)
LABELS = ["Zeta", "Omega"]


def _make(
    path: Path,
    *,
    labelled: bool = False,
    two_column: bool = False,
    caption: bool = True,
    extra=None,
    below_first: bool = False,
) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    if below_first:
        # The stream writes the below-figure paragraph BEFORE the above-figure one.
        y = FIG.y1 + 44
        for line in BELOW:
            page.insert_text((72, y), line, fontname="helv", fontsize=10)
            y += 14
    y = 72.0
    for line in ABOVE:
        page.insert_text((72, y), line, fontname="helv", fontsize=10)
        y += 14
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 300, 200), False)
    pix.set_rect(pix.irect, (235, 235, 235))
    page.insert_image(FIG, pixmap=pix)
    if labelled:
        for i, label in enumerate(LABELS):
            page.insert_text(
                (FIG.x0 + 20, FIG.y0 + 40 + 30 * i), label, fontname="helv", fontsize=10
            )
    if caption:
        page.insert_text((72, FIG.y1 + 16), CAPTION, fontname="helv", fontsize=10)
    if extra is not None:
        extra(page)
    y = FIG.y1 + 44
    if not below_first:
        for line in BELOW:
            page.insert_text((72, y), line, fontname="helv", fontsize=10)
            y += 14
    if two_column:
        for i in range(4):
            page.insert_text((400, 72 + 14 * i), f"Right column row {i} of text", fontsize=10)
    doc.save(path)
    doc.close()
    return path


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert (
        Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")
    ), socr.__file__


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    from socr.core.providers import PROFILE_QWEN_LOCAL

    monkeypatch.setattr(
        UnifiedPipeline, "_available_engines_for_agentic", lambda self: [PROFILE_QWEN_LOCAL]
    )
    monkeypatch.setattr(UnifiedPipeline, "_resolve_judge_model", lambda self, *a, **kw: "")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_figure_boxes_one_record_per_drawn_placement(tmp_path: Path) -> None:
    pdf = _make(tmp_path / "p.pdf")
    with fitz.open(pdf) as doc:
        boxes = figure_boxes(doc[0])
    assert boxes is not None and len(boxes) == 1
    assert boxes[0] == pytest.approx(tuple(FIG), abs=1.0)


def test_figure_boxes_skips_a_small_placement(tmp_path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 30, 20), False)
    pix.set_rect(pix.irect, (10, 10, 10))
    page.insert_image(fitz.Rect(72, 72, 102, 92), pixmap=pix)
    assert figure_boxes(page) == []


def test_word_owners_assign_every_word_exactly_once(tmp_path: Path) -> None:
    pdf = _make(tmp_path / "p.pdf", labelled=True)
    with fitz.open(pdf) as doc:
        plan = figure_crops.plan_figure_page(doc[0])
        words = doc[0].get_text("words")
    assert plan is not None
    assert len(plan.owners) == len(words)
    assert plan.owner_counts()["figure:1"] == len(LABELS)
    assert plan.owner_counts()[figure_crops.OWNER_CAPTION] == len(CAPTION.split())
    assert plan.unread_figures == []
    assert plan.figure_word_counts == {1: len(LABELS)}


def test_unlabelled_figure_is_unread(tmp_path: Path) -> None:
    with fitz.open(_make(tmp_path / "p.pdf")) as doc:
        plan = figure_crops.plan_figure_page(doc[0])
    assert plan is not None and plan.unread_figures == [1]


def test_two_column_page_declines(tmp_path: Path) -> None:
    with fitz.open(_make(tmp_path / "p.pdf", two_column=True)) as doc:
        assert figure_crops.plan_figure_page(doc[0]) is None


# ---------------------------------------------------------------------------
# End-to-end through process()
# ---------------------------------------------------------------------------


class _StubEngine:
    name = "qwen"

    def __init__(self) -> None:
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        from socr.core.result import PageOutput, PageStatus

        self.calls += 1
        return [
            PageOutput(page_num=n, text="PROVIDER_READ", status=PageStatus.SUCCESS, engine="qwen")
            for n in page_nums
        ]


class _AcceptingJudge:
    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        return AcceptDecision(accept=True, reason="stub accepts all")


def _process(tmp_path, tag, monkeypatch, *, provider, neutralised, **shape):
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch

    engine = _StubEngine()
    pdf = tmp_path / f"{tag}.pdf"
    if not pdf.exists():
        _make(pdf, **shape)
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: engine)
        if neutralised:
            m.setattr(figure_crops, "plan_figure_page", lambda page: None)
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                native_first=True,
                write_manifest=False,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                save_figures=False,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL] if provider else []
        pipe._build_page_judge = lambda state: _AcceptingJudge()
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        result = pipe.process(pdf, output_dir=tmp_path / f"out-{tag}")
    return result, engine


def _one(tmp_path, tag, pattern) -> Path:
    found = sorted((tmp_path / f"out-{tag}").rglob(pattern))
    assert len(found) == 1, found
    return found[0]


def _sidecar(tmp_path, tag) -> dict:
    return json.loads(_one(tmp_path, tag, "pages/00001.json").read_text())


def _page_text(tmp_path, tag) -> str:
    return _one(tmp_path, tag, "pages/00001.md").read_text()


def _native_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("![")]


@pytest.mark.parametrize("provider", [True, False])
def test_crop_lands_between_the_paragraphs_and_flags_the_page(
    tmp_path, monkeypatch, provider
) -> None:
    kw = dict(provider=provider)
    off, eng_off = _process(tmp_path, "off", monkeypatch, neutralised=True, **kw)
    on, eng_on = _process(tmp_path, "on", monkeypatch, neutralised=False, **kw)
    t_off, t_on = _page_text(tmp_path, "off"), _page_text(tmp_path, "on")

    # Control: without the crop path the whole page is appended after all the prose.
    assert "chart_page_1.png" in t_off and "chart_region_p1_1.png" not in t_off
    # Crop path: the figure ref sits between the two paragraphs and no whole-page ref remains.
    assert "chart_page_1.png" not in t_on
    lines = t_on.splitlines()
    ref = next(i for i, ln in enumerate(lines) if "chart_region_p1_1.png" in ln)
    assert any(ABOVE[-1] in ln for ln in lines[:ref])
    assert any(BELOW[0] in ln for ln in lines[ref + 1 :])
    assert not any(BELOW[0] in ln for ln in lines[:ref])

    # The crop is smaller than the page.
    from PIL import Image

    crop = next((tmp_path / "out-on").rglob("chart_region_p1_1.png"))
    page_png = next((tmp_path / "out-off").rglob("chart_page_1.png"))
    with Image.open(crop) as c, Image.open(page_png) as p:
        assert c.width * c.height < p.width * p.height

    # Every native line survives, in order.
    assert _native_lines(t_on) == _native_lines(t_off)

    # Status: a figure box with no native word is unread. Only status and failure mode move.
    a, b = _sidecar(tmp_path, "off"), _sidecar(tmp_path, "on")
    assert a["failure_mode"] != FailureMode.FIGURE_WORDS_UNREAD.value
    assert b["status"] == "warning"
    assert b["failure_mode"] == FailureMode.FIGURE_WORDS_UNREAD.value
    assert b.get("audit_passed") == a.get("audit_passed")
    assert on.status is not DocumentStatus.SUCCESS
    # The page is not re-routed: same provider calls.
    assert eng_on.calls == eng_off.calls


@pytest.mark.parametrize("provider", [True, False])
def test_figure_with_its_own_words_ships_them_once_and_is_not_flagged(
    tmp_path, monkeypatch, provider
) -> None:
    kw = dict(provider=provider, labelled=True)
    _process(tmp_path, "off", monkeypatch, neutralised=True, **kw)
    on, _ = _process(tmp_path, "on", monkeypatch, neutralised=False, **kw)
    t_off, t_on = _page_text(tmp_path, "off"), _page_text(tmp_path, "on")
    for label in LABELS:
        assert t_off.count(label) == 1
        assert t_on.count(label) == 1, t_on
    a, b = _sidecar(tmp_path, "off"), _sidecar(tmp_path, "on")
    assert b["failure_mode"] != FailureMode.FIGURE_WORDS_UNREAD.value
    assert (a["status"], a["failure_mode"]) == (b["status"], b["failure_mode"])
    assert sorted(_native_lines(t_on)) == sorted(_native_lines(t_off))


@pytest.mark.parametrize("provider", [True, False])
def test_two_column_page_keeps_the_whole_page_route(tmp_path, monkeypatch, provider) -> None:
    kw = dict(provider=provider, two_column=True)
    off, _ = _process(tmp_path, "off", monkeypatch, neutralised=True, **kw)
    on, _ = _process(tmp_path, "on", monkeypatch, neutralised=False, **kw)
    assert _page_text(tmp_path, "on") == _page_text(tmp_path, "off")
    assert "chart_page_1.png" in _page_text(tmp_path, "on")
    a, b = _sidecar(tmp_path, "off"), _sidecar(tmp_path, "on")
    assert (a["status"], a["failure_mode"]) == (b["status"], b["failure_mode"])
    assert on.status is off.status


def test_flagged_page_is_not_restored_from_the_ledger(tmp_path, monkeypatch) -> None:
    """A WARNING page is never terminal: the next run reprocesses it and the mode survives."""
    monkeypatch.setattr(UnifiedPipeline, "_resume_skip", lambda self, *a, **k: None)
    loaded: list = []
    real = UnifiedPipeline._load_terminal_page

    def spy(self, *a, **k):
        out = real(self, *a, **k)
        loaded.append(out)
        return out

    monkeypatch.setattr(UnifiedPipeline, "_load_terminal_page", spy)
    _process(tmp_path, "r", monkeypatch, provider=True, neutralised=False)
    first = _sidecar(tmp_path, "r")
    loaded.clear()
    result, _ = _process(tmp_path, "r", monkeypatch, provider=True, neutralised=False)
    assert loaded and all(o is None for o in loaded)
    second = _sidecar(tmp_path, "r")
    assert second["failure_mode"] == first["failure_mode"] == FailureMode.FIGURE_WORDS_UNREAD.value
    assert result.status is not DocumentStatus.SUCCESS


# ---------------------------------------------------------------------------
# Astra round 1: nothing a figure box touches is suppressed or duplicated
# ---------------------------------------------------------------------------


def _lines(tmp_path, tag, **shape) -> list[str]:
    return [ln.strip() for ln in _page_text(tmp_path, tag).splitlines() if ln.strip()]


def _assert_crop_route(tmp_path, tag) -> None:
    """The per-figure route ran: a crop is referenced and the whole-page PNG is not."""
    text = _page_text(tmp_path, tag)
    assert "chart_region_p1_1.png" in text and "chart_page_1.png" not in text, text


def _straddle(page) -> None:
    # "10 10": the first 10 is inside the box, the second is outside it; the line is mostly inside.
    page.insert_text((300, 300), "alpha 10              10", fontname="helv", fontsize=10)


def test_straddling_line_ships_both_tokens(tmp_path, monkeypatch) -> None:
    with fitz.open(_make(tmp_path / "chk.pdf", extra=_straddle)) as doc:
        plan = figure_crops.plan_figure_page(doc[0])
        w = [x for x in doc[0].get_text("words") if x[4] == "10"]
    assert plan is not None and len(w) == 2, "setup: one 10 inside the box, one outside"
    _process(tmp_path, "on", monkeypatch, provider=True, neutralised=False, extra=_straddle)
    _assert_crop_route(tmp_path, "on")
    assert sum(ln.split().count("10") for ln in _lines(tmp_path, "on")) == 2


def _straddle_placeholder_tokens(page) -> None:
    # Every token of this line also occurs in the placeholder text, so a bag-of-tokens
    # suppression would drop the whole line, including the "1" outside the box.
    page.insert_text((300, 300), "figure 1              1", fontname="helv", fontsize=10)


def test_straddling_line_of_placeholder_tokens_ships_whole(tmp_path, monkeypatch) -> None:
    kw = dict(extra=_straddle_placeholder_tokens)
    _process(tmp_path, "on", monkeypatch, provider=True, neutralised=False, **kw)
    _assert_crop_route(tmp_path, "on")
    assert "figure 1              1".split() in [ln.split() for ln in _lines(tmp_path, "on")]


def _tick(page) -> None:
    page.insert_text((100, 300), "1", fontname="helv", fontsize=10)


def test_tick_label_equal_to_a_placeholder_token_ships(tmp_path, monkeypatch) -> None:
    _process(tmp_path, "on", monkeypatch, provider=True, neutralised=False, extra=_tick)
    _assert_crop_route(tmp_path, "on")
    assert _lines(tmp_path, "on").count("1") == 1


def _label_and_notes(page) -> None:
    # A label wholly inside the box, in one block with notes below it (the block is < half inside).
    for i, ln in enumerate(["Zeta", "Notes: source data", "Notes: second line", "Notes: third"]):
        page.insert_text((100, 392 + 12 * i), ln, fontname="helv", fontsize=10)


def test_label_in_a_mostly_outside_block_ships_exactly_once(tmp_path, monkeypatch) -> None:
    kw = dict(extra=_label_and_notes, caption=False)
    _process(tmp_path, "on", monkeypatch, provider=True, neutralised=False, **kw)
    _assert_crop_route(tmp_path, "on")
    assert _page_text(tmp_path, "on").count("Zeta") == 1


def test_out_of_order_stream_keeps_the_whole_page_route(tmp_path, monkeypatch) -> None:
    kw = dict(provider=True, below_first=True)
    off, _ = _process(tmp_path, "off", monkeypatch, neutralised=True, **kw)
    on, _ = _process(tmp_path, "on", monkeypatch, neutralised=False, **kw)
    assert _page_text(tmp_path, "on") == _page_text(tmp_path, "off")
    assert "chart_page_1.png" in _page_text(tmp_path, "on")
    a, b = _sidecar(tmp_path, "off"), _sidecar(tmp_path, "on")
    assert (a["status"], a["failure_mode"]) == (b["status"], b["failure_mode"])
    assert on.status is off.status


def test_table_path_still_drops_a_represented_line_by_default() -> None:
    from socr.core.born_digital import BornDigitalDetector

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Revenue 10", fontname="helv", fontsize=10)
    page.insert_text((72, 300), "Closing paragraph", fontname="helv", fontsize=10)
    region = (fitz.Rect(60, 85, 200, 110), "| Revenue | 10 |\n| --- | --- |")
    det = BornDigitalDetector()
    kept = det.interleave_table_regions_into_page(page, [region])
    assert "Revenue 10" not in kept and "| Revenue | 10 |" in kept
    plain = det.interleave_table_regions_into_page(page, [region], suppress_represented=False)
    assert "Revenue 10" in plain and "| Revenue | 10 |" in plain


def _spanning_block(page) -> None:
    # One block whose first line is above the box top (200) and whose last line is inside it.
    page.insert_text(
        (72, 195),
        "Spanning first line above the figure top\nSpanning second line\nSpanning last line inside",
        fontsize=10,
    )


def test_block_spanning_the_insertion_point_keeps_the_whole_page_route(
    tmp_path, monkeypatch
) -> None:
    with fitz.open(_make(tmp_path / "chk.pdf", extra=_spanning_block, caption=False)) as doc:
        blocks = [b for b in doc[0].get_text("dict")["blocks"] if b.get("type") == 0]
        spanning = [
            b
            for b in blocks
            if b["bbox"][1] < FIG.y0 and any(ln["bbox"][1] >= FIG.y0 for ln in b["lines"])
        ]
        assert spanning, "setup: one block must straddle the box top"
        assert figure_crops.plan_figure_page(doc[0]) is None
    kw = dict(provider=True, extra=_spanning_block, caption=False)
    off, _ = _process(tmp_path, "off", monkeypatch, neutralised=True, **kw)
    on, _ = _process(tmp_path, "on", monkeypatch, neutralised=False, **kw)
    assert _page_text(tmp_path, "on") == _page_text(tmp_path, "off")
    assert "chart_page_1.png" in _page_text(tmp_path, "on")
