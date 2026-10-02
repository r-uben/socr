"""#961: a scan with a baked-in (invisible) OCR text layer must not ship as trusted native.

Detector: render-mode-3 text (``get_texttrace`` type 3) on a page whose raster coverage is
>= ``RASTER_DOMINANCE_RATIO``. A hit leaves the trusted-native lane for OCR.

Hermetic: synthetic fitz PDFs, stub provider, no judge. Routing pins are DIFFERENCES
(detector neutralised vs live in the same process), never absolute outcomes (CLAUDE.md, #257).
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from socr.core.born_digital import BornDigitalDetector
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import DocumentStatus, FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline

_PROSE = (
    "The estimated effect of the policy change on output is reported below for each "
    "specification, with robust standard errors clustered by region and year. "
)


def _pixmap(w=200, h=200):
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, w, h), False)
    pix.set_rect(pix.irect, (235, 235, 235))
    return pix


def _make(path: Path, *, image: str, render_mode: int) -> Path:
    """``image``: full | small | none. ``render_mode`` 3 = invisible, 0 = visible."""
    doc = fitz.open()
    page = doc.new_page()
    if image == "full":
        page.insert_image(page.rect, pixmap=_pixmap())
    elif image == "small":
        page.insert_image(fitz.Rect(72, 500, 172, 600), pixmap=_pixmap(100, 100))
    y = 72
    for _ in range(8):
        page.insert_text((72, y), _PROSE, fontname="helv", fontsize=10, render_mode=render_mode)
        y += 14
    doc.save(path)
    doc.close()
    return path


SHAPES = {
    "scan_invisible_over_full_raster": ("full", 3, True),
    "born_digital_visible_text": ("none", 0, False),
    "small_figure_plus_invisible_text": ("small", 3, False),
    "full_raster_visible_text": ("full", 0, False),
    "invisible_text_no_raster": ("none", 3, False),
}


def _detect(pdf: Path) -> bool:
    with fitz.open(pdf) as doc:
        return BornDigitalDetector()._has_invisible_text_over_raster(doc[0])


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_detector_shapes(tmp_path: Path, name: str) -> None:
    image, mode, expected = SHAPES[name]
    assert _detect(_make(tmp_path / "p.pdf", image=image, render_mode=mode)) is expected


def test_texttrace_is_skipped_without_enough_raster(tmp_path: Path) -> None:
    """The cost guard: a born-digital page never pays for the text trace."""
    pdf = _make(tmp_path / "p.pdf", image="small", render_mode=3)
    with fitz.open(pdf) as doc:
        page = doc[0]
        calls = []
        real = page.get_texttrace
        page.get_texttrace = lambda *a, **k: calls.append(1) or real(*a, **k)
        assert BornDigitalDetector()._has_invisible_text_over_raster(page) is False
        assert calls == [], "get_texttrace must not run below the raster ratio"


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


def _analyze(pdf: Path, native_only: bool = False):
    pipe = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.DEEPSEEK,
            enabled_engines=[EngineType.GEMINI],
            agentic=True,
            quiet=True,
            native_first=True,
            native_only=native_only,
        )
    )
    state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    pipe._phase_analyze(state)
    return pipe, state


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_routing_difference_pin(tmp_path: Path, name: str, monkeypatch) -> None:
    image, mode, expected = SHAPES[name]
    pdf = _make(tmp_path / "p.pdf", image=image, render_mode=mode)
    with monkeypatch.context() as m:
        m.setattr(BornDigitalDetector, "_has_invisible_text_over_raster", lambda self, p: False)
        pipe_off, state_off = _analyze(pdf)
        trusted_off = pipe_off._is_agentic_trusted_native(1, state_off.pages[1])
        kinds_off = [e.kind for e in state_off.events]
    pipe_on, state_on = _analyze(pdf)
    trusted_on = pipe_on._is_agentic_trusted_native(1, state_on.pages[1])
    events_on = [e for e in state_on.events if e.kind == "invisible_text_scan"]

    assert state_off.pages[1].is_born_digital
    assert trusted_off, "baseline (main): every shape ships trusted native"
    assert "invisible_text_scan" not in kinds_off
    if expected:
        assert not trusted_on
        assert [(e.page_num, e.data["error"]) for e in events_on] == [(1, False)]
    else:
        assert trusted_on and events_on == []


def test_event_is_recomputed_not_replayed() -> None:
    assert "invisible_text_scan" not in UnifiedPipeline.resume_restore_kinds()


def test_event_text_routed_retained_and_error(tmp_path: Path, monkeypatch) -> None:
    pdf = _make(tmp_path / "p.pdf", image="full", render_mode=3)
    routed = [e for e in _analyze(pdf)[1].events if e.kind == "invisible_text_scan"][0]
    retained = [e for e in _analyze(pdf, True)[1].events if e.kind == "invisible_text_scan"][0]
    assert "routed to OCR" in routed.detail and routed.data["error"] is False
    assert "RETAINED" in retained.detail and "routed to OCR" not in retained.detail

    def _boom(self, page):
        raise RuntimeError("x")

    with monkeypatch.context() as m:
        m.setattr(BornDigitalDetector, "_has_invisible_text_over_raster", _boom)
        failed = [e for e in _analyze(pdf)[1].events if e.kind == "invisible_text_scan"][0]
    assert failed.data["error"] is True and "FAILED" in failed.detail


# ---------------------------------------------------------------------------
# End-to-end through process()
# ---------------------------------------------------------------------------

_OCR_MARK = "PROVIDER_READ_OF_THE_PAGE"


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
            PageOutput(page_num=n, text=_OCR_MARK, status=PageStatus.SUCCESS, engine="qwen")
            for n in page_nums
        ]


class _AcceptingJudge:
    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        return AcceptDecision(accept=True, reason="stub accepts all")


def _chart_pdf(path: Path) -> Path:
    """Visible prose plus a large embedded raster: ``has_chart_marks`` fires, the chart lane
    takes the page. The raster is below the dominance ratio, so the REAL detector is quiet and
    the ``forced`` mode below stands in for it (this test is about the lane, not the detector)."""
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for _ in range(8):
        page.insert_text((72, y), _PROSE, fontname="helv", fontsize=10)
        y += 14
    page.insert_image(fitz.Rect(72, y + 20, 372, y + 320), pixmap=_pixmap(300, 300))
    doc.save(path)
    doc.close()
    return path


def _e2e(
    tmp_path,
    tag,
    monkeypatch,
    *,
    provider,
    native_only=False,
    detector="live",
    chart=False,
    recover_corrupt_math=True,
):
    """detector: live | neutralised | raising | forced (always fires)."""
    if detector not in {"live", "neutralised", "raising", "forced"}:
        raise ValueError(detector)
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch

    engine = _StubEngine()
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: engine)
        if detector == "neutralised":
            m.setattr(BornDigitalDetector, "_has_invisible_text_over_raster", lambda s, p: False)
        elif detector == "forced":
            m.setattr(BornDigitalDetector, "_has_invisible_text_over_raster", lambda s, p: True)
        elif detector == "raising":

            def _boom(self, page):
                raise RuntimeError("detector exploded")

            m.setattr(BornDigitalDetector, "_has_invisible_text_over_raster", _boom)
        if chart:
            pdf = _chart_pdf(tmp_path / f"{tag}.pdf")
        else:
            pdf = _make(tmp_path / f"{tag}.pdf", image="full", render_mode=3)
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                native_first=True,
                native_only=native_only,
                recover_corrupt_math=recover_corrupt_math,
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


def _sidecar(tmp_path, tag) -> dict:
    found = sorted((tmp_path / f"out-{tag}").rglob("pages/00001.json"))
    assert len(found) == 1, found
    return json.loads(found[0].read_text())


def _page_text(tmp_path, tag) -> str:
    found = sorted((tmp_path / f"out-{tag}").rglob("pages/00001.md"))
    assert len(found) == 1, found
    return found[0].read_text()


def test_e2e_provider_present_text_comes_from_the_provider(tmp_path, monkeypatch) -> None:
    _, eng_off = _e2e(tmp_path, "a_off", monkeypatch, provider=True, detector="neutralised")
    _, eng_on = _e2e(tmp_path, "a_on", monkeypatch, provider=True)
    assert eng_off.calls == 0 and _OCR_MARK not in _page_text(tmp_path, "a_off")
    assert "estimated effect" in _page_text(tmp_path, "a_off"), "main: the old OCR ships"
    assert eng_on.calls == 1
    assert _OCR_MARK in _page_text(tmp_path, "a_on")
    assert "estimated effect" not in _page_text(tmp_path, "a_on")


def test_e2e_provider_absent_is_never_trusted_success(tmp_path, monkeypatch) -> None:
    off, _ = _e2e(tmp_path, "b_off", monkeypatch, provider=False, detector="neutralised")
    on, _ = _e2e(tmp_path, "b_on", monkeypatch, provider=False)
    assert _sidecar(tmp_path, "b_off")["status"] == "success"
    assert off.status is DocumentStatus.SUCCESS
    side = _sidecar(tmp_path, "b_on")
    assert side["status"] != "success"
    assert side["failure_mode"] == FailureMode.NATIVE_INVISIBLE_TEXT_SCAN.value
    assert on.status is not DocumentStatus.SUCCESS
    assert "estimated effect" in _page_text(tmp_path, "b_on"), "no content dropped"


def test_e2e_native_only_retains_the_page_but_demotes_it(tmp_path, monkeypatch) -> None:
    off, _ = _e2e(
        tmp_path, "c_off", monkeypatch, provider=True, native_only=True, detector="neutralised"
    )
    on, eng_on = _e2e(tmp_path, "c_on", monkeypatch, provider=True, native_only=True)
    assert _sidecar(tmp_path, "c_off")["status"] == "success"
    assert off.status is DocumentStatus.SUCCESS
    side = _sidecar(tmp_path, "c_on")
    assert eng_on.calls == 0
    assert side["status"] == "warning"
    assert side["failure_mode"] == FailureMode.NATIVE_INVISIBLE_TEXT_SCAN.value
    assert on.status is not DocumentStatus.SUCCESS
    assert "estimated effect" in _page_text(tmp_path, "c_on")


def test_e2e_detector_exception_fails_closed(tmp_path, monkeypatch) -> None:
    _, eng_off = _e2e(tmp_path, "d_off", monkeypatch, provider=True, detector="neutralised")
    _, eng_on = _e2e(tmp_path, "d_on", monkeypatch, provider=True, detector="raising")
    assert eng_off.calls == 0
    assert eng_on.calls == 1
    assert _OCR_MARK in _page_text(tmp_path, "d_on")


def test_e2e_native_only_with_a_raising_detector_is_demoted_too(tmp_path, monkeypatch) -> None:
    _e2e(tmp_path, "e_off", monkeypatch, provider=True, native_only=True, detector="neutralised")
    assert _sidecar(tmp_path, "e_off")["status"] == "success"
    on, _ = _e2e(tmp_path, "e_on", monkeypatch, provider=True, native_only=True, detector="raising")
    side = _sidecar(tmp_path, "e_on")
    assert side["status"] == "warning"
    assert side["failure_mode"] == FailureMode.NATIVE_INVISIBLE_TEXT_SCAN.value
    assert on.status is not DocumentStatus.SUCCESS


def test_e2e_chart_asset_page_with_a_hit_is_demoted_too(tmp_path, monkeypatch) -> None:
    """The chart lane ships retained native prose; it must not bypass the demotion."""
    off, _ = _e2e(
        tmp_path,
        "f_off",
        monkeypatch,
        provider=True,
        native_only=True,
        chart=True,
        detector="neutralised",
    )
    side_off = _sidecar(tmp_path, "f_off")
    assert side_off["engine"] == "chart_asset", "setup: the page must take the chart lane"
    assert side_off["status"] == "success" and off.status is DocumentStatus.SUCCESS

    on, _ = _e2e(
        tmp_path,
        "f_on",
        monkeypatch,
        provider=True,
        native_only=True,
        chart=True,
        detector="forced",
    )
    side_on = _sidecar(tmp_path, "f_on")
    assert side_on["engine"] == "chart_asset"
    assert side_on["status"] == "warning"
    assert side_on["failure_mode"] == FailureMode.NATIVE_INVISIBLE_TEXT_SCAN.value
    assert on.status is not DocumentStatus.SUCCESS


def _run_twice(tmp_path, monkeypatch, *, detector="live"):
    """process() the SAME pdf into the SAME output dir twice (a resume), fresh pipeline each."""
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch

    pdf = _make(tmp_path / "r.pdf", image="full", render_mode=3)
    out = tmp_path / "out-r"
    engines = []
    for _ in range(2):
        engine = _StubEngine()
        engines.append(engine)
        with monkeypatch.context() as m:
            m.setattr(orch, "get_engine", lambda engine_type, e=engine: e)
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
            pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
            pipe._build_page_judge = lambda state: _AcceptingJudge()
            pipe._resolve_crop_vlm_model = lambda: None
            pipe._resolve_judge_model = lambda *a, **k: ""
            pipe.process(pdf, output_dir=out)
    return out, engines


def _audit_kinds(out: Path) -> list[str]:
    kinds = []
    for f in sorted(out.rglob("*audit*.json")):
        text = f.read_text()
        if text.strip().startswith("{"):
            kinds += [e["kind"] for e in json.loads(text).get("events", [])]
    return kinds


def test_resume_run_recomputes_the_event_once_and_keeps_the_ocr_page(tmp_path, monkeypatch) -> None:
    out, engines = _run_twice(tmp_path, monkeypatch)
    assert engines[0].calls == 1, "first run: the scan page goes to OCR"
    assert engines[1].calls == 0, "second run: the finished page is resumed, not re-read"
    kinds = _audit_kinds(out)
    assert kinds.count("invisible_text_scan") == 1, (
        "the second run recomputes the event; replaying it as well would double it"
    )
    found = sorted(out.rglob("pages/00001.md"))
    assert len(found) == 1 and _OCR_MARK in found[0].read_text()


# ---------------------------------------------------------------------------
# Astra fixes on #963: union coverage, majority-invisible, corrupt-math lane, D3
# ---------------------------------------------------------------------------


def _mixed_pdf(path: Path, *, visible_lines: int, invisible_lines: int, images) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    for rect in images:
        page.insert_image(rect, pixmap=_pixmap())
    y = 72
    for mode, n in ((0, visible_lines), (3, invisible_lines)):
        for _ in range(n):
            page.insert_text((72, y), _PROSE, fontname="helv", fontsize=10, render_mode=mode)
            y += 14
    doc.save(path)
    doc.close()
    return path


def test_background_image_with_mostly_visible_text_and_a_little_invisible_is_quiet(tmp_path):
    """A born-digital page with a full-page background image plus invisible accessibility text."""
    rect = fitz.Rect(0, 0, 595, 842)
    pdf = _mixed_pdf(tmp_path / "p.pdf", visible_lines=8, invisible_lines=1, images=[rect])
    assert _detect(pdf) is False
    # Control: the same page with the text mostly invisible fires.
    pdf = _mixed_pdf(tmp_path / "q.pdf", visible_lines=1, invisible_lines=8, images=[rect])
    assert _detect(pdf) is True


def test_invisible_majority_boundary(tmp_path):
    """Exactly half invisible is not a majority; one more invisible line is."""
    rect = fitz.Rect(0, 0, 595, 842)
    half = _mixed_pdf(tmp_path / "h.pdf", visible_lines=4, invisible_lines=4, images=[rect])
    more = _mixed_pdf(tmp_path / "m.pdf", visible_lines=4, invisible_lines=5, images=[rect])
    assert _detect(half) is False
    assert _detect(more) is True


def test_overlapping_images_do_not_inflate_coverage(tmp_path):
    """Two images stacked on the same 60% of the page: summed area 1.2 passes the ratio,
    the union (0.6) does not."""
    r = fitz.Rect(0, 0, 595, 505)
    pdf = _mixed_pdf(tmp_path / "o.pdf", visible_lines=0, invisible_lines=8, images=[r, r])
    with fitz.open(pdf) as doc:
        page = doc[0]
        summed = sum(fitz.Rect(i["bbox"]).get_area() for i in page.get_image_info())
        assert summed / page.rect.get_area() >= BornDigitalDetector.RASTER_DOMINANCE_RATIO
        assert BornDigitalDetector()._has_invisible_text_over_raster(page) is False


def test_union_coverage_boundary(tmp_path):
    """Two side-by-side images whose union is just under / just over the ratio."""
    ratio = BornDigitalDetector.RASTER_DOMINANCE_RATIO
    w, h = 595.0, 842.0

    def build(frac, name):
        top = fitz.Rect(0, 0, w, h * frac / 2)
        bottom = fitz.Rect(0, h * frac / 2, w, h * frac)
        return _mixed_pdf(tmp_path / name, visible_lines=0, invisible_lines=8, images=[top, bottom])

    assert _detect(build(ratio - 0.02, "under.pdf")) is False
    assert _detect(build(ratio + 0.02, "over.pdf")) is True


def test_union_area_helper() -> None:
    from socr.core.born_digital import _union_area

    clip = fitz.Rect(0, 0, 100, 100)
    a = fitz.Rect(0, 0, 60, 100)
    b = fitz.Rect(40, 0, 100, 100)
    assert _union_area([a, b], clip) == pytest.approx(10000.0)
    assert _union_area([a, a], clip) == pytest.approx(6000.0)
    assert _union_area([fitz.Rect(-50, -50, 50, 50)], clip) == pytest.approx(2500.0)
    assert _union_area([], clip) == 0.0


def _scan_state(tmp_path, monkeypatch, *, detector_live=True):
    pdf = _make(tmp_path / "s.pdf", image="full", render_mode=3)
    with monkeypatch.context() as m:
        if not detector_live:
            m.setattr(BornDigitalDetector, "_has_invisible_text_over_raster", lambda s, p: False)
        pipe, state = _analyze(pdf)
    return pipe, state.pages[1]


def test_scan_page_never_enters_the_corrupt_math_lane(tmp_path, monkeypatch) -> None:
    """Native prose must not survive a corrupt-math hybrid on a scan page: whole-page OCR."""
    pipe_off, ps_off = _scan_state(tmp_path, monkeypatch, detector_live=False)
    pipe_on, ps_on = _scan_state(tmp_path, monkeypatch)
    for pipe, ps in ((pipe_off, ps_off), (pipe_on, ps_on)):
        ps.has_corrupt_math = True
        pipe.config.recover_corrupt_math = True
    assert pipe_off._is_corrupt_math_recovery_page(1, ps_off), "main: the hybrid lane owns it"
    assert not pipe_on._is_corrupt_math_recovery_page(1, ps_on)
    # A failed scan is unknown, and treated as a scan.
    ps_off.invisible_text_scan_failed = True
    assert not pipe_off._is_corrupt_math_recovery_page(1, ps_off)


def test_d3_native_retention_of_a_scan_page_ships_error_not_success(tmp_path) -> None:
    """The D3 floor (unverifiable native table) on a scan page ships the failure marker."""
    from socr.core.manifest import _select_page_output_tagged

    def ship(scan: bool):
        pdf = tmp_path / f"d3-{scan}.pdf"
        doc = fitz.open()
        doc.new_page().insert_text((54, 72), "text layer long enough to count as native here.")
        doc.save(pdf)
        doc.close()
        state = DocumentState(handle=DocumentHandle.from_path(pdf))
        p = state.pages[1]
        p.is_born_digital = True
        p.native_text = "native body"
        p.native_table_structure_failed = True
        p.native_table_unverifiable = True
        p.has_tables = True
        p.needs_ocr_enhancement = True
        p.d3_floor_png_ref = "![p1](figures/p001.png)"
        p.invisible_text_over_raster = scan
        native = PageOutput(
            page_num=1,
            text="native body",
            status=PageStatus.WARNING,
            engine="native",
            audit_passed=False,
        )
        p.attempts.append(native)
        p.best_output = native
        return _select_page_output_tagged(state, 1)[0]

    base, scan = ship(False), ship(True)
    assert base.status.value == "error", "setup: the D3 conjunction ships ERROR on main"
    assert scan.status == base.status
    assert scan.status.value != "success"
    assert scan.failure_mode == base.failure_mode


def _e2e_corrupt_math(tmp_path, tag, monkeypatch, *, detector):
    """process() a scan page that ALSO trips corrupt-math, recovery enabled.

    ``has_corrupt_math`` is stamped after analysis (a synthetic PDF cannot produce it), so
    the only thing that differs between the runs is whether the scan detector fires.
    """
    orig = UnifiedPipeline._phase_analyze

    def analyze(self, state, *a, **k):
        out = orig(self, state, *a, **k)
        state.pages[1].has_corrupt_math = True
        state.pages[1].needs_ocr_enhancement = True
        return out

    with monkeypatch.context() as m:
        m.setattr(UnifiedPipeline, "_phase_analyze", analyze)
        return _e2e(
            tmp_path, tag, monkeypatch, provider=True, detector=detector, recover_corrupt_math=True
        )


def test_e2e_scan_page_with_corrupt_math_goes_to_whole_page_ocr(tmp_path, monkeypatch) -> None:
    _, eng_off = _e2e_corrupt_math(tmp_path, "m_off", monkeypatch, detector="neutralised")
    _, eng_on = _e2e_corrupt_math(tmp_path, "m_on", monkeypatch, detector="live")

    assert "estimated effect" in _page_text(tmp_path, "m_off"), (
        "main: the hybrid lane owns the page and the old native prose ships"
    )
    assert _OCR_MARK not in _page_text(tmp_path, "m_off")
    assert eng_on.calls == 1
    assert _OCR_MARK in _page_text(tmp_path, "m_on")
    assert "estimated effect" not in _page_text(tmp_path, "m_on")
