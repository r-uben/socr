"""#1100: a page with no text and no ink never reaches a model.

Hermetic: synthetic fitz PDFs, stub provider, no judge. process() pins are
DIFFERENCES (detector live vs always-False in the same process).
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from socr.core.blank_page import blank_page_marker, is_blank_page
from socr.core.config import EngineType, PipelineConfig
from socr.core.result import DocumentStatus
from socr.pipeline.orchestrator import UnifiedPipeline

PROSE = "This paragraph is the only content on a non-blank page."


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


def _write_pages(path: Path, *builders) -> Path:
    doc = fitz.open()
    for build in builders:
        build(doc.new_page())
    doc.save(path)
    doc.close()
    return path


def _blank(page) -> None:
    return None


def _prose(page) -> None:
    page.insert_text((72, 72), PROSE, fontname="helv", fontsize=11)


def _faint_mark(page) -> None:
    """One 2×2 grey pixmap: any non-white pixel is ink (no new threshold)."""
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 2, 2), False)
    pix.set_rect(pix.irect, (200, 200, 200))
    page.insert_image(fitz.Rect(72, 72, 74, 74), pixmap=pix)


def _near_white_revenue(page) -> None:
    """Raster 'Revenue 123.45' in RGB (255, 255, 250) on white. No text layer."""
    src = fitz.open()
    tmp = src.new_page(width=420, height=80)
    tmp.insert_text((10, 50), "Revenue 123.45", fontsize=24, color=(1.0, 1.0, 250 / 255))
    pix = tmp.get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB)
    src.close()
    page.insert_image(fitz.Rect(36, 36, 400, 100), pixmap=pix)


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


def _process(tmp_path, tag, monkeypatch, *, provider, detect, builders):
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch
    from socr.core import blank_page as blank_mod

    engine = _StubEngine()
    pdf = tmp_path / f"{tag}.pdf"
    if not pdf.exists():
        _write_pages(pdf, *builders)
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: engine)
        if not detect:
            m.setattr(blank_mod, "is_blank_page", lambda page: False)
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


def _sidecar(tmp_path, tag, page: int = 1) -> dict:
    return json.loads(_one(tmp_path, tag, f"pages/{page:05d}.json").read_text())


def _page_text(tmp_path, tag, page: int = 1) -> str:
    return _one(tmp_path, tag, f"pages/{page:05d}.md").read_text()


def test_detector_blank_vs_faint_mark(tmp_path: Path) -> None:
    blank = _write_pages(tmp_path / "b.pdf", _blank)
    mark = _write_pages(tmp_path / "m.pdf", _faint_mark)
    with fitz.open(blank) as doc:
        assert is_blank_page(doc[0])
    with fitz.open(mark) as doc:
        assert not is_blank_page(doc[0]), (
            "rule: any native word or colour-channel below 255 at DETECT_DPI is ink"
        )


def test_near_white_coloured_raster_text_is_not_blank(tmp_path: Path) -> None:
    """P0: RGB (255, 255, 250) text is ink. Greyscale collapses it to 255."""
    pdf = _write_pages(tmp_path / "cream.pdf", _near_white_revenue)
    with fitz.open(pdf) as doc:
        assert not doc[0].get_text("words")
        assert not is_blank_page(doc[0]), (
            "rule: any colour-channel below 255 is ink; (255,255,250) is not white"
        )


@pytest.mark.parametrize("provider", [True, False])
def test_blank_page_makes_zero_provider_calls_and_ships_the_note(
    tmp_path, monkeypatch, provider
) -> None:
    result, engine = _process(
        tmp_path, "blank", monkeypatch, provider=provider, detect=True, builders=(_blank,)
    )
    assert engine.calls == 0
    assert _page_text(tmp_path, "blank") == blank_page_marker()
    side = _sidecar(tmp_path, "blank")
    assert side["status"] == "success"
    assert side["engine"] == "blank"
    assert side["failure_mode"] == "none"
    assert side.get("audit_passed") is True
    doc_metas = [
        p for p in (tmp_path / "out-blank").rglob("metadata.json") if p.parent.name != "out-blank"
    ]
    assert len(doc_metas) == 1, doc_metas
    assert json.loads(doc_metas[0].read_text()).get("blank_pages") == [1]
    assert result.status is DocumentStatus.SUCCESS


@pytest.mark.parametrize("provider", [True, False])
def test_faint_mark_is_not_blank(tmp_path, monkeypatch, provider) -> None:
    """A 2×2 grey pixmap is ink (non-white DETECT_DPI pixel): not blank."""
    off, eng_off = _process(
        tmp_path, "mark-off", monkeypatch, provider=provider, detect=False, builders=(_faint_mark,)
    )
    on, eng_on = _process(
        tmp_path, "mark-on", monkeypatch, provider=provider, detect=True, builders=(_faint_mark,)
    )
    assert _page_text(tmp_path, "mark-on") == _page_text(tmp_path, "mark-off")
    assert (_sidecar(tmp_path, "mark-on")["engine"], on.status) == (
        _sidecar(tmp_path, "mark-off")["engine"],
        off.status,
    )
    assert "blank" not in _sidecar(tmp_path, "mark-on")["engine"]


@pytest.mark.parametrize("provider", [True, False])
def test_blank_gate_is_a_difference_on_the_blank_page_only(tmp_path, monkeypatch, provider) -> None:
    """Two-page PDF: only the blank page moves vs the detector-off baseline."""
    builders = (_blank, _prose)
    off, eng_off = _process(
        tmp_path, "mix-off", monkeypatch, provider=provider, detect=False, builders=builders
    )
    on, eng_on = _process(
        tmp_path, "mix-on", monkeypatch, provider=provider, detect=True, builders=builders
    )
    assert _page_text(tmp_path, "mix-on", 2) == _page_text(tmp_path, "mix-off", 2)
    assert _page_text(tmp_path, "mix-on", 1) == blank_page_marker()
    assert _page_text(tmp_path, "mix-off", 1) != blank_page_marker()
    assert eng_on.calls < eng_off.calls or (eng_on.calls == 0 and provider is False)
    assert _sidecar(tmp_path, "mix-on", 1)["engine"] == "blank"
    assert _sidecar(tmp_path, "mix-off", 1)["engine"] != "blank"


def test_blank_page_is_terminal_on_resume(tmp_path, monkeypatch) -> None:
    loaded: list = []
    real = UnifiedPipeline._load_terminal_page

    def spy(self, *a, **k):
        out = real(self, *a, **k)
        loaded.append(out)
        return out

    monkeypatch.setattr(UnifiedPipeline, "_resume_skip", lambda self, *a, **k: None)
    monkeypatch.setattr(UnifiedPipeline, "_load_terminal_page", spy)
    _process(tmp_path, "r", monkeypatch, provider=True, detect=True, builders=(_blank,))
    first = _sidecar(tmp_path, "r")
    loaded.clear()
    result, engine = _process(
        tmp_path, "r", monkeypatch, provider=True, detect=True, builders=(_blank,)
    )
    assert loaded and any(o is not None for o in loaded)
    assert engine.calls == 0
    second = _sidecar(tmp_path, "r")
    assert second["engine"] == first["engine"] == "blank"
    assert second["status"] == "success"
    assert result.status is DocumentStatus.SUCCESS


def test_blank_page_event_is_in_the_resume_allowlist() -> None:
    assert "blank_page" in UnifiedPipeline.resume_restore_kinds()


def test_blank_page_event_survives_resume(tmp_path, monkeypatch) -> None:
    """The page-loop event is replayed; a membership pin alone would miss a filter drop."""
    from socr.core.document import DocumentHandle
    from socr.core.result import PageOutput, PageStatus
    from socr.core.state import DocumentState

    _process(tmp_path, "ev", monkeypatch, provider=True, detect=True, builders=(_blank,))
    side = _sidecar(tmp_path, "ev")
    assert any(e.get("kind") == "blank_page" for e in side.get("audit_events", [])), side

    pdf = tmp_path / "ev.pdf"
    out_dir = tmp_path / "out-ev"
    pipe = UnifiedPipeline(
        PipelineConfig(agentic=True, quiet=True, write_manifest=False, judge_backend="heuristic")
    )
    pipe._scan_root = pdf.parent
    resumed = DocumentState(handle=DocumentHandle.from_path(pdf))
    pipe._restore_terminal_page_state(
        resumed,
        1,
        PageOutput(
            page_num=1,
            text=blank_page_marker(),
            status=PageStatus.SUCCESS,
            engine="blank",
            audit_passed=True,
        ),
        out_dir,
    )
    assert any(e.kind == "blank_page" for e in resumed.events), [e.kind for e in resumed.events]


def test_halted_blank_page_is_not_reported_as_shipped(tmp_path, monkeypatch) -> None:
    """A blank page past a cascade halt ships a failure marker, not the blank note."""
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.agentic import PageDecision, ProviderAttempt
    from socr.pipeline import orchestrator as orch

    # p1 has ink but no text layer so it is an OCR page (timeout → halt).
    # p2 is blank: detected before the loop, never shipped if the halt wins.
    pdf = _write_pages(tmp_path / "halt.pdf", _faint_mark, _blank)
    captured: dict = {}
    real_assemble = UnifiedPipeline._phase_assemble

    def _spy_assemble(self, state, output_dir):
        captured["state"] = state
        return real_assemble(self, state, output_dir)

    def _timeout(page_num, ladder, run_provider, judge, **kwargs):
        out = PageOutput(
            page_num=page_num, text="", status=PageStatus.ERROR, engine="qwen", audit_passed=False
        )
        prof = ladder[0]
        att = ProviderAttempt(
            engine=prof.engine,
            output=out,
            cost_usd=0.0,
            accepted=False,
            reason="provider timeout",
            provider_id=prof.id,
            model=prof.model,
            backend=prof.backend,
        )
        return PageDecision(page_num=page_num, final_output=out, attempts=[att])

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
    monkeypatch.setattr(UnifiedPipeline, "_phase_assemble", _spy_assemble)
    monkeypatch.setattr(orch, "route_page", _timeout)
    monkeypatch.setattr(orch, "probe_ollama_idle", lambda *a, **k: False)
    result = pipe.process(pdf, output_dir=tmp_path / "out-halt")

    assert "PARTIAL_SAVE_VLM_TIMEOUT" in (result.error or "")
    state = captured["state"]
    # Detection-time flag is still set; the shipped body is not the note.
    assert state.pages[2].is_blank is True
    assert state.pages[2].not_processed_after_halt is True
    assert UnifiedPipeline._shipped_blank_pages(state) == []
    note = blank_page_marker()
    bodies = [
        getattr(p, "text", "") or ""
        for p in (result.pages or [])
        if getattr(p, "page_num", None) == 2
    ]
    for frag in (tmp_path / "out-halt").rglob("*.md"):
        bodies.append(frag.read_text())
    assert all(note not in body for body in bodies), bodies
    doc_metas = [
        p for p in (tmp_path / "out-halt").rglob("metadata.json") if p.parent.name != "out-halt"
    ]
    assert len(doc_metas) == 1, doc_metas
    meta = json.loads(doc_metas[0].read_text())
    assert 2 not in (meta.get("blank_pages") or [])
