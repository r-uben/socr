"""GH-994: a table detection missed, flattened to prose, must not ship clean SUCCESS.

Detector: ``has_tables`` False AND a "Table N" caption line AND (>= ``MIN_TABLE_ROWS``
horizontal rules sharing an x-extent OR ``has_recurring_numeric_columns`` OR the GH-64 flag).
Consequence: WARNING + ``FailureMode.TABLE_NOT_RECONSTRUCTED``, text unchanged, no re-routing.

Hermetic: synthetic fitz PDFs, stub provider, no judge. process() pins are DIFFERENCES
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
from socr.core.result import DocumentStatus, FailureMode
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.reconstruct import MIN_TABLE_ROWS

_PROSE = (
    "The estimated effect of the policy change on output is reported below for each "
    "specification, with robust standard errors clustered by region and year. "
)
_CAPTION = "Table 3: Effect of the policy change on output"
# Two-word labels: keeps the page under the GH-64 single-token-line shape, so the numeric-column
# shape below is exercised by ``has_recurring_numeric_columns`` and not by the GH-64 flag.
_LABELS = [f"{w} model" for w in ("Baseline", "Treated", "Control", "Placebo", "Lagged", "Pooled")]


def _make(path: Path, *, caption: bool, rules: int = 0, numeric: bool = False) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    y = 72.0
    for _ in range(4):
        page.insert_text((72, y), _PROSE, fontname="helv", fontsize=10)
        y += 14
    y += 10
    if caption:
        page.insert_text((72, y), _CAPTION, fontname="helv", fontsize=10)
        y += 20
    rule_ys = []
    for i, label in enumerate(_LABELS):
        rule_ys.append(y - 9)
        page.insert_text((72, y), label, fontname="helv", fontsize=10)
        if numeric:
            page.insert_text((300, y), f"{0.1 * (i + 1):.3f}", fontname="helv", fontsize=10)
            page.insert_text((400, y), f"{1.7 + 0.3 * i:.3f}", fontname="helv", fontsize=10)
        y += 16
    for ry in rule_ys[:rules]:
        page.draw_line((72, ry), (480, ry), width=0.5)
    doc.save(path)
    doc.close()
    return path


SHAPES = {
    "caption_plus_rules": dict(caption=True, rules=MIN_TABLE_ROWS, numeric=False, fires=True),
    "caption_plus_numeric_columns": dict(caption=True, rules=0, numeric=True, fires=True),
    "caption_no_structure": dict(caption=True, rules=0, numeric=False, fires=False),
    "caption_too_few_rules": dict(
        caption=True, rules=MIN_TABLE_ROWS - 1, numeric=False, fires=False
    ),
    "rules_no_caption": dict(caption=False, rules=MIN_TABLE_ROWS, numeric=False, fires=False),
    "numeric_columns_no_caption": dict(caption=False, rules=0, numeric=True, fires=False),
}


def _spec(name):
    s = dict(SHAPES[name])
    return s.pop("fires"), s


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


def _analyze(pdf: Path):
    pipe = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.DEEPSEEK,
            enabled_engines=[EngineType.GEMINI],
            agentic=True,
            quiet=True,
            native_first=True,
        )
    )
    state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    pipe._phase_analyze(state)
    return state


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_detector_shapes(tmp_path: Path, name: str) -> None:
    fires, kw = _spec(name)
    state = _analyze(_make(tmp_path / "p.pdf", **kw))
    ps = state.pages[1]
    assert ps.is_born_digital
    assert not ps.has_tables, "setup: detection must have missed the table"
    assert ps.table_not_reconstructed is fires
    events = [e for e in state.events if e.kind == "table_not_reconstructed"]
    assert [e.page_num for e in events] == ([1] if fires else [])


def test_has_tables_true_stays_quiet(tmp_path: Path, monkeypatch) -> None:
    """Same caption-plus-rules page, but detection reports a table: no fire."""
    _, kw = _spec("caption_plus_rules")
    pdf = _make(tmp_path / "p.pdf", **kw)
    with monkeypatch.context() as m:
        m.setattr(BornDigitalDetector, "_detect_tables", lambda self, page: True)
        state = _analyze(pdf)
    assert state.pages[1].has_tables
    assert state.pages[1].table_not_reconstructed is False
    assert not [e for e in state.events if e.kind == "table_not_reconstructed"]


def test_gh64_flag_with_caption_fires(tmp_path: Path, monkeypatch) -> None:
    """The GH-64 flag is folded in: with a caption it now demotes; without, it stays quiet."""
    _, kw = _spec("caption_no_structure")
    pdf = _make(tmp_path / "p.pdf", **kw)
    no_cap = _make(tmp_path / "q.pdf", caption=False)
    with monkeypatch.context() as m:
        m.setattr(BornDigitalDetector, "_detect_columnar_numbers", staticmethod(lambda page: True))
        assert _analyze(pdf).pages[1].table_not_reconstructed is True
        assert _analyze(no_cap).pages[1].table_not_reconstructed is False


def test_event_is_recomputed_not_replayed() -> None:
    assert "table_not_reconstructed" not in UnifiedPipeline.resume_restore_kinds()


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


def _chart_pdf(path: Path) -> Path:
    """Prose plus a large embedded raster: ``has_chart_marks`` fires and the chart lane owns it."""
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for _ in range(8):
        page.insert_text((72, y), _PROSE, fontname="helv", fontsize=10)
        y += 14
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 300, 300), False)
    pix.set_rect(pix.irect, (235, 235, 235))
    page.insert_image(fitz.Rect(72, y + 20, 372, y + 320), pixmap=pix)
    doc.save(path)
    doc.close()
    return path


def _process(
    tmp_path, tag, monkeypatch, *, provider, shape, neutralised, native_only=False, forced=False
):
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch

    engine = _StubEngine()
    _, kw = _spec(shape) if shape != "chart" else (None, {})
    pdf = tmp_path / f"{tag}.pdf"
    if not pdf.exists():
        _chart_pdf(pdf) if shape == "chart" else _make(pdf, **kw)
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: engine)
        if neutralised:
            m.setattr(BornDigitalDetector, "_detect_flattened_table", classmethod(lambda *a: False))
        if forced:
            m.setattr(BornDigitalDetector, "_detect_flattened_table", classmethod(lambda *a: True))
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                native_first=True,
                native_only=native_only,
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


@pytest.mark.parametrize("provider", [True, False])
@pytest.mark.parametrize("shape", ["caption_plus_rules", "caption_plus_numeric_columns"])
def test_e2e_fire_demotes_page_and_document_text_unchanged(
    tmp_path, monkeypatch, provider, shape
) -> None:
    kw = dict(provider=provider, shape=shape)
    off, eng_off = _process(tmp_path, "off", monkeypatch, neutralised=True, **kw)
    on, eng_on = _process(tmp_path, "on", monkeypatch, neutralised=False, **kw)
    side_off, side_on = _sidecar(tmp_path, "off"), _sidecar(tmp_path, "on")
    assert side_off["failure_mode"] != FailureMode.TABLE_NOT_RECONSTRUCTED.value
    assert side_off["status"] == "success" and off.status is DocumentStatus.SUCCESS
    assert side_on["status"] == "warning"
    assert side_on["failure_mode"] == FailureMode.TABLE_NOT_RECONSTRUCTED.value
    assert on.status is not DocumentStatus.SUCCESS
    # No re-routing: identical engine calls, identical text.
    assert eng_on.calls == eng_off.calls
    assert _page_text(tmp_path, "on") == _page_text(tmp_path, "off")
    assert "Baseline" in _page_text(tmp_path, "on")


@pytest.mark.parametrize("provider", [True, False])
def test_e2e_quiet_shape_is_identical_with_and_without_detector(
    tmp_path, monkeypatch, provider
) -> None:
    kw = dict(provider=provider, shape="caption_no_structure")
    off, _ = _process(tmp_path, "qoff", monkeypatch, neutralised=True, **kw)
    on, _ = _process(tmp_path, "qon", monkeypatch, neutralised=False, **kw)
    a, b = _sidecar(tmp_path, "qoff"), _sidecar(tmp_path, "qon")
    assert (a["status"], a["failure_mode"]) == (b["status"], b["failure_mode"])
    assert on.status is off.status
    assert _page_text(tmp_path, "qon") == _page_text(tmp_path, "qoff")


def test_e2e_native_only_demotes_too(tmp_path, monkeypatch) -> None:
    kw = dict(provider=True, shape="caption_plus_rules", native_only=True)
    _process(tmp_path, "noff", monkeypatch, neutralised=True, **kw)
    on, _ = _process(tmp_path, "non", monkeypatch, neutralised=False, **kw)
    assert _sidecar(tmp_path, "noff")["status"] == "success"
    assert _sidecar(tmp_path, "non")["status"] == "warning"
    assert on.status is not DocumentStatus.SUCCESS


def test_resume_does_not_restore_a_cached_success_for_a_flagged_page(tmp_path, monkeypatch) -> None:
    """A SUCCESS sidecar written with the detector off is not reused once it is live.

    The document-level gate is bypassed (it keys on a source-digest fingerprint a monkeypatch
    cannot move); the per-page ledger gate under test is the same either way.
    """
    kw = dict(provider=True, shape="caption_plus_rules")
    monkeypatch.setattr(UnifiedPipeline, "_resume_skip", lambda self, *a, **k: None)
    loaded: list = []
    real = UnifiedPipeline._load_terminal_page

    def spy(self, *a, **k):
        out = real(self, *a, **k)
        loaded.append(out)
        return out

    monkeypatch.setattr(UnifiedPipeline, "_load_terminal_page", spy)
    _process(tmp_path, "r", monkeypatch, neutralised=True, **kw)
    assert _sidecar(tmp_path, "r")["status"] == "success"

    # Control: detector still off, the finished page IS restored from the ledger.
    loaded.clear()
    _process(tmp_path, "r", monkeypatch, neutralised=True, **kw)
    assert any(o is not None for o in loaded), "setup: the ledger gate must restore a clean page"

    # Detector live: the same cached SUCCESS page is refused and reprocessed.
    loaded.clear()
    result, _ = _process(tmp_path, "r", monkeypatch, neutralised=False, **kw)
    assert loaded and all(o is None for o in loaded)
    side = _sidecar(tmp_path, "r")
    assert side["status"] == "warning"
    assert side["failure_mode"] == FailureMode.TABLE_NOT_RECONSTRUCTED.value
    assert result.status is not DocumentStatus.SUCCESS


def test_chart_asset_lane_demotes_but_keeps_body_and_audit_passed(tmp_path, monkeypatch) -> None:
    """The chart lane ships retained native prose; a flagged page must not bypass the demotion.

    The synthetic chart page has no caption, so the detector is forced for the live arm; the
    test is about the lane, not the detector.
    """
    kw = dict(provider=True, shape="chart", native_only=True)
    off, _ = _process(tmp_path, "coff", monkeypatch, neutralised=True, **kw)
    on, _ = _process(tmp_path, "con", monkeypatch, neutralised=False, forced=True, **kw)
    a, b = _sidecar(tmp_path, "coff"), _sidecar(tmp_path, "con")
    assert a["engine"] == "chart_asset", "setup: the page must take the chart lane"
    assert a["status"] == "success" and off.status is DocumentStatus.SUCCESS
    assert b["engine"] == "chart_asset"
    assert b["status"] == "warning"
    assert b["failure_mode"] == FailureMode.TABLE_NOT_RECONSTRUCTED.value
    assert b["winning_output"]["audit_passed"] is a["winning_output"]["audit_passed"] is True
    assert _page_text(tmp_path, "con") == _page_text(tmp_path, "coff")
    assert "estimated effect" in _page_text(tmp_path, "con")
    assert on.status is not DocumentStatus.SUCCESS


def _select(tmp_path, *, engine: str, flagged: bool):
    from socr.core.manifest import _select_page_output_tagged
    from socr.core.result import PageOutput, PageStatus

    pdf = tmp_path / f"sel-{engine}-{flagged}.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((54, 72), "text layer long enough to count as native here.")
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    p = state.pages[1]
    p.is_born_digital = True
    p.native_text = "native body"
    p.table_not_reconstructed = flagged
    # Not a prefix extension of ``native_text``: a re-derived fallback would drop it.
    out = PageOutput(
        page_num=1,
        text="rewritten native body\n\n$$x = 1$$",
        status=PageStatus.SUCCESS,
        engine=engine,
        audit_passed=True,
    )
    p.attempts.append(out)
    p.best_output = out
    return _select_page_output_tagged(state, 1)[0]


@pytest.mark.parametrize("engine", ["native", "native+equations", "chart_asset"])
def test_selected_native_winner_keeps_exact_bytes_only_status_differs(tmp_path, engine) -> None:
    base = _select(tmp_path, engine=engine, flagged=False)
    flag = _select(tmp_path, engine=engine, flagged=True)
    assert base.text == flag.text == "rewritten native body\n\n$$x = 1$$"
    assert base.status.value == "success" and flag.status.value == "warning"
    assert flag.failure_mode == FailureMode.TABLE_NOT_RECONSTRUCTED
    assert flag.audit_passed is base.audit_passed is True
    assert flag.engine == base.engine == engine


def test_selected_model_winner_is_not_demoted(tmp_path) -> None:
    base = _select(tmp_path, engine="qwen", flagged=False)
    flag = _select(tmp_path, engine="qwen", flagged=True)
    assert (flag.status, flag.failure_mode, flag.text) == (
        base.status,
        base.failure_mode,
        base.text,
    )


def test_resume_restores_a_cached_model_winner_on_a_flagged_page(tmp_path, monkeypatch) -> None:
    """The refusal is for cached NATIVE/chart text only; a model reading is not suspect."""
    kw = dict(provider=True, shape="caption_plus_rules")
    monkeypatch.setattr(UnifiedPipeline, "_resume_skip", lambda self, *a, **k: None)
    _process(tmp_path, "m", monkeypatch, neutralised=True, **kw)
    sidecar = next((tmp_path / "out-m").rglob("pages/00001.json"))
    meta = json.loads(sidecar.read_text())
    meta["winning_output"]["engine"] = "qwen"
    sidecar.write_text(json.dumps(meta))
    loaded: list = []
    real = UnifiedPipeline._load_terminal_page

    def spy(self, *a, **k):
        out = real(self, *a, **k)
        loaded.append(out)
        return out

    monkeypatch.setattr(UnifiedPipeline, "_load_terminal_page", spy)
    _process(tmp_path, "m", monkeypatch, neutralised=False, **kw)
    assert loaded and all(o is not None for o in loaded), "model winner must be restored"


def test_flagged_page_with_no_winner_and_no_attempts_is_demoted_not_clean(tmp_path) -> None:
    """The synthetic native fallback: nothing selected, nothing attempted, no other defect."""
    from socr.core.manifest import SelectionProvenance, _select_page_output_tagged

    def ship(flagged: bool):
        pdf = tmp_path / f"fb-{flagged}.pdf"
        doc = fitz.open()
        doc.new_page().insert_text((54, 72), "text layer long enough to count as native here.")
        doc.save(pdf)
        doc.close()
        state = DocumentState(handle=DocumentHandle.from_path(pdf))
        p = state.pages[1]
        p.is_born_digital = True
        p.native_text = "native body"
        p.table_not_reconstructed = flagged
        assert p.best_output is None and not p.attempts
        return _select_page_output_tagged(state, 1)

    (base, base_tag), (flag, flag_tag) = ship(False), ship(True)
    assert (base.status.value, base.failure_mode) == ("success", FailureMode.NONE)
    assert base_tag is SelectionProvenance.NATIVE_CLEAN
    assert flag.text == base.text == "native body"
    assert flag.status.value == "warning"
    assert flag.failure_mode == FailureMode.TABLE_NOT_RECONSTRUCTED
    assert flag_tag is SelectionProvenance.NATIVE_FALLBACK


def test_e2e_flagged_page_with_no_winner_makes_the_document_not_success(
    tmp_path, monkeypatch
) -> None:
    """Assemble sees a flagged page with no best_output and no attempts: page WARNING, document not SUCCESS."""
    orig = UnifiedPipeline._phase_assemble

    def assemble(self, state, output_dir):
        for ps in state.pages.values():
            ps.best_output = None
            ps.attempts.clear()
        return orig(self, state, output_dir)

    kw = dict(provider=True, shape="caption_plus_rules")
    with monkeypatch.context() as m:
        m.setattr(UnifiedPipeline, "_phase_assemble", assemble)
        off, _ = _process(tmp_path, "woff", monkeypatch, neutralised=True, **kw)
        on, _ = _process(tmp_path, "won", monkeypatch, neutralised=False, **kw)
    assert _sidecar(tmp_path, "woff")["status"] == "success"
    assert off.status is DocumentStatus.SUCCESS, "setup: unflagged no-winner page is clean"
    assert _sidecar(tmp_path, "won")["status"] == "warning"
    assert on.status is not DocumentStatus.SUCCESS


@pytest.fixture(autouse=True)
def _whole_page_chart_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """#1053: this file pins the chart lane's whole-page route, not the per-figure crop route.

    Its raster-figure fixture would otherwise take the crop route (and ship WARNING for the
    unread figure words), which ``tests/test_1053_figure_crops.py`` owns.
    """
    from socr.figures import figure_crops

    monkeypatch.setattr(figure_crops, "plan_figure_page", lambda page: None)
