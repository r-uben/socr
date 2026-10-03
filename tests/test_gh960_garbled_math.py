"""#960: a page whose native text layer garbled its mathematics must not ship as trusted native.

Four signals, measured on the trusted-native population: private-use glyphs, math-alphanumeric
codepoints, letters of a script the corpus is not written in (Cambria math extracting as
Syriac/Tamil), and spans in a math font ``_MATH_FONT_RE`` does not list. Any hit routes the
page to a whole-page OCR read (the P4-R region lane keeps the garbled native text as its floor)
and off the corrupt-math hybrid lane.

Hermetic: synthetic fitz PDFs, no provider, no judge. Routing pins are DIFFERENCES (the
detector neutralised vs live in the same process), never an absolute outcome (CLAUDE.md, #257).
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from socr.core import born_digital
from socr.core.born_digital import (
    MISDECODED_MATH_SCRIPTS,
    GarbledMathSignals,
    detect_garbled_math,
)
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import DocumentStatus, FailureMode
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline

_PROSE = (
    "The estimated effect of the policy change on output is reported below for each "
    "specification, with robust standard errors clustered by region and year. "
)

#: Lines written with a TextWriter: PyMuPDF falls back to its bundled Noto fonts for glyphs
#: Helvetica lacks, so a math-italic letter lands in a "Noto Sans Math" span (a math font
#: ``_MATH_FONT_RE`` does not list) and Syriac letters extract as Syriac.
HITS = {
    "unlisted_math_font": "the coefficient \U0001d465 is reported here",
    "misdecoded_script": "the coefficient ܐܒ is reported here",
}
CONTROL = "the coefficient x is reported here"


def _page_pdf(path: Path, line: str, *, chart: bool = False) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for _ in range(8):
        page.insert_text((72, y), _PROSE, fontname="helv", fontsize=10)
        y += 14
    tw = fitz.TextWriter(page.rect)
    tw.append((72, y + 10), line, fontsize=10)
    tw.write_text(page)
    if chart:
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 300, 300), False)
        pix.set_rect(pix.irect, (200, 30, 30))
        page.insert_image(fitz.Rect(72, y + 30, 372, y + 330), pixmap=pix)
    doc.save(path)
    doc.close()
    return path


def _signals(pdf: Path) -> GarbledMathSignals:
    with fitz.open(pdf) as doc:
        page = doc[0]
        return detect_garbled_math(page, page.get_text("text"))


# ---------------------------------------------------------------------------
# The detector itself.
# ---------------------------------------------------------------------------


def test_setup_fonts_really_produce_the_signals(tmp_path: Path) -> None:
    """Canary: if PyMuPDF's fallback fonts changed, every page-level pin below is vacuous."""
    with fitz.open(_page_pdf(tmp_path / "m.pdf", HITS["unlisted_math_font"])) as doc:
        fonts = {f[3] for f in doc[0].get_fonts()}
    assert any("Math" in f for f in fonts), fonts
    with fitz.open(_page_pdf(tmp_path / "s.pdf", HITS["misdecoded_script"])) as doc:
        assert "ܐ" in doc[0].get_text("text")


def test_unlisted_math_font_fires(tmp_path: Path) -> None:
    sig = _signals(_page_pdf(tmp_path / "p.pdf", HITS["unlisted_math_font"]))
    assert sig.unlisted_math_font_chars > 0 and sig.fired


def test_misdecoded_script_fires(tmp_path: Path) -> None:
    sig = _signals(_page_pdf(tmp_path / "p.pdf", HITS["misdecoded_script"]))
    assert sig.misdecoded_script_letters == 2 and sig.fired


def test_control_page_is_quiet(tmp_path: Path) -> None:
    sig = _signals(_page_pdf(tmp_path / "p.pdf", CONTROL))
    assert not sig.fired and sig.nonzero() == {}


class _NoSpans:
    def get_text(self, *a, **k):
        return {"blocks": []}


@pytest.mark.parametrize(
    ("text", "field"),
    [
        ("a  b", "private_use"),
        ("a \U000f0001 b", "private_use"),
        ("a \U0001d465 b", "math_alphanumeric"),
        ("a க b", "misdecoded_script_letters"),  # Tamil
        ("a ܐ b", "misdecoded_script_letters"),  # Syriac
    ],
)
def test_text_signals(text: str, field: str) -> None:
    sig = detect_garbled_math(_NoSpans(), text)
    assert sig.nonzero() == {field: 1}


@pytest.mark.parametrize(
    "text",
    [
        "plain ascii 0.47 and -1.2",
        "Greek αβσ is printed by the corpus",
        "Cyrillic ж and CJK 中 too",
        "accented Holmström and Gürkaynak",
        "a script-L symbol ℒ and aleph ℵ",  # letter-like symbols, not a script
        "© 2023 and − 1",
    ],
)
def test_text_signals_quiet_on_legitimate_text(text: str) -> None:
    assert not detect_garbled_math(_NoSpans(), text).fired


@pytest.mark.parametrize(
    ("font", "fires"),
    [
        ("UniMath-Regular", True),
        ("ABCDEF+MTMI", True),
        ("RMTMI", True),
        ("MTSYN", True),
        ("MnSymbol10", True),
        ("Cambria Math", True),
        ("Fourier-Math-Symbols", True),
        ("CMMI10", False),  # listed: the P4-R region lane owns it
        ("ABCDEF+CambriaMath", False),
        ("STIXMath-Regular", False),
        ("TimesNewRomanPSMT", False),
        ("ArialMT", False),
        ("Helvetica", False),
    ],
)
def test_math_font_names(font: str, fires: bool) -> None:
    class _Page:
        def get_text(self, *a, **k):
            return {"blocks": [{"lines": [{"spans": [{"font": font, "text": "xy"}]}]}]}

    assert (detect_garbled_math(_Page(), "").unlisted_math_font_chars > 0) is fires


def test_script_set_excludes_scripts_the_corpus_prints() -> None:
    assert not MISDECODED_MATH_SCRIPTS & {"LATIN", "GREEK", "CYRILLIC", "CJK", "MODIFIER"}
    assert {"SYRIAC", "TAMIL"} <= MISDECODED_MATH_SCRIPTS


def test_loaded_source_is_this_checkout() -> None:
    import socr

    src = Path(__file__).resolve().parents[1] / "src"
    assert Path(socr.__file__).resolve().is_relative_to(src), socr.__file__


# ---------------------------------------------------------------------------
# Routing: differences against the neutralised detector.
# ---------------------------------------------------------------------------


def _neutral(page, text):
    return GarbledMathSignals()


def _boom(page, text):
    raise RuntimeError("detector exploded")


def _pipeline() -> UnifiedPipeline:
    return UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.DEEPSEEK,
            enabled_engines=[EngineType.GEMINI],
            agentic=True,
            quiet=True,
            native_first=True,
        )
    )


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    from socr.core.providers import PROFILE_QWEN_LOCAL

    monkeypatch.setattr(
        UnifiedPipeline, "_available_engines_for_agentic", lambda self: [PROFILE_QWEN_LOCAL]
    )
    monkeypatch.setattr(UnifiedPipeline, "_resolve_judge_model", lambda self, *a, **kw: "")


def _analyze(pdf: Path, monkeypatch, detector=None):
    with monkeypatch.context() as m:
        if detector is not None:
            m.setattr(born_digital, "detect_garbled_math", detector)
        pipe = _pipeline()
        state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
        pipe._phase_analyze(state)
    return pipe, state


@pytest.mark.parametrize("name", sorted(HITS) + ["control"])
def test_routing_difference_pin(tmp_path: Path, name: str, monkeypatch) -> None:
    pdf = _page_pdf(tmp_path / "p.pdf", HITS.get(name, CONTROL))
    pipe_off, state_off = _analyze(pdf, monkeypatch, _neutral)
    pipe_on, state_on = _analyze(pdf, monkeypatch)
    trusted_off = pipe_off._is_agentic_trusted_native(1, state_off.pages[1])
    trusted_on = pipe_on._is_agentic_trusted_native(1, state_on.pages[1])
    events = [e for e in state_on.events if e.kind == "garbled_math_native"]

    assert "garbled_math_native" not in [e.kind for e in state_off.events]
    if name in HITS:
        assert trusted_off, "baseline (main behaviour): the page ships trusted native"
        assert not trusted_on
        assert state_on.pages[1].needs_ocr_enhancement
        assert [e.page_num for e in events] == [1]
        assert events[0].data["signals"] == state_on.pages[1].garbled_math_signals
    else:
        assert trusted_on == trusted_off
        assert events == []


def test_event_is_recomputed_not_replayed() -> None:
    assert "garbled_math_native" not in UnifiedPipeline.resume_restore_kinds()


def test_event_says_routed_or_retained_and_flags_errors(tmp_path, monkeypatch) -> None:
    pdf = _page_pdf(tmp_path / "p.pdf", HITS["misdecoded_script"])

    def _events(native_only: bool, detector=None):
        with monkeypatch.context() as m:
            if detector is not None:
                m.setattr(born_digital, "detect_garbled_math", detector)
            pipe = _pipeline()
            pipe.config.native_only = native_only
            state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
            pipe._phase_analyze(state)
        return [e for e in state.events if e.kind == "garbled_math_native"]

    routed, retained = _events(False)[0], _events(True)[0]
    assert "routed to OCR" in routed.detail and routed.data["error"] is False
    assert "RETAINED" in retained.detail and "routed to OCR" not in retained.detail
    failed = _events(False, _boom)[0]
    assert failed.data["error"] is True and "FAILED" in failed.detail


def test_garbled_page_leaves_the_corrupt_math_hybrid_lane(tmp_path, monkeypatch) -> None:
    """The hybrid keeps the garbled native prose; a flagged page goes whole-page instead."""
    pdf = _page_pdf(tmp_path / "p.pdf", HITS["misdecoded_script"])
    pipe_off, state_off = _analyze(pdf, monkeypatch, _neutral)
    pipe_on, state_on = _analyze(pdf, monkeypatch)
    ps_off, ps_on = state_off.pages[1], state_on.pages[1]
    for pipe, ps in ((pipe_off, ps_off), (pipe_on, ps_on)):
        ps.has_corrupt_math = True
        pipe.config.recover_corrupt_math = True
    assert pipe_off._is_corrupt_math_recovery_page(1, ps_off), "main: the hybrid lane owns it"
    assert not pipe_on._is_corrupt_math_recovery_page(1, ps_on)
    ps_off.garbled_math_scan_failed = True  # unknown is treated as a hit
    assert not pipe_off._is_corrupt_math_recovery_page(1, ps_off)


# ---------------------------------------------------------------------------
# End-to-end through process().
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


def _e2e(
    tmp_path: Path,
    tag: str,
    monkeypatch,
    *,
    provider: bool,
    native_only: bool = False,
    detector: str = "live",
    chart: bool = False,
    reprocess: bool = False,
    recover_corrupt_math: bool = False,
    line: str = HITS["misdecoded_script"],
):
    """Run process() once. ``detector``: live | neutralised | raising."""
    modes = {"live": None, "neutralised": _neutral, "raising": _boom}
    if detector not in modes:
        raise ValueError(f"unknown detector mode {detector!r}")
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch

    engine = _StubEngine()
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: engine)
        if modes[detector] is not None:
            m.setattr(born_digital, "detect_garbled_math", modes[detector])
        pdf = tmp_path / f"{tag}.pdf"
        if not pdf.exists():  # a second run on the same tag must see the same bytes
            _page_pdf(pdf, line, chart=chart)
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                native_first=True,
                native_only=native_only,
                reprocess=reprocess,
                write_manifest=False,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                recover_corrupt_math=recover_corrupt_math,
                save_figures=False,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL] if provider else []
        pipe._build_page_judge = lambda state: _AcceptingJudge()
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        result = pipe.process(pdf, output_dir=tmp_path / f"out-{tag}")
    return result, engine


def _sidecar(tmp_path: Path, tag: str) -> dict:
    found = sorted((tmp_path / f"out-{tag}").rglob("pages/00001.json"))
    assert len(found) == 1, found
    return json.loads(found[0].read_text())


def _page_text(tmp_path: Path, tag: str) -> str:
    found = sorted((tmp_path / f"out-{tag}").rglob("pages/00001.md"))
    assert len(found) == 1, found
    return found[0].read_text()


def _audit_kinds(tmp_path: Path, tag: str) -> set[str]:
    text = "".join(f.read_text() for f in sorted((tmp_path / f"out-{tag}").rglob("*audit*.json")))
    events = json.loads(text)["events"] if text.strip().startswith("{") else []
    return {e["kind"] for e in events}


@pytest.mark.parametrize("name", sorted(HITS))
def test_e2e_provider_present_text_comes_from_the_provider(tmp_path, monkeypatch, name) -> None:
    line = HITS[name]
    _, eng_off = _e2e(
        tmp_path, "a_off", monkeypatch, provider=True, detector="neutralised", line=line
    )
    _, eng_on = _e2e(tmp_path, "a_on", monkeypatch, provider=True, line=line)

    assert eng_off.calls == 0 and "estimated effect" in _page_text(tmp_path, "a_off")
    assert eng_on.calls == 1
    assert _OCR_MARK in _page_text(tmp_path, "a_on")
    assert _sidecar(tmp_path, "a_on")["engine"] != "native"


def test_e2e_control_page_is_unchanged(tmp_path, monkeypatch) -> None:
    _, eng_off = _e2e(
        tmp_path, "k_off", monkeypatch, provider=True, detector="neutralised", line=CONTROL
    )
    _, eng_on = _e2e(tmp_path, "k_on", monkeypatch, provider=True, line=CONTROL)
    assert eng_on.calls == eng_off.calls
    assert _page_text(tmp_path, "k_on") == _page_text(tmp_path, "k_off")
    assert _sidecar(tmp_path, "k_on")["status"] == _sidecar(tmp_path, "k_off")["status"]


def test_e2e_never_trusted_success_without_a_model_read(tmp_path, monkeypatch) -> None:
    """No provider: the page ships its native text demoted. Pinned as a difference."""
    off, _ = _e2e(tmp_path, "b_off", monkeypatch, provider=False, detector="neutralised")
    on, _ = _e2e(tmp_path, "b_on", monkeypatch, provider=False)
    side_off, side_on = _sidecar(tmp_path, "b_off"), _sidecar(tmp_path, "b_on")

    assert side_off["status"] == "success" and off.status is DocumentStatus.SUCCESS
    assert side_on["status"] != "success"
    assert side_on["failure_mode"] == FailureMode.NATIVE_GARBLED_MATH.value
    assert on.status is not DocumentStatus.SUCCESS
    assert "estimated effect" in _page_text(tmp_path, "b_on"), "no content dropped"


def test_e2e_native_only_retains_the_page_but_demotes_it(tmp_path, monkeypatch) -> None:
    off, _ = _e2e(
        tmp_path, "c_off", monkeypatch, provider=True, native_only=True, detector="neutralised"
    )
    on, eng_on = _e2e(tmp_path, "c_on", monkeypatch, provider=True, native_only=True)
    side_off, side_on = _sidecar(tmp_path, "c_off"), _sidecar(tmp_path, "c_on")

    assert side_off["status"] == "success" and off.status is DocumentStatus.SUCCESS
    assert eng_on.calls == 0
    assert side_on["status"] == "warning"
    assert side_on["failure_mode"] == FailureMode.NATIVE_GARBLED_MATH.value
    assert on.status is not DocumentStatus.SUCCESS
    assert "estimated effect" in _page_text(tmp_path, "c_on")
    assert "native_garbled_math_retained" in _audit_kinds(tmp_path, "c_on")
    assert "native_garbled_math_retained" not in _audit_kinds(tmp_path, "c_off")


def test_e2e_detector_exception_fails_closed(tmp_path, monkeypatch) -> None:
    _, eng_off = _e2e(tmp_path, "d_off", monkeypatch, provider=True, detector="neutralised")
    _, eng_on = _e2e(tmp_path, "d_on", monkeypatch, provider=True, detector="raising")
    assert eng_off.calls == 0
    assert eng_on.calls == 1
    assert _OCR_MARK in _page_text(tmp_path, "d_on")


def test_chart_asset_lane_demotes_on_garbled_math(tmp_path, monkeypatch) -> None:
    """The chart lane ships retained native prose; it must not bypass the demotion."""
    _e2e(
        tmp_path,
        "f_off",
        monkeypatch,
        provider=True,
        native_only=True,
        chart=True,
        detector="neutralised",
    )
    side_off = _sidecar(tmp_path, "f_off")
    assert side_off["engine"] == "chart_asset", "setup: chart lane"
    assert side_off["status"] == "success"
    on, _ = _e2e(tmp_path, "f_on", monkeypatch, provider=True, native_only=True, chart=True)
    side = _sidecar(tmp_path, "f_on")
    assert side["engine"] == "chart_asset"
    assert side["status"] == "warning"
    assert side["failure_mode"] == FailureMode.NATIVE_GARBLED_MATH.value
    assert on.status is not DocumentStatus.SUCCESS


def _e2e_corrupt_math(tmp_path, tag, monkeypatch, *, detector):
    """process() a garbled-math page that ALSO trips corrupt-math, recovery enabled.

    ``has_corrupt_math`` is stamped after analysis (a synthetic PDF cannot produce it), so the
    only difference between the runs is whether the #960 detector fires.
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


def test_e2e_corrupt_math_page_goes_whole_page_when_garbled(tmp_path, monkeypatch) -> None:
    _, eng_off = _e2e_corrupt_math(tmp_path, "m_off", monkeypatch, detector="neutralised")
    _, eng_on = _e2e_corrupt_math(tmp_path, "m_on", monkeypatch, detector="live")

    assert "estimated effect" in _page_text(tmp_path, "m_off"), (
        "main: the hybrid lane owns the page and the native prose ships"
    )
    assert _OCR_MARK not in _page_text(tmp_path, "m_off")
    assert eng_on.calls == 1
    assert _OCR_MARK in _page_text(tmp_path, "m_on")
    assert "estimated effect" not in _page_text(tmp_path, "m_on")


# ---------------------------------------------------------------------------
# Resume: the CURRENT analysis wins over a cached native-text terminal page.
# ---------------------------------------------------------------------------


def _spy_restores(monkeypatch) -> list:
    restored: list = []
    orig = UnifiedPipeline._load_terminal_page

    def spy(self, *a, **kw):
        out = orig(self, *a, **kw)
        if out is not None:
            restored.append(out)
        return out

    monkeypatch.setattr(UnifiedPipeline, "_load_terminal_page", spy)
    return restored


@pytest.mark.parametrize("chart", [False, True], ids=["native_lane", "chart_lane"])
@pytest.mark.parametrize("second", ["live", "raising"])
def test_resume_does_not_restore_a_cached_clean_page_when_the_scan_now_fires(
    tmp_path, monkeypatch, chart, second
) -> None:
    kw = dict(provider=True, native_only=True, chart=chart)
    _e2e(tmp_path, "r", monkeypatch, detector="neutralised", **kw)
    first = _sidecar(tmp_path, "r")
    assert first["status"] == "success", "setup: the clean first run caches SUCCESS"
    assert first["engine"] == ("chart_asset" if chart else "native"), first["engine"]

    restored = _spy_restores(monkeypatch)
    res, _ = _e2e(tmp_path, "r", monkeypatch, detector=second, reprocess=True, **kw)
    again = _sidecar(tmp_path, "r")
    assert restored == [], "the cached native page must not be restored"
    assert again["status"] == "warning"
    assert again["failure_mode"] == FailureMode.NATIVE_GARBLED_MATH.value
    assert res.status is not DocumentStatus.SUCCESS


@pytest.mark.parametrize("chart", [False, True], ids=["native_lane", "chart_lane"])
def test_resume_still_restores_when_the_scan_stays_clean(tmp_path, monkeypatch, chart) -> None:
    """Control: neutralised both times, the cached page IS restored, so the refusal above is
    caused by the fresh flag and nothing else."""
    kw = dict(provider=True, native_only=True, chart=chart, detector="neutralised")
    _e2e(tmp_path, "c", monkeypatch, **kw)
    restored = _spy_restores(monkeypatch)
    res, _ = _e2e(tmp_path, "c", monkeypatch, reprocess=True, **kw)
    assert len(restored) == 1, "setup: the ledger gate restores a clean cached page"
    assert _sidecar(tmp_path, "c")["status"] == "success"
    assert res.status is DocumentStatus.SUCCESS
