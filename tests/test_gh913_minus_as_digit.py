"""#913: a minus the text layer extracts as the digit "2" must not ship as trusted native.

#217 rebuilds a missing ToUnicode map, but a font it cannot repair still hands back the
raw byte, so ``-0.12`` ships as ``20.12`` under SUCCESS. The detector runs on the page as
it looks AFTER that repair; any hit routes the page off the trusted-native lane to OCR.

Every test is hermetic: synthetic PDFs built with fitz, no provider, no judge. The
routing pin is a DIFFERENCE (the same pipeline with the detector neutralised vs live),
never an absolute outcome measured on one machine (CLAUDE.md, #257).
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from socr.core import born_digital
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import DocumentStatus, FailureMode
from socr.core.glyph_recovery import MINUS_AS_DIGIT_SIZE_TOLERANCE_PT, count_minus_as_digit_hits
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline

_PROSE = (
    "The estimated effect of the policy change on output is reported below for each "
    "specification, with robust standard errors clustered by region and year. "
)
_TAIL = " was the coefficient on lagged output."


def _page_pdf(path: Path, draw) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for _ in range(8):
        page.insert_text((72, y), _PROSE, fontname="helv", fontsize=10)
        y += 14
    draw(page, y + 10)
    doc.save(path)
    doc.close()
    return path


def _pair(page, y, first, first_font, first_size, rest, rest_font="helv", first_color=(0, 0, 0)):
    """``first`` then ``rest`` on one baseline, flush against each other (no space span)."""
    page.insert_text((72, y), first, fontname=first_font, fontsize=first_size, color=first_color)
    x = 72 + fitz.get_text_length(first, fontname=first_font, fontsize=first_size)
    page.insert_text((x, y), rest + _TAIL, fontname=rest_font, fontsize=10)


def _fake_minus_symbol_font(page, y):
    # The shape #913 measured: a lone "2" from a symbol font, then the number in the
    # text font, at the same size.
    _pair(page, y, "2", "symb", 10, "0.12")


def _lone_two_other_font(page, y):
    _pair(page, y, "2", "tiro", 10, "3.4")


def _superscript_two(page, y):
    _pair(page, y, "2", "symb", 6, "0.12")


def _ordinary_decimal(page, y):
    page.insert_text((72, y), "2.5" + _TAIL + " (year 2020)", fontname="helv", fontsize=10)


def _same_font_lone_two(page, y):
    # Same font and size, but a colour change still splits the span: only the font
    # comparison keeps this from being read as a fake minus.
    _pair(page, y, "2", "helv", 10, "3.4", first_color=(1, 0, 0))


def _two_then_letter(page, y):
    _pair(page, y, "2", "symb", 10, "x")


def _lone_three(page, y):
    _pair(page, y, "3", "symb", 10, "0.12")


HIT_SHAPES = {
    "symbol_font_two": _fake_minus_symbol_font,
    "lone_two_other_font": _lone_two_other_font,
}
CONTROL_SHAPES = {
    "superscript": _superscript_two,
    "ordinary_decimal": _ordinary_decimal,
    "same_font_lone_two": _same_font_lone_two,
    "two_then_letter": _two_then_letter,
    "lone_three": _lone_three,
}


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


def _analyze(pdf: Path):
    pipe = _pipeline()
    state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    pipe._phase_analyze(state)
    return pipe, state


def _open_hits(pdf: Path) -> int:
    with fitz.open(pdf) as doc:
        return count_minus_as_digit_hits(doc[0])


@pytest.mark.parametrize("name", sorted(HIT_SHAPES))
def test_detector_counts_the_fake_minus(tmp_path: Path, name: str) -> None:
    assert _open_hits(_page_pdf(tmp_path / "p.pdf", HIT_SHAPES[name])) == 1


@pytest.mark.parametrize("name", sorted(CONTROL_SHAPES))
def test_detector_ignores_controls(tmp_path: Path, name: str) -> None:
    assert _open_hits(_page_pdf(tmp_path / "p.pdf", CONTROL_SHAPES[name])) == 0


def test_size_tolerance_is_a_named_constant() -> None:
    assert MINUS_AS_DIGIT_SIZE_TOLERANCE_PT > 0


def test_loaded_source_is_this_checkout() -> None:
    """Canary: the mutation harness runs this suite from a copy and must be measuring it."""
    import socr

    assert (
        Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")
    ), socr.__file__


@pytest.mark.parametrize("name", sorted(HIT_SHAPES) + sorted(CONTROL_SHAPES))
def test_routing_difference_pin(tmp_path: Path, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same PDF, same pipeline; only the detector differs. Hits move off trusted-native."""
    shapes = {**HIT_SHAPES, **CONTROL_SHAPES}
    pdf = _page_pdf(tmp_path / "p.pdf", shapes[name])

    with monkeypatch.context() as m:
        m.setattr(born_digital, "count_minus_as_digit_hits", lambda page: 0)
        pipe_off, state_off = _analyze(pdf)
        trusted_off = pipe_off._is_agentic_trusted_native(1, state_off.pages[1])
        kinds_off = [e.kind for e in state_off.events]

    pipe_on, state_on = _analyze(pdf)
    trusted_on = pipe_on._is_agentic_trusted_native(1, state_on.pages[1])
    events_on = [e for e in state_on.events if e.kind == "minus_extracted_as_digit"]

    assert state_off.pages[1].is_born_digital
    assert trusted_off, "baseline (the main behaviour): the page ships trusted native"
    assert "minus_extracted_as_digit" not in kinds_off
    if name in HIT_SHAPES:
        assert not trusted_on
        assert [(e.page_num, e.data["hits"]) for e in events_on] == [(1, 1)]
    else:
        assert trusted_on
        assert events_on == []


def test_event_is_recomputed_not_replayed() -> None:
    """analyze runs on every run, so replaying the event from a sidecar would double it."""
    assert "minus_extracted_as_digit" not in UnifiedPipeline.resume_restore_kinds()


# ---------------------------------------------------------------------------
# End-to-end through process(): every pin is a DIFFERENCE against the run with the
# detector neutralised (the behaviour of main), never an absolute outcome.
# ---------------------------------------------------------------------------

_OCR_MARK = "PROVIDER_READ_OF_THE_PAGE"


class _StubEngine:
    """The provider seam: returns a recognisable text, counts calls, never spawns."""

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
):
    """Run process() once. ``detector``: live | neutralised | raising."""
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch

    engine = _StubEngine()
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: engine)
        if detector == "neutralised":
            m.setattr(born_digital, "count_minus_as_digit_hits", lambda page: 0)
        elif detector == "raising":

            def _boom(page):
                raise RuntimeError("detector exploded")

            m.setattr(born_digital, "count_minus_as_digit_hits", _boom)
        pdf = _page_pdf(tmp_path / f"{tag}.pdf", _fake_minus_symbol_font)
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


def _sidecar(tmp_path: Path, tag: str) -> dict:
    """The per-page sidecar: the only place the PAGE's status/failure mode is recorded
    (``EngineResult.pages`` is one whole-document blob)."""
    import json

    found = sorted((tmp_path / f"out-{tag}").rglob("pages/00001.json"))
    assert len(found) == 1, found
    return json.loads(found[0].read_text())


def _page_text(tmp_path: Path, tag: str) -> str:
    found = sorted((tmp_path / f"out-{tag}").rglob("pages/00001.md"))
    assert len(found) == 1, found
    return found[0].read_text()


def test_e2e_provider_present_text_comes_from_the_provider(tmp_path, monkeypatch) -> None:
    off, eng_off = _e2e(tmp_path, "a_off", monkeypatch, provider=True, detector="neutralised")
    on, eng_on = _e2e(tmp_path, "a_on", monkeypatch, provider=True)

    assert eng_off.calls == 0 and _OCR_MARK not in _page_text(tmp_path, "a_off")
    assert "0.12" in _page_text(tmp_path, "a_off"), "main behaviour: the native layer ships"
    assert eng_on.calls == 1
    assert _OCR_MARK in _page_text(tmp_path, "a_on")
    assert "0.12" not in _page_text(tmp_path, "a_on")
    assert _sidecar(tmp_path, "a_on")["engine"] != "native"


def test_e2e_provider_absent_is_never_trusted_success(tmp_path, monkeypatch) -> None:
    off, _ = _e2e(tmp_path, "b_off", monkeypatch, provider=False, detector="neutralised")
    on, _ = _e2e(tmp_path, "b_on", monkeypatch, provider=False)

    assert _sidecar(tmp_path, "b_off")["status"] == "success", "main: trusted native SUCCESS"
    assert off.status is DocumentStatus.SUCCESS
    assert _sidecar(tmp_path, "b_on")["status"] != "success"
    assert _sidecar(tmp_path, "b_on")["failure_mode"] == FailureMode.NATIVE_MINUS_AS_DIGIT.value
    assert on.status is not DocumentStatus.SUCCESS
    assert "0.12" in _page_text(tmp_path, "b_on"), "no content dropped"


def test_e2e_native_only_retains_the_page_but_demotes_it(tmp_path, monkeypatch) -> None:
    off, _ = _e2e(
        tmp_path, "c_off", monkeypatch, provider=True, native_only=True, detector="neutralised"
    )
    on, eng_on = _e2e(tmp_path, "c_on", monkeypatch, provider=True, native_only=True)
    side_off, side_on = _sidecar(tmp_path, "c_off"), _sidecar(tmp_path, "c_on")

    assert side_off["status"] == "success" and off.status is DocumentStatus.SUCCESS
    assert eng_on.calls == 0, "--native-only never calls a provider"
    assert side_on["status"] == "warning"
    assert side_on["failure_mode"] == FailureMode.NATIVE_MINUS_AS_DIGIT.value
    assert on.status is not DocumentStatus.SUCCESS
    assert "0.12" in _page_text(tmp_path, "c_on"), "the text is retained, only the status differs"


def test_e2e_detector_exception_fails_closed(tmp_path, monkeypatch) -> None:
    _, eng_off = _e2e(tmp_path, "d_off", monkeypatch, provider=True, detector="neutralised")
    _, eng_on = _e2e(tmp_path, "d_on", monkeypatch, provider=True, detector="raising")

    assert eng_off.calls == 0
    assert eng_on.calls == 1
    assert _OCR_MARK in _page_text(tmp_path, "d_on")


def test_event_text_says_routed_or_retained_and_flags_errors(tmp_path, monkeypatch) -> None:
    pdf = _page_pdf(tmp_path / "p.pdf", _fake_minus_symbol_font)

    def _events(native_only: bool, detector=None):
        with monkeypatch.context() as m:
            if detector is not None:
                m.setattr(born_digital, "count_minus_as_digit_hits", detector)
            pipe = _pipeline()
            pipe.config.native_only = native_only
            state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
            pipe._phase_analyze(state)
        return [e for e in state.events if e.kind == "minus_extracted_as_digit"]

    routed = _events(False)[0]
    retained = _events(True)[0]
    assert "routed to OCR" in routed.detail and routed.data["error"] is False
    assert "RETAINED" in retained.detail and "routed to OCR" not in retained.detail

    def _boom(page):
        raise RuntimeError("x")

    failed = _events(False, _boom)[0]
    assert failed.data["error"] is True and "FAILED" in failed.detail


def test_detector_counts_a_two_followed_by_dot_five(tmp_path: Path) -> None:
    """The `2` + `.5` shape: the next span starts with a point, then a digit."""

    def draw(page, y):
        _pair(page, y, "2", "symb", 10, ".5")

    assert _open_hits(_page_pdf(tmp_path / "p.pdf", draw)) == 1


def test_e2e_native_only_with_a_raising_detector_is_demoted_too(tmp_path, monkeypatch) -> None:
    """Unknown is not clean: a failed scan under --native-only demotes like a hit."""
    off, _ = _e2e(
        tmp_path, "e_off", monkeypatch, provider=True, native_only=True, detector="neutralised"
    )
    assert (
        _sidecar(tmp_path, "e_off")["status"] == "success" and off.status is DocumentStatus.SUCCESS
    )
    on, _ = _e2e(tmp_path, "e_on", monkeypatch, provider=True, native_only=True, detector="raising")
    side = _sidecar(tmp_path, "e_on")
    assert side["status"] == "warning"
    assert side["failure_mode"] == FailureMode.NATIVE_MINUS_AS_DIGIT.value
    assert on.status is not DocumentStatus.SUCCESS
