"""#990: a control byte where the PDF prints a minus must not ship as trusted native.

The text layer hands back a C0 byte for a glyph it cannot decode, so ``-0.47`` ships as
``\\x010.47`` under SUCCESS (the byte is invisible on render). The detector reads the page text
AFTER #217's repair; any hit routes the page off the trusted-native lane to OCR, through
#913's surfacing path with its own audit kind.

Hermetic: synthetic fitz PDFs, no provider, no judge. Routing pins are DIFFERENCES (the
detector neutralised vs live in the same process), never an absolute outcome (CLAUDE.md, #257).
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from socr.core import born_digital
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.glyph_recovery import count_control_byte_before_digit_hits
from socr.core.result import DocumentStatus, FailureMode
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline

_PROSE = (
    "The estimated effect of the policy change on output is reported below for each "
    "specification, with robust standard errors clustered by region and year. "
)

HITS = {
    "stx_before_digit": "the coefficient is \x020.47 here",
    "soh_before_digit": "the coefficient is \x010.47 here",
    "eot_before_dot_digit": "the coefficient is \x04.47 here",
    "one_space_between": "the coefficient is \x02 0.47 here",
}
CONTROLS = {
    "tab": "the coefficient is \t0.47 here",
    "plain_number": "the coefficient is 0.47 here",
    "control_then_letter": "the coefficient is \x02abc here",
    "control_then_dot_letter": "the coefficient is \x02.abc here",
    "control_at_end": "the coefficient is here \x02",
    "two_spaces_between": "the coefficient is \x02  0.47 here",
}


def _page_pdf(path: Path, line: str, *, chart: bool = False) -> Path:
    """Prose plus one line carrying ``line``; ``chart`` adds a large raster."""
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for _ in range(8):
        page.insert_text((72, y), _PROSE, fontname="helv", fontsize=10)
        y += 14
    page.insert_text((72, y + 10), line, fontname="helv", fontsize=10)
    if chart:
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 300, 300), False)
        pix.set_rect(pix.irect, (200, 30, 30))
        page.insert_image(fitz.Rect(72, y + 30, 372, y + 330), pixmap=pix)
    doc.save(path)
    doc.close()
    return path


def _text(pdf: Path) -> str:
    with fitz.open(pdf) as doc:
        return doc[0].get_text("text")


def test_extractor_really_keeps_the_control_byte(tmp_path: Path) -> None:
    """Setup canary: if fitz stopped emitting the byte, every test below would be vacuous."""
    assert "\x02" in _text(_page_pdf(tmp_path / "p.pdf", HITS["stx_before_digit"]))


@pytest.mark.parametrize("name", sorted(HITS))
def test_detector_fires_on_a_control_byte_before_a_number(tmp_path: Path, name: str) -> None:
    text = _text(_page_pdf(tmp_path / "p.pdf", HITS[name]))
    assert count_control_byte_before_digit_hits(text) == 1


@pytest.mark.parametrize("name", sorted(CONTROLS))
def test_detector_is_quiet_on_controls(tmp_path: Path, name: str) -> None:
    text = _text(_page_pdf(tmp_path / "p.pdf", CONTROLS[name]))
    assert count_control_byte_before_digit_hits(text) == 0


@pytest.mark.parametrize("sep", ["\t", "\n", "\r"])
def test_tab_newline_cr_before_digits_stay_quiet(sep: str) -> None:
    assert count_control_byte_before_digit_hits(f"value{sep}0.47 and{sep}.5") == 0


def test_every_other_c0_code_fires() -> None:
    fired = {c for c in range(0x20) if count_control_byte_before_digit_hits(f"a{chr(c)}1")}
    assert fired == set(range(0x20)) - {9, 10, 13}


def test_a_unicode_decimal_digit_counts_as_a_digit() -> None:
    """Ramey p80: the digit after the byte is a mis-decoded non-ASCII digit; `[0-9]` missed it."""
    assert count_control_byte_before_digit_hits("a \x1f٣ b") == 1


def test_counts_each_occurrence() -> None:
    assert count_control_byte_before_digit_hits("a \x011 b \x02.5 c \x04x \x0307") == 3


def test_loaded_source_is_this_checkout() -> None:
    import socr

    src = Path(__file__).resolve().parents[1] / "src"
    assert Path(socr.__file__).resolve().is_relative_to(src), socr.__file__


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


@pytest.mark.parametrize("name", sorted(HITS) + sorted(CONTROLS))
def test_routing_difference_pin(tmp_path: Path, name: str, monkeypatch) -> None:
    pdf = _page_pdf(tmp_path / "p.pdf", {**HITS, **CONTROLS}[name])
    with monkeypatch.context() as m:
        m.setattr(born_digital, "count_control_byte_before_digit_hits", lambda text: 0)
        pipe_off, state_off = _analyze(pdf)
        trusted_off = pipe_off._is_agentic_trusted_native(1, state_off.pages[1])
        kinds_off = [e.kind for e in state_off.events]
    pipe_on, state_on = _analyze(pdf)
    trusted_on = pipe_on._is_agentic_trusted_native(1, state_on.pages[1])
    events = [e for e in state_on.events if e.kind == "control_byte_before_digit"]

    assert trusted_off, "baseline (main behaviour): the page ships trusted native"
    assert "control_byte_before_digit" not in kinds_off
    # The two kinds stay distinguishable: #913's detector does not see this page.
    assert not [e for e in state_on.events if e.kind == "minus_extracted_as_digit"]
    if name in HITS:
        assert not trusted_on
        assert [(e.page_num, e.data["hits"]) for e in events] == [(1, 1)]
    else:
        assert trusted_on
        assert events == []


def test_event_is_recomputed_not_replayed() -> None:
    assert "control_byte_before_digit" not in UnifiedPipeline.resume_restore_kinds()


# ---------------------------------------------------------------------------
# End-to-end through process(): differences against the detector-neutralised run.
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
    line: str = HITS["stx_before_digit"],
    minus_raising: bool = False,
):
    """Run process() once. ``detector``: live | neutralised | raising."""
    if detector not in {"live", "neutralised", "raising"}:
        raise ValueError(f"unknown detector mode {detector!r}")
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch

    engine = _StubEngine()
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: engine)
        if detector == "neutralised":
            m.setattr(born_digital, "count_control_byte_before_digit_hits", lambda t: 0)
        elif detector == "raising":

            def _boom(text):
                raise RuntimeError("detector exploded")

            m.setattr(born_digital, "count_control_byte_before_digit_hits", _boom)
        if minus_raising:  # #913's scan failing on the second run

            def _boom913(page):
                raise RuntimeError("913 scan exploded")

            m.setattr(born_digital, "count_minus_as_digit_hits", _boom913)
        pdf = tmp_path / f"{tag}.pdf"
        if not pdf.exists():  # a second run on the same tag must see the same bytes (checksum)
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


def test_e2e_provider_present_text_comes_from_the_provider(tmp_path, monkeypatch) -> None:
    _, eng_off = _e2e(tmp_path, "a_off", monkeypatch, provider=True, detector="neutralised")
    _, eng_on = _e2e(tmp_path, "a_on", monkeypatch, provider=True)

    assert eng_off.calls == 0 and "0.47" in _page_text(tmp_path, "a_off")
    assert eng_on.calls == 1
    assert _OCR_MARK in _page_text(tmp_path, "a_on")
    assert _sidecar(tmp_path, "a_on")["engine"] != "native"


def test_e2e_provider_absent_is_never_trusted_success(tmp_path, monkeypatch) -> None:
    off, _ = _e2e(tmp_path, "b_off", monkeypatch, provider=False, detector="neutralised")
    on, _ = _e2e(tmp_path, "b_on", monkeypatch, provider=False)

    assert _sidecar(tmp_path, "b_off")["status"] == "success"
    assert off.status is DocumentStatus.SUCCESS
    side = _sidecar(tmp_path, "b_on")
    assert side["status"] != "success"
    assert side["failure_mode"] == FailureMode.NATIVE_MINUS_AS_DIGIT.value
    assert on.status is not DocumentStatus.SUCCESS
    assert "0.47" in _page_text(tmp_path, "b_on"), "no content dropped"


def test_e2e_native_only_retains_the_page_but_demotes_it(tmp_path, monkeypatch) -> None:
    off, _ = _e2e(
        tmp_path, "c_off", monkeypatch, provider=True, native_only=True, detector="neutralised"
    )
    on, eng_on = _e2e(tmp_path, "c_on", monkeypatch, provider=True, native_only=True)
    side_off, side_on = _sidecar(tmp_path, "c_off"), _sidecar(tmp_path, "c_on")

    assert side_off["status"] == "success" and off.status is DocumentStatus.SUCCESS
    assert eng_on.calls == 0
    assert side_on["status"] == "warning"
    assert side_on["failure_mode"] == FailureMode.NATIVE_MINUS_AS_DIGIT.value
    assert on.status is not DocumentStatus.SUCCESS
    assert "0.47" in _page_text(tmp_path, "c_on")
    assert "native_minus_as_digit_retained" in _audit_kinds(tmp_path, "c_on")
    assert "native_minus_as_digit_retained" not in _audit_kinds(tmp_path, "c_off")


def test_e2e_detector_exception_fails_closed(tmp_path, monkeypatch) -> None:
    _, eng_off = _e2e(tmp_path, "d_off", monkeypatch, provider=True, detector="neutralised")
    _, eng_on = _e2e(tmp_path, "d_on", monkeypatch, provider=True, detector="raising")
    assert eng_off.calls == 0
    assert eng_on.calls == 1
    assert _OCR_MARK in _page_text(tmp_path, "d_on")


def test_e2e_raising_detector_under_native_only_is_demoted(tmp_path, monkeypatch) -> None:
    _e2e(tmp_path, "e_off", monkeypatch, provider=True, native_only=True, detector="neutralised")
    assert _sidecar(tmp_path, "e_off")["status"] == "success"
    on, _ = _e2e(tmp_path, "e_on", monkeypatch, provider=True, native_only=True, detector="raising")
    side = _sidecar(tmp_path, "e_on")
    assert side["status"] == "warning"
    assert side["failure_mode"] == FailureMode.NATIVE_MINUS_AS_DIGIT.value
    assert on.status is not DocumentStatus.SUCCESS


def test_event_says_routed_or_retained_and_flags_errors(tmp_path, monkeypatch) -> None:
    pdf = _page_pdf(tmp_path / "p.pdf", HITS["stx_before_digit"])

    def _events(native_only: bool, detector=None):
        with monkeypatch.context() as m:
            if detector is not None:
                m.setattr(born_digital, "count_control_byte_before_digit_hits", detector)
            pipe = _pipeline()
            pipe.config.native_only = native_only
            state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
            pipe._phase_analyze(state)
        return [e for e in state.events if e.kind == "control_byte_before_digit"]

    routed, retained = _events(False)[0], _events(True)[0]
    assert "routed to OCR" in routed.detail and routed.data["error"] is False
    assert "RETAINED" in retained.detail and "routed to OCR" not in retained.detail

    def _boom(text):
        raise RuntimeError("x")

    failed = _events(False, _boom)[0]
    assert failed.data["error"] is True and "FAILED" in failed.detail


def test_chart_asset_lane_demotes_on_a_control_byte(tmp_path, monkeypatch) -> None:
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
    assert _sidecar(tmp_path, "f_off")["engine"] == "chart_asset", "setup: chart lane"
    assert _sidecar(tmp_path, "f_off")["status"] == "success"
    on, _ = _e2e(tmp_path, "f_on", monkeypatch, provider=True, native_only=True, chart=True)
    side = _sidecar(tmp_path, "f_on")
    assert side["engine"] == "chart_asset"
    assert side["status"] == "warning"
    assert side["failure_mode"] == FailureMode.NATIVE_MINUS_AS_DIGIT.value
    assert on.status is not DocumentStatus.SUCCESS


# ----------------------------------------------------------------------------
# Resume: the CURRENT analysis wins over a cached native-text terminal page.
# A clean first run caches SUCCESS; a later (--reprocess, so the doc-level skip is off and the
# per-page ledger gate is what decides) run whose scan hits or fails must not restore it.
# ----------------------------------------------------------------------------


def _spy_restores(monkeypatch) -> list:
    """Record every non-None return of the per-page ledger gate."""
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
    assert again["status"] == "warning", "must be demoted, never restored as SUCCESS"
    assert again["failure_mode"] == FailureMode.NATIVE_MINUS_AS_DIGIT.value
    assert res.status is not DocumentStatus.SUCCESS


@pytest.mark.parametrize("chart", [False, True], ids=["native_lane", "chart_lane"])
def test_resume_still_restores_when_the_scan_stays_clean(tmp_path, monkeypatch, chart) -> None:
    """Control for the pin above: the same two runs with the detector neutralised both times
    DO restore the cached page, so the refusal is caused by the fresh flag and nothing else."""
    kw = dict(provider=True, native_only=True, chart=chart, detector="neutralised")
    _e2e(tmp_path, "c", monkeypatch, **kw)
    restored = _spy_restores(monkeypatch)
    res, _ = _e2e(tmp_path, "c", monkeypatch, reprocess=True, **kw)
    assert len(restored) == 1, "setup: the ledger gate restores a clean cached page"
    assert _sidecar(tmp_path, "c")["status"] == "success"
    assert res.status is DocumentStatus.SUCCESS


@pytest.mark.parametrize("chart", [False, True], ids=["native_lane", "chart_lane"])
def test_resume_hole_is_shared_by_913s_scan(tmp_path, monkeypatch, chart) -> None:
    """#913 uses the same predicate, so its failing scan had the same resume hole."""
    kw = dict(provider=True, native_only=True, chart=chart, line="the coefficient is 0.47 and 2.5")
    _e2e(tmp_path, "m", monkeypatch, detector="neutralised", **kw)
    assert _sidecar(tmp_path, "m")["status"] == "success", "setup: clean first run"
    restored = _spy_restores(monkeypatch)
    res, _ = _e2e(
        tmp_path, "m", monkeypatch, detector="neutralised", reprocess=True, minus_raising=True, **kw
    )
    assert restored == []
    assert _sidecar(tmp_path, "m")["status"] == "warning"
    assert _sidecar(tmp_path, "m")["failure_mode"] == FailureMode.NATIVE_MINUS_AS_DIGIT.value
    assert res.status is not DocumentStatus.SUCCESS
