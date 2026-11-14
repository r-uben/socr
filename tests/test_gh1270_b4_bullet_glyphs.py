"""#1270 (B4, class 1): a list bullet is not shown to be unrecovered mathematics, so the mark says
what is true. NOTHING is released: a page with private-use codepoints keeps its mark and its
status exactly as on main (text-layer and ink evidence cannot certify a page: coordinator ruling
on PR #1341).

The detector records ``private_use_bullets_only`` (the #960 span-level count found no private-use
codepoint in a maths-font span), keeping the font-blind ``has_unmapped_math_glyphs`` that routing
and the maths accounting read. The verdict clause keeps its check, result (FAIL) and scope and
changes only its words: "private-use bullet or dingbat glyphs", with no claim about maths fonts
(the scan checked codepoints, and a page may typeset maths in a maths font elsewhere) and none
about what else was or was not checked (the pass trace may hold PASS or FAIL for other checks).
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fitz
import pytest
from test_gh165_unresolved_math_outcome import _prose_pdf, _state

from socr.core.born_digital import BornDigitalDetector, count_pua_chars
from socr.core.config import EngineType, PipelineConfig
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.core.state import PageState
from socr.core.verdict import (
    BULLET_GLYPHS_REASON,
    Check,
    CheckResult,
    _native_glyph_clause,
)
from socr.math.accounting import UNRESOLVED_MATH_KIND
from socr.pipeline.orchestrator import UnifiedPipeline

BULLET = chr(0xF0B7)
WINGDINGS_BULLET = chr(0xF0A7)
MATH_MINUS = chr(0xF02D)

_PROSE = (
    "Optimists set a lift-off date and the heterogeneous agents consume more, with enough "
    "trailing words to stay above the floor for a born digital clean text layer classification."
)


# ------------------------------------------------------------------------------ the detector


def _assess(tmp_path: Path, monkeypatch, *, glyph: str, font: str, scan_fails: bool = False):
    """Run the REAL ``_assess_page`` on a page whose text layer carries ``glyph`` in a span of
    ``font``. Only the glyph's font differs between arms."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = _prose_pdf(tmp_path / "doc.pdf")
    doc = fitz.open(str(pdf))
    page = doc[0]
    original = page.get_text
    crafted = f"{_PROSE}\n{glyph} first list item\n"

    def fake(*args, **kwargs):
        mode = args[0] if args else kwargs.get("option", "text")
        if mode == "text":
            return crafted
        if mode == "dict":
            spans = [{"font": font, "text": f"{glyph} first list item", "bbox": (0, 0, 1, 1)}]
            return {"blocks": [{"type": 0, "lines": [{"spans": spans}]}]}
        return original(*args, **kwargs)

    monkeypatch.setattr(page, "get_text", fake)
    if scan_fails:
        monkeypatch.setattr(
            "socr.core.born_digital.detect_garbled_math",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("scan failed")),
        )
    pa = BornDigitalDetector()._assess_page(page, 1)
    doc.close()
    return pa


@pytest.mark.parametrize(
    ("glyph", "font"),
    [(BULLET, "SymbolMT"), (WINGDINGS_BULLET, "Wingdings-Regular"), (chr(0xF074), "Wingdings")],
)
def test_a_bullet_is_recorded_as_bullets_only_and_the_flag_is_unchanged(
    tmp_path, monkeypatch, glyph, font
):
    pa = _assess(tmp_path, monkeypatch, glyph=glyph, font=font)
    assert pa.is_born_digital
    assert count_pua_chars(pa.native_text) == 1, "premise: the codepoint is in the shipped text"
    # the flag stays font-blind (routing and the maths accounting read it) ...
    assert pa.has_unmapped_math_glyphs is True
    # ... and the new fact says no private-use codepoint is mathematics
    assert pa.private_use_bullets_only is True
    assert any("bullet or dingbat glyphs" in n for n in pa.notes), "the note says what was found"
    assert pa.needs_ocr_enhancement is False


def test_only_the_font_differs_between_a_bullet_and_maths(tmp_path, monkeypatch):
    """Same codepoint-count, same text; a font that is not a Symbol-encoded font owns its own
    glyph, which may be maths (the #960 rule)."""
    bullet = _assess(tmp_path / "a", monkeypatch, glyph=BULLET, font="SymbolMT")
    maths = _assess(tmp_path / "b", monkeypatch, glyph=BULLET, font="CIDFont+F6")
    real_minus = _assess(tmp_path / "c", monkeypatch, glyph=MATH_MINUS, font="SymbolMT")
    assert bullet.native_text == maths.native_text
    assert bullet.private_use_bullets_only is True
    assert maths.private_use_bullets_only is False
    assert real_minus.private_use_bullets_only is False
    assert all(x.has_unmapped_math_glyphs for x in (bullet, maths, real_minus))
    assert any("unmapped math glyphs" in n for n in real_minus.notes)
    assert not any("list bullets" in n for n in real_minus.notes)


def test_a_failed_scan_shows_nothing(tmp_path, monkeypatch):
    """Fail closed: an unreadable page is unknown, not bullets."""
    failed = _assess(tmp_path / "a", monkeypatch, glyph=BULLET, font="SymbolMT", scan_fails=True)
    assert failed.garbled_math_scan_failed is True
    assert failed.has_unmapped_math_glyphs is True
    assert failed.private_use_bullets_only is False


def test_a_page_with_no_private_use_codepoint_is_not_bullets_only(tmp_path, monkeypatch):
    pa = _assess(tmp_path, monkeypatch, glyph="-", font="SymbolMT")
    assert pa.has_unmapped_math_glyphs is False and pa.private_use_bullets_only is False


# ------------------------------------------------------------------------------ the verdict


def _page(**over) -> SimpleNamespace:
    base = dict(
        page_num=1,
        has_unmapped_math_glyphs=True,
        private_use_bullets_only=True,
        text_coverage_verdict="passed",
        is_born_digital=True,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _native(text: str = _PROSE) -> PageOutput:
    return PageOutput(
        page_num=1, text=text, status=PageStatus.WARNING, engine="native", audit_passed=True
    )


def test_the_bullet_mark_states_only_what_is_true():
    clause = _native_glyph_clause(_page(), _native())
    assert clause is not None and clause.check is Check.NATIVE_GLYPHS
    assert clause.reason == BULLET_GLYPHS_REASON
    # no claim about mathematics, maths fonts, or what else was checked
    for word in ("mathematics", "maths font", "otherwise checked", "not checked"):
        assert word not in clause.reason


def test_no_evidence_changes_the_mark():
    """Differences, not values: the same page with every piece of independent evidence present
    (ink check passed) and absent yields the identical clause. Nothing releases a page."""
    out = _native()
    with_evidence = _native_glyph_clause(_page(text_coverage_verdict="passed"), out)
    without = _native_glyph_clause(_page(text_coverage_verdict=""), out)
    assert with_evidence == without and with_evidence is not None


def test_a_bullet_page_differs_from_a_maths_page_only_in_the_words():
    out = _native()
    bullets = _native_glyph_clause(_page(), out)
    maths = _native_glyph_clause(_page(private_use_bullets_only=False), out)
    assert bullets is not None and maths is not None
    assert (bullets.check, bullets.result, bullets.scope) == (
        maths.check,
        maths.result,
        maths.scope,
    )
    assert bullets.result is CheckResult.FAIL
    assert bullets.reason != maths.reason
    assert "unrecovered mathematics" in maths.reason


def test_the_other_native_glyph_modes_are_untouched():
    from socr.core.result import FailureMode
    from socr.core.verdict import _mode_clause

    out = PageOutput(
        page_num=1,
        text="x",
        status=PageStatus.WARNING,
        engine="native",
        failure_mode=FailureMode.NATIVE_GARBLED_MATH,
    )
    clause = _mode_clause(out)
    assert (
        clause is not None
        and clause.check is Check.NATIVE_GLYPHS
        and clause.result is CheckResult.FAIL
    )


def test_the_clause_follows_the_flag_and_only_the_flag() -> None:
    out = _native("x")
    off, on = PageState(page_num=1), PageState(page_num=1)
    on.has_unmapped_math_glyphs = True
    assert _native_glyph_clause(off, out) is None
    assert _native_glyph_clause(on, out) is not None


# ------------------------------------------------------------- the shipping boundary, both states


def _run(tmp_path: Path, *, text: str, bullets_only: bool, coverage: bool, providers: list):
    """The real agentic loop and assemble on ONE native page. Only the arguments differ between
    arms. Engines pinned; the ladder and the judge model supplied explicitly."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "doc.pdf"
    if not pdf.exists():
        _prose_pdf(pdf)
    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            agentic=True,
            judge_backend="heuristic",
            quiet=True,
            save_figures=False,
            write_manifest=False,
            native_first=True,
            check_text_coverage=coverage,
        )
    )
    pipeline._scan_root = pdf.parent
    state = _state(pdf, native_text=text, damaged=True)
    state.pages[1].private_use_bullets_only = bullets_only
    out_dir = tmp_path / "out"
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(pipeline, "_available_engines_for_agentic", return_value=providers)
        )
        stack.enter_context(patch.object(pipeline, "_resolve_judge_model", return_value=""))
        route = stack.enter_context(patch("socr.pipeline.orchestrator.route_page"))
        pipeline._phase_agentic(state, out_dir)
        result = pipeline._phase_assemble(state, out_dir)
        route.assert_not_called()
    return state, result, out_dir


def _outcome(state, result):
    from socr.core.manifest import finalized_page_record

    rec = finalized_page_record(state, 1)
    clauses = [(c.check.value, c.result.value, c.reason) for c in rec.verdict.clauses]
    notes = list(rec.output.audit_notes or [])
    events = [e.detail for e in state.events if e.kind == UNRESOLVED_MATH_KIND]
    return rec.output.status, rec.output.failure_mode, clauses, notes, events, result.status


WITH_BULLET = f"{_PROSE}\n{BULLET} first list item"


@pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []], ids=["provider", "no-provider"])
def test_the_page_keeps_its_status_and_only_the_words_change(tmp_path, providers):
    """Arms differ in ONE thing each: ``bullets_only`` (the wording) and the ink check (the
    would-be evidence). Status, failure mode and the glyph clause's result are identical across
    all of them: nothing is released."""
    arms = {}
    for name, kw in {
        "bullets-ink": dict(bullets_only=True, coverage=True),
        "bullets-no-ink": dict(bullets_only=True, coverage=False),
        "maths-ink": dict(bullets_only=False, coverage=True),
    }.items():
        state, result, _ = _run(tmp_path / name, text=WITH_BULLET, providers=providers, **kw)
        arms[name] = _outcome(state, result)

    def glyph(arm):
        return [c for c in arms[arm][2] if c[0] == "native_glyphs"]

    # no release: status, mode and document status do not depend on the evidence or the wording
    assert arms["bullets-ink"][0] is arms["bullets-no-ink"][0] is arms["maths-ink"][0]
    assert arms["bullets-ink"][0] is not PageStatus.SUCCESS
    assert arms["bullets-ink"][1] == arms["bullets-no-ink"][1] == arms["maths-ink"][1]
    assert arms["bullets-ink"][5] is arms["bullets-no-ink"][5] is arms["maths-ink"][5]
    assert arms["bullets-ink"] == arms["bullets-no-ink"], "the ink evidence changes nothing"
    # the wording differs, the clause result does not
    assert [c[1] for c in glyph("bullets-ink")] == [c[1] for c in glyph("maths-ink")] == ["FAIL"]
    assert [c[2] for c in glyph("bullets-ink")] == [BULLET_GLYPHS_REASON]
    assert "unrecovered mathematics" in glyph("maths-ink")[0][2]
    # page note and document event say what is true
    notes, events = arms["bullets-ink"][3], arms["bullets-ink"][4]
    assert any("bullet or dingbat glyphs" in n for n in notes)
    assert events and all("bullet or dingbat glyphs" in d for d in events)
    assert not any("mathematics" in d and "bullet" in d for d in events)
    maths_events = arms["maths-ink"][4]
    assert maths_events and not any("bullet" in d for d in maths_events)


def test_the_sidecar_records_the_bullet_fact_with_the_flag(tmp_path):
    _s, _r, out_dir = _run(
        tmp_path,
        text=WITH_BULLET,
        bullets_only=True,
        coverage=True,
        providers=[PROFILE_QWEN_LOCAL],
    )
    meta = json.loads(next(out_dir.rglob("00001.json")).read_text())
    assert meta["has_unmapped_math_glyphs"] is True
    assert meta["private_use_bullets_only"] is True
    # maths: the flag is there, the bullet fact is not (sparse: other sidecars keep their bytes)
    _s2, _r2, out2 = _run(
        tmp_path / "maths",
        text=WITH_BULLET,
        bullets_only=False,
        coverage=True,
        providers=[PROFILE_QWEN_LOCAL],
    )
    maths_meta = json.loads(next(out2.rglob("00001.json")).read_text())
    assert maths_meta["has_unmapped_math_glyphs"] is True
    assert "private_use_bullets_only" not in maths_meta


def test_the_bullet_fact_is_restored_on_resume_and_the_wording_follows(tmp_path):
    """A resumed page is not re-assessed: the sidecar alone restores ``private_use_bullets_only``,
    and without it the restored page would say "unrecovered mathematics" again."""
    from socr.core.document import DocumentHandle
    from socr.core.manifest import finalized_page_record
    from socr.core.state import DocumentState

    state, _r, out_dir = _run(
        tmp_path, text=WITH_BULLET, bullets_only=True, coverage=True, providers=[PROFILE_QWEN_LOCAL]
    )
    pipeline = UnifiedPipeline(PipelineConfig(quiet=True, write_manifest=False))
    resumed = DocumentState(handle=DocumentHandle.from_path(state.handle.path))
    out = PageOutput(
        page_num=1, text=WITH_BULLET, status=PageStatus.WARNING, engine="native", audit_passed=True
    )
    pipeline._restore_terminal_page_state(resumed, 1, out, out_dir)
    assert resumed.pages[1].has_unmapped_math_glyphs is True
    assert resumed.pages[1].private_use_bullets_only is True
    rec = finalized_page_record(resumed, 1)
    reasons = [c.reason for c in rec.verdict.clauses if c.check is Check.NATIVE_GLYPHS]
    assert BULLET_GLYPHS_REASON in reasons
    assert any("bullet or dingbat glyphs" in n for n in rec.output.audit_notes or [])


def test_the_detector_fact_reaches_the_page_state(tmp_path):
    """``PageState.apply_born_digital`` copies ``private_use_bullets_only`` from the assessment."""
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState

    pdf = _prose_pdf(tmp_path / "doc.pdf")
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    assessment = BornDigitalDetector().detect(str(pdf))
    for fact in (True, False):
        assessment.pages[0].private_use_bullets_only = fact
        state.apply_born_digital(assessment)
        assert state.pages[1].private_use_bullets_only is fact
