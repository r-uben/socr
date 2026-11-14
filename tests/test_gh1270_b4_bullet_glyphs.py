"""#1270 (B4, class 1) under the RELEASE RULE: a list bullet is not unrecovered mathematics, and a
page's mark is lifted only on independent page-scope evidence.

Before: ``has_unmapped_math_glyphs = pua_count > 0`` counted every private-use codepoint,
font-blind, so SymbolMT / Wingdings list bullets (U+F0B7, U+F0A7, U+F074, U+F06E) marked 292 of
298 ``native_glyphs=FAIL`` pages in run 51270eb5 as "unrecovered mathematics".

Now (a) the detector records ``private_use_bullets_only`` (the #960 span-level maths count found
none), keeping the font-blind flag that routing and the maths accounting read; (b) the verdict
REWORDS the mark to what is true (bullet glyphs, page not otherwise checked) and lifts it, with
the page's demotion, only when the page has independent evidence on the text it ships:
a native page, a digest-verified #1050 ink-vs-layer PASS and no FAIL/UNKNOWN entry; a model
reading, ``independent_page_evidence``; and no private-use codepoint may still ship.

Scope: only the PUA clause (``_native_glyph_clause``). ``NATIVE_GARBLED_MATH`` and
``NATIVE_MINUS_AS_DIGIT`` also map to ``native_glyphs`` and are untouched here.
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
from verdict_marks_helper import assert_differs_only_by_marks

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
    independent_native_evidence,
    native_coverage_outcome,
    private_use_bullets_released,
    stamp_text_digest,
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
    assert any("list bullets or dingbats" in n for n in pa.notes), "the note says what was found"
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

LAYER = "Output grew 2.1 percent in the quarter"


def _page(**over) -> SimpleNamespace:
    base = dict(
        page_num=1,
        has_unmapped_math_glyphs=True,
        private_use_bullets_only=True,
        text_coverage_verdict="passed",
        own_layer_text=LAYER,
        own_layer_word_reference=LAYER,
        own_layer_figure_words="",
        is_born_digital=True,
        prior_ocr_layer_suspect=False,
        has_unrecovered_symbol_glyphs=False,
        table_ladder_disposition=None,
        has_figures=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _native(text: str = LAYER, status=PageStatus.WARNING) -> PageOutput:
    return PageOutput(page_num=1, text=text, status=status, engine="native", audit_passed=True)


def _model(text: str = LAYER, status=PageStatus.WARNING) -> PageOutput:
    return PageOutput(page_num=1, text=text, status=status, engine="qwen", audit_passed=True)


def test_a_native_bullet_page_with_ink_evidence_is_released():
    page, out = _page(), _native()
    assert private_use_bullets_released(out, page) is True
    assert _native_glyph_clause(page, out) is None


@pytest.mark.parametrize(
    "missing",
    [
        pytest.param({"text_coverage_verdict": "flagged"}, id="ink-unread"),
        pytest.param({"text_coverage_verdict": "unknown"}, id="coverage-unknown"),
        pytest.param({"text_coverage_verdict": ""}, id="coverage-not-run"),
        pytest.param({"prior_ocr_layer_suspect": True}, id="prior-ocr-layer"),
    ],
)
def test_each_missing_piece_of_evidence_keeps_the_reworded_mark(missing):
    """The control arms: the SAME page, one piece of evidence removed at a time."""
    out = _native()
    assert private_use_bullets_released(out, _page()) is True, "premise"
    page = _page(**missing)
    assert private_use_bullets_released(out, page) is False
    clause = _native_glyph_clause(page, out)
    assert clause is not None
    assert clause.check is Check.NATIVE_GLYPHS and clause.result is CheckResult.UNKNOWN
    assert clause.reason == BULLET_GLYPHS_REASON
    assert "mathematics" not in clause.reason


def test_a_bullet_that_still_ships_as_a_private_use_codepoint_is_not_released():
    """tealbook-partb 2018-05-02 p28 (a layer that repeats a paragraph four times) and three
    tealbook-parta pages are hand-graded defective, carry coverage PASS, and ship raw bullets:
    the evidence rule alone released all four. Same page and evidence; only the shipped bytes
    differ."""
    assert private_use_bullets_released(_native(LAYER), _page()) is True
    raw = _native(LAYER + f"\n{BULLET} first item")
    assert private_use_bullets_released(raw, _page()) is False
    clause = _native_glyph_clause(_page(), raw)
    assert clause is not None and clause.reason == BULLET_GLYPHS_REASON


def test_maths_keeps_the_maths_claim_and_is_never_released():
    page = _page(private_use_bullets_only=False)
    out = _native()
    assert private_use_bullets_released(out, page) is False
    clause = _native_glyph_clause(page, out)
    assert clause is not None and clause.result is CheckResult.FAIL
    assert "unrecovered mathematics" in clause.reason


def test_a_control_byte_in_the_text_is_a_fail_entry_and_blocks_the_release():
    out = _native(LAYER + "\x03")
    assert private_use_bullets_released(out, _page()) is False


def test_the_native_evidence_is_digest_bound_to_the_shipped_text():
    """A PASS computed on other text is dropped, so the evidence is gone."""
    page = _page()
    entry = stamp_text_digest(native_coverage_outcome(_native(), page).entry, LAYER)
    assert independent_native_evidence([entry], LAYER) is True
    assert independent_native_evidence([entry], LAYER + " edited") is False


def test_a_model_reading_needs_independent_page_evidence():
    page = _page()
    assert private_use_bullets_released(_model(), page) is True
    # the numbers it ships are not in the layer: the numbers check FAILs, no evidence
    assert (
        private_use_bullets_released(_model("Output grew 9.9 percent in the quarter"), page)
        is False
    )
    # the reading drops a word the layer prints: the words check FAILs
    assert private_use_bullets_released(_model("Output grew 2.1 in the quarter"), page) is False


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


def _run(
    tmp_path: Path,
    *,
    text: str,
    bullets_only: bool,
    coverage: bool,
    providers: list,
    fresh_flags: bool = False,
):
    """The real agentic loop and assemble on ONE native page. Only the arguments differ between
    arms. Engines pinned; the ladder and the judge model supplied explicitly."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "doc.pdf"
    if not pdf.exists():  # a resume reads the SAME file: its checksum is part of the ledger gate
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
    if (
        fresh_flags
    ):  # a resumed run: nothing of the page's analysis is recomputed, only its sidecar speaks
        state.pages[1].has_unmapped_math_glyphs = False
        state.pages[1].private_use_bullets_only = False
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
    _NOTES[id(state)] = (
        list(rec.output.audit_notes or []),
        [e.detail for e in state.events if e.kind == UNRESOLVED_MATH_KIND],
    )
    return rec.output.status, clauses, rec.output.text, result.status


_NOTES: dict = {}


CLEAN = _PROSE
WITH_BULLET = f"{_PROSE}\n{BULLET} first list item"


@pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []], ids=["provider", "no-provider"])
def test_the_shipped_page_differs_only_by_the_evidence(tmp_path, providers):
    """Both provider states. Arm A has the evidence (no private-use byte ships, the ink check ran
    and passed); B ships the raw bullet; C has the same bytes as A with the ink check off; D is a
    maths page (not bullets only). Differences, not absolute tuples."""
    arms, notes = {}, {}
    for name, kw in {
        "evidence": dict(text=CLEAN, bullets_only=True, coverage=True),
        "raw-bullet": dict(text=WITH_BULLET, bullets_only=True, coverage=True),
        "no-ink-check": dict(text=CLEAN, bullets_only=True, coverage=False),
        "maths": dict(text=CLEAN, bullets_only=False, coverage=True),
    }.items():
        state, result, _ = _run(tmp_path / name, providers=providers, **kw)
        arms[name] = _outcome(state, result)
        notes[name] = _NOTES[id(state)]

    def glyph_marks(arm):
        return [c for c in arms[arm][1] if c[0] == "native_glyphs"]

    assert arms["evidence"][0] is PageStatus.SUCCESS and glyph_marks("evidence") == []
    for name in ("raw-bullet", "no-ink-check"):
        assert arms[name][0] is PageStatus.WARNING, name
        assert [c[1:] for c in glyph_marks(name)] == [("UNKNOWN", BULLET_GLYPHS_REASON)], name
    assert arms["maths"][0] is PageStatus.WARNING
    assert [c[1] for c in glyph_marks("maths")] == ["FAIL"]
    assert "unrecovered mathematics" in glyph_marks("maths")[0][2]
    # the released page and the reworded page ship the same bytes apart from the mark block
    assert_differs_only_by_marks(arms["no-ink-check"][2], arms["evidence"][2], 1)
    # the page note and the document event say what is true: bullets, not mathematics
    for name in ("raw-bullet", "no-ink-check"):
        page_notes, doc_events = notes[name]
        assert any("bullet glyphs" in n and "not otherwise checked" in n for n in page_notes), name
        assert doc_events and all("bullet glyphs" in d for d in doc_events), name
    assert notes["evidence"] == ([], []), "released: no demotion note, no document event"
    maths_notes, maths_events = notes["maths"]
    assert any("unmapped math glyphs" in n for n in maths_notes)
    assert maths_events and not any("bullet glyphs" in d for d in maths_events)
    # the document follows the pages: only the evidence arm is clean of the maths accounting
    assert arms["evidence"][3] is not arms["no-ink-check"][3]


def test_the_sidecar_records_the_bullet_fact_with_the_flag(tmp_path):
    _state_, _result, out_dir = _run(
        tmp_path, text=WITH_BULLET, bullets_only=True, coverage=True, providers=[PROFILE_QWEN_LOCAL]
    )
    meta = json.loads(next(out_dir.rglob("00001.json")).read_text())
    assert meta["has_unmapped_math_glyphs"] is True
    assert meta["private_use_bullets_only"] is True
    # maths: the flag is there, the bullet fact is not (sparse: other sidecars keep their bytes)
    _s2, _r2, out2 = _run(
        tmp_path / "maths",
        text=CLEAN,
        bullets_only=False,
        coverage=True,
        providers=[PROFILE_QWEN_LOCAL],
    )
    maths_meta = json.loads(next(out2.rglob("00001.json")).read_text())
    assert maths_meta["has_unmapped_math_glyphs"] is True
    assert "private_use_bullets_only" not in maths_meta


@pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []], ids=["provider", "no-provider"])
def test_a_released_page_stays_released_on_resume_and_its_bullet_fact_is_restored(
    tmp_path, providers
):
    """The page ships SUCCESS, so run 2 skips it from its terminal sidecar. The restored page
    state must carry the bullet fact again (the flag alone would restore a maths claim)."""
    first_state, first, _ = _run(
        tmp_path, text=CLEAN, bullets_only=True, coverage=True, providers=providers
    )
    assert _outcome(first_state, first)[0] is PageStatus.SUCCESS, "premise"
    again_state, again, _ = _run(
        tmp_path,
        text=CLEAN,
        bullets_only=True,
        coverage=True,
        providers=providers,
        fresh_flags=True,
    )
    ps = again_state.pages[1]
    assert ps.has_unmapped_math_glyphs is True and ps.private_use_bullets_only is True
    assert _outcome(again_state, again)[:3] == _outcome(first_state, first)[:3]
