"""#140: math-font pages ship trusted-native with no audit of known-lossy math.

`_detect_math_fonts` (born_digital.py) flags a page whose maths is typeset in
math fonts (Computer Modern, STIX, ...) as known-lossy by its OWN docstring:
"subscripts flatten, Greek letters drop, reading order breaks around
equations". That signal feeds `has_math_font_typesetting` (a strict subset of
the broader `has_equations`, which also folds in a raw-LaTeX-string fallback
that extracts fine), which is metadata only -- `needs_ocr_enhancement =
has_corrupt_math` never reads it -- so the page ships SUCCESS with no audit
trail at all unless the equation lane happens to run and happens to cover
every region it finds.

The adjacent PUA class (#165) already gets this treatment via
`native_math_unrecovered`. This file is the mirror for the math-font class:
`native_math_font_unrecovered`, sourced from the equation lane's OWN
region-outcome evidence (`PageState.equation_region_evidence`) rather than
from a residual-glyph count, because there is no PUA byte signal for this
damage class.

Sections:
  (a) pure accounting (`math_font_unrecovered_detail`) -- outcome-based, not
      configuration-based, and the no-double-report guard against #165's own
      event (criterion 3).
  (b) the manifest reporting guard (`_apply_math_font_unrecovered_guard`).
  (c) the real agentic lane end to end: covered vs uncovered, the flags-off
      case, and persistence into the page sidecar (criterion 1).

Hermetic: `_available_engines_for_agentic` is patched explicitly on every
pipeline instance, `_resolve_judge_model` returns "", `route_page` is asserted
un-called, and the only equation "model" is a deterministic spy patched at
`socr.math.equation_latex.latex_for_crop`. No ollama, no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.result import DocumentStatus, PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState, PageState  # noqa: E402
from socr.math.accounting import (  # noqa: E402
    MATH_FONT_UNRECOVERED_KIND,
    UNRESOLVED_MATH_KIND,
    math_font_unrecovered_detail,
)

# ---------------------------------------------------------------------------
# (a) Pure accounting: outcome-based, not configuration-based.
# ---------------------------------------------------------------------------


def test_no_math_signal_is_untouched() -> None:
    assert (
        math_font_unrecovered_detail(
            has_math_font_typesetting=False,
            has_corrupt_math=False,
            has_unmapped_math_glyphs=False,
            evidence=None,
        )
        is None
    )


def test_a_generic_equation_page_with_no_font_signal_is_untouched() -> None:
    """The `has_equations` union also includes a raw-LaTeX-string fallback
    (`_detect_equations`) that extracts as ordinary text -- nothing lossy.
    Gating on that union instead of the font-metadata signal alone regressed
    #269 (BLOCKING 1's "equation-only pages ship unaffected by S1"); the
    guard must never demote a page whose ONLY equation signal is the generic
    one, `has_math_font_typesetting` False."""
    from socr.core.manifest import _apply_math_font_unrecovered_guard

    out = PageOutput(page_num=1, text="prose", status=PageStatus.SUCCESS, engine="native")
    p = PageState(
        page_num=1,
        is_born_digital=True,
        native_text="prose",
        has_equations=True,
        has_math_font_typesetting=False,
    )

    result = _apply_math_font_unrecovered_guard(out, p)

    assert result is out


def test_corrupt_math_pages_are_not_double_reported() -> None:
    """A font-map-corrupted page already has its own recovery lane and its own
    AUDIT_FAILED bucket (`corrupt_math_hybrid_pages`). It must not also raise
    this event for the same maths."""
    assert (
        math_font_unrecovered_detail(
            has_math_font_typesetting=True,
            has_corrupt_math=True,
            has_unmapped_math_glyphs=False,
            evidence=None,
        )
        is None
    )


def test_pua_pages_are_not_double_reported() -> None:
    """Criterion 3: a page already carrying `UNRESOLVED_MATH_KIND` for PUA
    damage must not also raise this event for the same maths."""
    assert (
        math_font_unrecovered_detail(
            has_math_font_typesetting=True,
            has_corrupt_math=False,
            has_unmapped_math_glyphs=True,
            evidence=None,
        )
        is None
    )


def test_no_evidence_at_all_is_unresolved() -> None:
    """Criterion 1: recovery flags off (the lane never ran, so it never
    recorded an attempt) still emits a durable record naming the loss."""
    detail = math_font_unrecovered_detail(
        has_math_font_typesetting=True,
        has_corrupt_math=False,
        has_unmapped_math_glyphs=False,
        evidence=None,
    )
    assert detail is not None
    assert "no equation-lane recovery evidence" in detail.reason
    assert detail.regions_total == 0
    assert MATH_FONT_UNRECOVERED_KIND == "native_math_font_unrecovered"


def test_lane_ran_but_located_nothing_is_unresolved() -> None:
    detail = math_font_unrecovered_detail(
        has_math_font_typesetting=True,
        has_corrupt_math=False,
        has_unmapped_math_glyphs=False,
        evidence={"regions_total": 0, "regions_covered": 0},
    )
    assert detail is not None
    assert "located no display-equation region" in detail.reason


def test_partial_coverage_is_unresolved() -> None:
    detail = math_font_unrecovered_detail(
        has_math_font_typesetting=True,
        has_corrupt_math=False,
        has_unmapped_math_glyphs=False,
        evidence={"regions_total": 3, "regions_covered": 1},
    )
    assert detail is not None
    assert "2 of 3" in detail.reason
    assert detail.regions_total == 3
    assert detail.regions_covered == 1


def test_full_coverage_clears_the_signal() -> None:
    """Criterion 2: turning the recovery flags on does not by itself silence
    the event -- only actual coverage does. Same evidence shape, only the
    covered count changes."""
    assert (
        math_font_unrecovered_detail(
            has_math_font_typesetting=True,
            has_corrupt_math=False,
            has_unmapped_math_glyphs=False,
            evidence={"regions_total": 2, "regions_covered": 2},
        )
        is None
    )


def test_the_outcome_is_the_same_whether_or_not_flags_were_on() -> None:
    """The #165 falsifier, restated for this class: identical (zero) coverage
    must report identically regardless of what produced it -- the lane never
    running (flags off) or the lane running and finding nothing (flags on, no
    region located)."""
    flags_off = math_font_unrecovered_detail(
        has_math_font_typesetting=True,
        has_corrupt_math=False,
        has_unmapped_math_glyphs=False,
        evidence=None,
    )
    flags_on_no_region = math_font_unrecovered_detail(
        has_math_font_typesetting=True,
        has_corrupt_math=False,
        has_unmapped_math_glyphs=False,
        evidence={"regions_total": 0, "regions_covered": 0},
    )
    assert flags_off is not None and flags_on_no_region is not None
    assert flags_off.reason != flags_on_no_region.reason, (
        "the two causes are distinguishable in the record"
    )
    assert flags_off.regions_total == flags_on_no_region.regions_total == 0


# ---------------------------------------------------------------------------
# (b) The manifest reporting guard.
# ---------------------------------------------------------------------------


def _page_state(**overrides) -> PageState:
    base = dict(
        page_num=1,
        is_born_digital=True,
        native_text="prose",
        has_math_font_typesetting=True,
        has_corrupt_math=False,
        has_unmapped_math_glyphs=False,
        equation_region_evidence=None,
    )
    base.update(overrides)
    return PageState(**base)


def test_guard_notes_the_loss_without_demoting_status() -> None:
    """Reports, does not demote: the measured trigger rate (36.1% of the free
    lane, ~15x the PUA class this mirrors, per docs/log/2026-09-02_p4m-
    trigger-rates.md) makes an unconditional demotion an unclearable
    AUDIT_FAILED for the 8.0% slice with no display equation to locate."""
    from socr.core.manifest import _apply_math_font_unrecovered_guard

    out = PageOutput(page_num=1, text="prose", status=PageStatus.SUCCESS, engine="native")
    p = _page_state()

    result = _apply_math_font_unrecovered_guard(out, p)

    assert result.status is PageStatus.SUCCESS
    assert any("math-font typesetting" in n for n in result.audit_notes)


def test_guard_leaves_a_covered_page_untouched() -> None:
    from socr.core.manifest import _apply_math_font_unrecovered_guard

    out = PageOutput(page_num=1, text="prose", status=PageStatus.SUCCESS, engine="native")
    p = _page_state(equation_region_evidence={"regions_total": 1, "regions_covered": 1})

    result = _apply_math_font_unrecovered_guard(out, p)

    assert result.status is PageStatus.SUCCESS
    assert result.audit_notes == out.audit_notes


def test_guard_leaves_a_math_free_page_untouched() -> None:
    from socr.core.manifest import _apply_math_font_unrecovered_guard

    out = PageOutput(page_num=1, text="prose", status=PageStatus.SUCCESS, engine="native")
    p = _page_state(has_math_font_typesetting=False)

    result = _apply_math_font_unrecovered_guard(out, p)

    assert result is out


def test_guard_never_touches_status_whatever_it_starts_as() -> None:
    from socr.core.manifest import _apply_math_font_unrecovered_guard

    out = PageOutput(page_num=1, text="prose", status=PageStatus.ERROR, engine="native")
    p = _page_state()

    result = _apply_math_font_unrecovered_guard(out, p)

    assert result.status is PageStatus.ERROR
    assert any("math-font typesetting" in n for n in result.audit_notes)


# ---------------------------------------------------------------------------
# (c) The real agentic lane, end to end.
# ---------------------------------------------------------------------------

#: The native text of the math-font page. Its numeric token is the presence
#: oracle for the attach guard, same convention as the P4-R equation-lane
#: pipeline tests.
_EQ_SOURCE = "y = 2x + 1 (3)"
_EQ_NATIVE = f"The result follows directly.\n\n{_EQ_SOURCE}\n\nWe conclude in section 4."
#: A reading whose numbers are all present on the page -- the attach guard's OK arm.
_LATEX_CONTAINED = r"y = 2x + 1 \tag{3}"


@pytest.fixture(autouse=True)
def _isolate_backend_resolution(monkeypatch):
    """Same isolation as the P4-R lane suite: `qwen_backend` defaults to
    `auto`, which resolves to vLLM whenever `VLLM_BASE_URL` is exported."""
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)


def _make_pdf(tmp_path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "The result follows directly.", fontsize=11)
    page.insert_text((240, 200), _EQ_SOURCE, fontsize=11)
    page.insert_text((72, 300), "We conclude in section 4.", fontsize=11)
    path = tmp_path / "mathfont.pdf"
    doc.save(str(path))
    doc.close()
    return path


def _make_pipeline(*, equation_region_lane: bool = True, **overrides):
    from socr.core.config import EngineType, PipelineConfig
    from socr.pipeline.orchestrator import UnifiedPipeline

    cfg = PipelineConfig(
        primary_engine=EngineType.DEEPSEEK,
        enabled_engines=list(EngineType),
        agentic=True,
        quiet=True,
        native_first=True,
        equation_region_lane=equation_region_lane,
        **overrides,
    )
    return UnifiedPipeline(cfg)


def _make_state(pdf_path: Path) -> DocumentState:
    from socr.core.born_digital import DocumentAssessment, PageAssessment
    from socr.core.document import DocumentHandle

    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf_path, page_count=1)
    state = DocumentState(handle=handle)
    state.pages[1] = PageState(
        page_num=1,
        is_born_digital=True,
        native_text=_EQ_NATIVE,
        needs_ocr_enhancement=False,
        has_tables=False,
        has_equations=True,
        has_math_font_typesetting=True,
        has_corrupt_math=False,
        has_unmapped_math_glyphs=False,
    )
    state._last_assessment = DocumentAssessment(
        path=pdf_path,
        pages=[
            PageAssessment(
                page_num=1,
                is_born_digital=True,
                native_text=_EQ_NATIVE,
                confidence=0.9,
                needs_ocr_enhancement=False,
                has_tables=False,
                has_equations=True,
                has_math_font_typesetting=True,
                has_corrupt_math=False,
                has_unmapped_math_glyphs=False,
            )
        ],
    )
    return state


def _region_result():
    from socr.math.detect_equations import EquationDetectionResult, EquationRegion

    bbox = (230.0, 185.0, 400.0, 215.0)
    region = EquationRegion(
        page_num=1,
        bbox=bbox,
        source_bbox=bbox,
        has_eq_number=True,
        equation_label="(3)",
        source_text=_EQ_SOURCE,
    )
    return EquationDetectionResult(page_num=1, regions=[region], detection_time_s=0.0)


class _ModelSpy:
    def __init__(self, latex: str):
        self.latex = latex
        self.calls = 0

    def __call__(self, *args, **kwargs) -> str:
        self.calls += 1
        return self.latex


def _run(
    tmp_path: Path,
    *,
    lane: bool = True,
    provider: bool,
    latex: str = _LATEX_CONTAINED,
    regions: bool = True,
    tag: str = "",
):
    """One hermetic agentic+assemble run over the math-font fixture page."""
    from socr.core.providers import PROFILE_QWEN_LOCAL

    out_root = tmp_path / f"run_{tag or lane}_{provider}"
    out_root.mkdir(parents=True, exist_ok=True)
    pdf = _make_pdf(out_root)
    pipeline = _make_pipeline(equation_region_lane=lane)
    pipeline._scan_root = pdf.parent
    state = _make_state(pdf)
    pipeline._last_assessment = state._last_assessment

    spy = _ModelSpy(latex)

    def _detect(page, page_num):
        from socr.math.detect_equations import EquationDetectionResult

        if not regions:
            return EquationDetectionResult(page_num=page_num, regions=[], detection_time_s=0.0)
        return _region_result()

    out_dir = out_root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("socr.math.detect_equations.detect_display_equations", side_effect=_detect),
        patch("socr.math.equation_latex.latex_for_crop", side_effect=spy),
        patch("socr.pipeline.orchestrator.route_page") as route,
        patch.object(
            pipeline,
            "_available_engines_for_agentic",
            return_value=[PROFILE_QWEN_LOCAL] if provider else [],
        ),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
    ):
        pipeline._phase_agentic(state, out_dir)
        result = pipeline._phase_assemble(state, out_dir)
        route.assert_not_called()

    return pipeline, state, result, out_dir


def _math_font_events(state: DocumentState) -> list:
    return [e for e in state.events if getattr(e, "kind", "") == MATH_FONT_UNRECOVERED_KIND]


def test_fully_covered_page_raises_no_event(tmp_path: Path) -> None:
    """The lane locates the one region, the reading is accepted and aligned:
    covered, so no event and the page is not demoted for this reason."""
    pipeline, state, result, _ = _run(tmp_path, provider=True, tag="covered")

    assert _math_font_events(state) == []
    assert "math-font typesetting" not in (result.error or "")
    assert not any("math-font typesetting" in n for n in result.audit_notes)
    from socr.core.manifest import finalized_page_record

    assert finalized_page_record(state, 1).output.status is PageStatus.SUCCESS


def test_no_provider_leaves_the_math_uncovered_and_reported(tmp_path: Path) -> None:
    """With no provider the lane locates the region but every call is skipped
    (`equation_region_reading_unvalidated`): uncovered, so the event fires and
    the document note names it. Status stays SUCCESS -- the demotion was
    considered and rejected on measured blast radius (see the decision log);
    this is "reported and visible", not "fails the document"."""
    pipeline, state, result, out_dir = _run(tmp_path, provider=False, tag="noprovider")

    events = _math_font_events(state)
    assert len(events) == 1
    assert events[0].page_num == 1
    assert events[0].data["regions_total"] == 1
    assert events[0].data["regions_covered"] == 0
    assert state.status is DocumentStatus.SUCCESS
    assert "math-font typesetting" not in (result.error or ""), (
        "the note must never land in `error` -- that field is load-bearing "
        "(cli.py greps it for LOST_CONTENT_NOTE, GH-177 documents it as "
        "'already AUDIT_FAILED'); a populated error on a success=True result "
        "would smuggle the full failure blast radius back in through a field "
        "a reasonable caller checks before status"
    )
    assert any("math-font typesetting" in n for n in result.audit_notes)

    from socr.core.manifest import finalized_page_record

    assert finalized_page_record(state, 1).output.status is PageStatus.SUCCESS

    # Criterion 1: the event survives into the page sidecar.
    sidecar = json.loads(next(out_dir.rglob("pages/00001.json")).read_text())
    kinds = [e.get("kind") for e in sidecar.get("audit_events", [])]
    assert MATH_FONT_UNRECOVERED_KIND in kinds


def test_lane_off_and_no_evidence_is_reported_the_same_way(tmp_path: Path) -> None:
    """Criterion 1, the literal case: the equation lane disabled entirely (no
    `--equation-region-lane`) means no evidence is EVER retained for this
    page. The event must still fire -- this is the exact silent-by-default
    gap #140 exists to close -- even though the document itself still reports
    SUCCESS (the demotion is deliberately withheld)."""
    pipeline, state, result, _ = _run(tmp_path, lane=False, provider=True, tag="laneoff")

    events = _math_font_events(state)
    assert len(events) == 1
    assert events[0].data["regions_total"] == 0
    assert state.status is DocumentStatus.SUCCESS


def test_covered_vs_uncovered_is_the_pinned_difference(tmp_path: Path) -> None:
    """Repo convention: pin the DIFFERENCE the change under test controls, not
    an absolute value that provider-dependent machinery could legitimately
    vary in CI. Here both runs are fully hermetic (provider ladder is an
    explicit parameter), so the difference is exact: the event fires or not,
    document status is unaffected either way (see the decision log)."""
    _, covered_state, _, _ = _run(tmp_path, provider=True, tag="diff_covered")
    _, uncovered_state, _, _ = _run(tmp_path, provider=False, tag="diff_uncovered")

    assert _math_font_events(covered_state) == []
    assert len(_math_font_events(uncovered_state)) == 1
    assert covered_state.status is DocumentStatus.SUCCESS
    assert uncovered_state.status is DocumentStatus.SUCCESS


def test_a_pua_page_never_also_raises_the_math_font_event(tmp_path: Path) -> None:
    """Criterion 3, end to end: a page carrying BOTH signals (math-font
    typesetting detected AND unmapped/PUA glyphs) raises only
    `UNRESOLVED_MATH_KIND`, never `MATH_FONT_UNRECOVERED_KIND` too."""
    pipeline, state, _, _ = _run(tmp_path, provider=False, tag="pua_overlap")
    state.pages[1].has_unmapped_math_glyphs = True
    state._last_assessment.pages[0].has_unmapped_math_glyphs = True

    # Re-finalize with the overlap signal set, mirroring how `_phase_assemble`
    # reconciles per page.
    from socr.core.manifest import finalized_page_record

    finalized_page_record(state, 1)

    assert (
        math_font_unrecovered_detail(
            has_math_font_typesetting=True,
            has_corrupt_math=False,
            has_unmapped_math_glyphs=True,
            evidence=state.pages[1].equation_region_evidence,
        )
        is None
    )


def test_the_event_and_the_note_survive_a_resume(tmp_path: Path) -> None:
    """A resumed page is not re-assessed and not re-recovered (PP-5): the
    equation lane never runs a second time, so the ONLY way an uncovered
    page's record can still be visible on resume is if the sidecar-persisted
    ``equation_region_evidence`` and the ``MATH_FONT_UNRECOVERED_KIND`` event
    are both actually restored -- neither is exercised by any test above.
    Pins the identical event and page-level outcome (still SUCCESS, per the
    criterion-4 decision) across run 1 -> resume."""
    from socr.core.document import DocumentHandle
    from socr.core.manifest import finalized_page_record

    pipeline, state, result, out_dir = _run(tmp_path, provider=False, tag="resume")
    assert _math_font_events(state)
    assert result.status is DocumentStatus.SUCCESS

    sidecar = json.loads(next(out_dir.rglob("pages/00001.json")).read_text())
    assert sidecar.get("equation_region_evidence") == {
        "regions_total": 1,
        "regions_covered": 0,
    }
    assert MATH_FONT_UNRECOVERED_KIND in [e.get("kind") for e in sidecar["audit_events"]]

    resumed = DocumentState(handle=DocumentHandle.from_path(state.handle.path))
    assert not resumed.events, "nothing restored yet; the test would be vacuous otherwise"
    page_out = PageOutput(
        page_num=1,
        text=_EQ_NATIVE,
        status=PageStatus.SUCCESS,
        engine="native",
        audit_passed=True,
    )
    pipeline._restore_terminal_page_state(resumed, 1, page_out, out_dir)

    assert resumed.pages[1].equation_region_evidence == {
        "regions_total": 1,
        "regions_covered": 0,
    }
    resumed_events = _math_font_events(resumed)
    assert len(resumed_events) == 1, "the record vanished (or duplicated) on resume"
    assert (
        resumed_events[0].data
        == sidecar["audit_events"][
            [e.get("kind") for e in sidecar["audit_events"]].index(MATH_FONT_UNRECOVERED_KIND)
        ]["data"]
    )
    assert finalized_page_record(resumed, 1).output.status is PageStatus.SUCCESS
