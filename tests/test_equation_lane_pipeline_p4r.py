"""P4-R: the equation-region lane driven through the agentic loop.

Covers the acceptance criteria the tests stage left unwritten because the
executor's signature did not exist yet:

  * the executor's failure floors and its one accepting arm (plan t5 / critique
    t6-t7): zero regions, no provider, model refusal, 1A failure, presence
    rejection, encoding-suspect abstain, unaligned slice, and an accepted
    reading;
  * the difference pin, lane OFF vs ON, over BOTH provider states (plan t6 /
    critique t8);
  * the presence guard's rejection path end to end (plan t7);
  * the P2 structure-class floor's unreachability on a table-free page with the
    lane on and every reading refused (plan t8 / critique t9);
  * provisional/final fragment byte identity and resume semantics (plan t9 /
    critique t10).

Method, per CLAUDE.md and #257: every assertion is a DIFFERENCE against the
same fixture run with the lane off, or against the page's own native text --
never an absolute (status, audit_passed, failure_mode) tuple measured on this
machine. Both provider states are parametrised.

Hermeticity: `_available_engines_for_agentic` and `_resolve_judge_model` are
patched on the pipeline, `route_page` is stubbed, and the equation model is
injected by patching `socr.math.equation_latex.latex_for_crop`. No ollama, no
network, no real VLM. `detect_display_equations` is injected too: the point
under test is the lane, not the detector (which has its own suite), and text
inserted with fitz's default font does not reliably trip the real math-font
geometry.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

# The native text of the equation page. Its numeric tokens are the presence
# oracle, so the readings below are written against it deliberately.
_EQ_SOURCE = "y = 2x + 1 (3)"
_EQ_NATIVE = f"The result follows directly.\n\n{_EQ_SOURCE}\n\nWe conclude in section 4."

#: A reading whose numbers are all present on the page -- the guard's OK arm.
_LATEX_CONTAINED = r"y = 2x + 1 \tag{3}"
#: A reading carrying a number that is nowhere on the page -- the guard's FAIL
#: arm, and the whole reason the guard exists.
_LATEX_INVENTED = r"y = 2x + 0.9137 \tag{3}"


@pytest.fixture(autouse=True)
def _isolate_backend_resolution(monkeypatch):
    """Decide backend resolution here, never inherit it from the shell.

    Cold review round 4, N3 -- and wider than the finding scoped it. The lane
    authorises a provider by the backend it RESOLVES to, and `qwen_backend`
    defaults to `auto`, which means vLLM whenever `VLLM_BASE_URL` is exported.
    That is the documented HPC deployment, so on such a machine EVERY test in
    this file that expects a model call was failing (22 of them) while
    production was behaving correctly. The variable is cleared for the module;
    the one test that cares about the exported state sets it back explicitly.
    """
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)


# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------


def _make_pdf(tmp_path: Path) -> Path:
    """Three born-digital pages: prose, a display equation, a table."""
    doc = fitz.open()

    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Introductory prose that carries no mathematics.", fontsize=11)

    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "The result follows directly.", fontsize=11)
    page.insert_text((240, 200), _EQ_SOURCE, fontsize=11)
    page.insert_text((72, 300), "We conclude in section 4.", fontsize=11)

    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Table 1", fontsize=11)
    y = 130
    for row in range(6):
        page.insert_text((72, y), f"Row{row}   {row}.10   {row}.20", fontsize=9)
        y += 16

    path = tmp_path / "p4r.pdf"
    doc.save(str(path))
    doc.close()
    return path


def _make_pipeline(*, equation_region_lane: bool, **overrides):
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


def _make_state(pdf_path: Path):
    """A DocumentState whose page 2 carries the equation signal and no table."""
    from socr.core.born_digital import DocumentAssessment, PageAssessment
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState, PageState

    natives = {
        1: "Introductory prose that carries no mathematics.",
        2: _EQ_NATIVE,
        3: "Table 1\n\n| Row | A | B |\n| --- | --- | --- |\n| 0 | 0.10 | 0.20 |",
    }
    flags = {
        1: dict(has_tables=False, has_equations=False),
        2: dict(has_tables=False, has_equations=True),
        3: dict(has_tables=True, has_equations=False),
    }

    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf_path, page_count=3)
    state = DocumentState(handle=handle)
    for pn in (1, 2, 3):
        state.pages[pn] = PageState(
            page_num=pn,
            is_born_digital=True,
            native_text=natives[pn],
            needs_ocr_enhancement=False,
            **flags[pn],
        )
    state._last_assessment = DocumentAssessment(
        path=pdf_path,
        pages=[
            PageAssessment(
                page_num=pn,
                is_born_digital=True,
                native_text=natives[pn],
                confidence=0.9,
                needs_ocr_enhancement=False,
                **flags[pn],
            )
            for pn in (1, 2, 3)
        ],
    )
    return state


def _region(page, *, source_text: str = _EQ_SOURCE):
    """One injected display-equation region with a real, renderable bbox."""
    from socr.math.detect_equations import EquationDetectionResult, EquationRegion

    bbox = (230.0, 185.0, 400.0, 215.0)
    region = EquationRegion(
        page_num=2,
        bbox=bbox,
        source_bbox=bbox,
        has_eq_number=True,
        equation_label="(3)",
        source_text=source_text,
    )
    return EquationDetectionResult(page_num=2, regions=[region], detection_time_s=0.0)


class _ModelSpy:
    """Stands in for the equation VLM. Records every call it receives."""

    def __init__(self, latex: str):
        self.latex = latex
        self.calls = 0

    def __call__(self, *args, **kwargs) -> str:
        self.calls += 1
        return self.latex


def _run(
    tmp_path: Path,
    *,
    lane: bool,
    provider: bool,
    latex: str = _LATEX_CONTAINED,
    regions: bool = True,
    source_text: str = _EQ_SOURCE,
    detect_raises: bool = False,
    pipeline=None,
    out_root: Path | None = None,
):
    """One hermetic agentic+assemble run. Returns (state, result, spy, out_dir)."""
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.agentic import PageDecision

    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = _make_pdf(tmp_path)
    pipeline = pipeline or _make_pipeline(equation_region_lane=lane)
    state = _make_state(pdf)
    pipeline._last_assessment = state._last_assessment

    spy = _ModelSpy(latex)

    def _detect(page, page_num):
        from socr.math.detect_equations import EquationDetectionResult

        if detect_raises:
            raise RuntimeError("injected detector failure")
        if page_num != 2 or not regions:
            return EquationDetectionResult(page_num=page_num, regions=[], detection_time_s=0.0)
        return _region(page, source_text=source_text)

    def _routed(page_num, *args, **kwargs):
        out = PageOutput(
            page_num=page_num,
            text="| Row | A | B |\n| --- | --- | --- |\n| 0 | 0.10 | 0.20 |",
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        return PageDecision(page_num=page_num, final_output=out, accepted=True)

    out_dir = (out_root or tmp_path) / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("socr.math.detect_equations.detect_display_equations", side_effect=_detect),
        patch("socr.math.equation_latex.latex_for_crop", side_effect=spy),
        patch("socr.pipeline.orchestrator.route_page", side_effect=_routed),
        patch.object(
            pipeline,
            "_available_engines_for_agentic",
            return_value=[PROFILE_QWEN_LOCAL] if provider else [],
        ),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
    ):
        pipeline._phase_agentic(state, out_dir)
        result = pipeline._phase_assemble(state, out_dir)
    return state, result, spy, out_dir


def _page_view(state, page_num: int) -> dict:
    """The observable bundle compared between runs -- never asserted absolutely."""
    bo = state.pages[page_num].best_output
    return {
        "text": bo.text if bo else None,
        "engine": bo.engine if bo else None,
        "status": bo.status if bo else None,
        "failure_mode": bo.failure_mode if bo else None,
        "audit_passed": bo.audit_passed if bo else None,
    }


def _kinds(state, page_num: int) -> set[str]:
    return {e.kind for e in state.events if e.page_num == page_num}


# ---------------------------------------------------------------------------
# The difference pin: lane off vs on, both provider states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", [True, False], ids=["with-provider", "no-provider"])
def test_only_the_equation_page_changes_route(tmp_path: Path, provider: bool) -> None:
    """Exactly page 2 moves; pages 1 and 3 are untouched in both provider states."""
    off, _, off_spy, _ = _run(tmp_path / "off", lane=False, provider=provider)
    on, _, on_spy, _ = _run(tmp_path / "on", lane=True, provider=provider)

    moved = {pn for pn in (1, 2, 3) if _page_view(off, pn) != _page_view(on, pn)}
    if provider:
        assert moved == {2}
    else:
        # With no provider nothing may move at all: the page must ship exactly
        # as it does today, which is the stricter half of the same pin.
        assert moved == set()

    for pn in (1, 3):
        assert _page_view(off, pn) == _page_view(on, pn)

    # The model is never consulted with the lane off, in either provider state.
    assert off_spy.calls == 0


def test_no_provider_means_no_model_call_and_no_new_failure_surface(tmp_path: Path) -> None:
    """CI's exact state: the page ships the lane-off bytes and nothing is stamped."""
    off, off_result, _, _ = _run(tmp_path / "off", lane=False, provider=False)
    on, on_result, spy, _ = _run(tmp_path / "on", lane=True, provider=False)

    assert spy.calls == 0
    assert _page_view(on, 2) == _page_view(off, 2)
    assert on_result.status == off_result.status
    # The skip is recorded rather than silent, but it changes nothing shipped.
    assert "equation_region_reading_unvalidated" in _kinds(on, 2)


def test_accepted_reading_is_additive_and_never_replaces_prose(tmp_path: Path) -> None:
    """Native prose survives verbatim; the reading arrives beside its own region."""
    from socr.math.equation_latex import EQUATION_SIDECAR_HEADER

    off, _, _, _ = _run(tmp_path / "off", lane=False, provider=True)
    on, _, spy, _ = _run(tmp_path / "on", lane=True, provider=True)

    assert spy.calls == 1
    native = off["text"] if isinstance(off, dict) else _page_view(off, 2)["text"]
    attached = _page_view(on, 2)["text"]

    assert attached != native
    # Every line of the lane-off text survives, in order.
    prev = -1
    for line in [line for line in native.splitlines() if line.strip()]:
        found = attached.find(line)
        assert found > prev, f"native line lost or reordered: {line!r}"
        prev = found
    assert EQUATION_SIDECAR_HEADER in attached
    assert "```latex" in attached
    assert "equations/" in attached
    # In place, not appended at the page end.
    assert attached.index(EQUATION_SIDECAR_HEADER) < attached.index("We conclude in section 4.")
    assert _page_view(on, 2)["engine"] != _page_view(off, 2)["engine"]
    assert "equation_region_reading_attached" in _kinds(on, 2)


def test_status_and_audit_flag_are_not_moved_by_an_accepted_reading(tmp_path: Path) -> None:
    """audit_passed is winner selection: the lane must never touch it."""
    off, off_result, _, _ = _run(tmp_path / "off", lane=False, provider=True)
    on, on_result, _, _ = _run(tmp_path / "on", lane=True, provider=True)

    for key in ("status", "failure_mode", "audit_passed"):
        assert _page_view(on, 2)[key] == _page_view(off, 2)[key]
    assert on_result.status == off_result.status


# ---------------------------------------------------------------------------
# The presence guard as a rejection guard
# ---------------------------------------------------------------------------


def test_invented_number_rejects_the_reading_without_demoting_the_page(
    tmp_path: Path,
) -> None:
    off, off_result, _, _ = _run(tmp_path / "off", lane=False, provider=True)
    on, on_result, spy, _ = _run(tmp_path / "on", lane=True, provider=True, latex=_LATEX_INVENTED)

    assert spy.calls == 1
    # Nothing attached: byte-for-byte the lane-off page.
    assert _page_view(on, 2) == _page_view(off, 2)
    assert on_result.status == off_result.status

    rejected = [
        e for e in on.events if e.page_num == 2 and e.kind == "equation_region_reading_rejected"
    ]
    assert len(rejected) == 1
    assert "0.9137" in rejected[0].data["invented"]
    assert rejected[0].data["presence_reason"]
    assert "equation_region_reading_attached" not in _kinds(on, 2)


def test_encoding_suspect_abstains_and_a_numeric_reading_is_not_attached(tmp_path: Path) -> None:
    """Ruling 4 holds, and GH-522 narrows what the abstention permits.

    The page still ABSTAINS rather than FAILs: it is not rejected, not demoted,
    and keeps its native prose. What changed is that an abstention no longer
    licenses ATTACHING a reading whose numbers nobody could check -- a
    crop-backed LaTeX sidecar sat beside the authoritative native slice carrying
    possibly-invented values, tagged only by `presence_status` on an audit event
    the .md never shows.

    Scoped to readings that HAVE numbers; the number-free case below still
    attaches, because there is nothing to invent.
    """
    pipeline = _make_pipeline(equation_region_lane=True)
    pdf_dir = tmp_path / "suspect"

    # Same invented number as the rejection case above -- only the page's
    # encoding-hygiene flag differs, so the abstain is what changed the outcome.
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.agentic import PageDecision

    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf = _make_pdf(pdf_dir)
    state = _make_state(pdf)
    state.pages[2].has_encoding_hygiene_suspect = True
    pipeline._last_assessment = state._last_assessment
    spy = _ModelSpy(_LATEX_INVENTED)

    def _detect(page, page_num):
        from socr.math.detect_equations import EquationDetectionResult

        if page_num != 2:
            return EquationDetectionResult(page_num=page_num, regions=[], detection_time_s=0.0)
        return _region(page)

    def _routed(page_num, *args, **kwargs):
        return PageDecision(
            page_num=page_num,
            final_output=PageOutput(
                page_num=page_num, text="x", status=PageStatus.SUCCESS, engine="qwen"
            ),
            accepted=True,
        )

    out_dir = pdf_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (
        patch("socr.math.detect_equations.detect_display_equations", side_effect=_detect),
        patch("socr.math.equation_latex.latex_for_crop", side_effect=spy),
        patch("socr.pipeline.orchestrator.route_page", side_effect=_routed),
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
    ):
        pipeline._phase_agentic(state, out_dir)
        pipeline._phase_assemble(state, out_dir)

    attached = [
        e for e in state.events if e.page_num == 2 and e.kind == "equation_region_reading_attached"
    ]
    assert attached == [], (
        "an unchecked numeric reading was attached to the page under an abstention (GH-522)"
    )

    unverifiable = [
        e
        for e in state.events
        if e.page_num == 2 and e.kind == "equation_region_reading_unverifiable"
    ]
    assert len(unverifiable) == 1, "the refusal must be recorded, not silent"
    assert unverifiable[0].data["presence_status"] == "unverifiable"
    assert unverifiable[0].data["unchecked_values"], "the event does not say WHICH values"
    # The FILE, not the path string (cubic P2 on #537). "the crop stays on disk
    # as evidence" is the promise that makes refusing acceptable -- a reader can
    # still check the reading by hand -- and a non-empty string proves nothing
    # about whether anything was written.
    crop = Path(unverifiable[0].data["crop_path"])
    assert crop.exists(), (
        f"the crop was not written to {crop}; refusing the reading is only "
        "acceptable because the evidence survives"
    )

    # Ruling 4, still: abstain is not reject, and the page is not demoted.
    assert "equation_region_reading_rejected" not in _kinds(state, 2), (
        "an encoding-suspect page was REJECTED; ruling 4 says it abstains"
    )


def test_an_abstaining_page_still_attaches_a_reading_with_no_numbers(tmp_path: Path) -> None:
    """The other half of GH-522, and the reason the refusal is scoped.

    A pure-symbol equation carries nothing that can be invented, so refusing it
    would discard safe LaTeX on exactly the pages where the text layer is
    already damaged -- the "dropped is worse than missing" half of the corpus
    rule. Without this, a blanket refusal on UNVERIFIABLE would satisfy the test
    above.
    """
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.agentic import PageDecision

    pipeline = _make_pipeline(equation_region_lane=True)
    pdf_dir = tmp_path / "suspect_symbols"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf = _make_pdf(pdf_dir)
    state = _make_state(pdf)
    state.pages[2].has_encoding_hygiene_suspect = True
    pipeline._last_assessment = state._last_assessment
    spy = _ModelSpy(r"\alpha + \beta = \gamma")

    def _detect(page, page_num):
        from socr.math.detect_equations import EquationDetectionResult

        if page_num != 2:
            return EquationDetectionResult(page_num=page_num, regions=[], detection_time_s=0.0)
        return _region(page)

    def _routed(page_num, *args, **kwargs):
        return PageDecision(
            page_num=page_num,
            final_output=PageOutput(
                page_num=page_num, text="x", status=PageStatus.SUCCESS, engine="qwen"
            ),
            accepted=True,
        )

    out_dir = pdf_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (
        patch("socr.math.detect_equations.detect_display_equations", side_effect=_detect),
        patch("socr.math.equation_latex.latex_for_crop", side_effect=spy),
        patch("socr.pipeline.orchestrator.route_page", side_effect=_routed),
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
    ):
        pipeline._phase_agentic(state, out_dir)
        pipeline._phase_assemble(state, out_dir)

    attached = [
        e for e in state.events if e.page_num == 2 and e.kind == "equation_region_reading_attached"
    ]
    assert len(attached) == 1, (
        "a number-free reading was refused; nothing in it can be invented, so "
        "the presence guard has no grounds to withhold it"
    )
    assert attached[0].data["presence_status"] == "unverifiable"


# ---------------------------------------------------------------------------
# Executor failure floors
# ---------------------------------------------------------------------------


def test_zero_regions_costs_no_model_call_and_ships_the_floor(tmp_path: Path) -> None:
    off, _, _, _ = _run(tmp_path / "off", lane=False, provider=True)
    on, _, spy, _ = _run(tmp_path / "on", lane=True, provider=True, regions=False)

    assert spy.calls == 0
    assert _page_view(on, 2) == _page_view(off, 2)
    assert "equation_lane_no_region" in _kinds(on, 2)


def test_refused_model_output_ships_the_floor(tmp_path: Path) -> None:
    off, _, _, _ = _run(tmp_path / "off", lane=False, provider=True)
    on, _, spy, _ = _run(tmp_path / "on", lane=True, provider=True, latex="")

    assert spy.calls == 1
    assert _page_view(on, 2) == _page_view(off, 2)
    assert "equation_region_reading_unvalidated" in _kinds(on, 2)


def test_structurally_invalid_latex_ships_the_floor(tmp_path: Path) -> None:
    off, _, _, _ = _run(tmp_path / "off", lane=False, provider=True)
    on, _, _, _ = _run(tmp_path / "on", lane=True, provider=True, latex=r"\frac{1}{ y = 2x + 1")

    assert _page_view(on, 2) == _page_view(off, 2)
    assert "equation_region_reading_unvalidated" in _kinds(on, 2)


def test_unaligned_source_slice_drops_the_reading_and_says_so(tmp_path: Path) -> None:
    """A reading with no anchor is discarded, never appended at the page end."""
    off, _, _, _ = _run(tmp_path / "off", lane=False, provider=True)
    on, _, _, _ = _run(
        tmp_path / "on",
        lane=True,
        provider=True,
        source_text="a slice that is nowhere in the page text",
    )

    assert _page_view(on, 2) == _page_view(off, 2)
    assert "equation_region_reading_unaligned" in _kinds(on, 2)


def test_detector_failure_ships_the_floor(tmp_path: Path) -> None:
    """A detector that raises costs no model call and loses no prose.

    NOTE on the event: the GH-36a seam `_detect_and_crop_equation_page` already
    fails soft -- it logs and returns no regions -- so a raising detector
    reaches the lane as "no region found" rather than as an exception. The lane
    keeps its own `equation_lane_detection_failed` handler for failures raised
    outside that seam. Either way the page is no worse off than with the lane
    off, since neither run reads the equation; the imprecision is in provenance
    only and is recorded in docs/log/2026-09-02_p4r-equation-lane.md.
    """
    off, _, _, _ = _run(tmp_path / "off", lane=False, provider=True)
    on, _, spy, _ = _run(tmp_path / "on", lane=True, provider=True, detect_raises=True)

    assert spy.calls == 0
    assert _page_view(on, 2) == _page_view(off, 2)
    assert _kinds(on, 2) & {"equation_lane_detection_failed", "equation_lane_no_region"}


def test_legacy_gh36_block_does_not_double_attach(tmp_path: Path) -> None:
    """Lane + both legacy flags: one crop pass, one sidecar, not two."""
    from socr.math.equation_latex import EQUATION_SIDECAR_HEADER

    pipeline = _make_pipeline(
        equation_region_lane=True, detect_equations=True, recover_clean_equations=True
    )
    on, _, spy, _ = _run(tmp_path / "both", lane=True, provider=True, pipeline=pipeline)

    text = _page_view(on, 2)["text"]
    assert text.count(EQUATION_SIDECAR_HEADER) == 1
    assert spy.calls == 1


# ---------------------------------------------------------------------------
# P2 structure-class floor: unreachable on a table-free page
# ---------------------------------------------------------------------------


def test_p2_structure_class_floor_cannot_fire_on_the_equation_page(tmp_path: Path) -> None:
    """Ruling 2, end to end: every reading refused, and no prose is deleted."""
    from socr.core.manifest import _reaches_structure_class_branch

    on, _, _, _ = _run(tmp_path / "refused", lane=True, provider=True, latex="")

    ps = on.pages[2]
    assert ps.is_structure_class() is False
    assert _reaches_structure_class_branch(ps) is False

    text = _page_view(on, 2)["text"]
    assert "failed: unverifiable table" not in text
    for line in [line for line in _EQ_NATIVE.splitlines() if line.strip()]:
        assert line in text, "P2's floor deleted prose it must never reach"


def test_manifest_selection_ships_the_attached_output_unchanged(tmp_path: Path) -> None:
    """The winner seam must not re-stamp or revert the page to plain native.

    `native+equations` starts with `native` on purpose, so every existing
    `startswith("native")` guard treats it as a native lane. This pins that the
    guard chain does not then discard the attachments.
    """
    from socr.core.manifest import WinnerKind, _select_page_output_tagged, _winning_page_output

    on, _, _, _ = _run(tmp_path / "winner", lane=True, provider=True)

    selected, kind = _select_page_output_tagged(on, 2)
    assert kind is WinnerKind.PASSING_BEST_OUTPUT
    assert selected is on.pages[2].best_output
    shipped = _winning_page_output(on, 2)
    assert shipped.text == on.pages[2].best_output.text
    assert "```latex" in shipped.text


# ---------------------------------------------------------------------------
# Fragment identity and resume
# ---------------------------------------------------------------------------


def _fragment_path(out_dir: Path, page_num: int) -> Path:
    matches = sorted(out_dir.rglob(f"pages/*{page_num}.md"))
    assert matches, f"no fragment written for page {page_num} under {out_dir}"
    return matches[-1]


def test_final_fragment_matches_the_assembled_markdown(tmp_path: Path) -> None:
    """`_rewrite_all_fragments` and the stitched .md agree, page for page.

    Cold review round 1, finding 6: the earlier version asserted only
    `fragment.strip() in whole`. A substring check passes even when the document
    has been cut into the WRONG number of fragments, which is exactly the
    corruption finding 1 produced -- every truncated fragment is still a
    substring. This compares the exact fragment set against the exact page
    bodies the contract splits out of the final markdown.
    """
    from ocr_output_contract import split_native_pages

    on, _, _, out_dir = _run(tmp_path / "frag", lane=True, provider=True)

    body = [p for p in out_dir.rglob("*.md") if p.parent.name != "pages"]
    assert body, "no assembled markdown written"
    whole = body[-1].read_text()

    page_bodies = split_native_pages(whole)
    fragments = sorted(out_dir.rglob("pages/*.md"))
    assert len(fragments) == 3, f"fragment count: {[f.name for f in fragments]}"
    assert len(page_bodies) == 3

    for frag_path, page_body in zip(fragments, page_bodies, strict=True):
        assert frag_path.read_text().strip() == page_body.strip(), (
            f"{frag_path.name} diverged from its page in the assembled markdown"
        )
    assert "```latex" in fragments[1].read_text()


def test_provisional_flush_matches_the_final_fragment(tmp_path: Path) -> None:
    """The in-loop crash-recovery copy and `_rewrite_all_fragments` agree.

    Comparing the final fragment to the final markdown is not enough: the
    provisional flush is written mid-loop, BEFORE assemble, and is the copy a
    crash would leave behind. It must already carry the attachments.
    """
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.agentic import PageDecision
    from socr.pipeline.orchestrator import UnifiedPipeline

    root = tmp_path / "flush"
    root.mkdir(parents=True, exist_ok=True)
    pdf = _make_pdf(root)
    out_dir = root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = _make_pipeline(equation_region_lane=True)
    state = _make_state(pdf)
    pipeline._last_assessment = state._last_assessment
    spy = _ModelSpy(_LATEX_CONTAINED)
    provisional: dict[int, str] = {}
    real_flush = UnifiedPipeline._flush_page_fragment

    def _capture(self, st, page_num, text, output_dir):
        provisional.setdefault(page_num, text)
        return real_flush(self, st, page_num, text, output_dir)

    def _detect(page, page_num):
        from socr.math.detect_equations import EquationDetectionResult

        if page_num != 2:
            return EquationDetectionResult(page_num=page_num, regions=[], detection_time_s=0.0)
        return _region(page)

    def _routed(page_num, *args, **kwargs):
        return PageDecision(
            page_num=page_num,
            final_output=PageOutput(
                page_num=page_num, text="x", status=PageStatus.SUCCESS, engine="qwen"
            ),
            accepted=True,
        )

    with (
        patch.object(UnifiedPipeline, "_flush_page_fragment", _capture),
        patch("socr.math.detect_equations.detect_display_equations", side_effect=_detect),
        patch("socr.math.equation_latex.latex_for_crop", side_effect=spy),
        patch("socr.pipeline.orchestrator.route_page", side_effect=_routed),
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
    ):
        pipeline._phase_agentic(state, out_dir)
        pipeline._phase_assemble(state, out_dir)

    assert 2 in provisional, "no provisional flush captured for the lane page"
    assert "```latex" in provisional[2], "the crash-recovery copy lost the attachment"
    assert provisional[2] == _fragment_path(out_dir, 2).read_text()
    # And the authoritative rewrite left the fragment SET intact, not merely
    # this one file (finding 6).
    assert len(sorted(out_dir.rglob("pages/*.md"))) == 3


def test_resume_restores_the_attached_page_without_a_second_model_call(
    tmp_path: Path,
) -> None:
    """Run twice with the same config: page 2 comes back from the ledger."""
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.agentic import PageDecision

    root = tmp_path / "resume"
    root.mkdir(parents=True, exist_ok=True)
    pdf = _make_pdf(root)
    out_dir = root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _detect(page, page_num):
        from socr.math.detect_equations import EquationDetectionResult

        if page_num != 2:
            return EquationDetectionResult(page_num=page_num, regions=[], detection_time_s=0.0)
        return _region(page)

    def _routed(page_num, *args, **kwargs):
        return PageDecision(
            page_num=page_num,
            final_output=PageOutput(
                page_num=page_num, text="x", status=PageStatus.SUCCESS, engine="qwen"
            ),
            accepted=True,
        )

    spy = _ModelSpy(_LATEX_CONTAINED)
    texts = []
    for _ in range(2):
        pipeline = _make_pipeline(equation_region_lane=True)
        state = _make_state(pdf)
        pipeline._last_assessment = state._last_assessment
        with (
            patch("socr.math.detect_equations.detect_display_equations", side_effect=_detect),
            patch("socr.math.equation_latex.latex_for_crop", side_effect=spy),
            patch("socr.pipeline.orchestrator.route_page", side_effect=_routed),
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
        ):
            pipeline._phase_agentic(state, out_dir)
            pipeline._phase_assemble(state, out_dir)
        texts.append(_page_view(state, 2)["text"])

    assert texts[0] == texts[1], "resumed page drifted from the first run"
    assert spy.calls == 1, f"the model was called again on resume ({spy.calls} calls)"
    # The lane's disposition must survive the resume, or a refusal becomes
    # invisible the moment a run is restarted.
    assert "equation_region_reading_attached" in _kinds(state, 2)


def test_a_rejected_reading_survives_a_resume(tmp_path: Path) -> None:
    """The presence guard's refusal is still in the audit trail after resume."""
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.agentic import PageDecision

    root = tmp_path / "resume_rejected"
    root.mkdir(parents=True, exist_ok=True)
    pdf = _make_pdf(root)
    out_dir = root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _detect(page, page_num):
        from socr.math.detect_equations import EquationDetectionResult

        if page_num != 2:
            return EquationDetectionResult(page_num=page_num, regions=[], detection_time_s=0.0)
        return _region(page)

    def _routed(page_num, *args, **kwargs):
        return PageDecision(
            page_num=page_num,
            final_output=PageOutput(
                page_num=page_num, text="x", status=PageStatus.SUCCESS, engine="qwen"
            ),
            accepted=True,
        )

    spy = _ModelSpy(_LATEX_INVENTED)
    states = []
    for _ in range(2):
        pipeline = _make_pipeline(equation_region_lane=True)
        state = _make_state(pdf)
        pipeline._last_assessment = state._last_assessment
        with (
            patch("socr.math.detect_equations.detect_display_equations", side_effect=_detect),
            patch("socr.math.equation_latex.latex_for_crop", side_effect=spy),
            patch("socr.pipeline.orchestrator.route_page", side_effect=_routed),
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
        ):
            pipeline._phase_agentic(state, out_dir)
            pipeline._phase_assemble(state, out_dir)
        states.append(state)

    assert spy.calls == 1
    for state in states:
        rejected = [
            e
            for e in state.events
            if e.page_num == 2 and e.kind == "equation_region_reading_rejected"
        ]
        assert rejected, "the refusal disappeared on resume"
        assert "0.9137" in rejected[0].data["invented"]


def test_toggling_the_lane_invalidates_the_resume_ledger(tmp_path: Path) -> None:
    """The flag is fingerprinted, so a terminal page produced under the other
    setting is not reused. `_run_fingerprint` is run-wide by design, so this is
    asserted on the fingerprint rather than on which pages reprocess."""
    on = _make_pipeline(equation_region_lane=True)._run_fingerprint()
    off = _make_pipeline(equation_region_lane=False)._run_fingerprint()
    assert on != off


# ---------------------------------------------------------------------------
# Cold review round 1 -- reproductions, written before the fixes
# ---------------------------------------------------------------------------


def _drive(pipeline, state, out_dir: Path, *, spy, profiles, source_text: str = _EQ_SOURCE):
    """One agentic+assemble pass over an existing state/out_dir."""
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.agentic import PageDecision

    def _detect(page, page_num):
        from socr.math.detect_equations import EquationDetectionResult

        if page_num != 2:
            return EquationDetectionResult(page_num=page_num, regions=[], detection_time_s=0.0)
        return _region(page, source_text=source_text)

    def _routed(page_num, *args, **kwargs):
        return PageDecision(
            page_num=page_num,
            final_output=PageOutput(
                page_num=page_num, text="x", status=PageStatus.SUCCESS, engine="qwen"
            ),
            accepted=True,
        )

    with (
        patch("socr.math.detect_equations.detect_display_equations", side_effect=_detect),
        patch("socr.math.equation_latex.latex_for_crop", side_effect=spy),
        patch("socr.pipeline.orchestrator.route_page", side_effect=_routed),
        patch.object(pipeline, "_available_engines_for_agentic", return_value=list(profiles)),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
    ):
        pipeline._phase_agentic(state, out_dir)
        return pipeline._phase_assemble(state, out_dir)


def test_r1_f1_model_page_marker_cannot_split_the_document(tmp_path: Path) -> None:
    """Finding 1: a `## Page N` line inside a model reading must never reach the
    assembled body, where `split_native_pages` would read it as a real page
    boundary and cut the native prose in half.

    The injected marker uses page number 3 deliberately: `3` occurs in the
    native text exactly once, so the numeric-presence guard passes the reading
    and cannot be the thing that saves us here.
    """
    from ocr_output_contract import PAGE_MARKER_RE, split_native_pages

    injected = "y = 2x + 1\n## Page 3"
    on, result, spy, out_dir = _run(tmp_path / "marker", lane=True, provider=True, latex=injected)

    body = sorted(p for p in out_dir.rglob("*.md") if p.parent.name != "pages")
    assert body, "no assembled markdown written"
    assembled = body[-1].read_text()

    assert len(PAGE_MARKER_RE.findall(assembled)) == 3, (
        "a model-authored page marker reached the assembled body"
    )
    assert len(split_native_pages(assembled)) == 3, (
        "the document splits into the wrong number of logical pages"
    )
    fragments = sorted(out_dir.rglob("pages/*.md"))
    assert len(fragments) == 3, f"fragment count changed: {[p.name for p in fragments]}"
    assert spy.calls == 1


def test_r1_f2_invented_exponent_is_a_candidate_token(tmp_path: Path) -> None:
    """Finding 2: an exponent is a number the model wrote, so the guard must
    see it. `x^9` against a page that has no 9 is an invented value."""
    from socr.tables.escalation_canary import PRESENCE_INVENTED, text_value_tokens

    assert text_value_tokens("x^9") == {"9": 1}
    assert text_value_tokens("x^999") == {"999": 1}

    verdict = region_presence_verdict_for("the page mentions 2 and nothing else", "x^9")
    assert verdict.status == PRESENCE_INVENTED
    assert "9" in verdict.invented


def region_presence_verdict_for(native: str, candidate: str):
    from socr.tables.escalation_canary import region_presence_verdict

    return region_presence_verdict(native, candidate)


def test_r1_f3_an_executed_call_is_metered_live_and_on_resume(tmp_path: Path) -> None:
    """Finding 3: the lane's model call must appear in `engine_runs` and must
    not turn `total_cost` into None when the page is resumed."""
    from socr.core.providers import PROFILE_QWEN_LOCAL

    root = tmp_path / "cost"
    root.mkdir(parents=True, exist_ok=True)
    pdf = _make_pdf(root)
    out_dir = root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    spy = _ModelSpy(_LATEX_CONTAINED)

    pipeline = _make_pipeline(equation_region_lane=True)
    state = _make_state(pdf)
    pipeline._last_assessment = state._last_assessment
    _drive(pipeline, state, out_dir, spy=spy, profiles=[PROFILE_QWEN_LOCAL])

    lane_runs = [r for r in state.engine_runs if (r.engine or "").startswith("native+equations")]
    assert lane_runs, "the executed equation-model call was never recorded in engine_runs"
    assert state.total_cost is not None
    live_total = state.total_cost

    pipeline2 = _make_pipeline(equation_region_lane=True)
    state2 = _make_state(pdf)
    pipeline2._last_assessment = state2._last_assessment
    _drive(pipeline2, state2, out_dir, spy=spy, profiles=[PROFILE_QWEN_LOCAL])

    assert spy.calls == 1, "the page did not resume"
    assert state2.total_cost is not None, "resume turned the run's total cost unknown"
    assert state2.total_cost == live_total


def test_r1_f4_an_unrelated_local_profile_does_not_authorise_the_call(
    tmp_path: Path,
) -> None:
    """Finding 4: availability must mean the profile that serves the equation
    model, not any local-tier provider. Marker is local and cannot serve it."""
    from socr.core.providers import PROFILE_MARKER

    root = tmp_path / "wrongprofile"
    root.mkdir(parents=True, exist_ok=True)
    pdf = _make_pdf(root)
    out_dir = root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    spy = _ModelSpy(_LATEX_CONTAINED)

    pipeline = _make_pipeline(equation_region_lane=True)
    state = _make_state(pdf)
    pipeline._last_assessment = state._last_assessment
    _drive(pipeline, state, out_dir, spy=spy, profiles=[PROFILE_MARKER])

    assert spy.calls == 0, "the lane called the equation model with no provider able to serve it"


def test_r1_f5_a_no_provider_page_is_retried_when_a_provider_appears(
    tmp_path: Path,
) -> None:
    """Finding 5: a skip caused by transient unavailability must not freeze as a
    terminal success. The next run, with the provider back and the fingerprint
    unchanged, must call the model."""
    from socr.core.providers import PROFILE_QWEN_LOCAL

    root = tmp_path / "retry"
    root.mkdir(parents=True, exist_ok=True)
    pdf = _make_pdf(root)
    out_dir = root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    spy = _ModelSpy(_LATEX_CONTAINED)

    pipeline = _make_pipeline(equation_region_lane=True)
    state = _make_state(pdf)
    pipeline._last_assessment = state._last_assessment
    _drive(pipeline, state, out_dir, spy=spy, profiles=[])
    assert spy.calls == 0

    pipeline2 = _make_pipeline(equation_region_lane=True)
    state2 = _make_state(pdf)
    pipeline2._last_assessment = state2._last_assessment
    _drive(pipeline2, state2, out_dir, spy=spy, profiles=[PROFILE_QWEN_LOCAL])

    assert spy.calls == 1, (
        "the equation page was restored from a no-provider terminal and never re-read"
    )
    assert "```latex" in _page_view(state2, 2)["text"]


# ---------------------------------------------------------------------------
# Cold review round 2 -- reproductions, written before the fixes
# ---------------------------------------------------------------------------


def _paid_profile(rate: float, *, backend: str = "ollama", model: str | None = None, engine=None):
    """A profile serving the lane's model at a NON-zero price.

    Every round-1 budget test used the free local rung, which cannot show
    whether a cap is applied at all (finding 3).
    """
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import TIER_CLOUD, ProviderProfile

    return ProviderProfile(
        engine=engine or EngineType.QWEN,
        tier=TIER_CLOUD,
        cost_per_page_usd=rate,
        id=f"paid-{backend}-{rate}",
        backend=backend,
        model=model or PipelineConfig().clean_equation_model,
    )


def _lane_run(tmp_path: Path, *, profiles, latex: str = _LATEX_CONTAINED, regions: int = 1, **cfg):
    """Drive one lane page with an explicit provider list and config."""
    root = tmp_path
    root.mkdir(parents=True, exist_ok=True)
    pdf = _make_pdf(root)
    out_dir = root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = _make_pipeline(equation_region_lane=True, **cfg)
    state = _make_state(pdf)
    pipeline._last_assessment = state._last_assessment
    spy = _ModelSpy(latex)

    from socr.core.result import PageOutput, PageStatus
    from socr.math.detect_equations import EquationDetectionResult, EquationRegion
    from socr.pipeline.agentic import PageDecision

    def _detect(page, page_num):
        if page_num != 2:
            return EquationDetectionResult(page_num=page_num, regions=[], detection_time_s=0.0)
        boxes = [(230.0, 185.0, 400.0, 215.0), (230.0, 90.0, 400.0, 120.0)]
        regs = [
            EquationRegion(
                page_num=2,
                bbox=boxes[i],
                source_bbox=boxes[i],
                has_eq_number=True,
                equation_label="(3)",
                source_text=_EQ_SOURCE if i == 0 else "The result follows directly.",
            )
            for i in range(regions)
        ]
        return EquationDetectionResult(page_num=2, regions=regs, detection_time_s=0.0)

    def _routed(page_num, *args, **kwargs):
        return PageDecision(
            page_num=page_num,
            final_output=PageOutput(
                page_num=page_num, text="x", status=PageStatus.SUCCESS, engine="qwen"
            ),
            accepted=True,
        )

    with (
        patch("socr.math.detect_equations.detect_display_equations", side_effect=_detect),
        patch("socr.math.equation_latex.latex_for_crop", side_effect=spy),
        patch("socr.pipeline.orchestrator.route_page", side_effect=_routed),
        patch.object(pipeline, "_available_engines_for_agentic", return_value=list(profiles)),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
    ):
        pipeline._phase_agentic(state, out_dir)
        pipeline._phase_assemble(state, out_dir)
    return state, spy


def test_r2_f3_per_page_cost_cap_is_applied_before_calling(tmp_path: Path) -> None:
    """Finding 3: `--max-cost-per-page` must filter the lane's candidate exactly
    as `_phase_agentic` filters the ladder. A profile above the cap is not a
    provider, so no call may be made."""
    state, spy = _lane_run(tmp_path / "cap", profiles=[_paid_profile(0.05)], max_cost_per_page=0.01)
    assert spy.calls == 0, "the lane called a provider priced above --max-cost-per-page"


def test_r2_f3_region_calls_count_cumulatively_against_the_page_cap(
    tmp_path: Path,
) -> None:
    """Two region reads on one page cost twice the rate; the cap is per PAGE."""
    state, spy = _lane_run(
        tmp_path / "cum",
        profiles=[_paid_profile(0.03)],
        regions=2,
        max_cost_per_page=0.05,
    )
    assert spy.calls == 1, f"{spy.calls} calls at 0.03 each blew through a 0.05 per-page cap"


def test_r2_f3_remaining_cost_budget_is_checked_before_calling(tmp_path: Path) -> None:
    """`--cost-budget` must gate the lane the way it gates the generic branch."""
    state, spy = _lane_run(tmp_path / "budget", profiles=[_paid_profile(0.05)], cost_budget=0.02)
    assert spy.calls == 0, "the lane spent past the remaining --cost-budget"


def test_r2_f4_same_model_on_another_backend_is_not_a_provider(tmp_path: Path) -> None:
    """Finding 4: the call is hardwired to the Ollama generate API, so a profile
    that does not serve that API cannot serve it.

    Round 3 replaced the synthetic profile this test used to build. A QWEN rung's
    executed backend comes from the live config, not from the registry's
    declaration, so a hand-made QWEN profile stamped "vllm" resolves back to
    Ollama and is CORRECTLY addressable. The production shape is the canonical
    Ollama-DECLARED profile resolving to vLLM through `qwen_backend`; the raw
    field still decides for engines the resolver does not rewrite.
    """
    from socr.core.config import EngineType
    from socr.core.providers import PROFILE_QWEN_LOCAL

    state, spy = _lane_run(tmp_path / "backend", profiles=[PROFILE_QWEN_LOCAL], qwen_backend="vllm")
    assert spy.calls == 0, "a vLLM-resolved profile authorised an Ollama-transport call"

    state2, spy2 = _lane_run(
        tmp_path / "backend_raw",
        profiles=[_paid_profile(0.0, backend="vllm", engine=EngineType.GEMINI)],
    )
    assert spy2.calls == 0, "a non-Ollama rung authorised an Ollama-transport call"


# -- Finding 5: the REAL entry path (process() / _resume_skip) ----------------


def _process_pdf(tmp_path: Path) -> Path:
    """Two born-digital pages: prose, then the equation page. No table page, so
    a no-provider run still completes cleanly (the reviewer's scenario)."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Introductory prose that carries no mathematics.", fontsize=11)
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "The result follows directly.", fontsize=11)
    page.insert_text((240, 200), _EQ_SOURCE, fontsize=11)
    page.insert_text((72, 300), "We conclude in section 4.", fontsize=11)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "proc.pdf"
    doc.save(str(path))
    doc.close()
    return path


def _process_run(pdf: Path, out_dir: Path, *, provider: bool, spy, **cfg):
    """One REAL `process()` run: the entry path with the document resume gate."""
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.math.detect_equations import EquationDetectionResult, EquationRegion

    pipeline = _make_pipeline(equation_region_lane=True, **cfg)

    real_detect = pipeline.bd_detector.detect

    def _detect_with_equations(path):
        assessment = real_detect(path)
        for pa in assessment.pages:
            pa.has_equations = pa.page_num == 2
            pa.has_tables = False
        return assessment

    pipeline.bd_detector.detect = _detect_with_equations

    def _detect_regions(page, page_num):
        if page_num != 2:
            return EquationDetectionResult(page_num=page_num, regions=[], detection_time_s=0.0)
        bbox = (230.0, 185.0, 400.0, 215.0)
        return EquationDetectionResult(
            page_num=2,
            regions=[
                EquationRegion(
                    page_num=2,
                    bbox=bbox,
                    source_bbox=bbox,
                    has_eq_number=True,
                    equation_label="(3)",
                    source_text=_EQ_SOURCE,
                )
            ],
            detection_time_s=0.0,
        )

    # Hermetic: any page that still reaches the ladder must not launch a real
    # engine just because a provider profile is present.
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.agentic import PageDecision

    def _routed(page_num, *args, **kwargs):
        return PageDecision(
            page_num=page_num,
            final_output=PageOutput(
                page_num=page_num,
                text=f"routed page {page_num}",
                status=PageStatus.SUCCESS,
                engine="qwen",
                audit_passed=True,
            ),
            accepted=True,
        )

    with (
        patch("socr.math.detect_equations.detect_display_equations", side_effect=_detect_regions),
        patch("socr.math.equation_latex.latex_for_crop", side_effect=spy),
        patch("socr.pipeline.orchestrator.route_page", side_effect=_routed),
        patch.object(
            pipeline,
            "_available_engines_for_agentic",
            return_value=[PROFILE_QWEN_LOCAL] if provider else [],
        ),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
    ):
        return pipeline.process(pdf, output_dir=out_dir)


def test_r2_f5_document_gate_lets_a_pending_equation_page_be_retried(
    tmp_path: Path,
) -> None:
    """Finding 5: no-provider then provider must re-read the page.

    Round 1 pinned this on `_phase_agentic` only. `process()` consults
    `_resume_skip` BEFORE any page ledger, so the document-level gate is what
    actually decides, and it never saw the latch.
    """
    root = tmp_path / "docgate"
    pdf = _process_pdf(root)
    out_dir = root / "out"
    spy = _ModelSpy(_LATEX_CONTAINED)

    _process_run(pdf, out_dir, provider=False, spy=spy)
    assert spy.calls == 0

    result = _process_run(pdf, out_dir, provider=True, spy=spy)
    assert result.status.value != "skipped", (
        "the document was skipped whole; the pending equation page was never reached"
    )
    assert spy.calls == 1, "the equation model was never called on the retry run"

    md = [p for p in out_dir.rglob("*.md") if p.parent.name != "pages"]
    assert md and "```latex" in md[-1].read_text()


def test_r2_f5_a_completed_document_still_skips_when_the_provider_goes_away(
    tmp_path: Path,
) -> None:
    """The other direction: provider then no provider must RESTORE, not re-fail."""
    root = tmp_path / "docgate2"
    pdf = _process_pdf(root)
    out_dir = root / "out"
    spy = _ModelSpy(_LATEX_CONTAINED)

    _process_run(pdf, out_dir, provider=True, spy=spy)
    assert spy.calls == 1
    md = [p for p in out_dir.rglob("*.md") if p.parent.name != "pages"][-1]
    before = md.read_text()

    result = _process_run(pdf, out_dir, provider=False, spy=spy)
    assert result.status.value == "skipped", "a finished document was reprocessed"
    assert spy.calls == 1
    assert md.read_text() == before


def test_r2_f5_a_transport_failure_sets_the_latch(tmp_path: Path) -> None:
    """A provider was present but the call failed, so the model did not run."""
    root = tmp_path / "transport"
    pdf = _process_pdf(root)
    out_dir = root / "out"
    failing = _ModelSpy("")

    _process_run(pdf, out_dir, provider=True, spy=failing)
    assert failing.calls == 1

    good = _ModelSpy(_LATEX_CONTAINED)
    result = _process_run(pdf, out_dir, provider=True, spy=good)
    assert result.status.value != "skipped", "a page whose call failed was frozen as done"
    assert good.calls == 1


def test_r2_f5_strict_local_does_not_suppress_the_latch(tmp_path: Path) -> None:
    """The latch says the model did not run; that is true under strict-local too."""
    root = tmp_path / "strict"
    pdf = _process_pdf(root)
    out_dir = root / "out"
    spy = _ModelSpy(_LATEX_CONTAINED)

    _process_run(pdf, out_dir, provider=False, spy=spy, strict_local=True)
    assert spy.calls == 0

    result = _process_run(pdf, out_dir, provider=True, spy=spy, strict_local=True)
    assert result.status.value != "skipped"
    assert spy.calls == 1


# -- Finding 6: byte-exactness ------------------------------------------------


def test_r2_f6_fragments_are_byte_exact_without_stripping(tmp_path: Path) -> None:
    from ocr_output_contract import split_native_pages

    on, _, _, out_dir = _run(tmp_path / "bytes", lane=True, provider=True)

    whole = [p for p in out_dir.rglob("*.md") if p.parent.name != "pages"][-1].read_text()
    page_bodies = split_native_pages(whole)
    fragments = sorted(out_dir.rglob("pages/*.md"))
    assert len(fragments) == len(page_bodies) == 3
    for frag_path, page_body in zip(fragments, page_bodies, strict=True):
        assert frag_path.read_text() == page_body, f"{frag_path.name} is not byte-exact"


def test_r2_f6_a_rejected_reading_leaves_the_page_byte_identical_to_lane_off(
    tmp_path: Path,
) -> None:
    """The marker regression compared counts; this compares the bytes."""
    injected = "y = 2x + 1\n## Page 3"
    off, _, _, off_dir = _run(tmp_path / "off", lane=False, provider=True)
    on, _, _, on_dir = _run(tmp_path / "on", lane=True, provider=True, latex=injected)

    assert _page_view(on, 2)["text"] == _page_view(off, 2)["text"]
    off_frag = sorted(off_dir.rglob("pages/*.md"))
    on_frag = sorted(on_dir.rglob("pages/*.md"))
    assert len(off_frag) == len(on_frag) == 3
    for a, b in zip(off_frag, on_frag, strict=True):
        assert a.read_text() == b.read_text(), f"{a.name} differs from the lane-off run"


# ---------------------------------------------------------------------------
# Cold review round 3 -- reproductions, written before the fixes
# ---------------------------------------------------------------------------


def test_r3_f4_a_config_resolved_vllm_backend_is_not_a_provider(tmp_path: Path) -> None:
    """Finding 4: authorization must read the backend the profile RESOLVES to.

    `PROFILE_QWEN_LOCAL` is DECLARED as Ollama, but `resolved_provenance` runs it
    through `qwen_backend` and can land on vLLM. The lane posts to the Ollama
    generate API, so under that config the canonical local profile is not a
    provider for this call -- even though its raw `backend` field still says
    "ollama". Round 2 checked the raw field and missed exactly this shape.
    """
    from socr.core.config import PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL, resolved_provenance

    cfg = PipelineConfig(agentic=True, quiet=True, equation_region_lane=True, qwen_backend="vllm")
    # The premise, asserted rather than assumed.
    assert resolved_provenance(PROFILE_QWEN_LOCAL, cfg)[0] == "vllm"
    assert PROFILE_QWEN_LOCAL.backend == "ollama"

    from socr.pipeline.orchestrator import UnifiedPipeline

    pipeline = UnifiedPipeline(cfg)
    profile, reason = pipeline._equation_lane_provider([PROFILE_QWEN_LOCAL])
    assert profile is None, "a config-resolved vLLM deployment authorised an Ollama call"
    assert "backend" in reason


def test_r3_f4_an_explicit_ollama_backend_is_always_addressable(monkeypatch) -> None:
    """An EXPLICIT backend wins in both directions, exported variable or not."""
    from socr.core.config import PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline.orchestrator import UnifiedPipeline

    for env in (None, "http://vllm.example:8000/v1"):
        if env is None:
            monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        else:
            monkeypatch.setenv("VLLM_BASE_URL", env)
        cfg = PipelineConfig(
            agentic=True, quiet=True, equation_region_lane=True, qwen_backend="ollama"
        )
        profile, reason = UnifiedPipeline(cfg)._equation_lane_provider([PROFILE_QWEN_LOCAL])
        assert profile is PROFILE_QWEN_LOCAL, f"VLLM_BASE_URL={env}: {reason}"


@pytest.mark.parametrize("exported", [False, True], ids=["no-vllm-url", "vllm-url-exported"])
def test_r3_f4_auto_resolves_by_environment_not_by_ambient_luck(
    monkeypatch, exported: bool
) -> None:
    """`auto` means Ollama unless VLLM_BASE_URL is exported -- assert BOTH states.

    Cold review round 4, N3: the earlier control asserted only the Ollama arm and
    never cleared the variable, so it failed on the documented HPC setup, where
    production correctly resolves `auto` to vLLM and refuses the profile.
    """
    from socr.core.config import PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline.orchestrator import UnifiedPipeline

    if exported:
        monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.example:8000/v1")
    else:
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)

    cfg = PipelineConfig(agentic=True, quiet=True, equation_region_lane=True, qwen_backend="auto")
    profile, reason = UnifiedPipeline(cfg)._equation_lane_provider([PROFILE_QWEN_LOCAL])
    if exported:
        assert profile is None, "auto + VLLM_BASE_URL resolves to vLLM and cannot be addressed"
        assert "backend" in reason
    else:
        assert profile is PROFILE_QWEN_LOCAL, reason


def test_r3_f5_the_root_entry_is_never_persisted_completed_without_its_latch(
    tmp_path: Path,
) -> None:
    """Finding 5: the latch and the terminal record must land in ONE write.

    Round 2 called `index.record(...)` -- which saves a completed entry
    immediately -- and only then mutated the entry and saved again. An
    interruption or a failure of that second save leaves a valid completed root
    entry with no latch, and the next run skips the whole document. Snapshot
    every persisted state and assert that intermediate never exists.
    """
    import copy

    from ocr_output_contract import RootIndex

    root = tmp_path / "atomic"
    pdf = _process_pdf(root)
    out_dir = root / "out"
    spy = _ModelSpy(_LATEX_CONTAINED)

    snapshots: list[dict] = []
    real_save = RootIndex.save

    def _spy_save(self):
        real_save(self)
        snapshots.append(copy.deepcopy(self.files))

    with patch.object(RootIndex, "save", _spy_save):
        _process_run(pdf, out_dir, provider=False, spy=spy)

    assert spy.calls == 0
    # Every state the index was ever persisted in, not just the last one. Both
    # "completed" and "partial" are skippable shapes -- `_resume_skippable`
    # accepts a partial entry whose checksum, fingerprint and output all match --
    # so either one persisted WITHOUT the latch is the hole.
    persisted = [
        entry
        for snap in snapshots
        for entry in snap.values()
        if entry.get("status") in ("completed", "partial")
    ]
    assert persisted, "no resumable root entry was ever written"
    for entry in persisted:
        assert entry.get("equation_lane_retry_pending") is True, (
            f"a resumable root entry ({entry.get('status')}) was persisted without the "
            "pending-retry latch; an interruption in that window loses the retry forever"
        )


def test_r3_f5_a_failed_index_write_leaves_no_completed_record(tmp_path: Path) -> None:
    """Fail-closed: if the single write fails, the document is NOT recorded done."""
    from ocr_output_contract import RootIndex

    root = tmp_path / "failsave"
    pdf = _process_pdf(root)
    out_dir = root / "out"
    spy = _ModelSpy(_LATEX_CONTAINED)

    def _boom(self):
        raise OSError("simulated index write failure")

    with patch.object(RootIndex, "save", _boom):
        _process_run(pdf, out_dir, provider=False, spy=spy)

    entries = RootIndex(out_dir).files
    resumable = [e for e in entries.values() if e.get("status") in ("completed", "partial")]
    assert not resumable, (
        "a resumable root record survived a failed index write, so the document "
        "would be skipped with its equation page never read"
    )


def test_r3_n1_the_lane_uses_the_ladder_cost_filter(tmp_path: Path) -> None:
    """N1: pin the ladder's `max_cost_per_page` argument on its own.

    The end-to-end cap test cannot do this: with one region, the pre-call cap
    catches the same case, so removing the ladder argument alone leaves it
    green. Asserting on the candidate `_equation_lane_provider` returns fails
    the moment the lane stops passing the cap to `provider_ladder`.
    """
    from socr.core.config import PipelineConfig
    from socr.pipeline.orchestrator import UnifiedPipeline

    profile = _paid_profile(0.05)
    cfg = PipelineConfig(
        agentic=True, quiet=True, equation_region_lane=True, max_cost_per_page=0.01
    )
    selected, reason = UnifiedPipeline(cfg)._equation_lane_provider([profile])
    assert selected is None, "the lane selected a rung the routing ladder prices out"
    assert "priced above" in reason

    uncapped = PipelineConfig(agentic=True, quiet=True, equation_region_lane=True)
    selected2, _ = UnifiedPipeline(uncapped)._equation_lane_provider([profile])
    assert selected2 is profile, "the control arm must select the same rung with no cap"


# ---------------------------------------------------------------------------
# Cold review round 4 -- reproductions, written before the fixes
# ---------------------------------------------------------------------------


def test_r4_f5_a_stale_root_entry_cannot_survive_a_failed_terminal_write(
    tmp_path: Path,
) -> None:
    """Finding 5 residual: fail closed at the START, not only at the end.

    The single terminal write is atomic, but atomicity only protects what it
    writes. A matching OLDER entry -- completed, latch-free, from a build that
    predates this lane, its output since deleted -- survives a failed terminal
    write. The run will have re-created the output it points at, so it becomes
    resumable again with no latch, and the next run skips the whole document
    with the equation page never read.

    The stale entry is seeded directly rather than produced by a successful run:
    a run that DID read the page leaves terminal page sidecars, and restoring
    those is correct behaviour, not a lost retry.
    """
    from ocr_output_contract import (
        DocMetadata,
        RootIndex,
        Status,
        doc_dir_for,
        markdown_path_for,
        relative_key,
        safe_checksum,
        utc_timestamp,
    )

    root = tmp_path / "stale"
    pdf = _process_pdf(root)
    out_dir = root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = _make_pipeline(equation_region_lane=True)
    with patch.object(pipeline, "_resolve_judge_model", return_value=""):
        fingerprint = pipeline._run_fingerprint()
    rel_key = relative_key(pdf, pdf.parent)
    md_path = markdown_path_for(doc_dir_for(out_dir, rel_key), rel_key)

    # A completed, latch-free record whose output is not on disk.
    RootIndex(out_dir).record(
        rel_key,
        DocMetadata(
            status=Status.COMPLETED,
            checksum=safe_checksum(pdf),
            model="qwen",
            backend="socr",
            processing_time=1.0,
            timestamp=utc_timestamp(),
            output_path=str(md_path),
            pages=2,
            fingerprint=fingerprint,
        ),
    )
    assert RootIndex(out_dir).files[rel_key].get("equation_lane_retry_pending") is None
    assert not md_path.exists()

    # A no-provider run re-emits that output but cannot record its own result.
    stalled = _ModelSpy(_LATEX_CONTAINED)
    with patch.object(RootIndex, "save", lambda self: (_ for _ in ()).throw(OSError("boom"))):
        _process_run(pdf, out_dir, provider=False, spy=stalled)

    # The next run has a provider. It must NOT skip, and it must read the page.
    retry = _ModelSpy(_LATEX_CONTAINED)
    result = _process_run(pdf, out_dir, provider=True, spy=retry)
    assert result.status.value != "skipped", (
        "a stale latch-free root entry survived the failed write and skipped the document"
    )
    assert retry.calls == 1, "the equation page was never read"
    md = [p for p in out_dir.rglob("*.md") if p.parent.name != "pages"]
    assert md and "```latex" in md[-1].read_text()


def test_r4_n2_an_unaddressable_backend_is_a_settled_refusal(tmp_path: Path) -> None:
    """N2: a refusal decided by configuration must not latch for retry.

    `qwen_backend=vllm` is a supported deployment, not a transient outage. If it
    sets the latch, every rerun refuses the document skip, restores the same
    page, and writes the latch again -- idempotent resume is defeated forever
    for that configuration.
    """
    root = tmp_path / "settled"
    pdf = _process_pdf(root)
    out_dir = root / "out"
    spy = _ModelSpy(_LATEX_CONTAINED)

    first = _process_run(pdf, out_dir, provider=True, spy=spy, qwen_backend="vllm")
    assert spy.calls == 0
    assert first.status.value != "skipped"

    second = _process_run(pdf, out_dir, provider=True, spy=spy, qwen_backend="vllm")
    assert spy.calls == 0
    assert second.status.value == "skipped", (
        "a configuration-settled refusal keeps the document permanently unskippable"
    )


def test_a_no_oracle_page_still_attaches_notation_only_latex(tmp_path: Path) -> None:
    """GH-543: the GH-522 refusal was scoped too broadly, and this is the case.

    UNVERIFIABLE covers two situations. The one #522 is about is a DAMAGED text
    layer -- an oracle exists but cannot be trusted. The other is a page with no
    numeric oracle at all, and there a numeral in the reading is usually
    NOTATION: the 2 in `E = mc^2`, an equation tag. Refusing on that convicts
    notation-only LaTeX on prose pages, which
    `test_a_page_with_no_numbers_is_unverifiable_not_invented` protects
    deliberately on the legacy path.

    So the refusal is now scoped to damaged text. This page is NOT
    encoding-suspect and has no numeric oracle: it must still attach.
    """
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.agentic import PageDecision

    pipeline = _make_pipeline(equation_region_lane=True)
    pdf_dir = tmp_path / "no_oracle"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf = _make_pdf(pdf_dir)
    state = _make_state(pdf)
    # Deliberately NOT encoding-suspect: the page text is fine, it just has no
    # numbers for the oracle.
    # The region's source slice must appear in the page text, or
    # `attach_equation_sidecars_in_place` drops the reading as UNALIGNED and the
    # test passes without the attachment it claims to pin (cubic P2 on #545 --
    # the first version asserted only the absence of a refusal). The slice is
    # chosen number-free so the page still has no numeric oracle, which is the
    # condition under test.
    source = "E equals m c squared"
    state.pages[2].native_text = f"Prose before.\n\n{source}\n\nProse after."
    pipeline._last_assessment = state._last_assessment
    spy = _ModelSpy(r"E = mc^2")

    def _detect(page, page_num):
        from socr.math.detect_equations import EquationDetectionResult

        if page_num != 2:
            return EquationDetectionResult(page_num=page_num, regions=[], detection_time_s=0.0)
        return _region(page, source_text=source)

    def _routed(page_num, *args, **kwargs):
        return PageDecision(
            page_num=page_num,
            final_output=PageOutput(
                page_num=page_num, text="x", status=PageStatus.SUCCESS, engine="qwen"
            ),
            accepted=True,
        )

    out_dir = pdf_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (
        patch("socr.math.detect_equations.detect_display_equations", side_effect=_detect),
        patch("socr.math.equation_latex.latex_for_crop", side_effect=spy),
        patch("socr.pipeline.orchestrator.route_page", side_effect=_routed),
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
    ):
        pipeline._phase_agentic(state, out_dir)
        pipeline._phase_assemble(state, out_dir)

    refused = [
        e
        for e in state.events
        if e.page_num == 2 and e.kind == "equation_region_reading_unverifiable"
    ]
    assert refused == [], (
        "notation-only LaTeX was refused on a page whose text layer is FINE and "
        "simply has no numbers; the exponent in `E = mc^2` is not a data value"
    )

    # And it ATTACHED. Asserting only the absence of a refusal would pass on a
    # reading dropped for some entirely different reason -- which is exactly
    # what happened before the source slice above was aligned.
    attached = [
        e for e in state.events if e.page_num == 2 and e.kind == "equation_region_reading_attached"
    ]
    assert len(attached) == 1, (
        f"the notation-only reading did not attach: {[e.kind for e in state.events if e.page_num == 2]}"
    )
    assert attached[0].data["presence_status"] == "unverifiable"
