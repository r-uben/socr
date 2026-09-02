"""Cold review round 3 on the P3+P5 branch (`fix/p3-p5-judged-bytes-ship`).

The round-2 review re-tested every fix and found four of them incomplete. These
are the cold reviewer's own canaries, adapted to run in-repo and to exercise the
production seams rather than local helpers:

1. **Operational-failure laundering.** The crop lane is reachable on ladder
   exhaustion with a non-empty operational failure (a truncated provider read).
   The re-judge rewrote the candidate to SUCCESS/NONE before judging and then
   promoted that state while the old `error` string stood. A crop repairs
   tables; it does not repair a truncated read.
2. **Guard reuse.** `origin/main` already carries PR #518's guards, and the
   local re-implementations were weaker: the delimiter regex missed the
   contract's own case-insensitive / leading-whitespace / trailing-text forms,
   and the LaTeX tokenizer's exponent exclusion is the exact hole #518 closed
   (an invented 9 hiding in `x^9`).
3. **Judge-event ownership.** A swapped instance-global sink cannot isolate a
   judge that TIMED OUT: its worker keeps running and its late event reached
   `state.events`, or another re-judge's scratch list.
4. **Metering attribution and durability.** A heuristic re-judge was journalled
   as the winning OCR engine at that engine's page price, and the extra spend
   did not survive a terminal resume.

Hermetic: no provider, no network, no live model. The equation cases patch the
crop reader (`latex_for_crop`) so the REAL `process_equation_region` chain runs.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

fitz = pytest.importorskip("fitz")

from ocr_output_contract import PAGE_MARKER_RE  # noqa: E402

from socr.core.audit_log import AuditEvent  # noqa: E402
from socr.core.config import PipelineConfig  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import SOCR_MARKER_RE  # noqa: E402
from socr.core.providers import PROFILE_GEMINI  # noqa: E402
from socr.core.result import (  # noqa: E402
    DocumentStatus,
    EngineResult,
    FailureMode,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState  # noqa: E402
from socr.pipeline.agentic import AcceptDecision  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:[.,]\d+)?")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text or ""))


class _AcceptSuccessOnly:
    """Stand-in for the composed judge: accepts only a SUCCESS-status reading."""

    def assess(self, output, provider):
        return AcceptDecision(
            accept=output.status is PageStatus.SUCCESS,
            reason="success status" if output.status is PageStatus.SUCCESS else "not success",
        )


def _state(tmp_path: Path, text: str = "old"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "source page")
    doc.save(pdf)
    doc.close()
    state = DocumentState(DocumentHandle(pdf))
    bo = PageOutput(
        page_num=1,
        text=text,
        status=PageStatus.SUCCESS,
        engine="gemini",
        provider_id=PROFILE_GEMINI.id,
        cost_usd=PROFILE_GEMINI.cost_per_page_usd,
    )
    ps = state.pages[1]
    ps.is_born_digital = False
    ps.attempts.append(bo)
    ps.best_output = bo
    return state, ps, bo


# ---------------------------------------------------------------------------
# 1 — an operational failure is not something a table crop repairs
# ---------------------------------------------------------------------------


class TestOperationalFailureIsNotLaundered:
    def test_rejudge_does_not_erase_an_operational_truncation(self, tmp_path: Path) -> None:
        state, ps, bo = _state(tmp_path, text="patched table")
        bo.status = PageStatus.ERROR
        bo.failure_mode = FailureMode.TRUNCATED
        bo.error = "completion stopped at token limit"
        pipeline = UnifiedPipeline(PipelineConfig(quiet=True, judge_backend="heuristic"))

        direct = _AcceptSuccessOnly().assess(bo, PROFILE_GEMINI)
        assert not direct.accept, "control: the same bytes cannot be accepted as an ERROR output"

        pipeline._rejudge_crop_patched_page(
            state, 1, ps, bo, "old table", _AcceptSuccessOnly(), PROFILE_GEMINI
        )
        assert bo.status is PageStatus.ERROR
        assert bo.failure_mode is FailureMode.TRUNCATED
        assert bo.text == "old table", "the unjudgeable patch must not ship either"
        refusals = [e for e in state.events if e.kind == "table_reread_rejudged"]
        assert refusals and not (refusals[0].data or {}).get("accepted", True)

    def test_a_clean_incumbent_is_still_promoted(self, tmp_path: Path) -> None:
        """Control: the refusal above is about the operational failure, not a
        blanket refusal of every promotion."""
        state, ps, bo = _state(tmp_path, text="patched table")
        pipeline = UnifiedPipeline(PipelineConfig(quiet=True, judge_backend="heuristic"))
        pipeline._rejudge_crop_patched_page(
            state, 1, ps, bo, "old table", _AcceptSuccessOnly(), PROFILE_GEMINI
        )
        assert bo.text == "patched table"
        assert bo.audit_passed is True


# ---------------------------------------------------------------------------
# 2 — the equation seam reuses main's guards, and they are stronger
# ---------------------------------------------------------------------------


def _drive_equation_seam(tmp_path: Path, raw_latex: str, native_text: str):
    """Run the REAL GH-36b seam with only the crop reader stubbed.

    `process_equation_region` (PR #518's choke point) therefore runs for real,
    including its assembly-contract gate.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    pipeline = UnifiedPipeline(PipelineConfig(quiet=True))
    state = MagicMock(spec=DocumentState)
    state.events = [
        AuditEvent(
            page_num=1,
            kind="equation_region_detected",
            engine="",
            detail="",
            data={"crop_path": str(tmp_path / "eq_p1_r0.png")},
        )
    ]
    page_state = MagicMock()
    page_state.native_text = native_text
    page_state.has_corrupt_math = False
    page_state.has_encoding_hygiene_suspect = False
    state.pages = {1: page_state}

    out = PageOutput(page_num=1, text=native_text, status=PageStatus.SUCCESS, engine="qwen")
    with patch("socr.math.equation_latex.latex_for_crop", return_value=raw_latex):
        pipeline._attach_equation_latex_sidecars(state, [out])
    return out, state


class TestEquationGuardsAreMainsGuards:
    @pytest.mark.parametrize("latex", [r"x^9", r"10^9", r"x^{9}"])
    def test_exponent_digit_must_not_evade_the_presence_guard(
        self, tmp_path: Path, latex: str
    ) -> None:
        """PR #518 removed the exponent exclusion from the tokenizer precisely
        because `x^9` let an invented 9 through. Re-adding one at a second seam
        would reopen the closed hole."""
        # The scratch directory name must carry no digits: it lands in the
        # crop path inside the sidecar, and a digit there would be mistaken for
        # the invented one this test is looking for.
        out, state = _drive_equation_seam(
            tmp_path / "exponent",
            raw_latex=latex,
            native_text="The estimate is 42.8 per cent in 2024.",
        )
        assert "```latex" not in out.text, f"an invented 9 hid inside {latex!r} and was attached"
        assert any(e.kind == "equation_sidecar_refused" for e in state.events)

    @pytest.mark.parametrize(
        "marker", ["## page 2", "    ## Page 2", "## Page 2 trailing"], ids=[0, 1, 2]
    )
    def test_guard_matches_the_actual_page_contract(self, tmp_path: Path, marker: str) -> None:
        """The contract's own regex is case-insensitive, allows leading
        whitespace and trailing text. A guard that misses those forms lets a
        model reading split the document."""
        assert PAGE_MARKER_RE.search(marker), "control: the contract splits on this"
        out, _state = _drive_equation_seam(
            tmp_path / f"m{abs(hash(marker))}",
            raw_latex=f"y = 2x + 1\n{marker}",
            native_text="The line is y equals 2 x plus 1.",
        )
        assert not PAGE_MARKER_RE.search(out.text), (
            "a model-authored page boundary reached the shipped body"
        )

    def test_a_grounded_sidecar_is_still_attached(self, tmp_path: Path) -> None:
        out, state = _drive_equation_seam(
            tmp_path / "clean",
            raw_latex="x = 42.8",
            native_text="The estimate is 42.8 per cent.",
        )
        assert "42.8" in out.text
        assert "```latex" in out.text, "the guard must not refuse a grounded sidecar"
        assert not any(e.kind == "equation_sidecar_refused" for e in state.events)

    def test_a_page_with_no_numbers_is_unverifiable_not_invented(self, tmp_path: Path) -> None:
        """`region_presence_verdict` abstains when the page has no numeric
        oracle, so notation-only LaTeX on a prose page is not convicted."""
        out, state = _drive_equation_seam(
            tmp_path / "abstain",
            raw_latex=r"E = mc^2",
            native_text="E equals m c squared linearised",
        )
        assert "```latex" in out.text
        assert not any(e.kind == "equation_sidecar_refused" for e in state.events)


# ---------------------------------------------------------------------------
# 2b — the fabricated-ref sanitizer adds only marker prose
# ---------------------------------------------------------------------------


class TestImageRefSanitizerIsSubtractiveOutsideMarkers:
    def test_image_ref_exception_is_really_subtractive(self, tmp_path: Path) -> None:
        """The round-2 version of this test was vacuous: its MagicMock state had
        no usable handle, the URL-index lookup raised, and only phantom deletion
        ran. With the index forced empty the fabricated-ref gate really runs --
        and it REPLACES the ref with socr's own marker.

        The invariant is therefore about CONTENT: every token added lies inside
        a recognised socr marker span, and no content token changes."""
        pipeline = UnifiedPipeline(PipelineConfig(quiet=True, save_figures=False))
        state = MagicMock()
        state.events = []
        state.pages = {}
        before = "Revenue 42.8.\n\n![chart](https://example.invalid/invented999.png)"
        out = PageOutput(page_num=1, text=before, status=PageStatus.SUCCESS, engine="qwen")

        with patch.object(pipeline, "_source_url_index", return_value=frozenset()):
            pipeline._sanitize_agentic_page_image_refs(state, 1, out, tmp_path)

        assert out.text != before, "the fixture must actually trigger the fabricated-ref gate"
        outside_markers = SOCR_MARKER_RE.sub(" ", out.text)
        assert _tokens(outside_markers) <= _tokens(before), (
            "content outside socr's own markers must be subtractive only"
        )
        assert "invented999" not in out.text


# ---------------------------------------------------------------------------
# 3 — judge-event ownership per invocation
# ---------------------------------------------------------------------------


class TestJudgeEventOwnership:
    def test_timed_out_rejudge_cannot_leak_a_late_event(self, tmp_path: Path) -> None:
        state, ps, bo = _state(tmp_path, text="patched")
        pipeline = UnifiedPipeline(PipelineConfig(quiet=True, judge_backend="heuristic"))
        release = threading.Event()
        finished = threading.Event()

        class _LateEventJudge:
            def assess(self, output, provider):
                release.wait(timeout=2)
                pipeline._record_judge_event(
                    state,
                    AuditEvent(
                        page_num=1, kind="late_candidate_hard_fail", engine="gemini", detail="x"
                    ),
                )
                finished.set()
                return AcceptDecision(accept=False, reason="late hard fail")

        judge = pipeline._TimeoutJudge(_LateEventJudge(), timeout_sec=0.01)
        pipeline._rejudge_crop_patched_page(state, 1, ps, bo, "old", judge, PROFILE_GEMINI)
        release.set()
        assert finished.wait(timeout=2)
        assert "late_candidate_hard_fail" not in [e.kind for e in state.events]

    def test_a_timed_out_worker_cannot_reach_a_later_rejudge(self, tmp_path: Path) -> None:
        """Cross-talk: the abandoned worker of re-judge A must not land in the
        scratch list of re-judge B, whose events DO reach the document."""
        state, ps, bo = _state(tmp_path, text="patched-a")
        pipeline = UnifiedPipeline(PipelineConfig(quiet=True, judge_backend="heuristic"))
        a_running = threading.Event()
        b_running = threading.Event()
        a_emitted = threading.Event()

        class _SlowJudgeA:
            def assess(self, output, provider):
                a_running.set()
                b_running.wait(timeout=2)
                pipeline._record_judge_event(
                    state,
                    AuditEvent(page_num=1, kind="late_from_a", engine="gemini", detail="x"),
                )
                a_emitted.set()
                return AcceptDecision(accept=False, reason="late")

        class _JudgeB:
            def assess(self, output, provider):
                b_running.set()
                assert a_emitted.wait(timeout=2)
                pipeline._record_judge_event(
                    state,
                    AuditEvent(page_num=1, kind="from_b", engine="gemini", detail="x"),
                )
                return AcceptDecision(accept=True, reason="b accepted")

        judge_a = pipeline._TimeoutJudge(_SlowJudgeA(), timeout_sec=0.01, owner=pipeline)
        pipeline._rejudge_crop_patched_page(state, 1, ps, bo, "old", judge_a, PROFILE_GEMINI)
        assert a_running.wait(timeout=2)

        bo.text = "patched-b"
        judge_b = pipeline._TimeoutJudge(_JudgeB(), timeout_sec=5.0, owner=pipeline)
        pipeline._rejudge_crop_patched_page(state, 1, ps, bo, "old", judge_b, PROFILE_GEMINI)

        kinds = [e.kind for e in state.events]
        assert "late_from_a" not in kinds, "an abandoned worker's event reached the document"
        assert "from_b" in kinds, "the accepting re-judge's own events must be kept"


# ---------------------------------------------------------------------------
# 4 — the re-judge is attributed to the judge, and survives resume
# ---------------------------------------------------------------------------


class TestRejudgeMeteringAttribution:
    def test_heuristic_rejudge_is_not_charged_as_gemini_ocr(self, tmp_path: Path) -> None:
        state, ps, bo = _state(tmp_path, text="patched")
        pipeline = UnifiedPipeline(PipelineConfig(quiet=True, judge_backend="heuristic"))
        pipeline._rejudge_crop_patched_page(
            state, 1, ps, bo, "old", _AcceptSuccessOnly(), PROFILE_GEMINI
        )
        run = state.engine_runs[-1]
        assert run.engine == "heuristic"
        assert run.cost == 0.0

    def test_the_event_names_the_judge_that_actually_ran(self, tmp_path: Path) -> None:
        """`state.agentic_judge_model` records the judge that was BUILT; a
        degraded-to-heuristic run must not be journalled as the VLM that never
        ran."""
        state, ps, bo = _state(tmp_path, text="patched")
        state.agentic_judge_model = "heuristic"
        pipeline = UnifiedPipeline(PipelineConfig(quiet=True, judge_backend="ollama"))
        with patch.object(pipeline, "_resolve_judge_model", return_value="some-vlm:latest"):
            pipeline._rejudge_crop_patched_page(
                state, 1, ps, bo, "old", _AcceptSuccessOnly(), PROFILE_GEMINI
            )
        event = [e for e in state.events if e.kind == "table_reread_rejudged"][-1]
        assert (event.data or {})["judge_model"] == "heuristic"
        assert state.engine_runs[-1].engine == "heuristic"

    def test_rejudge_cost_survives_terminal_resume(self, tmp_path: Path) -> None:
        state, ps, bo = _state(tmp_path, text="patched")
        config = PipelineConfig(quiet=True, judge_backend="heuristic", reprocess=True)
        pipeline = UnifiedPipeline(config)
        # Round 5: per-page spend is a RECORDED FACT. Production records it at
        # the same site that journals the EngineResult, so the fixture does too;
        # deriving it from the attempt list is the defect round 5 removed.
        state.record_engine_run(
            EngineResult(
                document_path=state.handle.path,
                engine="gemini",
                status=DocumentStatus.SUCCESS,
                cost=PROFILE_GEMINI.cost_per_page_usd,
            )
        )
        UnifiedPipeline._add_page_cost(ps, PROFILE_GEMINI.cost_per_page_usd)
        pipeline._rejudge_crop_patched_page(
            state, 1, ps, bo, "old", _AcceptSuccessOnly(), PROFILE_GEMINI
        )
        original_cost = state.total_cost
        out_dir = tmp_path / "out"
        with patch.object(pipeline, "_resolve_judge_model", return_value=None):
            pipeline._flush_page_fragment(state, 1, bo.text, out_dir)
            pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)

        resumed = DocumentState(DocumentHandle(state.handle.path))
        resumed_pipe = UnifiedPipeline(config)
        with patch.object(resumed_pipe, "_resolve_judge_model", return_value=None):
            loaded = resumed_pipe._load_terminal_page(resumed, 1, out_dir)
        assert loaded is not None, "control: a promoted SUCCESS/audit_passed page is terminal"
        resumed_pipe._restore_terminal_page_state(resumed, 1, loaded, out_dir)
        assert resumed.total_cost == original_cost
