"""TICKET-E1 (GH-353): mechanical binding evidence at the table judge gate.

``tables/binding.py bind()`` is a pure, cloud-free geometric check that
catches the failure shape judges have been measured to miss: a value
multiset that is completely correct but bound to the wrong row/column
(GH-273 -- two frontier judges both blessed that exact defect). This file
pins the gate-level composition (``_run_table_judge_gate`` /
``UnifiedPipeline.process``), not ``bind()`` itself (see ``test_binding.py``
for the oracle's own unit tests, including the identical GH-273 fixture this
file's PDF-based fixture mirrors).

Hermetic throughout, matching ``test_table_judge_gate.py``'s own contract:
CI has no ollama and no ``gemini`` binary, so every rung here is an injected
fake; ``_available_engines_for_agentic`` and ``_resolve_judge_model`` are
patched wherever ``process()`` runs. Per the #253/#257 trap, the flag-on/
flag-off comparison never pins an absolute outcome measured on one machine
-- it runs both configurations and asserts the DIFFERENCE the flag makes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

from socr.core.audit_log import AuditEvent
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import DocumentStatus, FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.judge.table_verdict import (
    TABLE_LADDER_ACCEPTED_KIND,
    RungResult,
    TableJudgeVerdict,
)
from socr.pipeline.orchestrator import UnifiedPipeline

# ---------------------------------------------------------------------------
# Fixtures: a ruled table whose native text is FIXED (OLS/IV header, RowA ->
# 100/200, RowB -> 300/400) -- the exact GH-273 shape (identical value
# multiset, only the row-label binding differs) built as a real PDF so the
# gate's own ``open_pdf``/``get_text("words")`` read exercises real geometry,
# not synthetic word tuples (that unit-level coverage already lives in
# ``test_binding.py::test_numeric_row_anchors_reject_shifted_row_labels``).
# ---------------------------------------------------------------------------

_CORRECT_MD = (
    "|      | OLS | IV  |\n| ---- | --- | --- |\n| RowA | 100 | 200 |\n| RowB | 300 | 400 |\n"
)
_SHIFTED_MD = (
    "|      | OLS | IV  |\n| ---- | --- | --- |\n| RowB | 100 | 200 |\n| RowA | 300 | 400 |\n"
)


def _row_shift_pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
    """One page, one ruled table: header (OLS, IV) + RowA(100,200) +
    RowB(300,400) -- the native ground truth ``_CORRECT_MD`` matches and
    ``_SHIFTED_MD`` contradicts (same values, swapped row labels)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    cols = [100, 250, 380]
    rows = [100, 122, 144]
    page.insert_text((cols[1] + 4, rows[0] + 12), "OLS", fontsize=9)
    page.insert_text((cols[2] + 4, rows[0] + 12), "IV", fontsize=9)
    page.insert_text((cols[0] + 4, rows[1] + 12), "RowA", fontsize=9)
    page.insert_text((cols[1] + 4, rows[1] + 12), "100", fontsize=9)
    page.insert_text((cols[2] + 4, rows[1] + 12), "200", fontsize=9)
    page.insert_text((cols[0] + 4, rows[2] + 12), "RowB", fontsize=9)
    page.insert_text((cols[1] + 4, rows[2] + 12), "300", fontsize=9)
    page.insert_text((cols[2] + 4, rows[2] + 12), "400", fontsize=9)
    for yy in [*rows, rows[-1] + 22]:
        page.draw_line((100, yy), (460, yy))
    for xx in [*cols, 460]:
        page.draw_line((xx, rows[0]), (xx, rows[-1] + 22))
    pdf_path = tmp_path / name
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _no_native_words_pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
    """A ruled box with the SAME geometry, but zero native text -- the
    ticket's own acceptance fixture ("no native words is not demoted by
    binding alone"). ``locate_tables`` finds the box from the ruling lines
    alone; ``page.get_text("words")`` is empty."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    cols = [100, 250, 380]
    rows = [100, 122, 144]
    for yy in [*rows, rows[-1] + 22]:
        page.draw_line((100, yy), (460, yy))
    for xx in [*cols, 460]:
        page.draw_line((xx, rows[0]), (xx, rows[-1] + 22))
    pdf_path = tmp_path / name
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _pass_verdict(confidence: str = "high") -> TableJudgeVerdict:
    return TableJudgeVerdict(verdict="PASS", confidence=confidence, findings=[])


class _QueueRung:
    """A ``RungCallable`` that returns pre-baked results in order and records
    every call (crop_path, markdown, prior_findings) for assertion."""

    def __init__(self, results: list[RungResult], rung_id: str = "fake") -> None:
        self._results = list(results)
        self.rung_id = rung_id
        self.calls: list[tuple] = []

    def __call__(self, crop_path, markdown, prior_findings):
        self.calls.append((crop_path, markdown, prior_findings))
        if not self._results:
            raise AssertionError(f"{self.rung_id} called more times than results provided")
        return self._results.pop(0)


def _accept_rung(confidence: str = "high") -> _QueueRung:
    return _QueueRung([RungResult(rung="fake1", ok=True, verdict=_pass_verdict(confidence))])


def _fail_verdict(confidence: str = "high") -> TableJudgeVerdict:
    from socr.judge.table_verdict import Finding, FindingCode

    return TableJudgeVerdict(
        verdict="FAIL",
        confidence=confidence,
        findings=[Finding(code=FindingCode.FABRICATED_VALUE, where="cell", detail="bad value")],
    )


def _reject_rung() -> _QueueRung:
    return _QueueRung([RungResult(rung="fake1", ok=True, verdict=_fail_verdict("high"))])


# ---------------------------------------------------------------------------
# Pipeline / state helpers (mirrors test_table_judge_gate.py's own)
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> PipelineConfig:
    kwargs = dict(
        primary_engine=EngineType.QWEN,
        agentic=True,
        judge_backend="heuristic",
        enabled_engines=[EngineType.QWEN],
        quiet=True,
        save_figures=False,
        write_manifest=False,
        table_judge_ladder=True,
    )
    kwargs.update(overrides)
    return PipelineConfig(**kwargs)


def _make_pipeline(config: PipelineConfig | None = None) -> UnifiedPipeline:
    return UnifiedPipeline(config or _make_config())


def _make_state(pdf_path: Path, page_count: int = 1) -> DocumentState:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf_path, page_count=page_count)
    return DocumentState(handle=handle)


def _bo(text: str, engine: str = "qwen") -> PageOutput:
    return PageOutput(
        page_num=1,
        text=text,
        status=PageStatus.SUCCESS,
        engine=engine,
        audit_passed=True,
    )


def _events_of_kind(state: DocumentState, kind: str) -> list[AuditEvent]:
    return [e for e in state.events if e.kind == kind]


# ---------------------------------------------------------------------------
# Gate-level: direct _run_table_judge_gate calls
# ---------------------------------------------------------------------------


class TestNoNativeWordsStaysNeutral:
    def test_no_native_words_not_demoted_by_binding_alone(self, tmp_path: Path) -> None:
        """The ticket's own acceptance criterion: a witnessed table with zero
        native words must not be demoted by binding alone -- only ONE rung
        call happens (the real one), never a synthesized mechanical FAIL."""
        pipeline = _make_pipeline()
        pdf_path = _no_native_words_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_CORRECT_MD)

        rung = _accept_rung()
        pipeline._run_table_judge_gate(state, 1, ps, bo, [rung])

        assert len(rung.calls) == 1
        assert ps.table_ladder_disposition is None
        events = _events_of_kind(state, TABLE_LADDER_ACCEPTED_KIND)
        assert len(events) == 1
        assert events[0].data["rung_trail"] == [
            {"rung": "fake1", "ok": True, "executing": pipeline.config.table_judge_rung1_model}
        ]


class TestRowLabelShiftComposesWithLadder:
    def test_shifted_labels_do_not_seed_findings_and_cap_at_unverified(
        self, tmp_path: Path
    ) -> None:
        """CONSILIUM (panel #3, GH-353 E1) originally specified: "the
        mechanical FAIL is prepended as rung 0 and tiebreaks into the
        real rung with the contradiction's findings attached."

        GH-359 ruling 4 overturns the findings-injection half of that
        composition: judge input is crop + markdown, nothing else, so the
        mechanical check cannot occupy a fake ladder slot. The rest of
        panel #3 stands: a genuine contradiction caps acceptance at
        UNVERIFIED (not REJECTED -- GH-334: the native text layer can be
        the culprit), does not inject findings, and does not overrule a
        ladder REJECTED.
        """
        pipeline = _make_pipeline()
        pdf_path = _row_shift_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_SHIFTED_MD)

        rung = _accept_rung("high")
        with patch.object(pipeline, "_transcribe_cell_token", return_value=None):
            pipeline._run_table_judge_gate(state, 1, ps, bo, [rung])

        assert len(rung.calls) == 1
        _crop_path, _markdown, prior_findings = rung.calls[0]
        assert prior_findings is None
        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        events = _events_of_kind(state, "table_ladder_unverified")
        assert len(events) == 1
        assert "mechanical binding check found a contradiction" in events[0].detail

    def test_shifted_labels_with_weak_real_pass_stays_unverified(self, tmp_path: Path) -> None:
        """A LOW-confidence PASS after the mechanical tiebreak is not
        corroboration (A4: "a lone weak witness is not consensus") -- the
        page lands UNVERIFIED, neither silently accepted nor REJECTED --
        the same terminal the clamp would have produced anyway."""
        pipeline = _make_pipeline()
        pdf_path = _row_shift_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_SHIFTED_MD)

        rung = _accept_rung("low")
        pipeline._run_table_judge_gate(state, 1, ps, bo, [rung])

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED

    def test_shifted_labels_with_rejecting_real_rung_is_unverified_by_the_contradiction(
        self, tmp_path: Path
    ) -> None:
        """A reader FAIL over an ACTIVE row-label contradiction ends UNVERIFIED.

        GH-575: the mechanical contradiction is terminal and the blind
        adjudicator is never asked, so the readers' FAIL cannot become a
        content rejection or a withhold on this page."""
        pipeline = _make_pipeline()
        pdf_path = _row_shift_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_SHIFTED_MD)

        rung = _reject_rung()
        pipeline._run_table_judge_gate(state, 1, ps, bo, [rung])

        # GH-575 (cold review round 1, finding 1). This fixture's native
        # geometry ACTIVELY contradicts the emitted grid -- that is the whole
        # point of a row-label shift -- and an active contradiction is now a
        # terminal: the table ends UNVERIFIED and the blind adjudicator is
        # never asked. Withholding needs a blind reader to have disagreed, and
        # here nobody was allowed to look.
        #
        # GH-575 SUPERSEDES ruling 5's "the clamp never claims a rejected
        # page" for the contradiction case, deliberately and in one direction
        # only: mechanical evidence still cannot turn anything into a content
        # REJECTION, but an active contradiction now settles the terminal
        # before the readers' verdict is consulted, so the event that ships is
        # the contradiction's. The reader FAIL reaching neither REJECTED nor
        # WITHHELD is the assertion that says so.
        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        assert not _events_of_kind(state, "table_ladder_withheld")
        assert not _events_of_kind(state, "table_ladder_rejected")
        events = _events_of_kind(state, "table_ladder_unverified")
        assert len(events) == 1
        assert "mechanical binding check found a contradiction" in events[0].detail

    def test_shifted_labels_with_no_rungs_available_is_unverified(self, tmp_path: Path) -> None:
        """GH-359 ruling 5: no CLI available plus a mechanical
        contradiction is UNVERIFIED, not REJECTED. Empty-rungs REJECTED
        was leftover of panel #3's rung-0 composition (mechanical FAIL as
        last rung). Mechanical evidence withholds accept; it does not
        force a content verdict (GH-334)."""
        pipeline = _make_pipeline()
        pdf_path = _row_shift_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_SHIFTED_MD)

        pipeline._run_table_judge_gate(state, 1, ps, bo, [])

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        events = _events_of_kind(state, "table_ladder_unverified")
        assert len(events) == 1
        assert "mechanical binding check found a contradiction" in events[0].detail
        assert events[0].data["rung_trail"] == []

    def test_correct_labels_with_no_rungs_available_stays_unverified(self, tmp_path: Path) -> None:
        """Sanity: the SAME PDF (same native geometry), but candidate
        markdown that actually agrees with it -- no contradiction, so the
        pre-E1 fail-open behaviour (empty rungs -> UNVERIFIED, no ladder
        call) is unchanged."""
        pipeline = _make_pipeline()
        pdf_path = _row_shift_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_CORRECT_MD)

        pipeline._run_table_judge_gate(state, 1, ps, bo, [])

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        assert not _events_of_kind(state, "table_ladder_rejected")


# ---------------------------------------------------------------------------
# Full process() runs -- the ticket's required flag on/off comparison,
# asserting the DIFFERENCE only (never an absolute outcome pinned).
# ---------------------------------------------------------------------------


def _route_page_returning(text: str, engine: str = "qwen"):
    def _fake_route(page_num, ladder, run_provider, judge, **kwargs):
        from socr.pipeline.agentic import PageDecision, ProviderAttempt

        out = PageOutput(
            page_num=page_num,
            text=text,
            status=PageStatus.SUCCESS,
            engine=engine,
            audit_passed=True,
        )
        prof = ladder[0]
        att = ProviderAttempt(
            engine=prof.engine,
            output=out,
            cost_usd=prof.cost_per_page_usd,
            accepted=True,
            reason="ok",
            provider_id=prof.id,
            model=prof.model,
            backend=prof.backend,
        )
        return PageDecision(page_num=page_num, final_output=out, attempts=[att], accepted=True)

    return _fake_route


class TestProcessFlagDifference:
    def test_flag_off_ships_unaffected_flag_on_demotes(self, tmp_path: Path) -> None:
        pdf_off = _row_shift_pdf(tmp_path / "off", "doc.pdf")
        pdf_on = _row_shift_pdf(tmp_path / "on", "doc.pdf")

        # -- flag off: baseline, the shifted-label defect ships untouched --
        cfg_off = _make_config(table_judge_ladder=False)
        pipeline_off = _make_pipeline(cfg_off)
        with (
            patch(
                "socr.pipeline.orchestrator.route_page",
                side_effect=_route_page_returning(_SHIFTED_MD),
            ),
            patch.object(
                pipeline_off, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline_off, "_resolve_judge_model", return_value=""),
        ):
            result_off = pipeline_off.process(pdf_off, tmp_path / "off_out")

        assert result_off.status == DocumentStatus.SUCCESS
        assert "table_rejected" not in (result_off.error or "")

        # -- flag on, no CLI rungs available: mechanical check alone -------
        cfg_on = _make_config(table_judge_ladder=True)
        pipeline_on = _make_pipeline(cfg_on)
        with (
            patch(
                "socr.pipeline.orchestrator.route_page",
                side_effect=_route_page_returning(_SHIFTED_MD),
            ),
            patch.object(
                pipeline_on, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline_on, "_resolve_judge_model", return_value=""),
            patch.object(pipeline_on, "_build_table_judge_rungs", return_value=[]),
        ):
            result_on = pipeline_on.process(pdf_on, tmp_path / "on_out")

        assert result_on.status == DocumentStatus.AUDIT_FAILED
        assert "table_unverified" in (result_on.error or "")
        assert "table_rejected" not in (result_on.error or "")

    def test_flag_on_no_native_words_ships_success(self, tmp_path: Path) -> None:
        """The ticket's no-native-words acceptance criterion at the full
        process() level: flag on, zero native text on the page, a real
        (accepting) rung -- the page ships SUCCESS, not demoted."""
        pdf_path = _no_native_words_pdf(tmp_path)
        cfg = _make_config(table_judge_ladder=True)
        pipeline = _make_pipeline(cfg)

        with (
            patch(
                "socr.pipeline.orchestrator.route_page",
                side_effect=_route_page_returning(_CORRECT_MD),
            ),
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch.object(pipeline, "_build_table_judge_rungs", return_value=[_accept_rung()]),
        ):
            result = pipeline.process(pdf_path, tmp_path / "no_words_out")

        assert result.status == DocumentStatus.SUCCESS
        assert "table_rejected" not in (result.error or "")


# ---------------------------------------------------------------------------
# P1 (task t5): the three-way binding evidence classification.
#
# ``_binding_contradiction_for_witness`` was CONTRADICTION-ONLY: it returns a
# ``BindingResult`` iff a genuine cell/row-label contradiction fired, else
# None -- collapsing "structurally proven correct" and "nothing checkable"
# into the same falsy value. Q1/Q2's guard chain needs to tell those apart:
# a full PASS overrules a reader outright; an ABSTAIN falls through to the
# blind-cell adjudicator instead of stopping the chain.
#
# ``classify_binding_evidence`` is the pure, BindingResult-only classifier
# (no PDF, no pipeline) this ticket adds; the gate-level refactor
# (``_binding_evidence_for_witness`` returning ``(BindingResult, BindingEvidence)``)
# is exercised via the fixtures already defined above in this file.
# ---------------------------------------------------------------------------

from socr.tables.binding import (  # noqa: E402
    BindingEvidence,
    BindingResult,
    ColumnHeaderPath,
    ContradictedCell,
    MatchedCell,
    RowLabelContradiction,
    classify_binding_evidence,
)


def _fully_checked_result(*, contradicted: bool) -> BindingResult:
    """A BindingResult that IS ``fully_checked`` -- every lane/row/cell was
    resolved -- and either agrees (PASS candidate) or disagrees (CONTRADICT)."""
    matched = (
        [] if contradicted else [MatchedCell(row_path=("RowA",), col_path=("OLS",), value="100")]
    )
    contradicted_cells = (
        [
            ContradictedCell(
                row_path=("RowA",),
                col_path=("OLS",),
                native_token="100",
                model_token="999",
            )
        ]
        if contradicted
        else []
    )
    return BindingResult(
        matched_cells=matched,
        contradicted_cells=contradicted_cells,
        row_label_contradictions=[],
        native_unbound=[],
        model_unbound=[],
        ambiguous_count=0,
        row_binding_unverifiable=False,
        row_label_unverifiable=False,
        column_binding_unverifiable=False,
        column_header_paths=[ColumnHeaderPath(lane=0, path=("OLS",), spans_lanes=1)],
    )


class TestClassifyBindingEvidence:
    def test_fully_checked_no_contradiction_is_pass(self) -> None:
        result = _fully_checked_result(contradicted=False)
        assert result.structural_agreement is True
        assert classify_binding_evidence(result) is BindingEvidence.PASS

    def test_fully_checked_with_contradicted_cells_is_contradict(self) -> None:
        result = _fully_checked_result(contradicted=True)
        assert classify_binding_evidence(result) is BindingEvidence.CONTRADICT

    def test_row_label_contradiction_is_contradict_even_if_otherwise_checked(self) -> None:
        result = BindingResult(
            row_label_contradictions=[
                RowLabelContradiction(row_path=("RowA",), candidate_label="RowB", native_bbox=None)
            ],
            row_binding_unverifiable=False,
            row_label_unverifiable=False,
            column_binding_unverifiable=False,
        )
        assert classify_binding_evidence(result) is BindingEvidence.CONTRADICT

    def test_default_unverifiable_result_is_abstain(self) -> None:
        """The dataclass default (no box / no native words path): nothing was
        checked, nothing to disagree with -- ABSTAIN, not PASS."""
        result = BindingResult()
        assert result.structural_agreement is False
        assert classify_binding_evidence(result) is BindingEvidence.ABSTAIN

    def test_partially_checked_with_no_contradiction_is_abstain_not_pass(self) -> None:
        """The numeric multiset alone must never produce PASS: a row/column
        left unverifiable means real coverage gaps, even with zero
        contradictions recorded."""
        result = BindingResult(
            row_binding_unverifiable=True,
            row_label_unverifiable=False,
            column_binding_unverifiable=False,
        )
        assert result.no_known_contradiction is True
        assert result.fully_checked is False
        assert classify_binding_evidence(result) is BindingEvidence.ABSTAIN

    def test_ambiguous_cell_with_no_contradiction_is_abstain(self) -> None:
        result = BindingResult(
            row_binding_unverifiable=False,
            row_label_unverifiable=False,
            column_binding_unverifiable=False,
            ambiguous_count=1,
        )
        assert classify_binding_evidence(result) is BindingEvidence.ABSTAIN

    def test_unbound_native_or_model_cells_are_contradict_not_abstain(self) -> None:
        """``no_known_contradiction`` already treats native_unbound/model_unbound
        (the dropped/invented-digit C4 signal) as evidence of disagreement --
        the classifier must not silently soften that back to ABSTAIN."""
        from socr.tables.binding import UnboundCell

        result = BindingResult(
            row_binding_unverifiable=False,
            row_label_unverifiable=False,
            column_binding_unverifiable=False,
            native_unbound=[UnboundCell(row_path=("RowA",), col_path=("OLS",), token="100")],
        )
        assert classify_binding_evidence(result) is BindingEvidence.CONTRADICT


class TestBindingEvidenceIsClosed:
    def test_exactly_three_members(self) -> None:
        assert {e.name for e in BindingEvidence} == {"PASS", "CONTRADICT", "ABSTAIN"}


# ---------------------------------------------------------------------------
# Gate-level refactor: the E1 clamp's existing behaviour (contradiction ->
# TABLE_UNVERIFIED, absence of coverage -> neutral) must survive verbatim
# through the new evidence-typed helper. These reuse the row-shift/no-native
# fixtures already defined in this file.
# ---------------------------------------------------------------------------


class TestGateBindingEvidenceHelper:
    def test_shifted_labels_classify_as_contradict_at_the_gate(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        pdf_path = _row_shift_pdf(tmp_path)
        state = _make_state(pdf_path)

        from socr.tables.witness import prepare_table_witnesses

        with prepare_table_witnesses(pdf_path, 1, _SHIFTED_MD) as witnesses:
            assert witnesses, "fixture premise: the table region must be located"
            binding_result, evidence = pipeline._binding_evidence_for_witness(
                state, 1, witnesses[0]
            )

        assert binding_result is not None
        assert evidence is BindingEvidence.CONTRADICT

    def test_correct_labels_classify_as_pass_at_the_gate(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        pdf_path = _row_shift_pdf(tmp_path)
        state = _make_state(pdf_path)

        from socr.tables.witness import prepare_table_witnesses

        with prepare_table_witnesses(pdf_path, 1, _CORRECT_MD) as witnesses:
            assert witnesses
            _binding_result, evidence = pipeline._binding_evidence_for_witness(
                state, 1, witnesses[0]
            )

        assert evidence is BindingEvidence.PASS

    def test_no_native_words_classifies_as_abstain_at_the_gate(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        pdf_path = _no_native_words_pdf(tmp_path)
        state = _make_state(pdf_path)

        from socr.tables.witness import prepare_table_witnesses

        with prepare_table_witnesses(pdf_path, 1, _CORRECT_MD) as witnesses:
            assert witnesses
            _binding_result, evidence = pipeline._binding_evidence_for_witness(
                state, 1, witnesses[0]
            )

        assert evidence is BindingEvidence.ABSTAIN

    def test_e1_clamp_unchanged_for_high_pass_with_contradiction(self, tmp_path: Path) -> None:
        """The GH-367 E1 clamp/adjudicate path for an already-accepted high
        PASS with a contradiction must be untouched by this refactor -- this
        task does not let structural_agreement lift an existing conviction."""
        pipeline = _make_pipeline()
        pdf_path = _row_shift_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_SHIFTED_MD)

        rung = _accept_rung("high")
        with patch.object(pipeline, "_transcribe_cell_token", return_value=None):
            pipeline._run_table_judge_gate(state, 1, ps, bo, [rung])

        # Unchanged from TestRowLabelShiftComposesWithLadder above: a genuine
        # contradiction still caps acceptance at UNVERIFIED, never REJECTED,
        # even though the underlying evidence is now typed CONTRADICT.
        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
