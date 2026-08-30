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
    def test_shifted_labels_seed_prior_findings_on_the_real_rung(self, tmp_path: Path) -> None:
        """The mechanical FAIL is prepended as rung 0 and tiebreaks into the
        real rung with the contradiction's findings attached -- proving the
        composition (not a post-hoc overwrite) actually happened, even
        though a subsequent high-confidence PASS still wins per A4's own
        "never held hostage by an earlier B" rule."""
        pipeline = _make_pipeline()
        pdf_path = _row_shift_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_SHIFTED_MD)

        rung = _accept_rung("high")
        pipeline._run_table_judge_gate(state, 1, ps, bo, [rung])

        assert len(rung.calls) == 1
        _crop_path, _markdown, prior_findings = rung.calls[0]
        assert prior_findings, "the real rung must see the mechanical contradiction's findings"
        assert all(f.code.value == "WRONG_BINDING" for f in prior_findings)
        # A4's own contract: a later high-confidence PASS still accepts.
        assert ps.table_ladder_disposition is None

    def test_shifted_labels_with_weak_real_pass_stays_unverified(self, tmp_path: Path) -> None:
        """A LOW-confidence PASS after the mechanical tiebreak is not
        corroboration (A4: "a lone weak witness is not consensus") -- the
        page lands UNVERIFIED, neither silently accepted nor REJECTED."""
        pipeline = _make_pipeline()
        pdf_path = _row_shift_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_SHIFTED_MD)

        rung = _accept_rung("low")
        pipeline._run_table_judge_gate(state, 1, ps, bo, [rung])

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED

    def test_shifted_labels_with_no_rungs_available_forces_rejected(self, tmp_path: Path) -> None:
        """No CLI available at all (strict_local, or every rung otherwise
        unavailable): the mechanical rung is the sole -- and therefore
        last -- rung in the sequence, so its FAIL exhausts the ladder to
        REJECTED unconditionally. The mechanical check needs no cloud
        egress, so an empty ``rungs`` list does not exempt it."""
        pipeline = _make_pipeline()
        pdf_path = _row_shift_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_SHIFTED_MD)

        pipeline._run_table_judge_gate(state, 1, ps, bo, [])

        assert ps.table_ladder_disposition == FailureMode.TABLE_REJECTED
        events = _events_of_kind(state, "table_ladder_rejected")
        assert len(events) == 1
        assert events[0].data["rung_trail"] == [
            {"rung": "mechanical:binding", "ok": True, "executing": "mechanical binding check"}
        ]

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
        assert "table_rejected" in (result_on.error or "")

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
