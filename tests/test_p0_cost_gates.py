"""Hermetic regression guards for P0 cost gates and free-geometry short-circuit.

Covers five cost-governance defects surfaced by the 2026-11-27 audit:
1. ``--strict-local`` and explicit ``--max-cost-per-page 0`` must never fire a
   live Gemma Cloud generation probe or auto-select a cloud engine.
2. Local corrupt-math recovery (default ``qwen3-vl:30b-a3b-instruct``) must
   record a known ``0.0`` cost rather than ``None`` so it does not poison
   ``DocumentState.total_cost`` and lock ``--cost-budget`` to $0.00 for later
   pages; an unpriced ``:cloud`` math model still records ``None`` (fail-closed).
3. Blind-cell adjudication (``kimi-k2.6:cloud``, default $0) and cloud cell
   transcription (``glm-5.3-flash:cloud``) must refuse when ``--cost-budget``
   is exhausted or prior spend is unknown; ``PageLadderBudget`` skips must not
   bill ``_record_call``; ``adjudicate`` must skip doomed cloud transcriptions.
4. ``EngineType.GEMMA`` has a bounded entry in ``DEFAULT_PROVIDER_TIMEOUTS``
   and ``GemmaCloudEngine.process_pages`` catches ``subprocess.TimeoutExpired``
   into ``FailureMode.TIMEOUT``.
5. ``BindingEvidence.PASS`` on an admissible born-digital table short-circuits
   before paid cloud table-judge rungs (0 cloud calls, ``verified_by_geometry``),
   except when ``bo.judge_outcome == JUDGE_OUTCOME_TIMEOUT`` (#713 timeout
   credential still requires an executed visual rung).

Hermetic: no live Ollama, no network, no external CLI.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest

from socr.core.born_digital import DocumentAssessment, PageAssessment
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.providers import (
    PROFILE_GEMINI,
    PROFILE_QWEN_LOCAL,
)
from socr.core.result import (
    JUDGE_OUTCOME_TIMEOUT,
    EngineResult,
    FailureMode,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState
from socr.engines.gemma_cloud import GemmaCloudEngine
from socr.engines.registry import resolve_auto_engine_for_config
from socr.judge.ladder_budget import PageLadderBudget
from socr.judge.table_cell_guard import GuardDisposition, evaluate_cell_guard
from socr.judge.table_rung_ollama import BlindCellResult
from socr.judge.table_verdict import (
    RungResult,
    TableJudgeVerdict,
)
from socr.math.recover import CorruptMathRegion
from socr.pipeline.agentic import DEFAULT_PROVIDER_TIMEOUTS, AcceptDecision
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.adjudication import ContradictionItem, adjudicate
from socr.tables.binding import BindingEvidence


def _pdf_with_pages(tmp_path: Path, texts: list[str], name: str = "doc.pdf") -> Path:
    pdf = tmp_path / name
    doc = fitz.open()
    for text in texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
        else:
            # Draw non-white ink so blank-page detection does not treat a
            # synthetic scanned page as blank.
            page.draw_rect(fitz.Rect(72, 72, 160, 100), fill=(0, 0, 0))
    doc.save(pdf)
    doc.close()
    return pdf


def _ruled_table_pdf(tmp_path: Path, name: str = "table.pdf") -> Path:
    """Two-row ruled born-digital table whose text layer matches _TABLE_MD."""
    pdf = tmp_path / name
    doc = fitz.open()
    page = doc.new_page()
    cols = [100, 250, 380]
    rows = [100, 122, 144]
    page.insert_text((cols[1] + 4, rows[0] + 12), "OLS", fontsize=9)
    page.insert_text((cols[2] + 4, rows[0] + 12), "IV", fontsize=9)
    page.insert_text((cols[0] + 4, rows[1] + 12), "Alpha", fontsize=9)
    page.insert_text((cols[1] + 4, rows[1] + 12), "100", fontsize=9)
    page.insert_text((cols[2] + 4, rows[1] + 12), "200", fontsize=9)
    page.insert_text((cols[0] + 4, rows[2] + 12), "Beta", fontsize=9)
    page.insert_text((cols[1] + 4, rows[2] + 12), "300", fontsize=9)
    page.insert_text((cols[2] + 4, rows[2] + 12), "400", fontsize=9)
    for yy in [*rows, rows[-1] + 22]:
        page.draw_line((100, yy), (460, yy))
    for xx in [*cols, 460]:
        page.draw_line((xx, rows[0]), (xx, rows[-1] + 22))
    doc.save(pdf)
    doc.close()
    return pdf


_TABLE_MD = (
    "|       | OLS | IV  |\n| ----- | --- | --- |\n| Alpha | 100 | 200 |\n| Beta  | 300 | 400 |\n"
)


# ---------------------------------------------------------------------------
# 1. Strict-local & pinned-zero must never probe Gemma Cloud or auto-pick cloud
# ---------------------------------------------------------------------------


class TestStrictLocalNoGemmaCloudLeak:
    @pytest.mark.parametrize(
        "policy_kwargs",
        [
            {"strict_local": True},
            {"max_cost_per_page": 0.0, "max_cost_per_page_pinned": True},
        ],
        ids=["strict_local", "pinned_zero_cap"],
    )
    def test_is_available_for_config_blocks_before_cli_or_network_probe(
        self, policy_kwargs: dict
    ) -> None:
        engine = GemmaCloudEngine()
        cfg = PipelineConfig(enable_gemma_cloud=True, **policy_kwargs)
        with (
            patch.object(engine, "is_available", side_effect=AssertionError("must not probe CLI")),
            patch(
                "socr.engines.gemma_cloud.gemma_cloud_model_available",
                side_effect=AssertionError("must not POST to Ollama Cloud"),
            ),
        ):
            ok, reason = engine.is_available_for_config(cfg)
        assert ok is False
        assert reason

    @pytest.mark.parametrize(
        "policy_kwargs",
        [
            {"strict_local": True},
            {"max_cost_per_page": 0.0, "max_cost_per_page_pinned": True},
        ],
        ids=["strict_local", "pinned_zero_cap"],
    )
    def test_probe_engines_for_agentic_skips_gemma_probe_under_local_policy(
        self, policy_kwargs: dict
    ) -> None:
        cfg = PipelineConfig(
            agentic=True,
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.GEMMA],
            enable_gemma_cloud=True,
            **policy_kwargs,
        )
        pipe = UnifiedPipeline(cfg)
        with patch(
            "socr.engines.gemma_cloud.GemmaCloudEngine.is_available_for_config",
            return_value=(False, "should never be probed"),
        ) as probe:
            available, _unservable, _qwen_pin, gemma_pin = pipe._probe_engines_for_agentic()
        probe.assert_not_called()
        assert available == []
        assert gemma_pin == ""

    def test_resolve_auto_engine_for_config_skips_cloud_under_strict_local(self) -> None:
        """Paired difference: same engine availability, only strict_local flips."""

        def _fake_avail(self) -> bool:
            return self.name in {"gemma", "gemini", "glm"}

        with (
            patch("socr.engines.qwen.QwenEngine.is_available", return_value=False),
            patch("socr.engines.marker.MarkerEngine.is_available", return_value=False),
            patch("socr.engines.gemma_cloud.GemmaCloudEngine.is_available", _fake_avail),
            patch("socr.engines.gemini.GeminiEngine.is_available", _fake_avail),
            patch("socr.engines.glm.GLMEngine.is_available", _fake_avail),
        ):
            open_choice = resolve_auto_engine_for_config(
                PipelineConfig(strict_local=False, enable_gemma_cloud=True)
            )
            no_gemma_choice = resolve_auto_engine_for_config(
                PipelineConfig(strict_local=False, enable_gemma_cloud=False)
            )
            local_choice = resolve_auto_engine_for_config(
                PipelineConfig(strict_local=True, enable_gemma_cloud=True)
            )
            zero_cap_choice = resolve_auto_engine_for_config(
                PipelineConfig(
                    strict_local=False,
                    max_cost_per_page=0.0,
                    max_cost_per_page_pinned=True,
                    enable_gemma_cloud=True,
                )
            )

        assert open_choice is EngineType.GEMMA
        assert no_gemma_choice is EngineType.GEMINI
        assert local_choice is EngineType.GLM
        assert zero_cap_choice is EngineType.GLM


# ---------------------------------------------------------------------------
# 2. Local corrupt-math records 0.0 (doesn't poison budget); cloud stays None
# ---------------------------------------------------------------------------


class TestCorruptMathCostBudgetIntegrity:
    def test_local_corrupt_math_preserves_budget_for_later_paid_page(self, tmp_path: Path) -> None:
        """Page 1 runs default local corrupt-math; page 2 needs paid Gemini.
        With local cost=0.0, page 2 still gets its --cost-budget; with None it
        would have been starved to $0.00."""
        pdf = _pdf_with_pages(
            tmp_path,
            ["Clean prose\nPðAÞ ¼ 1", ""],
            name="two_page_budget.pdf",
        )
        assessment = DocumentAssessment(
            path=pdf,
            pages=[
                PageAssessment(
                    page_num=1,
                    is_born_digital=True,
                    native_text="Clean prose\nPðAÞ ¼ 1",
                    confidence=0.9,
                    needs_ocr_enhancement=True,
                    has_equations=True,
                    has_corrupt_math=True,
                    word_count=80,
                ),
                PageAssessment(
                    page_num=2,
                    is_born_digital=False,
                    native_text="",
                    confidence=0.0,
                    needs_ocr_enhancement=True,
                    word_count=0,
                ),
            ],
        )

        gemini_calls: list[int] = []

        class _StubEngine:
            def __init__(self, name: str, text: str) -> None:
                self.name = name
                self._text = text

            def is_available(self) -> bool:
                return True

            def process_pages(self, pdf_path, page_nums, config, dpi=200, subprocess_timeout=None):
                if self.name == "gemini":
                    gemini_calls.extend(page_nums)
                return [
                    PageOutput(
                        page_num=p,
                        text=self._text,
                        status=PageStatus.SUCCESS,
                        engine=self.name,
                        audit_passed=True,
                    )
                    for p in page_nums
                ]

        class _RejectLocalAcceptGemini:
            def assess(self, output: PageOutput, provider):
                if provider.engine is EngineType.GEMINI:
                    return AcceptDecision(accept=True, reason="accepted paid")
                return AcceptDecision(accept=False, reason="reject local")

        cfg = PipelineConfig(
            agentic=True,
            native_first=True,
            recover_corrupt_math=True,
            table_judge_ladder=False,
            check_text_coverage=False,
            describe_figure_crops=False,
            save_figures=False,
            write_manifest=False,
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN, EngineType.GEMINI],
            judge_backend="heuristic",
            cost_budget=0.05,
            quiet=True,
        )
        pipe = UnifiedPipeline(cfg)
        pipe.bd_detector = MagicMock()
        pipe.bd_detector.detect.return_value = assessment
        pipe._available_engines_for_agentic = MagicMock(
            return_value=[PROFILE_QWEN_LOCAL, PROFILE_GEMINI]
        )
        pipe._build_page_judge = MagicMock(return_value=_RejectLocalAcceptGemini())

        engines = {
            EngineType.QWEN: _StubEngine("qwen", "local draft"),
            EngineType.GEMINI: _StubEngine("gemini", "Clean paid transcription of page two."),
        }

        with (
            patch("socr.pipeline.orchestrator.get_engine", side_effect=lambda et: engines[et]),
            patch(
                "socr.math.recover.recover_math_regions",
                return_value=[
                    CorruptMathRegion(
                        rect=object(),
                        source_text="PðAÞ ¼ 1",
                        crop_path="equations/corrupt_math_p00001_r001.png",
                        raw_latex=r"P(A) = 1",
                        validation_ok=True,
                        validation_reason="ok",
                        model_id=cfg.math_model,
                        attempts=1,
                    )
                ],
            ),
            patch.object(pipe, "_resolve_crop_vlm_model", return_value=None),
            patch.object(pipe, "_resolve_judge_model", return_value=""),
        ):
            result = pipe.process(pdf, tmp_path / "out")

        assert gemini_calls == [2], "local corrupt-math on p1 must not poison budget for p2"
        assert result.cost == pytest.approx(PROFILE_GEMINI.cost_per_page_usd)

    def test_cloud_corrupt_math_model_still_records_unknown_none(self, tmp_path: Path) -> None:
        """Unpriced :cloud math_model keeps Round 6 fail-closed cost=None."""
        pdf = _pdf_with_pages(tmp_path, ["PðAÞ ¼ 1"], name="cloud_math.pdf")
        cfg = PipelineConfig(
            agentic=True,
            recover_corrupt_math=True,
            math_model="custom-math:cloud",
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
        )
        pipe = UnifiedPipeline(cfg)
        state = DocumentState(DocumentHandle(pdf))
        state.pages[1].native_text = "PðAÞ ¼ 1"
        state.pages[1].is_born_digital = True
        state.pages[1].has_corrupt_math = True

        with patch("socr.math.recover.recover_math_regions", return_value=[]):
            out = pipe._recover_corrupt_math_page(state, 1, tmp_path / "out")

        assert out.cost_usd is None
        assert state.total_cost is None


# ---------------------------------------------------------------------------
# 3. Blind-cell guard, cell transcriber budget gates, and adjudication early stop
# ---------------------------------------------------------------------------


class TestCellGuardAndTranscriberBudgets:
    def _state(self, tmp_path: Path) -> DocumentState:
        pdf = _pdf_with_pages(tmp_path, ["body"])
        return DocumentState(DocumentHandle(pdf))

    @pytest.mark.parametrize("prior_cost", [None, 0.05], ids=["unknown_total", "budget_exhausted"])
    def test_zero_priced_adjudicator_refused_when_budget_unknown_or_exhausted(
        self, tmp_path: Path, prior_cost: float | None
    ) -> None:
        state = self._state(tmp_path)
        state.record_engine_run(
            EngineResult(document_path=state.handle.path, engine="gemini", cost=prior_cost),
            page_nums=[1],
        )
        cfg = PipelineConfig(
            cost_budget=0.05,
            table_judge_adjudicator_cost_per_call_usd=0.0,
        )
        called: list[int] = []

        def _adj(crop, refs):
            called.append(1)
            return BlindCellResult(rung="adjudicator:kimi", ok=True, tokens={"R1C2": "1.5"})

        decision = evaluate_cell_guard(
            state=state,
            page_num=1,
            crop_path=None,
            extraction_tokens={"R1C2": "1.5"},
            requested_refs=["R1C2"],
            geometry_evidence=BindingEvidence.ABSTAIN,
            adjudicator=_adj,
            config=cfg,
        )
        assert called == []
        assert decision.adjudicator_ran is False
        assert decision.disposition is GuardDisposition.NOT_CLEARED

    def test_page_ladder_budget_skip_does_not_bill_adjudicator(self, tmp_path: Path) -> None:
        state = self._state(tmp_path)
        cfg = PipelineConfig(table_judge_adjudicator_cost_per_call_usd=0.02)
        clock = iter([0.0, 100.0, 100.0]).__next__
        budget = PageLadderBudget(1.0, clock=clock)

        def _adj(crop, refs):
            raise AssertionError("wrapped adjudicator should have been skipped")

        _adj.rung_id = "adjudicator:kimi"
        wrapped = budget.wrap_adjudicator(_adj)

        decision = evaluate_cell_guard(
            state=state,
            page_num=1,
            crop_path=None,
            extraction_tokens={"R1C2": "1.5"},
            requested_refs=["R1C2"],
            geometry_evidence=BindingEvidence.ABSTAIN,
            adjudicator=wrapped,
            config=cfg,
        )
        assert decision.adjudicator_ran is False
        assert decision.disposition is GuardDisposition.NOT_CLEARED
        assert state.engine_runs == ()
        assert state.total_cost == 0.0

    @pytest.mark.parametrize("prior_cost", [None, 0.05], ids=["unknown_total", "budget_exhausted"])
    def test_cell_transcribe_budget_helper_blocks_when_budget_unavailable(
        self, tmp_path: Path, prior_cost: float | None
    ) -> None:
        state = self._state(tmp_path)
        state.record_engine_run(
            EngineResult(document_path=state.handle.path, engine="gemini", cost=prior_cost),
            page_nums=[1],
        )
        pipe = UnifiedPipeline(PipelineConfig(cost_budget=0.05))
        assert pipe._cell_transcribe_budget_allows(state) is False

        open_state = self._state(tmp_path)
        open_state.record_engine_run(
            EngineResult(document_path=open_state.handle.path, engine="gemini", cost=0.01),
            page_nums=[1],
        )
        assert pipe._cell_transcribe_budget_allows(open_state) is True

    def test_adjudicate_skips_doomed_cloud_transcriptions(self) -> None:
        item1 = ContradictionItem(
            kind="row_label",
            row_path=("RowA",),
            col_path=(),
            native_token="RowA",
            model_token="RowB",
            cell_bbox=(0.0, 0.0, 10.0, 10.0),
        )
        item2 = ContradictionItem(
            kind="row_label",
            row_path=("RowC",),
            col_path=(),
            native_token="RowC",
            model_token="RowD",
            cell_bbox=(0.0, 10.0, 10.0, 20.0),
        )
        item_unaddressed = ContradictionItem(
            kind="row_label",
            row_path=("RowE",),
            col_path=(),
            native_token="RowE",
            model_token="RowF",
            cell_bbox=None,
        )

        # Case A: unaddressable non-garbage item makes lifting impossible -> 0 cloud calls.
        calls_a: list = []
        rec_a = adjudicate(
            (item1, item_unaddressed),
            markdown="md",
            transcribe=lambda bbox: calls_a.append(bbox) or "RowB",
        )
        assert rec_a.status == "held"
        assert calls_a == []
        assert len(rec_a.items) == 2

        # Case B: first item fails to disprove -> stop before paying for item 2.
        calls_b: list = []

        def _fail_first(bbox):
            calls_b.append(bbox)
            return "RowA"  # matches native, does not disprove

        rec_b = adjudicate((item1, item2), markdown="md", transcribe=_fail_first)
        assert rec_b.status == "held"
        assert len(calls_b) == 1
        assert len(rec_b.items) == 2


# ---------------------------------------------------------------------------
# 4. Gemma timeout default + clean TimeoutExpired handling
# ---------------------------------------------------------------------------


class TestGemmaTimeoutGuard:
    def test_gemma_present_in_default_provider_timeouts(self) -> None:
        assert EngineType.GEMMA in DEFAULT_PROVIDER_TIMEOUTS
        assert 0 < DEFAULT_PROVIDER_TIMEOUTS[EngineType.GEMMA] <= 300.0

    def test_gemma_process_pages_catches_subprocess_timeout(self, tmp_path: Path) -> None:
        pdf = _pdf_with_pages(tmp_path, ["hello"])
        engine = GemmaCloudEngine()
        cfg = PipelineConfig(enable_gemma_cloud=True)
        crop = tmp_path / "crop.png"
        crop.write_bytes(b"png")

        with (
            patch(
                "socr.engines.gemma_cloud.render_gemma_crops",
                return_value=([("p1_top", crop)], False),
            ),
            patch(
                "socr.engines.gemma_cloud.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["qwen-ocr"], timeout=30.0),
            ),
        ):
            outputs = engine.process_pages(pdf, [1], cfg, subprocess_timeout=60.0)

        assert len(outputs) == 1
        assert outputs[0].status is PageStatus.ERROR
        assert outputs[0].failure_mode is FailureMode.TIMEOUT
        assert "timed out" in (outputs[0].error or "").lower()


# ---------------------------------------------------------------------------
# 5. Free native geometry (BindingEvidence.PASS) skips paid cloud table readers
# ---------------------------------------------------------------------------


class TestBindingPassSkipsCloudTableReaders:
    def _gate_state(
        self, tmp_path: Path, name: str = "doc.pdf"
    ) -> tuple[UnifiedPipeline, DocumentState]:
        pdf = _ruled_table_pdf(tmp_path, name=name)
        cfg = PipelineConfig(
            agentic=True,
            table_judge_ladder=True,
            judge_backend="heuristic",
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
        )
        pipe = UnifiedPipeline(cfg)
        pipe._resolve_judge_model = lambda: ""  # type: ignore[method-assign]
        state = DocumentState(DocumentHandle(pdf))
        state.pages[1].is_born_digital = True
        state.pages[1].has_tables = True
        return pipe, state

    def test_binding_pass_skips_cloud_rungs_while_abstain_calls_them(self, tmp_path: Path) -> None:
        """Paired difference: same PDF, same candidate, same rung — PASS skips
        the cloud rung completely; ABSTAIN invokes it."""
        calls_pass: list[str] = []
        calls_abstain: list[str] = []

        def _make_rung(log: list[str]):
            def _rung(crop_path, markdown, prior_findings):
                log.append(markdown)
                return RungResult(
                    rung="ollama:glm",
                    ok=True,
                    verdict=TableJudgeVerdict(verdict="PASS", confidence="high", findings=[]),
                )

            _rung.rung_id = "ollama:glm"
            return _rung

        pipe_pass, state_pass = self._gate_state(tmp_path, "pass.pdf")
        bo_pass = PageOutput(
            page_num=1,
            text=_TABLE_MD,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        pipe_pass._run_table_judge_gate(
            state_pass, 1, state_pass.pages[1], bo_pass, [_make_rung(calls_pass)]
        )

        pipe_abstain, state_abstain = self._gate_state(tmp_path, "abstain.pdf")
        bo_abstain = PageOutput(
            page_num=1,
            text=_TABLE_MD,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        with patch.object(
            pipe_abstain,
            "_binding_evidence_for_witness",
            return_value=(None, BindingEvidence.ABSTAIN),
        ):
            pipe_abstain._run_table_judge_gate(
                state_abstain, 1, state_abstain.pages[1], bo_abstain, [_make_rung(calls_abstain)]
            )

        assert calls_pass == [], "BindingEvidence.PASS must skip paid cloud table rungs"
        assert len(calls_abstain) == 1, "control: BindingEvidence.ABSTAIN still invokes rungs"
        assert state_pass.pages[1].table_ladder_disposition is None
        accepted = [e for e in state_pass.events if e.kind == "table_ladder_accepted"]
        assert len(accepted) == 1
        assert accepted[0].data["reason"] == "verified_by_geometry"
        assert accepted[0].data["rung_trail"] == []
        assert "readers skipped" in accepted[0].detail

    def test_binding_pass_still_runs_rung_on_page_judge_timeout_for_credential(
        self, tmp_path: Path
    ) -> None:
        """#713 timeout credential requires an executed visual rung, so
        JUDGE_OUTCOME_TIMEOUT does not skip the ladder on BindingEvidence.PASS."""
        calls: list[str] = []

        def _rung(crop_path, markdown, prior_findings):
            calls.append(markdown)
            return RungResult(
                rung="ollama:glm",
                ok=True,
                verdict=TableJudgeVerdict(verdict="PASS", confidence="high", findings=[]),
            )

        _rung.rung_id = "ollama:glm"
        _rung.executing = "ollama:glm"

        pipe, state = self._gate_state(tmp_path, "timeout.pdf")
        bo = PageOutput(
            page_num=1,
            text=_TABLE_MD,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
            judge_outcome=JUDGE_OUTCOME_TIMEOUT,
        )
        pipe._run_table_judge_gate(state, 1, state.pages[1], bo, [_rung])

        assert len(calls) == 1
        assert bo.table_acceptance_credential is not None
