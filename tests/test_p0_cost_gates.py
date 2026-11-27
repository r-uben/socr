"""Hermetic regression guards for P0 cost gates, strict-local policy, and Gemma timeouts.

Covers four cost-governance defects surfaced by the 2026-11-27 audit:
1. ``--strict-local`` and explicit ``--max-cost-per-page 0`` must never fire a
   live Gemma Cloud generation probe or auto-select a cloud engine (including
   cloud-pinned ``--qwen-model x:cloud`` and unknown engines).
2. Local corrupt-math recovery (default ``qwen3-vl:30b-a3b-instruct``) must
   record a known ``0.0`` cost rather than ``None`` so it does not poison
   ``DocumentState.total_cost`` and lock ``--cost-budget`` to $0.00 for later
   pages; an unpriced ``:cloud`` math model still records ``None`` (fail-closed).
3. Blind-cell adjudication (``kimi-k2.6:cloud``, default $0) and cloud cell
   transcription (``glm-5.3-flash:cloud``) must refuse when ``--cost-budget``
   is exhausted or prior spend is unknown (verified at both real ``process()``
   callers: ``_adjudicate_clamped_table`` and ``_repair_native_table_cells``);
   budget-skipped native cell repairs must record ``cell_repair_budget_skipped``
   without escalating to whole-page OCR; ``PageLadderBudget`` skips must not
   bill ``_record_call``; ``adjudicate`` must skip doomed cloud transcriptions
   and label skipped items ``"skipped"``.
4. ``EngineType.GEMMA`` has a bounded entry in ``DEFAULT_PROVIDER_TIMEOUTS``,
   ``GemmaCloudEngine.process_pages`` classifies ``subprocess.TimeoutExpired``
   via a typed timeout error (not substring matching), and cloud-rung timeouts
   never trigger the local-backend ``PARTIAL_SAVE_VLM_TIMEOUT`` cascade halt.

Hermetic: no live Ollama, no network, no external CLI.
"""

from __future__ import annotations

import json
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
    PROFILE_GEMMA_CLOUD,
    PROFILE_QWEN_LOCAL,
)
from socr.core.result import (
    EngineResult,
    FailureMode,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState
from socr.engines.gemma_cloud import GemmaCloudEngine
from socr.engines.registry import resolve_auto_engine_for_config
from socr.judge.ladder_budget import (
    TABLE_LADDER_BUDGET_EXHAUSTED_KIND,
    PageLadderBudget,
)
from socr.judge.table_cell_guard import GuardDisposition, evaluate_cell_guard
from socr.judge.table_rung_ollama import BlindCellResult
from socr.judge.table_verdict import RungResult, TableJudgeVerdict
from socr.math.recover import CorruptMathRegion
from socr.pipeline.agentic import (
    DEFAULT_PROVIDER_TIMEOUTS,
    REASON_PROVIDER_TIMEOUT,
    AcceptDecision,
    PageDecision,
    ProviderAttempt,
)
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.adjudication import ContradictionItem, adjudicate
from socr.tables.binding import BindingEvidence
from socr.tables.native_first import (
    CELL_BUDGET_SKIPPED_REASON,
    CELL_MISMATCH_REASON,
    CELLS,
    FailingCell,
    NativeTableFirstWork,
    NativeTablePlan,
)


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


def _two_page_spend_then_row_shift_pdf(tmp_path: Path, name: str = "shift2.pdf") -> Path:
    """Page 1 is a scanned page (to spend budget); page 2 is a ruled born-digital table."""
    pdf = tmp_path / name
    doc = fitz.open()
    p1 = doc.new_page()
    p1.draw_rect(fitz.Rect(72, 72, 160, 100), fill=(0, 0, 0))

    p2 = doc.new_page()
    cols = [100, 250, 380]
    rows = [100, 122, 144]
    p2.insert_text((cols[1] + 4, rows[0] + 12), "OLS", fontsize=9)
    p2.insert_text((cols[2] + 4, rows[0] + 12), "IV", fontsize=9)
    p2.insert_text((cols[0] + 4, rows[1] + 12), "RowA", fontsize=9)
    p2.insert_text((cols[1] + 4, rows[1] + 12), "100", fontsize=9)
    p2.insert_text((cols[2] + 4, rows[1] + 12), "200", fontsize=9)
    p2.insert_text((cols[0] + 4, rows[2] + 12), "RowB", fontsize=9)
    p2.insert_text((cols[1] + 4, rows[2] + 12), "300", fontsize=9)
    p2.insert_text((cols[2] + 4, rows[2] + 12), "400", fontsize=9)
    for yy in [*rows, rows[-1] + 22]:
        p2.draw_line((100, yy), (460, yy))
    for xx in [*cols, 460]:
        p2.draw_line((xx, rows[0]), (xx, rows[-1] + 22))
    doc.save(pdf)
    doc.close()
    return pdf


_SHIFTED_MD = (
    "|      | OLS | IV  |\n| ---- | --- | --- |\n| RowB | 100 | 200 |\n| RowA | 300 | 400 |\n"
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
        """Paired difference: same engine availability, only policy flips."""

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

    @pytest.mark.parametrize(
        "policy_kwargs",
        [
            {"strict_local": True},
            {"max_cost_per_page": 0.0, "max_cost_per_page_pinned": True},
        ],
        ids=["strict_local", "pinned_zero_cap"],
    )
    def test_resolve_auto_engine_refuses_cloud_pinned_qwen_and_unknown_profiles(
        self, policy_kwargs: dict, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A cloud-pinned ``--qwen-model x:cloud`` must never be auto-selected or used
        as fallback under strict-local / pinned zero cap, and an engine missing from
        ``DEFAULT_PROVIDERS`` fails closed."""
        cfg = PipelineConfig(
            qwen_model="qwen3-vl:30b-cloud",
            qwen_model_pinned=True,
            enable_gemma_cloud=True,
            **policy_kwargs,
        )

        # Case 1: QWEN (cloud-pinned) and GLM are available -> must skip QWEN and pick GLM.
        with (
            patch("socr.engines.qwen.QwenEngine.is_available", return_value=True),
            patch("socr.engines.marker.MarkerEngine.is_available", return_value=False),
            patch("socr.engines.glm.GLMEngine.is_available", return_value=True),
            patch("socr.engines.gemma_cloud.GemmaCloudEngine.is_available", return_value=True),
            patch("socr.engines.gemini.GeminiEngine.is_available", return_value=True),
        ):
            picked = resolve_auto_engine_for_config(cfg)
        assert picked is EngineType.GLM

        # Case 2: No local engine available (only cloud-pinned QWEN) -> fallback must NOT
        # be QWEN, and warning names both strict-local and pinned zero cost cap.
        caplog.clear()
        with (
            caplog.at_level("WARNING", logger="socr.engines.registry"),
            patch("socr.engines.qwen.QwenEngine.is_available", return_value=True),
            patch("socr.engines.marker.MarkerEngine.is_available", return_value=False),
            patch("socr.engines.glm.GLMEngine.is_available", return_value=False),
            patch("socr.engines.gemma_cloud.GemmaCloudEngine.is_available", return_value=True),
            patch("socr.engines.gemini.GeminiEngine.is_available", return_value=True),
        ):
            fallback = resolve_auto_engine_for_config(cfg)
        assert fallback is EngineType.MARKER
        assert any(
            "strict-local" in rec.message and "zero cost cap" in rec.message
            for rec in caplog.records
        )

        # Case 3: Engine with prof=None in DEFAULT_PROVIDERS is refused (fail-closed).
        with (
            patch("socr.engines.registry.AUTO_ENGINE_ORDER", [EngineType.AUTO, EngineType.GLM]),
            patch.dict(
                "socr.engines.registry._ENGINES", {EngineType.AUTO: MagicMock()}, clear=False
            ),
            patch("socr.engines.glm.GLMEngine.is_available", return_value=True),
        ):
            picked_unknown = resolve_auto_engine_for_config(PipelineConfig(**policy_kwargs))
        assert picked_unknown is EngineType.GLM


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
# 3. Blind-cell guard, caller-level cell transcriber gates, and adjudication
# ---------------------------------------------------------------------------


class TestCellGuardAndTranscriberBudgets:
    def _state(self, tmp_path: Path) -> DocumentState:
        pdf = _pdf_with_pages(tmp_path, ["body"])
        return DocumentState(DocumentHandle(pdf))

    @pytest.mark.parametrize(
        ("prior_cost", "expected_substr"),
        [
            (None, "document spend is unmetered"),
            (0.05, "document budget is exhausted"),
        ],
        ids=["unknown_total", "budget_exhausted"],
    )
    def test_zero_priced_adjudicator_refused_when_budget_unknown_or_exhausted(
        self, tmp_path: Path, prior_cost: float | None, expected_substr: str
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
        assert expected_substr in (decision.detail or "")
        assert "0.0 exceeds" not in (decision.detail or "")

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

    def test_process_clamped_adjudication_skips_transcriber_when_cost_budget_exhausted(
        self, tmp_path: Path
    ) -> None:
        """Real ``process()`` caller test for ``_adjudicate_clamped_table._transcribe``.

        Page 1 spends $0.05 on Gemini; page 2 has a row-shifted born-digital table
        whose clamped adjudication would transcribe cell crops. Paired difference:
        with ``cost_budget=0.05`` (exhausted), 0 transcriber calls run; with
        ``cost_budget=1.00`` (open), the transcriber runs.
        """
        pdf = _two_page_spend_then_row_shift_pdf(tmp_path)

        def _run_arm(tag: str, cost_budget: float) -> list[Path]:
            cfg = PipelineConfig(
                agentic=True,
                native_first=True,
                table_judge_ladder=True,
                check_text_coverage=False,
                describe_figure_crops=False,
                save_figures=False,
                write_manifest=False,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN, EngineType.GEMINI],
                judge_backend="heuristic",
                cost_budget=cost_budget,
                quiet=True,
            )
            pipe = UnifiedPipeline(cfg)
            transcribe_calls: list[Path] = []

            def _fake_route(page_num, ladder, run_provider, judge, **kwargs):
                if page_num == 1:
                    out = PageOutput(
                        page_num=1,
                        text="Paid page one text.",
                        status=PageStatus.SUCCESS,
                        engine="gemini",
                        audit_passed=True,
                        cost_usd=0.05,
                    )
                    att = ProviderAttempt(
                        engine=EngineType.GEMINI,
                        output=out,
                        cost_usd=0.05,
                        accepted=True,
                        reason="ok",
                        provider_id=PROFILE_GEMINI.id,
                        model=PROFILE_GEMINI.model,
                        backend=PROFILE_GEMINI.backend,
                    )
                    return PageDecision(page_num=1, final_output=out, attempts=[att], accepted=True)
                out = PageOutput(
                    page_num=2,
                    text=_SHIFTED_MD,
                    status=PageStatus.SUCCESS,
                    engine="qwen",
                    audit_passed=True,
                    cost_usd=0.0,
                )
                att = ProviderAttempt(
                    engine=EngineType.QWEN,
                    output=out,
                    cost_usd=0.0,
                    accepted=True,
                    reason="ok",
                    provider_id=PROFILE_QWEN_LOCAL.id,
                    model=PROFILE_QWEN_LOCAL.model,
                    backend=PROFILE_QWEN_LOCAL.backend,
                )
                return PageDecision(page_num=2, final_output=out, attempts=[att], accepted=True)

            def _accept_rung(crop_path, markdown, prior_findings):
                return RungResult(
                    rung="fake1",
                    ok=True,
                    verdict=TableJudgeVerdict(verdict="PASS", confidence="high", findings=[]),
                )

            _accept_rung.rung_id = "fake1"

            def _transcribe(crop: Path) -> str:
                transcribe_calls.append(crop)
                return "RowB"

            with (
                patch("socr.pipeline.orchestrator.route_page", side_effect=_fake_route),
                patch.object(
                    pipe,
                    "_available_engines_for_agentic",
                    return_value=[PROFILE_QWEN_LOCAL, PROFILE_GEMINI],
                ),
                patch.object(pipe, "_resolve_judge_model", return_value=""),
                patch.object(pipe, "_resolve_crop_vlm_model", return_value=None),
                patch.object(pipe, "_plan_native_table_first", return_value=None),
                patch.object(pipe, "_withhold_contradicted_unverified_tables", return_value=None),
                patch.object(pipe, "_build_table_judge_rungs", return_value=[_accept_rung]),
                patch.object(pipe, "_transcribe_cell_token", side_effect=_transcribe),
            ):
                pipe.process(pdf, tmp_path / f"out_{tag}")
            return transcribe_calls

        exhausted_calls = _run_arm("exhausted", cost_budget=0.05)
        open_calls = _run_arm("open", cost_budget=1.00)
        assert exhausted_calls == [], "exhausted --cost-budget must skip clamped cell transcription"
        assert len(open_calls) >= 1, "control: open --cost-budget still transcribes clamped cells"

    def test_process_native_cell_repair_budget_skip_blocks_transcriber_and_ocr_escalation(
        self, tmp_path: Path
    ) -> None:
        """Real ``process()`` caller test for ``_repair_native_table_cells`` cost-budget
        AND wall-clock ``PageLadderBudget`` gates.

        When either budget skips the cell repair on page 2:
        - ``_transcribe_cell_token`` is NOT called;
        - page 2 does NOT escalate to whole-page OCR (``route_page`` not called for p2);
        - ``audit_log.json`` records ``native_table_cell_unresolved`` with
          ``reason == "cell_repair_budget_skipped"`` (not ``"cell_mismatch"``).
        """
        pdf = _two_page_spend_then_row_shift_pdf(tmp_path, name="repair2.pdf")
        cell = FailingCell(
            row_line="| RowA | 109 | 200 |",
            cell_index=1,
            grid_token="109",
            native_token="100",
            bbox=(250.0, 122.0, 380.0, 144.0),
        )
        cells_work = NativeTableFirstWork(
            plan=NativeTablePlan(CELLS, cells=(cell,), reason=CELL_MISMATCH_REASON)
        )

        def _run_repair_arm(
            tag: str,
            *,
            cost_budget: float,
            wall_clock_exhausted: bool = False,
        ) -> tuple[list[Path], list[int], list[dict]]:
            cfg = PipelineConfig(
                agentic=True,
                native_first=True,
                table_judge_ladder=False,
                check_text_coverage=False,
                describe_figure_crops=False,
                save_figures=False,
                write_manifest=False,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN, EngineType.GEMINI],
                judge_backend="heuristic",
                cost_budget=cost_budget,
                table_judge_page_budget_sec=1.0 if wall_clock_exhausted else 0.0,
                quiet=True,
            )
            pipe = UnifiedPipeline(cfg)
            transcribe_calls: list[Path] = []
            routed_pages: list[int] = []

            def _fake_route(page_num, ladder, run_provider, judge, **kwargs):
                routed_pages.append(page_num)
                out = PageOutput(
                    page_num=page_num,
                    text="Paid page text.",
                    status=PageStatus.SUCCESS,
                    engine="gemini",
                    audit_passed=True,
                    cost_usd=0.05,
                )
                att = ProviderAttempt(
                    engine=EngineType.GEMINI,
                    output=out,
                    cost_usd=0.05,
                    accepted=True,
                    reason="ok",
                    provider_id=PROFILE_GEMINI.id,
                    model=PROFILE_GEMINI.model,
                    backend=PROFILE_GEMINI.backend,
                )
                return PageDecision(
                    page_num=page_num, final_output=out, attempts=[att], accepted=True
                )

            def _plan(state, page_num, ps):
                return cells_work if page_num == 2 else None

            def _transcribe(crop: Path) -> str:
                transcribe_calls.append(crop)
                return "999"  # does not match native, triggers normal mismatch control

            out_dir = tmp_path / f"repair_{tag}"
            clock_vals = iter([0.0, 100.0, 100.0, 100.0, 100.0]).__next__
            budget_patch = (
                patch(
                    "socr.judge.ladder_budget.PageLadderBudget",
                    side_effect=lambda limit, **kw: PageLadderBudget(limit, clock=clock_vals, **kw),
                )
                if wall_clock_exhausted
                else patch.object(pipe, "_page_ladder_budget_sec", return_value=0.0)
            )
            with (
                patch("socr.pipeline.orchestrator.route_page", side_effect=_fake_route),
                patch.object(
                    pipe,
                    "_available_engines_for_agentic",
                    return_value=[PROFILE_QWEN_LOCAL, PROFILE_GEMINI],
                ),
                patch.object(pipe, "_resolve_judge_model", return_value=""),
                patch.object(pipe, "_resolve_crop_vlm_model", return_value=None),
                patch.object(pipe, "_plan_native_table_first", side_effect=_plan),
                patch.object(pipe, "_transcribe_cell_token", side_effect=_transcribe),
                budget_patch,
            ):
                pipe.process(pdf, out_dir)
            audit_path = out_dir / "repair2" / "audit_log.json"
            events = json.loads(audit_path.read_text())["events"] if audit_path.exists() else []
            return transcribe_calls, routed_pages, events

        # Arm 1: --cost-budget exhausted after p1 -> no transcriber, no p2 OCR escalation,
        # recorded with CELL_BUDGET_SKIPPED_REASON.
        t_cost, r_cost, ev_cost = _run_repair_arm("cost_exhausted", cost_budget=0.05)
        assert t_cost == []
        assert r_cost == [1], f"budget-skipped native repair must not escalate p2 to OCR: {r_cost}"
        unresolved_cost = [
            e for e in ev_cost if e["kind"] == "native_table_cell_unresolved" and e["page_num"] == 2
        ]
        assert len(unresolved_cost) == 1
        assert unresolved_cost[0]["data"]["reason"] == CELL_BUDGET_SKIPPED_REASON

        # Arm 2: PageLadderBudget wall-clock exhausted -> no transcriber, no p2 OCR escalation,
        # records both table_ladder_budget_exhausted and CELL_BUDGET_SKIPPED_REASON.
        t_wall, r_wall, ev_wall = _run_repair_arm(
            "wall_exhausted", cost_budget=1.00, wall_clock_exhausted=True
        )
        assert t_wall == []
        assert r_wall == [1], f"wall-clock-skipped repair must not escalate p2 to OCR: {r_wall}"
        unresolved_wall = [
            e for e in ev_wall if e["kind"] == "native_table_cell_unresolved" and e["page_num"] == 2
        ]
        assert len(unresolved_wall) == 1
        assert unresolved_wall[0]["data"]["reason"] == CELL_BUDGET_SKIPPED_REASON
        assert any(
            e["kind"] == TABLE_LADDER_BUDGET_EXHAUSTED_KIND and e["page_num"] == 2 for e in ev_wall
        )

        # Control: budget open -> transcriber runs and mismatch still escalates p2 normally.
        t_open, r_open, _ev_open = _run_repair_arm("open", cost_budget=1.00)
        assert len(t_open) == 1
        assert r_open == [1, 2]

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

        # Case A: unaddressable non-garbage item makes lifting impossible -> 0 cloud calls;
        # skipped addressable item is recorded as "skipped" (not "held").
        calls_a: list = []
        rec_a = adjudicate(
            (item1, item_unaddressed),
            markdown="md",
            transcribe=lambda bbox: calls_a.append(bbox) or "RowB",
        )
        assert rec_a.status == "held"
        assert calls_a == []
        assert [o.outcome for o in rec_a.items] == ["skipped", "abstained"]
        assert [d["outcome"] for d in rec_a.to_dict()["items"]] == ["skipped", "abstained"]

        # Case B: first item fails to disprove -> stop before paying for item 2, and
        # distinguish held (attempted) from skipped (never attempted).
        calls_b: list = []

        def _fail_first(bbox):
            calls_b.append(bbox)
            return "RowA"  # matches native, does not disprove

        rec_b = adjudicate((item1, item2), markdown="md", transcribe=_fail_first)
        assert rec_b.status == "held"
        assert len(calls_b) == 1
        assert [o.outcome for o in rec_b.items] == ["held", "skipped"]
        assert [d["outcome"] for d in rec_b.to_dict()["items"]] == ["held", "skipped"]


# ---------------------------------------------------------------------------
# 4. Gemma timeout default, typed TimeoutExpired handling, and no local halt
# ---------------------------------------------------------------------------


class TestGemmaTimeoutGuard:
    def test_gemma_present_in_default_provider_timeouts(self) -> None:
        assert EngineType.GEMMA in DEFAULT_PROVIDER_TIMEOUTS
        assert 0 < DEFAULT_PROVIDER_TIMEOUTS[EngineType.GEMMA] <= 300.0

    def test_gemma_process_pages_classifies_timeout_by_type_not_stderr_substring(
        self, tmp_path: Path
    ) -> None:
        pdf = _pdf_with_pages(tmp_path, ["hello"])
        engine = GemmaCloudEngine()
        cfg = PipelineConfig(enable_gemma_cloud=True)
        crop = tmp_path / "crop.png"
        crop.write_bytes(b"png")

        # Arm 1: actual subprocess.TimeoutExpired -> FailureMode.TIMEOUT.
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
            timed_out = engine.process_pages(pdf, [1], cfg, subprocess_timeout=60.0)

        assert len(timed_out) == 1
        assert timed_out[0].status is PageStatus.ERROR
        assert timed_out[0].failure_mode is FailureMode.TIMEOUT

        # Arm 2: non-zero CLI exit whose stderr merely contains "timed out" -> CLI_ERROR.
        completed = subprocess.CompletedProcess(
            args=["qwen-ocr"],
            returncode=1,
            stdout="",
            stderr="upstream proxy timed out while forwarding request",
        )
        with (
            patch(
                "socr.engines.gemma_cloud.render_gemma_crops",
                return_value=([("p1_top", crop)], False),
            ),
            patch("socr.engines.gemma_cloud.subprocess.run", return_value=completed),
        ):
            cli_err = engine.process_pages(pdf, [1], cfg, subprocess_timeout=60.0)

        assert len(cli_err) == 1
        assert cli_err[0].status is PageStatus.ERROR
        assert cli_err[0].failure_mode is FailureMode.CLI_ERROR

    def test_gemma_cloud_timeout_does_not_trigger_local_vlm_halt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Paired ``process()`` test: a Gemma Cloud provider timeout with an unhealthy
        local Ollama probe must NOT halt the document as ``PARTIAL_SAVE_VLM_TIMEOUT``;
        a local Qwen provider timeout under the same probe still halts."""
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        pdf = _pdf_with_pages(tmp_path, ["", ""], name="two_scanned.pdf")

        def _run_halt_arm(tag: str, profile) -> tuple[list[int], bool, int]:
            cfg = PipelineConfig(
                agentic=True,
                native_first=True,
                table_judge_ladder=False,
                check_text_coverage=False,
                describe_figure_crops=False,
                save_figures=False,
                write_manifest=False,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[profile.engine],
                enable_gemma_cloud=True,
                judge_backend="heuristic",
                quiet=True,
            )
            pipe = UnifiedPipeline(cfg)
            routed: list[int] = []
            probes: list[int] = []

            def _fake_route(page_num, ladder, run_provider, judge, **kwargs):
                routed.append(page_num)
                out = PageOutput(
                    page_num=page_num,
                    text="",
                    status=PageStatus.ERROR,
                    engine=profile.engine.value,
                    failure_mode=FailureMode.TIMEOUT,
                    audit_passed=False,
                )
                att = ProviderAttempt(
                    engine=profile.engine,
                    output=out,
                    cost_usd=0.0,
                    accepted=False,
                    reason=REASON_PROVIDER_TIMEOUT,
                    provider_id=profile.id,
                    model=profile.model,
                    backend=profile.backend,
                )
                return PageDecision(
                    page_num=page_num, final_output=out, attempts=[att], accepted=False
                )

            def _busy_probe(*a, **k):
                probes.append(1)
                return False

            with (
                patch.object(pipe, "_available_engines_for_agentic", return_value=[profile]),
                patch.object(pipe, "_resolve_judge_model", return_value=""),
                patch.object(pipe, "_resolve_crop_vlm_model", return_value=None),
                patch("socr.pipeline.orchestrator.route_page", side_effect=_fake_route),
                patch("socr.pipeline.orchestrator.probe_ollama_idle", side_effect=_busy_probe),
            ):
                res = pipe.process(pdf, tmp_path / f"halt_{tag}")
            halted = "PARTIAL_SAVE_VLM_TIMEOUT" in (res.error or "")
            return routed, halted, len(probes)

        cloud_routed, cloud_halted, cloud_probes = _run_halt_arm("gemma", PROFILE_GEMMA_CLOUD)
        local_routed, local_halted, local_probes = _run_halt_arm("qwen", PROFILE_QWEN_LOCAL)

        assert cloud_routed == [1, 2], "cloud timeout must not halt p2"
        assert cloud_halted is False
        assert cloud_probes == 0
        assert local_routed == [1], "control: local VLM timeout still halts after p1"
        assert local_halted is True
        assert local_probes >= 1
