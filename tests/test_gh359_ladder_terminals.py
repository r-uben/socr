"""GH-359: enforce the seven pinned ladder terminals at process().

The spec is ``docs/log/2026-08-31_gh359-ladder-terminals.md``. Helper-unit
coverage of the same transitions lives in ``test_table_ladder.py`` and
``test_ladder_binding_evidence.py``. This file pins DIFFERENCE at
``UnifiedPipeline.process()``: agentic produces content terminals, assemble
backfills missing ones as UNVERIFIED. A green helper suite is not a gate.

Hermetic: CI has no ollama and no ``gemini`` binary. Rungs are injected.
``_available_engines_for_agentic`` is patched; ``_resolve_judge_model`` is
patched to ``""``. Never pin an absolute outcome measured on one machine
(#253/#257): every assertion is a same-process difference.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import fitz

from socr.core.config import EngineType, PipelineConfig
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import DocumentStatus, FailureMode, PageOutput, PageStatus
from socr.judge.table_verdict import Finding, FindingCode, RungResult, TableJudgeVerdict
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.binding import BindingResult, ContradictedCell

_TABLE_MD = (
    "| c0 | c1 | c2 | c3 |\n"
    "| --- | --- | --- | --- |\n"
    "| 10 | 11 | 12 | 13 |\n"
    "| 20 | 21 | 22 | 23 |\n"
    "| 30 | 31 | 32 | 33 |\n"
)


def _ruled_pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    cols = [100, 220, 300, 380]
    rows = [100 + i * 22 for i in range(4)]
    for r, y in enumerate(rows):
        for c, x in enumerate(cols):
            page.insert_text((x + 4, y + 12), f"{r}{c}", fontsize=9)
    for yy in rows:
        page.draw_line((100, yy), (460, yy))
    for xx in cols + [460]:
        page.draw_line((xx, rows[0]), (xx, rows[-1]))
    pdf_path = tmp_path / name
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _pass(confidence: str = "high") -> RungResult:
    return RungResult(
        rung="fake",
        ok=True,
        verdict=TableJudgeVerdict(verdict="PASS", confidence=confidence, findings=[]),
    )


def _fail(code: FindingCode = FindingCode.FABRICATED_VALUE) -> RungResult:
    return RungResult(
        rung="fake",
        ok=True,
        verdict=TableJudgeVerdict(
            verdict="FAIL",
            confidence="high",
            findings=[Finding(code=code, where="cell", detail="judge rejects")],
        ),
    )


def _not_s1() -> RungResult:
    return RungResult(rung="fake", ok=False, error="simulated infra failure")


class _QueueRung:
    def __init__(self, results: list[RungResult]) -> None:
        self._results = list(results)
        self.calls: list[tuple] = []

    def __call__(self, crop_path, markdown, prior_findings):
        self.calls.append((crop_path, markdown, prior_findings))
        if not self._results:
            raise AssertionError("rung called more times than results provided")
        return self._results.pop(0)


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


def _process(
    pipeline: UnifiedPipeline,
    pdf_path: Path,
    out_dir: Path,
    rungs: list | None,
    *,
    isolate_mechanical: bool = True,
):
    """Run ``process()`` with hermetic provider/judge patches.

    ``isolate_mechanical=True`` (default) stubs the GH-273 detector to
    False so rulings 1-4/6-7 cannot be confounded by bind() on the
    fixture. Ruling 5 turns this off and patches ``bind()`` itself.
    """
    patches = [
        patch(
            "socr.pipeline.orchestrator.route_page",
            side_effect=_route_page_returning(_TABLE_MD),
        ),
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
    ]
    if rungs is not None:
        patches.append(patch.object(pipeline, "_build_table_judge_rungs", return_value=rungs))
    if isolate_mechanical:
        patches.append(
            patch.object(pipeline, "_binding_contradiction_for_witness", return_value=None)
        )
    from contextlib import ExitStack

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        return pipeline.process(pdf_path, out_dir)


def _rejected(error: str | None) -> bool:
    return "table_rejected" in (error or "")


def _unverified(error: str | None) -> bool:
    return "table_unverified" in (error or "")


# ---------------------------------------------------------------------------
# Ruling 1 — last-rung PASS+low is UNVERIFIED
# ---------------------------------------------------------------------------


class TestRuling1LastRungPassLow:
    def test_pass_high_vs_pass_low_difference(self, tmp_path: Path) -> None:
        """Same process(), only the last-rung confidence changes."""
        pdf_high = _ruled_pdf(tmp_path / "high", "doc.pdf")
        pdf_low = _ruled_pdf(tmp_path / "low", "doc.pdf")

        result_high = _process(
            UnifiedPipeline(_make_config()),
            pdf_high,
            tmp_path / "high_out",
            [_QueueRung([_pass("high")])],
        )
        result_low = _process(
            UnifiedPipeline(_make_config()),
            pdf_low,
            tmp_path / "low_out",
            [_QueueRung([_pass("low")])],
        )

        assert _unverified(result_high.error) is False
        assert _unverified(result_low.error) is True
        assert result_high.status != result_low.status


# ---------------------------------------------------------------------------
# Ruling 2 — CLI₂ may overrule CLI₁ FAIL
# ---------------------------------------------------------------------------


class TestRuling2Cli2MayOverruleFail:
    def test_cli2_high_pass_vs_cli2_fail_difference(self, tmp_path: Path) -> None:
        pdf_over = _ruled_pdf(tmp_path / "over", "doc.pdf")
        pdf_rej = _ruled_pdf(tmp_path / "rej", "doc.pdf")

        result_over = _process(
            UnifiedPipeline(_make_config()),
            pdf_over,
            tmp_path / "over_out",
            [_QueueRung([_fail()]), _QueueRung([_pass("high")])],
        )
        result_rej = _process(
            UnifiedPipeline(_make_config()),
            pdf_rej,
            tmp_path / "rej_out",
            [_QueueRung([_fail()]), _QueueRung([_fail()])],
        )

        assert _rejected(result_over.error) is False
        assert _rejected(result_rej.error) is True


# ---------------------------------------------------------------------------
# Ruling 3 — mixed B then C is UNVERIFIED
# ---------------------------------------------------------------------------


class TestRuling3MixedBThenC:
    def test_b_then_c_vs_b_then_b_difference(self, tmp_path: Path) -> None:
        pdf_bc = _ruled_pdf(tmp_path / "bc", "doc.pdf")
        pdf_bb = _ruled_pdf(tmp_path / "bb", "doc.pdf")

        result_bc = _process(
            UnifiedPipeline(_make_config()),
            pdf_bc,
            tmp_path / "bc_out",
            [_QueueRung([_fail()]), _QueueRung([_not_s1()])],
        )
        result_bb = _process(
            UnifiedPipeline(_make_config()),
            pdf_bb,
            tmp_path / "bb_out",
            [_QueueRung([_fail()]), _QueueRung([_fail()])],
        )

        assert _unverified(result_bc.error) is True
        assert _rejected(result_bc.error) is False
        assert _rejected(result_bb.error) is True
        assert _unverified(result_bb.error) is False


# ---------------------------------------------------------------------------
# Ruling 4 — crop + markdown, nothing else
# ---------------------------------------------------------------------------


class TestRuling4NoFindingsPayload:
    def test_cli2_sees_no_prior_findings(self, tmp_path: Path) -> None:
        pdf_path = _ruled_pdf(tmp_path)
        cli2 = _QueueRung([_pass("high")])
        _process(
            UnifiedPipeline(_make_config()),
            pdf_path,
            tmp_path / "out",
            [_QueueRung([_fail()]), cli2],
        )

        assert cli2.calls, "CLI₂ must have been called (ruling 2 override path)"
        _crop, markdown, prior = cli2.calls[0]
        assert markdown.strip() == _TABLE_MD.strip()
        assert prior is None


# ---------------------------------------------------------------------------
# Ruling 5 — mechanical contradiction → UNVERIFIED; fully_checked is not a gate
# ---------------------------------------------------------------------------


class TestRuling5MechanicalUnverified:
    def test_incomplete_coverage_does_not_demote(self, tmp_path: Path) -> None:
        """``fully_checked is False`` with no contradiction is NEUTRAL."""
        pdf_neutral = _ruled_pdf(tmp_path / "neutral", "doc.pdf")
        pdf_shift = _ruled_pdf(tmp_path / "shift", "doc.pdf")

        incomplete = BindingResult()  # defaults: fully_checked is False
        assert incomplete.fully_checked is False
        assert not incomplete.contradicted_cells
        assert not incomplete.row_label_contradictions

        contradiction = BindingResult(
            contradicted_cells=[
                ContradictedCell(
                    row_path=("RowA",),
                    col_path=("OLS",),
                    native_token="100",
                    model_token="100",
                )
            ]
        )
        assert contradiction.fully_checked is False

        with patch("socr.tables.binding.bind", return_value=incomplete):
            result_neutral = _process(
                UnifiedPipeline(_make_config()),
                pdf_neutral,
                tmp_path / "neutral_out",
                [_QueueRung([_pass("high")])],
                isolate_mechanical=False,
            )
        with patch("socr.tables.binding.bind", return_value=contradiction):
            result_shift = _process(
                UnifiedPipeline(_make_config()),
                pdf_shift,
                tmp_path / "shift_out",
                [_QueueRung([_pass("high")])],
                isolate_mechanical=False,
            )

        assert _unverified(result_neutral.error) is False
        assert _unverified(result_shift.error) is True
        assert _rejected(result_shift.error) is False


# ---------------------------------------------------------------------------
# Ruling 6 — agentic produces content terminals; assemble backfills misses
# ---------------------------------------------------------------------------


class TestRuling6ChokePoint:
    def test_skipped_helper_cannot_ship_success_with_assemble_live(self, tmp_path: Path) -> None:
        """An emitted table that never produced a ladder event cannot ship
        SUCCESS. Difference: assemble backfill live vs no-op'd; the
        per-page helper is no-op'd in both.

        Not parametrized over an empty provider ladder: that path ships a
        page-failure marker, not markdown tables, so assemble has nothing
        to backfill and cannot pin this hole.
        """
        active_pdf = _ruled_pdf(tmp_path / "active", "doc.pdf")
        unwired_pdf = _ruled_pdf(tmp_path / "unwired", "doc.pdf")

        def _run(pdf_path: Path, output_dir: Path, *, assemble_backfill: bool):
            pipeline = UnifiedPipeline(_make_config())
            patches = [
                patch(
                    "socr.pipeline.orchestrator.route_page",
                    side_effect=_route_page_returning(_TABLE_MD),
                ),
                patch.object(
                    pipeline,
                    "_available_engines_for_agentic",
                    return_value=[PROFILE_QWEN_LOCAL],
                ),
                patch.object(pipeline, "_resolve_judge_model", return_value=""),
                patch.object(
                    pipeline, "_build_table_judge_rungs", return_value=[_QueueRung([_fail()])]
                ),
                patch.object(pipeline, "_binding_contradiction_for_witness", return_value=None),
                patch.object(pipeline, "_run_table_judge_gate", return_value=None),
            ]
            if not assemble_backfill:
                patches.append(
                    patch.object(
                        pipeline,
                        "_backfill_missing_table_ladder_terminals",
                        return_value=None,
                    )
                )
            from contextlib import ExitStack

            with ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                return pipeline.process(pdf_path, output_dir)

        active = _run(active_pdf, tmp_path / "active_out", assemble_backfill=True)
        unwired = _run(unwired_pdf, tmp_path / "unwired_out", assemble_backfill=False)

        assert _unverified(active.error) is True
        assert active.status == DocumentStatus.AUDIT_FAILED
        assert _unverified(unwired.error) is False
        assert unwired.status == DocumentStatus.SUCCESS

    def test_live_helper_reject_differs_from_skipped_helper_unverified(
        self, tmp_path: Path
    ) -> None:
        """Content terminals still come from the agentic helper."""
        pdf_live = _ruled_pdf(tmp_path / "live", "doc.pdf")
        pdf_skip = _ruled_pdf(tmp_path / "skip", "doc.pdf")

        result_live = _process(
            UnifiedPipeline(_make_config()),
            pdf_live,
            tmp_path / "live_out",
            [_QueueRung([_fail()])],
        )

        pipeline_skip = UnifiedPipeline(_make_config())
        with (
            patch(
                "socr.pipeline.orchestrator.route_page",
                side_effect=_route_page_returning(_TABLE_MD),
            ),
            patch.object(
                pipeline_skip, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline_skip, "_resolve_judge_model", return_value=""),
            patch.object(
                pipeline_skip, "_build_table_judge_rungs", return_value=[_QueueRung([_fail()])]
            ),
            patch.object(pipeline_skip, "_binding_contradiction_for_witness", return_value=None),
            patch.object(pipeline_skip, "_run_table_judge_gate", return_value=None),
        ):
            result_skip = pipeline_skip.process(pdf_skip, tmp_path / "skip_out")

        assert _rejected(result_live.error) is True
        assert result_live.status == DocumentStatus.AUDIT_FAILED
        assert _unverified(result_skip.error) is True
        assert _rejected(result_skip.error) is False
        assert result_skip.status == DocumentStatus.AUDIT_FAILED


# ---------------------------------------------------------------------------
# Ruling 7 — NOT_A_TABLE is REJECTED, not a figure reroute
# ---------------------------------------------------------------------------


class TestRuling7NotATableIsRejected:
    def test_not_a_table_rejects_and_keeps_the_markdown(self, tmp_path: Path) -> None:
        pdf_nat = _ruled_pdf(tmp_path / "nat", "doc.pdf")
        pdf_ok = _ruled_pdf(tmp_path / "ok", "doc.pdf")

        result_nat = _process(
            UnifiedPipeline(_make_config()),
            pdf_nat,
            tmp_path / "nat_out",
            [_QueueRung([_fail(FindingCode.NOT_A_TABLE)])],
        )
        result_ok = _process(
            UnifiedPipeline(_make_config()),
            pdf_ok,
            tmp_path / "ok_out",
            [_QueueRung([_pass("high")])],
        )

        assert _rejected(result_nat.error) is True
        assert _rejected(result_ok.error) is False
        # Not a figure reroute: the emitted table markdown still ships.
        assert "10" in result_nat.markdown
        assert "| 11 |" in result_nat.markdown or "11" in result_nat.markdown
        assert not result_nat.figures
