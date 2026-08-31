"""GH-367: binding-clamp adjudication lift, at the real caller.

The spec is ``docs/log/2026-08-31_gh367-adjudication-lift.md``. Disproof is
exact identity under bind()'s own normalizers: encoding-garbage native
tokens, or an independent cell-raster transcription that matches markdown
and not native. An ordinary judge PASS never lifts the GH-359 ruling 5
clamp. A helper-unit suite is not a gate — this file drives
``_run_table_judge_gate`` and ``UnifiedPipeline.process()``.

Every process()-level assertion is a same-process DIFFERENCE (#253/#257): two
runs changing only the thing under test. Absolute status pins were removed in
review -- the pipeline is mocked here so they are deterministic today, but the
repo's rule exists because exactly that kind of pin broke main before, and a
future unpinned audit step must not turn this file red.

Hermetic: CI has no ollama. Transcriber is injected. ``_available_engines
_for_agentic`` and ``_resolve_judge_model`` are patched wherever
``process()`` runs. Never pin an absolute outcome measured on one machine
(#253/#257): every process() assertion is a same-process difference.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import fitz

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import DocumentStatus, FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.judge.table_verdict import (
    TABLE_BINDING_ADJUDICATED_KIND,
    TABLE_LADDER_ACCEPTED_KIND,
    RungResult,
    TableJudgeVerdict,
)
from socr.pipeline.orchestrator import UnifiedPipeline

_SHIFTED_MD = (
    "|      | OLS | IV  |\n| ---- | --- | --- |\n| RowB | 100 | 200 |\n| RowA | 300 | 400 |\n"
)


def _row_shift_pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
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


class _QueueRung:
    def __init__(self, results: list[RungResult], rung_id: str = "fake") -> None:
        self._results = list(results)
        self.rung_id = rung_id
        self.calls: list[tuple] = []

    def __call__(self, crop_path, markdown, prior_findings):
        self.calls.append((crop_path, markdown, prior_findings))
        if not self._results:
            raise AssertionError(f"{self.rung_id} called more times than results provided")
        return self._results.pop(0)


class _QueueTranscriber:
    def __init__(self, tokens: list[str | None]) -> None:
        self._tokens = list(tokens)
        self.calls: list = []

    def __call__(self, crop_path):
        self.calls.append(crop_path)
        if not self._tokens:
            return None
        return self._tokens.pop(0)


def _accept_rung() -> _QueueRung:
    return _QueueRung(
        [
            RungResult(
                rung="fake1",
                ok=True,
                verdict=TableJudgeVerdict(verdict="PASS", confidence="high", findings=[]),
            )
        ]
    )


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


def _process(pipeline: UnifiedPipeline, pdf_path: Path, out_dir: Path, rungs, transcribe):
    with (
        patch(
            "socr.pipeline.orchestrator.route_page",
            side_effect=_route_page_returning(_SHIFTED_MD),
        ),
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_build_table_judge_rungs", return_value=rungs),
        patch.object(pipeline, "_transcribe_cell_token", side_effect=transcribe),
    ):
        return pipeline.process(pdf_path, out_dir)


# ---------------------------------------------------------------------------
# Gate-level: _run_table_judge_gate is the real caller of the clamp.
# ---------------------------------------------------------------------------


class TestGateLift:
    def test_transcriber_matching_markdown_lifts_clamp(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        pdf_path = _row_shift_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        transcribe = _QueueTranscriber(["RowB", "RowA"])
        with patch.object(pipeline, "_transcribe_cell_token", side_effect=transcribe):
            pipeline._run_table_judge_gate(state, 1, ps, _bo(_SHIFTED_MD), [_accept_rung()])

        assert ps.table_ladder_disposition is None
        accepted = [e for e in state.events if e.kind == TABLE_LADDER_ACCEPTED_KIND]
        adjudicated = [e for e in state.events if e.kind == TABLE_BINDING_ADJUDICATED_KIND]
        assert len(accepted) == 1
        assert len(adjudicated) == 1
        assert adjudicated[0].data["status"] == "lifted"
        assert ps.binding_adjudication[accepted[0].data["table_id"]]["status"] == "lifted"
        assert transcribe.calls

    def test_transcriber_matching_native_does_not_lift(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        pdf_path = _row_shift_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        transcribe = _QueueTranscriber(["RowA", "RowB"])
        with patch.object(pipeline, "_transcribe_cell_token", side_effect=transcribe):
            pipeline._run_table_judge_gate(state, 1, ps, _bo(_SHIFTED_MD), [_accept_rung()])

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        assert list(ps.binding_adjudication.values())[0]["status"] == "held"

    def test_partial_transcription_does_not_lift(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        pdf_path = _row_shift_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        transcribe = _QueueTranscriber(["RowB", "RowB"])
        with patch.object(pipeline, "_transcribe_cell_token", side_effect=transcribe):
            pipeline._run_table_judge_gate(state, 1, ps, _bo(_SHIFTED_MD), [_accept_rung()])

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        record = list(ps.binding_adjudication.values())[0]
        assert record["status"] == "held"
        disproofs = [item["disproof"] for item in record["items"]]
        assert None in disproofs
        assert "raster_transcription" in disproofs

    def test_encoding_garbage_lifts_without_transcriber(self, tmp_path: Path) -> None:
        """fitz insert_text sanitizes PUA/U+FFFD, so the character class is
        pinned in ``test_binding_adjudication``. Here the real gate runs on
        the GH-273 PDF and only the detector flips — the transcriber must
        not be consulted."""
        pipeline = _make_pipeline()
        pdf_path = _row_shift_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        with (
            patch("socr.tables.adjudication.token_is_encoding_garbage", return_value=True),
            patch.object(pipeline, "_transcribe_cell_token", return_value=None) as transcribe,
        ):
            pipeline._run_table_judge_gate(state, 1, ps, _bo(_SHIFTED_MD), [_accept_rung()])

        assert ps.table_ladder_disposition is None
        assert list(ps.binding_adjudication.values())[0]["status"] == "lifted"
        transcribe.assert_not_called()

    def test_prior_lift_survives_transcriber_unavailable(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        pdf_path = _row_shift_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        transcribe = _QueueTranscriber(["RowB", "RowA"])
        with patch.object(pipeline, "_transcribe_cell_token", side_effect=transcribe):
            pipeline._run_table_judge_gate(state, 1, ps, _bo(_SHIFTED_MD), [_accept_rung()])
        assert ps.table_ladder_disposition is None

        state2 = _make_state(pdf_path)
        ps2 = state2.pages[1]
        ps2.binding_adjudication = dict(ps.binding_adjudication)
        with patch.object(pipeline, "_transcribe_cell_token", return_value="RowA") as again:
            pipeline._run_table_judge_gate(state2, 1, ps2, _bo(_SHIFTED_MD), [_accept_rung()])

        assert ps2.table_ladder_disposition is None
        assert list(ps2.binding_adjudication.values())[0]["status"] == "lifted"
        again.assert_not_called()


# ---------------------------------------------------------------------------
# process() DIFFERENCE pins — the content gate. A helper-unit green while
# the agentic `if` / the adjudication loop is commented out is not a gate.
# ---------------------------------------------------------------------------


class TestProcessDifference:
    def test_transcriber_markdown_vs_native_is_the_lift(self, tmp_path: Path) -> None:
        pdf_hold = _row_shift_pdf(tmp_path / "hold", "doc.pdf")
        pdf_lift = _row_shift_pdf(tmp_path / "lift", "doc.pdf")

        result_hold = _process(
            _make_pipeline(),
            pdf_hold,
            tmp_path / "hold_out",
            [_accept_rung()],
            _QueueTranscriber(["RowA", "RowB"]),
        )
        result_lift = _process(
            _make_pipeline(),
            pdf_lift,
            tmp_path / "lift_out",
            [_accept_rung()],
            _QueueTranscriber(["RowB", "RowA"]),
        )

        assert result_hold.status != result_lift.status
        assert "table_unverified" in (result_hold.error or "")
        assert "table_unverified" not in (result_lift.error or "")
        assert "table_rejected" not in (result_lift.error or "")

        lift_sidecars = list((tmp_path / "lift_out").rglob("00001.json"))
        hold_sidecars = list((tmp_path / "hold_out").rglob("00001.json"))
        assert lift_sidecars and hold_sidecars
        lift_meta = json.loads(lift_sidecars[0].read_text(encoding="utf-8"))
        hold_meta = json.loads(hold_sidecars[0].read_text(encoding="utf-8"))
        lift_statuses = [
            v.get("status") for v in lift_meta.get("binding_adjudication", {}).values()
        ]
        hold_statuses = [
            v.get("status") for v in hold_meta.get("binding_adjudication", {}).values()
        ]
        assert "lifted" in lift_statuses
        assert "held" in hold_statuses
        assert lift_meta.get("table_ladder_disposition") is None
        assert hold_meta.get("table_ladder_disposition") != lift_meta.get(
            "table_ladder_disposition"
        )

    def test_encoding_garbage_vs_well_formed_is_the_lift(self, tmp_path: Path) -> None:
        """Same PDF, same shifted markdown, same accepting rung, transcriber
        inert. Only ``token_is_encoding_garbage`` flips. Well-formed labels
        stay clamped; treating them as encoding garbage lifts."""
        pdf_held = _row_shift_pdf(tmp_path / "held", "doc.pdf")
        pdf_lift = _row_shift_pdf(tmp_path / "lift", "doc.pdf")

        def _run(pdf_path, out_dir, garbage: bool):
            pipeline = _make_pipeline()
            patches = [
                patch(
                    "socr.pipeline.orchestrator.route_page",
                    side_effect=_route_page_returning(_SHIFTED_MD),
                ),
                patch.object(
                    pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
                ),
                patch.object(pipeline, "_resolve_judge_model", return_value=""),
                patch.object(pipeline, "_build_table_judge_rungs", return_value=[_accept_rung()]),
                patch.object(pipeline, "_transcribe_cell_token", return_value=None),
                patch(
                    "socr.tables.adjudication.token_is_encoding_garbage",
                    return_value=garbage,
                ),
            ]
            from contextlib import ExitStack

            with ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                return pipeline.process(pdf_path, out_dir)

        result_held = _run(pdf_held, tmp_path / "held_out", garbage=False)
        result_lift = _run(pdf_lift, tmp_path / "lift_out", garbage=True)

        assert result_held.status != result_lift.status
        assert "table_unverified" in (result_held.error or "")
        assert "table_unverified" not in (result_lift.error or "")

    def test_flag_off_does_not_adjudicate(self, tmp_path: Path) -> None:
        """Ladder stays default-off: flag off vs on, same shifted table,
        transcriber would lift if it ran. Flag off ships; flag on with a
        native-agreeing transcriber still demotes — the clamp is not
        weakened, and off does not grow a new path."""
        pdf_off = _row_shift_pdf(tmp_path / "off", "doc.pdf")
        pdf_on = _row_shift_pdf(tmp_path / "on", "doc.pdf")

        pipeline_off = _make_pipeline(_make_config(table_judge_ladder=False))
        with (
            patch(
                "socr.pipeline.orchestrator.route_page",
                side_effect=_route_page_returning(_SHIFTED_MD),
            ),
            patch.object(
                pipeline_off, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline_off, "_resolve_judge_model", return_value=""),
            patch.object(pipeline_off, "_transcribe_cell_token", return_value="RowB"),
        ):
            result_off = pipeline_off.process(pdf_off, tmp_path / "off_out")

        pipeline_on = _make_pipeline()
        with (
            patch(
                "socr.pipeline.orchestrator.route_page",
                side_effect=_route_page_returning(_SHIFTED_MD),
            ),
            patch.object(
                pipeline_on, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline_on, "_resolve_judge_model", return_value=""),
            patch.object(pipeline_on, "_build_table_judge_rungs", return_value=[_accept_rung()]),
            patch.object(pipeline_on, "_transcribe_cell_token", return_value="RowA"),
        ):
            result_on = pipeline_on.process(pdf_on, tmp_path / "on_out")

        assert result_off.status != result_on.status
        assert "table_unverified" in (result_on.error or "")
