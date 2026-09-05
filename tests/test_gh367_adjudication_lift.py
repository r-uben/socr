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
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

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
    def test_all_abstained_records_unverified_cause_without_transcribing(
        self, tmp_path: Path
    ) -> None:
        from socr.judge.table_verdict import TABLE_LADDER_UNVERIFIED_KIND

        pipeline = _make_pipeline()
        pdf_path = _row_shift_pdf(tmp_path)
        outcomes = []
        for missing_origin in (False, True):
            state = _make_state(pdf_path)
            with (
                patch.object(
                    pipeline,
                    "_transcribe_cell_token",
                    side_effect=_QueueTranscriber(["RowB", "RowA"]),
                ) as transcribe,
                patch("socr.tables.locate.ordinal_origin", return_value=None)
                if missing_origin
                else nullcontext(),
            ):
                pipeline._run_table_judge_gate(
                    state, 1, state.pages[1], _bo(_SHIFTED_MD), [_accept_rung()]
                )
            outcomes.append(state.pages[1].table_ladder_disposition)
            if missing_origin:
                transcribe.assert_not_called()
                events = [
                    event for event in state.events if event.kind == TABLE_LADDER_UNVERIFIED_KIND
                ]
                assert len(events) == 1
                assert events[0].data["cause"] == "abstained"
                record = next(iter(state.pages[1].binding_adjudication.values()))
                assert record["status"] == "held"
                assert all(item["outcome"] == "abstained" for item in record["items"])
            else:
                assert transcribe.call_count == 2
        assert outcomes[0] != outcomes[1]

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


def test_geometry_address_four_controls_in_one_process(tmp_path: Path) -> None:
    """Hold the printed page fixed while independently breaking each prefix."""
    from socr.tables.binding import RowLabelContradiction, bind
    from socr.tables.locate import label_column_edge, ordinal_origin, row_bands
    from socr.tables.witness import prepare_table_witnesses

    pdf_path = tmp_path / "geometry.pdf"
    markdown = (
        "| Label | OLS |\n| --- | --- |\n| First | 100 |\n"
        "| Target complete | 200 |\n| Last | 300 |\n"
    )
    with fitz.open() as doc:
        page = doc.new_page()
        for baseline, label, value in (
            (112, "Label", "OLS"),
            (132, "First", "100"),
            (146, "Target complete", "200"),
            (160, "Last", "300"),
        ):
            page.insert_text((104, baseline), label, fontsize=9)
            page.insert_text((254, baseline), value, fontsize=9)
        for y in (100, 120, 168):
            page.draw_line((100, y), (350, y))
        doc.save(pdf_path)

    with prepare_table_witnesses(pdf_path, 1, markdown) as witnesses:
        witness = witnesses[0]
        region, table_id = witness.box.bbox, witness.table_id
    with fitz.open(pdf_path) as doc:
        page = doc[0]
        binding = bind(page.get_text("words"), markdown, region=region)
        origin = ordinal_origin(page, region)
        bands = [band for band in row_bands(page, region) if band.y0 >= origin]
        edge = label_column_edge(page, region)
    assert len(binding.native_rows) == len(bands) == 3
    assert binding.row_binding == {0: 0, 1: 1, 2: 2}
    target = binding.native_rows[1]
    full = target.label_bbox
    truncated = (full[0], full[1], (full[0] + full[2]) / 2, full[3])
    # The native row's words anchor it. Its rounded y and the disputed bbox
    # are deliberately unsuitable as independent addresses.
    binding.native_rows = [replace(row, y=origin) for row in binding.native_rows]
    binding.row_label_contradictions = [
        RowLabelContradiction(target.row_path, "Target complete", truncated)
    ]
    binding.row_label_contradictions[0] = replace(
        binding.row_label_contradictions[0], row_path=("Target",)
    )
    binding.native_rows[1] = replace(binding.native_rows[1], row_path=("Target",))

    missing = replace(binding, native_rows=binding.native_rows[1:], row_binding={1: 0, 2: 1})
    inserted = replace(binding, row_binding={0: 0, 2: 1, 3: 2})
    shifted_markdown = markdown.replace("| Target complete", "| Inserted | |\n| Target complete")
    ambiguous_bands = [replace(bands[0], ambiguity="merge-heuristic"), *bands[1:]]
    pipeline = _make_pipeline()
    state = _make_state(pdf_path)
    rendered = []
    real_pixmap = fitz.Page.get_pixmap

    def capture_pixmap(page, *args, **kwargs):
        rendered.append(tuple(kwargs["clip"]))
        return real_pixmap(page, *args, **kwargs)

    records = []
    for current, candidate, geometry, expected_calls in (
        (binding, markdown, bands, 1),
        (missing, markdown, bands, 0),
        (inserted, shifted_markdown, bands, 0),
        (binding, markdown, ambiguous_bands, 0),
    ):
        with (
            patch("socr.tables.locate.row_bands", return_value=geometry),
            patch.object(fitz.Page, "get_pixmap", capture_pixmap),
            patch.object(
                pipeline, "_transcribe_cell_token", return_value="Target complete"
            ) as transcribe,
            patch.object(
                pipeline, "_render_adjudication_crop", wraps=pipeline._render_adjudication_crop
            ) as render,
        ):
            record = pipeline._adjudicate_clamped_table(
                state, 1, table_id, candidate, current, state.pages[1], region=region
            )
        assert transcribe.call_count == expected_calls
        if expected_calls:
            cell = (region[0], bands[1].y0, edge, bands[1].y1)
            assert record.items[0].item.cell_bbox == cell
            assert render.call_args.args[2] == cell
            assert cell != truncated
            assert record.items[0].disproof == "raster_transcription"
        else:
            assert record.items[0].outcome == "abstained"
            assert record.items[0].disproof is None
        records.append(record)
    assert [r.status for r in records] == ["lifted", "held", "held", "held"]
    assert [r.items[0].item.abstain_reason for r in records[1:]] == [
        "native chain breaks at native row 0 (band 1)",
        "model index 2 != band 1",
        "prefix crosses ambiguous band(s) [0]",
    ]
    # Assert the rectangle actually rendered, including both padding clamps.
    assert len(rendered) == 1
    assert rendered[0] == pytest.approx(
        (
            region[0],
            (bands[0].y1 + bands[1].y0) / 2,
            edge,
            (bands[1].y1 + bands[2].y0) / 2,
        )
    )


def test_address_metadata_preserves_resume_and_final_markdown(tmp_path: Path) -> None:
    from socr.tables.adjudication import ContradictionItem

    pdf_path = _row_shift_pdf(tmp_path)
    pipeline = _make_pipeline()
    full_dir, legacy_dir = tmp_path / "full", tmp_path / "legacy"
    full_result = _process(
        pipeline, pdf_path, full_dir, [_accept_rung()], _QueueTranscriber(["RowB", "RowA"])
    )
    new_fields = {"cell_bbox", "address_source", "abstain_reason", "outcome"}
    original = ContradictionItem.to_record

    def legacy_record(item, disproof):
        return {
            key: value for key, value in original(item, disproof).items() if key not in new_fields
        }

    with patch.object(ContradictionItem, "to_record", legacy_record):
        legacy_result = _process(
            _make_pipeline(),
            pdf_path,
            legacy_dir,
            [_accept_rung()],
            _QueueTranscriber(["RowB", "RowA"]),
        )
    assert full_result.status == legacy_result.status
    full_md = list(full_dir.rglob("doc.md"))
    legacy_md = list(legacy_dir.rglob("doc.md"))
    assert len(full_md) == len(legacy_md) == 1
    assert full_md[0].read_bytes() == legacy_md[0].read_bytes()
    assert full_result.markdown == legacy_result.markdown

    sidecar = next(full_dir.rglob("pages/00001.json"))
    baseline = json.loads(sidecar.read_text())
    legacy = json.loads(next(legacy_dir.rglob("pages/00001.json")).read_text())
    baseline_items = next(iter(baseline["binding_adjudication"].values()))["items"]
    legacy_items = next(iter(legacy["binding_adjudication"].values()))["items"]
    assert all(new_fields <= item.keys() for item in baseline_items)
    assert all(new_fields.isdisjoint(item) for item in legacy_items)

    # Keep fingerprint and every other field fixed; strip only the new item
    # metadata, then compare the real terminal ledger gate and restoration.
    decisions = []
    for include_fields in (True, False):
        meta = json.loads(json.dumps(baseline))
        if not include_fields:
            for table in meta["binding_adjudication"].values():
                for item in table["items"]:
                    for key in new_fields:
                        item.pop(key)
        sidecar.write_text(json.dumps(meta))
        state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
        decision = pipeline._load_terminal_page(state, 1, full_dir)
        decisions.append(decision)
        if decision is not None:
            pipeline._restore_terminal_page_state(state, 1, decision, full_dir)
            assert state.pages[1].binding_adjudication == meta["binding_adjudication"]
    assert (decisions[0] is None) == (decisions[1] is None)
    assert decisions[0] is not None, "the control must exercise a skippable terminal page"
    if decisions[0] is not None:
        assert decisions[0].text == decisions[1].text
        assert decisions[0].status == decisions[1].status
        assert decisions[0].audit_passed == decisions[1].audit_passed
