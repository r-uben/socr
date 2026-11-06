"""#1267 at ``process()``: the row-binding check's effect on a shipped page, as DIFFERENCES.

Every arm runs the same pipeline on the same synthetic PDF with the same fake model output;
the control arm replaces ``_run_row_binding_check`` with a no-op, so the only thing that
changes between arms is whether the check runs.  No absolute status is pinned: provider-
dependent machinery may legitimately move it (CLAUDE.md, the no-provider trap).
"""

from __future__ import annotations

import contextlib
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fitz")

from test_gh1267_row_binding_check import GROUP, build_page, md  # noqa: E402

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.providers import PROFILE_QWEN_LOCAL  # noqa: E402
from socr.core.result import PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402
from socr.pipeline.agentic import PageDecision, ProviderAttempt  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

CONTRADICTED_KIND = "table_row_binding_contradicted"
MARKER = "[page 1 failed: unverifiable table — see image]"
LINK = "![Failed table page 1](figures/failed_table_p1.png)"
WITHHELD_KIND = "table_ladder_withheld"

CLEAN_ROWS = list(GROUP)
SHIFTED_ROWS = [
    ("Real GDP", GROUP[0][1]),
    ("Final sales", []),
    ("Priv. dom. final purchases", GROUP[1][1]),
    ("", GROUP[2][1]),
    ("Net exports", GROUP[3][1]),
]


SWAPPED_ROWS = [
    ("Real GDP", GROUP[0][1]),
    ("Priv. dom. final purchases", GROUP[1][1]),
    ("Final sales", GROUP[2][1]),
    ("Net exports", GROUP[3][1]),
]


def _pdf(tmp_path: Path, rotation: int, native=None) -> Path:
    page = build_page(native or GROUP, rotation)
    path = tmp_path / f"grouped_{rotation}_{len(native or GROUP)}.pdf"
    page.parent.save(path)
    return path


def _run(
    tmp_path: Path,
    rows,
    rotation: int,
    *,
    check_runs: bool,
    out_name: str,
    pdf: Path | None = None,
    ladder_replaces: list | None = None,
    patch_replaces: list | None = None,
    native=None,
    markdown: str | None = None,
):
    pdf = pdf or _pdf(tmp_path, rotation, native)
    out_dir = tmp_path / out_name
    text = markdown if markdown is not None else md(rows)
    cfg = PipelineConfig(
        primary_engine=EngineType.QWEN,
        local_engine=EngineType.QWEN,
        agentic=True,
        strict_local=True,
        judge_backend="heuristic",
        enabled_engines=[EngineType.QWEN],
        table_judge_ladder=ladder_replaces is not None,
        quiet=True,
        save_figures=False,
        write_manifest=True,
    )
    pipeline = UnifiedPipeline(cfg)

    def _fake_route(page_num, ladder, run_provider, judge, **kwargs):
        out = PageOutput(
            page_num=page_num,
            text=text,
            status=PageStatus.SUCCESS,
            engine="qwen",
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

    captured: dict = {}
    original_assemble = pipeline._phase_assemble

    def _assemble_wrapper(state, od):
        captured["state"] = state
        return original_assemble(state, od)

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("socr.pipeline.orchestrator.route_page", side_effect=_fake_route))
        stack.enter_context(
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            )
        )
        stack.enter_context(patch.object(pipeline, "_resolve_judge_model", return_value=""))
        stack.enter_context(patch.object(pipeline, "_plan_native_table_first", return_value=None))
        stack.enter_context(patch.object(pipeline, "_build_table_judge_rungs", return_value=[]))
        stack.enter_context(patch.object(pipeline, "_is_chart_asset_page", return_value=False))
        stack.enter_context(
            patch.object(pipeline, "_reconcile_chart_table_grids", lambda *a, **k: 0)
        )
        stack.enter_context(
            patch.object(pipeline, "_suppress_chart_table_skeletons", lambda *a, **k: 0)
        )
        stack.enter_context(patch("socr.tables.reconstruct.chart_region_bboxes", return_value=[]))
        if ladder_replaces is not None:
            # The ladder / escalation REPLACES the table after the model read it: a fake gate that
            # swaps the text, and the native-contradiction withhold (which would ship a marker) off.
            new_text = md(ladder_replaces)
            stack.enter_context(
                patch.object(
                    pipeline,
                    "_run_table_judge_gate",
                    side_effect=lambda state, page_num, ps, bo, rungs: setattr(
                        bo, "text", new_text
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    pipeline, "_withhold_contradicted_unverified_tables", lambda *a, **k: None
                )
            )
        if patch_replaces is not None:
            # A crop patch that passed the value check: it runs before the ladder, releases the
            # unverified state, and leaves a value-equal table.
            patched = md(patch_replaces)

            def _patch(state, page_num, bo, *a, **k):
                bo.text = patched
                ps = state.pages[page_num]
                ps.unchecked_table_kind = ""
                ps.table_ladder_disposition = None

            stack.enter_context(
                patch.object(pipeline, "_guard_agentic_page_table_repetition", side_effect=_patch)
            )
        if not check_runs:
            stack.enter_context(
                patch.object(pipeline, "_run_row_binding_check", lambda *a, **k: None)
            )
        stack.enter_context(
            patch.object(pipeline, "_phase_assemble", side_effect=_assemble_wrapper)
        )
        result = pipeline.process(pdf, out_dir)
    return pipeline, result, captured.get("state"), out_dir, pdf


def _sidecar(out_dir: Path) -> dict:
    return json.loads(next(out_dir.rglob("pages/00001.json")).read_text(encoding="utf-8"))


def _fragment(out_dir: Path) -> str:
    return next(out_dir.rglob("pages/00001.md")).read_text(encoding="utf-8")


def _assembled(out_dir: Path) -> str:
    docs = [p for p in out_dir.rglob("*.md") if p.parent.name != "pages"]
    assert len(docs) == 1, docs
    return docs[0].read_text(encoding="utf-8")


def _trust(out_dir: Path) -> dict:
    return json.loads(next(out_dir.rglob("tables_trust.json")).read_text(encoding="utf-8"))


def _events(out_dir: Path, kind: str) -> list[dict]:
    log = json.loads(next(out_dir.rglob("audit_log.json")).read_text(encoding="utf-8"))
    return [e for e in log.get("events", []) if e.get("kind") == kind]


# Run-to-run noise that is not part of the table verdict.
_NOISE = {
    "timings_s",
    "row_binding",
    "row_binding_contradicted_tables",
    "row_binding_unverifiable_tables",
    "row_binding_table_count",
    "row_binding_withhold_only",
    "row_binding_mark_only",
}


def _stable(sidecar: dict) -> dict:
    return {k: v for k, v in sidecar.items() if k not in _NOISE}


@pytest.mark.parametrize("rotation", [0, 90])
def test_clean_table_the_check_changes_nothing_but_its_record(tmp_path, rotation):
    _, res_on, _, on, pdf = _run(tmp_path, CLEAN_ROWS, rotation, check_runs=True, out_name="on")
    _, res_off, _, off, _ = _run(
        tmp_path, CLEAN_ROWS, rotation, check_runs=False, out_name="off", pdf=pdf
    )
    assert _fragment(on) == _fragment(off)
    assert _assembled(on) == _assembled(off)
    assert _stable(_sidecar(on)) == _stable(_sidecar(off))
    assert res_on.status == res_off.status
    assert _events(on, CONTRADICTED_KIND) == []
    assert _trust(on) == _trust(off)
    # and the check did run: its record is the only new field
    assert _sidecar(on).get("row_binding")
    assert "row_binding" not in _sidecar(off)


def _withheld_trust(out_dir: Path) -> int:
    return _trust(out_dir)["counts_by_kind"].get(WITHHELD_KIND, 0)


@pytest.mark.parametrize("rotation", [0, 90])
def test_shifted_table_is_withheld_with_marker_and_page_picture(tmp_path, rotation):
    """Owner ruling 2026-11-05: a proven contradiction withholds the table, at every surface."""
    _, res_on, state_on, on, pdf = _run(
        tmp_path, SHIFTED_ROWS, rotation, check_runs=True, out_name="on"
    )
    _, res_off, _, off, _ = _run(
        tmp_path, SHIFTED_ROWS, rotation, check_runs=False, out_name="off", pdf=pdf
    )
    frag_on, frag_off = _fragment(on), _fragment(off)
    # the bytes: gone from the page, the marker and ONE picture link in their place
    for line in md(SHIFTED_ROWS).splitlines()[2:]:
        assert line in frag_off and line not in frag_on
    assert frag_on.count(MARKER) == 1 and MARKER not in frag_off
    assert frag_on.count(LINK) == 1 and LINK not in frag_off
    assert any(p.name == "failed_table_p1.png" for p in on.rglob("*.png"))

    # the page: failure mode / disposition WITHHELD, only because the check ran
    side_on, side_off = _sidecar(on), _sidecar(off)
    assert side_on["winning_output"]["failure_mode"] == "table_withheld"
    assert side_on["table_ladder_disposition"] == "table_withheld"
    assert side_off["table_ladder_disposition"] != "table_withheld"
    assert side_on["winning_output"]["status"] == PageStatus.ERROR.value
    assert side_off["winning_output"]["status"] != PageStatus.ERROR.value

    # the events: the evidence and the withheld terminal, one each
    assert len(_events(on, CONTRADICTED_KIND)) == 1 and _events(off, CONTRADICTED_KIND) == []
    withheld = _events(on, WITHHELD_KIND)
    assert len(withheld) == 1 and withheld[0]["data"]["reason"] == "row_binding_contradiction"
    assert {"emptied", "displaced", "label_lost"} <= {
        r["verdict"] for r in withheld[0]["data"]["rows"]
    }

    # the unshipped model table stays in the sidecar beside its evidence
    record = side_on["row_binding"]["p1-t0"]
    assert record["withheld_markdown"] == md(SHIFTED_ROWS).rstrip("\n")
    assert record["outcome"] == "contradicted"

    # trust, metadata, document status: reported as a withheld table, +1 against the control
    assert _withheld_trust(on) == _withheld_trust(off) + 1
    assert _trust(on)["untrusted_pages"] == [1]
    note_on = next(iter(json.loads(next(on.rglob("metadata.json")).read_text())["files"].values()))
    assert "row labels do not bind" in (note_on.get("error") or "")
    assert state_on.pages[1].table_ladder_disposition.value == "table_withheld"
    assert res_on.status.value == "error" and res_off.status.value != "error"


TWO_NATIVE = [
    *GROUP,
    ("Exports", ["2.1", "2.2", "2.3"]),
    ("Imports", ["3.1", "3.2", "3.3"]),
    ("Services", ["4.1", "4.2", "4.3"]),
]
SECOND_TABLE = [
    ("Exports", ["2.1", "2.2", "2.3"]),
    ("Imports", ["3.1", "3.2", "3.3"]),
    ("Services", ["4.1", "4.2", "4.3"]),
]


def _two_tables(first_rows) -> str:
    return md(first_rows) + "\nProse between the tables.\n\n" + md(SECOND_TABLE)


@pytest.mark.parametrize("rotation", [0, 90])
def test_only_the_contradicted_table_is_withheld(tmp_path, rotation):
    markdown = _two_tables(SHIFTED_ROWS)
    _, _, _, on, pdf = _run(
        tmp_path,
        None,
        rotation,
        check_runs=True,
        out_name="on",
        native=TWO_NATIVE,
        markdown=markdown,
    )
    _, _, _, off, _ = _run(
        tmp_path,
        None,
        rotation,
        check_runs=False,
        out_name="off",
        native=TWO_NATIVE,
        markdown=markdown,
        pdf=pdf,
    )
    frag_on = _fragment(on)
    assert frag_on.count(MARKER) == 1 and frag_on.count(LINK) == 1
    for line in md(SHIFTED_ROWS).splitlines()[2:]:
        assert line not in frag_on and line in _fragment(off)
    # the bound sibling and the prose between ship untouched
    for line in md(SECOND_TABLE).splitlines():
        assert line in frag_on
    assert "Prose between the tables." in frag_on
    withheld = _events(on, WITHHELD_KIND)
    assert [e["data"]["table_id"] for e in withheld] == ["p1-t0"]
    assert _sidecar(on)["row_binding_contradicted_tables"] == [0]
    assert _sidecar(on)["row_binding"]["p1-t1"]["outcome"] == "bound"


@pytest.mark.parametrize("rotation", [0, 90])
def test_a_withheld_table_and_an_unverifiable_sibling_each_keep_their_own_outcome(
    tmp_path, rotation
):
    second_slip = [("Exports", ["2.1", "2.2", "2.3"]), ("Imxorts", ["3.1", "3.2", "3.3"])]
    native = [*GROUP, *SECOND_TABLE]
    markdown = md(SHIFTED_ROWS) + "\nProse between the tables.\n\n" + md(second_slip)
    _, _, _, on, _ = _run(
        tmp_path, None, rotation, check_runs=True, out_name="on", native=native, markdown=markdown
    )
    frag = _fragment(on)
    assert frag.count(MARKER) == 1
    # the unverifiable sibling is marked with ITS number (the second table), the withheld one is gone
    assert frag.count("row binding not verified on the page's text layer for table 2") == 1
    assert "table 1;" not in frag
    for line in md(second_slip).splitlines()[2:]:
        assert line in frag


@pytest.mark.parametrize("rotation", [0, 90])
def test_withheld_page_rerun_and_resume_are_byte_identical(tmp_path, rotation):
    _, _, _, first, pdf = _run(tmp_path, SHIFTED_ROWS, rotation, check_runs=True, out_name="out")
    keys = (
        "table_ladder_disposition",
        "row_binding",
        "row_binding_contradicted_tables",
        "row_binding_table_count",
        "row_binding_withhold_only",
        "d3_floor_png_ref",
    )

    def view(out):
        side = _sidecar(out)
        win = side["winning_output"]
        # (the restored sidecar text is read back from the fragment, which has no trailing newline)
        return (
            {k: side.get(k) for k in keys}
            | {k: win.get(k) for k in ("status", "failure_mode")}
            | {"text": (win.get("text") or "").strip()}
        )

    frag, side = _fragment(first), view(first)
    # a withheld page is a content terminal: the second run restores it instead of re-reading it
    _run(tmp_path, SHIFTED_ROWS, rotation, check_runs=True, out_name="out", pdf=pdf)
    assert _fragment(first) == frag
    assert view(first) == side
    assert frag.count(MARKER) == 1 and frag.count(LINK) == 1
    assert len(_events(first, CONTRADICTED_KIND)) == 1 and len(_events(first, WITHHELD_KIND)) == 1


@pytest.mark.parametrize("rotation", [0, 90])
def test_withheld_state_survives_flush_restore_and_refinalize(tmp_path, rotation):
    from socr.core.document import DocumentHandle
    from socr.core.manifest import _apply_ladder_disposition_guard

    markdown = _two_tables(SHIFTED_ROWS)
    pipeline, _, state, out_dir, pdf = _run(
        tmp_path,
        None,
        rotation,
        check_runs=True,
        out_name="out",
        native=TWO_NATIVE,
        markdown=markdown,
    )
    fragment = _fragment(out_dir)
    assert pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True) is not None
    resumed = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    resumed.pages[1] = state.pages[1]
    restored = PageOutput(
        page_num=1, text=fragment, status=PageStatus.SUCCESS, engine="qwen", audit_passed=True
    )
    pipeline._restore_terminal_page_state(resumed, 1, restored, out_dir)
    ps = resumed.pages[1]
    kinds = [e.kind for e in resumed.events if e.page_num == 1]
    assert kinds.count(CONTRADICTED_KIND) == 1 and kinds.count(WITHHELD_KIND) == 1
    assert ps.row_binding_contradicted_tables == [0] and ps.row_binding_withhold_only
    assert ps.row_binding_table_count == 2
    assert ps.table_ladder_disposition.value == "table_withheld"
    # re-finalizing the already-spliced text changes nothing: the marker is not stacked
    again = _apply_ladder_disposition_guard(restored, 1, ps)
    assert again.text == fragment


def test_a_splice_that_cannot_be_placed_falls_back_to_the_page_floor():
    """Fail closed: ordinals that do not fit the text withhold the whole page, never ship the table."""
    from socr.core.manifest import _apply_ladder_disposition_guard
    from socr.core.result import FailureMode

    ps = type("P", (), {})()
    ps.table_ladder_disposition = FailureMode.TABLE_WITHHELD
    ps.row_binding_contradicted_tables = [3]
    ps.row_binding_table_count = 4
    ps.row_binding_withhold_only = True
    ps.d3_floor_png_ref = ""
    out = PageOutput(
        page_num=1,
        text=md(SHIFTED_ROWS),
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    guarded = _apply_ladder_disposition_guard(out, 1, ps)
    assert "Final sales" not in guarded.text and guarded.status is PageStatus.ERROR


def _pair(tmp_path, first_rows, rotation, **kw):
    _, res_on, _, on, pdf = _run(
        tmp_path, first_rows, rotation, check_runs=True, out_name="on", **kw
    )
    _, res_off, _, off, _ = _run(
        tmp_path, first_rows, rotation, check_runs=False, out_name="off", pdf=pdf, **kw
    )
    return res_on, on, res_off, off


@pytest.mark.parametrize("first", ["clean", "shifted"])
@pytest.mark.parametrize("rotation", [0, 90])
def test_a_shifted_replacement_is_withheld_whatever_the_model_first_read(tmp_path, rotation, first):
    """The ladder replaces the table with a DIFFERENT shifted one: the verdict is about what ships."""
    first_rows = CLEAN_ROWS if first == "clean" else SHIFTED_ROWS
    _, on, _, off = _pair(tmp_path, first_rows, rotation, ladder_replaces=SWAPPED_ROWS)
    frag_on = _fragment(on)
    assert frag_on.count(MARKER) == 1 and MARKER not in _fragment(off)
    for line in md(SWAPPED_ROWS).splitlines()[2:]:
        assert line not in frag_on  # the replacement's bytes are the ones withheld
    assert len(_events(on, CONTRADICTED_KIND)) == 1 and _events(off, CONTRADICTED_KIND) == []
    assert _withheld_trust(on) >= _withheld_trust(off) + 1
    assert _sidecar(on)["row_binding"]["p1-t0"]["withheld_markdown"] == md(SWAPPED_ROWS).rstrip(
        "\n"
    )


@pytest.mark.parametrize("rotation", [0, 90])
def test_a_corrected_replacement_is_not_withheld(tmp_path, rotation):
    """The model's shifted table is replaced by a correct one: the check adds nothing."""
    res_on, on, res_off, off = _pair(tmp_path, SHIFTED_ROWS, rotation, ladder_replaces=CLEAN_ROWS)
    assert _events(on, CONTRADICTED_KIND) == [] and _events(on, WITHHELD_KIND) == _events(
        off, WITHHELD_KIND
    )
    assert MARKER not in _fragment(on)
    assert _fragment(on) == _fragment(off)
    assert _stable(_sidecar(on)) == _stable(_sidecar(off))
    assert res_on.status == res_off.status


@pytest.mark.parametrize("rotation", [0, 90])
def test_a_value_correct_crop_patch_that_is_still_shifted_is_withheld(tmp_path, rotation):
    """A patch the value check accepted releases the unverified state; the table is still shifted."""
    _, on, _, off = _pair(tmp_path, CLEAN_ROWS, rotation, patch_replaces=SHIFTED_ROWS)
    assert _sidecar(on)["table_ladder_disposition"] == "table_withheld"
    assert _sidecar(off)["table_ladder_disposition"] != "table_withheld"
    assert _fragment(on).count(MARKER) == 1 and MARKER not in _fragment(off)
    assert len(_events(on, CONTRADICTED_KIND)) == 1


UNVERIFIABLE_KIND = "table_row_binding_unverifiable"
UNVERIFIABLE_MARK = "row binding not verified on the page's text layer for table 1"
# A label the text layer does not print (a transcription slip) on a row whose values it does print:
# the values are found, the label cannot be confirmed.
ABSENT_VALUES_ROWS = [*GROUP[:3], ("Net xprts", GROUP[3][1])]


@pytest.mark.parametrize("rotation", [0, 90])
def test_unverifiable_table_is_marked_without_claiming_a_contradiction(tmp_path, rotation):
    _, res_on, _, on, pdf = _run(
        tmp_path, ABSENT_VALUES_ROWS, rotation, check_runs=True, out_name="on"
    )
    _, res_off, _, off, _ = _run(
        tmp_path, ABSENT_VALUES_ROWS, rotation, check_runs=False, out_name="off", pdf=pdf
    )
    side_on, side_off = _sidecar(on), _sidecar(off)
    assert side_on["winning_output"]["status"] == PageStatus.WARNING.value
    assert side_off["winning_output"]["status"] != PageStatus.WARNING.value
    assert side_on["winning_output"]["failure_mode"] == "table_unverified"
    assert side_on["unchecked_table_kind"] == "unverified"
    assert side_on["row_binding_mark_only"] is True
    assert side_on["row_binding_unverifiable_tables"] == [0]
    frag = _fragment(on)
    assert frag.count(UNVERIFIABLE_MARK) == 1 and UNVERIFIABLE_MARK not in _fragment(off)
    assert (
        "do not bind" not in frag and MARKER not in frag
    )  # not withheld, no contradiction claimed
    assert side_on["table_ladder_disposition"] != "table_withheld"
    assert _assembled(on).count(UNVERIFIABLE_MARK) == 1
    assert len(_events(on, UNVERIFIABLE_KIND)) == 1 and _events(off, UNVERIFIABLE_KIND) == []
    assert _events(on, CONTRADICTED_KIND) == []
    assert _trust(on)["counts_by_kind"].get(UNVERIFIABLE_KIND) == 1
    assert UNVERIFIABLE_KIND not in _trust(off)["counts_by_kind"]
    assert side_on["winning_output"]["status"] != side_off["winning_output"]["status"]


@pytest.mark.parametrize("rotation", [0, 90])
def test_unverifiable_state_survives_flush_and_restore(tmp_path, rotation):
    from socr.core.document import DocumentHandle

    pipeline, _, state, out_dir, pdf = _run(
        tmp_path, ABSENT_VALUES_ROWS, rotation, check_runs=True, out_name="out"
    )
    assert pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True) is not None
    resumed = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    resumed.pages[1] = copy.deepcopy(state.pages[1])
    # start from a page that carries none of the row-binding state, so only the restore supplies it
    UnifiedPipeline.rollback_row_binding_page_state(resumed, 1, resumed.pages[1])
    resumed.pages[1].unchecked_table_kind = ""
    restored = PageOutput(
        page_num=1, text="x", status=PageStatus.SUCCESS, engine="qwen", audit_passed=True
    )
    pipeline._restore_terminal_page_state(resumed, 1, restored, out_dir)
    ps = resumed.pages[1]
    assert [e.kind for e in resumed.events if e.page_num == 1].count(UNVERIFIABLE_KIND) == 1
    assert ps.row_binding_unverifiable_tables == [0] and ps.row_binding_contradicted_tables == []
    assert ps.row_binding_mark_only is True
    assert ps.unchecked_table_kind == "unverified"


def test_a_check_that_errors_is_marked_unverifiable(tmp_path):
    """The check's own failure is not a pass: the page is marked, with the error recorded.
    Differential (merge gate, Fable P2): the same failing check, run vs not run."""
    with patch(
        "socr.tables.row_binding_check.check_page_row_binding", side_effect=RuntimeError("boom")
    ):
        _, _, _, on, pdf = _run(tmp_path, CLEAN_ROWS, 0, check_runs=True, out_name="err")
        _, _, _, off, _ = _run(tmp_path, CLEAN_ROWS, 0, check_runs=False, out_name="off", pdf=pdf)
    side_on, side_off = _sidecar(on), _sidecar(off)
    assert next(iter(side_on["row_binding"].values()))["error"] == "check_error:RuntimeError"
    assert "row_binding" not in side_off
    assert side_on["winning_output"]["status"] != side_off["winning_output"]["status"]
    assert side_on["winning_output"]["failure_mode"] == "table_unverified"
    assert side_off["winning_output"]["failure_mode"] != "table_unverified"
    assert _fragment(on).count(UNVERIFIABLE_MARK) == 1
    assert "[table unverified" not in _fragment(off)
    assert len(_events(on, UNVERIFIABLE_KIND)) == 1 and _events(off, UNVERIFIABLE_KIND) == []


@pytest.mark.parametrize("rotation", [0, 90])
def test_an_unverifiable_table_does_not_mark_its_bound_sibling(tmp_path, rotation):
    """Merge gate (Fable P2): when the row-binding check alone made the page unverified, only the
    unverifiable table carries a mark; the table it BOUND ships as it would without the check."""
    first_slip = [*GROUP[:3], ("Net xprts", GROUP[3][1])]
    markdown = md(first_slip) + "\nProse between the tables.\n\n" + md(SECOND_TABLE)
    kw = dict(native=TWO_NATIVE, markdown=markdown)
    _, _, _, on, pdf = _run(tmp_path, None, rotation, check_runs=True, out_name="on", **kw)
    _, _, _, off, _ = _run(
        tmp_path, None, rotation, check_runs=False, out_name="off", pdf=pdf, **kw
    )
    side_on = _sidecar(on)
    assert side_on["row_binding"]["p1-t0"]["outcome"] == "unverifiable"
    assert side_on["row_binding"]["p1-t1"]["outcome"] == "bound"
    frag_on, frag_off = _fragment(on), _fragment(off)
    assert "[table unverified" not in frag_off
    assert frag_on.count("[table unverified") == 1
    assert frag_on.count(UNVERIFIABLE_MARK) == 1
    # the bound sibling's bytes, from the line before it to its end, are the control arm's
    second = md(SECOND_TABLE).rstrip("\n")  # the fragment ends without the final newline
    tail_on = frag_on[frag_on.index("Prose between the tables.") :]
    tail_off = frag_off[frag_off.index("Prose between the tables.") :]
    assert second in tail_on and tail_on == tail_off


def test_a_row_binding_withhold_is_layer_caused():
    """Merge gate (Fable P2): under a suspect layer at resume, a row-binding withhold is re-read like
    the native-contradiction withhold, not replayed as a content terminal."""
    from socr.judge.table_verdict import REASON_ROW_BINDING_CONTRADICTION
    from socr.pipeline import orchestrator as orch

    def meta(reason):
        return {"audit_events": [{"kind": WITHHELD_KIND, "data": {"reason": reason}}]}

    assert orch._sidecar_layer_caused_records(meta(REASON_ROW_BINDING_CONTRADICTION)) == [
        WITHHELD_KIND
    ]
    assert orch._sidecar_layer_caused_records(meta("rung_outage")) == []


def test_stale_row_binding_state_is_cleared_when_the_check_does_not_run(tmp_path):
    """Merge gate (Fable P2): the rollback runs before every guard, so a call that returns early
    (here: a native-engine page) leaves no earlier verdict behind."""
    from socr.core.audit_log import AuditEvent
    from socr.core.document import DocumentHandle
    from socr.core.state import PageState

    pdf = _pdf(tmp_path, 0)
    pipeline = UnifiedPipeline(PipelineConfig(quiet=True))
    state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    ps = PageState(page_num=1)
    ps.is_born_digital = True
    ps.row_binding = {"p1-t0": {"outcome": "unverifiable"}}
    ps.row_binding_unverifiable_tables = [0]
    ps.row_binding_table_count = 1
    ps.row_binding_mark_only = True
    state.pages[1] = ps
    state.events.append(AuditEvent(page_num=1, kind=UNVERIFIABLE_KIND, detail="stale"))
    bo = PageOutput(page_num=1, text=md(CLEAN_ROWS), status=PageStatus.SUCCESS, engine="native")
    assert pipeline._run_row_binding_check(state, 1, ps, bo) is False
    assert ps.row_binding == {}
    assert ps.row_binding_unverifiable_tables == [] and ps.row_binding_table_count == 0
    assert ps.row_binding_mark_only is False
    assert [e for e in state.events if e.kind == UNVERIFIABLE_KIND] == []
