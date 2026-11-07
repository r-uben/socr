"""#1282 at ``process()``: the point-repair step's effect on a shipped page, as DIFFERENCES.

Every pair of arms runs the same pipeline on the same synthetic PDF with the same fake model
output; the control arm replaces ``UnifiedPipeline._repair_dropped_points`` with a no-op, so the
only thing that changes between arms is whether the step runs.  No absolute status is pinned:
provider-dependent machinery may legitimately move it (CLAUDE.md, the no-provider trap).  The
helper ``_run`` pins ``primary_engine``, ``local_engine`` and ``enabled_engines`` and patches
``_available_engines_for_agentic``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fitz")

from test_gh1267_row_binding_check import build_page, md  # noqa: E402
from test_gh1267_row_binding_pipeline import (  # noqa: E402
    _assembled,
    _events,
    _fragment,
    _run,
    _sidecar,
)

from socr.core.born_digital import BornDigitalDetector  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

KIND = "table_point_repaired"
NATIVE = [
    ("Real GDP", ["1.1", "2.2", "3.3"]),
    ("Net exports", ["-.7", "1.5", "2.5"]),
    ("Exports", ["2.1", "2.2", "2.3"]),
    ("Imports", ["3.1", "3.2", "3.3"]),
]
# the model dropped the point of -.7; the rest is faithful
DROPPED = [NATIVE[0], ("Net exports", ["-7", "1.5", "2.5"]), NATIVE[2], NATIVE[3]]
# the same page plus a label slip, so the page stays marked in BOTH arms
SLIPPED = [*DROPPED[:3], ("Imxorts", NATIVE[3][1])]
# the model displaced the values one label down AND dropped a point: CONTRADICTED
DISPLACED = [
    NATIVE[0],
    ("Net exports", []),
    ("Exports", ["-7", "1.5", "2.5"]),
    ("", NATIVE[2][1]),
    NATIVE[3],
]


def _noop(self, state, page_num, ps, bo):
    ps.point_repairs = {}
    return 0


def _pair(tmp_path, rows, rotation, **kw):
    _, _, st_on, on, pdf = _run(
        tmp_path, rows, rotation, check_runs=True, out_name="on", native=NATIVE, **kw
    )
    with patch.object(UnifiedPipeline, "_repair_dropped_points", _noop):
        _, _, st_off, off, _ = _run(
            tmp_path, rows, rotation, check_runs=True, out_name="off", pdf=pdf, native=NATIVE, **kw
        )
    return (on, st_on), (off, st_off)


def _without_text_layer_marks(text: str) -> str:
    """*text* without the paragraphs of the ``text_layer_*`` page marks."""
    blocks = text.split("\n\n")
    return "\n\n".join(b for b in blocks if "text_layer_" not in b)


@pytest.mark.parametrize("rotation", [0, 90])
def test_the_step_changes_exactly_the_repaired_cell_and_adds_its_event(tmp_path, rotation):
    (on, st_on), (off, st_off) = _pair(tmp_path, SLIPPED, rotation)
    frag_on, frag_off = _fragment(on), _fragment(off)
    assert "| -7 |" in frag_off and "| -7 |" not in frag_on
    assert "| -.7 |" in frag_on and "| -.7 |" not in frag_off
    # the difference is that cell and the two text-layer marks it caused, nothing else: the page
    # stays marked for its other (slipped) row in BOTH arms
    assert _without_text_layer_marks(frag_off).replace("| -7 |", "| -.7 |") == frag_on
    assert "text_layer_numbers=FAIL" in frag_off and "text_layer_numbers" not in frag_on
    assert "row binding not verified" in frag_on and "row binding not verified" in frag_off
    assert _without_text_layer_marks(_assembled(off)).replace("| -7 |", "| -.7 |") == _assembled(on)
    # the sidecar's winning output is the repaired text, so a resume replays the repaired bytes
    win_on = _sidecar(on)["winning_output"]["text"]
    win_off = _sidecar(off)["winning_output"]["text"]
    assert _without_text_layer_marks(win_off).replace("| -7 |", "| -.7 |") == win_on
    # the event: only in the enabled arm, with the evidence
    assert _events(off, KIND) == []
    (event,) = _events(on, KIND)
    assert event["data"]["table_id"] == "p1-t0"
    (row,) = event["data"]["rows"]
    assert (row["label"], row["old"], row["new"]) == ("Net exports", "-7", "-.7")
    # the sidecar's row-binding record carries the same evidence
    assert _sidecar(on)["row_binding"]["p1-t0"]["repairs"] == event["data"]["rows"]
    assert "repairs" not in _sidecar(off)["row_binding"]["p1-t0"]

    # and the repaired row is BOUND in the enabled arm, UNVERIFIABLE in the control
    def row_verdicts(out):
        rec = _sidecar(out)["row_binding"]["p1-t0"]
        return {r["shipped_label"]: r["verdict"] for r in rec["rows"]}

    assert "Net exports" in row_verdicts(off) and "Net exports" not in row_verdicts(on)


@pytest.mark.parametrize("rotation", [0, 90])
def test_a_clean_page_is_untouched_by_the_step(tmp_path, rotation):
    (on, _), (off, _) = _pair(tmp_path, NATIVE, rotation)
    assert _fragment(on) == _fragment(off)
    assert _assembled(on) == _assembled(off)
    assert _events(on, KIND) == []
    assert "repairs" not in _sidecar(on).get("row_binding", {}).get("p1-t0", {})


@pytest.mark.parametrize("rotation", [0, 90])
def test_a_contradicted_table_is_withheld_with_its_text_unrepaired(tmp_path, rotation):
    """Owner ruling: nothing derived from the text layer reaches the re-read's input."""
    (on, _), (off, _) = _pair(tmp_path, DISPLACED, rotation)
    assert _events(on, KIND) == []
    rec_on = _sidecar(on)["row_binding"]["p1-t0"]
    rec_off = _sidecar(off)["row_binding"]["p1-t0"]
    assert rec_on["outcome"] == rec_off["outcome"] == "contradicted"
    assert rec_on["withheld_markdown"] == rec_off["withheld_markdown"]
    assert "| -7 |" in rec_on["withheld_markdown"] and "-.7" not in rec_on["withheld_markdown"]
    assert "repairs" not in rec_on
    assert _fragment(on) == _fragment(off)


@pytest.mark.parametrize("rotation", [0, 90])
def test_a_second_run_reaches_the_page_ledger_and_reproduces_the_repaired_page(tmp_path, rotation):
    """The document-level gate would skip the whole document and the second run would never
    look at the page (the first version of this test was vacuous that way).  The root index is
    removed, so the page loop runs and the per-page ledger (`_load_terminal_page`) is consulted.

    A repaired page keeps its row-binding mark (no repair lifts one), so it is not SUCCESS and the
    ledger sends it back to be read again: the second run repairs it again.  What is pinned: the
    ledger was consulted, and the repair is deterministic and not doubled (same bytes, one event)."""
    from ocr_output_contract import RootIndex

    kw = dict(native=NATIVE)
    _, _, _, out, pdf = _run(tmp_path, DROPPED, rotation, check_runs=True, out_name="out", **kw)
    frag, doc, win = _fragment(out), _assembled(out), _sidecar(out)["winning_output"]["text"]
    ev = _events(out, KIND)
    assert len(ev) == 1 and "| -.7 |" in frag
    Path(RootIndex(out).path).unlink()
    ledger = {"calls": 0, "loaded": []}
    original = UnifiedPipeline._load_terminal_page

    def _spy(self, *a, **k):
        ledger["calls"] += 1
        got = original(self, *a, **k)
        ledger["loaded"].append(got is not None)
        return got

    with patch.object(UnifiedPipeline, "_load_terminal_page", _spy):
        _run(tmp_path, DROPPED, rotation, check_runs=True, out_name="out", pdf=pdf, **kw)
    assert ledger["calls"] > 0  # the second run really consulted the per-page ledger
    assert not any(ledger["loaded"])  # a marked page is read again, never skip-and-kept
    assert _fragment(out) == frag
    assert _assembled(out) == doc
    assert _sidecar(out)["winning_output"]["text"] == win
    assert _events(out, KIND) == ev


@pytest.mark.parametrize("rotation", [0, 90])
def test_the_page_level_restore_replays_the_event_and_the_evidence(tmp_path, rotation):
    """`_restore_terminal_page_state` (what a resumed page goes through) brings back the
    `table_point_repaired` event exactly once, and the repair evidence in the row-binding record."""
    from socr.core.document import DocumentHandle
    from socr.core.result import PageOutput, PageStatus
    from socr.core.state import DocumentState

    pipeline, _, state, out_dir, pdf = _run(
        tmp_path, DROPPED, rotation, check_runs=True, out_name="out", native=NATIVE
    )
    fragment = _fragment(out_dir)
    assert pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True) is not None
    resumed = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    resumed.pages[1] = state.pages[1]
    restored = PageOutput(
        page_num=1, text=fragment, status=PageStatus.SUCCESS, engine="qwen", audit_passed=True
    )
    pipeline._restore_terminal_page_state(resumed, 1, restored, out_dir)
    replayed = [e for e in resumed.events if e.page_num == 1 and e.kind == KIND]
    assert len(replayed) == 1
    assert replayed[0].data == _events(out_dir, KIND)[0]["data"]
    assert resumed.pages[1].row_binding["p1-t0"]["repairs"] == replayed[0].data["rows"]


@pytest.mark.parametrize("rotation", [0, 90])
def test_a_repair_lifts_no_mark_even_when_every_row_now_binds(tmp_path, rotation):
    """Ruling 2: the ladder judged the unrepaired bytes, so the shipped text is not the judged
    text.  DROPPED binds completely after the repair; the page keeps its row-binding mark and status."""
    (on, st_on), (off, st_off) = _pair(tmp_path, DROPPED, rotation)
    assert "| -.7 |" in _fragment(on) and "| -7 |" in _fragment(off)
    for out in (on, off):
        assert "row binding not verified" in _fragment(out)
        assert len(_events(out, "table_row_binding_unverifiable")) == 1
    side_on, side_off = _sidecar(on), _sidecar(off)
    for key in ("status", "failure_mode", "table_ladder_disposition"):
        assert side_on[key] == side_off[key], key
    rec = side_on["row_binding"]["p1-t0"]
    assert rec["outcome"] == side_off["row_binding"]["p1-t0"]["outcome"] == "unverifiable"
    assert (
        rec["rows"] == [] and rec["repairs"]
    )  # every row binds; the record still says unverifiable
    assert side_on["row_binding_unverifiable_tables"] == [0]


@pytest.mark.parametrize("rotation", [0, 90])
def test_repair_evidence_follows_its_table_when_an_earlier_table_is_withheld(tmp_path, rotation):
    """Two tables: t0 is CONTRADICTED (withheld, its reading untouched), t1 carries a dropped point.
    The evidence is attached to p1-t1, by block, and t0's withheld markdown is the model's own."""
    t1_native = [("Wages", ["5.1", "-.2", "5.3"]), ("Prices", ["6.1", "6.2", "6.3"])]
    t1_shipped = [("Wages", ["5.1", "-2", "5.3"]), ("Prices", ["6.1", "6.2", "6.3"])]
    markdown = md(DISPLACED) + "\nProse between the tables.\n\n" + md(t1_shipped)
    native = [*NATIVE, *t1_native]
    _, _, _, on, pdf = _run(
        tmp_path, None, rotation, check_runs=True, out_name="on", native=native, markdown=markdown
    )
    side = _sidecar(on)
    t0, t1 = side["row_binding"]["p1-t0"], side["row_binding"]["p1-t1"]
    assert t0["outcome"] == "contradicted" and "repairs" not in t0
    assert t0["withheld_markdown"] == md(DISPLACED).rstrip("\n")
    assert [(r["old"], r["new"]) for r in t1["repairs"]] == [("-2", "-.2")]
    assert [e["data"]["table_id"] for e in _events(on, KIND)] == ["p1-t1"]


@pytest.mark.parametrize("rotation", [0, 90])
def test_a_page_whose_layer_may_not_judge_gets_no_repair_and_no_event(tmp_path, rotation):
    """The check's own guard: a layer refused for more than spacing is not evidence."""
    (on, _), (off, _) = _pair(
        tmp_path, SLIPPED, rotation, classifier_refuses=True, spacing_alone=False
    )
    assert _fragment(on) == _fragment(off)
    assert _events(on, KIND) == []


@pytest.mark.parametrize("rotation", [0, 90])
def test_a_suspect_prior_ocr_layer_gets_no_repair_and_no_event(tmp_path, rotation):
    from socr.core import state as state_mod

    with (
        patch.object(state_mod, "layer_is_admissible_evidence", lambda ps: False),
        patch("socr.pipeline.orchestrator.layer_is_admissible_evidence", lambda ps: False),
    ):
        _, _, _, out, _ = _run(
            tmp_path, SLIPPED, rotation, check_runs=True, out_name="out", native=NATIVE
        )
    assert _events(out, KIND) == []
    assert "| -7 |" in _fragment(out)


def test_the_repair_and_the_check_share_one_guard():
    """Drift guard: both steps call the one predicate (the repair must never touch a page the
    check would not judge)."""
    import inspect

    check = inspect.getsource(UnifiedPipeline._run_row_binding_check)
    repair = inspect.getsource(UnifiedPipeline._repair_dropped_points)
    assert check.count("self._row_binding_applies(") == 2
    assert repair.count("self._row_binding_applies(") == 2
    assert (
        "layer_is_admissible_evidence" not in check and "layer_is_admissible_evidence" not in repair
    )
