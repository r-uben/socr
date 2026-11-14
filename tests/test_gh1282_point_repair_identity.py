"""#1282: a repair is recorded IFF the repaired reading's bytes ship, decided by IDENTITY.

The rule: a ``table_point_repaired`` event and ``row_binding[...]["repairs"]`` evidence exist if and
only if the repaired model reading's own bytes are what ships, and a repaired number never ships
without its record.  Deciding by whether *some* line of the shipped table equals ``row_after`` is
wrong: another reading that read the cell faithfully prints the same line.

Every test is a pair of arms through ``process()`` run on the same PDF and fake model output, and
pins a DIFFERENCE, never an absolute outcome (CLAUDE.md, the no-provider trap):

* ``on`` runs the repair step, ``off`` replaces it with a no-op.  Where the repaired reading does
  not ship, ``on`` and ``off`` must be indistinguishable in every file the run writes.
* both provider states (``_run`` patches ``_available_engines_for_agentic`` and pins the engines).
"""

from __future__ import annotations

import dataclasses
from unittest.mock import patch

import pytest

pytest.importorskip("fitz")

from test_gh1267_row_binding_check import ROTATIONS, md  # noqa: E402
from test_gh1267_row_binding_pipeline import (  # noqa: E402
    _assembled,
    _events,
    _fragment,
    _run,
    _sidecar,
)
from test_gh1282_point_repair_pipeline import (  # noqa: E402
    DISPLACED,
    DROPPED,
    NATIVE,
    PROVIDER_STATES,
    _noop,
    _two_tables,
)
from test_gh1282_point_repair_winner import (  # noqa: E402
    OLDER_MARKER,
    audit_failed_best_output,
)

from socr.core.manifest import SelectionProvenance, finalized_page_record  # noqa: E402
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402
from socr.pipeline.orchestrator import POINT_REPAIR_NOT_SHIPPED, UnifiedPipeline  # noqa: E402


KIND = "table_point_repaired"


def _run_arms(tmp_path, providers, runner, **kw):
    """(on, off): *runner* with the repair step live, then replaced by a no-op."""
    st_on, on, pdf = runner(tmp_path, "on", providers, None, **kw)
    with patch.object(UnifiedPipeline, "_repair_dropped_points", _noop):
        st_off, off, _ = runner(tmp_path, "off", providers, pdf, **kw)
    return (st_on, on), (st_off, off)


def _files(root):
    """Every file but the content-addressed replay cache, which stores the model's reading (the
    repaired one, whether or not it ships) under its own digest."""
    return sorted(
        p.relative_to(root) for p in root.rglob("*") if p.is_file() and "cache" not in p.parts
    )


def _assert_no_trace(on, off):
    """The two runs are the same run: same files, every Markdown file byte-identical, the same
    sidecar (timings aside), the same event kinds -- and no repair anywhere."""
    assert _files(on) == _files(off)
    for rel in _files(on):
        if rel.suffix == ".md":
            assert (on / rel).read_bytes() == (off / rel).read_bytes(), rel
    side_on, side_off = _sidecar(on), _sidecar(off)
    for side in (side_on, side_off):
        side.pop("timings_s", None)
    assert side_on == side_off
    assert _events(on, KIND) == _events(off, KIND) == []
    assert all("repairs" not in r for r in side_on.get("row_binding", {}).values())


# ---------------------------------------------------------------------------
# 1. An older attempt that read the cell faithfully ships
# ---------------------------------------------------------------------------


def _faithful_older(p):
    """An older attempt that read the point correctly: its row is the repaired row, byte for byte."""
    return PageOutput(
        page_num=p.best_output.page_num,
        text=md(NATIVE) + "\n\n" + OLDER_MARKER,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )


def _older_wins(tmp_path, name, providers, pdf):
    holder = {}

    def _winner(p):
        if p.best_output is None:
            return None
        return holder.setdefault("o", _faithful_older(p))

    with (
        audit_failed_best_output(UnifiedPipeline),
        patch("socr.core.manifest._reaches_structure_class_branch", return_value=True),
        patch("socr.core.manifest.structure_class_grid_winner", side_effect=_winner),
    ):
        _, _, st, out, pdf = _run(
            tmp_path, DROPPED, 0, check_runs=True, out_name=name, native=NATIVE,
            providers=providers, pdf=pdf,
        )  # fmt: skip
    return st, out, pdf


@PROVIDER_STATES
def test_an_older_attempt_that_read_the_row_faithfully_is_not_credited_with_the_repair(
    tmp_path, providers
):
    """The older attempt ships and its table already prints ``| Net exports | -.7 | ... |``, the
    line the repair of ``best_output`` produced.  Text equality would record a repair for bytes
    the repair never touched and demote the older table's clean BOUND record to UNVERIFIABLE."""
    (_, on), (_, off) = _run_arms(tmp_path, providers, _older_wins)
    if providers:
        assert OLDER_MARKER in _fragment(on) and "| -.7 |" in _fragment(on)
    _assert_no_trace(on, off)
    if providers:
        rec = _sidecar(on)["row_binding"]["p1-t0"]
        assert rec["outcome"] == "bound"
        assert not _events(on, "table_row_binding_unverifiable")


# ---------------------------------------------------------------------------
# 2. Native bytes never keep a model repair event
# ---------------------------------------------------------------------------


def _native_ships(tmp_path, name, providers, pdf):
    """The page ships the native fallback, and the native text is a pipe table whose row is the
    repaired row byte for byte (a reconstructed native table)."""
    original = UnifiedPipeline._apply_text_coverage

    def _wrapped(self, state, page_num, ps, *a, **k):
        if ps.best_output is not None:
            ps.best_output.audit_passed = False
            ps.needs_ocr_enhancement = True
            ps.native_text = md(NATIVE)
            ps.has_tables = False
        return original(self, state, page_num, ps, *a, **k)

    with patch.object(UnifiedPipeline, "_apply_text_coverage", _wrapped):
        pipeline, _, st, out, pdf = _run(
            tmp_path, DROPPED, 0, check_runs=True, out_name=name, native=NATIVE,
            providers=providers, pdf=pdf,
        )  # fmt: skip
    return st, out, pdf


@PROVIDER_STATES
def test_native_bytes_that_print_the_repaired_row_keep_no_repair_event(tmp_path, providers):
    (st_on, on), (_, off) = _run_arms(tmp_path, providers, _native_ships)
    if providers:
        # the pin is real: the page ships NATIVE_FALLBACK bytes and they print the repaired row
        rec = finalized_page_record(st_on, 1)
        assert rec.selection_provenance is SelectionProvenance.NATIVE_FALLBACK
        assert rec.output.engine == "native" and not rec.output.point_repaired
        assert "| Net exports | -.7 | 1.5 | 2.5 |" in rec.output.text
    _assert_no_trace(on, off)


# ---------------------------------------------------------------------------
# 3. The error path: repaired bytes never ship without their record
# ---------------------------------------------------------------------------


def _check_raises(tmp_path, name, providers, pdf):
    """A step between the repair and the commit raises (the page's flush then lands in its
    ``except Exception: continue``)."""

    def _boom(self, *a, **k):
        raise RuntimeError("boom")

    with patch.object(UnifiedPipeline, "_run_row_binding_check", _boom):
        _, _, st, out, pdf = _run(
            tmp_path, DROPPED, 0, check_runs=True, out_name=name, native=NATIVE,
            providers=providers, pdf=pdf,
        )  # fmt: skip
    return st, out, pdf


@PROVIDER_STATES
def test_a_failure_between_the_repair_and_its_commit_ships_the_model_reading_unrepaired(
    tmp_path, providers
):
    (st_on, on), (_, off) = _run_arms(tmp_path, providers, _check_raises)
    if providers:
        assert "| -7 |" in _fragment(on) and "| -.7 |" not in _fragment(on)
        assert "| -.7 |" not in _assembled(on)
        assert "| -.7 |" not in _sidecar(on)["winning_output"]["text"]
        assert all(not ps.point_repairs for ps in st_on.pages.values())
    assert _events(on, KIND) == []
    if not providers:
        return  # no model reading, no page fragment to compare
    # the failed step is the same failure in both arms: what the repair step changed is nothing
    assert _fragment(on) == _fragment(off)
    assert _assembled(on) == _assembled(off)


# ---------------------------------------------------------------------------
# 4. A dropped repair leaves a named reason, not an empty UNVERIFIABLE record
# ---------------------------------------------------------------------------


@PROVIDER_STATES
@pytest.mark.parametrize("rotation", ROTATIONS)
def test_a_table_demoted_only_for_a_repair_that_was_dropped_says_why(tmp_path, rotation, providers):
    """t0 is CONTRADICTED, so the whole page is withheld and t1's repaired cell does not ship.
    t1's record was written UNVERIFIABLE because of the repair; with the evidence removed it keeps
    a named reason.  The control page (t0 faithful) ships the repair with its evidence."""
    _, _, _, held, pdf = _two_tables(tmp_path, rotation, DISPLACED, "held", providers)
    _, _, _, ctrl, _ = _two_tables(tmp_path, rotation, NATIVE, "ctrl", providers, pdf=pdf)
    if not providers:
        assert _events(held, KIND) == _events(ctrl, KIND) == []
        return
    assert _sidecar(ctrl)["row_binding"]["p1-t1"]["repairs"]
    assert "error" not in _sidecar(ctrl)["row_binding"]["p1-t1"]
    rec = _sidecar(held)["row_binding"]["p1-t1"]
    assert "repairs" not in rec
    assert rec["outcome"] == "unverifiable"
    assert rec["error"] == POINT_REPAIR_NOT_SHIPPED


# ---------------------------------------------------------------------------
# 5. A page whose record changes only at assemble
# ---------------------------------------------------------------------------


def _invalid_at_assemble(tmp_path, name, providers, pdf):
    """The page loop ships the repaired table; the authoritative assemble-time records replace the
    page (a figure description introduced a malformed table: TABLE_EMISSION_INVALID)."""
    import socr.core.manifest as manifest

    original = manifest.finalized_page_records

    def _assemble_records(state, saved_body=None):
        records = original(state, saved_body)
        if saved_body is None:
            return records
        return [
            dataclasses.replace(
                r,
                output=dataclasses.replace(
                    r.output,
                    text=f"[page {r.output.page_num} failed: invalid table]",
                    status=PageStatus.ERROR,
                    failure_mode=FailureMode.TABLE_EMISSION_INVALID,
                    error="table emission invalid: ragged_rows",
                ),
            )
            for r in records
        ]

    with patch.object(manifest, "finalized_page_records", _assemble_records):
        pipeline, _, st, out, pdf = _run(
            tmp_path, DROPPED, 0, check_runs=True, out_name=name, native=NATIVE,
            providers=providers, pdf=pdf,
        )  # fmt: skip
    return st, out, pdf


def _plain(tmp_path, name, providers, pdf):
    _, _, st, out, pdf = _run(
        tmp_path, DROPPED, 0, check_runs=True, out_name=name, native=NATIVE,
        providers=providers, pdf=pdf,
    )  # fmt: skip
    return st, out, pdf


@PROVIDER_STATES
def test_a_page_replaced_at_assemble_loses_the_events_and_evidence_of_its_repair(
    tmp_path, providers
):
    """Control: the same run without the assemble-time replacement records the repair.  Replaced
    at assemble, the page ships no repaired cell, so no event and no evidence survive, in the
    document's audit log, its sidecar, and what a resume would restore."""
    st_ctrl, ctrl, pdf = _plain(tmp_path, "ctrl", providers, None)
    st_inv, inv, _ = _invalid_at_assemble(tmp_path, "inv", providers, pdf)
    if not providers:
        assert _events(ctrl, KIND) == _events(inv, KIND) == []
        return
    # control: the repair ships and is recorded
    assert "| -.7 |" in _fragment(ctrl) and len(_events(ctrl, KIND)) == 1
    assert _sidecar(ctrl)["row_binding"]["p1-t0"]["repairs"]
    # replaced at assemble: the document ships the failure body, and nothing claims a repair
    assert "invalid table" in _fragment(inv) and "-.7" not in _fragment(inv)
    assert "-.7" not in _assembled(inv)
    assert _events(inv, KIND) == []
    assert all("repairs" not in r for r in _sidecar(inv)["row_binding"].values())
    assert all(not ps.point_repairs for ps in st_inv.pages.values())
    assert not [e for e in st_inv.events if e.kind == KIND]
    # the fragment stays the assembled document's page (byte-identity is not disturbed)
    assert _fragment(inv).strip() in _assembled(inv)


@PROVIDER_STATES
def test_a_terminal_resumed_page_restores_exactly_the_committed_events_and_evidence(
    tmp_path, providers
):
    """A resumed page starts from a fresh ``PageState`` and the terminal sidecar.  It gets back
    what the run committed: one event and the evidence when the repair shipped, nothing when the
    assemble reconcile removed it."""
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState, PageState

    if not providers:
        pytest.skip("no model reading, nothing to restore")
    restored_counts = {}
    for name, runner in (("ctrl", _plain), ("inv", _invalid_at_assemble)):
        st, out, pdf = runner(tmp_path, name, providers, None)
        pipeline = _pipeline_like()
        resumed = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
        resumed.pages[1] = PageState(page_num=1)
        restored = PageOutput(
            page_num=1,
            text=_fragment(out),
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        pipeline._restore_terminal_page_state(resumed, 1, restored, out)
        replayed = [e for e in resumed.events if e.page_num == 1 and e.kind == KIND]
        restored_counts[name] = len(replayed)
        on_disk = _events(out, KIND)
        assert [e.data for e in replayed] == [e["data"] for e in on_disk]
        evidence = resumed.pages[1].row_binding.get("p1-t0", {}).get("repairs")
        assert bool(evidence) == bool(replayed)
    assert restored_counts == {"ctrl": 1, "inv": 0}


def _pipeline_like():
    from socr.core.config import EngineType, PipelineConfig

    return UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            agentic=True,
            strict_local=True,
            judge_backend="heuristic",
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            save_figures=False,
            write_manifest=True,
        )
    )


# ---------------------------------------------------------------------------
# 6. #1300: a repair dropped at commit restores the model's text, on every later path
# ---------------------------------------------------------------------------


def _dropped_at_commit(tmp_path, name, providers, pdf, *, drop):
    """The repair step runs; with *drop* the commit finds no repair that ships (the selector's
    verdict is forced), without it the commit is the real one.  Returns the in-loop flushed
    fragment (before assemble rewrites it) as well."""
    flushed: list[str] = []
    real_flush = UnifiedPipeline._flush_page_fragment

    def _spy(self, state, page_num, body, output_dir, *a, **k):
        flushed.append(body)
        return real_flush(self, state, page_num, body, output_dir, *a, **k)

    if drop:
        with (
            patch.object(UnifiedPipeline, "_shipped_repairs", staticmethod(lambda ps, out: {})),
            patch.object(UnifiedPipeline, "_flush_page_fragment", _spy),
        ):
            _, _, st, out, pdf = _run(
                tmp_path, DROPPED, 0, check_runs=True, out_name=name, native=NATIVE,
                providers=providers, pdf=pdf,
            )  # fmt: skip
    else:
        with patch.object(UnifiedPipeline, "_flush_page_fragment", _spy):
            _, _, st, out, pdf = _run(
                tmp_path, DROPPED, 0, check_runs=True, out_name=name, native=NATIVE,
                providers=providers, pdf=pdf,
            )  # fmt: skip
    st.flushed_bodies = flushed
    return st, out, pdf


@PROVIDER_STATES
def test_a_repair_dropped_at_commit_ships_the_models_cell_through_assemble(tmp_path, providers):
    """The drop is FORCED: ``_shipped_repairs`` is patched to keep nothing (``_valid_repairs``
    calls ``_repairs_ship`` directly and still sees the repair), so this pins the restore, not a
    real drop path.  The restore must reach the in-loop flushed fragment, the sidecar and
    assemble alike."""
    st_kept, kept, pdf = _dropped_at_commit(tmp_path, "kept", providers, None, drop=False)
    st_drop, drop, _ = _dropped_at_commit(tmp_path, "drop", providers, pdf, drop=True)
    assert _events(drop, KIND) == []
    assert all(not ps.point_repairs for ps in st_drop.pages.values())
    if not providers:
        assert _events(kept, KIND) == []
        return  # no model reading, nothing repaired
    # control: a kept repair still ships repaired, with its event
    assert "| -.7 |" in _assembled(kept) and _events(kept, KIND)
    # dropped: the model's own cell ships everywhere, assemble included, with no event
    for text in (_fragment(drop), _assembled(drop), _sidecar(drop)["winning_output"]["text"]):
        assert "| -7 |" in text and "| -.7 |" not in text
    # the crash-recovery fragment flushed in the loop (before assemble) carries the model's cell
    assert st_drop.flushed_bodies
    assert all("| -7 |" in b and "| -.7 |" not in b for b in st_drop.flushed_bodies)
    # the persisted row-binding records no repair and names why the table is not bound
    rb = _sidecar(drop)["row_binding"]
    assert rb and all("repairs" not in r for r in rb.values())
    assert all(
        r["outcome"] != "bound" and r.get("error") == "point_repair_not_shipped"
        for r in rb.values()
    )
    assert all(
        not getattr(o, "point_repaired", False)
        for ps in st_drop.pages.values()
        for o in [ps.best_output]
        if o
    )
    # the page flags the mark is drawn from agree with those records (Fable, PR #1304): every table
    # the record calls UNVERIFIABLE is in the page's unverifiable list, which the mark reads
    for ps in st_drop.pages.values():
        for table_id, record in ps.row_binding.items():
            if record.get("error") == "point_repair_not_shipped":
                assert int(table_id.rsplit("-t", 1)[1]) in ps.row_binding_unverifiable_tables
    assert any(ps.row_binding_unverifiable_tables for ps in st_drop.pages.values())


def test_the_restore_marks_a_table_the_check_had_recorded_bound():
    """Fable, PR #1304: the restore sets a record UNVERIFIABLE, and the page flags the mark is drawn
    from must say the same.  The page here is the one the pipeline cannot reach through the check
    (record BOUND, flags empty), so the commit is called directly with a forced drop."""
    from socr.core.state import PageState

    ps = PageState(page_num=1)
    ps.point_repairs = {"p1-t0": [object()]}
    ps.row_binding = {"p1-t0": {"outcome": "bound", "repairs": [{"x": 1}]}}
    bo = PageOutput(page_num=1, text="repaired", status=PageStatus.SUCCESS)
    bo.point_repaired = True
    shipped = PageOutput(page_num=1, text="repaired", status=PageStatus.SUCCESS)
    shipped.point_repaired = True
    pipe = UnifiedPipeline.__new__(UnifiedPipeline)
    with patch.object(UnifiedPipeline, "_shipped_repairs", staticmethod(lambda ps, out: {})):
        pipe._commit_point_repairs(
            type("S", (), {"events": []})(), 1, ps, shipped, bo, "model text"
        )
    assert bo.text == "model text"
    assert ps.row_binding["p1-t0"]["outcome"] == "unverifiable"
    assert ps.row_binding["p1-t0"]["error"] == POINT_REPAIR_NOT_SHIPPED
    assert ps.row_binding_unverifiable_tables == [0]
    assert bo.status is PageStatus.WARNING  # the policy ran: the page is marked, not SUCCESS
