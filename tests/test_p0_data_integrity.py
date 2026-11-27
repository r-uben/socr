"""Hermetic regression guards for P0 silent-data-loss and crash-integrity fixes.

Covers all 7 P0 bugs on ``fix/p0-data-integrity`` at real shipping boundaries:
  1. Cross-document ``_paid_lane_undo`` isolation across ``process()`` / ``process_batch()``.
  2. Resume replay of skipped-page ``TABLE_DISTRUST_KINDS``, ``table_escalation_accepted``,
     and recent PR kinds (with deduplication of analyze-time ``table_structure_failed``).
  3. Chronological resolution in ``build_tables_trust`` (earlier accept never erases later
     distrust on the same page/table).
  4. Fail-closed blind-cell guard when any reader FAIL finding has an empty or non-cell
     ``where`` (e.g. ``"last two rows"``).
  5. ``_phase_assemble`` publication order (provisional until fragments/sidecars/manifest/
     audit finish), atomic ``.md`` writes, fragment-error demotion, and stale
     ``audit_log.json`` cleanup on clean reruns.
  6. ``socr library`` nonzero exit code on skipped ``PARTIAL`` documents.
  7. Refinalizers (``refinalize_unshipped_figures``, ``refinalize_boilerplate``,
     ``refinalize_second_reader``) and allowlisted HTML tag stripping in ``second_reader``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

fitz = pytest.importorskip("fitz")

from socr import cli as socr_cli  # noqa: E402
from socr.core.audit_log import AuditEvent  # noqa: E402
from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.result import (  # noqa: E402
    DocumentStatus,
    EngineResult,
    FailureMode,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState, PageState  # noqa: E402
from socr.core.tables_trust import build_tables_trust  # noqa: E402
from socr.judge.table_cell_guard import BindingEvidence, BlindCellResult  # noqa: E402
from socr.judge.table_ladder import RungResult, TableLadderOutcome, TableLadderResult  # noqa: E402
from socr.judge.table_verdict import (  # noqa: E402
    Finding,
    FindingCode,
    TableJudgeVerdict,
)
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402
from socr.tables.second_reader import Agreement, compare_readings, row_numbers  # noqa: E402


def _prose_pdf(path: Path, body: str = "Federal Reserve economic projection text.") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for row in range(8):
        page.insert_text((72, 100 + row * 22), f"{body} Sentence {row + 1}.", fontsize=10)
    doc.save(str(path))
    doc.close()
    return path


def _hermetic_pipeline(**overrides) -> UnifiedPipeline:
    cfg = PipelineConfig(
        agentic=True,
        quiet=True,
        primary_engine=EngineType.QWEN,
        local_engine=EngineType.QWEN,
        enabled_engines=[EngineType.QWEN],
        native_first=True,
        write_manifest=True,
        judge_backend="heuristic",
        dual_pass_tables=False,
        detect_equations=False,
        save_figures=False,
        describe_figures=False,
        describe_figure_crops=False,
        **overrides,
    )
    pipe = UnifiedPipeline(cfg)
    pipe._available_engines_for_agentic = lambda: []
    pipe._resolve_crop_vlm_model = lambda: None
    pipe._resolve_judge_model = lambda *a, **k: ""
    return pipe


# ---------------------------------------------------------------------------
# 1. Cross-document _paid_lane_undo isolation
# ---------------------------------------------------------------------------


def test_paid_lane_undo_does_not_leak_across_documents(tmp_path: Path) -> None:
    """Doc A's undo entry must never overwrite Doc B, and running Doc B on the same pipeline
    must not clear Doc A's active entry (keyed by ``(id(state), page_num)``)."""
    pipe = _hermetic_pipeline()
    pdf_a = _prose_pdf(tmp_path / "doc_a.pdf", "Document A confidential text.")
    pdf_b = _prose_pdf(tmp_path / "doc_b.pdf", "Document B distinct text.")

    state_a = DocumentState(handle=DocumentHandle.from_path(pdf_a))
    free_a = PageOutput(
        page_num=1, text="DOC_A_FREE_TEXT", status=PageStatus.WARNING, engine="qwen"
    )
    paid_a = PageOutput(
        page_num=1, text="DOC_A_PAID_TEXT", status=PageStatus.SUCCESS, engine="gemini"
    )
    entry_a = {
        "state": state_a,
        "free": free_a,
        "paid": paid_a,
        "events_before": [AuditEvent(page_num=1, kind="table_ladder_withheld", detail="doc A")],
        "trial": [],
        "commit_len": 0,
        "accepted_event": AuditEvent(page_num=1, kind="paid_reading_after_table_withhold"),
        "snapshot": PageState(page_num=1),
        "kept_fields": frozenset(),
    }
    pipe._paid_lane_undo[(id(state_a), 1)] = entry_a

    # Direct cross-state call must refuse Doc A's undo record for Doc B's state,
    # and must leave Doc A's own entry intact for concurrent execution.
    state_b = DocumentState(handle=DocumentHandle.from_path(pdf_b))
    ps_b = state_b.pages[1]
    out_b = PageOutput(page_num=1, text="DOC_B_OWN_TEXT", status=PageStatus.WARNING, engine="qwen")
    ps_b.best_output = out_b
    state_b.events.append(AuditEvent(page_num=1, kind="native_fallback", detail="doc B event"))

    undone = pipe._undo_paid_after_table_withhold(state_b, 1, ps_b, "late table withhold")
    assert undone is None
    assert ps_b.best_output is out_b
    assert ps_b.best_output.text == "DOC_B_OWN_TEXT"
    assert [e.detail for e in state_b.events] == ["doc B event"]
    assert pipe._paid_lane_undo.get((id(state_a), 1)) is entry_a

    # Running process() on Doc B must neither leak Doc A's entry nor wipe Doc A's active entry.
    out_dir = tmp_path / "out"
    res = pipe.process(pdf_b, output_dir=out_dir)
    assert res.status is DocumentStatus.SUCCESS
    assert pipe._paid_lane_undo.get((id(state_a), 1)) is entry_a
    assert "DOC_A_FREE_TEXT" not in res.pages[0].text
    assert "Document B distinct text." in res.pages[0].text

    # Doc A can still undo its own page 1 cleanly after Doc B finished.
    ps_a = state_a.pages[1]
    ps_a.best_output = paid_a
    restored_a = pipe._undo_paid_after_table_withhold(state_a, 1, ps_a, "late table withhold")
    assert restored_a is free_a
    assert ps_a.best_output is free_a
    assert (id(state_a), 1) not in pipe._paid_lane_undo


# ---------------------------------------------------------------------------
# 2. Resume replays TABLE_DISTRUST_KINDS & deduplicates analyze-time duplicates
# ---------------------------------------------------------------------------


def test_resume_replays_distrust_events_and_deduplicates_analyze_table_structure_failed(
    tmp_path: Path,
) -> None:
    """Skipped page-loop distrust events survive resume into tables_trust.json without doubling
    analyze-time ``table_structure_failed``."""
    pdf = _prose_pdf(tmp_path / "paper.pdf")
    out_dir = tmp_path / "out"
    doc_dir = out_dir / "paper"
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True)

    analyze_event = AuditEvent(
        page_num=1,
        kind="table_structure_failed",
        engine="native",
        detail="grid_shape defect",
        data={"defects": ["grid_shape"]},
    )
    model_struct_event = {
        "kind": "table_structure_failed",
        "engine": "qwen",
        "detail": "header_unattributed on model table",
        "data": {"defects": ["header_unattributed"]},
    }
    sidecar = {
        "page_num": 1,
        "terminal": True,
        "audit_passed": True,
        "status": "success",
        "winning_output": {
            "page_num": 1,
            "text": "Table body",
            "status": "success",
            "engine": "qwen",
            "audit_passed": True,
        },
        "audit_events": [
            {
                "kind": "table_value_drift_unadjudicated",
                "engine": "qwen",
                "detail": "value drift on p1",
                "data": {"table_id": "p1-t0"},
            },
            {
                "kind": "table_structure_failed",
                "engine": "native",
                "detail": "grid_shape defect",
                "data": {"defects": ["grid_shape"]},
            },
            model_struct_event,
            dict(model_struct_event),
            {
                "kind": "numbers_disagree_with_second_reader",
                "engine": "second_reader",
                "detail": "8.8 vs -8.8",
                "data": {},
            },
            {
                "kind": "chart_table_arbitration",
                "engine": "native",
                "detail": "chart owns region",
                "data": {},
            },
        ],
    }
    (pages_dir / "00001.json").write_text(json.dumps(sidecar), encoding="utf-8")

    pipe = _hermetic_pipeline()
    pipe._scan_root = pdf.parent
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    state.events.append(analyze_event)

    shipped = PageOutput.from_dict(sidecar["winning_output"])
    pipe._restore_terminal_page_state(state, 1, shipped, out_dir)

    kinds = [e.kind for e in state.events if e.page_num == 1]
    assert "table_value_drift_unadjudicated" in kinds
    assert "numbers_disagree_with_second_reader" in kinds
    assert "chart_table_arbitration" in kinds
    # Native analyze-time table_structure_failed is NOT doubled; BOTH genuine identical
    # model-side table_structure_failed events ARE preserved.
    tsf = [e for e in state.events if e.page_num == 1 and e.kind == "table_structure_failed"]
    assert len(tsf) == 3
    assert sorted(e.engine for e in tsf) == ["native", "qwen", "qwen"]

    # Writing audit_log + tables_trust must keep page 1 untrusted in tables_trust.json.
    pipe._write_audit_log(state, doc_dir)
    trust_data = json.loads((doc_dir / "tables_trust.json").read_text(encoding="utf-8"))
    assert trust_data["untrusted_pages"] == [1]
    assert "table_value_drift_unadjudicated" in trust_data["pages"]["1"]["reasons"]


# ---------------------------------------------------------------------------
# 3. Chronological table resolution in build_tables_trust
# ---------------------------------------------------------------------------


def test_build_tables_trust_resolution_is_chronological() -> None:
    """An earlier accept must NEVER erase a later rejection, withhold, or drift event."""
    # Table-scoped: accept THEN reject on same table -> untrusted.
    trust_after_accept = build_tables_trust(
        "doc.pdf",
        [
            AuditEvent(page_num=2, kind="table_ladder_accepted", data={"table_id": "t0"}),
            AuditEvent(page_num=2, kind="table_ladder_rejected", data={"table_id": "t0"}),
        ],
    )
    assert trust_after_accept.untrusted_pages == [2]
    assert trust_after_accept.pages[2].reasons == ["table_ladder_rejected"]

    # Whole-page: escalation accept THEN later drift -> untrusted.
    trust_after_esc = build_tables_trust(
        "doc.pdf",
        [
            AuditEvent(page_num=3, kind="dualpass_flagged"),
            AuditEvent(page_num=3, kind="table_escalation_accepted"),
            AuditEvent(page_num=3, kind="table_value_drift_unadjudicated"),
        ],
    )
    assert trust_after_esc.untrusted_pages == [3]
    assert trust_after_esc.pages[3].reasons == ["table_value_drift_unadjudicated"]

    # Reject -> accept -> withhold: only the post-accept withhold survives.
    trust_reopened = build_tables_trust(
        "doc.pdf",
        [
            AuditEvent(page_num=4, kind="table_ladder_rejected", data={"table_id": "t0"}),
            AuditEvent(page_num=4, kind="table_ladder_accepted", data={"table_id": "t0"}),
            AuditEvent(page_num=4, kind="table_ladder_withheld", data={"table_id": "t0"}),
        ],
    )
    assert trust_reopened.untrusted_pages == [4]
    assert trust_reopened.pages[4].reasons == ["table_ladder_withheld"]

    # Normal chronological resolution: reject THEN accept still resolves cleanly.
    trust_resolved = build_tables_trust(
        "doc.pdf",
        [
            AuditEvent(page_num=5, kind="table_ladder_rejected", data={"table_id": "t0"}),
            AuditEvent(page_num=5, kind="table_ladder_accepted", data={"table_id": "t0"}),
        ],
    )
    assert trust_resolved.untrusted_pages == []


# ---------------------------------------------------------------------------
# 4. Blind-cell guard fails closed on non-cell FAIL findings
# ---------------------------------------------------------------------------


def test_blind_cell_guard_refuses_clearance_when_any_finding_is_non_cell(tmp_path: Path) -> None:
    """A mixed FAIL verdict (one cell ref + one non-cell finding like 'last two rows') must
    stay UNVERIFIED and never call the blind-cell adjudicator, while a pure cell-only FAIL
    still clears when the adjudicator agrees."""
    table_md = "| Country | Q1 | Q2 |\n| --- | --- | --- |\n| US | 1.5 | 2.5 |"
    witness = SimpleNamespace(table_id="p1-t0", markdown=table_md, crop_path=tmp_path / "crop.png")
    witness.crop_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    adj_calls: list[list[str]] = []

    def agreeing_adjudicator(crop_path: Path, refs):
        adj_calls.append(list(refs))
        return BlindCellResult(rung="adjudicator:test", ok=True, tokens={"R1C2": "1.5"})

    pipe = _hermetic_pipeline()
    state = DocumentState(handle=DocumentHandle(path=tmp_path / "d.pdf", page_count=1))

    # Arm A: mixed cell + structural finding ("last two rows") -> fail closed, no adjudicator call.
    mixed_verdict = TableJudgeVerdict(
        verdict="FAIL",
        confidence="high",
        findings=[
            Finding(code=FindingCode.FABRICATED_VALUE, where="R1C2", detail="digit check"),
            Finding(
                code=FindingCode.STRUCTURE_MERGED, where="last two rows", detail="dropped footer"
            ),
        ],
    )
    mixed_res = TableLadderResult(
        table_id="p1-t0",
        outcome=TableLadderOutcome.REJECTED,
        final_verdict=mixed_verdict,
        rung_results=[RungResult(rung="r1", ok=True, verdict=mixed_verdict)],
    )
    out_mixed = pipe._resolve_table_guard_chain(
        state,
        1,
        witness,
        mixed_res,
        BindingEvidence.ABSTAIN,
        agreeing_adjudicator,
        {},
        {},
        set(),
        set(),
        {},
        {},
    )
    assert out_mixed.outcome is TableLadderOutcome.UNVERIFIED
    assert adj_calls == [], "adjudicator must not be called when non-cell findings exist"

    # Arm A2: mixed cell + empty-where finding -> also fail closed.
    empty_where_verdict = TableJudgeVerdict(
        verdict="FAIL",
        confidence="high",
        findings=[
            Finding(code=FindingCode.FABRICATED_VALUE, where="R1C2", detail="digit check"),
            Finding(code=FindingCode.NOT_A_TABLE, where="", detail="chart merged in"),
        ],
    )
    empty_res = TableLadderResult(
        table_id="p1-t0",
        outcome=TableLadderOutcome.REJECTED,
        final_verdict=empty_where_verdict,
        rung_results=[RungResult(rung="r1", ok=True, verdict=empty_where_verdict)],
    )
    out_empty = pipe._resolve_table_guard_chain(
        state,
        1,
        witness,
        empty_res,
        BindingEvidence.ABSTAIN,
        agreeing_adjudicator,
        {},
        {},
        set(),
        set(),
        {},
        {},
    )
    assert out_empty.outcome is TableLadderOutcome.UNVERIFIED
    assert adj_calls == []

    # Arm A3: FAIL verdict with empty findings alongside another rung's cell ref -> fail closed.
    from socr.judge.table_ladder import TableLadderPending

    fail_no_findings_res = TableLadderResult(
        table_id="p1-t0",
        outcome=TableLadderOutcome.REJECTED,
        final_verdict=TableJudgeVerdict(verdict="FAIL", confidence="high", findings=[]),
        rung_results=[
            RungResult(
                rung="r0",
                ok=True,
                verdict=TableJudgeVerdict(
                    verdict="FAIL",
                    confidence="high",
                    findings=[Finding(code=FindingCode.FABRICATED_VALUE, where="R1C2", detail="d")],
                ),
            ),
            RungResult(
                rung="r1",
                ok=True,
                verdict=TableJudgeVerdict(verdict="FAIL", confidence="high", findings=[]),
            ),
        ],
    )
    out_no_findings = pipe._resolve_table_guard_chain(
        state,
        1,
        witness,
        fail_no_findings_res,
        BindingEvidence.ABSTAIN,
        agreeing_adjudicator,
        {},
        {},
        set(),
        set(),
        {},
        {},
    )
    assert out_no_findings.outcome is TableLadderOutcome.UNVERIFIED
    assert adj_calls == []

    # Arm A4 & A5: TWO_LOW_PASS with empty doubt or unparseable doubt -> fail closed.
    for bad_doubts in (["R1C2", ""], ["R1C2", "footer note"]):
        doubt_verdict = TableJudgeVerdict(verdict="PASS", confidence="low", doubts=bad_doubts)
        doubt_res = TableLadderResult(
            table_id="p1-t0",
            outcome=TableLadderOutcome.UNVERIFIED,
            pending=TableLadderPending.TWO_LOW_PASS,
            final_verdict=doubt_verdict,
            rung_results=[RungResult(rung="r1", ok=True, verdict=doubt_verdict)],
        )
        out_doubt = pipe._resolve_table_guard_chain(
            state,
            1,
            witness,
            doubt_res,
            BindingEvidence.ABSTAIN,
            agreeing_adjudicator,
            {},
            {},
            set(),
            set(),
            {},
            {},
        )
        assert out_doubt.outcome is TableLadderOutcome.UNVERIFIED
        assert adj_calls == []

    # Arm B: same table with ONLY the parseable cell finding -> blind transcription clears it.
    cell_only_verdict = TableJudgeVerdict(
        verdict="FAIL",
        confidence="high",
        findings=[Finding(code=FindingCode.FABRICATED_VALUE, where="R1C2", detail="digit check")],
    )
    cell_only_res = TableLadderResult(
        table_id="p1-t0",
        outcome=TableLadderOutcome.REJECTED,
        final_verdict=cell_only_verdict,
        rung_results=[RungResult(rung="r1", ok=True, verdict=cell_only_verdict)],
    )
    out_cell_only = pipe._resolve_table_guard_chain(
        state,
        1,
        witness,
        cell_only_res,
        BindingEvidence.ABSTAIN,
        agreeing_adjudicator,
        {},
        {},
        set(),
        set(),
        {},
        {},
    )
    assert out_cell_only.outcome is TableLadderOutcome.ACCEPTED
    assert adj_calls == [["R1C2"]]


# ---------------------------------------------------------------------------
# 5. _phase_assemble publication order, fragment failure, atomic .md, stale audit cleanup
# ---------------------------------------------------------------------------


def test_phase_assemble_publishes_final_metadata_last_and_cleans_stale_audit_log(
    tmp_path: Path,
) -> None:
    """No non-provisional metadata write may happen before fragments, sidecars, manifest, and
    audit_log finish; a clean rerun removes any stale ``audit_log.json``."""
    pdf = _prose_pdf(tmp_path / "clean.pdf")
    out_dir = tmp_path / "out"
    doc_dir = out_dir / "clean"
    doc_dir.mkdir(parents=True)
    stale_audit = doc_dir / "audit_log.json"
    stale_audit.write_text(
        json.dumps({"pdf_filename": "clean.pdf", "event_count": 1, "events": [{"kind": "stale"}]}),
        encoding="utf-8",
    )

    pipe = _hermetic_pipeline()
    # Suppress the informational figure_presence_pass event so this rerun has zero audit events
    # and exercises stale audit_log.json removal end to end.
    pipe._agentic_figure_presence_page = lambda *a, **k: None
    events_order: list[tuple[str, object]] = []

    orig_write_meta = pipe._write_metadata
    orig_rewrite = pipe._rewrite_all_fragments
    orig_manifest = pipe._write_manifest
    orig_audit = pipe._write_audit_log

    def spy_meta(state, result, output_dir, has_text, provisional=False, **kw):
        events_order.append(("write_metadata", provisional))
        return orig_write_meta(state, result, output_dir, has_text, provisional=provisional, **kw)

    def spy_rewrite(*a, **kw):
        events_order.append(("rewrite_fragments", None))
        return orig_rewrite(*a, **kw)

    def spy_manifest(*a, **kw):
        events_order.append(("write_manifest", None))
        return orig_manifest(*a, **kw)

    def spy_audit(*a, **kw):
        events_order.append(("write_audit_log", None))
        return orig_audit(*a, **kw)

    pipe._write_metadata = spy_meta
    pipe._rewrite_all_fragments = spy_rewrite
    pipe._write_manifest = spy_manifest
    pipe._write_audit_log = spy_audit

    result = pipe.process(pdf, output_dir=out_dir)
    assert result.status is DocumentStatus.SUCCESS
    assert not stale_audit.exists(), "clean rerun must remove stale audit_log.json"
    assert not list(doc_dir.glob("*.md.tmp")), "atomic markdown save must leave no .tmp file"

    final_writes = [i for i, item in enumerate(events_order) if item == ("write_metadata", False)]
    assert len(final_writes) == 1, f"expected single final metadata publish, got {events_order}"
    final_idx = final_writes[0]
    for step in ("rewrite_fragments", "write_manifest", "write_audit_log"):
        step_indices = [i for i, (name, _) in enumerate(events_order) if name == step]
        assert step_indices and max(step_indices) < final_idx, (
            f"{step} must complete before final non-provisional metadata publish: {events_order}"
        )


def test_fragment_rewrite_failure_keeps_sidecars_and_metadata_provisional(tmp_path: Path) -> None:
    """When ``_rewrite_all_fragments`` fails, assemble must NOT stamp sidecars ``terminal=True``
    or publish a non-provisional ``completed`` root metadata entry."""
    pdf = _prose_pdf(tmp_path / "fragfail.pdf")
    out_dir = tmp_path / "out"
    pipe = _hermetic_pipeline()

    # Force Path.write_text inside _rewrite_all_fragments to fail via _page_fragment_path.
    def bad_frag_path(output_dir, state, page_num):
        raise OSError("disk full during fragment rewrite")

    pipe._page_fragment_path = bad_frag_path

    pipe.process(pdf, output_dir=out_dir)
    doc_dir = out_dir / "fragfail"
    sidecar = json.loads((doc_dir / "pages" / "00001.json").read_text(encoding="utf-8"))
    meta = json.loads((doc_dir / "metadata.json").read_text(encoding="utf-8"))
    root = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))

    assert sidecar["terminal"] is False
    assert meta["status"] == "partial"
    assert meta["fingerprint"].endswith(":pre-figures")
    root_entry = next(iter(root["files"].values()))
    assert root_entry["status"] == "partial"
    assert root_entry["fingerprint"].endswith(":pre-figures")


def test_tables_trust_and_manifest_write_failures_keep_metadata_provisional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Swallowed errors inside ``_write_tables_trust`` or ``_write_manifest`` must keep final
    doc and root metadata provisional instead of publishing ``completed`` over stale/missing
    artifacts."""
    import socr.core.manifest as manifest_mod
    from socr.core.tables_trust import TablesTrust

    # Arm 1: _write_tables_trust fails via TablesTrust.save raising OSError.
    pdf1 = _prose_pdf(tmp_path / "trustfail.pdf")
    out1 = tmp_path / "out1"
    pipe1 = _hermetic_pipeline()

    def boom_trust(*a, **k):
        raise OSError("disk full writing tables_trust.json")

    with monkeypatch.context() as m:
        m.setattr(TablesTrust, "save", boom_trust)
        # Inject one distrust event so _write_tables_trust actually calls save().
        orig_audit = pipe1._write_audit_log

        def inject_and_write(state, doc_dir, records=None):
            state.events.append(AuditEvent(page_num=1, kind="dualpass_flagged", detail="x"))
            return orig_audit(state, doc_dir, records=records)

        pipe1._write_audit_log = inject_and_write
        pipe1.process(pdf1, output_dir=out1)

    meta1 = json.loads((out1 / "trustfail" / "metadata.json").read_text(encoding="utf-8"))
    root1 = json.loads((out1 / "metadata.json").read_text(encoding="utf-8"))
    assert meta1["status"] == "partial"
    assert meta1["fingerprint"].endswith(":pre-figures")
    assert next(iter(root1["files"].values()))["fingerprint"].endswith(":pre-figures")

    # Arm 2: _write_manifest fails inside build_manifest/save.
    pdf2 = _prose_pdf(tmp_path / "manifestfail.pdf")
    out2 = tmp_path / "out2"
    pipe2 = _hermetic_pipeline()

    def boom_manifest(*a, **k):
        raise OSError("disk full writing manifest.json")

    with monkeypatch.context() as m:
        m.setattr(manifest_mod, "build_manifest", boom_manifest)
        pipe2.process(pdf2, output_dir=out2)

    meta2 = json.loads((out2 / "manifestfail" / "metadata.json").read_text(encoding="utf-8"))
    root2 = json.loads((out2 / "metadata.json").read_text(encoding="utf-8"))
    assert meta2["status"] == "partial"
    assert meta2["fingerprint"].endswith(":pre-figures")
    assert next(iter(root2["files"].values()))["fingerprint"].endswith(":pre-figures")


def test_resumed_or_non_clean_document_preserves_existing_audit_log(tmp_path: Path) -> None:
    """When ``audit.events`` is empty on a resumed or non-SUCCESS run, ``_write_audit_log`` must
    NOT unlink an existing ``audit_log.json`` (preventing silent evidence destruction)."""
    pdf = _prose_pdf(tmp_path / "resumed.pdf")
    pipe = _hermetic_pipeline()

    # Case 1: resumed page present -> preserve existing audit_log.json.
    doc_dir_resumed = tmp_path / "doc_resumed"
    doc_dir_resumed.mkdir()
    prior_log = doc_dir_resumed / "audit_log.json"
    prior_log.write_text(
        json.dumps({"pdf_filename": "resumed.pdf", "event_count": 1, "events": [{"kind": "old"}]}),
        encoding="utf-8",
    )
    state_resumed = DocumentState(handle=DocumentHandle.from_path(pdf))
    state_resumed.status = DocumentStatus.SUCCESS
    state_resumed.resumed_pages.add(1)
    assert pipe._write_audit_log(state_resumed, doc_dir_resumed) is True
    assert prior_log.exists(), "resumed doc must never silently delete prior audit_log.json"

    # Case 2: non-clean status -> preserve existing audit_log.json.
    doc_dir_failed = tmp_path / "doc_failed"
    doc_dir_failed.mkdir()
    failed_log = doc_dir_failed / "audit_log.json"
    failed_log.write_text(prior_log.read_text(encoding="utf-8"), encoding="utf-8")
    state_failed = DocumentState(handle=DocumentHandle.from_path(pdf))
    state_failed.status = DocumentStatus.AUDIT_FAILED
    assert pipe._write_audit_log(state_failed, doc_dir_failed) is True
    assert failed_log.exists(), "non-SUCCESS doc must not delete prior audit_log.json"

    # Control: fresh clean SUCCESS with zero events unlinks stale audit_log.json.
    doc_dir_clean = tmp_path / "doc_clean"
    doc_dir_clean.mkdir()
    clean_log = doc_dir_clean / "audit_log.json"
    clean_log.write_text(prior_log.read_text(encoding="utf-8"), encoding="utf-8")
    state_clean = DocumentState(handle=DocumentHandle.from_path(pdf))
    state_clean.status = DocumentStatus.SUCCESS
    assert pipe._write_audit_log(state_clean, doc_dir_clean) is True
    assert not clean_log.exists()


# ---------------------------------------------------------------------------
# 6. socr library exit code on skipped PARTIAL documents
# ---------------------------------------------------------------------------


def test_socr_library_exits_nonzero_on_skipped_partial_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``socr library`` must exit nonzero when a skipped document carries a partial error,
    matching ``socr process``."""
    import yaml
    from test_gh964_library import _fake_doc, _write_cfg

    def _run_cli(arm: str, skip_error: str | None):
        arm_dir = tmp_path / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = _write_cfg(arm_dir)
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        root = Path(cfg["root"])
        _prose_pdf(root / "pdf" / "paper.pdf")
        _fake_doc(root / "text", "paper", status="completed")

        class _StubPipeline:
            def __init__(self, config):
                self.config = config

            def process(self, pdf: Path, out_root: Path, scan_root: Path | None = None):
                _fake_doc(out_root, pdf.stem, status="partial" if skip_error else "completed")
                return EngineResult(
                    document_path=pdf,
                    engine="socr",
                    status=DocumentStatus.SKIPPED,
                    error=skip_error,
                )

        monkeypatch.setattr("socr.pipeline.orchestrator.UnifiedPipeline", _StubPipeline)
        monkeypatch.setattr(socr_cli, "_preflight_gate", lambda *a, **k: None)
        runner = CliRunner()
        return runner.invoke(
            socr_cli.cli,
            ["library", "--config", str(cfg_path), "--rerun", "paper"],
        )

    res_partial = _run_cli("partial", "recorded as partial; pass --reprocess to retry")
    assert res_partial.exit_code != 0, res_partial.output

    res_clean = _run_cli("clean", None)
    assert res_clean.exit_code == 0, res_clean.output


# ---------------------------------------------------------------------------
# 7. Refinalizers + allowlisted HTML tag stripping in second_reader
# ---------------------------------------------------------------------------


def test_refinalize_unshipped_figures_demotes_completed_to_partial_and_recounts_after_sidecar_write(
    tmp_path: Path,
) -> None:
    """``refinalize_unshipped_figures`` must demote ``completed`` -> ``partial`` in both doc
    and root metadata, and recompute table counts AFTER writing updated sidecars."""
    from test_gh1187_unshipped_figure_pages import _CAPTION, _PROSE, _pdf_with_a_figure, _run_dir

    from socr.pipeline import refinalize_unshipped_figures as tool

    pdf = _pdf_with_a_figure(tmp_path / "paper.pdf", draw_figure=False)
    table = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    body = f"{_CAPTION}\n\n{table}\n\n{_PROSE}"
    out, doc_dir = _run_dir(tmp_path, pdf, body)

    # Seed verified table counts matching the initially clean sidecar.
    meta_path = doc_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["status"] = "completed"
    meta["tables"] = {"shipped_text": 1, "verified_text": 1, "withheld": 0}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    root_path = out / "metadata.json"
    root = {
        "files": {
            meta["key"]: {
                "status": "completed",
                "output_path": str(doc_dir / f"{pdf.stem}.md"),
            }
        }
    }
    root_path.write_text(json.dumps(root, indent=2), encoding="utf-8")

    changes = tool.refinalize_document(doc_dir, pdf_dir=tmp_path)
    assert len(changes) == 1

    meta_after = json.loads(meta_path.read_text(encoding="utf-8"))
    root_after = json.loads(root_path.read_text(encoding="utf-8"))
    assert meta_after["status"] == "partial"
    assert root_after["files"][meta["key"]]["status"] == "partial"
    # Sidecar became WARNING before _recount_tables ran, so verified_text must drop to 0.
    assert meta_after["tables"]["verified_text"] == 0

    # Crash recovery: simulate interruption right after sidecars were written (before
    # metadata/root/audit_log finished) and verify rerun reconciles them idempotently.
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    root_path.write_text(json.dumps(root, indent=2), encoding="utf-8")
    audit_path = doc_dir / "audit_log.json"
    audit_path.write_text(
        json.dumps({"pdf_filename": "paper.pdf", "event_count": 0, "counts": {}, "events": []}),
        encoding="utf-8",
    )
    assert tool.refinalize_document(doc_dir, pdf_dir=tmp_path) == []
    meta_recovered = json.loads(meta_path.read_text(encoding="utf-8"))
    root_recovered = json.loads(root_path.read_text(encoding="utf-8"))
    audit_recovered = json.loads(audit_path.read_text(encoding="utf-8"))
    assert meta_recovered["status"] == "partial"
    assert meta_recovered["tables"]["verified_text"] == 0
    assert root_recovered["files"][meta["key"]]["status"] == "partial"
    assert audit_recovered["event_count"] == 1


def test_refinalize_boilerplate_preserves_unrelated_md_and_does_not_recreate_doc_md_on_reocr(
    tmp_path: Path,
) -> None:
    """When a doc has both ``re_ocr`` and ``strip_preamble`` pages plus an unrelated ``notes.md``,
    ``refinalize_boilerplate`` must unlink ONLY canonical ``<stem>.md``, keep ``notes.md``, and
    NOT recreate ``<stem>.md`` (which would defeat the resume gate)."""
    from test_boilerplate_output import JCS_P2_TEXT

    from socr.pipeline.refinalize_boilerplate import refinalize_document

    doc_dir = tmp_path / "paper"
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True)

    preamble = (
        "Here is the transcription of the text in the image, rendered in clean Markdown format:\n\n"
    )
    p1_text = preamble + "Real economic report body paragraph on page one."
    p2_text = JCS_P2_TEXT

    (pages_dir / "00001.md").write_text(p1_text, encoding="utf-8")
    (pages_dir / "00001.json").write_text(
        json.dumps(
            {
                "page_num": 1,
                "terminal": True,
                "status": "success",
                "audit_passed": True,
                "winning_output": {
                    "page_num": 1,
                    "text": p1_text,
                    "engine": "qwen",
                    "status": "success",
                    "audit_passed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (pages_dir / "00002.md").write_text(p2_text, encoding="utf-8")
    (pages_dir / "00002.json").write_text(
        json.dumps(
            {
                "page_num": 2,
                "terminal": True,
                "status": "success",
                "audit_passed": True,
                "winning_output": {
                    "page_num": 2,
                    "text": p2_text,
                    "engine": "qwen",
                    "status": "success",
                    "audit_passed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    canonical_md = doc_dir / "paper.md"
    canonical_md.write_text(f"## Page 1\n\n{p1_text}\n\n## Page 2\n\n{p2_text}", encoding="utf-8")
    notes_md = doc_dir / "notes.md"
    notes_md.write_text("my hand-written notes", encoding="utf-8")
    (doc_dir / "metadata.json").write_text(
        json.dumps({"key": "paper.pdf", "status": "completed", "output_path": str(canonical_md)}),
        encoding="utf-8",
    )

    changes = refinalize_document(doc_dir, apply=True)
    assert {(c.page_num, c.action) for c in changes} == {(1, "strip_preamble"), (2, "re_ocr")}
    assert notes_md.read_text(encoding="utf-8") == "my hand-written notes"
    assert not canonical_md.exists(), "canonical doc markdown must stay removed when re_ocr runs"
    assert (pages_dir / "00001.md").read_text(encoding="utf-8") == (
        "Real economic report body paragraph on page one."
    )
    meta = json.loads((doc_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["status"] == "partial"

    # Second pass after <stem>.md is gone AND metadata.json lacks output_path (leaving
    # notes.md as the sole *.md in doc_dir): notes.md must NEVER be treated as canonical
    # doc markdown or unlinked/overwritten.
    (doc_dir / "metadata.json").write_text(
        json.dumps({"key": "paper.pdf", "status": "partial"}), encoding="utf-8"
    )
    changes_again = refinalize_document(doc_dir, apply=True)
    assert [(c.page_num, c.action) for c in changes_again] == [(2, "re_ocr")]
    assert notes_md.read_text(encoding="utf-8") == "my hand-written notes"

    # Malformed metadata.json must not crash refinalize_document (dry-run or apply),
    # and must still leave unrelated notes.md untouched.
    (doc_dir / "metadata.json").write_text("{broken json", encoding="utf-8")
    dry_changes = refinalize_document(doc_dir, apply=False)
    assert [(c.page_num, c.action) for c in dry_changes] == [(2, "re_ocr")]
    apply_changes = refinalize_document(doc_dir, apply=True)
    assert [(c.page_num, c.action) for c in apply_changes] == [(2, "re_ocr")]
    assert notes_md.read_text(encoding="utf-8") == "my hand-written notes"


def test_refinalize_second_reader_recovers_after_crash_between_fragment_and_sidecar_write(
    tmp_path: Path,
) -> None:
    """If ``refinalize_second_reader`` crashed after writing ``pages/00001.md`` and ``doc.md``
    but before writing ``pages/00001.json``, a rerun must complete cleanly instead of crashing."""
    from test_gh1267_row_binding_check import GROUP
    from test_gh1267_row_binding_pipeline import _run
    from test_gh1409_refinalize_second_reader import MODEL, Reader, _disputing, _doc_dir

    from socr.pipeline import refinalize_second_reader as tool

    _, _, _, out, pdf = _run(tmp_path, GROUP, 0, check_runs=True, out_name="run", providers=True)
    doc_dir = _doc_dir(out)
    sidecar_path = doc_dir / "pages" / "00001.json"
    orig_sidecar_bytes = sidecar_path.read_bytes()

    # First run with disputing reading, then revert ONLY the sidecar to simulate a crash
    # after fragment + doc_md write and before sidecar write.
    disputing_text = _disputing(out)
    res1 = tool.refinalize_document(
        doc_dir,
        pdf.parent,
        apply=True,
        read_page=Reader(disputing_text),
        reader_model=MODEL,
        prompt_sha256="sha256:prompt",
    )
    assert len(res1) == 1 and res1[0].outcome == "disagree"
    sidecar_path.write_bytes(orig_sidecar_bytes)

    # Second run must recover cleanly using the stored reading without raising RuntimeError.
    res2 = tool.refinalize_document(
        doc_dir,
        pdf.parent,
        apply=True,
        read_page=Reader(disputing_text),
        reader_model=MODEL,
        prompt_sha256="sha256:prompt",
    )
    assert len(res2) == 1 and res2[0].outcome == "disagree"
    sidecar_after = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar_after["status"] == "warning"
    assert sidecar_after["failure_mode"] == FailureMode.NUMBERS_DISAGREE_WITH_SECOND_READER.value


def test_second_reader_preserves_inequalities_and_non_html_angle_tokens() -> None:
    """Allowlisted HTML stripping must preserve inequalities, single-letter comparisons,
    escaped literal tags, and non-HTML tokens while cleanly stripping quote-aware attributes."""
    from socr.tables.second_reader import _fold_notation

    assert row_numbers("p < 0.05 and z > 1.96") == ("0.05", "1.96")
    assert row_numbers("x &lt; 10 and y &gt; 2") == ("10", "2")
    assert row_numbers("Table <1> series <PR> grew 3.5 percent") == ("1", "3.5")
    assert row_numbers("<td>1.5</td><td><b>2.5</b></td>") == ("1.5", "2.5")
    # Single-letter allowlisted tag names followed by non-attribute digits must not eat content.
    assert row_numbers("x<a 3 and y>2") == ("3", "2")
    assert row_numbers("n<b 12>") == ("12",)
    # Quoted '>' inside an HTML attribute must not prematurely terminate tag stripping.
    assert row_numbers('<td title="x > 1">3</td>') == ("3",)
    # Strip runs BEFORE html.unescape, so escaped tags survive as literal text.
    assert "<br>" in _fold_notation("literal &lt;br&gt; 4")
    assert "<sup>" in _fold_notation("literal &lt;sup&gt;5&lt;/sup&gt;")
    assert row_numbers("literal &lt;br&gt; 4 &lt;sup&gt;5&lt;/sup&gt;") == ("4", "5")

    # End-to-end comparator: a wrong number inside an inequality must disagree, not vanish.
    shipped = "Threshold p < 0.05 and z > 1.96"
    reader_same = "```markdown\nThreshold p &lt; 0.05 and z &gt; 1.96\n```"
    reader_wrong = "```markdown\nThreshold p &lt; 0.50 and z &gt; 1.96\n```"
    assert compare_readings(shipped, reader_same).outcome is Agreement.AGREE
    assert compare_readings(shipped, reader_wrong).outcome is Agreement.DISAGREE
