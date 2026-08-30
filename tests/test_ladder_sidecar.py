"""TICKET-D1a (GH-353): table judge ladder sidecar persistence + restore.

``_flush_page_sidecar`` (``pipeline/orchestrator.py``) is the single writer
of ``pages/NNN.json``; ``_restore_terminal_page_state`` is the single reader
that repopulates in-memory ``PageState``/``DocumentState`` on resume. Before
this ticket, the sidecar never carried ``PageState.table_ladder_disposition``
and the restore path never replayed the per-table ``table_ladder_*`` audit
events back into ``state.events`` -- so a page resumed from a terminal
sidecar silently lost its REJECTED/UNVERIFIED verdict (C3's manifest guard
reads ``PageState.table_ladder_disposition``) and dropped out of
``tables_trust.json`` / the assemble-time metadata note (both of which are
derived from ``state.events`` -- see ``_tables_trust_note`` /
``build_tables_trust``).

Hermetic throughout: ``judge_backend="heuristic"`` keeps ``_run_fingerprint``
away from any live Ollama probe (``_resolve_judge_model`` is only consulted
on the VLM-judge branch), and no rung / witness machinery is exercised here
-- ``_flush_page_sidecar`` and ``_restore_terminal_page_state`` are called
directly, never through the gate (``_run_table_judge_gate`` is B1's, not
this ticket's, surface).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import fitz

from socr.core.audit_log import AuditEvent
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.core.tables_trust import build_tables_trust, trust_note
from socr.judge.table_verdict import (
    TABLE_LADDER_ACCEPTED_KIND,
    TABLE_LADDER_REJECTED_KIND,
    TABLE_LADDER_UNVERIFIED_KIND,
)
from socr.pipeline.orchestrator import UnifiedPipeline

# ---------------------------------------------------------------------------
# Fixtures / helpers (mirrors tests/test_table_judge_gate.py's pattern)
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


def _pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    doc.new_page()
    pdf_path = tmp_path / name
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _bo(text: str = "| a | b |\n| --- | --- |\n| 1 | 2 |\n", engine: str = "qwen") -> PageOutput:
    return PageOutput(
        page_num=1,
        text=text,
        status=PageStatus.SUCCESS,
        engine=engine,
        audit_passed=True,
    )


def _sidecar_path(pipeline: UnifiedPipeline, state: DocumentState, output_dir: Path) -> Path:
    from ocr_output_contract import doc_dir_for, relative_key

    scan_root = pipeline._scan_root or state.handle.path.parent
    doc_dir = doc_dir_for(output_dir, relative_key(state.handle.path, scan_root))
    return doc_dir / "pages" / "00001.json"


def _seed_ladder_page(state: DocumentState, disposition: FailureMode, engine: str = "qwen") -> None:
    """Mirror what ``_run_table_judge_gate`` (B1) would have set for a
    two-table page: one table rejected, one accepted -- any REJECTED wins
    at the page reduction (A4's rule)."""
    ps = state.pages[1]
    ps.best_output = _bo(engine=engine)
    ps.attempts = [ps.best_output]
    ps.table_ladder_disposition = disposition
    state.events.append(
        AuditEvent(
            page_num=1,
            kind=TABLE_LADDER_ACCEPTED_KIND,
            engine=engine,
            detail="table p1-t1 accepted by the judge ladder",
            data={"table_id": "p1-t1"},
        )
    )
    kind = (
        TABLE_LADDER_REJECTED_KIND
        if disposition is FailureMode.TABLE_REJECTED
        else TABLE_LADDER_UNVERIFIED_KIND
    )
    state.events.append(
        AuditEvent(
            page_num=1,
            kind=kind,
            engine=engine,
            detail="table p1-t2 demoted by the judge ladder",
            data={"table_id": "p1-t2"},
        )
    )
    # An unrelated, non-ladder event on the SAME page: must never be picked
    # up by the ladder-scoped restore below (only TABLE_LADDER_EVENT_KINDS
    # members round-trip through this path).
    state.events.append(
        AuditEvent(page_num=1, kind="native_fallback", engine="native", detail="unrelated")
    )


# ---------------------------------------------------------------------------
# Persist: _flush_page_sidecar carries the disposition, deterministically.
# ---------------------------------------------------------------------------


class TestSidecarPersist:
    def test_disposition_persisted_as_bare_value(self, tmp_path: Path) -> None:
        pdf_path = _pdf(tmp_path / "doc")
        pipeline = _make_pipeline()
        state = _make_state(pdf_path)
        _seed_ladder_page(state, FailureMode.TABLE_REJECTED)

        out_dir = tmp_path / "out"
        sidecar_path = pipeline._flush_page_sidecar(state, 1, out_dir)
        meta = json.loads(sidecar_path.read_text(encoding="utf-8"))

        assert meta["table_ladder_disposition"] == FailureMode.TABLE_REJECTED.value
        kinds = {ev["kind"] for ev in meta["audit_events"]}
        assert TABLE_LADDER_ACCEPTED_KIND in kinds
        assert TABLE_LADDER_REJECTED_KIND in kinds
        assert "native_fallback" in kinds  # unrelated events still logged verbatim

    def test_no_disposition_persists_as_null(self, tmp_path: Path) -> None:
        """Flag off / no ladder record: the field is present but null -- the
        `test_no_disposition` control for the restore round-trip below."""
        pdf_path = _pdf(tmp_path / "doc")
        pipeline = _make_pipeline()
        state = _make_state(pdf_path)
        ps = state.pages[1]
        ps.best_output = _bo()
        ps.attempts = [ps.best_output]

        out_dir = tmp_path / "out"
        sidecar_path = pipeline._flush_page_sidecar(state, 1, out_dir)
        meta = json.loads(sidecar_path.read_text(encoding="utf-8"))

        assert meta["table_ladder_disposition"] is None
        assert meta["audit_events"] == []

    def test_flush_is_byte_stable_across_repeated_calls(self, tmp_path: Path) -> None:
        """Two consecutive flushes of an UNCHANGED state produce byte-identical
        sidecars -- guards against nondeterministic serialisation (dict/set
        ordering) of the new field."""
        pdf_path = _pdf(tmp_path / "doc")
        pipeline = _make_pipeline()
        state = _make_state(pdf_path)
        _seed_ladder_page(state, FailureMode.TABLE_UNVERIFIED)

        out_dir = tmp_path / "out"
        first = pipeline._flush_page_sidecar(state, 1, out_dir).read_bytes()
        second = pipeline._flush_page_sidecar(state, 1, out_dir).read_bytes()

        assert first == second


# ---------------------------------------------------------------------------
# Restore: _restore_terminal_page_state rebuilds disposition + events, never
# re-judges.
# ---------------------------------------------------------------------------


class TestSidecarRestore:
    def _flush_original(
        self, tmp_path: Path, disposition: FailureMode
    ) -> tuple[UnifiedPipeline, DocumentState, Path]:
        pdf_path = _pdf(tmp_path / "doc")
        pipeline = _make_pipeline()
        state = _make_state(pdf_path)
        _seed_ladder_page(state, disposition)
        out_dir = tmp_path / "out"
        pipeline._flush_page_sidecar(state, 1, out_dir)
        return pipeline, state, out_dir

    def test_restore_reproduces_disposition_and_ladder_events(self, tmp_path: Path) -> None:
        pipeline, original_state, out_dir = self._flush_original(
            tmp_path, FailureMode.TABLE_REJECTED
        )

        resumed_pipeline = _make_pipeline()
        resumed_state = _make_state(original_state.handle.path)
        page_out = _bo()

        with patch(
            "socr.tables.witness.prepare_table_witnesses",
            side_effect=AssertionError("restore must never re-judge"),
        ):
            resumed_pipeline._restore_terminal_page_state(resumed_state, 1, page_out, out_dir)

        rps = resumed_state.pages[1]
        assert rps.table_ladder_disposition == FailureMode.TABLE_REJECTED

        restored_ladder_events = [
            e
            for e in resumed_state.events
            if e.kind in (TABLE_LADDER_ACCEPTED_KIND, TABLE_LADDER_REJECTED_KIND)
        ]
        assert len(restored_ladder_events) == 2
        table_ids = {e.data.get("table_id") for e in restored_ladder_events}
        assert table_ids == {"p1-t1", "p1-t2"}
        # The unrelated event from the ORIGINAL run is not replayed by this
        # ladder-scoped restore -- other tickets own their own resume fields.
        assert not any(e.kind == "native_fallback" for e in resumed_state.events)

    def test_restore_reproduces_tables_trust_and_note(self, tmp_path: Path) -> None:
        pipeline, original_state, out_dir = self._flush_original(
            tmp_path, FailureMode.TABLE_UNVERIFIED
        )
        original_trust = build_tables_trust(
            original_state.handle.filename, original_state.events
        ).to_dict()
        original_note = trust_note(
            build_tables_trust(original_state.handle.filename, original_state.events)
        )

        resumed_pipeline = _make_pipeline()
        resumed_state = _make_state(original_state.handle.path)
        resumed_pipeline._restore_terminal_page_state(resumed_state, 1, _bo(), out_dir)

        restored_trust = build_tables_trust(
            resumed_state.handle.filename, resumed_state.events
        ).to_dict()
        restored_note = trust_note(
            build_tables_trust(resumed_state.handle.filename, resumed_state.events)
        )

        assert restored_trust == original_trust
        assert restored_note == original_note
        assert restored_note is not None

    def test_restore_reproduces_metadata_note(self, tmp_path: Path) -> None:
        pipeline, original_state, out_dir = self._flush_original(
            tmp_path, FailureMode.TABLE_REJECTED
        )
        original_note = UnifiedPipeline._table_judge_ladder_note(original_state)

        resumed_pipeline = _make_pipeline()
        resumed_state = _make_state(original_state.handle.path)
        resumed_pipeline._restore_terminal_page_state(resumed_state, 1, _bo(), out_dir)

        restored_note = UnifiedPipeline._table_judge_ladder_note(resumed_state)
        assert restored_note == original_note
        assert "table_rejected" in restored_note

    def test_no_ladder_record_restores_none_and_no_events(self, tmp_path: Path) -> None:
        """Flag off / no ladder record: restore is byte-identical to today --
        disposition stays None and no ladder events are appended."""
        pdf_path = _pdf(tmp_path / "doc")
        pipeline = _make_pipeline()
        state = _make_state(pdf_path)
        ps = state.pages[1]
        ps.best_output = _bo()
        ps.attempts = [ps.best_output]
        out_dir = tmp_path / "out"
        pipeline._flush_page_sidecar(state, 1, out_dir)

        resumed_pipeline = _make_pipeline()
        resumed_state = _make_state(state.handle.path)
        resumed_pipeline._restore_terminal_page_state(resumed_state, 1, _bo(), out_dir)

        assert resumed_state.pages[1].table_ladder_disposition is None
        assert resumed_state.events == []

    def test_restore_never_calls_witness_or_rung_machinery(self, tmp_path: Path) -> None:
        """Sentinel: restoring a REJECTED page must not re-judge anything --
        no crop render, no rung transport, no ladder state machine call."""
        pipeline, original_state, out_dir = self._flush_original(
            tmp_path, FailureMode.TABLE_REJECTED
        )

        resumed_pipeline = _make_pipeline()
        resumed_state = _make_state(original_state.handle.path)

        with (
            patch(
                "socr.tables.witness.prepare_table_witnesses",
                side_effect=AssertionError("must not prepare witnesses on restore"),
            ),
            patch(
                "socr.judge.table_ladder.run_table_ladder",
                side_effect=AssertionError("must not run the ladder on restore"),
            ),
            patch.object(
                resumed_pipeline,
                "_build_table_judge_rungs",
                side_effect=AssertionError("must not build rungs on restore"),
            ),
        ):
            resumed_pipeline._restore_terminal_page_state(resumed_state, 1, _bo(), out_dir)

        assert resumed_state.pages[1].table_ladder_disposition == FailureMode.TABLE_REJECTED

    def test_restore_survives_garbage_disposition_value(self, tmp_path: Path) -> None:
        """A hand-corrupted / future-version sidecar with an unknown
        disposition string must not raise -- the flag-restore block is
        non-fatal by contract (body text is already correct)."""
        pdf_path = _pdf(tmp_path / "doc")
        pipeline = _make_pipeline()
        state = _make_state(pdf_path)
        _seed_ladder_page(state, FailureMode.TABLE_REJECTED)
        out_dir = tmp_path / "out"
        sidecar_path = pipeline._flush_page_sidecar(state, 1, out_dir)

        meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
        meta["table_ladder_disposition"] = "not_a_real_failure_mode"
        sidecar_path.write_text(json.dumps(meta), encoding="utf-8")

        resumed_pipeline = _make_pipeline()
        resumed_state = _make_state(pdf_path)
        # Must not raise.
        resumed_pipeline._restore_terminal_page_state(resumed_state, 1, _bo(), out_dir)
        # Non-fatal failure: the disposition and any later flags in the same
        # try-block are left at their defaults rather than half-applied.
        assert resumed_state.pages[1].table_ladder_disposition is None
