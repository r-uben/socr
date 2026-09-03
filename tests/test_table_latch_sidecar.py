"""P1 prep item 1: sparse sidecar persistence of the table-judge retry latch.

``PageState.table_judge_retry_pending`` (docs/log/2026-09-02_gh359-ladder-
terminals-design.md, "Panel and synthesis"; PR #518's shape) must be
persisted DIFFERENTLY from ``equation_lane_retry_pending``: that field is
always written to the sidecar (present, ``false`` by default -- see the
canonical key set in ``tests/test_p6_disposition_persistence.py``). The
table latch must be SPARSE -- the key is written only when True -- because
plan task t4 forbids adding it to that canonical key set or to
``tests/fixtures/p6/prechange_assemble.json``: any always-present key would
change every default-off sidecar's bytes and violate P6's additive-field
contract (``tests/test_p6_stage_ab_difference.py``).

Contract these tests hold the sidecar to:
  * ``PageState.table_judge_retry_pending: bool = False`` (see also the
    derivation tests in ``tests/test_table_judge_gate.py``).
  * ``_flush_page_sidecar`` omits the key entirely when the latch is False,
    and writes ``"table_judge_retry_pending": true`` when it is True.
  * ``_restore_terminal_page_state`` defaults an absent key to False and
    restores a True latch from a sidecar that carries it.

Mirrors the pattern of ``tests/test_ladder_sidecar.py`` (same pipeline/state
helpers, same ``_flush_page_sidecar`` / ``_restore_terminal_page_state``
surface) rather than driving the gate or the agentic loop.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

from ocr_output_contract import Status
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline


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


class TestSparseSidecarWrite:
    def test_default_latch_omits_the_key_entirely(self, tmp_path: Path) -> None:
        pdf_path = _pdf(tmp_path / "doc")
        pipeline = _make_pipeline()
        state = _make_state(pdf_path)
        ps = state.pages[1]
        ps.best_output = _bo()
        ps.attempts = [ps.best_output]

        out_dir = tmp_path / "out"
        sidecar_path = pipeline._flush_page_sidecar(state, 1, out_dir)
        meta = json.loads(sidecar_path.read_text(encoding="utf-8"))

        assert "table_judge_retry_pending" not in meta, (
            "an unset latch must add NO key -- a present-but-false key would "
            "already break sidecar byte-identity for every flag-off/default page"
        )

    def test_true_latch_is_written_as_true(self, tmp_path: Path) -> None:
        pdf_path = _pdf(tmp_path / "doc")
        pipeline = _make_pipeline()
        state = _make_state(pdf_path)
        ps = state.pages[1]
        ps.best_output = _bo()
        ps.attempts = [ps.best_output]
        ps.table_judge_retry_pending = True

        out_dir = tmp_path / "out"
        sidecar_path = pipeline._flush_page_sidecar(state, 1, out_dir)
        meta = json.loads(sidecar_path.read_text(encoding="utf-8"))

        assert meta.get("table_judge_retry_pending") is True

    def test_default_off_sidecar_is_byte_identical_with_and_without_this_field_existing(
        self, tmp_path: Path
    ) -> None:
        """A page latched False must serialize identically to a page that never
        went through the ladder gate at all -- the sparse encoding's whole
        point is that default-off output cannot move."""
        pdf_path = _pdf(tmp_path / "doc")
        pipeline = _make_pipeline()

        state_a = _make_state(pdf_path)
        ps_a = state_a.pages[1]
        ps_a.best_output = _bo()
        ps_a.attempts = [ps_a.best_output]
        # explicit False, mirrors what the gate leaves behind on a
        # content-only (non-latching) terminal
        ps_a.table_judge_retry_pending = False

        state_b = _make_state(pdf_path)
        ps_b = state_b.pages[1]
        ps_b.best_output = _bo()
        ps_b.attempts = [ps_b.best_output]

        out_a = pipeline._flush_page_sidecar(state_a, 1, tmp_path / "a").read_bytes()
        out_b = pipeline._flush_page_sidecar(state_b, 1, tmp_path / "b").read_bytes()
        assert out_a == out_b


class TestSidecarRestore:
    def test_absent_key_restores_false(self, tmp_path: Path) -> None:
        pdf_path = _pdf(tmp_path / "doc")
        pipeline = _make_pipeline()
        state = _make_state(pdf_path)
        ps = state.pages[1]
        ps.best_output = _bo()
        ps.attempts = [ps.best_output]

        out_dir = tmp_path / "out"
        pipeline._flush_page_sidecar(state, 1, out_dir)

        restored_state = _make_state(pdf_path)
        pipeline._restore_terminal_page_state(restored_state, 1, ps.best_output, out_dir)
        assert restored_state.pages[1].table_judge_retry_pending is False

    def test_true_latch_round_trips_through_restore(self, tmp_path: Path) -> None:
        pdf_path = _pdf(tmp_path / "doc")
        pipeline = _make_pipeline()
        state = _make_state(pdf_path)
        ps = state.pages[1]
        ps.best_output = _bo()
        ps.attempts = [ps.best_output]
        ps.table_judge_retry_pending = True

        out_dir = tmp_path / "out"
        pipeline._flush_page_sidecar(state, 1, out_dir)

        restored_state = _make_state(pdf_path)
        pipeline._restore_terminal_page_state(restored_state, 1, ps.best_output, out_dir)
        assert restored_state.pages[1].table_judge_retry_pending is True, (
            "a page legitimately restored while the rung remains unavailable "
            "must keep carrying the latch forward"
        )

    def test_old_sidecar_missing_the_key_is_compatible(self, tmp_path: Path) -> None:
        """A sidecar written before this feature existed has no such key at
        all (not even absent-by-omission logic tested above -- the raw JSON
        never mentions it). Restore must default to False, not raise."""
        pdf_path = _pdf(tmp_path / "doc")
        pipeline = _make_pipeline()
        state = _make_state(pdf_path)
        ps = state.pages[1]
        ps.best_output = _bo()
        ps.attempts = [ps.best_output]

        out_dir = tmp_path / "out"
        sidecar_path = pipeline._flush_page_sidecar(state, 1, out_dir)
        meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
        meta.pop("table_judge_retry_pending", None)
        sidecar_path.write_text(json.dumps(meta), encoding="utf-8")

        restored_state = _make_state(pdf_path)
        pipeline._restore_terminal_page_state(restored_state, 1, ps.best_output, out_dir)
        assert restored_state.pages[1].table_judge_retry_pending is False


@pytest.mark.parametrize("latch", [False, True])
def test_equation_lane_sidecar_encoding_is_unchanged_by_the_new_field(
    tmp_path: Path, latch: bool
) -> None:
    """Control: adding the table latch must not touch the (always-present,
    non-sparse) equation-lane key's own encoding."""
    pdf_path = _pdf(tmp_path / "doc")
    pipeline = _make_pipeline()
    state = _make_state(pdf_path)
    ps = state.pages[1]
    ps.best_output = _bo()
    ps.attempts = [ps.best_output]
    ps.equation_lane_retry_pending = latch

    out_dir = tmp_path / "out"
    sidecar_path = pipeline._flush_page_sidecar(state, 1, out_dir)
    meta = json.loads(sidecar_path.read_text(encoding="utf-8"))

    assert meta["equation_lane_retry_pending"] is latch


class TestLatchedDocMetadata:
    """Task t5: generalize root-index latch wrapper to _LatchedDocMetadata."""

    def test_delegates_attributes_to_doc_metadata(self) -> None:
        from ocr_output_contract import DocMetadata, Status
        from socr.pipeline.orchestrator import _LatchedDocMetadata

        base_meta = DocMetadata(
            status=Status.COMPLETED,
            checksum="sha256:abcd",
            model="qwen",
            backend="socr",
            processing_time=1.23,
            timestamp="2026-09-03T00:00:00Z",
            output_path="doc/doc.md",
            pages=3,
        )
        latched = _LatchedDocMetadata(base_meta, ("table_judge_retry_pending",))
        assert latched.status == Status.COMPLETED
        assert latched.checksum == "sha256:abcd"
        assert latched.model == "qwen"
        assert latched.backend == "socr"
        assert latched.pages == 3

    @pytest.mark.parametrize(
        "pending_keys,expected_equation,expected_table",
        [
            ((), False, False),
            (("equation_lane_retry_pending",), True, False),
            (("table_judge_retry_pending",), False, True),
            (("equation_lane_retry_pending", "table_judge_retry_pending"), True, True),
        ],
    )
    def test_to_entry_sparse_wire_format(
        self,
        pending_keys: tuple[str, ...],
        expected_equation: bool,
        expected_table: bool,
    ) -> None:
        from ocr_output_contract import DocMetadata, Status
        from socr.pipeline.orchestrator import _LatchedDocMetadata

        base_meta = DocMetadata(
            status=Status.COMPLETED,
            checksum="sha256:1234",
            model="qwen",
            backend="socr",
            processing_time=0.5,
            timestamp="2026-09-03T00:00:00Z",
            output_path="doc/doc.md",
            pages=1,
        )
        latched = _LatchedDocMetadata(base_meta, pending_keys)
        entry = latched.to_entry()

        if expected_equation:
            assert entry.get("equation_lane_retry_pending") is True
        else:
            assert "equation_lane_retry_pending" not in entry

        if expected_table:
            assert entry.get("table_judge_retry_pending") is True
        else:
            assert "table_judge_retry_pending" not in entry


class TestTerminalMetadataRootIndexLatches:
    """Task t5: verify _write_metadata computes latches across pages and records in one save."""

    def test_clean_pages_omit_latches_from_root_entry(self, tmp_path: Path) -> None:
        from ocr_output_contract import RootIndex
        from socr.core.result import DocumentStatus, EngineResult

        pdf_path = _pdf(tmp_path / "src")
        out_dir = tmp_path / "out"
        pipeline = _make_pipeline()
        state = _make_state(pdf_path, page_count=2)
        state.pages[1].table_judge_retry_pending = False
        state.pages[1].equation_lane_retry_pending = False
        state.pages[2].table_judge_retry_pending = False
        state.pages[2].equation_lane_retry_pending = False

        res = EngineResult(
            document_path=pdf_path,
            status=DocumentStatus.SUCCESS,
            engine="qwen",
            processing_time=1.0,
        )
        pipeline._write_metadata(state, res, out_dir, has_text=True)

        idx = RootIndex(out_dir)
        rel_key = list(idx.files.keys())[0]
        entry = idx.files[rel_key]
        assert "table_judge_retry_pending" not in entry
        assert "equation_lane_retry_pending" not in entry

    def test_multipage_mixed_latches_recorded_in_single_root_entry(self, tmp_path: Path) -> None:
        from ocr_output_contract import RootIndex
        from socr.core.result import DocumentStatus, EngineResult

        pdf_path = _pdf(tmp_path / "src")
        out_dir = tmp_path / "out"
        pipeline = _make_pipeline()
        state = _make_state(pdf_path, page_count=2)
        state.pages[1].equation_lane_retry_pending = True
        state.pages[1].table_judge_retry_pending = False
        state.pages[2].equation_lane_retry_pending = False
        state.pages[2].table_judge_retry_pending = True

        res = EngineResult(
            document_path=pdf_path,
            status=DocumentStatus.SUCCESS,
            engine="qwen",
            processing_time=1.0,
        )
        pipeline._write_metadata(state, res, out_dir, has_text=True)

        idx = RootIndex(out_dir)
        rel_key = list(idx.files.keys())[0]
        entry = idx.files[rel_key]
        assert entry.get("equation_lane_retry_pending") is True
        assert entry.get("table_judge_retry_pending") is True


class TestResumeSkippableTableLatch:
    """Task t7: test _resume_skippable with table_judge_retry_blocks."""

    def test_lazy_predicate_evaluation_only_when_latched(self, tmp_path: Path) -> None:
        from ocr_output_contract import DocMetadata, RootIndex, Status
        from socr.pipeline.orchestrator import _LatchedDocMetadata, _resume_skippable

        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = RootIndex(out_dir)

        # 1. Unlatched entry: predicate MUST NOT be called.
        idx.record(
            "unlatched.pdf",
            DocMetadata(
                status=Status.COMPLETED,
                checksum="sha256:1111",
                model="qwen",
                backend="socr",
                processing_time=1.0,
                timestamp="2026-09-03T00:00:00Z",
                output_path="doc/unlatched.md",
                pages=1,
                fingerprint="fp1",
            ),
        )
        (out_dir / "doc").mkdir(parents=True, exist_ok=True)
        (out_dir / "doc" / "unlatched.md").write_text("done")

        calls = []

        def _predicate(rung_kinds=None):
            calls.append(True)
            return True

        res = _resume_skippable(
            idx,
            "unlatched.pdf",
            "sha256:1111",
            "fp1",
            out_dir,
            table_judge_retry_blocks=_predicate,
        )
        assert res is True
        assert calls == []

        # 2. Latched entry: predicate IS called.
        latched_meta = _LatchedDocMetadata(
            DocMetadata(
                status=Status.COMPLETED,
                checksum="sha256:2222",
                model="qwen",
                backend="socr",
                processing_time=1.0,
                timestamp="2026-09-03T00:00:00Z",
                output_path="doc/latched.md",
                pages=1,
                fingerprint="fp2",
            ),
            ("table_judge_retry_pending",),
        )
        idx.record("latched.pdf", latched_meta)
        (out_dir / "doc" / "latched.md").write_text("done")

        res2 = _resume_skippable(
            idx,
            "latched.pdf",
            "sha256:2222",
            "fp2",
            out_dir,
            table_judge_retry_blocks=_predicate,
        )
        assert res2 is False
        assert len(calls) == 1

    @pytest.mark.parametrize(
        "status,reachable,expected_skippable",
        [
            (Status.COMPLETED, True, False),
            (Status.COMPLETED, False, True),
            (Status.PARTIAL, True, False),
            # Cold review round 1, finding 1: a PARTIAL latched entry whose rung
            # is STILL unreachable stays skippable, exactly like the COMPLETED
            # one. The latch says "an outage happened", not "reprocess forever";
            # only reachability-now may refuse the skip.
            (Status.PARTIAL, False, True),
        ],
    )
    def test_root_gate_decisions_for_latched_entry(
        self, tmp_path: Path, status, reachable: bool, expected_skippable: bool
    ) -> None:
        from ocr_output_contract import DocMetadata, RootIndex, Status
        from socr.pipeline.orchestrator import _LatchedDocMetadata, _resume_skippable

        out_dir = tmp_path / f"out_{status.value}_{reachable}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "doc").mkdir(parents=True, exist_ok=True)
        (out_dir / "doc" / "doc.md").write_text("sample output")
        idx = RootIndex(out_dir)

        latched_meta = _LatchedDocMetadata(
            DocMetadata(
                status=status,
                checksum="sha256:3333",
                model="qwen",
                backend="socr",
                processing_time=1.0,
                timestamp="2026-09-03T00:00:00Z",
                output_path="doc/doc.md",
                pages=1,
                fingerprint="fp3",
            ),
            ("table_judge_retry_pending",),
        )
        idx.record("doc.pdf", latched_meta)

        res = _resume_skippable(
            idx,
            "doc.pdf",
            "sha256:3333",
            "fp3",
            out_dir,
            table_judge_retry_blocks=lambda rung_kinds=None: reachable,
        )
        assert res is expected_skippable


class TestLoadTerminalPageTableLatch:
    """Task t7: test _load_terminal_page with table retry latch."""

    def test_mixed_latched_rejected_reprocesses_when_reachable(self, tmp_path: Path) -> None:
        from socr.core.manifest import FailureMode

        pdf_path = _pdf(tmp_path / "src")
        out_dir = tmp_path / "out"
        pipeline = _make_pipeline()
        state = _make_state(pdf_path)
        ps = state.pages[1]
        ps.best_output = _bo()
        ps.attempts = [ps.best_output]
        ps.table_ladder_disposition = FailureMode.TABLE_REJECTED
        ps.table_judge_retry_pending = True

        pipeline._flush_page_sidecar(state, 1, out_dir)
        frag_path = out_dir / "doc" / "pages" / "00001.md"
        frag_path.write_text("page text")

        with patch.object(pipeline, "_table_judge_rung_available_now", return_value=True):
            loaded = pipeline._load_terminal_page(state, 1, out_dir)
            assert loaded is None, "reachable now must return None and reprocess the page"

    def test_mixed_latched_rejected_restores_and_carries_latch_when_unreachable(
        self, tmp_path: Path
    ) -> None:
        from socr.core.manifest import FailureMode

        pdf_path = _pdf(tmp_path / "src")
        out_dir = tmp_path / "out"
        pipeline = _make_pipeline()
        state = _make_state(pdf_path)
        ps = state.pages[1]
        ps.best_output = _bo()
        ps.attempts = [ps.best_output]
        ps.table_ladder_disposition = FailureMode.TABLE_REJECTED
        ps.table_judge_retry_pending = True

        pipeline._flush_page_sidecar(state, 1, out_dir)
        frag_path = out_dir / "doc" / "pages" / "00001.md"
        frag_path.write_text("page text")

        with patch.object(pipeline, "_table_judge_rung_available_now", return_value=False):
            loaded = pipeline._load_terminal_page(state, 1, out_dir)
            assert loaded is not None, "unreachable now must allow D1b restore"
            restored_state = _make_state(pdf_path)
            pipeline._restore_terminal_page_state(restored_state, 1, loaded, out_dir)
            assert restored_state.pages[1].table_judge_retry_pending is True

    def test_content_only_rejected_restores_without_latch(self, tmp_path: Path) -> None:
        from socr.core.manifest import FailureMode

        pdf_path = _pdf(tmp_path / "src")
        out_dir = tmp_path / "out"
        pipeline = _make_pipeline()
        state = _make_state(pdf_path)
        ps = state.pages[1]
        ps.best_output = _bo()
        ps.attempts = [ps.best_output]
        ps.table_ladder_disposition = FailureMode.TABLE_REJECTED
        ps.table_judge_retry_pending = False

        pipeline._flush_page_sidecar(state, 1, out_dir)
        frag_path = out_dir / "doc" / "pages" / "00001.md"
        frag_path.write_text("page text")

        with patch.object(pipeline, "_table_judge_rung_available_now", return_value=True):
            loaded = pipeline._load_terminal_page(state, 1, out_dir)
            assert loaded is not None, "content-only REJECTED restores regardless of reachability"
            restored_state = _make_state(pdf_path)
            pipeline._restore_terminal_page_state(restored_state, 1, loaded, out_dir)
            assert restored_state.pages[1].table_judge_retry_pending is False
