"""GH-819: three ``_agentic_native_page`` audit kinds vanish on resume.

``native_encoding_hygiene_suspect`` (#136), ``native_unrecovered_symbol_glyphs``
(#217) and ``possible_table_structure_not_reconstructed`` (GH-64) are all
emitted from ``_agentic_native_page``, which does NOT run for a page a resumed
run skips as terminal (``_load_terminal_page`` short-circuits before the
``elif is_native:`` branch that calls it -- see ``_phase_agentic``). Before
this ticket none of the three were in ``resume_restore_kinds()``, so the
sidecar kept the record and a resumed run's ``audit_log.json`` / CLI line
silently lost it -- the sidecar and the run report disagree, and the run
report is what an operator reads.

The contrast that must NOT be broken: ``orphan_word_dropped`` is emitted by
``_phase_analyze``, which runs unconditionally on EVERY run (before the
per-page resume-skip loop, see ``process()`` around ``_phase_analyze`` then
``_phase_agentic``) -- so it is deliberately absent from the allowlist, and
adding it would double it (once replayed from the sidecar, once re-emitted
this run).

These tests drive the REAL machinery: ``_agentic_native_page`` to emit the
events, ``_flush_page_sidecar`` to write the real ``pages/00001.json``, and
``_restore_terminal_page_state`` to read it back -- a full flush/restore
cycle, not a hand-built meta dict. Every assertion pins the event COUNT after
that cycle, not allowlist membership -- a membership-only test would pass
while replay stayed broken (see ``resume_restore_kinds`` returning the set
without the filter at :12235 ever consulting it correctly).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from socr.core.audit_log import AuditEvent
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState, PageState
from socr.judge.table_verdict import TABLE_WRAPPED_LABEL_MERGED_KIND
from socr.pipeline.orchestrator import UnifiedPipeline

#: The three kinds this ticket adds. Established by measurement, not by
#: absence: each is emitted only from ``_agentic_native_page`` (grep-verified
#: at src/socr/pipeline/orchestrator.py:9776/9800/9826), which the
#: ``if resumed is not None: ...; continue`` gate in ``_phase_agentic``
#: (:8701-8706) skips entirely for a terminal page -- so nothing re-emits
#: them on resume and the sidecar copy is the only record.
_LOST_KINDS = (
    "native_encoding_hygiene_suspect",
    "native_unrecovered_symbol_glyphs",
    "possible_table_structure_not_reconstructed",
)

#: Emitted by ``_phase_analyze``, which runs unconditionally before the
#: per-page resume loop on every single run (grep-verified: the
#: ``orphan_word_dropped`` AuditEvent sits inside ``_phase_analyze`` at
#: :1827, and ``process()`` calls ``self._phase_analyze(state)`` at :1466
#: unconditionally, before ``self._phase_agentic(state, out_dir)`` at :1468
#: which contains the per-page resume-skip gate). Replaying it from the
#: sidecar on top of this run's own re-emission would double-count it, so it
#: must stay OFF the allowlist.
_REEMITTED_KIND = "orphan_word_dropped"


def _pipeline() -> UnifiedPipeline:
    return UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.DEEPSEEK,
            enabled_engines=list(EngineType),
            agentic=False,
            quiet=True,
            native_first=True,
        )
    )


def _handle(pdf: Path, page_count: int = 1) -> DocumentHandle:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        return DocumentHandle(path=pdf, page_count=page_count)


def _native_page_state(*, all_flags: bool) -> PageState:
    ps = PageState(page_num=1)
    ps.is_born_digital = True
    ps.native_text = "Alpha 0.143 Beta"
    ps.has_encoding_hygiene_suspect = all_flags
    ps.has_unrecovered_symbol_glyphs = all_flags
    ps.possible_table_structure_not_reconstructed = all_flags
    return ps


def _run_native_page_and_flush(
    tmp_path: Path, *, all_flags: bool
) -> tuple[UnifiedPipeline, DocumentState, Path]:
    """Run 1: emit via the real ``_agentic_native_page``, flush a real sidecar."""
    pdf = tmp_path / "doc.pdf"
    out_dir = tmp_path / "out"
    pipeline = _pipeline()

    state = DocumentState(handle=_handle(pdf))
    ps = _native_page_state(all_flags=all_flags)
    state.pages[1] = ps

    pipeline._agentic_native_page(state, 1, ps)
    # The control: an already-allowlisted, unrelated kind riding along.
    state.events.append(
        AuditEvent(
            page_num=1,
            kind=TABLE_WRAPPED_LABEL_MERGED_KIND,
            engine="native",
            detail="control event, already allowlisted",
        )
    )
    # An analyze-phase kind: emitted every run, must never be replayed.
    state.events.append(
        AuditEvent(
            page_num=1,
            kind=_REEMITTED_KIND,
            engine="native",
            detail="analyze-phase event, re-emitted every run",
        )
    )

    sidecar = pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)
    assert sidecar is not None and sidecar.exists()

    return pipeline, state, out_dir


def _resume(
    pipeline: UnifiedPipeline, out_dir: Path, tmp_path: Path, *, all_flags: bool
) -> DocumentState:
    """Run 2: resume from run 1's sidecar with a fresh, event-free state."""
    pdf = tmp_path / "doc.pdf"
    resumed_state = DocumentState(handle=_handle(pdf))
    resumed_state.pages[1] = _native_page_state(all_flags=False)

    restored_out = PageOutput(
        page_num=1,
        text="Alpha 0.143 Beta",
        status=PageStatus.WARNING if all_flags else PageStatus.SUCCESS,
        engine="native",
        audit_passed=not all_flags,
        failure_mode=FailureMode.NONE,
        cost_usd=0.0,
    )
    pipeline._restore_terminal_page_state(resumed_state, 1, restored_out, out_dir)
    return resumed_state


def _counts(events: list[AuditEvent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ev in events:
        counts[ev.kind] = counts.get(ev.kind, 0) + 1
    return counts


class TestNativeAuditKindsSurviveResume:
    def test_run1_actually_emits_all_three(self, tmp_path: Path) -> None:
        """Sanity: before touching resume, confirm run 1's own emission."""
        _pipeline_obj, state, out_dir = _run_native_page_and_flush(tmp_path, all_flags=True)
        counts = _counts(state.events)
        for kind in _LOST_KINDS:
            assert counts.get(kind, 0) == 1
        assert out_dir.exists()

    def test_lost_kinds_replay_exactly_once_after_flush_and_restore(self, tmp_path: Path) -> None:
        pipeline, _state, out_dir = _run_native_page_and_flush(tmp_path, all_flags=True)
        resumed = _resume(pipeline, out_dir, tmp_path, all_flags=True)

        counts = _counts(resumed.events)
        for kind in _LOST_KINDS:
            assert counts.get(kind, 0) == 1, (
                f"{kind}: expected exactly 1 replayed event after a real "
                f"flush/restore cycle, got {counts.get(kind, 0)} -- {resumed.events}"
            )

    def test_control_kind_still_survives_resume(self, tmp_path: Path) -> None:
        """The allowlisted control must replay -- proves the mechanism works
        at all, so a failure on the three new kinds is the filter, not a
        broken harness."""
        pipeline, _state, out_dir = _run_native_page_and_flush(tmp_path, all_flags=True)
        resumed = _resume(pipeline, out_dir, tmp_path, all_flags=True)

        counts = _counts(resumed.events)
        assert counts.get(TABLE_WRAPPED_LABEL_MERGED_KIND, 0) == 1

    def test_analyze_phase_kind_is_not_double_counted(self, tmp_path: Path) -> None:
        """``orphan_word_dropped`` must NOT replay from the sidecar: it is
        re-emitted by ``_phase_analyze`` every run, and this resume path does
        not simulate that re-emission, so any count above 0 here would prove
        it was wrongly added to the allowlist."""
        pipeline, _state, out_dir = _run_native_page_and_flush(tmp_path, all_flags=True)
        resumed = _resume(pipeline, out_dir, tmp_path, all_flags=True)

        counts = _counts(resumed.events)
        assert counts.get(_REEMITTED_KIND, 0) == 0, (
            f"orphan_word_dropped replayed {counts.get(_REEMITTED_KIND, 0)} time(s) "
            "from the sidecar -- it must stay off resume_restore_kinds() because "
            "_phase_analyze re-emits it every run"
        )

    def test_page_carrying_none_of_these_kinds_resumes_unchanged(self, tmp_path: Path) -> None:
        """Both directions: a clean page must not gain events from thin air."""
        pipeline, _state, out_dir = _run_native_page_and_flush(tmp_path, all_flags=False)
        resumed = _resume(pipeline, out_dir, tmp_path, all_flags=False)

        counts = _counts(resumed.events)
        for kind in _LOST_KINDS:
            assert counts.get(kind, 0) == 0
        assert counts.get(_REEMITTED_KIND, 0) == 0

    def test_all_three_kinds_are_members_of_resume_restore_kinds(self) -> None:
        """Membership alone (kept alongside the count assertions above, never
        instead of them) -- documents the allowlist decision directly."""
        allowed = UnifiedPipeline.resume_restore_kinds()
        for kind in _LOST_KINDS:
            assert kind in allowed
        assert _REEMITTED_KIND not in allowed
