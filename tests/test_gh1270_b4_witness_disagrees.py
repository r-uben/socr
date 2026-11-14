"""#1270 (B4, class 4): a witness disagreement is not a hallucination.

Before: both fail-closed endings of ``SourceEvidenceTableJudge`` except no-witness stamped
``FailureMode.HALLUCINATION``, including a witness that could not examine the table at all
(90 pages in run 51270eb5: 85 disagreements, 5 unexaminable). Now: ``WITNESS_DISAGREES`` and
``TABLE_UNVERIFIED_WITNESS``. The ENDING is unchanged, so every assertion pins a DIFFERENCE:
the arms ship the same text, status and ``audit_passed`` and differ only in the recorded cause.

Hermetic: ``_available_engines_for_agentic`` patched, ``primary_engine`` / ``local_engine`` /
``enabled_engines`` pinned, ``_resolve_judge_model`` patched to "", no real tesseract.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest
from test_gh658_no_witness_backend_reason import (
    CANDIDATE_TABLE,
    CONTRADICTING_WITNESS_TEXT,
    _assess,
    _gate_attempt,
    _witness,
    state_of,
)

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.manifest import SelectionProvenance, _select_page_output_tagged
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import WITNESS_REJECT_MODES, FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.agentic import (
    AcceptDecision,
    PageDecision,
    ProviderAttempt,
    SourceEvidenceTableJudge,
)
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.source_evidence import (
    WITNESS_READING,
    SourceEvidenceResult,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _pipeline() -> UnifiedPipeline:
    return UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            agentic=True,
            judge_backend="heuristic",
            quiet=True,
            save_figures=False,
            write_manifest=False,
            native_first=False,
            dual_pass_tables=False,
            escalate_ambiguous_tables=False,
            table_judge_ladder=False,
        )
    )


def _unverifiable_assess():
    """The gate with a witness that could not examine the table (``verifiable`` False)."""
    from socr.core.audit_log import AuditEvent

    events: list[AuditEvent] = []
    inner = MagicMock()
    inner.assess.return_value = AcceptDecision(accept=False, reason="inner refuses")
    doc = fitz.open()
    page = doc.new_page(width=500, height=700)
    judge = SourceEvidenceTableJudge(
        inner=inner,
        get_fitz_page=lambda pn: page,
        record_event=events.append,
        native_trusted=lambda pn: False,
    )
    output = PageOutput(
        page_num=3,
        text=CANDIDATE_TABLE,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    result = SourceEvidenceResult(
        verifiable=False, passed=False, reason="the table region could not be localized"
    )
    with patch("socr.tables.source_evidence.verify_scanned_table", return_value=result):
        decision = judge.assess(output, MagicMock())
    return decision, output, events


# ---------------------------------------------------------------- the gate (producer)


def test_gate_labels_each_ending_by_what_the_witness_did_and_changes_nothing_else() -> None:
    disagree_dec, disagree, disagree_ev = _assess(
        witness=_witness(WITNESS_READING, text=CONTRADICTING_WITNESS_TEXT)
    )
    unver_dec, unver, unver_ev = _unverifiable_assess()

    assert disagree.failure_mode is FailureMode.WITNESS_DISAGREES
    assert unver.failure_mode is FailureMode.TABLE_UNVERIFIED_WITNESS
    # The ending is the old ending, in both arms: same status, same audit flag, still
    # rejected, and the load-bearing latch token the scanned-table floor reads is intact.
    assert disagree_dec.accept is unver_dec.accept is False
    assert disagree_dec.reason.startswith("source_evidence_table")
    assert unver_dec.reason.startswith("source_evidence_table")
    assert disagree.status is unver.status is PageStatus.ERROR
    assert disagree.audit_passed is unver.audit_passed is False
    # Still counted as a distrusted table (tables_trust reads this kind).
    assert any(e.kind == "source_evidence_table_reject" for e in disagree_ev)
    assert any(e.kind == "source_evidence_table_reject" for e in unver_ev)
    # The carried evidence follows the cause: a disagreement names the values, an
    # unexaminable table names the reason.
    assert disagree.table_disputed and not disagree.table_unverifiable
    assert unver.table_unverifiable and not unver.table_disputed
    # Nothing is labelled a hallucination any more.
    assert FailureMode.HALLUCINATION not in {disagree.failure_mode, unver.failure_mode}


def test_every_new_mode_is_in_the_control_flow_set_and_the_legacy_one_stays() -> None:
    assert WITNESS_REJECT_MODES == {
        FailureMode.HALLUCINATION,
        FailureMode.WITNESS_DISAGREES,
        FailureMode.TABLE_UNVERIFIED_WITNESS,
    }


# -------------------------------------------------- the floor (recorded cause, one ending)


def _floored_state(tmp_path: Path, *, unverifiable: bool, mode: FailureMode | None = None):
    pdf = tmp_path / "doc.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.touch()
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf, page_count=1)
    state = DocumentState(handle=handle)
    ps = state.pages[1]
    ps.is_born_digital = False
    ps.has_tables = True
    ps.native_text = ""
    ps.best_output = PageOutput(
        page_num=1, text=CANDIDATE_TABLE, status=PageStatus.SUCCESS, engine="qwen"
    )
    _pipeline()._apply_scanned_table_floor(ps, pdf, 1, None, witness_unverifiable=unverifiable)
    if mode is not None:  # a legacy record: the sidecar says hallucination
        ps.best_output.failure_mode = mode
        ps.scanned_table_witness_unverifiable = False
    ps.attempts.append(ps.best_output)
    ps.d3_floor_png_ref = "![p1](figures/p001.png)"
    return state


def test_floor_records_the_cause_and_ships_the_same_page(tmp_path: Path) -> None:
    disagree = _floored_state(tmp_path / "a", unverifiable=False)
    unver = _floored_state(tmp_path / "b", unverifiable=True)
    legacy = _floored_state(tmp_path / "c", unverifiable=False, mode=FailureMode.HALLUCINATION)

    d_out, d_prov = _select_page_output_tagged(disagree, 1)
    u_out, u_prov = _select_page_output_tagged(unver, 1)
    l_out, l_prov = _select_page_output_tagged(legacy, 1)

    assert d_out.failure_mode is FailureMode.WITNESS_DISAGREES
    assert u_out.failure_mode is FailureMode.TABLE_UNVERIFIED_WITNESS
    # shipping boundary: identical bytes, status, audit flag and provenance on all three arms
    assert d_out.text == u_out.text == l_out.text
    assert "failed: unverifiable table" in d_out.text
    assert d_out.status is u_out.status is l_out.status is PageStatus.ERROR
    assert d_out.audit_passed is u_out.audit_passed is l_out.audit_passed is False
    assert d_prov is u_prov is l_prov is SelectionProvenance.UNVERIFIABLE_TABLE_SCANNED
    assert disagree.pages[1].scanned_table_witness_unverifiable is False
    assert unver.pages[1].scanned_table_witness_unverifiable is True
    assert unver.pages[1].scanned_table_no_witness is False


@pytest.mark.parametrize("unverifiable", [False, True])
def test_the_new_causes_survive_the_sidecar_and_the_resume(
    tmp_path: Path, unverifiable: bool
) -> None:
    """new durable state restores: DERIVED from the persisted mode, no new sidecar key."""
    run1 = _floored_state(tmp_path / "r1", unverifiable=unverifiable)
    pipeline = _pipeline()
    out_dir = tmp_path / "out"
    sidecar = pipeline._flush_page_sidecar(run1, 1, out_dir, terminal=True)
    assert sidecar is not None
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    expected = (
        FailureMode.TABLE_UNVERIFIED_WITNESS if unverifiable else FailureMode.WITNESS_DISAGREES
    )
    assert meta["failure_mode"] == expected.value
    assert "scanned_table_witness_unverifiable" not in meta  # P6 key set stays frozen

    run2 = _floored_state(tmp_path / "r2", unverifiable=False)
    run2.pages[1].scanned_table_witness_unverifiable = False
    restored = PageOutput(
        page_num=1,
        text=CANDIDATE_TABLE,
        status=PageStatus(meta["status"]),
        engine="qwen",
        audit_passed=False,
        failure_mode=FailureMode(meta["failure_mode"]),
    )
    run3 = DocumentState(handle=run2.handle)
    pipeline._restore_terminal_page_state(run3, 1, restored, out_dir)
    ps = run3.pages[1]
    assert ps.scanned_table_evidence_failed is True
    assert ps.scanned_table_witness_unverifiable is unverifiable
    assert ps.scanned_table_no_witness is False
    assert ps.scanned_table_withheld_unconfirmed is False


# ------------------------------- the ladder (control flow), both provider states


def _attempt(mode: FailureMode, reason: str, disputed: str = "") -> ProviderAttempt:
    a = _gate_attempt(no_witness=False, engine_name="qwen", reason_suffix=reason)
    a.output.failure_mode = mode
    a.output.table_disputed = disputed
    return a


def _run(tmp_path: Path, attempts: list, providers: list):
    pdf = tmp_path / "doc.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=500, height=700)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 2, 2), False)
    pix.set_rect(pix.irect, (200, 200, 200))
    page.insert_image(fitz.Rect(72, 72, 74, 74), pixmap=pix)
    doc.save(str(pdf))
    doc.close()
    pipeline = _pipeline()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    ps = state.pages[1]
    ps.is_born_digital = False
    ps.has_tables = True
    ps.native_text = ""

    def _route(page_num, *a, **kw):
        return PageDecision(
            page_num=page_num,
            final_output=attempts[-1].output,
            attempts=list(attempts),
            accepted=False,
        )

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=providers),
        patch("socr.pipeline.orchestrator.route_page", side_effect=_route),
        patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
    ):
        pipeline._phase_agentic(state, tmp_path / "out")
    shipped, _ = _select_page_output_tagged(state, 1)
    return state.pages[1], shipped


def _outcome(ps, shipped):
    return (
        shipped.text,
        shipped.status,
        shipped.audit_passed,
        ps.scanned_table_evidence_failed,
        ps.scanned_table_no_witness,
        ps.scanned_table_withheld_unconfirmed,
    )


@pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []], ids=["provider", "no-provider"])
def test_ladder_floors_every_cause_identically_and_records_only_the_cause(
    tmp_path: Path, providers: list
) -> None:
    arms = {
        "legacy": [_attempt(FailureMode.HALLUCINATION, "numeric tokens unsupported: ['62.5']")],
        "disagree": [
            _attempt(FailureMode.WITNESS_DISAGREES, "numeric tokens unsupported: ['62.5']")
        ],
        "unverifiable": [_attempt(FailureMode.TABLE_UNVERIFIED_WITNESS, "region not localized")],
    }
    results = {k: _run(tmp_path / k, v, providers) for k, v in arms.items()}
    outcomes = {k: _outcome(*v) for k, v in results.items()}
    # no-provider state: the loop bails before routing, so the arms are indistinguishable
    # and the old/new labels cannot matter. Provider state: floored, identical but for cause.
    assert outcomes["legacy"] == outcomes["disagree"] == outcomes["unverifiable"]
    if providers:
        assert outcomes["disagree"][3] is True, "the floor must still apply"
        assert results["legacy"][1].failure_mode is FailureMode.WITNESS_DISAGREES
        assert results["disagree"][1].failure_mode is FailureMode.WITNESS_DISAGREES
        assert results["unverifiable"][1].failure_mode is FailureMode.TABLE_UNVERIFIED_WITNESS
        assert results["unverifiable"][0].scanned_table_witness_unverifiable is True
    else:
        assert results["disagree"][1].failure_mode is not FailureMode.WITNESS_DISAGREES


@pytest.mark.parametrize(
    "later", [FailureMode.WITNESS_DISAGREES, FailureMode.TABLE_UNVERIFIED_WITNESS]
)
def test_a_witness_that_examined_the_page_beats_an_earlier_missing_witness(
    tmp_path: Path, later: FailureMode
) -> None:
    """Both new modes are witness verdicts for the no-witness reduction: the page was
    read or examined by something, so it is not 'no witness' (the #658 rule, new modes)."""
    first = _gate_attempt(no_witness=True, engine_name="qwen", reason_suffix="no OCR witness")
    second = _attempt(later, "second rung")
    ps, shipped = _run(tmp_path, [first, second], [PROFILE_QWEN_LOCAL])
    assert ps.scanned_table_evidence_failed is True
    assert ps.scanned_table_no_witness is False
    assert shipped.failure_mode is later
    assert UnifiedPipeline._no_witness_backend_pages(state_of(ps)) == []


# ---------------------------- the readers that must not fail open (manifest sites)


GRID = "| A | B |\n| --- | --- |\n| x | 1 |\n| y | 2 |\n"


@pytest.mark.parametrize("mode", sorted(WITNESS_REJECT_MODES, key=lambda m: m.value))
def test_a_witness_rejected_table_is_never_a_candidate_grid(mode: FailureMode) -> None:
    """Three manifest predicates read ``ERROR or mode in WITNESS_REJECT_MODES``. The status
    is WARNING here so the MODE alone decides: a missed site would let a witness-rejected
    grid back into the pool a floor is meant to exclude (fail open)."""
    from socr.core.manifest import (
        _grid_authored_attempt,
        _grid_reading_attempt,
        _grid_shaped_attempt,
    )

    def out(m: FailureMode) -> PageOutput:
        return PageOutput(
            page_num=1,
            text=GRID,
            status=PageStatus.WARNING,
            engine="qwen",
            audit_passed=True,
            failure_mode=m,
        )

    for pred in (_grid_shaped_attempt, _grid_reading_attempt, _grid_authored_attempt):
        assert pred(out(FailureMode.NONE)) is True, f"{pred.__name__}: premise"
        assert pred(out(mode)) is False, f"{pred.__name__} admitted {mode}"


@pytest.mark.parametrize("mode", sorted(WITNESS_REJECT_MODES, key=lambda m: m.value))
def test_the_two_page_level_keepers_refuse_a_witness_rejected_model_table(
    mode: FailureMode,
) -> None:
    """``flagged_model_page_output`` and ``d3_floor_kept_model_output`` read the same set.
    Each is shown to KEEP a model table on the control arm (mode NONE) before the mode arm
    is asserted to refuse it, so a premise that never reached the line fails loudly."""
    from socr.core.manifest import d3_floor_kept_model_output, flagged_model_page_output
    from socr.core.result import REJECTION_AMBIGUOUS_DEFERRED
    from socr.core.state import PageState

    def page(m: FailureMode) -> PageState:
        ps = PageState(page_num=1)
        ps.is_born_digital = True
        ps.native_text = "native words"
        ps.native_table_structure_failed = True
        ps.native_table_unverifiable = True
        ps.native_table_structure_defective = True
        out = PageOutput(
            page_num=1,
            text=GRID,
            status=PageStatus.WARNING,
            engine="qwen",
            audit_passed=False,
            failure_mode=m,
            rejection_class=REJECTION_AMBIGUOUS_DEFERRED,
        )
        ps.best_output = out
        ps.attempts.append(out)
        return ps

    control = page(FailureMode.NONE)
    assert d3_floor_kept_model_output(control) is not None, "premise: D3 keeper reaches the line"
    flagged_control = page(FailureMode.NONE)
    flagged_control.native_table_unverifiable = False  # the flagged-but-not-floored shape
    assert flagged_model_page_output(flagged_control) is not None, "premise: #259 reaches the line"

    arm = page(mode)
    assert d3_floor_kept_model_output(arm) is None
    flagged_arm = page(mode)
    flagged_arm.native_table_unverifiable = False
    assert flagged_model_page_output(flagged_arm) is None


@pytest.mark.parametrize(
    ("mode", "result"),
    [(FailureMode.WITNESS_DISAGREES, "FAIL"), (FailureMode.TABLE_UNVERIFIED_WITNESS, "UNKNOWN")],
)
def test_the_verdict_names_the_cause_instead_of_the_unknown_fallback(
    mode: FailureMode, result: str
) -> None:
    """Without a ``_MODE_CLAUSES`` row the page's mark falls to ``unknown=UNKNOWN`` -- a mark
    that says socr did not record why. The legacy mode still does (pinned as the premise)."""
    from socr.core.state import PageState
    from socr.core.verdict import build_page_verdict

    def clauses(m: FailureMode) -> dict[str, str]:
        out = PageOutput(
            page_num=1,
            text="[page 1 failed: unverifiable table — see image]",
            status=PageStatus.ERROR,
            engine="qwen",
            audit_passed=False,
            failure_mode=m,
        )
        v = build_page_verdict(1, out, PageState(page_num=1))
        return {c.check.value: c.result.value for c in v.clauses}

    assert clauses(FailureMode.HALLUCINATION).get("unknown") == "UNKNOWN", "premise"
    got = clauses(mode)
    assert "unknown" not in got
    assert got.get("source_evidence") == result


def test_a_pre_1270_sidecar_keeps_its_stored_hallucination_on_resume(tmp_path: Path) -> None:
    """Review of PR #1325: a floor page whose sidecar stored ``hallucination`` (written before
    the label was split) is not relabelled when selection runs again on resume. The keep needs
    the floor's own marker in the saved text as well as the stored mode."""
    state = _floored_state(tmp_path / "a", unverifiable=False)
    floor, _prov = _select_page_output_tagged(state, 1)
    assert floor.failure_mode is FailureMode.WITNESS_DISAGREES

    # Resume: the saved floor output comes back with the mode the old run stored.
    old = _floored_state(tmp_path / "b", unverifiable=False)
    stored = PageOutput(
        page_num=1,
        text=floor.text,
        status=PageStatus.ERROR,
        engine=floor.engine,
        audit_passed=False,  # as the floor writes it; True would win the passing-output shortcut
        failure_mode=FailureMode.HALLUCINATION,
    )
    old.pages[1].best_output = stored
    old.pages[1].attempts = [stored]
    resumed, _ = _select_page_output_tagged(old, 1)
    assert resumed.failure_mode is FailureMode.HALLUCINATION
    assert resumed.text == floor.text and resumed.status is floor.status

    # Difference arm: the same saved floor output stored by a run of this code (the new mode)
    # keeps the new mode: only the stored mode differs between the two resumed arms.
    new = _floored_state(tmp_path / "c", unverifiable=False)
    stored_new = PageOutput(
        page_num=1,
        text=floor.text,
        status=PageStatus.ERROR,
        engine=floor.engine,
        audit_passed=False,  # as the floor writes it; True would win the passing-output shortcut
        failure_mode=FailureMode.WITNESS_DISAGREES,
    )
    new.pages[1].best_output = stored_new
    new.pages[1].attempts = [stored_new]
    again, _ = _select_page_output_tagged(new, 1)
    assert again.failure_mode is FailureMode.WITNESS_DISAGREES
    assert again.text == resumed.text
