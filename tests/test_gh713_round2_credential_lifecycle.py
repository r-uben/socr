"""#713 round 2: the credential's LIFECYCLE, not just its first admission.

Astra's round-2 review of ``20d3df8`` (REQUEST_CHANGES) found four P1 holes and
one P2, all of them one hop past the fresh-admission path round 1 pinned:

1. a credentialed page RESTORED from the ledger failed selection again, because
   the restored body carries the disclosure note and was re-verified against the
   pre-note ``candidate_sha256``. The resume destroyed the reading the first run
   shipped and replaced it with the fail-closed marker;
2. a COMPLETED rejection could coexist with an earlier rung's timeout authority,
   in two shapes: the judge boundary left ``judge_outcome`` on timeout after a
   completed refusal, and a later rejected attempt over the SAME bytes was
   walked past by the timeout search;
3. the production deadline adapter turned its own timeout into an
   ``AcceptDecision``, so the real loop never typed a real timeout;
4. ``table_unexplained_lanes`` -- a REPORTED omission of native lane values --
   was exempted as if it were merely an unscorable page;
5. document ``metadata.json`` said "ladder exhausted; fail-closed floor shipped"
   for a page whose credentialed candidate actually shipped.

Every test pins a DIFFERENCE between two runs in the same process that vary
exactly one thing, per this repo's CI rule.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from test_gh713_judge_timeout_credential import (
    MODEL_TABLE,
    MODEL_TEXT,
    SECOND_TABLE,
    _credential,
    _pdf,
    _pipeline,
    _Prof,
    _state,
)

from socr.core.manifest import (
    CREDENTIAL_BLOCKING_EVENT_KINDS,
    CREDENTIAL_NON_BLOCKING_EVENT_KINDS,
    SelectionProvenance,
    _select_page_output_tagged,
    _winning_page_output,
)
from socr.core.result import (
    JUDGE_OUTCOME_COMPLETED,
    JUDGE_OUTCOME_TIMEOUT,
    REJECTION_JUDGE_ONLY,
    REJECTION_VERIFIER_ERROR,
    FailureMode,
    PageOutput,
    PageStatus,
)
from socr.core.tables_trust import TABLE_DISTRUST_KINDS


def _credentialed(pdf: Path):
    """A structure-class page whose one grid candidate holds a valid credential."""
    state = _state(pdf)
    out = state.pages[1].attempts[1]
    out.table_acceptance_credential = _credential(
        state, candidate_text=MODEL_TEXT, tables=[MODEL_TABLE]
    )
    return state, out


# ---------------------------------------------------------------------------
# P1-1. Restore -> reselection -> reassembly.
# ---------------------------------------------------------------------------


def test_restored_credentialed_page_reassembles_byte_identically(tmp_path: Path) -> None:
    """Astra P1-1: the resumed document must be the document, byte for byte.

    The full sequence, not just the ledger read: assemble, load the terminal
    page, install it as the page's attempt, run selection AGAIN, and reassemble.
    Round 1 stopped at the successful load; selection then re-verified the
    finalized body against ``candidate_sha256``, which describes bytes that
    deliberately predate the disclosure note, and the page collapsed to
    ``PAGE_JUDGE_TIMEOUT`` plus the marker -- a resume silently destroying a
    reading the first run had shipped.
    """
    pdf = _pdf(tmp_path)
    state, _ = _credentialed(pdf)
    pipeline = _pipeline()
    original = pipeline._phase_assemble(state, tmp_path / "first")

    fresh = _state(pdf)
    restored = pipeline._load_terminal_page(fresh, 1, tmp_path / "first")
    assert restored is not None, "the ledger gate must restore the credentialed page"
    ps = fresh.pages[1]
    ps.attempts = [restored]
    ps.best_output = restored

    winning, tag = _select_page_output_tagged(fresh, 1)
    assert winning.failure_mode is FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED
    assert winning.status is PageStatus.WARNING
    assert winning.audit_passed is False
    assert tag is SelectionProvenance.STRUCTURE_CLASS_JUDGE_TIMEOUT_RESTORED

    resumed = pipeline._phase_assemble(fresh, tmp_path / "second")
    assert resumed.markdown == original.markdown


def test_a_note_is_never_appended_twice_across_two_resumes(tmp_path: Path) -> None:
    """The restored ending ships VERBATIM, so the note count cannot grow.

    DIFFERENCE across three assemblies of the same page: original, resumed,
    resumed again. An ending that rebuilt the body instead of returning it would
    add one note per resume and still look "successful" at every other level.
    """
    pdf = _pdf(tmp_path)
    state, _ = _credentialed(pdf)
    pipeline = _pipeline()
    counts = []
    current = state
    for round_num in range(3):
        out_dir = tmp_path / f"round{round_num}"
        result = pipeline._phase_assemble(current, out_dir)
        counts.append(result.markdown.count("TIMED OUT"))
        fresh = _state(pdf)
        restored = pipeline._load_terminal_page(fresh, 1, out_dir)
        assert restored is not None
        fresh.pages[1].attempts = [restored]
        fresh.pages[1].best_output = restored
        current = fresh
    assert counts[0] >= 1
    assert counts == [counts[0]] * 3


def test_arbitrary_finalized_bytes_are_not_blessed_as_judged_bytes(tmp_path: Path) -> None:
    """DIFFERENCE: the restored body, vs the same body with one byte changed.

    The restored ending must verify against the digest finalization RECORDED,
    never merely accept whatever the ledger holds because it is flagged.
    """
    pdf = _pdf(tmp_path)
    state, _ = _credentialed(pdf)
    pipeline = _pipeline()
    pipeline._phase_assemble(state, tmp_path / "first")

    def _reselect(mutate: bool) -> FailureMode:
        fresh = _state(pdf)
        restored = pipeline._load_terminal_page(fresh, 1, tmp_path / "first")
        assert restored is not None
        if mutate:
            restored = replace(restored, text=restored.text.replace("13.9", "13.8"))
        fresh.pages[1].attempts = [restored]
        fresh.pages[1].best_output = restored
        return _winning_page_output(fresh, 1).failure_mode

    assert _reselect(False) is FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED
    assert _reselect(True) is FailureMode.PAGE_JUDGE_TIMEOUT


# ---------------------------------------------------------------------------
# P1-2. A completed rejection retires the authority.
# ---------------------------------------------------------------------------


def test_completed_rejection_at_the_judge_boundary_retires_the_credential(
    tmp_path: Path,
) -> None:
    """DIFFERENCE: the same credentialed output, judged by two judges.

    One judge times out (the verdict is missing, the credential stands and the
    reading ships flagged); the other REFUSES the same bytes (the verdict
    exists, so nothing may ship under an exception meant for a missing one).
    """
    from socr.pipeline import agentic

    pdf = _pdf(tmp_path)

    class _Rejecting:
        def assess(self, output, prof):
            return agentic.AcceptDecision(accept=False, reason="completed rejection")

    class _TimingOut:
        def assess(self, output, prof):
            raise TimeoutError("timed out")

    modes = {}
    for label, judge in (("rejected", _Rejecting()), ("timed_out", _TimingOut())):
        state, out = _credentialed(pdf)
        agentic.route_page(1, [_Prof()], lambda prof, page, _o=out: _o, judge)
        modes[label] = (
            out.judge_outcome,
            out.table_acceptance_credential is not None,
            _winning_page_output(state, 1).failure_mode,
        )

    assert modes["timed_out"] == (
        JUDGE_OUTCOME_TIMEOUT,
        True,
        FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED,
    )
    assert modes["rejected"][0] == JUDGE_OUTCOME_COMPLETED
    assert modes["rejected"][1] is False
    assert modes["rejected"][2] is not FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED


def test_a_later_rejection_of_the_same_bytes_invalidates_the_incumbent(
    tmp_path: Path,
) -> None:
    """Astra P1-2 case (b): the timeout search must not walk past a refusal.

    DIFFERENCE: the later rejected attempt carries the SAME bytes as the
    credentialed incumbent, vs a DIFFERENT candidate (a re-cropped reading). The
    first is an applicable verdict about this reading and voids it; the second
    is a verdict about other text and must leave an unchanged incumbent alone --
    otherwise any rejected escalation attempt would delete a good page.
    """
    pdf = _pdf(tmp_path)

    def _ships_with(rejected_text: str) -> FailureMode:
        state, out = _credentialed(pdf)
        rejected = replace(
            out,
            text=rejected_text,
            judge_outcome=JUDGE_OUTCOME_COMPLETED,
            table_acceptance_credential=None,
            rejection_class=REJECTION_JUDGE_ONLY,
            judge_reason="completed rejection",
        )
        state.pages[1].attempts.append(rejected)
        state.pages[1].best_output = rejected
        return _winning_page_output(state, 1).failure_mode

    same_bytes = _ships_with(MODEL_TEXT)
    other_bytes = _ships_with(MODEL_TEXT.replace(SECOND_TABLE, "") + "\nRe-cropped reading.\n")
    assert same_bytes is not FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED
    assert other_bytes is FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED


def test_a_verifier_error_is_not_a_completed_refusal(tmp_path: Path) -> None:
    """DIFFERENCE: ``REJECTION_JUDGE_ONLY`` vs ``REJECTION_VERIFIER_ERROR``.

    The verifier breaking is another MISSING verdict, not a negative one, so it
    must not retire authority the way a refusal does -- the same distinction the
    whole ticket rests on, applied to the invalidation side.
    """
    pdf = _pdf(tmp_path)

    def _ships_with(rejection_class: str) -> FailureMode:
        state, out = _credentialed(pdf)
        later = replace(
            out,
            judge_outcome="",
            table_acceptance_credential=None,
            rejection_class=rejection_class,
        )
        state.pages[1].attempts.append(later)
        state.pages[1].best_output = later
        return _winning_page_output(state, 1).failure_mode

    assert _ships_with(REJECTION_JUDGE_ONLY) is not FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED
    assert _ships_with(REJECTION_VERIFIER_ERROR) is FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED


def test_the_restored_ending_refuses_on_the_same_grounds_as_the_fresh_one(
    tmp_path: Path,
) -> None:
    """A page that fails closed on the first run must not be admitted on resume.

    DIFFERENCE: the same restored body, with and without a blocking event on the
    page. Both endings share ``_credential_admission_refusal`` for exactly this;
    a restored page that skipped those checks would be a way to launder a
    contradiction through the ledger.
    """
    from socr.core.audit_log import AuditEvent

    pdf = _pdf(tmp_path)
    state, _ = _credentialed(pdf)
    pipeline = _pipeline()
    pipeline._phase_assemble(state, tmp_path / "first")

    def _reselect(with_event: bool) -> FailureMode:
        fresh = _state(pdf)
        restored = pipeline._load_terminal_page(fresh, 1, tmp_path / "first")
        assert restored is not None
        fresh.pages[1].attempts = [restored]
        fresh.pages[1].best_output = restored
        if with_event:
            fresh.events.append(
                AuditEvent(page_num=1, kind="table_structure_failed", detail="row count drift")
            )
        return _winning_page_output(fresh, 1).failure_mode

    assert _reselect(False) is FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED
    assert _reselect(True) is FailureMode.PAGE_JUDGE_TIMEOUT


# ---------------------------------------------------------------------------
# P1-3. The wrapper the real loop uses.
# ---------------------------------------------------------------------------


def test_the_production_deadline_adapter_types_its_own_timeout(tmp_path: Path) -> None:
    """Astra P1-3: test the WRAPPER the real loop installs, not a bare judge.

    ``_phase_agentic`` wraps every page judge in ``_TimeoutJudge``. Round 1's
    typed field was only ever exercised against judges that raise directly, and
    the adapter converted its OWN deadline into ``AcceptDecision(False, "judge
    timeout")`` -- so on the production path the field it added was never set,
    a credential could never be minted, and a real timeout was indistinguishable
    from a real refusal.

    DIFFERENCE: three judges behind the SAME real adapter -- one that exceeds
    the deadline, one that raises a timeout of its own, one that answers.
    """
    import time

    from socr.pipeline import agentic
    from socr.pipeline.orchestrator import UnifiedPipeline

    class _Slow:
        def assess(self, output, prof):
            time.sleep(10)
            return agentic.AcceptDecision(accept=True, reason="never reached")

    class _RaisesTimeout:
        def assess(self, output, prof):
            raise TimeoutError("inner timed out")

    class _Answers:
        def assess(self, output, prof):
            return agentic.AcceptDecision(accept=False, reason="completed rejection")

    outcomes = {}
    reasons = {}
    for label, inner in (
        ("deadline", _Slow()),
        ("inner", _RaisesTimeout()),
        ("answered", _Answers()),
    ):
        out = PageOutput(page_num=1, text=MODEL_TEXT, status=PageStatus.SUCCESS, engine="qwen")
        judge = UnifiedPipeline._TimeoutJudge(inner, timeout_sec=0.05)
        decision = agentic.route_page(1, [_Prof()], lambda prof, page, _o=out: _o, judge)
        outcomes[label] = out.judge_outcome
        reasons[label] = decision.attempts[-1].reason or ""

    assert outcomes["deadline"] == JUDGE_OUTCOME_TIMEOUT
    assert outcomes["inner"] == JUDGE_OUTCOME_TIMEOUT
    assert outcomes["answered"] == JUDGE_OUTCOME_COMPLETED
    # The cascade-halt probe in ``_phase_agentic`` decides whether a wedged
    # backend stops the document by scanning attempt reasons for "timeout".
    # Raising must not have taken that signal away.
    assert "timeout" in reasons["deadline"].lower()


# ---------------------------------------------------------------------------
# P1-4. The blocking policy, in both directions.
# ---------------------------------------------------------------------------


def test_an_unexplained_native_lane_blocks_the_stand_in(tmp_path: Path) -> None:
    """Astra P1-4: a reported omission is a contradiction, not a missing score.

    ``table_unexplained_lanes`` fires when native lanes carry VALUES in matched
    rows that map to no emitted column. Admitting a stand-in over an outstanding
    one ships a table known to be missing values, under a credential asserting
    every table was accepted.

    DIFFERENCE: the omission event vs ``table_not_scorable``, which stays exempt
    because inability to measure is not a positive rejection.
    """
    from socr.core.audit_log import AuditEvent

    pdf = _pdf(tmp_path)

    def _ships_with(kind: str) -> FailureMode:
        state, _ = _credentialed(pdf)
        state.events.append(
            AuditEvent(
                page_num=1,
                kind=kind,
                detail="1 native lane carries values in matched rows but maps to no column",
                data={"unexplained_lanes": 1},
            )
        )
        return _winning_page_output(state, 1).failure_mode

    assert _ships_with("table_unexplained_lanes") is FailureMode.PAGE_JUDGE_TIMEOUT
    assert _ships_with("table_not_scorable") is FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED


def test_the_blocking_policy_pins_its_exclusions_not_only_its_members() -> None:
    """A newly added distrust kind must not become exempt by default.

    Round 1 asserted only that every blocker belongs to the trust index. That
    subset relation is satisfied by an EMPTY policy, and it stays green when a
    distrust kind is added and silently left out. The complement is therefore
    pinned explicitly, so adding a kind to ``TABLE_DISTRUST_KINDS`` fails here
    until someone decides, in writing, which side it belongs on.
    """
    assert CREDENTIAL_BLOCKING_EVENT_KINDS <= TABLE_DISTRUST_KINDS
    assert CREDENTIAL_NON_BLOCKING_EVENT_KINDS <= TABLE_DISTRUST_KINDS
    assert not (CREDENTIAL_BLOCKING_EVENT_KINDS & CREDENTIAL_NON_BLOCKING_EVENT_KINDS)
    assert (
        CREDENTIAL_BLOCKING_EVENT_KINDS | CREDENTIAL_NON_BLOCKING_EVENT_KINDS
    ) == TABLE_DISTRUST_KINDS
    assert CREDENTIAL_NON_BLOCKING_EVENT_KINDS == {
        # inability to SCORE is not a positive rejection
        "table_not_scorable",
        # the four S1 kinds this very selection emits from its own result
        "structure_class_model_table_kept",
        "structure_class_ladder_exhausted_floor",
        "structure_class_row_corroborated",
        "structure_floor_overrode_ladder",
    }
    assert "table_unexplained_lanes" in CREDENTIAL_BLOCKING_EVENT_KINDS


# ---------------------------------------------------------------------------
# P2. Document metadata must describe what shipped.
# ---------------------------------------------------------------------------


def test_document_metadata_does_not_claim_a_floor_for_a_page_that_shipped(
    tmp_path: Path,
) -> None:
    """Astra P2: page sidecars cannot rescue metadata that contradicts the body.

    DIFFERENCE: credential present vs absent, through the real assemble phase,
    read off the document's own ``metadata.json``. With it, the candidate ships
    and the document must say so; without it, the floor ships and the document
    must say THAT -- naming the timeout, because "every candidate was refused"
    and "nothing ever judged the candidate" call for different operator action.
    """
    pdf = _pdf(tmp_path)
    errors = {}
    for label in ("credentialed", "floor"):
        state, out = _credentialed(pdf)
        if label == "floor":
            out.table_acceptance_credential = None
        clean = PageOutput(
            page_num=2,
            text="Closing prose.",
            engine="qwen",
            status=PageStatus.SUCCESS,
            audit_passed=True,
        )
        state.pages[2].attempts = [clean]
        state.pages[2].best_output = clean
        out_dir = tmp_path / label
        _pipeline()._phase_assemble(state, out_dir)
        # The DOCUMENT's metadata, under its own stem directory -- not the
        # run-level file beside it, which carries no per-document error.
        paths = [p for p in out_dir.rglob("metadata.json") if p.parent != out_dir]
        assert paths, "assemble must write document metadata"
        errors[label] = json.loads(paths[0].read_text()).get("error") or ""

    assert "fail-closed floor shipped" not in errors["credentialed"]
    assert "TIMED OUT" in errors["credentialed"]
    assert "INCOMPLETE" in errors["credentialed"]
    assert "fail-closed floor shipped" in errors["floor"]
    assert "TIMED OUT" in errors["floor"]
    # The ordinary "structure-class ladder exhausted" reason belongs to neither
    # page: nothing here exhausted a ladder.
    assert "structure-class ladder exhausted" not in errors["credentialed"]
    assert "structure-class ladder exhausted" not in errors["floor"]
