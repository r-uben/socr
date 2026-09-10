"""#713 rounds 3-4: WHICH bytes a refusal is about, and what counts as a refusal.

Astra's round-3 review of ``4d3f3b0`` (REQUEST_CHANGES) found one P1 and two P2,
all of them about identity rather than policy:

1. the restored ending tested supersession by hashing ``out.text`` -- which for a
   restored page is the FINALIZED body, note included -- so a later completed
   refusal of the ORIGINAL candidate bytes never matched and the restored
   authority shipped over a live rejection of the very reading it vouches for.
   The ledger gate had the same hole from the other side: it validated the saved
   record against itself and never looked at what the live run knows;
2. the deadline adapter re-raised an inner timeout UNCHANGED, so a judge raising
   builtin ``TimeoutError("timed out")`` reached the trail as "judge raised:
   timed out" -- the cascade-halt probe scanned for the contiguous substring
   "timeout", found none, and never armed on a wedged backend;
3. ``_reject_unverified`` returns a negative decision when the deterministic
   table VERIFIER RAISED. The boundary stamped that COMPLETED, so the
   verifier-error exemption one line later was bypassed and an infrastructure
   crash retired a credential nothing had contradicted.

Round 4 added the state-transition half of (3): the boundary detected "the
verifier ran" by comparing a PERSISTENT field before and after the call, which
cannot see a same-value assignment, so a SECOND consecutive verifier crash on the
same output was recorded as a completed rejection. The typed outcome now rides on
the decision that call returned.

Every test pins a DIFFERENCE between two runs in the same process that vary
exactly one thing, per this repo's CI rule.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from test_gh713_judge_timeout_credential import (
    MODEL_TABLE,
    MODEL_TEXT,
    _credential,
    _pdf,
    _pipeline,
    _Prof,
    _state,
)

from socr.core.manifest import _winning_page_output
from socr.core.page_credential import sha256_text
from socr.core.result import (
    JUDGE_OUTCOME_COMPLETED,
    JUDGE_OUTCOME_TIMEOUT,
    JUDGE_OUTCOME_VERIFIER_ERROR,
    REJECTION_JUDGE_ONLY,
    REJECTION_VERIFIER_ERROR,
    FailureMode,
    PageOutput,
    PageStatus,
)


def _fresh_dir(base: Path, name: str) -> Path:
    """A private directory per sub-run: every DIFFERENCE here runs twice."""
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _restored(tmp_path: Path):
    """Assemble a credentialed timeout page, then load it back off the ledger.

    Returns the pipeline, the ledger directory, a FRESH state with the restored
    output installed, and the restored output itself -- the exact shape resume
    hands to the second ``_phase_assemble``.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = _pdf(tmp_path)
    state = _state(pdf)
    out = state.pages[1].attempts[1]
    out.table_acceptance_credential = _credential(
        state, candidate_text=MODEL_TEXT, tables=[MODEL_TABLE]
    )
    pipeline = _pipeline()
    directory = tmp_path / "ledger"
    pipeline._phase_assemble(state, directory)

    fresh = _state(pdf)
    restored = pipeline._load_terminal_page(fresh, 1, directory)
    assert restored is not None
    return pipeline, directory, fresh, restored


def _refusal_of(out: PageOutput, text: str) -> PageOutput:
    """A later attempt that COMPLETED a refusal of ``text``."""
    return replace(
        out,
        text=text,
        judge_outcome=JUDGE_OUTCOME_COMPLETED,
        table_acceptance_credential=None,
        rejection_class=REJECTION_JUDGE_ONLY,
        judge_reason="completed rejection",
    )


# ---------------------------------------------------------------------------
# P1. The restored body is not the judged bytes; the credential says which are.
# ---------------------------------------------------------------------------


def test_a_refusal_of_the_original_candidate_retires_the_restored_authority(
    tmp_path: Path,
) -> None:
    """Astra round 3, P1: supersession is keyed to the CANDIDATE identity.

    DIFFERENCE: the later refusal carries the ORIGINAL candidate bytes (the
    credential's ``candidate_sha256``, which are NOT the restored body's bytes)
    vs a different reading entirely. The first is a verdict about this page's
    reading and must void the restored authority; the second is about other text
    and must leave it alone.
    """

    def _ships_with(rejected_text: str) -> FailureMode:
        _, _, fresh, restored = _restored(_fresh_dir(tmp_path, str(len(rejected_text))))
        rejected = _refusal_of(restored, rejected_text)
        fresh.pages[1].attempts = [restored, rejected]
        fresh.pages[1].best_output = rejected
        return _winning_page_output(fresh, 1).failure_mode

    _, _, probe, restored_probe = _restored(_fresh_dir(tmp_path, "identity"))
    # The premise: the shipped body is NOT the judged candidate, and the
    # credential is the only record that knows the judged one.
    assert restored_probe.text != MODEL_TEXT
    assert restored_probe.table_acceptance_credential["candidate_sha256"] == sha256_text(MODEL_TEXT)
    assert probe is not None

    original_bytes = _ships_with(MODEL_TEXT)
    other_bytes = _ships_with(MODEL_TEXT + "\nA re-cropped reading of the same page.\n")
    assert original_bytes is not FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED
    assert other_bytes is FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED


def test_the_ledger_gate_refuses_a_credential_the_live_run_already_refused(
    tmp_path: Path,
) -> None:
    """Astra round 3, P1, the sidecar half: the record verifies against itself.

    A sidecar written before any rung refused these bytes still passes every
    check in ``_load_terminal_page`` -- credential, fingerprint and fragment
    digest all agree with each other. Only the LIVE state knows a completed
    verdict has since refused this candidate.

    DIFFERENCE: the same ledger read against the same directory, with and
    without that rejection present on the live page.
    """
    pipeline, directory, fresh, restored = _restored(tmp_path)
    assert pipeline._load_terminal_page(fresh, 1, directory) is not None

    fresh.pages[1].attempts = [restored, _refusal_of(restored, MODEL_TEXT)]
    fresh.pages[1].best_output = fresh.pages[1].attempts[-1]
    assert pipeline._load_terminal_page(fresh, 1, directory) is None


def test_supersession_does_not_rank_a_reloaded_credential_by_list_position(
    tmp_path: Path,
) -> None:
    """Astra round 3, P1: chronology survives the restore.

    Resume appends the restored output to a FRESH ``PageState``, so an authority
    minted before the rejection lands positionally after it. Ranking by index
    would let a reload out-vote a live refusal purely by arriving later.

    DIFFERENCE: the same two attempts in both orders.
    """
    modes = []
    for order in ("rejection-first", "restore-first"):
        _, _, fresh, restored = _restored(_fresh_dir(tmp_path, order))
        rejected = _refusal_of(restored, MODEL_TEXT)
        fresh.pages[1].attempts = (
            [rejected, restored] if order == "rejection-first" else [restored, rejected]
        )
        fresh.pages[1].best_output = rejected
        modes.append(_winning_page_output(fresh, 1).failure_mode)
    assert modes[0] is modes[1]
    assert FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED not in modes


# ---------------------------------------------------------------------------
# P2. An inner timeout arms the wedge probe.
# ---------------------------------------------------------------------------


def test_an_inner_timeout_arms_the_production_cascade_halt_predicate() -> None:
    """Astra round 3, P2: through the real adapter, real routing, real predicate.

    The trigger reads the TYPED outcome, so nothing about an exception's wording
    decides whether a wedged backend halts the document.

    DIFFERENCE: three judges behind the SAME adapter -- one raising a timeout of
    its own whose words contain no contiguous "timeout", one exceeding the
    adapter's deadline, one completing a refusal. The first two must arm the
    probe and the third must not.
    """
    import time

    from socr.pipeline import agentic
    from socr.pipeline.orchestrator import UnifiedPipeline

    class _RaisesTimeout:
        def assess(self, output, prof):
            raise TimeoutError("timed out")

    class _Slow:
        def assess(self, output, prof):
            time.sleep(10)
            return agentic.AcceptDecision(accept=True, reason="never reached")

    class _Answers:
        def assess(self, output, prof):
            return agentic.AcceptDecision(accept=False, reason="completed rejection")

    armed = {}
    for label, inner in (
        ("inner", _RaisesTimeout()),
        ("deadline", _Slow()),
        ("answered", _Answers()),
    ):
        out = PageOutput(page_num=1, text=MODEL_TEXT, status=PageStatus.SUCCESS, engine="qwen")
        judge = UnifiedPipeline._TimeoutJudge(inner, timeout_sec=0.05)
        decision = agentic.route_page(1, [_Prof()], lambda prof, page, _o=out: _o, judge)
        armed[label] = UnifiedPipeline._attempts_show_timeout(decision.attempts)

    assert armed["inner"] is True
    assert armed["deadline"] is True
    assert armed["answered"] is False


# ---------------------------------------------------------------------------
# P2. A crashed verifier is not a verdict.
# ---------------------------------------------------------------------------


def test_a_raised_verifier_is_not_a_completed_refusal_through_the_real_helper(
    tmp_path: Path,
) -> None:
    """Astra round 3, P2: the PRODUCTION combination, not a blanked outcome.

    ``_UnverifiedTableRejection._reject_unverified`` returns a negative
    ``AcceptDecision`` when the deterministic table verifier RAISES. Round 2's
    test hand-set an empty ``judge_outcome`` on the later attempt, so it never
    saw what the boundary actually stamps.

    DIFFERENCE: the same routing, with the same bytes, where the only thing that
    changes is whether the negative decision came from ``_reject_unverified`` or
    from a judge that looked and said no.
    """
    from socr.pipeline import agentic

    class _Crashes(agentic._UnverifiedTableRejection):
        def _emit_event(self, **kwargs) -> None:
            pass

        def assess(self, output, provider):
            return self._reject_unverified(output, ValueError("verifier exploded"), 1)

    class _Refuses:
        def assess(self, output, provider):
            return agentic.AcceptDecision(accept=False, reason="judge looked and said no")

    outcomes = {}
    modes = {}
    for label, judge in (("verifier-error", _Crashes()), ("refusal", _Refuses())):
        pdf = _pdf(_fresh_dir(tmp_path, label), name=f"{label}.pdf")
        state = _state(pdf)
        incumbent = state.pages[1].attempts[1]
        incumbent.table_acceptance_credential = _credential(
            state, candidate_text=MODEL_TEXT, tables=[MODEL_TABLE]
        )
        later = replace(incumbent, judge_outcome="", table_acceptance_credential=None)
        decision = agentic.route_page(1, [_Prof()], lambda prof, page, _o=later: _o, judge)
        later.audit_passed = decision.accepted
        state.pages[1].attempts.append(later)
        state.pages[1].best_output = later
        outcomes[label] = later.judge_outcome
        modes[label] = _winning_page_output(state, 1).failure_mode

    assert outcomes["verifier-error"] == JUDGE_OUTCOME_VERIFIER_ERROR
    assert outcomes["refusal"] == JUDGE_OUTCOME_COMPLETED
    assert modes["verifier-error"] is FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED
    assert modes["refusal"] is not FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED


def test_a_raised_verifier_does_not_retire_an_existing_typed_timeout() -> None:
    """A missing verdict cannot retire another missing verdict.

    The judge boundary writes onto a LIVE ``PageOutput`` a previous rung may
    already have stamped ``JUDGE_OUTCOME_TIMEOUT`` (and the table gate may have
    minted a credential against). A later rung whose verifier crashes answered
    nothing, so the timeout -- and the stand-in this ticket exists for -- stands.

    DIFFERENCE: a crashed verifier vs a real refusal, both arriving on an output
    that already carries the typed timeout and a credential.
    """
    from socr.pipeline import agentic

    class _Crashes(agentic._UnverifiedTableRejection):
        def _emit_event(self, **kwargs) -> None:
            pass

        def assess(self, output, provider):
            return self._reject_unverified(output, ValueError("verifier exploded"), 1)

    class _Refuses:
        def assess(self, output, provider):
            return agentic.AcceptDecision(accept=False, reason="judge looked and said no")

    seen = {}
    for label, judge in (("verifier-error", _Crashes()), ("refusal", _Refuses())):
        out = PageOutput(
            page_num=1,
            text=MODEL_TEXT,
            status=PageStatus.SUCCESS,
            engine="qwen",
            judge_outcome=JUDGE_OUTCOME_TIMEOUT,
            table_acceptance_credential={"candidate_sha256": sha256_text(MODEL_TEXT)},
        )
        agentic.route_page(1, [_Prof()], lambda prof, page, _o=out: _o, judge)
        seen[label] = (out.judge_outcome, out.table_acceptance_credential is not None)

    assert seen["verifier-error"] == (JUDGE_OUTCOME_TIMEOUT, True)
    assert seen["refusal"] == (JUDGE_OUTCOME_COMPLETED, False)


def test_the_verifier_error_class_from_an_earlier_rung_cannot_disguise_a_refusal() -> None:
    """The boundary compares against a PRE-CALL snapshot, not the class alone.

    ``rejection_class`` is a live field an earlier rung may already have set. If
    the exemption keyed on its VALUE, one crashed verifier would make every
    later refusal on the same output inadmissible as a verdict.

    DIFFERENCE: the same refusing judge, on an output that does and does not
    already carry ``REJECTION_VERIFIER_ERROR``.
    """
    from socr.pipeline import agentic

    class _Refuses:
        def assess(self, output, provider):
            return agentic.AcceptDecision(accept=False, reason="judge looked and said no")

    seen = {}
    for label, stale in (("stale-verifier-error", REJECTION_VERIFIER_ERROR), ("clean", None)):
        out = PageOutput(page_num=1, text=MODEL_TEXT, status=PageStatus.SUCCESS, engine="qwen")
        out.rejection_class = stale
        agentic.route_page(1, [_Prof()], lambda prof, page, _o=out: _o, _Refuses())
        seen[label] = out.judge_outcome

    assert seen["stale-verifier-error"] == JUDGE_OUTCOME_COMPLETED
    assert seen["clean"] == JUDGE_OUTCOME_COMPLETED


# ---------------------------------------------------------------------------
# Round 4. The outcome is a fact about ONE call, not about a persistent field.
# ---------------------------------------------------------------------------


def test_a_second_verifier_crash_is_still_not_a_completed_verdict(tmp_path: Path) -> None:
    """Astra round 4, P2: a same-value assignment is invisible to a snapshot.

    Round 3 detected "the verifier ran" by comparing ``output.rejection_class``
    before and after the call. That field persists across rungs, so the SECOND
    consecutive crash on the same output found the class already set, the
    boundary concluded a judge had answered, and it stamped COMPLETED and deleted
    the credential -- destroying a reading no judge ever refused.

    DIFFERENCE: the same real ``_reject_unverified`` through the same routing,
    once and then twice, on the same live output. Both iterations must look
    identical: no crash is a verdict, however many precede it.
    """
    from socr.pipeline import agentic

    class _Crashes(agentic._UnverifiedTableRejection):
        def _emit_event(self, **kwargs) -> None:
            pass

        def assess(self, output, provider):
            return self._reject_unverified(output, ValueError("verifier exploded"), 1)

    pdf = _pdf(_fresh_dir(tmp_path, "repeated"))
    state = _state(pdf)
    out = state.pages[1].attempts[1]
    out.table_acceptance_credential = _credential(
        state, candidate_text=MODEL_TEXT, tables=[MODEL_TABLE]
    )

    seen = []
    for _ in range(3):
        agentic.route_page(1, [_Prof()], lambda prof, page, _o=out: _o, _Crashes())
        seen.append(
            (
                out.judge_outcome,
                out.table_acceptance_credential is not None,
                _winning_page_output(state, 1).failure_mode,
            )
        )

    assert seen[0] == seen[1] == seen[2]
    assert seen[0] == (JUDGE_OUTCOME_TIMEOUT, True, FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED)


def test_the_missing_verdict_rides_on_the_decision_not_on_the_output() -> None:
    """The typed outcome is produced BY the call that failed to answer.

    DIFFERENCE: three judges, one output object each -- a crashed verifier, a
    real refusal, and a real refusal on an output an earlier rung already marked
    ``REJECTION_VERIFIER_ERROR``. The stale class must not disguise the refusal,
    which is what a naive "is the class set?" test would have done once the
    before/after snapshot was removed.
    """
    from socr.pipeline import agentic

    class _Crashes(agentic._UnverifiedTableRejection):
        def _emit_event(self, **kwargs) -> None:
            pass

        def assess(self, output, provider):
            return self._reject_unverified(output, ValueError("verifier exploded"), 1)

    class _Refuses:
        def assess(self, output, provider):
            return agentic.AcceptDecision(accept=False, reason="judge looked and said no")

    seen = {}
    for label, judge, stale in (
        ("verifier-error", _Crashes(), None),
        ("refusal", _Refuses(), None),
        ("refusal-after-stale-class", _Refuses(), REJECTION_VERIFIER_ERROR),
    ):
        out = PageOutput(page_num=1, text=MODEL_TEXT, status=PageStatus.SUCCESS, engine="qwen")
        out.rejection_class = stale
        agentic.route_page(1, [_Prof()], lambda prof, page, _o=out: _o, judge)
        seen[label] = out.judge_outcome

    assert seen["verifier-error"] == JUDGE_OUTCOME_VERIFIER_ERROR
    assert seen["refusal"] == JUDGE_OUTCOME_COMPLETED
    assert seen["refusal-after-stale-class"] == JUDGE_OUTCOME_COMPLETED


def test_a_real_timeout_after_a_verifier_crash_is_typed_as_a_timeout() -> None:
    """The two missing-verdict outcomes are ordered by what actually happened.

    A verifier crash records its own kind; a judge that then times out on the
    same bytes records the timeout, which is the only outcome that licenses the
    credentialed stand-in.

    DIFFERENCE: the same two calls in sequence, checked after each.
    """
    from socr.pipeline import agentic
    from socr.pipeline.orchestrator import UnifiedPipeline

    class _Crashes(agentic._UnverifiedTableRejection):
        def _emit_event(self, **kwargs) -> None:
            pass

        def assess(self, output, provider):
            return self._reject_unverified(output, ValueError("verifier exploded"), 1)

    class _TimesOut:
        def assess(self, output, provider):
            raise TimeoutError("timed out")

    out = PageOutput(page_num=1, text=MODEL_TEXT, status=PageStatus.SUCCESS, engine="qwen")
    agentic.route_page(1, [_Prof()], lambda prof, page, _o=out: _o, _Crashes())
    after_crash = out.judge_outcome
    agentic.route_page(
        1,
        [_Prof()],
        lambda prof, page, _o=out: _o,
        UnifiedPipeline._TimeoutJudge(_TimesOut(), timeout_sec=1.0),
    )
    assert after_crash == JUDGE_OUTCOME_VERIFIER_ERROR
    assert out.judge_outcome == JUDGE_OUTCOME_TIMEOUT


def test_a_foreign_candidate_digest_cannot_authorize_a_fresh_body(tmp_path: Path) -> None:
    """``reading_digests`` widens refusals; it never widens ADMISSION.

    The credential's own digests are trusted only to decide which refusals are
    about this reading. Fresh admission still recomputes the candidate digest
    from the bytes in hand, so a credential naming foreign bytes withholds the
    page rather than vouching for whatever is there.

    DIFFERENCE: the honest credential vs the same credential with a foreign
    ``candidate_sha256``.
    """
    from socr.core.manifest import reading_digests

    modes = {}
    for label, foreign in (("honest", None), ("foreign", sha256_text("a different candidate"))):
        pdf = _pdf(_fresh_dir(tmp_path, label), name=f"{label}.pdf")
        state = _state(pdf)
        out = state.pages[1].attempts[1]
        cred = _credential(state, candidate_text=MODEL_TEXT, tables=[MODEL_TABLE])
        if foreign:
            cred["candidate_sha256"] = foreign
            assert foreign in reading_digests(replace(out, table_acceptance_credential=cred))
        out.table_acceptance_credential = cred
        modes[label] = _winning_page_output(state, 1).failure_mode

    assert modes["honest"] is FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED
    assert modes["foreign"] is not FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED
