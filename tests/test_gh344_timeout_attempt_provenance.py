"""GH-344: every ProviderAttempt records which provider it was.

The timeout branch in ``route_page`` constructed its attempt without
``provider_id`` / ``model`` / ``backend``. Budget skip, provider raise, judge
raise and the accepted path all populate them, so a timed-out rung was the one
journal entry that could not say WHICH provider timed out — exactly the entry
an operator reads first when a run stalls.

Pinned as a DIFFERENCE against a sibling branch rather than against literals:
the timeout attempt must carry the same provenance the budget-skip attempt
carries for the same profile, so neither can drift alone.
"""

from __future__ import annotations

from socr.core.providers import PROFILE_QWEN_CLOUD, PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.pipeline.agentic import route_page


def _judge_rejects(page_num, output, profile=None, **kwargs):
    """Never reached on the timeout / budget-skip paths, but must not be a
    landmine if the fixture grows: the previous version imported a
    ``JudgeDecision`` symbol that does not exist."""
    raise AssertionError("judge must not be reached on a timeout or budget skip")


def _timeout_attempt():
    """Drive route_page's timeout branch and return the recorded attempt."""

    def _hangs(profile, page_num):
        import time

        time.sleep(5)
        return PageOutput(
            page_num=page_num,
            text="never",
            status=PageStatus.SUCCESS,
            engine=profile.engine.value,
            audit_passed=True,
        )

    decision = route_page(
        1,
        [PROFILE_QWEN_LOCAL],
        _hangs,
        _judge_rejects,
        provider_timeout={PROFILE_QWEN_LOCAL.engine: 0.05},
    )
    timeouts = [a for a in decision.attempts if a.reason == "provider timeout"]
    assert timeouts, "fixture did not reach the timeout branch"
    return timeouts[0]


def _budget_skip_attempt():
    """The sibling branch that already recorded provenance."""

    def _never_called(profile, page_num):  # pragma: no cover - must not run
        raise AssertionError("provider must be skipped on budget")

    paid = PROFILE_QWEN_LOCAL.__class__(
        **{**PROFILE_QWEN_LOCAL.__dict__, "cost_per_page_usd": 10.0}
    )
    decision = route_page(1, [paid], _never_called, _judge_rejects, remaining_budget=0.0)
    skips = [a for a in decision.attempts if a.reason == "budget exceeded"]
    assert skips, "fixture did not reach the budget-skip branch"
    return skips[0], paid


class TestEveryAttemptSaysWhichProviderItWas:
    def test_the_timeout_attempt_records_the_provider(self) -> None:
        attempt = _timeout_attempt()

        assert attempt.provider_id == PROFILE_QWEN_LOCAL.id
        assert attempt.model == PROFILE_QWEN_LOCAL.model
        assert attempt.backend == PROFILE_QWEN_LOCAL.backend

    def test_timeout_and_budget_skip_record_the_same_way(self) -> None:
        """Difference pin against a sibling, not against literals: if the two
        branches ever disagree about what identifies a provider, that is the
        bug, whichever one moved."""
        timeout = _timeout_attempt()
        skip, paid = _budget_skip_attempt()

        assert (
            (timeout.provider_id, timeout.model, timeout.backend)
            == (
                skip.provider_id,
                skip.model,
                skip.backend,
            )
            == (paid.id, paid.model, paid.backend)
        )

    def test_a_timed_out_attempt_is_still_not_accepted(self) -> None:
        """Inertness: recording provenance must not make a timeout look like a
        usable result."""
        attempt = _timeout_attempt()

        assert attempt.accepted is False
        assert attempt.cost_usd == 0.0


class TestSharedEngineProfilesAreDisambiguated:
    """GH-344 AC #2, and the canary GH-227 actually needs.

    ``PROFILE_QWEN_LOCAL`` and ``PROFILE_QWEN_CLOUD`` share ``EngineType.QWEN``,
    so ``prof.engine`` cannot say which of them timed out -- only ``prof.id``
    can. Without two Qwen rungs on one ladder that disambiguation is argued
    rather than measured.
    """

    def test_the_cloud_rung_is_named_when_it_is_the_one_that_times_out(self) -> None:
        seen: list = []

        def _local_ok_cloud_hangs(profile, page_num):
            seen.append(profile.id)
            if profile.id == PROFILE_QWEN_CLOUD.id:
                import time

                time.sleep(5)
            return PageOutput(
                page_num=page_num,
                text="local output the judge rejects",
                status=PageStatus.SUCCESS,
                engine=profile.engine.value,
                audit_passed=True,
            )

        def _always_escalate(page_num, output, profile=None, **kwargs):
            class _D:
                accepted = False
                reason = "escalate"
                raw_verdict = None

            return _D()

        decision = route_page(
            1,
            [PROFILE_QWEN_LOCAL, PROFILE_QWEN_CLOUD],
            _local_ok_cloud_hangs,
            _always_escalate,
            provider_timeout={PROFILE_QWEN_LOCAL.engine: 0.05},
        )

        timeouts = [a for a in decision.attempts if a.reason == "provider timeout"]
        assert timeouts, f"cloud rung never timed out; providers seen: {seen}"
        assert timeouts[0].provider_id == PROFILE_QWEN_CLOUD.id, (
            "the journal named the wrong Qwen rung; engine alone cannot "
            "disambiguate two profiles sharing EngineType.QWEN"
        )
        assert timeouts[0].provider_id != PROFILE_QWEN_LOCAL.id
        assert PROFILE_QWEN_LOCAL.engine == PROFILE_QWEN_CLOUD.engine, (
            "precondition: if these stop sharing an engine this pins nothing"
        )
