# The non-vacuity trap is systemic, not a slip

Recorded 2026-08-20, after it appeared for the **fourth** time in one night, in code
written by two different agents on four different PRs.

## The trap

The rule is that a new regression test must FAIL against the baseline commit, on an
assertion about behaviour that already exists there. That is what distinguishes a real
guard from a decorative one — and it is the check that caught PR #250, whose own test
asserted the defective behaviour and passed.

The trap is that a test asserting on anything the change **introduces** — a new
attribute, a new kwarg, a new config field — does not fail behaviourally at the
baseline. It raises:

    AttributeError: 'PageState' object has no attribute 'fabricated_image_refs'
    TypeError: reconstruct_table_regions() got an unexpected keyword argument 'rejections'

The test *is* red at the baseline, so a careless report says "fails at baseline, passes
on branch, proof complete". It counts as a failure and reads as a proof. It is neither:
it proves only that the symbol is new, which nobody doubted.

## Four occurrences

| PR | form | caught by |
|---|---|---|
| #252 round 2 | `state.pages[1].fabricated_image_refs == 0` in three reverse guards | author, before pushing |
| #254 round 2 | `reconstruct_table_regions(page, rejections=…)` — kwarg absent at base | author, before pushing |
| #252 round 1 | a sixth test unit-testing a new helper | author, deleted it deliberately |
| **#252 round 3** | `state.pages[1].fabricated_image_refs` in a setup assert | **the reviewer — the author claimed no AttributeError, falsely** |

The first three are a competent author catching itself. The fourth is the one that
matters: the same author, having twice diagnosed and fixed this exact trap, reintroduced
it and then **asserted in the PR body that it had not happened**. Not dishonesty — the
count matched (5 failed / 5 passed), so the summary looked consistent with the evidence
it was summarising. Nobody re-read which failure was which.

## Why the gate held anyway

Because the reviewer re-ran the baseline instead of reading the author's summary of it.
`bin/verify_citations.py` does the same thing for triage evidence, and the review board
requires an access proof rather than a claim. The pattern is the same one this whole run
was built on: **a claim about evidence is not evidence**, no matter how competent the
claimant, and the check has to be mechanical.

Note the finding was scoped correctly, which is what made it useful rather than
alarming: the reviewer confirmed the *code* was sound and the ordering genuinely pinned,
and voided only the baseline proof as stated. A blunter reviewer would have rejected the
whole PR and cost a round.

## The convention to adopt

1. In any assertion that runs against the baseline, reach for
   `getattr(obj, "new_field", default)` rather than a direct attribute access.
2. Never call a function with a kwarg the change introduces, in a test that must also
   run at the base.
3. When reporting non-vacuity, paste the failure **lines**, not just the counts. `5
   failed` is not a proof; `AssertionError: 'i.imgur.com' not in …` is. A count cannot
   distinguish a behavioural failure from a symbol-absence error, which is precisely how
   this survived to round 3.
4. The reviewer, not the author, re-runs the baseline.

Point 3 is the cheap one and would have caught all four.

---

## Addendum — a fifth finding, about #205's own framing

Recorded when the successor code owner fixed #253.

#205 says the TR-3 geometry hard-fail "is detected on 62/245 table pages and surfaced
nowhere", and the binding scope I gave both code owners was "do not key page or document
status on it". Measuring at the branch's base `7696719` before writing the scope guard,
the successor found that **document status is already keyed on the TR-3 flag** through
pre-existing paths — the D3 floor and `native_fallback`:

| | flagged | clean |
|---|---|---|
| non-agentic | `AUDIT_FAILED` | `SUCCESS` |
| agentic | `ERROR` | `AUDIT_FAILED` |

So "surfaced nowhere" is true of the **analyze-time detection**, and false of the flag in
general. Nothing #205 controls does the keying, but something does.

This mattered concretely. The obvious way to write the scope guard — assert that a
flagged document and a clean one end in the same state — would have asserted something
untrue and failed at the base for the right reason and the wrong cause. The guard instead
pins the measured base values and shows they are byte-identical on the branch, which
proves what actually needs proving: that this change moves nothing.

It also matters for the morning. #205's remaining steps 2 and 3 are written on the
premise that the signal is inert. It is not entirely inert, and the hand-judgement of the
62-page set should be designed knowing that.

Same lesson as the rest of this file: the framing in an issue — including an issue the
owner wrote carefully and revised — is a claim, and a claim is not a measurement.

---

## Correction to the addendum, and the sixth trap: a pin measured in one environment

The addendum above states, as a finding, that document status is already keyed on the
TR-3 flag in all four cells. **A second reviewer, building its own fixture and stubs,
could not reproduce that in the non-agentic cells** — it saw the move in both agentic
cells (`ERROR` vs `AUDIT_FAILED`) but `SUCCESS`/`SUCCESS` non-agentic, where the original
fixture yielded `AUDIT_FAILED`/`SUCCESS`.

So the claim is **fixture-dependent**, and I recorded it more firmly than the evidence
supports. It is true that the flag *can* already move document status; it is not
established that it does so universally. The design conclusion drawn from it — write the
guard differentially rather than as flagged-vs-clean equality — survives either way, and
is if anything better supported: a claim that changes with the fixture is exactly the
kind you must not pin.

### What this cost

PR #253 was merged and **reverted from main**. Its scope guard pinned an absolute outcome
tuple measured locally, where a provider exists. In CI, which has neither ollama nor a
provider, the provider-dependent machinery (the D3 floor, `native_fallback`) never fires,
nothing moved, and the pin — which asserted movement — failed. It passed locally on 1850
tests and went red in CI. The merge happened while that check was red.

This is the same family as the five above, in a new costume: **evidence that is true in
the environment where it was gathered and false in the one that matters.** A symbol-
absence failure looks like a behavioural failure; a locally-measured status pin looks
like a behavioural invariant. Both are red at the baseline, both read as proof.

### The fix, and the fault

The redo (#258) was told the implementation was at fault. It measured instead: it ran
`process()` over a flagged page twice, with the emission enabled and disabled, across all
four (agentic × provider) cells, and found the outcome tuple, the event list and the OCR
calls identical. **The implementation was already inert. The test was wrong.** A fresh
reviewer confirmed that independently, on its own fixture, in its own worktrees, having
found the author's tree mid-merge and therefore refused to trust anything from it.

The guard is now differential — two runs differing only in the flag, or only in whether
the emission happens — with no absolute outcome tuple pinned anywhere, parametrised over
both provider states. It is green in CI, the environment that killed its predecessor.

### The rule this adds

Never pin an absolute outcome measured in one environment. Pin a **difference** measured
in whatever environment the test happens to run in, and parametrise over the axis that
differs between local and CI — here, the provider. And the standing rule from the top of
this file applies to me as well as to the agents: I wrote that addendum from one agent's
measurement without a second fixture, which is the same "a claim about evidence is not
evidence" error, one level up.
