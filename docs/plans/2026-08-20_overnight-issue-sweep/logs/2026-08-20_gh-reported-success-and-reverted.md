# `gh pr edit` returned 0, printed the URL, and did not persist the edit

Found 2026-08-20 ~10:50 by the successor code owner, while adding a banner it had
previously reported adding.

## What happened

Two separate PR body edits silently reverted:

- **#252** — a `SUPERSEDED` banner over a stale round-1 proof section. Reported as
  applied; not present. Caught by the reviewer (Kimi) during round-4 verification.
- **#255** — the **entire round-2 body rewrite** reverted to the round-1 text. Found
  only because the author went back to re-apply #252's banner and checked.

In both cases `gh pr edit` **exited 0 and printed the PR URL**. `#253`'s edits from the
same period survived. Cause not established from here — a race with another writer, or
an interaction with the rebase tooling. What is established is the failure mode.

## Why this one matters more than it looks

PR bodies are where this run's **non-vacuity proofs** live: the new test run against the
baseline, the raw outputs, the honest account of which surfacing levels are covered. A
body that silently reverts to an earlier round does not merely lose formatting — it
restores a **stale proof for a commit that is no longer the head**, and it reads as
current. That is the same shape as every other trap this run has hit: something that
looks like evidence, isn't, and gives no error.

`#254` shows the milder version: its body is still the round-1 proof against `53b0637`
with `5 failed, 3 passed`, while the branch has moved twice. Its round-2 evidence exists
only as a PR comment. Nothing was lost, but a reader trusting the body would be reading
a proof about a commit that is not the head.

## What was done

Every one of the four PR bodies now carries a verified banner giving the current head,
the current base, the reviewer, and an explicit warning that **any SHA quoted further
down predates the rebase**, with the round comments authoritative over the body where
they disagree. Each was applied and then **read back** — all four confirmed present.

## The rule

**An exit code is not evidence that a remote mutation happened.** Read it back.

This run already applied that rule in three places and got value from it each time:
`bin/verify_citations.py` re-resolves every citation rather than trusting the model that
wrote it; `bin/apply_tracker_actions.sh` re-reads each issue after writing and records the
observed state; the review board requires an access proof rather than a claim. The gap
was that nobody had extended it to `gh pr edit` — the one remote write whose success we
took on faith, and the one that silently failed.

The code owner adopted the read-back rule for the rest of the run on its own initiative
after finding this, which is the right response.
