# Stacked PRs get no CI at all — a structural conflict in the plan

Found 2026-08-20 ~04:30 while checking PR #252.

## What happens

`.github/workflows/ci.yml` at `main_sha` is triggered by:

```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
```

`pull_request: branches: [main]` filters on the **base** branch. Tonight's stack is
built exactly the other way:

| PR | head | base | CI |
|---|---|---|---|
| #251 | `fix/161-resume-ledger-audit` | `main` | **test (3.11) pass, typecheck pass** |
| #252 | `fix/225-phantom-image-urls` | `fix/161-resume-ledger-audit` | **no test job ran** |
| #253+ | each on its predecessor | not `main` | will also not run |

So only the bottom of the stack is CI-verified. Every PR above it reports
`NO-CHECKS` — which TICKETS.md defines as a terminal state that releases the
successor, so the night was not blocked, but the plan's "CI to a terminal state"
and its "stacked branches, each PR based on its predecessor" mandates cannot both
deliver a green tick. The panel merged those two decisions without noticing they
collide.

## Why the stack is still right

The reason for stacking was concrete: `#161`, `#205` and `#225` all write
`orchestrator.py::_phase_agentic`, and parallel branches from one baseline would
collide inside a 900-line function and could silently undo each other — the #250
shape. That risk is real and CI would not have caught it either. Unstacking to buy
a CI tick would trade a verified risk for an unverified one.

## The compensating control that is actually in place

The code owner runs the full suite locally in the worktree, with the isolation
canary passed first, and pastes the raw counts into each PR body. For #252:

    PYTHONPATH=<worktree>/src ~/venvs/socr/bin/pytest -q
    1835 passed, 3 xfailed, 5 warnings in 184.33s

plus the non-vacuity pair (new test FAILS at `main_sha`, PASSES on the branch, both
raw outputs pasted) and `uvx ruff@0.16.0 format --check .` clean.

That is stronger evidence than a CI tick on its own — but it is *this machine's*
evidence, not GitHub's, and the owner should treat it that way.

## For the morning — three options, cheapest first

1. **Merge bottom-up.** Merge #251 to `main`; #252's base auto-retargets to `main`
   and CI runs on it for real. Repeat up the stack. No repo change needed. This is
   the recommended path.
2. **Widen the trigger** so PRs into any branch run CI:
   `pull_request:` with no `branches:` filter (or add the `fix/**` pattern). One
   line, and it makes every future stacked PR self-verifying.
3. Rebase each PR onto `main` individually — only safe if the collisions the stack
   exists to prevent are re-checked by hand, so this is the worst of the three.

---

## Addendum — a conflicting PR runs no CI at all, and nothing says so

Found 2026-08-20 ~13:20 by the #253 redo author, after #256 had supposedly fixed CI for
every PR.

`#256` removed the base-branch filter, so a stacked PR now self-verifies. But there is a
second, quieter hole: **if a PR has a merge conflict with its base, GitHub cannot compute
the merge commit, so no `pull_request` workflow runs at all.** The checks tab then shows
only the advisory reviewers (CodeRabbit, cubic) — which reads exactly like "nothing
failed". `gh pr checks` does not distinguish "all green" from "nothing ran".

That is the same shape as everything else this run kept hitting: absence of a signal
presented as a passing signal. And it is more dangerous than the original CI gap, because
the original produced an obvious `NO-CHECKS`, whereas this produces a checks list with
green ticks in it.

The redo hit it when `main` moved to `49e491b` (#254, #255) mid-review and its branch
conflicted. It merged `main` into the branch rather than rebasing (a rebase would have
needed a force-push) and CI then ran.

### Verified for this run's merges

All three PRs merged after the revert were checked against their exact merged heads:

    fix/195-197-198-destruction-check  pull_request completed success  c306d41
    fix/222-probe-interface            pull_request completed success  6e40e6b
    fix/205-tr3-surfacing-inert        pull_request completed success  013c1b2

So none of them merged un-CI'd. Note the query that shows this: `gh run list --branch
<branch> --workflow CI`. **`gh run list --commit <branch-head>` returns nothing**, because
for `pull_request` events the run is recorded against the synthesised merge commit, not
the branch head — a false alarm that looks identical to "CI never ran".

### The check to actually use

Before merging, confirm a CI run exists **for the current head**, not merely that the
checks list has no failures:

    gh run list --repo <repo> --branch <branch> --workflow CI --limit 3 \
      --json status,conclusion,headSha

If the newest run's `headSha` is not the PR's head, CI has not seen what you are about to
merge — regardless of how green the checks tab looks.
