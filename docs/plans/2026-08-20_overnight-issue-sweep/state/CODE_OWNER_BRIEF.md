# CODE OWNER BRIEF — Stream E, wave 4

**You are the single code owner for the whole night.** No other agent writes code.
This is not caution for its own sake: the candidate fixes all write
`orchestrator.py::_phase_agentic` and `manifest.py::_winning_page_output`, or
`reconstruct.py` and `born_digital.py`. Parallel branches from one baseline would
collide inside a 900-line function and can silently undo each other. That is the
exact shape of the #250 defect this repo shipped today.

## Stacked branches, never parallel

Each PR is based on its **predecessor's branch**, not on the baseline:

    E1 (#161)            base: main_sha 53b0637        branch: fix/161-resume-ledger-audit
    E2 (#225)            base: E1's branch             branch: fix/225-phantom-image-urls
    E3 (#205 surfacing)  base: E2's branch             branch: fix/205-tr3-auditevent
    E4 (#147)            base: E3's branch             branch: fix/147-landscape-axis
    E5 (#195+197+198)    base: E4's branch             branch: fix/195-197-198-destruction-check
    E6 (#222)            base: E5's branch             branch: fix/222-probe-interface
    E7 (#221+#227)       base: E6's branch             branch: fix/221-227-cascade-latch

Work in ONE worktree, one branch at a time, moving forward through the stack.

## Per-ticket loop

implement → independent review (a different model) → revise → push → open PR →
CI to a terminal state → final verify. Terminal CI states are `GREEN`, `FAILED`,
`TIMED-OUT`, `NO-CHECKS`. **All four release the successor.** None blocks the
report. A red CI is a fact for the morning, not a reason to stall the stack.

## Non-vacuity proof — the thing that is actually checked

"CI green" is cheatable: delete the assertion, or write a test that exercises a
symbol your fix introduces so the reverted version fails with `ImportError`
instead of a behavioural assertion. Both look green. Neither proves anything.

So for every ticket, record in the PR body:

1. the new test run **against `main_sha`** — it must **FAIL**, and fail on an
   assertion about behaviour that **already exists** at `main_sha`, not on an
   ImportError for something your fix added;
2. the same test run **on your branch** — it must **PASS**;
3. **both raw outputs pasted**, not summarised.

Run (1) in the pristine baseline worktree
`/Users/rubenffuertes/repos/.worktrees/socr-night-base` — leave it clean when you
are done, it is the run's reference tree.

## Non-negotiables

- `export PYTHONPATH=<your-worktree>/src` and pass
  `bin/isolation_canary.sh <your-worktree>` **in every worktree**, every time.
  Paste the output in the PR body.
- **Lint gate is exactly `uvx ruff@0.16.0 format --check .`** Do not use
  `~/venvs/socr/bin/ruff` — it is older, it silently reports clean on files CI
  rejects, and that gap turned `main` red and blocked four PRs.
- **CI has no ollama and no provider.** A test driving `_phase_agentic` or
  `process()` in agentic mode must patch `_available_engines_for_agentic` or it
  passes here and fails in CI.
- Stage files **by name**. Never `git add -A`. No `Co-Authored-By` trailer.
- **Never `gh pr merge`. Never `git push --force`.** PRs are proposed.
- Never touch `/Users/rubenffuertes/repos/tools/socr` as a working tree.
- Check `state/ABORT` before every push. If it exists, stop pushing and write up
  where you got to.
- **No magic thresholds.** Derive from data or use a named, documented constant.
- **Cardinal rule: no silent content loss.** A wrong or dropped number is worse
  than a missing one, and a failure must surface at page, document, metadata AND
  CLI level — not just one of them.

## If a ticket will not land

Record it `SKIPPED` with the reason and move to the next one. A half-built fix
pushed at 04:00 is worse than an honest skip. E7 has an explicit gate: if the
timeout stub cannot carry backend identity (the #159 dependency), skip it and say
so rather than shipping half of a two-part change that #227 warns makes behaviour
worse on its own.
