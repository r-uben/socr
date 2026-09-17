# GH-671 — a truncated best_output must not win by short-circuit

## Defect

`_select_page_output_tagged` (`src/socr/core/manifest.py`) returns early on any
non-native `best_output` with `audit_passed=True`, gated only on
`native_distrusted` / `native_text_shredded`. S1's own truncation filtering
(`_truncated_grid_reading_ids` / `structure_class_truncated_engines` /
`_strict_grid_authored_pool`'s drop) runs later, under
`_reaches_structure_class_branch`, and was never consulted in the
short-circuit region. A judge-cleared TRUNCATED model that happened to be
`best_output` shipped unchallenged, and a COMPLETE alternative reading in the
wider pool was never considered — a silent truncation win over a complete
reading (the live #645 bulletin p2 shape, here with the truncated candidate
promoted to `best_output` itself).

#665's pin (`test_cross_pool_truncated_strict_loses_to_complete_wide_pool_only`)
deliberately keeps `best_output.audit_passed=False` so S1 is reachable at all
— it structurally cannot exercise this short-circuit.

## Fix (narrow option, per ticket)

Made a truncated `best_output` ineligible for the `PASSING_BEST_OUTPUT`
short-circuit, mirrored in both places that gate on the same three
preconditions so they cannot drift apart again:

- `_select_page_output_tagged`'s own early return (the short-circuit itself).
- `_reaches_structure_class_branch`'s own entry gate (so falling through
  the short-circuit actually reaches S1, where `_strict_grid_authored_pool`
  already drops a truncated candidate in favour of a complete one from the
  wider pool — this machinery pre-existed from #665/TICKET-A2 and needed no
  changes).

Both add a third disjunct, `best_output_truncated = id(p.best_output) in
_truncated_grid_reading_ids(p)`. `_truncated_grid_reading_ids` already returns
an empty set whenever there is nothing to compare against (a single
candidate, or every candidate truncates), so a genuinely-only-candidate
truncated `best_output` still ships via the short-circuit exactly as before —
TICKET-A2's own "if it is the only candidate, it still ships, flagged" clause
is preserved.

Did NOT touch the broader question at `orchestrator.py:200` (whether a
native-lane verdict should outrank a passing winner) — out of scope per the
ticket.

## Files changed

- `src/socr/core/manifest.py` — two mirrored edits (see above).
- `tests/test_gh671_truncated_best_output_short_circuit.py` — new, hermetic
  pin at the real caller (`_select_page_output_tagged`).

## Evidence

**New tests** (`tests/test_gh671_truncated_best_output_short_circuit.py`):
2 passed.

- `test_truncated_best_output_does_not_win_by_short_circuit`: best_output =
  judge-cleared truncated qwen (`audit_passed=True`), a complete gemini
  reading only in `p.attempts` (`audit_passed=False`, wide-pool-only). Winner
  is the complete reading, provenance is not a bare `PASSING_BEST_OUTPUT`, and
  a `candidate_truncated` audit event fired for `qwen`.
- `test_untruncated_passing_best_output_still_takes_the_short_circuit`: both
  directions — an ordinary clean, non-truncated `best_output` still takes
  `PASSING_BEST_OUTPUT` unchanged, with no `candidate_truncated` event. This
  guards against a fix that routes every clean model page down the long S1
  path.

**Full suite, measured**:
- Baseline (`git archive HEAD` before the fix, frozen to `/tmp/gh671-baseline`,
  `PYTHONPATH=.../src`, no `.git`): 5635 passed, 1 skipped, 4 xfailed
  (5640 collected). The 1 skip (`test_gh592_scoped_positional_emission.py`:
  "origin/main ref not present") is a `git archive`-only artifact — a bare
  archive has no `.git`, so that test's own git-ref lookup skips; the same
  test runs (not skipped) in a real worktree, both before and after the fix.
- Post-fix (`/tmp/wt-671`, the real worktree, same `PYTHONPATH` scheme):
  5638 passed, 0 skipped, 4 xfailed (5642 collected).
- Reconciled: 5638 = 5635 (baseline passed) + 1 (the `test_gh592` test, which
  is a real pass in a worktree, invisible as a skip only in the archive
  baseline) + 2 (this ticket's new tests). **Blast radius: zero existing
  tests changed behaviour.** No `FAILED` in either run.

**Mutation round** (required by the ticket): copied `src/`, `tests/`, and
`pyproject.toml` (its `pythonpath = ["src"]` shadows external `PYTHONPATH`) to
`/tmp/gh671-mutant`, reverted both edits (uncapped anchor `count() == 1`
asserted before each revert) back to the pre-fix early-return / gate.
Canary — `os.path.realpath(socr.__file__) ==
os.path.realpath('/tmp/gh671-mutant/src/socr/__init__.py')` inside the same
Python process the mutant pytest run used — confirmed the mutant source, not
the editable install, was under test.
- `test_truncated_best_output_does_not_win_by_short_circuit`: **reddened**
  (`AssertionError: assert 'qwen' == 'gemini'` — the truncated best_output
  shipped, exactly the pre-fix defect).
- `test_untruncated_passing_best_output_still_takes_the_short_circuit`:
  **stayed green** (1 failed, 1 passed).

Cleaned up `/tmp/gh671-mutant` and `/tmp/gh671-baseline` after measuring.

## Lint

`uvx ruff@0.16.0 format --check .` — clean (737 files already formatted).

## Deviation / follow-up

None. The narrow option (2) was sufficient; option 1 (reordering the main
selection path) was not needed. `orchestrator.py:194`'s prior "reordering
around `PASSING_BEST_OUTPUT` has flipped SUCCESS/AUDIT_FAILED before" caution
did not apply here — nothing was reordered, only a third disjunct added to an
existing gate condition, mirrored in the one other place that gate is
duplicated.
