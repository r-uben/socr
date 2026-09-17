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

**Both sites, not scope creep.** `_reaches_structure_class_branch`'s own
docstring commits to "mirroring EVERY precondition [the S1] branch sits
behind, in the SAME order" and records the fallout the one time this pair
drifted (#269 BLOCKING 2): a page's real winner shipped via one branch while
a document-level bucket, reading only the OTHER function, believed a
different branch had fired (a PROSE page landed in
`structure_class_model_pages` and flipped its document to AUDIT_FAILED). This
ticket's defect is a THIRD case of the same shape: patching only the
selector's short-circuit would leave `_reaches_structure_class_branch`
returning `False` for the identical page (same three-condition gate, same
missing disjunct), so a truncated `best_output` would fall through the
short-circuit yet still bounce off the S1 entry gate one line later. Editing
both sites is the minimum change that keeps the invariant the first function's
own docstring commits to, not a widening of the narrow option.

**`id()` keying safety.** Both edits use
`id(p.best_output) in _truncated_grid_reading_ids(p)`, matching
`_truncated_grid_reading_ids`'s own dedup-by-`id()` construction. Identity
keying is fragile in general (a `dataclasses.replace()` copy has a new `id()`
even with identical content — this repo does use `replace()` elsewhere, e.g.
#259's substitution and the `math_hybrid`/D3/`flagged_model` branches in this
same function). It is safe here specifically because `p.best_output` is never
reassigned or replaced anywhere between the two reads: grepping
`src/socr/core/manifest.py` for `best_output\s*=` finds no assignment to
`p.best_output` inside the selection functions (`replace()` calls build NEW
`PageOutput`s for the RETURN value; they never write back to `p.best_output`
on the stored `PageState`). Both `id()` lookups — one in the selector's own
gate, one in `_reaches_structure_class_branch` if reached — read the SAME
attribute off the SAME `p` object within a single `_select_page_output_tagged`
call, so they always see the same object and the same `id()`. Pinned directly
by `test_the_two_mirrored_gates_agree_on_the_truncated_case`, rather than
resting on this argument alone.

## Files changed

- `src/socr/core/manifest.py` — two mirrored edits (see above).
- `tests/test_gh671_truncated_best_output_short_circuit.py` — new, hermetic
  pin at the real caller (`_select_page_output_tagged`), plus the
  gate-agreement pin.

## Evidence

**New tests** (`tests/test_gh671_truncated_best_output_short_circuit.py`):
3 passed.

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
- `test_the_two_mirrored_gates_agree_on_the_truncated_case`: on the same
  truncated-best_output page, `_select_page_output_tagged` does not
  short-circuit AND `_reaches_structure_class_branch` independently returns
  `True` — the two duplicated gates agree. Catches a future drift between the
  two copies before it reaches production (the #269 BLOCKING 2 shape).

**Targeted blast-radius check** (before the full-suite run below): the
orchestrator pre-measured 11 test files mentioning `PASSING_BEST_OUTPUT` on
`main@66d5d7c` (`test_p6_disposition_finalization.py`,
`test_s1_structure_class_winner_gh_reachability.py`,
`test_p6_disposition_persistence.py`, `test_r7_winner_kind_tags.py`,
`test_p6_stage_c_bucket_contract.py`, `test_p6_stage_ab_difference.py`, plus
five more single-mention files) and flagged
`test_s1_structure_class_winner_gh_reachability.py` (50 tests, including
`_reaches_structure_class_branch`'s own #269 BLOCKING-2 regression suite) as
the one most likely to encode an existing assumption about this gate. Ran it
alone and the other ten files together: **all 50 + 175 = 225 tests pass
unchanged**, including
`test_ordinary_already_passing_non_native_winner_is_not_an_s1_event` (an
ordinary non-truncated non-native `best_output` still takes the short-circuit
via `assert winner is model_attempt`) and every `_reaches_structure_class_branch`
regression case named in that file's own BLOCKING-2 section. None of the 11
moved, matching the narrow option's prediction (only a page with a genuinely
truncated `best_output` is affected, and none of these fixtures constructs
one).

**Full suite, measured** (re-run after the third pinning test was added, from
a fresh `git archive 66d5d7c` — the real pre-fix commit, not the fix commit —
so the baseline is genuinely pre-fix):
- Baseline (frozen to `/tmp/gh671-baseline2`, no `.git`): 5635 passed,
  1 skipped, 4 xfailed (5640 collected). The 1 skip
  (`test_gh592_scoped_positional_emission.py`: "origin/main ref not present")
  is a `git archive`-only artifact — a bare archive has no `.git`, so that
  test's own git-ref lookup skips; the same test runs (not skipped) in a real
  worktree, both before and after the fix.
- Post-fix (`/tmp/wt-671`, the real worktree, same `PYTHONPATH` scheme):
  5639 passed, 0 skipped, 4 xfailed (5643 collected).
- Reconciled: 5639 = 5635 (baseline passed) + 1 (the `test_gh592` test, which
  is a real pass in a worktree, invisible as a skip only in the archive
  baseline) + 3 (this ticket's new tests). **Blast radius: zero existing
  tests changed behaviour.** No `FAILED` in either run.

**Mutation round** (required by the ticket; re-run after the third pinning
test was added): copied `src/`, `tests/`, and `pyproject.toml` (its
`pythonpath = ["src"]` shadows external `PYTHONPATH`) to
`/tmp/gh671-mutant2`, reverted both edits (uncapped anchor `count() == 1`
asserted before each revert) back to the pre-fix early-return / gate.
Canary — `os.path.realpath(socr.__file__) ==
os.path.realpath('/tmp/gh671-mutant2/src/socr/__init__.py')` inside the same
Python process the mutant pytest run used — confirmed the mutant source, not
the editable install, was under test.
- `test_truncated_best_output_does_not_win_by_short_circuit`: **reddened**
  (`AssertionError: assert 'qwen' == 'gemini'` — the truncated best_output
  shipped, exactly the pre-fix defect).
- `test_the_two_mirrored_gates_agree_on_the_truncated_case`: **reddened**
  (`AssertionError: ... PASSING_BEST_OUTPUT != PASSING_BEST_OUTPUT` — with
  both edits reverted, the selector DOES take the short-circuit again).
- `test_untruncated_passing_best_output_still_takes_the_short_circuit`:
  **stayed green** (2 failed, 1 passed).

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
