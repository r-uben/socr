# GH-817 — the `fabricated_image_refs` (GH-225) demotion must survive resume

## What changed

`#225`'s guard (`_guard_fabricated_image_refs`) demotes the DOCUMENT to
`AUDIT_FAILED` via the `fabricated_ref_pages` orthogonal bucket in
`_phase_assemble` (`orchestrator.py:727`, keyed off `PageState.
fabricated_image_refs`), but that counter was never written to the sidecar
meta block and never restored in `_restore_terminal_page_state`. A resumed
run that restores the page as terminal came back with the counter at 0: the
demotion silently disappeared while the cleaned (redacted) text — the
fabricated refs already stripped — still shipped under `SUCCESS`.

This is the third instance of the family `#814` (GH-682, three
`chart_region_*` flags) and `#815` (GH-674, `equation_sidecar_skipped`) closed
today. Followed their idiom exactly:

1. **Persisted** in the sidecar meta write block (`_flush_page_sidecar`,
   `orchestrator.py:~11203`), beside `chart_region_*` / `equation_sidecar_skipped`.
2. **Max-restored** in `_restore_terminal_page_state` (`orchestrator.py:~12150`)
   — see the counter-vs-bool judgement call below.

No new document-level note was needed: `_fabricated_url_note`
(`orchestrator.py:4931`) already surfaces this at the document level, reading
`state.events` for `fabricated_image_ref` audit events — which are themselves
already persisted/replayed via the existing `audit_events` sidecar field and
`_restore_terminal_page_state`'s event-replay path. This ticket's gap was
purely the counter that drives `pages_ok`, not the surfacing.

## Counter vs. bool: max(), not OR

`#814`/`#815`'s fields are dispositions (booleans): "did X happen to this
page", where OR is exactly right — any one occurrence is enough, and
"any-of-two-runs" is definitionally an OR. `fabricated_image_refs` is a
**count** of how many refs were redacted. A boolean-ised OR (`bool(a) or
bool(b)`, or worse, `a or b` on two truthy ints) would either collapse the
count to a fixed 1/0 or silently pick whichever operand is non-zero,
changing the *number* reported on resume even though the demotion (which
only needs "any nonzero") would still be correct either way.

`max(existing, sidecar)` is the right restore for a count:

- Preserves the demotion invariant: any nonzero value on either side still
  makes the result nonzero, so `fabricated_ref_pages` still catches the page.
- Never lets a resume DROP a higher count this run already recorded down to a
  lower (or missing, i.e. 0) sidecar value — the OR-restore's whole point.
- Also never lets a resume SILENTLY GROW a count via re-detection double
  counting (an accidental `existing + sidecar` would inflate the number on
  every resume of an already-terminal page); `max()` treats the two counts as
  two measurements of the same fact, not an accumulator.

A count that silently changes value across a resume (even while never going
to zero) is its own small defect for anything downstream that reports the
number (e.g. a CLI/log line reading `len(removed)` history) — `max()` is the
restore that keeps both the demotion AND the number honest.

## Files changed

- `src/socr/pipeline/orchestrator.py` — persist `fabricated_image_refs` in the
  sidecar meta write block; max-restore it in `_restore_terminal_page_state`.
- `tests/test_gh817_fabricated_refs_resume.py` (new) — 5 tests, all pinning the
  OUTCOME (`DocumentStatus`), not the field's mere presence: a real fabrication
  demotes; a clean run (no fabrication) still succeeds (reverse regression); a
  full real `_guard_fabricated_image_refs` → `_flush_page_sidecar` →
  `_restore_terminal_page_state` → `_phase_assemble` cycle reproduces the
  reported RUN1/RUN2 bug and proves it fixed; a value set this run is not
  cleared by an older sidecar missing the key (the OR/max case, missing-key
  side); a sidecar with a HIGHER count than the in-memory value wins (the
  max-not-OR-as-bool case — the test that would have failed under a naive
  `bool(a) or bool(b)` restore floored at 1).
- `tests/test_p6_disposition_persistence.py` — added `"fabricated_image_refs"`
  to the frozen pre-disposition sidecar key-set
  (`test_sidecar_only_additive_key_is_disposition`), same treatment `#814`/
  `#815` gave their keys.
- `tests/fixtures/p6/prechange_assemble.json` — purely additive:
  `"fabricated_image_refs": 0` added to all 12 sidecar entries (verified via a
  scripted walk: 12 insertions, 0 deletions, `git diff --stat` confirms).

## Mutation proof (write and restore, separately)

Harness: `cp -R src tests` into `/tmp/wt-817-mut-write` and
`/tmp/wt-817-mut-restore` (outside the repo, both deleted after). Ran with
`pytest -o pythonpath=src` (the sweep scout's cleaner alternative to copying
`pyproject.toml`) plus a canary test asserting `socr.__file__` resolves under
the mutant tree via `os.path.realpath` — confirmed passing (not silently
skipped) before trusting either mutation's result. Each mutation asserted its
anchor's uncapped `str.count(anchor) == 1` and asserted the source actually
changed, aborting the harness otherwise (both anchors were unique on first
try, no abort needed).

| Guard | Mutation | Failing test(s) | Count |
|---|---|---|---|
| Write (persist) | Replaced the `"fabricated_image_refs": (int(...) if ps else 0)` meta-dict entry with a hardcoded `0` | `test_full_flush_restore_reassemble_cycle_still_demotes` | 1 failed, 5 passed |
| Restore (max) | Deleted the `ps.fabricated_image_refs = max(...)` assignment entirely | `test_full_flush_restore_reassemble_cycle_still_demotes`, `test_max_restore_does_not_silently_shrink_a_lower_run_value` | 2 failed, 4 passed |

Each mutation caught by a distinct, non-overlapping subset (the max-shrink
test only fails under the restore mutation, never the write mutation, as
expected — it never touches the write path). Reverted between rounds by
discarding the copy (never `git checkout` in the shared worktree).

## Test results (measured)

- `tests/test_gh817_fabricated_refs_resume.py`: 5 passed.
- `tests/test_p6_disposition_persistence.py` + `tests/test_p6_stage_c_difference.py`
  + `tests/test_p6_stage_ab_difference.py` + `tests/test_table_latch_sidecar.py`
  + `tests/test_s1_structure_class_winner_gh_reachability.py`: 156 passed.
- Full suite, baseline vs branch, both measured directly (not assumed from a
  prior ticket's numbers, per the standing "quote baselines by measuring"
  rule):
  - Baseline (`git archive HEAD` at `42a3e49`, fresh copy,
    `PYTHONPATH=.../src pytest tests -q`): **5655 passed, 1 skipped, 4
    xfailed** (the skip is `test_gh592_scoped_positional_emission.py`, the
    documented archive artifact per `#815`'s log — a real git checkout has
    the ref it needs, an archive copy does not).
  - Branch (this worktree, real git history so `gh592` runs instead of
    skipping): **5661 passed, 4 xfailed**, 0 skipped.
  - Reconciles exactly: +6 passed = +5 (this ticket's new tests) + 1
    (`gh592` moving from skip → pass, the archive artifact, not a
    regression).
- `uvx ruff@0.16.0 format --check .`: clean (two files needed reformatting on
  first draft — the orchestrator edit's line length and the new test file —
  fixed with `uvx ruff@0.16.0 format` on those two files, then re-verified
  clean across the whole repo).

## Framing check

The orchestrator's framing was accurate throughout — the reproduction, the
sidecar/restore location, and the "third instance of a closed-twice family"
characterization all matched what I found. The one deliberate judgement call
was the one the orchestrator explicitly asked me to make rather than dictate:
max() over a boolean-ised OR for the restore, reasoned above. I considered
whether to also add a `_fabricated_image_refs_note` mirroring `#815`'s
`_equation_sidecar_skipped_note`, but `_fabricated_url_note` already exists
and already does that job for this exact bucket (it predates this ticket), so
adding a second note would have been a duplicate surface, not a gap closed.
