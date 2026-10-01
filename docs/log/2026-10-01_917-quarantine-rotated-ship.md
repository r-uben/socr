# GH-917 / GH-916: quarantine rotated native-first SHIP (2026-10-01)

## Why
On main d139ce8, `attempt_rotated_native_table` returned SHIP / exact_pass on 35 rotated
pages in the 400-PDF library. A vision audit (Fable) plus mechanical confirmation found
14/35 wrong: detached minus signs (a bare minus cell followed by an unsigned number, rest
of the row shifted one column), dropped rows or panels, text in numeric columns.
`plan_native_table` cannot see these: native_verifier excludes standalone signs from
numeric tokens, and EXACT_PASS pairs rows by numeric multiset (#916). Rotated SHIP skips
route_page, the page judge and the table ladder, so these shipped SUCCESS.

## Change (containment only)
- `src/socr/tables/native_first.py`: new constant `ROTATED_SHIP_QUARANTINED`; in
  `attempt_rotated_native_table`, a SHIP from `plan_native_table` is replaced by
  `NativeTablePlan(DEFER, reason=ROTATED_SHIP_QUARANTINED)`. The grid is still returned.
- `src/socr/pipeline/orchestrator.py` (`_plan_native_table_first`): on that reason, log a
  warning and append an `AuditEvent` of kind `rotated_native_table_quarantined` (same
  record class and sidecar path as `landscape_page_refused`; no new persisted record
  type, so no new resume site). The planner already returns None for non-SHIP, so the page
  stays on route_page + judges + table ladder. Upright native-first is untouched.
- The orchestrator's rotated SHIP compose branch is now unreachable; left in place so the
  fix to restore it (order-aware verifier) is a one-line removal of the quarantine.

## Tests (tests/test_rotated_native_table_first.py)
- 90/270 attempt returns DEFER with the quarantine reason; `plan_native_table` on the same
  words+markdown still returns SHIP (difference pin). Grid-order checks (#902) kept.
- `process()` on the rotated fixture, parametrised over provider state (one provider / none):
  `native_table_exact_pass` absent, `rotated_native_table_quarantined` present,
  route_page called once with a provider and not called with an empty ladder.
- Upright SHIP unchanged: `tests/test_native_table_first.py` passes untouched.

## Mutation
Copy of src+tests+pyproject outside the repo, canary asserting `socr.__file__` inside the
copy (passed), anchor `count == 1` asserted uncapped. Replacing the quarantine condition
with `if False:` fails 5 tests (DEFER pin, both provider states of the process() test, the
two TestRotationSign attempt tests).

## Results
Focused: 12 passed, 2 skipped (FOMC fixture absent). Full suite (default
OLLAMA_HOST): 5917 passed, 2 skipped, 4 xfailed. `uvx ruff@0.16.0 format --check .` clean.

## Follow-up
#916 (order-aware verifier) and #917's own questions (running-head exclusion, label+year
cells) are the real fix; lift the quarantine only after the 35-page set re-measures clean.

## Round 2 (PR #918 review, 3 cubic findings)
1. Resume: `rotated_native_table_quarantined` added to `resume_restore_kinds()`.
   `_plan_native_table_first` only runs over `ocr_pages` after resumed pages are removed,
   so nothing re-emits it on resume (no double count). New test drives the real emit,
   `_flush_page_sidecar`, `_restore_terminal_page_state` and pins the count (1 emitted,
   1 restored). Mutation (allowlist entry removed, canary + uncapped anchor count):
   both new resume tests fail; 13 others pass.
2. `test_rotated_exact_pass_is_quarantined_not_shipped` now keeps the result and pins a
   difference: the same page run with and without the quarantine (pre-#917 SHIP restored
   via a wrapper), in both provider states. Exact-pass event only without the quarantine,
   markdown and engine/status differ, routed output ships with a provider. No absolute
   no-provider outcome is pinned.
3. Event detail reworded: "deferred to normal routing" (was "page routed to OCR", false
   with an empty ladder).
