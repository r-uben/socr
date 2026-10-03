# GH-994 (a): a flattened table no longer ships SUCCESS

Branch `fix/994-flattened-table-demote`, from origin/main 54fe285a. Part (b), re-routing to a
model read, is NOT done: the native table path refuses these pages and a re-route is about 209
VLM reads per corpus run plus judging, and a model can invent tables (the 4 prose false
positives). It needs its own A/B first.

## What changed

- `BornDigitalDetector._detect_flattened_table` (`core/born_digital.py`). Fires only when
  `has_tables` is False, the page has a `Table N` caption line, AND one of: `_MIN_TABLE_ROWS`
  horizontal rules (distinct y) sharing an x-extent; `has_recurring_numeric_columns(words,
  _MIN_COLS)`; the GH-64 flag. This is the first consumer of the GH-64 flag. New
  `PageAssessment.table_not_reconstructed` / `PageState.table_not_reconstructed`. Two new named
  constants for what counts as a rule stroke (1.0 pt slope for drawn lines, 2.0 pt height for
  rects), taken from the measured detector (`gh994/signals.py`). It never routes.
- `FailureMode.TABLE_NOT_RECONSTRUCTED` (`core/result.py`).
- Page status (`core/manifest.py`): demoted via `native_demoted` to WARNING, `audit_passed`
  untouched (winner selection, see memory "audit_passed selects the winner"). It is also added to
  the `native_distrusted` short-circuit so a restored SUCCESS winner cannot outrank the flag.
  Not gated on `p.attempts`: this page never reaches the OCR ladder. Text unchanged.
- Chart-asset lane (`orchestrator.py`) ships native prose too, so it gets the same demotion.
- Document status: `flattened_table_pages` (same shape as `invisible_retained_pages`) blocks
  `pages_ok`, so the document is AUDIT_FAILED, not SUCCESS; plus the `table_not_reconstructed_retained`
  event and a console line.
- Audit event `table_not_reconstructed`, emitted at analyze, recomputed each run (not in
  `_RESUME_REPLAYED`).
- Resume: `_load_terminal_page` refuses a cached native/chart winner when this run's analysis
  flags the page (as #913 does). A page demoted this way is WARNING, so it is not terminal-SUCCESS
  on the next run either.
- `docs/OUTPUT.md`: failure mode row (33 members) and both event kinds.

## Measurements (real detector, `detect_page`, worktree source, canary passed, CPU only)

- 8 real-miss pages (Forsythe p8/p22/p25, hansen p20, bernanke p32, Barrot p41, bybee p30/p32):
  **8/8** fire.
- Trusted-native population, 9891 pages: **241 fires, 209 outside the chart lane**, matches the
  housekeeping sweep. Pre-filtered to caption plus any signal (275 candidates), which is exact
  because the caption is a necessary condition.
- 127-page census: **0 fires** (all 127 have `has_tables=True`), 0 changes.
- Known cost, from the same sweep: about 4 of 20 sampled fires (20%) are false positives
  (figure pages, prose pages with a `Table N ...` sentence at line start). They are demoted to
  WARNING with text unchanged, which is the price of not re-routing.

## Tests (`tests/test_gh994_flattened_table.py`, 18)

Hermetic synthetic fitz PDFs. Detector shapes: caption+rules fires, caption+numeric columns
fires (two-word labels so the GH-64 flag does not carry it), caption alone quiet, two rules
quiet, rules without caption quiet, numeric columns without caption quiet, `has_tables=True`
quiet, GH-64 flag with/without caption. `process()` difference pins (detector neutralised vs
live, both provider states) on page status, failure mode, document status, identical text and
engine calls; native-only; quiet shape identical on/off; resume refuses a cached SUCCESS.

Mutations, in an external copy of src, tests and pyproject (the test asserts
`socr.__file__` is inside the copy; the no-op control survived, as it must): removing the
`has_tables` gate, the caption gate, the rules threshold (3 -> 1), the numeric-columns signal,
the GH-64 fold, the manifest demotion, the manifest failure mode, the document-status term, the
resume refusal, the analyze event and the state copy were all killed. The first run of the
numeric-columns mutant SURVIVED: the fixture also tripped the GH-64 flag, so the signal was
untested. The fixture labels were made two-word and it is killed now.

## Not covered

- The chart-asset lane demotion has no dedicated test (shares the manifest predicate; the lane
  needs a chart-marks fixture).
- Doc-level resume (`_resume_skip`) keys on the source-digest fingerprint, so it is moved by a
  real code change but not by a monkeypatch; the test bypasses it and exercises the per-page gate.
- Collision: #993 edits metadata/library; this touches `born_digital.py`, `state.py`,
  `result.py`, `manifest.py` and `orchestrator.py` only.

## Review fixes (Astra, ACCEPT-WITH-FIXES)

1. Demotion moved into the passing-winner short-circuit in `_winning_page_output`: the selected
   native/chart winner is `replace`d in place (status WARNING, `TABLE_NOT_RECONSTRUCTED`), so its
   exact bytes ship (`native+equations` bodies are not re-derived through the prefix-only
   `_native_text_with_appends`). The flag is no longer in `native_distrusted`, and the now
   unreachable `table_flattened` terms in the native-fallback branch were removed rather than left
   unguarded. Pinned by bytes-identical-with-and-without-flag tests for native, native+equations
   and chart_asset; a model winner is untouched.
2. Resume refusal is guarded to `native*` / `chart_asset*` cached winners, like #990. A cached
   model winner on a flagged page is restored (regression test).
3. Chart-asset lane test: body and `audit_passed` kept, WARNING + failure mode, document not SUCCESS.
   Honest limit: the lane's own `flattened_suspect` term is backstopped by the manifest demotion,
   so removing only that term survives; removing both is killed by the chart test.
