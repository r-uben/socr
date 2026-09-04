# GH-581: name the real UNVERIFIED cause, stop the false "retryable" promise

2026-09-04. Fixes r-uben/socr#581, found by the 2026-09-04 ladder corpus run
(`docs/log/2026-09-04_ladder-corpus-run.md`).

## What

`_run_table_judge_gate`'s per-table message loop (`orchestrator.py`) built
two of its four `table_ladder_unverified` detail strings from fixed text:

- the binding-contradiction branch always said "(acceptance withheld;
  retryable on resume)";
- the default branch always said "unverified by the judge ladder (infra
  problem, retryable on resume)".

Neither checked whether the page's own retry latch
(`ps.table_judge_retry_pending`) had actually fired. A deterministic
binding contradiction never latches (GH-575); a rung that answered but was
not accepted (¬S1, no outage) does not latch either. Both branches promised
a retry that resume does not perform.

Separately, the audit trail dropped the cause. `RungResult.error` /
`unavailable` / `refusal` never reached the `rung_trail` entries, and the
guard chain's own decision (`guard_detail_by_table`, populated by
`_resolve_table_guard_chain`) was read only by the ACCEPTED and WITHHELD
messages -- never by UNVERIFIED, which is where it matters most.

## Fix (current state)

The bullets immediately below describe what the code actually does today,
after seven review rounds. Everything under "Sites changed" onward is kept
as a dated history of how it got there, round by round -- read it for
provenance, not as a second, competing description of current behaviour.

- **Wording is built from FIXED phrases only, never raw text -- on every
  path that has been through this ticket.** Most `table_ladder_unverified`
  detail construction reads ONE constant table keyed by cause,
  `CAUSE_DETAIL_PHRASES` (`table_verdict.py`, all eleven causes): the
  default branch and the no-witness branch in `_run_table_judge_gate`, the
  assemble-time completeness backfill, the legacy-sidecar normalizer
  (`_normalize_legacy_unverified_event`), and the `tables_trust.json`
  empty-detail floor (`_unverified_fallback_detail`, round 7, finding 1).
  The binding-contradiction branch (the table-level `forced_by_binding`
  clamp) is the one exception: it builds its own two fixed literal strings
  (held / not-held) directly, not through the shared table -- still no raw
  text, just not that specific dict. Every one of these appends the
  controlled retry clause the same way: `latched = bool(unavailable_kinds)`
  -- the SAME set that sets `ps.table_judge_retry_pending` -- and
  "; retryable on resume" only when it is true. Raw provider/exception/guard
  diagnostics (a rung's ¬S1 `error`, a guard decision's free-text `detail`,
  an exception's own message) are NEVER interpolated into `detail` on any
  of these paths -- they live only in structured data (`rung_trail[].error`,
  `data["guard_detail"]`, and a dedicated `data["witness_error"]` for the
  witness-preparation exception path). This is deliberate: an untrusted
  diagnostic can itself contain the reserved substring "retryable" and
  would otherwise trip the literal acceptance invariant below on a
  genuinely unlatched event (cold review round 4, finding 1).
- **The literal acceptance invariant holds for every UNVERIFIED path**:
  `("retryable" in detail) == data["latched"]`. Verified by a parametrized
  test over all ten production-reachable causes plus a paired test that two
  otherwise-identical fixtures, one with a benign raw error and one whose
  raw error contains the word "retryable", produce IDENTICAL `detail`
  strings (cold review round 4, finding 1's paired-fixture requirement).
- **Audit trail**: each `rung_trail` entry now also carries `error`,
  `unavailable`, `refusal` (straight off the `RungResult`). EVERY
  `table_ladder_unverified` event -- including the witness-preparation
  exception path, which never reaches the per-table message loop and is
  populated by hand (cold review round 1, finding 3) -- now carries
  `data["guard_detail"]` (or `null`), `data["latched"]` (bool), and
  `data["cause"]`: one of a closed set of ELEVEN strings
  (`socr/judge/table_verdict.py`, `TABLE_LADDER_UNVERIFIED_CAUSES`):
  `CAUSE_NO_WITNESS`/`CAUSE_RUNG_UNAVAILABLE`/`CAUSE_RUNG_NOT_ACCEPTED`/
  `CAUSE_BINDING_CONTRADICTION`/`CAUSE_GUARD_NOT_CLEARED`/`CAUSE_GUARD_ERROR`/
  `CAUSE_NO_RUNGS_CONFIGURED`/`CAUSE_INSUFFICIENT_CORROBORATION`/
  `CAUSE_WITNESS_PREPARATION_ERROR`/`CAUSE_MISSING_TABLE_TERMINAL`/
  `CAUSE_UNKNOWN` (reserved for a genuinely unclassified state; cold review
  round 1, finding 2 moved two previously-`unknown` production paths -- an
  empty configured rung list, and a lone low-confidence PASS with no
  corroborating witness -- onto their own named causes; round 2, finding 1
  added `CAUSE_MISSING_TABLE_TERMINAL` for the assemble-time completeness
  backfill, a SECOND `table_ladder_unverified` emit site outside this
  message loop entirely -- see the round-2/3 sections below). Selection is
  by TABLE MEMBERSHIP in `guard_detail_by_table`, never by truthiness of the
  detail string (round 3, finding 1 -- a valid guard answer can carry an
  empty detail). `_resolve_table_guard_chain` gained a `guard_cause_by_table`
  output dict to distinguish `guard_not_cleared` from `guard_error` (an
  exception inside the chain) and from `binding_contradiction` (the guard's
  own `GuardDisposition.CONTRADICTED`, a different code path from the
  table-level `forced_by_binding` clamp but the same fact); and a
  `guard_decision_by_table` dict (round 2, finding 3) carrying the TYPED
  decision -- `guard_disposition`, `geometry_evidence`, `adjudicator_ran`,
  `guard_unavailable`, `guard_refusal`, `adjudicator_suppressed`,
  `requested_refs` -- spread into the event's data additively.
- **The literal acceptance invariant** (round 3, finding 3): for every
  `table_ladder_unverified` event, `("retryable" in detail) == data["latched"]`.
  A negative claim must avoid the substring `"retryable"` entirely, not
  merely negate it -- the no-witness branch's "not retryable" phrasing
  violated this even though it was semantically correct; reworded to "will
  not be retried".
- **Document/CLI wording** (`_unverified_wording_split`,
  `_table_judge_ladder_note`, the verbose CLI summary): the old two-bucket
  split (retryable / unwitnessed) silently dropped a page whose only
  UNVERIFIED event was a non-latched binding contradiction or default-branch
  terminal -- it belonged in neither bucket. Now FOUR buckets --
  `retryable`, `not_retryable`, `unwitnessed`, `incomplete` -- with
  `retryable`/`not_retryable`/`incomplete` mutually exclusive per page in
  that fixed precedence (round 3, finding 2: a page that is both latched
  and incomplete is told about the underlying fact once, not twice).
  `unwitnessed` stays orthogonal and can co-occur with any of the three.
  The assemble-time completeness backfill event now derives its own
  `latched` from the page's real `table_judge_retry_pending` (round 3,
  finding 2 -- it used to hard-code `False`, which could contradict an
  already-fired page latch from a sibling table) and includes the
  page-scoped positive retry clause when true, exactly like the mixed
  no-witness path.

## Sites changed

- `src/socr/judge/table_verdict.py`: eleven `CAUSE_*` constants +
  `TABLE_LADDER_UNVERIFIED_CAUSES` (see round 2/3 below for the four added
  after cold review); the terminal-events comment no longer equates
  UNVERIFIED with "infra problem" (round 3, finding 5).
- `src/socr/pipeline/orchestrator.py`: `_resolve_table_guard_chain` (new
  `guard_cause_by_table` and `guard_decision_by_table` params);
  `_run_table_judge_gate`'s message loop (both named branches, cause
  selected by table MEMBERSHIP not detail truthiness, `rung_trail`
  construction, event `data`); `_unverified_wording_split` (4-tuple return,
  mutually-exclusive precedence); `_table_judge_ladder_note` and the
  verbose CLI summary block; `_backfill_missing_table_ladder_terminals`
  (round 2, finding 1; round 3, finding 2 fixed its `latched` derivation).

## Correction to an existing test's assumption

`tests/test_gh560_unwitnessed_wording.py` had a case ("a located witness
handed an empty `rungs` list") that it called "a genuine rung outage" and
asserted stays worded "retryable on resume". An empty `rungs` list produces
no `RungResult` at all, so the P1 latch cannot fire for it structurally --
it is a configuration fact (`strict_local`, or no CLI found), not a
transient outage a later identical call resolves. That assumption was the
same empty promise #560 itself was filed against, for a different
non-latching cause. Renamed the test
(`test_a_located_witness_with_no_rungs_is_not_retryable`) and re-pinned it,
plus `test_the_document_note_separates_the_two`, to the corrected behaviour.

## Tests

`tests/test_gh581_unverified_cause.py` CURRENT state, after all seven review
rounds: 23 `def test_*` definitions, 32 collected test cases
(`pytest --collect-only`) -- one of the 23 is `@pytest.mark.parametrize`'d
over the ten production causes with a dedicated event builder each,
contributing 10 of the 32. Imports the exported `CAUSE_*` constants and
`TABLE_LADDER_UNVERIFIED_CAUSES` from `socr.judge.table_verdict` and asserts
against them / membership in the closed set, not raw strings (round 3,
finding 4). Built up incrementally: seven focused hermetic difference tests
from round 1 (two paired-difference tests and five controlled path
assertions), five more from round 2, five more from round 3, three more
from round 4 (`CAUSE_DETAIL_PHRASES`/no-raw-text and the guard-prep-inside-
the-try regression), and one more from round 5 (the paired ConnectionError/
RuntimeError latch difference below). The round-1 list (below) is kept as
the original difference pins, not a current inventory:

1. same table, latch absent vs. present (one `RungResult.unavailable=True`):
   `"retryable"` in `detail` and `data["latched"]` track the difference.
2. a rung with `ok=False` and a non-empty error: the trail entry carries the
   error text, `cause == "rung_not_accepted"`.
3. binding `CONTRADICT` with no unavailability: `cause ==
   "binding_contradiction"`, no `"retryable"`; the adjudicator is never
   called (GH-575 unchanged).
4. two-low-pass with no adjudicator configured (guard chain reaches
   `NOT_CLEARED`): `guard_detail` equals the guard's own decision detail.
   Superseded by round 4, finding 1: since then `guard_detail` is
   structured-only and deliberately NOT interpolated into `detail`, and the
   round-1 test itself was updated to assert `guard_detail` is absent from
   `detail`, not present in it. `cause == "guard_not_cleared"` still holds.

Updated for the new `rung_trail`/event-`data` keys (exact-dict-equality
assertions): `tests/test_table_judge_gate.py`, `tests/test_ladder_binding_
evidence.py` (1 assertion + the `test_flag_off_ships_unaffected_flag_on_
demotes` doc-note check, which now passes because the `not_retryable`
bucket keeps the page's mention in the note instead of dropping it),
`tests/test_gh560_unwitnessed_wording.py` (2 tests re-pinned, see above).
This was round 1's state: four test modules touched (three tracked-file
changes plus the new `test_gh581_unverified_cause.py`), with
`tests/test_ladder_sidecar.py` run alongside them as an UNCHANGED regression
module. Superseded by round 6 (the legacy-sidecar normalizer required a
fixture fix in `test_ladder_sidecar.py`) and round 7, finding 3 (a dedicated
new-fields round-trip test added to the same file) -- `test_ladder_sidecar.py`
is modified, not unchanged. CURRENT total: six test modules touched
(`test_gh581_unverified_cause.py` new; `test_table_judge_gate.py`,
`test_ladder_binding_evidence.py`, `test_gh560_unwitnessed_wording.py`,
`test_ladder_sidecar.py`, and `test_gh95_tables_trust.py` modified).
`tests/test_p1_ladder_retry_latch.py` and the rest of
`tests/test_p1_tiebreak_and_withhold.py` / `test_table_cell_guard.py` needed
no changes -- they read `rung_trail` by index/subset, not exact-dict
equality.

## Cold review round 1 (NOT MERGEABLE, 3 major + 2 minor)

- **Finding 1** (`orchestrator.py`, no-witness branch): `latched` is PAGE-scoped
  (any table's rung was unavailable) but the no-witness branch's own "not
  retryable" claim is TABLE-scoped (its own witness will never exist). A
  mixed page (one unwitnessed table, one table with an unavailable rung) had
  the unwitnessed table's event carry `latched: true` with a detail that only
  said "not retryable" -- no positive clause even though the page itself is
  retryable. Fixed by appending an explicit page-scoped clause ("the page
  itself is retryable on resume: another table on this page had an
  unavailable rung") when `latched` is true, while keeping the table's own
  "not retryable" fact unchanged. Test:
  `TestMixedPageLatchIsPageScoped::test_positive_clause_present_only_when_the_page_latch_fires`.
- **Finding 2** (`orchestrator.py`, default UNVERIFIED branch): two known
  production paths fell into `cause: unknown` instead of a specific cause --
  a located witness handed an empty rung list (a config fact: strict_local,
  or no CLI found), and a lone low-confidence PASS with no corroborating
  witness (`table_ladder.py` ruling 1, ordinary S1, never reaches the P1
  guard chain). Added `CAUSE_NO_RUNGS_CONFIGURED` and
  `CAUSE_INSUFFICIENT_CORROBORATION` to the closed set in
  `table_verdict.py`, assigned before the `CAUSE_UNKNOWN` fallback. Updated
  `test_table_judge_gate.py`'s empty-rung-list pin from `unknown` to
  `no_rungs_configured`. Tests:
  `TestNoRungsConfiguredIsItsOwnCause`, `TestInsufficientCorroborationIsItsOwnCause`.
- **Finding 3** (`orchestrator.py`, witness-preparation exception path): the
  outer fail-closed path (B0's documented-never-to-raise contract broken)
  emitted `table_ladder_unverified` with an empty `data` dict -- no `cause`,
  `latched`, `guard_detail`, `rung_trail`, or `witness_scope`. Added
  `CAUSE_WITNESS_PREPARATION_ERROR` and emit the full additive contract on
  this path by hand (`rung_trail: []`, `witness_scope: "none"`,
  `guard_detail: None`, `latched: False`). Terminal and latch unchanged.
  Extended `test_witness_preparation_exception_is_unverified_not_raised` in
  `test_table_judge_gate.py` with the `data` assertion.
- **Finding 4** (this log): "see report below" pointed at nothing. Fixed by
  this section.
- **Finding 5** (`tests/test_gh581_unverified_cause.py`): unsorted import
  block (`I001`). Fixed with `uvx ruff@0.16.0 check --select I --fix`.

Each of the three MAJOR fixes was verified fail-then-pass: the fix was
temporarily reverted in place, the corresponding new test was confirmed to
fail with the exact defect the review reproduced, then the fix was restored
and the test re-confirmed green.

## Cold review round 2 (NOT MERGEABLE, 3 major + 2 minor)

- **Finding 1** (`_backfill_missing_table_ladder_terminals`, the assemble-
  time completeness backfill -- a SECOND `table_ladder_unverified` emit
  site outside the per-table message loop entirely): still said "infra
  problem, retryable on resume" with an empty `data` dict, even though
  nothing here ever reaches a rung to latch. Added `CAUSE_MISSING_TABLE_
  TERMINAL`; the event now carries the full additive contract
  (`rung_trail: []`, `witness_scope: "none"`, `guard_detail: None`,
  `latched: False`, `cause`) and a completeness-specific detail ("no
  table-judge terminal was recorded; the page will be reprocessed because
  its terminal is incomplete") instead of the false retry promise. Because
  `table_ladder_incomplete` has its own resume semantics (distinct from
  `table_judge_retry_pending`), `_unverified_wording_split` gained a FOURTH
  bucket, `incomplete`, read from the page's own `table_ladder_incomplete`
  flag and excluded from both the latch-retryable and not-retryable
  witnessed buckets -- the document note and CLI summary each gained a
  fourth, distinct sentence for it. No change to latch, terminal,
  disposition, or completeness behavior. Tests:
  `TestAssembleBackfillNamesItsOwnCause` (two tests: the event itself, and
  the document note).
- **Finding 2** (`orchestrator.py`, default UNVERIFIED branch): cause
  classification was entered only when `RungResult.error` was truthy, but
  `error` defaults to `""` independently of the typed `unavailable`/
  `refusal` fields -- a real outage with no error text fell through to
  `CAUSE_UNKNOWN` despite firing the latch. Reclassified from the typed
  facts first: any `not rr.ok` names the table's ¬S1 rungs; `unavailable`
  or `refusal` on any of them is `rung_unavailable`, otherwise `rung_not_
  accepted`; `cause_text` is built independently (joined non-empty errors,
  or the neutral fallback when all are empty). Test:
  `TestCauseClassificationIsTyped::test_empty_error_with_unavailable_is_
  still_rung_unavailable` (same fixture, error `""` vs `"boom"`, both
  `unavailable=True`, same cause).
- **Finding 3** (`_resolve_table_guard_chain` / the message loop): only
  `CellGuardDecision.detail` (free text) reached the audit data --
  `cause: guard_not_cleared` and a latch that may belong to a SIBLING table
  cannot establish whether THIS table's adjudicator ran, was never
  configured, or was suppressed. Added `guard_decision_by_table`, a
  per-table serialized decision populated alongside `guard_detail_by_table`,
  and spread its keys into the UNVERIFIED event's `data` when present:
  `guard_disposition`, `geometry_evidence`, `adjudicator_ran` (bool),
  `guard_unavailable`, `guard_refusal`, `adjudicator_suppressed`,
  `requested_refs`. Additive only -- not consulted for outcome or latch.
  Tests: `TestGuardDecisionTypedFields` (unavailable vs. refusal as a
  controlled difference on the same real call; configured-absent vs.
  suppressed as a controlled difference on `adjudicator_ran`/
  `adjudicator_suppressed`).
- **Finding 4** (this log): stale inventory -- fixed by this section and the
  "Sites changed" / "Tests" sections above.
- **Finding 5** (`orchestrator.py`, no-witness branch): the fourteen-line
  explanatory comment for the finding-1-round-1 fix had been duplicated
  (an artifact of an earlier revert-and-restore cycle during fail-then-pass
  verification). Removed the second copy.

Each of the three round-2 MAJOR fixes was verified fail-then-pass the same
way as round 1: reverted in place, the corresponding new test confirmed to
fail with the exact defect the review reproduced, restored, test
re-confirmed green.

## Full report (round 2)

Format gate: `uvx ruff@0.16.0 format --check .` -> `560 files already
formatted`, exit 0. Import order:
`uvx ruff@0.16.0 check --select I tests/test_gh581_unverified_cause.py` ->
`All checks passed!`, exit 0.

Focused modules, foreground:
```
PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-581/src ~/venvs/socr/bin/pytest \
  tests/test_gh581_unverified_cause.py tests/test_gh560_unwitnessed_wording.py \
  tests/test_table_judge_gate.py tests/test_ladder_binding_evidence.py \
  tests/test_ladder_sidecar.py tests/test_gh359_ladder_terminals.py \
  tests/test_p1_document_surfaces.py -q
```
-> `112 passed`, exit 0.

Full suite, foreground:
```
PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-581/src ~/venvs/socr/bin/pytest tests/ -q
```
-> `4055 passed, 4 xfailed`, exit 0 (165.96s).

## Cold review round 3 (NOT MERGEABLE, 3 major + 2 minor)

- **Finding 1** (`orchestrator.py`, default UNVERIFIED branch): the guard
  cause was selected with `if guard_detail:`, so a valid guard answer with
  an EMPTY detail (`BlindCellResult(unavailable=True, error="")`) was
  discarded to the generic fallback despite `guard_cause_by_table` already
  holding the specific cause. Reselected by table MEMBERSHIP
  (`result.table_id in guard_detail_by_table`) with a neutral `cause_text`
  fallback only when the detail string itself is empty. Test:
  `TestGuardCauseSelectedByMembership` (paired empty/non-empty detail, both
  keep `guard_not_cleared` and `guard_unavailable: True`).
- **Finding 2** (`orchestrator.py`, assemble backfill + `_unverified_
  wording_split`): the backfill event hard-coded `latched: False` and could
  disagree with an already-fired page latch from a sibling table; the
  document note then listed the same page in BOTH the retryable and
  incomplete sentences for the same underlying fact. The backfill now
  derives `latched` from the page's real `table_judge_retry_pending` and
  appends the page-scoped positive clause when true. `_unverified_wording_
  split`'s four buckets are now mutually exclusive per page in a fixed
  precedence: `retryable` > `incomplete` > `not_retryable` (`unwitnessed`
  stays orthogonal). Test: `TestMixedPageBucketPrecedence` (a page latched
  by one table AND incomplete via a second table's backfill: `backfill_
  latched == page_latch`, and the note names page 1 exactly once).
- **Finding 3** (`orchestrator.py`, no-witness branch): the acceptance
  criterion is the literal invariant `("retryable" in detail) ==
  data["latched"]`. The no-witness branch's unconditional "not retryable"
  phrasing contains the substring `"retryable"` even when unlatched,
  failing the literal check despite being semantically correct. Reworded to
  "will not be retried". Test:
  `TestRetryableSubstringMatchesLatchedForEveryPath`, parametrized over all
  ten UNVERIFIED causes (no-witness, rung unavailable, rung not accepted,
  binding contradiction, guard not cleared, guard error, witness
  preparation error, no rungs configured, insufficient corroboration,
  missing table terminal), asserting the invariant for each.
- **Finding 4** (this log): stale inventory -- fixed by the corrected counts
  throughout this log and this section.
- **Finding 5** (`table_verdict.py`, terminal-events comment): the block
  above `TABLE_LADDER_ACCEPTED_KIND`/`REJECTED`/`UNVERIFIED` still called
  UNVERIFIED "infra problem, distrust". Reworded to "no accepted verdict"
  and pointed readers at the event's `cause`/`latched` for the actual
  reason and retry semantics, since this diff's own causes list five
  non-infrastructure reasons.

Each of the three round-3 MAJOR fixes was verified fail-then-pass the same
way as rounds 1 and 2: reverted in place, the corresponding new test
confirmed to fail with the exact defect the review reproduced (including a
SEPARATE revert-check isolating just the bucket-precedence half of finding
2 from its backfill-latch half), restored, tests re-confirmed green.

## Full report (round 3)

Format gate: `uvx ruff@0.16.0 format --check .` -> `560 files already
formatted`, exit 0. Import order:
`uvx ruff@0.16.0 check --select I tests/test_gh581_unverified_cause.py` ->
`All checks passed!`, exit 0.

Focused modules, foreground:
```
PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-581/src ~/venvs/socr/bin/pytest \
  tests/test_gh581_unverified_cause.py tests/test_gh560_unwitnessed_wording.py \
  tests/test_table_judge_gate.py tests/test_ladder_binding_evidence.py \
  tests/test_ladder_sidecar.py tests/test_gh359_ladder_terminals.py \
  tests/test_p1_document_surfaces.py tests/test_p1_tiebreak_and_withhold.py \
  tests/test_table_cell_guard.py tests/test_p1_ladder_retry_latch.py -q
```
-> `230 passed`, exit 0.

Full suite, foreground:
```
PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-581/src ~/venvs/socr/bin/pytest tests/ -q
```
-> `4067 passed, 4 xfailed`, exit 0 (161.16s).

## Cold review round 4 (NOT MERGEABLE, 2 major)

- **Finding 1** (`orchestrator.py`, the default UNVERIFIED branch, the
  witness-preparation exception path): `detail` interpolated raw
  provider/exception/guard text (a rung's ¬S1 error, a guard decision's
  free-text detail, an exception's own message). An untrusted diagnostic
  containing the reserved substring `"retryable"` therefore violated the
  round-3 literal invariant even on a genuinely unlatched event. Fixed by
  building EVERY UNVERIFIED `detail` from ONE fixed per-cause phrase table,
  `CAUSE_DETAIL_PHRASES` (`table_verdict.py`) -- keyed by `cause`, covering
  all eleven causes including `CAUSE_BINDING_CONTRADICTION` (the guard
  chain's own `CONTRADICTED` disposition, reachable through the default
  branch, not just the table-level `forced_by_binding` clamp's own two
  fixed held/not-held phrases) -- plus the controlled retry clause. Raw
  diagnostics now live ONLY in structured data: `rung_trail[].error` (already
  existed), `guard_detail` (already existed, no longer echoed into
  `detail`), and a NEW `witness_error` field on the witness-preparation
  exception path carrying `f"{type(exc).__name__}: {exc}"`. The no-witness
  and assemble-backfill branches were also switched to read their phrase
  from the SAME dict, removing the last inline duplicates. Tests:
  `TestDetailNeverInterpolatesRawText` (two: paired benign/reserved-word
  rung errors produce IDENTICAL details with the raw text visible only in
  `rung_trail`; witness-preparation exception text lands only in
  `data["witness_error"]`).
- **Finding 2** (`_resolve_table_guard_chain`): `_doubted_cell_refs` and
  `resolve_cell_refs` ran BEFORE the guard chain's own fail-closed `try`, so
  an exception there escaped to the OUTER per-witness `except` in
  `_run_table_judge_gate`, which fabricates a synthetic `RungResult(rung=
  "unknown", ...)` and discards the real reader trail that had already run
  -- reported as `cause="rung_not_accepted"` with `guard_detail=None`,
  contradicting the log's own claim that `CAUSE_GUARD_ERROR` covers every
  exception inside the chain. Moved both calls inside the `try`, so the
  SAME handler now converts a prep failure to `CAUSE_GUARD_ERROR` while
  `result` (and its real `rung_results`) passes through `_unverified()`
  untouched. Test: `TestGuardPrepFailureStaysInsideTheGuardChain` (patches
  `socr.judge.table_verdict.resolve_cell_refs` to raise; asserts `cause ==
  guard_error` and the trail keeps the two real rung identities, `{r1, r2}`,
  not a fabricated `unknown`).

Both round-4 MAJOR fixes were verified fail-then-pass the same way as every
prior round: reverted in place, the corresponding new test confirmed to
fail with the exact defect the review reproduced, restored, tests
re-confirmed green (including the witness-preparation-error sub-case of
finding 1, checked as an independent revert).

## Full report (round 4)

Format gate: `uvx ruff@0.16.0 format --check .` -> `560 files already
formatted`, exit 0. Import order:
`uvx ruff@0.16.0 check --select I tests/test_gh581_unverified_cause.py` ->
`All checks passed!`, exit 0.

`tests/test_gh581_unverified_cause.py` now has 18 `def test_*` definitions,
27 collected test cases.

Focused modules, foreground:
```
PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-581/src ~/venvs/socr/bin/pytest \
  tests/test_gh581_unverified_cause.py tests/test_gh560_unwitnessed_wording.py \
  tests/test_table_judge_gate.py tests/test_ladder_binding_evidence.py \
  tests/test_ladder_sidecar.py tests/test_gh359_ladder_terminals.py \
  tests/test_p1_document_surfaces.py tests/test_p1_tiebreak_and_withhold.py \
  tests/test_table_cell_guard.py tests/test_p1_ladder_retry_latch.py -q
```
-> `233 passed`, exit 0.

Full suite, foreground:
```
PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-581/src ~/venvs/socr/bin/pytest tests/ -q
```
-> `4070 passed, 4 xfailed`, exit 0 (171.48s).

## Cold review round 5 (NOT MERGEABLE, 1 major + 2 minor)

- **Finding 1** (`_resolve_table_guard_chain`, MAJOR): round 4's move of
  `_doubted_cell_refs`/`resolve_cell_refs` inside the guard chain's own
  `try` (finding 2) fixed the discarded-trail bug but silently changed
  LATCH behaviour for the subset of exceptions `is_availability_exception`
  classifies as transient: on `origin/main`, such an exception used to
  escape to the OUTER per-witness handler, which synthesizes a
  `RungResult(rung="unknown", unavailable=True, ...)` that DOES feed the P1
  latch (`rung_kind("unknown")` returns `"unknown"` verbatim, no colon).
  After round 4's move, the SAME exception is caught by the inner handler,
  labelled `guard_error`, and never touches `guard_unavailable_kinds` --
  the terminal stays UNVERIFIED but the latch silently flips from `True` to
  `False`. Fixed by feeding the SAME `"unknown"` kind into
  `guard_unavailable_kinds` when `is_availability_exception(exc)` is true,
  while keeping the real reader trail and the `guard_error` cause (a
  deterministic defect -- `KeyError`, `TypeError`, a plain `RuntimeError`
  -- still does not latch, per GH-575). Test:
  `TestGuardPrepAvailabilityFailureStillLatches` (a `ConnectionError` from
  `resolve_cell_refs` latches on both trees; a `RuntimeError` does not on
  either), plus the reviewer's own reproducer re-run directly and confirmed
  passing.
- **Finding 2** (this log, MINOR): the "Fix" section's top summary described
  round-1 behaviour (raw text in `detail`, ~10 causes) as if it were
  current, and the "Tests" section's counts were stale from round 3.
  Rewritten -- see the new "current state" preface on "Fix" above and the
  corrected counts under "Tests".
- **Finding 3** (`orchestrator.py`, MINOR): two inline comments still
  described the pre-round-3 false semantics -- one calling a witnessed
  empty rung list "genuinely retryable" (round 3 reclassified it as
  `CAUSE_NO_RUNGS_CONFIGURED`, a non-latching config fact), the other
  defining `TABLE_UNVERIFIED` as "infra problem, retryable on resume" in
  the document-status aggregation code (the same equivalence round 3,
  finding 5 already removed from `table_verdict.py`'s own module comment).
  Both reworded to point at `data["cause"]`/`data["latched"]` instead of
  asserting a fixed semantics.

The round-5 MAJOR fix was verified fail-then-pass: reverted in place, the
new test (and the reviewer's own reproducer, re-run directly) confirmed to
fail with the exact defect the review reproduced, restored, tests
re-confirmed green.

## Full report (round 5)

Format gate: `uvx ruff@0.16.0 format --check .` -> `560 files already
formatted`, exit 0. Import order:
`uvx ruff@0.16.0 check --select I tests/test_gh581_unverified_cause.py` ->
`All checks passed!`, exit 0.

`tests/test_gh581_unverified_cause.py`: 19 `def test_*` definitions, 28
collected test cases.

Focused modules, foreground:
```
PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-581/src ~/venvs/socr/bin/pytest \
  tests/test_gh581_unverified_cause.py tests/test_gh560_unwitnessed_wording.py \
  tests/test_table_judge_gate.py tests/test_ladder_binding_evidence.py \
  tests/test_ladder_sidecar.py tests/test_gh359_ladder_terminals.py \
  tests/test_p1_document_surfaces.py tests/test_p1_tiebreak_and_withhold.py \
  tests/test_table_cell_guard.py tests/test_p1_ladder_retry_latch.py -q
```
-> `234 passed`, exit 0.

Full suite, foreground:
```
PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-581/src ~/venvs/socr/bin/pytest tests/ -q
```
-> `4071 passed, 4 xfailed`, exit 0 (177.82s).

## Cold review round 6 (NOT MERGEABLE, 1 major + 1 minor)

- **Finding 1** (MAJOR, `_restore_terminal_page_state`): a sidecar written by
  a pre-GH-581 build has no `latched`/`cause`/`guard_detail` key at all on
  its `table_ladder_unverified` events, and its raw `detail` may still say
  "retryable on resume" unconditionally -- fresh-emission coverage does not
  close the defect for the existing corpus of already-written sidecars,
  which keep replaying the exact false promise this ticket exists to fix on
  every resume. Fixed with a new static helper,
  `_normalize_legacy_unverified_event`, called from the event-replay loop
  whenever a `table_ladder_unverified` event's restored `data` lacks
  `latched`. It sets `latched` from the page's ALREADY-restored real latch
  (`ps.table_judge_retry_pending`, set earlier in the same method -- never
  re-derived), adds the structured defaults (`rung_trail` kept if present
  else `[]`, `guard_detail: None`, `witness_scope` kept if present else
  `"none"`), and selects a specific `cause` ONLY where the legacy event's
  own typed fields prove it: a trail entry with `unavailable: true` proves
  `CAUSE_RUNG_UNAVAILABLE`; the pre-existing GH-560 `retryable: False`
  marker (an explicit typed fact, not a shape guess) proves
  `CAUSE_NO_WITNESS`; anything else -- including an empty trail with no
  witness-scope key, which merely LOOKS like the no-witness case -- falls
  to `CAUSE_UNKNOWN`. `detail` is rebuilt from `CAUSE_DETAIL_PHRASES` plus
  the controlled latch clause, exactly like a fresh emission. Never changes
  which pages the resume gate admits (`_load_terminal_page` reads
  `table_judge_retry_pending`/`table_ladder_incomplete` directly, both
  restored before this normalizer runs, never this event's `data`) or the
  page's disposition. The `_unverified_wording_split` docstring's
  "compatibility fallback" paragraph, which had described the pre-normalizer
  "no latched key -> retryable" default as an intentional, documented
  behaviour, now says what it actually is: a defensive fallback for a path
  outside both this normalizer and the live gate, not a real byte the split
  should ever see in production. Tests:
  `TestLegacySidecarEventsAreNormalizedOnRestore` (an end-to-end mixed-page
  case -- a WITHHELD content-terminal sibling plus this deterministic
  UNVERIFIED table -- through the real `_load_terminal_page`/
  `_restore_terminal_page_state` path, matching the reviewer's own
  reproducer verbatim; plus a unit-level paired-latch test of the
  normalizer itself for the `CAUSE_RUNG_UNAVAILABLE`-proven case, since a
  genuinely latched page is correctly NOT admitted by the resume gate at
  all and so cannot be exercised end-to-end). Also fixed a real regression
  this surfaced: `tests/test_ladder_sidecar.py`'s `_seed_ladder_page`
  fixture built a hand-written `table_ladder_unverified` event without the
  current-contract keys, which the new normalizer correctly (but
  unintentionally, for that test's purpose) treated as legacy and rewrote
  on restore, breaking `test_restore_reproduces_tables_trust_and_note`
  (comparing an unnormalized original against its now-normalized restored
  copy). Fixed by adding the modern keys to that fixture's UNVERIFIED-kind
  event, matching what `_run_table_judge_gate` actually emits.
- **Finding 2** (MINOR, this log): the "Fix" heading had three lines each
  prefixed with `##` (rendering three headings instead of one), and its
  history note hard-coded an incomplete round range. Rewritten as one H2
  with the round-count note as an ordinary paragraph, without pinning a
  round number that will go stale again next round.

The round-6 MAJOR fix was verified fail-then-pass: the normalizer branch was
disabled in place, the new end-to-end test confirmed to fail with the exact
defect the review reproduced (and the reviewer's own reproducer script,
re-run directly, failed identically before the fix and passed after),
restored, tests re-confirmed green.

## Full report (round 6)

Format gate: `uvx ruff@0.16.0 format --check .` -> `560 files already
formatted`, exit 0. Import order:
`uvx ruff@0.16.0 check --select I tests/test_gh581_unverified_cause.py` ->
`All checks passed!`, exit 0.

`tests/test_gh581_unverified_cause.py`: 21 `def test_*` definitions, 30
collected test cases.

Focused modules, foreground:
```
PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-581/src ~/venvs/socr/bin/pytest \
  tests/test_gh581_unverified_cause.py tests/test_gh560_unwitnessed_wording.py \
  tests/test_table_judge_gate.py tests/test_ladder_binding_evidence.py \
  tests/test_ladder_sidecar.py tests/test_gh359_ladder_terminals.py \
  tests/test_p1_document_surfaces.py tests/test_p1_tiebreak_and_withhold.py \
  tests/test_table_cell_guard.py tests/test_p1_ladder_retry_latch.py -q
```
-> `236 passed`, exit 0.

Full suite, foreground:
```
PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-581/src ~/venvs/socr/bin/pytest tests/ -q
```
-> `4073 passed, 4 xfailed`, exit 0 (194.62s).

## Cold review round 7 (NOT MERGEABLE, 1 major + 4 minor)

- **Finding 1** (MAJOR, `src/socr/core/tables_trust.py`): the trust-index
  consumer's empty-detail floor substituted a fixed "infra problem,
  retryable on resume" for a `table_ladder_unverified` event with no
  `detail` of its own -- the exact false promise this ticket removes
  everywhere else, reintroduced at a THIRD consumer this ticket had not
  updated. `LADDER_TERMINAL_NOTES["table_ladder_unverified"]` removed;
  replaced with `_unverified_fallback_detail(data)`, which reads the
  event's own `cause` (via `CAUSE_DETAIL_PHRASES`, falling back to
  `CAUSE_UNKNOWN`'s phrase for a legacy event with neither field) and
  appends "; retryable on resume" only when `data["latched"] is True`.
  Grepped the whole tree afterward for "retryable on resume" and "infra
  problem" outside `orchestrator.py`: the only remaining hits are inside
  `tables_trust.py`'s own new fixed strings/comments (correct, conditional)
  and a documentation comment in `table_verdict.py` explaining the false
  equivalence this ticket removed -- no other producer exists. Test:
  replaced the absolute-fallback pin in `tests/test_gh95_tables_trust.py`
  with `test_unverified_fallback_reads_the_events_own_latch_not_a_fixed_
  string`, a paired latched/unlatched/legacy difference test.
- **Finding 2** (MINOR, the round-6 legacy normalizer): recognized only the
  GH-560 `retryable: False` marker and a trail row's `unavailable: true`,
  so a row with `ok=False`, `unavailable=False`, `refusal=False` (typed
  bits explicitly present, explicitly ruling out an outage) fell to the
  generic `CAUSE_UNKNOWN` instead of the more specific `CAUSE_RUNG_NOT_
  ACCEPTED` it actually proves. Reclassified: any row with `unavailable:
  true` OR `refusal: true` -> `CAUSE_RUNG_UNAVAILABLE`; a row with both
  bits explicitly present and both `False` -> `CAUSE_RUNG_NOT_ACCEPTED`;
  bits absent entirely -> `CAUSE_UNKNOWN`. Corrected the round-6 test that
  had (incorrectly, per this finding) pinned `CAUSE_UNKNOWN` for exactly
  this partially-modern shape, and added a true pre-581 row-shape control
  (`rung`/`ok`/`executing` only, no `unavailable`/`refusal` keys at all)
  that legitimately stays `CAUSE_UNKNOWN`, plus a `refusal: true` pairing.
- **Finding 3** (MINOR, `tests/test_ladder_sidecar.py`): the round-6/round-1
  seeded fixtures all predate this ticket's `error`/`unavailable`/`refusal`
  trail keys, so the "exact row round-trips" assertions proved only the
  legacy three-key shape. Added a dedicated test seeding one current-shape
  row with non-default values for all three fields and asserting the exact
  row survives both flush (to the sidecar JSON) and restore (back into
  `state.events`).
- **Finding 4** (MINOR, this log): several "current state" claims had gone
  stale across rounds 4-6 -- narrowed the phrase-table claim to the paths
  that actually use `CAUSE_DETAIL_PHRASES` (the binding-contradiction branch
  builds its own two fixed strings, not through that dict), corrected the
  round-1 test-4 description (guard detail is structured-only since round 4,
  not present in `detail`), and updated every stale round/count/module
  inventory to the current diff (23 definitions / 32 cases, six test modules
  touched, `test_ladder_sidecar.py` modified not unchanged).
- **Finding 5** (MINOR, `orchestrator.py`): a new line in the round-6
  normalizer exceeded 100 columns. Wrapped.

The round-7 MAJOR and MINOR-2 fixes were verified fail-then-pass: each was
reverted in place, the corresponding (new or corrected) test confirmed to
fail with the exact defect the review reproduced, restored, tests
re-confirmed green.

## Full report (round 7)

Format gate: `uvx ruff@0.16.0 format --check .` -> `560 files already
formatted`, exit 0. `uvx ruff@0.16.0 check --select E501,I` on all seven
touched/reviewed files: only pre-existing findings remain (three E501 lines
in `orchestrator.py` and two `I001` import-order blocks in `orchestrator.py`
and `test_table_judge_gate.py`, none touched this round, all noted as
pre-existing in round 5's own report) -- zero NEW findings from this
round's edits.

`tests/test_gh581_unverified_cause.py`: 23 `def test_*` definitions, 32
collected test cases.

Focused modules, foreground:
```
PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-581/src ~/venvs/socr/bin/pytest \
  tests/test_gh581_unverified_cause.py tests/test_gh560_unwitnessed_wording.py \
  tests/test_table_judge_gate.py tests/test_ladder_binding_evidence.py \
  tests/test_ladder_sidecar.py tests/test_gh359_ladder_terminals.py \
  tests/test_p1_document_surfaces.py tests/test_p1_tiebreak_and_withhold.py \
  tests/test_table_cell_guard.py tests/test_p1_ladder_retry_latch.py \
  tests/test_gh95_tables_trust.py -q
```
-> `272 passed`, exit 0.

Full suite, foreground:
```
PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-581/src ~/venvs/socr/bin/pytest tests/ -q
```
-> `4077 passed, 4 xfailed`, exit 0 (193.12s).
