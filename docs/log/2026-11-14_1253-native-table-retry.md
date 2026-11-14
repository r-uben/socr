# #1253 B3: a refused native table gets one model attempt

Branch `fix/issue-1253-b3`. Brief: `scratch/army/phaseBC/BRIEF-B3.md`; decision: `DECISIONS.md` (B3);
binding conditions: Carbon's B3 section in `advice-carbon2.md`.

## Problem

513 pages of the 156-document run ended as "failed table, see image" with `attempts_summary = [native]`
and no model attempt. `_plan_native_table_first` runs before the provider ladder; a REFUSE plan removed
the page from `ocr_pages`, so `route_page` never saw it. Reasons: row_count 267, structure_defective 111,
unlocalizable_cells 97, region_unverifiable 29, numeric_orphan 9.

## Change

* `MODEL_RETRY_REFUSAL_REASONS` (`tables/native_first.py`) is an allowlist by name of five reasons.
  `header_unattributed`, `label_binding` and `text layer unreadable` stay refused. A new refusal reason
  stays refused until it is added there.
* In `_phase_agentic`, per page, once the ladder exists: a REFUSE plan whose reason is on the list, with a
  non-empty ladder, runs `_refuse_native_table_first` unchanged (flags, floor PNG, native attempt,
  `native_table_cell_unresolved`), sets `ps.native_table_model_retry = <reason>`, and drops the plan so the
  page takes the whole-page route with every downstream table check (reread, row-binding, sign/point
  checks, table judge gate). No new acceptance path.
* Ordering (condition 3): `_plan_native_table_first` is untouched and still runs before the ladder is
  built. The gate sits in the page loop, where the ladder is known. An empty ladder (no provider, or
  `--strict-local` with no local rung) never reaches it: behaviour identical to main.
* One `native_table_model_retry` event per retried page, written from the first finalized record:
  `reason`, `attempted`, `accepted`, `why_not_accepted`, `model_attempts`, `selection_provenance`,
  `ending`, `primary_reason`, `shipped_refused_native_grid`. `accepted` means selection took a passing
  non-native best output. A later row-binding withhold can still demote an accepted page; that is
  recorded by its own events.
* The event is in `_RESUME_REPLAYED` and `tests/test_resume_restore_kinds.py` (74 kinds). The
  `native_table_model_retry` reason string is persisted in the sidecar and restored.

## Binding conditions

* P0-1: the retry calls the refusal bookkeeping first, so `native_table_structure_failed` and
  `native_table_unverifiable` are set. With them set, a qwen attempt that is not accepted meets the D3
  floor before `NATIVE_FALLBACK`. The event's `shipped_refused_native_grid` is the tripwire.
* P0-2 and #262: `d3_floor_kept_model_output` returns `None` on a page with `native_table_model_retry`.
  #262 would otherwise keep a REJECTION_JUDGE_ONLY / AMBIGUOUS_DEFERRED grid over the floor as
  UNACCEPTED_OUTPUT_KEPT, which on a chart-like row_count page is a grid read from axis ticks. On a retried
  page a reading is accepted (replaces the marker) or the marker ships. #262 is unchanged on every
  other page. The 75 baseline pages that shipped unaccepted via #262 are on pages that reached qwen
  without this flag (never retried); they are untouched.
* Prose: the not-accepted ending is the D3 floor, which reads no model state, so its body equals the
  refused arm's body. Pinned by a byte-equality test between the retry arm and main.

## Tests (`tests/test_native_table_model_retry_1253.py`)

Driven through `process()`, `route_page` stubbed, engines pinned, `_resolve_judge_model` patched to "". Each
pins a difference between two runs in the same process (retry allowlist on vs emptied = main), in both
provider states. No absolute status tuple is pinned.

* provider: retry arm routes the page and records it; off arm does not; markdown differs only by the model reading.
* none: routes, markdown, status, event kinds identical to the off arm; no retry event.
* rejected reading: same markdown as main, status error, refusal flags set, not `native_fallback`, not
  `unaccepted_output_kept`.
* accepted and rejected counted apart; no event reports a shipped refused grid.
* each of the five reasons retried; the other three stay refused.
* 513 pages: `SOCR_RETRY_FULL_SCALE=1` runs the measured count (this run: 513 events, 513 routed, 0
  shipped refused grids, 0 unaccepted-kept; about 65 s). The default suite runs 48 pages.
* resume: flag and event survive flush and restore.

## Mutants (copy of src + tests outside the repo, `socr.__file__` inside the copy, anchor count 1)

| Mutant | Tests that fail |
| --- | --- |
| P0-1: skip `_refuse_native_table_first` in the retry branch | `retry_differs_from_main...[provider]`, `rejected_reading_ends...[provider]` |
| P0-2: remove the `native_table_model_retry` return in `d3_floor_kept_model_output` | `rejected_reading_ends...[provider]` (a model reading ships over the floor), `rejected_reading_is_kept_by_262_without_the_exemption` |
| allowlist: drop the reason condition (retry every REFUSE) | `other_refusal_reasons_stay_refused` x3, `retry_differs...[provider]` |
| allowlist: add `header_unattributed` | `other_refusal_reasons_stay_refused[header_unattributed]`, `allowlist_is_the_six_named_reasons` |

## Review round 1 (PR #1320)

* An ACCEPTED retry lifts the refusal (`_lift_native_table_refusal`): `native_table_structure_failed`,
  `native_table_unverifiable`, its ordinals and the floor picture go back to their pre-refusal values, as
  `_crop_repair` does. Before this, the accepted page landed in `d3_floor_pages` (false
  `table_region_unverifiable`, red CLI line, document error). A rejected retry keeps them. The native grid's own
  `table_structure_failed` / `table_not_scorable` events are facts about the layer and still reach the
  document note; that is untouched here.
* The retry event is recorded after the row-binding withhold and the point repairs, and `accepted` also
  requires the shipped page not to be ERROR. Assemble's final-body guard restates it
  (`_restate_native_table_model_retry`, an edit of the existing event so a resumed page keeps its record).
* `--retry-failed-pages`: the dropped plan and the pre-refusal flags are kept by the loop (not on `PageState`,
  which the reset replaces) and the plan is put back for pass two, so the refusal, the #262 exemption and the
  retry flag are applied again.
* A `cell_mismatch` page whose repair could not be confirmed, or whose spliced grid re-plans to an
  allowlisted REFUSE, is retried too. `cell_mismatch` joins the allowlist (six reasons); it can only be a CELLS
  plan, never a plan-time REFUSE, and the plan is replaced only when a provider exists, so no-provider runs end
  as before.

Mutants (copy outside the repo, `socr.__file__` inside the copy, anchor count 1), all killed: skip the lift
(`retry_differs...[provider]`); drop the shipped-status condition
(`accepted_then_withheld...[provider]`); drop the assemble restate
(`final_body_guard_conviction...[provider]`); drop the pass-two plan restore
(`retry_failed_pages_second_pass...[provider]`); drop the cells retry
(`cell_mismatch_page...` x2). 513-page stub, accepted arm: 513 routed, 513 accepted, 0
`table_region_unverifiable`, document SUCCESS.

## Not measured

Recovery rate on the 513 pages and the added qwen time need the cluster re-run (brief says expect fewer than
130 recovered; row_count on chart-like pages will mostly be noise that the checks reject). The gold-page
guard counts (D2, D4) also come from that run. The brief's own done test ("pages with a cell_unresolved
event and no qwen attempt: 513 -> 0") is gameable because that event is emitted inside the refusal
function; the pin here is the retry event count per refused page instead.
