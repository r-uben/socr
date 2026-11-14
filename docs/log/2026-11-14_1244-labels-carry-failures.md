# #1244 / #1204 (fix plan C1): document and page labels carry failures up

Branch `fix/issue-1244-c1`. Labels only: page text, winner selection and page statuses are unchanged.

## What changed

1. **Document status invariant** (`pipeline/orchestrator.py`, `_pages_not_success`). A document is
   `completed` only if EVERY page 1..page_count has a final record with status SUCCESS whose PUBLISHED
   `audit_passed` is true (review round: a SUCCESS page with a FAIL clause, or with an internal acceptance
   that is not `True`, publishes false and blocks `completed`). A page with no record counts as incomplete,
   so an empty or partial record list can never read as clean (no `all()` over a list). It runs twice:
   - on `pre_records`, in the `pages_ok` chain, after the existing buckets. The recovered-prose bucket
     (`scanned_prose_recovered_pages`, computed and reported before, but not gated) is now gated in the
     chain too (#697); a recovered-prose document is named by its own note, not the generic one;
   - on the FINAL records (after the figure phase, the chart-region pass and the final-body guard), so a
     late demotion reaches `final_result`, `state.status`, `metadata.json` and the CLI. It re-writes the
     metadata (provisional flag preserved, GH-503) and runs before the authoritative sidecar and fragment
     flush, so terminal sidecars carry the same record. When the final body does not split into
     `page_count` pages no final record can be read, so this check FAILS CLOSED: every page counts as
     a page without a final record and a document that was SUCCESS cannot stay `completed`.
   A document demoted only by this invariant carries the named reason
   `page(s) N did not finish with status success and a passing audit` in `error`. A document already demoted by a bucket keeps
   that bucket's reason and gets nothing appended. No new audit event kind, so nothing new to restore on
   resume (`tests/test_resume_restore_kinds.py` unaffected).
2. **Published `audit_passed`** (`core/verdict.py`, `published_audit_passed`). The sidecar's top-level
   `audit_passed` is derived at record time: status SUCCESS, internal acceptance exactly `True`, and no
   FAIL clause in the verdict. UNKNOWN does not block. WARNING and ERROR are false regardless of clauses.
   The internal acceptance is an input, never an output.

   Internal acceptance for resume/replay is `winning_output.audit_passed` (the serialised
   `PageOutput.audit_passed`, the #252 winner-selection flag). It is untouched.
   Including the internal flag in the derivation is deliberate: the published value can only be `true`
   where it was `true` before.

## Reader audit of `audit_passed`

Top-level sidecar key (now the derived label):

| Reader | Treatment |
| --- | --- |
| `benchmark/scoreboard.py:152` | wants the label; its two `labels.*` counters now measure the fix |
| `benchmark/acceptance.py:365` (`rule_qwen`, table-only bypass) | `bypassed and published true`; derived is a subset of internal, so it can only flag fewer |
| `benchmark/acceptance.py:345-347` (`success_common`) | reads top level AND nested; a SUCCESS page with a FAIL clause now trips `SUCCESS_AUDIT_FALSE`, which is correct |
| `pipeline/refinalize_{own_text,prior_ocr,unshipped_figures,point_drops}.py` | each copied the nested flag to the top level; now call `published_audit_passed` so a refinalized sidecar publishes the same derivation. `refinalize_own_text` also re-derives it on its repair path (a page that already carries the clause), so a stale `true` left by an earlier pass is corrected on the next run |
| `pipeline/refinalize_boilerplate.py:165` | set nested False and left the top level stale `true`; now also sets the top level False |
| `pipeline/refinalize_prior_ocr.py:272` | reads cached attempt dicts (`r.get`), not the sidecar |

Nested `winning_output.audit_passed` (internal, unchanged): `orchestrator.py` `_load_terminal_page`
(gate at the GH-161 check), `_resume_prior_ocr_reading_was_accepted` (raw_winning), the JUDGE_TIMEOUT
ladder resume (`winning.get("audit_passed") is not False`), `_record_ledger_audit_reject`,
`review/html.py:151`, `acceptance.py:345-347`, and the `PageOutput.to_dict/from_dict` round trip.
`result.py:705/778` is the `PageOutput` / `EngineResult` field, not the sidecar key.

Resume eligibility is unchanged because it reads only the nested value.

## Tests (`tests/test_gh1244_labels_carry_failures.py`)

- `published_audit_passed`: WARNING with UNKNOWN-only clauses, SUCCESS with FAIL, SUCCESS with UNKNOWN only,
  internal-false SUCCESS, missing/None acceptance, ERROR, dict form.
- `_pages_not_success`: missing record, empty list, subset, every non-SUCCESS status.
- Late demotion (`pre`, `final`, `both`), both provider states: baseline, fix-off and fix-on arms. Text
  byte-identical across all three, nested acceptance identical, document status / metadata / error / result
  audit_passed move only with the fix.
- No-demotion scope guard: fix on/off give identical selection and the same document status.
- Prior-OCR fixture (WARNING page, internal true), `accept` x provider: selection identical on/off, published
  label derived, nested unchanged.
- Resume: skip decisions identical on/off for the accepted-prior-OCR exception (page 1 skipped, restored
  sidecar byte-identical, nested true, published false) and the rejected arm (page 1 re-read).
- Review round: `test_a_success_page_that_publishes_false_blocks_completed` (SUCCESS page with a FAIL
  clause, and with internal acceptance False, demoted in the pre and the final records; fix-off arm equals
  the baseline, fix-on is AUDIT_FAILED naming only page 2),
  `test_a_body_that_does_not_split_fails_closed` (the `_phase_assemble` final split yields one body too
  many; the named reason covers every page),
  `test_a_recovered_prose_document_names_its_own_bucket_not_the_generic_reason`,
  `test_a_rerun_rederives_the_published_audit_passed_of_a_page_it_already_marked`
  (`test_gh1185_refinalize_tool_unit.py`), and
  `test_the_control_helper_accepts_only_the_designed_unchecked_demotion` (the control helper rejects a page
  outside the fixture's design, an unrelated note, a failed document and a missing C1 note).
  `test_a_demoted_page_stops_the_document_reading_completed` and
  `test_the_fix_changes_no_selection_on_a_document_with_no_demotion` now compare against the baseline
  unconditionally.

### Mutants

Procedure, per mutant: copy `src` and `tests` (and `pyproject.toml`) to a fresh directory outside the
repository, run with `PYTHONPATH` set to the copy and assert `socr.__file__` resolves inside it, assert the
guard's anchor text occurs exactly once (uncapped count) before editing, remove or invert the guard, and run
the test that owns it. A mutant counts only if that test fails.

| Mutant | Result |
| --- | --- |
| M1 missing record counts as SUCCESS | `test_pages_not_success_counts_missing_records_as_incomplete` fails |
| M2 published flag copied from internal | 8 failures, including the helper table, `test_published_label_differs_from_the_internal_acceptance[True-True]`, `test_resume_eligibility_is_unchanged_by_the_label[True]` |
| M3 internal acceptance corrupted in fresh selection (published value written into `winning_dict`) | `final-*`/`both-*` demotion tests, the label test, the resume test, and 6 of `test_gh1192_resume_accepted_prior_ocr.py` fail |
| M4 no late demotion | `final-True`, `final-False` fail |
| M5 no pre-record gate | `pre-True`, `pre-False` fail |
| M6 FAIL clause ignored | two helper cases fail |
| M7 no fail-closed on a body that does not split | `test_a_body_that_does_not_split_fails_closed` fails (both provider states) |
| M8 invariant reads status only, not the published label | 8 failures in `test_a_success_page_that_publishes_false_blocks_completed` |
| M9 recovered-prose bucket not gated | `test_a_recovered_prose_document_names_its_own_bucket_not_the_generic_reason` fails (both provider states) |
| M10 refinalize repair path does not re-derive the label | `test_a_rerun_rederives_the_published_audit_passed_of_a_page_it_already_marked` fails |
| M11 control helper ignores the design's page set | `test_the_control_helper_accepts_only_the_designed_unchecked_demotion` fails |
| M12 control helper accepts any other note | same test fails |

## Fallout in existing tests

57 tests failed on the first full run. Cause, per `_pages_not_success` spy over the failures: most held
a control document whose model-read pages are `numbers_unchecked` WARNING (a model reading is CHECKED only
against a born-digital layer of its own that shares a number with it; blank-PDF and hand-built fixtures have
none) and which the old gate reported SUCCESS. Fixed by giving control pages a layer and a number where the
test needs a clean page, by `tests/c1_support.py::assert_clean_or_only_unchecked` where the test is about the
loop, or by moving a difference test off the document status onto the named reason. The p6 stage-C oracle
gained `PUBLISHED_AUDIT_PASSED_1244_DIFFERENCES` (pages 4, 5, 7: `true` -> `false`); nothing else moved and
`winning_output.audit_passed` stays out of that list.
`test_1053_figure_crops.py::test_native_only_ships_inline_crops_and_demotes_the_document` pinned
`on.status is not off.status`, which was the hole itself; it now asserts the crop page WARNING, the
neutralised document AUDIT_FAILED, and the reason text on each arm.

Review round on the changed tests: `tests/c1_support.py::assert_clean_or_only_unchecked` now takes the pages
the fixture is designed to leave unchecked (`pages=`) and the other notes it is designed to carry
(`other_notes=`), and fails on any other page, note or status. The two difference tests that lost their
failing-arm check regained it (`test_gh359_ladder_terminals.py::TestRuling1LastRungPassLow` low arm, and the
three arms in `test_gh367_adjudication_lift.py`, with `result_off.error` free of `table_unverified`).
`test_gh1050_text_coverage.py` carries a per-test reason: two of its five off arms are plain SUCCESS pages
and keep their strict assertion, the other three ship a figure without its values.

Full suite after the review round: 10,407 passed, 49 skipped, 12 xfailed, 0 failed. `ruff@0.16.0 format --check .` clean.

## Measurement (cached run `fed_rerun_51270eb5_words`, 156 documents)

Computed offline on the cached sidecars and metadata with the shipped `published_audit_passed` and the
invariant's rule (every page 1..N has a sidecar with status `success`). The pipeline was not re-run.

| Metric | Before | After |
| --- | ---: | ---: |
| `completed` documents with non-SUCCESS pages | 11 | 0 |
| pages with a FAIL clause and published `audit_passed=true` | 1,087 | 0 |
| non-SUCCESS pages published `audit_passed=true` | 1,636 | 0 |
| documents `completed` / `partial` | 26 / 130 | 15 / 141 |

Review-round check on the same cache (8,109 SUCCESS page sidecars): SUCCESS pages carrying a FAIL clause: 0;
SUCCESS pages whose internal acceptance is not `True`: 0; completed documents blocked only by the published
label (all pages SUCCESS): 0. So gating the invariant on the published label moves nothing on this run; it
closes the shape for a future one.

Documents that flip `completed` -> `partial` (pages that block):

- beige books, one WARNING page each (page 3): 2009-09-23 (48 pp), 2010-03-16 (50), 2010-11-03 (48),
  2011-03-15 (51), 2011-06-22 (72), 2011-09-21 (52), 2012-04-25 (61), 2013-03-20 (50), 2015-04-29 (62)
- greenbook supplements 2003-01-29 and 2003-10-28 (5 pp each): WARNING page 3, ERROR pages 4 and 5. Their
  metadata `error` already names the withheld table; the status now agrees with it.

## Not changed

- Page statuses, failure modes, selected text, byte identity of the final `.md`, and which pages resume skips.
- The 1,636 WARNING pages keep their marks. Fixing labels does not fix content.
- UNKNOWN-only WARNING pages publish `audit_passed=false` (status), but a SUCCESS page carrying only UNKNOWN
  clauses still publishes `true` (DECISIONS, C1).

## Consequence to decide

The invariant makes "completed" require every page to have status SUCCESS. A model-read page whose reading
ships no number, or that has no layer of its own, is `text_layer_numbers=UNKNOWN` WARNING (#1185), not SUCCESS,
and so blocks `completed` for its document. On the Fed run that moved only the 11 documents above, but a purely scanned, number-free corpus would
never reach `completed` until #1208 provides an independent reader. The earlier #1185 note ("UNKNOWN stays at the
page") is superseded by the C1 ruling.

## Resume cost

A document that leaves `completed` is no longer skipped whole on the next run: its non-SUCCESS pages are
re-read every run (a model page is paid again, and its text can differ). For the 11 Fed documents that left
`completed` this is 15 pages in total (computed from the cached sidecars): 1 WARNING page each in the nine beige
books, and 1 WARNING + 2 ERROR pages in each of the two greenbook supplements. A scanned corpus whose pages
are all unchecked model readings would pay for every page of every document until #1208 provides an
independent reader.
