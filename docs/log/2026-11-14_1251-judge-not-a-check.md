# #1251 (C3): the page judge's acceptance is not a check

Branch `fix/issue-1251-c3`, on top of the merged C2 pass trace (#1332). Brief:
`scratch/army/phaseBC/BRIEF-C3.md`; rulings in `DECISIONS.md` (C3 line, C3 HOLD line, D2).

## Rule

A model-read page that would ship SUCCESS (any judge-field state: accepted with outcome completed,
an empty outcome as the crop-patched re-judge writes, or no judge field at all; the judge's reason
text is never read; ruling, review round below) stays SUCCESS only when its pass trace shows, on
the shipped text:

* `text_layer_words` PASS at page scope, and
* `text_layer_numbers` PASS at page scope, or NOT_APPLICABLE, and
* no entry of any check is FAIL or UNKNOWN.

Every PASS is digest-verified against the shipped text first (`verify_pass_digests`): a stale or
digest-less PASS is not evidence. Otherwise the page gets one clause, `page_judge=UNKNOWN`,
worded by what is missing ("no independent check confirmed this reading: neither the words nor
the numbers of the page were compared with the page's own text layer; verify it against the page
image before citing"). It never says who judged, because on an escalated page a cloud engine read
and the local model judged.

The judge keeps its power to warn and reject; nothing on that path changed.

Decisions taken in code, each open to the coordinator:

* **Only a page the other checks leave SUCCESS gets the clause.** A page already carrying any
  clause is demoted and marked; a second line says nothing new (Fable C3 point 1, "decide whether
  already-marked pages get the extra line": they do not). The mark is a line added at the top; the
  body text is byte-identical, so "final `.md` byte-identical" holds for the body and not for the
  page's mark block.
* **Distinct failure mode `reading_uncorroborated`**, in `PERMANENTLY_UNRESOLVABLE_MODES`, so a
  resume restores the page instead of re-reading it (Fable C3 point 2): the same layer and config
  give the same absence of evidence, and a re-read is not deterministic, so it would only change
  shipped text. The check name stays `page_judge`; `REPROCESSABLE_CLAUSE_CHECKS` (which keeps
  #1181's pages reprocessable) now asks `clause_is_reprocessable`, which exempts exactly this
  clause. `MODEL_OUTPUT_UNJUDGED` is untouched and stays reprocessable.
* **Not fired** when the page was accepted without a judge by #1181's table-exact path (#1181 owns
  that clause), or when the engine is not a model reading. (Earlier versions also exempted a page
  whose judge fields were not a completed acceptance; the review-round ruling removed that: the
  predicate keys on the absence of evidence.)
* The clause is derived on every pass (status SUCCESS, or already demoted by this mode), and
  `_MODE_CLAUSES` replays it from the mode, like #1181's, so a re-finalized page publishes the
  verdict it had. A resumed page takes its published verdict and trace as before.
* Demotion reaches document status, `metadata.json` and the CLI through the existing page-status
  paths; `tests/test_gh1251_judge_not_a_check.py` pins all three.

## Judge reply persistence

`pages/NNNNN.judge-calls`: a JSON list, one record per page-judge call, in the order they
happened. Record: `page`, `outcome` (`verdict`, `timeout`, `error`, `deadline_exceeded`),
`judge_model`, `judge_backend`, `prompt_sha256` (the digest the run manifest also records),
`reading_sha256`, and for a verdict the full `reply`, `reply_chars` and the `parsed` verdict; for a
call without one, the exception type and a capped message (`JUDGE_CALL_ERROR_MAX_CHARS`, 500, with
its derivation beside it). Recorded at the VLM leaf (`VLMPageJudge`), so the re-judge, the retries
and every rung are covered; our own wall-clock deadline adds a `deadline_exceeded` row from
`_TimeoutJudge`, and a worker that finishes after it still records its own `verdict` row.

* Written by `_flush_page_sidecar` (every sidecar write, so the in-loop, assemble and final flushes
  leave the same file) from `PageState.judge_calls`. A resumed page is loaded with the list its
  original run wrote (digest-checked against the sidecar's reference; a missing or altered file is
  logged and not rebuilt), so the file is preserved byte for byte. A page with no call has no
  file; a stale file from an earlier run of the page is removed.
* The sidecar keeps `judge_calls: {path, sha256, count}`, only when there is a call. Not in the run
  fingerprint (the test compares `run_fingerprint` across a resume).
* The file is named `NNNNN.judge-calls`: it does NOT end in `.json`, so no `*.json` glob reads it as
  a sidecar. Readers that globbed `pages/*.json` still go through
  `core/page_files.page_sidecar_paths` (scoreboard, library, table_counts, the refinalize tools),
  which refuses the companion by name as a second line of defence; the acceptance meter does not
  call the file (or its `.unrestored` copy) non-canonical.

## Part A (C2 hardening, separate commit a7551662)

Scoreboard `records.pages_pass_digest_mismatch` (expected 0); a `row_binding` PASS outside
`table:N` is NOT_RUN; `TraceEntry.to_dict` writes digest keys for a PASS only; test that the
fragment body, the sidecar's `winning_output.text` and every page-scope `text_sha` agree after
assemble (the fragment IS `winning_output.text`: marks are inside both); C2 log wording.

## Measured (cached run, no model)

Script: `docs/log/2026-11-14_1251-measure.py`. Run `fed_rerun_51270eb5_words` (12,413 sidecars,
the brief's 43 qwen SUCCESS pages); gold `gold/fed-2026-11-11.jsonl` joined on the `.md` digest.

**Limitation.** The sidecars carry no pass trace and no own-layer text. The predicate is
recomputed the way `socr-refinalize-own-text` does: socr's page analysis of each PDF (current code)
gives the layer inputs, and the current clause functions run on the sidecar's `winning_output`.
The number is "what this code would label the page", and it includes every check added since the
run. The judge fields come from the sidecar.

| | pages |
| --- | --- |
| qwen SUCCESS pages | 43 (all with typed judge acceptance; all reason "faithful") |
| keep SUCCESS under the rule | **43** |
| demoted by this rule | **0** |
| demoted by the other current checks | 0 |
| documents flipping completed to partial | **0** |
| gold-clean pages present in the run | 234 (D2 bound, 1 in 100: 2) |
| gold-clean pages demoted | **0** |
| gold-defective qwen SUCCESS pages, newly marked | **0** of 9 |

On the earlier `fed_rerun_51270eb5` (before the words refinalize) there are 59 qwen SUCCESS pages;
16 are demoted by the words check alone, 0 by this rule, 43 stay (the same 43).

All 43 pass the independent evidence: words PASS and numbers PASS on the shipped text. The 9
gold-defective ones (7 `lost_text`, 1 `lost_figure`, 1 with no defect type recorded) compare 123 to 320 owed words each with none
missing. So the rule as specified does not mark them: lost text and lost figures are not words the
page's own layer says are missing. Clean pages in the same 43 compare 4 to 334 words (median over
all 43: 126 compared, 304 shipped). A page can pass on very few compared words (min 4).

Consequences, stated plainly:

* The done-test line "the 9 `still_silent` SUCCESS pages become WARNING" is NOT met. The brief's
  blunt rule (demote every judge-approved qwen SUCCESS page) would have met it and cost 15 gold-clean
  pages, over the D2 bound (2 of 234). The independent-evidence rule costs 0 and marks 0.
* The rule bites where the layer owes no word (scanned pages, numbers-only pages), a failed
  trace entry exists (raw control byte, row-binding UNKNOWN), or the words check is NOT_RUN. On this
  run the existing numbers and words clauses already demote those pages. It is a fail-closed guard
  for pages the other checks cannot reach (see the fixtures below), not a fix for the 9.

## Tests

`tests/test_gh1251_judge_not_a_check.py` (44): the predicate table (words, numbers, FAIL/UNKNOWN
anywhere, table-scope pass, stale and absent digest); the clause (reason ignored, scope: accepted
model reading only, cloud engine wording, already-marked page gains no line); the seam (demotion,
mode, idempotence, body untouched); the resume gate; **process level, both provider states**: the
same page and the same judge, only the independent check removed, differs in page status,
`failure_mode`, `audit_passed`, the mark, document status, `metadata.json` and the CLI, and in
nothing else; with no provider the two arms are identical. Resume: the demoted page is restored,
not re-read, byte for byte. Side file: prompt digest, model, full reply, parsed verdict, sidecar
reference; a timeout and a crash recorded as such; kept across a resume with the same bytes and the
same fingerprint; no file on a page no model judged; an altered file is not restored; our own
deadline is recorded once; readers skip the companion file.

Mutants (copy of `src` and `tests` under the scratchpad, `socr.__file__` asserted inside the copy,
uncapped anchor count == 1 asserted before editing, all KILLED by the new tests):

| mutant | killed by |
| --- | --- |
| predicate ignores the words PASS | `test_words_that_did_not_pass_are_no_evidence[NOT_RUN]` |
| predicate ignores FAIL / UNKNOWN entries | `test_any_failure_or_unknown_entry_of_any_check_withdraws_the_evidence` |
| resume re-reads (exemption removed) | `test_the_demoted_page_is_restored_on_resume_not_re_read[provider]` and the clause test |
| clause removed | `test_a_page_whose_layer_owes_no_words_has_only_the_approval` |
| reply not persisted | `test_every_judge_call_leaves_its_prompt_digest_model_and_full_reply` |
| resume does not restore the calls | `test_the_side_file_and_its_reference_survive_a_resume_...` |
| numbers NOT_RUN counts as evidence | `test_numbers_that_did_not_pass_are_no_evidence[NOT_RUN]` |
| a stale digest is accepted | `test_a_pass_computed_on_other_text_is_no_evidence_for_this_text` |

## Existing tests this change failed, and the coordinator's rulings (applied)

Eleven existing tests failed. Rulings, as applied:

1. **Judge-accepted control pages with no independent check** (subject is something else; every
   assertion kept, only the control page's evidence added): `test_gh1013...::test_fields_set_by_
   verification_survive_to_the_ship` (the harness passes the layer's words; the reading carries
   them), `test_gh1185_text_layer_numbers_e2e` (the page also prints a prose line, different on
   every page because text repeated on every page is furniture the words check does not owe; the
   reading carries it), `test_gh1187...::test_the_document_status_turns_on_this_lane_alone` (the
   words check runs and passes in both arms), `test_gh881...::test_the_other_pages_are_processed_
   and_the_bad_page_is_failed` (the fixture's lines repeat on every page, so a real reference is
   empty; the words are given), `test_gh987...::test_short_circuited_pages_are_reprocessed_...`
   (the layer owes "text for", at a place of its own per page).
   Two refinalize-equality tests (`test_gh1294...::test_refinalize_turns_an_old_run_into_...`, and
   the two refinalize tests of gh1185) build their "old run" by patching a check out of the
   current code; the patched run is also a run before #1251, so `_judge_only_clauses` is patched
   out of it too (a run before #1294 or #1185 had no page_judge clause either). Without that the
   old run is demoted by the new clause and carries its mark line, which a real old run does not.
2. **The offline tools apply the clause.** `core.verdict.with_judge_only_clause` is the one
   predicate (accepted model reading, no #1181 clause, `independent_page_evidence`) evaluated on
   `reconcile_pass_trace` of the sidecar's old trace plus the tool's recomputed checks, stamped
   with the text it ships. A page that gains another clause drops the judge-only one (a live run
   gives it only to a page nothing else marks; the seam re-chooses the mode). Wired into
   `socr-refinalize-own-text`, `-point-drops` and `-unshipped-figures`;
   `socr-refinalize-prior-ocr` goes through `finalized_page_record`, the live seam, so it has it.
   Test: `tests/test_gh1251_refinalize_judge_clause.py` (12): an old run (code before #1251, no
   layer evidence) refinalized equals the run of this code (sidecar, body, document md, document
   and root status), idempotent; the same old run on a page WITH evidence is left byte for byte;
   the helper's table (words and numbers pass, digest of other text, no trace, other clause,
   scope).
3. **gh987 byte equality.** A page the breaker short-circuited made no judge call, so its
   `judge_calls` reference legitimately differs from the slow run's: `judge_calls` is volatile in
   those tests' `_shape` (only that key), with a comment.

Mutants of this part (copies outside the repo, `socr.__file__` asserted inside the copy, anchor
count == 1 before editing; all KILLED):

| mutant | killed by |
| --- | --- |
| own_text tool does not call the helper | `test_refinalize_turns_an_old_judge_only_page_into_what_this_code_writes` |
| helper keeps the judge clause beside another | `test_helper_drops_the_clause_when_the_page_gains_another` |
| helper adds the clause whatever the trace holds | `test_a_page_with_an_independent_check_is_left_alone_by_the_tool` and two helper tests |
| helper ignores the scope (accepted model reading, SUCCESS) | the four `test_helper_is_scoped_...` cases |

## Measured finding on the Fed run

On the Fed run C3 demotes **0 of 43** judge-approved qwen SUCCESS pages: all 43 have the words
check and the numbers check at PASS on the shipped text (table above). The 9 gold-defective pages
that are still silent (7 `lost_text`, 1 `lost_figure`, 1 untyped) are invisible to every
text-layer check: what is lost is text or figures that exist in the page IMAGE, not words the page's
own text layer says are missing. Marking them needs an image-level reader (a model or a pixel
comparison of the page against the reading), not another layer comparison. C3 as ruled cannot
reach them and is a fail-closed guard for pages the layer checks cannot reach.

## Review round (PR #1338): rulings applied

PR text now says `Refs #1251`, not `Closes`: the 9 gold-defective pages are not reached (above).

1. **Resume trusts the trace, not an empty verdict.** `resumed_verdict_with_judge_only_clause`
   (`core/verdict.py`), called by `_select_and_finalize_page` before the seam, completes a resumed
   page's published verdict with the clause when the trace the page WILL WRITE (the sidecar's
   replay, each PASS kept only if it names the shipped text; no trace writes every slot NOT_RUN)
   holds no independent evidence. Any other published clause is kept as it stands; a verdict that
   carries a clause is returned unchanged. Test: `test_gh1251_round2.py` resume tests (two arms of
   one resume that differ only in the sidecar's words PASS, in both provider states).
2. **The predicate keys on absence of evidence.** `_judge_accepted` is gone: every model reading
   that would ship SUCCESS needs `independent_page_evidence`, else the clause. Applies to the live
   seam, `with_judge_only_clause` and the resume helper. **This turns 55 existing tests red** (stub
   model readings with no layer evidence and no completed judge outcome); they were left as they
   were and are listed in the PR report for the coordinator.
3. **Evidence is visible.** The words PASS entry now reads "... (compared N word(s); the page ships
   M)". Scoreboard: `records.model_success_without_verified_evidence` (a SUCCESS model page whose
   own trace holds no digest-verified evidence of its own text; expected 0) and
   `records.pages_weak_words_pass` (a page-scope words PASS whose sidecar `text_layer_words.compared`
   is below `WEAK_WORDS_PASS_BELOW`). The release rule was not changed on this basis.
   *Weak words passes.* On the 43 qwen SUCCESS pages of the Fed words run, `compared` ranges 4 to
   550 (deciles 4, 5, 19, 19, 38, 126, 165, 235, 278, 334, 550); 13 of 43 are below 20, 20 of 43
   at or below 30. `WEAK_WORDS_PASS_BELOW = 20`: about one sentence of running text, the lower
   quartile of the measured run (19 to 25). It is a reporting bucket and it separates nothing the
   gold labels call defective (one defective page compared 5 words, the other eight 123 or more),
   so it is a counter, not a gate. Recompute it from `docs/log/2026-11-14_1251-measure.py report`.
4. **Control fixtures run the real check.** gh1187, gh881, gh1013 no longer stub the words
   comparison: the stub readings carry the layer's words and the tests assert the real words PASS
   on the sidecar trace. gh987 keeps `judge_calls` out of `_shape` and now asserts it: the slow
   judge's pages have one record (`deadline_exceeded`), the one page that paid the deadline in the
   wedged run has the same, every short-circuited page has no reference.
5. **Judge-calls durability.**
   * Seeded: `_load_rejudge_candidate` (after its fingerprint and checksum match) and
     `_reset_page_for_retry` (`--retry-failed-pages`) carry the page's existing calls forward, so
     the next flush extends the file; the first attempt's timeout is not lost.
   * Unreadable (missing, altered, unparsable) is KEPT: the writer never unlinks it while the page
     holds no call, keeps the sidecar reference with `unavailable: <reason>`; if new calls arrive,
     the unreadable file moves aside once to `NNNNN.judge-calls.unrestored` and the reference gets
     `prior_unavailable: {path, sha256, count, reason, kept_as}`, carried through later flushes.
   * Raw reply before parsing: `parse_verdict` raises `JudgeReplyUnparseableError` (a
     `ValueError`, as before) holding the full reply; a truncated reply
     (`JudgeReplyTruncatedError`) holds what the backend returned; both are kept in the record.
   * The call record is built inside the fail-open boundary (`VLMPageJudge._record`).
   * Late records. Design chosen: **record `deadline_exceeded` and ignore later appends** from the
     abandoned worker. The deadline row and an abandon flag are set under one lock
     (`_JUDGE_CALLS_LOCK`), the worker's own record checks the flag under the same lock, and the
     writer snapshots the list under it. The late answer is discarded by the pipeline anyway
     ("late page-judge result discarded"), so the file and the page never differ by a record that
     lands after a flush. Cost: the late reply itself is not kept (the deadline row says the call
     was given up on). Test: `test_a_call_that_answers_after_our_deadline_is_not_recorded...`.
   * One prompt identity: `judge.judge.page_judge_prompt_sha256` is the single helper; the run
     manifest's `page_judge_prompt_digest` calls it, and an unreadable prompt yields the same
     unique marker form for both.
   * **Limit:** the table-escalation canary (`tables/escalation_canary.py`) also calls a judge and
     leaves no record: it has no page state to hang one on, so recording it is not cheap.
6. **`socr-refinalize-point-drops`** derives the clause after the repaired row-binding payloads
   exist (they are built before the verdict and handed to the helper as `row_binding_records`).
   Test: the spy in `test_gh1251_round2.py`. Imports sorted (ruff `I`); `docs/OUTPUT.md` marks the
   `judge-calls` record fields as conditional; the measure script parses `run OUT [--words]` and
   `report OUT` with `argparse`.

Mutants of this round (copies of `src`, `tests` and `docs` outside the repo, `socr.__file__`
asserted inside the copy, uncapped anchor count == 1 asserted before editing; all KILLED):

| mutant | killed by |
| --- | --- |
| resume trusts the empty published verdict (helper call removed) | the resume test, both provider states, and the helper table |
| predicate gated on the judge fields again | the three judge-field cases and the live-seam table |
| re-judge does not seed the first run's calls | `test_a_rejudge_extends_the_first_runs_calls...` |
| unreadable file deleted when the page holds no call | the writer test and the resumed-page test |
| unreadable file overwritten by new calls (not moved aside) | `test_an_unreadable_calls_file_is_kept_and_flagged...` |
| late record after the deadline is recorded | `test_a_call_that_answers_after_our_deadline...` |
| retry reset drops the first attempt's calls | `test_a_retried_page_keeps_the_first_attempts_judge_calls...` |
| raw reply dropped on a parse error | `test_a_reply_that_does_not_parse_is_still_kept_in_full` |
| call record built outside the fail-open boundary | `test_a_failure_to_build_the_call_record...` |
| point-drops derives the clause without the repaired payloads | the spy test |
| words PASS does not name what it compared | `test_the_words_pass_names_how_many_words...` |
| scoreboard does not count model SUCCESS without evidence | the scoreboard test |

**Existing tests turned red by ruling 2 (50 in the final full run).** Not edited. Each builds a stub model reading
(`engine="qwen"`, SUCCESS, a heuristic or hand-set acceptance with no completed judge outcome) on a
page whose layer gives no words evidence, and asserts it stays SUCCESS / the document completes.
They pass on the previous head of this branch; restoring the judge-field gate in a copy makes them
pass again, so ruling 2 is the only cause (resume helper and offline helper included). Per file:

  * `tests/test_a1c_header_binding_unverified_surfacing.py`: 1
  * `tests/test_canon_remediation.py`: 2
  * `tests/test_canon_round2.py`: 3
  * `tests/test_canon_round3.py`: 3
  * `tests/test_chart_lane.py`: 2
  * `tests/test_gh1005_native_status_follows_shipped.py`: 1
  * `tests/test_gh1145_page_verdict_marks.py`: 2
  * `tests/test_gh1156_part_a_interim.py`: 1
  * `tests/test_gh1181_exact_pass_unjudged_clause.py`: 3
  * `tests/test_gh1185_refinalize_tool_unit.py`: 2
  * `tests/test_gh1218_drops_clause.py`: 1
  * `tests/test_gh1267_row_binding_pipeline.py`: 2
  * `tests/test_gh161_resume_ledger_audit_gate.py`: 2
  * `tests/test_gh190_empty_table_surfacing.py`: 1
  * `tests/test_gh225_fabricated_image_urls.py`: 1
  * `tests/test_gh226_table_emission_guard.py`: 1
  * `tests/test_gh851_escalation_latch_evidence.py`: 1
  * `tests/test_gh974_review_pins.py`: 1
  * `tests/test_gh993_readable_tables.py`: 3
  * `tests/test_ladder_e2e.py`: 3
  * `tests/test_ladder_resume.py`: 1
  * `tests/test_native_cell_advisory_quiet_side_effects.py`: 2
  * `tests/test_orchestrator.py`: 1
  * `tests/test_p6_stage_ab_difference.py`: 1
  * `tests/test_p6_stage_c_difference.py`: 2
  * `tests/test_resf_native_cell_pipeline.py`: 4
  * `tests/test_s1_structure_class_winner_gh_reachability.py`: 2
  * `tests/test_structural_gate_b1_gh151.py`: 1

The coordinator rules: either each fixture gets the layer's words (as the five of round 1), or
the rule is narrowed.

## Not done

* The native-engine relabel (brief item 5): untouched, per Fable (`core/manifest.py:2528` retires a
  credentialed stand-in on `page_judge_completed and not audit_passed`).
* `judge_model` stamping on unjudged pages (carbon F-B-008) is a C2 item and is not changed here.
* `socr-refinalize-boilerplate` does not rewrite a verdict at all (it strips a preamble and keeps
  the verdict as it stands), so it does not add the clause; a preamble-stripped judge-only page is
  not demoted by it. `-point-drops` evaluates the clause on the old row-binding entries (its own
  new records are written after the verdict); only `-own-text` has an end to end test.
