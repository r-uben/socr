# #1251 (C3): the page judge's acceptance is not a check

Branch `fix/issue-1251-c3`, on top of the merged C2 pass trace (#1332). Brief:
`scratch/army/phaseBC/BRIEF-C3.md`; rulings in `DECISIONS.md` (C3 line, C3 HOLD line, D2).

## Rule

A model-read page whose acceptance is the page judge's (the typed fields `judge_outcome ==
page_judge_completed` and `judge_accepted is True`; the judge's reason text is never read) stays
SUCCESS only when its pass trace shows, on the shipped text:

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
* **Not fired** when the page was accepted without a judge (#1181 owns that), when the judge did
  not accept it, or when the engine is not a model reading.
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
* The file name matches `*.json`, so every reader that globbed `pages/*.json` for sidecars now goes
  through `core/page_files.page_sidecar_paths` (scoreboard, library, table_counts, the four
  refinalize tools); the acceptance meter does not call it non-canonical.

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

## Not done

* The native-engine relabel (brief item 5): untouched, per Fable (`core/manifest.py:2528` retires a
  credentialed stand-in on `page_judge_completed and not audit_passed`).
* `judge_model` stamping on unjudged pages (carbon F-B-008) is a C2 item and is not changed here.
* `socr-refinalize-boilerplate` does not rewrite a verdict at all (it strips a preamble and keeps
  the verdict as it stands), so it does not add the clause; a preamble-stripped judge-only page is
  not demoted by it. `-point-drops` evaluates the clause on the old row-binding entries (its own
  new records are written after the verdict); only `-own-text` has an end to end test.
