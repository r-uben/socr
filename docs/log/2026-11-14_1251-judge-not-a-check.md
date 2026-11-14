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

## Existing tests this change fails (NOT edited; coordinator ruling needed)

Full suite on the commit: 14 fail, 10,636 pass.

Group 1, the rule's own effect. Each builds a model-read page, accepted by a fixed judge, on a
rotated or otherwise layer-less fixture where no words are owed, and asserts it is clean SUCCESS.
Under the rule that page has only the approval, so it gets the clause:

* `test_gh1013_rejudge_timed_out_candidate.py::test_fields_set_by_verification_survive_to_the_ship`
* `test_gh1185_text_layer_numbers_e2e.py::test_the_reading_that_drops_the_point_is_marked_and_the_faithful_one_is_not`
* `test_gh1185_text_layer_numbers_e2e.py::test_refinalize_then_resume_equals_a_run_of_this_code`
* `test_gh1185_text_layer_numbers_e2e.py::test_a_disagreement_reaches_the_document_the_root_index_and_the_cli`
* `test_gh1185_text_layer_numbers_e2e.py::test_refinalize_carries_a_disagreement_to_the_document_and_the_root_index`
* `test_gh1187_unshipped_figure_pages.py::test_the_document_status_turns_on_this_lane_alone`
* `test_gh1294_layer_words.py::test_refinalize_turns_an_old_run_into_what_a_run_of_this_code_writes`
* `test_gh881_per_page_load_failure.py::test_the_other_pages_are_processed_and_the_bad_page_is_failed`
  (the document note counts 4 untrusted pages, not 3)

Two of these are a real design question: the offline tools (`socr-refinalize-own-text`) do not
apply this clause, so "refinalize an old run equals a run of this code" no longer holds for a
judge-only page. Either the tools apply it too, or the equality tests exempt it.

Group 2, the side-file name. `test_gh987_judge_breaker_e2e.py` (6 tests: wedged judge, breaker
surfaced once, raising probe, vLLM probe, short-circuited resume, breaker event on resume) reads
`out_dir.rglob("pages/*.json")` and does `int(path.stem)`, which fails on `00001.judge-calls`. The
brief fixes the name; moving the file out of `pages/` (for example `judge/NNNNN.json`) or
filtering the helper are the two ways out.

## Not done

* The native-engine relabel (brief item 5): untouched, per Fable (`core/manifest.py:2528` retires a
  credentialed stand-in on `page_judge_completed and not audit_passed`).
* `judge_model` stamping on unjudged pages (carbon F-B-008) is a C2 item and is not changed here.
* The offline refinalize tools do not add the clause (see above).
