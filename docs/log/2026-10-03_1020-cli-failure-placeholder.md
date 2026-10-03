# 2026-10-03 GH-1020: a CLI failure placeholder is CLI_ERROR, not SUCCESS

## Bug

`BaseEngine.process_pages` (`src/socr/engines/base.py`) treated any non-empty page text as
SUCCESS with `audit_passed=True`. qwen-ocr-cli, on a backend error (Ollama 500), exits 1
and writes `*[OCR failed for page N]*` into the page slot (`qwen_ocr/processor.py::_ocr_pages`,
reason in `DocResult.page_errors`). socr read the marker as text, judged it, and could ship it.

## Engine survey (sibling repos read only)

- qwen-ocr-cli: `*[OCR failed for page {idx}]*` per page (idx = position in the image dir);
  `*[OCR Failed]*` for a document with no pages (`processor.py` `_write_document`).
- mistral-ocr-cli: `*[OCR Failed]*` for a document with no pages (`processor.py`).
- gemini, marker, glm, deepseek, nougat: no sibling source on this machine writes a
  placeholder; they skip the page file and exit non-zero, already CLI_ERROR/EMPTY_OUTPUT.
  Not verified beyond socr's own wrappers.

## Change

- `CLI_FAILURE_PLACEHOLDERS` + `is_cli_failure_placeholder()` in `engines/base.py`: two exact
  regexes with sources cited, matched against the WHOLE stripped text (never a substring, so
  a page quoting the marker is untouched).
- `process_pages`: a placeholder page (checked raw and after `_clean_output`, which covers the
  aggregated qwen read-back) becomes `ERROR` / `CLI_ERROR` / `audit_passed=False`, carrying the
  CLI exit note. Applies whatever the exit code.
- `process_document`: an exit-0 whole-document placeholder becomes an `ERROR` / `CLI_ERROR` result.

The structured signal (`page_errors` in the CLI's metadata.json) was not used: it is keyed by the
CLI's own image-dir index and is only written for qwen; the marker is the one signal common to
both CLIs that emit one. The placeholder-to-ERROR mapping reuses the existing CLI_ERROR path, so
page, document, metadata and CLI surfacing are the existing ones.

"No judge call": the shipped judges refuse a non-SUCCESS/empty output without a model call
(`agentic.py` "empty/error output"), and `route_page` then escalates to the next rung.

## Tests

`tests/test_gh1020_cli_failure_placeholder.py` (12): marker recognition and non-recognition
(substring, embedded, near-miss); real `process_pages` with a faked subprocess, success vs
placeholder+exit 1 vs placeholder+exit 0; agentic loop run twice, differing only in what rung one
wrote: the judge saw the real text in one leg and never the marker in the other, the next rung ran
only in the failure leg, and the marker is absent from written markdown. Hermetic (ladder patched,
judge model "", `get_engine` stubbed, subprocess faked); no absolute status pinned.

Mutant (external copy of src+tests, `socr.__file__` canary, anchor count 1): neutering the matcher
(`return False and any(...)`) fails 6 of the 12 tests.

## Follow-up (not done, other repo)

qwen-ocr-cli `backends/base.py`: include `resp.text[:300]` in the 5xx error and retry once.

## Review fixes (Astra: ACCEPT-WITH-FIXES)

1. **Embedded markers.** The matcher is now per LINE: a marker alone on a line anywhere in a
   page means part of it is missing, so the whole page is ERROR / CLI_ERROR with no text (the
   readable remainder is not shipped as a partial SUCCESS). `process_pages` checks each page
   (aggregate sections are split per page first, so one bad section fails only that page);
   `process_document` fails an exit-0 aggregate that contains a marker line.
2. **Best-effort.** `agentic.is_failed_candidate()` (status ERROR or failure_mode CLI_ERROR).
   `_best_effort` excludes such attempts from its usable pool. If every attempt failed it still
   returns a failure (unchanged contract), never a promoted one. VLM path: `VLMPageJudge` and
   `HeuristicPageJudge` refuse non-SUCCESS before any render or model call; the test uses the
   real classes and asserts the model/renderer/checker were never touched.
3. **Table rungs.** The rungs read raw markdown with no status check, so the gate is where the
   candidate is handed over: `_run_table_judge_gate` returns early, and the
   `_escalate_table_page` call site skips, on `is_failed_candidate`. A failed output normally has
   empty text (already skipped by `not bo.text`); the gate makes that independent of the text.
   The escalation call-site gate is not covered by a mutant (needs a full-loop fixture).
   **Correction:** the earlier claim that "every judge refuses such output" was too broad. The
   page judges (Heuristic, VLM, the table-verifier wrappers when delegating) refuse non-SUCCESS;
   the table ladder and its rungs do not, which is why they are gated above.
4. **Known false positive.** A page whose entire text is a marker string is classed as a CLI
   failure. A real paper will not consist solely of that string. A marker quoted inside a
   sentence is not a line of its own and is untouched (pinned).
5. **Mutants** (external copies, canary, uncapped anchor count 1 each): per-line matching
   reverted to whole-text fails 6 tests; dropping the `_best_effort` exclusion fails
   `test_best_effort_never_selects_a_failed_candidate`; dropping the table-gate check fails
   `test_table_ladder_is_never_entered_for_a_failed_candidate`.

## Round 3

- **Decision (coordinator): a page with a marker line stays a whole-page CLI_ERROR.** Astra asked
  to keep the readable remainder as WARNING. Overruled: the rung failed, so the ladder moves on
  and the next rung re-reads the whole page. A partial read is never better than a full re-read,
  and if every rung fails the existing floor/fallback applies.
- **`_best_effort`.** The correction to round 2: the earlier "unchanged contract" claim was not
  true (all-failed fell back to every attempt even when a non-empty failed one existed). Now the
  pre-PR selection (`usable` = non-empty, else all attempts, same `max` key) is kept verbatim and
  the single change is that a healthy (non-failed) non-empty candidate always wins over a failed
  one. `test_best_effort_all_failed_matches_the_pre_pr_selection` compares to a copy of the main
  implementation on all-failed inputs; the other test pins the one intended difference and that
  ranking among healthy candidates is untouched.
- **Raw text.** The marker is matched on the page text as read, before `_clean_output` (one check,
  not two). Lines inside a fenced block are skipped. Limit: the aggregate read-back (qwen) is
  cleaned before it is split into sections, so there the check necessarily sees cleaned text; the
  raw guarantee holds for per-page files, where `_clean_output` would unwrap a whole-page fence
  into a bare marker (pinned).
- Tests: agentic legs now both use exit 0, so the marker alone triggers the next rung; added an
  exit-0 whole-document `*[OCR Failed]*` case.
- Mutants (external copies, canary, anchor count 1): dropping the healthy-preference fails 2 tests;
  removing the fence skip fails 2; checking cleaned instead of raw text fails 1.
