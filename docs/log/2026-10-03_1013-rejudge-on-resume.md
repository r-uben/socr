# #1013: re-judge a timed-out model candidate on resume

Part 3 of #1004 (design notes: `2026-10-03_1004-judge-timeout-untrusted-native.md`).

## Design

- **Persist.** A page that shipped `NATIVE_UNTRUSTED_JUDGE_TIMEOUT` writes
  `judge_timeout_candidate = {text_sha256, candidate: PageOutput.to_dict()}` into its sidecar
  (`_flush_page_sidecar`). Sparse: only those pages carry the key, so every other sidecar keeps
  its bytes. Engine/provider/model ride inside `candidate`; fingerprint and input checksum are the
  sidecar's own top-level fields.
- **Gate (`_load_rejudge_candidate`).** Reuse only if: sidecar parses and is `terminal`, run
  fingerprint equals this run's, input checksum equals this input's, the page still has
  `needs_ocr_enhancement`, the candidate text hashes to `text_sha256`, and `provider_id` is a rung of
  THIS run's ladder. Any doubt returns None and the page is routed as before (one re-OCR).
- **Re-judge (`_rejudge_kept_candidate`, called in the route branch before `route_page`).** Same
  `judge` object the ladder uses (deadline adapter + #987 breaker). `rejudge_candidate` asks it up
  to `PipelineConfig.rejudge_attempts` times (default 1, 0 disables); only a timeout is retried.
  Only an ACCEPT returns a `PageDecision`; the page then flows through the normal table/credential
  gates, with `cost_usd=0` for the skipped OCR. Reject, timeout or error returns None and
  `route_page` runs; native is never shipped on a second timeout without the ladder having run.
- **Surfacing.** Events `rejudge_accepted|rejected|timeout|error` (page sidecar `audit_events`,
  audit log, metadata), replayed on resume via `_RESUME_REPLAYED`, plus one CLI line per kind. The
  page/document verdict of a rejected or timed-out re-judge is whatever the ladder then produces
  (still `NATIVE_UNTRUSTED_JUDGE_TIMEOUT` / AUDIT_FAILED if the judge times out again). An accepted
  page ships the model text at its own status; no new failure mode.

## Not done / notes

- `rejudge_attempts` is not in the run fingerprint (it changes no byte on a page without a kept
  candidate, and a kept candidate is checked against the fingerprint itself).
- A document recorded AUDIT_FAILED is skipped at the root index; the real second run is
  `--reprocess`, which the tests use (the flag is excluded from the fingerprint).
- Candidate bytes are not passed through `_ingest_candidate` again: they already crossed it in the
  run that judged them, and the judge must see the exact bytes.

## Tests

`tests/test_gh1013_rejudge_timed_out_candidate.py` (8), one output dir, run 1 vs run 2 differing in
one thing: (a) timeout then accept: candidate ships, 0 OCR calls, judge saw the kept bytes once;
(b) accept vs reject: OCR 0 vs 1 calls, candidate not shipped on reject; second timeout runs the
ladder; attempts bounded for n in 0/1/3; (c) stale fingerprint, tampered bytes, wrong input
checksum: no reuse (OCR runs, judge sees only the ladder's candidate). Unaffected page sidecar has no
new key. `test_resume_restore_kinds.py` expected set updated (41 -> 45).
Hermetic: `_available_engines_for_agentic` patched, `_resolve_judge_model` returns "".

## Mutants

External copy of src, tests and pyproject (so the loaded-source canary in the test file holds),
uncapped anchor count == 1 asserted before editing. See the PR for results.

Results (each applied once, anchor count 1, copy loads its own `src` per the test file's canary):

- skip the text checksum check: 1 failed (`test_c_...tampered_bytes_means_no_reuse`).
- ship on a timeout: 3 failed (second-timeout runs the ladder, bounded attempts, stop-on-first-answer).
- unbounded retry (`range(attempts + 3)`): 2 failed (second-timeout, bounded attempts).

Also updated: `test_gh987_judge_breaker_e2e.py::test_short_circuited_pages_are_reprocessed_on_resume...`
now pins the difference (`rejudge_attempts=0` re-OCRs all four pages; default re-judges them with
0 OCR); `test_cli_flag_agentic_status_gh142.py` lists `rejudge_attempts` as `_UNEXERCISED` with the
observing test named. The kept candidate blanks `judge_reason`/`skip_reason` so a breaker-tripped
sidecar stays shape-equal to a real-deadline one (#987 contract). Full suite: 6767 passed.
