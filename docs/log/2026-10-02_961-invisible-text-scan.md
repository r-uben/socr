# GH-961: a scan with an invisible baked-in OCR layer leaves the trusted-native lane

Branch `fix/961-invisible-text-scan`, from origin/main e06aabd5.

## Problem
Scanned PDFs that carry an old OCR text layer (render mode 3 over a page-sized raster) pass
as born-digital, so the old OCR ships verbatim under SUCCESS, bypassing every model and judge.

## Change
- `core/born_digital.py`: `BornDigitalDetector._has_invisible_text_over_raster(page)`.
  Fires when the page's raster coverage is >= `RASTER_DOMINANCE_RATIO` (the existing 0.90, no
  new threshold) AND `get_texttrace()` has a type-3 (invisible) span with characters. Raster
  coverage is computed first, with the same sum-of-bbox arithmetic as `_raster_coverage`,
  and the texttrace is skipped when it is below the ratio. `_assess_page_signals` sets
  `needs_ocr_enhancement` (so `_is_native_eligible_without_ocr` is false: same mechanism as
  #913) and the new `PageAssessment.invisible_text_over_raster`. A raising detector sets
  `invisible_text_scan_failed`, treated as a hit (fail closed).
- `core/state.py`: both flags copied onto `PageState`.
- `core/result.py`: `FailureMode.NATIVE_INVISIBLE_TEXT_SCAN`.
- `core/manifest.py`: `_invisible_text_suspect(p)` joins `native_distrusted`, so under
  `--native-only` the page falls through to the native fallback: same text, WARNING, new
  failure mode, NATIVE_FALLBACK provenance. `audit_passed` is untouched (it selects the winner).
- `pipeline/orchestrator.py`: `_phase_analyze` emits `invisible_text_scan` (page, `data.error`,
  `data.native_only`); the chart-asset lane demotes the same way; `invisible_retained_pages`
  bucket in `_phase_assemble` clears `pages_ok`, emits `native_invisible_text_retained`, and
  prints a CLI line (a native-only page is not in `native_fallback_pages`).
- Resume: the event is derived in `_phase_analyze` from the PDF on every run, so it is NOT in
  `_RESUME_REPLAYED`. The run fingerprint does not change with this ticket's flags; a
  previously finished scan page is re-read because `socr_source_digest` (part of the
  fingerprint) changes with the source, so every earlier terminal page is invalidated. The
  test is a real second `process()` run into the same output dir: the finished OCR page is
  resumed (no provider call) and the event appears exactly once.

## Cost (bertsekas, 411 pages, 405 fire; CPU, machine at load ~16)
Detector 5.1 ms/page mean, 12.7 ms max, against 104 ms/page mean for `_assess_page` (~5%).
Every bertsekas page has a raster, so this is the worst case: all pages pay the texttrace.
A born-digital page without raster pays only `get_image_info`; a test asserts the texttrace is
not called below the ratio.

## Routing, main vs branch (analysis phase only, no model; counts, basenames, pages)
Same source, detector neutralised (= main) vs live, `_phase_analyze` plus
`_is_agentic_trusted_native`, over every document that has a trusted-native page in
`native-audit/population.jsonl`. Script: `~/.local/state/socr-housekeeping/gh961/route.py`.
- 379 documents, 0 errors: trusted-native pages 9891 on main, 9115 on the branch; 776 pages
  re-routed in 17 documents; 825 events (49 on pages already off the lane on main).
- The set of re-routed pages equals the detector-E3 set computed from `features.jsonl`
  exactly (0 differences either way).
- Pages that fire but were already off the lane on main (needs_ocr already true) get the
  event but no routing change.
- 0 pages moved the other way. This is also structural: the change only ever sets
  `needs_ocr_enhancement`.
- The issue quoted 728 pages / 15 documents; the measured population here is 776 pages / 17
  documents (the population file was regenerated after that figure). Biggest: Bertsekas 382,
  Romer-Romer 50+50, Bernanke-Blinder 46, Christiano-Eichenbaum-Evans 40.

## Tests
`tests/test_gh961_invisible_text_scan.py` (hermetic fitz PDFs, ladder and judge patched,
stub provider; pins are differences, detector neutralised vs live in the same process):
- detector shapes: scan (full raster + invisible text) fires; visible born-digital, small figure
  + invisible text, full raster + visible text, invisible text without raster stay quiet;
- texttrace not called below the ratio;
- routing difference over all shapes; event not replayed; event wording (routed / retained / error);
- process(): provider present, provider absent, `--native-only`, detector raising, raising +
  `--native-only`, chart-asset lane (detector forced; the real one is quiet on a small figure).

Suite: 6233 passed, 2 skipped, 4 xfailed (default OLLAMA_HOST, 6m48s).

Mutations (external copy of src + tests + pyproject, uncapped `count(anchor) == 1`, baseline
green, canary in suite, run against the 961 and 913 suites): 16 of 16 killed: route flag not
set, ratio ignored, type-3 ignored, fail open, state copy of either flag dropped, native-only
distrust term removed, failure mode dropped, scan-failure term removed from the manifest
predicate, event not emitted, event error flag wrong, retained wording, document bucket not
blocking, bucket ignores scan failure, chart lane status, chart lane failure mode. The first run
left the two chart mutants alive (no chart test existed); the chart test was added and both die.

## Residuals
- A cosmetic scan (invisible text that is a clean, correct OCR) is re-OCR'd: cost is one model
  read per page (the measured false positives, 2 of 7 sampled).
- Invisible text via opacity 0 rather than render mode 3 is not detected (E3 as specified).

## Round 2 (Astra on #963: ACCEPT-WITH-FIXES)
- P1a, detector narrowed: raster coverage is now the UNION area of the image boxes clipped to
  the page (`_union_area`, sweep over x with merged y intervals), not a sum; and the page's
  text must be MAJORITY invisible by characters (`invisible * 2 > total`), not "any invisible
  span". Controls: background image + mostly visible text + a little invisible text is quiet;
  exactly half invisible is quiet, one more line fires; two stacked images whose summed area
  passes the ratio but whose union is 60% are quiet; a union coverage boundary pin (ratio -
  0.02 quiet, ratio + 0.02 fires); `_union_area` unit pin (overlap, clip, empty).
  Routing re-run on the 18 documents that had any event: 776 pages re-routed (unchanged, no
  page lost or gained, 0 the other way); events 825 -> 824, the lost one is the single
  population page with some but not majority invisible text, which was already off the lane
  on main.
- P1b: `_is_corrupt_math_recovery_page` now refuses a scan-suspect page (hit or failed scan),
  so it takes whole-page OCR instead of a corrupt-math hybrid that would have shipped the old
  OCR prose while the event said OCR replaced it. Difference pin: with the detector off the
  hybrid lane owns the page (main); live, it does not; a failed scan is refused too. The
  retained-bucket exclusion for the hybrid (`n not in corrupt_math_hybrid_pages`) therefore
  never sees a scan page.
- D3: pin that a scan page with the D3 conjunction (unverifiable native table) ships ERROR,
  identical status and failure mode to the same page without the scan flag, never SUCCESS.
- Mutations (external copy, uncapped anchor count 1, baseline 57 passed over the 961 + 913
  suites): 22 of 22 killed, including union replaced by sum, union not clipped, any-invisible
  instead of majority, majority off by one, and each corrupt-math lane term dropped.
- Suite: 6240 passed, 1 failed, 2 skipped, 4 xfailed; the failure was
  `tests/test_timings.py::test_native_page_records_extract_not_route`, a wall-clock tolerance
  (50 ms observed vs 1 ms) tripped while a routing sweep ran alongside; it passes alone (13
  passed).
