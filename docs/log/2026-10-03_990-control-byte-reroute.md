# GH-990: a control byte before a number re-routes the page off trusted-native

Branch `fix/990-control-byte-reroute`, from origin/main 7420bd87 (after #985).

## Problem
Trusted-native pages ship the repaired text layer verbatim. Where the PDF prints a minus (or
another symbol) in a font #217 cannot decode, the layer holds a C0 control byte
(`\x01`, `\x02`, `\x04`, ...) that is invisible on render. `-0.47` ships as `<byte>0.47` under
SUCCESS. #913's detector does not see it (it looks for the digit `2`). The same code stands
for different glyphs in different documents, so mapping it to a minus is not viable
(`~/.local/state/socr-housekeeping/minus-ctrl/report.md`).

## Change
Follows #913's pattern, no parallel mechanism.
- `core/glyph_recovery.py`: `count_control_byte_before_digit_hits(text)`. Regex: a C0 character
  other than tab, newline, CR, then at most one space, then an optional `.`, then `\d`. Runs on
  `raw_text` in `_assess_page`, i.e. the page text after #217's repair; no extra page walk.
- `core/born_digital.py`: `PageAssessment.control_byte_digit_hits` /
  `control_byte_scan_failed`. Any hit (or a raising scan: fail closed) sets
  `needs_ocr_enhancement`, which already takes the page off the free lane.
- `core/state.py`: the two fields copied to `PageState`.
- `core/manifest.py`: `minus_as_digit_suspect` also tests the new fields. It is the single
  predicate behind the `--native-only` demotion in `_select_page_output_tagged` and the
  failure mode, so no new branch was added there.
- `pipeline/orchestrator.py`: new audit kind `control_byte_before_digit`
  (`data.hits`, `data.error`, `data.native_only`), emitted in `_phase_analyze` beside
  `minus_extracted_as_digit`; distinct so a reader can tell the two apart. Recomputed every
  run, deliberately NOT in `_RESUME_REPLAYED`. The chart-asset demotion and the
  `minus_retained_pages` document bucket now call `minus_as_digit_suspect` instead of
  inlining the #913 fields (same behaviour for #913).
- Failure mode: `NATIVE_MINUS_AS_DIGIT` is reused ("a sign or glyph in the text layer is
  unreliable"). `docs/OUTPUT.md` and the enum comment widened; new audit kind documented.
  Under `--native-only` the page keeps its text and ships WARNING; the document is not SUCCESS
  (`native_minus_as_digit_retained` event).

Deviations from the brief, both from measurement:
- At most one space between the byte and the number is allowed. 10 of the 35 measured
  trusted hits are `<byte> <digit>` (copyright before a year, a spaced binary minus); the strict
  "immediately before" form fires on 20 of the 30 measured pages.
- `\d` (any Unicode decimal digit), not `[0-9]`: Ramey p80 is followed by a mis-decoded
  non-ASCII digit and the ASCII class missed it.

## Measurements (branch source via PYTHONPATH + `socr.__file__` canary; basenames/pages/counts only)
Main behaviour = the same branch with the new detector neutralised in-process.
- The 30 measured pages: 30 of 30 re-routed (trusted on main, not trusted on the branch),
  0 in the other direction (0 pages became trusted anywhere).
- Trusted-native population (380 PDFs, 22,162 pages, ~9.1k trusted on main): the detector fires
  on 74 trusted pages in 33 documents. 54 are ordinary pages, 20 are chart-asset-lane pages
  (the issue's 30 excluded the chart lane). All 74 are re-routed. 0 re-routed the other way.
- This is more than the issue's "about 30". The 24 non-chart extras beyond the 30 are control
  bytes whose left neighbour is a letter, digit or punctuation (the issue's analysis
  required whitespace or a delimiter before the byte). By the same argument they are
  undecoded glyphs directly before a number (a letter-byte-digit shape is the binary minus
  `t-1`), and the cost of a false positive is one VLM read, but I did NOT render them, so
  "all corrupted" is unverified for those 24. Adding the left-boundary clause reproduces the 30.
- 127-page census (gh936 inputs): 2 pages fire, 0 of them trusted on main, so 0 routing change.
- Cost: the regex over the page text, 13 us mean per page (max 2.7 ms) over 22,162 pages, against
  108 ms for `_assess_page` (#913 log). Negligible.

## Tests
`tests/test_gh990_control_byte.py` (65 passed together with #913's file): hermetic fitz PDFs
(provider ladder patched, judge model `""`). Detector: `\x01`, `\x02`, `\x04` before a digit or
`.digit`, one space; quiet on tab, newline, CR, plain number, byte then letter, byte then
`.letter`, byte at end, two spaces; every non-whitespace C0 code fires; Unicode digit.
Routing difference pin (detector neutralised vs live, same process). process() difference pins:
provider present (text comes from the provider), absent (never SUCCESS, failure mode kept,
text kept), `--native-only` (WARNING + failure mode + retained event), raising scan (fails
closed, also under `--native-only`), chart-asset lane. Event: routed vs retained wording,
`data.error`. Event not in the resume replay set. Setup canary: fitz really emits the byte.

Mutations (external copy of src + tests + pyproject; uncapped `count(anchor) == 1` asserted;
baseline 65 passed with the source canary inside the copy): 22 of 22 killed: route flag, tab
allowed, newline allowed, dot-digit dropped, letter accepted, ASCII digits only, optional space
dropped, any whitespace allowed, count capped, scan failure treated clean, hits not stored on the
assessment, hits not copied to page state, scan flag not copied, suspect ignores hits, suspect
ignores scan failure, chart lane not demoted, retained bucket ignores the byte, event kind
renamed, event hits zeroed, event error flag dropped, native-only wording swapped, event loop
skips a failed scan.

## Not done
- No glyph mapping; the page is read by a model.
- Other detector gaps of #913 (same-font corruption etc.) are unchanged.

## Suite
Full suite, default OLLAMA_HOST: 6562 passed, 2 skipped, 4 xfailed. Format gate clean (`uvx ruff@0.16.0 format --check .`). The layering test caught a cross-package private import (`_minus_as_digit_suspect`); the predicate is now public as `minus_as_digit_suspect`.
