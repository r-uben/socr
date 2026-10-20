# #1185: a model-read page is CHECKED only if its numbers are the page's own (2026-10-20)

Owner ruling (2026-10-17, confirmed by the orchestrator 2026-10-20 with `compared > 0`): on a
model reading, CHECKED requires every number the reading ships to occur in the page's OWN
text layer. Disagreement is WARNING with the numbers NAMED; no own layer, or no number in
the reading, is MARKED. Nothing is corrected from the layer. Drops are #1218.

## What was built

- `tables/reading_agreement.py`: one comparator (`compare_numbers`) with a provenance field,
  ported from 0d045283. A page MULTISET, not position-by-position: the layer's order is not
  the reading's (two columns, column-major tables, rotated tables), and position-by-position
  turns one inserted number into a dispute at every later position. Order only NAMES the
  counterpart ("text layer says -.8, reading says -8"). Tokenizer rules and why are in the
  module docstring; each was added for a measured noise class or an Argon finding.
- `core/verdict.py`: the silent-success clause `text_layer_numbers` (FAIL / UNKNOWN), the
  #1156 template. Prior-OCR pages are left to #1156's clause; page-failed markers are
  skipped; a damaged layer (detector refused it, or digit-damage flags) turns a dispute
  into UNKNOWN. Both modes are permanent on resume, but only on a WARNING page.
- `core/born_digital.py` / `core/state.py`: `own_layer_text`, the cleaned raw layer, on
  EVERY page (it was empty on routing-refused pages). In memory only.
- `pipeline/refinalize_own_text.py` (`socr-refinalize-own-text`): applies the clause to a
  finished run through the same seam, with no OCR.
- `core/manifest.py`: a `## Page N` header under leading marks is stripped too (it doubled
  under any mark; found by the conformance test).

## Why not #1206's per-cell binding

Run offline on Tealbook p125 it located 0 table regions (rotated page), witness MISSING: it
abstains on the motivating case, covers tables only, and the calibration run predates it.

## Measured choices

- Union of raw layer + structured `native_text`: rejected. It cleared 5 of 289 disputes and
  can manufacture agreement (a detached minus in one extraction).
- Furniture: on calibration, 0 disputed pages are disputed only by a page number or running
  head; on run 23319732, 2 (folio `628`, a "Published online" header), both on damaged
  layers (UNKNOWN, not FAIL). No exclusion added.
- Drops (#1218): no clean line; see the issue.

## Test changes (expectations, and 16 approved control fixtures)

Expectations: ~40 tests whose model-read fixture page now carries the clause; each edit is
commented `# #1185:` with its reason. Approved fixture changes (orchestrator, 2026-10-20),
each control proven with `gh1185_agreement_spy.assert_agreed_on` (compared > 0, agreed):

1-2. gh1145 fail-closed fallback [warning]/[error]: the CHECKED control page needs its own layer and a number.
3. gh1181 mark turns on the EXACT_PASS accept only: the judged clean arm (both arms share the layer so only the event differs).
4. gh1181 a page without the event is untouched: the clean page in the loop.
5. gh1181 a page the gate delegated to the judge is untouched: the clean page in the loop.
6. structural gate b1, passing non-native winner unaffected: the passing qwen winner.
7. s1 ordinary passing winner not an S1 event: the model attempt.
8. s1 legacy sidecar resumes as an ordinary passing page: the legacy page (p6 fixture helper; its two #1185 oracle leaves removed).
9. a1c clean SUCCESS sidecar is resume-skippable: the clean qwen page.
10. gh1013 fields set by verification survive to the ship: the `plain` arm.
11. gh161 audit-passed SUCCESS page is resume-skippable: the accepted page.
12. gh161 legacy SUCCESS + audit_passed=False is not skippable: the page under test (approved separately).
13. gh987 short-circuited pages are reprocessed: control run AND wedged run (approved separately).
14. ladder resume, no-ladder baseline: the no-ladder page.
15. pp5 non-success page reprocessed: p1 (p2 untouched; it passes because the resume now keeps a numbers_* page only when it is WARNING).
16. canon remediation replay bit-identical: the hand-built page set.

## Argon

Plan review: union risk, starred cells in maths, spaced brackets, resume non-convergence --
all acted on. Diff review: 7 real (currency `$` opening maths, `A-8`/`t-1` manufacturing a
negative, `142^a` losing its base, model-transcribed quotes dropped, loose year and bracket
relations, mark doubled on a re-finalize retry, `\frac` read as `12`), all fixed with tests.
Not acted on, with reason: the fallback firing when chart/equation clauses co-occur (those
modes are in `_MODE_CLAUSES`, so the status is explained); the clause setting a mode on a
non-SUCCESS page (the demotion only touches SUCCESS); stale `socr_source_digest` on pages
the tool does not change (as #1217's tool; the run fingerprint is what resume reads).

## Follow-ups

- #1218: drop detection.
- The equation/chart permanent modes are kept on resume on the mode alone, whatever the
  status; #1185 closed that only for its own two modes.
- A number in a caption ("Table 2") counts as a reading number.
