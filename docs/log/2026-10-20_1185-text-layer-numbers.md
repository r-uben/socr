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

## Round 2 (2026-10-20): Fable and Argon reviews of PR #1219

Every finding was reproduced before any fix; commands and outputs are in the lead's
report-r2.md. Deferred items are #1220.

- **A FAIL reaches every level** (Fable P1-A, Argon 1). Reproduced: the `-8` arm was
  `success`/`completed` at document, metadata, root index and CLI. Now a page whose verdict
  carries `text_layer_numbers=FAIL` is in the document rule (`numbers_disagree_pages` in
  `pages_ok`: AUDIT_FAILED, `partial`, "completed with warnings"), gets one
  `numbers_disagree_with_text_layer` audit event and a line in the document's `error` note.
  The tool writes the same status, note and events (document, root index, sidecar copy),
  derived from the sidecars as written so an interrupted run is completed by the next.
  **UNKNOWN stays at the page**: it says the check could not run, not that a number is
  contradicted, which is the #1145 rule for the equation / chart / unjudged clauses.
- **Analysis-only earlier-OCR pages are UNKNOWN, not skipped** (Fable P1-B). In a run,
  #1156's clause marks such a page in the same pass; this tool applies only #1185's, so
  abstaining left 22 of the calibration copy's 43 "stay CHECKED" pages unmarked. Chosen
  over counting them in `skipped`: a skipped page still ships unmarked until someone runs
  `socr-refinalize-prior-ocr`, and the rule is that no such page comes out unmarked. The
  clause's no-own-text UNKNOWN is true of it, with a reason naming #1156's tool.
- **Agreement with a refused or damaged layer is not a CHECK** (Argon 4, overruling the
  round-1 asymmetry and Fable P2-4 by the orchestrator's ruling). Every early return of the
  page analysis sets `is_born_digital=False`, so this also covers #961 pages that return
  before the invisible-OCR test runs. The cosmetic flag (#136: lost spaces, fused words) is
  NOT damage: the detector routes digit damage to OCR before it sets that flag, so an
  agreement counts and a dispute is UNKNOWN. Cost: 589 of run 23319732's 3,071 pages
  move from CHECKED to UNKNOWN. A layer refused only for being short ("insufficient text
  layer") is refused all the same.
- **Comparator** (Argon 2, 3, 6): the year-callout relation needs numbers written bare on
  both sides and a callout that does not start with 0; labels run into digits (`Q1`, `A-8`,
  `c1`) are split on the reading too, claim first, and are neither compared nor disputed
  (a reading may name columns the page leaves unnamed); a `tabular` is converted to table
  rows (its `\cline`/`\multicolumn` arguments dropped); HTML tags are stripped after the
  maths step; a numeric callout on a table cell keeps the cell's number, except a power of
  ten or a base after a times sign. Measured on run 23319732 by reading every new FAIL:
  a first, wider version (the layer's full word/digit split on the reading, every non-maths
  environment kept) added 16 FAILs, 13 of them false (`2018:II`, `5.¹¹`, `\cline{3-4}`,
  `2 \times 10^6`, `\includegraphics[width=0.9...]`); the narrowed version adds 3: two real
  (Hansen-McMahon p17, layer `0.29`, reading `0.28`, inside a `tabular` round 1 dropped)
  and one false (Kuttner p11, a layer writing the minus as `@`, in #1220).
- **The tool places the clause where a fresh run does** (Argon 7.1). Reproduced on a WARNING
  page with a figure clause: page bytes were already identical, the sidecar's clause order
  was not. The position is the count of silent-success clauses the RUN derived, from its
  own prior-OCR flag and its `audit_log.json` event kinds; a document without an audit log
  is counted in `skipped`.
- **Nothing is passed over silently** (Fable P2-6, Argon 7.3): unfinished pages and
  documents without `metadata.json` or `audit_log.json` are counted in `skipped`.
- Argon 7.2, partly real: a manifest missing the page raised after the fragments and before
  the sidecars, and every rerun raised at the same place. The manifest is now checked before
  the first write. The rest of the claim does not hold: on any other interruption the
  order is deliberate and the rerun completes it: `_swap_once` accepts an already-written
  section, the fragment check accepts either text, and `refinalize_page` re-derives from the
  untouched sidecar.

The round-1 control spy proved only that the comparison agreed, which stays true on a
refused layer; it now also asserts the clause decision was "none" (the page is CHECKED).
Five controls were given a born-digital layer (Argon 8, approved): gh161 ×2, ladder,
canon, pp5. The p3 fixture PDF now prints the "Table 2" caption its reading quotes, and
its round-1 expectation edits are reverted.
gh993's end-to-end tests get their own PDF (`_ruled_pdf_as_read`): the shared grid's 48
characters fall under the analysis's 50-character floor, and a caption alone tripped the
native table trust check because the grid's header (`00..03`) is not the reading's
(`c0..c3`); it prints the reading's table and a caption without digits. No gh993
assertion depended on the short layer, no expectation changed, the shared builder is
untouched (approved by the orchestrator, 2026-10-20).

## Round 3 (2026-10-20): round-2 reviews of 9e90eb39

- **Label digits, narrow and in any script** (Fable P1-1, Argon P1-1). The rule was ASCII
  only (`β1`, `σ2` still freed a digit: 252 born-digital pages of run 23319732 run a Greek
  letter into a digit), and it tagged values: the last letter of `\approx`, `\sim`, `\pm`,
  `\le`, `\bf` made the following number a label, never disputed. A label is now a name of
  one to three letters of any script, starting a word and not a LaTeX command, an optional
  `.` or hyphen, and a one- or two-digit integer with no digit, comma, percent or decimal
  after it; plus `1Q`..`4Q`, `1H`, `1st`..`4th`, and a one-name subscript `$Q_1$` joined
  to `Q1`. `USD2,000` and `Korea-8` are not tagged; they are writings the comparator does
  not read, listed in #1220.
- **A multi-line `tabular` row stays one row** (Argon P1-2): `\hline` / `\toprule` on its own
  line left the row without its leading `|`. Rows are now whitespace-collapsed; an optional
  `[t]` and nested-brace column specs are matched.
- **An equation in a table cell is not the cell's number**: read as table rows, 2018
  Tealbook p111's Taylor-rule table made a false FAIL on a formula's `0.5` (the layer prints
  that maths as glyphs the analysis cannot read). Dropping any lettered maths lost three
  real FAILs (`$142^a$` cells); dropping every `=` lost `$\rho^{m} = 0.75$`. Kept: an
  equation whose right side is one number; dropped: any other equation.
- **The tool** (Argon P1-3): a finished document without `audit_log.json` had no events (a
  run writes the file only then), so it is checked with empty event kinds, and the log is
  created when the tool adds the first event. `PageChange.result` reads this clause, not
  the verdict's last one.

Measured on run 23319732 against round 2: the FAIL set is identical (32); one page moves
between UNKNOWN reasons. Calibration copy: see report-r3.
