# 2026-10-08 - a table-only floor keeps corroborated prose (#1043)

Pages and counts only; the corpus is copyrighted. Full phase-1 write-up (kept outside the repo):
`~/.local/state/socr-housekeeping/prose-splice/MEASURE.md`.

## Problem

When every model reading of a page is rejected only because of a table, the cascade ships the whole page
as a fail-closed floor (marker plus image). Correct prose is lost as text. Case: 1990 Forsythe-Lundholm
p25 (scan, invisible OCR layer): qwen dropped a column, the judge rejected it, gemini failed
`table_structure_failed: grid_shape`, the page became `invisible_scan_unread`.

## Prior art read first

- `native_prose_floor_text` (#649/#652): ships the page's OWN layer around withheld numeric bands. #652 round 10
  deleted `_prose_corroboration_ok` (model prose vs a withheld table's vocabulary) after six reproduced
  fabrication paths. Model prose has not shipped from a floor since.
- `table_floor_text_for_source` / `splice_all_table_regions` (GH-90/371/520): the splice itself, reused as is. Its
  coverage proof needs detected table geometry, which a scan does not have.
- #1027 (invisible-scan floor), #993 (`count_page_tables`, whose `[page N failed: unverifiable table` marker
  regex is what makes the withheld tables count with no change).

The new check differs from the deleted guard in kind: it compares by ORDER (bigrams), requires the layer to be
COVERED by the reading, and refuses any numeral the layer does not print, rather than testing vocabulary
overlap with the table.

## Phase 1 numbers (measure, no src change)

Two output sets (archive re-OCR, 7 docs; HPC socr-h2, 14 docs): 146 floor page-records, 114 unique pages;
112 records rejected only for a table. Negative controls: 3,444 readings of OTHER pages of the same document were BUILT;
3,363 had both sides to score (the rest had no prose or no layer band). Of the 3,363, 4 scored 1.0, all with <= 16 prose tokens, so the token floor is 17. Among
3,222 scored at >= 17 tokens the maximum is 0.892, so the cutoff is 0.90.

| | records | unique pages |
|---|---|---|
| would recover text (scratch scripts) | 52 | 39 |
| same, replayed through the shipped code (`corroborate_prose`, completed-verdict and no-blank-reason rules) | 47 | 34 |
| prose false-accepts | 0 | 0 |

The 5-record gap is the shipped rule being stricter than the scratch rule: a verdict must be COMPLETED and
a blank reason is unknown, not table-only. 17 renders read against the reading's prose, plus a token diff of
all 52: every unmatched token was a layer hyphenation split or dropout. Failing pages differ for real (LaTeX
wrappers, layer misreading numerals, mostly-figure pages). 3 failing pages had visibly correct prose and floor
only because the invisible layer is wrong: conservative, not false accepts. I read 17 renders, not the ~20
asked.

The 6 over-withheld tables of REGRADE (#1-6 = Forsythe p17/p25/p28, Hansen p15/p19/p20): p17, p25, p20
recover; p28 floors (also rejected for omitted figure axes and page number, so not table-only); p15 and p19
floor (layer prints "21"/"5%" differently from the page). 3 of 6.

Residual: prose recall is 0.90-1.00, so a recovered page can drop a few words with no marker. Less loss than
the floor (whole page), but silent. No recovered page had a numeric-majority prose line besides page numbers,
download watermarks and "Volume 95, Number 2, 2020".

## Change

- `tables/prose_corroboration.py`: `rejection_is_table_only`, `corroborate_prose`, constants
  `PROSE_CORROBORATION_MIN` (0.90) and `PROSE_CORROBORATION_MIN_TOKENS` (17) with their derivation.
- `core/manifest.py`: the cascade is renamed `_select_page_output_cascade` (R7's AST pins read it);
  `_select_page_output_tagged` is now cascade + `_corroborated_prose_over_floor`. Fires only on the four table
  floors (scanned-table, native-D3, invisible-scan, structure-class), only when the floor is a bare marker, the
  words are cached, every model reading was refused by a COMPLETED verdict whose reason is table-only, and a
  reading's prose corroborates. New `SelectionProvenance`, `PagePrimaryReason`, `FailureMode`
  `table_withheld_prose_corroborated`. Status WARNING, `audit_passed` False (a resume re-reads the page).
- `pipeline/orchestrator.py`: own bucket (keeps the document out of SUCCESS), audit event, CLI line, document
  note, native words also cached for invisible-layer pages.
- `docs/OUTPUT.md`, `core/audit_log.py` rank.

Deliberately NOT covered: judge-timeout and text-table structure-class floors (nothing refused / not numeric
evidence), the rotated-shred floor, ladder-issued `table_withheld`.

## Weak links, stated

- The table-only test reads model-authored judge text. It is a filter; the corroboration is the safeguard.
  49 of the 52 scratch recoveries are table-only by gate prefix alone.
- Cutoff margin 0.892 vs 0.90 on same-document negatives is thin.
- A reading that drops text the invisible layer also lacks cannot be caught.

## Tests

`tests/test_gh1043_table_only_floor_prose.py` (23): corroborated vs uncorroborated prose as a difference,
non-table rejection floors, one mixed reading blocks, timeout / never-judged / blank reason floor, no layer,
no table block, numeral absent from layer, accepted reading untouched, #993 withheld count, e2e through
`process()` with a twin run. Hermetic. Frozen pins moved: R7 (cascade still 23 tags, enum 24), p6 contract.

Mutants (external copy of `src` + `tests`, anchor count asserted == 1, `socr.__file__` canary = the suite's own
`test_loaded_source_is_this_checkout` passing inside the copy): corroboration check off -> 3 fail; table-only
check off -> 3 fail.

## Review round (cubic P2s on #1047)

- Mixed rejection text: a clause that also names a figure, axis, caption, equation, footnote, paragraph,
  text or legend is no longer table-only ("The table is malformed and the figure axis labels are missing"
  floors). Pinned with that sentence and Forsythe p28's real reason.
- Image refs: the reading's image references were stripped for scoring but still shipped. They are now
  stripped from the shipped body too; the only image is the floor's own (`d3_floor_png_ref` /
  `invisible_scan_png_ref`, written by this document's run).
- Numerals: the tokeniser was ASCII-only and silently dropped non-ASCII digits, so the unmatched-numeral veto
  never saw them. It is now Unicode-aware (`[^\W_]+`, `isnumeric`).
- Wording: the banner no longer says word-for-word. It says corroborated by the page's text layer, with the
  similarity and the cutoff; the audit event carries `similarity`.
- Safety: the pass test already ANDs the score cutoff with `unmatched_numerals == 0` (and the token floor);
  now pinned by a mutant (veto off -> 1 failure).
- Counts: built vs scored negatives distinguished above.
- Survivors: replay through the shipped code is now 46 records, 34 unique pages (was 47 / 34); no page lost.
- Mutants (external copy, anchor count 1): mixed-clause veto off -> 3 fail; ASCII-only tokens -> 1 fail;
  numeral veto off -> 1 fail; image strip off -> 1 fail.

## Round 2: the bag-of-words guard is REPLACED (Fable REJECT on #1047)

**Everything above about the 0.90 similarity cutoff, the 34 surviving pages and the "0 false-accepts" is
superseded.** A review ran counterexamples against the real `corroborate_prose` and shipped: dropped sentences
carrying numerals (recall skipped every layer line with a digit), 4.2%->4.8%, 2.5->2.7, an inserted minus
sign, 1987->1978 (1978 printed in a reference), a leaked table row 21.0->12.0, number words, flipped meanings,
a deleted "not". The numeral veto compared digit runs against the whole layer, tables included. Measuring only
mutual precision and a frequency recall cannot satisfy "a wrong number is worse than a missing one".

New guard (`tables/prose_corroboration.py`): an ORDERED, STRICT alignment (`difflib.SequenceMatcher`, reading
prose vs the layer's bands in page order). Every non-equal step must be OCR noise ONLY (`NOISE_CLASS`: case,
surrounding punctuation, markdown decoration, Unicode compatibility forms, soft hyphen / zero-width, the Unicode
minus; and an ALPHABETIC run split or joined differently). A digit-bearing token has no noise: whole token, sign
and decimal included. Any inserted, deleted or replaced word, number, sign or negation floors the page. The
reading must cover every layer band (numeric-bearing ones included) except the withheld table itself, a band
whose alphabetic words all occur in the reading's own table block (numerals free), within that table's token
budget. A reading that reproduces a table band as prose is refused (a leaked row ships un-withheld).
The similarity score, `PROSE_CORROBORATION_MIN` and the unmatched-numeral veto are gone; the 17-token floor
stays (identical short captions score 1.0 under any measure). Banner: "corroborated by this page's text layer
(ordered match, N tokens)"; the audit event carries `matched_tokens`.

### Re-measure (same two output sets, same table-only + completed-verdict rules)

| | records | unique pages |
|---|---|---|
| floor pages | 139 | 109 |
| recover under the strict rule | 2 | **1** |

The one page is Forsythe p9 (read against its render: the prose is right). The bag-of-words guard recovered 34.
Of the 6 over-withheld tables (#1-6): **0 of 6 recover**. Forsythe p25, the motivating case, now fails: the
invisible layer prints words the model read correctly differently (OCR errors such as "plorr" for "Plott" are not
noise the guard may excuse), plus running-header and watermark lines. The strict rule is working as specified:
an invisible OCR layer is too noisy to vouch for a reading word for word, so almost nothing can be vouched for.

### Known limits (not closed)

- Common mode: a line missing from BOTH the layer and the reading is invisible to a two-witness comparison. socr
  has no measure of an invisible layer's completeness (checked: `_check_token_coverage` is a numeric-orphan
  diagnostic, `_raster_coverage` is image area), so this is not closed. A layer missing a line that the reading
  HAS floors (pinned); both missing it passes (not pinned, cannot be).
- A dropped prose line built only from the table's own words, within the table's token budget, hides in the table
  band allowance.
- The table-only test still reads model-authored judge text.

### Tests and mutants

`tests/test_gh1043_table_only_floor_prose.py` (52): every Fable counterexample is a parametrised test
(dropped numeric sentences, dropped sentences, meaning flips, deleted negation, ASCII and Unicode minus inserted,
three changed decimals, reversed paragraphs, two-column row-wise read, other page sharing a header, year swap,
number words, layer missing a line, leaked table row true and misread) plus the earlier mixed-clause, image-ref
and Unicode-numeral pins. Mutants (external copy of `src` + `tests`, anchor count asserted == 1, canary =
`test_loaded_source_is_this_checkout` passing in the copy): noise class widened to any word -> 4 fail;
numeric bands dropped from coverage -> 4 fail; reading-only text tolerated -> 1; uncovered layer line tolerated
-> 4; table-row leak tolerated -> 1; mixed-clause veto off -> 3; image strip off -> 1.
