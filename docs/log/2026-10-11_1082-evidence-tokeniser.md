# 2026-10-11 — GH-1082 step 1: scanned-table evidence tokeniser (reduced to one fix)

**Branch:** `fix/issue-1082` (off `origin/main@90f92838`). **Status:** step 1 of 2, reduced after review.

## What was tried

A first draft added tokeniser rules so Tesseract's spelling quirks would not read as a different
number: minus folding on OCR evidence (`~`, dash runs, `_`), decimal-comma folding (`0,809`), splitting
comma-joined pairs (`(-5.52,1.86)`), edge stripping of stars/quotes/braces/brackets, and a `$...$` strip.
Replayed on the 181 stored pages (`scratch/routing-baseline/step1/measure.py`; stored witness texts stand in
for re-running Tesseract) it moved the check from 19 to 20 passing pages, and disputed numbers on the 12
hand-judged pages from 50 to 38 (`tess_sign_or_punct`) and 10 to 3 (`other`). Forney p22 still failed.

## Why the loosening rules were withdrawn

An adversarial review (Astra) reproduced four fail-open paths through `verify_scanned_table`:

1. Edge stripping destroyed a Markdown link in a MODEL cell: `[999](https://x/note)` + `1` against evidence
   `1` passed (also reaches `escalation_canary` via `collect_table_tokens`).
2. The OCR minus fold let evidence `~5` ("about five") or a decorative `--5--` support a model `-5`.
3. Pair splitting let evidence `1.234,56` (decimal comma, 1234.56) support model `1.234` and `56`.
4. Decimal-comma folding let evidence `1,5` (a list) support a model `1.5`.

Each is a number the page does not print being supported. The repo ranks a wrong number above a dropped
page, and the whole set gained one page of 162. None of these rules can be shown unable to create support, so
they are gone. They are pinned as regression guards in `tests/test_gh1082_evidence_tokeniser.py`.

## What remains

`_strip_word_edges` in `tables/source_evidence.py`: the evidence-side plain strip removed a leading `.`, so a
printed `.45` became `45` and could SUPPORT a model `45`. A leading `.` before a digit is now kept (`.45` ->
`0.45` through `strip_presentation`). This removes a false support and adds support only for the number the page
prints. Content-label tokens see the same stripped word as before. Applies to every evidence text (text layer and
OCR) since the old strip did; there is no OCR-only treatment anywhere (pinned by
`test_text_layer_path_has_no_ocr_only_treatment`).

## Measurement, reduced change

19 of 181 pages pass before and after; Forney 1988 p22 fails (`124`, `4.76`, `72`, `4.01`, `4.31`, `4.52` missing);
no page with a `model_wrong` token passes. Disputes on the 12 hand-judged pages are unchanged (92): the
remaining ones are genuine digit differences (dropped minus, lost decimal point, missing cell, digit misread)
or shapes no safe string rule resolves. They belong to step 2: re-read each disputed number from a crop and ship
the table only if the re-read agrees.

## Noted, not changed

Evidence `—5` (em dash) already supports a model `-5` because `strip_presentation` folds the em dash to a minus.
Pre-existing, outside this ticket; worth its own issue.

## Mutants

External copy of src and tests, `socr.__file__` canary, anchors counted == 1: restoring the old plain strip
fails the leading-dot tests; multiset -> set check fails the count-shortfall guard.

## #1085 — the two leftovers

**Branch:** `fix/issue-1085`. Both changes are in `tables/source_evidence.py`; `strip_presentation` is untouched
(11 call sites in 7 modules, several born-digital). A test pins its dash fold so a future "fix" there is caught.

1. **Fail-open, em/en dash.** `_tokens_from_plain_text` now drops an evidence numeric token with an en or em dash
   before its first digit (`_is_dash_led`). The token supports NOTHING, not even the positive `5`: a page printing
   `—5` may mean "n/a, 5", a range, or a typeset minus, and the text cannot say which, so only "no support" can never
   create support for a number the page does not print. ASCII `-` and U+2212 keep their sign meaning; a trailing
   dash (`5—`) is unchanged; the model side is unchanged.
2. **Fail-closed, dot then punctuation.** After the leading-dot rule, `_strip_word_edges` strips leading punctuation
   again, so `...(0.025)`, `.(45)`, `.,45`, `."45"`, `.[45]` give `0.025` / `45` as before #1084. `.45` still
   gives `0.45`.

**Measurement** (181 stored pages; stored `witness_texts` stand in for re-running Tesseract; script
`scratch/routing-baseline/step-1085/measure.py`, scratch and uncommitted like the #1084 one): passing pages 19 -> 17.
Forney 1988 p22 still fails; no passing page contains a `model_wrong` token. The two pages lost are
`2020__bochkay_hales_chava...TAR` p18 (`—11.543***`, `—3.644`, `(—3.53)`) and
`2023__bauer_swanson...ONLINEAPPENDIX` p3 (eight `—0.xxx` values): Tesseract read typeset minuses as em dashes, and the
model's negatives were probably right. This is the price of the fail-open fix; recovering them is step 2 (re-read from a
crop), not a string rule.

**Mutants** (external copy of src and tests, `socr.__file__` canary, anchors counted == 1): restoring dash support fails
the dash-led tests (2 failed); removing the second strip fails the `_strip_word_edges` cases and the `0.025` end-to-end
test (6 failed).

**Review P0s (Astra), fixed.** (1) Dropping dash-led tokens could empty the bundle and activate
`_corroborate_via_distrusted_layer`, whose parser folds `—5` and ignores header rows. Now a dropped dash-led number is
counted (`SourceEvidenceBundle.ambiguous_numeric`) and keeps `has_content_evidence` true, so the page fails as
unsupported, not unwitnessed; the rescue DECLINES when the layer has any dash-led number (round 3: deleting the words hid an
omitted row and let `corroborate_rows` tolerate one unsupported number in fifty; the shared parser and its tolerance are
untouched, the tolerance is #1087). (2) `_is_dash_led` decodes entities (and a whole-token markdown link) to a
fixed point before checking, so `&amp;amp;mdash;5` / `&#8212;5` / `[—5](u)` are caught. The header-row blind spot of the
rescue is reachable on origin/main with no #1085 code: untrusted layer `-5`, silent OCR witness, header 999 passes.
