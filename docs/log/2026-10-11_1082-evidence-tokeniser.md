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
