# Table-detection firing rate across the corpus

2026-08-30. GH-328, salvaged from `TODO.md` before GH-156 deleted it.

Counts how often each of socr's two table detectors fires, so that corpus-level
claims about "table pages" rest on a measured denominator rather than an assumed
one. Content-free per the 2026-08-22 convention: identifiers and aggregate counts
only.

## Method

All 528 pages of the 9 papers named in
`docs/log/2026-08-20_lane-comparison-manifest.json` — the whole documents, not the
21-page sample the manifest itself lists. For each page, two calls made
**unconditionally** — that is what was measured, and it is NOT how production
routes:

- `page.find_tables()` — PyMuPDF's ruled-table detector.
- `socr.tables.reconstruct.has_numeric_columns(page)` — socr's second-pass gate.

**In production these are not independent.** `BornDigitalDetector._detect_tables`
(`src/socr/core/born_digital.py`) returns True immediately when `find_tables()`
finds anything and only then falls through to `return has_numeric_columns(page)`.
So the production quantity is `find_tables() == 0` ∩ heuristic — the
"second-pass only" row below — and the raw `has_numeric_columns` row counts
pages the gate would never have asked about.

Measuring unconditionally is what makes the 14-page overlap (102 vs 88) visible
at all. Under the documented gate those two numbers would be equal, and a
reader following the gate would conclude the table does not add up. It adds up
because the instrument deliberately differs from the router. (The gate itself is
what GH-348 names.)

Measured twice: on `main` before GH-248's lane-reuse fix, and after it.

## Result

| | before GH-248 | after |
|---|---|---|
| pages | 528 | 528 |
| `find_tables` fires | 23 | 23 |
| `has_numeric_columns` fires | 102 | 92 |
| **second-pass only** | **88** | **78** |

"Second-pass only" is the number that matters for the HEURISTIC: pages routed to
the table path on the numeric-lane gate alone, with no ruled table found.
**78 of 528 pages, 15%.**

That 15% is the heuristic-only slice, NOT socr's table-page rate. socr routes a
page to the table path when EITHER detector fires, so after GH-248 that is
`find_tables` 23 + second-pass-only 78 = **101 pages, 19%**. GH-248's 391 hits
were `has_tables` — both paths — so quoting 15% as "socr finds N table pages"
mixes denominators, which is the same silent inflation this log exists to warn
about.

`find_tables` is unchanged at 23, confirming GH-248 touched only the heuristic path.
That fix removed 10 second-pass firings, 11% of them.

## What this does NOT establish

**These are firings, not errors.** A borderless table is a real table, and the
second-pass gate exists precisely to catch it. On a clean paper most of the 78 are
probably legitimate.

Establishing how many are wrong needs page-level ground truth, which this
measurement does not have and cannot derive: the native reconstruction cannot
adjudicate its own routing decision, and that is the same circularity
`2026-08-30_model-vs-native-table-rows.md` had to break with blind transcription.

What is established is the direction of the error. The Glaeser–Sacerdote–Scheinkman
paper (mirrored OCR text layer) put four pure-prose pages into this bucket before
GH-248 — `find_tables` returned 0 on all four while the heuristic returned True — so
the count inflates on scanned or damaged sources and does so silently.

## Why the number is worth having

- It is the denominator for any "socr finds N table pages" claim. GH-248's issue
  records a triage that filtered on `has_tables`, got 391 hits, and found prose among
  the top ones by hand.
- 15% on a clean corpus is the *floor*. A corpus containing scanned papers will sit
  higher, and nothing in the pipeline signals which case it is in.
- ~~GH-326's re-measurement wants a trustworthy denominator, which is what this
  is.~~ **Retracted (GH-357).** This is two detector booleans over 528 pages;
  GH-326's re-measurement is the 14 real candidates plus Nakamura p42, pinned at
  `process()`. Different instrument, different question — and GH-326 was already
  closed when this log was written. GH-353 is the current gate. Claiming this as
  that denominator is the overclaim class GH-338 §5 names.

## Reproducing

Open each page of each manifest paper, call `find_tables()` and
`has_numeric_columns()`, and tally the three counts. No model runs and no network.
PDFs are read from `~/Dropbox/backups/…` — note the
`~/Dropbox/research/jmp/references/` path holds 0-byte online-only placeholders that
read as empty files.
