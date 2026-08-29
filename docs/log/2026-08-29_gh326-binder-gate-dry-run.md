# GH-326 step 1 — what the proposed binder gate would actually do

2026-08-29. Dry-run measurement taken **before** wiring `binding.bind` into the
production SUCCESS disposition, because the issue's step 1 ("wire it") and step 2
("re-measure") are in the wrong order: nobody had measured what the gate would do,
and the only prior measurement explicitly warns its contradiction counts "must not be
used as a precision claim".

Content-free by the convention of `2026-08-22_binding-oracle-corpus-measurement.md`:
method, identifiers, aggregate counts and content-free diagnostics only. The papers
are copyrighted; no page content, candidate markdown or token text is recorded here.

## Method — the self-bind experiment

The 2026-08-22 measurement bound **model** candidates against native geometry and
found 0 of 14 fully checked. That leaves two competing explanations, which it could
not separate:

1. the model grids were too damaged to bind, or
2. the binder cannot fully check a real table at all.

This run separates them by removing the model entirely. For each table page in
`2026-08-20_lane-comparison-manifest.json`, the candidate is **socr's own native
reconstruction of that same page** — `BornDigitalDetector.extract_structured`, whose
table regions are built by the word-geometry rowizer from the very words the binder
then binds against.

This is the easiest possible input. Every value in the candidate provably originates
in that page's own text layer. If the binder cannot fully check *this*, no model
candidate will do better.

- Corpus: the 9-paper manifest; the 15 pages marked `kind: table`.
- `bind(page.get_text("words"), <largest pipe block>)`, no model calls, no network.
- PDFs read from a local materialised copy. Note the `Dropbox/research/jmp/references`
  path used by earlier notes now holds **0-byte online-only placeholders**; the
  materialised copies are under `Dropbox/backups/`.

## Result

| Measure | Count |
|---|---|
| Pages measured | 15 |
| **Fully checked** | **0 / 15** |
| Row binding unverifiable | 15 / 15 |
| Column binding unverifiable | 15 / 15 |
| Produced no pipe grid at all | 3 / 15 |
| **`model_unbound` non-empty** | **6 / 15** |

Row *and* column binding were unverifiable on **every single page**, including pages
whose ambiguity count was zero.

## What this settles

**The coverage failure is in the binder, not in the model candidates.** 0/14 in the
prior measurement was not a statement about Gemini or Qwen output. The binder cannot
fully check a table even when the candidate is derived from the same word geometry it
is being compared against.

**`model_unbound` cannot be read as an invented-value signal today.** It fires on 6 of
15 pages in a run where **no model was involved at all**. On the worst page, 66 tokens
were reported `model_unbound`; **all 66 are literally present in that page's own word
layer** (checked by set membership against `get_text("words")`, counts only recorded
here). The signal is reporting "the binder could not place this value", which is a
coverage failure — not "this value was invented", which is what #270 needs.

**Wiring the gate now would be actively harmful.** It would demote roughly 40% of
table pages on a signal that is provably wrong in the easiest possible case, trading
silent fabrication for wholesale false demotion. In a citation corpus a false demotion
is cheaper than a false accept, but not when it fires this often on correct pages: the
demotion stops carrying information, which is the same failure as the 327 unread flags
that motivated #317.

**The #270 fixture is not reachable.** Nakamura p42 — the page the issue names as the
re-measurement target — produced **no pipe grid** from the native path, so the binder
never sees it. Two other pages behaved the same way.

## Consequence for the ticket

#326 as written ("wire it, then re-measure") cannot be executed in that order. The
prerequisite is the one the 2026-08-22 log already identified and this run confirms
from the opposite direction: **candidate-to-native-table scoping and coverage must be
fixed first**, so that some known-good real grid becomes fully checked and the known
shifted-label page is actually reached.

Until then the gate has nothing trustworthy to gate on, and #322 stays blocked — which
is the correct outcome, just for a sharper reason than when the issue was filed.

## Reproducing

The measurement is a read-only sweep: open each manifest page, take the largest pipe
block from `extract_structured`, call `bind`, and tally `fully_checked`,
`row_binding_unverifiable`, `column_binding_unverifiable`, `ambiguous_count`,
`len(model_unbound)` and `len(native_unbound)`. No socr source is modified and no
engine runs.
