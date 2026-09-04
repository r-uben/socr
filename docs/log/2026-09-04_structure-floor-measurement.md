# Structure-class floor: why every grid candidate was refused (measurement)

2026-09-04, read from the audit logs of the two ladder corpus runs (docs/log/2026-09-04_ladder-corpus-run.md, docs/log/2026-09-04_ladder-corpus-rerun.md). Content-free. Decision input for #589. Caveat: run 2 spans three source digests (see the re-run log correction).


Read-only measurement. Data: `scratchpad/ladder-run/` (run 1) and `scratchpad/ladder-run2/`
(run 2), `audit_log.json` + `pages/NNNNN.json` per doc, `manifest.json` for original page
numbers. Code read (not modified): `src/socr/pipeline/orchestrator.py`,
`src/socr/core/manifest.py`, `src/socr/core/tables_trust.py`.

## CAVEAT — the two runs are not a clean repeat

Every page's `socr_source_digest` differs between run 1 (`5ae7b5b4…`) and run 2. Worse,
run 2 itself carries **three different digests** across its own docs (`ed3520d4…` for
doc00–doc04, `f55e62cf…` for doc05/doc07, `77ed229b…` for doc08) — the checkout was edited
*while run 2 was executing*. So "did the same page float in both runs" cannot be read as a
pure non-determinism signal; some of the divergence is confounded with code changes made
between and during the runs. Flagged explicitly below wherever it matters.

## Headline

| | Run 1 | Run 2 |
|---|---|---|
| Floor pages | 4 (doc00 p7, doc01 p15, doc02 p42, doc03 p34) | 3 (doc01 p14, doc01 p15, doc02 p42) |

Only **2 pages floor in both runs**: doc01 p15 (Gertler-Karadi) and doc02 p42
(Nakamura-Steinsson). The other 3 floor pages appear in only one run each — given the
digest churn above, that is *not* strong evidence of flakiness on its own.

Reason families across all 5 distinct floor pages (7 page-occurrences total):

| Family | Occurrences (page-level) | Judgment from logs alone |
|---|---|---|
| `native_table_verifier_hard_fail: value_guard_multiset_mismatch` (dropped/invented/shifted numeric tokens) | 4 pages (doc01 p15 ×2 runs, doc02 p42 ×2 runs, doc01 p14) | **Real damage** — a mechanical multiset diff, not a style call |
| `native_table_verifier_hard_fail: value_guard_label_binding` (interleaved row pattern) | 1 page (doc03 p34) | **Real damage** — confirmed independently by a binding adjudication that held (see below) |
| `native_table_verifier_warn: ambiguous_lane_count_mismatch` (deferred to VLM, not a hard reject) | 2 pages (doc00 p7, doc01 p14) | **Not evidence of damage** — this is a defer, not a refusal |
| `value_guard_row_count_warning` (native row count ≠ output row count) | 3 pages | Ambiguous on its own; needs pairing with a hard-fail to mean anything |
| `text_grid_rejected` (native text-strategy grid rebuilt by word-geometry rowizer, "shipped values are correct") | 2 pages (doc00 p7, doc03 p34) | **Says it is NOT damage** — the event's own detail states the rebuilt values are correct; this is a routing/robustness gap, not a content defect |
| `table_not_scorable` (native parse doesn't form a grid / no ground-truth label match) | 5 pages | Can't judge — this is an absence-of-signal event, not a positive refusal reason |
| `table_ladder_unverified` — infra cause (`table witness could not be prepared` / judge-ladder `infra problem, retryable`) | 3 pages | **Not damage** — an infrastructure failure to run the judge, not a content verdict |
| `landscape_page_refused` (rotated page) | 1 page (doc02 p42) | Structural, plausibly correct to refuse for that lane, but doesn't explain why *no* candidate (model or text-grid) covered it |
| `table_escalation_timeout` (120s, lane disabled for rest of doc) | 1 page (doc03 p34) | **Infra**, not content |
| `table_binding_adjudicated … held: 0/3 contradictions disproved` | 1 page (doc03 p34) | Corroborates real damage on that page |
| `table_binding_adjudicated … lifted: 1/1 contradictions disproved` + `table_ladder_accepted` | 1 page (doc01 p15, both runs) | **Contradicts the floor** — see finding below |

## Per-page detail

### doc00 p7 (run 1 only) — Cochrane-Piazzesi, "bond risk premia"
Order of events: `table_value_drift_unadjudicated` (qwen, gemini — ambiguous, row-count
discrepancy prevents pairing) → `text_grid_rejected` (native: *"1 text-strategy table
grid(s) rejected: a lane boundary split 38 native numeric token(s) … Rebuilt with the
lossless word-geometry rowizer — the shipped values are correct, but this page's layout is
adversarial to find_tables(strategy='text')"*) → `value_guard_row_count_warning` (qwen,
gemini: 16 output rows vs 14 native) → `native_table_verifier_warn` (qwen, gemini:
*"ambiguous_lane_count_mismatch: native_lanes=19, output_cols=11/12 … deferring to
VLM"* — a WARN, not a hard fail) → `table_not_scorable` (*"native text layer parsed 4
row(s) that do not form a grid"*) → `table_ladder_unverified` (qwen: *"infra problem,
retryable on resume"*).

**No hard fail anywhere on this page.** The native text-strategy grid was explicitly
*rebuilt correctly* by the rowizer; both model attempts only ever got a WARN-level
lane-count ambiguity deferred to VLM judging, and that judging never completed
(infra problem). This page floored without any positive evidence of a wrong number.

### doc01 p15 (both runs) — Gertler-Karadi, "monetary policy surprises"
Order: `structure_class_ladder_exhausted_floor` → `table_structure_failed` (native: ragged
widths / detached label row) → `native_table_verifier_hard_fail` ×2 qwen (*"9 paired row(s)
… multiset mismatch"*, then *"1 paired row(s) …"*), ×1 gemini (*"5 paired row(s) …"*) →
`table_not_scorable` → `table_escalation_rejected` (qwen: *"exactness 0.0% -> 0.0% (0 vs 0
of 36 cells)"*) → `table_binding_adjudicated` (qwen: *"table p2-t0 binding adjudication
lifted: 1/1 contradictions disproved"*) → `table_ladder_accepted` (qwen: *"table p2-t0
accepted by the judge ladder"*).

**Contradiction:** the same table (`p2-t0`) that hard-failed the winner-side value guard
(multiset mismatch) is later reported `table_ladder_accepted` after its one binding
contradiction was disproved by adjudication — yet the page still ends in
`structure_class_ladder_exhausted_floor` and ships the fail-closed marker. Reading
`src/socr/core/manifest.py::_grid_authored_attempt`, a candidate only counts as
"grid-authored" if `audit_passed` is True (or explicitly `REJECTION_AMBIGUOUS_DEFERRED`);
the multiset-mismatch hard fail sets `audit_passed=False` on that attempt and is never
revisited once the (separately-tracked) judge ladder accepts it downstream. So the winner
selection and the judge-ladder acceptance are **two different gates that can disagree on
the same candidate**, and selection does not consult the ladder's later verdict. This is
the strongest "floor may be over-refusing" signal in the data — it needs a code read to
confirm whether that's intended (winner-side guard is meant to be final and stricter) or a
gap (the ladder's disproof should have reopened the candidate).

### doc02 p42 (both runs) — Nakamura-Steinsson, "high frequency identification"
Order: `structure_class_ladder_exhausted_floor` → `landscape_page_refused` (native:
*"dominant text direction is rotated; prose retained, page routed to OCR"*) →
`native_table_verifier_hard_fail` ×2 qwen + ×1 gemini, all identically *"9 paired row(s)
have numeric-token multiset mismatch"* → `table_not_scorable` (*"parsed 0 row(s)"*) →
`table_ladder_unverified` (qwen: *"not judged: no table witness could be prepared, so no
rung ran (not retryable — a re-run reaches the same empty witness)"*).

Three independent attempts (qwen ×2, gemini) all report the identical 9-row multiset
mismatch — that consistency is real evidence of damage, not noise. Native itself refused
to author a grid at all (rotated page). This is the cleanest "floor is doing its job" case
in the set.

### doc03 p34 (run 1 only) — Pflueger-Rinaldi, "fed moves market"
Order: `table_structure_failed` (native: ragged widths/detached label) →
`text_grid_rejected` (native: *"a lane boundary split 12 native numeric token(s) …
Rebuilt with the lossless word-geometry rowizer — the shipped values are correct"*) →
`value_guard_row_count_warning` (9 vs 8) → `native_table_verifier_hard_fail` ×2
(qwen, gemini: *"value_guard_label_binding: 4 adjacent interleaved pair(s) >= 4 clean
labeled row(s) — name/value binding is broken (systematic interleaved row pattern)"*) →
`table_escalation_timeout` (qwen: *"escalation exceeded 120s; lane disabled for the rest of
this document"*) → `table_binding_adjudicated` (gemini: *"held: 0/3 contradictions
disproved"*) → `table_ladder_unverified` (gemini: *"mechanical binding check found a
contradiction that adjudication did not disprove"*).

Here the label-binding hard fail is corroborated independently: the binding adjudication
was asked to disprove the contradiction and could not (0/3 held). That's two independent
mechanisms agreeing — real damage. Same page also has a `text_grid_rejected` event whose
own detail says the rebuilt native values are correct, so that particular candidate wasn't
the reason for the floor; the label-binding fail was.

### doc01 p14 (run 2 only) — Gertler-Karadi, page before p15
Order: `table_value_drift_unadjudicated` (qwen: rows where *model wrote numbers* and *"the
page reads (nothing)"* — the reverse direction from doc00's case: the model may be
inventing values, not dropping them) → `table_structure_failed` (native) →
`value_guard_row_count_warning` (qwen: 5 output vs 10 native; gemini: 10 output vs 21
native — large native tables) → `native_table_verifier_warn` (qwen, gemini:
`ambiguous_lane_count_mismatch`, deferred) → `native_table_verifier_hard_fail` (qwen: *"5
paired row(s) … multiset mismatch"*) → `table_structure_failed` (gemini:
`header_unattributed`) → `table_not_scorable` → `table_ladder_unverified` (gemini: *"infra
problem, retryable on resume"*).

The row-count gaps here are large (native has roughly 2x the rows either model captured),
consistent with a genuinely dense/complex table rather than a borderline call.

## Stability

- doc01 p15 and doc02 p42 floor identically in both runs (both directions of evidence:
  same hard-fail reason strings, same candidate sources tried).
- doc00 p7 and doc03 p34 (run 1) and doc01 p14 (run 2) each appear only once. Given the
  source-digest churn documented above, this is not attributable to model
  non-determinism alone without re-running on a pinned commit.

## Candidate sources tried per floor page

Every floor page tried the same three candidate sources: native geometry grid (via
`text_grid_rejected` / `table_structure_failed` / `landscape_page_refused`), the qwen VLM
rung, and the gemini VLM rung. None reached an accepted, `audit_passed=True` grid before
the floor fired. doc01 p15 is the exception: qwen's grid *was* later accepted by the judge
ladder, but too late to affect winner selection (see contradiction above).

## What I can and cannot judge from logs alone

Can judge: whether a hard-fail reason string represents a mechanical, checkable defect
(multiset mismatch, label-binding interleave, adjudication-confirmed contradiction) vs.
an infra/defer signal (WARN lane-count ambiguity, judge-ladder infra failure, escalation
timeout) that carries no positive evidence of damage.

Cannot judge from logs alone: whether the *specific* numbers flagged by a multiset
mismatch are actually wrong in the shipped/refused candidate, or whether a `text_grid_rejected`
grid that the event itself calls "correct" would in fact render acceptably — that needs the
page image and the two candidate texts side by side.

## Next measurement to settle "floor is right" vs "over-refuses"

Render the page image for the two purely-WARN/infra floor pages (doc00 p7, and doc01 p14's
`ambiguous_lane_count_mismatch` candidates) alongside the *refused* text-strategy grid
(`text_grid_rejected`'s rebuilt output, which the event log itself asserts is correct) and
the qwen/gemini candidate outputs, and hand-check cell-by-cell against the source PDF. This
isolates the two pages with **no hard-fail evidence** from the three pages that do have a
hard-fail (doc01 p15, doc02 p42, doc03 p34), where the multiset/binding mismatches are
themselves mechanically checkable and don't need visual judgment — the question there is
narrower: for doc01 p15 specifically, trace why `table_ladder_accepted` doesn't reopen
`_grid_authored_attempt`'s `audit_passed` gate, since that is the one page where two
different gates in the same run disagree about the same candidate.

Needed data: rendered PNG crops of doc00 p7 / doc01 p14's table regions at 300 DPI (already
available via `page_images` outputs in the run dirs, or re-render from the source PDFs
listed in `manifest.json`), plus the exact refused candidate texts (already in
`audit_log.json`/`NNNNN.json`) for a side-by-side.
