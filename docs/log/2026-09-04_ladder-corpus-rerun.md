# Ladder corpus re-run — after the #582 inline-math binder fix

2026-09-04. Run 1 (baseline): 12:17–14:25 local, socr `main` @ `e830d9b` (before #582),
logged in `docs/log/2026-09-04_ladder-corpus-run.md`. Run 2 (this log): 14:57–17:37 local,
socr `main` @ `f434019` (`#584`, merging `fix(tables): unwrap inline-math presentation
before the binder and adjudicator compare cells (GH-582)`). Same manifest, same 8 papers,
same 20 pages, same `socr process <pdf> --write-manifest --verbose` invocation, sequential.
Content-free: identifiers, counts and dispositions only.

## Headline

| | run 1 (before #582) | run 2 (after #582) |
|---|---|---|
| pages | 20 | 20 |
| tables that reached the ladder | 18 | 18 |
| ACCEPTED | 5 | 7 |
| WITHHELD | 1 | 0 |
| UNVERIFIED | 12 | 11 |
| page endings | 14 `model_output`, 6 `fail_closed_marker` | 15 `model_output`, 5 `fail_closed_marker` |
| page status | 5 SUCCESS, 9 WARNING, 6 ERROR | 4 SUCCESS, 11 WARNING, 5 ERROR |
| wall clock | 2 h 07 min (6.4 min/page) | 2 h 40 min (8.0 min/page) |
| cloud cost | $0.0010 | $0.0020 |

Exit code 1 on 7 of 8 papers in both runs — every one surfaced at document level, as
designed.

## Contradiction classes, before and after

Classified every item in every table's `binding_adjudication[<table>].items[]`
(`native_token`, `model_token`, `kind`), using `strip_math_presentation` /
`tokens_agree` from `src/socr/tables/native_verifier.py` and `adjudication.py` on the
`f434019` tree (`PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-rerunlog/src`) to
confirm what the fix does and does not normalise, then checking each remaining item by
inspection.

| class | run 1 | run 2 |
|---|---|---|
| (a) inline-math wrapping (`$-0.06$`, `Adjusted $\text{R}^2$`) | 6 | 0 |
| (b) sibling LaTeX commands (`\Delta`, `\log`, `\&`, `\mathcal`) | 6 | 6 |
| (c) native row-label defect (truncated / run-on / shredded) | 12 | 12 |
| (d) lane shift (one side empty) | 4 | 4 |
| **total items** | **28** | **22** |

Class (a) is gone in run 2, at every locus it appeared: the 5 numeric cells on doc02 p2
(`$-0.06$` vs `−0.06` and siblings) and the one text row label on doc03 p1 (`Adjusted
$\text{R}^2$` vs `Adjusted R2`) no longer appear in `binding_adjudication` at all — the
binder does not convict them, so they never reach the item list. Every other class is
unchanged item-for-item: same count, and by inspection the same or an equivalent
wrapping variant (OCR nondeterminism changes `$\Delta \log \text{ Comm. price (3m)}$` to
`$\Delta \log$ Comm. price (3m)` between runs on doc05/doc07, but both still fail
`tokens_agree` for the same reason).

**Why (b) does not close under #582**: `strip_math_presentation(label=True)` unwraps a
whole `$…$`/`\(…\)` pair, `\text{}`/`\mathrm{}`/`\textbf{}`, and flattens `^`/`_` script
markers — it does not translate a bare LaTeX command. `\Delta` and `\log` never resolve
to `∆` and `log`, so `∆Slope (3m)` and `\Delta Slope (3m)` still normalise to different
strings. This is exactly the follow-up scope #582 recorded and did not attempt.

**Why (c) does not close under #582**: these are native-side defects, not presentation —
`1t 1t` (doc04 p3) is a shredded subscript, `Sample 1988:1–2019:12 1994:1–2019:12 …`
(doc05/doc07 p1) is four sample windows run together where the model reads one column
header `Sample`, and `Treasury inst. forward rate` (doc02 p3/p4) is missing its `3Y`/`5Y`/
`10Y`/`2Y` maturity prefix. No amount of unwrapping the model side fixes a native token
that dropped content; per project convention (#331 pattern) these are native row-label
loss, not a binder/adjudicator predicate gap.

**Why (d) does not close under #582**: `tokens_agree` refuses an empty key
(`bool(left_key) and left_key == right_key`) by design — an empty native or model token
is a structural lane mismatch, not a formatting difference math-stripping could paper
over.

## Per-table diff (same doc/page/table id, run 1 → run 2)

| table | run 1 | run 2 | cause |
|---|---|---|---|
| doc00 p1-t0 | UNVERIFIED (infra, label lost — #581) | UNVERIFIED (infra, label lost — #581) | unchanged |
| doc00 p2-t0 | ACCEPTED | ACCEPTED | unchanged |
| doc00 p2-t1 | **WITHHELD** (readers disagreed on `H1C3`) | **ACCEPTED** | reader disagreement did not recur — no binding-contradiction items on this table in either run; not a #582 effect |
| doc00 p3-t0 | UNVERIFIED (no table witness) | UNVERIFIED (no table witness) | unchanged |
| doc00 p4-t0 | UNVERIFIED (binding contradiction) | UNVERIFIED (binding contradiction) | unchanged — contradiction items not captured for this table |
| doc01 p1-t0 | UNVERIFIED (rung 2 answered `ok:false`) | UNVERIFIED (infra, both rungs `ok:true`, label lost — #581) | rung 2 (`agy`) failed in run 1 and succeeded in run 2; the table stayed unverified either way, cause differs — nondeterminism in rung 2 availability, not #582 |
| doc01 p2-t0 | ACCEPTED | ACCEPTED | unchanged |
| doc01 p3-t0 | ACCEPTED | ACCEPTED | unchanged |
| doc01 p3-t1 | ACCEPTED | ACCEPTED | unchanged |
| doc02 p1-t0 | UNVERIFIED (no table witness) | UNVERIFIED (no table witness) | unchanged |
| doc02 p2-t0 | UNVERIFIED (binding contradiction: 5 wrapped numeric cells) | **ACCEPTED** | direct #582 effect — the only contradiction items on this table were class (a); zero items post-fix |
| doc02 p3-t0 | UNVERIFIED (binding: 3 native-truncated labels) | UNVERIFIED (binding: same 3 items) | unchanged, class (c) |
| doc02 p4-t0 | UNVERIFIED (binding: 4 native-truncated labels) | UNVERIFIED (binding: same 4 items) | unchanged, class (c) |
| doc03 p1-t0 | UNVERIFIED (binding: 3 items — 2 lane-shift + 1 wrapping) | UNVERIFIED (binding: 2 items — the 2 lane-shift items only) | the wrapping item resolved (#582); the 2 lane-shift items (class d) still hold the table unverified |
| doc04 p2-t0 | ACCEPTED | ACCEPTED | unchanged |
| doc04 p3-t0 | UNVERIFIED (binding: shredded native subscript) | UNVERIFIED (binding: same item) | unchanged, class (c) |
| doc05 p1-t0 | UNVERIFIED (binding: 6 items) | UNVERIFIED (binding: 6 items) | unchanged — 3 class (b), 1 class (d), 2 class (c) in both runs |
| doc07 p1-t0 | UNVERIFIED (binding: 5 items) | UNVERIFIED (binding: 5 items) | unchanged — 3 class (b), 1 class (d), 1 class (c) in both runs |

Two of the nine binding-contradiction/withheld tables in run 1 flipped to ACCEPTED in
run 2. One (doc02 p2-t0) is attributable to #582 — its contradiction items were entirely
class (a). The other (doc00 p2-t1) is a reader-agreement guard unrelated to the binder,
and did not recur in run 2 for reasons the audit log does not record (no items were ever
logged for that table in either run) — treat it as run-to-run variance, not a fix effect.

One table (doc02 p2, at the *page* level, not the table-ladder level) also flipped the
other way: `warning`/`model_output` in run 1 to `error`/`fail_closed_marker`
(`native_table_unverifiable`) in run 2. That disposition comes from the native-trust
judgment, a separate mechanism from the table ladder, and is further evidence the OCR
output differs between runs on this page.

**Variance caveat.** `qwen3-vl:30b-a3b-instruct` is not deterministic run to run: table
ids, witnessed-table counts, and even which pages get a table witness at all can differ.
Every table in the diff above matched on `doc/page/table_id` between runs — none
appeared in only one run this time — but that is not guaranteed on a future re-run and
should not be assumed.

## What the re-run settles

1. **#582 closes exactly the class it targeted, and only that class.** Every recorded
   inline-math-wrapping contradiction (6 items, class a) is gone in run 2; every
   sibling-LaTeX, native-defect, and lane-shift item (22 items, classes b/c/d) is
   unchanged. One table (doc02 p2-t0) moved ACCEPTED as a direct, attributable result.
2. **The sibling LaTeX-command class (`\Delta`, `\log`, `\&`, `\mathcal`) is confirmed
   still open**, by code inspection of `strip_math_presentation` (unwraps delimiters and
   `\text{}`, does not translate bare commands) and by the unchanged item count on
   doc05/doc07 across both runs. Filed as follow-up scope on #582.
3. **The native row-label defects on #331's pattern are confirmed still open** — same 12
   items, same loci, both runs. Not in #582's scope; a native-extraction problem, not a
   binder/adjudicator predicate gap.
4. **#581 (UNVERIFIED label lying about retryability) reproduces in both runs** — the
   "infra problem" default-label case appears in run 1 (1 table) and run 2 (2 tables,
   including one that changed cause between runs), and the underlying reason is still not
   written to the trail in either.
5. **Wall clock and cost both rose** (2 h 07 → 2 h 40, $0.0010 → $0.0020) between runs.
   The extra cost tracks more `$0.0002` rung-1 calls in run 2 (doc00 p1–p4 all routed
   through a rung call in run 2, only some did in run 1); this looks like routing
   variance from the different OCR output per page, not a #582 cost regression — #582
   touches comparison logic, not routing.

## What it does not settle

- Whether doc00 p2-t1's WITHHELD→ACCEPTED flip is meaningful or noise: no
  `binding_adjudication` items were ever recorded for that table in either run, so there
  is nothing to diff — the guard that flipped it is not instrumented finely enough to
  tell without a further code read.
- A clean pre/post rate for #582 beyond this one corpus and this one flip: 8 of 20 pages
  are OCR-nondeterministic enough that table ids and witnessed-table counts could differ
  on a third run, which would change the denominator.
- Whether the routing/cost increase in run 2 is systematic or an artifact of this
  particular OCR pass; would need a same-commit re-run to isolate.

## Reproduce

Run 1 data: `/private/tmp/claude-501/-Users-rubenffuertes-repos-tools-socr/de2d9763-56e2-4f37-bcbb-0cb876a7f07f/scratchpad/ladder-run/`.
Run 2 data: the same session scratchpad, `ladder-run2/` — identical layout
(`manifest.json`, `progress.txt`, `out/<doc>/<doc>/audit_log.json`,
`out/<doc>/<doc>/pages/NNNNN.json`, `tabulate.py`). `tabulate.py <run dir>` reprints the
per-page table; the contradiction-item and `table_ladder_*` event extraction used here
reads `pages/NNN.json` (`binding_adjudication[<table>].items[]`) and `audit_log.json`
(events whose `kind` starts with `table_ladder_`, `data.rung_trail`, top-level `detail`).
