# STATUS — extraction defects, cross-plan coordinator

Last updated: 2026-08-12

Owns the **global** wave order and file ownership across five sibling plans. Each folder
keeps its own tickets; none of them may define its own dispatch order any more, because
three of them write the same files and each was scheduled in isolation.

| Plan | Issue | Folder |
|---|---|---|
| Figures extracted as tables | #150 | [`../gh150-figures-as-tables/`](../gh150-figures-as-tables/TICKETS.md) |
| Structure lost at full recall | #151 | [`../gh151-structural-gate/`](../gh151-structural-gate/TICKETS.md) |
| Landscape pages transposed | #147 | [`../gh147-landscape-pages/`](../gh147-landscape-pages/TICKETS.md) |
| Rowizer destroys values | #144 | [`../gh144-rowizer-destroys-values/`](../gh144-rowizer-destroys-values/TICKETS.md) |
| Side-by-side tables merged | #152 | [`../gh152-side-by-side-tables/`](../gh152-side-by-side-tables/TICKETS.md) |

Ticket detail — Problem / Do / Files / Done when — lives in each folder's `TICKETS.md` and
is unchanged. This file decides only **what may run at the same time**.

## Wave 0 — merge gate — **DONE 2026-08-12**

Both PRs are merged to `main`; the four tickets they gated are unblocked.

- **PR #148** (GH-145) — `dominant_text_direction()` is on `main`, so GH-147 A1 reduces to
  exposing it on `PageAssessment`.
- **PR #149** (GH-146) — the `reconstruct.py` header work. GH-144 A2/A2b and GH-152 A1 may
  now branch from `main`.

**Issue #146 is deliberately still open.** PR #149 fixed only cause 1 (a data row promoted to
header). Cause 2 — the region excluding the header band — is tracked as **GH-144 A2b**, in the
GH-144 folder because it is the same class of `reconstruct.py` boundary error and must
serialize on the same file.

## File ownership (the reason waves exist)

One file, one wave. Collisions are what the per-folder schedules could not see.

| File | Claimed by | Serialized as |
|---|---|---|
| `src/socr/tables/reconstruct.py` | GH-144 A2 · GH-144 A2b · GH-152 A1 · GH-152 A2 | W2 → W2 → W3 → W4 |
| `src/socr/pipeline/orchestrator.py` | GH-150 B1 · GH-147 A2 · GH-151 B1 | W1 → W2 → W3 |
| `src/socr/core/born_digital.py` | GH-147 A1 · GH-147 A2 · GH-151 B1 | W1 → W2 → W3 |
| `tests/test_chart_detection_gh150.py` | GH-150 A2 · GH-150 B2 | W2 → W3 |
| `src/socr/figures/extractor.py` | GH-150 A1 | W1 only |
| `src/socr/tables/native_verifier.py` | GH-151 A2 | W1 only |
| `src/socr/tables/structure_check.py` (new) | GH-151 A1 | W1 only |
| `src/socr/core/state.py` | GH-151 B1 | W3 only |

## Waves

Everything on one row dispatches in parallel — the write sets are disjoint by construction.
A wave closes when every ticket in it is reviewed and accepted, not when its code is written.

### Wave 1 — 6 parallel, no blockers

| Ticket | Writes | Note |
|---|---|---|
| GH-150 A1 | `figures/extractor.py` | thin-stroke vector plots as chart marks |
| GH-150 B1 | `pipeline/orchestrator.py` | chart-vs-table arbitration |
| GH-151 A1 | `tables/structure_check.py` (new) | grid-shape checks |
| GH-151 A2 | `tables/native_verifier.py` | x-position binding check |
| GH-147 A1 | `core/born_digital.py` | dominant text direction on `PageAssessment` |
| GH-144 A1 | `logs/` only | read-only diagnosis |

GH-147 A1 assumes wave 0 merged #148; if it has, the ticket reduces to exposing the existing
`dominant_text_direction()` on `PageAssessment`.

### Wave 2 — 3 lanes

| Ticket | Writes | Depends on |
|---|---|---|
| GH-144 A2 **then** A2b | `tables/reconstruct.py` | GH-144 A1 |
| GH-147 A2 | `core/born_digital.py`, `pipeline/orchestrator.py` | GH-147 A1 |
| GH-150 A2 | `tests/test_chart_detection_gh150.py` | GH-150 A1 |

A2 and A2b are one lane held by **one agent in sequence**, not two parallel tickets. Waves
bound concurrency, not how much work an agent may do on a file it already owns.

### Wave 3 — 4 parallel

| Ticket | Writes | Depends on |
|---|---|---|
| GH-152 A1 | `tables/reconstruct.py` | GH-144 A2b |
| GH-151 B1 | `core/born_digital.py`, `core/state.py`, `pipeline/orchestrator.py` | GH-151 A1+A2 · GH-147 A2 |
| GH-150 B2 | `tests/test_chart_detection_gh150.py` | GH-150 A1+B1 |
| GH-147 B1 | `tests/test_landscape_refusal_gh147.py`, `logs/` | GH-147 A2 |

### Wave 4 — 3 parallel

| Ticket | Writes | Depends on |
|---|---|---|
| GH-152 A2 | `tables/reconstruct.py` | GH-152 A1 |
| GH-144 A3 | `logs/` | GH-144 A2, A2b |
| GH-151 B2 | GH-49 issue comment | GH-151 B1 |

### Wave 5

| Ticket | Writes | Depends on |
|---|---|---|
| GH-152 B1 | `tests/test_side_by_side_tables_gh152.py`, `logs/` | GH-152 A2 |

## Critical path

`GH-144 A1 → A2 → A2b → GH-152 A1 → GH-152 A2 → GH-152 B1` — six tickets, all on
`reconstruct.py`. Nothing else in the program is longer, so this chain sets the schedule;
every other lane has slack. Staff it first and keep **one** agent on it — the file admits no
concurrency, and handing it between agents costs more than it saves.

## Decisions taken here

**GH-144 A2 precedes GH-152 A1 on `reconstruct.py`.** Both rewrite lane geometry and neither
folder knew about the other. GH-144 goes first because it destroys numeric values in a citation
corpus (worse than merging two tables), because a boundary-in-whitespace constraint is local and
survives banding, and because GH-152 A1's `Done when` demands the suite pass unchanged — which is
cheaper to satisfy after the boundary bug is gone than before.

**Revisit if GH-144 A1 says otherwise.** A1 is diagnosis; if it finds the false gutter is an
artifact of full-page-width clustering, then x-banding (GH-152 A1) is the actual root cause and
this order inverts. A1 must state explicitly whether banding subsumes the boundary fix.

**GH-147 A2 precedes GH-151 B1** on `born_digital.py` + `orchestrator.py`. GH-147 A2 adds a
narrow "refuse this native page" path; GH-151 B1's own ticket says to mirror the existing
fail-closed pattern, so it should mirror a pattern that already exists rather than invent one
concurrently.

## Relationship to PR #179

PR #179 (`docs: open-issue priority graph`) schedules all 41 open issues in Waves 1–8 and
overlaps these five plans. Where they disagree, **this file wins for #144/#145/#146/#147/#150/
#151/#152** and #179 governs the other 34 issues. Two specific corrections to #179's Wave 1:

- It ranks #146 fourth and #145 sixth, but both are implemented and awaiting merge, and both
  gate work it ranks earlier. They are wave 0 here, not mid-wave-1.
- Its prose says same-wave issues run in parallel while its graph draws Wave 1 as a serial
  chain. The waves above are parallel by write set, which is the only definition that holds.

## Wave 1 — CLOSED 2026-08-12

All six tickets **accepted**. `main` at `712ee90`; 1571 passed / 2 xfailed;
`uvx ruff@0.16.0 format --check .` clean (280 files).

| Ticket | Landed as |
|---|---|
| GH-151 A1 | `d4f6154` |
| GH-144 A1 | PR #180 (`ea27edd`) |
| GH-150 A1 | PR #182 (`dfa35ea`) |
| GH-147 A1 | PR #185 |
| GH-150 B1 | PR #186 (`712ee90`) |

**GH-151 A2 (PR #184) is NOT part of the closed wave — parked as draft.** Its column
binding is modal consensus: the most common row signature is taken as the correct column
structure. In a citation corpus the minority rows are frequently the ones that matter (a
wrapped header band, a totals row, a merged cell), and the primitive cannot distinguish
"deviant because malformed" from "deviant because genuinely different". The ordinal
predicate that would qualify it is gated on `value_col_count == len(centres)`, so outside
that exact case the modal vote is unqualified. Being inert today is not a defence — the
risk is adoption later on the strength of having been merged. **GH-151 B1 (wave 3) must
not build its gate on this primitive**; it needs a design saying what column binding
should key on instead.

### Residual, still unfiled

The chart-loss gap is **not fixed by B1 and has no issue**. Chart region placeholders are
resolved only when the judge *rejects* the ladder output. If the judge *accepts* a VLM
extraction on a mixed page, that output governs and the placeholder never reaches the
final markdown — the chart is silently dropped. This is the "no silent content loss" red
line.

## Wave 2 — CLOSED 2026-08-13

All three tickets **accepted and merged**. `main` at `13033a3`; 1591 passed / 1 xfailed;
`uvx ruff@0.16.0 format --check .` clean (284 files).

| Ticket | Landed as |
|---|---|
| GH-150 A2 | PR #194 (`4ac487d`) |
| GH-144 A2+A2b | PR #192 (`d645b24`) |
| GH-147 A2 | PR #193 (`13033a3`) |

Dispatched as one `ticket-relay` workflow (GPT+Opus pair → Fable rules → Grok-4.6+GPT
converge → Fable ratifies → Sonnet implements → Gemini reports), then one review round,
one fix round per PR, then a second review round. Both #192 and #193 needed a fix round;
#194 merged on the first pass.

### Measured outcome, not just "merged"

GH-144's acceptance is a measurement, not a green suite. On the NS QJE reference pages
(sha256 `6611c6af…`), decimal-token loss through `extract_structured`:

| Page (A1's naming) | fitz idx | before | after |
|---|---|---|---|
| p17 (TABLE II) | 16 | 5 lost | **0** |
| p42 (Appendix A.1) | 41 | 35 lost | **0** |
| p43 (Appendix A.2) | 42 | 13 lost | **0** |

Before the fix, output also *exceeded* raw on two pages (duplication); after, output
matches raw exactly. This measures token survival only — it does not certify table
*shape*.

### A review false-alarm worth not repeating

#192's second reviewer returned REQUEST_CHANGES claiming the fix caused **100% table
loss** on p17 and p42. It measured `reconstruct_table_regions` in isolation, loading the
PR's source as a standalone module because the editable install resolves `import socr` to
the checkout rather than the PR branch.

The isolated observation was correct — that function does return `[]` on those pages. The
conclusion was not: `extract_structured`'s `if not table_regions:` gate
(`born_digital.py`) then fires `rowize_from_words_chart_aware`, which recovers the table
whole. **Measuring one rung of the ladder is not evidence about the ladder.** Any claim
about content loss must be measured at the caller, end to end. The rest of that review
stood up and was useful — it confirmed the four earlier fixes and that the new tests are
load-bearing rather than vacuous.

### Residuals

- **GH-195 undersells its own risk.** It is scoped as "the rejection ships as a quiet
  WARNING". The review round showed the underlying failure mode can be whole-table, not
  a few obscured values. Widen it.
- The chart-loss gap below is still unfiled.
- One nit left on `orchestrator.py`: `_is_trusted_native_without_ocr` still re-derives
  "was this refused?" from `text_is_rotated and has_tables` instead of reading
  `PageAssessment.native_table_lane_refused`. Provably equivalent at that call site
  today (it is gated behind `is_born_digital`), so not a live bug — but it is the same
  pattern GH-147 A2 removed from the audit predicate.

## Wave 3 — CLOSED 2026-08-14 — one built, three RETARGETED

The headline result is not the code. **Three of four tickets were sent back rather than
built**, because their premises no longer held. A premise check was added to the dispatch
harness for this wave — it caught all three on its first run.

| Ticket | Outcome |
|---|---|
| GH-151 B1 | **BUILT** — PR #200, branch `feat/151-b1-structural-gate` (`9dbc5fe` + `2455279`) |
| GH-152 A1 | **RETARGETED** — detector-only; integration moves to a recut A2 |
| GH-150 B2 | **RETARGETED** — fixtures + pinned xfail; the real fix becomes new ticket C1 |
| GH-147 B1 | **RETARGETED** — the metric was invalidated by GH-147 A2's own fix |

Each retarget is written into its own folder's `TICKETS.md`. In one line each:

- **GH-152 A1** — the ticket blamed the rowizer. Measurement showed the page-wide
  text-strategy `find_tables` at `reconstruct.py:141` merges the tables *first* and
  suppresses the fallback, so wiring only the rowizer is a measured end-to-end no-op. A1 is
  now a detector; A2 is recut to consume it at **both** rungs.
- **GH-150 B2** — the defect still reproduces (both PDFs copied, hashed, measured through the
  installed package), but the production fix is a merge inside `born_digital.py`, which B2
  does not own. B2 becomes fixtures plus a strict xfail; **new TICKET-C1** owns the fix and is
  blocked on PR #200 for that file.
- **GH-147 B1** — the ticket's own metric is dead. A2 sets `native_text = raw_text.strip()`
  on refused pages (`born_digital.py:915-931`), so word recall there is **~1.0 by
  construction** and the 20/40 figure cannot survive a correct fix. Retargeted onto refusal
  rate plus a structural witness.

### What this says about the remaining waves

Three of four is not bad luck. This plan folder was authored against a codebase that has since
moved twice (waves 1 and 2). **Waves 4 and 5 were written the same day and should be assumed
stale until checked.** Treat a retarget as the expected outcome of a premise check, not as a
failure — and check the premise before staffing, not after.

### Ownership amendments

- `src/socr/tables/structure_check.py` — released from wave-1-only; GH-151 B1 extends it.
- `src/socr/core/manifest.py` — **added**, claimed by GH-151 B1. A flag the manifest does not
  read re-stamps `audit_passed=True` (the PP-7-R1 bug shape), which would make the gate inert.
- `src/socr/core/born_digital.py` — contended, and now the **single bottleneck for all remaining
  work**. Four claimants, in order: GH-151 B1 holds it (PR #200); **GH-150 C1 needs it next** and
  cannot dispatch until #200 merges; GH-152 A2 may also need it if left-to-right reading order
  stays in its `Done when` — `born_digital.py:1201` re-sorts by `y0` alone, so ordering cannot be
  delivered from `reconstruct.py`; and **fake-native B1** (`docs/plans/fake-native-pages/`, merged
  in #209) queues fourth. Nothing about that queue is parallelisable — resolving #200's direction
  is what unsticks all four.

### Process notes worth keeping

- **Measure at the caller.** Any content-loss claim must go through
  `BornDigitalDetector().extract_structured` or `process()`. Wave 2's #192 review produced a
  false blocking finding — "100% table loss" — by measuring one rung in isolation; end to end,
  the loss was zero. This is now written into the dispatch harness.
- **A green suite is not a guard.** GH-151 B1 shipped with five of six wiring tests missing and
  undisclosed; you could delete the guard enforcing the `--native-only` ruling and the whole
  suite still passed. The follow-up round then found the *replacement* test could not
  distinguish the correct fix from a simpler wrong one. Require proof that each new test fails
  when the production line is reverted.
- **The corpus lives in two places and neither is complete.** iCloud has 407 PDFs with 45
  evicted to 0-byte placeholders; ProtonDrive has 277, essentially all real — and covers all 45
  gaps. The union is complete. Google Drive holds a third archive copy that must not be read
  from (kept quit by design, streams rather than stores). Copy to `/tmp` and verify byte size;
  never open in place.
- **Size a vivid failure before letting it block work.** The fake-native plan was opened as an
  interrupt on the belief that scanned-with-bad-OCR pages were contaminating the table-defect
  numbers, and its STATUS said nothing on #200 should be decided until it landed. Measurement
  (#209) put the population at 2.4% of pages and the overlap with TR-3 at 6/68 — it blocked
  nothing. The read-only measurement was cheap and worth running; the *blocking claim* attached
  to it was a generalisation from one dramatic page. Measure first, then decide what it stops.

## Next action

**Dispatch wave 3b — the three retargeted tickets**, now that each carries a scope grounded in
measurement: GH-152 A1 (`tables/reconstruct.py`), GH-150 B2 (`tests/` + `fixtures/` + `logs/`),
GH-147 B1 (`tests/` + `fixtures/` + `logs/`). Write sets are disjoint.

**Then wave 4, gated on PR #200 merging:** GH-150 C1, GH-152 A2 and fake-native B1 all want
`born_digital.py`, so they serialize behind #200 and behind each other.

**The standing blocker is #200's direction, not its code.** It is rebuilt as an escalation signal
(`docs/log/2026-08-14_gh151-b1-escalation-decision.md`); what remains undecided is what the signal
should *do* — Fable's proposal is a stop-condition veto on the ladder with a fail-closed marker at
the top rung. Until that is settled, four tickets sit still.


