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
| `src/socr/tables/structure_check.py` (new) | GH-151 A1 · GH-151 B1 | W1 → W3 |
| `src/socr/core/state.py` | GH-151 B1 | W3 only |
| `src/socr/core/manifest.py` | GH-151 B1 | W3 only |

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
| GH-151 B1 | `core/born_digital.py`, `core/state.py`, `pipeline/orchestrator.py`, `tables/structure_check.py`, `core/manifest.py` | GH-151 A1+A2 · GH-147 A2 |
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

## Next action

**Dispatch wave 3 — three of its four tickets are ready:** GH-152 A1
(`tables/reconstruct.py`), GH-150 B2 (`tests/test_chart_detection_gh150.py`), GH-147 B1
(`tests/test_landscape_refusal_gh147.py`, `logs/`). Write sets are disjoint.

**GH-151 B1 is NOT dispatchable and must not be included.** It depends on GH-151 A2,
which is parked as draft PR #184 and was deliberately excluded from wave 1: its column
binding is modal consensus, which cannot distinguish "deviant because malformed" from
"deviant because genuinely different". B1 needs a design saying what column binding
should key on instead — a `socr-designer` pass, not an implementer.

