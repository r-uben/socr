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

## Wave 0 — merge gate (not a ticket)

Four downstream tickets are blocked on two open PRs, both already implemented:

- **PR #148** (`fix/145-region-overlap-drops-prose`) — adds `dominant_text_direction()`.
  Until it merges, GH-147 A1 either duplicates it or is written twice.
- **PR #149** (`fix/146-data-row-promoted-to-header`) — the `reconstruct.py` header work.
  GH-144 A2 and GH-152 A1 rewrite the same file and must not branch from before it.

Wave 1 does not depend on either and may start immediately. Waves 2+ may not.

## File ownership (the reason waves exist)

One file, one wave. Collisions are what the per-folder schedules could not see.

| File | Claimed by | Serialized as |
|---|---|---|
| `src/socr/tables/reconstruct.py` | GH-144 A2 · GH-152 A1 · GH-152 A2 | W2 → W3 → W4 |
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

### Wave 2 — 3 parallel

| Ticket | Writes | Depends on |
|---|---|---|
| GH-144 A2 | `tables/reconstruct.py` | GH-144 A1 · PR #149 |
| GH-147 A2 | `core/born_digital.py`, `pipeline/orchestrator.py` | GH-147 A1 |
| GH-150 A2 | `tests/test_chart_detection_gh150.py` | GH-150 A1 |

### Wave 3 — 4 parallel

| Ticket | Writes | Depends on |
|---|---|---|
| GH-152 A1 | `tables/reconstruct.py` | GH-144 A2 |
| GH-151 B1 | `core/born_digital.py`, `core/state.py`, `pipeline/orchestrator.py` | GH-151 A1+A2 · GH-147 A2 |
| GH-150 B2 | `tests/test_chart_detection_gh150.py` | GH-150 A1+B1 |
| GH-147 B1 | `tests/test_landscape_refusal_gh147.py`, `logs/` | GH-147 A2 |

### Wave 4 — 3 parallel

| Ticket | Writes | Depends on |
|---|---|---|
| GH-152 A2 | `tables/reconstruct.py` | GH-152 A1 |
| GH-144 A3 | `logs/` | GH-144 A2 |
| GH-151 B2 | GH-49 issue comment | GH-151 B1 |

### Wave 5

| Ticket | Writes | Depends on |
|---|---|---|
| GH-152 B1 | `tests/test_side_by_side_tables_gh152.py`, `logs/` | GH-152 A2 |

## Critical path

`GH-144 A1 → GH-144 A2 → GH-152 A1 → GH-152 A2 → GH-152 B1` — five deep, all on
`reconstruct.py`. Nothing else in the program is longer, so this chain sets the schedule;
every other lane has slack. Staff it first and keep one agent on it.

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

## Next action

Wave 0: merge PR #148 and PR #149. Then dispatch wave 1 as six parallel tickets.
