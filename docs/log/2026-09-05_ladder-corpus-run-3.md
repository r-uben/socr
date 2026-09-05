# Ladder corpus run 3 — after verifier-independence (A2, C2a, C2b)

2026-09-05, 16:51–18:05 local. socr `main@d82a0f2` (#621, VI-C2b merged), run under B2's
discipline: dedicated checkout `~/repos/.worktrees/socr-vi-C3` detached at `d82a0f2`,
`PYTHONPATH=<checkout>/src` on every process, resolved package path verified, intended
`socr_source_digest` `720a5822…` recorded before launch. **Gate: 20/20 sidecars carry the
intended digest, 0 mismatches.** Same manifest, same 8 papers, same 20 pages as runs 1–2,
sequential. Outputs in `~/Data/socr/vi-C3-2026-09-05/`. Content-free: identifiers, counts,
dispositions, seconds. Exit 1 on 7 of 8 papers, as in every run — surfaced at document level by
design. **This is a report, not the gate**: the gates were A2's and C2b's frozen replays.

## Headline vs run 2

| | run 2 (`f434019`, 3 digests) | **run 3 (`d82a0f2`, 1 digest)** |
|---|---|---|
| pages | 20 | 20 |
| ladder terminals | 7 ACCEPTED · 0 WITHHELD · 11 UNVERIFIED (18) | **8 ACCEPTED · 1 WITHHELD · 9 UNVERIFIED** (18) |
| tables adjudicated | 7 → 1 lifted / 6 held | **4 → 0 lifted / 4 held (3 addressed, 11 abstained of 14 items)** |
| page status | 4 SUCCESS · 11 WARNING · 5 ERROR | **7 SUCCESS · 8 WARNING · 5 ERROR** |
| cloud cost | $0.0020 | **$0.0010** |
| wall-clock | 2 h 40 min (confounded) | **2 h 17 min = 6.8 min/page** (B2: 7.0) |

## What moved, and why

- **doc02 p3 / p4: UNVERIFIED (7 held items) → ACCEPTED, no contradiction.** A2's fix: the
  seven `2Y/3Y/5Y/10Y` stubs sat 0.001–0.002 pt outside the witness region and top-left
  containment dropped them; centroid membership keeps them. The only change in the plan that
  turns a held table into an accepted one, and it did so live in under 4 min per page.
- **doc01 p2: ACCEPTED-by-padding-accident → ACCEPTED with no adjudication.** The same clip
  fix removed the contradiction that run 2 lifted by luck (A1b).
- **Every one of the 14 remaining items matched the committed prediction artifact
  item-for-item on the live pipeline**: doc03 ×2 abstained (column test; native chain break at
  row 3), doc04 ×1 abstained (no origin), doc05 3 addressed on cells (4,4,4) (6,6,6) (8,8,8) +
  3 abstained (native chain break at row 13), doc07 ×5 abstained (no column edge).
- **The three addressed items were transcribed on their geometry cells (adjudication 4.4 s)
  and none was disproved.** The re-read agreed with neither side: model `$\Delta \text{ Slope
  (3m)}$`, native `∆Slope (3m)` — the comparison layer does not translate `\Delta` to `∆`.
  That is **#585** verbatim from the run-2 log. The address is now independent of both lanes;
  the remaining reason a correct re-read fails to lift is a normalisation gap with an issue
  number, not a geometry problem.
- **1 WITHHELD (doc00 p2)** where run 2 had 0: readers rejected and the blind cell disagreed —
  the #560 terminal working as designed on a page that also carries an accepted table.

## Cause taxonomy for every held table

| table | items | cause | owner |
|---|---|---|---|
| doc03 p1 | 2 | lane_mismatch (column test) · native row split above (chain) | class (d); #600 family |
| doc04 p3 | 1 | abstained — no origin (scanned, no rules) | #603 |
| doc05 p1 | 3 addressed, not disproved | presentation — `\Delta` ≠ `∆` in `tokens_agree` | **#585** |
| doc05 p1 | 3 | abstained — native chain breaks at row 13 (extra native row) | #600 family |
| doc07 p1 | 5 | abstained — no column edge (ragged label x0s 72 / 74 / 78) | C1 §(f) design item |

No held table is attributed to `model_wrong`; failure to disprove never became a conviction.

## Latency (B1 timings, exclusive seconds)

| stage | run 3 | B2 baseline |
|---|---|---|
| route | 3153 | 3186 |
| extract | 2857 | 3185 |
| ladder | 1403 | 1100 |
| tables | 644 | 599 |
| equations | 130 | 274 |
| adjudication | **4** | 57 |
| total | 8192 (6.8 min/page) | 8402 (7.0) |

The geometry path costs nothing measurable; adjudication fell because only 3 cells were
transcribed. Route overhead still equals extraction — B2's finding stands.

## Plan outcome

The verifier-independence plan closes with: the native lane witnesses doc02 correctly (8 of 22
disputed items gone at the source); the recovery crop is addressed by page geometry through an
ordinal chain that neither lane can steer, with every uncertainty surfaced as an abstention
carrying its reason; and the next blocker to an actual lift is named (#585). Coverage today is
3 addressable of 14 remaining items — small and sound.
