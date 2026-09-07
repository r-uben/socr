# STATUS — defect-census fixes

Last updated: 2026-09-08

## Stage
Waves 1-7 merged: A1a #630, C1 #631, D1 #632, A1b #640, A1c #644, A2 #647, B1 #651, E1 #654,
E2 #657. D3 measured 2026-09-07 (`docs/log/2026-09-07_D3-fed-table-lane-remeasure.md`); it
surfaced two bugs, both filed and worked: #658 merged (#666); #659 fix in review (PR #668).
D2, F1a, F1b, F2 still TODO.

## Base state (clean before tickets)
- `main@eb14c82`; census + plan on branch `docs/fed-ecb-census` (7015f46, d00fb11, 86db834, 7fea35a, +panel revision).
- Pinned measurement checkout `~/repos/.worktrees/socr-census` (detached at `eb14c82`).
- Fixtures under `~/Data/socr/census-ecb-2026-09-06/`, `~/Data/socr/census-591-recheck/`.

## Ticket board
| Ticket | Stream | Status | depends-on | Wave |
|--------|--------|--------|------------|------|
| A1a | corroboration fn | DONE (#630 merged 2026-09-06) | — | 1 |
| C1 | native geometry (#592) | DONE (#631 merged 2026-09-06) | — | 1 |
| D1 | throughput | DONE (#632 merged 2026-09-06) | — | 1 |
| A1b | selection | DONE (#640 merged 2026-09-07; #639 filed) | A1a | 2 |
| A1c | surfacing + resume | DONE (#644 merged 2026-09-07; live: report p1-p3 99-100% flagged) | A1b | 3 |
| A2 | truncation guard | DONE (#647 merged 2026-09-07) | A1c | 4 |
| B1 | marker scope (#591) | DONE (#651 merged 2026-09-07; #591 open: #649, #650) | A2 | 5 |
| D3 | Fed re-measure | DONE (measured 2026-09-07; surfaced #658 merged #666, #659 PR #668 in review) | A2 | 5 |
| E1 | scan ≠ chart (#511) | DONE (#654 merged 2026-09-07; #653) | B1 | 6 |
| E2 | table_not_scorable scope | DONE (#657 merged 2026-09-07) | E1 | 7 |
| D2 | route cost (measure) | TODO | E1 | 7 |
| F1a | ditto text (#625) | TODO | A2 | 7 |
| F1b | derived-cell provenance | TODO | F1a | 8 |
| F2 | nbsp hierarchy (#624) | TODO | F1b | 9 |

## Dispatch waves
- Wave 1: A1a (`tables/row_corroboration.py`), C1 (`born_digital.py`), D1 (`providers.py`) — disjoint.
- Waves 2–5 serial on `manifest.py`/`orchestrator.py`: A1b → A1c → A2 → B1 (D3 is docs-only, runs beside B1).
- Wave 6: E1.
- Wave 7: E2, D2 (depends-on E1), F1a.
- Waves 8–9: F1b, F2.

## Known gaps
- Fed page sum (issue #636 item 5): lead-in total vs the 2,690 + 2,765 split is unreconciled;
  still open, needs measurement.
- Six ECB fixture pages (issue #636 item 7): not yet named for `EXTRA_NUMBERS_MAX_SHARE` /
  A1c's Done-when; still open, needs measurement.

## Next action
F1a/F1b/F2 and D2; #643 parked (needs font-size/bottom-rule evidence); #189 and #659 PRs in
final review.
