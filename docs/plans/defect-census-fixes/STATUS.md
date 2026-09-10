# STATUS — defect-census fixes

Last updated: 2026-09-11

## Stage
Waves 1-9 merged: A1a #630, C1 #631, D1 #632, A1b #640, A1c #644, A2 #647, B1 #651, E1 #654,
E2 #657. D3 measured 2026-09-07 (`docs/log/2026-09-07_D3-fed-table-lane-remeasure.md`); it
surfaced two bugs, both merged: #658 (PR #666), #659 (PR #668). Separately, #189 (mixed
chart+table page) merged as PR #672. F1a (#625, ditto marks) merged as PR #686. F2 (#624,
`&nbsp;` hierarchy) landed across PR #689 and PR #691 (#624b's font-evidence merge); #624 and
#601 are both closed, but the decode does not survive to the shipped `.md` — that remainder is
tracked as the still-open #688. #643 (recurring column lanes) remains parked: PR #663 is open
as a draft. D2 (route cost measurement) and F1b (derived-cell provenance) are still TODO.

Overnight 2026-09-10 (Codex-reviewed, one Astra round per finding): #592 round 2 merged as PR #704
(bounded immediate-band adoption; `Refs`, remainder #706) and its continuation as PR #710 (closes
#706); #696 merged as PR #699 (flatten spanning headers, nine rounds; a same-column lower table or a
footnote carrying the leaf row's distinct tokens freezes the fold as a no-op, never a deletion); BoE
third-institution census merged as PR #705 (#703 filed). #695 (#652/#649) merged after sixteen rounds:
model-prose corroboration deleted by ruling (no band-gap geometry proves a block's role), a scan's own
native layer ships flagged, and the review viewer now honours CommonMark escapes. Filed: #707 (measure native-fallback
fidelity on the six Fed minutes), #709 (#704's immediate band hoists a staff row above a beside-heading).

Second cycle (05:40–08:00): #703 merged as PR #715 — A2's row-shortfall term (b) now abstains unless the native
page shows recurring numeric token lanes (own-occupancy seeds, same-band co-occurrence, rejoin within one rounding
bin); the BoE 2018 text-table page clears `table_truncated` but still ships the marker because its ladder-accepted
candidate lost the page judge to a timeout (#713); A1b's twin has the same blind spot (#714). #688 merged as PR
#716 — a label cell's leading indentation (`&nbsp;` runs) is stripped once at candidate ingestion and at region
extraction before identities, so the shipped `.md` carries plain labels; interior entities are never decoded.
#709 accepted as PR #720 (beside-heading unit: heading, adopted pair and run emit contiguously at the
heading's block key). Filed: #713, #714, #717 (conftest neuters `shutil.which`).

Third cycle and evening (2026-09-10 08:00–24:00): desk leftovers merged — #708 (PR #722, CI pin for the
unequal-block search), #712 (PR #723, the withholding lane now escapes native lines), #718+#719 (PR #724, which
also fixed a #716 regression: canonicalised labels printed twice as loose prose). #713 merged as PR #725: a typed
page-judge outcome and an acceptance credential let a ladder-accepted candidate ship flagged after a judge
timeout (four Codex rounds; a fresh BoE run was never performed, so the census page is still unmeasured). #714
as PR #726: A1b declines the numeric-corroboration route on a text table instead of refusing it — the first
version admitted a fabricated sentence on two numeric rows; text tables now need a page-judge acceptance or the
#713 credential, so the cached BoE page still ships its marker, under an honest reason.

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
| F1a | ditto text (#625) | DONE (#686 merged) | A2 | 7 |
| F1b | derived-cell provenance | TODO | F1a | 8 |
| F2 | nbsp hierarchy (#624) | DONE, remainder open (#689, #691 merged; decode-to-shipped-md gap tracked as #688) | F1b | 9 |

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
D2 (route cost measurement) and F1b (derived-cell provenance) are the only remaining tickets
on this board. Open issues still needing work: #643 (parked; PR #663 draft), #707 (native fallback fidelity measurement, the
only evidence behind #695's retention claim), #717 (test hygiene). Next measurement: a fresh run of the BoE 2018
excerpt with a live judge, to see whether #713's credential is minted and the text-table page ships.
#688/#703/#709/#713/#714 closed by PRs #716/#715/#720/#725/#726. #649/#652 closed by PR #695.
