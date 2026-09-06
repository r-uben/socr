# STATUS — defect-census fixes

Last updated: 2026-09-06

## Stage
Plan drafted from the two-institution census and revised after the advisory panel (GPT,
DeepSeek, Kimi; Gemini transport-failed, Grok quota-exhausted). Awaiting owner approval;
nothing dispatched.

## Base state (clean before tickets)
- `main@eb14c82`; census + plan on branch `docs/fed-ecb-census` (7015f46, d00fb11, 86db834, 7fea35a, +panel revision).
- Pinned measurement checkout `~/repos/.worktrees/socr-census` (detached at `eb14c82`).
- Fixtures under `~/Data/socr/census-ecb-2026-09-06/`, `~/Data/socr/census-591-recheck/`.

## Ticket board
| Ticket | Stream | Status | depends-on | Wave |
|--------|--------|--------|------------|------|
| A1a | corroboration fn | DONE-LOCAL (92133fa; reviewer + Astra; edge-row blind spot -> A1b; unpushed) | — | 1 |
| C1 | native geometry (#592) | DONE-LOCAL (4ce7ecc; reviewer + 2 Astra rounds; residual pinned; unpushed) | — | 1 |
| D1 | throughput | DONE-LOCAL (0fa11fc, reviewer ACCEPT, live on agentic path; Astra skipped: Codex quota <10%; unpushed) | — | 1 |
| A1b | selection | TODO | A1a | 2 |
| A1c | surfacing + resume | TODO | A1b | 3 |
| A2 | truncation guard | TODO | A1c | 4 |
| B1 | marker scope (#591) | TODO | A2 | 5 |
| D3 | Fed re-measure | TODO | A2 | 5 |
| D2 | route cost (measure) | TODO | B1 | 6 |
| E1 | scan ≠ chart (#511) | TODO | B1 | 6 |
| E2 | table_not_scorable scope | TODO | E1 | 7 |
| F1a | ditto text (#625) | TODO | A2 | 7 |
| F1b | derived-cell provenance | TODO | F1a | 8 |
| F2 | nbsp hierarchy (#624) | TODO | F1b | 9 |

## Dispatch waves
- Wave 1: A1a (`tables/row_corroboration.py`), C1 (`born_digital.py`), D1 (`providers.py`) — disjoint.
- Waves 2–5 serial on `manifest.py`/`orchestrator.py`: A1b → A1c → A2 → B1 (D3 is docs-only, runs beside B1).
- Wave 6: D2 and E1 both touch orchestrator → E1 first, then D2.
- Waves 7–9: E2, F1a, F1b, F2.

## Next action
Wave 1 complete locally 2026-09-06: A1a 92133fa (socr-a1a), C1 4ce7ecc (socr-c1), D1 0fa11fc (socr-d1); each reviewed, A1a/C1 Astra-checked, D1 Astra skipped (Codex quota). Pushed 2026-09-06; PRs #629 (docs), #630 (A1a), #631 (C1), #632 (D1). Next: confirm a CI run exists per head, wait green, merge; then wave 2 A1b (depends A1a merged) on a fresh branch off main.
branches off main (`feat/NN-row-corroboration`, `fix/592-aligned-runs`, `fix/NN-nougat-ladder`).
