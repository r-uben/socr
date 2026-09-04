# STATUS — verifier-independence

Last updated: 2026-09-04

## Stage

Plan drafted 2026-09-04 from a four-round Claude ↔ Codex conversation in the socr `brain`
tab, then attacked by a three-seat panel (Codex: fidelity; agy: coverage/size; grok:
gating/CI/resume) — 24 findings, all verified against the tree and folded in. Biggest
corrections: A2 retargeted from `native_rows.py` to `binding.py` (the binder builds its own
rows); `bands_from_rules` yields table boxes not row bands (C2a builds the helper); `lifted`
replaced as A2's counter; C3's live quota replaced by A2's frozen-replay gate; `tests/` is
flat. Nothing dispatched.

## Base state (clean before tickets)

- `main` @ `b7323f7` (#587 merged). Plan branch `feat/verifier-independence-plan`.
- Frozen evidence: `~/Data/socr/ladder-run2-2026-09-04/` (153 files, `SHA256SUMS`),
  copied from the run-2 scratchpad. Verify with `shasum -a 256 -c SHA256SUMS`.
- Measured baseline (run 2, `main@f434019`): 18 tables to ladder → 7 ACCEPTED /
  11 UNVERIFIED; 7 adjudicated → 1 lifted / 6 held; 3 held by native row-label defects;
  $0.0020 cloud; 8.0 min/page, no stage timings exist.
- Full suite last known green on `main`; ~1070 tests.

## Ticket board

| Ticket | Stream | Status | depends-on | Wave |
|--------|--------|--------|------------|------|
| A1  | native reference | TODO | — | 1 |
| B1  | latency | TODO | — | 1 |
| A1b | native reference | TODO | A1 | 2 (claude) |
| B2  | latency | TODO | B1 | 2 (claude) |
| A2  | native reference | TODO | A1b | 3 |
| C1  | verifier | TODO | A1b | 3 |
| C2a | verifier | TODO | C1 | 4 |
| C2b | verifier | TODO | C2a, A2, B1 | 5 |
| C3  | verifier | TODO | C2b, B2 | 6 (claude) |

## Dispatch waves

- Wave 1: A1 (`benchmark/replay_binding.py`, import-only from orchestrator) ∥ B1
  (`orchestrator.py`, `state.py`, `manifest.py`, `cli.py`) — disjoint.
- Wave 2: A1b (autopsy, Claude inline) ∥ B2 (breakdown run, Claude inline).
- Wave 3: A2 (`binding.py` + tests) ∥ C1 (design log) — disjoint.
- Wave 4: C2a (`locate.py` only).
- Wave 5: C2b (`adjudication.py`, `binding.py`, `orchestrator.py` two hunks).
- Wave 6: C3 (corpus re-run, Claude inline).

## Next action

Commit the plan folder on `feat/verifier-independence-plan`, open the PR, then `/plan next`
→ dispatch A1 and B1 on separate branches (`feat/vi-A1-replay-binding`,
`feat/vi-B1-stage-timings`).
