# STATUS — verifier-independence

Last updated: 2026-09-05

## Stage

Plan drafted 2026-09-04 from a four-round Claude ↔ Codex conversation in the socr `brain`
tab, then attacked by a three-seat panel (Codex: fidelity; agy: coverage/size; grok:
gating/CI/resume) — 24 findings, all verified against the tree and folded in. A second
revise panel (Codex / Claude / Cursor / Bugbot; Gemini silent) on 2026-09-05 folded
additional findings (base SHA, A1 replay gate, A2 denominator, B2 baseline framing, C2b
file split). Biggest corrections: A2 retargeted from `native_rows.py` to `binding.py` (the
binder builds its own rows); `bands_from_rules` yields table boxes not row bands (C2a builds
the helper); `lifted` replaced as A2's counter; C3's live quota replaced by A2's frozen-replay
gate; `tests/` is flat. Nothing dispatched.

## Base state (clean before tickets)

- `main` @ `92c1527` (#588 merged; docs-only digest-correction atop #587). Plan branch
  `feat/verifier-independence-plan`.
- Frozen evidence: `~/Data/socr/ladder-run2-2026-09-04/` (153 files, `SHA256SUMS`),
  copied from the run-2 scratchpad. Verify with `shasum -a 256 -c SHA256SUMS`.
- Measured baseline (run 2, `main@f434019`): 18 tables to ladder → 7 ACCEPTED /
  11 UNVERIFIED; 7 adjudicated → 1 lifted / 6 held; 3 held by native row-label defects;
  $0.0020 cloud; **8.0 min/page is confounded** — run-2 wall-clock spanned **three**
  `socr_source_digest` values (venv repoint mid-run); no stage timings exist (B1/B2 establish
  a fresh baseline under one digest).
- Full suite last known green on `main`; ~1070 tests.

## Ticket board

| Ticket | Stream | Status | depends-on | Wave |
|--------|--------|--------|------------|------|
| A1  | native reference | DONE | — | 1 |
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

Wave 1 dispatch: A1 on `feat/vi-A1-replay-binding` ∥ B1 on `feat/vi-B1-stage-timings`.

A1 done 2026-09-05 on `feat/vi-A1-replay-binding`
(`/Users/rubenffuertes/repos/.worktrees/socr-vi-A1`), see
`docs/log/2026-09-05_vi-a1-replay-binding.md`. Corpus replay: 7/7 rows,
exact multiset match on the unchanged tree (one row needed a D3
fail-closed-marker cache fallback, folded into the harness). A1b can
dispatch once this branch merges.
