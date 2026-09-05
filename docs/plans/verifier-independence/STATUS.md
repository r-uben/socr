# STATUS — verifier-independence

Last updated: 2026-09-05 (wave 1 merged, wave 2 started)

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
| B1  | latency | DONE | — | 1 |
| A1b | native reference | WIP | A1 | 2 (claude) |
| B2  | latency | WIP | B1 | 2 (claude) |
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

## Wave 1 closed (2026-09-05)

- **A1** merged (#593, `a49e0b4`): replay harness. Candidate identity is provenance-only
  (two Codex brain-seat rounds). Findings: persisted adjudication records carry **no
  `native_bbox`**; one frozen page's `winning_output.text` is the D3 marker.
- **B1** merged (#594, `4c1b284`): `/curia` codex vs grok + Sonnet seat; grok's build. **Recorded
  decision:** `total` is the independent page wall-clock and `total − Σ exclusive` is reported as
  unattributed — Done-when (a)'s literal "keys sum to total" is superseded (Astra: absorbing the
  remainder into `route` would conceal missing instrumentation). Page lifecycle in `try/finally`.
  Full suite 4108 passed.
- **Second corpus fixture:** `~/Data/socr/fed-sample-2026-09-05/in/fed-1989-11-14-minutes.pdf`
  (5 pages, $0.0002, 8 min). p3: table rejected at every rung for lack of native evidence, then
  the fail-closed marker discarded the page's prose too (#591). p1: two-column attendee list
  column-interleaved and shipped SUCCESS (#592). A1b includes p3; C1 must consider p1.
- **A1b baseline protocol (Astra):** a fresh crop is not what the historical adjudicator saw.
  A1b records source digest + rendered crops as the baseline artefact *before* A2, and labels
  reconstructed-baseline geometry separately from current geometry.

## Next action

Wave 2, both Claude inline: **B2** fresh timed baseline on the frozen 20 pages under B2's run
discipline (pinned checkout `mainc1b284`, explicit `PYTHONPATH`, one digest) ∥ **A1b** autopsy
of the 8 failed-disproof crops + census of all 12 class-(c) items + Fed p3.
