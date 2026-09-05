# STATUS — verifier-independence

Last updated: 2026-09-05 (C2b merged, C3 running)

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
| A1b | native reference | DONE | A1 | 2 (claude) |
| B2  | latency | DONE | B1 | 2 (claude) |
| A2  | native reference | DONE | A1b | 3 (curia, 2/3 — doc04 → #603) |
| C1  | verifier | DONE | A1b | 3 |
| C2a | verifier | DONE | C1 | 4 (#616, 13 review rounds) |
| A1c | native reference | DONE | — | 4 (#612, #595 closed) |
| C2b | verifier | DONE | C2a, A2, B1 | 5 (#621; single seat — grok 402) |
| C3  | verifier | WIP | C2b, B2 | 6 (claude; running) |

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

## Waves 2–3 closed (2026-09-05)

- **A1b** (autopsy): 8/8 target crops truncated by `native_bbox`; three native defect shapes; the
  corpus's one lift was padding luck. `labels.json` + `a1b-crops/` frozen with the corpus.
- **B2** (baseline): 7.0 min/page on one digest; `route` overhead 38 % = `extract` 38 %; ladder 13 %;
  no stage > 50 %. `logs/2026-09-05_B2-latency.md`, `_B2-timings.tsv`.
- **C1** (design, 4 Codex rounds): no table on the corpus has per-row or vertical rules — the
  rules-only stream C addressed 0/22. Ruling: rules-else-text-line bands with the same
  origin-relative ordinal chain on native, model and band; abstain otherwise. **3/22 addressable
  today; 3 of 14 remaining after A2.** Two negative controls mandated for C2b. Surfaced #600
  (`round(y0)` row split), #601 (model spacer row breaks the chain).
- **A2** (#602, curia codex vs grok, 4 rounds): root cause was NOT the rowizer — seven stubs sit
  0.001–0.002 pt outside the witness region and top-left containment dropped them. Fix: word
  centroid in the closed region box. Subscript fold for doc04 abstained (no geometry separates a
  subscript from a `(a)` note) → gate revised to 2/3, N=7, doc04 → #603. Codex's build projected
  the missing stub from a donor row (native inventing text) — disqualified.

## Wave 4 closed (2026-09-05)

- **A1c** (#612): replay rows are UNREPLAYABLE on witness failure and UNCHECKED without
  per-row bind evidence (frozen candidate index → `row_binding`, no label similarity). Four
  Codex rounds.
- **C2a** (#616): `row_bands` / `label_column_edge` / `ordinal_origin` + `BindingResult`
  surface. C1 §(a)'s rule-merge and band rules were both wrong on implementation; corrected
  in the note. Every non-`separate` verdict is flagged; C2b abstains across flagged prefixes
  (#614). Corpus: origins as measured; bands 13/23/23/12/13/17/33; ambiguity doc01 4,
  doc04 2, others 0.

## Wave 5 closed (2026-09-05)

- **C2b** merged (#621, `d82a0f2`): the recovery crop is addressed by page geometry through the
  ordinal chain (native and model chains from the header origin, i = j = b), the #614 prefix rule,
  and the column test; row-label items only; abstain is a separate outcome that never lifts;
  lift signatures carry address + method provenance. Frozen gate PASS: 14 remaining items, 3
  addressed (doc05 (4,4,4) (6,6,6) (8,8,8)), 11 abstained, A2 clears preserved. Curia degraded to
  one seat (grok: 402 Grok Build balance exhausted); Sonnet ACCEPT ×2, Codex brain seat
  APPROVE-FOR-PR after two holes past the first ACCEPT (numeric cells addressed; legacy lifts
  bypassing geometry).

## Next action

**C3 running** since 16:51 on `~/repos/.worktrees/socr-vi-C3` pinned at `d82a0f2`, intended
`socr_source_digest` `720a5822…`, outputs `~/Data/socr/vi-C3-2026-09-05/`. On completion: verify all
20 sidecars carry the intended digest; tabulate ACCEPTED / WITHHELD / UNVERIFIED, lifted / held /
abstained, cost, per-stage minutes; compare to run 2 line by line with the full cause taxonomy;
`docs/log/2026-09-05_ladder-corpus-run-3.md`. The hard gate was A2's frozen replay; this is the
report.
