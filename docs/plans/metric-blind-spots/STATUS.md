# STATUS — metric blind spots (socr #123)

Last updated: 2026-08-01

## Stage

Wave 1 done on `feat/123-metric-blind-spots`. B1 landed (`402395c`) and was
reviewed (ACCEPT-WITH-FOLLOWUP, `docs/log/2026-08-01_TICKET-B1-review.md`). A1
landed (`docs/log/2026-08-01_TICKET-A1.md`) and surfaced a second, untracked
defect (wrapped-label row identity) — now TICKET-B5.

**Wave 2's first attempt was reverted.** B2 as written made the markdown side
positional while the ground truth stayed compacted; on a leading-gap sparse table a
faithful transcription then scored 80% while one that dropped the column entirely
scored 100% — the metric rewarding structure-destroying OCR inside a production accept
rule. The implementation followed the ticket exactly; the ticket's seam was wrong. B2
and the former B3 are now **one ticket**, and two new invariants
(`test_a_perfect_transcription_scores_100`,
`test_dropping_a_column_never_beats_keeping_the_gap`) guard that regression. See
`docs/log/2026-08-01_TICKET-B2-review.md`.

Nothing here changes OCR behaviour. It changes what socr can *measure* — which
matters because `escalation_decision` uses the metric as a production accept rule.

## Base state (clean before tickets)

- `r-uben/socr`, `main` at `18b3b64`, clean, no open PRs
- full suite green at base: **1365 collected** (1364 passed, 1 xfailed). The figure
  originally recorded here — 1363 passed, 1 xfailed — was **off by one**; see
  "Known landmines".
- lint gate: `uvx ruff@0.16.0 format --check .` clean (235 files at base, 239 after B1)
  (**not** `~/venvs/socr/bin/ruff` — that version cannot check Markdown; see CLAUDE.md)
- reference artifacts preserved, since neither engine is deterministic:
  - `~/data/fiscal-ballast/_experiments/2026-07-31_gh96-engine-parity/`
  - `~/data/fiscal-ballast/_experiments/2026-08-01_gh96-corpus-rerun/`
- rendered visual comparisons already built (rendered PDF beside each engine):
  - `~/Desktop/socr-qwen-failures/`
  - `~/Desktop/socr-engine-compare/`

## Ticket board

| Ticket | Stream | Status | depends-on | Wave |
|--------|--------|--------|------------|------|
| A1 | grade the metric | DONE | — | 1 |
| B1 | scoring correctness | DONE | — | 1 |
| B2 | scoring correctness | TODO (reverted once) | B1 | 2 |
| B3 | — | CLOSED — merged into B2 | — | — |
| B4 | scoring correctness | TODO | B2 | 3 |
| B5 | scoring correctness | TODO | B4 | 4 |
| C1 | pipeline response | TODO | B4 | 4 |

## Active Agents

| Ticket | Agent | Status |
|--------|-------|--------|
| B1 | socr-implementer | DONE (`402395c`) |
| B1 | socr-reviewer | STOPPED — mutated the shared tree; findings void |
| B1 | orchestrating session | REVIEWED — ACCEPT-WITH-FOLLOWUP |
| A1 | socr-implementer | DONE — `tests/test_metric_corruption_battery.py`, see `docs/log/2026-08-01_TICKET-A1.md` |
| B2 | socr-implementer | REVERTED — `ad649b5`, reverted by `717914d` |
| B2 | orchestrating session | REVIEWED — REJECT; the ticket seam was wrong, B2+B3 merged |

## Dispatch waves

- **Wave 1:** B1 then A1, **sequential — not parallel as originally planned.** The
  editable install resolves `import socr` to the main checkout, so a separate worktree
  would test the *main* tree's source and agents must share one working tree. A1 and B1
  are write-disjoint but not behaviour-disjoint: A1's battery is written over
  `score_page`, whose contract B1 changes. B1 first, A1 against the settled contract.
- **Wave 2:** B2 (now includes the former B3 — markdown positions and ground-truth lanes
  land together, never separately)
- **Wave 3:** B4
- **Wave 4:** B5 and C1 — both depend only on B4, but they are **not** parallel (one
  shared working tree). Either order; B5 touches `native_rows.py`, C1 does not.

B2/B4 both touch `table_exactness.py` or `native_rows.py` and are strictly ordered;
do not parallelise them. In practice **no ticket in this plan can be parallelised with
another** — one shared working tree, one branch.

## Validation that the plan worked

Re-score the two preserved runs before and after. **Some historical escalation
accepts should flip to rejects** once column shifts become visible. If nothing flips,
the change did nothing — that diff is the real acceptance test, not the unit suite.

## Known landmines

- An earlier x-clustering attempt failed because it clustered numerics *to find* the
  label boundary; a numeral inside a label dragged the boundary left. B3 decouples
  them deliberately — do not re-merge those steps.
- Five footnote spellings exist (bare digit, `$^1$`, `^{1}`, `<sup>1,2</sup>`,
  unicode). `normalize_label` already folds all five; do not add a sixth special case.
- Tables with non-numeric data columns (a literal `Met` / `Not Met` column) are
  currently absorbed into the label by the last-non-numeric-word rule. B3's lane
  support test may fix this as a side effect; it is not a goal.
- Run-to-run variance of the local model exceeds most effects being measured. Do not
  validate any change against a single fresh run.
- **STATUS.md's original base test count was wrong by one** (recorded 1363 passed + 1
  xfailed; the true base collected 1365). This burned a review cycle chasing a
  non-existent anomaly. Re-measure before trusting a documented count.
- **Never `git checkout` / `switch` / `stash` / `reset` in this repo while agents are
  running.** B1's reviewer did it twice and spent part of its run reading the *pre-B1*
  tree. Read other revisions with `git show <rev>:<path>`. A throwaway `git clone` is
  also not isolation: the editable install resolves `import socr` to the main checkout,
  so tests in a clone still exercise the main tree's source.

## Findings carried forward

- **B1 finding 1 (HIGH) → folded into C1.** `ceiling_note` reaches no surface; a
  not-scorable page is invisible rather than loudly wrong.
- **B1 finding 2 (MEDIUM) → open.** The ticket's headline benefit (engine aggregates
  stop counting no-table pages as failures) is *not* delivered by B1 alone — there is no
  aggregate helper in `src/`; corpus scoring lives in
  `~/data/fiscal-ballast/_experiments/` and has not been re-scored against B1.
- Findings 3 and 4 (LOW) in `docs/log/2026-08-01_TICKET-B1-review.md`.
- **A1 finding → now TICKET-B5** (own ticket, sequenced after B4). `native_rows_from_page`
  never reconstructs a label split across two visual bands: whichever band carries the
  values keeps only its own text, the other line is dropped, and a perfect transcription
  scores as if the row were missing. Reproduced independently against the parser at 9pt
  and 12pt gaps (label becomes `'debt'`) and at 6pt (bands merge, label scrambles to
  `'Central debt government net'`). Held as a `strict=True` xfail in
  `tests/test_metric_corruption_battery.py`. **Deliberately not folded into B3** — this
  is label-*boundary* work and B3 takes the boundary as given; the landmine list records
  that merging the two is what killed the earlier attempt.

## Note on Stream A

B5 is the first defect in this plan caught **before** it produced published numbers
rather than after. The battery found it on its first run, in a transform the ticket had
listed as *benign*. Weigh that when judging whether Stream A earned its place.

## Next action

Dispatch Wave 2 (B2).
