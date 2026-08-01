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
| B2 | scoring correctness | DONE | B1 | 2 |
| B3 | — | CLOSED — merged into B2 | — | — |
| B4 | — | CLOSED — merged into B2 | — | — |
| B5 | scoring correctness | DONE | B2 | 3 |
| C1 | pipeline response | TODO | B2 | 3 |

## Active Agents

| Ticket | Agent | Status |
|--------|-------|--------|
| B1 | socr-implementer | DONE (`402395c`) |
| B1 | socr-reviewer | STOPPED — mutated the shared tree; findings void |
| B1 | orchestrating session | REVIEWED — ACCEPT-WITH-FOLLOWUP |
| A1 | socr-implementer | DONE — `tests/test_metric_corruption_battery.py`, see `docs/log/2026-08-01_TICKET-A1.md` |
| B2 | socr-implementer (attempt 1) | REVERTED — `ad649b5`, reverted by `717914d` |
| B2 | orchestrating session | REVIEWED — REJECT; the ticket seam was wrong, B2+B3 merged |
| B2 | socr-implementer (attempt 2) | DONE — see `docs/log/2026-08-01_TICKET-B2.md` |
| B2 | socr-implementer (reopen fix) | DONE — see `docs/log/2026-08-01_TICKET-B2-reopen-fix.md` |
| B2 | socr-implementer (Otsu-cut fix) | DONE — see `docs/log/2026-08-01_TICKET-B2-otsu-cut.md` |
| B2 | socr-implementer (paired-columns fix) | DONE — see `docs/log/2026-08-01_TICKET-B2-paired-columns-fix.md` |
| B2 | socr-implementer (widest-row-cap fix) | DONE — see `docs/log/2026-08-01_TICKET-B2-widest-row-cap.md` |
| B5 | socr-implementer | DONE — see `docs/log/2026-08-01_TICKET-B5.md` |

## Dispatch waves

- **Wave 1:** B1 then A1, **sequential — not parallel as originally planned.** The
  editable install resolves `import socr` to the main checkout, so a separate worktree
  would test the *main* tree's source and agents must share one working tree. A1 and B1
  are write-disjoint but not behaviour-disjoint: A1's battery is written over
  `score_page`, whose contract B1 changes. B1 first, A1 against the settled contract.
- **Wave 2:** B2 (now includes the former B3 **and** B4 — lanes, markdown positions, and the
  global monotone map land together, never separately)
- **Wave 3:** B5 and C1 — both depend only on B2, but they are **not** parallel (one shared
  working tree). Either order; B5 touches `native_rows.py`, C1 does not.

B2 is now the whole metric change and is strictly ordered before everything else. In practice
**no ticket in this plan can be parallelised with another** — one shared working tree, one
branch.

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
- **B1 finding 2 (MEDIUM) → CLOSED by the corpus re-score.** The headline benefit is
  confirmed: 17 pages of spurious `0.0%` become not-scorable and the mean over each
  metric's own scorable set rises 44.3% → 49.1%. See "Next action".
- Findings 3 and 4 (LOW) in `docs/log/2026-08-01_TICKET-B1-review.md`.
- **B2 limitation (MEDIUM, accepted) — sparse tables where NO row is complete under-split.**
  The widest-row cap (`8dae02c`) bounds lane count by the widest row's value count. That is
  presented as structural but is really an assumption: **it holds only if at least one row is
  complete.** Reproducer — 4 real columns, 4 rows, each omitting a different one, so the
  widest row has 3 values:

  ```
  Alpha  values=('1.1','1.2','1.3')  lanes=(0, 0, 1)   <- columns 0 and 1 merged
  distinct lanes = 3 (truth = 4);  faithful transcription = 75%, not 100%
  ```

  This *is* the wrong direction (faithful OCR penalised), so it is accepted on evidence, not
  on principle: it never bites on the reference corpus, because real financial tables carry a
  total row. It replaced a far worse over-split — mean pct went 49.6% → **84.5%**, 12 of 18
  pages at 100%, zero pages over-splitting. Revisit if a corpus without complete rows appears.
  A related unproven risk the fix log flags: duplicated page content could manufacture row
  support without raising the widest row's value count.
- **B2 limitation (LOW, accepted) — decimal-aligned columns lose lane resolution.** The
  paired-column fix keys on an exact-zero gap magnitude existing under some anchor (left,
  right or centre). Decimal-aligned numbers with varying integer widths have no exact-zero
  floor under *any* of the three, so the rightmost column can fall below the two-row lane
  support rule and come back lane-ambiguous (`-1`). Measured: `(0,1,2,-1)` on a 4-column
  decimal-aligned table. **This is not a wrong-direction defect** — a faithful transcription
  still scores 100%, and a column swap is still detected (83.3% vs 100%), just less sharply
  than under right alignment (75%). Accepted rather than fixed; decimal alignment is common
## Next action

**B2 is DONE.** The corpus re-score — the plan's stated real acceptance test — has been run
and both tickets are now validated against real pages rather than fixtures.

### Measured outcome

| | before B1/B2 | after |
|---|---|---|
| pages scoring a spurious `0.0%` | 17 | 0 (correctly not-scorable) |
| mean over each metric's own scorable set | 44.3% | 49.1% (B1 alone) |
| mean pct after the lane fixes | 49.6% (over-split state) | **84.5%** |
| pages at 100% | — | **12 of 18** |
| pages over-splitting lanes | 16 of 18 | **0** |

Method: scored the preserved run's 68 emitted pages against the source PDF under the
pre-B1/B2 modules (loaded from `18b3b64` into an isolated process — no checkout) and under
HEAD. No OCR was re-run; local-model variance exceeds the effect.

### Still below 100%, cause not yet separated

p39 (98.6), p45 (98.0), p46 (74.7), p55 (50.0), p24 and p53 (0.0). **Unknown whether these
are genuine qwen failures or residual metric defects** — p53 is the page the plan cites as
carrying a real column shift, so 0% there may be the metric working. Rendered comparisons
already exist in `~/Desktop/socr-qwen-failures/` and `~/Desktop/socr-engine-compare/`.
Until this is separated, **84.5% is not known to be the metric's ceiling or the engine's.**

### Remaining work

- **C1** — surface unexplained lanes *and* B1's `ceiling_note` through
  `TABLE_DISTRUST_KINDS` to the document level. Carries B1 review finding 1.
- Optionally: separate engine failure from metric defect on the six pages above.

### Standing lesson

Seven attempts on one ticket, six of them design-level failures, every one caught by a probe
outside the fixtures. **144 synthetic combinations passed while the real document failed 16
of 18 pages.** The corpus gate (`tests/test_corpus_rescore_gate.py`) exists so that cannot
recur silently; it skips in CI, so it protects local work only. Validate lane geometry
against the preserved corpus before believing any change — and never against a fresh OCR run.
