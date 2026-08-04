# STATUS — metric blind spots (socr #123)

Last updated: 2026-08-02

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
| C1 | pipeline response | DONE | B2 | 3 |
| C2 | pipeline response | DONE | C1 | follow-up |
| B6 | scoring correctness | TODO | B2 | follow-up |
| B7 | engine defect | TODO | — | follow-up |
| B8 | — | CLOSED — invalid, premise was a misreading | — | — |

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
| C1 | socr-implementer | DONE — see `docs/log/2026-08-01_TICKET-C1.md` |
| C2 | socr-implementer | DONE — see `docs/log/2026-08-02_TICKET-C2.md` |

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

- **B1 finding 1 (HIGH) → CLOSED by C1 + C2.** `ceiling_note` now reaches `tables_trust`
  (`untrusted_pages`, `counts_by_kind`, per-page `reasons`) via the new `table_not_scorable`
  kind, on **every run**, not only cloud-enabled ones. C1 wired the emitter to
  `_table_page_needs_escalation`; C2 added a call site (`_surface_table_scoring`) reachable
  with no escalation profile, so a local-only run (socr's default configuration) no longer
  ships all 17 not-scorable pages with no trace. Cost: ~137ms mean per page.
  **The rationale originally recorded for that cost was wrong** and was corrected in
  `e6a6242`: it compared against per-page VLM inference, but the reference document's run log
  records **65 of 68 pages as born-digital trusted native text**, which never run a VLM — so
  the cost was close to pure addition on exactly the pages that were previously nearly free.
  Now gated on `_page_has_tables`: prose pages skip the scoring entirely, chart pages still
  reach it (`has_tables` is True there, which is why B1 existed) and still surface as
  not-scorable. See `docs/log/2026-08-02_TICKET-C2.md`.
- **B1 finding 2 (MEDIUM) → CLOSED by the corpus re-score.** 17 pages of spurious `0.0%`
  become not-scorable; mean over each metric's own scorable set rises 44.3% → 49.1%.
- Findings 3 and 4 (LOW) in `docs/log/2026-08-01_TICKET-B1-review.md`.
- **B5 verified, corpus value UNPROVEN.** The reproducer now yields
  `'Central government net debt'` at 6pt, 9pt and 12pt, where before it gave `'debt'` or the
  scrambled `'Central debt government net'`. But diffing ground-truth labels across all 68
  corpus pages, only 7 change and **every one is a chart page already in B1's not-scorable
  set** — no scorable table page moved. The OBR document's real tables carry no band-split
  labels. Defect genuine and fixed; whether it matters on any real corpus is unknown.
- **B2 limitation (MEDIUM, accepted) — sparse tables where NO row is complete under-split.**
  The widest-row cap bounds lane count by the widest row's value count, which is presented as
  structural but really assumes at least one row is complete. Reproducer: 4 real columns, each
  row omitting a different one → columns 0 and 1 merge, faithful transcription scores 75%.
  Wrong direction, accepted **on evidence not principle** — it does not bite on the reference
  corpus (financial tables carry total rows) and it replaced a far worse over-split (mean
  49.6% → 84.5%). A related unproven risk: duplicated page content could manufacture row
  support without raising the widest row's value count.
- **B2 limitation (LOW, accepted) — decimal-aligned columns lose lane resolution.** No
  exact-zero gap exists under any anchor, so the rightmost column can come back
  lane-ambiguous (`(0,1,2,-1)`). Not the wrong direction: faithful still 100%, shifts still
  caught (83.3% vs 100%), just less sharply than right alignment's 75%.
- **SEPARATED, 2026-08-02 — the residual is mostly the METRIC, plus one real engine defect.**
  Compared emitted markdown against the native layer and against a render of the printed page:
  - **p53 (0.0%) — metric.** The printed table has a literal `Met`/`Not Met` column; socr emits
    it correctly as a column, and the last-non-numeric-word rule swallows it into the row
    label, so 0 of 6 labels match and all 17 rows fail. `agy` and `gemini-ocr` also score
    **0.0%** on this page — three engines failing identically is the scorer. → **TICKET-B6.**
  - **p53 also carries a real engine defect — diagnosed 2026-08-03, and the first framing of
    it was wrong.** It is not "repeated values dropped". The model emitted a header declaring
    **four** data columns for a table that needs **five** (the `Met`/`Not Met` status column
    plus four numbers), so `Met` takes the first numeric slot and every fully-populated row
    pushes its last value off the right edge. Confirmed against the raw model cache
    (`engine: qwen`, `page_num: 53`) — the loss is present before any socr processing, so it
    is the model, not post-processing. → **TICKET-B7** (rewritten around the real mechanism).
  - **The damage was detected AND surfaced — B8 was filed on a misreading and is closed.**
    `audit_passed` in a cached page blob is the per-page hallucination check, not the table
    trust surface. The run's own `tables_trust.json` has page 53 in `untrusted_pages` with
    three reasons, one of them naming the exact shortfall (`output_cols=5` against
    `native_lanes=14`) and another the missing values (`19 numeric rows vs 21 native`).
    The surfacing works. **This also lowers B7's severity**: the loss is real but flagged, not
    silent, so it is not the no-silent-content-loss violation first claimed.
  - **p46 (74.7%) and p55 (50.0%) — metric.** Every ground-truth value is present in the
    emitted markdown and rows match; the loss is in alignment/label matching, not extraction.
  - **p24 (0.0%) — engine.** No table emitted at all. A real zero, correctly measured.
  So **84.5% is largely the metric's ceiling, not qwen's** — but not purely: real content loss
  exists underneath it.

## Note on Stream A

The corruption battery found the eighth defect (B5) on its first run, in a transform the
ticket had listed as *benign* — the first defect in this plan caught before it produced
published numbers rather than after. It then caught two more during B5's own implementation.

## Next action

**All six tickets are DONE** (A1, B1, B2, B5, C1, C2). Branch `feat/123-metric-blind-spots`,
1406 passed / 1 xfailed, lint clean, **not pushed, no PR opened.**

Remaining, in order of value:

1. **Separate engine failure from metric defect** on p46/p55/p24/p53, using the rendered
   comparisons already on the Desktop. Without it we do not know whether 84.5% is the
   metric's ceiling or qwen's.
2. Open the PR when the above is settled.

## Standing lessons from this plan

- **Synthetic fixtures encode the geometry their author already thought of.** 144 synthetic
  combinations passed while the real document failed 16 of 18 pages. `tests/test_corpus_rescore_gate.py`
  exists so that cannot recur silently — but it **skips in CI**, so it protects local work only.
- **Seven attempts on TICKET-B2, six of them design-level failures.** Every one shared a root:
  a rule that assumes the data has exactly the structure the rule can represent — one seam,
  one distinguished gap, two groups, a clean zero floor, one complete row. Prefer formulations
  that select their own structure, and probe real pages before believing a green run.
- **Never validate against a fresh OCR run.** Local-model variance exceeds every effect here.
  Re-score the preserved runs in `~/data/fiscal-ballast/_experiments/`; to get a genuine
  "before", load old modules from git into an isolated process rather than checking out.
- **Never `git checkout` / `switch` / `stash` / `reset` while agents are running** — one
  shared working tree. A `git clone` is not isolation either: the editable install resolves
  `import socr` to the main checkout.
