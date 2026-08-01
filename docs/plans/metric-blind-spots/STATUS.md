# STATUS — metric blind spots (socr #123)

Last updated: 2026-08-01

## Stage

Planned, nothing dispatched. The graph exists and the design behind it was already
attacked adversarially (recorded on #123), so this is ready to execute cold in a new
session.

Nothing here changes OCR behaviour. It changes what socr can *measure* — which
matters because `escalation_decision` uses the metric as a production accept rule.

## Base state (clean before tickets)

- `r-uben/socr`, `main` at `18b3b64`, clean, no open PRs
- full suite green: 1363 passed, 1 xfailed
- lint gate: `uvx ruff@0.16.0 format --check .` clean over 235 files
  (**not** `~/venvs/socr/bin/ruff` — that version cannot check Markdown; see CLAUDE.md)
- reference artifacts preserved, since neither engine is deterministic:
  - `~/data/fiscal-ballast/_experiments/2026-07-31_gh96-engine-parity/`
  - `~/data/fiscal-ballast/_experiments/2026-08-01_gh96-corpus-rerun/`

## Ticket board

| Ticket | Stream | Status | depends-on | Wave |
|--------|--------|--------|------------|------|
| A1 | grade the metric | TODO | — | 1 |
| B1 | scoring correctness | TODO | — | 1 |
| B2 | scoring correctness | TODO | B1 | 2 |
| B3 | scoring correctness | TODO | B2 | 3 |
| B4 | scoring correctness | TODO | B3 | 4 |
| C1 | pipeline response | TODO | B4 | 5 |

## Dispatch waves

- **Wave 1:** A1 + B1 — file-disjoint (A1 is a new test file, B1 touches
  `table_exactness.py`)
- **Wave 2:** B2
- **Wave 3:** B3
- **Wave 4:** B4
- **Wave 5:** C1

B2/B3/B4 all touch `table_exactness.py` or `native_rows.py` and are strictly ordered;
do not parallelise them.

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

## Next action

Dispatch Wave 1: A1 and B1, in parallel.
