# 2026-08-01 — GH-96 corpus re-run: the lane measured after the fixes

The first full-document run of the escalation lane was made with a trigger that fired
on non-table pages and a metric that penalised engines for footnote syntax. Its
numbers were unusable. This is the same document re-run with #113 (grid trigger),
#118 (footnote normalization) and #98 (guard scoping) in place.

Document: OBR *Economic and fiscal outlook* November 2022, 68 pages, born-digital.
Agentic mode, no `--strict-local`, so a cloud rung is in the ladder.

## Result

| | before fixes | after |
|---|---:|---:|
| escalation calls | 18 | **10** |
| of which on pages with no table | **7** | **0** |
| accepted | 3 | **5** |
| rejected as regression, wrongly | ≥2 | **0** |
| metered cost | $0.0038 | **$0.0022** |

Every call now lands on a genuine table.

## The ten decisions

```
p13  ACCEPT   39.1% -> 100.0%
p39  reject   86.5% -> 86.5%     tie
p48  ACCEPT   17.6% -> 100.0%
p51  ACCEPT   94.9% -> 100.0%
p53  reject    0.0% ->   0.0%
p55  ACCEPT   40.9% ->  50.0%
p63  ACCEPT   77.4% ->  90.3%
p65  reject   95.2% ->  95.2%    tie
p66  reject   94.1% ->  94.1%    tie
p67  reject   94.1% -> 94.1%     tie
```

Five accepted, four ties refused (no churn for no gain), one unresolvable.

`tables_trust.json` records `resolved_by_escalation: [13, 48, 51, 55, 63]` and drops
those pages from the untrusted list, so a consumer gating on the sidecar is no longer
steered away from pages the pipeline fixed.

## What the fixes actually changed

**#113, the grid trigger.** The previous run spent seven calls comparing an empty
result against an empty result on cover pages, chart pages and prose — the native row
parser had read "November=['2022']" and chart axis labels as tables. Requiring at
least two value columns and at least two rows sharing that width removed all seven and
kept every real table.

**#118, footnote normalization.** The previous run rejected pages 61 and 62 as
regressions (82.1% -> 50.0%) when both candidates were in fact 100% correct; it
rejected p51 as a one-cell regression when the candidate was perfect. Those were
LaTeX/HTML footnote markers the normalizer did not fold, not quality differences. p51
is now an accepted +5.1 points, and the phantom regressions are gone.

## p53 — the irreducible cost, as predicted

A real 52-cell table whose labels the ground-truth parser cannot match, so both
engines score 0 and the call is wasted. The obvious guard — do not escalate when the
incumbent matched zero rows — would also refuse pages 64 and 65, which in the July
baseline were genuine 0% -> ~90% recoveries. The two are indistinguishable before the
call is made.

One wasted call per page where the GT parser fails on a real table. Known and
accepted, not a defect to engineer around.

## Caveats that still hold

**Run-to-run variance exceeds the lane's effect.** Page 48 scored 17.6% here and
~100% in the previous run, from the same local model on the same page. Page 65 was a
95-point escalation win in the July baseline and needs no escalation here. Single-run
per-page comparisons are close to meaningless; only aggregates and mechanisms
transfer.

**Document status is still `partial`, 19 pages untrusted, 40 flags.** Escalation fixed
five pages. It did not make the document trustworthy, and the dual-pass detector still
flags essentially every table page — the saturation recorded in the GH-95 calibration.

**The repetition guard (GH-97) has still never fired live.** Three full runs, zero
`table_row_repetition_truncated` events. The failure it guards is real — the July
artifact has 865 blank rows — but it has not recurred, so the guard's live behaviour
remains assumed rather than observed.

**One document, no negative controls.** Orderings have been stable across runs;
absolute values have not.
