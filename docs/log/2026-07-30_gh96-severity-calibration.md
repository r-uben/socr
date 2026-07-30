# 2026-07-30 — GH-96 Unit A: the hierarchy-aware metric, and why the proposed accept rule is dead

Unit A of the #96 plan exists to answer one question before any of the escalation
lane is written:

> Does `defect_severity` improve on exactly the pages where cell exactness improves?

It does not. On **none** of them. The proposed accept rule would have rejected every
win the issue cites.

## First: the existing metric cannot see this failure mode at all

`BenchmarkScorer.score_table_cells` compares numeric cells positionally and, when the
predicted grid shape differs from ground truth, falls back to **multiset recall**. The
GH-96 failure is a *permutation* — every digit present, right column, wrong row — so
recall is perfect. On OBR EFO November 2022 page 13:

| engine | `score_table_cells` | hierarchy-aware |
|---|---|---|
| socr strict-local | **100.0%** | **38.4%** |
| escalated (vision) | **100.0%** | **100.0%** |

The existing scorer gives the identical 100.0% to a scrambled table and a perfect one.
Any past benchmark number for a hierarchical table is uninformative about this mode.

## The calibration

Scored against each page own native text layer (born-digital, so ground truth is free),
cross-tabulated with `defect_severity(snapshot_structural_defects(...))`:

| page | socr % | escalated % | severity before | severity after | severity improves? | accept rule verdict |
|-----:|-------:|------------:|-----------------|----------------|--------------------|---------------------|
| 13 | 38.4 | 100.0 | (0, 0, 0, 0, 2) | (0, 0, 0, 0, 2) | no | REJECT — loses a real gain |
| 39 | 86.5 | 86.5 | (0, 0, 0, 0, 8) | (0, 0, 0, 0, 8) | no | REJECT |
| 45 | 100.0 | 100.0 | (0, 0, 0, 0, 10) | (0, 0, 0, 0, 10) | no | REJECT |
| 46 | 0.0 | 100.0 | (0, 0, 0, 0, 0) | (0, 0, 0, 0, 8) | no | REJECT — loses a real gain |
| 48 | 0.0 | 100.0 | (0, 0, 0, 0, 0) | (0, 0, 0, 0, 4) | no | REJECT — loses a real gain |
| 51 | 11.1 | 85.9 | (0, 0, 0, 0, 6) | (0, 0, 0, 0, 10) | no | REJECT — loses a real gain |
| 53 | 0.0 | 0.0 | (0, 0, 0, 0, 8) | (0, 0, 0, 0, 8) | no | REJECT |
| 55 | 0.0 | 50.0 | (0, 0, 0, 0, 0) | (0, 0, 0, 0, 5) | no | REJECT — loses a real gain |
| 59 | 100.0 | 100.0 | (0, 0, 0, 0, 1) | (0, 0, 0, 0, 1) | no | REJECT |
| 60 | 100.0 | 100.0 | (0, 0, 0, 0, 2) | (0, 0, 0, 0, 2) | no | REJECT |
| 61 | 82.1 | 82.1 | (0, 0, 0, 0, 4) | (0, 0, 0, 0, 4) | no | REJECT |
| 62 | 7.7 | 32.1 | (0, 0, 0, 0, 1) | (0, 0, 0, 0, 1) | no | REJECT — loses a real gain |
| 63 | 54.8 | 83.9 | (0, 0, 0, 0, 1) | (0, 0, 0, 0, 1) | no | REJECT — loses a real gain |
| 64 | 0.0 | 97.3 | (0, 0, 0, 0, 1) | (0, 0, 0, 0, 2) | no | REJECT — loses a real gain |
| 65 | 0.0 | 92.4 | (0, 0, 0, 0, 1) | (0, 0, 0, 0, 4) | no | REJECT — loses a real gain |
| 67 | 85.3 | 85.3 | (0, 0, 0, 0, 1) | (0, 0, 0, 0, 1) | no | REJECT |

**9 of 9 genuine gains rejected. Severity improves on zero pages.**

## Why, precisely

Two independent reasons, both visible in the table above.

**1. Only `lane_gap` ever moves, and it moves the wrong way.** The severity tuple is
`(verifier_hard_fail, label_binding_failure, header_collapsed, header_col_gap, lane_gap)`.
The first four are `0` on every page here. So severity reduces to
`lane_gap = abs(output_col_count - native_lane_count)` — and on pages 46, 48 and 55,
where socr emitted **no table at all**, `lane_gap` starts at 0 and *rises* to 8, 4 and 5
once a real multi-column table is produced. The metric is anti-correlated with quality:
emitting nothing scores better than emitting a correct table.

**2. `label_binding_failure` is 0 on page 13** — the canonical hierarchical-shift page,
whose entire defect *is* label binding. So it fails as a trigger as well as an accept
test. The Unit B sketch proposed it as the primary trigger signal; that is now known to
be wrong before a line of it was written.

This also confirms the design-review point that severity carries no *safety* weight: a
well-formed fabrication improves exactly these structural components. It was never
independent evidence, and it is not even a usable well-formedness filter here.

## Consequences for Unit B

- **Do not use `defect_severity` improvement as the accept test.** Not as the gate, and
  not as a secondary filter — it rejects 9 of 9 real wins.
- **Do not use `label_binding_failure` as the trigger.** It does not fire on the mode it
  names.
- The accept test has to be the canary alone (token containment calibrated against the
  incumbent), plus a well-formedness check that is *not* drawn from this tuple.
- A trigger still needs designing from signals that demonstrably fire. Candidates
  observable here: the page declared tables but emitted no table blocks (46/48/55), and
  the reconcile `count mismatch (no safe patch target)` note. Both need the same
  cross-tabulation before being trusted.

## Aggregate, restated with the better metric

| | cells | exact | |
|---|---:|---:|---|
| socr strict-local | 2361 | 1044 | **44.2%** |
| escalated | 2361 | 2006 | **85.0%** |

The bake-off scratch scorer reported 38.6% vs 74.0% over 2254 cells. Both engines score
higher here because the scratch scorer duplicate-label collision and its parser ceiling
were suppressing real matches; the *gap* is essentially unchanged. Page 13 escalated is
now measured at exactly **100.0%**, confirming the bake-off 95.2% was entirely the
collision artifact.

## Method

Ground truth: `src/socr/benchmark/table_exactness.py::native_rows_from_page`. Rows are
scoped to located table regions, grouped by vertical overlap (parameter-free — words on
a visual row have overlapping y-intervals), and split at the last non-numeric word so a
numeral inside a label ("Growth Plan after **17** October reversals") is not read as
data. Hierarchy comes from x-indentation. Matching is by normalized label in document
order, so a label reused under two parents ("Other measures") cannot be credited twice.

Reproduce: the paths in the bake-off note Setup table, plus `score_page(page, markdown)`.

---

# Addendum (2026-07-31) — the trigger, calibrated

The section above closed by saying a trigger "still needs designing from signals
that demonstrably fire", and named two candidates. With the metric in hand, all
candidates can be cross-tabulated against the ground truth of *does escalation
actually help this page* (a gain of more than one percentage point).

A candidate the original Unit B sketch never considered turns out to dominate: on a
born-digital page the exactness metric is computable **at runtime, model-free, at
zero cost**, because the native text layer is right there. That is a direct measure
of the defect rather than a structural proxy for it.

| trigger | fires | true pos | false pos | missed | precision | recall |
|---|---:|---:|---:|---:|---:|---:|
| **native-exactness < 100%** | 13/16 | 9 | 4 | 0 | **69%** | **100%** |
| a ground-truth row is missing | 13/16 | 9 | 4 | 0 | 69% | 100% |
| orphan value rows present | 4/16 | 4 | 0 | 5 | 100% | 44% |
| tables declared, none emitted | 3/16 | 3 | 0 | 6 | 100% | 33% |
| `dualpass_flagged` (current) | 16/16 | 9 | 7 | 0 | 56% | 100% |

## Recommendation

**Escalate when the emitted table disagrees with its own native text layer.**

It strictly dominates the signal socr flags on today: identical recall (100%, no
missed gains), better precision (69% vs 56%), and it fires on three fewer pages.

The discrimination is exactly what #95 found missing. Pages 45, 59 and 60 are
100% correct and this trigger stays silent on all three — while `dualpass_flagged`
fires on them, because it fires on every table page in the document.

Properties that matter for the lane:

- **Parameter-free.** "Disagrees at all" — not a tuned threshold. Equivalent
  formulations (`exactness < 100%`, `rows_not_found > 0`) select the same 13 pages.
- **Model-free and free.** No engine call to decide whether to make an engine call.
- **Fail-safe direction.** The four false positives (39, 53, 61, 67) cost one cloud
  call each, ~$0.0002, and escalation was never worse than the incumbent on any
  page in this document. Over-triggering costs money, not correctness.
- **Same oracle as the canary.** The trigger asks "does the incumbent disagree with
  native?"; the canary asks "does the candidate invent tokens native does not have?"
  One oracle, two questions, and it degrades coherently: where the native layer is a
  poor oracle the trigger fires and the canary then rejects, so the page is left
  alone at the cost of a wasted call.

## Limits, stated plainly

- **Born-digital only.** A scanned page has no native oracle, so neither trigger nor
  canary can run. Those pages stay out of the lane, as already planned.
- **One document, no negative controls for precision.** 16 table pages from one
  report. The 69% precision figure is a point estimate on a small, unrepresentative
  sample; the *ordering* against `dualpass_flagged` is the robust part.
- **The trigger cannot see permutations the canary also cannot see.** Both are token
  containment against the same layer. This trigger works on the OBR mode-(a) pages
  because the shift also *drops rows* (parents emptied, orphans trailing), which
  `rows_not_found` catches — not because it detects misalignment as such. A pure
  permutation with every label intact would score below 100% on cell values and so
  would still fire; a permutation that also preserved cell-to-label binding would be
  invisible, but that is not a failure mode observed here.
- Page 53 fires and never improves (0.0% → 0.0%): its native layer does not parse
  into scorable rows at all. It is a permanent false positive until that parse is
  fixed, and it is the reason precision is 69% rather than 75%.
