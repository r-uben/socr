# GH-862 — only a single displaced word may re-attach to a printed line

Branch `fix/862-band-fold-guards`. Filed against `#860` (issue `#600`) after that PR
merged; both reported shapes were reproduced on `main@d07d6e2` by calling
`_assign_bands` directly, so this is a real defect in shipped code, not a review opinion.

## What was wrong

`_assign_bands` folds a group of words into a neighbouring band when the two appear to
belong to one printed line that `round(y0)` tore into two keys. GH-600 widened that fold.
Two shapes get through:

1. A third group bridges two rows that the x-overlap guard refuses when compared to each
   other directly. The bridge is never compared to the pair it joins.
2. Two stacked rows that are x-disjoint — one carrying content only on the left, the next
   only on the right — pass the x-overlap guard with no tear involved at all.

Both are row collapse: two table rows become one. In a citation corpus that outranks a
missed repair (CLAUDE.md, "no silent content loss").

## What ships

At the top of the fold loop:

```python
if len(row_words) > 1:
    continue
```

Only a SINGLE displaced word may be re-attached. Two or more words already form their own
horizontal run, and a horizontal run of words at a shared top IS what a table row is;
re-attaching one run to another is a row collapse, not the repair of a torn line. This is
a categorical distinction — fragment versus run — not a tuned size, so it does not breach
the no-magic-numbers rule.

## Two candidates measured and rejected

**FIX 1 — compare the candidate against the accumulated union-find component, not the
original group.** Order-dependent, and the order is word emission order. With words
emitted RowA → tear → RowB it gives the correct two bands; with RowA → RowB → tear it
collapses the two rows and splits the tear off. The guard refuses whichever fold is
evaluated *second*, so it cannot be the discriminator. Dropped.

**FIX 2 — require the two band keys to be adjacent (`abs(other_key - y_key) <= 1`).** Its
premise is false. A rounding tear is not confined to adjacent keys: a superscript marker
on the same printed line sits several keys away. Measured in two existing fixtures —
`test_superscript_same_line_marker_folds_into_numeric_group_using_metadata` (marker at
y [104,108], row at [107,113]) and `test_text_fold_does_not_hop_into_a_numeric_row`
(y [104,111] against [110,116]). Both tests fail under FIX 2. Dropped.

FIX 1 was additionally shown INERT once FIX 2 was present — mutating the component walk
back out left every new test passing. Per the repo rule, a guard never seen to fail has
not been shown to guard anything, so it was removed rather than kept "for safety".

## Mutation evidence — all three guards on this path are load-bearing

Each mutation was applied to a full copy of `src`, `tests` and `pyproject.toml` outside
the repo, with a canary asserting `socr.__file__` resolves inside the copy, and an
uncapped `count(anchor) == 1` assertion before editing.

| Guard deleted in the mutant | Test that fails |
| --- | --- |
| single-word restriction (this fix) | `test_gh862_union_composition_does_not_bridge_two_rows`, `test_gh862_x_disjoint_unequal_stacked_rows_do_not_merge` |
| GH-600 x-overlap refusal | `test_gh862_x_disjoint_unequal_stacked_rows_do_not_merge` |
| strictly-larger-group rule | `test_gh862_equal_size_single_word_groups_x_disjoint_do_not_merge` |

The fourth new test, `test_gh862_rounding_tear_across_adjacent_keys_still_heals`, is the
control: it fails if the fold is disabled outright, so the suite cannot be satisfied by
deleting the feature instead of fixing it. The union-composition test is parametrised
over three word emission orders (`tear-first`, `row-first`, `by-x`) — that single test
would have caught FIX 1 immediately.

`_assign_bands` is pure and never reaches a provider, so absolute band counts are
hermetic in CI; the "pin a difference, not a value" rule is satisfied here by the
mutation pairs rather than by a runtime flag.

## A correction carried over from #860

PR #860 and issues #862/#863 state that the GH-600 x-guard "prevents 1,023 false merges".
What was measured is that it REFUSED 1,023 folds. Nothing established those folds were
wrong merges, and a guard that refused everything would score higher still. The 62.65%
figure is a refusal rate, not a precision figure. Corrected publicly on #860. Separating
the two needs a labelled sample that has not been taken.

## Open, and deliberately not resolved in this branch

**The trade is unmeasured.** The single-word rule refuses an unknown number of genuine
line repairs in exchange for refusing the two collapse shapes. The corpus scan that would
size it is queued behind other work; see `#863`, whose third arm counts folds attempted,
folds refused by this rule, and torn lines still healed on this branch head.

**An alternative exists.** A second opinion proposed splitting the two mechanisms the
fold conflates — a rounding tear (tops differ by less than the `round` quantum) versus a
displaced marker (tops differ by a real superscript raise) — and allowing
numeric-and-numeric folds only when the resulting component's y0 span is below 1.0, the
quantum the partition already uses. Evaluated on the component rather than the group, that
rule is order-independent. Its named breaker is a *numeric* footnote marker (a superscript
`1` rather than `*`), which would take the numeric path, be refused, and become an orphan
band. That rule is strictly narrower than what ships here and would heal more lines; it
is not adopted because its breaker is unmeasured and because it turns on a constant that a
reviewer may reasonably class as a threshold. Revisit when #863's arm-3 counts land.
