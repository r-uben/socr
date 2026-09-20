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

Inside the fold loop, gating the destination:

```python
if len(row_words) > 1 and abs(other_key - y_key) > 1:
    continue
```

A MULTI-word group may be re-attached only across an ADJACENT band key. Two or more
words already form their own horizontal run, and a horizontal run of words at a shared
top IS what a table row is; re-attaching one run to another across a real vertical gap is
a row collapse, not the repair of a torn line. `1` is the quantum of the `round(y0)` key
function itself, so it is the widest span a rounding tear can produce — derived, not a
tuned tolerance. A LONE displaced word may still fold further, because a raised marker
(GH-330) legitimately lands several keys away.

That disjunction is the whole point: this code path serves TWO different repairs, and
every single-rule fix fails one shape or the other.

## Measured, across every shape

Band counts from calling `_assign_bands` directly. "want" is the correct answer.

| shape | want | main@d07d6e2 | single-word-only | shipped |
|---|---|---|---|---|
| multi-word span tear (keys 100/101) | 1 | 1 | **2** | 1 |
| lone-word tear (GH-600 archetype) | 1 | 1 | 1 | 1 |
| bridging third group (CASE 1) | 2 | **1** | 2 | 2 |
| CASE 1, reversed emission order | 2 | **1** | 2 | 2 |
| x-disjoint stacked rows (CASE 2) | 2 | **1** | 2 | 2 |
| x-disjoint stacked rows, text | 2 | **1** | 2 | 2 |
| lone-word section label stacked under a data row | 2 | **1** | **1** | **1** |

Bold is wrong. The shipped rule is correct on six of seven; `main` on two of seven.

### The one shape still wrong

A one-word non-numeric section label ("Liabilities") carrying spurious shared line
identity with the numeric row above it folds into that row, welding a row stub onto a
data row. It is a LABEL loss rather than a numeric-row collapse, but under this corpus's
doctrine a stub attributed to the wrong row is still silent content loss. **It fails
identically on `main`, so this branch neither creates nor closes it.** Filed separately
rather than bolted on here.

## The first attempt: single-word-only, and why it was replaced

The first version of this fix allowed a fold only when the displaced group was a SINGLE
word. It closes all four collapse shapes, but it loses a genuine heal: PyMuPDF jitters
tops per SPAN, not per word, so a styled run straddling the `.5` boundary tears WHOLE.
Three words at y0 100.46 and four at 100.54 are one printed line that `main` heals and
the single-word rule refuses.

That miss was found by an independent adversarial review, not by the 113 tests in the
first commit — every fixture there had a single-word fragment by construction. It is now
pinned by `test_gh862_multi_word_span_tear_across_adjacent_keys_still_heals`, whose
control half is the same words displaced by a real row gap, so the test cannot be
satisfied by dropping the adjacency restriction either.

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
| the adjacency arm (leaving single-word-only) | `test_gh862_multi_word_span_tear_across_adjacent_keys_still_heals` |
| the whole GH-862 clause (leaving `main`) | the three `test_gh862_union_composition_does_not_bridge_two_rows` orders, `test_gh862_x_disjoint_unequal_stacked_rows_do_not_merge`, and the multi-word tear test's control half |
| GH-600 x-overlap refusal | `test_gh862_x_disjoint_unequal_stacked_rows_do_not_merge` |
| strictly-larger-group rule | `test_gh862_equal_size_single_word_groups_x_disjoint_do_not_merge` |

Neither arm of the disjunction is inert: removing either one turns a passing test red.

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
