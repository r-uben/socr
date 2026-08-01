# TICKET-B2 review — 2026-08-01 · REJECT (reverted)

Reviewed commit `ad649b5`. Verdict **REJECT**; reverted by `717914d`.
The implementation followed the ticket faithfully. **The ticket was wrong.**

## What the change did

`markdown_rows` stopped compacting a row's cells, storing `positional_values`
(blanks as `""`) on `LabeledRow.values` instead. `score_rows` was untouched.
The A1 strict xfail on `shift_into_adjacent_empty_cell` was removed and passed.
Full suite green: 1378 passed, 2 xfailed. Lint clean. `cells` semantics genuinely
unchanged, as reported — it is computed from ground-truth rows only.

All of that is accurate. It is also beside the point.

## The defect

The markdown side became positional while the ground truth stayed **compacted**
(`native_rows_from_page` emits a flat x-ordered list with no gaps). Comparing a
positional list against a compacted one misaligns every row with a **leading** gap.

Measured on a three-row, two-column page whose middle row's only value sits in
column 2:

| transcription | pct | exact/cells |
|---------------|-----|-------------|
| faithful — reproduces the empty cell | **80.0** | 4/5 |
| sloppy — drops the column entirely | **100.0** | 5/5 |

An engine that correctly reproduces the gap is penalised; one that destroys the
column structure scores perfect. `escalation_decision` accepts on
`candidate.exact > incumbent.exact`, so this is not a reporting curiosity — the
production accept rule would prefer the worse engine.

This is the same wrong-direction family as the original seven defects, newly
introduced by a ticket meant to remove one.

## Root cause: the seam, not the code

The plan split one change in two — markdown positions in B2, ground-truth lanes in
B3 — and told B2 to "compare positionally against the preserved markdown shape"
in the interim. There is no coherent interim: a positional list cannot be compared
against a compacted one. Any faithful implementation of B2-alone produces this.

**Resolution:** B2 and B3 merged into a single TICKET-B2. B3 is CLOSED, number
retained so references resolve.

## Why the battery missed it

A1's corruption battery passed throughout. Two reasons, both now fixed:

1. **Every assertion is relative** (`corrupted.pct < base.pct`). A uniformly
   depressed baseline still satisfies all of them. No test asserted that a perfect
   transcription scores 100% — the most basic metamorphic property there is.
2. **The base fixture's sparse row has a *trailing* gap** (`| 0.5 |  |`), which stays
   aligned even when the two sides disagree about what a gap is. Only a **leading**
   gap exposes the asymmetry.

Added, and passing on the reverted tree:

- `test_a_perfect_transcription_scores_100`
- `test_dropping_a_column_never_beats_keeping_the_gap`

Both would have failed on `ad649b5` — measured 80.0 against an asserted 100.0, and
faithful `exact`=4 against sloppy `exact`=5. They are listed in the merged B2's
done-when and **must not be weakened** to suit an implementation.

## Note for the next implementer

The relative-only assertion style is a general weakness of this battery, not a
one-off. When adding corrupting transforms, also pin at least one absolute anchor,
or a future regression can depress every score uniformly and go unnoticed again.
