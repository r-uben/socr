# #213 Task 1 — which branch admits a book index?

Date: 2026-08-15
Plan: `docs/plans/gh213-index-routing/PLAN.md` (Task 1)
Issue: <https://github.com/r-uben/socr/issues/213>

## Answer

**Branch B — `has_numeric_columns`, the numeric-lane gate — fires. Branch A —
PyMuPDF's `find_tables()` — does not fire at all.**

This is the opposite of the plan's leading candidate. The plan's Branch-B exclusion was a
code reading, not a measurement, and the measurement contradicts it.

| Page | A `len(find_tables().tables)` | B `has_numeric_columns` |
|---|---|---|
| `2003__woodford.pdf` p798 | 0 | **True** |
| `2003__woodford.pdf` p799 | 0 | **True** |
| `2021__nagel__ml_ap.pdf` p157 | 0 | **True** |

Both files were materialized at read time (4.9 MB and 84.9 MB); neither was a 0-byte iCloud
placeholder.

## Why the plan's reasoning failed

The plan (and issue #213's later analysis, and the TR-3 log's conclusion 3) all rest on this
premise:

> An index row has exactly **one** numeric token, its page number, so it occupies one lane.
> It cannot satisfy a three-lane rule.

**That premise is false for a real book index.** An index entry does not carry one page
number — it carries a *list* of them:

```
price stability: allocation of resources
  under, 410–11; asymmetric, 418;
  choice of index, 13–14, 440–41, 442f,
```

Each of those is a separate numeric token at a separate x-position, so a single visual line
populates several lanes at once. The premise confuses "one page number per *entry*" with "one
numeric token per *row*", and the wrapped continuation lines are where it breaks.

Measured lane occupancy (gate needs `grid_rows >= 3`, each with `>= 3` lanes):

| Page | numeric tokens | lanes | rows | **grid_rows** |
|---|---|---|---|---|
| woodford p798 | 52 | 19 | 27 | **7** |
| woodford p799 | 82 | 15 | 33 | **10** |
| nagel p157 | 124 | 14 | 44 | **13** |

The threshold is 3. The pages clear it by 2–4×. Example qualifying row (woodford p799,
y=185): `350, 353, 362, 427, 430, 434, 435, 439,` — eight tokens, eight lanes, one prose line.

## What this means for Task 2

Per the plan's own routing, this is the "Branch B fires → the code reading is wrong somewhere,
find out why before changing constants" case. The why is above, and it is not a tuning
problem: no value of `_MIN_LANES_PER_ROW` separates these pages from a real table, because a
dense index genuinely has more numeric tokens per row than many real tables do.

Consequences to carry into Task 2:

1. **Do not touch `_MIN_LANES_PER_ROW` / `_MIN_TABLE_ROWS`.** The gate is doing exactly what
   its docstring says; the shape it was built to reject (a *sparse* reference list, one
   trailing number per line) is not the shape that is actually getting through.
2. `find_tables()` is **not** implicated. Arbitration against PyMuPDF is not the remedy.
3. The needed signal is a positive index-shape discriminator that runs *before* or *alongside*
   the lane gate. What distinguishes these pages from a table is not lane count — it is that
   the numeric tokens are **inline within running text** (comma-suffixed, irregular x, no
   vertical alignment across rows) rather than **column-aligned**. Note the lane counts: 14–19
   lanes for a page whose rows use 3–8 of them, i.e. the lanes barely overlap between rows.
   A real numeric table has *few* lanes that *most* rows share. That ratio —
   lanes-per-page vs. lanes-per-row, or per-lane row density — is derivable from the page's own
   geometry and needs no tuned constant.

## Correction to earlier records

`docs/log/2026-08-15_tr3-hand-judgement.md` conclusion 3 states "#213's stated index mechanism
is disproven" on the one-token-per-row argument. That disproof is itself wrong: the mechanism
is the lane gate, just not for the reason the issue originally gave. The plan's Task 2 branch
list should be read with this log on top of it.

## Reproduction

```
~/venvs/socr/bin/python -c '
import fitz, os
from socr.tables.reconstruct import has_numeric_columns
p = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/library/Papers/papers/2003__woodford.pdf")
page = fitz.open(p)[797]
print(len(page.find_tables().tables), has_numeric_columns(page))
'
# -> 0 True
```

Diagnosis only. No source changed, no corpus content committed.
