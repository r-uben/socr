# TICKETS — GH-152 side-by-side tables merged

Status keys: `TODO` · `WIP` · `DONE` · `BLOCKED`. `depends-on` gates dispatch.

Context: `2025__haim` p31 prints TABLE A5 and TABLE A6 side by side. The rowizer
segments rows by y across the full page width, so the two tables are read as
single rows spanning both and neither survives. 54 tokens are also lost outright.
`extract_structured` already documents the x-band limitation for READING ORDER;
this is the same limitation damaging table reconstruction.

⚠️ `src/socr/tables/reconstruct.py` carries the GH-146 work, committed and open as
**PR #149**, and is then held by GH-144 A2 for the whole of wave 2. Do not dispatch A1
until both have landed.

## Stream A — column segmentation

### TICKET-A1 — an x-band DETECTOR, wired into nothing · TODO · depends-on: none · wave 3b
**Problem:** Row clustering spans the full page width, merging two side-by-side tables.

⚠️ **RETARGETED 2026-08-13 by a wave-3 ruling.** The original text said "detect gutters,
split the page into bands, reuse the clip-then-rowize approach per band" — i.e. detect AND
integrate. Measurement during the wave-3 design pass showed integration cannot land here:
wiring only the rowizer rung is a **measured end-to-end no-op** on the aligned fixture,
because `reconstruct_table_regions` has already returned a merged grid before the rowizer is
reached (`born_digital.py:1169-1192` short-circuits when `table_regions` is non-empty). The
page-wide text-strategy `find_tables` at `reconstruct.py:141` has no clip; that is what merges
the two tables, and it also suppresses the fallback that would have coped.

The defect is confirmed to still reproduce on current `main`. Only the scope changed.

**Do:** Add a private word-list → x-bands helper in `reconstruct.py`. **Detector only — do
NOT wire it into `reconstruct_table_regions` or `rowize_from_word_list`.** Integration is A2.

Candidate gutters are empty intervals in the word-x projection, judged by persistence across
the page's own repeated y-rows. Accept a split only when **both** sides show repeated row
structure **and** each side has a label column (a non-numeric word left of that side's
leftmost numeric lane, in at least `_MIN_TABLE_ROWS` rows). On any doubt — including a
bridged gutter — return the original full-width band.

**Binding constraints from the ruling:**
- Do **not** reuse `has_numeric_columns` / `_MIN_LANES_PER_ROW` as the per-band gate. That
  gate requires three co-occupied numeric lanes; the motivating A5/A6 token list looks like
  1- and 2-lane schemas, and a standalone 2-numeric-column table already fails both rungs.
- The per-band **label-column requirement is required**, not optional. The weaker
  "both sides look tabular" rule is the over-split failure mode: a halved wide table's right
  half has no label column.
- Do **not** promise left-to-right emission from `reconstruct.py`. `born_digital.py:1201`
  re-sorts by `y0` only and is owned by a sibling ticket.

**Files:** `src/socr/tables/reconstruct.py`
**Done when:** a synthetic two-table page yields two left-to-right bands; a single-table page
and an adversarial page with a wide internal label/value gap each yield one band. Plus a
separate characterization test, through the **installed package**
(`BornDigitalDetector().extract_structured`), pinning today's merge behaviour — A1 must not
flip it.
**Acceptance must fail today** because no x-band helper exists and the current whole-page
calls merge the two-table fixture. A green suite is not proof. Any content-loss claim must go
through `extract_structured` or `process()` — never an isolated rung, never a
standalone-module import. (Wave 2's #192 review produced a false blocking finding exactly
this way.)

### TICKET-A2 — consume the helper at BOTH merging rungs · TODO · depends-on: A1 · wave 4
**Problem:** Two tables need two grids, emitted left-to-right then top-to-bottom.

⚠️ **RECUT 2026-08-13, per A1's retarget ruling.** The original A2 ("rowize each band
independently, in reading order") addressed only one of the two rungs that merge. A1 is now
detector-only, so A2 owns all integration.

**Do:** Consume A1's helper at **both** merging rungs:
1. `page.find_tables(clip=band)` per band, instead of the current unclipped page-wide call;
2. band-scoped `rowize_from_word_list` when the clipped grid is empty or fails `_looks_tabular`.

Also fix, and document as a **second, distinct A2 defect**: the rowizer's snap-radius check at
`reconstruct.py:1348-1354` can drop the right-hand table's labels.

**Files:** `src/socr/tables/reconstruct.py`, and `src/socr/core/born_digital.py` **only if**
left-to-right reading order stays in this ticket's `Done when` — `born_digital.py:1201`
re-sorts by `y0` alone, so ordering cannot be delivered from `reconstruct.py`. The coordinator
must grant that file explicitly before dispatch; it is claimed elsewhere.
**Done when:** the p31 fixture emits two distinct markdown tables, and TABLE A5's correlation
values do not appear interleaved with A6's means — measured through the installed package.


## Stream B — evidence

### TICKET-B1 — end-to-end on the motivating page · TODO · depends-on: A2 · wave 3
**Problem:** Must be demonstrated on the real page, not only a synthetic one.
**Do:** Measure p31 word recall before/after and assert the 54 previously-missing
tokens (`Measure`, `Correlation`, `Mean`, `SD`, `Guidance`, `Legislative`, `0.95`,
`0.78`, …) are present.
**Files:** `tests/test_side_by_side_tables_gh152.py`
**Done when:** p31 recall ≥ 95% and the named tokens are all present; recorded in `logs/`.
