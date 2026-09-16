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

### TICKET-A1 — an x-band DETECTOR, wired into nothing · DONE · depends-on: none · wave 3b
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

### TICKET-A2 — consume the helper at BOTH merging rungs · DONE · depends-on: A1 · wave 4 · related: GH-418, GH-780
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

⚠️ **Partial closure — read before treating GH-152 as fully resolving side-by-side pages.**
This ticket closes the merge, and the "second, distinct defect" above, **only when
`_detect_column_gutter` fires**. When it does not fire on a genuine two-table page (measured
to happen, not hypothetical — hit twice while building this ticket's own fixtures), the
already-open **GH-418** (a word beyond every lane's snap radius is silently dropped in
`_rowize_segment`, found during #342, two prior fixes already rejected there for regressing
prose-page handling) still applies, and a page can still misattribute exactly as this ticket
describes. "GH-152 fixed" does not mean "side-by-side tables are safe" unconditionally — it
means safe specifically when the gutter is detected. See `docs/log/2026-09-16_152.md` for the
full trace. **related: GH-418** (cross-link this both ways — GH-418's issue should point back
here too).

⚠️ **Second partial-closure note — GH-780, a page-sized density floor applied
to a band.** Distinct from the GH-418 gap above. `rowize_from_word_list`
(the fallback rung, reached only when the PRIMARY rung's band-clipped
`find_tables` rejects or empties for a band — e.g. GH-146's ruling-line
character-destruction failure mode) inherits a pre-existing, page-sized
density floor in `_rowize_segment` (`>= 9` raw numeric tokens per segment,
`reconstruct.py:2412`). A narrow band — e.g. a one-value-column table like
the motivating page's actual TABLE A5 ("Measure/Correlation, 3 rows") —
can fall below it where the merged whole page would not, reverting that
page to the pre-GH-152 merge. **Conditional, not absolute:** measured
directly that a clean page (no ruling lines) splits this exact shape
correctly via the PRIMARY rung, which does not consult the floor at all —
the gap only bites when `find_tables` also rejects the band. **Whether the
real A5/A6 page is affected is UNKNOWN**, unverified rather than guessed
(checking needs the corpus, out of bounds here). The floor itself is NOT
changed — rescaling it per band is GH-780's separate design decision, not
this ticket's; filed by team-lead, cross-linked here and to GH-418. Pinned
as a tripwire (documents current behaviour, not desired behaviour) in
`tests/test_gh152_column_aware_rowize.py::TestGH780DensityFloorTripwire`.
See `docs/log/2026-09-16_152.md` for the full trace.


## Stream B — evidence

### TICKET-B1 — end-to-end on the motivating page · NOT SATISFIABLE · depends-on: A2 · wave 3
**Problem:** Must be demonstrated on the real page, not only a synthetic one.
**Do:** Measure p31 word recall before/after and assert the 54 previously-missing
tokens (`Measure`, `Correlation`, `Mean`, `SD`, `Guidance`, `Legislative`, `0.95`,
`0.78`, …) are present.
**Files:** `tests/test_side_by_side_tables_gh152.py`
**Done when:** p31 recall ≥ 95% and the named tokens are all present; recorded in `logs/`.


## 2026-09-16 dispatch note

A1 and A2 landed together (`src/socr/tables/reconstruct.py`,
`tests/test_gh152_column_aware_rowize.py`,
`tests/test_gh152_reconstruct_band_clip.py`). The detector
(`_detect_column_gutter`) and the label-column guard (`_has_row_labels`) are
shared by both merging rungs, exactly as A1 specified, and neither reuses
`has_numeric_columns` / `_MIN_LANES_PER_ROW` as the per-band gate, per the
wave-3 ruling.

B1 is NOT SATISFIABLE under this dispatch's instruction (synthetic fixtures
only, do not open the `2025__haim` corpus paper). If the ticket owner wants
B1's specific evidence (p31 recall, the 54 named tokens), it needs a
corpus-machine run they authorise separately — this is not an oversight.

A residual gap was found and is NOT fixed here: GH-418 (silent drop of a
word beyond every lane's snap radius, already open, already twice attempted
and rejected in #342) still fires on a two-table page if `_detect_column_gutter`
itself fails to find the gutter (e.g. unusual line/block grouping). A2's
band split makes the gutter-detected case safe; the gutter-not-detected case
is #418's pre-existing, separately-tracked defect, not new. See
`docs/log/2026-09-16_152.md`.
