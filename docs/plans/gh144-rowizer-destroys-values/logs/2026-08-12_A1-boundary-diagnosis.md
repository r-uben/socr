# GH-144 TICKET-A1 — boundary misplacement diagnosis

Date: 2026-08-12
Scope: diagnosis only. No files under `src/` touched. No new tests added, no existing
test edited. All probe scripts ran from `/tmp/gh144_a1/` and are not part of this
commit's diff.

## 1. Checkout state

Working tree at the time this diagnosis was performed: `main` @
`775c260a61a4e906a401c0b9781e66507716b55d` (`fix(145): relegate counter-directional
text instead of deleting it`).

The plan documents for this ticket (`STATUS.md`, `TICKETS.md`) do **not** exist on
`main` — the `gh144-rowizer-destroys-values/` folder on `main` contains only an empty
`logs/`. They exist on branch `docs/plans-wave1-defects` @
`9cb4d6d524079e9c371185c9bbb0f7b28097bcce` (one docs commit ahead of the `a073408`
commit the dispatching ruling cited). Both were read via
`git show docs/plans-wave1-defects:docs/plans/gh144-rowizer-destroys-values/{STATUS,TICKETS}.md`
— the branch was never checked out and the working tree was never switched away from
`main`/its ticket branch.

This reflects a parallel session moving the shared checkout mid-ticket (this repo's
tickets are dispatched against a single shared clone, not one worktree per ticket).
**This log is being written on `main`'s working tree but semantically belongs on
`docs/plans-wave1-defects`, where `STATUS.md`/`TICKETS.md` live.** The implementer
must not commit this file without coordinator sign-off on which branch receives it —
see the final section.

## 2. Synthetic fixture — rung-first ladder diagnosis

Fixture: `tests/test_region_overlap_gh145.py::table_page`, rebuilt verbatim in
`/tmp/gh144_a1/probe.py` (fixture-construction code copied, not imported — the test
file itself was not modified or executed for this purpose beyond the smoke check in
the test plan). A whitespace-gutter, unruled six-row table (`3M`…`5Y Treasury yield`,
three decimal columns, each row followed by a parenthetical row) with two notes lines
beneath it.

### Rung-by-rung trace through `extract_structured` (`src/socr/core/born_digital.py:1027-1101`)

1. **`page.find_tables()` (default lines strategy)** → `0` tables. Confirmed by
   direct call: `PROBE1 default find_tables count: 0`. The lane-stacked
   (`_is_lane_stacked`, born_digital.py:334) branch is therefore never entered — there
   is nothing to check for lane-stacking.
2. **`reconstruct_table_regions(page)`** (`src/socr/tables/reconstruct.py:92-128`)
   is called next because `table_regions` is empty. It **wins with exactly 1 region**:
   `PROBE2 reconstruct_table_regions count: 1`, `rect = Rect(60.0, 96.96, 227.60,
   313.52)`. Internally: `page.find_tables(vertical_strategy="text",
   horizontal_strategy="text")` returns one table over that same bbox; `table.extract()`
   produces a 27-row raw grid; `_clean_grid` (reconstruct.py:165) drops the fully-empty
   interleave rows; `_looks_tabular` (reconstruct.py:192) accepts it (6/8 non-empty
   rows have ≥2 numeric cells, well above `_MIN_DATA_ROW_FRAC`); `_grid_to_markdown`
   (reconstruct.py:252) renders it.
3. Because `table_regions` is non-empty after step 2, **`rowize_from_words_chart_aware`
   is never called** — the `if not table_regions:` guard at born_digital.py:1097
   short-circuits.

### The loss

Raw decimal multiset (`page.get_text("text")`, regex `\d+\.\d+`):
`Counter({'0.06': 4, '1.06': 4, '0.04': 4, '0.67': 2, '0.85': 2, '0.61': 2, '0.80': 2,
'0.05': 2, '1.00': 2, '0.94': 2, '1.10': 2, '1.02': 2, '0.73': 2, '0.64': 2, '0.09': 2})`
— 30 tokens total, each label value appearing twice (once as the bare value, once in
its parenthetical).

`extract_structured` output multiset loses **exactly**:

```
Counter({'0.67': 1, '0.85': 1, '1.00': 1, '1.10': 1, '1.06': 1, '0.73': 1})
```

— each of the six lost values had two occurrences in the raw text (the row's own
value and its parenthetical repeat) and exactly **one** of the two survives; the
parenthetical copy always survives, the bare row-label copy is always the one
destroyed. All parenthetical values, all second/third-column values, and all six
labels themselves survive intact.

**Falsification check for "not just a downstream cleaning/composition problem":**
the *producer* (`table.extract()`, before `_clean_grid`/`_grid_to_markdown` run at
all) already emits the split fragments. Row 0 of the raw grid is
`['3M', 'Treasury', 'yield 0', '.67', '0.61', '0.06']` — `.67` is already sundered
from `yield 0` at the `table.extract()` call itself. `_clean_grid` only strips
whitespace and empty rows/columns; `_grid_to_markdown` only renders. Neither touches
cell content. The value is destroyed at **grid construction** (the text-strategy
`find_tables` call), not by any later repair/composition step.

### Per-value boundary-vs-native-bbox evidence (all six)

`table.rows[n].cells` for this table is the **same 6-column boundary set for every
row** (the grid is a fixed rectangular table): column boundaries at native-PDF x
`60.0 | 73.51 | 112.91 | 152.05197143554688 | 177.55796813964844 | 210.08 | 227.60`.
The label→value column boundary is `x = 152.05197143554688` for all rows.

| Value | Row (label) | Native word bbox (x0, y0, x1, y1) | Boundary x | Strictly inside? |
|---|---|---|---|---|
| 0.67 | 3M Treasury yield | (146.526, 94.325, 164.040, 106.691) | 152.05197… | **yes** (146.53 < 152.05 < 164.04) |
| 0.85 | 6M Treasury yield | (146.526, 126.325, 164.040, 138.691) | 152.05197… | **yes** |
| 1.00 | 1Y Treasury yield | (145.032, 158.325, 162.546, 170.691) | 152.05197… | **yes** |
| 1.10 | 2Y Treasury yield | (145.032, 190.325, 162.546, 202.691) | 152.05197… | **yes** |
| 1.06 | 3Y Treasury yield | (145.032, 222.325, 162.546, 234.691) | 152.05197… | **yes** |
| 0.73 | 5Y Treasury yield | (145.032, 254.325, 162.546, 266.691) | 152.05197… | **yes** |

(`1.06` also occurs a second time on the page as the 2Y row's *Real*-column value,
word bbox x0 = `177.55796813964844`, i.e. exactly at the *next* boundary and fully
inside that column's cell — that occurrence is **not** split, which is why the raw→out
loss for `1.06` is exactly 1, not 2. Confirms the destruction is per-token-position,
not per-string.)

This is the canonical case the ratified contract named — `0.67`'s bbox
`146.53..164.04` straddling boundary `x=152.05` — reproduced exactly, and shown to
hold identically (same boundary x, same column) for the other five.

### Control: the rowizer is lossless on this fixture

`rowize_from_word_list(page.get_text("words"))` (`reconstruct.py:710-800`), run
directly on the same page's raw words (bypassing the ladder, i.e. simulating what
*would* happen if it had won): `PROBE4 rowize output decimal multiset` reproduces the
**full raw 30-token multiset with zero loss**. `rowize lost: Counter()`.

The rowizer's snap-radius logic (`_rowize_segment`, reconstruct.py:~867,
`_LANE_X_TOL_PT * _LANE_SNAP_MULT` snap margin) is therefore **not implicated** in
this fixture's loss — it never ran (`reconstruct_table_regions` won the ladder first),
and when forced to run on the identical words it does not lose the values.

## 3. Required silence finding (house-rule breach)

On this exact lossy page:

- `verify_native_table_region(page, region_md, region_rect)` (`src/socr/tables/native_verifier.py:1053`)
  returns **`hard_fail=False`, `warn=False`** (`reason=""`).
- After `extract_structured` runs, `BornDigitalDetector._last_extraction_had_unverifiable`
  is **`False`**. `_verify_regions` (`born_digital.py:1241`) only ever sets its return
  value `True` on a region `hard_fail`; a clean two-tier geometry check with no
  hard-fail leaves it `False` regardless of the six missing values.
- `_check_token_coverage` (`born_digital.py:1295`) is **DEBUG-only and non-gating**
  by its own docstring ("a deterministic safety net — it never suppresses output").
  It is also **structurally blind to boundary destruction on this class of loss**:
  it iterates native `page.get_text("words")` tokens (the *original*, unsplit `0.67`
  etc.) and checks whether each token's own `(x0, y0)` falls inside `0`, `1`, or `>1`
  region rects. The word `0.67` — the native token, never mutated — sits at
  `x0=146.53`, well inside the single table region rect (`x: 60..227.6, y:
  96.96..313.52`). It registers as covered exactly once. The check never inspects the
  *grid cell text* the token was rendered into, so it cannot see that the cell reads
  `.67` instead of `0.67`. The orphan/double-counted lists stay empty for all six.

**Silent content loss surfaced at NONE of: page status, document status, metadata, or
CLI.** The page ships as a normal, unflagged, non-warned success carrying six missing
numbers. This is the GH-144 defect exactly as filed, and it independently confirms
the house rule ("no silent content loss… must surface at every level") is currently
violated by this code path.

## 4. NS PDF materialization

Per the mustNotDo constraint, no tooling was run against the ProtonDrive or iCloud
directories directly. Files were copied read-only to `/tmp/gh144_a1/ns/` first.

| File | Source | Size | SHA-256 | `fitz.Page.count` |
|---|---|---|---|---|
| QJE | `~/Library/CloudStorage/ProtonDrive-.../Papers/2018__nakamura_steinsson__high_frequency_identification_of_monetary_non_neutrality_the_information_effect__QJE.pdf` | 851,838 B | `6611c6af964edccf2a28b1fbecc1d47a227aae1fa2ae83e374359fc4ba043a38` | 48 |
| WP | `~/Library/CloudStorage/ProtonDrive-.../Papers/2018__nakamura_steinsson__high_frequency_identification_of_monetary_non_neutrality_information_effect__WP.pdf` (note: no `_the_` before `information_effect` — the contract's predicted WP filename does not exist verbatim in this folder; this is the actual sibling) | 850,826 B | `21dc8249943a48a836752e862b90fc6c13fc856a7c0d7aae8bcfa3ff7e74ee1b` | 48 |

Both sizes match the ratified contract's stated byte counts exactly.

**iCloud placeholder confirmed 0 bytes** as the contract warned:
`~/Library/Mobile Documents/com~apple~CloudDocs/library/Papers/papers/2018__nakamura_steinsson__high_frequency_identification_of_monetary_non_neutrality_the_information_effect__QJE.pdf`
is `0` bytes on disk (`ls -la` confirmed). Never opened with `fitz`.

**Page-identity mapping — resolved.** Both PDFs carry the *same* QJE journal running
head and printed pagination (`page N (1-indexed, N=1..48) ⇒ fitz index N-1 ⇒ printed
QJE page 1283 + (N-1)`), confirmed by reading the printed page-number line on each
page (e.g. PDF page 13 → printed `1295`, PDF page 42 → printed `1324`). Content was
also diff-checked page-by-page between QJE and WP: all 48 pages differ by exactly 2
characters of trailing/whitespace noise, i.e. **WP is not a distinct pre-print with
different pagination** — it is a near-duplicate re-scan carrying the *same* journal
layout and pagination as QJE, not the independently-typeset NBER/author draft the
ticket's "WP pagination, not QJE" caveat anticipated. This is itself a finding worth
flagging to the coordinator: the corpus's WP sibling for this paper does not actually
diverge from the QJE version, so the reconciliation risk the ticket warned about does
not materialize for this particular pair (it might for other papers).

Section identity check (`"TABLE" in text`, table title text) confirms the ticket's
implied p-numbers are **1-indexed PDF page numbers**, and all measured pages carry
their expected table:

| PDF page | fitz idx | Printed QJE page | Table |
|---|---|---|---|
| 13 | 12 | 1295 | TABLE I — Response of Interest Rates and Inflation… |
| 17 | 16 | 1299 | TABLE II — Allowing for Background Noise… |
| 22 | 21 | 1304 | TABLE III — Response of Expected Output Growth… (control) |
| 30 | 29 | 1312 | TABLE IV — Estimates of Structural Parameters… (control) |
| 42 | 41 | 1324 | APPENDIX TABLE A.1 — Response of Interest Rates… (loss) |
| 43 | 42 | 1325 | APPENDIX TABLE A.2 — Response to a Fed Funds Rate Shock… (loss) |
| 44 | 43 | 1326 | APPENDIX TABLE A.3 (control) |

All measurements below use `QJE.pdf` (the file the ticket's `p13`/`p17`/`p42`/`p43`
values were written against, since WP carries identical pagination and did not need
to be substituted).

## 5. NS measurements

### Rung identification (mirrors §2's ladder, replicated from the real
`extract_structured` code path, `src/socr/core/born_digital.py:1027-1101`)

| Page | default `find_tables()` | lane-stacked? | `reconstruct_table_regions` | chart-aware rowizer needed? | **Winning rung** |
|---|---|---|---|---|---|
| p13 (Table I) | 0 | — | 0 (text-strategy table found but rejected: `_looks_tabular=False`, whole-page over-capture) | yes | **rung2: `rowize_from_words_chart_aware`** |
| p17 (Table II, loss) | 0 | — | 1 | no | **rung1: text-strategy `reconstruct_table_regions`** |
| p22 (Table III, control) | 0 | — | 0 | yes | rung2 |
| p30 (Table IV, control) | 0 | — | 0 | yes | rung2 |
| p42 (Appendix A.1, loss) | 0 | — | 1 | no | **rung1: text-strategy `reconstruct_table_regions`** |
| p43 (Appendix A.2, loss) | 0 | — | 1 | no | **rung1: text-strategy `reconstruct_table_regions`** |
| p44 (Appendix A.3, control) | 0 | — | 1 | no | rung1 |

All three loss pages win on the **same rung as the synthetic fixture** —
`reconstruct_table_regions`'s text-strategy grid, not the word-geometry rowizer.
The rowizer never wins on any of the seven measured pages of this paper.

### Raw-vs-output decimal loss, page level (`extract_structured` output vs raw `get_text("text")`)

| Page | raw decimals | lost (count) | verifier result |
|---|---|---|---|
| p13 | 54 | 0 | `hard_fail=True` (`value_guard_multiset_mismatch`, 4 paired rows) — **flagged but zero actual loss** |
| p17 | 108 | 5 lost: `{'0.08':1,'0.45':1,'0.35':1,'0.38':2}` | `warn=True` (`ambiguous_lane_count_mismatch`), not hard_fail |
| p22 | 8 | 0 | none |
| p30 | 65 | 0 | `hard_fail=True` (`value_guard_multiset_mismatch`, 6 paired rows) — flagged, zero loss |
| p42 | 152 | 35 lost, e.g. `0.67:1, 0.85:1, 1.00:1, 1.06:1, 1.21:1, 0.24:1, 0.36:2, …` | `warn=True` only, not hard_fail |
| p43 | 54 | 13 lost: `0.50:1, 0.16:4, 0.10:1, 0.41:1, 0.11:2, 0.12:1, 0.09:1, 0.07:1, 0.13:1` | `warn=True` only, not hard_fail |
| p44 | 54 | 0 | `warn=True` (same ambiguous_lane pattern as p43), zero loss |

Two falsifications worth recording explicitly:

- **p13 and p30 hard-fail the verifier yet lose zero values.** The `hard_fail` gate
  and actual content loss are not the same signal on this document — a page can
  hard-fail (flagged, would route to D3 fail-closed if wired through) while still
  being numerically lossless, and (per §3, generalised here) a page can lose values
  while passing the verifier clean. p17's 5-value loss and p42/p43's much larger
  losses all ship with **`hard_fail=False`** — the same silence class as the synthetic
  fixture, just with a `warn` breadcrumb (DEBUG-level, non-gating) instead of total
  silence.
- **p44 (control) also warns `ambiguous_lane_count_mismatch` with zero loss.** The
  warn signal alone does not distinguish loss from no-loss pages; it fires on lane-count
  geometry regardless of whether values were actually destroyed.

### Per-value boundary evidence — p43 (Appendix A.2, `reconstruct_table_regions` / text-strategy winner)

Scanning every native numeric token on the page whose bbox lies inside the winning
table's bbox, and checking whether any `table.rows[n].cells` column boundary falls
strictly inside that token's own native bbox (same lever as §2, real document):

| Value | Native bbox (x0,y0,x1,y1) | Column boundary inside | Net page-level loss confirmed? |
|---|---|---|---|
| 0.50 | (235.18, 128.87, 250.68, 136.84) | 246.435 | yes (in the lost multiset) |
| 0.59 | (235.18, 148.80, 250.68, 156.77) | 246.435 | no (another `0.59` elsewhere survives) |
| 0.41 | (235.18, 168.72, 250.68, 176.69) | 246.435 | yes |
| 0.48 | (235.18, 188.65, 250.68, 196.62) | 246.435 | no (masked by a surviving duplicate) |
| 0.38 | (235.18, 208.57, 250.68, 216.54) | 246.435 | no (masked) |
| 0.11 | (235.18, 228.50, 250.68, 236.47) | 246.435 | yes (2 of its occurrences lost) |
| 0.29 | (235.18, 272.34, 250.69, 280.31) | 246.435 | no (masked) |
| 0.07 | (235.18, 292.26, 250.69, 300.23) | 246.435 | yes |

All eight tokens sit in the **same column** and are split by the **same single
boundary**, `x = 246.43504333496094` — the identical single-boundary-inside-token
mechanism as the synthetic fixture's `x = 152.05197143554688`. Four of the eight
register as a *net* page-level loss (the Counter diff is nonzero); the other four have
a duplicate value elsewhere on the page that masks the Counter-level symptom even
though the mechanism destroyed that specific token's cell exactly the same way. This
is stated explicitly rather than only citing the four that show as net loss, because
the destructive mechanism operates on all eight regardless of whether a coincidental
duplicate elsewhere hides it from a naive multiset diff — the same masking risk exists
in any real table with repeated coefficient values (e.g. `0.11` appearing twice on a
page with 54 raw decimals is not rare).

Combined with the six synthetic values in §2, this gives **14 destroyed/mechanism-confirmed
values with per-value boundary-vs-bbox evidence**, exceeding the ticket's ≥10 floor.

### p42 (Appendix A.1) — a second, distinct defect present on the same page

p42's `reconstruct_table_regions` region is **not a clean single-boundary split** like
p43/synthetic. Its `table.extract()` raw grid shows the page's *rotated running head*
("THE QUARTERLY JOURNAL OF ECONOMICS", printed sideways in the margin) merged as
extra spurious columns into the **same** table object that spans the whole page
(`y: 58.2..578.4`, i.e. essentially the full page height). This produces rows where
single characters of the sideways running head are stacked one-per-row down extra
columns, alongside genuinely well-formed data rows (e.g. row 33 of the raw grid reads
`Ba | Nomin | 0.67 | (0.14) | 0.85 | (0.11) | 1.00 | (0.14) | 1.10 | (0.33) | 1.06 |
(0.36) | 0.73 | (0.20) | …` — **intact**, not split). Manual inspection confirms this
page's `0.67` etc. *do* survive in the primary grid row; the specific instances the
page-level Counter reports lost (§ table above, 35 values) come from **other**
occurrences of the same table split across duplicate/secondary passes of the
over-captured grid, not from the primary row.

This is the `_looks_tabular` localization-guard comment's own documented failure mode
("PyMuPDF's text-strategy grids EVERYTHING on a page by whitespace alignment") firing
in a **more severe** form than the clean single-boundary split: on this page the guard
did not reject the merge (the data-row fraction stayed high enough despite the
character-shredded margin columns). This is a **related but distinct** defect from
the boundary-inside-token mechanism A2 targets, and is flagged for the coordinator
rather than folded into A2's scope — fixing lane-boundary-vs-token placement will not
by itself fix a whole-page column over-capture that swallows a rotated running head.

### Header y-extent (#146 residual, feeds A2b) — p13, Table I

Measured region (winning rung: rung2, `rowize_from_words_chart_aware` →
`rowize_from_word_list` → `_rowize_segment`): `y0 = 128.87`, `y1 = 368.32`.
Header-band words directly above it: `"Nominal"`, `"Real"`, `"Inﬂation"` at
`y0 = 112.14, y1 = 120.11`.

**Named code path that excludes the header band:** `rowize_from_word_list`'s
y-gap segmentation (`src/socr/tables/reconstruct.py:747-767`). Reproducing the exact
segmentation on this page (same gap formula the function uses:
`split_threshold = max(_SPLIT_GAP_MULT * median_gap, _SPLIT_GAP_MIN_PT)`):

- `median_gap` on this page = `10.0` pt → `split_threshold = max(1.5 × 10.0, 10.0) =
  15.0` pt.
- The gap between the header row's y-group (`~112`) and the first data row's y-group
  (`~129`) is **17 pt** — *above* `split_threshold`, so the header is split into its
  **own segment** (`y = 81..112`, containing `TABLE I RESPONSE INTEREST RATES … Nominal
  Real Inﬂation`), separate from the data segment (`y = 129..342`).
- That header-only segment is then handed to `_rowize_segment` (reconstruct.py:801),
  which requires `len(num_words) >= _MIN_LANES_PER_ROW * _MIN_TABLE_ROWS` numeric
  tokens to even attempt lane detection (reconstruct.py:~815). The header segment has
  **zero** numeric tokens (`"Nominal"`, `"Real"`, `"Inﬂation"` are words, not numbers),
  so `_rowize_segment` returns `None` immediately and the header band is silently
  dropped **before** it is ever a region candidate — it never reaches
  `_looks_tabular`.

**This falsifies the ticket's assumption that the gap is "~9 pt, below the 10 pt
floor."** The measured gap on this page is **17 pt**, which is *above* both
`_SPLIT_GAP_MIN_PT` (10 pt) and the page-derived `1.5×median_gap` threshold (15 pt).
The header is excluded not because a too-small gap fails to split it out, but because
the gap-based segmenter correctly (by its own logic) isolates it into a segment that
then has no numeric content to build a grid from. `_SPLIT_GAP_MIN_PT`/`_SPLIT_GAP_MULT`
are not misconfigured for this page; the segment that receives the header is simply
discarded downstream for lacking numbers, which is a different mechanism than a
misplaced split boundary.

## 6. Answers to STATUS.md's two mandated questions, plus A2 guidance

**Q1 — Does full-page-width row clustering cause the false gutter?**
**NO**, on every page measured (synthetic and all seven NS pages). On the synthetic
fixture, default `find_tables()` is empty and the text-strategy `reconstruct_table_regions`
wins outright — the word-geometry rowizer never runs, so no row-clustering geometry
of its own is even evaluated for the six lost values. On NS, the three loss pages
(p17, p42, p43) all win on the **same text-strategy rung**, not the rowizer; the one
page where the rowizer *does* win (p13) is lossless on the numbers it does emit — its
only defect is the separate header-exclusion mechanism in §5, which is a segment
receiving zero numeric words, not a false gutter inside a populated row. **Nothing
measured here justifies GH-152's ladder inversion** (running the rowizer before
text-strategy reconstruct); the rowizer is not implicated in any of the loss cases.

**Q2 — Which code path excludes the header band?**
Measured and named for p13/Table I: `rowize_from_word_list`'s y-gap segmentation
(`reconstruct.py:747-767`) splits the header into its own y-segment (gap 17 pt,
above both the fixed floor and the page-derived threshold — not below either), and
`_rowize_segment` (`reconstruct.py:~815`) then discards that segment for having zero
numeric tokens, before `_looks_tabular` is ever consulted. See §5 for the full trace.

**A2 guidance.** The rejection predicate A2 needs is **native-token multiset loss
against the `table.extract()` grid** — not a boundary-intersection *count*.
`table.rows[n].cells` already exposes each cell's x-boundaries; the fix is to detect,
per numeric native token, whether any lane boundary falls strictly inside that
token's own bbox (exactly the check used throughout this log), and on detection fall
through — inside `reconstruct_table_regions` (`reconstruct.py:92-128`) or its caller
gate in `extract_structured` (`born_digital.py:~1076-1082`) — to the already-proven-lossless
`rowize_from_word_list` / `rowize_from_words_chart_aware` path for that region.

**Explicit warning on A2's ticket text as written:** "constrain lane boundaries to
fall in whitespace between word bboxes... a candidate boundary intersecting a word's
bbox must be moved to the nearest gap" targets `_rowize_segment`'s lane-boundary
placement. `_rowize_segment` is **lossless on both the synthetic fixture and the real
document** (§2 control, and it is the winning producer on none of the three NS loss
pages). Implementing A2 exactly as scoped would touch code that is not the source of
the loss — it would pass code review and CI (the strict-xfail test targets the
synthetic fixture, which never reaches the rowizer either) while leaving the actual
defect, the text-strategy grid built by `page.find_tables(vertical_strategy="text",
horizontal_strategy="text")` inside `reconstruct_table_regions`, untouched. **The fix
belongs in `reconstruct_table_regions`'s handling of the text-strategy grid** (or the
`extract_structured` gate around it), not in `_rowize_segment`.

Separately, p42 shows a related-but-distinct whole-page column over-capture (rotated
running head merged into the data table's grid) that boundary-vs-token detection
alone will not fix — flagged for the coordinator, out of A1's own scope to resolve.

---

## Handoff

This log was written on `main`'s working tree (branch `docs/144-a1-boundary-diagnosis`,
cut from `main` @ `775c260`) but belongs on `docs/plans-wave1-defects`, where
`STATUS.md`/`TICKETS.md` for this plan actually live. **No commit has been made.**
Awaiting coordinator sign-off on how this file should land on
`docs/plans-wave1-defects` before any commit happens.
