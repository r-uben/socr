# 2026-09-06 — TICKET-C1: baseline-aligned adjacent blocks are one line (GH-592)

## Problem

A tab-aligned two-column attendee list (e.g. `PRESENT: Mr. Greenspan, Chairman`
laid out as a "Mr./Ms." column and a separate name column) native-extracts via
`page.get_text("text")` as one column's lines in full, then the other's — 11
bare `Mr.`/`Ms.` lines followed by 11 names on the Fed 1989-11-14 minutes p.1
fixture — and ships `SUCCESS` with no flag. `page.get_text("text", sort=True)`
fixes this one case but was rejected: it changes whitespace/padding
byte-for-byte on every page and would incorrectly interleave genuine
independent multi-column prose.

## Fix

`src/socr/core/born_digital.py`: a geometry-only line assembler
(`_assemble_prose_with_aligned_runs` and helpers) runs before the existing
no-table-regions flat-text path. It walks `page.get_text("dict")` blocks,
finds maximal contiguous block ranges that form a bijection of baseline
-aligned lines across two x-disjoint column bands (mode-seeded nearest
-neighbor clustering on trimmed word-level left edges — see
`_cluster_two_bands`'s docstring for why naive largest-gap clustering on
either edge fails), and merges each row left-to-right. It returns `None`
whenever no such run is found, so the caller falls through to the
**unmodified** `page.get_text("text").strip()` path — every page without this
exact defect shape is byte-for-byte identical to before this ticket.

No lexical rule anywhere ("PRESENT", "Mr.", a name pattern) — the detector is
purely geometric, per the ticket's requirement.

## Measurement: `ALIGNED_RUN_GAP_MAX_WORD_SPACES`

The discriminator between "this is one row split into two columns" and "this
is a genuine two-column page layout" is the horizontal gap between the two
blocks, expressed as a multiple of the page's own median word-space width
(`_median_word_space_width`, the median horizontal gap between adjacent words
sharing one line — not a hardcoded constant, always measured on the page
under test).

### Positive case: Fed 1989-11-14 minutes, page 1

`/Users/rubenffuertes/Data/socr/fed-sample-2026-09-05/in/fed-1989-11-14-minutes.pdf`,
page 1 (real fixture; not committed to the repo — see Deviations below).

Median word-space width on this page: **6.42pt**.

The aligned run found spans dict blocks 6–10 (11 rows). Per-row honorific→name
gap, in points and as a multiple of the word-space width:

| Left | Right | gap (pt) | ratio (× word space) |
|---|---|---|---|
| `PRESENT: Mr.` | `Greenspan, Chairman` | 6.36 | 0.99 |
| `Mr.` | `Corrigan, Vice Chairman` | 7.00 | 1.09 |
| `Mr.` | `Angell` | 6.00 | 0.93 |
| `Mr.` | `Guffey` | 7.00 | 1.09 |
| `Mr.` | `Johnson` | 7.00 | 1.09 |
| `Mr.` | `Keehn` | 7.00 | 1.09 |
| `Mr.` | `Kelley` | 11.31 | 1.76 |
| `Mr.` | `LaWare` | 11.02 | 1.72 |
| `Mr.` | `Melzer` | 10.02 | 1.56 |
| `Ms.` | `Seger` | 6.36 | 0.99 |
| `Mr.` | `Syron` | 5.92 | 0.92 |

Range: **0.92–1.76** word spaces.

### Negative control: real two-column body text (round 2 — corrected)

No two-column-journal PDF fixture was found anywhere in this repo (`docs/`,
`tests/fixtures/`, `tests/data/` searched — see Deviations), so an earlier
pass of this log reported a hand-built synthetic negative control at ≈17–89×
a word space. That synthetic gutter was far wider than any real two-column
journal actually uses, and understated how close the real margin is. This
round replaces it with measurements against three real two-column papers
already on disk (`~/Library/Mobile Documents/com~apple~CloudDocs/Library/Papers/papers/`),
using the same `_median_word_space_width` / per-row gap measurement as the
positive case above, on each paper's first two-column body-text page:

| Paper | word space (pt) | narrowest body-text column gap (pt) | ratio (× word space) |
|---|---|---|---|
| Fama & French 1997, JFE, p.2 | 4.32 | 9.94 | **2.30** |
| Morris & Shin 1998, AER, p.2 | 2.36 | 10.02 | 4.25 |
| Grossman & Stiglitz 1980, AER, p.2 | 4.15 | 18.86 | 4.54 |

("Narrowest" = the smallest single-row gap between adjacent columns found on
that page — the binding case, since `_try_aligned_run` requires *every* row
in a candidate run to be at or under the threshold; one wide row does not
protect a page whose narrowest row is close to it.)

The narrowest real gutter measured, across all three papers and every row
examined, is **2.30×** a word space (Fama-French).

### Derived value — and what actually carries the safety margin

```python
ALIGNED_RUN_GAP_MAX_WORD_SPACES = 2.0
```

The measured positive ceiling is 1.76× (Fed fixture); the measured negative
floor is 2.30× (Fama-French). **The gap threshold alone has only ~15% margin
on each side** — `(2.0 − 1.76) / 1.76 ≈ 14%` above the positive case, and
`(2.30 − 2.0) / 2.30 ≈ 13%` below the negative case. This is not the wide
margin an earlier round of this log claimed (that round measured a synthetic
gutter, not a real one); stated plainly, a real page whose column gutter
happens to run a little narrower than Fama-French's would clear the gap
threshold on some rows.

What actually protects genuine two-column prose is not the gap threshold in
isolation — it's that `_try_aligned_run` also requires a **bijection over
the whole candidate run**: every line in the range must pair up left-to-right
with no leftover, and every pair's vertical bands must overlap (same row).
Real multi-column body text reads down one column for many lines before
moving to the next, so a candidate range spanning both columns almost never
produces equal left/right counts with correct row-by-row vertical alignment
even where an individual gap is narrow — the three papers measured above
never once satisfied the bijection over more than an isolated row or two,
regardless of gap width. The gap threshold is the second gate, not the first;
the run-shape requirement (bijection + row overlap + minimum row count) is
what makes the two-column case structurally different from the aligned-run
case, independent of how tight the gap margin is.

## Round 3: Astra false-merge finding and `LABEL_COLUMN_WIDTH_SHARE`

Astra (a Codex reviewer) found that gap-and-bijection alone is not sufficient
evidence of a label/value pair. Repro (Courier 10, `insert_textbox`):

- left rect `(72,100,200,200)`: `"We stopped the trial.\nThe drug was unsafe."`
- right rect `(201,100,400,200)`: `"We approved the drug.\nThe trial was sound."`

Trimmed gaps are 3pt and 9pt against a 6pt word space (both under the 2.0×
gap threshold on at least one row), and the two blocks form a perfect 2-row
bijection with overlapping vertical bands — every check `_try_aligned_run`
had now passes. The assembler merged this into
`"We stopped the trial. We approved the drug."` /
`"The drug was unsafe. The trial was sound."` — every word preserved,
content-preservation checks green, meaning destroyed: these are two
independent sentences, not a label and its value. The assumption that a
narrow gap plus complete row-by-row alignment establishes reading order is
false — two prose columns of comparable width can satisfy both.

### Fix: `LABEL_COLUMN_WIDTH_SHARE` width discriminator

A genuine label column (`Mr.`/`Ms.`) is always much narrower than the value
column it labels. Two independent prose columns are not — each runs close to
its own full column width. `_try_aligned_run` now additionally requires the
left block's median line width to be at most `LABEL_COLUMN_WIDTH_SHARE` ×
the right block's median line width, computed only over the actual candidate
items in the block range under test (median of `x1 - x0` per line, left vs.
right cluster).

**Measurement pitfall**: the first attempt measured this ratio via
`_cluster_two_bands` over the *whole page*, not the specific candidate range
`_find_aligned_runs` operates on. On the Fed fixture this produced a
misleading 0.134 — the whole-page right cluster ended up containing only 2
outlier-wide name lines (`"Corrigan, Vice Chairman"`, `"Greenspan,
Chairman"`), excluding the true bare-name rows (`"Angell"`, `"Guffey"`, …,
median ≈35pt) that the real candidate-range run actually includes. Restricting
the measurement to the true candidate range (dict blocks 6–10, the same 11
rows tabulated above) gives the correct ratio.

| Fixture | left median width (pt) | right median width (pt) | ratio |
|---|---|---|---|
| Fed 1989-11-14 minutes p.1 (candidate blocks 6–10, restricted) | 18.0 | 35.28 | **0.51** |
| Astra's independent-columns repro | 123.0 | 123.0 | **1.00** |
| Fama & French 1997, JFE, p.2 (whole-page approx.) | — | — | 0.97–1.03 |
| Morris & Shin 1998, AER, p.2 (whole-page approx.) | — | — | 0.97–1.03 |
| Grossman & Stiglitz 1980, AER, p.2 (whole-page approx.) | — | — | 0.97–1.03 |

The three real-paper values are whole-page approximations, not true
candidate-range measurements: none of the three papers ever forms a
candidate run across the column boundary in the first place (see the
bijection argument above), so there is no "restricted range" to measure —
their body-text columns are simply close to equal width by inspection,
consistent with the whole-page clustering not being distorted by outliers
the way the Fed's mixed-content leading row was.

```python
LABEL_COLUMN_WIDTH_SHARE = 0.65
```

Positive ceiling (must merge): 0.51 (Fed). Negative floor (must not merge):
Astra's 1.0, and the real papers' ~0.97. 0.65 sits with ~27% margin above the
Fed ceiling and ~33% margin below the tightest negative case, erring toward
the conservative (lower, rejects more) side per "fail toward not merging."

### Redundancy analysis: page-text-width check (ticket item 2)

The ticket also asked to require the merged line not exceed the page's
normal text width for its block pair, and to record whether this is
redundant with the width-ratio check above. Checked directly against Astra's
exact repro page: it contains exactly 2 dict blocks total (the left and
right columns being merged) and no third "normal single-column prose" line
anywhere on the page to serve as a width reference.  With no external
reference line, a page-text-width check is inapplicable to the one fixture
it was proposed to catch — it neither strengthens nor weakens the outcome
there, so it is redundant with the width-ratio check for every case
examined (Fed, Astra, and the three real papers). Not implemented as a
separate gate: adding an un-demonstrated, uncalibrated second magic
threshold contradicts this project's "no magic numbers" convention when the
one check already in place (`LABEL_COLUMN_WIDTH_SHARE`) already catches
every known false-merge case.

## Other constants (not measured — structural bounds, justified in-line)

- `_ALIGNED_RUN_MIN_ROWS = 2` — minimum evidence for "a repeating pattern",
  not a measured threshold.
- `_ALIGNED_RUN_FAIL_STREAK_LIMIT = 4` — search-cost bound for the greedy
  run-growing loop; the Fed fixture's true run only balances (11 vs. 11)
  after all 5 underlying dict blocks are included, passing through 3
  mismatched intermediate totals first. Set with one block of margin over
  that observed case.
- `_COLUMN_SEED_TOLERANCE = 3.0` — points within which consecutive left
  edges are treated as the same column seed when clustering; font/rendering
  jitter of a shared column start is well under a point in every fixture
  examined, this is a generous multiple of that.

## Round 4: Astra residual finding and `RIGHT_BLOCK_FILL_TOLERANCE_WORD_WIDTHS` / `MEASURE_FILL_SHARE_MAX`

Astra found a residual false-merge on `619666b` (round 3's commit): a short
repeated left label ("Note" x4) beside four INDEPENDENT, unrelated, very
unequal-length sentences. This passes bijection, the gap check, AND
`LABEL_COLUMN_WIDTH_SHARE` — the left block genuinely is much narrower than
the right (ratio ~0.16), because "Note" really is short next to these
sentences. Width asymmetry alone identifies "the left block is narrow", not
"the left block is a genuine label column".

**Geometry discrepancy found while reproducing it.** Astra's rects as
reported, (72,100,105,180) for the left block and (115,100,400,180) for the
right, do NOT reproduce a merge under round-3 code. `insert_textbox`
left-aligns text at the box's own left edge but does not stretch trailing
text to the box's right edge: "Note" at Courier 10pt is 24pt wide, so it
ends its actual glyphs at x=96, not the box's declared x=105 (9pt of unused
padding inside the box). Against a right column starting at x=115, the TRUE
trimmed-word-extent gap is 115-96=19pt, i.e. ~3.17x this page's measured
6pt word-space width — already above `ALIGNED_RUN_GAP_MAX_WORD_SPACES`
(2.0x) on its own, with or without any round-4 change. Confirmed by
stashing all round-4 changes and running the literal repro against pure
`619666b`: it also returns `None` there. The literal numbers as reported
were an approximation that didn't survive precise measurement, not an
actual round-3 bug.

Swept the right block's start x (holding the left "Note" box fixed) and
found the described shape genuinely merges under round-3 code for x in
[98, 108]. x=100 was chosen (trimmed gap 4pt, ~0.67x word space) as a clean
value in that window to faithfully test the concept Astra raised — see
`_build_astra_residual_narrow_label_page`'s docstring in
`tests/test_born_digital_aligned_runs.py` for the full account of this
deviation, documented there rather than silently substituted.

**The new rule.** A genuine label's VALUE column is a list of independent
short entries, each ending wherever its own content ends. A WRAPPED
PARAGRAPH's lines (all but the last) each run up against the column's own
measure, because greedy line-wrapping fills every line until the next word
would overflow — this is the structural difference a value list does not
share. Added `_median_word_width` (median individual-word glyph width, a
different yardstick from `_median_word_space_width`, which measures the gap
BETWEEN words) as the tolerance unit, and a fill-share check: over the
right block's lines in the candidate range, the share whose x1 lands within
`RIGHT_BLOCK_FILL_TOLERANCE_WORD_WIDTHS` word-widths of the block's own
widest line's x1. If that share exceeds `MEASURE_FILL_SHARE_MAX`, decline
to merge.

**Measured fill share** (tolerance = 2.0 word-widths, per
`RIGHT_BLOCK_FILL_TOLERANCE_WORD_WIDTHS`):

| Case | Right-block shape | Fill share | Merges? |
| --- | --- | --- | --- |
| Fed 1989 p1 name column | 10 bare names + 1 long "Corrigan, Vice Chairman" outlier | 0.182 | Yes (correct — unaffected) |
| Real wrapped Fed-minutes paragraph, 4 lines beside "Note"x4 | genuine wrapped body prose | 0.75 | No (correctly declines) |
| Astra's residual (x=100 repro) | 4 independent, wildly unequal one-line sentences beside "Note"x4 | 0.50 | Yes (known, accepted residual — see below) |

`MEASURE_FILL_SHARE_MAX = 0.5` sits with wide margin above the Fed positive
(0.182), comfortably below the real wrapped-paragraph case (0.75), and
exactly at — not above — Astra's residual, so with a strict `>` comparison
the residual is NOT caught. This was deliberate: 0.5 reads naturally as "a
strict majority", and pushing the threshold down to also catch the
residual (e.g. to something at or below 0.50) would risk pulling in real
attendee-list-style value columns whose entries happen to cluster near the
column's own natural width by coincidence of short vocabulary (first names
are commonly similar lengths) — a false rejection of the very case this
detector exists to serve, in exchange for catching one contrived,
low-evidence shape (four wildly unequal, unrelated one-line sentences
beside a repeated short label is not a shape observed in any real document
examined in this or the prior rounds).

**Residual, explicitly**: Astra's narrow-label / unequal-independent-
sentences case (`_build_astra_residual_narrow_label_page`,
`test_astra_residual_still_merges_documented_known_gap`) still merges after
round 4. This is a known, accepted limitation, pinned as a test documenting
CURRENT behaviour (not a regression) so a future change to it is
measurable. Per team-lead's round-4 instruction, this was measured and
reported honestly rather than closed by loosening the threshold past what
the Fed/real-paragraph evidence supports.

**Existing-fixture repairs (not a loosening of the new guard).** Two
round-2/round-3 synthetic fixtures broke when the fill-share check first
landed: `_build_uniform_gap_bijection_page` and
`_build_uniform_width_ratio_page` both used right-column text with
insufficient length variance (near-identical or fully-identical row
lengths), which trivially maximizes fill share regardless of the actual
dimension (gap or width ratio) each test isolates. Re-verified directly
against the real Fed PDF fixture first, confirming this was a fixture-
realism gap and not a genuine threat to the Fed positive (no CONSILIUM-GATE
triggered). Fixed both fixtures by adding realistic length variance
mirroring the Fed fixture's own shape (several similar-length short entries
plus one long outlier): `_build_uniform_gap_bijection_page`'s right column
is now 4 similar-length surnames plus one long "Corrigan, Vice Chairman of
the Committee" outlier; `_build_uniform_width_ratio_page`'s right column is
now 4 identical short lines plus one long outlier (preserves the median
width each test's threshold math depends on, while dropping fill share
below the merge-blocking level).

**The complete rule, four parts** (a candidate left/right item range
merges only if ALL of):

1. **Bijection** — the range's left lines and right lines pair up 1:1 with
   no leftover on either side (`_ALIGNED_RUN_MIN_ROWS` minimum evidence).
2. **Gap** — the trimmed word-level gap between the two columns is at most
   `ALIGNED_RUN_GAP_MAX_WORD_SPACES` (2.0) times the page's own median
   word-space width.
3. **Width ratio** — the left block's median item width is at most
   `LABEL_COLUMN_WIDTH_SHARE` (0.65) of the right block's, i.e. the left
   column reads as substantially narrower (a label), not a peer column.
4. **Measure fill** (new, round 4) — at most `MEASURE_FILL_SHARE_MAX`
   (0.5) of the right block's own lines land within
   `RIGHT_BLOCK_FILL_TOLERANCE_WORD_WIDTHS` (2.0) word-widths of the right
   block's own widest line; above that share, the right block reads as
   wrapped body prose, not a value list, and the range is declined.

## Verification

- `tests/test_born_digital_aligned_runs.py` (10 tests):
  - an inline `fitz`-built attendee-list fixture asserts the merge (0 bare
    honorific lines, all 5 `Mr./Ms. <name>` pairs present) through both
    `extract_structured` and the assembler directly;
  - an inline two-column-journal fixture asserts the assembler declines
    (`None`) and that `extract_structured`'s output is byte-identical
    whether the assembler runs or is monkeypatched to return `None`;
  - a uniform-gap bijection fixture pins both sides of the gap threshold
    directly: gap 1.5× a word space MUST merge, gap 2.3× (the real
    Fama-French floor measured above) MUST NOT merge;
  - Astra's exact independent-columns repro asserts the assembler declines
    (`None`) and that `extract_structured`'s output is byte-identical
    whether the assembler runs or is monkeypatched off, matching the
    two-column-journal test's pattern;
  - a uniform-width-ratio bijection fixture pins both sides of
    `LABEL_COLUMN_WIDTH_SHARE` directly: ratio 0.5 (below the Fed ceiling)
    MUST merge, ratio 0.8 (above the threshold, below Astra's 1.0) MUST NOT
    merge.
- Manually re-verified against the real Fed fixture: `extract_structured`
  produces `PRESENT: Mr. Greenspan, Chairman` as one line, 0 bare honorific
  lines, all 11 names correctly paired, with `LABEL_COLUMN_WIDTH_SHARE`
  active — no CONSILIUM-GATE was needed.
- Manually re-verified against the three real two-column papers listed above:
  none of them produce a bijective run across the column boundary at any
  candidate range tried, regardless of gap width — confirming the run-shape
  requirement, not just the gap threshold, is what keeps them from merging.
- `PYTHONPATH=.../src ~/venvs/socr/bin/pytest tests/test_born_digital_aligned_runs.py tests/test_born_digital.py -q` (round 4, 2 new tests added: the residual repro and its real-body-text sibling) → 68 passed.
- `PYTHONPATH=.../src ~/venvs/socr/bin/pytest tests -q` (full suite, round 4) → 4246 passed, 4 xfailed, 0 failed.
- `uvx ruff@0.16.0 format --check .` → clean (585 files already formatted, 1 reformatted in this round for a long comprehension line in the new fill-share check).

## Deviations from the ticket text

1. **Test file location**: the ticket names
   `tests/core/test_born_digital_aligned_runs.py`. This repo has no
   `tests/core/` subdirectory — every test file lives flat under `tests/`
   (confirmed via `find tests -type d`, which only shows `tests/fixtures`
   and `tests/data`). Filed at `tests/test_born_digital_aligned_runs.py` to
   match the repo's actual convention instead.
2. **Negative-control fixture**: the ticket implies testing against an
   "existing" two-column journal PDF. No such fixture exists anywhere in the
   repo, so the automated regression tests use `fitz`-built synthetic pages
   (following this test module's own established pattern for synthetic
   PDFs, `tests/test_born_digital.py`) rather than a checked-in real one.
   The *measurements* backing the threshold, however, are now against real
   two-column papers already present on this machine (see above), not a
   synthetic guess — only the checked-in regression fixture is synthetic.

## Files changed

- `src/socr/core/born_digital.py` — the assembler, its helpers, and the
  no-table-regions call site.
- `tests/test_born_digital_aligned_runs.py` — new differential test file.
- `docs/log/2026-09-06_C1-aligned-runs.md` — this file.
