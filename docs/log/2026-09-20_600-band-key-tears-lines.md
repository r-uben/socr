# GH-600 — `_assign_bands` keys rows on `round(y0)`, tearing printed lines in half

**Date:** 2026-09-20
**Branch:** `fix/600-band-key-tears-lines`
**Scope:** `src/socr/tables/binding.py` (`_assign_bands` only), `tests/test_binding.py`.

## Step 0 — narrowing the whole-page bound to production scope (MEASURED)

The issue's whole-page scan (`/tmp/zz600c.py`) measured 7,611 torn printed lines corpus-wide
(3,766 numeric-on-both-sides) as an explicit UPPER BOUND — `_assign_bands` only ever sees words
inside a located table region in production (`bind(words, markdown, region=witness.box.bbox)`
at `orchestrator.py:6315`, region from `TableWitness.box`, itself from
`socr.tables.locate.locate_tables`, a purely geometric/vector detector — no OCR, no provider).

Restricting the same scan to `_words_in_region(words, box.bbox)` for every `locate_tables(page)`
box on the same 380-doc corpus (`/tmp/zz600_step0.py`):

```
pages=4419  pages_with_table_boxes=2967  table_boxes=4803
in-region printed_lines(>=2 words)=70277
ARM1 raw round(y0) key splits the line:       5374
ARM2 _assign_bands FINAL partition splits it: 4740
  of those, both halves carry a numeric:      2958
```

4,740 (2,958 numeric-both-sides) is far past the "under a hundred, stop" bar the ticket set —
proceeded to a fix.

## Design — measuring the open question before choosing

The ticket's open question: can PyMuPDF's own `(block_no, line_no)` be trusted as row evidence
more broadly than the existing numeric-free-only fold trusts it, and what breaks (two genuinely
different rows sharing one `(block_no, line_no)`; rotated/multi-column regions)?

**Naive full key swap, rejected by measurement.** A synthetic column of 29 stacked `"-"`
placeholders in the real corpus (`mpr-2011-03.pdf` p50, `(block_no, line_no)=(16, 0)`) each
round to a DIFFERENT `round(y0)` and are each their own printed row (`/tmp/zz600_key_safety.py`
found this as the single largest y0-span/height-ratio outlier, 42x). Keying purely on
`(block_no, line_no)` would collapse all 29 into one row — this PDF's own line segmentation is
not one-row-per-`(block_no,line_no)` in general, so the swap was rejected outright.

**Generalizing the existing fold instead.** The existing fold already has the right shape:
fold group B into group A only when B's line identity resolves to exactly ONE other group A,
corroborated by bbox intersection against A's own words — no distance tolerance. It was
artificially restricted to "numeric-free group folds into numeric-bearing group, never the
reverse, never numeric-into-numeric." Removing that restriction and applying the SAME
uniqueness+overlap test symmetrically (union-find, so multi-hop chains resolve) is a minimal
generalization using only evidence already in the algorithm.

Measured against the corpus (`/tmp/zz600_proto.py`, symmetric fold with NO word-count
direction): in-region torn lines dropped from 4,740 to 4,198 (2,958 → 2,829 numeric-both-sides).
Inspecting the still-torn population showed ALL of it (`/tmp/zz600_diff.py` classification: 2,829
rotated-like, 0 horizontal-like, 0 ambiguous) is rotated text — consecutive "same line" words
share x0 and vary in y0, the opposite of the vertical-overlap evidence this fold uses, so it
cannot corroborate them. This is the SAME rotated-content abstention the pre-existing docstring
already documented (`doc04 p3 "1t"` under `ROTATED PCs`), not a new gap; the affected 48 documents
are exclusively `fomcminutes*` / `fomcprojtabl*` (SEP dot-plot/projection-table chart pages,
GH-734/GH-739 territory) plus 4 lines in `doc04.pdf`, the fixture the docstring already names.
Extending the fold to a rotation-aware (x-overlap) corroboration axis is real added complexity for
a residual concentrated in already-known chart pages, out of scope here — **left as measured, not
guessed, future work**, not silently declared solved.

**Regression found and fixed: naive symmetric folding is unsafe.** Running the full symmetric
fold against the existing suite failed
`test_vertical_band_ambiguity_from_word_extents_not_lane_gap_constant`: two DELIBERATELY
y-overlapping synthetic rows ("RowA"/1.0 at y∈[100,112], "RowB"/2.0 at y∈[108,120]), both using
the test helper `w()`'s universal `block=0, line=0`, both satisfied uniqueness+overlap in BOTH
directions and got merged into one band — destroying the exact row-extent-overlap ambiguity
signal that test exists to check. Real PyMuPDF extraction keys visually distinct lines
separately, but nothing in the algorithm enforced that, and the test proves the naive version's
safety net (uniqueness) is not sufficient on its own.

**Fix: require the destination to hold STRICTLY more words** on the shared line identity than
the source (a word count, not a distance/tolerance — already-present data). Re-measured
(`/tmp/zz600_proto2.py`): 4,740 → 4,402 (338 fixed), numeric-both-sides 2,958 → 2,819 (139
fixed) — smaller than the naive symmetric fold's 542/129, the remainder being exactly the
equal-count ties the naive version was unsafely merging. Re-ran the tie fixture: two equal-size
groups never merge, `_ambiguous_bands` signal preserved. Full binding suite green (362 passed, 1
xfailed) with this version; re-measured with the ACTUAL fixed `_assign_bands` (not the prototype)
and got the identical 4,402 / 2,819 numbers, confirming the prototype and the shipped code agree.

## What changed

`src/socr/tables/binding.py::_assign_bands`: replaced the numeric-free-only, one-directional
fold with a symmetric union-find fold. A group B folds into group A iff (1) B's `(block_no,
line_no)` metadata resolves to exactly one other group A, corroborated by an exact bbox
intersection against one of A's words, and (2) A holds strictly more words on that line identity
than B. Both conditions were already-present evidence (word membership, box geometry, word
count) — no new threshold, tolerance, or proximity radius introduced. The doc04 rotated-PCs
abstention behaviour is unchanged (its ambiguity — more than one candidate destination — is
untouched by this generalization).

## Tests added (`tests/test_binding.py`)

- `test_gh600_line_key_split_torn_by_rounding_boundary_is_healed` — matched control pair: one
  printed line (7 words, "rate rose to 3 percent by 2012", numeric on both sides of the tear —
  "3" in the 6-word majority, "2012" alone) with word tops EXACTLY aligned, and the same line
  with realistic sub-point jitter (100.46 vs 100.54) straddling the `round()` `.5` boundary. A
  `_pre_gh600_assign_bands` control (the exact pre-fix algorithm, kept for comparison) is
  asserted to AGREE on the aligned input and TEAR the jittered one — proving the control fires on
  the defect and not on a correct input. The current `_assign_bands` is asserted to produce one
  band for both.
- `test_gh600_equal_size_groups_sharing_line_identity_do_not_merge` — the safety regression
  above, pinned directly: two equal-word-count groups sharing line identity and overlapping in y
  must stay two bands.

## Mutation guard

Copied `src`, `tests`, `pyproject.toml` to `/tmp/socr-600-mutant` (outside the repo). Canary:
`os.path.realpath(socr.__file__)` asserted to start with `/private/tmp/socr-600-mutant/` (macOS
`/tmp` symlink) — passed. Anchor `def _assign_bands(words: list) -> tuple[list[float],
dict[float, int]]:` asserted `src.count(anchor) == 1` before editing — passed (both times: once
for the initial symmetric-fold edit, once for the word-count-majority correction). Reverted
`_assign_bands` in the mutant copy to the exact pre-GH-600 body (verified against
`git show origin/main:src/socr/tables/binding.py`) and re-ran the new tests:
`test_gh600_line_key_split_torn_by_rounding_boundary_is_healed` FAILED (`2 == 1`— the mutant is
killed), `test_gh600_equal_size_groups_sharing_line_identity_do_not_merge` still passed (expected
— the old code never merges numeric groups either, so this safety test doesn't distinguish
old/new; it exists to pin the NEW code's safety property, not to kill this particular mutant).
Deleted `/tmp/socr-600-mutant` afterward.

## Test results

Baseline measured directly (not a quoted figure): an isolated copy of this worktree with
`src/socr/tables/binding.py` and `tests/test_binding.py` reset to `origin/main` content, full
suite run — **5,721 passed, 4 xfailed**. With the fix in place: **5,723 passed, 4 xfailed** — a
clean `+2` delta (exactly the two new GH-600 tests), zero regressions, zero removed/altered
tests. With the fix:

- `tests/tables/` + every `test_*binding*`/`test_gh*band*` file: 362 passed, 1 xfailed (was 361
  passed, 1 xfailed before the word-count-majority correction turned the one regression back
  green).
- Full suite (`tests/`): **5,723 passed, 4 xfailed**, 0 failed.
- `uvx ruff@0.16.0 format --check .`: `751 files already formatted` — clean.

## Addendum — delegation to `cluster_band_words` investigated and rejected (MEASURED)

After the above was committed, a course-correction narrowed the "no distance tolerance" ban:
it was aimed at inventing a NEW magic number, not at reusing `row_corroboration.cluster_band_words`
— an already-shipped, GH-600-aware, data-derived (median word height × the named
`_ROW_BAND_TOLERANCE_FRACTION`) y-centre clusterer, factored out in #652. The preferred shape, if
it could be made to work, was `_assign_bands` delegating its row partition to it instead of
`round(word[1])`. This was investigated and MEASURED, not assumed:

**Re-measured Step 0 in this session** (`/tmp/zz600_fold_redundancy.py`, a single paired pass over
the same 380-doc corpus, computing all three arms from the same in-region word list per printed
line — not independent marginal totals): `70,278` in-region multi-word printed lines,
`5,376` torn under naive `round(y0)` (`2,958` numeric-both-sides). This total-torn count differs
from the number recorded above (`5,374`/`4,740`) — table-region membership shifted slightly as
`origin/main` moved from `205eeb4` to `e207a8a` during this ticket (unrelated merges landed) — but
the numeric-both-sides figure, the ticket's actual content-loss concern, is IDENTICAL (`2,958`
both times), so the acceptance bar and the measured gain below are unaffected.

**No import-cycle risk.** `row_corroboration.py` imports only from `native_verifier`; nothing in
`socr/tables/` imports `binding` from `row_corroboration`, so `binding.py` importing
`cluster_band_words` would be safe.

**Delegation tried three ways, all measured against the same corpus:**

1. **Pure `cluster_band_words` in place of `round(y0)`, no fold at all.** Recovers almost none of
   the ticket's target defect: `4,555` torn (`2,953` numeric-both-sides) — a reduction of only
   `5` numeric-both-sides lines out of `2,958` (0.17%), versus `139` (4.7%) from the committed
   fold. The GH-600 half-point tear is not, in practice, what this clusterer's tolerance mostly
   catches.

2. **`cluster_band_words`'s bands used as the base partition, with the existing metadata+bbox+
   word-count-majority fold layered on top** (same guards as the committed fix, operating on
   cluster indices instead of `round(y0)` keys): `2,327` torn (`947` numeric-both-sides) — looks
   like the best number of any arm measured, but is an artifact, not a fix (see below).

3. **Whether `cluster_band_words` ever chains adjacent, genuinely distinct printed rows into one
   band** (`/tmp/zz600_chain_check.py`): measured directly on the same corpus by checking, per
   returned band, whether it contains words from more than one `(block_no, line_no)` identity with
   NO vertical bbox overlap between them — i.e., the band spans a real y-gap. **211 such events
   across 4,630 in-region table boxes.** Inspected six concrete instances
   (`/tmp/zz600_chain_inspect.py`); all are genuine content collapse, not benign:
   `mpr-2007-02.pdf` p7 merges a 6-row table legend/header block ("MEMO" / "Indicator" / "2006
   actual" / "Central" / "Central" / "Range", `y0` 634.6→645.6, each its own printed row) into ONE
   band; `mpr-2007-07.pdf` p29 merges three ordinary PROSE paragraph lines the located table's
   bbox happens to catch at its edge into one band. This is exactly the failure mode
   `_assign_bands`'s docstring says `round(y0)` cannot produce ("it cannot make a run of nearby
   printed rows collapse into one band") — `cluster_band_words`'s tolerance is derived from the
   REGION's overall median word height, which is a single constant across a region that can mix
   dense multi-row headers/legends (narrow true line pitch) with ordinary body-pitch table rows;
   on the dense sub-area the tolerance exceeds the true pitch and multiple distinct rows fall
   inside one clustering window.

   Arm 2's apparently-best "torn" number is this same defect showing up as a false improvement:
   once `cluster_band_words` has already merged several distinct printed lines into one
   mega-band, no single line's own words can ever land in TWO different bands, so the "torn"
   metric trivially collapses — not because the rounding tear is healed, but because the
   partition has stopped distinguishing rows at all in the affected region. A metric that
   improves by discarding the row structure it is meant to protect is not a fix.

**Conclusion: delegation rejected, committed fix (`cd0caf1`) kept unchanged.** `round(word[1])`
stays the base partition specifically because it is provably incapable of ever merging two
distinct printed rows (a rounded-integer bucket collision requires two rows within 1pt of each
other's y0, i.e., visually the same line); `cluster_band_words`'s median-height tolerance carries
no such guarantee and is measured, not assumed, to violate it on real corpus pages. This is not a
rejection of "reuse what exists" — the committed fold already reuses only pre-existing evidence
(`_boxes_vertically_overlap`, PyMuPDF's own `(block_no, line_no)`, a word count) and introduces no
new constant; `cluster_band_words` was evaluated in good faith as the preferred shape and
measured unsafe for this specific call site's invariant, which a different call site
(`row_corroboration`'s own row-corroboration matching) can tolerate but `_assign_bands` cannot.

**The pre-existing metadata/bbox/word-count fold is therefore NOT redundant — it is the only
arm measured to recover the numeric-both-sides defect without introducing new content loss.**
It does not double-count with anything, since no clusterer delegation is in the shipped code to
double-count against.

All corpus-wide numbers in this addendum are freshly measured in this session
(`/tmp/zz600_fold_redundancy.py`, `/tmp/zz600_chain_check.py`, `/tmp/zz600_chain_inspect.py`,
`/tmp/zz600_combined.py`, `/tmp/zz600_adversarial.py`, `/tmp/zz600_doc04_detail.py`), all scratch,
deleted after use — none are quoted from memory. The 29-dash (`mpr-2011-03.pdf` p50) and doc04
rotated-PCs adversarial fixtures were also re-run directly against `cluster_band_words` for
completeness: both happen to band identically to the committed fix on those two fixtures (58/58
bands and 13/13 bands respectively, word-for-word identical content on doc04) — `cluster_band_words`
is not unsafe on every input, only on the header/legend-density pattern above, which is why the
211-event corpus scan (not just these two known fixtures) is the evidence that matters.

## Explicitly inferred, not measured

- That real PyMuPDF extraction "virtually always" gives distinct visual lines distinct
  `(block_no, line_no)` is inferred from PyMuPDF's own line-segmentation algorithm design, not
  measured directly; the dash-column and rotated-text cases are the two places this repo's own
  corpus already falsifies a naive version of that assumption, which is exactly why the
  word-count-majority guard exists rather than trusting uniqueness alone.
- That the rotated-text residual is fully covered elsewhere by GH-734/GH-739's chart handling
  was NOT re-verified end-to-end here (would require driving the routing/witness pipeline); the
  claim is limited to what was measured: the residual's page set is exclusively
  `fomcminutes*`/`fomcprojtabl*`, which is the corpus those tickets already target.
