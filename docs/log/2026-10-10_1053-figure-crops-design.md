# #1053 — per-figure crops: design (Fable + Astra, converged after one rebuttal round)

Status: **design only, no code.** One ruling needed from the owner (below) before an implementer is
dispatched. Part of `docs/plans/routing/` (figure lane).

## Problem, measured

The chart-asset lane ships a page's native text and appends one PNG of the **entire page** as its
"figure" (`_agentic_chart_asset_page`). Corpus run on Rorqual (array 22889958, socr main @
c4937472, 395 papers, 17,091 pages): **1,503 pages** take this lane. Random sample of 30 (seed
10530), sorted by eye:

| Kind | Share | Handled by |
|---|---|---|
| Prose plus one or more figures | 16 / 30 | **this design** |
| No figure at all | 9 / 30 | entry fixes: #1059 (never-drawn images, 206 pages), boxed tables, title-page rasters |
| Figure is the page | 5 / 30 | the lane's intended case |

## Agreed design

1. **Figure boxes.** Raster: the page's **drawn** placements (`get_image_info(xrefs=True)`, the
   #1059 reader), one record per placement, keeping GH-511/656 scan/decorative handling. Vector:
   `_vector_regions` / `_cluster_drawings`, each cluster gated the way `has_chart_marks` gates it.
   Merge panels only with shared-caption evidence and no intervening prose or table region.
   **Not** `chart_region_bboxes`: it extends boxes to the page edge across the chart's height and
   swallows a table beside a chart (the #1055 lesson). One new `figure_boxes(page)`;
   `has_chart_marks` becomes `bool(figure_boxes(page))`, so trigger and inventory cannot drift.
   A box overlapping a detected table keeps the GH-150 B1 mixed route.
2. **Split at detection (option A).** The lane's `native_text` is flat `get_text("text")` with no
   positions today. Build figure pages through the same region interleaver tables use
   (`interleave_table_regions_into_page`), one `(rect, content)` per box, plus a **word-to-owner
   map** (prose / caption / figure *i*) shared with #1050. Every word is assigned exactly once;
   boundary overlaps stay explicit, never deleted. Figure-free pages stay byte-identical.
   Rejected: option B (split at assembly) — it leaves figure words duplicated in the prose.
   Re-OCR cost is not a reason to prefer B: `_socr_source_digest` invalidates every fingerprint on
   any source change anyway.
3. **Placement.** Reuse the `(rect, placeholder)` machinery `rowize_from_words_chart_aware`
   already writes, rendered by `_render_chart_region_pngs`, so each figure lands at its y-position
   between the paragraphs around it. The whole-page `chart_page_N.png` is skipped when
   placeholders exist.
4. **Reading and checking figure words** (the figure quality check in the routing design):
   - Tesseract reads **every** crop (CPU, no model) and is compared with the PDF's own words
     inside the box.
   - Native words in the box are exact and ship as they are. **The model reads a crop only when
     Tesseract finds content the native layer lacks** (or the box has no native words): that is
     #1050's missing-text test with a witness instead of raw ink.
   - Digits are compared mechanically (signs, decimals, multiplicities); words by the text-only
     judge once it exists (#1051). A digit disagreement is re-read, never arbitrated by a judge.
   - Every "Figure N" caption must match a crop identity, not just a count.
5. **When a check is unavailable or undecided: preserve and flag.** Keep the crop and the
   readings that exist, labelled unverified, separate from accepted text. Demote the page
   (status + a new failure mode such as `FIGURE_WORDS_UNREAD`, never `audit_passed`) and add a
   **document-level term**: `pages_ok` is a conjunction of named page sets, so a page WARNING
   alone never reaches document status. Escalate **the crop** to the next permitted rung; never
   the page (a whole-page read would replace exact native prose with an unchecked model read).
6. **Whole-page lane.** Not a size class. It stays only as the fallback when figure boxes are
   unknown or a box is refused; a page whose one figure accounts for all content yields the same
   result through the crop path.

## Owner ruling needed (one question)

Point 5 changes rule 4 of `docs/plans/routing/README.md` ("a check that cannot decide sends the
page to the whole-page route"). For **figures**, both reviewers now recommend:

- **Preserve and flag + escalate the crop** (recommended), or
- **Keep rule 4 as written**: an undecided figure check sends the whole page to the model.

## Smallest first PR

Raster figures only, single-column pages, **no model yet**: `figure_boxes` raster half, the
detection-time split with the word-to-owner map, inline placement, and the unread-words status
(WARNING + `FIGURE_WORDS_UNREAD` + document term). Guard: a same-process, two-run difference on a
synthetic page with prose above and below a raster figure; assert the ref sits between them, the
crop is smaller than the page, every native line survives in order, and the page is not SUCCESS.
Mutant: delete the placeholder branch → the whole-page ref is appended → the test fails.

Measure before and after on the corpus: chart-asset pages (1,503 → ?), crops per page, crop area
over page area, words present both in prose and inside a box (today: all of them), captions
without a crop.

## Weakest points

- Caption association and reading order on multi-column pages: the interleaver assumes one column
  (`born_digital.py:4306`). The first PR is single-column only for that reason.
- The "refuse a box that owns a prose row" rule is unmeasured on the corpus.
- Tesseract is installed on the Mac but not on Rorqual (no module), so the witness needs a cluster
  install before any corpus run uses it.

Sources (local, not committed): `scratch/routing-baseline/design-1053-{fable,astra}.md`,
`rebuttal-1053-{fable,astra}`.
