# #1225 — a chart a model-read page shows and names nowhere is marked, not lost

2026-10-24. Branch `fix/issue-1225`, on `origin/main` 2a57f05d (#1229, #1230).
Measurement material: `scratch/routing-baseline/leads/1225/` (`report-phase1.md`,
`report-phase2-measure.md`, `eval_v2.jsonl`, the page renders under `sample/`, `sample2/`).

## The issue's population is mostly not figures

The issue counted 541 model-read CHECKED pages on run 23319732 that name no figure, ship no
picture, and have a region `FigureExtractor.extract_page` locates; "213 with `has_chart_marks`,
48 with `_vector_chart_found`". Re-derived with the merged #1187 lane's own predicates:

- The 541 reproduces (`cap == 0` and a raw region; 542 on a rerun). The 213/48 come from a
  DIFFERENT cell of the crosstab (`cap_anchored == 0`, 622 pages); inside the 541 they are 154/12.
- **408 of the 542 have only a whole-page box**: the raster of a scanned page, which
  `locate_figure_regions` already drops (`SCAN_RASTER_PAGE_COVERAGE_MIN`). `has_chart_marks`
  returns True on a page-sized raster with no text layer, so on scans it is not chart evidence.
- 13 of the 134 sub-page ones carry a LOOSE caption: #1187's.
- **Residual: 121 pages in 59 documents** (chart 16, vec 9).

Looked at against the page image (seeded stratified sample, 15/12/12/12, plus a census of the
residual's 16 chart pages; 59 pages): every real figure in the sample carried a caption #1187
already handles. Uncaptioned pages were logos, watermark blocks, text columns, tables, scans,
annotations: **0 of 19** sampled residual pages were real figures, and the census of the 16
residual chart pages found one chart grid (stock_watson 2012 p51) and one fully transcribed flow
diagram (forestal p101, not a loss).

So option (a) of the issue -- fire on `has_chart_marks` + a located region -- would fire on 16
run pages with 1–2 real, and on the Fed calibration copy it catches 1 of the 4 real chart pages
(greenbook-part2 2004-09 p46/50/52/54): those are monochrome hairline charts, rejected by the
neutral-colour rule, which stays. Flipping `save_figures` was not taken either: on scans it ships
page rasters as figures, and with `figures_max_total` = 25 it truncates 54/395 run documents and
29/70 Fed documents (extractor run on every page).

## The evidence that does separate them is in the text

On every real uncaptioned loss the VLM transcribed the chart's SKELETON and dropped the plot:
panel titles, units, legend entries and **the axis tick labels** (`1996 1998 2000 2002 2004`,
`18 / 16 / 14 / … / 0`). Plotted values almost never form an arithmetic progression; ticks do.

**Signal** (`unshipped_figures.tick_run`): the longest run of evenly spaced numbers on a line
every token of which is a number, or down consecutive one-number lines (blank lines do not break
a column); never inside a fenced or `<pre>` block.

- **`MIN_TICK_RUN = 3`, from the definition, not fitted.** Any two numbers are evenly spaced;
  evenness first constrains at the third. On both corpora no model-read page scores 2 or 3, so
  3 and 4 fire on exactly the same pages.
- Table rows (Markdown, GFM, LaTeX) and list items never count because their delimiters are
  tokens that are not numbers. Explicit exclusions for each were written first and could never
  change an answer (the production function agrees with the measured prototype on all 19,567
  pages of both corpora), so they were removed.

### Precision and recall

- **In scope** (model-read, CHECKED, no picture; 2,751 run + 44 Fed pages): **10 firings, all
  10 real charts with the plot dropped** (each checked against the page image). 6 carry a caption
  #1187 already answers (3 crops, 3 marks). **4 are new: Fed greenbook-part2 2004-09 p12, p50,
  p52, p54.** Run 23319732: 0 new.
- **All model-read pages, any status** (4,210 run + 865 Fed, including typed 1978–1993
  greenbooks and bluebooks): **38 firings, census, 38/38 chart pages.** Rule of three: the false
  fire rate is ≤ 8% at 95%.
- **Recall: 4 of the 6 known uncaptioned losses.** The two misses, recorded as a known gap:
  - greenbook-part2 2004-09 **p46**: transcribed as `## Title / Percent / - legend item / Note.`
    with no tick numbers at all.
  - stock_watson 2012 **p51**: the whole body is eight lettered panel titles
    (`i. Short-term (< 27 weeks) unemp. rate (4Q change, percent)`), indistinguishable from a
    numbered list.
  No text-only signal for either was positive enough to justify a mark.
- **False fires, adversarial set** (NATIVE text layers, out of this lane's scope, #1228):
  161 firings; seeded samples of 20 + 12 (seed 12252) gave 30/32 chart pages. **The two false
  fires are pipe-less tables with a consecutive-year header**: ando_fu_roch 2022 p23 (regression
  tables; the value cells carry `***`) and the Fed 2016-06-15 transcript p21 (economic projections
  table, `2016 2017 2018`). That is the residual risk on model-read text, should a VLM ever emit
  a table with no delimiters; none of the 38 model-read firings is one.

### What was measured and rejected

"Suppress when a non-progression number row of the same length sits next to the ticks" (the
advisor's proposal in phase 2): on model-read pages it removed 11 firings, all real charts,
including the positive p12; 11 of 12 sampled native suppressions were charts. Recall loss with
no measured precision gain. Rejected.

## The fix

One new branch in `unshipped_figures.decide`, so the live lane and
`socr-refinalize-unshipped-figures` cannot drift:

- **No caption, and `tick_run(body) >= MIN_TICK_RUN`** → `CHART_SKELETON_KIND`, a MARK naming
  "uncaptioned chart". **Never a crop, and the locator is not called** (owner ruling, and p12 has
  no region the locator finds).
- **A LOOSE-only label with no region** (#1187's `NOT_FOUND`, a cross-reference-shaped label such
  as `Chart 3 Selected Treasury yields`), **and ticks** → the same mark. Measured cases: 0. Added
  on the advisor's diff review: the ticks are the same evidence whether or not a label opens a
  line, and leaving such a page unmarked would be silent loss by construction.
- A captioned page keeps #1187's answer: on all 6 captioned firings #1187 already acts.

Surfaces, each pinned by a test that differs only in the tick lines: the page (verdict clause
`figure_text=UNKNOWN`, the mark as the first block, status WARNING, failure mode
`FIGURE_NOT_SHOWN` -- deterministic, so not retried on resume); the document (not completed:
the kind is in `MARKED_KINDS`); `metadata.json` (its own note sentence, because #1187's
"name a figure" sentence would be false here); the CLI (its own line, same reason); the audit
log and the resume replay list (`unshipped_figure_chart_skeleton`, 64 → 65 in the pin).

The mark says what was checked and no more: axis ticks and no picture of the chart. Its first
wording claimed "none of its plotted values", which `tick_run` does not check (advisor).

## Review (opus advisor, diff review)

No P0. Dispositions, each verified against the code:

- P1-a, "the tool's note may drop #1187's pages": not a defect. `_answered_sidecars` rebuilds
  from every sidecar on disk plus the pending ones; #1187's tool writes nothing for a page it
  answers `None`, so a no-caption page is never pre-answered.
- P1-b, "the mode may not be figure-scoped": pinned (`FIGURE_NOT_SHOWN` asserted in the verdict,
  tool and resume tests).
- P1-c, the mark over-claims: **fixed** (wording above).
- P1-d, pipe-less tables / one-number-per-line tables / calendars can fire: true, measured
  (two native false fires, zero model-read), and the proposed guard was measured and rejected.
- P1-e, loose caption + ticks unmarked: **fixed** (branch above).
- P2: a one-line `<pre>…</pre>` blinded the rest of the page, and `<pre` matched `<prefix`:
  **fixed** (misses only, never false fires). A decimal comma reads as thousands and indented
  code is not excluded: misses only, left.

## Gates

- **Tests:** `tests/test_gh1225_uncaptioned_chart_skeleton.py` (39 tests), with #1187's file
  and the resume-kind pin: 143 passed. Full suite under the lock: **8,980 passed, 48 skipped,
  9 xfailed, 0 failed** (379 s).
- **Format gate:** `uvx ruff@0.16.0 format --check .` clean.
- **Mutants:** 22/22 killed, external copy of `src`+`tests`+`pyproject.toml`,
  `socr.__file__` asserted in the copy, uncapped `count(anchor) == 1` before each edit
  (`scratch/routing-baseline/leads/1225/mutants_1225.py`, `mutants-final.txt`). Two needed a
  test fixed before they died: the CLI test matched a phrase the document note also prints, and
  taking the kind out of `MARKED_KINDS` survived every behavioural test because the fixture page
  also carried an independent #1185 mark; the `process()` fixture now holds the skeleton in its
  text layer so the status difference is this lane's alone.
- **Re-finalize on a copy of the Fed calibration run** (`chain.sh`, prior-ocr, own-text,
  unshipped-figures, then the reverse order, on two copies):
  - **meter 0 of 2,476 after every step, both orders**;
  - this lane: 11 pages `chart_skeleton`, 3 crops (#1187), 8 `unlocated` (#1187);
  - model-read CHECKED: 53 → 15 after own-text → **14** after this lane. The one page it moves
    from CHECKED to MARKED is greenbook-part2 2004-09 **p12**. p50/52/54 had already been
    demoted by #1185's own-text tool (numbers the layer does not hold), and 7 more were MARKED
    by other lanes; none of those marks said a chart was missing, and now each carries the
    figure clause too;
  - **a second pass of all three changes 0 files**, both orders;
  - forward vs reverse: **every `.md` byte-identical**; 0 pages differ in status, mode or clause
    set. 39 non-text files differ: `refinalized_from` and the tools' run logs (history: each
    order saw a different prior state), and the clause LIST ORDER on 5 pages that also carry a
    `page_judge` status clause -- the known #1228, logged by #1187 on 3 pages; two of the 5 are
    this lane's;
  - **re-finalize then resume**: all 22 lane pages restored from their sidecars and pushed back
    through `manifest.refinalize_unshipped_figure_page` with the published verdict give 0 byte
    differences in text, status, fragment and document `.md`, both copies
    (`resume_check_1225.py`; shown to fail on a tampered fragment).

## Not measured

- The full Fed run: it has not run yet (queued on Rorqual). The re-finalize path is shown on a
  copy of the 70-document calibration run only.
- Recall beyond the 6 known uncaptioned losses: no census of all chart pages exists.
- `socr-refinalize-boilerplate` is not in the order check: its dry run flags 0 pages on the
  calibration copy.
