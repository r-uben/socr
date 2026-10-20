# #1187 — a figure a page NAMES never ships as a caption with nothing under it

2026-10-20. Branch `fix/issue-1187`, cut from `origin/main` at `4d5946aa`.

## The diagnosis is not the issue's

The issue proposes one mechanism: these are monochrome thin-line vector plots, `has_chart_marks`
counts only non-neutral fills or thick strokes as data marks, so a hairline plot falls to the
framed-cluster fallback and that fallback misses it.

I probed the detectors directly on the 12 pages a human adjudicated against the page image
(run 23319732, `scratch/routing-baseline/leads/1187/measure-23319732.txt`):

```
herbst_johannsen p30/33/57/58   chart=True  vec=True  rasterboxes=[]  ndraw=51
leeper p4/11/14                 chart=False vec=False rasterboxes=[]  nimg=1 ndraw=0
strongin p19                    chart=False vec=False rasterboxes=[]  nimg=1 ndraw=0
mendoza-fernandez p19           chart=True  vec=True  rasterboxes=[]  ndraw=319
romer_romer p10                 chart=True  vec=False rasterboxes=[(61.5,68.0,253.7,217.0)]
golosov p37                     chart=False vec=False rasterboxes=[]  ndraw=72
jorda p24                       chart=True  vec=True  rasterboxes=[]  ndraw=190
```

`has_chart_marks` fires on 7 of 12 and `_vector_chart_found` on 6. **The detector is not the
problem on most of these pages; everything downstream of it is.** Four mechanisms, all measured:

1. **The chart-vs-table arbitration promises a lane a model-read page never enters.** A page
   with both signals is held out of the page-level chart lane and gets this event:
   `chart_table_arbitration`, "the chart is represented inline at its own position rather than
   appended after the page text". The code comment names the mechanism that is supposed to do
   it: `rowize_from_words_chart_aware` emits a placeholder, `_render_chart_region_pngs`
   resolves it. Both run on `ps.native_text`, and the only agentic call site of the second is
   inside `if not decision.accepted and (self._page_has_tables(...) ...)`. On these pages the
   model's reading WAS accepted, so the shipped text is the model's, carries no placeholder,
   and no crop is ever rendered. 15 of the first-pass hits.

2. **The chart lane is never asked at all on a page that goes to the model.**
   `_is_chart_asset_page` opens with `if not self._is_native_eligible_without_ocr(...)`, on the
   stated assumption that "the page would go to the OCR ladder anyway; there is nothing to
   intercept". 97 of 113 first-pass hits carry `garbled_math_native` or a sibling, which is
   exactly what makes a page not native-eligible. **The assumption that a model reading covers
   the chart is false: a VLM transcribes the caption and drops the plot.**

3. **The general figure phase did not run.** `runs_figures = (save_figures or describe_figures)
   and has_text`, and `save_figures` defaults to False. No worker log of run 23319732 carries an
   "Extracted N figures" or "No figures detected" line. Every picture that shipped on that whole
   corpus came from a chart lane or a fail-closed floor. `FigureExtractor.extract_page`, run by
   hand, locates a region on **all 12** adjudicated pages.

4. **#1030's scanned-figure lane has a caption-length cap that rejects real captions.** Its gate
   is `has_figure_caption`, which requires `len(line.strip()) <= MAX_CAPTION_LINE` with
   `MAX_CAPTION_LINE = 50`. The leeper and strongin caption lines are 95–100 characters, so that
   lane abstains on exactly the scans this issue names. Not touched here: a cap derived from a
   98-page scanned sample is #1030's to revisit, and this lane rescues those pages anyway.

## The rate

`measure3.py` / `measure4.py` on run 23319732: 17,091 pages, 3,082 model-read and CHECKED,
2,751 of those ship no picture at all, and **147 name a figure in their own shipped body** —
132 where `FigureExtractor.extract_page` locates a region, 15 where it does not. 147/3,082 is
**4.8%**, inside the issue's estimate of 5.1% (95% CI 1.9–12.8%).

`measure5.py` runs the FINISHED decision function over the same pages. 142 of the 2,751 are
acted on — **103 ship a crop, 38 are marked, 1 is partial** (it names two figures and one crop
shipped) — and 7 are abstentions, where a figure label opens a line, nothing is located, and the
label is not caption-shaped. 4.6% of the 3,082.

The mark branch is larger than the first measurement because of the whole-page rule below: a
scan's only located region is the page raster, which this lane refuses to call a figure crop, so
those pages take the mark. Every one of them still NAMES the figure that did not ship.

The 15 original mark-branch pages were checked by eye against the page image
(`scratch/routing-baseline/leads/1187/img-mark/`). 14 are genuine captions over a real figure
(stock_watson p48 is a six-series colour line chart; herskovic p20 is three network diagrams).
One, du_li p4, names "Figure 3:" over a table of prompts the page DID ship as a table — a false
mark, 1 of 15.

## The fix, and why this shape

A new lane that does **not** add a detector. It asks the one question no existing lane asks —
*does the body name a figure that the body does not show?* — and answers it with the locator the
repo already has.

* **Positive evidence, from the page's own shipped body.** The caption is read from the SHIPPED
  text, not the layer, because the harm is a caption a READER can see with nothing under it.
* **Ship the picture where the locator is positive**, mark where it is not. Shipping is better,
  and the ruling says so.
* **Two matchers, split on what a false fire COSTS.** The loose one (a label followed by a
  terminator, end of line, or whitespace and a non-lowercase character) gates the branch that
  ships a crop, where a false fire adds one picture to a page that already has a located figure
  region. The strict one (terminator or end of line) gates the branch that MARKS, where a false
  fire demotes a page. Measured: strict removes the three `nagel` cross-references and the
  `damato` one from the mark branch and keeps every genuine caption.
* **The label is parsed ATOMICALLY (`(?>...)`).** Without it the regex backtracks a dotted number
  to its first component and reads the decimal point as the caption's terminator. That is how
  `Figure 4.5 shows contour maps ...` became a caption, and how `Figure 3.E provides the
  distribution of participants' judgments ...` marked a prose page in the FIRST run of the
  re-finalize tool on the Fed calibration copy.
* **`#1030`'s shape, not a new one.** The orchestrator decides and renders; `manifest.
  _apply_unshipped_figure_guard` appends at finalisation; `verdict._unshipped_figure_clauses`
  marks. One writer per statement, and the flush, the stitch and a resume replay see one body.

### What I did NOT do, and why

`save_figures` defaulting to False means every figure on every page is silently dropped unless a
chart lane claims it. Measured: 541 further pages carry a located region and name no figure (213
of them with `has_chart_marks` true). Argon argued for folding them in. I did not: that is five
times this ticket's scope, the evidence that a figure was LOST on an uncaptioned page is weaker
than a dangling caption, and this ticket's ruling is specifically about a caption with nothing
under it. It belongs in its own ticket, with its own measurement.

Nor did I touch `has_chart_marks`'s neutral-colour rule, or `#1030`'s `MAX_CAPTION_LINE`.

### A whole-page box is not a figure region

On a scan the extractor's only region is the page raster itself. Shipping a crop of it put a
second picture of the page under a figure caption, and #1030's own end-to-end test caught it on
its control page. The bar is `SCAN_RASTER_PAGE_COVERAGE_MIN`, the repo's existing measured
"this raster IS the page" floor — no new number. Such a page takes the mark branch instead,
which still satisfies the ruling: it names the figure that did not ship.

### Known limitation, stated rather than implied

The crop is APPENDED, as every sibling lane appends its asset, with the alt text naming the
figure. It is not spliced under the caption: that needs a caption-to-region binding this lane
does not have. `stock_watson` p48 is the shape that would most reward one —
`_looks_like_table_grid` refuses its chart (a hairline coloured plot with gridlines scores 0 data
marks), so it takes the mark branch although a region is plainly there.

## Evidence

* **Mutation check** (`scratch/routing-baseline/leads/1187/mutants.py`): 21 mutations, **21
  killed**. External copy of `src` AND `tests` AND `pyproject.toml`, `socr.__file__` asserted
  inside the copy, uncapped `count(anchor) == 1` asserted before each edit. Anchors are read from
  the live source each run, so a mutation that no longer applies is an ERROR, not a silent pass.
  Both production call sites are mutated, and so is the ORDER of the page-loop call against
  GH-86's sentinel strip.
  **Two guards I had written were found UNKILLABLE and were DELETED rather than given a test:**
  a `(?![0-9])` lookahead the atomic group already subsumes, and a per-ref idempotence check the
  body gate already subsumes.
  It also found two of my own tests vacuous. The `process()` fixture shipped the page's clean
  NATIVE text, so it never drove a model-read page; and the sentinel test replaced
  `process_pages` with a wrong-arity lambda, which made the provider fail and the page fall back
  to that same native text. Both now pin `engine == "qwen"`, and the stub engine's body is a
  class attribute so a test can change the one thing it is about without re-declaring a
  signature.
* **The Fed calibration run, on a COPY** (`leads/1187/fedcopy`): 3 pages shipped a crop and 8
  were marked; a second run of the tool changed **0** of 15,227 files; the acceptance METER is
  **0 of 2,476 before and after**, with two pages moving from CHECKED to MARKED (the other six
  already carried a mark of another lane's) — the intended demotion, and the only movement.
* The marked Fed page (2019-10-04 presentation materials p4) was checked against the page image:
  two full-colour area charts under "Exhibit 2", no picture shipped. A true positive.

## Review

Argon reviewed the diagnosis and design BEFORE the build and the finished diff after it.
Design round: it was right that a VLM writes Markdown captions (the matcher now consumes `**`
and `#`) and right to doubt the whole-line caption text (the mark now names the LABEL, after a
900-character Fed cross-reference landed inside one). It proposed a "previous line ends
mid-sentence" guard; I measured it and it drops 90 of 141 real captions, so it was rejected in
favour of the loose/strict split. It doubted that `FigureExtractor` fires on the scanned pages;
it does, on all 12.

**Diff round: five P0s, all real, all taken.**

1. `_merge_fragment` and `_swap_once` tested the OLD text before the NEW one. This lane only
   ever ADDS, so `old` is a substring of `new`, and a run interrupted after the fragment write
   and redone appended the block to itself. Both now test `new` first, and two tests restore
   only the sidecars and re-run.
2. A render failure emitted the failure event while writing no clause, so the CLI and the
   document note called the page marked while it shipped SUCCESS unmarked. Once a region is
   located, geometry is the positive evidence, so the loose caption may name it.
3. A page naming more figures than crops shipped got neither crop nor mark for the rest. There
   is now a `partially_shipped` outcome that marks naming all of them — this lane has no
   caption-to-region binding and must not pretend otherwise.
4. The lane read the body BEFORE GH-86 stripped VLM sentinel refs, so it abstained on exactly
   the pages the strip then left with a bare caption. Moved after.
5. Neither production call site was covered: deleting them left the file green. Both are now
   mutated, and so is the ordering in (4).

Also taken: the alt text names every caption, not the first (a wrapped cross-reference above a
real caption would otherwise label the real figure's crop with the wrong number);
`_published_verdict` keys on the whole clause, not `(check, scope)`, which dropped a recorded
clause sharing a region ordinal; the document note is replaced rather than appended and carries
no `;` of its own; and the guard uses `isinstance` on the state fields, because a bare `Mock`
attribute is truthy and `or []` kept it.

Not taken: Argon proposed widening the lane to the 541 uncaptioned pages with a located region
(see "What I did NOT do"), and in the design round proposed a prose-continuation guard that I
measured at 90 of 141 real captions lost.

---

# Round 2 — the reviews of PR #1224

Argon **BLOCK**, Fable **MERGE-AFTER-FIX**. Every P0 and P1 was reproduced against the real
tree before anything was changed; the command-and-output record is
`scratch/routing-baseline/leads/1187/report-r2.md`. All eight findings were REAL in substance
and all are fixed.

**Argon's quotes were of a codebase that does not exist.** Nine of the ten identifiers it cites
as "exact lines" are absent from the tree (`UnshippedFigureDecision`, `inspect_and_render`,
`exclude_tables`, `body_has_markdown_table`, `_refinalize_page`, `blob_updates`,
`apply_persisted_events`, `render_located_regions`, `table_is_the_figure`). Its answer to one
review prompt — that `du_li` p4 does not false-mark, because of a table suppression it quotes —
is an answer about code nobody wrote; my own round-1 report records that page as a false mark.
**Nothing was dismissed on the strength of that**: each finding was tested against the real
code, and the substance held in every case.

## What the reviews were right about

1. **A caption count cannot establish how many figures a page carries** (Argon P0-1). Fed
   calibration p3 heads itself "Exhibit 1" and carries six numbered panels; the locator finds
   two regions, both in the left column; three charts shipped nowhere and the page counted
   CHECKED. So no outcome of this lane claims completeness any more: **every page it acts on is
   marked**, and the mark says the crops are not proven complete. That is also what the
   page-level chart lane already does with a whole-page image, which is strictly stronger
   evidence than a crop.
2. **A mark that does not reach the document is half a mark** (Argon P0-2). `pages_ok` had no
   term for this lane, so a document whose only defect was a lost figure finished SUCCESS with a
   populated `error`; and the CLI lines sat inside a block gated on 27 other page buckets.
3. **The borrowed failure mode said something false** (Argon P0-3).
   `VISUAL_VALUES_UNTRANSCRIBED` means a picture shipped without its values; on the unlocated
   branch nothing shipped. It also put a TRANSIENT render failure in
   `PERMANENTLY_UNRESOLVABLE_MODES`, so a resume would never retry it. New mode
   `FIGURE_NOT_SHOWN`, selected by the clause's scope kind, absent from `_MODE_CLAUSES` (this
   lane's state survives a resume, so the real clause is re-derived and a row would duplicate
   it), with `_status_is_explained` taught about it.
4. **A fail-closed floor must stay a floor** (Fable P1-1). `is_page_failed_marker` admits "a
   marker plus at most one image"; appending two crops made the page stop reading as a marker to
   the disposition classifier, to selection and to the resume gate. The guard now abstains —
   which only loses the picture and keeps the mark because of (1).
5. **One gate on the bytes about to ship** (Argon P1-1). The lane decided on `bo` before both
   table-withholding gates could replace that body with a floor. Moved after both, immediately
   before the flush — #1203's lesson, in the same loop.
6. **The label regex** (Argon P1-2, Fable P1-2). Four prose cross-references matched as strict
   captions and would have marked a prose page; six real caption forms matched nothing,
   including `Figure E.2:` — `herbst_johannsen` p57 and p58, **two of the twelve pages this
   issue adjudicates**. Round 1's "the locator found a region on all 12" was true of the locator
   and false of the lane, which never reached it on those two.
7. **The tool's own drift** (Argon P1-3). Pages it gave a PNG had no `assets` record, so
   `stale_assets` never checked them and the next resume would add them; and its sidecar events
   carried a `page_num` key the live writer omits, so a resume would rewrite the sidecar.

## Restated

154 corpus pages acted on (was 142) — 112 ship a crop and are marked, 42 are marked — **5.0%**
of the 3,082 model-read CHECKED pages, almost exactly the issue's 5.1% estimate. The 12 extra
are the appendix captions. Fed calibration METER **0 of 2,476 before and after**; CHECKED-qwen
53 → 49; a second tool run changes 0 files. **28/28 mutations killed.**

One mutation was **retired rather than pinned**: `_merge_fragment`'s ordering guard. Probed on
a page another lane had already marked, `new.startswith(old)` and `old.startswith(new)` are both
False, because marking rewrites the leading banner block — the two orders are no longer
distinguishable, so a test for it could not fail. The order is kept as the safe one with the
reason at the line. `_swap_once` is not in that position and is still mutated.
