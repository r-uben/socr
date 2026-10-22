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

# Round 3 — merging #1185 into this branch

`ed53a8ba` (#1219, the `text_layer_numbers` lane) landed on `main` first. Merged in, no rebase,
no force-push; merge commit `f357f564`. Two conflicts, both resolved by keeping **both**
behaviours — a figure a page does not show and numbers its text layer disputes are different
facts, and neither lane may undo the other:

* `verdict.silent_success_clauses`: #1187's clauses go **inside** #1185's
  `_clauses_ahead_of_text_layer_numbers` tuple, so that tuple stays the single ordering
  authority and #1185's slot arithmetic can see this lane's clause.
* `verdict._status_is_explained`: two independent early returns, one per lane.
* `manifest._verdict_demotion_mode`: this lane keys on the clause's **scope kind**, #1185's on
  its **result**; both lookups stand, then the shared by-check map.

Two cross-lane gaps the merge exposed were fixed rather than left: `refinalize_own_text` could
not count this lane's clauses when building its slot-reference state (extracted as a named
`slot_reference_state`, because a source-text assertion let the mutation that emptied it
survive); and this lane's tool appended **below** a numbers clause instead of ahead of one.

Measured and **not** fixed: on the 7 calibration pages carrying both clauses, the rendered mark
ORDER follows tool order, because `_apply_verdict_marks` appends the marks not already present
and never reorders an existing banner block. Both marks are present either way and the meter is
0; carried to #1226, because reordering a written block is a behaviour change for every lane.

# Round 4 — the second round of reviews of PR #1224

Argon BLOCK, Fable MERGE-AFTER-FIX, bot REQUEST_CHANGES. **Ten findings, all REAL**; the
reproduction record is in the PR. As in round 2, 15 of the 16 symbols Argon quotes do not exist
in this tree — and as in round 2, nothing was dismissed on that basis, because the substance
held every time.

1. **This lane had no business on a native page at all** (bot blocker 1 · Fable P1-2 · Argon
   P0-B, standing three rounds). The lane decides in the page loop, on `bo`; the paragraph
   reflow runs later, at finalisation. On a born-digital page a wrapped cross-reference —
   `… see` / `Figure 4. However, …` — is a strict caption on the raw layer and nothing at all
   on the text that ships. The owner measured ~7 such false fires in 70 pages of raw layer, and
   since round 2 one of them takes the whole document from `completed` to `partial`.
   The fix is not a better regex: **every number this PR has is of model-read pages** (the
   crosstab filtered `engine not in ("", "native", "native+equations")`; the Fed tool reads
   finalized, already-reflowed sidecars), so the mark branch on native pages was never measured
   at all. New predicate `page_was_read_by_a_model(engine)` gates both the live call site and
   the tool. A native page keeps whatever the native lanes say about it.
2. **A retryable mode made every acted-on page re-OCR forever** (Fable P1-1 · Argon P1-C).
   Round 1 was safe by accident: a shipped page was SUCCESS (ordinary skip) and an unlocated one
   borrowed a permanent mode. Round 2 made every acted-on page WARNING with `FIGURE_NOT_SHOWN`
   deliberately retryable, so the per-page ledger gate ("status MUST be SUCCESS") fell through
   and re-read the page on every partial resume. Split the mode in two: `FIGURE_NOT_SHOWN` is a
   determination about the page and is **permanent**; `FIGURE_RENDER_FAILED` is a transient
   crop failure and stays retryable. The owner's round-1 "second run made 0 engine calls"
   measured the whole-document RootIndex fast path, not the ledger — pinned now with a test that
   crashes after page one and resumes through the real `_restore_terminal_page_state`.
3. **The failure mode must be a function of the clause SET, not of arrival order** (Argon P0-A).
   `_verdict_demotion_mode` returned the first matching clause's mode and
   `_warn_success_status_for_verdict` fires only on a SUCCESS page, so whichever tool demoted
   the page first owned the mode: `own-text → figures` recorded `numbers_disagree_with_text_layer`,
   `figures → own-text` recorded `figure_not_shown`. Two harms — a numbers **FAIL**, the stronger
   finding, was shadowed by this lane's UNKNOWN; and because the numbers modes are permanent and
   the figure mode was not, the same page was restored or re-OCR'd depending on tool order.
   Now the clauses are sorted into fresh-run order (`in_fresh_run_order`, over the one
   `_SILENT_SUCCESS_ORDER` tuple) and FAIL is scanned before UNKNOWN, and an already-WARNING page
   recomputes a verdict-derived mode. All four paths — live, resume, and the tools in either
   order — converge on the same mode.
4. **The live run and the tool told the document two different stories** (Argon P1-A, P1-B).
   `RENDERED_KIND` is in `MARKED_KINDS`, so `_unshipped_figure_note`'s `elif` for a shipped crop
   was unreachable and a live run never reported one; the tool's two independent `if`s reported
   the page as both shipped and marked. Made the live note two independent `if`s too. The tool
   also wrote a non-empty `error` while leaving the document `status` as it found it, and never
   recounted `tables.verified_text` although it demotes pages from `success` — both fixed, the
   recount re-using `refinalize_own_text._recount_tables`.
5. **Two lanes cropping the same page** (Argon P1-D). GH-189's chart-region pass takes **every**
   page with tables as a candidate and knew nothing about this lane; a chart+table page is
   exactly round 1's mechanism 1. It now skips a page this lane has cropped.
6. **The regex, again** (bot 2 residual, bot 3). `[—–\-]` accepted a dash **range** as a caption
   terminator, so `Figure 3-5 show …` marked prose; and `IA.1` / `OA.3` appendix labels matched
   nothing. Both fixed, both pinned. The roman alternative's own tail was **deleted** rather
   than tested: the two-letter alternative consumes `II-3` and `I.2` first, so the mutation check
   could not kill it.
7. **One order for the marks, so byte identity can be STATED** (Argon P0-A, the half round 3
   measured and deferred to #1226). `_apply_verdict_marks` appended the lines not already
   present and never reordered an existing block, so the two re-finalize tools left the page
   reading in the order they RAN. New `_ordered_page_marks` re-sorts the canonical lines
   already in the leading block together with the new ones, by the key `marks_to_write`
   itself uses, and only on a page that is gaining a line (so a complete page stays
   byte-identical). Lines that are not canonical clause marks keep their place ahead of them:
   #1129's `> [page unaccepted: …]` headline has its own contract. Measured on the
   calibration copy with the three tools forwards and backwards: **every page fragment, every
   document `.md` and every `manifest.json` is now byte-identical across the two orders**
   (before: 7 fragments and 4 documents differed).

## What is still order-dependent, measured and NOT fixed

Across the two orders, 0 of 2,476 pages differ in status, failure mode or clause SET. What
differs is:

* **the recorded clause ORDER on 3 pages.** All three carry `page_judge` — a STATUS clause —
  ahead of both lanes' clauses. The marks on those pages are identical and complete; it is the
  sidecar's list order that differs.
  **This attribution was wrong, and Fable's round-3 review corrected it** (round 5): it is not
  #1185's count slot. `in_fresh_run_order` — *this* lane's sort, which ran last in the forward
  order — ranks `(Check.PAGE_JUDGE, SCOPE_PAGE)` as the silent `_exact_pass_unjudged_clause`
  whatever the clause's result, while a judge REFUSAL is a status clause that `derive_clauses`
  puts last. So the forward record is this PR's own sort and the reverse one is #1185's, and a
  fresh run gives a third order again (`[figure_text, text_layer_numbers, page_judge]`).
  Fixing it means ranking silent only on `result is UNKNOWN`, which changes the rank of every
  status clause in both lanes — its own change, with its own measurement: **#1228**.
* **`refinalized_from` on 7 pages, and the 3 tools' own run logs.** These record the status and
  source digest each tool FOUND, so two different orders genuinely saw two different prior
  states. That is history, not disagreement about the page.

# Round 5 — the round-3 reviews of PR #1224

Argon and Fable MERGE-AFTER-FIX, the bot REQUEST_CHANGES ("one line left"). Four REAL defects,
one real measurement that is not a defect, and two of my own round-4 tests that could not fail
on what they named. Argon's quotes check out this round; the invented symbols of rounds 2 and 4
are gone.

1. **One question, one predicate** (the bot's remaining blocker · Fable P1-1 · Argon P1-1).
   Round 4 closed the native blocker with a hand-written set of three engine strings while the
   reflow decides by `startswith(("native", "chart_asset"))`. Four engines that ship the page's
   own line-wrapped layer were still judged on it before reflow: `native+math` (37 of the 2,476
   calibration pages, 1,118 of run 23319732), `chart_asset`, `native_fallback` and
   `native+tables` — two of them reported, two found in reproducing it. Fable drove rollet p33's
   false mark through `process()` on a `native+math` page.
   The set is DELETED: `page_was_read_by_a_model` is now `bool(engine) and not
   engine_ships_native_text(engine)`, and the test pins the PROPERTY (the lane's answer is the
   negation of the reflow's, for every engine string) rather than a list, because a list is what
   drifted. `blank` still answers True and is deliberately left: a blank page has no body, the
   first gate abstains, and an exclusion that cannot fire is a guard this repo deletes.
   **Round 4's claim "every number was measured on exactly this predicate" was wrong** and is
   corrected here: that crosstab filtered three engine strings, on finalized (reflowed)
   sidecars, which is not what the live lane reads.
2. **A mode a floor set is not this chooser's to recompute** (bot P2, which Fable P1-2 and Argon
   P1-2c both measured and upgraded). `PRIOR_OCR_LAYER_UNVERIFIED` has two writers — this
   chooser, from a clause on an ACCEPTED reading, and the #1027/#1156 floors, which set it
   directly on a page whose reading was NOT accepted. Round 4's recompute claimed in its comment
   to leave a guard's mode alone while keying on a set containing the floor's, so a floor page
   that also carried the chart lane's clause came out `VISUAL_VALUES_UNTRANSCRIBED` — which IS
   permanent, making an unaccepted page one a resume restores for ever, under a reason the floor
   did not give. 4 of 2,476 calibration pages carry that pair. `_a_floor_set_this_mode` tells
   the writers apart by what only a floor sets beside the mode (`audit_passed=False` and the
   page's own native text); provenance would say it directly and is not available on a
   re-finalize pass, which rebuilds the output from the sidecar.
3. **Both of this lane's modes are explained by its clause** (Argon P1-2b). The round-4 split
   gave the render failure its own mode so it could stay retryable, and `_status_is_explained`
   named only `FIGURE_NOT_SHOWN`: on the second pass — which assemble makes over every page — a
   RENDER_FAILED page gained the fail-closed `unknown` clause while its bytes did not change.
   The idempotence test is now parametrised over every acted-on kind, which is what would have
   caught it; pinning one kind hid the other.
4. **The transience cannot live in the mode alone** (Argon P1-2c). Another lane's clause takes
   the mode over: a render-failed page with a disputed number records
   `numbers_disagree_with_text_layer`, which is permanent, so the failed crop was skipped for
   ever. The ledger now reads the lane's own record of what happened.
5. **Two round-4 tests could not fail on the behaviour they named** (Fable P2-4), and one of
   them sat directly over defect 4. `test_the_chart_region_pass_skips_a_page_this_lane_cropped`
   asserted the ORDER of three lines of source; it now puts a chart+table page through
   `process()` and counts the PNGs. `test_a_failed_crop_is_retried_on_resume` asserted set
   membership; it now crashes a two-page run after page one and asserts the page IS re-read.

## Measured, and NOT a defect

Argon P1-2a: adding this lane's clause takes 4 Fed pages out of #1192's prior-OCR resume
exception, so they are re-read on every partial resume. The measurement holds; the behaviour is
that exception's own documented refusal — "no second clause of any kind is admitted … the one
that stops a page whose table went unjudged or **whose figure crop went unread** from riding
along". The direction is conservative, and widening another lane's guard to admit exactly the
clause it names needs the owner. Recorded in #1228.

## Three tests that could not fail on what they named

Fable's P2-4 named two of mine from round 4; the third was one I wrote this round and the
mutation check caught it. Recorded because the pattern is the same each time — the test drove
something adjacent to the behaviour and the assertion was true either way.

1. The chart-region skip. Round 4 asserted the ORDER of three lines of source. The first
   replacement drove `process()` and still passed with the guard deleted: `_preserve_chart_regions`
   needs the born-digital detector to say the page has TABLES, and the lane's ordinary fixture
   has none, so the pass was never a candidate (measured: `has_tables` False without the aligned
   numeric rows, True with). On a fixture that IS a candidate, the table reconstruction wins the
   page and this lane is correctly out of scope — so the two lanes do not meet end to end on any
   fixture available here. The test now drives `_preserve_chart_regions` itself on a real
   chart+table page, with the field the skip reads as the only difference.
2. The retried crop. Round 4 asserted set membership. The first rewrite drove `process()` with a
   crash-and-resume and STILL passed with the guard deleted, because the ledger reads the mode
   off `winning_output` while the fixture edited only the sidecar's top-level copy — so both arms
   were re-read for an unrelated reason. It now resumes one sidecar twice, changing only the kind.
3. The scope test parametrised the three strings in the predicate's set, which is how the set
   drifted from the reflow's in the first place. It now pins the property.

## Gates (round 5)

| gate | result |
|---|---|
| full suite, frozen tree, shared lock | **8947 passed, 2 skipped, 9 xfailed, 0 failed** |
| `uvx ruff@0.16.0 format --check .` | clean, 1022 files |
| mutation check, both lanes' suites | **50/50 killed**, no survivors, no stale anchors |
| calibration, three tools in order and in reverse | METER **0 of 2,476** at every step |
| second pass of all three, both orders | **0 files changed** |
| forward vs reverse | every fragment, document `.md` and `manifest.json` IDENTICAL; 0 pages differ in status, mode or clause set |
