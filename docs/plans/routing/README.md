# Routing design — OCR only what needs it, check everything that ships

Opened 2026-10-09; rewritten 2026-10-11 with what has been built, measured and ruled since. Dashboard:
[`index.html`](index.html). This file is the source of truth; the dashboard summarises it.

## Goal

Send a model only the parts of a page the PDF's own text cannot provide, and ship nothing that nothing
checked. A wrong or dropped number is worse than a missing one; extra GPU time is the lesser cost
(owner, 2026-10-10).

## Owner rulings so far

| Date | Ruling | Where |
|---|---|---|
| 2026-10-08 | A scanned-page table that could not be checked ships marked "unverified"; a table the check contradicts stays withheld. | #1048 |
| 2026-10-09 | OCR'd text outside a figure is substituted into the prose in place; text inside a figure goes beneath the linked figure image. | this file, Target 1 |
| 2026-10-10 | An undecided figure check keeps the page's native text, flags the page (WARNING, `FIGURE_WORDS_UNREAD`) and escalates the figure crop, never the page. | #1053, #1075 |
| 2026-10-10 | The formula lane is judged on correct maths first, GPU time second. | #1078 |
| 2026-10-10 | Every run checks the tools its enabled features need before reading a page, and stops when one is missing. A plain check, not a model. | #1080, #1081 |
| 2026-10-11 | On a scanned table, the same model may confirm its own number from a one-cell crop (default off). Built, then parked: see Tables. | #1082 |
| standing | Digit disagreements never go to the text-only judge (Jev); they are settled mechanically or by a vision read. | Target 3 |

## Where it stands (measured)

Reference run: Rorqual array job 22922695, main @ a257fd1d (verified from `pip-freeze.txt`), 2026-10-10,
395 papers, 17,091 pages, Qwen3-VL-30B-A3B-Instruct on vLLM, one H100 per shard.

| Lane the page shipped from | Pages | Share |
|---|---|---|
| Native text, no model | 10,578 | 61.9% |
| Whole-page model read (qwen) | 4,042 | 23.6% |
| Figure lane (`chart_asset`) | 1,353 | 7.9% |
| Native text plus maths handling | 1,118 | 6.5% |

Against the 2026-10-01 baseline (job 22889958): papers halted mid-run 186 → 0; pages with no usable output
517 → 0; all 300 text pages wrongly sent to the figure lane (#1059, #1066, #1069) left it. Whole-page reads
rose from 1,266 to 4,042: pages that used to halt now reach the model, and 1,995 pages in 301 papers that
shipped native in the baseline now get a whole-page read, mostly because of the maths detector (#960). Cloud spend $0.00; about
4 GPU-hours in total, of which roughly 18 minutes per shard is loading the model.

## The four lanes

### Text

- **Built:** native prose ships as paragraphs, with line-end hyphens rejoined only when the same document
  shows the joined word (#1074, merged as #1077).
- **Check on native text:** missing-text step 1 is built (#1050, 2026-10-13). Every trusted-native page
  (`native` / `native+math`) gets a model-free ink-vs-visible-text-layer verdict (`passed` / `flagged` /
  `unknown`). A flagged page keeps its native prose, ships WARNING / `text_ink_unread`, and links
  `figures/uncovered_text_pN_rI.png`. Repeated ink is flagged like any other uncovered text, so a logo
  page ships a warning. No model call, no OCR of crops (that is step 2). Wrong text (#1051) is still
  designed, not built.
- **Known whole-page triggers besides maths:** invisible text layers over a scan (226 pages), control bytes
  before digits (152 pages).

### Formulas

- **Today:** a page whose text layer looks like damaged maths (#960) is read whole by the model, replacing the
  native prose. On the reference run the re-route fired 1,207 times; of the 1,655 maths-flagged pages the model
  read whole, 1,203 were flagged by the math-font signal alone.
- **Measured:** a hand check of 20 such pages found 9 with correct maths (false alarm), 8 flattened (lost
  sub/superscripts or fractions) and 3 with junk characters. Flattened maths is also common on native pages
  the detector never inspects: a layout probe flagged 30 of 66 sampled pages, about 24 of them really flattened.
- **Tried and parked:** two character rules to stop false alarms (#1078). Each freed pages whose maths then
  shipped worse, because the damage is in the layout, not the characters.
- **Record-only formula lane** (branch `impl/exp-formula-shadow`, not merged): find each formula from its
  layout, crop it with neighbouring text masked out, have the model read only the crop, and check the reading
  against the PDF's characters (digits, letters both ways, accents). Third round: 20 passing readings checked
  by eye, 16 fully correct. Reading the formulas of a page took about 45 s on the Mac against 18.8 s (median)
  for one whole-page read on the cluster: the lane is about correctness, not time.
- **Blocker:** both existing maths readers speak only the Ollama API (#1063). On the vLLM cluster 2,748
  formula regions on 1,159 pages were detected and none was read.

### Figures

- **Built:** pages whose images are listed but never drawn, reader highlights, and page-sized background fills
  no longer count as charts (#1059, #1066, #1069; merged as #1061, #1067, #1070). Chart-page numbers are no
  longer hidden as axis labels (#1055, merged as #1056; 6,337 numbers on 321 pages in the 2026-10-01 run). Figures on single-column pages are cropped one by one and placed inline; words inside a figure that
  nothing read flag the page `FIGURE_WORDS_UNREAD` (#1053, merged as #1075).
- **Open:** figures out of reading order (93 pages), side-by-side panels (55), drawn vector charts, crops
  merged with neighbouring figures (#1076), one real figure lost by the #1069 rule (Bianchi p58).

### Tables on scanned pages

The check: every number in the model's table must appear, as often as it occurs, in a Tesseract reading of
the page. Otherwise the page ships `[page N failed: unverifiable table — see image]` and no text.

- **Measured:** 181 pages in 74 papers were dropped because the cluster had no Tesseract. Tesseract now runs
  there (Apptainer image, `venv-socr-main`). With it, 19 of the 181 pass (17 after #1085); the rest fail because Tesseract and
  the model disagree on numbers. Of 92 disputed numbers on 12 random failing pages, checked against the page
  images, **the model was right on 90**; Tesseract dropped signs and decimal points, missed cells and misread
  digits. 11 of the 12 pages had a fully correct table. The one real error (Forney 1988 p22) is a row the model
  shifted next to blank cells.
- **Built:** two safety fixes to the comparison: a printed `.45` no longer supports a model `45` (#1084), and
  an en/em dash before a number no longer supports a model negative (#1085, merged as #1088).
- **Tried and parked:** re-reading each disputed number from a crop of its cell (#1082 step 2). Three review
  rounds each found a real way a wrong number could pass (a match elsewhere on the page, a neighbour's dash,
  a shifted row); closing them cut the rescued pages from 17 to 12 to 7 of 122. Every confirmed cell checked by
  eye was right. The limit is locating cells and confirming row identity on old scans.
- **Open defect:** the fallback check against a page's untrusted text layer never checks header-row numbers and
  tolerates one unsupported number in 50 (#1087). Wrong numbers can ship through it.

## Run-level safety

- Every run checks its tools first (#1080): model server (a real generation, Ollama or vLLM), OCR engine
  executable, page judge, Tesseract, maths readers and their models, table judge. A missing tool stops the run;
  `--allow-missing <check>` continues and records the gap as `ran_without` in metadata, manifest and summary.
- The vLLM liveness probe asks for the model the run actually uses (#1062, merged as #1064); one slow call no
  longer halts a document.
- The cluster corpus job records the installed socr version from `pip-freeze.txt` and stops when it differs from
  the label (the probe and fixes job scripts do not yet).

## Open decisions for the owner, in order

1. **Scanned tables that disagree with Tesseract.** Today they are withheld (2026-10-08 ruling: a contradicted
   table stays withheld). On old scans most disagreements are Tesseract misreads: on the 12 hand-checked pages,
   shipping the model's tables would have shipped 11 correct pages and one page (Forney p22) with two wrong
   numbers. Options:
   - **A. Keep withholding** (today). About 160 pages stay dropped; no wrong number ships through this path.
   - **B. Ship with a visible "unchecked" mark** in the page text and library text. About 160 pages return;
     roughly 1 in 12 of them carries a wrong number under the mark.
   No mechanical test found so far separates a Tesseract misread from a model error on these pages; the cell
   re-read was the attempt, and it rescued 7.
2. **May a checked formula reading replace the native text?** The September equation-lane ruling says model
   readings are attached beside the native text, never replace it. The formula check proves "nothing invented
   or dropped", not sub/superscript placement (`x_t^2` and `x^t_2` both pass).
3. **A formula the lane cannot read:** keep the page's native text with a warning (as figures do), or send the
   whole page to the model (today).

## Next work (no ruling needed)

- #1063: give the maths readers a vLLM route, so formulas are read on the cluster at all.
- #1087: the untrusted-layer fallback must check header numbers and allow no unsupported number.
- Upgrade the cluster's socr to current main; until #1063 lands, its jobs need `--allow-missing maths-readers`.
- Price local model calls by GPU time (not filed).

## Target (unchanged from 2026-10-09)

Native text ships only after model-free checks; models see gaps and bad lines, not pages.

1. **Text quality check, step 1: missing text (#1050).** Built as flag-and-link (2026-10-13). Render the page,
   find ink, subtract visible text-layer char boxes. Text-like ink with no words over it flags the page
   (`text_ink_unread`) and saves the region PNG; native prose is unchanged. A check that cannot decide is
   `text_coverage_unknown` (WARNING, no model in step 1; Target 4's whole-page route is step 2). Step 2
   (not built) OCRs the crop only. Inside a figure, the figure's marks are ignored but text-shaped ink still
   counts (axis labels, legends, box labels). Where the OCR'd text would land:

   ```markdown
   ![Figure 1: …](figures/p002_fig1.png)

   > Text in figure (OCR): ESG Score · Environmental: Greenhouse Gas Emissions, …
   ```
2. **Text quality check, step 2: wrong text (#1051).** Re-read the rendered page with Tesseract, align line by
   line with the native text; disagreeing lines → crop to the vision model.
3. **Jev as the text-only judge (#1051).** Jev judges text pairs (a real word difference or a misread). Digit
   disagreements never go through Jev.
4. **Fail closed.** A check that cannot run or decide sends the page to today's whole-page route; figures are
   the exception (2026-10-10 ruling above). #1050 step 1 has no model: `unknown` ships WARNING /
   `text_coverage_unknown` and keeps the native prose; the whole-page route is step 2.
5. **Honest local price.** GPU seconds enter the ladder's cost.

Order of work for the text checks: measure #1050 on trusted-native pages first (stop if there are about zero
real hits), then #1051 against a plain string-distance rule (drop Jev if it is no better), then build what
survived.

## Prior evidence that constrains this

- **`docs/plans/fake-native-pages`:** 72 of 2,972 pages (2.4%) are old scans with a baked-in OCR layer; #961
  routes an invisible text layer over a page raster to OCR. The missing-text step must not re-solve that.
- **Same plan, ticket B2 (lexical quality signal): closed, not built,** measured as noise. Jev's garble role in
  #1051 must beat that finding or be dropped.
- **Judge agreement is not corroboration.** Two vendor judges agreed on a page and both missed a shifted row.
  Hence the digit rule, and hence the scanned-table result above: a shifted row is the error that matters.

## Where the evidence lives

Measurements and hand checks are in gitignored scratch (`scratch/routing-baseline/`) and in the decision logs
under `docs/log/2026-10-1*`. Page images are not committed: the repo is public and the pages are third-party
papers.
