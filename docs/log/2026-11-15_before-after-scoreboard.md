# Before/after scoreboard: Phase A-C fixes and #1344 (2026-11-15)

Status: complete for the measurement asked. The 100 hard pages are measured, with and without the Hydrogen rung, and the
blocked pages were re-run with the #1346 fix (section 3c). The gold pages are measured in both arms
(section 4). The #1350 witness fix is measured in section 3d. Section 6 lists
what was not done.

- Before: `51270eb5` (the commit the audited Fed run was made on, before Phase A).
- After: `044e9443` (current main: Phase A, B2 part 1, B3, B4 relabel, C1-C3, #1344).
- Pages: the 100 random hard Fed pages of the #1343 test, and the 481 gold pages
  (`gold/fed-2026-11-11.jsonl`, run `51270eb5`).
- Paid calls: $1.37 of the $2 the owner approved ($1.08 Hydrogen arm, $0.25 fix re-run, $0.04 combined-fix run), all
  Gemini 4 Hydrogen. Nothing was merged.

## 1. What could and could not be compared

| Question | Measured? | How |
|---|---|---|
| Reader quality, four readers, 100 pages | Yes, one value per reader | Cached readings of the #1343 test |
| Reader before/after for Hydrogen, Flash, Sonnet | No such thing | Their cached readings are raw model output on a fixed prompt; no merge changes them |
| Pipeline before/after, 100 hard pages | Yes | Both commits re-run locally, same machine, same model, local only |
| Pipeline after with Hydrogen escalation | Yes | Section 3b; owner approved $2 |
| Blocked pages with the #1346 fix | Yes, 12 pages | Section 3c |
| Pipeline before, gold pages | Yes | Scoreboard on the cached audited run |
| Pipeline after, gold pages | Yes | Both commits re-run locally on each gold page |
| Invented / wrong printed numbers by image | Hydrogen only, 10 pages | The #1343 Opus image audit; no image audit exists for Flash, Sonnet or qwen |

## 2. Readers on the 100 hard pages (cached, same images, same prompt)

Yardstick: `socr.benchmark.scoreboard` number tokens, multiset difference against each page's raw
PDF text layer (8,705 numbers; 22 pages have a near-empty layer). "Dense" = the 78 pages with a
usable layer. qwen is the pipeline's shipped text from the cluster run at `51270eb5`.

| | Hydrogen | Gemini 3.8 Flash | Sonnet | qwen3-vl-30b |
|---|---|---|---|---|
| Numbers missing vs text layer | 391 | 919 | 1,380 | 1,782 |
| Missing on table pages only | 197 | 583 | 1,005 | 1,303 |
| Pages with zero numbers missing | 81 | 58 | 66 | 45 |
| Numbers shipped (dense pages) | 8,677 | 8,006 | 7,513 | 7,218 |
| Shipped but not in layer (dense) | 379 (4.4%) | 236 (2.9%) | 204 (2.7%) | 310 (4.3%) |
| of those, no other reader ships it | 109 (1.3%) | 53 (0.7%) | 12 (0.2%) | 102 (1.4%) |
| Possibly altered, upper bound (dense) | 30 (0.3%) | 88 (1.1%) | 40 (0.5%) | 154 (2.1%) |
| Invented, by image audit | 0 of about 700 (10 pages) | not audited | not audited | not audited |
| Wrong printed number, by image audit | 0 of about 700 (10 pages) | not audited | not audited | not audited |
| Cost per page | $0.0207 | not computed | not recorded | $0 |

Reading the proxies:

- "Shipped but not in layer" is a lead list, not an invention count. A correct reading from the
  image counts when the layer splits or lacks the number.
- "No other reader ships it" is the stricter proxy for the invented-number rate. For Hydrogen it
  includes the 40 bar heights it estimated on the Monetary Policy Report chart page.
- "Possibly altered" is min(missing, not-in-layer) per page, summed: the most numbers that could
  have been swapped for a different value. It is the proxy for the wrong-printed-number rate.
- Flash cost: 114,642 input and 158,938 output-plus-thinking tokens for the 100 pages; no verified
  Flash price is recorded in the repo, so no dollar figure is given. Sonnet usage was not recorded.

### Hydrogen through the production reader command (10 cached pages)

`gemini-ocr` with the production prompt, 10 of the 100 pages (1,057 layer numbers):

| | Numbers missing |
|---|---|
| Hydrogen, production command | 38 |
| Hydrogen, test prompt | 57 |
| qwen, cluster run | 267 |

37 of the 38 are on one page (p016).

## 3. Pipeline before/after on the 100 hard pages (local re-run)

Each page as a single-page PDF, `socr process --strict-local --no-gemma-cloud --qwen-backend ollama`,
cloud keys removed from the environment, `qwen3-vl:30b-a3b-instruct` on the Mac. The before arm
loaded its source from an exported copy of `51270eb5` (asserted per run). Cost $0 in both arms.
About 88 s per page before, 90 s after.

| | Before | After |
|---|---|---|
| Pages SUCCESS | 5 | 3 |
| Pages WARNING | 69 | 70 |
| Pages ERROR (text withheld) | 26 | 27 |
| Pages with a visible mark | 95 | 97 |
| Unmarked SUCCESS pages | 5 | 3 |
| of those, with numbers missing vs layer | 2 (p008: 1, p039: 5) | 1 (p008: 1) |
| Numbers missing vs layer, all pages | 4,306 | 4,329 |
| Numbers missing, pages that ship text | 792 | 791 |
| Pages with zero numbers missing | 38 | 37 |
| WARNING pages recorded as `audit_passed: true` | 30 | 0 |
| Pages with a recorded check trace | 0 | 98 |
| Model SUCCESS without verified evidence | 3 | 0 |
| Pages with a dropped decimal point | 5 | 4 |

What changed:

- 77 of 100 pages ship the same text once marks are stripped; 51 are byte-identical. The fixes
  changed labels far more than content.
- Two pages left unmarked SUCCESS for WARNING (`words_missing_from_page`): p039, which was missing
  5 layer numbers and shipped 6 the layer lacks, and p086. Those are silent defects now marked.
- p080 went from WARNING to ERROR and its text is withheld; that is the whole +23 in numbers
  missing.
- Four ERROR pages changed reason from `hallucination` to `witness_disagrees` (B4 relabel).
- p008 still ships as unmarked SUCCESS with one layer number missing.
- Numbers lost did not move. On these pages the fixes mark defects; they do not recover numbers.
  Recovery is what the Hydrogen rung is for; see section 3b.

Limit: this local run is not the cluster run. Under `--strict-local` 26 pages end as ERROR
(19 native, 5 marker), against 4 in the cluster run, so absolute counts here do not compare with
section 2. Only the before/after difference is the finding.

## 3b. After, with the Hydrogen rung on (100 hard pages)

Same pages, current main `044e9443`, Gemini key present:
`socr process --no-gemma-cloud --max-paid-pages 1 --qwen-backend ollama` (not `--strict-local`, so
the cloud table-judge rung of the ollama daemon is also reachable; that is a second difference from
section 3, see p089 below). Owner approved up to $2. About 160 s per page.

| | Before (local) | After (local) | After, Hydrogen on |
|---|---|---|---|
| Pages SUCCESS | 5 | 3 | 6 |
| Pages WARNING | 69 | 70 | 81 |
| Pages ERROR (text withheld) | 26 | 27 | 13 |
| Pages with a visible mark | 95 | 97 | 94 |
| Numbers missing vs layer, all pages | 4,306 | 4,329 | 2,575 |
| Numbers missing, pages that ship text | 792 | 791 | 675 |
| Pages with zero numbers missing | 38 | 37 | 53 |
| Shipped but not in layer, dense pages | not split | 163 | 279 |
| Shipped but not in layer, sparse pages | not split | 624 | 708 |
| Cost, 100 pages | $0 | $0 | $1.08 |

Hydrogen in the run:

- Called on 29 pages, accepted on 23, shipped on 24 (4 SUCCESS, 20 WARNING). On those 24 pages
  numbers missing fell from 2,230 to 62.
- Rejected 6 times by the pipeline's own checks: `table_truncated` 3, `header_unattributed` 2,
  page judge 1.
- **Blocked on 12 pages** (`paid escalation blocked (source_evidence_table)`): p007 p009 p027 p028
  p034 p044 p051 p053 p063 p067 p093 p097. The rule skips the paid reader when the pixel witness
  refuted the local model's table. Issue #1346, draft PR #1348.
- Cost is $0.037 per page on which a paid call ran, above the $0.0207 reader price: the paid
  table-check call is charged too.

Unmarked SUCCESS pages (6): p000, p074, p075 (Hydrogen; 0 layer numbers missing), p026, p045
(native; 0 missing), p002 (Hydrogen; 13 "missing"). p002 was checked against the page image: nothing
is missing; the text layer repeats the axis labels of two charts as hidden duplicates.

Pages that got worse than the local after arm (2):

- p007 (`Status of 13(3) facilities` table): local arm shipped the qwen table as WARNING; here it is
  withheld (`table_withheld`, 53 numbers). Checked against the image: the local arm's table has a
  merged header and, on the five rows with an empty "Potential Size as Announced" cell, values one
  column left; its dispute mark names only 37.3 and 17.5. Withholding is the safer ending. Hydrogen
  was blocked on this page.
- p089: local arm shipped the qwen table as WARNING `table_unverified` (3 missing); here the table
  ladder withheld it (382 numbers) after a paid table check, with no Gemini reading attempted. This
  comes from the table ladder being reachable in this arm, not from the reader.

Defects seen on Hydrogen pages (owner looked at p024):

- **Banner lists every unconfirmed number.** p024 (small-multiple charts, near-empty layer) opens
  with a 5,534-character banner: 130 entries of "reading says N, not in the text layer". Owner
  ruling: unreadable. Issue #1347, draft PR #1349 (banner becomes 191 characters; the sidecar keeps
  the list).
- **Chart shipped twice** on the same page: chart-region images with a no-values description, and
  the reader's axis-by-axis transcription. Recorded in #1347, not fixed.

## 3c. The 12 blocked pages re-run with the #1346 fix (draft PR #1348)

Same command as section 3b, source = branch `fix/issue-1346` (`23212500`). Cost $0.25.

Result: **no page changed.** Hydrogen was called on all 12 and its reading shipped on none. Status,
failure mode, numbers missing and numbers not in the layer are identical to section 3b on every
page. Lifting the block alone buys nothing and costs $0.0207 per blocked page.

Three reasons:

1. **Hydrogen's table fails the same witness check as the local model's** (6 pages: p007 p009 p034
   p044 p063 p097). On five of them the list of "unsupported" numbers is identical for qwen and for
   Hydrogen. Two independent readers read the same numbers and the witness has neither.
   - The witness is a classical OCR reading of the page image
     (`tables/source_evidence.py`, `verify_table_tokens`). A number it fails to read counts as a
     refutation of the table.
   - p007: the refuted numbers are 37.3 and 17.5. Both are printed in the PDF's own text layer
     (`Primary Dealer Credit Facility 3/17/2020 37.3`, `... 54.1 $ 17.5`). The witness is wrong.
   - p034: of Hydrogen's five refuted numbers, 2.1, 1.4, 1.2 and 6.6 are in the text layer; `-0.8`
     is not found as written (the Fed prints `-.8`; not checked further).
   - p009 p044 p063 p097 have a near-empty text layer, so the layer cannot settle them. Not checked
     against the images.
2. **Page judge accepts Hydrogen, then the same witness withholds its table** (3 chart pages: p053
   p067 p093). Page judge verdict `faithful`; final status ERROR `witness_disagrees`; the page ships
   trusted prose bands only (133, 210 and 0 layer numbers missing).
3. **Page judge accepts Hydrogen, but the local model's disputed table keeps the page** (3 chart
   pages: p027 p028 p051). The local model emitted a "table" of image names (`image1` ...), the
   witness disputed it, and an accepted later reading that carries no passed table does not lift
   that floor (#1124). The local model's page ships.

Reading: the paid block was a symptom. The part that stops Hydrogen is the classical-OCR witness
treating its own misses as refutations, on 9 of the 12 pages directly. PR #1348 should not be
merged for a measured gain; it has none until the witness is fixed.

## 3d. Rotated pages and the witness fix (#1350, draft PR #1351)

Cause of the p007 rejection in section 3c: on pages with `/Rotate` 90 or 270 the witness's
visible-text filter tested each character box in the wrong coordinate space and discarded the
character as inkless. Text-layer numbers surviving the filter on the 100 hard pages:

| Rotation | Pages | Layer numbers | Before | After fix |
|---|---|---|---|---|
| 0 | 91 | 6,208 | 5,850 | 5,850 (identical per page) |
| 90 | 5 | 104 | 26 | 104 |
| 270 | 4 | 1,373 | 629 | 1,373 |

Pipeline re-runs:

- **Nine rotated pages, local only, #1350 alone:** no page ends differently. The false rejection of
  37.3 and 17.5 on p007 is gone; the page is still WARNING `table_unverified` because the local arm
  has no table judge. On p034 the witness now rejects the local model on real misreadings (`-8`,
  `-4`, `-3` where the page prints `-.8`, `-.4`, `-.3`).
- **p007 and p034, #1346 and #1350 together, Hydrogen on ($0.04):**
  - p034: ERROR with 332 of 333 numbers missing becomes WARNING with Hydrogen's text shipped,
    **0 missing, 0 not in the layer**, marked (`table_unverified`: the table ladder reached no
    verdict). This is the first blocked page recovered.
  - p007: still ERROR `table_withheld`, 53 numbers missing. The local model's table now passes the
    witness and the page judge, so the paid reader is never called; the table ladder then withholds
    the table because a blind cell transcription disagrees. Withholding is right for that table
    (section 3b), but nothing escalates to Hydrogen after a ladder withhold. Open.

Neither fix has a measured effect alone. Together they recover p034. Pages with a near-empty text
layer and chart pages (section 3c, reasons 1-3) are not addressed by either.

## 4. Pipeline scoreboard on the gold pages

481 gold pages, each re-run as a single-page PDF, local models only (`--strict-local`), both
commits. "Audited run" is the cached cluster run `fed_rerun_51270eb5` the auditors graded.

| | Audited run (before) | Local before | Local after |
|---|---|---|---|
| Pages marked good (unmarked SUCCESS) | 272 | 264 | 239 |
| Silent defects on pages marked good | 62 (22.8%) | 56 (21.2%) | 35 (14.6%) |
| Defective pages with a mark | 102 | 108 | 129 |
| Clean pages | 235 | 235 | 235 |
| False alarms (clean pages with a mark) | 25 (10.6%) | 32 (13.6%) | 33 (14.0%) |
| of those, WARNING / ERROR | 19 / 6 | 24 / 8 | 25 / 8 |
| Pages the auditors classed withheld or marked | 82 | 82 | 82 |

The auditors' verdict belongs to the audited bytes. It carries over strictly only where the local
run reproduces those bytes: 222 pages (native-text pages). On those:

| 222 byte-reproducing pages | Before | After |
|---|---|---|
| Unmarked SUCCESS | 216 | 199 |
| Silent defects | 24 (11.1%) | 8 (4.0%) |
| False alarms among 195 clean | 3 | 4 |

Movement before to after, all 481:

- 56 defective pages were silent before. 28 are now marked (17 `visual_values_untranscribed`,
  11 `words_missing_from_page`); 28 are still silent.
- 7 defective pages were marked before and are unmarked after (g316 g327 g335 g360 g361 g364
  g367). All seven: the before arm replaced prose with "corrupt equation" crops, dropped 13 to 62
  words of the page's text layer and carried an `equation_fidelity=FAIL` mark; the after arm ships
  every text-layer word, no crop, no mark. The shipped text changed, so the auditors' "defective"
  verdict does not carry over; they are counted in the 35 above, which is therefore an upper bound.
  Not checked against the images.
- 1 new false alarm: g155 (clean page, now WARNING `words_missing_from_page`).
- Of the 82 withheld-or-marked pages, 5 were unmarked in the local before arm; 2 still are.

Limits: the gold set is not a random sample of the corpus, so these are rates on audited pages,
not corpus rates. 259 pages do not reproduce the audited bytes (a model reads them, and the local
backend differs from the cluster's), so on those the verdict is carried loosely. 43 pages
(g314-g356) and g313 were re-run after the main loop because the judge model was missing when
their turn first came.

## 5. PDF-based comparison against image-based comparison

Image-based evidence available: the Hydrogen 10-page audit, the auditors' gold verdicts (10 of the
100 pages), and agreement between readers (a number shipped by at least two other readers), used
here as a stand-in for the image where no audit exists.

Discrepancies:

1. **22 pages with a near-empty text layer** (p009 p014 p017 p026 p027 p028 p029 p030 p041 p043
   p044 p045 p050 p051 p056 p063 p064 p068 p070 p084 p093 p097). The PDF comparison reports zero
   numbers missing for every reader and counts nearly everything shipped as not in the layer.

   | Sparse pages | Hydrogen | Flash | Sonnet | qwen |
   |---|---|---|---|---|
   | Missing per text layer | 0 | 0 | 0 | 1 |
   | Missing per reader agreement | 1 | 118 | 280 | 683 |
   | Shipped, not in layer | 1,421 | 1,122 | 913 | 510 |
   | of those, two other readers agree | 861 | 841 | 840 | 482 |
   | of those, no other reader ships it | 229 | 48 | 1 | 0 |

   The layer-based table in section 2 therefore understates qwen's and Sonnet's losses: on these
   pages they miss 683 and 280 numbers the other readers agree on, and the layer sees none of it.
   Hydrogen's 229 unshared numbers are either real extra reading or inventions; the layer cannot
   say which.
2. **Hydrogen image audit against the layer count.** The 10 audited pages are those with the most
   not-in-layer numbers. The layer count flags over a thousand numbers there; the image audit found
   0 invented and 0 wrong among about 700 checked, plus 40 estimated bar values on one chart page.
3. **Layer numbers no reader ships (dense pages).** 124 of Hydrogen's 391 misses are shipped by no
   reader (Flash 154, Sonnet 133, qwen 183). These are candidates for layer-only tokens (chart axis
   ticks, hidden text) that should not count as losses. Not checked against images.
4. **Hidden duplicate labels in the layer** (p002, checked against the image). The layer count says
   13 numbers missing from Hydrogen's reading; nothing is missing on the page. The text layer
   repeats the axis labels of two charts as hidden duplicates.
5. **Pixel witness against the text layer** (p007, section 3c). The image-based witness rejects
   37.3 and 17.5 as unsupported; the text layer prints both.
6. **Gold verdicts against the layer count** (10 pages in both sets):
   - p096: auditors say clean; the layer count says 30 numbers missing for both qwen and Hydrogen.
   - p019 and p057: auditors say defective; the layer count shows 0 and 2 numbers missing for qwen.
     The defect is one the unsigned number count cannot see.
   - p099 and p089: layer count 38 and 4 missing; auditors class them withheld or marked, consistent.

## 6. Not done

- Image check of the witness-refuted numbers on the four sparse blocked pages (p009 p044 p063
  p097).
- No paid-reader escalation after a table-ladder withhold (p007, section 3d).
- Hydrogen rejections on sparse-layer and chart pages (section 3c); chart shipped twice (#1347).
- Image audit of Flash, Sonnet and qwen readings.
- Image check of the seven gold pages that lost their mark (section 4).

Working files: scratchpad `ba/` (`reader_rows.json`, `consensus_rows.json`, `arm_rows.json`,
`arm_rows_h.json`, `fix_rows.json`, `gold_before_rows.json`, `sb_100.json`, `run_arm.sh`,
`run_arm_h.sh`, `run_arm_fix.sh`).

Caveats on sections 3b and 3c: the page-judge model `qwen3.8:27b` was re-downloaded before these
runs and may not be the build the earlier arms used; these arms are not `--strict-local`; a local
run is not the cluster run.
