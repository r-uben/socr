# #1218: a model reading that drops table-row numbers its own text layer prints is marked (2026-11-06)

Overnight policy (orchestrator brief, 2026-11-06): safest option that keeps content. MARK the
page (WARNING, a visible mark, metadata, CLI), never withhold the reading, never CHECKED.
Scans and earlier-OCR layers are out of scope (#1156: the old layer never votes). Built on
#1185's normaliser (`docs/log/2026-10-20_1185-text-layer-numbers.md`).

Full suite (6f926249, army suite lock, `socr.__file__` in the worktree): 9,222 passed, 48 skipped,
9 xfailed, 1 failed, 8 errors, 405 s. The failure (`test_gh637_nougat_dropped_from_auto`) and the
8 errors (`test_gemma_cloud_opt_in` 2, `test_gh842_unservable_engines` 3, `test_orchestrator` 3)
reproduce on a clean export of the base 4e170962 on this machine (`scratch/army/1218/baseline`):
local engine environment, not this change. Format gate `uvx ruff@0.16.0 format --check .` clean.
Re-finalize parity (Kimi K5, Astra P0-A/P0-B): fixed and pinned by whole-artifact tests (fresh vs
re-finalize vs resume: audit_log.json, document and root metadata, sidecars, fragments, document
.md; an interrupted pass; a failed or different-count reassessment; an artifact in the earlier
tool's order), guarded by mutants M9, M11-M13, M16 below.

## Problem

#1185 compares the numbers a reading SHIPS with the page's own text layer and only counts the
reverse direction (`reference_only`, never decisive). A reading that omits numbers the page
prints therefore ships SUCCESS unmarked. Fed run aebc444c (precision team,
`scratch/army/precision/REPORT.md` section 3): qwen-read SUCCESS pages miss 1,312 of 27,648
printed body numbers (4.75%) on 44 pages, 16 of them miss >= 20; the native lane misses 0.

The issue body measured the obvious detector, layer-minus-reading over the whole page, as 95%
noise on calibration run 23319732: formulas typed as text, DOIs, section numbers, footnote
callouts, page numbers, running heads, chart ticks. Re-measured here (v2 row below), still
2,211 of 2,691 calibration pages fire.

## What was built

One new page-scoped check, `text_layer_drops` (FAIL only), not a result of #1185's
`text_layer_numbers`. Fable's reason (`advisors/fable.md` A4): under the same check the mode,
audit event and document note would say "ships a number the layer does not hold", which is
false of a drop; and `socr-refinalize-own-text` skips any page already carrying a
`text_layer_numbers` clause, so a run finished under #1185 (the aebc444c Fed run) could never
receive the drop clause.

- **Candidates** (`tables/layer_body.py`): the numeric TAIL of each LABEL ROW of the page's
  own layer, selected by geometry and independent of any reading. A label row is a line with a
  label (>= 1 word with >= 2 letters) followed by >= `LABEL_ROW_MIN_NUMBERS` words that are all
  numbers, no math operator on the line, not a constant-step axis run. Removed first:
  furniture (the detector's repeated word slots and folio slots), superscripts (span flag, else
  height ratio), words inside a chart or image box, digits glued to label words. Rotated pages
  are handled (line grouping by reading direction). Computed once at assessment and stored as
  `own_layer_drop_reference` on `PageAssessment` / `PageState`
  (`core/born_digital.py`, `core/state.py`), in memory only.
- **Comparison** (`tables/reading_agreement.py::dropped_numbers`): multiset of reference numbers
  minus a GENEROUS multiset of the reading (math kept, link URLs kept, `^ _ { }` split so
  `$R^2$` and `\tag{14}` yield their digits). Generosity can hide a drop and never invent one,
  so the count is a lower bound. Nothing is subtracted from the reading: an earlier version
  removed the page's running-head and folio numbers from the reading's multiset (Fable A2.2), and
  Astra showed in two rounds that every form of it can invent a drop (a faithful `Day 4` row
  against a bare folio `4`; a faithful `| Year | 2020 | ... |` row against a running head
  `Year 2020`) or, in its line-matching form, hide one (a head wrapped over two reading lines).
  Removed (OVERNIGHT.md, 2026-11-06 ~07:00): the price is that a transcribed running head's
  number can absorb a dropped body copy of the same value; measured at v3 that subtraction
  changed 27 of 2,052 dropped numbers (calibration) and 0 (Fed). A minus is part of the number.
- **Clause** (`core/verdict.py::text_layer_drops_clause`, `page_dropped_numbers`): FAIL when
  any number is dropped. Emits nothing (never UNKNOWN) on a non-model reading, an empty
  reading, a page-failed marker, `prior_ocr_layer_suspect`, a damaged layer
  (`_layer_is_damaged`), or an empty reference (empty means not checked, not zero dropped).
  Last in `_SILENT_SUCCESS_ORDER`, so #1185's slot count is unchanged.
- **Surfaces**:
  - page: status WARNING, never CHECKED;
  - mark: count only, "at least N number(s) printed in table rows on this page's own text
    layer are missing from this reading (listed in the audit log, #1218)", through the
    canonical clause-mark grammar (`marks.py` `Check.TEXT_LAYER_DROPS`);
  - failure mode: `NUMBERS_DROPPED_FROM_READING` (`core/result.py`, `manifest.py` `by_check`),
    in `PERMANENTLY_UNRESOLVABLE_MODES`;
  - audit event: kind `numbers_dropped_from_reading`, `data = {"page": n, "dropped": [...]}`,
    one per page, so the numbers live there and nowhere on the page;
  - document: `pages_ok` false, status AUDIT_FAILED / `partial`, a line in the document
    `error` note (`numbers_dropped_note`);
  - CLI: one console line listing the page numbers;
  - resume guard: `NUMBER_CHECK_MODES` = #1185's modes plus the new one; a resumed page is kept
    only when its mode is in it AND its status is WARNING (as #1185);
  - tool: `socr-refinalize-own-text` applies the clause through the same seam, idempotence
    keyed on the new check, clause placed by `in_fresh_run_order`, event written after #1185's.
- `NO_NUMBER_LAYER_HAS_SOME_REASON` (a reading that ships no number while the layer prints
  some) now points at this check for table-row numbers; other dropped numbers are still not
  checked.
- Tests: `tests/test_gh1218_layer_body.py`, `tests/test_gh1218_drops_clause.py`; the pins are
  DIFFERENCES (same page with and without the dropped row differs only in the clause, in both
  provider states), not values measured locally.

## Measured path

Two corpora, both with the production functions run offline over the saved readings: calibration
run 23319732 (2,691 qwen born-digital pages with a usable own layer; 1,511 more skipped as not
born-digital or prior-OCR) and the Fed precision pages (222 qwen SUCCESS pages). Sources:
`scratch/army/1218/data/summary*.txt`; per-page rows in `pages*.jsonl`.

| version | rule | calibration pages with >= 1 drop (of 2,691, all statuses) | calibration SUCCESS (of 2,129) | Fed SUCCESS (of 222) |
|---|---|---|---|---|
| v2 (`v2_body_run_rule`, `summary.txt`) | every geometry BODY layer number, axis runs left in | 2,211 | not printed | 97 |
| v2 axis-excluded (`v2_body_axis_excluded`) | same, same-line constant-step runs removed | 2,197 | not printed | 97 |
| v3 (`summary_v3.txt`) | tail numbers of label rows, G = 3 | 114 | 39 | 12 |
| v4 (`summary_v4.txt`) | + chart and image boxes, damage gate, tokenisation fixes | 23 (76 before the gate) | 6 (29 before the gate) | 10 |
| v5 (`summary_v5.txt`), table-overlap recovery on | + boxes that overlap a detected table are kept | 68 (89 before the gate) | 19 (32 before the gate) | 10 |
| v5, `boxes_without_table_recovery` (SHIPPED) | v4 boxes, new tokenisation | 54 | not printed | 10 |

Notes on the table. The v4 gate in `summary_v4.txt` is not the gate that shipped (see "Damaged
layers only" below); v4's 23 and the shipped 54 differ by that gate and by later tokenisation
changes, and the summary files do not isolate the two. v1 (the page multiset of the issue body)
has no summary file; `summary.txt` is byte-identical to `summary_v3.txt` (checked with `cmp`)
and carries the v2 numbers only as variant rows.

Reason for each step:

- v1 / v2 to v3: geometry BODY filtering (furniture, superscripts) still leaves folios,
  footnote markers, formulas, equation numbers and chart ticks. Only label-row tails are owed.
  Chart axes are excluded on purpose (see Choices).
- v3 to v4: the adjudication below found chart and figure pages as every Fed false alarm
  (5 of 5) and comparator artifacts on present numbers as the calibration false alarms (minus
  read as `2` on damaged layers, decimal commas, split decimals). Added: chart boxes
  (`chart_region_bboxes`) and image boxes, the production damage gate, zero-comma decimals
  (`0,332`), `:` as a decimal point, split-decimal joins.
- v4 to v5: chart boxes also hid real tables (#1268: tables inside chart boxes). Round 4
  kept a chart box that overlaps a detected table. Measured on calibration: 16 pages fire only
  because of that, adding 667 dropped numbers; 0 real drops recovered, +3 false alarms on the
  sample. Net negative, REMOVED (see Choices). Fed: 0 pages affected.

### Eye adjudication of 102 pages

Sample drawn at v3 (`data/adj/key.json`: stem, page, `fires` = v3 fired): 4 Sonnet readers, one
batch each (`out0..3.jsonl`), each page checked by image against the shipped markdown with a
quoted markdown line required. Sample strata (`summary.txt`): Fed fires 12, calibration fires by
count 13 / 13 / 13 + 1 fill, forced equation-or-reference pages 8, calibration non-firing 22,
Fed non-firing 20. Counts computed by this script, run with `python3 -I`:

```python
import json, glob, collections

key = json.load(open("data/adj/key.json"))
rows = [json.loads(l) for f in sorted(glob.glob("data/adj/out[0-3].jsonl")) for l in open(f)]
c = collections.Counter()
for r in rows:
    k = key[r["id"]]
    corpus = "fed" if k["stem"].startswith("fed-") else "calibration"
    c[(corpus, "fires" if k["fires"] else "nofire", r["verdict"])] += 1
```

Output (102 pages, 102 unique ids, 102 keys):

| corpus | v3 fired | REAL_DROP | FALSE_ALARM | v3 did not fire | CLEAN | MISSED_DROP |
|---|---|---|---|---|---|---|
| calibration | 40 | 9 | 31 | 30 | 29 | 1 |
| Fed | 12 | 7 | 5 | 20 | 20 | 0 |
| total | 52 | 16 | 36 | 50 | 49 | 1 |

The one missed drop is a wrong digit, #1185's domain. At v3 the firing pages were 16 real of 52.
The SHIPPED detector, measured end to end on the committed source (HEAD 3ba517eb exported and
imported from the export; its detector is byte-identical to the final commit's except a comment;
`scratch/army/1218/scripts/measure_final.py`, `rescore_final.py`,
`data/summary_final_3ba517eb.txt`, `data/rescore_final_3ba517eb.jsonl`), re-scored against the
same 102 eye verdicts: 14 of the 16 real drops fire, 5 false alarms fire, 0 of 49 clean pages and
the 1 missed drop do not fire; every firing page's dropped multiset equals the one the readers
checked, so no sample page needed a new eye check. Three of the 5 false alarms (Fed 2006-05-10
gb2 p28, 2007-09-18 gb2 p7, 2008-10-29 gb2 p8) are chart pages the precision team confirmed were
lost unmarked, so the mark is true there. The two real drops no longer caught (P012
jarocinski-karadi p13, P041 acosta p61) are tables under a chart or image box, which the adapter
excludes (#1268: `chart_region_bboxes` boxes some tables). Removing the reading-side furniture
subtraction changed 8 calibration pages from 1 dropped number to 0 (Heston p20, p22, p24, p32;
lanza p53, p56; lin-hsu p7; gomez-cram p10); the two of them in the sample were false alarms.
Intermediate figures quoted elsewhere in this log for v4/v5 were measured at the old decimal cut
(0.25) and are superseded by these.

Out of sample: the 36 calibration pages the shipped code fires on that the sample did not contain
were eye-adjudicated the same way (`data/adj2/out0-1.jsonl`): 28 REAL_DROP, 8 FALSE_ALARM (page
numbers taken as tail numbers, a `0.00` split as `0` `.00`, a printed `5.94 −10` written
`5.94-10`, a chart page, numbers present). Several real ones are wrong values rather than empty
cells (`1,877` shipped `1.877`, `0.343` shipped `0.243`): the printed number is absent, which is
what the mark says. All firing pages, shipped code: calibration 35 of 45 real (7 of 9 in the
sample, 28 of 36 outside); Fed 7 of 10 real, the other 3 the unmarked chart losses above.

Whole corpora, shipped code: calibration run 23319732 success pages 11 of 3,074 fire (133
numbers), warning 34 of 1,048, error 0 of 80; 1,559 calibration pages are gated off (damaged,
not born-digital or prior-OCR) and 3,538 have no label-row candidate (empty reference: not
checked). Fed run aebc444c qwen SUCCESS pages 10 of 222 fire (168 numbers); Fed was measured
through a words shim (no PDFs locally): slots from the available pages only, chart/image boxes
fetched on Rorqual with the same socr, superscript boxes absent, and the damage gate not
evaluated (no assessment), so Fed is an approximation of the shipped path.

Real drops by drop count at v3 (firing pages only, joined to `pages_v3.jsonl` on stem and page;
52 pages):

| dropped numbers on the page | REAL_DROP | FALSE_ALARM |
|---|---|---|
| 1-2 | 2 | 5 |
| 3-4 | 1 | 8 |
| 5-9 | 2 | 10 |
| 10-19 | 2 | 5 |
| 20+ | 9 | 8 |

## Constants and their derivations

- `LABEL_ROW_MIN_NUMBERS = 3`. Prose control: Fed native-transcript SUCCESS pages, 5,423 pages
  of 61 documents, 19,591 layer lines carrying a number. Lines that qualify as label rows with
  tail >= k: k=1 1,085; k=2 94; k=3 0, and 0 for every larger k up to 8. G is the smallest k
  with zero, so 3. Caveat (Kimi K4): the control is speech. It is not a null sample for
  bibliographies, regression output, numeric lists or slides, and it shares layout and lane
  selection with the pages it controls. Zero of 19,591 does not estimate the academic false-line
  rate. Kimi's own counterexample is a coalesced bibliography line (Costain; `Econometrica 36 1
  150 159` qualifies), suppressed today only by the damaged-layer gate. Not re-derived here; the
  remedy Kimi proposes is an eye-labelled candidate-line control per genre with upper confidence
  bounds.
- `DECIMAL_GAP_FRACTION = 0.06`, `DECIMAL_FRAGMENT_MIN_PAIRS = 2`
  (`data/fragment_hist.txt`). When the layer does not extract a decimal point, `0.004` is two
  words. Over 10,642 integer+digits pairs in label-row tails on both corpora, the pair counts
  have no empty valley (closely set integer cells are common). The evidence that a pair is ONE
  number is that its joined decimal occurs in the reading: 171 pairs. That support is 97 of 288
  pairs below ratio 0.06, 14 of 436 in 0.06-0.10, and zero from 0.10 to 0.54. Max F1 at
  ratio < 0.06 (precision 0.34, recall 0.57, F1 0.42). Two pairs: tails with exactly one close
  pair were supported 1 of 42 times (2%), tails with two or more 96 of 246 (39%). A wrong
  join usually adds a false drop; it can hide a real one when the reading ships the joined
  decimal (`1.23` for printed cells `1` and `23`). #1185's clause MAY then dispute it, but not
  if the page prints a genuine `1.23` elsewhere or the reading writes it inside mathematics
  that check excludes. The lower bound holds only for a correct reference. F1 0.42 is low; the cut is the
  best available, not a good separator.
- No count threshold (the design's `DROP_MIN_NUMBERS` does not exist). Any dropped number fires.
  The table above shows real drops in every count band (2, 1, 2, 2, 9), and false alarms in
  every band too, so no cut separates them; a threshold would trade real drops for nothing
  measurable. Fable's alternative (A5.4: no threshold, positive-evidence rules, k=1) is the one
  taken.
- Inherited from the precision team, derivations in the source docstrings: `FOOT_RATIO = 0.75`
  (decides only when no span flags are supplied; Fable noted it was chosen, not derived, and
  the span flag is the definition), `AXIS_MIN_RUN = 3` (two numbers always make a constant
  step), `_SLOT_DECIMALS = 1` (the detector's 0.1 pt slot rounding). The header/footer band
  fraction of `frame2.py` is not used.

## Choices made overnight

1. **Damaged layers only gate the clause, not the cosmetic flag** (OVERNIGHT.md ~05:00). The
   clause is suppressed on a layer #1185 calls damaged (not born-digital, or digit-damage
   flags) and not on #1185's cosmetic flag (lost spaces, fused words). On the 102-page sample at
   v4 (decimal cut 0.25): damaged-only 15 real drops / 7 false alarms fire; damaged plus cosmetic 11 / 4. The
   cosmetic gate costs 4 real drops (P021, P035, P038, P051). A false alarm costs an extra mark;
   a hidden drop is silent content loss. A fused pair can only hide a layer number from the
   reference, which lowers the count.
2. **The mark states the count only** (OVERNIGHT.md ~03:40). Naming layer numbers on the page
   re-shipped a number socr had withheld: in `test_native_contradiction`, a dropped-minus table
   was withheld and the mark printed `0.237`. It is also loose table text (owner rulings #1144,
   #1155). The numbers go to the audit event's `data`. Cost: a reader must open `audit_log.json`
   to see which numbers. Fable's caution agrees: a chart page can print 100 layer numbers under
   a mark.
3. **Chart axes are excluded** (STATUS.md). The brief says chart axes must not fire and
   loss of charts belongs to the deferred chart-loss team. This overrides Fable's A2.1 idea
   (count an axis run as dropped when ALL its members are missing from the reading), which
   rests on a real measurement: 419 of 587 (71%) of the first missing tokens on the 16 worst
   pages sit in a vertical constant-step column. Cost, measured: three Fed chart pages still
   fire with no chart box detected (2006-05-10 gb2 p28, 2007-09-18 gb2 p7, 2008-10-29 gb2 p8),
   and the precision team confirmed those charts were lost unmarked, so the mark is true
   there; charts that carry only ticks stay invisible.
4. **Table-overlap recovery removed** (v5). Keeping a chart box that overlaps a detected table
   put 16 calibration pages and 667 dropped numbers into the output; on the sample 0 real
   drops were recovered and 3 false alarms added. `boxes_without_table_recovery` is the shipped
   behaviour. The cost is the real tables the chart-as-table boxes hide, tracked as #1268.

## Merge-gate round (PR #1277, review in `scratch/army/1218/MERGE-GATE.md`)

Astra blocked on three P0s; Fable said merge with P2s. New commits on top of 763e03f4:

- **Resume kept the evidence (Astra P0-3).** A resumed terminal page recomputed its dropped list,
  and a reassessment that returned an empty reference wrote `dropped: []` beside a FAIL clause. The
  page's persisted `numbers_dropped` events are now restored with the page and resolved through the
  same `refinalize_own_text.resolve_dropped` the tool uses; an unconfirmable list is written as
  `unresolved`, never as an empty list. Pinned through the page-level resume path (metadata.json and
  a sibling page deleted, `_restore_terminal_page_state` spied).
- **Withheld tables (Fable P2).** The clause emits nothing on a page whose table socr withheld
  (`TABLE_WITHHELD` / `TABLE_WITHHELD_UNCONFIRMED`, the ladder disposition, or socr's withheld marker
  in the body), so the mark never calls a withhold a model drop.
- **Event kinds pinned both ways (Fable P2).** The pin found two kinds missing from
  `EVENT_KINDS_AFTER_NUMBERS_DROPPED` (`figure_description_dropped`, `figure_recoverable_labels`).
- **Axis rule needs evidence (Astra P0-2).** A constant-step tail is excluded only when every value is
  a calendar year, or the tail lies within one text height of a chart or image box. `China 10 20 30`
  is checked. Measured (`scratch/army/1218/data/summary_c2_12.txt`, `rescore_c2_12.jsonl`): 0 of 49
  adjudicated clean pages fire, adjudicated real and false-alarm pages unchanged; on the corpus 5
  calibration pages newly fire, all eye-checked false alarms (chart legends and ticks where no chart
  box was detected: Du p13, Fortier p15, p19, p24, p37; `data/adj3/out0.jsonl`). Dropping the rule
  entirely, or excluding only near a box, also fired on an adjudicated clean page (a year axis).
- **Writing forms (Astra/Fable P2).** The reading gets the layer's decimal-comma repair
  (`0,332`); a space-thousands group (`1 234`) and a bracketed negative written with a minus
  (`-1.2` for `(1.2)`) match only a layer number nothing else matched, each reading token once.
- **Row-local comparison (Astra P0-1) tried and not shipped.** Pairing layer rows with reading rows
  caught Astra's input (a dropped `China 32.7 32.2 32.9` row whose values prose repeats) but on the
  corpus added 5 pages that are all eye-checked false alarms (a reference year, a page number, a
  table written as a column of values, a prose list of maturities) and caught no real drop the
  multiset missed. Known recall gap, pinned by
  `test_known_gap_prose_repeating_a_dropped_rows_values_hides_it`; this check can only add marks.

## Mutation results

`scratch/army/1218/mutants/RESULTS.md`. Round 3, on the merge-gate tree f2e12081 (control 86
passed): 16 mutants, 14 killed at once (M1, M4, M5, M9, M11, M18 resume evidence, M20 withheld gate,
M21-M23 axis evidence, M24-M26 writing forms, M27a); the two holes, M19 (an unconfirmable list must be
recorded as `unresolved`, never `dropped: []`) and M27b (one space-thousands group satisfies one layer
number), were closed by tests in cfd877e0, each shown to fail under its mutant. Round 2, on the final commit 6f926249 (copies built
with `git archive HEAD`, `socr.__file__` asserted inside each copy, uncapped anchor count 1, same
collected count as the control; control 74 passed): 16 mutants applied, 15 killed (M1-M6, M8-M14,
M16, M17); M11b (only the persisted-list rule of the P0-A repair removed) survives because the
document-copy fallback and the count check cover it, while removing all three together (M11) is
killed. Kills resting on a single test: M6, M8, M11, M12, M14, M17. M7 and M15 targeted the
reading-side furniture subtraction and went with it. Round 1 (an earlier tree): 10 mutants, 10 applied, 10 killed. Control (unmutated
copy): 59 passed. Per mutant, failing tests of the suite that owns it:

| mutant | what | killed by |
|---|---|---|
| M1 | `verdict.py:1263` | 8 tests |
| M2 | `layer_body.py:468` | 3 |
| M3 | `layer_body.py:532` | 2 |
| M4 | `verdict.py:1174` | 4 |
| M5 | `verdict.py:1200` | 3 |
| M6 | `layer_body.py:362` | 1 (thin) |
| M7 | `reading_agreement.py:554` | 2 |
| M8 | `orchestrator.py:17395` | 1 (thin) |
| M9 | `refinalize_own_text.py:190` | 3 |
| M10 | `orchestrator.py:19502` | 3 |

M6 (zero-comma rule) and M8 (resume keeps a dropped page) are guarded by one test each.
Line numbers are those of the mutated copy at the time of the run and drift with the rebase.

## Known limits and follow-ups

Kimi (`advisors/kimi.md`), on the v3 code:

- **K1 recall holes.** Only a single geometry line with a suffix of >= 3 numbers qualifies. Not
  seen: label on one line and numbers on the next, rows of one or two cells (Filippou p69 has
  221 layer numbers and 0 label rows; a dropped `(2.07)` was found by eye), year-first rows
  (`2004 1.2 2.1 3.0`), side-by-side tables on one line (only the right one is owed), separate
  `%` / `$` cells or a trailing unit, a rotated table inside upright prose, and a genuine
  constant-step row such as `Revenue 100 200 300`, which `_tail_is_axis` rejects.
- **K2 false-alarm classes left.** Trailing-zero rounding (`2.10` against `2.1`); non-zero
  decimal commas (`1,332` against `1.332`, `1,33` against `1.33`); a glued callout
  (`1431` against `143^1`); chart labels outside the inferred boxes (Andrade p58, eight ticks);
  bibliography lines such as `Econometrica 36 1 150 159`.
- **K3 lower bound.** `dropped_numbers` consumes one global reading counter. Rows that repeat
  values are absorbed: on 2003-05-06 Greenbook p41 the two rows adjudicated absent hold 24
  cells and the missing list holds 7 (3 of 12 and 4 of 12), 29% row-local recall. Hence "at
  least N" in the mark. Kimi's remedy is an ordered row alignment with labels as anchors; the
  layer rows are already kept (`LayerNumber.row`, `row_label`) but `DropReference` flattens
  them. Not built.
- **K4** is under Constants.
- **K5 re-finalize parity** is blocking in Kimi's report: an interrupted pass writes an event
  with empty data and the `have` guard prevents repair; event order and document `error` note
  order differ from a fresh run. The fix and whole-artifact parity tests are tracked in the
  placeholder at the top.

Fable (`advisors/fable.md`):

- Equation-number lines (`(14)`): a one-token drop per numbered equation whenever the model
  omits the tag. Label-row selection removes most of these, not all.
- Line numbers in a margin (`5 10 15 20 25 30`, one per line) and flattened exponents and
  indices (`R2`, `σ2`, `xt`, `t-1`) are one full-height word each; the height rule does not
  catch them.
- A 1-page document has no furniture in either source: its folio and running-head numbers are
  BODY there.
- Tables inside chart boxes are unchecked (#1268).
- A dropped footnote block is real loss and is counted by design; section and theorem numbers
  are neutral when transcribed.

Other:

- Any number that is not a table-row tail (prose numbers, captions, a lone value) is not
  checked. This clause is a lower-bound witness for table rows, not a completeness check.
- Native lane copies the layer and emits nothing; scans and earlier-OCR layers emit nothing.

## Side findings for #1185

Adjudicators saw two real defects on pages they called FALSE_ALARM for #1218, both in #1185's
territory: P027 (2012 Krishnamurthy and Vissing-Jorgensen p16), image `[4.33]`, markdown
`[.433]`; P023, lost minus signs. Neither is flagged by this clause. Kimi's `2.10` against
`2.1` and decimal-comma false fires live in the same comparator.
