# #1261 / #1208 (plan B2): withhold the table, not the page

Status: partial. The prose of 22 pages comes back (round 5); the done bar (words missing on ERROR pages
<= 94,011) is NOT met: 154,351 -> 153,544 (round 5; 146,965 on round 4). Reaching it needs a protective gate relaxed, which this
ticket's binding conditions forbid. Evidence and the numbers that would decide it are below.

## What was measured first (before any gate moved)

`docs/log/2026-11-14_1261-measure.py` replays the born-digital geometry floor on the cached 156-doc
run (12,413 pages; no OCR). For each ERROR page with disposition `fail_closed_marker` (1,464) it
rebuilds the page state (`BornDigitalDetector.detect_page` -> `apply_born_digital`, the run's own
sidecar count and boxes, `native_words` from the PDF) and runs `table_floor_text_for_source` twice
in one process, changing only whether the new fallback runs. `table_floor_text_for_source` gained an
optional `refusals` list: each whole-page return records its reason. Production passes nothing.

Fidelity: of the 1,359 born-digital pages replayed, 1,175 reproduce the run's floor bytes exactly.
The other 184 differ only by what is added after the floor (a chart-region image on 20 no-detection
pages, a chart-region link on the recovered complete-box pages; checked on a diff). Those 184 are
excluded from every old-versus-new total below (their old arm is not the run's), counted
separately by the script; none of them changed between arms. The refusal table below counts all
1,359 replayed pages.

Why each page refused, old arm, with the text-layer words the page owes
(scoreboard definition, `textlayer.words_missing`):

| class (sidecar) | reason | pages | words missing |
|---|---|---:|---:|
| complete boxes | recovered already | 201 | 10,581 |
| complete boxes | `box_unsane` | 267 | 38,597 |
| complete boxes | `order_gate` | 136 | 20,318 |
| complete boxes | `structure_outside` | 116 | 13,225 |
| complete boxes | `rotated_lane` / `rotated_page` | 96 / 57 | 11,301 / 5,694 |
| complete boxes | `no_recoverable_prose` | 9 | 892 |
| complete boxes | `prior_ocr` | 1 | 160 |
| no detection (count 0, no box) | no boxes, no chart floor | 300 | 31,737 |
| no detection | `prior_ocr` | 124 | 14,621 |
| no detection | `rotated_lane` | 52 | 3,967 |

Another 105 floor pages are scans (not born-digital, out of scope).

Two findings that change the brief:

1. **There is no partial box inventory on this run.** Every sidecar with a detector count has
   `len(boxes) == count`; the other 600+ pages have count 0 and no box, and a fresh
   `detect_page` on all 476 born-digital ones confirms count 0 / 0 boxes. The brief's "539 pages
   where a box is missing" does not appear in the replay. The partial-box path the advisor asked
   to build (and the owner approved in DECISIONS) would fire on zero pages here, so it was
   built, found to touch zero pages and removed rather than shipped as dead loosening of a fail-closed
   gate. If a later corpus has such pages, the replay hook tells which gate refuses them.
2. **The 682 complete-box refusals are not one population.** The refusing gate is `box_unsane`
   (267), `order_gate` (136), `structure_outside` (116), rotation (153), no recoverable prose (9),
   prior OCR (1). The sidecar never
   recorded this; the `refusals` hook plus this replay is the record.

## What changed

`_native_order_prose_body` (manifest.py), called from the end of the geometry floor only when the
bands body fails the reading-order gate (the 136 `order_gate` pages: two columns welded by
y-banding, the layer's own order keeps them apart).

* It runs after `_each_table_box_sane` and `_structure_outside_boxes`, so a page those refuse is
  still the whole-page marker. Every detected box stays withheld (all table regions, trusted or
  not: no ordinal-to-box pairing is used anywhere).
* It takes the partition the floor already took (`_page_prose_partition`, keep-sentences) and
  re-emits the surviving words in layer line order (block, line). It admits no word the bands
  body would not ship, and it withholds more (below).
* **Per line and per column.** A layer line is narrower than a band, so each line is held to the
  keep-sentences rule on its own words: a line that is mostly numerals ("+-0.3", "2008", "1|") is
  withheld even when its band is not. A numerals-withheld line also takes down every line on its
  side of the band's widest gap between layer lines (its column): `Real GDP 2.3` beside `2.5`
  withhold together, and the stub labels and legend entries of a table whose values were withheld
  go with them (Fable 1 and 2). Applying the rule to the side's pooled words adds nothing beyond
  that (a pool is mostly numerals only when one of its lines is), so it is not a separate check.
* **Table continuations (Astra's blocker).** A detected box can be clipped: the ruled final row
  `Total pension obligation` / `12 million dollars` sat just outside it, with one numeral against
  two alphabetic tokens, so neither the per-line rule nor the structure gate (no recurring numeric
  lanes) withheld it, and the old order gate, which used to refuse the page, no longer does. Now
  the walk goes away from each box, above and below, band by band: the lines of a band that
  overlap the box's horizontal span are withheld, all of them, while any of them carries a
  numeral; the walk stops at the first band whose lines in that span carry none. A box whose
  geometry cannot be read (fewer than four numbers, non-finite, inverted) makes the body abstain:
  the whole-page marker. Not covered: a continuation that is numeral-free (a row of Yes/No or of
  labels) and a continuation outside the box's horizontal span; those keep shipping, as before.
* A line with no word of two or more letters (shredded inline math: `t`, `y`, `+`) is withheld
  (Fable 3). Six replayed pages shipped such lines on the earlier revision: bluebook 2006-03-28
  p22, 2007-01-31 p29, 2007-10-31 p35, 2008-04-30 p40, 2009-09-23 p69, 2009-11-04 p73. None
  ships one now.
* A line holding both shipped and withheld words (a baseline shared by prose and a table) is
  withheld whole; a line that parses as markdown table syntax is withheld; the shipped text must
  pass `text_layer_trusted`, and the result must pass the same `_prose_follows_native_order`.
* **Marks (Fable 4).** The table marker plus image appears once per run of withheld lines that
  holds a line inside or continuing a detected box. Any other run gets a neutral `[line withheld]`
  or `[N lines withheld]` (not a page mark; the order check treats it as furniture). The page keeps
  the table marker at least once. The earlier revision printed it per run: median 6 per changed
  page, 26 at most, most standing for a page number or an axis tick. Now: median 2, at most 3.
* A paragraph breaks where the layer block (`w[5]`) changes (Fable 5).
* Unchanged: scans, prior-OCR pages, rotation (both flags), native-only prose (the words are the
  PDF's), the ERROR status and failure mode, the floor credential digest (the writers hash the
  returned bytes as before), sibling tables (still withheld; second PR).

### Limits stated, not fixed

* **The order gate is close to a tautology here.** The fallback emits `get_text("words")` order
  and is checked against `get_text()` order, the same source. It still refused 20 pages, but order
  is only as good as the PDF's block order: on greenbook-part1 2010-03-16 p27 and bluebook
  2006-03-28 p22 the page header ships last. This body is not "held to the same order gate" in any
  stronger sense than that.
* **"Paragraphs" in the banner includes chart titles, legend entries and stub labels** of charts
  and unboxed tables that print no numeral (greenbook-part1 2010-03-16 p27 ships chart titles and
  legend entries as paragraphs). They carry no number; they are not prose either.
* **Hand read.** Fable read the shipped bodies of all 127 changed pages of the first revision
  (no table row and no loose value found); the rest of the leak evidence is the rule-based listing
  below. This revision's bodies were not read page by page; they differ from the read ones only
  by withholding more and by the marks.

## Result on the cached run

* Pages changed: 79 (order gate refusals 136 -> 57), all of them among the 1,175 pages whose old
  arm reproduces the run. Words recovered: 7,386 (9,737 on round 3, 10,388 on round 2 and 11,643
  on the first; the difference is what the span rule costs: a paragraph under a table ships only
  when it is closed). `textlayer.words_missing.error`: 154,351 -> 146,965. No page gained missing words.
* Leak diff, row level (multiset: a line the new arm ships once more than the old arm counts).
  On every changed page the old arm refused with `order_gate`, i.e. shipped only the marker, so
  every value line the new arm ships is new. 362 new lines carry a digit (72 pages); 7 of them
  are the neutral `[N lines withheld]` marks (the digit is the count), leaving 355 content lines.
  The listing is the replay output (`new_value_lines`, written by `2026-11-14_1261-measure.py`;
  the output is not committed, rerun the script). Content lines whose tokens are at least half
  numerals: 0. Lines with two or more numerals and at most six tokens: 0 (round 3 had 5 running
  heads). The rest are sentences, footnote text and running heads whose words outnumber
  their numerals: the keep-sentences class the floor already ships on the pages it recovered
  before. Lines without a two-letter word on the changed pages: 0. (The previous revision's
  "450 lines, 73 pages" counted those marks too.)
  No box content ships by construction (the shipped set excludes every band meeting a box), and
  the span rule withholds every line in a box's span that is not part of a closed paragraph.
* Not measured: whether any of the 355 lines is a table footnote that a reader would want
  withheld; they are not table rows. Whether the walk fires on the corpus is not separated from
  the other rules: the replay shows only the result.

## What the done bar needs (owner decision)

The target 94,011 needs about 52,900 more words (146,965 - 94,011). Remaining reasons, by words owed:
`box_unsane` 38,597 (267 pages), no-detection pages with a TR-3 failed region but no detector box
31,737 (300), `structure_outside` 13,225 (116). Each is a protective gate:

* `box_unsane`: a box that captures no table band or swallows prose is not the table it claims to
  bound. Relaxing it needs a per-box repair or the structure-outside check as the only guard;
  the advisor's single-row, categorical and clipped-box cases must be tested first.
* No-detection pages: the detector drew no rect, so nothing marks where the table is. The TR-3
  verifier knows the failed region's y-range (logged, not stored). Storing that rect in the sidecar
  and withholding it is the brief's "store rects" step and is a larger change (new durable audit
  field, resume restore, #1144's recorded 41 flattened categorical rows without a box).
* `structure_outside`: refuses by definition because table structure is printed outside every box.

The sidecar still does not record the refusing gate; a durable per-page reason is the brief's C2
record gap, left for its ticket (a new durable event must restore on resume).

## Tests and mutants

`tests/test_gh1261_native_order_floor.py` (29 tests after round 4): fallback on/off difference with exact
recovered prose and order; same-words subset check against the bands body; structure-outside and
box-sanity gates (recorded reasons); order gate on the fallback; rotation, prior OCR, scan;
shared-baseline line, loose value line, table-syntax line; floor credential and repeat selection;
`process()` in both provider states (ladder populated and empty; `primary_engine`,
`local_engine`, `enabled_engines` pinned by the shared helper) plus a real restore and re-assemble
(fragment and final .md byte-identical). Mutants (src and tests copied outside the repo,
`socr.__file__` asserted inside the copy, anchor count asserted 1): remove the fallback call,
structure-outside gate, box sanity gate, shipped-words filter, per-line numeral rule, order gate on
the fallback, rotation gate, table-syntax filter. All eight killed.
The prior-OCR gate in `structure_class_floor_text` is shadowed by an identical check inside
`native_prose_floor_text`, so removing either alone is an equivalent mutant; the test asserts the
outcome, not the branch.

Review round (PR #1324): the same file gained the clipped-box case at the shipping boundary
(`table_floor_text_for_source`, with and without the ruled rows; the prose after them ships), the
abstention, the column withholding, the two-letter rule, the marks and the block break. Mutants,
same method (src and tests copied outside the repo, `socr.__file__` asserted inside the copy,
anchor count asserted 1): remove the continuation walk (2 tests fail), remove the same-column
withholding (2 fail), mark every run as a table run (1 fails), drop the block break (1 fails),
drop the two-letter rule (1 fails), drop the abstention (1 fails). All six killed.

Review round 3 (PR #1324): the numeral-based continuation walk is replaced by a sentence-based
one. From each box the walk goes down and up through the bands that overlap the box's span and
withholds every line there until a band whose text in the span is a sentence
(`_is_sentence`: the floor's prose test, numeral-free, at least
`_NATIVE_ORDER_SENTENCE_MIN_WORDS` = 5 words; the constant's reasoning is at its definition).
`Pension liabilities`, `Coverage included` / `Yes`, stub labels and legends are not sentences.
The column split is replaced by a spread: a numerals-withheld line withholds the lines beside it
in its band, in x order both ways, until a sentence line, so no cell-gap width leaves a fragment
of a split row. Cost: a short opening line of a paragraph, or a numeral-bearing one, directly
under a table in its span is withheld under a neutral mark (9,737 words recovered vs 10,388).
Tests: Astra's two reproductions (both cells withheld, difference against the same page without
the rows, the sentence below ships), Fable's 60 pt cell-gap row (difference against the row
without the loose cell). Mutants (src and tests copied outside the repo, `socr.__file__` asserted
inside the copy, anchor count asserted 1): the walk stops at a numeral-free band (2 tests fail);
no spread (3 fail). Both killed.

Review round 4 (PR #1324): Astra showed a word count does not establish a table boundary
(`Pension liabilities for retired employees` then `Total pension obligation` / `12 million
dollars`; `Coverage for all employees included` / `Yes`: both shipped, a longer label reopens
every leak a threshold closes). The sentence-count stop (`_is_sentence`,
`_NATIVE_ORDER_SENTENCE_MIN_WORDS`) and the walk are removed. The rule is now per layer line:
within the horizontal span of any detected box, and in the band of any numerals-withheld line, a
line ships ONLY if its layer block (`w[5]` of `get_text("words")`) (a) has every line shippable
(no word inside a box, no numerals, table-syntax or shredded line), (b) has at least two layer
lines, and (c) its last line ends in sentence-final punctuation (`.`, `?`, `!`, then optional
closing quotes or brackets: `_SENTENCE_FINAL_RE`). Every other line in the span is withheld, on
the whole page (no walk, so nothing to stop early). A shredded-math line (cubic P1) makes (a)
fail for its block, so it cannot release the lines after it. Outside every span the earlier rules
apply unchanged. Clause (a)'s "no word inside a box" is implied by the shipped-words filter
(the partition withholds every box), so it is not a separate check. The spread of numerals
withholding and the walk collapse into this one predicate. Marks: a withheld run is a table run
when it holds a line in a box or in a box's span; a run withheld only through a numerals band
gets the neutral mark, as before.
Cost: a single-line paragraph, or a paragraph whose last line does not end in a stop (a heading
followed by text, a list), directly in a table's column is withheld under a mark: 9,737 -> 7,386
words recovered, 116 -> 79 pages. The previous 116 included pages whose only recovered prose was
exactly such lines. The fixtures that used one-line blocks in the left column now use real
two-line paragraphs (`insert_text` with a newline keeps them in one layer block).
Tests added: Astra's two long-label reproductions, each as separate-block lines and as one
block, through `table_floor_text_for_source` (nothing of the table ships; difference against
the same page without the rows); a real two-line paragraph after a table ships; a one-line
block, a block without a final stop, and a block holding a shredded-math line in a span are
withheld, with the same words shipping outside every span (difference); a closing quote after
the stop counts. Mutants (src and tests copied outside the repo, `socr.__file__` asserted inside
the copy, anchor count asserted 1): drop (a) (1 test fails), drop (b) (1 fails), drop (c)
(8 fail). All three killed. The full replay was rerun with
`docs/log/2026-11-14_1261-measure.py` (output directory redirected to a scratch copy).

## Review round 5 (Astra recheck 3, cubic P2 on b0eeb077)

Astra: a ruled row whose two cells are two-line blocks ending in a period passes the
closed-paragraph rule; block shape does not establish prose. Reproduced as a real PDF (clipped
box, cells at x=76 and x=230, borders at y=176/212 and x=72/220/320): both cells shipped.
Coordinator rule, structural, replacing the exception: within the horizontal span of ANY detected
box (any word whose x-range overlaps a box's x-range, top of page to bottom) NOTHING ships. Only
lines entirely outside every box's x-span ship, under the earlier rules (numerals band,
two-letter word, keep-sentences, block breaks). `_closed_paragraph`, `_SENTENCE_FINAL_RE` and the
per-block index are removed. The band of a numerals-withheld line is withheld whole, as before.
Cubic P2 (the quote-boundary test): the case lies inside a box span now, so the test and the rule
it pinned are dropped.
Cost, from the full replay (`2026-11-14_1261-measure.py`, output redirected to a scratch copy):
pages changed 79 -> 22 (order-gate refusals 136 -> 114), words recovered 7,386 -> 807
(`words_missing.error` 154,351 -> 153,544; no page gained missing words). A two-column page
whose table box spans one column loses that whole column. The done bar (94,011) is further away;
see #1323.
Leak diff: 32 new lines carrying a digit on 6 pages, 5 of them neutral `[N lines withheld]`
marks, 27 content lines (running heads and sentences). Lines at least half numerals: 0; without
a two-letter word: 0; two or more numerals in six tokens or fewer: 0; inside a box: 0.
Tests: the fixture's left column (inside GRID_BOX's span) now ships nothing and only the right
column ships, in order. Added: Astra's multiline ruled-cell row through
`table_floor_text_for_source` (both cells withheld; difference against the page without the row),
a closed two-line paragraph in a span withheld (and shipping outside every span), the other
column's prose in order. All earlier long-label and categorical-row reproductions still withhold.
Mutant (src and tests copied outside the repo, `socr.__file__` asserted inside the copy, anchor
count asserted 1): `continuation = band_span` (allow any line in a box span): 17 tests fail.
Killed.
