# #1261 / #1208 (plan B2): withhold the table, not the page

Status: partial. The prose of 127 pages comes back; the done bar (words missing on ERROR pages
<= 94,011) is NOT met: 154,351 -> 142,708. Reaching it needs a protective gate relaxed, which this
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
pages, a chart-region link on the recovered complete-box pages; checked on a diff).

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
2. **The 215 complete-box refusals are not one population.** The refusing gate is `box_unsane`
   (267), `order_gate` (136), `structure_outside` (116), rotation (153). The sidecar never
   recorded this; the `refusals` hook plus this replay is the record.

## What changed

`_native_order_prose_body` (manifest.py), called from the end of the geometry floor only when the
bands body fails the reading-order gate (the 136 `order_gate` pages: two columns welded by
y-banding, the layer's own order keeps them apart).

* It runs after `_each_table_box_sane` and `_structure_outside_boxes`, so a page those refuse is
  still the whole-page marker. Every detected box stays withheld (all table regions, trusted or
  not: no ordinal-to-box pairing is used anywhere).
* It takes the partition the floor already took (`_page_prose_partition`, keep-sentences) and
  changes only the ORDER the surviving words are emitted in: layer line order (block, line).
  It admits no word the bands body would not ship.
* A layer line is narrower than a band, so each line is additionally held to the keep-sentences
  rule on its own words: a line that is mostly numerals ("+-0.3", "2008", "1|") is withheld even
  when its band is not. The first measurement without this shipped 143 lines at least half numerals
  (dates and page numbers among them; also MPR 2008-07 p45 "+-0.3 +-0.7 +-1.0", p27 the year axis); with it, zero.
* A line holding both shipped and withheld words (a baseline shared by prose and a table) is
  withheld whole; a line that parses as markdown table syntax is withheld; the shipped text must
  pass `text_layer_trusted`, and the result must pass the same `_prose_follows_native_order`.
* Unchanged: scans, prior-OCR pages, rotation (both flags), native-only prose (the words are the
  PDF's), the ERROR status and failure mode, the floor credential digest (the writers hash the
  returned bytes as before), sibling tables (still withheld; second PR).

## Result on the cached run

* Pages changed: 127 (order gate refusals 136 -> 9). Words recovered: 11,643. `textlayer.words_missing.error`:
  154,351 -> 142,708. No page gained missing words.
* Leak diff, row level. On every changed page the old arm shipped only the marker, so any value
  line the new arm ships is new. All 565 new lines carrying a digit were listed
  (`b2-replay.jsonl`, `new_value_lines`). Lines whose tokens are at least half numerals: 0. Lines
  with two or more numerals and at most six tokens: 6, all running heads ("MONETARY POLICY REPORT:
  JUNE 2022 49") or a dated sentence ("June 14-15, 2022, meeting participants"). The rest are
  sentences, footnote text and running heads whose words outnumber their numerals: the
  keep-sentences class the floor already ships on the 201 pages it recovered before. No table row
  and no loose column value ships. No box content ships by construction (the shipped set excludes
  every band meeting a box). The 1,232 other pages are byte-identical between arms.
* Not measured: whether any of the 565 lines is a table footnote that a reader would want
  withheld; they are not table rows.

## What the done bar needs (owner decision)

The target 94,011 needs about 48,700 more words. Remaining reasons, by words owed:
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

`tests/test_gh1261_native_order_floor.py` (11 tests): fallback on/off difference with exact
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
