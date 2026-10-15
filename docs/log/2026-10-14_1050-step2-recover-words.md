# #1050 step 2 — read the words the PDF draws as shapes

Owner review of step 1 on real pages (2026-10-14): Schoenegger p10's framed
prompt box is "just text", not an image, and step 1 cut it into 4 crops;
Acosta p21's pasted newspaper clipping is a figure that could be OCR'd, and
step 1 cut it into 5. Two jobs: stop the over-splitting, then read the regions.

## 1. Over-splitting

Step 1 links rows only when their vertical gap is under a glyph height, so a
paragraph break or a column gutter inside one block starts a new region.

The fix takes grouping from the page's own geometry, not from a new gap rule:
two regions inside the SAME drawn rectangle (`get_drawings`) or the SAME placed
raster (`get_image_info`) are one block. Measured on the review pages:

| page | container the page itself declares | regions before | after |
|---|---|---|---|
| Schoenegger p10 | one drawn rect `(48.5, 81.2, 437.5, 571.5)` | 4 | 1 |
| Acosta p21 | one raster `(116.0, 70.8, 674.6, 923.4)` | 7 | 1 |

The merged box is the union of the member regions, **never the container**.
Acosta's clipping raster is placed over 85% of the page because the scan has
wide white margins; its ink is 5%. Cropping the container would hand the model
the whole page, native prose included.

Two guards, both using `crop_descriptions.PAGE_SIZED_BBOX_FRACTION` (already
measured for figure descriptions):

- the union a container would produce is never page-sized, so no merge can
  become a whole-page crop;
- a page-sized **drawn rect** is not a container at all. It is furniture — a
  border, a background fill — and contains every region by construction, so it
  would merge unrelated ink across the prose between them (Argon). A page-sized
  **raster** is still a container: it is one placed image and everything inside
  it came from that one object, which is exactly Acosta's 85% clipping scan.

Not handled: a frame drawn as four separate rule segments (LaTeX `\fbox`,
`mdframed`) encloses nothing, so its contents do not group. Schoenegger's box is
a single filled rect.

Over the 20-page sample below, 38 of 91 regions are merges.

## 2. Reading

`socr/core/text_recovery.py` (pure; the orchestrator injects the reader). The
reader is the one the number-free figure describer already builds —
`qwen3-vl:30b-a3b-instruct` via the run's Ollama backend. No new client, so it
is correct under `--strict-local` and a zero cost cap with no extra branch.
Flag: `--no-recover-uncovered-text` (on by default), in the run fingerprint.

Placement follows the 2026-10-09 ruling:

- region inside a raster (a figure) → step 1's picture stays and the words go
  **beneath** it, `> Text in figure (OCR): …`;
- region that is not in a figure (Schoenegger's box) → the words are
  substituted into the prose at their reading-order position,
  `> Text drawn as shapes (OCR): …`, and the picture link is dropped.

The anchor is the tail of the last native text block that ends above the
region; insertion is line-anchored and whitespace-tolerant, because a shipped
native page separates its printed lines with single newlines and can be one
blank-line "paragraph" end to end. An anchor that is not found falls back to
the page tail — it never guesses a position.

A region with no reading is exactly step 1: picture and mark, no text.

## 3. The witness, and the rule it produced

Tesseract on the same crop, through a public seam over the existing renderer
and reader. **Superseded:** the witness and that seam were deleted in round 5's
item G; nothing is verified now. Kept here because the measurements below were
taken with it.

**Measured 2026-10-14**, 91 regions on 22 pages (the 2 review pages plus a
seeded sample of 20 step-1-flagged silent-SUCCESS native pages from run
23117187, seed 20261014):

- 86 of 91 regions produced a model reading; 5 produced none.
- Tesseract returned **nothing at all** on 27 of 91, including pages of plain
  axis labels the model read correctly. On these crops it is the weaker reader.
- Content recall of the readings it could score has **no valley to cut at**:
  deciles 0.34 / 0.50 / 0.67 / 0.85 / 1.0 / 1.0 / 1.0 / 1.0 / 1.0.
- A hand check found correct and wrong readings sitting together at recall 0.0
  (correct: "VIRTUS INTERPRESS", a publisher logo; wrong: a decorative strip of
  crowns read as "PENNY E Z").

So no tuned agreement threshold was defensible. The rule is socr's **existing**
source-evidence rule applied to a crop: confirmed only when the witness
independently read every numeric token at least as often and every content
token at least once. Words the witness read but the model did not are not a
failure.

One vacuity hole, found by hand and closed: a textless squiggly line chart
(Menzly–Ozbas p21) was read as `y`, which is below the witness tokeniser's
content-token length and is not numeric, so containment held over two EMPTY
sets and the reading confirmed itself. A reading with no comparable token at
all now cannot confirm (`no_comparable_token`).

**Outcome on those same 91 regions (22 pages, pre-dating the whole-figure
rule, the dropped label and the mask), re-scored with the FINAL rule:** 23 confirmed, 63
unconfirmed, 5 unread, out of 1,075 words recovered. The two-sided and
order-sensitive rules and the unclassified-token rule cost 5 confirmations
against the first-round rule, every one of them a case the old rule could not
actually see. Unconfirmed costs nothing but a mark: the words still ship. Every
confirmed reading inspected is correct; the two readings the hand check found
wrong are both unconfirmed. Cost: 1 confirmed reading lost to the vacuity guard
that was correct (a date axis whose tokens are neither numeric nor content).

An unconfirmed reading is **not discarded**. It ships with the doubt in its own
mark (`(OCR, unconfirmed)`) and keeps step 1's picture — the same shape as the
2026-10-12 disputed-table ruling: ship marked, with the image to check against.

A confirmed mark says what was read was corroborated, not that everything on
the crop was read: Cochrane p9 confirms "RR y response / RR p response" and
leaves the axis ticks unread.

**Seconds** (Mac, shared GPU, local Ollama): model read per region median 5.3 s,
p90 11.8 s, max 120 s, 996 s over 91 regions. Tesseract per region median
0.23 s, max 1.07 s, 26 s in total. The witness costs about 2.6% of the model.

## Fix round 1 (Astra + Fable + Argon, 2026-10-14): 11 items + 2

Three independent reviews said do not merge. Every item below has a difference
test and an external mutant that the test kills; the mutant harness is
`scratch/1050s2/mutate.py` (external copy of `src`+`tests`+`pyproject.toml`,
`socr.__file__` canary inside the pytest process, single-occurrence anchor
assert before each edit).

**A — "confirmed" was too weak to drop the picture.** It was one-sided (only the
reading's tokens had to appear in the witness), so a TRUNCATED reading confirmed
and took the unread lines with it; and it compared bags, so `Alice 10 / Bob 20`
confirmed `Alice 20 / Bob 10`. Owner ruling: **every** recovered region keeps
its picture, confirmed or not. The check is now two-sided and numbers must
match as a SEQUENCE. No threshold was added.

**B — a merged crop could swallow native text.** Two bands enclosing a native
line produced a crop containing it, and the page shipped that line twice. A
merge whose union overlaps a visible native character box is refused.

**C — the native fallback dropped the words and the picture.** `manifest.py`
rebuilt the page from the frozen native snapshot and kept only the slice from
the page mark onward; a block placed in reading order sits BEFORE that mark.
It now carries every coverage part the snapshot lacks. Reading-order placement
is not preserved on that path — the parts land at the page tail.

**D / E — placement order.** Two regions sharing an anchor (or both above all
native text) were inserted second-before-first, and skipping only blockquotes
split a block from its own picture. Insertion now steps over any part step 2
already placed, picture included.

**F — a vector figure counted as "not a figure".** `in_figure` keyed on
`kind == "raster"`, so a matplotlib axes frame (a drawn rect) put outlined axis
labels into the prose. It now keys on the figure merge below.

**H — `--native-only` called a model.** It now carries the same gate the figure
describer has. Added a per-page crop-read budget derived from
`_page_ladder_budget_sec` (each region is one call, as each rung is one call) —
no new number — and a wedged reader now arms the EXISTING cascade-halt latch
rather than a second halt path.

**J — "no reader is exactly step 1" was nearly true.** The tail join used
`\n\n` where step 1 used `\n`. When nothing was recovered the output is now
byte-identical to step 1 again; verified against main's shipped Acosta page,
which differs only by the intended 7-crops-to-1 merge.

**K — the anchor picked the wrong column.** On a two-column page every
right-column block also ends above a left-column region. An anchor candidate
must now share some of the region's horizontal span.

**L — a token the check could not classify was skipped silently.** `2019q1`
begins with a digit so it is not a content word, and it is not a number, so
nothing compared it: measured on Ayivodji 2022 p22, the model read `2019q1`,
Tesseract read `2019q7`, and the region CONFIRMED. Unclassifiable tokens must
now match exactly, as a sequence. They can only BLOCK confirmation, never earn
it — a textless chart read as `y` must still fail the vacuity guard.

**I — the configured backend/host is ignored.** NOT FIXED. The reader is
`OllamaFigureEngine`, which hardcodes localhost; so does the figure describer it
borrows. Fixing it here would fork that engine's host handling for one caller.
Filed as its own follow-up rather than done badly.

### Whole-figure pictures (owner ruling, 2026-10-14)

A chart's labels each became their own crop: **Hansen et al. 2018 p23, 23 crops;
Bugel et al. p23, 18** — almost all single labels ("Quarters", "Top 1% share").
A label cut out of its chart cannot be checked by eye, which defeats keeping the
picture. Regions inside one figure now ship as ONE whole-figure crop, read once.

| page | step 1 | after this round |
|---|---|---|
| Hansen et al. 2018 p23 | 23 | 1 (whole figure) |
| Bugel et al. 2026 p23 | 18 | 1 (whole figure) |
| Schoenegger p10 | 4 | 1 (text, not a figure) |
| Acosta p21 | 7 regions (5 crops shipped) | 1 (figure) |

Both halves of "inside one figure" are existing detectors —
`figures.extractor.figure_boxes` (raster) and `tables.reconstruct.chart_region_bboxes`
(vector clusters). Neither separates a chart from a box of VECTOR-DRAWN TEXT:
measured, each returns exactly one cluster for Hansen p23 AND for Schoenegger
p10, and `has_chart_marks` is False on both. Shipping Schoenegger's prompt box
under a picture would contradict the owner directly ("it's just text").

The separation needs no new detector and no threshold, because the ink is
already segmented: **a chart carries drawn marks that lie outside its uncovered
text; a box of drawn text carries only letters.** Rules are excluded using
`MAX_THICK_PX`, the rule thickness the ink filter already uses — Schoenegger's
box contains exactly one non-text drawing, a 1 px rule; Hansen's chart contains
259 marks. A mutant that drops this condition flips Schoenegger to "figure",
checked on the real page, not only on the synthetic pair.

The native-text guard from item B is deliberately NOT applied to a figure: a
chart's own axis labels have a text layer, and excluding every figure that
contains one would leave nothing. A figure's label text can therefore appear
both in the page's native word-salad and in the recovered block.

## Fix round 2 (Astra + Argon, both: do not merge) + owner ruling G

**G — the "confirmed" label is gone (owner ruling).** Three review rounds found
token containment passing wrong readings: a label/value swap (content is an
unordered set), dash-led numbers (`—5` vs `—7` fall out of every bucket), glued
tokens (`pval=0.05` vs `pval=0.99`, `2019Jan` vs `2024Jan` both reduced to
"jan"). Rather than patch it a fourth time, the label was dropped. Every reading
now ships under ONE mark, `(OCR, not verified)`, beside its picture, and the
page mark never says "recovered" as if a check had happened. The Tesseract
witness is deleted from this path with its cost: it only ever decided that
label. What it incidentally guarded — a crop with no readable text at all, read
as a single letter — is now nobody's: that is #1152. `CONFIRMED_STYLE_RE` and
`test_no_block_can_ever_claim_it_was_checked` exist so the label cannot come
back by accident; a mutant that restores a confirmed branch fails 13 tests.

**A — figure grouping duplicated native prose.** `chart_region_bboxes` expands a
cluster to the page's LEFT EDGE, so a right-column chart's rect reached across
the left column; cropping and reading it shipped that prose twice. The figure
box is now the cluster's OWN extent (the union of the marks inside it), and a
framed callout is excluded outright: a native CONTENT word (socr's existing
token definition) sitting inside the box and touching no mark means the box
holds prose, not a figure. A frame's own bbox covers its interior, so the frame
is skipped when asking whether a word sits on a mark — otherwise every callout
looked like a chart. Measured: Hansen p23's only free words are the panel
labels "(A)" and "(B)", which are not content tokens.

**B — raster charts lost their picture.** A placed image IS a figure and its
bbox IS the picture; nothing further is asked of it. Requiring vector marks
shipped a 103-px label strip without its bars. A spans-the-cluster exemption for
thin plot lines was tried and REVERTED: a framed text box's divider rule spans
it too, and it flipped Schoenegger p10 to "figure". Rule-thinness is tested on
the SHORT side, so a data polyline — which rises and falls — is never rule-thin;
only a flat line is, and that is an axis.

**C — real timeouts never armed the cascade halt.** `OllamaFigureEngine.ask`
catches every exception and returns `None`, so the latch could only be armed by
an exception injected above the adapter, i.e. never in production. The reader is
now wrapped: a `None` that consumed the engine's whole deadline
(`FIGURE_ASK_TIMEOUT_SEC`, the existing 120 s, named not changed) raises
`TimeoutError`, which is the one thing a refusal cannot do. A wedge also stops
the remaining regions on that page, and the between-page consumption now has its
own two-page test — the mutant that drops it used to survive.

**D — the budget did not bound reader time.** It was `600 × (regions + 1)`, so
23 labels could burn 46 minutes. It is now the flat page budget,
`_page_ladder_budget_sec([])`, which does not grow with the region count.

**E — recovery off was not step 1.** Grouping ran unconditionally. It exists to
make ONE crop readable, so with no reader it only changed bytes. It is now
gated: with `--no-recover-uncovered-text` both review pages are **byte-identical
to main's shipped output**, verified against the committed run, not against two
runs of this branch.

**F — coverage.** Picture assertions now also cover the assembled document and a
resumed page, and the halt has the two-page test above.

## Fix round 3 (Fable, reproduced + guided Argon, static)

**1. Root fix: the reader never sees native text.** Most P0s across three rounds
were one class — the crop handed to the model contained words the text layer
already had, so the reading repeated them: a raster placed under prose, a
callout with an icon or a header band, a two-column page whose chart box
reached across. Rather than keep chasing shapes, what the model READS is now
decoupled from what the reader SEES: the crop sent to the model has every
visible native character painted out (`render_region_png(..., mask_native=True)`,
using the boxes `analyze_page` already computes), while the picture that ships
is the unmasked whole figure. The duplication class is closed at the source,
however imperfect the figure detection is. Driven in test by a REAL reader
(Tesseract) over the real crop, asserting the property — no content word of a
recovered block appears among the page's native words — rather than one
sentence, because a GARBLED copy of the native text is the same duplication.
The mutant that sends the unmasked crop fails that test.

A bug found while building it: the masking context manager wrapped its `yield`
in `try/except`, and a `@contextmanager` receives whatever the WITH BODY raises
at the yield point — so it swallowed the model read's `TimeoutError`, which is
the cascade-halt signal. The except now covers only the masking.

**2. The wedge latch leaked across documents.** `_recovery_backend_wedged` is a
class attribute and `process_batch` reuses one pipeline instance, so one wedged
document halted every document after it. It is reset per document in
`_prepare_text_coverage`, with a two-document batch test.

**3. Equation sidecars on NATIVE_FALLBACK — NOT REAL** (confirmed independently
by Fable in round 4; Argon's static trace was wrong).** Argon's claim was
static. Reproduced through the real selection seam
(`scratch/1050s2/probe_sidecar.py`): with a reading-order block inserted, the
equation, the recovered block, the picture and the page mark all survive
`SelectionProvenance.NATIVE_FALLBACK`. The trace misses round 1's item C fix —
`manifest._coverage_parts_to_carry` carries every coverage part the frozen
native snapshot lacks, wherever it sits, which is exactly what the claim says
is missing.

**4. Halt and budget edges.**
- The halt is armed from the READING CALL's own outcome, with no health probe.
  `_probe_backend_idle` asks `/api/tags`, which answers 200 in milliseconds on a
  jammed GPU, so gating on it made the latch unreachable in the case it exists
  for. **Deliberate deviation from Argon #6:** a fast failure does NOT arm the
  halt. A quick `None` is "no model", which is step-1 behaviour; halting a
  document because the reader is simply absent would be worse than the bug.
- A call may not START unless the budget can hold a whole one
  (`FIGURE_ASK_TIMEOUT_SEC`). `PageLadderBudget` documents that a call beginning
  inside the budget keeps its own deadline, so "> 0 left" let a 180 s budget
  reach ~300 s.

**5. P2s.**
- `_COMMENTARY_RE` dropped any reading opening "I ", "It appears" or "Here is".
  A label starting with "I", or a note reading "It appears in Table 3", is
  ordinary page text, and dropping it is silent content loss. Split in two: an
  exact lead-in ("Here is the transcription:") is STRIPPED and the rest kept;
  only a refusal returns nothing.
- `_already_present` compared single paragraphs, but the shipped piece is a
  block PLUS its picture — two paragraphs — so applying twice inserted it twice.
  It now matches a RUN of paragraphs, and still refuses a prefix.
- With no reachable reader, grouping still ran, so a host without Ollama shipped
  merged crops main would have shipped separately. Availability is resolved once
  per document and gates grouping: no reader, no grouping, step 1 exactly.
  `metadata.json` under `--no-recover-uncovered-text` is pinned clean too.

## Fix round 4 (Fable reproduced + guided Argon) — everything left was in the mask

**1. Invisible native text was not painted (P0).** The mask painted VISIBLE
chars only, so an alpha-0 OCR layer over a raster or over outlined text left its
ink readable in the crop and the sentence shipped twice (recovery ON 2, OFF 1).
The mask now covers every char box the shipped prose is built from, visible and
invisible — whatever `get_text` contributes.

**2. Char boxes erased uncovered ink (P1).** A char box spans ascender to
descender, so painting boxes wiped drawn marks just below a baseline: a drawn
"0.058" came back as "U.UD& arawn pelow". The mask now paints what step 1
already classes as COVERED — connected ink components whose pixels lie mostly
under a native char box — so the ink recovery exists to read is never painted.
`native_ink_masked` is that one function; its mutant (back to box painting)
fails the below-the-baseline test.

**3. A masking failure read the unmasked crop (P1).** Silently, at debug level,
reopening the very duplication the mask closes. It now degrades to NO reading:
step 1's picture and mark, plus an `uncovered_text_mask_failed` event.

**4. A fail-fast deadline was swallowed (P1).** `OllamaFigureEngine.ask` caught
`TotalDeadlineExceeded` — raised in under a millisecond by the outstanding-call
guard — and returned `None`, so a wedge in that shape was invisible. It is
re-raised. Nothing was needed on the orchestrator side:
`ollama_utils.TotalDeadlineExceeded` already subclasses `TimeoutError`, so the
existing test lands on it.

**5. The budget refusal was silent (P2).** `PageLadderBudget.call` only
announces when its own clock is spent, so a region refused for want of a WHOLE
call's room emitted nothing while OUTPUT.md promised an event. It now counts as
skipped and emits `uncovered_text_budget_exhausted`.

**6. The reader was probed with coverage off (P2).** `check_text_coverage` is
checked first: with no flagged page there is nothing to recover.

**7. The flag-off identity check is in the suite (P2).** It derives step 1's
expected tail from the un-grouped regions and compares bytes, so the "\n" vs
"\n\n" joiner mutant now dies inside the suite rather than only against the real
pages. `metadata.json` under `--no-recover-uncovered-text` is pinned too.

**8. `_apply_text_coverage` was only half-idempotent (P2).** A second call
duplicated the marker and the figure blocks. It now strips whatever the feature
itself wrote before writing it again, so the result does not depend on how often
the phase ran.

**9. Stale docstrings (P2)** citing the deleted Tesseract witness are rewritten.

## Fix round 5 (Fable reproduced + guided Argon)

**1. The mask painted glyphs WHITE (P0).** On a grey or dark fill -- a chart
exported as a raster with native labels over it -- that handed the model fat
white words, which it read: the mask created the duplication it exists to
prevent. Measured with the reviewer's metric (contrast pixels inside the native
char boxes, against the fill), white painting left **395% on a 0.6 grey and 389%
on a dark fill** -- MORE legible than before masking. Each covered component is
now filled with the MEDIAN of a ring around it, so a gradient is followed rather
than flattened: **0% on all four flat fills** (0.92, 0.85, 0.6, and
dark-with-white-text). On a vertical GRADIENT the same metric, taken against a
row-local background, is also 0% -- but the fill is one colour per component,
so on a steeper ramp than the one measured a faint flat word-shaped patch can
remain. Named as a limit below rather than claimed as solved.
`cv2.inpaint` was tried first and reverted: it reconstructs a ghost from the
anti-aliased boundary and left 119% on a near-white fill.

**2. Silent loss: the equation sidecar on NATIVE_FALLBACK (P1).**
`_native_text_with_appends` accepts an attempt only if it STARTS with the native
snapshot. A recovered block inserted at its reading-order position sits
mid-prose, so the decorated attempt failed that test and the GH-36b
`![equation crop]` sidecar was dropped -- from fragment and final `.md` alike,
with no event. The test now runs with this feature's own paragraphs spliced out,
and whitespace-normalised: inserting a block deliberately promotes the native
text's single newline to a blank line, which an exact prefix test cannot survive.

My round-3 probe missed this because it anchored at the TAIL, which keeps the
prefix. My first round-5 test missed it too, for a second reason worth
recording: it left an UNDECORATED attempt carrying the equation, and the plain
prefix test matched that one instead. A fixture that still passes with the fix
removed is not a test of the fix.

**3. The engine change was unguarded (P2).** Its test stubbed above the engine,
so deleting the `gemini_api` change survived. It now drives the real
`OllamaFigureEngine.ask` with the HTTP layer patched. Collateral, as the
reviewer predicted: `ask` re-raising broke `crop_descriptions`'s documented
never-raise contract, and its caller catches once per DOCUMENT -- one fail-fast
would have dropped every figure description on that document. The describer path
catches and returns `None`; #1050 wraps the engine itself and is unaffected.

**4. The below-baseline test is exact (P2).** 0% loss, not 90%. The halo
dilation that catches a glyph's anti-aliased edge never enters an UNCOVERED ink
component, so a drawn mark within a rule's width of a native glyph survives.

**5. Events and reader calls are idempotent (P2).** Byte idempotency was not
enough: a second apply re-read every crop and appended a second copy of every
event. A region whose picture and block are already in the body is reused; its
PLACEMENT is still re-derived, so the page lands identically.

**6. The flag no longer changes a run's identity when OFF (P2).** Hashing it
unconditionally made every OFF run invalidate main's fragments and force a
reprocess that changes nothing -- the GH-142 failure shape in the other
direction. The test inspects the fingerprint PAYLOAD, because a test on the
digest alone cannot fail: both configs would carry the key and still agree.

### A vacuous test, caught only by its mutant

`test_the_figure_describer_still_never_raises` passed against a mutant that
removed the guard entirely. conftest's autouse
`_figure_description_model_is_absent` replaces `_figure_description_ask` for
EVERY test, so the test measured the stub. It now reinstalls the real method,
captured at import before any fixture runs. Same shape as
`magicmock-makes-negative-assertions-vacuous`: the mutant was the only thing
that could have found it.

## Known limit (accepted, not a blocker)

Charts the whole-figure rule rejects fall back to per-region pictures. Nothing
is lost or unmarked — each region still ships its crop and its words — but the
picture is a sliver rather than the figure. Four shapes: hairline step charts,
tiny scatter markers, line axes without a frame, and a frame drawn as four
separate rule segments (LaTeX `\fbox`, `mdframed`), whose segments enclose
nothing so its contents never group.

Also accepted (round 6): on a steep vertical GRADIENT the ring-median fill is
one colour per component, so a faint flat word-shaped patch can remain where the
background ramps across a glyph. Measured 0% local contrast on the gradient
tested; a per-pixel background estimate would close it and was not attempted.

Also accepted (round 5): drawn capitals that OVERLAP a native descender lose
about 32% of their ink to the mask, because the covered-component rule cannot
separate two touching strokes. A fast `None` mid-document keeps step-1 crops.

## Not measured

- No cluster (vLLM) run: the reader path is the Ollama figure engine.
- Recovery is not applied to the `text_coverage_unknown` branch.
- A figure region's picture is still appended at the page tail, not placed at
  its own reading-order position.
- `ANCHOR_TAIL_WORDS = 8` is the one chosen constant in the diff. It is a match
  window, not a quality cut: too few words and the snippet repeats (the
  insertion is then refused, not guessed); too many and the native extractor's
  rejoining breaks the match. No data was collected on where it should sit.
- The 91 regions are not a corpus rate: they come from 22 pages, 2 of them the
  owner's review pages, and 18 of the 91 come from one document. They were also
  measured BEFORE the whole-figure rule, which collapses many of them; the
  per-region numbers above are therefore a bound on the old shape, not a
  prediction of the new one. No re-run was done.
- A frame drawn as four separate rule segments (LaTeX `\fbox`, `mdframed`)
  encloses nothing, so its contents still do not group. Parked, named here at
  the orchestrator's instruction.
- Item I (the configured vLLM backend and host are ignored) is not fixed; see
  above.
