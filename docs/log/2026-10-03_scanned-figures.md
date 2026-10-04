# 2026-10-03 - scanned figure pages ship the page image (#1030)

Pages and counts only; the corpus is copyrighted. Scripts, contact sheets and the full tables live
outside the repo in `~/.local/state/socr-housekeeping/figures-scan/` (`MEASURE.md`).

## Problem

A chart on a SCANNED page ships as whatever the invisible OCR layer or a model made of it. A scan
has no vector marks, so `has_chart_marks` / `_is_chart_asset_page` never claim it, and
`_describe_and_embed_figures` skips every page in `assessment.scanned_pages()` ("a scan has no
localizable figures"). #1027/#1028 cover only the case where every model read was rejected.

## Phase 1: measurement

Sample: the three raster documents among the finished archive re-OCR outputs (Forsythe-Lundholm
1990, Hansen 1995, Gleason-Lee 2003; 96 of 98 pages are a full-page image). The other 12 listed
documents are born digital. Ground truth: every page looked at on contact sheets: 19 figure pages
(Forsythe 12, Hansen 5, Gleason 2), 29 table pages, 50 other. Most figure pages also carry prose
and Hansen 13 also carries a table.

How the figure pages shipped (current run, Forsythe + Hansen, 17 pages; Gleason unfinished):

| class | pages |
|---|---|
| (a) junk (one letter per line) + page image | 1 |
| (b) fail-closed marker + page image | 6 |
| (c) readable model text + page image | 7 |
| (d) chart omitted, no image, no marker | 3 |

Older code (pre-aeb89727, 19 pages) shipped the invisible layer verbatim on 16 of 19. The one
current junk page is the shape #1028 now sends to the floor when a model attempt ran.

Detectors on the 98 pages (layer = pymupdf text of the invisible layer):

| detector | precision | recall | false fires |
|---|---|---|---|
| pymupdf image block | 0.19 | 1.00 | 79 |
| marker/surya layout | not run by this ladder; nothing recorded | | |
| run of one-char lines >= 4 / >= 7 | 0.64 / 0.60 | 0.37 / 0.16 | 4 / 2 table pages |
| tick-number lines >= 5 | 0.43 | 0.84 | 16 table, 5 prose |
| **figure caption line in the layer** | **1.00 (17/17)** | **0.89 (17/19)** | **0 table, 0 prose** |

The stop criterion ("zero prose or table pages lost in the sample") is met by the caption signal
alone; bursts and ticks are not precise because tables set vertical column headers. The shipped
function `has_figure_caption` was re-run over the 98 layers and reproduces the measurement
(TP 17, FP 0, FN 2).

## Decision (per page, not per region)

Nothing on a scan isolates the figure box (page-sized image block, no layout boxes recorded, no
vector marks). So the unit is the page:

- Fires when `invisible_text_over_raster` AND the layer has a caption line AND the page does not
  already ship an image of itself (D3 / rotated-shred / invisible-scan floors).
- Ships: the page text exactly as selected + the page image (`scanned_figure_page_p<N>.png`,
  forced like chart PNGs, with or without `--save-figures`). The caption stays text because
  nothing is removed. The doc-level figure extractor skips these pages so the raster is not shipped
  twice (under `--save-figures` a scan not classified scanned used to get an unnamed full-page
  "Figure N" block; this replaces it).
- Junk: only when the text that ships IS the layer (engine `native*`: no provider, `--native-only`,
  outage) are runs of >= `MIN_SPELLED_RUN` (7) one-character lines moved verbatim into a
  `socr:spelled-axis-residue` HTML comment, as GH-369 does for axis tick numbers (separated, never
  dropped). A model reading is never rewritten. 7 = smallest run on a page whose vertical text I
  confirmed as an axis title (Forsythe 16); longest run on any page with both a caption and a
  table = 4 (Hansen 13). On the real shipped text it fences 13 lines on each of the two junk pages
  and 0 on every caption page that carries a table.
- Where: `manifest._apply_scanned_figure_guard`, last of the finalize guards, so the flush, the
  stitch and a resume replay see one text (idempotent on its own ref). The page loop step only
  renders the PNG and records the event. Abstains on markers and floor pages (rewriting their bytes
  could change what the disposition classifier reads).
- Surfacing: audit event `scanned_figure_asset` (replayed on resume), sidecar
  `scanned_figure_png_ref`, document note in `metadata.error` (same channel as the chart-asset
  debt), CLI line. A render failure is status-only: SUCCESS -> WARNING, event `png_saved: false`,
  document note says the figure is preserved nowhere, red CLI line.

## What this does NOT fix (limits)

- Recall 0.89: a caption garbled in the layer (Forsythe 12, 16) is missed; those pages keep today's
  behaviour. A scan with NO text layer has no caption to read.
- Whole page, not a crop: the image repeats the prose around the figure. Cropping needs a region
  signal that does not exist here.
- Class (d) pages whose caption is garbled still lose the chart silently.
- The page image is not a transcription: the data inside the plot is unread (the existing
  chart-asset lane makes the same trade).
- Not run end to end on the corpus (a pinned re-OCR was running; I did not touch it). Evidence
  for the build is the 98-page layer measurement + the real-text fence counts above + hermetic
  e2e tests.
- Small sample: 3 documents, 2 journal layouts, one labeller.

## Tests

`tests/test_gh1030_scanned_figure_pages.py` (26): detector examples and non-examples; fence keeps
every character and leaves short runs alone; guard difference flag on/off, idempotent, abstains on
floors/markers/empty, fences only when the layer ships, render failure is status-only; e2e through
`process()` for (provider, no-provider) x (`--save-figures`, not): the SAME scan with and without a
caption (image present only with it), and the same caption scan with the step switched off (text
identical minus the ref; no duplicate raster), event / sidecar / metadata note, spelled-axis fence
in the no-provider run, second run (resume) does not duplicate the image. Hermetic:
`_available_engines_for_agentic` patched (both provider states), `_resolve_judge_model` -> "".
Every pin is a difference, not a value measured locally.

Mutants (external copy of `src` + `tests` + `pyproject.toml`; baseline 26 pass inside the copy
including `test_loaded_source_is_this_checkout`, which asserts `socr.__file__` is in the copy;
anchor count 1 asserted before each edit):

| mutant | result |
|---|---|
| guard call removed from the finalize chain | 6 fail |
| detector always true | 9 fail |
| page-loop hook not called | 6 fail |
| fence call removed | 2 fail |
| figure-phase de-dup removed | 2 fail |
| `MIN_SPELLED_RUN` = 1 | 1 fail |

Full suite and `uvx ruff@0.16.0 format --check .` clean before commit (see commit).

Frozen pins updated for the new member / sidecar key: `test_resume_restore_kinds` (45 -> 46,
`scanned_figure_asset`), `test_p6_disposition_persistence` (sidecar key set),
`p6_stage_c_oracle.VOLATILE_KEYS` (+`scanned_figure_png_ref`, empty on every page without a scanned
figure). Full suite: 6879 passed, 2 skipped, 4 xfailed.

## Round 2 (review of 7f4ad170)

- Fence is IN PLACE: each run of >= 7 one-character lines keeps its position, every line and every
  blank line; three wrapper lines are added (open, note, close), so deleting them returns the input
  byte for byte. Separate runs are not merged. This supersedes "moved to the end of the page" above.
- Fence abstains inside or beside a markdown table, inside `$...$` / `$$` math, beside a list item,
  and on a run of bare bullet markers (a vertical table header, an equation and a list of single
  characters all look like an axis title to a line counter). Conservative by construction: a stray
  `$` earlier on the page also makes later runs abstain.
- Caption needs caption-shaped evidence: the line is at most `MAX_CAPTION_LINE` = 50 characters
  (longest caption line in the sample; the 27 caption lines run 8 to 50, wrapped prose 79 to 87) OR the next
  non-blank line is figure furniture (a lone character or a bare number); and a label with nothing
  after it is rejected when the next line continues in lowercase ("Figure 3" / "shows ...").
  Re-scored with the shipped function on the same 98 pages: precision 17/17, recall 17/19, 0 table and
  0 prose false fires, i.e. unchanged. The sample contains no `Figure 3. We ...` style over-fire, so the new rules
  cost no recall HERE; they are pinned by synthetic cases, not by corpus evidence. Residual: a SHORT
  prose line that opens with a label and a terminator still passes.
- Reporting reads the FINALISED output: the event now says only "rendered"; document note, CLI and
  metadata classify each page as shipped (ref in the finalised text), suppressed (rendered, not
  referenced; yellow CLI line) or lost (render failed).
- Guard keeps the rendered image on a bare marker and on an empty page (a captioned scan never ends
  with neither text nor image; marker plus one image block still classifies as a marker). It still
  abstains when a floor image or a marker-with-image is present.
- Resume test now proves: engine not called on the second run (page-level skip, document ledger
  dropped to force it), resumed `.md` and page fragment byte identical, event replayed into the
  new `audit_log.json`, note in `metadata.error`, one image ref.
- Mutants (external copy, canary + anchor count 1, 43 pass at baseline): blank lines dropped 1 fail;
  table / math / list abstention removed 1 / 3 / 1 fail; caption length gate removed 2 fail;
  continuation gate removed 2 fail; guard abstains on bare marker again 1 fail.

## Round 3 (review of 52426fa5)

- The fence is VISIBLE: the one-line note `[unreadable figure text from scan, kept verbatim]`, then the
  run in a ```text block, in place. An HTML comment hides the text in every rendered view, which is
  silent loss for a reader. The wrapper-removal pin deletes the note and the opening and closing
  fence lines and still gets the input back byte for byte; a test asserts no `<!--` appears.
  This supersedes the HTML-comment wording of rounds 1 and 2.
- Math abstention also covers `\[ \]`, `\( \)` and `\begin{..}..\end{..}` (depth tracked across lines).
- List abstention uses the repo's `born_digital.LIST_MARKER_GLYPHS` plus dashes (en, em, minus,
  hyphen bullet); runs of bare marker glyphs and neighbouring `<glyph> item` lines abstain.
- Caption: text after `Figure N.` that reads as a sentence (opens with a subject pronoun, contains a
  results-sentence verb, or is longer than `MAX_CAPTION_WORDS` = 7, the longest caption remainder in the
  sample, and ends in a period) fires only when the next line is figure furniture. `Figure 3. We find no
  effect.` no longer fires alone. Re-scored with the shipped function on the 98 pages: precision 17/17,
  recall 17/19, 0 table / 0 prose false fires (unchanged: none of the 27 sample captions is a
  sentence by these rules). The verb and pronoun lists are heuristic and pinned by synthetic cases only.
- Mutants (baseline 55): fence back to an HTML comment 1 fail; sentence gate removed 3; LaTeX depth
  not tracked 3; Unicode bullets dropped 3. (A first LaTeX mutant that removed only the per-line
  "latex" flag survived: the depth already covers the run lines, so it was equivalent.)

## Round 4 (review of bcc69c89): the sentence gate is removed

Round 3's gate (subject-pronoun / verb / word-count lists) is deleted with its tests and mutant.
Review showed it both misses ("Figure 3. Prices rise." then a new paragraph; wrapped sentences
without a period) and misfires ("Figure 2. Annual reports." read "reports" as a verb). No word list
separates a short sentence from a short title.

Decision: the detector is the round-2 rule (label at the start of a line + terminator, then the
50-character line or figure-furniture-after rule and the lowercase-continuation rule). Re-scored with
the shipped function on the 98 pages: precision 17/17, recall 17/19, 0 table / 0 prose false fires.

Why precision beyond the measured 17/17 is not worth more heuristics: a false caption fire is
harm-bounded. It only ADDS one page-image reference beside unchanged text. The fence is an independent
gate (runs of 7 or more one-character lines, outside tables, math and lists), so a prose page that
opens a line with "Figure 3. We find ..." and has no such run is byte-identical apart from one image
link. Pinned by `test_a_short_prose_line_opening_with_a_label_fires_but_changes_nothing_but_an_image_link`.
Residual, accepted: short prose lines that open with a label can add that image link.

Also in round 4 (cubic P2): the fence no longer corrupts pages that hold fenced code. Runs inside an
existing ``` / ~~~ block abstain, and the delimiter is one backtick longer than the longest backtick
run on the page (3 when there is none), per CommonMark. Mutants: existing-fence check removed 1 fail;
fixed 3-backtick delimiter 1 fail. The other cubic P2 (`png_saved` recorded before the guard) is not
true at this head: `png_saved` now means "rendered" and the note / CLI / metadata read the finalised text.

Round 5 (Astra on dea003ae): an unclosed code fence runs to the end of the document, so the appended
image ref rendered as literal code while the report said it shipped. The guard now closes an open
``` / ~~~ block (matching delimiter, own line) before appending the ref, using the same tracker the run
fencing uses (`close_open_code_fence`). Pinned for ```, ~~~ and a 4-backtick fence. Mutant (close
skipped): 3 fail.
