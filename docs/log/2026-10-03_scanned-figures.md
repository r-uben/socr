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
