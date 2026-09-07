# 2026-09-07 — TICKET-E1: a page-sized raster with dense native words is the scan, not a chart (#511 large half)

## What changed

`has_chart_marks` (`src/socr/figures/extractor.py`) now refuses the raster fast path when a
page-covering raster carries native words at prose density: it is the scan itself (an OCR text
layer under a page photograph) or a decorative page export (real text under a slide background),
not a chart. New helper `_raster_is_scan_or_decorative(page, rect, page_area)` and two named
constants:

- `SCAN_RASTER_PAGE_COVERAGE_MIN = 0.90` — the raster must cover ~the whole page before the check
  even applies (guards #510's small-raster gate: a chart inset or embedded figure well under
  full-page size is untouched regardless of any text near it).
- `RASTER_TEXT_DENSITY_MIN = 0.75` words/100pt², measured over the raster's own placed area, using
  only words whose center falls inside the raster rect.

`_is_chart_asset_page` (`orchestrator.py`) is unaffected — it only calls `has_chart_marks`.

## The fixture correction (mid-ticket CONSILIUM-GATE)

The dispatch message asked me to preserve `ecb-speeches-2021-speech-p2-4.pdf` (all 3 pages) as
"real chart, must stay chart." I opened the actual rendered output
(`~/Data/socr/census-ecb-2026-09-06/out/ecb-speeches-2021-speech-p2-4/figures/chart_page_2.png`)
and it is a PowerPoint slide export — bullet text plus a SmartArt arrow diagram, no axes, no
plotted data. By #511's own text ("a half-page photograph, a full-bleed decorative background...
still routes into the chart lane") this is exactly the still-open decorative-raster case, not a
chart to protect.

I measured `words_inside_share` (always 1.000 on every fixture, useless as a discriminator),
word-bbox-area-ratio (overlapping bands, 0.10-0.33 Fed vs 0.15-0.17 ECB slide — not separable),
and word density (Fed 2.22-8.03, ECB slide 1.37-3.33 — the ranges nearly touch). I stopped and sent
the team lead the full measurement table rather than pick a value past that fork. Ruling: the
ecb2021 fixture's "must stay chart" label was wrong — move it to the must-NOT-be-chart set; ship
the ticket's literal density rule; do not gate on colorspace even though it happened to separate
every fixture (a genuine grayscale raster chart would misfire on colorspace alone, and no such
fixture exists in the corpus to calibrate a combined rule against).

## Measurements (2026-09-07, real PDFs, `page.get_images()` / `get_image_rects()` / `get_text("words")`)

| Fixture | coverage | colorspace (secondary, not gated on) | n_words in raster | density words/100pt² | verdict before | verdict after |
|---|---|---|---|---|---|---|
| fed-meetings-1969-05-27 p1 | 1.000 | DeviceGray | 151 | 3.08 | chart | **not chart** |
| fed-1989-11-14 p1 | 1.000 | DeviceGray | 161 | 3.28 | chart | **not chart** |
| fed-1989-11-14 p2 | 1.000 | DeviceGray | 268 | 5.46 | chart | **not chart** |
| fed-1989-11-14 p4 | 1.000 | DeviceGray | 394 | 8.03 | chart | **not chart** |
| fed-1989-11-14 p5 | 1.000 | DeviceGray | 109 | 2.22 | chart | **not chart** |
| ecb-speeches-2021-speech-p2-4 p1 (slide export) | 1.000 | DeviceRGB | 85 | 2.91 | chart | **not chart** |
| ecb-speeches-2021-speech-p2-4 p2 (slide export) | 1.000 | DeviceRGB | 97 | 3.33 | chart | **not chart** |
| ecb-speeches-2021-speech-p2-4 p3 (slide export) | 1.000 | DeviceRGB | 40 | 1.37 (floor anchor) | chart | **not chart** |
| ecb-speeches-2025-speech-p21-23 p2 (real data chart, vector path) | n/a — `images=0`, vector cluster | n/a | n/a | n/a | chart | **chart (unchanged)** |
| synthetic raster chart w/ axis labels (only raster-chart anchor in the corpus) | 1.000 | RGB (ICC sRGB) | 10 | 0.21 (ceiling anchor) | chart | **chart (unchanged)** |

`RASTER_TEXT_DENSITY_MIN = 0.75` sits strictly between the ceiling anchor (0.21, synthetic chart)
and the floor anchor (1.37, ecb2021 p3), with margin on both sides. Verified directly against the
real corpus PDFs via `has_chart_marks(page)` — all 5 Fed pages and all 3 ecb2021 pages now False;
the ecb2025 vector chart and the synthetic raster chart both stay True.

No genuine raster chart with an OCR/native text layer exists anywhere in the corpus fixtures I had
access to (the confirmed real chart, ecb2025 p2, goes through the vector-cluster path — `images=0`
— so it never touches this new code at all). The synthetic fixture is therefore the **only**
raster-chart anchor calibrating the density floor; this is recorded plainly per the team lead's
instruction, and it is why colorspace was rejected as a gate — there is no fixture to confirm a
grayscale raster chart wouldn't misfire on it.

## Tests

`tests/figures/test_has_chart_marks_scan_raster.py` (new, 7 tests, real PyMuPDF page objects, no
MagicMock):
- dense full-page grayscale scan (Fed-shaped) → not chart
- sparse RGB decorative slide at the exact measured floor (1.37) → not chart
- synthetic raster chart at the exact measured ceiling (0.21) → chart
- page-covering raster with zero native words → chart (unaffected; density 0)
- density pinned exactly on both sides of `RASTER_TEXT_DENSITY_MIN` → difference asserted directly
- a small (sub-`SCAN_RASTER_PAGE_COVERAGE_MIN`) raster packed with dense text → still chart (#510's
  gate untouched — the new check only applies once the raster covers the page)
- both new constants are named, and the density floor is asserted to sit strictly between the two
  measured anchors

`has_chart_marks` docstring updated to record the new behaviour and to narrow the "what this gate
still does NOT do" note to image-only pages carrying no native text at all (the true GH-511
residue).

## Verification

- `PYTHONPATH=.../socr-e1/src ~/venvs/socr/bin/pytest tests/figures/ tests/test_chart_lane.py -q`
  → 40 passed.
- Direct check against the 8 real corpus/synthetic fixtures via `has_chart_marks()` → all 8 match
  the expected before/after verdicts above.
- Full suite: `PYTHONPATH=.../socr-e1/src ~/venvs/socr/bin/pytest tests/ -q` → **4321 passed, 4
  xfailed**, 0 failed (208s).
- `uvx ruff@0.16.0 format --check .` → 2 files needed formatting (the new test file and the edited
  `extractor.py`), reformatted, then clean (598 files).

## Not done / follow-up

- E2 (`table_not_scorable` scope) depends on this ticket and is next.
- The true GH-511 residue (an image-only raster — no native text at all — that is a photograph or
  decorative background rather than a chart) remains open; nothing here changes that case, and the
  docstring says so explicitly.
- No genuine grayscale raster-chart-with-text fixture exists to calibrate a combined
  colorspace+density rule; if one turns up in a future census, worth revisiting whether colorspace
  can safely tighten the gate further.
