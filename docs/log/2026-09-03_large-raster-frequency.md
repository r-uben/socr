# GH-511 — how often does a large raster route a born-digital page to the chart lane?

**Date:** 2026-09-03
**Script:** `docs/log/2026-09-03_large-raster-frequency.py`
**Corpus:** 277 PDFs, the local papers library. 0 unreadable files; **64 pages excluded
as unreadable**, and **207 born-digital pages carried a raster whose placement could not
be measured**.

Both exclusions are reported by the script and are stated here because they bound the
number below. The 64 are outside the denominator entirely. The 207 are inside it but
under-counted in the numerator: production `has_chart_marks` fails *open* on an
unmeasurable placement, so up to 207 more pages could reach the lane than the 1530 below.
That moves the figure the same direction as the conclusion, so it does not weaken it.

(An earlier draft of this log said "1 unreadable page", inferred from the single MuPDF
error visible in the console rather than counted. The script did not report the tally at
all until cubic asked for it on #557; it is 64.)

#511 set its own acceptance: *"how often does a large non-chart raster actually appear on
a born-digital page in the corpus? If the answer is 'rarely', the current fail-toward-chart
behaviour is the right trade and this closes as WONTFIX with that number recorded."*

The answer is not "rarely". **#511 does not close.**

## The population

| | pages |
| --- | ---: |
| born-digital | 14356 |
| scanned | 894 |
| born-digital carrying any raster | 1848 |
| …raster below `CHART_MIN_CLUSTER_AREA` (spared by #510) | 318 |
| …raster at or above it (**the #511 population**) | **1530** |

**1530 / 14356 = 10.66% of born-digital pages, spread across 141 of the 277 papers.**

The distribution is heavily skewed: one scanned-with-OCR-layer book
(`1982__bertsekas__opt.pdf`) supplies 385 of the 1530, a quarter of the whole population,
and the top ten papers supply 825. But 141 papers is half the corpus — this is not one
pathological file.

## What those pages actually are

The population is a bound, not a classification: it counts pages where the chart-vs-photo
decision *arises*, and says nothing about which way it should go. So the sweep also draws a
sample, ordered by `sha256(paper|page)` so the draw is fixed by the corpus rather than
chosen by whoever ran it (`--sample N --out DIR`). Ten pages, classified by eye:

- **6 genuine figures** — scatter plots, line charts, bar charts, a timeline diagram. The
  chart lane is right about these.
- **4 full-page rasters of a page that carries a real text layer** — scanned book and
  journal pages (JSTOR-style) whose OCR layer makes the detector call them born-digital
  while the page image is the whole page. **Two of these four carry no chart at all**: one
  is a page of bibliography entries, one a page of unbroken prose.

So roughly 4 in 10 of the population are false positives, and the misclassification is not
the "half-page photograph" #511 imagined. It is a scan with an OCR text layer.

## A correction to #511's premise

#511 says the false-positive page "ships its prose as an untranscribed image". It does not.
`_agentic_chart_asset_page` retains `ps.native_text` and *appends* the PNG ref; the prose
stays in the markdown. The real exposures on a false positive are narrower, and both should
be stated as risks rather than as measured harm — neither was measured here:

1. **The table lane is skipped.** `bo.engine != "chart_asset"` gates table verification in
   `_phase_agentic`, so a scanned page carrying a real table is never verified.
2. **`fence_chart_axis_residue` runs.** #369 narrowed it to a *run* of bare-numeric lines,
   so a genuine numeric column on such a page could still be fenced as axis residue.
3. A spurious whole-page PNG per page — 385 of them for the Bertsekas book alone.

## What the sample suggests, and what it does not license

#511 declares geometry exhausted, on the grounds that a page-sized photo and a page-sized
chart share every measurable box property. The sample points at a discriminator it did not
consider: all four false positives are rasters covering **essentially the whole page** on a
page that *also* carries paragraph-shaped text. A genuine figure is placed *within* a page;
a full-bleed raster over prose is a scan.

That is a two-part predicate — near-full-page raster **and** prose-shaped (not label-shaped)
text — and the second half needs its own corpus measurement before it can be a rule, since
the repo forbids a guessed threshold. **Nothing was implemented here.** A full-page chart
with a caption is a real page type and must not be broken to fix this.

## A measurement that was wrong, recorded

Two, in fact.

- **The char-count proxy.** The first attempt used a text-length threshold instead of the
  real detector. It reported 99.75%, because a scanned book with an OCR layer passes a
  char-count test and those pages are full-page rasters. I had reasoned the proxy would
  over-count the *denominator* and so give a lower bound; it inflates the numerator too,
  by exactly the pages that carry a big image. The wrong version looked entirely
  reasonable.
- **"1 unreadable page".** Stated in the first draft of this log from the one MuPDF error
  visible in the console, not from a count -- because the script tallied the exclusions and
  never printed them. It is 64. A number that is not printed is a number nobody checks.
- **`reaches_the_lane`.** The sweep originally asked "would vector evidence route this page
  anyway?" via `has_chart_marks`. It cannot: `has_chart_marks` takes the raster fast path,
  so a page that just cleared the area gate answers True *because of that raster*. The two
  counts are equal by construction, not by measurement. The counter is kept, renamed
  `predicate_agrees`, as the gate-vs-predicate agreement check it actually is.

## Reproduce

```
PYTHONPATH=src ~/venvs/socr/bin/python docs/log/2026-09-03_large-raster-frequency.py \
    <papers dir> [--sample 10 --out <dir>]
```

Output is content-free — counts and basenames only.
