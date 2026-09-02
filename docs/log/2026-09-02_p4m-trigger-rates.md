# P4-M — trigger rates for the equation lane (no model calls)

2026-09-02. The measurement the P4 ruling required before any widening of the free
lane ships (`docs/log/2026-09-02_p4-structure-lane-design.md`, section 7).
Produced by `src/socr/benchmark/trigger_rates.py` over the Papers library
(367 born-digital PDFs on disk, 23,190 pages) with socr at the tip of
`docs/conceptual-revision-2026-09`. Content-free: counts and basenames only.
The every-sixth-paper subset (60 papers, 3,163 pages) gave the same shares
within two points, so the numbers are not a sampling artefact.

## What the numbers say

- Two thirds of all pages take the free lane today (67.5%); tables already route
  13% of pages to the ladder.
- The equation signal as detected is, in practice, the **math-font term alone**:
  the LaTeX-markup regex fires on 6 pages in 23,190 and the corrupt-math term on
  none of the free-lane pages (corrupt math already removes a page from the lane).
- **Routing on the signal as detected moves 36% of the free lane** — 5,642 pages —
  to a local model call. Nearly a quarter of those (8.0% of the lane) carry ten or
  fewer math-font characters: inline symbols in prose, not equations.
- The hygiene flags (unmapped or unrecovered glyphs — the native layer is actually
  damaged) mark 2.4% of the lane.
- The math-font character count has **no natural break**: the 1-10, 11-50 and
  51-200 buckets are each 8-15% of the lane. Any cut is an invented threshold.

## The table

- pages: 23190; born-digital: 21651; table pages (already routed): 3114
- free-lane pages today: 15644 of 23190 (67.5% of all pages)
- of which chart-asset lane (raster present, still no model): 2215
## Free-lane pages each candidate trigger would move to the ladder
| trigger | pages moved | share of free lane |
|---|---|---|
| has_equations as detected (font OR regex OR corrupt) | 5642 | 5642 (36.1%) |
| font term only | 5640 | 5640 (36.1%) |
| regex term only | 6 | 6 (0.0%) |
| corrupt-math term only | 0 | 0 (0.0%) |
| hygiene flags only (unmapped / unrecovered glyphs) | 383 | 383 (2.4%) |
| regex OR corrupt OR hygiene (no font term) | 389 | 389 (2.5%) |
## Math-font characters per free-lane page (distribution, not a threshold)
| bucket | pages | share of free lane |
|---|---|---|
| 0 | 10080 | 10080 (64.4%) |
| 1-10 | 1253 | 1253 (8.0%) |
| 11-50 | 1272 | 1272 (8.1%) |
| 51-200 | 2405 | 2405 (15.4%) |
| >200 | 634 | 634 (4.1%) |
## Per-document free-lane pages moved by `has_equations` as detected

Per-document rows are in the run output and not reproduced here.

## Ruling — 2026-09-02

Owner chose **the signal as detected** (`has_equations`, in practice the math-font
term), accepting that 36% of the free lane pays a local model call. Reasons: it is
the signal the ruling names and it is already on `PageState`; the panel's Q2 ruling
makes over-routing a cost, not a correctness risk (the model reading is region-scoped
and never replaces native prose); and the character-count distribution has no
break that would justify a threshold. Revisit only with a measured throughput
problem on the local model, and then with a documented cut derived from data.
