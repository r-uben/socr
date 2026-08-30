# GH-372 — booktabs rules misfired the in-table chart detector

Date: 2026-08-31. Branch `fix/372-booktabs-chart-misfire`, base `main@2ed5407`.

## Root cause

On Cochrane–Piazzesi p18, Table 6's own booktabs rules (>1pt thick strokes)
were union-find-clustered into a blob whose area cleared the 14,400pt² chart
threshold, and `_has_filled_rects_or_thick_strokes`'s thick-stroke branch
(`width > 1.0`) fired on them unconditionally. The whole table was routed to
the chart lane: every word excluded from the rowizer, no table candidate, and
a "chart region" crop that was a screenshot of the table. Full trace: the
2026-08-30 p15/p18 live-run diagnosis (page 18 section).

## Fix

`_RULE_THINNESS_RATIO = 0.02`: in the thick-stroke branch only, a stroke whose
bbox thickness/span ratio is ≤ 0.02 is a rule (booktabs \toprule ≈ 1pt over
≥ 200pt gives ≤ 0.005; a stroked line's bbox is flatter still) and no longer
qualifies the cluster. Fills and coloured strokes are untouched, so a real
chart whose axis is a thick rule still qualifies through its other marks.

## Accepted trade-off

A chart whose ONLY mark is a single flat axis-aligned thick stroke — no fill,
no colour, no frame (GH-150 A1's framed-cluster path) — is geometrically
indistinguishable from a table rule and is now missed. Judged rarer than the
booktabs tables this gate protects; recorded in the code comment at the
constant.
