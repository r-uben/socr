# #635 — staged chart-count reader

The current artifact contains five empty table skeletons and five preserved crops, with unresolved placement. Preserve the figures procedure: provisional classification, no asset discarded on uncertainty, image plus source caption and descriptive alt text; numeric readings require explicit series/unit binding and provenance.

**Constraint correction:** the issue's worked June counts sum to 19, not 16. Visual inspection of the supplied 2018 panel suggests that example misreads the dashed heights; it must not become the test oracle. Also, the 2021 panel has no visible June series, and the longer-run bars appear to total 15 for September and 14 for June. Those latter counts need independently checked annotation before becoming goldens. Neither “two series in every panel” nor unconditional equal totals is a safe assumption.

## Stage 0 — suppress empty derivations, preserve evidence

Add a structural skeleton check: after parsing headers/separators and identifying label versus data positions from the chart-table binding, every data cell is empty. Headers containing numeric bin limits are not observations. Literal zero is data; text-valued cells and unresolved-value tokens are not empty.

For a table proven to represent a chart region, remove the empty skeleton from published Markdown. Retain its original bytes/hash in provenance and record `chart_table_skeleton_suppressed`; publish the crop, caption, class-specific alt text and a concise “counts not extracted” status instead. Do not match words such as “Participants,” assume five tables imply five crops, or remove unrelated empty forms. Bind panels using native heading/axis geometry and unique source anchors. If binding remains ambiguous, quarantine the proposed skeleton as unresolved rather than presenting it as chart data; preserve the crops in the existing labelled unresolved-placement block.

Keep this mutation outside `reconcile_chart_region_refs`, whose current contract preserves model-authored text. Apply it once to candidate text before judging/identity creation, then let #189 reconcile references normally. Final assembly and fragments must use those same bytes.

## Stage 1 — geometric reader

Initially support discrete vertical bars/histograms only. Keep the original raster immutable. The first saved crop clips part of the two-line x labels: use native page word boxes and the PDF-to-crop transform, or generate a separate expanded evidence crop bounded by the required axes/legend. Do not invent missing upper bin endpoints.

1. Recover panel/horizon, legend entries and axis labels with their positions. Map solid fill to September and dashed outline to June using the legend, not colour names. Legend inheritance between panels requires explicit figure-level binding. Record series as present, absent or unresolved; never invent June values for 2021.
2. Fit the y calibration from at least two labelled tick positions and validate against the remaining ticks and baseline. Tick positions supply the scale: two participants per tick interval here, so one participant occupies half that pixel spacing. Do not mistake axis labels for bar counts.
3. Associate each bar with an x-bin using its horizontal footprint and label centres/intervals. Preserve the printed percent ranges as labels; record percent as the bin unit and participants as the count unit. Separate the solid silhouette from the dashed staircase using legend-derived fill/stroke and dash geometry.
4. Propagate measured stroke/edge and tick-localisation uncertainty into a count interval. Emit an integer only when exactly one nonnegative integer is supported within that uncertainty. Odd counts legitimately fall between the printed two-unit ticks.

Per cell persist source checksum/page, crop ID and SHA-256, crop transform/DPI, panel and series key, bin label/coordinates, bar pixel bbox and detected top/baseline, calibration tick pair with values/positions, uncertainty interval, reader version, status and constraint results. A zero needs evidence of an observable empty bin for a present series, not merely failure to detect a bar.

Emit `UNRESOLVED` for inseparable overlaps, occluded tops/legend, uncertain bin ownership, inconsistent calibration or no uniquely supported integer. Never allocate residual participants to make a sum work.

Refuse fixed pixel, colour-distance or model-confidence cutoffs and unconditional rounding. Accept calibration-derived bounds: half a count is Δy/(2Δn) pixels; measured uncertainty must be smaller and support one integer uniquely.

Acceptance combines independent geometric/integer checks with the issue's caller hook (`accept/reject/no_opinion`). Supply totals per series and horizon—September 2018's 16 comes from the caller, never code. Require same-series cross-checks and complete observed-bin accounting. Equal series totals are an additional check only when source/caller evidence establishes equal populations; it is not universally independent domain knowledge. A failed constraint rejects verification, not the image. Without an applicable hook, label the derivation unverified.

## Stage 2 — model-assisted proposals

Reuse the configured local Qwen VLM to propose legend, bin and bar-top annotations alongside geometry; optionally obtain an independent Gemini proposal through the existing provider configuration. Reconcile by cell identity. Agreement must satisfy calibration and constraints; disagreement stays unresolved. Never publish a model number contradicted by geometry, average competing counts, or use totals to force agreement.

## Implementation and tests

Extend `ChartRegionAsset` and `_render_chart_region_crops`; add `figures/chart_data.py` for skeleton detection, calibrated readings and validation. Integrate via orchestrator candidate ingestion and `_preserve_chart_regions`; reuse `FigureExtractor.extract_page`, `chart_region_anchors` and `reconcile_chart_region_refs`. Add caller-hook configuration and persist derivations in manifest/sidecars.

Pin five-panel asset retention, empty versus zero/text cells, ambiguous binding, repeated assembly/resume, scale/DPI changes, odd heights, overlap, missing legends, absent June, inconsistent totals and model/geometry disagreement. Golden counts require independent human annotation, not the issue's example.

**Owner decision:** may expected totals be keyed by survey and horizon, with absent series represented explicitly? I recommend yes; a universal equal-total/16 rule would reject valid panels.
