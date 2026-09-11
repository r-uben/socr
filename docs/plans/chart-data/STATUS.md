# chart-data (#635) — status

Design: [DESIGN.md](./DESIGN.md) (Astra, 2026-09-11). Three stages; only Stage 0 is built.

## Stage 0 — suppress empty derivations, preserve evidence — **DONE**

Branch `feat/635-stage0-chart-skeleton`. New module `src/socr/figures/chart_data.py`;
hook `UnifiedPipeline._suppress_chart_table_skeletons`, called from `route_page`'s
`on_candidate` boundary (and from the escalation and crop-repair replacements, with a
backstop crossing in `_phase_agentic` for producers that reach neither); tests in
`tests/test_gh635_chart_table_skeletons.py`.

Done-when, and how each is met:

| Done-when | Met by |
| --- | --- |
| The check is STRUCTURAL, never lexical | `find_empty_skeletons`: header row is a column key, column 0 of a body row is the row label (the convention `binding.parse_grid` / `label_canonical` already hold), every remaining cell carries no non-whitespace character. No word is matched. |
| A literal `0`, text, and unresolved tokens are data | The same single rule — only a cell that says nothing at all is empty. Pinned three ways. |
| Binding uses native geometry and unique source anchors | `region_interior_rows` (the complement of `chart_region_anchors`) plus two independent proofs, over words this region alone owns — `chart_region_bboxes` expands clusters independently and promises no non-overlap, so a word inside another region's box, or in a band two boxes share, is evidence for neither —: a label drawn inside exactly one region matching exactly one candidate line, AND every data column key drawn IN FULL along that region's axis — the keys' first atoms consecutive on one word-row in the header's order, each further atom of a range key in the row below and in the same column, which is how a two-line tick label is printed (`region_axis_rows` / `_axis_attested`). Sharing a token is not attestation. |
| Ambiguity quarantines, never deletes | Any failed proof, an intervening panel label, or a candidate panel order running backwards against the source leaves the text byte-identical and records `chart_table_skeleton_unbound`. "Quarantine" here means exactly that: the unbound grid stays in the published text and the event says it was looked at and kept. Nothing is moved to a separate artifact. |
| Provenance retains the original bytes and hash | `chart_table_skeleton_suppressed` carries page, table id, region index, crop filename, SHA-256 and the original grid text. Two candidates that bind a byte-identical grid to DIFFERENT panels keep one record each — the bytes and an ordinal do not establish that two derivations are about the same chart — while the page's withheld COUNT is taken over distinct grids, so candidate history and the withheld tally cannot inflate one another. |
| Mutation applied at candidate INGESTION, before judgment | `route_page`'s `on_candidate` boundary, right after `canonicalize_candidate(output)` and before `judge.assess` — so the page judge, table scoring, witness/identity creation, selection and the flush all see the withheld text, and no verdict is passed on bytes that later change. Three producers cross it: every provider rung, the table-escalation candidate and the crop-repaired text. Verified NOT to cross it: the trusted-native lane, the page-level chart and equation lanes, #649 recovery, the manifest fallbacks and #713's restored outputs — those reach only the `_phase_agentic` backstop crossing, and the native fallbacks among them can carry tables. Idempotent, and the records are deduplicated by the grid AND its binding, `(kind, table, sha256, region)`. Not in `reconcile_chart_region_refs`. |
| Surfaced at every level | Page sidecar note (`winning_output.audit_notes` + `audit_events`), document audit events, CLI line "N chart-table skeleton(s) suppressed; crops kept". |
| Byte-identity and resume hold | `_rewrite_all_fragments` still the sole authoritative writer; the pass is idempotent, so re-assembly and a second run reproduce the same bytes. Both event kinds are in `resume_restore_kinds`, so a restored page keeps the withheld grid's bytes and hash (which live nowhere else) and reports the same count. Pinned on events and provenance, not only on body bytes. |

What the dotplot page ships now: five panel headings, five "counts not extracted" notes in
place of the five empty grids, the page's own prose, and the five crops (still in #189's
labelled unresolved-placement block — see residuals).

## Residuals carried out of Stage 0

1. **Crop placement is unchanged.** The five crops still land in #189's unresolved block,
   because the native anchors around each panel are identical on this page and
   `_anchor_slot` correctly refuses them. Stage 0's binding is strictly stronger and could
   be handed to `reconcile_chart_region_refs` as a third binding source; that is a change
   to #189's contract and was deliberately left out of this ticket.
2. **Alt text is chart-specific, not chart-CLASS-specific.** There is no chart classifier
   in the tree. The note names the panel label, the label-column key and the bin range, all
   read off the source. A real class ("vertical bar histogram") arrives with Stage 1.
3. **Producers that reach only the backstop.** Three cross the ingestion seam: every
   provider rung, the table-escalation candidate, the crop-repair patch. The trusted-native
   lane, the page-level chart and equation lanes, #649 recovery, the manifest fallbacks and
   #713's restored outputs do not; they are caught by the `_phase_agentic` backstop
   crossing instead, which runs after their own judgment rather than before it. The native
   and manifest fallbacks can carry tables, so this is the gap that matters. Beyond the
   backstop, text that becomes the page body at assembly — the fallback selected there and
   `_phase_assemble`'s own rewriters — is not covered at all. None of these authors a
   chart-derived grid today, so this is a gap in coverage, not a known loss.
4. **Refusal events on ordinary chart pages.** A chart page carrying an unrelated empty
   form now emits one `chart_table_skeleton_unbound` event per run. Intended (the decision
   is visible), but it is new audit-log volume.

## Stage 1 — geometric reader — **DONE (vector charts only)**

Branch `feat/635-stage1-bar-reader`. New module `src/socr/figures/chart_reader.py`;
`suppress_chart_table_skeletons` gained a `derivations` argument so the block that ships
where the empty grid stood is the derived table instead of Stage 0's note; the reading
itself is driven by `UnifiedPipeline._derive_chart_counts`, called from the same
candidate-ingestion seam Stage 0 already crosses. Tests in
`tests/test_gh635_chart_reader.py`. Counts for the corpus page are in
[GOLDENS.md](./GOLDENS.md).

Done-when, and how each is met:

| Done-when | Met by |
| --- | --- |
| Panel, legend and axes from NATIVE geometry, not OCR of the crop | `read_chart_page` reads `page.get_drawings()` and `page.get_text("words")` only. The crop is never decoded. |
| Legend maps swatch geometry to a series name, never a colour | `read_legend`: a swatch is a mark that does not rest on the axis, is narrower than a bin AND covers no bin's printed label centre, and has a word-row beside it. Style is `solid_fill` (a filled rect) or `dashed_stroke` (any non-zero dash array). No colour channel is read anywhere in the module; the fixture's legend dash pattern differs from its staircase's, so pattern equality is deliberately not used either. |
| Legend inheritance needs explicit figure-level binding | Panels are grouped by (printed bin-label sequence, fitted tick-value set) — the page's own evidence that they are panels of one chart. A legend found in one member binds the group, and the source region is recorded per panel (`legend_from_region`). A group whose members carry two different legends binds nothing. The axis titles are resolved the same way. |
| Series presence is present / absent / unresolved, never forced to zero | A series with no marks in the panel is `unresolved`, with cells `()`. The 2021 panel publishes no June row at all. `absent` is reserved for evidence this fixture does not carry (see residual 2). |
| Calibration from ≥2 labelled ticks, validated against the rest and the baseline | `calibrate_y`: least squares over every stroked tick paired with the numeric word beside it (outside the plot's x span, within half the smallest tick gap). `residual` is the worst disagreement over all ticks AND the axis line, which the fit must place at zero. On the corpus page: 9 ticks, residual 0.0064 pt against a half-count of 1.780 pt. |
| One participant = half the 2-unit tick spacing, derived | `YCalibration.half_count_points` is `points_per_unit / 2`, fitted. No spacing constant exists in the module. |
| Bars from the PDF vector drawings, not raster pixels | Filled rects resting on the axis (the solid series) and dashed runs/risers (the staircase). **A raster fallback is explicitly out of scope for this round** — a page with no drawing operators is refused with that reason in the event. |
| Each bar assigned to a bin by its horizontal footprint | `_owned_bins`: the bins whose printed label centre lies inside the mark. Not an interval-containment test — a text bbox carries side bearings, so the printed label's centre sits ~0.7 pt off the drawn bin centre on this fixture, and an interval derived from it cuts a bar's own edge. A mark owning NO bin makes every bin it touches `UNRESOLVED`, and so does a BAR owning more than one: a histogram bar spans one bin by construction, so covering two printed label centres is the same evidence failure as covering none (round 2 finding — before it, such a bar was published at full height in both bins). The staircase reader keeps the multi-bin case, where a level legitimately runs across several bins. |
| Solid silhouette separated from the dashed staircase by fill vs stroke+dash | Two different readers, selected by the legend-bound style. The staircase is additionally reconstructed as a level function and CHECKED: every riser must join the level on its left to the level on its right, probed half a bin out. One riser that does not sends the whole series `UNRESOLVED`. |
| Count interval from measured uncertainty; one integer or UNRESOLVED | `_resolve`: the height widened by (calibration residual + half the mark's own stroke width) on each side, divided by the fitted points per participant. An integer only when exactly one non-negative integer lies inside, and only when the uncertainty is below half a count. |
| A zero needs an observable empty bin | A zero is emitted only when the series is PRESENT, the bin lies wholly inside the drawn plot, and nothing of that series is over it — for bars, a histogram resting on the axis, where a zero bar is invisible by construction; for the staircase, an outline the page draws DESCENDING to the axis where it meets the bin from a neighbouring level. Where that neighbouring level simply stops instead, the bin is `UNRESOLVED`: an outline that ended and an outline that fell to zero are the same picture (round 2 finding — before it, a gap read as zero and the cell claimed a riser check that had iterated an empty list). `Cell.empty_bin_observed` records it. |
| Never allocate residual participants to make a sum work | No code path reads a total. The acceptance hook is the only thing that ever sees one, it is the CALLER's, and its verdict can only accept, reject or abstain — it can never change a cell. Pinned by running one reading through three verdicts and asserting the counts are identical. |
| Per-cell provenance persisted | `PanelReading.to_dict` on the `chart_counts_derived` event: source checksum, page, crop filename + sha256 + DPI + clip, panel label, series key and style, every bin's label and coordinates, each cell's bar bbox, detected top and baseline, the calibration's tick pairs/residual/zero, the uncertainty interval, reader version, status, and the verification verdict with its detail. |
| Acceptance: internal check plus the caller hook; totals never in code | `_internally_checked` (every emitted count a non-negative integer; every bin of a present series accounted for) runs always. `verify_panel` then applies `PipelineConfig.chart_constraint_hook`, `(survey_key, horizon, {series: {bin: count}}) -> accept/reject/no_opinion`. `survey_key` is the source document's stem, `horizon` the panel's own printed heading. Without a hook the derivation is published labelled UNVERIFIED. A hook that raises leaves it unverified, never takes the reading with it. |
| A failed constraint rejects the derivation, never the image | A REJECTED panel publishes a note naming the refusal and NO table; the crop reference is unchanged. Pinned end to end through the pipeline. |
| Real table replaces Stage 0's note; crop stays | `panel_block` writes `\| Series \| <bins…> \|` with one row per present series, cells an integer or the literal `UNRESOLVED`. It ships through `suppress_chart_table_skeletons(derivations=…)`, so the binding that decided WHERE it goes is Stage 0's, unchanged. |
| Surfaced at page, document and CLI | Page: an audit note on the winning output, plus `PageState.chart_derivations`. Document: `chart_counts_derived` / `chart_counts_not_derived` events, both in `resume_restore_kinds`. CLI: "N chart derivation(s): V verified / U unverified / R rejected; C cell(s) read, X UNRESOLVED". |
| Byte identity and resume hold | The reading is a pure function of the page and is cached per page per run, so every candidate crossing produces the same bytes; `_rewrite_all_fragments` remains the sole authoritative writer. Resume replays both event kinds and recomputes `chart_derivations` from them. Pinned. |

What the dotplot page ships now: five panel headings, five tables of counts (two series
on four panels, one on 2021), the page's own prose, and the five crops.

## Residuals carried out of Stage 1

1. **Vector only.** A scanned or rasterised chart has no drawing operators; the reader
   refuses it by name (`chart_counts_not_derived`, "the page draws no vector operators")
   and Stage 0's note ships unchanged. A raster lane (column profiling against the same
   calibration) is deliberately not built here.
2. **`absent` is never emitted.** A panel that draws no marks for a legend-declared
   series yields `unresolved`, because "the series is absent from this panel" and "every
   bin of it is zero" are not distinguishable from that panel's geometry — an all-zero
   staircase would be drawn coincident with the axis and add no ink. The trichotomy is
   in the model and in the sidecar; only two of its three values are reachable from a
   chart shaped like this one.
3. **The derivation publishes only where an empty grid stood.** The block replaces the
   withheld skeleton, so a chart page whose model output never emitted a grid gets the
   event but no table in the body. Giving the reader its own insertion point is a change
   to #189's placement contract and was left out. Round 2 made the surfacing honest about
   it: the page note and the CLI both report only what the document actually holds, and
   the CLI names the unpublished readings on a separate line rather than counting them
   as shipped.
4. **The axis titles are read positionally.** The count unit is the topmost multi-token
   row above the highest tick that is not drawn wholly inside the plot; the bin unit is
   the first narrower row below the bin labels. That is the corpus figure's layout. A
   chart that titles its y axis inside the frame would lose the unit (and only the
   unit — no count depends on it).
5. **Crop digest is computed by re-rendering.** `crop_digest` mirrors
   `_render_chart_region_crops`'s matrix, rotation and clip and hashes the pixmap, so
   the provenance names the file that ships without waiting for assembly to write it.
   Pinned: a test renders the crop through `_render_chart_region_crops` and asserts the
   file on disk hashes to the digest the derivation recorded. If the two paths ever
   diverge, that test is what says so.
6. **The acceptance hook is fingerprinted by identity, not by behaviour.** Round 2 put
   `module.qualname` (empty when absent) into the run fingerprint, so adding a hook,
   removing one, or swapping one for another reprocesses pages that were already
   terminal — which closes the dangerous direction the reviewer measured, where a hook
   turned on to REJECT a table was skipped on resume and the rejected table stayed in the
   document. What remains: editing a hook's logic without renaming it is not detected,
   because a fingerprint cannot see a function body. Said out loud in the config
   docstring; the instruction to callers is to rename the hook when its rules change.
7. **Stage 2 is untouched.** No model is consulted; geometry is the only authority, as
   the design requires before proposals can be reconciled against it.
8. **One document is not a defect rate.** Every claim about accuracy here is measured on
   a single Fed SEP page, now read independently by a second reviewer through two
   channels that share no code with the implementation (raw operators with bin
   boundaries taken from the staircase risers, and a 600-DPI raster ink read): zero
   disagreements over all 65 cells. The claim is "correct on this page", not "correct".
   The two refusal paths added in round 2 are exercised by synthetic drawings only,
   because this fixture draws neither a bar spanning two bins nor a staircase that stops
   without descending.
9. **The corpus golden does not run in CI.** `test_dotplot_fixture_five_panels` is
   skipped when `~/Data/socr/fixtures/dotplot/dotplot-p20.pdf` is absent, which it is on
   CI. The strongest evidence in the ticket is therefore unenforced on every CI run; a
   green tick does not mean the golden was checked.

## Stage 2 — model-assisted proposals — **TODO**

Not started. Depends on Stage 1's geometry being the authority.

## Owner decision — **ANSWERED (yes)**

From DESIGN.md: *may expected totals be keyed by survey and horizon, with absent series
represented explicitly?* **Yes**, recorded with the Stage 1 brief. Implemented as
`PipelineConfig.chart_constraint_hook`, which receives the survey key and the horizon
separately and whose payload carries only the series the panel actually draws. No
expected total exists anywhere in this codebase.
