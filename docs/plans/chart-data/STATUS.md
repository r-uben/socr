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
| A zero needs an observable empty bin | A zero is emitted only when the series is PRESENT, the bin lies wholly inside the drawn plot, and nothing of that series is over it — for bars, a histogram resting on the axis, where a zero bar is invisible by construction; for the staircase, an outline the page draws DESCENDING to the axis. The question is asked of the whole undrawn stretch, not of two array indices: the reader walks out in each direction to the nearest bin the outline says anything about and asks THAT one for the descent, so a gap five bins wide is held to the same evidence as a gap one bin wide. A neighbour whose own level the reader could not resolve is evidence MISSING, never evidence not required. The walk's other admissible witness is a neighbouring run already drawn ON the axis, which never left it. BOTH witnesses are held to the same half-count bound as every height: a stroke too thick to locate the axis within half a participant certifies a zero by neither route. `Cell.empty_bin_observed` records it, and the cell detail names the bin whose descent or on-axis run was or was not drawn. (Round 2 finding: a gap read as zero while the cell claimed a riser check that had iterated an empty list. Round 3: that check reached only i±1 and was waived entirely by an unresolved neighbour, and its slack had no half-count bound. Round 4: the bound was asked of the descent but not of the on-axis witness, so at a 14 pt stroke a level 0.6 participants above the axis certified five zeros in a panel that refused that same level for the same uncertainty.) |
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
   docstring; the instruction to callers is to rename the hook when its rules change. The
   sharpest form of that limit, also stated there and pinned by a test: two hooks built by
   one factory share the qualname `make.<locals>.hook`, so a factory-shaped caller gets no
   discrimination at all.
7. **Stage 2 is untouched.** No model is consulted; geometry is the only authority, as
   the design requires before proposals can be reconciled against it.
8. **198 pages of measurement, and still no per-cell defect rate.** Accuracy is measured
   two ways, and they answer different questions. On the single Fed SEP fixture page
   (`dotplot-p20.pdf`) a second reviewer annotated all 65 cells independently through two
   channels sharing no code with the implementation — raw operators with bin boundaries
   taken from the staircase risers, and a 600-DPI raster ink read — with zero
   disagreements. That page still carries the only per-cell oracle there is. On the
   #735 branch the reader was then run over 198 pages: the 23 Fed SEP projection pages
   and the 175 FOMC minutes pages (35 meetings × Figures 3.A–3.E). Those have **no
   per-cell oracle**. What they have is an independent consistency check: across the 175
   minutes pages, 610 fully-resolved current-meeting series produce exactly one
   participant count per meeting, with the longer-run panels one below it and **zero
   stragglers over 35 meetings**, with no total supplied to the reader. That is strong
   corroboration and it is not a defect rate. The seven refusal paths now in the reader
   (rounds 2, 3 and 4 of #635, plus #735's compound-path guard, panel residual gate and
   uncorroborated-label-row refusal) are exercised by synthetic drawings only, because
   neither corpus draws a bar spanning two bins, a staircase that stops without
   descending, a gap whose far end is unresolvable, a stroke thick enough to defeat the
   half-count bound, or a page ground other than white.
9. **A stray bar refuses a bin it barely touches.** `_doubted_by` is a raw interval
   overlap with no tolerance, so a bar of unknown binning that reaches 0.14 pt into the
   next bin's interval refuses that whole bin. Round 2 made strays much more common (a bar
   spanning two label centres is now one), so this path carries more traffic than it did.
   It costs recall only, never correctness — a bin whose contents are contested publishes
   nothing rather than a guess — so it is recorded rather than charged.
10. **One unresolvable neighbour refuses a whole side.** The empty-bin walk stops at the
   nearest bin in each direction that says anything, so when that bin's own level is not
   established, every empty bin behind it walks back to the same ambiguity and refuses
   too. The fan-out is unbounded: one ambiguous level can cost every zero on its side of
   the panel. Reported by the round-3 reviewer. Like residual 9 it costs recall only —
   the alternative is publishing zeros witnessed by a level the reader could not read —
   so it is recorded rather than charged.
11. **The corpus golden does not run in CI.** `test_dotplot_fixture_five_panels` is
   skipped when `~/Data/socr/fixtures/dotplot/dotplot-p20.pdf` is absent, which it is on
   CI. The strongest evidence in the ticket is therefore unenforced on every CI run; a
   green tick does not mean the golden was checked.

12. **A region holding two panels reads one of them unless their tick columns differ.**
   Two plots in one region used to read as one: the lower axis was selected, the upper
   title was taken, and the lower chart's bars were published under it at a calibration
   residual of 0.0. The reader now counts the plot frames in a region and refuses a region
   that holds more than one — a refusal, not a split, because the region index is the
   identity the crops and the Stage 0 notes are keyed on, so the reader cannot manufacture
   new ones. Two things bound that count, and both were measured by the #735 reviewers.
   First, the count is by LADDER IDENTITY: `_span_groups` keys a ladder on its exact x
   span, so two plots stacked in the same column draw one span group holding both ladders'
   heights, every candidate axis reads that one merged ladder, and the region holds one
   frame by this count. What catches that drawing is the residual gate instead — a ladder
   of doubled length cannot fit one scale — so the outcome is still a refusal, by a
   different route. The refusal added here fires only where the plots' tick columns differ
   in x. Second, where the upper plot carries NO tick ladder of its own there is one frame
   to find, the calibration closes cleanly off the lower one, the lower panel reads
   correctly, and the upper panel is neither read nor refused — one crop covering both, one
   table showing one, no internal issue. That last case is the surviving silent loss, and
   closing it belongs to the region detector, which is what draws the boundary in the first
   place. Found by the #735 reviewers (`test_rev735b.py::test_d_...`, `test_rev735c.py`,
   `test_rev735f.py`, `test_astra_735.py::test_good_residual_...`); the ladder-identity
   limit is pinned as a control in `tests/test_gh735_sep_reader.py`.
13. **`_ladders_agree` has no bound relative to the tick pitch.** The two copies of a tick
   ladder are compared within half their own stroke width, which is the right resolution
   for a rounding difference but is not scaled to what a count is worth: a 4 pt
   disagreement against a 20 pt pitch is refused at a 0.4 pt stroke and accepted at a 9 pt
   one. It cannot produce a wrong number — a displaced ladder moves `zero_y`, which lands
   in `cal.residual`, which both the panel gate and `_resolve` charge — so it costs recall
   or refusal, never correctness. Recorded rather than charged.

14. **#734 is not closed by the #735 branch: a FILLED model grid never reaches the
   reader.** `_derive_chart_tables` in `src/socr/pipeline/orchestrator.py` opens with a
   structural pre-check, `if not find_empty_skeletons(text): return 0`, so chart derivation
   runs only where the model left an EMPTY grid. A model that fills its chart table with
   invented numbers bypasses the geometry path entirely, and nothing reconciles its values
   against what the page draws. The reader-side half of #734 (five panels collapsing into
   one region) is fixed; the reconciliation half is not, and #734 stays open. Found by the
   #735 reviewer by source inspection, not by a pipeline run.

15. **Labels drawn as vector art: the panel now REFUSES, and a caption at the bin centres
   still mislabels.** Where the bin labels are OUTLINED rather than set as text, the page
   carries no tokens on the label row at all. Through round 6 the bars then attested the
   nearest prose row and the panel published its counts under that row's words. **At this
   build it refuses**: the caption that used to take the bins is prose, and round 7's
   numeric gate rejects it (`/private/tmp/rev735/test_rev735g.py::test_k`, which published
   on every earlier commit of this branch and on `3cbf8a9`). The underlying weakness is NOT
   closed — a NUMERIC row on such a page would still take the bins, which is item 18 — but
   the fabrication this item described is not reachable here.

   **Still live, and a different thing from fabrication.** A four-word caption drawn at the
   bin centres BELOW a row of numeric labels is absorbed as a second LINE of those labels
   and joined into them: the bins become `1.0-Effective`, `2.0-federal`, `3.0-funds`,
   `4.0-rate` (`/private/tmp/rev735/test_rev735n.py::test_v`). The counts are right, the
   attested row is the real label row, and the page's own labels survive inside the joined
   string — so this is **mislabelling, not fabrication**, and it is pre-existing on main.
   In the partition rule's terms (`_aligned`, replaced in `c15cdcd`): a row below is a
   second line when its tokens partition one-per-column with none left over and the runs in
   order, and a caption with exactly as many words as the chart has bins satisfies that
   just as a printed lower bound does. Nothing geometric separates them. Harmless on both
   corpora, whose second lines are the range endpoints.

16. **A panel with no bar standing on its axis is refused outright.** Round 5 deleted the
   layout fallback: where no bar covers exactly one token of any row below the axis, the
   region has no bins and is refused. Four successive fallbacks were tried and each
   published a prose row as the bins with fabricated counts at a perfect residual — the
   densest row (the page's footnote), the only row in span, the first row, and both of the
   last two required together (#735 review rounds 1–4, and `read_bins`' docstring). What it
   costs is a dashed series with no bar anywhere on its axis, which now reads nothing. That
   branch is reached 0 times in 835 calls over the 198 corpus pages, and both corpus dumps
   are byte-identical across the change; the loss is therefore synthetic so far, and it is
   a refusal rather than a wrong number. The 14 synthetic drawings in
   `tests/test_gh635_chart_reader.py` whose point was the outline now draw one bar
   (`WITNESS`) so their own subject still reads, and the refusal itself is pinned as a
   difference against them.

17. **A bar standing on a prose row made that row the bins (round 6), and the geometric
   defence of it was abandoned in round 7.** Round 6 added two admission tests: a bar
   attests a row only if it lies INSIDE the bin that row's own centres derive
   (`_attesting_bars`), and the winner was discarded when a silent row below the axis was
   an equally good home for the same marks (`_unruled_out_rival`). Two reviewers then broke
   the second one in both directions — a fabrication through it, and a FALSE REFUSAL of a
   chart the reader otherwise reads correctly, caused by nothing more than a footnote
   printed under the chart. Round 7 **deletes `_unruled_out_rival` outright**; it is not
   patched or narrowed. `_attesting_bars` is kept, which neither reviewer could defeat.

   In its place, one rule: **every token of the chosen row must be a number, or a range of
   numbers, as printed** (`_numeric_row`). It uses Stage 0's own key grammar
   (`chart_data._key_atoms`) so the two halves of the feature cannot drift, and then
   requires each atom to parse as a number — the step that grammar does not take, since it
   accepts `B1` and `Effective` as well-formed keys. A row with any word token cannot be
   the bins, and where no row below the axis is all-numeric the panel is refused.

   **Measured cost: nothing on the corpus** — on all 840 `read_bins` calls the attested row
   is all-numeric, no call has zero numeric rows below its axis, and both corpus dumps stay
   byte-identical. The first version of the gate did NOT have that property and was caught
   by those dumps rather than by any test: it judged each token with the raw key grammar,
   which treats the corpora's own two-line form `0.13-` (upper bound on the next line) as
   malformed, so it rejected the real label row on every SEP call and published `0.37`
   where the page says `0.13-0.37`. A trailing range dash is now stripped before parsing,
   as `_join_atoms` already does, and the case is pinned by its own test. What it does cost is charts whose bins are NOT
   numeric: a histogram labelled by country or sector is now out of scope and refuses. The
   repository's own synthetic fixtures were relabelled from `B1..B5` to `1.0..5.0` for the
   same reason, and any reviewer probe still drawn with `B`-labels now refuses by design
   rather than by defect.

18. **OPEN, disclosed: ANY numeric row the bars attest becomes the bins.** The numeric
   gate stops all three round-6 constructions from publishing under prose, but "all three
   refuse" overstates it: two refuse outright, and the one whose stray mark sits among real
   numeric labels reads those labels and leaves the bins the stray contests UNRESOLVED. It
   does not close the class, and the class is wider than the first statement of
   this item allowed. What the reader requires of a row is only that it is all-numeric,
   that its tokens are inside the axis' span, and that the bars stand inside the bins its
   own centres derive. **Vertical order is not consulted at all** — with the rival rule
   deleted in round 7, nothing in the reader compares a candidate row's position against
   any other row's.

   So the route has at least two instances of one shape, both reproduced against this
   build. Print real bin labels `1.0 2.0 3.0 4.0` at x = 160, 180, 200, 220 and a numeric
   annotation `0 5 10 15` at x = 130, 190, 250, 310, then stand four 8pt bars of 3, 5, 4, 2
   on the annotation's positions. With the annotation **above** the labels the panel
   publishes 3, 5, 4, 2 under `0 | 5 | 10 | 15`; with the same annotation **below** them it
   publishes identically (`/private/tmp/rev735/test_rev735m.py::test_s`). Every annotation
   token parses, every bar lies wholly inside its annotation-derived 60pt bin, and no bar
   covers a real label centre.

   This is the owner-accepted class, not a new one: the reader establishes that a row is
   label-SHAPED and attested by the marks, and never that it IS the labels. It is pinned
   by `test_a_numeric_annotation_row_still_takes_the_bins` so it cannot drift silently, and
   it is unmeasured — no corpus page is known to draw it.

**Owner ruling, 2026-09-13 — best-effort numeric-chart extraction.** The owner has accepted
best-effort extraction of numeric charts as the product scope, against the alternative of
refusing every mapping the reader cannot prove. What that means, stated plainly so no
reader of these tables is misled:

* the reader establishes that a row is **label-shaped** (all numeric) and **attested** by
  the marks standing on the axis. It does **not** establish that the row IS the bin labels,
  and item 18 is a live construction in which it is not;
* published chart tables are **UNVERIFIED** unless a caller's `chart_constraint_hook`
  accepts them. The banner on every derived block says so, and nothing in Stage 1 promotes
  a reading to verified;
* the counts themselves are geometry, not guesses — an integer is emitted only where the
  measured interval admits exactly one — but the COLUMN those counts are published under
  rests on the attestation above.

**Scope of every number above.** All 835 measured `read_bins` calls come from the 198
dot-plot pages of the two Fed corpora, plus 5 more on the reference fixture. The other
corpora in `~/Data/socr` — 29 documents, 84 pages, BoE, ECB, Banxico and the older Fed
samples — produce **zero** calls into `read_bins`: `chart_region_bboxes` finds 49 chart
regions among them, and not one yields a frame and calibration the reader will read bins
for. So "no real page needs this" and "no real page is harmed by this" are measured on the
dot-plot corpus and *assumed* everywhere else, on the strength of a reader that declines
those pages earlier.

## Stage 2 — model-assisted proposals — **TODO**

Not started. Depends on Stage 1's geometry being the authority.

## Owner decision — **ANSWERED (yes)**

From DESIGN.md: *may expected totals be keyed by survey and horizon, with absent series
represented explicitly?* **Yes**, recorded with the Stage 1 brief. Implemented as
`PipelineConfig.chart_constraint_hook`, which receives the survey key and the horizon
separately and whose payload carries only the series the panel actually draws. No
expected total exists anywhere in this codebase.
