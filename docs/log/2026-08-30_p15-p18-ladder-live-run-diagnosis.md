# Diagnosis: p15 / p18 extraction failures (GH-353 first live run)

Doc: 2002__cochrane_piazzesi__bond_risk_premia__WP.pdf
Run artifacts: /tmp/socr-ladder-trial/2002__cochrane_piazzesi__bond_risk_premia__WP/
socr @ main c0d8952 (GH-353 table judge ladder, merged today)

## Page 15 — whole-page prose loss on a D3-floor table rejection

**Root cause:** by design, the D3 fail-closed floor (panel decision Q1=D3, GH-200/TR-3)
replaces the page's ENTIRE `PageOutput.text` with just the failure marker + image ref
when (a) `native_table_unverifiable` is set and (b) no model candidate authored a usable
grid (`d3_floor_kept_model_output` returns None). The docstring frames this as "route the
*region* to the image-asset lane," but the implementation has no region-level splice —
`PageOutput` carries one monolithic `text` field per page, so shipping the marker discards
every paragraph on the page, including prose that had nothing wrong with it.

This IS the coded behavior, not a transient bug: `src/socr/core/manifest.py:1051-1061`
constructs `PageOutput(text=d3_text, ...)` from `d3_marker` alone — `p.native_text` (which
holds the full page's prose) is never read or spliced in at this site. The comment block at
`manifest.py:997-1003` documents intent ("ship neither flawed table ... route the region to
the image-asset lane") but the code executes at page granularity, not region granularity.

**Evidence:**
- `pages/00015.json` `winning_output.text` = `"[page 15 failed: unverifiable table — see
  image]\n\n![Failed table page 15](figures/failed_table_p15.png)"` (103 bytes), `status:
  "error"`, `failure_mode: "native_table_structure_failed"`, `engine: "native"`.
- `native_table_unverifiable: true`, `table_ladder_disposition: "table_rejected"`.
- Audit event `table_region_geometry_hard_fail` (engine=native): "per-region geometry
  verifier hard-failed on this page's native table (numeric-token multiset mismatch...)".
- Audit event `table_region_unverifiable` (engine=native): "per-region geometry verifier
  hard-failed (geometry_impossible_collapse) and OCR ladder also failed; D3 fail-closed:
  explicit failed-table marker shipped".
- Both qwen and gemini candidate table extractions were attempted (`table_value_drift_
  unadjudicated` events, engine=qwen and engine=gemini) but landed AMBIGUOUS (row-count
  mismatch made per-row pairing unreliable), and the judge ladder ultimately rejected the
  qwen candidate (`table_ladder_rejected`, rung_trail: mechanical:binding →
  ollama:glm-5.3-flash:cloud → gemini(agy), all executed OK, content still rejected) — so
  `kept_model` came back None and the pure-marker branch fired.
- `src/socr/core/manifest.py:1004-1061` is the exact branch: lines 1022-1041 are the
  "kept_model present" branch (splices model text + supersede note + image, would have
  preserved SOME reading); lines 1043-1061 are the "kept_model is None" branch that emits
  the bare marker — this is the one that fired here, and it never touches `p.native_text`.

**Verdict:** the table-rejection-vs-prose-loss coupling is a genuine scoping bug, not
intended behavior at the page-content level — the design intent (stated in the comment) was
region-scoped, the implementation is page-scoped. Table 5's caption/intro prose (visible in
the earlier in-loop flush of 00015.md, later overwritten by the terminal marker) is citable
prose with zero defects, discarded solely because the table on the same page failed.

**Proposed ticket:** "D3 fail-closed table floor drops surrounding page prose it never
verified as bad" — When `_winning_page_output` ships the D3/TR-3 unverifiable-table marker
(`manifest.py:1051-1061`, and the scanned-table variant at `manifest.py:985-995`), splice the
marker in place of ONLY the failed table region within `p.native_text`, instead of replacing
the whole page's text. Requires locating the table's row-span within `native_text` (already
tracked via `native_words`/row rects used by `reconstruct.py`) and doing a targeted
substitution, falling back to today's whole-page marker only if the table's textual span
can't be isolated. Add a regression fixture: a synthetic page with an unrelated prose
paragraph before/after a D3-floored table, assert the prose survives in the final page text.

## Page 18 — chart-region false positive strips the whole table, ships flattened native prose

**Root cause:** `chart_region_bboxes()` (`src/socr/tables/reconstruct.py:1002-`) misclassifies
Table 6's own booktabs rule lines as a "chart" drawing cluster. The union-find clustering
(`_union_find_clusters`, gap=`_CHART_CLUSTER_GAP_PT`=30pt, `reconstruct.py:892,915-959`)
merges every horizontal/vertical rule of the table into ONE cluster because adjacent table
rows are well under 30pt apart; the merged cluster's area comfortably exceeds
`_CHART_MIN_CLUSTER_AREA_PT2` (14 400 pt², `reconstruct.py:888`) since it spans the table's
full width and height. `_has_filled_rects_or_thick_strokes` (`reconstruct.py:962-999`) then
returns True on the "thick stroke" branch (`width > 1.0`, line 996-998) — a plain black
booktabs double-rule is >1pt wide, no color/fill needed to trigger it.

Once `chart_region_bboxes` returns a bbox that covers the entire table, `rowize_from_
words_chart_aware` (`reconstruct.py:1132-1199`) excludes every word inside it from
`non_chart_words` (line 1181), so the table rowizer receives zero words for that region and
returns nothing tabular. With no table candidate to accept, `decision.accepted=False`; per
`_page_has_tables` this is a structure-class table page with no usable grid, so it falls to
`structure_class_native_fallback` (`orchestrator.py:5500,5640-5641,6066-6078`; sidecar event
`structure_class_native_fallback`, `tables_trust.py:157-163`): "native's prose ships instead,
flagged WARNING rather than SUCCESS." Native's raw `page.get_text()` extraction is untouched
by the chart-region exclusion (that exclusion only affects the *table rowizer's* input), so
it ships the full page including every table token, but as one flat word-order stream with no
grid — exactly the "run-on paragraph of numbers" observed. The `![chart region 1](...)`
placeholder gets spliced into `ps.native_text` mid-stream by `_render_chart_region_pngs`
(`orchestrator.py:1524-1618`, called from `orchestrator.py:3481-3487`), which crops and saves
the wrongly-detected chart bbox — confirmed by inspecting `figures/chart_region_p18_1.png`:
it is a perfectly legible screenshot of the ENTIRE Table 6 grid, not a chart.

**Evidence:**
- `pages/00018.md` line 18: `![chart region 1](figures/chart_region_p18_1.png)` immediately
  followed by flattened header tokens and 30 bare numeric lines — no markdown table anywhere.
- `pages/00018.json`: `native_table_structure_failed: true`, `native_table_unverifiable:
  false` (this is why P18 did NOT take the D3-floor path like p15 — no per-region geometry
  hard-fail was recorded, only value-guard multiset mismatches on qwen/gemini candidates),
  `table_ladder_disposition: "table_rejected"`, final audit event `structure_class_native_
  fallback` engine=native: "structure-class page (table); a model rung ran but authored no
  usable grid, and native may not author one either (C1) -- native's prose ships instead."
- `audit_log.json` `chart_asset_page` events fire on pages [8, 11, 22, 25, 27, 36] — **18 is
  NOT among them**, confirming the whole-page chart-asset router (`has_chart_marks` /
  `_is_chart_asset_page`) never routed p18 to the image lane; this is purely the in-table
  `chart_region_bboxes` false positive inside the native table rowizer, a different code path.
- `figures/chart_region_p18_1.png` renders as: a clean 6-row, 11-column table with headers
  `γᵀf (t) U (t) XLI (t) cay (t) cpi (t) R²` — visibly a table, not a chart.

**Verdict:** genuine misfire, not designed behavior. The chart detector's "thick stroke"
heuristic (`reconstruct.py:996-998`, any drawing width > 1.0pt) has no floor/ceiling
calibrated against booktabs table rules, and the 30pt cluster gap merges an entire
multi-row table's rule set into one blob that trivially clears the area threshold. Nothing
in `chart_region_bboxes` checks whether the drawing cluster's aspect ratio / rule-line
pattern looks like a table (parallel full-width horizontal lines, no vertical bars/points)
before calling it a chart.

**Proposed ticket:** "Chart-region detector misfires on booktabs table rules, discarding the
whole table" — `chart_region_bboxes` (`reconstruct.py:1002`) needs a table-rule exclusion
before accepting a cluster as chart-like: e.g. reject clusters whose drawings are dominated
by full-cluster-width horizontal strokes with no vertical bars/markers (the union-find
cluster IS the table's rule grid, not a bar/line chart). Cheapest fix: only accept
`_has_filled_rects_or_thick_strokes` on the "thick stroke" branch when the stroke's y-extent
also has non-full-width horizontal segments or accompanying colored/filled marks — a lone
top/bottom double-rule spanning 100% of the cluster's width should not qualify alone. Add a
regression fixture from p18's actual PDF page (or a synthetic booktabs table with a >1pt
double rule and no other vector marks) asserting `chart_region_bboxes` returns `[]` for it.

## Secondary: was gemini cloud OCR escalation attempted?

Yes, on both pages — but only inside the table JUDGE ladder, not as an alternate page-level
OCR engine. Rung trails for both p15 and p18 (`table_ladder_rejected` events) show
`ollama:glm-5.3-flash:cloud` then `gemini` (backend `agy`) both executed successfully
(`"ok": true`) as arbitration rungs over the qwen-authored candidate table; the ladder still
rejected the table on content grounds (value drift / multiset mismatch), not because gemini
was gated or skipped. Separately, gemini DID run as a candidate table-extraction engine on
p15 (`table_value_drift_unadjudicated` engine=gemini) and produced a table-verifier hard-fail
on p18 (`native_table_verifier_hard_fail` engine=gemini) plus a `dualpass_patched` repair
attempt — so gemini was not gated on either page; it ran and its output was rejected the same
as qwen's. Nothing here indicates an escalation-gating bug; both pages simply have genuinely
defective/ambiguous source tables that every engine (native, qwen, gemini) disagreed on.
