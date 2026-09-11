# chart-data (#635) — status

Design: [DESIGN.md](./DESIGN.md) (Astra, 2026-09-11). Three stages; only Stage 0 is built.

## Stage 0 — suppress empty derivations, preserve evidence — **DONE**

Branch `feat/635-stage0-chart-skeleton`. New module `src/socr/figures/chart_data.py`;
hook `UnifiedPipeline._suppress_chart_table_skeletons` at the #688 candidate seam in the
agentic per-page lifecycle; tests in `tests/test_gh635_chart_table_skeletons.py`.

Done-when, and how each is met:

| Done-when | Met by |
| --- | --- |
| The check is STRUCTURAL, never lexical | `find_empty_skeletons`: header row is a column key, column 0 of a body row is the row label (the convention `binding.parse_grid` / `label_canonical` already hold), every remaining cell carries no non-whitespace character. No word is matched. |
| A literal `0`, text, and unresolved tokens are data | The same single rule — only a cell that says nothing at all is empty. Pinned three ways. |
| Binding uses native geometry and unique source anchors | `region_interior_rows` (the complement of `chart_region_anchors`) plus two independent proofs: a label drawn inside exactly one region matching exactly one candidate line, AND every data column key attested by that region's own axis tokens. |
| Ambiguity quarantines, never deletes | Any failed proof, an intervening panel label, or a candidate panel order running backwards against the source leaves the text byte-identical and records `chart_table_skeleton_unbound`. |
| Provenance retains the original bytes and hash | `chart_table_skeleton_suppressed` carries page, table id, region index, crop filename, SHA-256 and the original grid text. |
| Mutation applied ONCE, at the candidate seam | `orchestrator._phase_agentic`, immediately after `canonicalize_candidate(bo)` — before table scoring, the crop reread, escalation, the judge gate's witness/identity creation and the flush. Not in `reconcile_chart_region_refs`. |
| Surfaced at every level | Page sidecar note (`winning_output.audit_notes` + `audit_events`), document audit events, CLI line "N chart-table skeleton(s) suppressed; crops kept". |
| Byte-identity and resume hold | `_rewrite_all_fragments` still the sole authoritative writer; the pass is idempotent, so re-assembly and a second run reproduce the same bytes. Pinned. |

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
3. **Agentic path only.** The hook sits in `_phase_agentic`'s per-page lifecycle. The
   phase-major single-engine / multi-engine / consensus paths still ship the empty grid.
4. **Refusal events on ordinary chart pages.** A chart page carrying an unrelated empty
   form now emits one `chart_table_skeleton_unbound` event per run. Intended (the decision
   is visible), but it is new audit-log volume.

## Stage 1 — geometric reader — **TODO**

Not started. Discrete vertical bars/histograms only; legend-derived series mapping,
two-tick calibration, per-cell provenance, `UNRESOLVED` rather than a guess. Golden counts
require independent human annotation — the issue's worked June example sums to 19, not 16,
and must not become the oracle.

## Stage 2 — model-assisted proposals — **TODO**

Not started. Depends on Stage 1's geometry being the authority.

## Owner decision — **PENDING**

From DESIGN.md: *may expected totals be keyed by survey and horizon, with absent series
represented explicitly?* Astra recommends yes; a universal equal-total/16 rule would reject
valid panels. Unanswered. Stage 1's acceptance hook cannot be specified until it is.
