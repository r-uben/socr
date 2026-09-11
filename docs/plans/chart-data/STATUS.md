# chart-data (#635) — status

Design: [DESIGN.md](./DESIGN.md) (Astra, 2026-09-11). Three stages; only Stage 0 is built.

## Stage 0 — suppress empty derivations, preserve evidence — **DONE**

Branch `feat/635-stage0-chart-skeleton`. New module `src/socr/figures/chart_data.py`;
hook `UnifiedPipeline._suppress_chart_table_skeletons`, called from `route_page`'s
`on_candidate` boundary (and from the escalation and crop-repair replacements); tests in
`tests/test_gh635_chart_table_skeletons.py`.

Done-when, and how each is met:

| Done-when | Met by |
| --- | --- |
| The check is STRUCTURAL, never lexical | `find_empty_skeletons`: header row is a column key, column 0 of a body row is the row label (the convention `binding.parse_grid` / `label_canonical` already hold), every remaining cell carries no non-whitespace character. No word is matched. |
| A literal `0`, text, and unresolved tokens are data | The same single rule — only a cell that says nothing at all is empty. Pinned three ways. |
| Binding uses native geometry and unique source anchors | `region_interior_rows` (the complement of `chart_region_anchors`) plus two independent proofs: a label drawn inside exactly one region matching exactly one candidate line, AND every data column key drawn IN FULL along that region's axis — the keys' first atoms consecutive on one word-row in the header's order, each further atom of a range key in the row below and in the same column, which is how a two-line tick label is printed (`region_axis_rows` / `_axis_attested`). Sharing a token is not attestation. |
| Ambiguity quarantines, never deletes | Any failed proof, an intervening panel label, or a candidate panel order running backwards against the source leaves the text byte-identical and records `chart_table_skeleton_unbound`. "Quarantine" here means exactly that: the unbound grid stays in the published text and the event says it was looked at and kept. Nothing is moved to a separate artifact. |
| Provenance retains the original bytes and hash | `chart_table_skeleton_suppressed` carries page, table id, region index, crop filename, SHA-256 and the original grid text. |
| Mutation applied at candidate INGESTION, before judgment | `route_page`'s `on_candidate` boundary, right after `canonicalize_candidate(output)` and before `judge.assess` — so the page judge, table scoring, witness/identity creation, selection and the flush all see the withheld text, and no verdict is passed on bytes that later change. The escalation candidate and the crop-repaired text cross the same seam, so a later reading cannot reintroduce the grid; the `_phase_agentic` crossing is now a backstop for pages that arrive by another door. Idempotent, and the records are deduplicated by the grid's own `(kind, table, sha256)`. Not in `reconcile_chart_region_refs`. |
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
3. **Candidate replacements outside the three covered seams.** Ingestion (every ladder
   rung), the table-escalation candidate and the crop-repair patch all cross the seam, and
   `_phase_agentic` keeps a backstop crossing for a page that arrives by another door (the
   native lane, a ledger restore). What is NOT covered is text that becomes the page body
   after that backstop: the native/manifest fallback selected at assembly, and
   `_phase_assemble`'s own rewriters. None of them authors a chart-derived grid today, so
   this is a gap in coverage, not a known loss.
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
