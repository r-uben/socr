# 2026-08-30 — GH-371: regional D3 table splice

## Decision

When the D3 table floor distrusts only some native table regions, socr preserves
the page's surrounding prose and replaces only those table blocks. Isolation is
proved with two pieces of evidence captured during native verification:

- the zero-based ordinals of the failed table regions; and
- the total number of parser-identified table regions examined.

The ordinal-plus-count pair is intentional. Table markdown is not an identity:
two tables can have identical markdown, so replacing by string equality can
replace the wrong occurrence. The ordinal identifies the occurrence and the
count proves that the parsed output still has the expected topology.

## Fail-closed boundary

Regional replacement is all-or-nothing. If the failed ordinals or expected
count are missing, malformed, duplicated, out of range, or disagree with the
table blocks parsed from the candidate text, isolation is not provable. The
splice then returns no regional result and the existing whole-page failure
marker remains the fallback. This keeps the previous fail-closed behavior for
partial, stale, or ambiguous provenance.

The same conservative rule applies to scanned pages under GH-90. A scanned
page's source-evidence floor distrusts the model's tables, so every table block
identified by the parser is removed/replaced. Removing all parser-identified
tables avoids shipping any unverified table merely because it was not the one
that caused the floor. If no table block can be identified, isolation is not
possible and the whole-page marker is used.

## Assembly and crash recovery

The in-loop text mutation was removed. Incremental fragment flushing is a
crash-recovery snapshot, while `_rewrite_all_fragments` and canonical assembly
are authoritative. Mutating the page text in the agentic loop could make the
fragment differ from the later winner/assembly calculation and could destroy
prose before the final disposition was known. The regional decision is therefore
derived at the winner/assembly boundary and persisted in the page sidecar so a
resume sees the same evidence.

## Marker semantics

A regional D3 result is not a whole-page failure marker. It contains a visible
failure marker for the distrustful table region, but verified surrounding prose
and any unaffected table blocks remain usable. `is_page_failed_marker` therefore
recognizes the whole-page fallback separately from regional output. Assembly
records the regional table event without incorrectly deriving a `page_failed`
event; only the fail-closed whole-page fallback carries that document-level
failure marker.

## Verification

The GH-371 tests cover duplicate-table identity, malformed-provenance fallback,
native and scanned prose preservation, sidecar persistence, and fragment/
canonical parity. The focused regression and full-suite gates were run without
Ollama, network access, or live providers:

- focused D3, GH-90, winner-selection, assembly, sidecar, and resume suites:
  **263 passed**;
- `~/venvs/socr/bin/pytest -q`: **2,585 passed, 5 xfailed**;
- `uvx ruff@0.16.0 format --check .`: **423 files already formatted**.


## Post-review fix (2026-08-31): the two-site collision

The adversarial review caught the working tree contradicting the "in-loop text
mutation was removed" paragraph above: the orchestrator's GH-90 handler still
stripped table blocks out of `best_output.text` in place, so the winner branch's
splice re-read already-stripped text, found no blocks, and fell back to the
whole-page marker — re-losing exactly the prose this ticket preserves (two
parallel implementers, one seam). Fixed by making the log's claim true: the
handler (`_apply_scanned_table_floor`) now demotes only, and
`_select_page_output_tagged` is the sole writer of the floor text. The dead
losing half (`remove_table_blocks` and its tests) was deleted, and
`test_floor_then_winner_chain_preserves_prose` chains the real handler with
winner selection so the collision cannot silently return.
