# Review packet — action 524e08eaf32d

- **Issue**: #146 — https://github.com/r-uben/socr/issues/146
- **Kind**: **COMMENT**  (issue STAYS OPEN; comment only)
- **Disposition**: PARTIALLY-IMPLEMENTED  (UNANIMOUS)
- **Adjudicator (may NOT review this)**: kimi  ·  batch batch-1
- **Triage seats for this batch (may NOT review this)**: grok, claude, ollama-minimax

> Note: the triage seats SPLIT on this issue. Read both readings before voting.

## Per-seat triage verdicts (machine-checked)

| seat | verdict | evidence_verified | fixed_by_commit |
|---|---|---|---|
| claude | PARTIALLY-IMPLEMENTED | yes | — |
| grok | PARTIALLY-IMPLEMENTED | yes | 356b727d4afb8149d1c96771c8cb7bf7d12296ce |
| kimi | ALREADY-FIXED | NO | 356b727 |
| ollama-minimax | PARTIALLY-IMPLEMENTED | yes | 356b727d4afb8149d1c96771c8cb7bf7d12296ce |

## Basis given by the adjudicator

Cause 1 (unconditional grid[0] promotion) is fixed as prescribed — reconstruct.py:572 guards with _is_data_row and emits an empty header. Cause 2 (region excludes the header band) has machinery (_extend_scope_for_header, _prepend_header_band) but it only runs on the rejected-grid fallback scope, not the rowizer path generally; the TICKET-D2 corpus measurement (17/23 wrong headers, 14 by this mechanism) postdates the fix and is unreproduced at main_sha. Two independent seats with verified evidence plus a third agree; the disposition is not ALREADY-FIXED because the issue's 'Then:' half is measurably incomplete on the corpus.

## Risk if this is wrong

A stale-but-harmless status comment; the band-demotion half is corroborated by a corpus measurement taken after the fix, so the residual is near-certain to be real.

## Evidence cited (already resolved against main_sha by bin/verify_citations.py)

[
 {
  "path": "src/socr/tables/reconstruct.py",
  "line": 572,
  "snippet": "    if not assume_header and _is_data_row(grid[0]):"
 },
 {
  "path": "src/socr/tables/reconstruct.py",
  "line": 1154,
  "snippet": "def _extend_scope_for_header(tight, words: list):"
 }
]

## EXACT text proposed for posting

        Status at 53b0637 (overnight sweep, machine-checked): the immediate lossless half landed — _grid_to_markdown tests _is_data_row(grid[0]) and emits an empty header rather than promoting a data row (356b727; tests/test_table_header_gh146.py green). The header-band half is partial: _extend_scope_for_header/_prepend_header_band exist but only on the rejected-grid fallback scope; on the plain rowizer path the header band still ships as prose above an empty-header grid, and TICKET-D2's corpus measurement (17 of 23 emitted tables with wrong headers, 14 by band demotion) postdates the fix and has not been re-run. Issue stays open.
