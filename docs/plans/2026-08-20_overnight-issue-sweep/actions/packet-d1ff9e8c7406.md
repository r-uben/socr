# Review packet — action d1ff9e8c7406

- **Issue**: #144 — https://github.com/r-uben/socr/issues/144
- **Kind**: **COMMENT**  (issue STAYS OPEN; comment only)
- **Disposition**: PARTIALLY-IMPLEMENTED  (UNANIMOUS)
- **Adjudicator (may NOT review this)**: kimi  ·  batch batch-1
- **Triage seats for this batch (may NOT review this)**: grok, claude, ollama-minimax

## Per-seat triage verdicts (machine-checked)

| seat | verdict | evidence_verified | fixed_by_commit |
|---|---|---|---|
| claude | PARTIALLY-IMPLEMENTED | yes | — |
| grok | PARTIALLY-IMPLEMENTED | yes | d645b24b7b8b4169d51e95a796de3147eaccd2ae |
| kimi | PARTIALLY-IMPLEMENTED | yes | — |
| ollama-minimax | PARTIALLY-IMPLEMENTED | yes | d645b24b7b8b4169d51e95a796de3147eaccd2ae |

## Basis given by the adjudicator

All three seats, on machine-checked evidence for two, agree the mechanism landed but the issue is not closable: the destruction check fires and rejects (reconstruct.py:171), yet its coverage holes are exactly #197/#198, its surfacing hole is #195, the fallback rowizer still misplaces a lane on the issue-comment synthetic, and the original corpus evidence (p17/42/43) is unremeasured. Not ALREADY-FIXED (acceptance not mapped, corpus claim unverified); not FIX-CANDIDATE tonight because the remaining core — the fallback rowizer's lane placement and the step-3 routing — needs measurement the plan defers by design (#144 step 3 is explicitly deferred in TICKETS.md).

## Risk if this is wrong

If actually fully fixed, an open issue lingers and a redundant comment is posted — cheap; the reverse (closing a live value-loss defect) is the unacceptable direction, so PARTIALLY-IMPLEMENTED is the safe error.

## Evidence cited (already resolved against main_sha by bin/verify_citations.py)

[
 {
  "path": "src/socr/tables/reconstruct.py",
  "line": 171,
  "snippet": "        if destroyed:"
 },
 {
  "path": "src/socr/tables/reconstruct.py",
  "line": 377,
  "snippet": "        if not (_NUM_TOKEN_RE.match(text) and _NUMERIC_RE.search(text)):"
 },
 {
  "path": "src/socr/tables/reconstruct.py",
  "line": 169,
  "snippet": "numeric_scope if numeric_scope is not None else fitz.Rect(table.bbox)"
 }
]

## EXACT text proposed for posting

        Status at 53b0637 (overnight sweep, machine-checked): the A2 destruction check lands and rejects value-splitting text-strategy grids with rowizer fallback (d645b24), and the per-region hard-fail reaches the D3 floor. Still open: (a) the check's own coverage — None numeric-scope falls back to table.bbox (#197), decorated numerics 0.67***/U+2212/$0.67 are never candidates (#198), rejection is log-only (#195); (b) on the issue-comment synthetic the fallback rowizer still emits `| 6M Treasury | yield | 0.85 |` and loses 0.06/0.80/0.05/0.94 from the grid; (c) the p17/42/43 corpus loss has not been re-measured at this SHA. Issue stays open; #195+#197+#198 are queued as one fix tonight.
