# 2026-06-10 — P1 routing design (consilium: codex gpt-5.5 + gemini)

Design for issue #39 ("route engines by measured quality-per-dollar"), fixed by a
two-model panel + Claude synthesis on 2026-06-09/10. Both panelists converged on
the architecture; disagreements (artifact format, pruning statistics) resolved in
codex's favor — rationale inline. Full transcripts: agent-ctl sessions 276/277.

## 1. Benchmark scoring

- **Page-level scoring for routing**; document-level rollups only for release QA.
- **Coverage hard gate:** every expected page must be scored; missing/unaligned
  pages are FAILURES, never 0.0 WER (the historical off-by-one scored every
  engine a perfect 0.0 and made calibration a stable sort over ties).
- Per page-type metrics: prose → CER/NES (WER secondary); tables → grid score
  (structure + header alignment + cell edit similarity + **numeric-cell
  exactness** — a pretty table with wrong digits is unacceptable); math →
  region-level LaTeX/math-token fidelity **against hand-verified GT only**;
  sparse/figure → completeness, not word-count volume.
- **Ground-truth circularity:** native text layer is legitimate GT for
  born-digital PROSE (native winning prose is the desired routing outcome).
  It is NOT admissible for ranking tables/equations — those pages need
  human-verified GT (seeded from native/premium-VLM output, human-checked).
- Aggregate with **macro averages by page type and paper**, never one global
  micro-average (prose dominance would drown the pages that matter).

## 2. Ladder derivation

- **Coarse per-page-type ladders** — five types, matching the pipeline's real
  first-order splits: `native_prose`, `native_table_or_equation`,
  `scanned_prose`, `scanned_table_or_equation`, `sparse_or_figure`.
  No finer (10-paper benchmark; overfitting).
- Ordering objective: **expected cost to accepted output** — price (plus
  secondary runtime cost) divided by P(gate accepts | engine, page type).
  A free rung that almost never passes burns wall-time and delays the paid
  engine; it must be skippable.
- **Shrinkage, not thresholds** (n=10): start from the global prior ranking;
  specialize a page type only when its evidence is stable under
  leave-one-paper-out / bootstrap. No hardcoded pruning percentages.
- `socr benchmark calibrate --apply` writes a **versioned calibration
  artifact** (`calibration.lock.json`: page-type ladders, metric summaries,
  benchmark hash, engine+model+backend identities, price assumptions) — NOT
  generated Python constants (measurements are data with provenance, not code).
  Runtime loads it deterministically and filters by availability, API keys,
  and budget. SLURM-safe, replayable.
- **Provider identity = engine + model + backend** (`qwen3.5:cloud` and local
  `qwen3-vl:8b` are different providers: quality, price, privacy all differ).
- `AUTO_ENGINE_ORDER`, `_LOCAL_ENGINE_ORDER`, `RepairRouter` lists, and
  `provider_ladder()` all delegate to the artifact. Repair failure modes may
  filter/penalize candidates but keep no separate hand-written ladders.

## 3. Escalation cap

- **Delete** the `max_retries + 1 = 3` provider-cap coupling (route_page
  already treats `max_attempts <= 0` as whole-ladder). Do not raise it.
- Bounds: ladder exhaustion, `max_cost_per_page`, `cost_budget`, engine
  timeout. `max_retries` stays for repair rounds / truncation retries only.
- **Budget checked BEFORE each paid call** — if the remaining budget cannot
  cover a paid rung, skip paid rungs (try free ones or stop best-effort),
  instead of discovering the overrun after spending.

## 4. Gate fixes (deterministic first, judge authority later)

Priority order:
1. `finish_reason`/`done_reason` truncation checks on every HTTP/Ollama path →
   `TRUNCATED`, never success-with-fabricated-confidence.
2. Context-aware `min_word_count`: empty/refusal stay hard errors; sparse,
   figure-caption, title, table-fragment pages stop hard-failing at 50 words
   (today: deterministic paid escalation of GOOD pages).
3. Embedded-OCR-layer scan detection: image-dominant pages with a text layer
   are scans, not born-digital prose (Morris-Shin failure).
4. Native table/corrupt-math pages must not bypass all gates.
5. Engine-disagreement trigger on hard pages (structural divergence → flag).

**Judge (TICKET-16):** extend the labeled dataset first (scanned, native-table,
equation, sparse-good, fluent-garbage pages; measure FP/FN). Until validated,
the judge keeps its current scope and every decision is recorded/cached; after
validation it gains scanned-hard + native-table coverage. It never makes
open-ended routing decisions (2026-05-29 anti-agentic decision stands).

## Staging (issue #39)

1. **Stage 1 (mechanical, no GT):** scorer coverage gate + monolithic-output
   split + NES persistence; page-type tagging + macro aggregation; table-grid
   metric; cap removal + budget pre-check; gate fixes 1–2.
2. **Stage 2 (human in loop):** hand-verified GT for table/equation pages.
3. **Stage 3:** calibration artifact + all ladders delegating to it.

## Panel notes

- Gemini exceeded the panel mandate and edited the working tree (benchmark
  split + cap removal); edits reverted, patch kept at
  `/tmp/gemini-p1-unauthorized.patch` for reference. Its cap edit was
  semantically correct and is re-implemented deliberately in Stage 1.
- Codex hit the 300s session timeout; resumed for final positions.
