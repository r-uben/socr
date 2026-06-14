# 2026-06-13 — Research findings behind the agentic local-first plan

Consolidated from a planning session (Claude + Codex gpt-5.5 + Gemini web research + an
own-hardware OCR benchmark). These are the facts the tickets rest on.

## Architecture (Codex gpt-5.5, two rounds)
- Make **agentic the default**, but the "cheapest-local-capable-first" policy lives in the
  deterministic Python loop (`provider_ladder` → `route_page` → judge), NOT an LLM.
- A Claude skill is optional and only as a **frozen preflight** (pick profile/budget/judge,
  freeze into manifest). Never the live routing authority — reproducibility is a hard
  requirement for a citable corpus.
- Reproducibility comes from a **rich frozen manifest**, not `temperature=0`. Today's manifest
  is too thin → enrich before flipping the default (TICKET-B3).
- Split provider identity into **engine + backend + model** (TICKET-A1) — `QWEN` currently
  means either local or cloud, which breaks routing semantics.

## Benchmark (own 64GB Mac, hard born-digital pages, 200 DPI = production)
5 Ollama vision models × hard pages (Kuttner Table 2, Morris-Shin eqs, Evans matrix, BK2005
prose) + the real Consensus Forecasts 11-column US forecaster table (202606 p4).

- **`qwen3-vl:30b-a3b-instruct` (A3B MoE, non-thinking) = local winner.** Prose clean; math =
  flawless LaTeX; 5-col table = perfect grid, digit+sign exact (recovers signs native corrupts:
  `@25`→`−25`, `þ4`→`+4`); **11-col Consensus table = 91s, each forecaster on its own row,
  digit-exact vs the Gemini reference** (Eaton, Goldman matched). Blemish: summary rows un-pair
  the year columns (→ TICKET-D1).
- **Thinking trap:** the default `qwen3-vl:30b` is a *thinking* build. On the dense 11-col
  table it emits 5000+ thinking tokens and never reaches the transcription within 400-600s.
  `think:false` (API) and `/no_think` (prompt) are BOTH ignored on Ollama 0.30.8. Only the
  `-instruct` variant disables thinking. → motivates TICKET-C1 (stall guard) and the
  hard rule "local model = the instruct variant."
- **Rejected:** `qwen3-vl:8b` (collapses dense tables, ~200s); `minicpm-v4.5:8b` (degenerate
  loop, broken on Ollama); `qwen3-vl-ocr` (timeout); OlmOCR-2 (context-limit / empty here).

## Landscape (Gemini web research, mid-2026)
- `qwen3-vl:30b` is the **local frontier** on 64GB. Genuinely better (Llama-4 Scout 109B,
  Qwen3-VL-235B) is too big or API-only. Smaller (Granite-Vision, Nanonets, dots.ocr, GOT-OCR2,
  PaddleOCR-VL, Florence-2) = hype/worse on dense grids.
- Dedicated table models (TableFormer/Docling) are subsumed by modern VLMs; but cropping the
  table *before* the VLM helps (validates socr's dual-pass crop design).
- Web free-tiers for dense tables if ever needed: Mathpix (gold standard), LlamaParse (10k
  pg/mo), Google DocAI (1k/mo), Azure DocIntelligence (500/mo).

## Infra gotcha discovered
`uv run` HANGS in this iCloud repo (venv sync stalls on fileproviderd). Use the venv binaries
directly: `~/venvs/socr/bin/{python,pytest,ruff}`. This is now a hard rule for all agents.

## Artifacts
Benchmark harness + outputs: `scratch/bench/` (run_bench.sh, pages200/, out200/, ce/).
Memory: `[[reference-local-ocr-benchmark-jun2026]]`, `[[reference-sococrbench]]`.
