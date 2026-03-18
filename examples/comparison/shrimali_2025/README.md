# OCR Comparison: Shrimali & Ahmad (2025)

**Paper:** "When the central banks are all ears?" (Finance Research Letters, 6 pages, 1.8 MB, born-digital)

## Methods

| # | Method | Command | Words | Time | Notes |
|---|--------|---------|-------|------|-------|
| 01 | Standard (gemini CLI) | `socr process paper.pdf --primary gemini` | 3,697 | 42s | Legacy pipeline, single engine |
| 02 | Unified (gemini CLI) | `socr process paper.pdf --unified --primary gemini` | 2,930 | 129s | 5-phase pipeline, repair loop ran 2x for 3 enhanced pages |
| 03 | Unified (gemini-api) | `socr process paper.pdf --unified --primary gemini-api` | 3,137 | 80s | Per-page API, no truncation risk |
| 04 | Multi-engine (gemini+mistral) | `socr process paper.pdf --multi-engine gemini,mistral` | 3,556 | 46s | Mistral unavailable, fell back to gemini only |
| 04b | Multi-engine (gemini-api+gemini) | `socr process paper.pdf --multi-engine gemini-api,gemini` | 3,213 | 125s | Both ran, consensus pending (whole-doc vs per-page gap) |

## Observations

- **Standard vs Unified:** Similar output quality. Unified runs born-digital detection + repair loop, adding ~90s overhead for a 6-page paper. The repair loop tried to enhance 3 pages with tables/figures but used the same engine (gemini).
- **CLI vs API:** Per-page API (gemini-api) avoids truncation risk entirely. Slightly fewer words because each page is processed independently (no cross-page context).
- **Multi-engine:** When both engines are available, the consensus engine picks the best output per page. Currently limited by CLI vs API output format mismatch (whole-doc vs per-page).

## Verdict

For short born-digital papers: **Standard pipeline with gemini CLI** is fastest and produces good output. The unified pipeline's overhead isn't worth it for 6-page papers.

For long papers (>30 pages): Use **unified with gemini-api** (per-page, no truncation) or **multi-engine** for best quality.
