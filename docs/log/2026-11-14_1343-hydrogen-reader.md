# #1343 Gemini 4 Hydrogen as the Gemini reader (2026-11-14)

## Change

- `DEFAULT_GEMINI_MODEL = "gemini-4-hydrogen"` (API name confirmed by `GET /v1beta/models`, 2026-11-14).
- `gemini-3.8-flash` joins `RETIRED_GEMINI_MODELS`, so runs recorded under it keep a Gemini price
  (`profile_by_model`) instead of reading as $0.
- `PROFILE_GEMINI.cost_per_page_usd` is `GEMINI_COST_PER_PAGE_USD`, derived from the measured run
  instead of the old 0.0002 estimate (about 100x too low).

## Measured cost

100 random hard Fed pages, gemini-4-hydrogen, 2026-11-14:

- input tokens: 118,042
- output + thinking tokens: 252,503 (thinking is billed as output)
- list price: $1.50 per 1M input, $7.50 per 1M output

(118,042 x 1.50 + 252,503 x 7.50) / 1,000,000 / 100 = $0.0207 per page, about $21 per 1,000 hard pages.

## Test results (issue body, same images, same prompt)

| | Hydrogen | Gemini 3.8 Flash | Sonnet | qwen3-vl-30b |
|---|---|---|---|---|
| Numbers missing vs text layer | 391 | 919 | 1380 | 1781 |
| Tables only | 197 | 583 | 1005 | 1303 |
| Pages with zero numbers missing | 81 | 58 | 66 | 45 |

Image audit of the 10 pages with the most not-in-layer numbers (two Opus checkers, cropped images,
about 700 numbers): 0 invented, 0 wrong printed numbers. One defect: on a bar-chart page (Monetary
Policy Report Figure 3.D) Hydrogen estimated 40 bar heights and wrote them as table cells (accurate,
none printed) and dropped the axis ticks. The prompt rule that fixes it lives in gemini-ocr-cli and
is a separate change. Hydrogen ignores temperature; it is not sent.

## Escalation: is Gemini reachable as the hard-page reader?

Yes, by default, with these conditions:

- `GEMINI_API_KEY` or `GOOGLE_API_KEY` set and the `gemini-ocr` binary on PATH
  (`GeminiEngine.is_available`, `src/socr/engines/gemini.py`).
- `EngineType.GEMINI` in `enabled_engines` (default: all engines).
- `PROFILE_GEMINI` is `auto_eligible` and `supports_per_page`, so it is a rung of the cost-sorted
  ladder (`provider_ladder`, `src/socr/core/providers.py`): qwen local (free) first, gemini after.
  `_phase_agentic` escalates up the ladder when the judge rejects a rung.
- Not reachable under `--strict-local` or `--max-cost-per-page 0`; `--max-cost-per-page` below
  the new price now also prices Gemini out (it used to fit under $0.001).
- Model override per run: `--gemini-model`.

## Open consequence of the new price

The ladder is sorted by cost. At $0.0207 Gemini is no longer cheaper than Mistral ($0.001) in any
ladder that includes both; Mistral is `auto_eligible=False`, so the default ladder is unaffected.
Twelve existing tests pinned the old 0.0002 price or the GLM, GEMINI, MISTRAL order; they now use
`GEMINI_COST_PER_PAGE_USD` and the order GLM, MISTRAL, GEMINI (stand-in engines swapped where a test
needed a cheap paid rung). Full suite green (10761 passed). The coordinator should confirm that
Gemini sitting above Mistral in an include_ineligible ladder is intended.
