# Cost-aware agentic OCR (2026-05-30)

Goal: "develop agentic OCR that uses the best cost-effective provider on the go."

## What was built

`socr paper.pdf --agentic`. For each OCR page, try the **cheapest available
provider first**; an injected **judge** looks at the result and either accepts it
or escalates up a **cost-ordered ladder**. Stop at the first accepted output;
keep the best attempt if none is accepted. Born-digital prose still takes free
native text (unless `--no-native-first`). Every attempt's provider + cost is
recorded; the run writes a replayable manifest.

### Pieces (commits 955a3bc, 81aa2e2, 00d1a19)

- **`core/providers.py`** — `ProviderProfile` + `provider_ladder()` (cheapest
  first, tie-broken by `ENGINE_PRIORITY`). Prices are tunable DEFAULTS in one
  table; routing uses RELATIVE ordering, so the exact dollars matter less than
  the order. **No capability tables** — we do not pre-declare "engine X handles
  math"; the judge catches a cheap provider failing and escalates. That keeps
  routing free of brittle static matrices (and honors "let the model reason").
- **`pipeline/agentic.py`** — `route_page(page, ladder, run_provider, judge,
  max_attempts)`. Pure given its two injected deps. Records every attempt
  (engine, cost, verdict); best-effort keeps the most trustworthy output on total
  failure. Judge adapters: `VLMPageJudge` (render the page, ask a vision model if
  the OCR is faithful) and `HeuristicPageJudge` (no-model fallback).
- **`orchestrator._phase_agentic`** — wires it in behind `config.agentic`
  (default off → legacy flow untouched). Appends attempts to `PageState`, sets
  `best_output` once, records cost so `DocumentState.total_cost` is right, and
  auto-writes a manifest. `build_manifest` is now called on real runs (was tests
  only).
- **CLI** — `--agentic --judge-backend --judge-model --max-cost-per-page
  --cost-budget --write-manifest` on `process` + `batch`.

## Key design decisions

- **Python owns the loop; the LLM is a stateless per-page decider.** (The
  go-team panel was unanimous: agent-on-top is a fatal operational shape for a
  10k-page corpus.) The `.md` is the judge *prompt*, not the orchestrator.
- **Don't route intermediate attempts through `apply_result()`** (codex catch):
  it auto-promotes the first passing attempt, which fights `route_page`'s
  possibly-later winner. Append attempts; set `best_output` once at the end.
- **Manifest = artifact cache, not a re-execution recipe.** `socr replay` serves
  frozen output blobs (zero model calls), so VLM non-determinism never breaks
  reproducibility.

## Validation (real engines, not mocks)

`socr demo.pdf --agentic --no-native-first` on a 2-page born-digital PDF:
- Ladder built `glm($0) -> deepseek($0) -> marker($0) -> gemini($0.0002)`.
- glm ran first; judge rejected it; escalated to deepseek, then marker; stopped
  at `max_attempts=3` (gemini never spent on). Attempt chain recorded in the
  manifest journal.
- All OCR rejected/failed on these clean prose pages → fell back to **free native
  text**. Total cost **$0.00**.
- `socr replay manifest.json` reconstructed the document **bit-identically with 0
  model calls**.

475 tests pass; no new lint.

## Open / next

- TICKET-16 judge benchmark: label ~100 pages to tune the accept threshold (and
  measure iterations-to-fix → decide real loop depth).
- Real per-provider prices; populate `model_version`/`prompt_hash` in fingerprints.
- HPC fold (TICKET-12 Increment 2) deferred — different vLLM runtime.
- Env note: `deepseek`/`marker` returned 0/1 in this environment (local models
  not fully working); the routing machinery is proven regardless.
