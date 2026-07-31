# 2026-07-31 — Local VLM evaluation: qwen3.5:27b rejected, two hazards found

**Decision: keep `qwen3-vl:30b-a3b-instruct`.** A candidate replacement that an
external benchmark ranked well above it produced byte-identical output on our own
fixtures at up to 5.9× the cost.

Recorded as a negative result so nobody re-runs it.

## Why we looked

[socOCRbench](https://noahdasanaike.github.io/posts/sococrbench.html) (Dasanaike,
last updated 2026-07-31) benchmarks OCR models on social-science documents and scores
several models socr ships. On its numbers:

| Model | Where we use it | Overall | TEDS (tables) |
|---|---|---|---|
| Qwen3 VL 30B | our local default | 0.4261 | 0.3253 |
| **Qwen3.5 27B** | candidate | **0.5417** | **0.5242** |
| GLM-OCR | `engines/glm.py` | 0.3679 | 0.4085 |
| DeepSeek-OCR | provider profile | 0.0855 | 0.0510 |

Constraint: local-only, MacBook Pro M3 Max / 64 GB. That rules out every model above
~0.55 (all cloud). `qwen3.5:27b` was the only local candidate promising a real gain,
and its profile — slow, better on tables — suited a GH-96 escalation tier rather than
a default.

## Method

Production path, one variable changed. `locate_tables` → `TableCropExtractor` →
`OllamaTableReader` with the production prompt and crop DPI; only the model name
differs between arms. `num_ctx` pinned to 16384 and `think=false` on **both** arms.

Fixtures: `tests/fixtures/table_repair/ce_like_p4.pdf` and `ce_like_p4_dense.pdf`,
scored against their committed ground-truth JSON. Metric is positional cell accuracy
plus whole-row exactness — cells compared by position, not by regex-extracting
numbers, so a non-numeric `na` still anchors the alignment.

Harness in scratch (not committed); it is a few dozen lines and cheaper to rewrite
than to maintain.

## Result

| Model | Case | Cell acc | Row exact | Cold s | Gen tok/s |
|---|---|---|---|---|---|
| `qwen3-vl:30b-a3b-instruct` | clean | 60.6% | 33.3% | 100.8 | 50.2 |
| `qwen3-vl:30b-a3b-instruct` | dense | 100.0% | 100.0% | 28.0 | 50.2 |
| `qwen3.5:27b-q4_K_M` | clean | 60.6% | 33.3% | 145.8 | 11.8 |
| `qwen3.5:27b-q4_K_M` | dense | 100.0% | 100.0% | 148.9 | 11.8 |

The clean-fixture outputs are **byte-identical**, not merely similar.

Timings are cold-run. Repeat runs are contaminated by KV-cache reuse (one repeat
returned in 6.1 s vs 100.8 s with identical scores) — do not read the medians.

## How to read the tie

Identical scores mean **the fixtures cannot discriminate these models**, not that the
models are equivalent. Both arms are blocked:

- **dense** — baseline already scores 44/44. Zero headroom. The fixture was built to
  catch the deterministic rowizer's name↔value interleaving, never the VLM's reading.
- **clean** — the 60.6% ceiling is the harness. The 13 "missed" cells are 4 `na` cells
  (both models correctly render them blank; the ground-truth comparison counts that as
  a miss) and the 9 cells of the second table (the production prompt transcribes one
  table per crop; the full-page fallback crop contains two).

So the claimed +0.19 TEDS advantage is **untested here, not refuted**. socOCRbench is
built on handwritten and degraded historical census forms; socr's corpus is printed
academic tables. The ranking did not transfer. Testing it properly needs genuinely
degraded pages with headroom — which these synthetic, license-clean fixtures
deliberately are not.

## Hazard 1 — `-mlx` builds can emit structurally-valid, content-free tables

`qwen3.5:27b-mlx`, same prompt and same crop as the GGUF above, returned:

```
| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
```

Correct pipe structure, every cell empty. The GGUF build of the same model produced
the full, correct 11×4 table. This is worse than a crash: GH-96's trust index would
accept this as a well-formed table, and any shape-only validator passes it clean. It
is exactly the "no silent content loss" red line in `CLAUDE.md`.

My first read blamed the model; that was wrong, and the GGUF re-test is what caught
it. Benchmarking a quant is not benchmarking a model.

## Hazard 2 — thinking-tagged models run away and emit nothing

| Build | `think` | Result |
|---|---|---|
| `27b-mlx` | true | 2048 tokens generated, **0 characters**, 225.8 s |
| `27b-q4_K_M` | true | 2048 tokens generated, **0 characters**, 320.2 s |
| `27b-q4_K_M` | false | 325 tokens, full correct table, 42.3 s |

`CLAUDE.md` already bans `qwen3-vl:30b` for exactly this. `qwen3.5` carries ollama's
`thinking` tag and fails identically on both builds. The ban should be generalised to
the tag, not maintained one model at a time.

## Discarded hypothesis

Both models load with a declared 262144-token context, and the first timeout looked
like KV-cache pressure on 64 GB. Capping `num_ctx` to 16384 changed **nothing** — the
run still hit the 600 s timeout. The cause was dense-model throughput plus the
thinking runaway. Recorded because it is a plausible wrong turn worth not repeating.

## Also worth taking from the same source

Independent of model choice, Dasanaike's write-up on digitizing 20M historical
documents reports two things we do not do:

- **Inference-parameter tuning** — image resolution, pre-call resize, max output
  tokens, repetition penalty. Reported ~0.10 validation swings from these alone,
  comparable to an entire model upgrade. We have never grid-searched them.
- **A degeneracy taxonomy in post-processing** — classify every output as fully
  valid / partially degenerate (trailing repetition) / fully degenerate, then
  truncate-and-dedupe the middle class and drop the last. Maps onto our page-status
  model, and would have caught Hazard 1.

## Cleanup

`qwen3.5:27b-q4_K_M` and `qwen3.5:27b-mlx` removed (36 GB reclaimed).
`qwen3-vl:30b-a3b-instruct` unchanged.
