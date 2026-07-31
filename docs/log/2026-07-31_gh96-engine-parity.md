# 2026-07-31 — GH-96 engine parity: the escalation engine is not the oracle

The bake-off measured escalation using **agy** (Antigravity CLI). The lane will call
**socr's own Gemini engine** (`gemini-ocr`, `PROFILE_GEMINI`, `gemini-3-flash-preview`).
Different CLI, different prompt, different post-processing, possibly a different
model. So "a second vision pass recovers these tables" was established; "socr's
Gemini engine recovers these tables" was not.

This note closes that gap, and the answer changes the accept rule.

## Artifacts

`~/data/fiscal-ballast/_experiments/2026-07-31_gh96-engine-parity/`

Preserved out of `/tmp` deliberately: neither agy nor `gemini-ocr` is deterministic,
and agy's model cannot be pinned (`agy-set-model` is a dead symlink), so these exact
outputs **cannot be regenerated**. Without them the numbers below are unverifiable
claims. Contains the 19 rendered page PNGs, agy's raw and table-extracted output,
and `gemini-ocr`'s output per page.

## Result

Scored with `benchmark/table_exactness.score_page` against each page's native layer.
16 table pages of OBR EFO November 2022, 2359 ground-truth cells.

| page | socr (qwen local) | agy | gemini-ocr (flash) |
|-----:|------------------:|----:|-------------------:|
| 13 | 38.4 | 100.0 | 100.0 |
| 39 | 86.5 | 86.5 | **79.7** |
| 45 | 100.0 | 100.0 | 100.0 |
| 46 | 0.0 | 100.0 | 100.0 |
| 48 | 0.0 | 100.0 | 85.1 |
| 51 | 11.1 | 100.0 | 93.9 |
| 53 | 0.0 | 0.0 | 0.0 |
| 55 | 0.0 | 50.0 | 50.0 |
| 59 | 100.0 | 100.0 | **79.3** |
| 60 | 100.0 | 100.0 | **75.9** |
| 61 | 82.1 | 82.1 | **50.0** |
| 62 | 7.7 | 32.1 | 50.0 |
| 63 | 54.8 | 83.9 | 83.9 |
| 64 | 0.0 | 100.0 | 88.0 |
| 65 | 0.0 | 95.2 | 85.6 |
| 67 | 94.1 | 94.1 | 94.1 |

**Aggregate: socr 45.0%, agy 86.9%, gemini-ocr 78.2%.**

## Two findings, both of which change the design

### 1. "Escalation is never worse" is false for the engine we will actually use

That property was measured on agy and generalised. `gemini-ocr` is **worse than the
incumbent on four pages** (39, 59, 60, 61), and on two of them socr was **perfect**.
Page 61 regresses 82.1% → 50.0%.

The trigger saves 59 and 60 — they score 100%, so "incumbent disagrees with native"
never fires. It does **not** save 39 and 61.

### 2. The canary cannot catch those regressions, because they are not fabrications

Every number in the regressed output is present in the native layer. Token
containment is satisfied. The canary accepts them.

## Consequence: the accept rule changes

Exactness is computable **at runtime, model-free, at zero cost** on a born-digital
page — the same property that made it the best trigger. So it can also be the gate:

> **Accept iff `exactness(candidate) > exactness(incumbent)`.**

Measured over the same 16 pages, escalating the 13 pages the trigger fires on:

| accept rule | aggregate |
|---|---|
| socr baseline, no escalation | 45.0% |
| canary-only (accept unless fabricated) | 81.7% |
| **accept iff exactness improves** | **85.0%** |

Better by 3.3 points, and — the part that matters more — **monotone by
construction**: no page can leave worse than it arrived. The canary cannot offer
that, because a regression is not an invention.

It also subsumes the canary's own job: a fabrication scores ~0 and is rejected, a
truncated candidate scores low and is rejected.

**The canary is therefore demoted, not deleted.** It remains the gate for pages where
the ground truth will not parse and exactness is uncomputable, and it is cheap
defence-in-depth. Its docstring should say so rather than read as authoritative.

## A third instance of one bug

The first parity run scored `gemini-ocr` at **60.9%** on page 13 while agy scored
100%, which would have argued for pinning Pro, switching engines, or wrapping agy.
It was an artifact: `gemini-ocr` emits section-total rows in **bold**, and
`BenchmarkScorer._is_numeric_cell` anchors its regex without stripping markdown, so
every parent row was discarded from scoring. Corrected score: **100.0%**, identical
to agy.

That is the same defect in a third tokenizer:

1. `source_evidence.collect_table_tokens` (GH-103) — the evidence gate passed vacuously
2. `native_rows._is_value` — parent rows dropped from ground truth and from scoring
3. `table_exactness._split_label_and_values` — bold values not recognised

**Rule worth generalising: anywhere this codebase applies an anchored numeric regex
to *markdown* rather than to PDF words, emphasis must be stripped first.** PDF words
never carry markdown; engine output routinely does.

## Open

- `gemini-ocr` accepts `--model gemini-3.1-pro`. The 78.2% → 86.9% gap to agy is
  plausibly closable at higher cost per page. **Untested.**
- `mistral-ocr` (`$0.001/page`, `auto_eligible=False`) is available and unmeasured.
- One document, no negative controls. Treat orderings as robust and absolute values
  as provisional.
