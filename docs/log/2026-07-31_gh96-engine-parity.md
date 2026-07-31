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

---

# Addendum — Pro does not close the gap, and a cloud CLI hung for 97 minutes

## `gemini-3.1-pro-preview` is *worse* than flash on this task

The open question above was whether pinning Pro closes the 78.2% → 86.9% gap to agy.
It does not. Over the 7 table pages that completed before the run wedged:

| page | socr | flash | **Pro** | agy |
|-----:|-----:|------:|--------:|----:|
| 13 | 38.4 | 100.0 | 100.0 | 100.0 |
| 39 | 86.5 | 79.7 | 86.5 | 86.5 |
| 45 | 100.0 | 100.0 | **79.6** | 100.0 |
| 46 | 0.0 | 100.0 | **25.3** | 100.0 |
| 48 | 0.0 | 85.1 | **25.7** | 100.0 |
| 51 | 11.1 | 93.9 | 100.0 | 100.0 |
| 53 | 0.0 | 0.0 | 0.0 | 0.0 |

Partial aggregate, 561 cells: **socr 31.6%, flash 85.0%, Pro 67.4%, agy 88.9%.**

**Decision: pin flash.** Better, cheaper, ~3× faster (11s vs 34s per page), and it
did not hang. "Bigger model, better OCR" is false here.

Sample caveat: 7 of 16 pages, so 67.4% should not be quoted as Pro's score. The
safe claim is the ordering — Pro does not close the gap, and is behind flash on
pages flash handles well.

Note the CLI's own `--help` advertises `--model gemini-3.1-pro`, which returns
`NOT_FOUND`. The working identifier is `gemini-3.1-pro-preview`.

## A cloud OCR CLI hung indefinitely — trap #4, evidenced

The Pro batch stopped after 10 of 19 pages and sat wedged mid-request on a single
page for **97 minutes** with the process still alive, until killed manually. No
timeout fired, because there is none on that path.

This was already listed as a design risk for the lane. It is now a reproduction:

- In the page-major agentic loop, escalation runs inline. A wedged cloud CLI stalls
  the **entire document**, not just the page.
- The existing cascade-halt cannot help: it triggers on `probe_ollama_idle()`, which
  says nothing about a cloud subprocess.
- So the lane needs its own wall-clock timeout and a document-level
  `_escalation_degraded` latch that disables escalation after the first hang —
  and must **not** set `backend_degraded`, which would emit a false
  `PARTIAL_SAVE_VLM_TIMEOUT` blaming a local GPU that was never involved.

## Consequence for the accept rule, now implemented

`tables/escalation_decision.py`:

    accept iff exactness(candidate) > exactness(incumbent)

Applied to the full 16-page set with flash: **45.0% → 85.0%**, with all four
regressions (39, 59, 60, 61) blocked and every page decided on measurement rather
than deferred to the canary.

One refinement found while validating: gate on whether the *ground truth* is usable,
not on the report's `scorable` flag. That flag is set when no prediction label
matched — evidence the prediction is bad, not that the measurement is invalid.
Gating on it handed the incumbent's worst failures to the weaker canary, including
two real 0% → ~86% recoveries.

---

# Correction (2026-08-01) — the engines are tied; this note's ranking was an artifact

**Everything above that compares `gemini-ocr` against agy is wrong.** The gap was
never real. It was produced by the metric penalising one engine for its footnote
syntax.

## What happened

The metric compares row labels by a normalized key, and every producer spells
footnote markers differently:

| spelling | producer |
|---|---|
| `Underlying differences1` | the PDF's native text layer |
| `Underlying differences$^1$` | LaTeX — socr's local qwen path |
| `Nominal GDP<sup>1</sup>` | HTML — agy |
| `Nominal GDP (£ billion)1,2` | multi-note marker, native |

Only the bare-digit form was folded. So identical rows differed by a single
character, never matched, and were scored as dropped. Each engine was penalised in
proportion to how often it used a spelling the normalizer did not know.

## Corrected aggregates, same 2359 cells

| engine | this note said | actual |
|---|---:|---:|
| socr local qwen | 45.0% | **47.4%** |
| agy | 86.9% | **94.1%** |
| gemini-ocr | 78.2% | **94.1%** |

**agy and `gemini-ocr` are tied.** The 8.7-point gap this note reports, and the
39-point gap on page 13 it reports as an earlier artifact, were both the same bug
at different magnitudes.

## What this retires

- The claim that `gemini-ocr` trails the oracle. It does not.
- The open question of whether agy should be wrapped as a socr engine. The engine
  already in the pipeline — pinnable, reproducible, costed — measures identically.
- The Pro-vs-flash comparison in the addendum above is also suspect for the same
  reason; it was scored with the biased metric. Flash remains the choice on cost,
  speed and the observed hang, but the quality ordering in that table is not
  evidence.

## What it does not retire

The socr-vs-cloud gap. Local 47.4% against cloud 94.1% is larger than this note
originally reported, so the case for escalation is stronger, not weaker.

## How it was found

By opening page 48 beside the **rendered PDF** and noticing the output looked
correct while the metric said 85.1%. It was 100%. Every earlier check had compared
the metric against other numbers rather than against the document — which is why
four successive measurement bugs survived until someone looked at the page.

Fixed in `tables/native_rows.normalize_label` with one generic rule covering HTML,
LaTeX, unicode and bare-digit markers.
