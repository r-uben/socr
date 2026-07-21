# OCR-04 bake-off — Unlimited-OCR (Baidu)

**Date:** 2026-07-22  
**Candidate:** `baidu/Unlimited-OCR` (HF; MIT; image-text-to-text)  
**Gate:** `ai-skills/ocr` Bake-off gate (smoke → CE digit exactness vs `socr process --strict-local`)  
**Verdict: FAIL at smoke (CUDA-only) — no adapter, no default change.**

## Environment

| Item | Value |
|---|---|
| Host | macOS (Apple Silicon) |
| torch | 2.12.0 — **CUDA false**, **MPS true** |
| HF cache | No prior `Unlimited-OCR` weights |
| Fixtures ready | `tools/socr/tests/fixtures/table_repair/ce_like_p4.pdf`, `ce_like_p4_dense.pdf` + GT JSON |
| Baseline engine | Ollama `qwen3-vl:30b-a3b-instruct` present |

## Step 1 — Smoke

Upstream README (HF `baidu/Unlimited-OCR`):

> Inference using Huggingface transformers on **NVIDIA GPUs**. Requirements tested on python 3.12.3 + **CUDA12.9**.

Also ships CUDA-specific Docker images; no Mac/MPS path documented.

**Result:** Smoke **blocked** on this Mac. Per skill gate: “Smoke … on Mac **(or document CUDA-only)**.” Documented: **CUDA-only**.

Did **not** download multi-GB weights solely to confirm CUDA import failure.

## Step 2 — CE digit exactness

**Skipped.** Candidate cannot run locally on the evaluation host. No cell-match rate vs GT; viral bench scores do not count.

When CUDA (HPC) is available, resume with:

```bash
# Baseline
socr process ce_like_p4_dense.pdf --strict-local \
  --qwen-model qwen3-vl:30b-a3b-instruct \
  -o /tmp/ocr04-baseline --reprocess --judge-backend heuristic

# Candidate: HF transformers path from Unlimited-OCR README on same PDF page images
# then score cells against ground_truth_dense.json (same harness as test_table_repair_parity)
```

## Pass bar

| Criterion | Status |
|---|---|
| Smoke on Mac **or** documented CUDA-only | CUDA-only documented |
| ≥ parity on CE digits vs strict-local qwen | **Not measured** |
| Clear win on qwen-failure class without born-digital regression | **Not measured** |

**Overall: FAIL / incomplete** for integration. Skill already forbids Unlimited-OCR as default until pass — unchanged.

## Follow-ups

| Ticket | Action |
|---|---|
| OCR-04 | Closed as fail-at-smoke on Mac; reopen bake-off on CUDA host to finish CE cells |
| OCR-05 | Still blocked (depends on OCR-04 pass) |
| OCR-06 | Remains DEFERRED |

## Non-goals observed

No registry entry, no `--primary` adapter, no skill default change.
