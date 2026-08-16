# 2026-08-16 — Free local OCR, verified end to end on Bocconi HPC

**Question asked:** can socr OCR PDFs using only local models, for free, on either the
Mac or the Bocconi cluster — and with which models?

**Answer: yes, on both.** Measured, not assumed. Details below.

---

## Headline results

| Run | Where | Model | Pages | Wall clock | Cost | Contaminated pages |
|-----|-------|-------|------:|-----------:|-----:|-------------------:|
| 628531 | H100 NVL 95 GB | `Qwen/Qwen3.5-35B-A3B-FP8` (vLLM) | 45 | 12:04 | $0.0000 | 8 |
| 628549 | same | same, patched client | 45 | 9:50 | $0.0000 | **0** |
| local | M3 Max 64 GB | `qwen3-vl:30b-a3b-instruct` (Ollama) | 1 (dense fixture) | ~2 min | $0.0000 | 0 — **44/44 cells correct** |

Document: `BHL_2026.pdf` (Bügel, Hidalgo & Luetticke 2026), 45 pages, dense regression
tables. Output at `/scratch/3179349/ocr_test_qwen35_fixed/BHL_2026/`.

---

## What was wrong, and what fixed it

### 1. Hybrid-thinking models corrupt the transcription (qwen-ocr #4)

Qwen3.5 and Qwen3.6 are hybrid-thinking models with thinking **on** by default. On 8 of
45 pages the model emitted its own reasoning instead of the page:

```
## Page 10

The user wants me to transcribe the provided image into Markdown.

**1. Analyze the Image:**
*   **Title:** "Table 2: Serial correlation and predictability of the shock series"
```

socr wrote that into the document as if it were the paper's text. For a citation corpus
this is content corruption — nothing downstream can distinguish commentary from content.

**Fix:** `chat_template_kwargs: {"enable_thinking": false}` (vLLM/OpenRouter),
`think: false` (Ollama), plus DashScope's top-level `enable_thinking`. Now the default in
qwen-ocr, with `--thinking` as the escape hatch. Verified live before and after.

**Verification that it was commentary, not content loss:** the output shrank 126 KB → 76 KB,
and the 50 KB delta is confined *exactly* to the 8 contaminated pages (10, 11, 18, 21, 36,
37, 40, 41). The other 37 pages differ by <400 chars each. Page 10 lost 20,253 chars and
still carries Table 2 in full — 32 rows, coefficients and standard errors in the right
cells, negative signs present.

### 2. Input resolution was never set (qwen-ocr #5) — **relevant to #213/#217**

qwen-ocr rendered at 300 DPI (an A4 page ≈ 8.7 MP) and let the server-side vision
processor downscale by an *unstated default*. The effective resolution was neither chosen
nor logged.

Qwen's own OCR cookbook ([QwenLM/Qwen3-VL `cookbooks/ocr.ipynb`](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/ocr.ipynb))
sets it explicitly:

```python
min_pixels = 512*32*32    #   524,288 px
max_pixels = 2048*32*32   # 2,097,152 px  (~2.1 MP)
```

(Qwen3-VL aligns to multiples of **32**; Qwen2.5-VL used 28.)

**Fix:** enforced client-side via a reimplementation of the cookbook's `smart_resize`,
because vLLM's OpenAI server does not accept per-request `min_pixels`/`max_pixels` in chat
content parts. Resizing before send makes the budget bind identically across all three
backends *and* makes the logged number the number the model actually saw. Measured:
`2480x3509 (8.70 MP) -> 1216x1696 (2.06 MP)`.

> **Hypothesis, and its refutation.** The initial thought was that small glyphs — digits,
> minus signs, sub/superscripts — die in an unmanaged downscale, and so the unmanaged budget
> might explain #217. **This is wrong for #217, and the log should not be read as supporting
> it.** The three-model panel (`docs/log/2026-08-16_gh217-minus-sign-panel.md`) confirmed
> #217 at the *font* layer: the PDF embeds subsetted pi fonts with no `/ToUnicode` and an
> empty `/Differences`, so code 50 (`/H11002`, minus) is emitted by PyMuPDF as ASCII `'2'`.
> That corruption happens in the **native text layer**, before any image is rendered, and no
> resolution setting can touch it.
>
> Where the budget *could* still matter is the VLM-read path — table structure damage
> (#213/#215-class), where a model reads a rendered crop rather than the text layer. That is
> untested and is a separate experiment from #217.
>
> A second-order use does follow from #217, though: a VLM read at a *known* resolution is an
> independent witness to a text layer that is known to lie. TR-3 currently checks emitted
> numbers against `page.get_text("words")` — the same corrupt string — so it cannot catch
> this class by construction.

---

## Model findings

- **`Qwen/Qwen3.5-35B-A3B-FP8`** — works on one H100 with thinking off. MoE, ~3B active
  params. Chosen for speed; note socOCRbench scores the *dense* 27B higher (~0.54 vs
  ~0.48 for the flash tier), so this is a deliberate speed-over-accuracy trade.
- **`qwen3-vl:30b-a3b-instruct`** (Ollama, Mac) — still the safest local option. No
  thinking leak, 44/44 cells on `ce_like_p4_dense.pdf`, ~138 s/page.
- **`qwen3.6:35b-a3b-nvfp4`** (Ollama, Mac) — **broken for OCR.** Answers text prompts
  normally but returns an *empty HTTP body* for any image request on Ollama 0.32.7. Not a
  thinking problem; the vision path itself fails. Do not use.
- socOCRbench's top 10 are all proprietary APIs. Best self-hostable entries: Qwen3.5 122B
  (0.575), Qwen3-VL 235B (0.548), Qwen3.5 27B (0.542).

---

## Cluster facts that contradict the `/hpc-bocconi` skill notes

The skill file is stale in three ways that cost time today:

1. **GPU partitions.** Real list: `gpua100` (MIG slices `4g.40gb`/`3g.40gb` — *not* full
   80 GB A100s), `gpunew` (H100 ×2), **`gpuh200` (H200 ×2)** — the H200 partitions are
   undocumented in the skill. Each has `debug_`/`short_`/`medium_`/`long_` variants.
2. **`--hpc-sequential` is DeepSeek-only.** `HPCPipeline` hardwires
   `DeepSeekVLLMEngine` for OCR and Nougat for math. It never touches Qwen. socr's own
   `docs/MODELS.md` scores DeepSeek-OCR at 0.085 ("dead weight"). The Qwen path on HPC is
   the *normal* pipeline with `--qwen-backend vllm`, not `--hpc-sequential`.
3. **Home quota is nearly full** — 175 GB of a 180 GB quota (200 GB hard limit). Model
   weights must go to `/scratch/$USER` (BeeGFS, 245 T, 93 T free). `HF_HOME` should point
   there.

Biggest home consumers if space is ever needed: `hf_cache` 32 G, `job_market_paper` 23 G,
`hf_local` 16 G, `socr-fiscal` 14 G, `greybark-colombia` 14 G, `tap` 12 G.

---

## Reusable assets left on the cluster

- `/scratch/3179349/jobs/dl_qwen35.sh` — CPU-only HF download job (no GPU, ~5 min).
- `/scratch/3179349/jobs/serve_and_test.sh` — serves vLLM on an H100, waits for health,
  runs a raw-vision sanity check, then socr end to end. `serve_and_test_fixed.sh` is the
  same against the patched client.
- `/scratch/3179349/src/qwen-ocr-cli/` — synced source; `qwen-ocr` installed from it.
- `/scratch/3179349/hf_cache/` — `Qwen3.5-35B-A3B-FP8` (35 G) and
  `Qwen3-VL-30B-A3B-Instruct` (58 G, pre-existing).

First vLLM start took 13:35 (torch.compile); the second took 5:20 from the compile cache.

---

## Issues filed on `r-uben/qwen-ocr-cli`

| # | Title | Status |
|---|-------|--------|
| [2](https://github.com/r-uben/qwen-ocr-cli/issues/2) | vLLM default model two generations stale (`Qwen3-VL-7B-Instruct`) | open |
| [3](https://github.com/r-uben/qwen-ocr-cli/issues/3) | Ollama default tag unpinned — can resolve to a thinking build that hangs | open |
| [4](https://github.com/r-uben/qwen-ocr-cli/issues/4) | No way to disable thinking | **fixed**, `ff7d817` |
| [5](https://github.com/r-uben/qwen-ocr-cli/issues/5) | Input resolution never set explicitly | **fixed**, `44e1560` |

---

## Process notes / caveats

- **The fix reached `origin/main` unreviewed.** The delegated agent was told to commit
  locally and not push; `origin/main` is now `44e1560`, fast-forward merged. The code is
  verified working on real hardware (both backends), but it did not get a diff review.
  Retroactive review is still owed.
- `#5`'s commit also bundles a `CLAUDE.md` change, including a real build-doc fix:
  `uv sync` alone does not install pytest/ruff (they live in the `dev` extra), so the
  documented `uv sync && uv run pytest` was silently erroring on collection. Now
  `uv sync --extra dev`.
- The `max_image_side` / 4000 px cap was **removed** — subsumed by the pixel budget, and
  dead code at 2.1 MP.
- `strip_thinking` (removes matched `<think>…</think>` blocks) was added as a backstop
  beyond the issue's scope, for servers that ignore the switch.

## Is socr production-ready?

Not yet, and today's work is not why.

- **Prose-heavy documents:** yes, today, on either machine, for free.
- **Table-heavy documents, unattended:** no. #213/#217 remain open, the hand-judged sample
  found 4 of 5 gate-flagged tables genuinely damaged, and this single 45-page run raised
  9 dual-pass disagreements plus 7 native-table-verifier warnings. The pipeline is telling
  you it is unsure about its tables, and the failure mode is silent number damage.

**Next action:** the resolution hypothesis does **not** apply to #217 (font-layer defect,
see above). If it is tested at all, the target is the VLM-read path — table structure damage
of the #213/#215 class — not the minus sign.
