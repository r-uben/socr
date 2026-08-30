# socr

[![PyPI](https://img.shields.io/pypi/v/socr)](https://pypi.org/project/socr/)
[![Python 3.11–3.12](https://img.shields.io/pypi/pyversions/socr)](https://pypi.org/project/socr/)
[![License](https://img.shields.io/github/license/r-uben/socr)](LICENSE)

Multi-engine document OCR with cascading fallback and quality audit.

`socr` orchestrates multiple OCR engines — calling each as a CLI subprocess, auditing output quality, and falling back to a different engine when results are poor. Each engine is a standalone CLI tool (`gemini-ocr`, `deepseek-ocr`, `marker-ocr`, etc.) that can also be used independently.

## Install

```bash
pip install socr

# With specific engine backends
pip install socr[gemini]          # Google Gemini (cloud)
pip install socr[local]           # DeepSeek + Nougat (local/free)
pip install socr[all]             # All engines
```

Engines are installed separately because they have different dependencies (torch, cloud SDKs, etc.). Install only what you need.

## Usage

```bash
# Process a PDF (deterministic mode)
socr paper.pdf

# Cost-aware agentic mode: processes + saves one page at a time (pages/NNN.md),
# resumable on re-run, byte-identical final output.
socr paper.pdf --agentic
socr paper.pdf --agentic --strict-local           # local-only (free), page-by-page
socr paper.pdf --agentic --cost-budget 0.05       # cap spend per document
# Interrupted a run? Just run it again — finished pages are skipped:
socr paper.pdf --agentic                          # resumes from the last saved page

# Choose engine (deterministic mode)
socr paper.pdf --primary gemini
socr paper.pdf --save-figures

# Batch process a directory
socr batch ~/Papers/ -o ./results/
socr batch ~/Papers/ --dry-run        # preview what would be processed

# Reproducibly rebuild a document from a manifest (no model calls)
socr replay output/paper/manifest.json -o paper.md

# Check which engines are available
socr engines
```

## How it works

socr routes **each page** to an OCR engine, checks the result, and re-tries on a
different engine when the result is poor. It runs in two modes that differ in how
the engine for a page is chosen.

### Deterministic mode (default)

```
PDF → classify each page → easy: local engine · hard: primary engine
    → heuristic audit → fallback on failed pages → Markdown
```

The engine is chosen **up front** from predicted page difficulty (tables,
equations, layout). Born-digital prose uses native text for free. Quality is
checked by heuristics; failed pages fall back to another engine.

### Agentic, cost-aware mode (`--agentic`)

```
PDF → for each page, in order:
        route → extract text → reconcile tables → place figures/charts → (equations)
        → verify → FLUSH pages/NNN.md to disk → next page
    → stitch fragments → Markdown (byte-identical to whole-doc assembly) + manifest
```

The engine is chosen **dynamically by cost**: try the cheapest available provider
first, let a judge (a vision model that looks at the page, or a heuristic
fallback) decide accept-or-escalate, and climb the cost ladder
(`local → cheap cloud → premium cloud`) only when the cheaper output is rejected.
Stops at the first accepted output, bounded by `--cost-budget` / `--max-cost-per-page`.
Each run records the winning provider + cost per page and writes a **manifest**
that `socr replay` can reconstruct with zero model calls.

**Progressive, page-by-page processing.** In agentic mode socr is *page-major*: it
finishes one page completely and **writes it to disk immediately** (`pages/NNN.md`
+ a `pages/NNN.json` status sidecar) before starting the next, rather than holding
the whole document in memory and saving once at the end. Concretely:

- **Crash-safe.** A hang or crash on page 30 leaves pages 1–29 already on disk. The
  final `<stem>.md` is the ordered stitch of the fragments and is **byte-identical**
  to the non-progressive whole-doc assembly.
- **Resume.** Re-running a partially-processed document **skips pages already finished**
  (matched by a per-page run-fingerprint + input checksum) and reprocesses only the
  rest. A model/prompt/flag change invalidates the fingerprint and forces re-OCR; the
  skip is conservative — on any doubt the page is reprocessed, never silently reused.
- **Clean halt on a wedged model.** If the local VLM stops responding, socr flushes
  the finished pages, marks the document `PARTIAL_SAVE_VLM_TIMEOUT`, and stops cleanly
  instead of feeding more work into a stuck GPU.
- **Tables / figures / charts, per page.** Born-digital table geometry is verified
  before paying for a VLM judge; figures are embedded **inline** within their page;
  chart/front-matter pages (vector charts that would otherwise become text word-salad)
  are saved as **image assets** with a note rather than transcribed.
- **Equations (opt-in).** `--detect-equations` saves model-free crop PNGs of display
  equations; `--recover-clean-equations` additionally reads each crop to LaTeX into a
  non-destructive sidecar (validated; bad LaTeX never replaces the native text/crop).

Each engine is a separate CLI binary. `socr` calls it as a subprocess, reads the
output markdown, and applies the quality pipeline. See `docs/ARCHITECTURE.md` for
the full design.

## Engines

Routing is **local-first → Ollama Cloud → paid cloud edge case**. See
`docs/MODELS.md` for the full per-sub-task policy and the measured data behind it.

| Engine | Package | Type | Routing role |
|--------|---------|------|------|
| Qwen | `qwen-ocr-cli` | Ollama Cloud / local | **Workhorse VLM** (`qwen3.5:cloud`, no extra key) |
| Gemini | `gemini-ocr-cli` | Cloud | Edge-case escalation, ~$0.0002/page |
| Marker | `marker-ocr-cli` | Local | Layout-aware fallback (Surya + Texify) |
| GLM | `glm-ocr-cli` | Local | Fast local emergency fallback |
| Nougat | `nougat-ocr-cli` | Local | Academic papers, Python <3.13 |
| Mistral | `mistral-ocr-cli` | Cloud | Manual only (`--primary mistral`); dominated by Gemini |
| DeepSeek | `deepseek-ocr-cli` | Local | Manual only (`--primary deepseek`); low quality |

Check availability:
```
$ socr engines

  [+] gemini       cloud, ~$0.0002/page
  [+] marker       local, layout-aware (Surya + Texify)
  [+] mistral      cloud, ~$0.001/page
  [+] deepseek     local via Ollama
  [x] nougat       local, academic papers
```

## CLI reference

```
socr process <PDF> [OPTIONS]
  -o, --output-dir PATH       Output directory
  --primary ENGINE             Primary OCR engine (gemini, marker, deepseek, etc.)
  --fallback ENGINE            Fallback engine
  --no-audit                   REMOVED - rejected with an error (GH-139)
  --no-native-first            OCR every page (don't use native text for prose)
  --save-figures               Extract figure PNGs + inline image refs (no captions)
  --describe-figures           Also add VLM captions (opt-in, non-authoritative)
  --timeout SECONDS            Subprocess timeout
  --profile NAME               Load ~/.config/socr/{name}.yaml
  --config PATH                Custom YAML config file
  -q, --quiet / -v, --verbose  Output verbosity
  --dry-run / --reprocess      List-only / force reprocess

  # Agentic cost-aware routing (page-major; progressive save + resume)
  --agentic                    Per page: cheapest provider first, judge escalates,
                               then flush pages/NNN.md to disk before the next page.
                               Re-running resumes from the last finished page.
  --strict-local               Only local/free rungs (no paid cloud)
  --judge-backend MODE         auto | vlm | heuristic (default: auto)
  --judge-model NAME           VLM model for the judge (e.g. qwen2-vl:7b)
  --max-cost-per-page USD      Skip providers above this price (0 = no cap)
  --cost-budget USD            Stop escalating once doc spend hits this (0 = ∞)
  --write-manifest             Write a replayable manifest + blob cache
  --detect-equations           Detect display-equation regions, save crop PNGs (model-free)
  --recover-clean-equations    Also read equation crops to LaTeX into a sidecar (opt-in)
  --legacy-routing             Use the old deterministic backbone instead of agentic

socr batch <DIR> [OPTIONS]
  Same options as process, plus:
  --limit N                    Process first N files

socr replay <MANIFEST> [-o OUT]  Rebuild a document from cache (no model calls)
socr judge-benchmark <DATASET>   Score the judge against labeled good/mangled pages
socr engines                     Show available engines
```

## Output

```
output/<doc_stem>/
├── <doc_stem>.md        # final OCR text (stitched from pages/)
├── metadata.json        # processing stats + status
├── pages/               # agentic mode: per-page progressive save + resume ledger
│   ├── 00001.md         #   one fragment per page (stitches to <doc_stem>.md, byte-identical)
│   ├── 00001.json       #   sidecar: status, terminal flag, engine, run-fingerprint
│   └── ...
├── figures/             # with --save-figures
│   └── figure_1_page3.png
└── audit_log.json       # per-page audit events (timeouts, chart-asset pages, failures)
```

The `pages/` directory is what makes a run **crash-safe and resumable**: each
`NNN.md` is written the instant its page finishes, and re-running reuses the ones
whose `NNN.json` is `terminal` with a matching fingerprint.

## Configuration

Create `~/.config/socr/config.yaml`:

```yaml
primary_engine: gemini
fallback_engine: marker
timeout: 300
save_figures: false
audit_min_words: 50
```

Every `PipelineConfig` field can be set here (see `socr/core/config.py`); keys under
`hpc:` are `HPCConfig` fields.

An unrecognised key is a **hard error** — the load fails and names the offending key.
This is deliberate: a silently-ignored config key is the bug this rule exists to prevent
(#240), where a `cost_budget` spend cap set in a file simply never took effect and
nothing said so.

Or use profiles: `~/.config/socr/fast.yaml` → `socr paper.pdf --profile fast`

## Engine CLIs

Each backend is an independent CLI tool:

- [gemini-ocr-cli](https://github.com/r-uben/gemini-ocr-cli) — Google Gemini
- [deepseek-ocr-cli](https://github.com/r-uben/deepseek-ocr-cli) — DeepSeek via Ollama
- [mistral-ocr-cli](https://github.com/r-uben/mistral-ocr-cli) — Mistral AI
- [marker-ocr-cli](https://github.com/r-uben/marker-ocr-cli) — Marker (Surya + Texify)
- [nougat-ocr-cli](https://github.com/r-uben/nougat-ocr-cli) — Meta Nougat

## License

MIT
