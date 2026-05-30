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

# Cost-aware agentic mode: cheapest provider first, escalate only if rejected
socr paper.pdf --agentic
socr paper.pdf --agentic --cost-budget 0.05      # cap spend per document
socr paper.pdf --agentic --max-cost-per-page 0.0  # local-only (free)

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
PDF → per page: try cheapest provider → judge the output
    → accept, or escalate up the cost ladder → Markdown (+ replayable manifest)
```

The engine is chosen **dynamically by cost**: try the cheapest available provider
first, let a judge (a vision model that looks at the page, or a heuristic
fallback) decide accept-or-escalate, and climb the cost ladder
(`local → cheap cloud → premium cloud`) only when the cheaper output is rejected.
Stops at the first accepted output, bounded by `--cost-budget` / `--max-cost-per-page`.
Each run records the winning provider + cost per page and writes a **manifest**
that `socr replay` can reconstruct with zero model calls.

Each engine is a separate CLI binary. `socr` calls it as a subprocess, reads the
output markdown, and applies the quality pipeline. See `docs/ARCHITECTURE.md` for
the full design.

## Engines

| Engine | Package | Type | Notes |
|--------|---------|------|-------|
| Gemini | `gemini-ocr-cli` | Cloud | Google Gemini, ~$0.0002/page |
| Mistral | `mistral-ocr-cli` | Cloud | Mistral AI |
| Marker | `marker-ocr-cli` | Local | Layout-aware (Surya + Texify) |
| DeepSeek | `deepseek-ocr-cli` | Local | Via Ollama |
| Nougat | `nougat-ocr-cli` | Local | Academic papers, Python <3.13 |

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
  --no-audit                   Skip quality audit
  --no-native-first            OCR every page (don't use native text for prose)
  --save-figures               Save extracted figure images
  --timeout SECONDS            Subprocess timeout
  --profile NAME               Load ~/.config/socr/{name}.yaml
  --config PATH                Custom YAML config file
  -q, --quiet / -v, --verbose  Output verbosity
  --dry-run / --reprocess      List-only / force reprocess

  # Agentic cost-aware routing
  --agentic                    Per page: cheapest provider first, judge escalates
  --judge-backend MODE         auto | vlm | heuristic (default: auto)
  --judge-model NAME           VLM model for the judge (e.g. qwen2-vl:7b)
  --max-cost-per-page USD      Skip providers above this price (0 = no cap)
  --cost-budget USD            Stop escalating once doc spend hits this (0 = ∞)
  --write-manifest             Write a replayable manifest + blob cache

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
├── <doc_stem>.md        # OCR text
├── metadata.json        # Processing stats
└── figures/             # With --save-figures
    └── figure_1_page3.png
```

## Configuration

Create `~/.config/socr/config.yaml`:

```yaml
primary_engine: gemini
fallback_engine: marker
timeout: 300
save_figures: false
audit_enabled: true
audit_min_words: 50
```

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
