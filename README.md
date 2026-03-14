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
# Process a PDF
socr paper.pdf

# Choose engine
socr paper.pdf --primary gemini
socr paper.pdf --primary marker

# Save extracted figures
socr paper.pdf --save-figures

# Batch process a directory
socr batch ~/Papers/ -o ./results/
socr batch ~/Papers/ --dry-run        # preview what would be processed
socr batch ~/Papers/ --reprocess      # force reprocess all

# Check which engines are available
socr engines
```

## How it works

```
PDF → Primary OCR → Quality Audit → (Fallback OCR if needed) → Markdown
```

1. **Primary OCR** — Calls the primary engine CLI on the whole PDF
2. **Quality audit** — Heuristic checks (word count, garbage ratio, repetition)
3. **Fallback** — If audit fails, tries a different engine

Each engine is a separate CLI binary. `socr` calls it as a subprocess, reads the output markdown, and applies the quality pipeline.

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
  --save-figures               Save extracted figure images
  --timeout SECONDS            Subprocess timeout (default: 300)
  --profile NAME               Load ~/.config/socr/{name}.yaml
  --config PATH                Custom YAML config file
  -q, --quiet                  Suppress non-error output
  -v, --verbose                Verbose output
  --dry-run                    List files without processing
  --reprocess                  Force reprocess already-done files

socr batch <DIR> [OPTIONS]
  Same options as process, plus:
  --limit N                    Process first N files

socr engines                   Show available engines
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
