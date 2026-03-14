# socr

[![PyPI](https://img.shields.io/pypi/v/socr)](https://pypi.org/project/socr/)
[![Python](https://img.shields.io/pypi/pyversions/socr)](https://pypi.org/project/socr/)
[![License](https://img.shields.io/github/license/r-uben/socr)](LICENSE)

Multi-engine OCR with cascading fallback, quality audit, and figure extraction.

Process academic papers and documents using free local models first, with automatic cloud fallback for failed pages. Extract and describe figures using vision models.

## Features

- **Multi-engine OCR** — Nougat, DeepSeek, Mistral, Gemini (via dedicated CLI tools)
- **Smart routing** — Free local engines first, cloud only when needed
- **Quality audit** — Heuristics + LLM review catches garbage text
- **Figure extraction** — Renders figures from PDFs, describes with vision models
- **Parallel processing** — Process multiple pages/figures concurrently
- **Batch processing** — Process entire directories of papers
- **CLI** — Live progress, colored panels, cost tracking
- **Modular engines** — Each OCR backend is a standalone CLI tool

## Quick Start

```bash
# Install globally (recommended)
pipx install socr --python python3.12

# Or install in a project
pip install socr

# Pull required Ollama models
ollama pull deepseek-r1:32b    # quality audit
ollama pull deepseek-ocr       # local OCR

# Process a paper
socr paper.pdf

# Process with figure images saved
socr paper.pdf --save-figures

# Batch process a folder
socr batch ~/Papers/ --limit 10
```

## Example Output

Processing a 22-page economics paper:

```
socr v0.1.0

kuttner_2001_monetary_policy.pdf
22 pages, 1.2 MB
type: academic

(1) primary ocr
    deepseek
    [+] page 1
    [+] page 2
    ...

(2) quality audit
    [!] page 10 (19.2% garbage)

(3) fallback ocr
    gemini
    [+] page 10

(4) figure processing
    [+] fig 1 (p.1): unknown
    [+] fig 2 (p.8): scatter_plot
    [+] fig 3 (p.12): scatter_plot

---

done 22/22 pages
     3 figures
     241.5s
     $0.0002
     deepseek (21) + gemini (1)

-> output/kuttner_2001_monetary_policy/kuttner_2001_monetary_policy.md
```

Output structure:
```
output/<doc_stem>/
├── <doc_stem>.md      # Full OCR text + figure descriptions
├── metadata.json      # Stats, engines used, cost
└── figures/           # With --save-figures
    ├── figure_1_page1.png
    └── ...
```

## Output Formats

Use `-f/--format` to choose the output format:

| Format | Content | Best For |
|--------|---------|----------|
| `markdown` (default) | Formatted text with page headers, inline figure descriptions | Human reading, documentation |
| `json` | Structured data with pages, figures, stats, engine info | Programmatic access, downstream processing |
| `txt` | Plain concatenated text with page separators | Simple text extraction |

### JSON format structure

```json
{
  "document": "/path/to/paper.pdf",
  "pages": [
    {
      "page_num": 1,
      "text": "...",
      "status": "success",
      "engine": "deepseek",
      "confidence": null,
      "figures": [
        {
          "figure_num": 1,
          "figure_type": "scatter_plot",
          "description": "A scatter plot showing...",
          "bbox": [100, 200, 400, 500],
          "image_path": "figures/figure_1_page1.png"
        }
      ]
    }
  ],
  "stats": {
    "total_pages": 22,
    "pages_success": 22,
    "figures_detected": 3,
    "total_cost": 0.0002,
    "total_time": 241.5
  }
}
```

The `metadata.json` contains:
```json
{
  "document": "/path/to/paper.pdf",
  "stats": {
    "total_pages": 22,
    "pages_success": 22,
    "total_cost": 0.0002,
    "total_time": 241.5
  },
  "engines_used": { "deepseek": 21, "gemini": 1 },
  "figures": 3,
  "pages_needing_reprocessing": []
}
```

See `examples/` for complete processed papers:
- `kuttner_2001/` - 22 pages, 3 figures (monetary policy)
- `sutskever_2014/` - 9 pages (sequence to sequence learning)
- `bernanke_kuttner_2005/` - 37 pages, 6 vector figures (stock market reaction)

## Pipeline Architecture

```
                                    ┌─────────────┐
                                    │  PDF Input  │
                                    └──────┬──────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │  Document Classifier   │
                              │  (academic vs general) │
                              └───────────┬────────────┘
                                          │
          ┌───────────────────────────────┼───────────────────────────────┐
          │                               │                               │
          ▼                               ▼                               ▼
    ACADEMIC PATH                   GENERAL PATH                    ROUTER AGENT
 (Nougat → DeepSeek)             (DeepSeek → Nougat)            (picks available engine)
          │                               │                               │
          └───────────────────────────────┴───────────────────────────────┘
                                          │
                          ┌───────────────┴───────────────┐
                          │   STAGE 1: PRIMARY OCR        │
                          │                               │
                          │  ┌─────────────────────────┐  │
                          │  │ Local Engines (FREE):   │  │
                          │  │                         │  │
                          │  │  • Nougat               │  │
                          │  │    nougat_ocr lib       │  │
                          │  │    model: 0.1.0-small   │  │
                          │  │                         │  │
                          │  │  • DeepSeek             │  │
                          │  │    Ollama               │  │
                          │  │    deepseek-ocr:latest  │  │
                          │  └─────────────────────────┘  │
                          │             OR                │
                          │  ┌─────────────────────────┐  │
                          │  │ Cloud Engines (PAID):   │  │
                          │  │                         │  │
                          │  │  • Gemini               │  │
                          │  │    Google genai SDK     │  │
                          │  │    gemini-3-flash       │  │
                          │  │    ~$0.0002/page        │  │
                          │  │                         │  │
                          │  │  • Mistral              │  │
                          │  │    Mistral SDK          │  │
                          │  │    pixtral-large        │  │
                          │  │    ~$0.001/page         │  │
                          │  └─────────────────────────┘  │
                          └───────────────┬───────────────┘
                                          │
                                   [Page Results]
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │   STAGE 2: QUALITY AUDIT      │
                          │                               │
                          │  ┌─────────────────────────┐  │
                          │  │ Heuristics Checker      │  │
                          │  │ (rule-based, instant):  │  │
                          │  │  • Word count ≥ 50      │  │
                          │  │  • Garbage ratio < 15%  │  │
                          │  │  • Avg word length ok   │  │
                          │  │  • No unicode issues    │  │
                          │  │  • No repeated patterns │  │
                          │  └──────────┬──────────────┘  │
                          │             │                 │
                          │     [pages flagged?]          │
                          │             │                 │
                          │             ▼                 │
                          │  ┌─────────────────────────┐  │
                          │  │ Cross-Check (optional): │  │
                          │  │  Try 2nd local engine   │  │
                          │  └──────────┬──────────────┘  │
                          │             │                 │
                          │     [still flagged?]          │
                          │             │                 │
                          │             ▼                 │
                          │  ┌─────────────────────────┐  │
                          │  │ LLM Auditor (FREE):     │  │
                          │  │  Ollama                 │  │
                          │  │  deepseek-r1:32b        │  │
                          │  │  (reasoning model)      │  │
                          │  │  • Can override heur.   │  │
                          │  │  • verdict: acceptable, │  │
                          │  │    needs_review, poor   │  │
                          │  └──────────┬──────────────┘  │
                          └─────────────┼─────────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                    [all pass]                   [some flagged]
                         │                             │
                         │                             ▼
                         │             ┌───────────────────────────────┐
                         │             │   STAGE 3: FALLBACK OCR       │
                         │             │                               │
                         │             │  Router selects different     │
                         │             │  engine (not primary):        │
                         │             │   Gemini → Mistral →          │
                         │             │   DeepSeek → Nougat           │
                         │             │                               │
                         │             │  Reprocess flagged pages      │
                         │             └───────────────┬───────────────┘
                         │                             │
                         └─────────────┬───────────────┘
                                       │
                                  [All Pages]
                                       │
                                       ▼
                          ┌────────────────────────────┐
                          │  STAGE 4: FIGURE AGENT     │
                          │  (if enabled)              │
                          │                            │
                          │  ┌──────────────────────┐  │
                          │  │ PyMuPDF Extractor:   │  │
                          │  │  • Vector figures    │  │
                          │  │    (charts, plots)   │  │
                          │  │  • IMAGE blocks      │  │
                          │  │  • Embedded images   │  │
                          │  │  • Filter by size    │  │
                          │  │  • Render at 150 DPI │  │
                          │  └──────────┬───────────┘  │
                          │             │              │
                          │             ▼              │
                          │  ┌──────────────────────┐  │
                          │  │ Vision Engine:       │  │
                          │  │  Gemini 3 Flash      │  │
                          │  │  DeepSeek-OCR        │  │
                          │  │  Pixtral-Large       │  │
                          │  └──────────┬───────────┘  │
                          │             │              │
                          │      [Figure Results]      │
                          └─────────────┬──────────────┘
                                        │
                                        ▼
                          ┌─────────────────────────────┐
                          │      OUTPUT ASSEMBLY        │
                          │                             │
                          │  Format: markdown/json/txt  │
                          │                             │
                          │  output/<doc_stem>/         │
                          │    ├─ document.md           │
                          │    ├─ metadata.json         │
                          │    └─ figures/              │
                          │         └─ figure_N.png     │
                          └─────────────────────────────┘
```

**Cost optimization:**
- Local engines first (Nougat, DeepSeek via Ollama) - FREE
- Quality audit with local Ollama (llama3.2/qwen2.5) - FREE
- Cloud fallback only for failed pages (Gemini ~$0.0002/page)
- Typical 22-page paper: $0.0002 total (only 1 page needed cloud)

## Requirements

**Base:**
- Python 3.11+
- [Ollama](https://ollama.ai) with `deepseek-r1:32b` (for quality audit) or `llama3.2`/`qwen2.5` as lighter alternatives

**OCR Engines** (install individually based on needs):

| Engine | Installation | Type | Cost |
|--------|-------------|------|------|
| **DeepSeek** | `pip install deepseek-ocr-cli` | Local (Ollama) | Free |
| **Nougat** | `pip install nougat-ocr-cli` | Local (Python) | Free |
| **Gemini** | `pip install gemini-ocr-cli` | Cloud API | ~$0.0002/page |
| **Mistral** | `pip install mistral-ocr-cli` | Cloud API | ~$0.001/page |

**Installation examples:**

```bash
# Install with local engines only (recommended to start)
pip install socr[local]

# Install with all engines
pip install socr[all]

# Install specific engines
pip install socr[deepseek,gemini]
```

**Setup:**

```bash
# 1. Install Ollama and pull audit model (reasoning model recommended)
ollama pull deepseek-r1:32b  # best quality, or llama3.2 for speed

# 2. If using DeepSeek OCR, pull its model
ollama pull deepseek-ocr:latest

# 3. If using cloud engines, set API keys
export GEMINI_API_KEY="your-key"
export MISTRAL_API_KEY="your-key"

# 4. Check engine status
socr engines
```

## OCR Engine CLIs

Each OCR backend is a standalone CLI tool that can be used independently:

- **[deepseek-ocr-cli](https://github.com/r-uben/deepseek-ocr-cli)** - Local OCR via Ollama/DeepSeek
- **[gemini-ocr-cli](https://github.com/r-uben/gemini-ocr-cli)** - Cloud OCR via Google Gemini
- **[mistral-ocr-cli](https://github.com/r-uben/mistral-ocr-cli)** - Cloud OCR via Mistral AI
- **[nougat-ocr](https://github.com/facebookresearch/nougat)** - Academic papers (used as Python library)

## CLI Commands

```bash
# Simple usage
socr paper.pdf                    # Process a PDF
socr paper.pdf --save-figures     # Save figure images

# Full options
socr process paper.pdf [OPTIONS]
  -o, --output PATH      Output file path
  -f, --format           markdown|json|txt
  --primary ENGINE       Force primary engine
  --fallback ENGINE      Force fallback engine
  --no-audit             Skip quality audit
  --no-figures           Skip figure processing
  --save-figures         Save figure images to disk
  --figures-engine       gemini|deepseek|mistral (default: auto-select)
  --timeout SECONDS      Timeout per page/figure (default: 300)
  --workers N            Parallel workers (default: 4, use 1 for sequential)

# Batch process directory
socr batch ~/Papers/ [OPTIONS]
  --limit N              Process first N files
  --save-figures         Save all figure images
  --figures-engine       gemini|deepseek|mistral (default: auto-select)
  --timeout SECONDS      Timeout per page/figure (default: 300)
  --workers N            Parallel workers (default: 4)

# Check engines
socr engines

# Check audit system
socr audit-status
```

## Parallel Processing

socr can process multiple pages and figures in parallel to speed up large documents:

```bash
# Fast processing with 8 parallel workers (recommended for M3 Pro/Max with 32GB+ RAM)
socr paper.pdf --workers 8

# Conservative processing for limited RAM
socr paper.pdf --workers 2

# Sequential processing (most reliable, slowest)
socr paper.pdf --workers 1

# Extended timeout for very complex pages (10 min per page)
socr paper.pdf --timeout 600
```

**Hardware recommendations:**

| RAM | Workers | Notes |
|-----|---------|-------|
| 8GB | 1-2 | Sequential or minimal parallelism |
| 16GB | 2-4 | Good for most documents |
| 32GB | 4-6 | Fast processing |
| 64GB+ | 6-8 | Maximum parallelism |

Parallel processing uses ThreadPoolExecutor for concurrent page/figure processing. Each worker loads the page image into memory, so higher worker counts require more RAM.

**Timeout configuration:**

The default 300s (5 min) timeout works for most pages. For documents with complex figures or slow API responses, increase the timeout:

```yaml
# socr.yaml
parallel_pages: 4
parallel_figures: 2
figure_timeout: 600  # 10 min per figure

nougat:
  timeout: 600
deepseek:
  timeout: 600
```

## Configuration

Create `socr.yaml` in your project or home directory:

```yaml
# Engine selection
primary_engine: deepseek
fallback_engine: gemini
figures_engine: gemini  # For figure descriptions

# Quality audit
audit:
  enabled: true
  min_word_count: 50
  garbage_threshold: 0.15

# Figure processing
include_figures: true
save_figures: false
figures_max_total: 25
figures_max_per_page: 3
figure_timeout: 180  # seconds per figure description

# Parallel processing (adjust based on your hardware)
parallel_pages: 4    # pages processed concurrently
parallel_figures: 2  # figures described concurrently

# Output
output_dir: output
output_format: markdown

# Engine-specific timeouts (seconds)
nougat:
  timeout: 300
deepseek:
  timeout: 300
gemini:
  timeout: 300
mistral:
  timeout: 300
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed module documentation.

```
src/socr/
├── cli.py              # Click CLI
├── core/
│   ├── config.py       # AgentConfig dataclass
│   ├── document.py     # PDF loading
│   └── result.py       # OCRResult, PageResult, FigureResult
├── engines/
│   ├── base.py         # BaseEngine ABC
│   ├── deepseek.py     # Ollama/DeepSeek
│   ├── gemini.py       # Google Gemini
│   ├── mistral.py      # Mistral AI
│   └── nougat.py       # Nougat (academic)
├── audit/
│   ├── heuristics.py   # Garbage detection
│   └── llm_audit.py    # Ollama-based review
├── pipeline/
│   ├── processor.py    # 4-stage pipeline
│   └── router.py       # Engine selection
└── ui/                 # Rich console output
```

## License

MIT
