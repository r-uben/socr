# Dev log — Failures on equation-heavy textbooks (2026-06-07)

Branch: `docs/annotate-math-textbook-failures` (off `main`).

First real-corpus run of socr against **born-digital but equation-heavy
textbooks** (as opposed to the ~journal-paper corpus the pipeline was tuned on).
Test document: Dougherty, *Pattern Recognition and Classification* (Springer
2013), 203 pp, born-digital, **clean text layer** (`pdftotext` extracts prose
char-exact). A 1220 pp companion (Theodoridis, *Machine Learning*, 2026) is
queued but not yet run — same class of document, so the same failures apply.

Run config (local-first, no cloud — the realistic offline mode):
`--unified --save-figures --primary qwen --qwen-backend ollama
--qwen-model qwen3-vl:8b --fallback glm`, with `GEMINI_API_KEY`/`MISTRAL_API_KEY`
unset. Hardware: M3 Max / 64 GB.

This log **annotates what does not work** — it is a failure inventory, not a fix.
Each item maps to a ticket (TICKET-19..24) in `TICKETS.md`.

## Summary: native is good, the routing around it is not

The clean native text layer is excellent (e.g. p8 acknowledgments: flawless,
ligatures intact). Every failure below is socr **routing away from that good
native text**, or **erroring on a usable result**, or **lacking a math-aware
path** — not the text layer itself.

## Failures

### F1 — `extract_structured()` shreds prose/reference pages into fake grids  (TICKET-19, highest)
Born-digital pages flagged `has_tables` get `extract_structured()` instead of
`get_text()`. On **p19** (an exercises + references page with one tiny embedded
data table), PyMuPDF's `find_tables` text-strategy over-fires and renders the
*entire page* — prose, exercises, the reference list — as a 9-column markdown
grid, splitting words mid-token:

```
| 3. Consider th | e | following | items bou | ght i | n a | supermarket |
| Alpaydin, E.: Intro | d | uction to Ma | chine learnin | g, 2nd | edn. | MIT Press |
```

- **~62 corrupted lines** document-wide (`grep -cE "\| [a-z] \|"`).
- **Reproduced byte-identically on BOTH a VLM-routed run AND a native-only run**
  → the corruption is the *native* structured-table path, NOT the VLM.
- Plain `pdftotext -f 19 -l 19` on the same page is perfectly clean.
- This is the exact false-positive TODO already flags ("firing-rate validation of
  structure-restore … false-positive check on prose/references"). Here is a
  concrete, checked-in reproducer. The numeric-column gate (≥3 lanes) is **not**
  catching this page.

### F2 — Over-classification of born-digital pages as "complex"  (TICKET-20)
`needs_ocr_enhancement = has_tables or has_figures or has_equations` flagged
**129 / 203** born-digital pages as complex, routing them off the free,
char-exact native path onto local VLMs that are **slower and worse on prose**.
For a document with a clean text layer this inverts the cost/quality tradeoff:
the pipeline pays VLM time to *degrade* text that `get_text()` already had right.

### F3 — Local-only run hard-errors on a fully-written output  (TICKET-21)
When `GEMINI_API_KEY` is absent and a page fails local audit, the escalation
target is the **cloud** engine → escalation is impossible → the run ends with
`Error: Processing failed: None` and a non-zero exit, **even though the complete
`.md` was written** (357 KB, all 203 pages). Also observed: p3 logged
"glm garbage -> deepseek (recovered by deepseek)" but the emitted page is
**blank**. There is no "local-only: accept best local output, exit 0" mode, and
"recovered" can mean "recovered to empty".

### F4 — "Scanned" over-count  (TICKET-22)
The detector reported **13 scanned pages**; ground truth (pages whose
`get_text()` is < 40 chars) is **4** — pages 1, 2, 3, 9, all decorative front
matter. The other 9 are full-page-figure / sparse pages tripping
`MIN_WORDS_PER_PAGE=15` / `MIN_CHARS_FOR_TEXT_LAYER=50` and getting needlessly
sent to OCR.

### F5 — No math-aware extraction path (the central gap for this corpus)  (TICKET-23)
Math pages are detected (`_MATH_FONT_RE` → routed to OCR pre-extraction), but the
only destinations are **general VLMs that do not emit LaTeX** or **native
`get_text()` that linearizes math** (flattens sub/superscripts, drops Greek,
breaks reading order). For equation-heavy textbooks this is the headline miss:
socr has no equation→LaTeX route. A `/consilium` panel (Codex; Gemini failed to
return) recommended a local math-aware step — **marker-pdf / Texify on MPS** —
run only on detected equation regions, always storing the crop PNG beside any
LaTeX so bad LaTeX never silently replaces a faithful image.

### F6 — No native-only / routing-threshold CLI knob  (TICKET-24)
To make socr trust the clean text layer and OCR only genuine scans, we had to
**patch `born_digital.py`** (a `SOCR_NATIVE_ONLY` env gate forcing
`needs_ocr_enhancement = False`). That patch was reverted to keep `main` canonical
for this annotation pass. There is no supported flag (`--native-only`,
`--enhancement-threshold`) to express "this is born-digital with a clean layer;
do not enhance prose." `--no-native-first` is the opposite lever.

## Cross-cutting takeaway
socr was tuned on born-digital **journal papers**, where the text layer is mostly
prose + a few tables. On **textbooks** the mix flips: lots of equations, lots of
embedded figures, long reference/exercise pages. That flip is what surfaces
F1–F5. The pipeline's instinct ("enhance complex pages with a model") is exactly
wrong when the native layer is already clean and the "model" is a general local
VLM with no math/table specialization.
