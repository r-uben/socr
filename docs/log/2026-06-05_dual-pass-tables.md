# Dev log — Dual-pass table extraction (Phase 4c) (2026-06-05)

## Context
Follow-on to the hard-page VLM judge (`feat/qwen-engine`, PR #28). The judge
*detects* corrupted tables by looking at the whole page; this feature *extracts*
them precisely: locate each table, crop it, re-read the crop with a
table-specialised VLM pass, and reconcile that reading against the whole-page OCR.
Crop-vs-page disagreement is a built-in corruption flag. Owner reads econ/finance
papers, so regression/summary tables are the high-value, high-corruption-risk
content.

Branch: `feat/dual-pass-tables` (off `feat/qwen-engine`). 511 tests pass (+20).

## Design decisions (owner's calls, recorded)
1. **On disagreement: auto-patch + flag.** The crop pass (higher effective
   resolution, table-focused) is authoritative — replace the page-markdown table
   with the crop reading AND surface the disagreement. Safety rails keep this from
   corrupting the corpus: patch only when the crop reading is a well-formed table
   whose column count matches the page table; otherwise flag without editing.
2. **v1 scope: ruled + booktabs only.** See the localization evidence below.
   Fully-borderless (whitespace-only) tables stay on the whole-page judge.
3. **Crop-pass model = the judge model ladder** (`_resolve_judge_model`:
   qwen3.5:cloud -> minicpm-v -> qwen3-vl). No new config, no extra key.

## Localization evidence (measured, not assumed)
Ran `page.find_tables()` against three synthetic table styles. The naive default
misses most econ tables; the fix is geometric:

| Style | `find_tables(lines)` | `find_tables(text)` | `get_drawings()` rule-band |
|-------|----------------------|---------------------|-----------------------------|
| Fully ruled | tight bbox | — | — |
| Booktabs (h-rules only — *the econ norm*) | **0 tables** | bbox swallows half the page | tight band from rule y-coords |
| Fully borderless | 0 tables | over-inclusive | no rules to find |

- `find_tables` default ("lines") only localizes fully-ruled tables.
- "text" strategy localizes booktabs/borderless but its bbox is unusable: on a
  realistic mixed page it spanned the section title, a prose paragraph, the
  caption AND the table (y 72->364 for a table at y 188->308). Too over-inclusive
  to crop -> would re-OCR half the page.
- **Booktabs fix:** read the horizontal rules (top/mid/bottom) from
  `page.get_drawings()`; their y-coords give a tight vertical band, their
  endpoints the horizontal extent. Validated: band (110,170,490,308) for a table
  whose text spans y 174->306, ignoring surrounding prose.

So v1 = ruled (`find_tables` lines) + booktabs (rule-band detector), both with
precise, auto-patch-safe bboxes. Borderless = documented limitation.

## What shipped
- `src/socr/tables/locate.py` — `locate_tables(page) -> [TableBox]`: ruled
  (find_tables) + booktabs (horizontal-rule-band from get_drawings). Dedup by IoU,
  reading order. Geometric thresholds only (rule flatness/width in PDF points), no
  tuned percentages.
- `src/socr/tables/extract.py` — `TableCropExtractor` crops each bbox to a
  high-DPI PNG (400, crops are small so they afford more DPI than a full page) and
  reads it via an injected `TableReader`. `OllamaTableReader` is the default
  backend (reuses the judge's image -> `/api/generate` path). Fail-open: a failed
  crop/read drops that table (-> count mismatch -> flag, not patch).
- `src/socr/prompts/table_extract.md` — policy-as-data: transcribe one table to
  markdown, preserve every digit/sign/paren/star, keep SEs in their own row/cell,
  don't invent.
- `src/socr/tables/reconcile.py` — parse markdown table blocks, structural
  cell-by-cell diff (no similarity threshold; any differing cell is a
  disagreement, exact (row,col,old,new) reported), patch crop -> page by
  reading-order index. Count mismatch or malformed crop -> flag without editing.
- Orchestrator **Phase 4c** `_phase_dual_pass_tables` — runs before assembly, for
  every mode, on `has_tables` pages with model-OCR'd (non-native) best_output.
  Patches `best_output.text`, appends a `dual-pass <action>: ...` audit note,
  surfaces in console. Gated by `config.dual_pass_tables` (default True),
  `--no-dual-pass-tables`. Fail-open per page.

## Validation
- 20 new unit/integration tests (`tests/test_dual_pass_tables.py`): localization
  (ruled/booktabs/borderless/mixed-prose), markdown parse + diff (incl.
  formatting-only differences ignored), reconcile (patch / agreement no-op /
  count-mismatch flag / malformed-crop flag), extractor (stub reader + fail-open),
  and the orchestrator phase on a **real synthetic PDF** (locate + crop-render run
  for real; only the VLM is stubbed).
- **Live end-to-end** (qwen3-vl:8b via Ollama): synthetic booktabs table, two
  cells corrupted in the simulated page OCR ((0.010)->(0.018), 1,204->1,294). The
  crop pass transcribed both correctly and reconcile patched both, naming the
  exact misreads. Full pipeline confirmed against a real model.

## Limitations / next
1. **Fully-borderless tables** have no geometric anchor -> not localized in v1.
   The whole-page hard-page judge still covers them. A future detector could
   cluster numeric text lines into a band, but text-strategy bboxes proved too
   sloppy to auto-patch.
2. Crop pass is one VLM call per located table on table pages — extra latency on
   table-heavy docs. Gated behind the flag; only model-OCR'd pages.
3. Disagreement records live in `best_output.audit_notes` + console. A durable
   per-run audit log (shared with RECITATION/judge rejections) is still the open
   item from the qwen session.

## Branch state
- `feat/dual-pass-tables` off `feat/qwen-engine`. Clean, 511 tests pass, new code
  ruff-clean (also removed a pre-existing dead `import fitz` in orchestrator).
- Not yet pushed / no PR (owner's call). Natural target: PR into `feat/qwen-engine`
  or stack behind PR #28.
