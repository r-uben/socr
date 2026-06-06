# Dev log — Table quality + text-layer integrity (2026-06-06)

Branch: `feat/audit-log` (off `main`). Continues the arc in
`2026-06-05_audit-log-and-real-corpus.md`. ~541 tests pass.

## The arc of this session
Started from "is the dual-pass table feature actually useful on the real corpus?"
Real-corpus evidence redirected the work twice, ending somewhere better than it
started:

1. **Dual-pass can't localize scanned tables** (scans have no vector rules) — but
   the corpus is **overwhelmingly born-digital**, so that barely matters.
2. **The real table defect is structure-loss on born-digital tables**, not VLM
   corruption — fixed deterministically (no model).
3. **"Born-digital" does not mean "trustworthy text layer"** — added corruption
   detection so PyMuPDF is no longer blindly trusted.

## What shipped (commits on `feat/audit-log`)
1. **Durable audit log** (`core/audit_log.py`) — `audit_log.json` per run
   (RECITATION / judge / dual-pass events), written always when events exist.
2. **Image-based table localization** (`tables/image_locate.py`) — connected-
   component rule detection on the raster for scanned pages (vector detectors are
   blind there). Validated: zero false positives on real scanned prose. Adds
   numpy + opencv-python-headless.
3. **Auto-patch -> flag-only by default** + `--auto-patch-tables` opt-in. Codex
   (gpt-5.5): a silent wrong patch to a research number is worse than a missed
   correction; column-count match is not protection. Dual-pass no longer edits the
   corpus by default.
4. **Structure-restore** (`tables/reconstruct.py`) — the headline fix. Born-digital
   booktabs tables (no vertical rules) make `find_tables(lines)` return 0, so they
   were dumped as a flat 1-D token stream (right numbers, no grid). Recover the
   grid with PyMuPDF's text strategy, clean it (drop empty rows/cols, strip page
   running-heads), keep only if still tabular. Char-exact native values, **no
   model, no Ollama**. Validated on Fama-French 1997 p8.
5. **Numeric-column gate** (replaces a crude columnar trigger) — the structural
   test: cluster numeric tokens into x-lanes, require several rows that co-occupy
   >=3 lanes (a grid), else skip. Strictly better than word-count and the old gate:
   references p40 (which used to hang text-strategy for minutes) skips in 0.005s;
   tables p16/p37 that the old gate MISSED are now caught. Word-cap kept only as a
   cost backstop. (Earlier I tested clipping text-strategy to the rule band — 4x
   faster but it SILENTLY TRUNCATED FF p8 from ~48 rows to 23 when the bottom rule
   was undetected, so rejected: truncation is unacceptable for a research corpus.)
6. **Text-layer corruption detection** (`born_digital._encoding_corruption_ratio`)
   — "born-digital" PDFs can have a broken font/ToUnicode map yielding valid chars
   in wrong positions ("Journal"->"Joumal", "1997"->"(/997)"), invisible to the
   garbage-char check, and native pages bypass the judge so it ships silently.
   Two-tier (calibrated: clean ~0.000, header-only glitch ~0.019, pervasive
   ~0.095): FLAG (>1%) records the page suspect; ESCALATE (>5%) sets
   is_born_digital=False -> route to OCR. FF (header-only) is flagged not escalated.

## Key evidence / decisions
- **Corpus composition:** overwhelmingly born-digital; only ONE scanned paper in
  the pre-1996 set (table-less). Scanned pages have ZERO vector drawings.
- **Native extraction is value-faithful but structure-destroying** on booktabs
  tables (find_tables=0 -> flat stream). This is the defect worth fixing.
- **PyMuPDF is NOT uniformly enough:** math -> already routed to VLM via font
  detection; figures -> VLM description; corrupted text layers -> now detected;
  clean prose/tables -> native stays the free default.
- **find_tables(text-strategy) is ~quadratic** in tokens and hangs on dense
  non-table pages (references, equation dumps). The numeric-column gate is the
  correctness-safe guard; clipping to the rule band truncates and was rejected.
- Auto-patch reversal and gate design both cross-checked with codex (gpt-5.5).

## Gotchas
- Papers library is on **Google Drive CloudStorage** — first open of each PDF
  triggers an on-demand download, so corpus sweeps are I/O-bound and slow even when
  per-page work is fast. Not a code bug.
- cv2 + numpy were added to both `~/venvs/socr` and the in-iCloud `.venv` (the
  stale test env). Canonical env is `~/venvs/socr`.

## End-to-end reality check + engine benchmark (the humbling part)
Ran a DEFAULT `socr process` on a 3-page born-digital FF excerpt and watched the
whole pipeline — which corrected the narrative:

- **Born-digital TABLE pages do NOT use native text in the default pipeline.**
  `has_tables -> has_complex_content -> needs_ocr_enhancement=True -> route to the
  VLM` (Gemini). So the clean grid that ships for a table page is Gemini's OCR;
  **structure-restore never fires on the default table path.** It computes the
  native grid only as a fallback / agentic cheap-rung / scoring baseline.
- **What DID run correctly in the default pipeline:** dual-pass (flagged 6 table
  mismatches, 0 patches — flag-only held, no corruption) and the audit log
  (`audit_log.json`, 6 events). Structure-restore + corruption detection live on
  the native path, exercised on PROSE pages and in agentic/offline/fallback runs.

**Table-digit benchmark (FF p8, 18 industries; PDF text layer = ground truth):**

| Engine | time | cell errors | notes |
|--------|------|-------------|-------|
| Gemini (cloud) | ~7s | **0 / 64** | cleanest full grid |
| minicpm-v:8b (local) | 15s | **0** (17/18 rows aligned) | fast + free; minor structure gaps |
| qwen3-vl:8b (local) | 88s | **0** (16/18 rows aligned) | slow on dense pages |

- **On clean born-digital table DIGITS, all three tie at zero errors.** Native
  structure-restore reads the same text layer, so it matches by construction.
- So "char-exact native beats VLM OCR" is theoretical here — Gemini (and the local
  VLMs) read every digit right. The structure-restore value is **cost / offline /
  fallback**, NOT a default-path accuracy win.
- **Local VLMs are viable for clean-table digits** (minicpm-v: 15s, free, 0 errors).
  The cloud's edge is structural cleanliness and harder content (scans, dense
  math), not numeric fidelity on clean tables. socOCRbench (qwen ~0.57 vs Gemini
  ~0.60-0.64) measures that broader doc quality, not table digits.
- **Decision:** keep Gemini as the default for tables (cleanest structure); local
  VLMs (minicpm-v) are a strong free/offline alternative worth the agentic ladder.
  No routing change — the accuracy case for preferring native didn't materialize.

## Open (see TODO.md / TICKETS.md TICKET-18)
- Per-page provenance record (engine + model version + native/model + table-
  reconstructed + corruption flag), written by default. The natural home is the
  audit log; closes the "which model read page N, and was native trusted?" gap.
- Populate `model_version` in manifest fingerprints (long-standing).
- Firing-rate validation of structure-restore across the corpus (sweeps stalled on
  Drive I/O; now fast enough to run).
- Optional: extend the judge to spot-check native *table* pages (catch non-
  corruption native table errors; native pages currently bypass the judge).
- Structure-restore polish: caption-as-header row, the empty separator column
  between stacked sub-tables, 2-numeric-column tables (gate needs >=3 lanes/row).
- Land `feat/audit-log` -> main (9 commits, all tested).
