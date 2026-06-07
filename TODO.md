# socr — TODO

Live, prioritized next-actions. Detail lives in `TICKETS.md` and `docs/log/`.
Last updated: 2026-06-07.

## Now / next
- [ ] **Textbook-class failures (TICKET-19..24)** — first equation-heavy textbook
      run surfaced 6 issues; see `docs/log/2026-06-07_math-textbook-failures.md`.
      Highest: **TICKET-19** — `extract_structured()` shreds a prose+references page
      (Dougherty p19) into a fake 9-col grid; contradicts TICKET-18's met criterion
      "references/prose never trigger reconstruction". Then **TICKET-23** (no
      equation→LaTeX path; the central gap for math books) and **TICKET-21**
      (local-only run hard-errors on a fully-written `.md`).
- [ ] **Land `feat/audit-log` -> main** — 9 commits, all tested (541 pass). The
      table-quality + text-integrity program (see `docs/log/2026-06-06_*`).
- [ ] **Per-page provenance, written by default** — record per page: engine,
      model version, native-vs-model, table-reconstructed flag, encoding-corruption
      score. Extend `core/audit_log.py`. Closes "which model read page N, and was
      native trusted?" — the manifest is opt-in and `model_version` is often blank.
- [ ] **Firing-rate validation of structure-restore** across the corpus — how often
      booktabs grids are recovered, false-positive check on prose/references. Sweeps
      stalled on Google Drive I/O; the numeric-column gate makes them fast now.

## Soon
- [ ] **Populate `model_version`** in manifest fingerprints (long-standing gap).
- [ ] **Judge spot-check on native table pages** — native pages currently bypass
      the VLM judge; a cheap image-vs-text check on native *tables* would catch
      non-corruption native errors (mis-ordered columns, dropped sub/superscripts).
- [ ] **Structure-restore polish** — strip caption-as-header rows; collapse the
      empty separator column between stacked sub-tables; handle 2-numeric-column
      tables (the gate currently needs >=3 numeric lanes/row).

## Later / optional
- [ ] **Content-aware multi-table localization** — the rule-band detector over-
      merges stacked tables and fragments wide ones; only matters once a use case
      needs precise per-table crops (dual-pass on scans).
- [ ] **Dual-pass on scans** — needs both image localization (built) AND a reliable
      crop-read VLM; deferred until a use case justifies it.
- [ ] `--anchor` opt-in flag (anchoring tested marginal; low priority).
- [ ] qwen-ocr-cli GitHub remote (local-only; create if backup wanted).

## Known limitations (documented, not bugs)
- Dual-pass localizes only vector-ruled / booktabs tables; scanned tables need the
  image detector + a reliable crop model.
- Auto-patch is opt-in (`--auto-patch-tables`); default is flag-only by design.
- `find_tables(text-strategy)` is ~quadratic in tokens; gated by numeric-column
  structure + a word-count backstop.
