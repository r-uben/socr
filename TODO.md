# socr — TODO

Live, prioritized next-actions. Detail lives in `TICKETS.md` and `docs/log/`.
Last updated: 2026-06-10.

> **Stale below this line (2026-08-12).** The body predates issues #139–#178 and the
> extraction-defect work. Two corrections, pending a proper rewrite under #156 or PR #179:
>
> - **Extraction defects (#144/#146/#147/#150/#151/#152) are scheduled in
>   [`docs/plans/extraction-defects/STATUS.md`](docs/plans/extraction-defects/STATUS.md)** —
>   the sole owner of wave order and file ownership across those five plans. Wave 0 (PRs #148,
>   #149) merged 2026-08-12; wave 1 is six parallel tickets.
> - The `--recover-clean-equations` bug leading "Now / next" below is now filed as **#157**.

## Now / next
- [ ] **Bug: `--recover-clean-equations` (GH-36b) no-ops on native-trusted pages** —
      found live on a real paper (40pp econ draft, inline display equations in the
      methods section). Native-first correctly glyph-scrambles those equations
      (stacked fraction/subscript layout read in wrong order), and GH-36a's
      `--detect-equations` correctly *finds* the equation region on those same
      pages — but GH-36b then skips splicing the recovered LaTeX back in every
      time, logging `"has detected equations but no PageOutput; skipping"`. Root
      cause: no `PageOutput` exists yet for a native-trusted page at the point
      GH-36b tries to attach the recovered LaTeX. This is exactly the failure mode
      the flag exists to fix, so today it only helps equations on pages that
      already route through OCR for other reasons (tables, scans) — the common
      case (equation embedded in an otherwise-prose page) is silently unfixed.
      Repro: `socr process paper.pdf --detect-equations --recover-clean-equations`
      on any born-digital paper with inline display equations; grep the run log
      for `no PageOutput`. Current workaround is `--no-native-first` (full-page
      OCR, bypasses native trust entirely) — expensive, not a real fix.
- [ ] **#39 Stage 2 — hand-verified ground truth** for table/equation pages of the
      10-paper benchmark set (seed candidates from native/premium-VLM output as
      `page_N.table.md` grids; human checks the numbers). Blocks Stage 3.
- [ ] **#39 Stage 3 — calibration artifact**: run `socr benchmark run` across the
      now-validated CLI fleet, `calibrate --apply` writes `calibration.lock.json`
      (page-type ladders, benchmark hash, engine+model+backend identity), and
      AUTO_ENGINE_ORDER / _LOCAL_ENGINE_ORDER / RepairRouter / provider_ladder all
      delegate to it. Design: `docs/log/2026-06-10_p1-routing-design.md`.
- [ ] **Agentic live test** — uncapped cheapest-first ladder on the 6-page
      Shrimali-Ahmad benchmark paper (waiting for the ocr-fleet session's
      marker/nougat chain to free local compute).
- [ ] **Regenerate the corrupted-era library copies** (Kuttner 2001, Bernanke-
      Kuttner 2005) with the fixed pipeline; full corpus re-sweep waits for Stage 3.
- [ ] **Textbook-class failures (TICKET-19..24)** — TICKET-19 prose-shredding fixed
      on main (dc9e773); TICKET-23 equation→LaTeX partially addressed by
      `--recover-corrupt-math` (opt-in, corrupt-math only); TICKET-20/22/24 open.
      See `docs/log/2026-06-07_math-textbook-failures.md`.
- [ ] **Per-page provenance, written by default** — record per page: engine,
      model version, native-vs-model, table-reconstructed flag, encoding-corruption
      score. Extend `core/audit_log.py`. Closes "which model read page N, and was
      native trusted?" — the manifest is opt-in and `model_version` is often blank.
      (Pairs naturally with Stage 3's engine+model identity work.)
- [ ] **Firing-rate validation of structure-restore** across the corpus — how often
      booktabs grids are recovered, false-positive check on prose/references. Sweeps
      stalled on Google Drive I/O; the numeric-column gate makes them fast now.

## Done 2026-06-09/10 (detail in docs/log/)
- P0 #38: pipeline no longer silently destroys content (html_tables, attempts
  fallback, repair AUTO crash, judge wiring, figures ordering, flagged native).
- P1 #39 Stage 1: uncapped escalation + budget pre-check, truncation gate,
  sparse-page gate at decision points, benchmark coverage hard gate + page types
  + table-cell fidelity. Two adversarial review rounds. 702 tests.
- Validated end-to-end twice on Bernanke-Kuttner 2005 (the old total-failure
  paper): 37/37 pages, judge 19/19 operational, clean tables, honest audit log.

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
