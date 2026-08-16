# TODO

Working list, ranked. Not a mirror of GitHub — the issue tracker holds all 55 open items;
this holds the ones that should be picked up next and why. Updated 2026-08-16.

Ranking rule: **silent content loss first**, then cost of the fix. socr's stated invariant is
that a wrong or dropped number is worse than a missing one, so a defect that ships bad output
while reporting `Success` outranks a defect that fails loudly.

---

## Now — silent loss, cheap to stop

- [ ] **#219 · flag detector disagreement on math pages.** When the page-level math detector
      fires but `detect_display_equations` returns zero regions, say so. This does *not*
      require solving math detection; it converts a silent class into a visible warning.
      Cheapest high-value change on the board.
- [ ] **#205 · TR-3 geometry hard-fail is surfaced nowhere.** Fires on 62/245 table pages and
      no one hears it. Same shape as above: detection already works, surfacing doesn't.
- [ ] **#195 · GH-144 grid rejection ships as a quiet WARNING**, not a page/document/CLI status.
- [ ] **#162 · table verifier exceptions fail open** into the accepting inner judge. An
      exception should never read as approval.

> These four share one root: socr *knows* something is wrong and doesn't tell anyone at a level
> the caller can act on. Fixing the surfacing is independent of fixing the underlying detection,
> and it is what makes the rest of the backlog safely deferrable.

## Next — table integrity (the concentration)

14 of the open issues are one failure mode: a table is damaged and validation doesn't notice.

- [ ] **#190 · all-empty but structurally-valid table passes validation.** Shape-only checks
      cannot see content loss.
- [ ] **#198 · destruction check skips decorated numerics** (`0.67***`, unicode minus, currency).
- [ ] **#152 · two side-by-side tables merged into one region** and flattened.
- [ ] **#212 · EXACT_PASS accepts a model table with no header-attribution check.**
- [ ] **#215 · header-attribution reject term parked** — destroyed header bands ship undetected.
- [ ] **#213 · book indexes routed to table reconstruction.**

## Then — routing correctness

- [ ] **#214 · resume fingerprint has no source-version component.** Fixing a bug does not
      invalidate already-cached pages, so correctness fixes are silently unapplied on resume.
      This one multiplies every other fix on this list and deserves to move up if any of the
      above ship.
- [ ] **#173 · fingerprint omits `auto_patch_tables` and equation model identity.**
- [ ] **#159 · ProviderProfile identity discarded** — the qwen-cloud rung runs the local backend.
- [ ] **#154 / #160 · cost caps do not constrain the $0.00-priced cloud rung.**

## Equations

- [ ] **#219 · rewrite display-math detection** (the deeper half). Font-family whitelisting fails
      for every non-TeX math package; `mathpazo` papers lose every display equation. Consider
      glyph identity (∑ ∫ ∏ √ ± ≤ ≥ ∈, Greek) plus geometry (centred, isolated, larger leading)
      instead of font names.
- [ ] **#165 · PUA-only math pages skip recovery routing.**
- [ ] **#164 · rejected recovery appends whole page native text per region.**
- [ ] **#157 · `--recover-clean-equations` can skip pages with detected crops.**

## Born-digital

- [ ] **#217 · minus-sign glyph forgery.** Root cause confirmed by the three-model panel (pi
      fonts, no `/ToUnicode`, empty `/Differences`; code 50 → `'2'`). Plan exists in
      `docs/log/2026-08-16_gh217-minus-sign-panel.md`; unimplemented. Blast radius is 295 forged
      glyphs across 20 pages — `+`, `−`, `=` all forged, not just the minus.

## Owed from 2026-08-16

- [ ] **Review the qwen-ocr fix retroactively.** `r-uben/qwen-ocr-cli` #4 and #5 reached
      `origin/main` unreviewed (`ff7d817`, `44e1560`). Verified working on both backends against
      real hardware, but no diff review happened.
- [ ] **Move `docs/log/2026-08-16_free-local-ocr-hpc.md`** off `docs/217-glyph-forgery-panel` —
      it was committed there because a parallel session shares the checkout.
- [ ] **Correct the `/hpc-bocconi` skill file**: A100s are 40 GB MIG slices not 80 GB cards;
      H100 (`gpunew`) and H200 (`gpuh200`) partitions exist and are undocumented; home quota is
      175/180 GB so weights belong on `/scratch`; `--hpc-sequential` runs DeepSeek-OCR, not Qwen.
- [ ] **qwen-ocr #2 / #3** — stale vLLM default model, unpinned Ollama tag. Both still open.

## Architecture (not blocking anything)

- [ ] **#155 · split the ~5.5k-LOC `pipeline/orchestrator.py` god-module.**
- [ ] **#174 · make agentic the only first-class path**; quarantine the legacy backbone.
- [ ] **#175 / #176 · package layering and the DocumentState blackboard.**
- [ ] **#156 · this file will drift** against closed issues. Reconcile it when you touch it.

---

# Carried over from the previous TODO (2026-06-10 body)

Retained because it is still open work, not superseded by the ranking above. Wave order and
file ownership for the extraction-defect plans (#144/#146/#147/#150/#151/#152) is owned by
[`docs/plans/extraction-defects/STATUS.md`](docs/plans/extraction-defects/STATUS.md) — not by
this file.

## Benchmark / calibration (#39)

- [ ] **Stage 2 — hand-verified ground truth** for table/equation pages of the 10-paper
      benchmark set. Seed candidates from native/premium-VLM output as `page_N.table.md`
      grids; a human checks the numbers. **Blocks Stage 3.**
- [ ] **Stage 3 — calibration artifact.** `socr benchmark run` across the validated CLI fleet;
      `calibrate --apply` writes `calibration.lock.json` (page-type ladders, benchmark hash,
      engine+model+backend identity); `AUTO_ENGINE_ORDER`, `_LOCAL_ENGINE_ORDER`,
      `RepairRouter` and `provider_ladder` all delegate to it.
      Design: `docs/log/2026-06-10_p1-routing-design.md`.

## Corpus / validation

- [ ] **Regenerate the corrupted-era library copies** (Kuttner 2001, Bernanke–Kuttner 2005)
      with the fixed pipeline. Full corpus re-sweep waits on Stage 3.
- [ ] **Firing-rate validation of structure-restore** across the corpus — how often booktabs
      grids are recovered, and a false-positive check on prose/references.
- [ ] **Judge spot-check on native table pages.** Native pages bypass the VLM judge entirely;
      a cheap image-vs-text check on native *tables* would catch non-corruption native errors
      (mis-ordered columns, dropped sub/superscripts). **Note 2026-08-16:** this is the same
      structural gap as #219 — native-trusted pages are never independently witnessed.

## Textbook-class failures

- [ ] **TICKET-20 / 22 / 24** open. TICKET-19 (prose-shredding) fixed on main (`dc9e773`);
      TICKET-23 (equation→LaTeX) partially addressed by `--recover-corrupt-math` (opt-in,
      corrupt-math only). See `docs/log/2026-06-07_math-textbook-failures.md`.

## Provenance

- [ ] **#158 · per-page provenance written by default** — engine, model version, native-vs-model,
      table-reconstructed flag, encoding-corruption score. Extend `core/audit_log.py`. Closes
      "which model read page N, and was native trusted?"; the manifest is opt-in and
      `model_version` is often blank. Pairs with Stage 3's engine+model identity work.

## Structure-restore polish

- [ ] Strip caption-as-header rows.
- [ ] Collapse the empty separator column between stacked sub-tables.
- [ ] Handle 2-numeric-column tables (the gate currently needs ≥3 numeric lanes/row).

## Later / optional

- [ ] **Content-aware multi-table localization** — the rule-band detector over-merges stacked
      tables and fragments wide ones. Only matters once a use case needs precise per-table crops.
- [ ] **Dual-pass on scans** — needs both image localization (built) *and* a reliable crop-read
      VLM; deferred until a use case justifies it.
- [ ] `--anchor` opt-in flag (anchoring tested marginal; low priority).
- [x] ~~qwen-ocr-cli GitHub remote~~ — exists: `github.com/r-uben/qwen-ocr-cli` (2026-08-16).

## Known limitations (documented, not bugs)

- Dual-pass localizes only vector-ruled / booktabs tables; scanned tables need the image
  detector plus a reliable crop model.
- Auto-patch is opt-in (`--auto-patch-tables`); default is flag-only by design.
- `find_tables(text-strategy)` is ~quadratic in tokens; gated by numeric-column structure and
  a word-count backstop.

---

## Deliberately not doing

- **Chasing a better OCR model.** Measured 2026-08-16: the models are not the bottleneck.
  Tables came back good and 44/44 on the hard fixture; the losses are socr-side detection
  and surfacing. See `docs/log/2026-08-16_free-local-ocr-hpc.md`.
- **Mistral OCR routing changes** (#202, #203) until there is a measurement.
