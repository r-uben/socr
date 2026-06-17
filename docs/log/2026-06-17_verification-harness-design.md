# Table-verification harness — design (GH-56)

Date: 2026-06-17
Panel: codex (gpt-5.5) + gemini (antigravity), strong consensus. Supersedes the brittle
whole-page `_value_guard` application.

## Why

The deterministic value-guard (per-row multiset + label-binding + row-count, in
`native_verifier.py`) is correct on simple pages but kept FALSE-REJECTING a CORRECT, complete
extraction of the real Consensus Economics page (`202401.pdf` p.4). Root cause, confirmed over five
rounds: it verifies the output table against the WHOLE PAGE's native tokens, and on a multi-element
page (two tables + a bar chart + a prose block) the chart-axis numbers and prose figures
contaminate the native-row set — inflating the count AND mis-aligning the per-row pairing, so the
multiset check hard-fails correctly-extracted rows. Geometric patches (chart-bbox exclusion, y-band
cutoffs) each hit the next confounder. The whole-page frame is the problem.

## The harness — tri-state, table-level

For each extracted table:

- **EXACT_PASS** — the deterministic check, **scoped to the table's own region** (not whole-page),
  finds clean 1:1 row alignment and matching numeric multisets → **ship** (no model; strongest
  guarantee).
- **CERTAIN_FAIL** — systematic label-binding offset, OR a multiset mismatch *under clean
  alignment* (a number is genuinely wrong/dropped) → **reject → image floor**. A model may NOT
  override this.
- **AMBIGUOUS** — multiset mismatch *concurrent with* a row-count / alignment / lane discrepancy
  (the pairing is polluted — the CE case), OR no trusted table bbox, OR a scanned page (no native
  tokens) → **defer to the VLM table-judge** → confirm: ship · reject/uncertain: image floor.

## The load-bearing insight: SCOPE TO THE TABLE BBOX

- **Deterministic check → per region.** Run it on the native words inside each table's region
  (forecaster table; historical table), excluding the chart + prose **by construction**. On CE this
  makes the alignment clean → exact all-cell verification → **CE ships deterministically**, no model.
- **VLM judge → cropped table image.** Show it a tight crop of *just the table*, never the full
  page — removes chart/prose distraction and keeps the digits large enough to read.

## VLM table-judge contract (a strict auditor, not a rubber stamp)

- Input: cropped table-region image + the extracted markdown.
- Adversarial, structured ask: "You are a strict data auditor. Report any cell that is missing,
  altered, or unreadable; ACCEPT only if every number matches." Return per-cell verdicts
  (`{row, col, image_value, status}`). The **harness** compares values (same N2 normalization), not
  the VLM's say-so. Malformed / uncertain / unreadable → reject.
- **Grid-size cap:** a VLM cannot reliably check ~500 cells. For a huge AMBIGUOUS table, skip the
  VLM and go straight to the **image floor** — never trust a model to spot one swapped digit in a
  sea of numbers.

## No-silent-loss floor

- The VLM never overrides a deterministic CERTAIN_FAIL.
- Born-digital: every candidate numeric token must exist in the table region's native tokens (hard
  subset guard) unless the page is routed to image.
- Image floor whenever nothing can verify. Scanned tables that pass the VLM are marked
  lower-confidence (`vlm_verified_scan`), never claimed as exactly verified.

## Cost / reproducibility

VLM fires only on AMBIGUOUS / scanned pages (bounded). `temperature=0`. Cache the verdict on
`hash(crop_bytes + markdown)` so replay is stable. `--strict-local`: local qwen only; if
unavailable, exact passes ship, certain-fails go to image, ambiguous/scanned go to image (no cloud).

## Build order

1. **Tri-state router + per-region deterministic scoping** (THIS IS NEXT). Scope the existing
   `_value_guard` to each table region instead of whole-page. Ships CE deterministically — the win
   chased all session. No model. → ticket **TR-6**.
2. **VLM table-judge layer** (cropped, adversarial, structured, grid-capped) for AMBIGUOUS-bbox +
   SCANNED pages — folds the "hard cases" roadmap (scanned, exotic layouts) into this architecture.
   → ticket **TR-7**.

Biggest residual failure mode (both panelists): **bad table localization** — if the region bbox /
crop captures the wrong table or omits a row/column, both the deterministic region check and the
VLM validate the wrong thing. Mitigation: make bbox trust a gate; require the VLM to confirm table
identity + no cropped-out rows; full-page image floor when bbox trust isn't established.
