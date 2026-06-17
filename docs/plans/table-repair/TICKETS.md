# TICKETS — reliable dense-table extraction (table-repair)

Canonical backlog for the **table-repair** initiative (GH-56). Design rationale + the
`/consilium` verdict live in `docs/log/2026-06-17_gh56-table-repair-design.md`; live execution
state in `STATUS.md`.

Status keys: `READY`, `NEEDS-DESIGN`, `BLOCKED`, `WIP`, `DONE`, `DEFERRED`.

## v1 scope (adopted from the panel verdict, section 6 of the design note)

**v1 is entirely DETERMINISTIC — no model in the repair or segmentation path.**

> per-region segmentation (fixes the monolithic-verifier root cause) → deterministic rowizer
> (Option B) recovers clean grids at zero model cost → D3 fail-closed floor catches whatever B
> can't.

The VLM only enters in **v2** (TR-4 A2 re-ask, TR-5 S3 VLM confirm/split), once we have
measured whether deterministic per-region + rowizer already yields clean CE grids.

## Dispatch rules

- One implementation ticket per `socr-implementer`; disjoint write set; `socr-reviewer` pass
  before acceptance. Commit on the initiative branch (`STATUS.md`), stage by name, never
  `git add -A`, one commit per ticket, do not push, **wait for CI green before merge**.
- TR-1, TR-2, TR-3 all touch `born_digital.py` / `orchestrator.py` regions — they are
  **serialized** by their dependency chain; never run concurrently.
- `uv run` or `~/venvs/socr/bin/*`. Never `python script.py`.
- Hit an architectural fork you can't resolve from the ticket → stop, return `CONSILIUM-GATE`
  with a one-sentence question.

Line numbers are approximate; method names are the stable anchors.

---

## TR-0 — License-clean CE-like fixture + cell-by-cell parity harness

GitHub: https://github.com/r-uben/socr/issues/56
Status: **DONE** · Priority: P0 · Agent: `socr-implementer` · Depends on: none · Wave 1

### Problem
No falsifiable acceptance gate exists. The design note's headline criterion ("better than native
on dense tables") is unfalsifiable until there is a frozen fixture with known ground-truth cells.
The real CE PDF is **licensed and cannot be committed** (see global google-drive/licensed-data
rule) — so the fixture must be a **synthesized born-digital page** that reproduces CE p.4's
structure with KNOWN values.

### Plan
1. Build a fixture generator (PyMuPDF `insert_text` at explicit coords, **no ruling lines** — the
   line-less, whitespace-gutter layout is the point) producing a born-digital page with: a main
   multi-schema numeric grid (N forecaster rows × C indicator×year columns, with `na` blanks and
   at least one short/ragged source row), a SECOND small table with a different schema, a vector
   **chart** region, and a prose text box. Emit the ground-truth as a committed JSON sidecar.
2. A parity helper `assert_table_parity(extracted_md, ground_truth_json)` that compares **per
   cell** (row label × column → value), reporting the exact mismatching cells, and asserts each
   region is a SEPARATE markdown grid + the chart is an image asset (no transcribed values) + the
   text box is prose, in reading order.
3. A test that runs the agentic path on the fixture and asserts parity — `xfail` today (collapsed
   output), to be flipped green by TR-1…TR-3.

### Write ownership
`tests/fixtures/table_repair/` (PDF generator + ground-truth JSON); `tests/test_table_repair_parity.py`.

### Acceptance
- The fixture is born-digital (has a real text layer) and **license-clean** (synthesized, not the
  CE PDF). The ground-truth JSON enumerates every expected cell.
- `assert_table_parity` fails on today's collapsed output with a precise per-cell diff.
- Hermetic (no ollama/provider): if it drives `_phase_agentic`, it patches
  `_available_engines_for_agentic` (CI has no provider).

### Verification
- `~/venvs/socr/bin/pytest tests/test_table_repair_parity.py -q`
- `~/venvs/socr/bin/ruff check tests/test_table_repair_parity.py`

---

## TR-1 — Deterministic rowizer for lane-stacked `find_tables()` regions (Option B)

GitHub: https://github.com/r-uben/socr/issues/56
Status: **DONE** · Priority: P0 · Agent: `socr-implementer` · Depends on: TR-0 · Wave 1

### Problem
`extract_structured()` trusts `page.find_tables()` whenever it returns ANY non-empty region
(`born_digital.py:~712`) and only reaches the word-geometry rowizer when `find_tables()` returns
nothing (`born_digital.py:~728`). A non-empty-but-lane-stacked region (CE's whitespace-gutter
grid) therefore never reaches the rowizer; `_table_to_markdown` (`born_digital.py:~813`) preserves
the embedded newlines and emits the collapsed cell. This is the highest-leverage, lowest-risk,
zero-cost, fully deterministic fix.

### Plan
1. Detect a lane-stacked / collapsed `find_tables()` region (a cell containing newline-stacked
   tokens, or output column count ≪ the region's native x-lane count — reuse the verifier's lane
   signal, no new magic threshold).
2. For such a region, run `reconstruct_table_regions()` / the word-geometry rowizer on the
   region's words instead of `_table_to_markdown`'s passthrough.
3. Preserve `na` blanks from geometry (a missing token in a lane is a blank cell, not a skip).

### Write ownership
`src/socr/core/born_digital.py` (the `extract_structured` find_tables→rowizer gate +
`_table_to_markdown` region path); `src/socr/tables/reconstruct.py` only if the rowizer needs a
region-scoped entry point.

### Acceptance
- On the TR-0 fixture, a single-schema lane-stacked region rowizes to a correct grid (parity
  passes for that region).
- Investigation recorded: does `find_tables()` return CE's two tables as SEPARATE regions or one
  merged region? If merged, the multi-schema split is **not** in TR-1 — file/flag TR-2's split
  sub-task; do NOT silently rowize a merged multi-schema region into a wrong grid.
- No regression on existing born-digital / reconstruct tests.

### Verification
- `~/venvs/socr/bin/pytest tests/test_reconstruct.py tests/test_born_digital*.py tests/test_table_repair_parity.py -q`
- `~/venvs/socr/bin/ruff check src/socr/core/born_digital.py src/socr/tables/reconstruct.py`

---

## TR-2 — Per-region verifier scoping + reading-order reassembly

GitHub: https://github.com/r-uben/socr/issues/56
Status: **DONE** · Priority: P0 · Agent: `socr-implementer` · Depends on: TR-1 · Wave 2

### Problem
The native verifier counts lanes over the WHOLE page (`native_verifier.py:~114-148`): it compared
CE p.4's 27 native lanes against one merged 11-column output and fired `geometry_impossible_collapse`
even though each individual table is internally consistent. Reassembly is `y0`-only
(`born_digital.py:~737`), which scrambles multi-column layouts.

### Plan
1. **Spike (socr-designer):** resolve TR-1's open question — when `find_tables()` over-merges two
   schemas into one region, what is the deterministic split signal (blank-row gap? lane-structure
   change? second header detection?)? If no robust deterministic split exists, this is the v1/v2
   boundary — escalate (the VLM-split is TR-5/v2).
2. Verify each table region against its **own** lanes, not the page total.
3. Deterministic **token-coverage post-check**: every native numeric token lands in exactly one
   region (no orphaned/double-counted token).
4. Column-aware reading-order reassembly; chart/figure regions → existing image-asset lane
   (`has_chart_marks`, `_is_chart_asset_page` `orchestrator.py:~1128`, chart_asset output
   `orchestrator.py:~1608-1635`).

### Write ownership
`src/socr/tables/native_verifier.py` (per-region scoping); `src/socr/pipeline/orchestrator.py`
(per-region table phase + reassembly — distinct method region from TR-3); `born_digital.py` reading
-order only (coordinate with TR-1's owner — serialize).

### Acceptance
- The TR-0 fixture's two tables each verify against their OWN lanes (no false
  `geometry_impossible_collapse`); token-coverage post-check passes.
- Reading order correct on the multi-column fixture; chart → image asset (no transcribed values).

### Verification
- `~/venvs/socr/bin/pytest tests/test_table_repair_parity.py tests/ -k "verifier or reconstruct or chart" -q`

---

## TR-3 — D3 fail-closed floor + selection-policy fix

GitHub: https://github.com/r-uben/socr/issues/56
Status: **DONE** · Priority: P0 · Agent: `socr-implementer` · Depends on: TR-2 · Wave 3

### Problem
When a table region still fails verification, `_winning_page_output` ships the collapsed native
text (it prefers native fallback before a failed `best_output`, `manifest.py:~257-276`) — the
plausible-but-wrong artifact. The panel's Q1 verdict is **D3: ship neither flawed table**.

### Plan
1. A table region that fails verification (after TR-1 rowizer + TR-2 per-region check) ships an
   **explicit failed-table marker** for that region and routes it to the **image-asset lane**
   (rendered PNG, no transcribed values) — never collapsed-native, never ragged row-major.
2. Update the per-region selection so a flagged region does not silently revert to collapsed
   native; emit a **distinct audit event** ("table_region_unverifiable → image-asset") vs the
   existing `native_fallback`.
3. Per-region status flags propagate to page/document status + metadata + CLI (no-silent-loss:
   surface at every level).

### Write ownership
`src/socr/core/manifest.py` (`_winning_page_output` region selection); `src/socr/pipeline/orchestrator.py`
(D3 image-asset routing for a failed region — distinct method region from TR-2);
`src/socr/core/audit_log.py` (new event kind).

### Acceptance
- On a deliberately unverifiable region, the output ships a failed-table marker + PNG ref, NOT a
  collapsed/ragged markdown table; a distinct audit event is recorded; document status is demoted.
- The TR-0 parity test passes end-to-end for the verifiable regions and fails-closed for any
  unverifiable one (no wrong cells ever emitted).

### Verification
- `~/venvs/socr/bin/pytest tests/test_table_repair_parity.py tests/test_silent_content_destruction.py tests/test_manifest_agentic.py -q`
- `~/venvs/socr/bin/ruff check src/socr/core/manifest.py src/socr/pipeline/orchestrator.py src/socr/core/audit_log.py`

---

## v2 — UNBLOCKED by real-CE evidence (2026-06-17)

The measured trigger the panel required has fired: deterministic v1 (TR-1…TR-3) gets the **values
and columns right on real CE but breaks the ROW structure** (top block merged, names offset from
values by one row — gap-segmentation can't handle CE's dense/variable spacing). See
`docs/log/2026-06-17_real-CE-v1-finding.md`. v2 brings the VLM in for STRUCTURE, geometry-guarded
for VALUES.

---

## TR-4a — Real-CE-geometry fixture (the failing gate) — DO FIRST

GitHub: https://github.com/r-uben/socr/issues/56
Status: **DONE** · Priority: P0 · Agent: `socr-implementer` · Depends on: none

### Problem
The TR-0 fixture used clean, evenly-spaced single-line rows → it went green while real CE stays
broken (false confidence). v2 needs a fixture that REPRODUCES real CE's row geometry so "green"
means green.

### Plan
1. From a real CE page (e.g. `202401.pdf` p.4), extract the **actual row y-positions and x-lane
   positions** (geometry only). License-clean: copy the COORDINATES, synthesize fake forecaster
   names + numbers placed at those real positions — never commit CE text/numbers.
2. Regenerate the fixture (or add a second `ce_like_p4_dense.pdf`) reproducing: the tight top
   block, the small name/value vertical offset, the summary rows (Consensus/High/Low). Update
   ground-truth.
3. The e2e parity test must now FAIL (`xfail(strict=True)`, or a new dense-fixture test) —
   reproducing the real-CE row-merge/offset failure. This is TR-4's acceptance gate.

### Write ownership
`tests/fixtures/table_repair/` + `tests/test_table_repair_parity.py`. No `src/` changes.

---

## TR-4 — Value-guarded VLM-for-structure (NEEDS-DESIGN → then implement)

GitHub: https://github.com/r-uben/socr/issues/56
Status: **NEEDS-DESIGN** · Priority: P0 · Agent: `socr-designer` then `socr-implementer` ·
Depends on: TR-4a

### Goal
When the deterministic rowizer's output fails its verifier (real CE) — or there is no geometry
(scanned page) — escalate to the VLM for the table STRUCTURE, accept it ONLY if geometry confirms
the VALUES, else fail closed to the image (D3). The rowizer stays the free first pass; the VLM is
the escalation.

### Design questions (socr-designer pass)
1. **Escalation trigger + where it wires.** The VLM fires when `native_table_verifier` hard-fails
   the rowized grid (the real-CE path) OR the page is not born-digital. Where in the agentic ladder
   does this sit, and how does it relate to TR-3's D3 floor?
2. **The value-guard (the crux).** The VLM proposes rows×columns; accept ONLY if its numeric-token
   multiset, **lane-aware**, EQUALS the page's native numeric tokens (born-digital) — no invented,
   no dropped, no shifted. Specify the exact check. For SCANNED pages there is no native-token net —
   what is the accept criterion (lower-confidence ship + flag)?
3. **Prompt constraint without magic numbers.** Feed native geometry (N rows, C columns, row
   labels) as a hard constraint; derive N/C from the page, not a constant.
4. **Fail-closed integration.** A VLM structure that fails the value-guard must route to the TR-3
   D3 image floor. Confirm D3 covers the agentic verifier path (the gap real CE exposed: D3 didn't
   fire because the failure was on the agentic `NativeTableVerifierJudge` path, not born-digital
   `_verify_regions`).

### Then implement against the TR-4a failing gate.

---

## TR-5 — S3 VLM confirm/split for segmentation (DEFERRED to v2)

Status: **DEFERRED** (v2) · Depends on: TR-2 spike outcome (only needed if deterministic
geometry-led splitting proves insufficient).

The full S3 hybrid: geometry proposes regions, the VLM confirms/splits over-merged or ambiguous
regions, with the deterministic token-coverage post-check from TR-2 as the safety net.
