# TICKETS — GH-151 structure lost at full recall

Status keys: `TODO` · `WIP` · `DONE` · `BLOCKED`. `depends-on` gates dispatch.

Context: `2024__bauer_pflueger_sunderam` p26 ships at **100% word recall with 0
tokens missing** and an unusable table — spanning headers in body cells, `R2` and
its values on different rows, coefficients unbound from their standard errors.
This falsifies word recall as a sufficient routing gate (proposed on GH-49):
recall measures TOKEN loss and is blind to STRUCTURAL loss.

## Stream A — structural signals (all deterministic, no model)

### TICKET-A1 — grid-shape checks · TODO · depends-on: none · wave 1
**Problem:** Nothing detects a grid whose rows disagree on width or that is mostly empty.
**Do:** Add pure functions over a parsed grid: row-width consistency, empty-cell
density, orphan rows (empty label cell with populated neighbours). Return a report
object, not a boolean — the caller decides policy.
**Files:** `src/socr/tables/structure_check.py` (new)
**Done when:** `~/venvs/socr/bin/pytest tests/test_structure_check_gh151.py -q` exits 0; the p26 grid (checked in as a fixture string) is reported defective and a clean 3x4 grid is not.

### TICKET-A2 — x-position binding check · SUPERSEDED by A2R · depends-on: none · wave 1
**Superseded 2026-08-13.** Implemented as draft PR #184 and **not merged**; the PR is to be
closed, not revived. Its verdict bit (`BindingReport.failed`) is derived from **modal
consensus**: `modal_sig = min(sig for sig, c in counts.items() if c == best_count)`, then
every row whose signature differs is recorded `misbound`. That cannot distinguish "deviant
because malformed" from "deviant because genuinely different", and in a citation corpus the
minority rows — a wrapped header band, a totals row, a merged cell — are frequently the ones
that matter. The `ordinal_mismatch` predicate does not rescue it: it is computed *from*
`modal_single_lane`, so it is downstream of the same vote, and it is gated on
`value_col_count == len(centres)` — precisely the equality that stops holding on a damaged
page, so it disables itself where it is needed. Worse, when the modal lane equals `col_idx`
the ordinal check is skipped entirely, so a correctly-bound majority hides a minority
misbind.

The underlying idea survives; the implementation does not. Refiled as **A2R** below.

### TICKET-A2R — absolute per-row binding checks · TODO · depends-on: none · wave 4+
**Problem:** A value sitting under the wrong column lane is detectable from native geometry
for free — but not by asking what most rows do.
**Do:** Two per-row predicates that need no clustering and no vote: **x-order inversion**
(within a row, emitted column order must not contradict native left-to-right x order) and
**intra-row lane collision** (two values of the same row resolving to the same native lane).
Both are decidable inside one row against raw coordinates. Carry forward #184's reusable
plumbing (pairing, exactly-once resolve); do not carry forward `BindingReport.failed`.
**Files:** `src/socr/tables/native_verifier.py`
**Done when:** a synthetic page whose grid has one value shifted a column reports a binding failure; the same page with the correct grid reports none; and no predicate references any other row's signature.
**Not a B1 dependency.** B1 ships without it — see B1's rewrite below.

## Stream B — consequence

### TICKET-B1 — surface structural failure at page level · DONE (shipped as an escalation gate per the 2026-08-15 panel ruling, feat/200-structural-escalation-gate) · depends-on: A1, GH-147 A2 · wave 3
**Problem:** A defect nothing consumes is not a gate. A page whose emitted table is
structurally broken ships as trusted native SUCCESS.

⚠️ **Rewritten 2026-08-13 after a design pass (Opus) and two independent critiques (Fable,
Grok-4.6).** Design note: `docs/log/2026-08-13_gh151-b1-design.md`. Four things changed and
an implementer must read this block before the Do.

**1. A2 is no longer a dependency.** The gate keys on **shape evidence alone**: A1's
`GridStructureReport` (`src/socr/tables/structure_check.py`, merged `d4f6154`) narrowed to
`ragged`, plus one new predicate below. `orphan_rows` is explicitly **excluded from the
gate** (it stays diagnostic): "blank label + any values" is exactly a legitimate
standard-error / t-statistic continuation row, which is why the unnarrowed report fires on
27 of 29 real native table blocks.

**2. New predicate — `FINDING_DETACHED_LABEL`.** A body row carrying a label and zero values,
immediately followed by a body row carrying values and no label, is one physically split row.
This is a **row-pair adjacency invariant** — decidable from one adjacent pair, containing no
numeric constant, and referencing no other row's signature. It is *not* unary; do not
describe it as "no reference to any other row" (the design note says this in one place and
contradicts itself in another). The constraint that binds is **no modal vote**, and this
satisfies it.

**3. The acceptance criterion is rewritten because the old one was not executable.** There is
no p26 PDF fixture — the only in-repo artifact is the `GH151_P26_MD` string at
`tests/test_structure_check_gh151.py:16-25`. Worse, after GH-144 A2/A2b the live motivating
page yields **zero** `find_table_blocks` on current `main`: it is now a flat text stream, not
a grid. **B1 is a markdown-grid gate and will not fire on that page. That is accepted and is
an explicit non-goal** — recovering it belongs to the flat-stream path, not to this gate.

**4. The fail-closed precedent named in the original ticket was the wrong one.** The old text
said mirror `native_table_unverifiable`; that is the D3 floor, which *deletes* the table and
ships a marker (`manifest.py:310-332`) — it would destroy a mostly-usable table over one
split row. Mirror `native_table_structure_failed` / `needs_repair` instead
(`state.py:68-89`, `manifest.py:340-350`): force repair, ship flagged native if repair fails.

**Do:**
1. Add `FINDING_DETACHED_LABEL` to `src/socr/tables/structure_check.py` and surface it on
   `GridStructureReport` (see the write-set note below).
2. Gate on `ragged` OR `detached_label_rows` across `check_markdown(...)` reports — never on
   `GridStructureReport.defective` as it stands.
3. Set a new `PageState` field `native_table_structure_defective` at analyze time, following
   the `PageAssessment.native_table_lane_refused` precedent from GH-147 A2 (merged
   `13033a3`): set at the moment the condition is detected, never re-derived downstream.
   A new field is required rather than reusing `native_table_structure_failed`, because
   `_score_per_page` **clears** that flag on a passing heuristic (`orchestrator.py:3702`,
   `:3727`) and would silently erase the structural verdict.
4. Emit an `AuditEvent` of kind `table_structure_failed`, keyed on the flag.
5. Surface at page status, document status, metadata **and** CLI — all four, per the house
   rule. Thread the new field through the sidecar and resume paths
   (`orchestrator.py:4282-4293`, `:4471-4477`) and `_clear_fail_closed_flags` (`:1810-1815`).

**Files:** `src/socr/core/born_digital.py`, `src/socr/core/state.py`,
`src/socr/pipeline/orchestrator.py`, `src/socr/tables/structure_check.py`,
`src/socr/core/manifest.py`, `tests/`.

**Write-set note — two amendments to the ownership table in
`docs/plans/extraction-defects/STATUS.md`, both to be made in the same commit:**
- `structure_check.py` is marked wave-1-only. Wave 1 is closed and nothing else claims it.
- **`manifest.py` must be added.** `_winning_page_output` only ORs the flags it knows
  (`manifest.py:340-350`). A new field the manifest does not read re-stamps
  `audit_passed=True` — this is the PP-7-R1 bug shape and would make the whole gate inert.

**Done when:**
- A synthetic PDF that `extract_structured` genuinely splits into a label-row/values-row pair
  yields an `AuditEvent` of kind `table_structure_failed`, and the page is not
  `audit_passed=True`.
- A seam test over the `GH151_P26_MD` string asserts the predicate fires on it.
- **Negative controls, required:** a legitimate standard-error / t-statistic continuation row
  does **not** fire; a group-heading row followed by an unlabelled column-number row does
  **not** fire; a footnote line mangled into a grid is classified deliberately, not by
  accident.
- Full suite passes; `uvx ruff@0.16.0 format --check .` clean.

**Known cost, accepted:** the narrowed predicate marks ~20 of 29 real regression-table blocks
defective. In agentic mode those pages already route to OCR (`orchestrator.py:1210`), so the
marginal effect there is only that the fallback lane can no longer silently ship a severed
grid as passed. The behaviour change lands on `--native-only` and non-agentic runs.

**SETTLED 2026-08-13 — the gate does NOT override `--native-only`.** Ruled independently by
Grok-4.6 (F8) and GPT, and ratified by the owner. `--native-only` is a documented contract,
not a hint: `cli.py:54-61` says born-digital pages are never OCR-enhanced "even with known
deficiencies", and `config.py:112-125` repeats it. Overriding would contradict socr's own
stated behaviour. GH-147 A2 is **not** the precedent — it refuses the native lane *before*
markdown exists (a rotated rowizer emits a transposed grid, garbage by construction), whereas
B1 would override an explicit user flag on a grid that does exist, on the strength of an
inferential row-pair predicate that can false-positive.

The decisive argument is not throughput: rerouting does not buy correctness. B1 does not
re-run its structural check over the VLM winner, so an override trades a conspicuously
flagged native grid for an unchecked, plausible-but-wrong one — and in a citation corpus
wrong numbers that look fine are the actual red line. If output-level safety is wanted later,
the principled answer is a fail-closed marker lane, not breaking the flag.

**BINDING CONSTRAINT — honouring the flag is only safe if the failure is loud end to end.**
An implementer must deliver all of:
- shipped page `WARNING` / `audit_passed=False`;
- document status `AUDIT_FAILED`;
- the flag persisted through the sidecar and manifest, and surviving resume;
- a `table_structure_failed` `AuditEvent`;
- visible CLI reporting.

And specifically: **the new defect flag must NOT feed an unconditional `needs_repair` branch
under `native_only`.** If it does, the implementation silently reroutes anyway and this
ruling is undone in code. If any of these surfaces cannot be guaranteed, stop and escalate —
without them, honouring the flag is not safe and the question reopens.

**Prevalence caveat:** the 20/29 figure is a measured *cost*, not a verified true-positive
rate. It was self-graded by the design pass over four finance/macro papers. Do not cite it as
a defect rate without independent re-inspection.

**SHIPPED 2026-08-15 as an escalation gate, per the panel ruling recorded in
`docs/plans/STATUS.md` context passed at dispatch and `docs/log/2026-08-15_tr3-hand-judgement.md`.**
The plumbing this ticket built (the `PageState` field, the `table_structure_failed` audit
event, the manifest D3-floor widening, `TABLE_DISTRUST_KINDS` membership, and
`structural_gate_fires` as the shared entry point) is unchanged. What changed:

1. B1's own predicate (`ragged OR detached_label_rows`) is kept exactly as designed — TR-3
   (numeric-token multiset) turned out to be blind to 3 of 5 defect classes the 2026-08-15
   hand judgement found (header loss, detached labels, star-only row deletion), so the plan
   to swap B1's predicate for TR-3 was abandoned. TR-3 and the shape gate are a disjunction,
   not a replacement of one by the other — measured non-overlap in
   `docs/log/2026-08-14_gh151-b1-predicate-design.md:225-239` (35/66 overlap, 27 pages the
   shape gate misses entirely).
2. A third disjunctive term, header-attribution (`src/socr/tables/header_attribution.py`),
   closes the header-loss gap TR-3 cannot see. `HARD` verdict only rejects; `SOFT` (tokens
   present but mis-columned) is recorded, never rejects — see
   `docs/log/2026-08-15_200-open-decision-1-resolved.md` for the pre-merge measurement that
   kept it advisory (1-2 of 3 classifiable damaged pages were mis-columned, under the
   promotion threshold).
3. The winner-side hole is closed: `NativeTableVerifierJudge.assess`
   (`src/socr/pipeline/agentic.py`) now runs the same structural/header check on whatever is
   ABOUT TO SHIP — including the `EXACT_PASS` accept path, which previously shipped a
   numerically-perfect, structurally-destroyed table at `confidence=1.0` without ever
   consulting the inner judge.
4. `--native-only` still never reroutes — record and surface only, following the existing
   settled ruling in this same file above. At the top rung with nowhere left to escalate, the
   D3 fail-closed floor (`manifest.py`) widened identically for the header-only defect.

Implemented on `feat/200-structural-escalation-gate` (rebased onto current `main`, 3 commits
ahead of the `feat/151-b1-structural-gate` base). Full suite 1648 passed / 1 xfailed;
`uvx ruff@0.16.0 format --check .` clean.

**Known follow-up, not blocking:** the firing rate of `table_output_defect` on VLM-produced
(non-native) accepted output is unmeasured — the 26.9%/35/66/27 figures in
`docs/log/2026-08-14_gh151-b1-predicate-design.md` were all measured on native markdown, a
different population. Recommended before wide rollout: run `table_output_defect` over an
existing agentic corpus run's accepted outputs and report the rate.

### TICKET-B2 — record the gate correction on GH-49 · TODO · depends-on: B1 · wave 3
**Problem:** GH-49 currently carries my claim that word recall is the routing signal.
p26 disproves sufficiency; the design note must not stay wrong.
**Do:** Comment on GH-49 with the p26 evidence and the revised rule: recall (token
loss) AND structure (binding/shape) together gate escalation.
**Files:** none (issue comment)
**Done when:** the comment exists and names p26 with its 100%-recall figure.
