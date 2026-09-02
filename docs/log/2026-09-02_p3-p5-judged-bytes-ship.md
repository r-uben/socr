# Decision Log — P3 + P5: Judged bytes are shipped bytes & dual-pass table reread on signal

**Date:** 2026-09-02  
**Ruling Reference:** [Conceptual revision — what socr is, and the shortest path to it](2026-09-01_conceptual-revision.md), programme items P3 and P5.

---

## 1. Summary of Decisions & Intent

1. **P3 — Judged Bytes Are Shipped Bytes:**
   - Shipped bytes must equal judged bytes across all pipeline flush and assemble stages.
   - Removed every post-route mutation and post-route recheck in `_phase_agentic` that caused accepted bytes to differ from shipped bytes.
   - Preserved table header repair strictly *inside* `NativeTableVerifierJudge` before verdict evaluation, ensuring evaluated candidate text is identical to accepted text.
   - Factored early-delegation geometry-bypass exits in `NativeTableVerifierJudge` through `_apply_structural_gate` (string-only), closing the single unique coverage gap left by deleting the post-route recheck.
   - Retired `post_route_recheck` rather than silently renaming it; audit events and status derivations remain intact for unchanged-route pages.

2. **P5 — Signal-Gated, Opt-In Dual-Pass Table Escalation:**
   - Dual-pass table crop reread (`--dual-pass-tables`) is default **off** (`PipelineConfig.dual_pass_tables = False`).
   - When enabled, crop reread never runs as a trunk pass on every accepted table page; it runs strictly as an escalation tool triggered by a table-verification or routing signal before the escalation/table terminal verdict.
   - Extractor construction is lazy and memoized per document, preventing unneeded model resolution and live probes on clean/flag-off runs.
   - Works across both provider states (escalation provider present vs absent/local-only).
   - `auto_patch_tables` remains conditionally recorded in `_run_fingerprint` when `dual_pass_tables=True` and ignored when `dual_pass_tables=False` (where the patch lane is unreachable).

---

## 2. P3 Deletions & Seam Relocations

### Deleted Blocks in Orchestrator (`src/socr/pipeline/orchestrator.py`)
Both post-route mutation blocks immediately following `route_page` in `_phase_agentic` were deleted:
1. **GH-56 Deterministic Header Repair block:** Previously called `repair_table_headers_on_page` after `route_page` on `ps.best_output.text`, mutating accepted text post-verdict.
2. **GH-200 String-Only Structural Recheck block (`post_route_recheck`):** Evaluated `table_output_defect` post-route and demoted selected outputs in place (`data.site == "post_route_recheck"`) without rerouting.

### Preserved In-Judge Table Header Repair
- `NativeTableVerifierJudge._maybe_repair_collapsed_headers` remains active inside the judge before assess renders an `AcceptDecision`.
- Evaluates candidate text for born-digital classified table pages with valid PDF geometry.
- Emits the `table_header_repair` audit event before the verdict, and passes the repaired candidate text to `_apply_structural_gate`.
- Repaired text is what is accepted, journaled, provisionally flushed, rewritten, and shipped.

### Early-Delegation String-Only Structural Gate Coverage
- In `NativeTableVerifierJudge.assess` (`src/socr/pipeline/agentic.py`), early-delegation exits (non-table page, missing `get_fitz_page`, non-success status, or empty text) previously bypassed `_apply_structural_gate`.
- Factored early-delegation exits so accepting inner decisions pass through `_apply_structural_gate(decision, output, page_num, words=None, rules=None)` (string-only evaluation, matching the deleted recheck's exact term set) before returning.
- Rejecting inner decisions return unchanged without emitting spurious structural events.

### Deliberately Removed Post-Verdict Behaviors
- **Header repair on rejected/best-effort outputs:** Deliberately removed because mutating text that failed acceptance violates the principle that unaccepted outputs are never modified after the ladder exhausts.
- **Header repair on non-table-classified pages:** Deliberately removed because running native column geometry repair without verified table classification risks corrupting non-table layouts.
- **`post_route_recheck` site:** Retired completely from production and test assertions; structural failures emit `table_structure_failed` via `_apply_structural_gate` prior to acceptance.
- **Document-status derivation:** Pages with unchanged routing retain standard status derivations; `audit_passed` is never toggled post-route merely to flag a page.

---

## 3. P5 Configuration, CLI & Orchestration Semantics

### Default & CLI Semantics
- `PipelineConfig.dual_pass_tables: bool = False` (default off).
- CLI commands (`process` and `batch`) expose paired `--dual-pass-tables/--no-dual-pass-tables` option with `default=None`, preserving configuration file and profile precedence.
- CLI `--auto-patch-tables` help clarifies that cell mutations occur only within an enabled, signal-triggered dual-pass reread.

### Signal Sources
Table escalation signal is evaluated once per eligible page using exact facts already computed by routing:
1. Calibrated table score via `_table_page_needs_escalation(state, page_num, ps, ...)`.
2. Explicit route escalation evidence: a rejected native verifier attempt followed by another attempt, or ladder exhaustion on a table page (`_route_page_table_escalation_signal(decision, ladder)`).
No broad frozenset of advisory events is turned into routing policy.

### Lazy Extractor Construction
- `_get_table_extractor()` closure is memoized once per document.
- `_resolve_crop_vlm_model` and `TableCropExtractor` instantiation occur only when `dual_pass_tables=True` AND a table escalation signal fires.
- Prevents probing Ollama / VLM endpoints during CI runs with the flag off or on clean documents.
- Missing model or construction failure fails open cleanly and is memoized to avoid repeated attempts.

### Provider State Independence
- Fired escalation signals trigger optional crop reread in both provider states:
  - When a cloud escalation provider profile is available (`_escalation_profile` present).
  - When running local-only with no cloud escalation provider (`_escalation_profile` None).
- Crop reread occurs in the escalation sequence before provider candidate acceptance/rejection and before the table-judge terminal verdict.

### Fingerprint Bookkeeping
- `_run_fingerprint` records `"auto_patch_tables": cfg.auto_patch_tables if cfg.dual_pass_tables else None`.
- When `dual_pass_tables=False`, the reread/patch lane is unreachable, so toggling `auto_patch_tables` leaves the fingerprint unchanged. When `dual_pass_tables=True`, the patch lane is reachable upon a fired signal, so toggling `auto_patch_tables` invalidates the cached fingerprint.

---

## 4. Code Additions and Deletions

Executed reproducible command:
```bash
git diff --numstat -- src/socr/pipeline/agentic.py src/socr/pipeline/orchestrator.py src/socr/core/config.py src/socr/cli.py tests docs/ARCHITECTURE.md
```

### Reproducible Output:
```text
5	2	docs/ARCHITECTURE.md
13	8	src/socr/cli.py
9	6	src/socr/core/config.py
26	12	src/socr/pipeline/agentic.py
135	152	src/socr/pipeline/orchestrator.py
79	0	tests/test_agentic.py
5	2	tests/test_dual_pass_tables.py
14	3	tests/test_gh190_empty_table_surfacing.py
295	4	tests/test_native_table_verifier.py
6	2	tests/test_r174b_config_schema.py
89	79	tests/test_structural_gate_b1_gh151.py
```

Untracked test files created:
- `tests/test_p3_judged_bytes_ship.py` (248 lines)
- `tests/test_p5_dual_pass_cli.py` (138 lines)
- `tests/test_p5_reread_on_signal.py` (367 lines)

---

## 5. Regression Test Suites

### New Regression Suites
- `tests/test_p3_judged_bytes_ship.py`:
  - Captures accepted text dynamically at `NativeTableVerifierJudge.assess` boundary.
  - Verifies exact byte equality across provisional flush bodies, authoritative `pages/00001.md` rewrite, and final stitched Markdown.
  - Asserts absence of `post_route_recheck` audit events.
- `tests/test_p5_reread_on_signal.py`:
  - Tests difference contract: clean page flag-off (0 reread calls), clean page flag-on (0 reread calls), fired signal flag-on (1 reread call).
  - Verifies lazy resolution of `_resolve_crop_vlm_model` and `TableCropExtractor`.
  - Exercises both provider states (escalation provider present vs absent).
- `tests/test_p5_dual_pass_cli.py`:
  - CLI runner and `build_config` tests covering default off, explicit `--dual-pass-tables`, explicit `--no-dual-pass-tables`, YAML default preservation, and CLI override of YAML configuration.

### Rewritten / Enhanced Tests
- `tests/test_native_table_verifier.py`: Added direct geometry-bypass tests covering non-table pages, missing `get_fitz_page`, error status, and empty text through `_apply_structural_gate`.
- `tests/test_structural_gate_b1_gh151.py`: Rewrote `TestPostRouteHeaderRepairRecheck` as a direct pre-verdict `NativeTableVerifierJudge` regression; verified `post_route_recheck` site retirement.
- `tests/test_agentic.py`: Added `test_structural_gate_covers_geometry_bypass_paths`.
- `tests/test_dual_pass_tables.py`: Updated default flag assertions to `False`.
- `tests/test_r174b_config_schema.py`: Updated schema default assertions to `False` while preserving explicit `True` round-trip deserialization.
- `tests/test_gh190_empty_table_surfacing.py`: Updated paired pipeline test to exercise the judge chain with post-route recheck removed.

---

## 6. Full Suite and Quality Gate Results

### Full Test Suite Run
```bash
PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-p35/src ~/venvs/socr/bin/pytest tests/ -q
```
**Result:**
```text
3010 passed, 4 xfailed, 5 warnings in 103.74s (0:01:43)
```

### Format Gate
```bash
uvx ruff@0.16.0 format --check .
```
**Result:**
```text
484 files already formatted
```
(Exit code: 0)

### Provenance Check
```bash
PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-p35/src uv run python -c 'import socr; print(socr.__file__)'
```
**Result:**
```text
/Users/rubenffuertes/repos/tools/socr-p35/src/socr/__init__.py
```

---

## 7. Cold review round 1

Cold reviewer report: `.troupe/runs/20260902-173952-standard-feature/outbox/review.md`
(worker `r173952-review-1`). Three findings, all worked reproducer-first: the failing
test was written and confirmed red against this tree before any source edit. The
reproducers live in **`tests/test_p35_cold_review_round1.py`** (9 tests; 5 of them were
red before the fixes, 4 are controls/converses that were already green).

### Finding 1 (medium) — the crop reread still shipped unjudged bytes. REPRODUCED.

`_reread_page_tables` assigns `bo.text = result.text` (`orchestrator.py`, "Patch and emit
AuditEvents OUTSIDE any try/except"), and P5 newly invokes it from the signal path, which
runs *after* `route_page` has accepted. So the moment P5's signal fired, P3's invariant
reopened on the path P5 added. `tests/test_p3_judged_bytes_ship.py` could not see it: it
runs with `dual_pass_tables=False`.

Red before the fix:
`TestJudgedBytesOnTheSignalPath::test_fired_signal_ships_the_judge_accepted_text[refused]`
shipped the patched text (`9.9`) where the judge had accepted `3.7`; the `[accepted]`
case shipped a caption line the judge never saw.

**Changed.** New `UnifiedPipeline._rejudge_crop_patched_page`. When the reread changed the
page's text, the patched text is treated as a NEW CANDIDATE and goes back through the same
`judge` object, with the provider profile whose reading won the route (`_winning_profile`,
recorded from `decision.attempts` where `att.output is decision.final_output`).

- The candidate is judged as a **copy** (`dataclasses.replace` with copied `audit_notes` /
  `figures`): the judge legitimately mutates what it is handed (pre-verdict header repair,
  `rejection_class`), and none of that may reach the shipped output unless it accepts.
- Judge accepts → the **candidate's** text ships, so any pre-verdict repair the judge made
  is part of the judged bytes. Audit event `table_reread_rejudged` with `accepted: true`.
- Judge refuses (or the re-judge raises) → the previously accepted bytes ship unchanged, an
  `audit_notes` line is added, and `table_reread_rejudged` is emitted with
  `accepted: false` and the refusal reason. A crop reading socr could not verify never
  quietly replaces a verified one.
- The re-judge call sits **outside** the caller's fail-open `try/except`, because
  `_reread_page_tables` patches text outside its own guard: a raise there must still not
  leave unjudged bytes on the page.

`table_reread_rejudged` is deliberately **not** added to `TABLE_DISTRUST_KINDS`. The
disagreement that produced the patch already surfaces through the `dualpass_*` kinds, and
adding a new distrust terminal would move document status for reasons this ticket did not
decide. Noted here so the omission is a decision, not an oversight.

Post-fix, `grep -n '\.text = ' ` over the whole of `_phase_agentic` (lines 3121-3996)
returns **nothing**: no post-accept mutation of `best_output.text` remains anywhere in the
phase, on any path.

### Finding 2 (medium) — TICKET-C2 scoring coverage shrank. REPRODUCED (in part).

The branch gated scoring on `not is_native and bo.engine not in {"native", "chart_asset"}`.
Pre-branch the reach set was the UNION of two arms: the TICKET-C2 arm (`bo.text and
_page_has_tables`, with no `is_native` condition) and the GH-96 escalation arm, which
called `_table_page_needs_escalation` itself for any page with `bo.text` and
`engine != "chart_asset"` whenever the lane was live — tables or not.

- **Native-bypass table pages: reproduced.** `TestTableScoringCoverage::
  test_native_bypass_table_page_is_still_scored` was red (0 scoring calls, expected 1).
  `tests/test_native_only_table_status_gh211.py` stubs `_surface_table_scoring` with a
  lambda, so it stayed green either way and hid the loss.
- **Chart-asset pages: NOT reproduced.** The reviewer's basis was
  `_surface_table_scoring`'s own docstring ("Chart pages still reach it — `has_tables` is
  True there"). That sentence is stale: `chart_winner_pages` is exactly `chart_only_pages`,
  and a chart page WITH a table signal is arbitrated into `chart_mixed_pages` and never
  enters the chart lane. A chart-asset winner therefore has `has_tables` False by
  construction and did not reach scoring before this branch either. The docstring is
  corrected in this commit. The restored condition covers the reviewer's hypothetical
  regardless, since it keeps the old `_page_has_tables` arm verbatim.

**Changed.** Scoring is lifted out of the P5 signal gate and restored to the pre-branch
union:

```python
_lane_live = _escalation_profile is not None and not _escalation_degraded
if bo.text and (self._page_has_tables(page_num, ps) or (_lane_live and bo.engine != "chart_asset")):
    _score_table_signal = bool(self._surface_table_scoring(state, page_num, ps, bo))
```

The crop reread keeps its own, narrower gate (an OCR table page, not native, not a chart
asset), which is the PP-3 gate it always had.

### Finding 3 (low) — route evidence drove the GH-96 provider escalation. REPRODUCED.

`_table_signal = _score_table_signal or _route_table_signal` was passed as
`needs_escalation`. Route evidence is true when an earlier rung was
`native_table_verifier:`-rejected even if the accepted winner scores 100% against the
native layer — a cloud re-read `decide_escalation` provably cannot keep.

Red before the fix in the opposite direction, which is the sharper symptom:
`TestEscalationUsesScoreEvidence::test_route_evidence_alone_does_not_trigger_the_provider_escalation`
failed with "the GH-96 lane never ran". Folding TICKET-C2 and GH-96 into one signal gate
had also made the escalation lane itself unreachable on a page with no signal, where
pre-branch it ran (and returned immediately after scoring).

**Changed.** `_escalate_table_page` is called from its own pre-branch gate again
(`_lane_live and bo.text and bo.engine != "chart_asset"`) and receives
`needs_escalation=_score_table_signal` — the score alone. Route evidence stays with the
crop reread, which is what it is evidence for. Passing the already-computed score keeps the
author's fix for duplicate `table_not_scorable` / `table_unexplained_lanes` events, with
one exception: when the crop reread actually changed the shipped bytes, `None` is passed so
the page is re-scored on the text that now ships rather than on a stale score describing
text that does not.

### Files changed in round 1

- `src/socr/pipeline/orchestrator.py` — `_rejudge_crop_patched_page` (new),
  `_winning_profile` capture after `route_page`, the scoring/crop/escalation block
  restructured, `_surface_table_scoring` docstring corrected.
- `tests/test_p35_cold_review_round1.py` — new, 9 tests.

### Gates after round 1

```bash
PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-p35/src ~/venvs/socr/bin/pytest tests/ -q
# 3019 passed, 4 xfailed, 5 warnings in 94.79s   (exit 0)

uvx ruff@0.16.0 format --check .
# 486 files already formatted                    (exit 0)
```

---

## 8. Cold review round 2

Independent cold review, verdict NOT MERGEABLE: three BLOCKING, two SHOULD-FIX.
Reproducers in **`tests/test_p35_cold_review_round2.py`** (10 tests; 7 red before the
fixes, 3 already-true invariants the ruling asked to pin). Every case pins a difference
between two runs of the real pipeline in one process.

### Finding 1 (BLOCKING) — an accepting re-judge was not promoted. REPRODUCED.

Round 1's re-judge copied only `candidate.text` onto an output that ladder exhaustion had
already stamped `audit_passed=False`, with `ps.native_table_structure_failed` set and the
fail-closed floor PNG rendered. `_grid_authored_attempt` refuses an attempt whose
`audit_passed` is false, so `structure_class_floor_applies` still held and the page shipped
the structure-class floor rather than the bytes the judge had just accepted. A crop could
never recover a page whose whole ladder was refused.

Red before the fix: both rungs CERTAIN_FAIL-rejected, ladder exhausts, the crop repairs the
grid, the judge accepts it — and the shipped page did not match a direct acceptance of the
same bytes.

**Changed.** An accepting re-judge now leaves exactly what a first-time acceptance leaves.
The candidate is built as a fresh reading (SUCCESS, no failure mode, no rejection class) and
on acceptance the page takes the candidate's status, failure mode and rejection class,
`audit_passed=True`, the verdict's reason and confidence, and the exhaustion stamps are
cleared: `native_table_structure_failed`, `scanned_table_evidence_failed`,
`d3_floor_png_ref`, `rotated_shred_png_ref`.

Deliberately NOT cleared: `native_table_unverifiable`, `native_table_header_unattributed`,
`native_rotated_text_shredded`. Those are facts about the page's native layer and are true
of it whether or not a model reading was accepted. One residue is knowingly left: when
exhaustion had already rewritten `ps.native_text` with rendered chart-region PNG paths, that
rewrite stands. It is unreachable on an accepted page, where the model output wins.

### Finding 2 (BLOCKING) — post-verdict helpers. REPRODUCED in part, and scoped by ruling.

The round-2 ruling scopes the invariant: "judged bytes are shipped bytes" means no
post-verdict step may ADD or ALTER content. Three helpers run after the last verdict:

- `_sanitize_agentic_page_image_refs` — SUBTRACTIVE, enumerated exception, stays. Pinned by
  a subtractive-only invariant test: every token in the output existed in the input, never
  the reverse. **Not a bug today**: the test was green on first run, so neither
  `strip_phantom_images` nor `redact_fabricated_image_refs` can add or alter a token.
- `_guard_agentic_page_table_repetition` — SUBTRACTIVE, enumerated exception, stays. Same
  invariant test, also green on first run: every dropped line is byte-identical to one kept.
- `_attach_equation_latex_sidecars` — ADDITIVE, and PR #518's choke-point guards are **not
  on this base**, so the sidecar was unguarded. Both halves reproduced: a sidecar carrying
  `## Page 2` reached shipped text, and so did a number present nowhere in the page's source.

**Changed.** New `_guard_equation_sidecar_block` at the orchestrator's attach seam, which is
where content enters shipped bytes. `src/socr/math/equation_latex.py` is deliberately left
untouched so PR #518 does not conflict.

- **Assembly delimiter.** A block matching the `## Page N` marker that
  `ocr_output_contract.split_native_pages` splits on is refused outright. There is no safe
  partial form of that defect.
- **Numeric presence.** Every value-token in the model's LaTeX must exist in the page's own
  source (its native text or the accepted page text). On a violation the LaTeX is refused
  and the sidecar is rebuilt with the crop PNG and native text kept, so refusing costs no
  content. The reason string handed to the rebuilt sidecar is digit-free on purpose: it is
  shipped text, and quoting the invented number there would put it back in the corpus by the
  back door. The numbers live in the `equation_sidecar_refused` audit event only.
- **Notation is not a value.** `_latex_value_tokens` strips superscripts, subscripts and
  command names before the check. The `2` in `E = mc^2` is an exponent, and a faithful
  linearisation says "squared"; refusing it would be a false positive, and one existing test
  (`test_attach_sidecars_modifies_page_output_text`) proved it immediately. A fraction's
  numerals are NOT stripped: those are values.

The three exceptions are enumerated in the `_phase_agentic` docstring as a standing
contract, so the next person adding a post-verdict step sees what it has to satisfy.

### Finding 3 (BLOCKING) — judge-side events leaked from a refused candidate. REPRODUCED.

The composed judge closes over `state.events` through `record_event`, so everything it
emitted while assessing a candidate landed on the document immediately — including for a
candidate about to be refused. `native_table_verifier_hard_fail` is in
`TABLE_DISTRUST_KINDS`, so one refused crop candidate was enough to mark the shipped
(accepted, clean) bytes untrusted.

Red before the fix, and worse than predicted: the baseline run produced no
`tables_trust.json` at all, while the refused-re-judge run produced one listing page 1 as
untrusted with reason `native_table_verifier_hard_fail`.

**Changed.** `record_event` now goes through `_record_judge_event`, and the re-judge runs
inside `_judge_events_to(scratch)`. On acceptance the captured events are appended to
`state.events`; on refusal they are discarded and exactly one `table_reread_rejudged` event
is recorded with the reason. Known residue, documented on the context manager: a judge that
TIMES OUT keeps running in `_TimeoutJudge`'s worker and may emit after the block exits,
where it falls back to `state.events` — the same behaviour as before this seam existed.

The round-1 note claiming `dualpass_*` already carried this state was wrong, as the review
said: `dualpass_patched` is in `RESOLVING_KINDS`, not a distrust kind. That claim is retracted
here. `table_reread_rejudged` is still not added to `TABLE_DISTRUST_KINDS`, and now for a
sound reason rather than the wrong one: with the leak closed, a refused re-judge leaves the
shipped bytes' trust result byte-identical to a run that never judged the candidate, which is
the correct answer for bytes that never shipped.

### Finding 4 (SHOULD-FIX) — the extra judge call was not metered. REPRODUCED.

Red before the fix: the re-judge run and the no-crop run had identical `engine_runs`.

**Changed.** Each re-judge appends one `EngineResult` attributed to the profile whose
reading won, at that profile's own per-page cost — 0.00 for a local profile, a known number
— with elapsed time and SUCCESS/AUDIT_FAILED by verdict. The `table_reread_rejudged` event
now carries `judge_model`, `provider_id`, `cost_usd` and the reason, so the event names the
judge that incurred the call rather than only the OCR engine that produced the incumbent.

One caveat, stated plainly: attributing the call at the WINNING PROFILE's page rate
overstates a cloud winner, because the extra call is a judge call and not a second page read.
That is the ruling's instruction, and it errs toward showing more spend than occurred, which
is the fail-closed direction for a budget.

`bo.judge_reason` is set only on acceptance. On refusal it still records the route verdict on
the bytes that actually ship; overwriting it with a verdict on bytes that do not would be the
GH-169 defect in reverse. `tests/test_gh169_judge_reason_persists.py` was retargeted rather
than weakened: it now pins the per-attempt site by its RHS and enumerates the only other
permitted assignment, so a third site still fails the test.

### Finding 5 (SHOULD-FIX) — ARCHITECTURE.md described an ordering the code does not have.

Documentation only, no test. `docs/ARCHITECTURE.md` now says what the code does: the crop
runs after `route_page`'s verdict, its reconciled text is a new candidate re-judged before
shipping, the previously accepted bytes ship on a refusal, an accepted re-judge promotes
exactly as a first-time acceptance does, and the crop still precedes the GH-96 escalation
candidate and the table-judge terminal.

### Files changed in round 2

- `src/socr/pipeline/orchestrator.py` — `_record_judge_event` / `_judge_events_to`,
  `_rejudge_crop_patched_page` rewritten (clean candidate, promotion, event isolation,
  metering), `_guard_equation_sidecar_block` (new) wired into
  `_attach_equation_latex_sidecars`, `PAGE_ASSEMBLY_DELIMITER_RE` / `_numeric_tokens` /
  `_latex_value_tokens`, the post-verdict contract in the `_phase_agentic` docstring.
- `tests/test_p35_cold_review_round2.py` — new, 10 tests.
- `tests/test_gh169_judge_reason_persists.py` — retargeted for the second assignment site.
- `docs/ARCHITECTURE.md` — the crop-reread ordering.

### Gates after round 2

```bash
PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-p35/src ~/venvs/socr/bin/pytest tests/ -q
# 3029 passed, 4 xfailed, 5 warnings in 93.52s   (exit 0)

uvx ruff@0.16.0 format --check .
# 487 files already formatted                    (exit 0)
```

---

## 9. Cold review round 3

The branch was rebased onto `origin/main` at `1fe1a93`, which brings PR #518's equation
region lane. Two files came back with conflict markers and were resolved by hand:

- `src/socr/cli.py` — kept main's GH-142 rejection of `--no-judge-hard-pages`, kept this
  branch's paired `--dual-pass-tables/--no-dual-pass-tables`, and dropped main's now-dead
  `if no_dual_pass_tables:` arm (the one-way flag no longer exists).
- `src/socr/pipeline/orchestrator.py` — kept both sides: main's `return results` at the end
  of the equation-region detector and this branch's sidecar guard method.

One rebase interaction followed, not a review finding: main's GH-142 flag registry named
`no_dual_pass_tables`, which this branch replaced. `tests/test_gh142_flag_audit.py` now
classifies `dual_pass_tables` instead.

The round-2 review then re-tested every earlier fix. Its canaries are adapted in-repo as
**`tests/test_p35_cold_review_round3.py`** (16 tests; 9 red before the fixes). They are
adapted rather than copied in two ways, both deliberate: the equation cases patch only the
crop reader so the REAL choke point runs, and the delimiter canary asserts against the
contract's own `PAGE_MARKER_RE` rather than importing a local constant this round deletes.

### Finding 1 residual (BLOCKING) — operational-failure laundering. REPRODUCED.

The crop lane is reachable on ladder exhaustion with a non-empty operational failure, such
as a partial `ERROR`/`TRUNCATED` provider read. The re-judge presented the patched text as a
clean SUCCESS candidate and, on acceptance, promoted that state while the old `error` string
stood, turning a truncated read into a terminal SUCCESS page.

**Changed.** The re-judge now refuses before spending a judge call when the winning attempt
is an operational failure (`status is ERROR`, `failure_mode is TRUNCATED`, or any non-empty
`error`). The patched text is reverted to the previously judged bytes and the refusal is one
`table_reread_rejudged` event with `judged: false`. A crop reread repairs a table; it has no
evidence about the part of the page the provider never returned. A control test pins that
this is about the operational failure and not a blanket refusal of promotion.

### Finding 2 residual (BLOCKING) — reuse main's guards. REPRODUCED in part.

The structural ruling was right, and stronger than the local implementation on both halves.

- **Delimiter — NOT reproduced as a live defect, but the local regex was deleted anyway.**
  The three contract forms (`## page 2`, an indented `## Page 2`, `## Page 2 trailing`) are
  already refused on this rebased tree, because `contract_delimiter_violation` at the
  `process_equation_region` choke point keys on the contract's own `PAGE_MARKER_RE` and both
  consumers of that function go through it. The local `PAGE_ASSEMBLY_DELIMITER_RE` was
  therefore both weaker AND redundant. Deleted.
- **Exponent — REPRODUCED.** `_LATEX_NOTATION_RE` hid the `9` in `x^9`, `10^9` and `x^{9}`.
  PR #518 removed exactly that lookbehind because an invented 9 produced an EMPTY candidate
  multiset and passed containment. Deleted, along with `_numeric_tokens` and
  `_latex_value_tokens`.

`_guard_equation_sidecar_block` now calls `region_presence_verdict` — the same one-way
containment guard the region lane uses, with normalized Counters, sign and leading-decimal
handling, and no exponent exclusion. It ABSTAINS when the page has no numeric oracle or its
text layer shows decode damage, which is why `E = mc^2` on a prose page is not convicted:
that is abstention, not an exemption. On an INVENTED verdict the LaTeX is refused and the
sidecar is rebuilt with the crop and native text kept.

Round 2's own two guard tests stubbed `process_equation_region` outright, so they could not
see the choke point at all. They are replaced by round 3's, which patch only the crop reader.
The coverage moved and got stronger; it was not dropped.

### New finding (BLOCKING) — the fabricated-ref sanitizer adds marker prose. REPRODUCED.

Round 2's subtractive test for this helper was vacuous: its `MagicMock(spec=DocumentState)`
had no usable handle, so `_source_url_index` raised, the exception was caught, and only
phantom deletion ran. With the index forced empty the fabricated-ref gate really runs, and it
replaces an invented reference with `[socr: fabricated image reference removed …]`.

**Changed.** The invariant is scoped by ruling to CONTENT tokens: a subtractive step that
leaves socr's own receipt where it removed something has not added content. `SOCR_MARKER_RE`
is defined ONCE, beside `is_page_failed_marker` in `manifest.py`, and recognises both the
`[socr: …]` notes and the `[page N failed: …]` markers. The test now forces the URL index
empty so the gate actually runs, strips recognised marker spans, and asserts what remains is
token-subset of the input. No production change was needed to the helper itself; the defect
was the vacuous test and the missing shared recognizer.

### Finding 3 residual (BLOCKING) — timed-out judge leaks and cross-talk. REPRODUCED.

Round 2 documented this as a harmless residue. It is not: `_TimeoutJudge` returns while its
worker keeps running, so once the instance-global sink was restored a late verifier event
appended to `state.events` and again described bytes that never shipped. The same shared list
could also capture an older abandoned worker's event into a LATER re-judge's scratch list.

**Changed.** Ownership is per invocation, not per instance.

- Each `_judge_events_to` block mints a token and binds `(token, sink)` to the calling
  thread. `_TimeoutJudge` now takes an `owner` and carries that binding into its worker
  thread, so the emitter always knows whose call it is on.
- The token is RETIRED when the block exits. An event arriving on a retired token is dropped:
  its candidate has already been disposed of.
- A worker whose wrapper could not carry a binding is still recognised as late — it is not
  the thread driving the page loop, and no judge call is in flight. That last rule is gated on
  at least one scratch block having run, so a run with no crop re-judge behaves exactly as
  before.

Pinned by both the late-event canary and a cross-talk canary in which re-judge A's abandoned
worker emits while re-judge B's block is open: A's event reaches neither the document nor B's
sink, and B's own events still reach the document on acceptance.

### Finding 4 (SHOULD-FIX) — attribution and durability. REPRODUCED.

Round 2 attributed the call to the winning OCR profile, which is wrong in both directions: a
heuristic decision was journalled as `engine=gemini` at Gemini's page price, and a paid remote
judge over a local winner recorded zero. My round-2 log flagged the overstatement as
deliberate; the review is right that the fix is attribution, not a caveat, and that stands
corrected here.

**Changed.** The call is priced by the model that ran it. `judge_model` is read from
`state.agentic_judge_model` — the judge `_build_page_judge` actually built — and is never
re-resolved from config, so a run degraded to heuristics cannot name the VLM that did not run.
`providers.profile_by_model` turns that name into a price; a judge model that is not a metered
rung runs on a host socr provides, so it costs the known 0.00 rather than "unknown".

Durability: the sidecar persists only the winning output's `cost_usd`, and resume rebuilds the
page's `EngineResult` from that one field. The judge cost is now folded into `bo.cost_usd`, the
same seam the equation lane uses, so a resumed run's arithmetic matches the live one and a
partial resume cannot regain budget already spent. Pinned by the reviewer's terminal-resume
canary.

Round 2's metering test asserted the OCR attribution; it is retargeted to the corrected rule
and now also asserts the negative (a judge call must not be journalled as the OCR engine).

### Finding 5 (SHOULD-FIX) — architecture text. Fixed.

`docs/ARCHITECTURE.md` no longer claims promotion is exactly a first-time acceptance without
qualification. It now names the exception the round-3 ruling created: a page whose winning
attempt was an operational failure is refused before a judge call is spent.

### Files changed in round 3

- `src/socr/cli.py`, `src/socr/pipeline/orchestrator.py` — conflict resolution.
- `src/socr/pipeline/orchestrator.py` — operational-failure refusal and `_refuse_crop_patch`,
  per-invocation judge-event ownership plus `_TimeoutJudge(owner=…)`, judge-attributed and
  durable metering, `_guard_equation_sidecar_block` rewritten onto `region_presence_verdict`,
  the local delimiter/tokenizer constants deleted, the post-verdict contract docstring.
- `src/socr/core/providers.py` — `profile_by_model`.
- `src/socr/core/manifest.py` — `SOCR_MARKER_RE`, the single marker recognizer.
- `tests/test_p35_cold_review_round3.py` — new, 16 tests.
- `tests/test_p35_cold_review_round2.py` — two superseded guard tests removed with a pointer,
  metering test retargeted.
- `tests/test_gh142_flag_audit.py` — the renamed dual-pass flag.
- `docs/ARCHITECTURE.md` — the promotion exception.

---

## 10. Cold review round 4

The round-3 review closed five items and the rebase audit (PR #518's equation lane intact,
44 passed; the flag audit intact, 37 + 10 passed). Two remained. Reproducers in
**`tests/test_p35_cold_review_round4.py`** (9 tests; 3 red before the fixes).

### Item 1 — the marker was a test oracle, not a contract. REPRODUCED.

The ruling stands that a recognised socr marker is not document content, and the review is
right that round 3 did not make it true of anything shipping. `SOCR_MARKER_RE` was read only
by the test and the docstring, while `url_provenance` still assembled its replacement prose
by hand. That makes the exception an assertion about a string literal rather than a property
of the code, and a hand-written second copy is precisely how a recognizer and an emitter
drift apart.

Reproducing it needed a different kind of check, and that is worth stating plainly: a value
assertion cannot distinguish "built from the contract" from "happens to look like it today",
and that difference IS the finding. The pin is therefore a source check on
`url_provenance.py`, the same shape `tests/test_gh169_judge_reason_persists.py` already uses
for its production site. It was red: the module contained no call to the builder and did
contain a hand-written marker.

**Changed.** `socr_marker(note)` sits beside `is_page_failed_marker` in `manifest.py`, next
to the recognizer, and is guaranteed by construction to emit only markers the recognizer
matches: the note is flattened to one line and its closing bracket replaced, so free-form
detail can ride inside a marker without escaping it. `FABRICATED_IMAGE_MARKER` is now built
through it. The emitted string is unchanged, which is the point: the behaviour was already
right, and what was missing was the thing that keeps it right.

Pinned three ways, with the reviewer's round-2 canary as the content-token check: every
marker the builder can emit is recognised; the fabricated reference is removed; and outside
recognised marker spans the sanitizer's output is a token subset of its input, with the
untouched prose surviving unaltered.

### Item 2 — an exhausted multi-rung page regained paid budget on resume. REPRODUCED.

Live routing journals `decision.total_cost_usd`, which covers every rung the ladder tried.
The terminal sidecar persisted only the WINNING output's own `cost_usd`. Those differ exactly
when the ladder paid for a rung it then rejected — which is this branch's own recovery path:
both rungs rejected, the paid rung spent, the crop repairs the table, and the re-judge
promotes the FREE local winner. The page's real spend sat on a rejected attempt, so a resumed
run restored zero and could spend that budget again.

**Changed.**

- `UnifiedPipeline._page_total_cost(ps)` sums every attempt's cost, which includes the
  re-judge already folded onto the winner. It returns `None` when any attempt is unmetered,
  matching `DocumentState.total_cost`: an unknown subtotal must never restore as zero.
- `_flush_page_sidecar` persists it as `page_cost_usd`.
- `_restore_terminal_page_state` now reads the sidecar BEFORE metering and rebuilds the
  page's `EngineResult` from `page_cost_usd`, falling back to the winner's cost for sidecars
  written before the field existed — which is what those runs actually recorded.

Pinned by a unit-level multi-rung resume canary (a rejected paid rung plus a free winner:
live and resumed totals equal), by a persistence canary on the real recovery path that reads
the sidecar the live run wrote and includes a control that the winner's own cost is NOT the
page's spend there, and by the round-3 no-double-count property kept as a control: folding
the judge cost onto the page must not inflate the live total, which sums `engine_runs` alone.

The end-to-end canary asserts on the persisted sidecar rather than driving a second
`process()` run, because reconstructing the harness's exact run fingerprint in-test is not
what the finding is about; the reconstruction half is pinned by the unit-level resume canary
above.

### Files changed in round 4

- `src/socr/core/manifest.py` — `socr_marker`, the builder beside the recognizer.
- `src/socr/core/url_provenance.py` — builds `FABRICATED_IMAGE_MARKER` from it.
- `src/socr/pipeline/orchestrator.py` — `_page_total_cost`, `page_cost_usd` in the sidecar,
  and the resume path metering from it.
- `tests/test_p35_cold_review_round4.py` — new, 9 tests.

---

## 11. Cold review round 5

The round-4 review closed the marker contract and rejected the page-total fix with the right
diagnosis: **per-page spend was DERIVED, and the list it derived from is both incomplete live
and collapsed on resume.** Round 5 makes it a recorded fact. Reproducers in
**`tests/test_p35_cold_review_round5.py`** (5 tests; 3 red before the fix).

### Item — page spend is a recorded fact, not a derivation. REPRODUCED, both halves.

Two independent losses, one root cause:

- **Incomplete live.** A GH-96 escalation pays for its call and journals the spend BEFORE the
  accept/reject branch, but the rejected branch returns without appending the candidate to
  `ps.attempts`. A derivation therefore missed real spend on the very FIRST sidecar write.
  Red before the fix: a refused paid escalation persisted `page_cost_usd` of 0.00 while
  `state.total_cost` correctly showed the call had been paid for.
- **Collapsed on resume.** `_restore_terminal_page_state` rebuilds `ps.attempts` as the single
  frozen winner, and assembly re-flushes every terminal sidecar. So the first resumed run
  recomputed the field from that collapsed list and wrote the loss back to disk. Red before
  the fix on the branch's own recovery path: live 0.0002, first resume 0.0002, second resume
  0.0000.

**Changed.**

- `PageState.page_cost_usd` is the fact. `UnifiedPipeline._add_page_cost(ps, cost)` records
  against it, with `None` absorbing in both directions exactly as `DocumentState.total_cost`
  treats it: one unmetered call makes the page total unknowable, and unknown never decays
  back to a number.
- It is incremented at every site that journals an `EngineResult` for the page: the
  `route_page` block (`decision.total_cost_usd`, so every rung the ladder tried, accepted or
  rejected), the GH-96 escalation ABOVE its accept/reject branch (which is the only place a
  refused candidate's spend is ever seen), the equation region lane, and the crop re-judge.
- `_flush_page_sidecar` persists the fact; `_restore_terminal_page_state` restores it
  VERBATIM into `PageState` and meters from it. Every later sidecar rewrite therefore writes
  the restored fact, not a recomputation.
- `_page_total_cost` survives only as a pure reader, and says so.
- A sidecar written before the field existed still falls back to the winner's cost — what
  those runs actually recorded — and that fallback becomes the page's fact, so the run after
  it is stable.

Pinned by: live / first resume / second resume all equal on a multi-rung page; the same three
totals on the real crop-recovery path end to end; a refused paid GH-96 escalation persisting
its spend on the FIRST sidecar write; the old-sidecar fallback written back as a fact and
stable on a third read; and the round-3 no-double-count control kept, since folding the judge
cost onto the page must still not inflate a live total that sums `engine_runs` alone.

One deliberate robustness choice, worth naming because it trades against this ticket's own
goal: `_add_page_cost` fails OPEN and logs a warning if the page object cannot carry the
field. The alternative was discovered by the existing GH-96 suite, whose `PageState` double
lacked it: the raise landed inside `_escalate_table_page`'s own fail-open guard, which
swallowed it and silently kept the incumbent text — losing an accepted escalation to a
metering bug. Under-recorded spend is a warning; lost content is not recoverable. The double
was also given the field.

### Superseded tests, retargeted rather than weakened

- Round 4's hand-built multi-rung resume case leaned on the derivation being fixed here. It is
  replaced by round 5's, which records spend the way production does and carries it through
  TWO resumes, with a pointer left in its place.
- Round 3's terminal-resume canary builds its fixture by hand; it now records the route spend
  at the same site it journals the `EngineResult`, mirroring production. Its assertion (live
  equals resumed) is unchanged.
- `tests/test_p35_cold_review_round2.py`'s harness now also returns the run's config and PDF
  path, so a later round can resume the page a live run left on disk under the SAME run
  fingerprint. Additive.

### Files changed in round 5

- `src/socr/core/state.py` — `PageState.page_cost_usd`, the recorded fact.
- `src/socr/pipeline/orchestrator.py` — `_add_page_cost` and its four recording sites,
  `_page_total_cost` reduced to a reader, verbatim restore on resume.
- `tests/test_p35_cold_review_round5.py` — new, 5 tests.
- `tests/test_p35_cold_review_round4.py`, `tests/test_p35_cold_review_round3.py`,
  `tests/test_p35_cold_review_round2.py`, `tests/test_gh96_escalation_lane.py` — retargeted
  fixtures and the pointer noted above.

---

## 12. Cold review round 6

Round 5's four canaries passed and the fail-open judgement was accepted. What the review
rejected was the claim of exhaustive coverage, and it was right in a way worth stating
plainly: **rounds 4 and 5 kept fixing the instance while the class stayed open.** Reproducers
in **`tests/test_p35_cold_review_round6.py`** (7 tests) plus one assertion added to the
existing corrupt-math end-to-end case; 7 red before the fix.

### Item — journaling and recording must be one call. REPRODUCED, both sites.

- **BLOCKING, corrupt-math recovery.** The lane journals a page `EngineResult` with
  `cost=None` — which is what it records whenever the model actually ran — and never touched
  the fact. So the live document total was correctly UNKNOWN while the page persisted a known
  `0.0`, and a resumed run read unmetered spend as no spend. Red before the fix on the
  existing reachable case at `tests/test_orchestrator.py`, which already drives this lane end
  to end and writes a terminal sidecar; it gained the page-total assertion rather than a new
  fixture.
- **MEDIUM, `DocumentState.apply_result`.** Journals and associates page outputs without
  touching the fact. No in-tree caller today, which is exactly why it was the next lane to
  bypass the contract silently.

**Changed.** `DocumentState.record_engine_run(result, page_nums=None)` is now the ONE place
`engine_runs` is appended to: it journals the run and charges its cost to the pages it ran on,
with `None` absorbing in both directions. The recording rule itself moved to
`socr.core.state.add_page_cost`, beside the journal helper, so the two cannot drift;
`UnifiedPipeline._add_page_cost` survives as a thin delegate because tests and call sites
reach for that name. Every site goes through it — the routing ladder, the GH-96 escalation
above its accept/reject branch, the equation region lane, the crop re-judge, the corrupt-math
lane, and `apply_result`. The resume path passes an empty `page_nums`, because it has just
restored the persisted fact verbatim and charging it again would double the page's spend on
every resume.

**The guard is the actual deliverable.** `TestNoLaneCanBypassTheJournalHelper` walks the AST
of every module under `src/socr` and fails on any `engine_runs` `append` / `extend` / `insert`
/ `+=` outside `record_engine_run`. A future lane written the obvious way now fails a test
instead of silently under-recording, which is the failure mode that survived three rounds. The
guard carries its own negative control: a synthetic bypass file it must detect, because a
guard that cannot fail is not a guard.

One consequence worth recording: the GH-96 suite's `_State` double had to grow the method. It
borrows `DocumentState.record_engine_run` rather than reimplementing it, so the double cannot
drift from the contract either. That is the second time a test double has had to model this
contract (round 5 added the field to `_PageState`), and both times the double's failure was a
true signal about a real seam.

### Files changed in round 6

- `src/socr/core/state.py` — `record_engine_run`, `add_page_cost`, `apply_result` routed
  through the helper.
- `src/socr/pipeline/orchestrator.py` — all six journal sites through the helper, the
  corrupt-math lane included; `_add_page_cost` reduced to a delegate.
- `tests/test_p35_cold_review_round6.py` — new, 7 tests including the bypass guard.
- `tests/test_orchestrator.py` — the corrupt-math case asserts the page total is unknown.
- `tests/test_gh96_escalation_lane.py` — the state double models the journal contract.

---

## 13. Cold review round 7

Production was accepted: all seven journal sites go through `record_engine_run`, resume
accounting is correct, and the corrupt-math canary passes. What the review rejected was the
ENFORCEMENT, and it was right — it defeated the round-6 AST guard four ways: an alias then
`append`, a `list(...) + [...]` reassignment, a `getattr` hop, and a subclass method that
simply took the exempt name. The round-6 log had called that guard the actual deliverable, so
the class it claimed to close was still open.

**Ruling accepted without reservation: a pattern-matcher cannot win that game.** Round 6 was
answering a structural problem with a lint, and every bypass above is ordinary Python that a
future lane would write without any intent to evade. The contract is now enforced by
ENCAPSULATION, where the failure happens at the line that writes it.

### The change

- `DocumentState._engine_runs` is the journal, private.
- `DocumentState.engine_runs` is a read-only view: a property with NO setter, returning a
  tuple. `append` / `extend` / `insert` raise `AttributeError` on a tuple; `+=` and plain
  assignment raise because there is no setter; a `getattr` hop or a local alias inherits the
  same tuple and raises identically.
- `record_engine_run` is the only member that touches the private list. Every reader —
  `total_cost`, `engines_used`, the manifest, the resume path, the tests — goes through the
  public view, and readers only iterate, index or count, all of which a tuple does.

### The guard, kept only in the role it can do

`tests/test_p35_cold_review_round7.py` keeps a SMALL scoped guard on the private NAME: no
module outside `state.py` may reference `_engine_runs`, and inside `state.py` only
`DocumentState.record_engine_run` and the read-only view may. It is scoped by enclosing CLASS
AND MEMBER, which is the specific hole the review found — round 6 exempted anything named
`record_engine_run`, so a subclass override with that name walked through.

The reviewer's five probes are the pin, each as its own test:

| probe | outcome |
|---|---|
| alias, then `append` | raises at runtime |
| `state.engine_runs = list(...) + [...]` | raises at runtime |
| `state.engine_runs += [...]` | raises at runtime |
| `getattr(state, "engine_runs").append(...)` | raises at runtime |
| subclass helper named `stash` | caught by the scoped guard |
| subclass override named `record_engine_run` | caught by the scoped guard |

Two further tests keep the exemption honest, since the owners are the one place the guard
does not look: the recorder must contain exactly one append to the private list, and
recording must still charge the page.

The defeated parts of the round-6 guard are deleted, with a note in their place saying where
the enforcement went and why. The one-call contract that file pins is unchanged and still
tested; only its mechanism moved.

### Call sites adapted

Nine test modules appended to the journal directly and now use `record_engine_run`; equality
assertions against `[]` became `()`. The GH-96 `_State` double mirrors the new shape — a
private list plus a read-only view — and borrows the real recorder rather than
reimplementing it. That is the third time a double has had to model this contract, and each
time it has been a true signal about a real seam rather than test friction.

### Files changed in round 7

- `src/socr/core/state.py` — the private journal and the read-only view.
- `tests/test_p35_cold_review_round7.py` — new, 10 tests including the five probes.
- `tests/test_p35_cold_review_round6.py` — the defeated guard deleted, with a pointer.
- `tests/test_gh96_escalation_lane.py`, `tests/test_manifest_agentic.py`,
  `tests/test_document_state.py`, `tests/test_orchestrator.py`,
  `tests/test_gh171_sidecar_carries_figures.py`,
  `tests/test_gh488_figure_sidecar_end_to_end.py`, and the round 3-5 files — call sites.

## 14. Cold review round 7 — the journal is not a constructor input

Round 7 accepted the encapsulation for every write through the public name and
found one legitimate residual: `_engine_runs` was still an `init=True` dataclass
field, so `DocumentState(handle, _engine_runs=[run])` and
`dataclasses.replace(state, _engine_runs=[run])` installed a journal no page was
charged for. Reproduced (total 0.25, page 0.0). Fixed by `field(init=False,
default_factory=list, repr=False)`; two probes pin it.

Ruled out of scope, deliberately: `object.__setattr__(state, "_engine_runs", …)`
and `state.__dict__[...]` writes. Those are Python reflection, not a contract a
class can close; the sole-writer guarantee covers every ordinary way a lane would
be written. Recorded so the next reviewer does not reopen it.
