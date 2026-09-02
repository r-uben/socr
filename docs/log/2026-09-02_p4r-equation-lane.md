# P4-R — the equation-region lane: what shipped

2026-09-02. Programme item P4(b) of `docs/log/2026-09-01_conceptual-revision.md`,
built to the rulings in `docs/log/2026-09-02_p4-structure-lane-design.md` section 7
and the trigger ruling in `docs/log/2026-09-02_p4m-trigger-rates.md`. This log
records what the build actually does, the presence-guard tokenizer, the
deviations from the plan, and what was deliberately not done.

## What shipped

A table-free born-digital page carrying the equation signal no longer takes the
free trusted-native bypass in **agentic mode**. It routes to a new lane that
reads its display-equation regions with the local VLM and attaches what survives
the gates beside the untouched native prose.

- **Flag:** `PipelineConfig.equation_region_lane`, default **True**. Kill switch
  `--no-equation-region-lane` on `process` and `batch`. Fingerprinted, so
  toggling it invalidates the per-page resume ledger.
- **Predicate:** `_is_equation_region_lane_page` (orchestrator.py:1372), beside
  `_is_corrupt_math_recovery_page`. True only for an agentic, native-eligible,
  born-digital page with native text and `has_equations`, and no tables, no
  corrupt math, no shredded rotated text, not `--native-only`.
- **Precedence in the fused loop:** corrupt math > chart asset > equation region
  > generic no-provider > plain native > whole-page `route_page`. The lane's
  pages are removed from `ocr_pages` and `native_fallback_pages` before provider
  setup, so an empty ladder cannot stamp one of them WARNING/MODEL_UNAVAILABLE.
- **Executor:** `_agentic_equation_region_page` (orchestrator.py:1614). It builds
  the ORDINARY native page first, via `_agentic_native_page`, and treats that as
  the floor. Everything after is additive.
- **Engine tag:** `native+equations` when at least one reading attaches, plain
  `native` otherwise. The `native` prefix is deliberate: every existing
  `startswith("native")` guard in `core/manifest.py` treats the lane as a native
  lane, mirroring `native+math`.
- **Status:** SUCCESS / `audit_passed=True` / `FailureMode.NONE` are **carried
  from the native floor**, never restated. The lane cannot upgrade or demote a
  page relative to the lane-off run, and it never touches `audit_passed` — that
  is the winner-selection flag, not a quality flag.
- **Attachment:** `attach_equation_sidecars_in_place` in
  `src/socr/math/equation_latex.py:308` (473 lines). It splices the existing
  1C sidecar block (crop image ref + `EQUATION_SIDECAR_HEADER` + fenced LaTeX)
  immediately after the region's own exact native slice. Every native byte
  survives, in order. It is idempotent, consumes repeated identical slices in
  region order, and reports a slice it cannot locate instead of appending a
  dangling block at the page end.

## The trigger

`has_equations` **as detected** — no threshold, no count, no new signal. That is
ruling 1, chosen from the measured table in the P4-M log (36.1% of today's free
lane moves; the math-character distribution has no natural break that would
justify inventing a cut).

The page trigger is **not** the same thing as region location. Every signalled
page enters the lane; the deterministic, model-free `detect_display_equations`
then decides whether there is anything to read. A page whose signal came from one
inline math glyph typically yields no display region, so it costs **zero model
calls** and ships bytes identical to a lane-off run. That is reuse of an existing
detector, not a narrower trigger and not an invented threshold — and it is not a
claim that the lane is cheap: no throughput measurement was made here.

## The presence guard

`region_presence_verdict` in `src/socr/tables/escalation_canary.py:386`, on top
of the new whole-text tokenizer `text_value_tokens` (:241).

- **Oracle:** `native_text_value_counts(native_text)` — the page's own numeric
  tokens, with multiplicity.
- **Candidate:** `text_value_tokens(raw_latex)` — the same single module-level
  numeric regex, applied to arbitrary text. LaTeX control words are alphabetic
  and simply do not match it, so `\frac{1}{2}` contributes `1` and `2` and
  `\tag{3}` contributes `3`, with no destructive stripping pass that could erase
  a real argument. A number directly following `^` is excluded as an exponent.
- **Direction:** containment is **one-way**. A region legitimately holds a subset
  of the page's numbers, so the verdict never returns `PRESENCE_LOST`.
- **ABSTAIN** (`PRESENCE_UNVERIFIABLE`) on an empty oracle, an encoding-suspect
  page, or a corrupt-math page. Never FAIL in those states — ruling 4.
- **FAIL** (`PRESENCE_INVENTED`) rejects **that one region's reading**. It does
  not demote the page, change the document status, or touch `audit_passed`. The
  refusal is an audit event carrying the invented tokens and the reason.

This is a rejection guard, not an acceptance contract. Presence proves "not
invented"; it never proves correctly placed, correctly bound, or complete. The
module says so and so does the verdict's docstring.

## Audit events

One terminal disposition per region: `equation_region_reading_attached`,
`equation_region_reading_rejected`, `equation_region_reading_unvalidated` (1A
failure, empty output, no crop, or a policy skip with its reason), or
`equation_region_reading_unaligned`. Two page-level outcomes:
`equation_lane_no_region` and `equation_lane_detection_failed`.

All six are in `UnifiedPipeline.EQUATION_LANE_EVENT_KINDS` and were added to the
allowlist `_restore_terminal_page_state` replays on resume. Without that, a
rejected reading would be written to the page sidecar and then vanish from
`audit_log.json` the moment the page resumed — the GH-353 D1a shape, and here it
would erase the only record that a reading was looked at and refused.

## With no provider

`_equation_lane_model_disabled_reason` refuses the call when the clean-equation
model is local and no local provider profile is available. The lane makes no HTTP
attempt, records the skip per region, and ships the native floor. This is CI's
exact state, and the difference pin asserts that in it **nothing at all moves**
between a lane-off and a lane-on run.

## Deviations from the plan, and why

1. **No new `src/socr/math/equation_lane.py`.** The plan's t2 created one; the
   critique forbade it and the tests stage wrote its acceptance tests against
   `socr.math.equation_latex`. Ruling 3 says to reuse the existing LaTeX-sidecar
   machinery rather than invent a new splice, so the helper extends
   `equation_latex.py` and consumes the existing `EquationLatexResult`. The
   ruling wins over the plan.
2. **The lane does not call `splice_math`.** That is the corrupt-math lane's
   REPLACING splice. P4-R is additive by ruling 3, so it uses its own in-place
   attachment helper instead.
3. **`_is_trusted_native_without_ocr` is untouched.** The widening lives in
   `_is_agentic_trusted_native` only. Changing the shared predicate would move
   equation pages in the non-agentic single-engine, consensus and repair paths,
   which have no region machinery to fall into.
4. **No `EngineResult` is appended for the lane.** `DocumentState.total_cost`
   sums `engine_runs`, and an unmetered local call there would make the remaining
   budget unknowable for every LATER page of any document containing an equation
   page (the fail-closed branch admits only free rungs). Provenance goes on the
   `PageOutput` (`provider_id`, `provider_model`, `provider_backend`, unknown
   `cost_usd`) and into the audit events instead.
5. **The E2E tests drive `_phase_agentic` + `_phase_assemble` rather than
   `process()`.** Same substance, far tighter control of the fixture, and it is
   the pattern the chart-lane suite already uses. Both provider states are
   parametrised and every assertion is a difference, per #257.
6. **Test corrections.** Two tests from the tests stage contradicted the code
   rather than a ruling and were fixed: `build_config`'s parameter is
   `config_path`, not `config_file`; and `_reaches_structure_class_branch` is a
   module-level function in `socr.core.manifest`, not a pipeline method — the
   original `getattr` probe SKIPPED both arms, making ruling 2's most important
   pin vacuous. A positive control was added so the corrected pin cannot pass
   because the gate is broken.

## Deliberately not done

- **`PageState.is_structure_class()` is unchanged** — still `has_tables` alone.
  This is ruling 2 and the whole safety margin of the design: BLOCKING 1 on #269
  reverted R3 for widening it, and after P2 (#490) that false-reject arm would
  delete the page's prose outright. Pinned behaviourally, both by unit tests over
  `_reaches_structure_class_branch` and by an end-to-end arm that runs the lane
  with every reading refused and asserts no `structure_class_floor_text` marker
  and no lost prose.
- **No whole-page model read of an equation page.** The unit of replacement is
  the region.
- **A rejected reading attaches nothing.** Its crop stays on disk as evidence.
  Attaching a crop-only block on rejection would be more visible but would change
  shipped bytes on pages where a lane-off run changes none.
- **Non-agentic paths keep the free native lane.**
- **Chart winners are not read.** A page with both chart marks and equation
  regions keeps the chart lane; its equations are not transcribed. Same class as
  the open `visual_values_not_transcribed` follow-up.
- **Figures are unchanged** (ruling 5); the chart-asset lane already covers
  table-free raster pages.
- **`reconstruct.py`, `native_verifier.py` and the chart-asset lane are
  untouched.**
- **A detection failure is not distinguishable from "no region" in the audit
  trail.** The GH-36a seam `_detect_and_crop_equation_page` already fails soft
  and returns no regions, so a raising detector reaches the lane as
  `equation_lane_no_region`. No content is lost either way — the page ships the
  same native prose it would with the lane off — so this is a provenance
  imprecision, not a loss. The lane keeps its own `equation_lane_detection_failed`
  handler for failures raised outside that seam.

## Evidence boundary

Synthetic fixtures and automated tests only. The lane was **not** run over a
corpus, and no accuracy, quality or throughput claim is made here. The only
measured numbers about this lane remain P4-M's trigger rates. In particular,
nothing here establishes how often a real page's reading is accepted, how often
the presence guard fires, or what the lane costs in wall-clock time on a real
document.

## Grounding canaries

```
src/socr/pipeline/orchestrator.py:1372  def _is_equation_region_lane_page
src/socr/pipeline/orchestrator.py:1614  def _agentic_equation_region_page
src/socr/math/equation_latex.py:308     def attach_equation_sidecars_in_place
wc -l src/socr/math/equation_latex.py -> 473
```

## Cold review round 1 — 2026-09-02

Verdict was NOT MERGEABLE on five blocking and four should-fix findings. Every
one was reproduced as a committed test BEFORE its fix, except where noted.

### 1. A model-authored `## Page N` split the document — reproduced, fixed

`assemble_pages` writes page boundaries and `split_native_pages` reads them back
from anywhere in the body, fence or no fence. The 1A gate is LaTeX SYNTAX only
and accepts `y = 2x + 1\n## Page 3` happily. **Reproduced end to end**: the
three-page fixture assembled with FOUR markers, split into four logical pages,
and tripped three separate consistency warnings — native prose cut in half and
reassigned by a model reading, which ruling 3 forbids.

There is no escape convention in the contract (the only existing handling is a
LEADING-marker strip in the provisional flush, useless against a marker in the
middle of a body), so per the round-1 ruling the reading is **rejected**, not
escaped. `contract_delimiter_violation` (`equation_latex.py:311`) refuses a page
marker or a triple-backtick run. It is applied inside `process_equation_region`,
so the legacy GH-36b sidecar path is covered by the same choke point, and again
in the lane so the refusal lands under its own audit kind
`equation_region_reading_unsafe_markup`. The canary asserts marker count,
logical page count and fragment count on the assembled document.

### 2. An invented exponent walked through the presence guard — reproduced, fixed

`_NUMERIC_TEXT_RE` began with `(?<!\^)`, so `text_value_tokens("x^9")` returned
NOTHING and an invented 9 was contained in every oracle. It was also wrong on its
own terms: only the first digit was skipped, so `x^999` tokenized as `99`. The
lookbehind is deleted. An exponent is a number the model wrote and ruling 4
requires the guard to see it.

The approving test encoded the gap and was rewritten: it now asserts the exponent
IS a candidate token, that a multi-digit exponent is not truncated, and that an
exponent absent from the page rejects the reading.

### 3. Executed model calls were unmetered — reproduced, fixed

No `EngineResult` was appended and the attached output carried `cost_usd=None`,
so on resume `_restore_terminal_page_state` rebuilt an UNMETERED run and
`state.total_cost` became None — failing every later paid rung closed and routing
the same document differently live than resumed. Reproduced on both arms: no lane
entry in `engine_runs`, and `total_cost is None` after a resume.

`_meter_equation_lane_calls` now appends one `EngineResult` per lane page that
actually called the model, priced from the serving profile's own
`cost_per_page_usd`. A local rung is 0.00, which is a **known** number, not an
unknown one. The cost is stamped on the shipped output before any early return,
so a REJECTED reading still records the spend its refused call cost, and live and
resumed totals agree.

### 4. "Provider available" meant any local profile — reproduced, fixed

The gate asked only whether any local-tier profile existed. Marker, Nougat and
DeepSeek are local tier and none can serve a crop to an Ollama vision model, so a
machine with Marker and no Ollama cleared it and the lane then POSTed and waited
out the 300s timeout. Reproduced with `[PROFILE_MARKER]`.

`_equation_lane_provider` now resolves the profile whose MODEL is the one about
to be called — the same `(engine, backend, model)` identity the ladder rungs on —
and returns it, so the cost and provenance come from that profile too. Fail-closed:
an operator pointing `--clean-equation-model` at a model no available profile
serves gets a recorded skip and native prose, never a speculative call.

### 5. A transient no-provider skip froze as a terminal SUCCESS — reproduced, fixed

Provider availability is transient and invisible to the run fingerprint, so a page
skipped because Ollama was down shipped a clean terminal SUCCESS and was restored
by the ledger forever: the default-on lane stayed permanently inert on that page
until `--reprocess`. Reproduced with two runs, the second with the provider back
and the model never called.

Fixed WITHOUT demoting the page, which would break the no-provider parity
requirement. `PageState.equation_lane_retry_pending` is persisted in the page
sidecar; `_load_terminal_page` refuses to skip such a page **when a provider is
available now**, and keeps skipping it when there is still none. The page's bytes,
status and audit verdict remain exactly what they are with the lane off. The latch
is set only for availability, never for config refusals (strict-local, an unpriced
cloud model), which the fingerprint already describes.

### 6. Tests did not assert the invariants they claimed — fixed

"Every native byte" checked three substrings; it now removes exactly the inserted
block and asserts equality to the original bytes. "Byte for byte" checked
`fragment.strip() in whole`, which the finding-1 corruption passes; it now
compares the exact fragment SET against the exact page bodies
`split_native_pages` yields, with counts. Every implementation-presence skip was
removed from the landed acceptance tests — deleting the feature used to turn them
green — and replaced with one explicit existence test per file.

### 7. The layering guard had been weakened — fixed

The allowlist entry blessing `benchmark/trigger_rates.py` importing
`socr.core.born_digital._MATH_FONT_RE` is REVERTED. `born_digital` now exposes
`is_math_font` and `math_font_char_count`, and the benchmark calls those, so the
P4-M measurement reads the math-font term through public core API and cannot
drift from the detector P4-R routes on. Pinned by a test that the public
accessor agrees with the private pattern and that the benchmark no longer names it.

### 8. A `process()`-driving test still probed Ollama — mechanism confirmed, fixed

`_run_fingerprint` resolves the judge identity and the resolver calls
`OllamaVisionJudge.is_available()`. Confirmed by counting: one fingerprint call
invokes the resolver once, so the flag-classification fixture made real HTTP
probes whose result depended on local Ollama state. `_resolve_judge_model` is now
pinned to `""` at that fixture's process boundary.

### 9. A wall-clock threshold in a unit test — fixed

`assert avg_ms < 500` is deleted. It is an empirical magic threshold: a loaded CI
worker fails it with correct geometry and a real regression under it passes.
Non-negative timing instrumentation remains; throughput acceptance belongs in a
benchmark with a measured baseline.

### Canaries after round 1

```
src/socr/pipeline/orchestrator.py:1372  def _is_equation_region_lane_page
src/socr/pipeline/orchestrator.py:1583  def _equation_lane_provider
src/socr/pipeline/orchestrator.py:1645  def _agentic_equation_region_page
src/socr/math/equation_latex.py:311     def contract_delimiter_violation
wc -l src/socr/math/equation_latex.py -> 529
```

## Cold review round 2 — 2026-09-02

Round 1 closed findings 1, 2, 7, 8 and 9. Findings 3, 4 and 5 were only
partly closed and 6 remained. The round-2 ruling named the common cause: the
lane called the model directly, outside the provider ladder, and then
re-implemented the ladder's guarantees piecemeal. It now takes authorization,
identity and price FROM that machinery instead.

### 3. The budget contract was bypassed — reproduced, fixed

Metering was in place, but `_phase_agentic` filters its ladder with
`max_cost_per_page` and then handed the lane the UNFILTERED list, and the lane
never computed remaining `cost_budget` at all. Every round-1 budget test used
the free local rung, which cannot show whether a cap is applied. Three
reproductions with a NON-zero-cost profile, all failing before the fix:

- a rung priced above `--max-cost-per-page` was called anyway;
- two region reads at 0.03 each ran under a 0.05 per-page cap, because reads
  were not counted cumulatively;
- a rung was called with less `--cost-budget` remaining than it costs.

`_equation_lane_provider` now runs candidates through `provider_ladder` with the
same `per_page_only` and `max_cost_per_page` arguments `_phase_agentic` uses, so
a rung priced out of the routing ladder is priced out of the lane by
construction. `_equation_lane_remaining_budget` reproduces the generic OCR
branch's computation including its fail-closed rule: an unmetered earlier call
makes the remainder unknowable, and an unknown subtotal is never treated as zero
spend. Both are checked BEFORE each region call, and reads accumulate against the
per-page cap, so the lane stops mid-page and records why.

### 4. Provider identity was still the model alone — reproduced, fixed

Round 1 matched `profile.model` only, while the call is hardwired to the Ollama
generate API. Reproduced with a profile serving the SAME model over vLLM: it
authorized the call and then lent it a price and provenance it never served.

Selection is now `(model, backend)`, with `EQUATION_LANE_BACKENDS` naming the
backends this lane's transport can actually address. The refusal reason
distinguishes the three cases an operator needs to tell apart: nothing available
serves the model, the model is only on a backend this transport cannot address,
or every rung serving it is priced out.

### 5. The document resume gate never saw the latch — reproduced, fixed

The round-1 latch lived in `_load_terminal_page`, but `process()` consults the
DOCUMENT gate before a `DocumentState` exists, so a no-provider run produced a
COMPLETED root entry and the next run skipped the whole document before any page
ledger was read. Reproduced through the real entry path: two `process()` runs,
provider absent then present, and the model was never called.

`_write_metadata` now lifts the page latch into the root index entry, and
`_resume_skippable` refuses the document skip when that marker is set and the
lane is enabled — on the single-file and the batch path both. Re-running is cheap
rather than free: pages that did finish are still restored by the provider-aware
per-page ledger, so an offline rerun re-opens the document without re-reading
anything. The marker is cleared as soon as a run finishes with nothing pending.

Two latch-semantics gaps were fixed with it, each with its own reproduction:

- **strict-local no longer suppresses the latch.** A strict-local run whose local
  rung is briefly down did not run the model either.
- **A transport failure now latches.** `latex_for_crop` returns `""` on an
  unreadable crop, a URLError, a timeout or a bodyless response. A provider was
  present at selection time and the model still did not run, so the page is not
  a finished result. Conservative by design: a model that genuinely answers with
  nothing is indistinguishable here and costs one retry, which is the safe
  direction.

The latch is deliberately NOT set for refusals the run fingerprint already
describes — strict-local versus a cloud model, a per-page cap, an exhausted
budget. Those reproduce identically on every rerun, so latching them would make
the document permanently unskippable while changing nothing about the outcome.
`_equation_lane_availability_refusal` draws that line in one place.

Both directions are pinned through `process()`: no-provider-then-provider must
re-call, and provider-then-no-provider must restore rather than re-fail.

### 6. Byte-exactness — gaps closed, no failure to reproduce

The remaining gaps were missing assertions rather than wrong behaviour: both new
tests passed the moment they were written, and that is reported as such rather
than as a fix. Fragments are now compared to the split page bodies WITHOUT
`.strip()`, for every fragment; and the marker regression now compares the
rejected page's bytes and every fragment against a lane-off run, instead of
comparing counts.

### Canaries after round 2

```
src/socr/pipeline/orchestrator.py:220   def _resume_skippable
src/socr/pipeline/orchestrator.py:431   def _equation_lane_retry_blocks_resume
src/socr/pipeline/orchestrator.py:1633  def _equation_lane_provider
src/socr/pipeline/orchestrator.py:1761  def _agentic_equation_region_page
```

## Cold review round 3 — 2026-09-02

Round 2 closed findings 3 and 6. Findings 4 and 5 were each down to one narrow
hole, plus a new should-fix about the shape of the round-2 cost tests.

### 4. Authorization read the DECLARED backend, not the resolved one — reproduced, fixed

`PROFILE_QWEN_LOCAL` declares `backend="ollama"`, but a QWEN rung's executed
backend comes from the live config: `resolved_provenance` runs it through
`qwen_backend` and returns vLLM on an HPC deployment. Round 2 compared the raw
registry field, so that deployment authorised an Ollama-transport call against a
rung serving somewhere else entirely — and then lent the call that rung's price
and provenance. Reproduced by asserting the premise first
(`resolved_provenance(PROFILE_QWEN_LOCAL, cfg)[0] == "vllm"`) and then that the
lane selected it anyway.

`_equation_lane_backend_addressable` now asks `resolved_provenance`, the same
function the manifest and the CLI invocation already agree on. `auto` is
resolved in context through `qwen_auto_resolves_to_openai`, because `auto` means
Ollama unless `VLLM_BASE_URL` is exported, which is the HPC deployment rather
than a corner case. A resolver failure is doubt and refuses.

The landed same-model/wrong-backend test now drives the PRODUCTION shape. Its old
synthetic profile turned out to prove nothing: a hand-made QWEN profile stamped
`vllm` is resolved BACK to Ollama by the very function this finding is about, so
accepting it is correct. The raw field still decides for engines the resolver
does not rewrite, and that arm is kept with a non-QWEN rung.

### 5. The latch and the terminal record were two writes — reproduced, fixed

`_write_metadata` called `index.record()`, which saves immediately, and only then
mutated the entry and saved again. Between those saves the index holds a
resumable entry with NO latch; an interruption or a failure of the second save
leaves it there permanently and the next run skips the whole document. Reproduced
by snapshotting every persisted state of the index and asserting none of them is
a resumable entry without the latch — the first snapshot was exactly that.

The latch now rides in on `to_entry` via a thin `_EquationRetryMetadata` wrapper,
so `RootIndex.record` remains the SOLE author of root metadata.json — a PP-5
invariant with its own regression test, which a first attempt at this fix broke
by assigning the entry directly — and there is still exactly one save. Fail-closed: if that single save raises,
nothing is recorded at all and the next run reprocesses the document — pinned by
a second test that makes the write fail and asserts no resumable record survives.
Note the hazard covers `partial` as well as `completed` entries: `_resume_skippable`
accepts a partial entry whose checksum, fingerprint and output all match, and a
no-provider run on a mixed document records exactly that.

### N1. The cost tests did not fail guard-by-guard — fixed

The round-2 max-cost test could not distinguish the ladder's cost-filter argument
from the lane's own pre-call cap: with one region they catch the same case, so
removing either alone left it green. A unit assertion on what
`_equation_lane_provider` RETURNS pins the ladder argument specifically.

Verified by mutation, each guard removed alone:

- drop `max_cost_per_page=` from the lane's `provider_ladder` call — only the new
  ladder test fails, and the end-to-end cap test still passes, exactly as the
  reviewer predicted;
- disable the pre-call per-page cap — only the cumulative-region test fails;
- disable the remaining-budget check — only the budget test fails.

### Canaries after round 3

```
src/socr/pipeline/orchestrator.py:1633  def _equation_lane_backend_addressable
src/socr/pipeline/orchestrator.py:1661  def _equation_lane_provider
src/socr/pipeline/orchestrator.py:220   class _EquationRetryMetadata  (the single write)
```

## Cold review round 4 — 2026-09-02

Round 3 closed finding 4 and N1. One residual on finding 5 remained, plus two
new should-fixes in the round-3 delta.

### 5 residual. A stale root entry survived a failed terminal write — reproduced, fixed

The single terminal write is atomic, but atomicity only protects what it writes.
A matching OLDER entry — completed, latch-free, from a build predating this
lane, its output since deleted — survives a failed terminal write. The run has
by then re-created the output that entry points at, so it becomes resumable
again with no latch and the next run skips the document with the equation page
never read: the original finding-5 outcome by a different route.

Fixed by failing closed at the START of the run rather than only at the end.
`_invalidate_root_entry_for_rerun` (orchestrator.py:464) runs after the resume
gate and before any output is emitted. When a record already exists it is
replaced, through `RootIndex.record` so that stays the sole author, with a
`Status.FAILED` marker carrying an explicit "run in progress" error — neither
`completed` nor `partial`, so both branches of `_resume_skippable` refuse it. If
THAT write fails the run refuses to start and surfaces the error, because
proceeding would re-create the output the stale entry points at. Refusing costs
a re-run; proceeding silently loses an unread equation page.

The reproduction seeds the stale entry directly rather than producing it from a
successful run. That distinction matters and cost one wrong iteration: a run
that DID read the page leaves terminal page sidecars, and restoring those on the
next run is correct behaviour, not a lost retry. Verified by mutation — with the
invalidation disabled, the test fails exactly as the reviewer's canary did.

### N2. A config-resolved backend refusal was latched as transient — reproduced, fixed

The round-3 refusal reason "resolves to backend(s) vllm" is settled by
configuration, not an outage: `qwen_backend=vllm` is a supported deployment. It
was setting the retry latch, so every rerun refused the document skip, restored
the same page and rewrote the latch — idempotent resume permanently defeated for
that configuration. It now joins the other fingerprint-described refusals in
`_equation_lane_availability_refusal`. Pinned by a two-run `process()` test:
the second identical run returns SKIPPED with zero calls.

### N3. Test environment leaked into backend resolution — fixed, and wider than scoped

The `auto` control did not clear `VLLM_BASE_URL`, so it failed on the documented
HPC setup where production correctly resolves `auto` to vLLM. Running the module
with that variable exported showed the exposure is not one test: **22 of 44
failed**, all because the lane correctly refused a provider the environment had
redirected. Production was right in every case; the tests were reading the shell.

An autouse fixture now clears the variable for the module, and the one test that
cares about it is parametrised over both states — unset asserts Ollama is
addressable, exported asserts the refusal. The module passes identically with
and without the variable. The other five P4-R test files were checked under the
same variable and are independent of it.

### Canaries after round 4

```
src/socr/pipeline/orchestrator.py:464   def _invalidate_root_entry_for_rerun
src/socr/pipeline/orchestrator.py:1850  def _equation_lane_availability_refusal
```
