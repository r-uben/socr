# Corrupt-math region recovery lane: default ON

2026-09-04. Owner ruling: `recover_corrupt_math` flips to `True` by default.

## Why

`docs/log/2026-09-03_p6-native-fallback-by-trigger.md` measured
`NATIVE_FALLBACK` triggers over the Papers library (23,190 pages, 21,651
born-digital): `needs_ocr_enhancement` fires on 15.6% of born-digital pages,
and corrupt math alone is 14.5 points of that — the dominant single trigger,
larger than every table trigger combined. `docs/log/2026-09-01_conceptual-revision.md`
frames the fourth ending (demoted native) as, in practice, the corrupt-math
ending. Leaving the lane opt-in meant the default agentic run kept shipping
font-mapped-corrupted mathematics ('=' → '¼', '(' → 'ð') silently inside
otherwise-clean native prose. That is the exact failure this repo's
"no silent content loss" rule exists to prevent (`CLAUDE.md`).

GH-233 is a related but DIFFERENT ticket ("fix(resume): fingerprint ignores
recover_corrupt_math and math_model") — it required the run fingerprint to
hash these two fields, which it already does (see "What a corpus run must
measure" below). It is not the default-on ruling; do not cite it as such.
The ruling itself lives only here, plus the measurement that motivated it in
`docs/log/2026-09-03_p6-native-fallback-by-trigger.md`.

## What changed

- `src/socr/core/config.py`: `PipelineConfig.recover_corrupt_math` default
  `False` → `True`, with a comment pointing at this log and the trigger-rate
  log (not GH-233), and correcting the prior "no-ops with no provider" claim
  (see "The real no-provider contract" below).
- `src/socr/math/recover.py`: **fixed a real bug found while writing the
  no-provider test** (see "Round 2 correction: the splice_math bug" below).
  `splice_math` was replacing an aligned region's native slice whenever a
  crop was retained, even when the region was UNRESOLVED — contradicting the
  GH-271 design record. Now only a RESOLVED region replaces its slice; an
  aligned, crop-retained but unresolved region keeps the native slice in
  place and inserts the crop reference + unresolved marker immediately after
  it. A SECOND bug was found and fixed in the same file while proving the
  fix byte-exact (see "Round 3 correction: the splice_math cursor bug"
  below): both the resolved and unresolved branches now walk `text` with a
  single forward-only cursor instead of always searching from the start, so
  two regions with identical `source_text` each consume their OWN occurrence
  in page order instead of colliding on the first one.
- `src/socr/pipeline/orchestrator.py`, `src/socr/core/state.py`: three
  stale "opt-in" mentions of the corrupt-math lane
  (`_is_corrupt_math_recovery_page` docstring, `_phase_agentic` docstring,
  `corrupt_math_hybrid` field comment) reworded to describe the lane as
  default-on with `--no-recover-corrupt-math` as the kill switch (see
  "Wording" below).
- `src/socr/cli.py`: `--recover-corrupt-math` was a one-way `is_flag` option
  (`build_config(recover_corrupt_math: bool = False)`, unconditional
  `if recover_corrupt_math: config.recover_corrupt_math = True`). Replaced
  with the paired `--recover-corrupt-math/--no-recover-corrupt-math`
  (default `None`), following the `--dual-pass-tables/--no-dual-pass-tables`
  (P5) and `--no-equation-region-lane` (P4-R) precedent: `build_config` now
  takes `recover_corrupt_math: bool | None = None` and only writes it when
  `_explicitly_given("recover_corrupt_math")`, so a config file/profile value
  is not silently clobbered by an unset CLI default, and `--no-recover-corrupt-math`
  is now the documented kill switch. Help text updated to state the default,
  cite the owner ruling by log path (not GH-233), and describe
  `--no-recover-corrupt-math` accurately: it restores ORDINARY whole-page
  routing for the page — native fallback happens only if no engine actually
  runs, it is not an unconditional "keep native prose" outcome.
- `docs/ARCHITECTURE.md`: the CLI module summary now lists the paired flag
  and its default-on status, citing this log.
- `tests/test_math_recover.py`: strengthened
  `test_splice_empty_latex_keeps_visible_failure_and_crop` to assert the
  corrupt native glyphs are KEPT (not replaced) and that the crop/marker
  follow them — this is the one existing test that had pinned the old buggy
  replace behavior (it asserted `"¼" not in out`, which the fix now
  contradicts by design). Strengthened
  `test_identical_source_occurrences_align_and_replace_independently` with
  an ordering assertion, and added
  `test_identical_unresolved_source_occurrences_each_get_own_adjacent_evidence`
  — the round-3 cursor-bug reproducer (see below).
- `tests/test_orchestrator.py`: replaced the no-provider difference test
  (now `test_agentic_corrupt_math_default_flip_no_provider_is_additive_only`)
  with an end-to-end version that fabricates nothing in
  `socr.math.recover` — real corrupt-math geometry on a real fixture PDF,
  real (refused) network call — and asserts a DIFFERENCE rather than an
  absolute status (see "The real no-provider contract" below), at both the
  document AND page-sidecar level, plus a byte-exact additivity check.
  Added the `_corrupt_math_pdf` fixture helper next to `_real_pdf`.
- `tests/test_corrupt_math_lane_default_cli.py`: new. Pins
  `PipelineConfig().recover_corrupt_math is True` directly, plus the full
  paired-CLI precedence rules (unconfigured default, explicit on/off, YAML
  true/false with no CLI override, YAML overridden either direction) for
  BOTH `process` and `batch`. Both commands' `--help` are checked for the
  paired flag AND the literal string "Default: on". Flipping the default
  back to `False` was confirmed to fail the `PipelineConfig()` pin (isolated
  check, reverted).

## Registries already covering this flag — verified, not modified

Three GH-142-style flag registries were already correct before this change
and did not need extending:

- `tests/test_gh142_flag_audit.py` — `recover_corrupt_math` was already
  classified `AGENTIC` ("corrupt-math recovery routing").
- `tests/test_cli_flag_agentic_status_gh142.py` — `recover_corrupt_math` was
  already in `_LIVE` (read by `_is_corrupt_math_recovery_page`).
- Run fingerprint (`orchestrator.py` around `_run_fingerprint`) already hashes
  both `recover_corrupt_math` and `math_model` (the latter only while the
  flag is on), so a corpus reprocesses under the new default without a
  separate pin. This is the actual scope of GH-233.

Ran `tests/test_gh142_flag_audit.py` (41 passed) and
`tests/test_cli_flag_agentic_status_gh142.py` + `tests/test_fingerprint_flag_coverage.py`
(52 passed) after the flip to confirm.

## P1 golden/byte-identity guard (`tests/test_p1_golden_flag_pinned.py`)

Not extended. That guard is scoped to `table_judge_ladder` specifically
because flipping *that* flag changes table-page routing through a ladder
gate that goes empty in CI (ollama absent), turning table pages UNVERIFIED
in a machine-dependent way. `recover_corrupt_math` is a different shape: a
region it detects and cannot resolve keeps the native slice untouched and
only ADDS a crop reference and unresolved marker next to it (see next
section) — the document's status does not silently move to a
machine-dependent `SUCCESS`, and the shipped native bytes for an unresolved
region do not change either.

`_AUDITED_GOLDEN_MODULES` in that guard lists 13 paths (not 14 — corrected
here). Seven modules in the suite reference `has_corrupt_math` (corrected
here — an earlier pass found only four, missing three by an incomplete
grep): `test_cli_flag_agentic_status_gh142.py`, `test_document_state.py`,
`test_encoding_corruption.py`, `test_equation_lane_routing_p4r.py`,
`test_gh326_presence_gate.py`, `test_orchestrator.py`, and
`test_p35_cold_review_round3.py`. None of these seven is in the 13-module
audited-golden list, so the guard's scope is unaffected by this flip.

Verified directly rather than assumed: reverted `recover_corrupt_math` to
its pre-flip default (`False`) in an isolated check and reran all seven
`has_corrupt_math`-referencing modules together with the full 13-module
audited-golden set (402 passed, 1 expected xfail) — nothing in that combined
set implicitly depended on the new default, and the guard's own tests still
passed unaffected. Reverted the default back to `True` immediately after
(confirmed via `grep -n "recover_corrupt_math: bool" src/socr/core/config.py`).
If a future golden module adds a corrupt-math fixture, it should pin
`recover_corrupt_math` explicitly at that point rather than pre-emptively
here.

## Round 2 correction: the splice_math bug

The FIRST version of this log, and the test it described, claimed the
no-provider (CI) contract was "identical native text in both flag states,
plus additive-only diagnostics when the flag is on." That test used a blank
one-page fixture PDF, so `recover_math_regions`'s geometry pass found no
corrupt region at all and returned `[]` before reaching `splice_math` — it
only exercised the OTHER branch (the page-level detector claims corrupt
math, but the PDF-geometry pass finds no matching region), not "a real
region is detected but the model cannot resolve it."

The SECOND version fixed the region-detection gap, but fabricated the
`CorruptMathRegion` return value directly via a patch on
`socr.math.recover.recover_math_regions` rather than exercising a real
page and a real (failing) model call. Running it surfaced the TRUE contract
— and the true contract was a bug: with the flag on, the shipped markdown
lost the raw corrupted glyphs entirely and gained a crop reference + an
unresolved marker IN THEIR PLACE. That is `splice_math` (`src/socr/math/
recover.py`) replacing an aligned region's native slice whenever a crop was
retained, even when the region was unresolved — which directly contradicts
the GH-271 design record
(`docs/log/2026-08-22_corrupt-equation-region-guardrail.md`): "Failure at
any boundary keeps the native source text and appends explicit evidence
instead of silently deleting content." A real code bug, not a documentation
gap — fixed in this ticket (see "What changed" above): only a RESOLVED
region now replaces its slice; an aligned, crop-retained but unresolved
region keeps the native slice untouched and inserts the crop + marker
immediately after it.

## The real no-provider contract

Rebuilt the test a third time, end-to-end and unmocked: a real fixture PDF
with an actual font-mapped corrupt equation line (`_corrupt_math_pdf` in
`tests/test_orchestrator.py`), read through the real, unpatched
`recover_math_regions`/`splice_math` seam, against a real but unreachable
Ollama endpoint (`ollama_host="http://127.0.0.1:1"`, connection refused —
not a mock). The OCR call is made for real, fails for real, and is retried
once for real (the documented empty-response retry in
`recover_math_regions`). Observed output before writing any assertion:

- **Flag off (`False`):** the gate `_is_corrupt_math_recovery_page` never
  fires. The page goes through the ordinary whole-page OCR path; the empty
  provider ladder means no engine runs, so it falls back to native text
  unchanged — the raw corrupted glyphs (`PðA or BÞ ¼ PðAÞ þ PðBÞ`) ship
  verbatim, no crop reference, no marker, `result.error is None`.
- **Flag on (`True`):** the region is detected, geometrically real, aligned
  against the native text, and the crop is retained (rendering needs only
  PyMuPDF, not a model) — but unresolved, because the model call fails
  closed against the unreachable endpoint. Per the now-fixed `splice_math`:
  the native slice (`PðA or BÞ ¼ PðAÞ þ PðBÞ`) ships UNCHANGED, and
  `![Corrupt equation crop](...)` plus `[corrupt equation unresolved: ...]`
  are inserted immediately after it. `result.error` is set to `"corrupt
  equation candidate unverified on page(s) 1"`.

**Per CLAUDE.md ("pin a difference, not a value")**, the test does not pin
an absolute `DocumentStatus` — that machinery is provider-dependent and this
repo has already been burned once (#253/#257) by pinning an outcome that
only held on one machine. What it does pin:

- The two runs reach the **same** page status as each other (not a specific
  named status).
- The **same** native slice — corrupt glyphs included — ships in **both**
  flag states.
- The flag's only effect with no provider is **additive**: a crop reference
  and an unresolved marker inserted immediately after the unchanged native
  slice, plus `result.error` set. Nothing is removed from the shipped text
  in either state.

`test_agentic_corrupt_math_default_flip_no_provider_is_additive_only` (in
`tests/test_orchestrator.py`) pins exactly this, at TWO levels: the
page-level sidecar status and winning-output engine (`"native"` vs.
`"native+math"`), and a byte-exact check — stripping precisely the
crop-plus-unresolved diagnostic block `splice_math` inserted from the
flag-on page fragment reproduces the flag-off fragment byte-for-byte.

## Round 3 correction: the splice_math cursor bug

Writing the byte-exact additivity test above surfaced a SECOND real bug in
`splice_math`, independent of the round-2 replace-vs-keep bug: two regions
sharing IDENTICAL `source_text` (a duplicate corrupt equation appearing
twice on one page — plausible; the same font-mapped symbol run can recur)
both matched `str.replace(source, ..., 1)`'s FIRST occurrence, because the
unresolved-insertion branch re-embeds `source` as the literal prefix of its
own replacement text. A from-the-start search for the second region then
re-matched that freshly-inserted text at the SAME location instead of
advancing to the real second occurrence — both crops piled up next to the
first occurrence, and the second occurrence shipped with no evidence
attached at all (silent-loss shape, the exact thing this lane exists to
prevent).

Reproduced first (`tests/test_math_recover.py::
test_identical_unresolved_source_occurrences_each_get_own_adjacent_evidence`),
confirmed failing against the pre-fix code, then fixed: `splice_math` now
walks `text` with a single forward-only cursor. Each region searches for its
source starting at the position immediately after wherever the PREVIOUS
region was spliced in (`text.find(source, cursor)`), and splices by index
rather than `str.replace(..., 1)`; the cursor advances to the end of
whatever was just inserted. The resolved (replace) branch had the same
latent shape and was fixed with the identical cursor in the same pass (it
happened to already pass its existing test because a full replacement does
not re-embed `source`, but a second identical resolved region would have hit
the same first-occurrence bug under different circumstances — e.g. if an
intervening replacement text ever contained the literal source bytes).
Regions are supplied in page/geometry order (`_recovery_groups`'s existing
contract), so the forward-only cursor also matches natural reading order.
Verified: `test_identical_source_occurrences_align_and_replace_independently`
(resolved case, strengthened with an ordering assertion) and the new
unresolved regression both pass; the full `tests/test_math_recover.py` suite
(29 tests) is green.

## Wording: "opt-in" corrected to reflect the default-on lane

`--no-recover-corrupt-math` help text, the `_is_corrupt_math_recovery_page`
docstring and the `_phase_agentic` docstring
(`src/socr/pipeline/orchestrator.py`), and the `corrupt_math_hybrid` field
comment (`src/socr/core/state.py`) all called the lane "opt-in" — accurate
before this ruling, stale after it. Reworded all four to say the lane is on
by default with `--no-recover-corrupt-math` as the kill switch. Checked
broadly for any other "opt-in" language describing this flag in `src/` and
`docs/ARCHITECTURE.md`; found none remaining. The one surviving "opt-in"
reference (`docs/log/2026-08-22_corrupt-equation-region-guardrail.md`) is a
dated historical decision record describing the state AT THE TIME it was
written and is correctly left alone.

## Cost

Per detected corrupt-math crop region (pages that are born-digital, have
native text, have `has_corrupt_math=True`, are not
native-rotated-shredded, and carry no table signal —
`_is_corrupt_math_recovery_page`): **up to two model calls, not one** — an
empty model response is retried once before the region fails closed
(`recover_math_regions` in `src/socr/math/recover.py`). A page can also
contain more than one corrupt region, each independently making up to two
calls. So the true unit cost is "up to two calls per unresolved region,
times however many regions a page has" — not "one call per page." At the
measured 14.5% of born-digital pages carrying this trigger, this is now an
always-on cost for any run with a provider configured — previously it was
zero unless explicitly opted in. No-provider runs (CI, `--strict-local` with
no local model, or an unreachable endpoint) pay no model cost — the crop
render still happens (it needs only PyMuPDF), and the OCR call is either
short-circuited (`model_disabled_reason`, e.g. `--strict-local`) or made and
fails closed (unreachable endpoint, up to two attempts) — and instead pay
the cost described above: a region that fails closed, keeping the native
slice untouched and appending a crop+marker beside it, rather than an
undetected silent corruption.

## What a corpus run afterwards must measure

- **Reprocess scope is run-wide, not corrupt-math-specific**: `_run_fingerprint`
  hashes the RESOLVED RUN config, not anything page-specific, and
  `recover_corrupt_math` is one top-level key in it. So flipping the default
  invalidates the fingerprint — and forces a one-time reprocess — for EVERY
  terminal page of EVERY document under a resumed run, not only pages that
  carry `has_corrupt_math=True`. A page with no corrupt math at all still
  reprocesses once; it simply produces byte-identical output on the far
  side, the same shape as any other run-wide config-key change. This is
  expected, not a regression. The reprocess-invalidation mechanism itself
  (hashing `recover_corrupt_math`/`math_model` into the fingerprint at all)
  is what GH-233 actually fixed; this ruling depends on that fix already
  being in place.
- **New AUDIT_FAILED rate**: documents that were previously `SUCCESS` (silent
  corrupt math shipped, native fallback never triggered because e.g. the
  whole-page ladder had a working rung) will now show `AUDIT_FAILED` when a
  region is detected and cannot resolve, because the region-lane gate takes
  the page before the whole-page ladder gets a chance to ship a clean
  result. Track this specifically on the ~14.5%-of-pages subset the
  trigger-rate log identified, not the corpus overall — the other ~85%
  should be unaffected. Shipped native bytes for an already-unresolved page
  are unchanged by the flip (see "Round 2 correction" above) — only the
  crop reference and marker are new, appended beside the untouched text.
- **Region-recovery yield**: of pages routed into the lane with a provider
  present, what fraction of DETECTED regions produce a 1A-syntax-valid,
  numerically-plausible LaTeX candidate vs. remain unresolved (crop+marker
  only) — this is the number that tells us whether the lane is worth its
  per-region model cost at scale, as opposed to just surfacing the problem.
- **Regions-per-page and retry rate**: how many corrupt regions a typical
  triggered page carries, and what fraction of first model calls come back
  empty and consume the retry — both multiply the raw per-page call count
  measured above and directly drive latency/cost.
- **Wall-clock/cost delta**: up to two calls per unresolved region, at
  corpus scale (14.5% of ~21,651 born-digital pages in the Papers library
  alone ≈ 3,130 pages, times however many regions and retries each carries)
  — measure actual latency/cost against the `escalation_timeout_sec` budget
  and the configured `math_model` price.
