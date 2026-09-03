# P6 Stage A/B — Finalization-Aware Page Disposition Contract and Bucket Derivation

Ticket: programme item P6 of `docs/log/2026-09-01_conceptual-revision.md`. Closes #176; implements Stage A and Stage B of `docs/log/2026-09-02_p6-selector-collapse-design.md`.

## 0. Grounding and Canary

Python environment and module location:
- `socr.__file__`: `/Users/rubenffuertes/repos/tools/socr-p6b/src/socr/__init__.py`

AST Canary Verification:
```bash
python3 -c "import ast;t=ast.parse(open('src/socr/core/manifest.py').read());f=[n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name=='_select_page_output_tagged'][0];print(f.name, f.lineno, f.end_lineno, len([r for r in ast.walk(f) if isinstance(r,ast.Return)]))"
```
Output: `_select_page_output_tagged 1107 1586 15`

The 15-return, 16-row selector structure is preserved intact in `_select_page_output_tagged` (with rows 11 and 12 sharing return line 1513 via conditional `SelectionProvenance.NATIVE_FALLBACK` / `SelectionProvenance.NATIVE_CLEAN`). All 16 selector endings remain in their load-bearing precedence order.

---

## 1. The Four-Ending Contract and Demoted-Native Exit Criterion

In accordance with §8 Q1 panel ruling (Codex & Gemini synthesis), `PageEnding` defines exactly four public endings naming the bytes that ship:

1. `NATIVE_PROSE`: Native digital text authored clean prose (`PageStatus.SUCCESS`, `audit_passed=True`).
2. `MODEL_OUTPUT`: Accepted or flagged-kept model reading (`PageStatus.SUCCESS` or `WARNING`).
3. `FAIL_CLOSED_MARKER`: Refused reading or no usable text produced, shipping a `[page N failed: ...]` failure marker (`PageStatus.ERROR`, `audit_passed=False`).
4. `DEMOTED_NATIVE`: Born-digital page with native text whose recovery was exhausted, shipping demoted native prose (`PageStatus.WARNING`, `audit_passed=False`).

### Demoted-Native Exit Criterion
`DEMOTED_NATIVE` is a measured, temporary deviation from the three-ending ruling. As documented in `PageEnding.DEMOTED_NATIVE`:
- Triggers: `needs_ocr_enhancement`, `chart_asset_render_failed`, `text_grid_rejected`, and residual native-table defects.
- Exit criterion: In a subsequent ticket, pages across the corpus exhibiting these triggers will be enumerated and hand-checked for fidelity. Each trigger will then be assigned independently to either `NATIVE_PROSE` (if the prose text layer is intact and trustworthy) or `FAIL_CLOSED_MARKER` (if the page cannot be safely shipped without OCR).

---

## 2. Normalized Reason vs Private Selection Provenance

- **`SelectionProvenance` (Internal)**: Formerly `WinnerKind`. Retains all 16 private selector rows in declaration and cascade order. It is strictly internal and never exposed as the public disposition or serialized to sidecars/manifests.
- **`PagePrimaryReason` (Normalized Cause)**: Normalizes the 16 internal selector rows into 12 distinct root causes, merging rows that answer the same fundamental question (e.g. structure class passing/flagged/floor normalize to `STRUCTURE_CLASS`; D3 model-kept and native floor normalize to `NATIVE_TABLE_UNVERIFIABLE`; unverified and flagged model attempts normalize to `UNACCEPTED_OUTPUT_KEPT`). It also includes `INVALID_TABLE_EMISSION` for post-selection guard rewrites.
- **`PageDisposition` (Public Contract)**: A frozen dataclass consisting of `(ending: PageEnding, primary_reason: PagePrimaryReason)` with JSON serialization (`to_dict()` and `from_dict()`).

---

## 3. Exact Six Disposition Pairs

The six assemble buckets derived unconditionally from `PageDisposition` in `_derive_disposition_buckets` are pinned to the following exact pairs:

| Assemble Bucket | Ending | Primary Reason | Source Provenance |
|---|---|---|---|
| `d3_model_table_pages` | `PageEnding.MODEL_OUTPUT` | `PagePrimaryReason.NATIVE_TABLE_UNVERIFIABLE` | `UNVERIFIABLE_TABLE_MODEL_KEPT` |
| `d3_floor_pages` | `PageEnding.FAIL_CLOSED_MARKER` | `PagePrimaryReason.NATIVE_TABLE_UNVERIFIABLE` | `UNVERIFIABLE_TABLE_NATIVE` |
| `flagged_model_pages` | `PageEnding.MODEL_OUTPUT` | `PagePrimaryReason.NATIVE_TABLE_DISTRUST` | `FLAGGED_MODEL_KEPT` |
| `structure_class_model_pages` | `PageEnding.MODEL_OUTPUT` | `PagePrimaryReason.STRUCTURE_CLASS` | `STRUCTURE_CLASS_GRID_PASSING`, `STRUCTURE_CLASS_GRID_FLAGGED` |
| `structure_class_floor_pages` | `PageEnding.FAIL_CLOSED_MARKER` | `PagePrimaryReason.STRUCTURE_CLASS` | `STRUCTURE_CLASS_FLOOR` |
| `corrupt_math_hybrid_pages` | `PageEnding.MODEL_OUTPUT` | `PagePrimaryReason.CORRUPT_MATH_HYBRID` | `CORRUPT_MATH_HYBRID` |

---

## 4. Why `native_fallback_pages` is Different

`native_fallback_pages` in `_phase_assemble` is an audit diagnostic bucket rather than a pure selection disposition. It answers: *"Did OCR get attempted for a non-S1/non-D3/non-flagged-model reason, with native bytes ultimately shipping demoted?"*
Because it incorporates pipeline-specific exclusion predicates (excluding D3, structure class, flagged models, and corrupt math hybrids), deriving it directly from `DEMOTED_NATIVE` would conflate selection outcome with pipeline execution history. Per §8 of the design doc, `native_fallback_pages` remains explicitly predicate-derived in `orchestrator.py` and is not derived from `PageDisposition`.

---

## 5. Single-Record Pre/Post-Transform Lifecycle

To eliminate drift between assembled Markdown, page fragments, sidecars, and manifest blobs:
1. `_select_and_finalize_page` executes selection and runs final emission and ladder guards in a single pass, producing an authoritative `FinalizedPageRecord(output, disposition, selection_provenance)`.
2. **Pre-transform Snapshot**: `_phase_assemble` invokes `finalized_page_records(state)` once to compute `pre_records` for bucket derivation and defect checks.
3. **Post-transform Snapshot**: After figure embedding and Markdown transformations, `final_records = finalized_page_records(state, saved_body=final_text)` is computed once. If post-transform text introduces an invalid table emission, the emission guard converts the page to `(PageEnding.FAIL_CLOSED_MARKER, PagePrimaryReason.INVALID_TABLE_EMISSION)` consistently across all surfaces.
4. `records=final_records` is threaded directly into `_rewrite_all_fragments`, `_flush_page_sidecar`, and `_write_manifest`, ensuring no downstream component performs redundant selection or guarding.

---

## 6. Additive Disposition Exception for Sidecar and Manifest Identity

- `ManifestEntry` and sidecar metadata (`pages/NNNNN.json`) now persist `disposition: {"ending": "...", "primary_reason": "..."}`.
- This addition is strictly additive: parsed equality holds everywhere else when `disposition` is omitted.
- Legacy sidecars and manifests lacking `disposition` load cleanly with `disposition=None`.
- `run_fingerprint` and `_page_blob_key` are computed exclusively from `winning_output` and configuration parameters, ensuring zero cache or resume fingerprint invalidation.

---

## 7. Resume Results

A 22-cell difference matrix across all 11 sidecar shapes (`clean_success`, `audit_rejected`, `error_status`, `warning_status`, `failure_marker`, `ladder_rejected_d1b`, `ladder_unverified`, `ladder_incomplete`, `structure_floor`, `equation_retry_pending`, `provisional`) was executed under both providerless and active-provider environments.
- In 100% of cases, `_load_terminal_page` yielded identical return-versus-`None` decisions and identical reconstructed `PageOutput` fields (`text`, `status`, `audit_passed`, `failure_mode`, `engine`) whether `disposition` was present, explicitly `None`, or omitted entirely.

---

## 8. Legacy Seam Deletions and Source LOC

- Legacy helpers `shipped_winner_kind` and `_finalize_page_output` were consolidated into `page_disposition` and `_select_and_finalize_page`.
- `WinnerKind` was replaced throughout with `SelectionProvenance` internally and `PageDisposition` publicly.

### LOC Statistics

`git diff --numstat main -- src/socr/core/manifest.py src/socr/pipeline/orchestrator.py`:
```
283	96	src/socr/core/manifest.py
168	192	src/socr/pipeline/orchestrator.py
```

`git diff --stat main -- src/socr/core/manifest.py src/socr/pipeline/orchestrator.py`:
```
 src/socr/core/manifest.py         | 379 ++++++++++++++++++++++++++++----------
 src/socr/pipeline/orchestrator.py | 360 +++++++++++++++++-------------------
 2 files changed, 451 insertions(+), 288 deletions(-)
```

Note: Net count is positive (+163 LOC) due to the addition of the Stage A disposition vocabulary, dataclasses, serialization logic, and single-record lifecycle plumbing. Net deletion will occur in subsequent merge stages (S3/S4).

---

## 9. Gate Verification

1. **Test Suite**:
   Command: `PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-p6b/src ~/venvs/socr/bin/pytest tests -q`
   Summary: `3335 passed, 4 xfailed, 5 warnings in 124.98s (0:02:04)`

2. **Code Formatting**:
   Command: `uvx ruff@0.16.0 format --check .`
   Result: `510 files already formatted` (clean, exit code 0)

3. **Structural and Safety Invariants**:
   - `git diff --check`: Clean (no whitespace or merge marker issues).
   - Forbidden modules: Only `manifest.py`, `orchestrator.py`, and test files touched.
   - Invariants confirmed:
     - No direct `DocumentState.engine_runs` mutations.
     - No unexpected `audit_passed` mutations.
     - No hardcoded empirical thresholds or page counts.
     - Selector preserves 15 returns and 16 selection provenance values.
     - No selector ending branches merged prematurely.

---

## 10. Cold review round 1

Reviewer: `r213053-review-1`, `.troupe/runs/20260902-213053-standard-feature/outbox/review.md`.
Five findings, all addressed. Reproduced first, in every case, before anything was changed.

### Finding 1 (High) — buckets derived from the disposition lost their old membership

**Reproduced: yes**, and the divergence is wider than the review found.

A seven-page corpus (`tests/p6_corpus_fixture.py`) was run through `_phase_assemble` twice:
once on the pre-change sources (`git archive HEAD src` into a temp tree) and once on this
tree. Pages 4 and 5 are the reviewer's two shapes — the full D3 flag conjunction plus a
passing non-native `best_output`, with and without a qualifying refused grid. Measured:

| bucket | pre-change | as shipped for review |
| --- | --- | --- |
| `d3_floor_pages` | `[2, 4]` | `[2]` |
| `d3_model_table_pages` | `[3, 5]` | `[3]` |

Downstream: the CLI D3 lines, the `table_region_unverifiable` / `d3_floor_model_table_kept`
audit events, the per-page sidecar `audit_events`, and the document's tables-trust note all
moved with them.

The suite-wide difference guard added for finding 2 then found a **third** bucket with the
same defect, in a real end-to-end fixture
(`tests/test_table_judge_gate.py::TestProcessFlagDifference::test_native_lane_is_witnessed_too`):
`flagged_model_pages` was `[1]` before and `[]` after.

**Root cause, one sentence:** the D3 conjunction and `flagged_model_page_output` are verdicts
on the NATIVE lane, and a page can carry either while selection ends somewhere else entirely —
a passing non-native `best_output` returns at the `PASSING_BEST_OUTPUT` ending long before
those branches are reached, leaving the page with the same disposition an ordinary clean model
page carries. No disposition pair can separate them; `tests/test_p6_stage_ab_difference.py::
TestColdReviewD3Shapes::test_the_reviewed_pages_are_not_disposition_derivable` pins that.

**Ruling applied: equality with the old predicate wins.** `_derive_disposition_buckets` now
splits three ways:

* **Three buckets stay flag-derived**, with the pre-change predicates verbatim:
  `d3_model_table_pages`, `d3_floor_pages`, `flagged_model_pages`.
* **Three buckets read the selection tag**: `structure_class_model_pages`,
  `structure_class_floor_pages`, `corrupt_math_hybrid_pages`. Their old predicates
  (`structure_class_grid_winner`, `structure_class_floor_applies`, `shipped_winner_kind(...)
  is CORRUPT_MATH_HYBRID`) are the functions the selector itself calls on the way to those
  endings, so the tag answers the same question by construction.
* They read `selection_provenance`, **not** `disposition`. The two differ by the
  post-selection guards: `_apply_table_emission_guard` can rewrite any ending to
  `(FAIL_CLOSED_MARKER, INVALID_TABLE_EMISSION)`, which would silently empty a bucket that
  feeds `pages_ok`, an audit event and a CLI line that are all still true.

**§8's inventory is amended: three of the six, not six.** §3 of this log (the "exact six
disposition pairs" table) is superseded by the split above. Whether a native-lane verdict
should outrank a passing winner is a real question; it is a behaviour change, and it is not
stage A/B's.

Two further silent behaviour changes were caught by the same corpus and reverted: the
`d3_floor_model_table_kept` and `structure_class_model_table_kept` audit events had been
switched from naming the CANDIDATE engine (`d3_floor_kept_model_output(state.pages[n])`,
`structure_class_grid_winner(state.pages[n])`) to naming the shipped record's engine. On page
5 that changed the recorded engine from `gemini` to `qwen`.

### Finding 2 (High) — the required difference test was missing

**Reproduced: yes** (the test did not exist). Added in three parts.

1. `tests/conftest.py` — the test-only seam. Holds the pre-change predicates, reconstructed
   verbatim from `git show HEAD:src/socr/pipeline/orchestrator.py`, plus an **autouse** guard
   that wraps `_derive_disposition_buckets` for the whole suite. Every fixture that drives
   `_phase_assemble` therefore asserts old-membership == new-membership on every real
   assemble — the 34 modules that drive it, the PP-2 golden corpus fixture included — with no
   per-module opt-in to forget. This guard is what found the `flagged_model_pages` case.
2. `tests/test_p6_stage_ab_difference.py::TestColdReviewD3Shapes` — the reviewer's two
   fixtures, asserted directly against those predicates.
3. `tests/test_p6_stage_ab_difference.py::TestPreChangeByteIdentity` — byte identity of
   document status, metadata notes, audit events, CLI summary, final `.md`, page fragments,
   sidecars and manifest against `tests/fixtures/p6/prechange_assemble.json`, which is the
   same corpus captured from the pre-change sources.

The comparison is a DIFFERENCE — same fixture, only the source tree varies — never an
absolute measured tuple, and `_phase_assemble` consults no provider, so it is hermetic in CI.

Two field families are excluded from the byte comparison, and the exclusion is itself pinned
(`test_disposition_is_the_only_added_key`): `disposition`, stage A's additive field, which is
proven to be the **only** key either surface gained or lost; and
`socr_source_digest` / `run_fingerprint` / `fingerprint` / `input_checksum` / `pdf_file_hash`,
of which the first three move whenever any source byte changes (that is their job) and the
last two hash a PyMuPDF-written PDF, which embeds a creation timestamp and is not byte-stable
between runs at all.

The seam lives only in `tests/`. Production carries no pre-change path.

### Finding 3 (Medium) — two record paths

**Reproduced: yes.** `_final_winning_outputs` was populated by nothing in production and by
four sites in `tests/test_gh317_structure_class_floor.py`; the orchestrator stitched the
injected output together with a freshly finalized record's disposition, so sidecar
`winning_output` and `disposition` could disagree.

Production now has ONE path: `_final_records`, or the explicit `record=` argument
`_flush_page_sidecar` already accepted. The attribute, its initialisation and the stitching
branch are deleted. The four test sites pass `record=_legacy_record(state, 1, winner)`; that
helper does the stitching in the test file, where the fixture that needs it is — these tests
are about sidecars written by OLDER builds, whose bodies the current selector cannot
reproduce.

### Findings 4 and 5 (Low) — unused import, helper parked mid-imports

**Reproduced: yes, both.** `page_disposition` is no longer imported by `orchestrator.py`
(nothing in the module calls it; the buckets read the record). `_derive_disposition_buckets`
now sits below the whole import block, asserted structurally: last top-level import is line
58, the helper starts at line 61.

### Tests retargeted, not weakened

* `tests/test_p6_disposition_buckets.py` — the pinned "six disposition pairs" table is
  replaced by `TAG_DERIVED_BUCKET_TAGS` (three buckets, pinned to their selection tags) and
  `FLAG_DERIVED_BUCKETS` (three, pinned to ignoring the record entirely). Two assertions were
  ADDED, not removed: `test_a_guard_rewritten_page_keeps_its_tag_derived_bucket` pins why
  these read the tag rather than the disposition, and
  `test_the_flag_derived_buckets_ignore_the_record_entirely` pins the inverse.
* `tests/test_gh292_hybrid_bucket_matches_the_tag.py` — the AST pin accepts a
  `CORRUPT_MATH_HYBRID` member of any tag vocabulary (`SelectionProvenance`,
  `PagePrimaryReason`, `WinnerKind`) instead of `PagePrimaryReason` alone. GH-292's actual
  demand — ask the manifest which ending selection took, never re-derive it — is unchanged,
  and `test_the_orchestrator_bucket_does_not_re_derive_the_page_state_flag`, the assertion
  that forbids reading the `PageState` flag back, is untouched. (Stage C then replaced this
  file's behavioural pins with `TestTheOrchestratorBucketReadsTheShippedDisposition`, which
  asks for the exact disposition pair; the AST pin above survived that rewrite.)
* `tests/test_gh317_structure_class_floor.py` — four sites retargeted from
  `_final_winning_outputs` to `record=`. The assertions themselves are unchanged.
* `tests/test_r7_winner_kind_tags.py`'s bijection pin is untouched by this round: 15 selector
  returns, 16 `SelectionProvenance` members, both re-verified by AST after the change.

### Gates

```
PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-p6b/src ~/venvs/socr/bin/pytest tests -q
3351 passed, 4 xfailed, 5 warnings in 126.60s     exit 0

uvx ruff@0.16.0 format --check .
514 files already formatted                       exit 0
```

`socr.__file__` = `/Users/rubenffuertes/repos/tools/socr-p6b/src/socr/__init__.py`.

---

## 11. Cold review round 2

Verdict: NOT MERGEABLE, ten findings. All addressed. Each was reproduced against
`git archive HEAD src` before anything was changed, and each fix is pinned by a test
verified to FAIL when the reviewed code is put back.

New pins live in `tests/test_p6_cold_review_round2.py` (14 tests).

### Finding 1 (blocking) — the selector predicate was widened

**Reproduced: yes.** HEAD's predicate is exactly `p.best_output and
p.best_output.audit_passed`; the reviewed tree added `or p.table_ladder_disposition is
not None`, promoting an audit-REJECTED `best_output` into the passing arm. Two shapes,
both measured HEAD-vs-tree:

| shape | HEAD | reviewed tree |
| --- | --- | --- |
| born-digital table page, native body, rejected qwen winner, `TABLE_REJECTED` | ships the D3 fail-closed marker, `ERROR/native` | ships native prose, `WARNING/native` |
| the same page without tables (nothing distrusts native) | `WARNING/native` | ships `REJECTED MODEL BODY`, `SUCCESS/qwen` |
| assemble-time backfill: native GFM table, rejected qwen winner, `table_judge_ladder=True` | sidecar `native/warning` | sidecar `qwen/success` |

**Reverted to HEAD's predicate.** Pinned by `TestFinding1SelectorIsNotWidenedByALadder
Disposition` — four tests, all four verified to fail with the widened predicate back in.
Three of them pin a DIFFERENCE: the same fixture with and without
`table_ladder_disposition` must produce the same winning text and engine, because a
ladder disposition changes a page's demotion, never which bytes win.

Note for the record: the widening was load-bearing for something. It kept the new
`disposition` field stable across resume. That is dealt with properly below.

### Finding 2 (blocking) — the ladder guard overwrote an existing terminal

**Reproduced: yes.** HEAD stamps the page's ladder disposition only when
`output.failure_mode is FailureMode.NONE`. The reviewed `or output.failure_mode in
_LADDER_TERMINAL_FAILURE_MODES` replaced one recorded terminal with another: an output
carrying `TABLE_REJECTED` on a page whose disposition is `TABLE_UNVERIFIED` came out
`TABLE_UNVERIFIED`. **Reverted.** Pinned by `TestFinding2LadderGuardDoesNotOverwriteA
Terminal`, with a control that the guard still fills an EMPTY failure mode.

### Finding 3 (blocking) — the disposition did not name what shipped

**Reproduced: yes.** Only `_TABLE_EMISSION_FAILED_RE` was recognised, so a page whose
body is `[page 1 failed: timeout during extraction]` was published as
`(MODEL_OUTPUT, UNACCEPTED_OUTPUT_KEPT)` — a page that shipped no content, described as
a kept model reading.

Fixed through the SHARED recogniser, per the ruling. `_select_and_finalize_page` now
asks `is_page_failed_marker(output.text)`, the same function every other consumer uses,
and any recognised whole-page marker is `FAIL_CLOSED_MARKER`. The reason names the
family via `_MARKER_FAMILY_REASONS`; when selection ALREADY knew the page was
fail-closed its own reason is kept, being strictly more specific than the family.

One new `PagePrimaryReason` member, `SHIPPED_FAILURE_MARKER`, for a marker selection did
not account for. The vocabulary is now 14 members, so
`test_page_primary_reason_is_a_normalized_cause_not_a_row_rename` (which requires fewer
than 15) is untouched. The unverifiable-table family is deliberately NOT given its own
member: its prose does not say which lane distrusted the table, and every path that
authors it already carries a fail-closed provenance whose lane-specific reason is kept.
Naming a lane the bytes do not carry would be an invention.

Pinned by `TestFinding3TheEndingIsReadFromTheShippedBytes`: every marker family plus the
reviewer's free-form timeout marker, plus a control that ordinary prose is untouched.

### Finding 4 (blocking) — three buckets read selection provenance

**Reproduced: yes** (the emission-guard-rewritten hybrid still emits
`corrupt_math_hybrid_shipped` and its CLI line). **Ruling applied as given: behaviour
preservation won in stage B.** The provenance derivation stayed in stage B to preserve
behaviour across the initial rollout, while documenting that the three selection-shaped
buckets would migrate to exact shipped `PageDisposition` pairs in stage C.

**Landed Stage-C resolution:** Stage C (`docs/log/2026-09-03_p6-stage-c-shipped-buckets.md`)
landed the migration of `structure_class_model_pages`, `structure_class_floor_pages`, and
`corrupt_math_hybrid_pages` to exact public `PageDisposition` pair equality in
`_derive_disposition_buckets`. `SelectionProvenance` is strictly internal and is never
read for bucket membership. The inverted test is
`tests/test_p6_disposition_buckets.py::test_a_guard_rewritten_hybrid_is_absent_from_migrated_buckets_but_keeps_flag_membership`
(alongside `tests/test_p6_stage_c_bucket_contract.py::TestGuardRewrittenPagesUsingTheRealCorpus::test_rewritten_hybrid_page_is_absent_from_all_three_migrated_buckets`),
which proves that guard-rewritten pages (`(FAIL_CLOSED_MARKER, INVALID_TABLE_EMISSION)`)
are excluded from all three migrated buckets while genuine structure-class floors
(`(FAIL_CLOSED_MARKER, STRUCTURE_CLASS)`) remain in `structure_class_floor_pages`.

### Finding 5 (blocking) — the flagged-model event lost the candidate's defect

**Reproduced: yes.** `_kept_defect` had been switched from
`state.pages[n].best_output` to the finalized record. A kept grid carrying a live
`\multicolumn` leak is a valid authored-grid candidate that the emission guard then
replaces, so the event's `grid_defect` went from `table_latex_leak` to `""` and the
detail clause vanished. **Reverted to inspecting `best_output`.** Pinned by
`TestFinding5TheFlaggedModelEventNamesTheCandidatesDefect`, verified to fail with the
record-reading version back in.

### Finding 6 (blocking) — provisional fragment vs sidecar: PRE-EXISTING, not fixed here

**Reproduced: yes. Pre-existing: YES**, with evidence. The in-loop provisional fragment
write is byte-identical to HEAD (`git show HEAD:src/socr/pipeline/orchestrator.py`, the
`structure_class_floor_applies` / `_raw_body` block), and the mismatch reproduces
identically on both trees.

Reproducing shape, run against `git archive HEAD src` and against this tree:

```
page 1: passing best_output, audit_passed=True, text =
    "Prose above.\n\n| a | b | c |\n|---|---|\n| 1 | 2 | 3 |\n\nProse below.\n"
    (a width-mismatched GFM table: 3 header cells, 2 delimiter cells)

_flush_page_fragment(state, 1, bo.text, out) ; _flush_page_sidecar(..., terminal=False)

HEAD          fragment holds the raw invalid table; sidecar winning_output is
              status=error, failure_mode=table_emission_invalid,
              text="[page 1 failed: invalid table emission — table_width_mismatch]"
this tree     identical
```

Per the ruling this is left alone in stage A/B and written up for filing. Note the
consequence for the acceptance bar: "byte identity between the in-loop fragment flush
and `_rewrite_all_fragments`" holds for the structure-class floor (which the provisional
path finalizes explicitly) but is NOT true in general, and was not true before this
work either.

### Finding 7 (should-fix) — the difference test's normalization and event snapshot

**Reproduced: yes**, all three parts.

* `VOLATILE_KEYS` dropped the manifest entry's whole `fingerprint` object, hiding its
  `engine`, `model_version`, `image_hash`, `prompt_hash`, `render_dpi` and both version
  fields. Narrowed to the PDF hash nested inside it; the rest of the object is now
  compared.
* `_events` dropped `AuditEvent.engine`, which the acceptance bar names. Added.
* No regeneration command was checked in. Added the
  `socr-regenerate-p6-prechange` project entry point, which exports the pre-change sources with
  `git archive HEAD src` and runs the same corpus against them.
  `--print-only` reports the normalized SHA-256 and whether the checked-in file matches.

The pin was regenerated under the narrowed normalization and the byte-identity tests
still pass, so nothing was hiding behind the wider exclusion.

### Finding 8 (should-fix) — the guard could be installed yet never exercised

**Reproduced: yes** (all three old tests were structural). `tests/conftest.py` now
appends to `GUARD_CALL_LOG` on every comparison it actually performs, cleared per test.
`test_the_guard_actually_compares_on_a_real_phase_assemble` drives a real
`_phase_assemble` and asserts exactly one comparison happened over the six buckets, on a
non-empty document. `test_removing_the_guard_leaves_nothing_compared` replaces the
wrapper with the production function and asserts the log stays empty, which is what makes
the first test load-bearing rather than self-satisfying.

### Finding 9 (should-fix) — two weakened tests

**Reproduced: yes**, both.

* GH-292's AST pin scanned the whole helper for any mention of a `CORRUPT_MATH_HYBRID`
  member. Restored to an RHS-specific DATAFLOW assertion: there must be a `Compare` whose
  LEFT side is a record field (`selection_provenance`, `disposition`, `primary_reason`,
  `ending`) or a local bound to one, and whose comparator is a `CORRUPT_MATH_HYBRID`
  member. Verified against two synthetic sources: the dataflow form is accepted, a bare
  mention beside a `PageState` flag read is rejected.
* `_legacy_record` combined injected historical bytes with a disposition recomputed from
  live state, then wrote that new field into fixtures described as older-build sidecars.
  Replaced by `_flush_legacy_sidecar`, which strips the `disposition` key after the
  flush, so an older-build artefact carries no such field at all — the builds these
  tests defend against predate it. `test_a_legacy_sidecar_fixture_carries_no_disposition
  _field` pins that, with a control that an ordinary flush from the current build does
  carry it.

### Finding 10 (nit) — stale import, contradictory docs

**Reproduced: yes**, both. `_winning_page_output` removed from `_flush_page_sidecar`'s
import (and from `tests/test_package_layering.py`'s private-symbol allowlist, which the
layering test then flagged as stale). The bucket test module docstring is rewritten to
match the implementation.

### Consequence of the finding-1 revert: the new field was not resume-stable

Reverting the selector exposed what the widening had been hiding.
`test_ladder_resume.py::test_sidecar_bytes_unchanged_after_resumed_run_reflushes` failed,
and the diff was **only** the new `disposition` key — HEAD is byte-identical, and the
winning output, status, engine and every other field match. Resume collapses
`p.attempts` to the single frozen winner, so a resumed re-flush recomputed
`UNACCEPTED_OUTPUT_KEPT` where run 1 had recorded `ACCEPTED_OUTPUT`, and a run that
reprocessed nothing rewrote the sidecar.

Fixed the way the same problem was already solved for
`structure_class_model_kept_on_resume`: `PageState.resumed_disposition` carries run 1's
published disposition forward, restored by `_restore_terminal_page_state` from the
sidecar it just read, and `_select_and_finalize_page` publishes it when present. A
sidecar written before the field existed restores `None` and the disposition is computed
exactly as on a fresh run. Nothing in the resume GATE consults it and no bucket derives
from it, so it cannot change which pages are skipped or how a document is scored. The
resume test is unchanged and passes.

### Gates

```
PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-p6b/src ~/venvs/socr/bin/pytest tests -q
3368 passed, 4 xfailed, 5 warnings                exit 0

uvx ruff@0.16.0 format --check .
516 files already formatted                       exit 0
```

---

## 12. Cold review round 3

Round 2's ten findings all CLOSED; finding 6 confirmed pre-existing and filed as #539.
One new blocking finding, addressed below.

### NEW 1 (blocking) — a restored disposition outranked the shipped bytes

**Reproduced: yes.** The `resumed_disposition` override added at the end of round 2 was
applied LAST — after the saved-body replacement, the table-emission guard and the shared
marker recogniser — so it replaced a correctly computed final disposition whenever the
final transform changed the bytes.

Measured on the reviewer's shape: an ordinary accepted page whose post-transform body is
a width-mismatched GFM table (three header cells, two delimiter cells), with
`resumed_disposition = {"ending": "model_output", "primary_reason": "accepted_output"}`.

| surface | without a restored disposition | with one |
| --- | --- | --- |
| sidecar `winning_output` | `error` / `table_emission_invalid` | same |
| sidecar `winning_output.text` | the invalid-table-emission marker | same |
| final `.md` | the marker | same |
| **sidecar `disposition`** | `FAIL_CLOSED_MARKER / INVALID_TABLE_EMISSION` | **`MODEL_OUTPUT / ACCEPTED_OUTPUT`** |

Every byte surface said fail-closed while the public disposition said the page shipped an
accepted model reading — the exact misclassification class the contract exists to close,
reintroduced on a supported final-body path.

**Fix, per the ruling: the restored value is a BASE, not a verdict.** It is now applied
where `provenance_to_disposition(provenance)` was, BEFORE the byte-derived classification,
and the emission-guard branch and the shared marker recogniser run after it and win
whenever the current bytes are a marker. A restored disposition may stabilise a resume
that changed nothing; it can never outrank a guard that rewrote what the page ships now.

**Pinned by `TestRound3RestoredDispositionNeverOutranksTheShippedBytes`**, three tests
that fix the boundary from both sides, on the same two-page document with one field
varied. Page 2 is a clean page rather than decoration: with a single failing page the
document has no text, so no `.md` and no manifest are written and the assertion would
have nothing to read.

Both directions were verified against deliberately broken sources:

* restoring the old last-applied ordering →
  `test_a_guard_rewritten_body_outranks_the_restored_disposition` fails;
* dropping the restored value entirely →
  `test_an_unchanged_resume_still_keeps_its_restored_disposition` AND the pre-existing
  `test_ladder_resume.py::TestRejectedSkipsAndKeeps::test_sidecar_bytes_unchanged_after
  _resumed_run_reflushes` both fail.

So neither half of the behaviour can be lost without a test going red. The
`test_the_same_page_without_a_restored_disposition_agrees` control pins that the resumed
and non-resumed paths reach the same answer on the rewritten page.

### Gates

```
PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-p6b/src ~/venvs/socr/bin/pytest tests -q
3371 passed, 4 xfailed, 5 warnings in 219.05s     exit 0

uvx ruff@0.16.0 format --check .
516 files already formatted                       exit 0
```
