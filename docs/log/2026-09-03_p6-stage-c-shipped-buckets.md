# P6 Stage C — Shipped-Bucket Disposition Derivation

Ticket: programme item P6 of `docs/log/2026-09-01_conceptual-revision.md`. Implements Stage C of `docs/log/2026-09-02_p6-selector-collapse-design.md`, completing the migration deferred by Stage B (`docs/log/2026-09-02_p6-stage-ab-disposition-contract.md`).

---

## 0. Grounding & Baseline Reference

- **Package Entry Point Resolution**: `socr.__file__` resolves to `/Users/rubenffuertes/repos/tools/socr-p6c/src/socr/__init__.py`.
- **Stage-A/B Baseline Commit ID**: `c2bb31abc7ce5463c58a2ac5cef481567ccbd702` (`Merge pull request #553 from r-uben/feat/p1-ladder-flip-prep`).
- **Baseline Capture Fixture**: `tests/fixtures/p6/prechange_assemble.json`.
  - Prior fixture SHA-256 (7-page Stage-A/B corpus): `238181b8a6314237561571f3f5d93ef6f70a9aaffd4eb9c4911a546ea8f0d936`.
  - First Stage-C regeneration, `disposition` stripped (superseded, cold review round 1 finding 1): `547f4c06fa7325e8de8118fe509b924bb0159f82e4e37ce17b90208e407627f6`.
  - Round-1 regeneration, `disposition` captured (superseded by round 2): `14e4fbe61814432a450912498721e784d3878c28d7135cfd19b4ab9059aac89e`.
  - **Current baseline SHA-256** (12-page corpus + the `legacy_resume` shape): `b3e0a8172c64f3fb1fd259be4a47af958ee4fa7da48e05e75fd0b788004b6802`.
- **Regeneration Tooling**: `socr-regenerate-p6-prechange` (defined under `[project.scripts]` in `pyproject.toml` and implemented in `src/socr/devtools/regenerate_p6_prechange.py`). Invoked as:
  ```bash
  PYTHONPATH=src uv run socr-regenerate-p6-prechange --rev c2bb31abc7ce5463c58a2ac5cef481567ccbd702
  ```
  Print-only verification:
  ```bash
  PYTHONPATH=src uv run socr-regenerate-p6-prechange --print-only --rev c2bb31abc7ce5463c58a2ac5cef481567ccbd702
  ```

---

## 1. The Landed Stage-C Disposition Bucket Contract

Stage C migrates the three selection-shaped assemble buckets from internal `SelectionProvenance` membership to exact public `PageDisposition` pair equality:

```python
_MIGRATED_DISPOSITION_BUCKETS: dict[str, PageDisposition] = {
    "structure_class_model_pages": PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.STRUCTURE_CLASS
    ),
    "structure_class_floor_pages": PageDisposition(
        PageEnding.FAIL_CLOSED_MARKER, PagePrimaryReason.STRUCTURE_CLASS
    ),
    "corrupt_math_hybrid_pages": PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.CORRUPT_MATH_HYBRID
    ),
}
```

In `_derive_disposition_buckets(state, records)` (`src/socr/pipeline/orchestrator.py`):
1. **Three Migrated Buckets**: Derived solely by comparing `record.disposition == _MIGRATED_DISPOSITION_BUCKETS[name]`. `record.selection_provenance` is never read for membership decisions.
2. **Three Flag-Derived Buckets**: `d3_model_table_pages`, `d3_floor_pages`, and `flagged_model_pages` remain predicate-derived from `PageState`. They evaluate native-lane verdicts (the D3 flag conjunction, `flagged_model_page_output`) that a page can carry while selection terminates on an independent branch (`PASSING_BEST_OUTPUT`). Neither `SelectionProvenance` nor `PageDisposition` can express these buckets.
3. **Seven Orthogonal Assemble Groups**: Extracted into the pure helper `_derive_orthogonal_assemble_buckets(state)`: `native_only_distrust_pages`, `value_drift_pages`, `fabricated_ref_pages`, `text_grid_rejected_pages`, `chart_detection_failed_pages`, `table_rejected_pages`, and `table_unverified_pages`.
4. **Leftovers**: `failed_pages` (shipped-text marker guard) and `native_fallback_pages` (multi-exclusion diagnostic bucket) remain outside both helpers.

---

## 2. Genuine Floor vs Guard-Rewritten Dispositions ("Only Failed" Resolution)

Stage C cleanly resolves the apparent "only failed" ambiguity without altering any ending or primary reason definition:

- **Genuine Structure-Class Floor**: A page where no usable grid candidate was authored or accepted (e.g. Page 10 / `STRUCT_FLOOR_PAGE`) finalizes with `(PageEnding.FAIL_CLOSED_MARKER, PagePrimaryReason.STRUCTURE_CLASS)`. Because its primary reason is `STRUCTURE_CLASS`, it matches `_MIGRATED_DISPOSITION_BUCKETS["structure_class_floor_pages"]`. It remains in `structure_class_floor_pages` and in `failed_pages`.
- **Guard-Rewritten Pages**: A page whose candidate grid or hybrid was selected but subsequently rewritten by the post-selection emission guard to a failure marker (e.g. Page 9 / `STRUCT_MODEL_REWRITTEN_PAGE`, Page 12 / `HYBRID_REWRITTEN_PAGE`) finalizes with `(PageEnding.FAIL_CLOSED_MARKER, PagePrimaryReason.INVALID_TABLE_EMISSION)`.
  - Its selection provenance remains `STRUCTURE_CLASS_GRID_FLAGGED` or `CORRUPT_MATH_HYBRID`.
  - Its finalized public disposition has primary reason `INVALID_TABLE_EMISSION`.
  - It does **not** match `STRUCTURE_CLASS` or `CORRUPT_MATH_HYBRID`.
  - It is therefore excluded from all three migrated buckets (`structure_class_model_pages`, `structure_class_floor_pages`, `corrupt_math_hybrid_pages`).
  - It appears in `failed_pages` and emits `table_structure_failed` (site: `final_body`), but no longer emits stale shipped-kind audit events (`structure_class_model_table_kept`, `corrupt_math_hybrid_shipped`) or CLI claims.

---

## 3. Named Case Inventory (12-Page Corpus)

The test corpus in `tests/p6_corpus_fixture.py` defines 12 named page cases. The
`PageDisposition` column is now machine-checked, not prose: `EXPECTED_PAGE_DISPOSITIONS`
in `tests/p6_stage_c_oracle.py` carries the same table and is asserted against all three
persisted surfaces of BOTH captures (cold review round 1, finding 1).

| Page | Named Constant | Label | Selection Provenance | Finalized PageDisposition | Failure Marker? | Migrated Bucket Membership |
|---|---|---|---|---|---|---|
| 1 | `CLEAN_BORN_DIGITAL_PAGE` | `clean_born_digital` | `NATIVE_CLEAN` | `(NATIVE_PROSE, CLEAN_NATIVE_PROSE)` | False | (None) |
| 2 | `D3_FLOOR_PAGE` | `d3_floor` | `UNVERIFIABLE_TABLE_NATIVE` | `(FAIL_CLOSED_MARKER, NATIVE_TABLE_UNVERIFIABLE)` | True | `d3_floor_pages` (flag-derived) |
| 3 | `D3_MODEL_KEPT_PAGE` | `d3_model_kept` | `UNVERIFIABLE_TABLE_MODEL_KEPT` | `(MODEL_OUTPUT, NATIVE_TABLE_UNVERIFIABLE)` | False | `d3_model_table_pages` (flag-derived) |
| 4 | `COLD_REVIEW_SHAPE_ONE_PAGE` | `cold_review_shape_one` | `PASSING_BEST_OUTPUT` | `(MODEL_OUTPUT, ACCEPTED_OUTPUT)` | False | `d3_floor_pages` (flag-derived) |
| 5 | `COLD_REVIEW_SHAPE_TWO_PAGE` | `cold_review_shape_two` | `PASSING_BEST_OUTPUT` | `(MODEL_OUTPUT, ACCEPTED_OUTPUT)` | False | `d3_model_table_pages` (flag-derived) |
| 6 | `NO_TEXT_FAILURE_PAGE` | `no_text_failure` | `NO_TEXT_MARKER` | `(FAIL_CLOSED_MARKER, NO_USABLE_OUTPUT)` | True | (None) |
| 7 | `PASSING_MODEL_PAGE` | `passing_model` | `PASSING_BEST_OUTPUT` | `(MODEL_OUTPUT, ACCEPTED_OUTPUT)` | False | (None) |
| 8 | `STRUCT_MODEL_PASSING_PAGE` | `struct_model_passing` | `STRUCTURE_CLASS_GRID_FLAGGED` | `(MODEL_OUTPUT, STRUCTURE_CLASS)` | False | `structure_class_model_pages` |
| 9 | `STRUCT_MODEL_REWRITTEN_PAGE` | `struct_model_rewritten` | `STRUCTURE_CLASS_GRID_FLAGGED` | `(FAIL_CLOSED_MARKER, INVALID_TABLE_EMISSION)` | True | Excluded from all 3 migrated buckets |
| 10 | `STRUCT_FLOOR_PAGE` | `struct_floor` | `STRUCTURE_CLASS_FLOOR` | `(FAIL_CLOSED_MARKER, STRUCTURE_CLASS)` | True | `structure_class_floor_pages` |
| 11 | `HYBRID_CLEAN_PAGE` | `hybrid_clean` | `CORRUPT_MATH_HYBRID` | `(MODEL_OUTPUT, CORRUPT_MATH_HYBRID)` | False | `corrupt_math_hybrid_pages` |
| 12 | `HYBRID_REWRITTEN_PAGE` | `hybrid_rewritten` | `CORRUPT_MATH_HYBRID` | `(FAIL_CLOSED_MARKER, INVALID_TABLE_EMISSION)` | True | Excluded from all 3 migrated buckets |

---

## 4. Complete Measured Leaf-Path Delta (17 Leaf Paths)

The shared exact-difference oracle in `tests/p6_stage_c_oracle.py` enumerates all 17 exact leaf-path differences between the Stage-A/B baseline and Stage C. There are zero wildcard paths.

### 1. Bucket Memberships (2 leaf paths)
- `("buckets", "structure_class_model_pages")`: `[8, 9]` $\to$ `[8]`
  - Page 9 leaves `structure_class_model_pages`; passing control page 8 remains.
- `("buckets", "corrupt_math_hybrid_pages")`: `[11, 12]` $\to$ `[11]`
  - Page 12 leaves `corrupt_math_hybrid_pages`; clean control page 11 remains.

### 2. State Events (1 leaf path)
- `("events",)`: 16 events $\to$ 14 events
  - Drops `structure_class_model_table_kept` on page 9.
  - Drops `corrupt_math_hybrid_shipped` on page 12.

### 3. Page Sidecars (2 leaf paths)
- `("sidecars", "p6_corpus/pages/00009.json", "audit_events")`: 3 events $\to$ 2 events
  - Drops `structure_class_model_table_kept`; `page_failed` and `table_structure_failed` remain.
- `("sidecars", "p6_corpus/pages/00012.json", "audit_events")`: 3 events $\to$ 2 events
  - Drops `corrupt_math_hybrid_shipped`; `page_failed` and `table_structure_failed` remain.

### 4. Audit Log (4 leaf paths)
- `("audit_log", 0, "counts", "structure_class_model_table_kept")`: `2` $\to$ `1`
- `("audit_log", 0, "counts", "corrupt_math_hybrid_shipped")`: `2` $\to$ `1`
- `("audit_log", 0, "event_count")`: `16` $\to$ `14`
- `("audit_log", 0, "events")`: 16 records $\to$ 14 records (omits page 9 structure-class and page 12 corrupt-math events).

### 5. Table Trust Artifact & Direct Derivatives (4 leaf paths)
Because `tables_trust.json` aggregates table flags including `structure_class_model_table_kept`:
- `("tables_trust", 0, "counts_by_kind", "structure_class_model_table_kept")`: `2` $\to$ `1`
- `("tables_trust", 0, "table_flags_n")`: `9` $\to$ `8`
- `("tables_trust", 0, "pages", "9", "reasons")`: `["structure_class_model_table_kept", "table_structure_failed"]` $\to$ `["table_structure_failed"]`
- `("tables_trust", 0, "pages", "9", "details")`: 2 detail lines $\to$ 1 detail line (removes the `structure_class_model_table_kept` line).

### 6. CLI Output (1 leaf path)
- `("cli",)`: Console summary text delta:
  - Crop-backed equation candidates: `2 page(s)... [11, 12]` $\to$ `1 page(s)... [11]`
  - Structure-class model grid: `2 structure-class page(s)... [8, 9]` $\to$ `1 structure-class page(s)... [8]`
  - Audit log summary: `(2 corrupt_math_hybrid_shipped, ..., 2 structure_class_model_table_kept)` $\to$ `(1 corrupt_math_hybrid_shipped, ..., 1 structure_class_model_table_kept)`
  - Table trust line: `8 page(s) with untrusted tables (9 flag(s))` $\to$ `8 page(s) with untrusted tables (8 flag(s))`

### 7. Result Error Note (1 leaf path)
- `("result_error",)`:
  - Pre-change: `"page(s) 2, 6, 9, 10, 12 produced no usable output; corrupt equation candidate unverified on page(s) 11, 12; untrusted tables on 7 page(s), 7 flag(s) (see tables_trust.json); page(s) 10: structure-class ladder exhausted; fail-closed floor shipped (marker plus page image, native geometry grid withheld); invalid final table emission on page(s) 9, 12"`
  - Stage C: `"page(s) 2, 6, 9, 10, 12 produced no usable output; corrupt equation candidate unverified on page(s) 11; untrusted tables on 6 page(s), 6 flag(s) (see tables_trust.json); page(s) 10: structure-class ladder exhausted; fail-closed floor shipped (marker plus page image, native geometry grid withheld); invalid final table emission on page(s) 9, 12"`

### 8. Metadata Error Fields (2 leaf paths)
- `("metadata", "metadata.json", "files", "p6_corpus.pdf", "error")`: Matches `result_error` delta.
- `("metadata", "p6_corpus/metadata.json", "error")`: Matches `result_error` delta.

---

## 5. Unaffected Invariant Surfaces

Every other surface across the entire capture is proven byte-for-byte identical against the normalized Stage-A/B baseline:
- `doc_status`: `DocumentStatus.AUDIT_FAILED`
- `result_status`: `DocumentStatus.AUDIT_FAILED`
- `result_audit_passed`: `False`
- All 13 output markdown files (`p6_corpus.md` and `pages/00001.md` through `pages/00012.md`).
- All 12 manifest entries (`manifest.json` blobs, replay journals, winning outputs).
- All 12 page contract records (`page_contract`).
- **All 36 persisted `disposition` leaves** (12 pages x sidecar / manifest entry / page-contract
  record). Added in cold review round 1: the field is no longer normalized away, so this is a
  measured identity rather than an untested assumption.
- Sidecars for pages 1, 2, 3, 4, 5, 6, 7, 8, 10, 11.
- Flag-derived bucket memberships (`d3_model_table_pages`, `d3_floor_pages`, `flagged_model_pages`).
- The `legacy_resume` capture section in full: disposition, provenance, winning output and all six
  bucket memberships for a page resumed from a pre-Stage-A sidecar (added in cold review round 2).
- All 7 orthogonal assemble bucket groups.

---

## 6. Ruling on Selection Provenance in Audit Events

**Ruling: Selection provenance is NOT added to audit-event data.**
`SelectionProvenance` is an internal implementation detail and optional feature metadata. Adding it to `AuditEvent.data` would mutate sidecar JSON structures for unaffected pages, violating whole-sidecar byte identity across unaffected controls.

---

## 7. Autouse Guard & Suite-Wide Verification

In `tests/conftest.py`, the autouse `_p6_bucket_difference_guard` fixture wraps both `orchestrator._derive_disposition_buckets` and `orchestrator._derive_orthogonal_assemble_buckets`:
- **Disposition Contract Assertion**: `assert_stage_c_disposition_buckets(state, records, new)` verifies that for all assemble executions across the test suite, the three flag-derived buckets equal `old_disposition_buckets(state)` and the three migrated buckets equal the exact `PageDisposition` pair matches.
- **Orthogonal Equality Assertion**: `assert_orthogonal_buckets_unchanged(state, new)` verifies that all 7 orthogonal bucket groups match `old_orthogonal_assemble_buckets(state)` on every assemble run.
- **Call-Log Reachability Proofs**:
  - `DISPOSITION_GUARD_CALL_LOG` and `ORTHOGONAL_GUARD_CALL_LOG` verify that a real `_phase_assemble` records active checks.
  - Removal controls prove that bypassing either wrapper leaves the corresponding log empty.
  - Three independent negative controls prove that perturbing a flag-derived bucket, a disposition-derived bucket, or an orthogonal bucket triggers an explicit assertion failure.

---

## 8. Resume & Legacy-Sidecar Invariance

- **Decision Equality Across 11 Terminal Shapes**: Verified in `tests/test_p6_disposition_persistence.py` across all 11 terminal page shapes under providerless and active-provider settings; `_load_terminal_page` yields identical return-versus-`None` decisions regardless of whether `disposition` is present, `None`, or omitted.
- **Modern Sidecars**: `PageState.resumed_disposition` carries the public disposition forward cleanly.
- **Legacy Sidecars**: A sidecar written before Stage A carries no `disposition` key, so there is nothing to restore, and resume has collapsed `p.attempts` to the single frozen winner. Selection therefore returns `PASSING_BEST_OUTPUT` and finalization calls the page `(PageEnding.MODEL_OUTPUT, PagePrimaryReason.ACCEPTED_OUTPUT)`: an ordinary passing model page, absent from `structure_class_model_pages`. **Stage C does not change this, deliberately** (cold review round 2). The shipped bytes are the model's grid either way; only the reporting differs from what run 1 published, and changing that reporting is a separate scoped change, not this one. Pinned two-sided by `tests/test_s1_structure_class_winner_gh_reachability.py::test_a_legacy_sidecar_without_a_disposition_resumes_as_an_ordinary_passing_page` and by the `legacy_resume` section of the baseline capture.
- **Failure Marker Precedence**: `_load_terminal_page` continues to reject failure markers. On final-body transformations, the post-selection emission guard and marker recognizer run after any restored disposition, guaranteeing that a failure marker body is never published as an accepted model output.

---

## 9. Stage D Demoted-Native Exit Work

The fourth ending `PageEnding.DEMOTED_NATIVE` and its associated `native_fallback_pages` diagnostic bucket remain a measured deviation from the three-ending target.

Stage D owns the exit work for `DEMOTED_NATIVE`. This work will be driven by the separately produced per-trigger corpus measurements for:
1. `needs_ocr_enhancement`
2. `chart_asset_render_failed`
3. `text_grid_rejected`
4. Residual native-table defects

Each trigger will be evaluated independently on its measured fidelity and assigned to either `NATIVE_PROSE` (if native text is intact and trustworthy) or `FAIL_CLOSED_MARKER` (if the page cannot safely ship without OCR). No N/F assignments or empirical data outside this worktree are assumed or hardcoded in Stage C.

---

## 10. Cold Review Round 1

Cold review at `.troupe/runs/20260903-040105-standard-feature/outbox/review.md`. All three
findings reproduced; a fourth stale citation was found by generalising finding 3 into a check.

### Finding 1 (bug) — the difference oracle could not see the disposition. REPRODUCED.

`grep -c '"disposition"' tests/fixtures/p6/prechange_assemble.json` returned **0**. Both
copies of `VOLATILE_KEYS` (`tests/p6_stage_c_oracle.py`, `src/socr/devtools/regenerate_p6_prechange.py`)
listed `disposition`, and the checked-in capture was written stripped, so the load-bearing
oracle compared every surface EXCEPT the public field this stage is about. The comment
justifying the exclusion ("vary by environmental source build or PDF generation timestamps")
was false for `disposition`: it was an additive-field exclusion carried over from Stage A,
and the Stage-A/B HEAD `c2bb31a` already persisted it (`manifest.py:210`).

Changed:
- `disposition` removed from both `VOLATILE_KEYS`, with a comment saying why it must stay out.
- Fixture regenerated from `c2bb31a` WITH disposition. It now carries **36** disposition
  leaves (12 pages x sidecar / manifest entry / page-contract record).
  New SHA-256 `14e4fbe...` at the time (superseded in round 2 by
  `b3e0a8172c64f3fb1fd259be4a47af958ee4fa7da48e05e75fd0b788004b6802`); `--print-only` reports
  `MATCH`.
- `tests/p6_stage_c_oracle.py` gained `EXPECTED_PAGE_DISPOSITIONS` (the 12-page table),
  `page_dispositions_by_surface`, and `collect_disposition_leaves`.
- `tests/test_p6_stage_c_difference.py::TestTheDispositionSurfaceIsCompared` (6 tests) asserts
  the field is not volatile, the baseline actually carries it (the anti-vacuity canary), the
  pinned table holds on all three surfaces of BOTH captures, the two guard-rewritten pages'
  disposition is unchanged while their bucket membership changes, no expected difference
  touches a disposition path, and every disposition leaf is identical leaf-by-leaf.

- `tests/test_p6_stage_ab_difference.py` shares the same capture, and its module docstring
  claimed `disposition` was excluded as Stage A's additive field. With the baseline
  regenerated at the Stage-A/B HEAD the field is present on both sides, so that framing was
  no longer true and `test_disposition_is_the_only_added_key` no longer proved what its name
  said. Renamed to `test_disposition_is_present_and_no_key_moved`, with the docstring
  rewritten to the presence-and-stability pin it actually is. The assertions are unchanged.

**The measured disposition delta of Stage C is EMPTY, and that is the correct result.** The
post-selection emission guard already rewrote pages 9 and 12 to
`(FAIL_CLOSED_MARKER, INVALID_TABLE_EMISSION)` at the Stage-A/B HEAD. That is precisely WHY
the migration drops them from the three buckets: the public disposition already said what
they shipped, and only the provenance-keyed buckets disagreed. So the expected differences
for the guard-rewritten pages are asserted as exact equality of the disposition plus exact
change of the bucket, together in one test, so the two cannot drift apart. The 17 enumerated
leaf differences are unchanged, and the full-capture oracle still passes with zero residual.

### Finding 2 (suggestion, treated as REQUIRED) — the selector was widened. REPRODUCED.

`_select_page_output_tagged`'s `PASSING_BEST_OUTPUT` early return had gained a third
disjunct, `structure_class_model_kept_on_resume`. RULING: the selector's predicates are out
of scope for Stage C; "do not touch the selector's 15 returns" governs their predicates, not
merely their count.

Changed: `src/socr/core/manifest.py` is reverted to HEAD at that return.

> **Superseded by cold review round 2, finding 1.** Round 1 then RELOCATED the widened
> predicate into `_select_and_finalize_page` as a base-disposition fallback, on the reading
> that the dependent test was pinning a real Stage-C need. That reading was wrong: the test
> was pinning the relocation's own result, one-sidedly, and the fallback changed a
> non-rewritten page's public reporting relative to HEAD. The fallback is now removed and
> `src/socr/core/manifest.py` is byte-identical to HEAD. See section 11.

### Finding 3 (nit) — a cited test does not exist. REPRODUCED.

`docs/log/2026-09-02_p6-selector-collapse-design.md:272` cited
`test_a_guard_rewritten_page_is_absent_from_migrated_buckets`; the real name is
`test_a_guard_rewritten_hybrid_is_absent_from_migrated_buckets_but_keeps_flag_membership`.
The Stage-A/B log repeated it and also cited class `TestGuardRewrittenCases`, actually
`TestGuardRewrittenPagesUsingTheRealCorpus`. Both fixed, in both logs.

### Finding 4 (found while fixing 3) — a fourth stale citation.

Rather than fix three names by hand, every `tests/*.py[::name]` citation in the three P6 logs
was resolved against the source. One more was stale:
`2026-09-02_p6-stage-ab-disposition-contract.md:258` cited
`test_the_orchestrator_bucket_itself_reads_the_tag`, which Stage C's own rewrite of
`tests/test_gh292_hybrid_bucket_matches_the_tag.py` had replaced. That bullet now names the
surviving AST pin, `test_the_orchestrator_bucket_does_not_re_derive_the_page_state_flag`, and
the class that replaced the behavioural pins. All citations in all three logs now resolve.

### Review item 4 — the packaged regeneration tool

- **Entry point**: `socr-regenerate-p6-prechange` is declared under `[project.scripts]` in
  `pyproject.toml`, per the repo rule that Python entry points are never `python <file>`.
  `uv run socr-regenerate-p6-prechange --print-only --rev c2bb31a...` resolves and runs inside
  this worktree, with and without `PYTHONPATH` set, because `uv` resolves the project from the
  working directory rather than from the editable install. CI never needs it: it regenerates a
  checked-in fixture and is run by a developer when the baseline revision moves.
- **Not imported by production**: `grep -rn devtools src/socr` outside the package returns
  nothing, and the package imports only the standard library. Its one `import socr` is inside
  the runner source string executed by an isolated child interpreter against the ARCHIVED
  tree, which is the whole point of the tool.
- **Now covered by the layering test**: it was not before. `tests/test_package_layering.py`
  gained `test_no_socr_module_imports_devtools` and `test_devtools_imports_only_the_standard_library`,
  plus five evasion proofs (absolute, relative, `from .. import devtools`, whole-module, and
  `__import__`) and a control proving `devtools` may import itself. The package ships in the
  wheel, so without this rule a runtime import could quietly grow and put a
  git-archive-and-subprocess harness on a user's OCR path.
- **Deleted script**: `tests/regenerate_p6_prechange.py` has no remaining references anywhere
  in the repository.

---

## 11. Cold Review Round 2

Independent cold review; NOT MERGEABLE on one finding. Reproduced, accepted, fixed.

### Finding 1 (blocking) — the relocated fallback changed a non-rewritten page. REPRODUCED.

Round 1 removed the selector widening and moved the same effect into
`_select_and_finalize_page`: when a resumed page carried `structure_class_model_kept_on_resume`
and no readable persisted disposition, the base disposition was forced to
`(MODEL_OUTPUT, STRUCTURE_CLASS)`. The reviewer's ruling is correct, and my round-1 reasoning
was wrong in a specific way worth naming: I treated the dependent test as evidence that
Stage C needed this, when the test asserted the new result one-sidedly and never compared it
with HEAD. A test that pins only the post-change value cannot tell you whether the change was
in scope.

**Reproducer** (`git archive c2bb31a src` versus the worktree, same scenario, isolated
interpreters, `socr.__file__` asserted on each side). Resume a pre-Stage-A sidecar carrying
`structure_class_model_kept: true` and no `disposition` key:

| | archived `c2bb31a` | worktree (round 1) |
| --- | --- | --- |
| selection provenance | `passing_best_output` | `passing_best_output` |
| disposition | `(model_output, accepted_output)` | `(model_output, structure_class)` |
| `structure_class_model_pages` | `[]` | `[1]` |
| winning bytes | the model grid | the model grid |

The BYTES are identical on both sides. What moved is the page's public reporting, and with the
bucket go `pages_ok`, the `structure_class_model_table_kept` audit kind, the structure-class CLI
line, the re-flushed sidecar, the audit log, the table-trust artifact and the error notes. The
page is not guard-rewritten, so `.troupe/spec.md:34` requires it byte-identical and
`.troupe/spec.md:47` requires resume unchanged. Out of scope, twice over.

**Fixed**: the fallback is removed. `git diff HEAD -- src/socr/core/manifest.py` is now EMPTY —
Stage C's entire production change lives in `src/socr/pipeline/orchestrator.py`.

**Why the oracle was green, and how that hole is closed.** The reviewer's diagnosis is exact:
the 12-page corpus cannot reach this shape, because a live run never sets the resume flag, so
no amount of enumerated leaf paths over that corpus could see the change. Rather than fix only
the instance, the shape is now IN the capture: `tests/p6_corpus_fixture.py` gained
`legacy_resume_capture()`, and `capture()` returns it under a `legacy_resume` key. It records
the restored flag, the absent `resumed_disposition`, the disposition pair, the provenance, the
winning output, and all six bucket memberships. The ordinary difference oracle now compares it
on both sides like any other surface.

It is a real guard, not a decorative one: reinstating the fallback fails
`test_p6_stage_c_difference.py` with `Unexpected diff at
legacy_resume.buckets.structure_class_model_pages`, and fails the retargeted named test at the
disposition. Both were confirmed by reinstating it and watching them fail.

**Why the pin goes through the checked-in capture and not `git archive` at test time.**
`.github/workflows/ci.yml` uses `actions/checkout@v4` with no `fetch-depth`, so CI clones
shallow and `c2bb31a` does not exist there. A test that shelled out to `git archive` would pass
locally and fail in CI — the same trap `CLAUDE.md` documents for provider-dependent tests. The
archived run happens once, in the regeneration tool, and its result is checked in. Baseline
regenerated: SHA-256 `b3e0a8172c64f3fb1fd259be4a47af958ee4fa7da48e05e75fd0b788004b6802`.

**The dependent test is reversed, not deleted.**
`test_legacy_sidecar_with_no_disposition_key_still_reconstructs_membership` became
`test_a_legacy_sidecar_without_a_disposition_resumes_as_an_ordinary_passing_page`. It now pins
HEAD's behaviour — `(MODEL_OUTPUT, ACCEPTED_OUTPUT)`, absent from the bucket, model grid still
shipped — and closes with a two-sided comparison against the `legacy_resume` section of the
archived capture, so the named assertions and the fixture cannot drift apart.

### Scope note for whoever wants the legacy-resume reporting changed

The underlying observation stands: on a pre-Stage-A sidecar the `structure_class_model_kept`
flag IS the only surviving record of what run 1 published, and reporting that page as an
ordinary passing model page loses it. Fixing that is defensible, and it is a SEPARATE,
explicitly scoped change. It needs its own ticket, and a two-sided archived difference fixture
enumerating every resume surface it moves: disposition, bucket, audit kind, CLI line,
re-flushed sidecar, audit log, table-trust artifact, document status and error notes. It must
not ride along under the selector constraint, which is what round 1 did. Nothing here is
deferred to Stage D; Stage D's scope is unchanged (section 9).

---

## 12. Gate Verification

1. **Package Resolution**:
   ```bash
   PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-p6c/src uv run python -c "import socr; print(socr.__file__)"
   ```
   Output:
   ```
   /Users/rubenffuertes/repos/tools/socr-p6c/src/socr/__init__.py
   ```
   Verified: Resolves inside this worktree.

2. **Full Test Suite**:
   ```bash
   PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-p6c/src ~/venvs/socr/bin/pytest tests -q
   ```
   Result after cold review round 2: **3691 passed, 4 xfailed, 5 warnings in 161.02s**, exit
   code 0. (Before round 1: 3677 passed. Round 1 added 6 disposition-surface tests and 8
   devtools layering tests; round 2 reversed one existing test rather than adding any.)

   Round 2 also caught a real gate failure on the first run: the new `PipelineConfig` in
   `legacy_resume_capture` did not pin `table_judge_ladder`, and
   `tests/test_p1_golden_flag_pinned.py` blocked it. Pinned to `False` with the audit named
   in a comment. The baseline SHA-256 is unchanged by that pin, which measures the flag inert
   for this capture rather than assuming it.
   (Historical Stage-A/B baseline recorded in [`docs/log/2026-09-02_p6-stage-ab-disposition-contract.md`](file:///Users/rubenffuertes/repos/tools/socr-p6c/docs/log/2026-09-02_p6-stage-ab-disposition-contract.md)).

3. **Code Formatting**:
   ```bash
   uvx ruff@0.16.0 format --check .
   ```
   Result: **530 files already formatted** (clean exit code 0).

4. **VCS Cleanliness Check**:
   ```bash
   git diff --check
   ```
   Result: Clean (zero whitespace errors or merge conflict markers).

   `git status --short` contains only the intended orchestrator/helper, test seam and tests, expanded fixture and regenerated JSON, regeneration entry-point files, `pyproject.toml`, and the three P6 log files.

5. **Archived Baseline Print-Only Match**:
   ```bash
   PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-p6c/src uv run socr-regenerate-p6-prechange --print-only --rev c2bb31abc7ce5463c58a2ac5cef481567ccbd702
   ```
   Output:
   ```
   c2bb31abc7ce5463c58a2ac5cef481567ccbd702: sha256=b3e0a8172c64f3fb1fd259be4a47af958ee4fa7da48e05e75fd0b788004b6802
   checked in: sha256=b3e0a8172c64f3fb1fd259be4a47af958ee4fa7da48e05e75fd0b788004b6802
   MATCH
   ```
