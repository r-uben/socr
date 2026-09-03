# 2026-09-03 — P1 Preparation: Transient Table Latch, Resume Machinery, Golden Flag Audit, and Pre-Flip Invariants

Date: 2026-09-03  
Status: PREPARED (`table_judge_ladder` remains default-off; all pre-flip invariants satisfied)  
Branch: `feat/p1-ladder-flip-prep`  

---

## 1. Executive Summary & Preparation Scope

This decision log records the complete engineering preparation for Phase 1 (P1) table-judge ladder integration prior to flipping the `table_judge_ladder` default from `False` to `True`.

### Core Objective
Prepare the OCR pipeline for table-judge ladder operation without changing default runtime behavior, breaking replay/golden test baselines, or risking machine-dependent CI failures:
1. **Causal Availability Classification**: Distinguish transient external reachability failures (missing binaries, transport timeouts, network/daemon outages) from deterministic content-judgment outcomes (PASS, FAIL, schema/parse failures, unreadable witnesses).
2. **Sparse Retry Latch Architecture**: Introduce `table_judge_retry_pending` on `PageState`, persisted sparsely in per-page sidecars and rolled up atomically into `<root>/metadata.json` via `RootIndex.record` without secondary writes or schema pollution.
3. **Resume Gate Ordering & Mixed Multi-Table Support**: Ensure document and page resume gates evaluate table retry latches lazily and handle mixed multi-table pages where a `TABLE_REJECTED` reduction would otherwise mask an unresolved unavailable table under D1b.
4. **Hermetic Two-Run Verification**: Prove single-file and batch process paths in both availability directions (unavailable-then-available recovery and available-then-unavailable restore) with zero network or provider calls.
5. **Golden, Replay, and Byte-Identity Audit**: Inventory all test modules reaching table processing, audit them across a two-arm flag harness (`table_judge_ladder=False` vs `True` with empty rungs), pin all constructor sites, and enforce coverage via an AST regression guard.
6. **CLI Startup Diagnostic**: Warn users at startup when `--strict-local` and `--table-judge-ladder` are concurrently enabled.
7. **Pre-Flip Invariant Preservation**: Confirm `table_judge_ladder` remains default `False`, unset-latch default artifacts remain byte-identical, and equation lane, P6 disposition, and winner selection semantics remain untouched.

---

## 2. Causal Availability Classification & Rung Contract

### RungResult Contract Extension
`RungResult` (`src/socr/judge/table_verdict.py`) gains an explicit boolean field:
```python
@dataclass
class RungResult:
    ...
    unavailable: bool = False
    refusal: bool = False
```
The sole semantic meaning of `unavailable=True` is that a configured rung could not be attempted or communicated with due to external or transient reachability failures. It defaults to `False` for backward compatibility. Cold review round 3 added a second, strictly stronger field, `refusal: bool = False`: the service itself refused us (quota, rate limit, credentials). A refusal always implies `unavailable`; the extra bit is what lets the gate trip a per-run circuit breaker for that rung.

### Cause-Specific Classification Matrix

Rewritten in cold review round 3 (findings 2 and 3). The earlier version described the round-1 code, which classified by *where* the failure happened rather than *what caused it*, and it survived round 2 unedited while the code moved underneath it. The classification now lives in ONE shared table in `src/socr/judge/table_verdict.py`, used by both rungs, the ladder's per-rung guard and the gate's whole-ladder guard.

The rule it encodes: **an outage is an external condition that can be restored without changing this code or this call. Anything the next identical call would hit again is a defect.** Both mistakes are real and asymmetric. A false negative permanently settles a table nobody ever judged; a false positive re-judges forever, re-paying timeout x tables x rungs to reproduce a failure that is not going to change.

| Subsystem | Event / Outcome | Classification | Rationale |
| :--- | :--- | :--- | :--- |
| **CLI spawn** (`table_rung_gemini.py`) | `FileNotFoundError` / errno `ENOENT` | outage | Binary absent; installing it makes the identical call work |
| **CLI spawn** | errno `ENOEXEC` / `EACCES` / `EPERM` | outage | Not an executable image, or not runnable by us: the ENVIRONMENT is wrong |
| **CLI spawn** | errno `E2BIG`, `EPIPE`, `EINVAL`, any other | defect | Describes THIS call; an oversized argv is reproduced by every retry |
| **CLI spawn** | `OSError` with no errno | defect | Nothing identifies an external cause; do not re-run the ladder every resume for a cause we cannot name |
| **CLI** | `subprocess.TimeoutExpired` | outage | The CLI never came back |
| **CLI exit** | nonzero exit, recognised refusal PHRASE in stdout OR stderr | outage + **refusal** | Quota, rate limit, revoked credentials, service down. Whole phrases with non-word guards: a flag named `--quota-project` is not a refusal |
| **CLI exit** | nonzero exit, health handshake fails | outage | The CLI itself is not usable |
| **CLI exit** | nonzero exit, healthy CLI, no refusal signature | defect | Usage or configuration error; identical every time |
| **CLI exit** | zero exit, valid verdict | answered | Substantive content verdict |
| **CLI exit** | zero exit, malformed / schema-invalid stdout | defect (¬S1) | Answered but unusable; a retry hits the same junk |
| **HTTP** (`table_rung_ollama.py`) | 401 / 403 / 407 | outage + **refusal** | Credentials, token or proxy can be restored; until then the rung never succeeds |
| **HTTP** | 408 / 429 | outage (429 also **refusal**) | Transient by definition |
| **HTTP** | 500-599 | outage | The service is unusable. Bounded at both ends: 600+ is not a status any server issues |
| **HTTP** | 404 whose `error` field matches the daemon's missing-model shape (`model '<name>' not found`, or a pull hint) | outage | Pulling the model makes the identical call work, and the reachability probe refuses for the same reason. A route error whose path merely contains `model` is the row below |
| **HTTP** | 404 otherwise (wrong route), 400, every other 4xx | defect | Our own request is wrong |
| **HTTP transport** | `httpx.TransportError` (connect, read, write, pool, proxy) | outage | No response at all |
| **HTTP transport** | `httpx.UnsupportedProtocol` | defect | The URL we built names a scheme httpx cannot speak: our configuration |
| **HTTP transport** | `httpx.DecodingError`, `httpx.TooManyRedirects` | defect | The daemon answered; we could not handle the answer |
| **Local input** | unreadable crop, missing crop path, empty witness | defect | Deterministic local input defect |
| **Parser** (`rung_result_from_output`) | garbage, empty, or schema-invalid JSON | defect (¬S1) | Answered but unusable |
| **Ladder / gate guards** | `ConnectionError`, `TimeoutError`, `subprocess.TimeoutExpired`, `httpx.TransportError` escaping a rung | outage | Typed by `is_availability_exception` |
| **Ladder / gate guards** | `TypeError`, `AssertionError`, `KeyError`, `ValueError`, `RuntimeError`, bare `OSError` | defect | Programming errors; retrying reproduces the crash |

Every row still ends the table `TABLE_UNVERIFIED`. The classification decides only whether the outcome carries the retry latch, never whether a table is trusted, so no content can be promoted by it.

**refusal** is a strictly stronger claim than outage: the service itself said no, so the next table in the same run gets the same answer. That trips the per-run breaker described in section 4.

### Page Latch Derivation in Gate
In `_run_table_judge_gate` (`src/socr/pipeline/orchestrator.py`), `PageState.table_judge_retry_pending` is derived before mechanical binding clamps (`clamped_results`):
- The latch is set to `True` if at least one table on the page has an original `TableLadderResult` of `UNVERIFIED` containing at least one `RungResult` with `unavailable=True`, and `PageState.table_judge_retry_rungs` records the rung KINDS (`"ollama"`, `"gemini"`) those results came from (cold review round 3, finding 1).
- It is NOT set for content-only rejections (`FAIL/FAIL`), content uncertainty with reachable rungs (low/low or lone-low), schema/parse failures, binding-only clamps, or empty rung lists resulting from flag-off or `strict_local`.

### Causal Ladder Transition Outcomes

| Ladder Scenario | Verdict Sequence | Final Terminal | Latch State | Reason |
| :--- | :--- | :--- | :--- | :--- |
| **C-then-B** | Unavailable -> Definite FAIL | `TABLE_REJECTED` | `False` | Substantive content rejection recorded |
| **C-then-high-PASS** | Unavailable -> High-conf PASS | `TABLE_ACCEPTED` | `False` | Authoritative approval achieved |
| **C-then-low-PASS** | Unavailable -> Low-conf PASS | `TABLE_UNVERIFIED` | `True` | Lone low PASS: the unavailable rung was the missing corroboration |
| **B-then-C** | Definite FAIL -> Unavailable | `TABLE_UNVERIFIED` | `True` | Ladder exhausted on an unreachable rung after a FAIL |
| **A-low-then-C** | Low-conf PASS -> Unavailable | `TABLE_UNVERIFIED` | `True` | Corroborating judge unavailable |
| **C-then-C** | Unavailable -> Unavailable | `TABLE_UNVERIFIED` | `True` | No rung could be reached |
| **A-low-then-A-low** | Low-conf PASS -> Low-conf PASS | `TABLE_ACCEPTED` | `False` | Ruling 1 quorum: two witnesses in agreement (see Q1 in §9) |
| **Transport exception** | `ConnectionError` / `TimeoutError` / `httpx.TransportError` / `TimeoutExpired` from a rung | `TABLE_UNVERIFIED` | `True` | Typed as an outage (`is_availability_exception`) |
| **Software defect** | `TypeError` / `AssertionError` / `KeyError` / `ValueError` / other from a rung | `TABLE_UNVERIFIED` | `False` | Deterministic; retrying it can only reproduce the crash (round 2, finding 2) |

---

## 3. Sparse Persistence & Atomic Root-Index Architecture

### Sparse Page-Sidecar Serialization (`t4`)
Per-page sidecars (`<output>/pages/XXXXX.json`) persist the retry state under strict sparsity:
- **True latch**: `"table_judge_retry_pending": true` is written to JSON, together with `"table_judge_retry_rungs": [...]` naming the rung kinds that were unavailable.
- **False or unset latch**: The key is **omitted entirely**.
- **Invariance**: Default-off runs and clean flag-on runs produce zero extra keys, preserving exact byte-identity with pre-change artifacts and respecting `tests/test_p6_stage_ab_difference.py` P6-only additive-field constraints.
- **Sidecar Restore**: `_restore_terminal_page_state` reads `meta.get("table_judge_retry_pending", False)` and `meta.get("table_judge_retry_rungs")`, defaulting cleanly to `False` / `[]` on older sidecars while carrying both forward on unfinished pages.

### Atomic Root-Index Latch Wrapper (`t5`)
To maintain the PP-5 architectural invariant that `RootIndex.record()` is the sole writer of `<root>/metadata.json`:
1. `_EquationRetryMetadata` was generalized into `_LatchedDocMetadata`.
2. `_LatchedDocMetadata` wraps `DocMetadata`, delegating all attribute lookups via `__getattr__` and merging an explicit mapping of latch keys to VALUES into `to_entry()` (values, not just flags, so the table lane can record which rung kinds it is waiting on).
3. In `UnifiedPipeline._write_metadata()`:
   - Scans `state.pages.values()` for active latches (`equation_lane_retry_pending` and `table_judge_retry_pending`).
   - If any page carries a latch, wraps metadata once with `_LatchedDocMetadata(meta, pending)`, where `pending` carries `equation_lane_retry_pending: true`, `table_judge_retry_pending: true`, and `table_judge_retry_rungs: [...]` -- the union of the rung kinds every latched page is waiting on. Cold review round 4: UNKNOWN is the top element of that union, not the empty set. If any latched page carries no kinds (a record written before the field existed), the key is OMITTED and the gate widens to any rung, rather than narrowing the document to the kinds the other pages happened to name.
   - Calls `RootIndex(output_dir).record(rel_key, index_meta)` in a **single atomic save**.
   - No direct mutations of `RootIndex` and no secondary disk writes are introduced.

---

## 4. Rung Reachability Seam, Refusal Breaker & Resume Gate Integration

### Rung Reachability Seam (rewritten in cold review rounds 2 and 3)

The original seam answered `True` on `shutil.which` alone, falling back to a bare `/api/tags` liveness ping. Both say yes to a rung that is guaranteed to fail again, so the gate was authorizing a state transition its own evidence could not support. That description is removed rather than annotated; what follows is the current design.

Each rung KIND owns one cheap, no-model reachability function, in the module that owns that rung's transport:

- `ollama_rung_reachable(model, host)` (`table_rung_ollama.py`) -- one bounded GET of `/api/tags`, then exact normalized `name:tag` membership. Daemon up **and** the configured judge model actually pulled. The exact-tag rule is #133's: availability means the pull, not the family.
- `gemini_rung_reachable(binary)` (`table_rung_gemini.py`) -- on PATH **and** a trivial no-model health handshake (`--version`, through its own module-local subprocess seam, under its own timeout) exits zero.

`UnifiedPipeline._table_judge_rung_available_now(rung_kinds=None)`:
- Returns `False` immediately if `table_judge_ladder=False` or `strict_local=True`.
- Asks only about the rung kinds the latch recorded (cold review round 3, finding 1). A healthy rung 1 is not evidence that a rung 2 which is still down has recovered; treating it as such reopened the document and re-ran the whole ladder on every resume.
- An empty or absent kind list means the record does not say which rung failed -- an entry written before the list existed -- and the question widens to "any rung", the conservative reading for an old record.
- Probes each kind at most once per RUN, clearing the cache at every public run boundary (`process` / `process_batch`), not only at construction. `process_batch` is ONE epoch: it resets on entry and its nested `process` calls do not reset again, so the pre-gate's admission decision and the per-file resume decision agree for the whole batch.
- Each probe is best effort: an unexpected exception means that rung is not attemptable, never a broken run.

### Per-Run Refusal Breaker (cold review round 3, new finding 2)

A rung result carrying `refusal=True` -- a recognised external refusal on a REAL call -- trips the breaker for that rung kind. A quota or credential refusal is not a per-table fact: the next table in the run gets the same answer.

The breaker acts at TWO boundaries, and cold review round 4 found the second one missing:

1. **The reachability seam** reports the kind unreachable for the rest of the run, so the resume gate stops admitting documents on the strength of it.
2. **The page/rung boundary in `_run_table_judge_gate`.** The seam alone is only a resume decision; the rungs are built once per run and every later page was still handed the unchanged list, so a three-page document paid the same refused call three times. The gate now filters the rung list per table through `_live_table_judge_rungs`.

The mapping from a refusal back to the callable that produced it is positional WITHIN the list actually executed: `run_table_ladder` calls the rungs in order and appends one result per call, so `rung_results[i]` came from `rungs_used[i]`.

Cold review round 5 added the other half. Identity alone cannot reach across documents, because `_build_table_judge_rungs` rebuilds the closures for every file while `process_batch` deliberately holds one epoch for the whole batch. So the rung closures now advertise themselves (`rung_kind`, `rung_id`, `executing`), and `_table_rung_callable_refused` asks three questions in order: did THIS object refuse (identity, authoritative); have we already called this object without a refusal (identity again, which is what keeps a healthy same-kind sibling alive); is this an object we have never called whose KIND already refused us this run (the same rung rebuilt for the next document). "Per run" therefore means per public call, and a refusal in file 1 spares every later file in the batch.

When the breaker empties the ladder entirely, the terminal is NOT the empty-rung terminal. The empty-rung case (`strict_local`, configured-off) is settled by configuration and must not latch; a refused ladder is transient, so `_refused_ladder_result` synthesizes results naming the refused kinds, the page latches, and a later run retries. Sparing a page must never settle it.

The breaker is per-run state only, cleared with the reachability cache at every run boundary. It holds however the document came to be reopened, including an equation-lane-forced reopen. All of these are pinned: the call count across a three-page document, across a two-file batch, and the next batch retrying.

### Document-Level Resume Gate (`_resume_skippable`) (`t7`)
- Generalized `_resume_skippable` to accept `table_judge_retry_blocks: bool | Callable[[list[str]], bool] = False`. The callable receives the rung kinds the entry recorded.
- **Lazy Evaluation**: The reachability predicate is evaluated **only** when `entry.get("table_judge_retry_pending") is True`. Flag-off runs and unlatched completed runs incur zero network probes or startup latency.
- A matching root entry with `table_judge_retry_pending=True` refuses skip if the ladder is enabled, `strict_local` is false, and one of the rungs THAT WAS UNAVAILABLE is attemptable now.

### Batch Pre-Gate Optimization (`process_batch`) (`t7`, `t9`)
- In `UnifiedPipeline.process_batch()`, the pre-gate predicate is per-file, because each entry names the rung kinds IT is waiting on. The run-local memoizing closure that used to live there would have collapsed those different questions into one answer; the per-kind cache on the pipeline memoizes instead, so a batch still probes each rung kind at most once.
- If no files are latched, reachability is never probed.
- If multiple files are latched, each rung KIND is probed **at most once** across the entire batch (and, since cold review round 3, across the whole batch run, not just its pre-gate scan).

### Page-Level Ledger Reader & Mixed Multi-Table D1b Case (`t7`)
Under D1b semantics, pages reduced to `TABLE_REJECTED` are normally restored from disk as terminal outcomes without reprocessing.
- **The Mixed Table Conflict**: If Page X has Table 1 (REJECTED by judge) and Table 2 (UNVERIFIED due to transient unavailable rung), the page-level status reducer gives precedence to `TABLE_REJECTED`. Under unpatched D1b, subsequent runs would restore Page X from disk and permanently bypass Table 2.
- **The Resolution in `_load_terminal_page`**:
  ```python
  if meta.get("table_judge_retry_pending") is True:
      if self._table_judge_retry_blocks_resume():
          return None  # Re-read and re-judge page
      # Still unavailable: allow D1b restore and carry latch forward
  ```
- If rungs are reachable now, `_load_terminal_page` returns `None`, forcing re-processing of the page and re-judging Table 2.
- If rungs remain unavailable, D1b restore is permitted and `_restore_terminal_page_state` carries `table_judge_retry_pending=True` forward into the next snapshot.
- Content-only `TABLE_REJECTED` pages (with no latch) continue to restore via D1b regardless of reachability.
- Pre-run invalidation (`_invalidate_root_entry_for_rerun`) executes before `DocumentState` creation, preventing stale entry resurrection if a rerun fails mid-flight (mirroring PR #518).

---

## 5. Hermetic Single-File and Batch Two-Run Evidence

Hermetic integration test suites in `tests/test_p1_ladder_retry_latch.py` verified the real entry paths across both availability directions:

### Single-File Matrix (`TestSingleFileRetryLatch`)
1. **Unavailable -> Available (Recovery & Partial Page Restore)**:
   - Run 1 (`available=False`): Page 1 table passes cleanly; Page 2 table encounters unavailable rung and latches `table_judge_retry_pending=True`.
   - Run 2 (`available=True`): Document is not skipped; recovered rung is invoked only for Page 2; Page 1 restores from sidecar without re-routing (`routed_pages == [2]`).
2. **Available -> Unavailable (Clean Restore)**:
   - Run 1 (`available=True`): Clean run with no latch.
   - Run 2 (`available=False`): Document skips cleanly at root level; unavailable rung is never called (`calls == []`); markdown output bytes match byte-for-byte.
3. **Content-Only Controls**:
   - Proved that content-only `REJECTED` and content-only `UNVERIFIED` do not create latches and do not reopen settled documents when reachability changes.
4. **Mixed Multi-Table Recovery**:
   - Proved that a page with both a REJECTED table and an unavailable table latches properly and reopens when rungs recover.
5. **RootIndex Snapshot Atomicity**:
   - Confirmed every intermediate and terminal `RootIndex.save` during an unavailable run carries `table_judge_retry_pending=True`.
6. **PR #518 Stale-Entry Regression Guard**:
   - Confirmed pre-run root invalidation prevents stale entry recovery if the final index save crashes.
7. **Flag-Off Control**:
   - Proved that when `table_judge_ladder=False`, the reachability seam raises `AssertionError` if probed, no latch is serialized, and repeat runs are byte-identical.

### Batch Matrix (`TestBatchRetryLatch`)
1. **Unavailable -> Available**:
   - Batch containing `latched.pdf` and `control.pdf`. Run 2 pre-gate admits only `latched.pdf` into `to_process`; `control.pdf` is skipped without invoking rungs or altering bytes.
2. **Available -> Unavailable**:
   - Completed batch files remain skipped at pre-gate with exact byte identity.
3. **Zero-Probe Guarantee**:
   - Unlatched batch files trigger zero reachability probes.
4. **Pre-Gate Memoization Spy**:
   - Proved that a batch with multiple latched files probes each rung kind at most once during pre-gate scanning (the predicate itself is now asked per file, since each entry names the rung kinds it is waiting on).
5. **Joint Equation Suite Compatibility**:
   - Ran alongside `test_equation_lane_pipeline_p4r.py` (58 tests passed); verified no regressions in equation retry semantics.

---

## 6. Golden, Replay, and Byte-Identity Flag Audit (`t10`, `t11`, `t12`)

### Audit Rationale & Threat Model
The pipeline fingerprint binds `table_judge_ladder`. A future default flip will intentionally change the fingerprint and trigger corpus-wide reprocessing.

The catastrophic failure guarded against is **test-suite machine dependence** (the `#253`/`#257` trap):
- Any golden, byte-identity, or replay test that constructs `PipelineConfig` using default values would, after a default flip, enable the table-judge ladder in CI.
- Because CI environments lack local Ollama instances and Gemini CLI binaries, every table page in those tests would be demoted to `TABLE_UNVERIFIED`.
- Golden markdown outputs and serialized sidecars would move silently, causing test failures dependent on whether local daemon processes happen to be running on the developer's machine.

### Complete AST Inventory & Exclusion Boundary
An AST parser inspected all 97 test modules containing `PipelineConfig` constructor sites (234 total call sites). Tests were classified by behavior:
- **In-Scope**: Fixtures executing table-bearing `UnifiedPipeline.process` runs or asserting serialized table artifacts.
- **Excluded**: Pure parser tests, `BlobStore`/`Manifest` replay without pipeline runs (`test_manifest_replay.py`), direct `BaseEngine` readback (`test_canonical_readback.py`, `test_canon_round2.py`), fragment flush tests (`test_pp1_fragment_flush.py`), native prose tests (`test_pp2_agentic_fuse.py`), figure-only tests (`test_pp4_inline_figures.py`, `test_gh493_resume_figure_sidecar.py`), and direct sidecar unit tests (`test_p6_disposition_persistence.py`, `test_ladder_sidecar.py`).

### Audited Module List & AST Guard Tuple
The exact 13 module paths audited and enforced by the AST regression guard in `tests/test_p1_golden_flag_pinned.py` (`_AUDITED_GOLDEN_MODULES`). Entries 1-3 were added by cold review round 1, finding 2 (see §10):
1. `tests/p6_corpus_fixture.py`
2. `tests/test_p6_cold_review_round2.py`
3. `tests/test_p6_disposition_finalization.py`
4. `tests/test_p3_judged_bytes_ship.py`
5. `tests/test_p35_cold_review_round1.py`
6. `tests/test_p35_cold_review_round2.py`
7. `tests/test_p35_cold_review_round4.py`
8. `tests/test_p35_cold_review_round5.py`
9. `tests/test_ladder_e2e.py`
10. `tests/test_table_repair_parity.py`
11. `tests/test_gh317_structure_class_floor.py`
12. `tests/test_gh190_empty_table_surfacing.py`
13. `tests/test_gh259_flagged_model_table_wins.py`

### Detailed Fixture Audit, Constructor Sites, and Moved vs. Unaffected Results

| Module Path | Test Node ID | Constructor Site & Owner | Two-Arm Classification | Pin Applied |
| :--- | :--- | :--- | :--- | :--- |
| `tests/test_gh190_empty_table_surfacing.py` | `test_paired_pipeline_run_empty_vs_populated_differs_at_all_surfaces` | Line 428 in test body | **Moved / Fails** (Flipped default turns populated run `AUDIT_FAILED` with `table_ladder_unverified`) | `table_judge_ladder=False,  # docs/log/2026-09-03_p1-prep-latch-and-audit.md` |
| `tests/test_gh317_structure_class_floor.py` | `test_paired_process_regression` | Line 489 (`_pipeline`), Line 872 (`test_phase_agentic_loop...`), Line 1003 (`test_phase_agentic_table_recovery...`), Line 1204 (`test_paired_process_regression`) | **Moved Artifact** (Assertions pass, but flipped arm adds `table_ladder_unverified` to serialized sidecar/audit) | `table_judge_ladder=False,  # docs/log/2026-09-03_p1-prep-latch-and-audit.md` at lines 489, 872, 1003, 1204 |
| `tests/test_table_repair_parity.py` | `TestEndToEndParity::test_agentic_parity_on_ce_like_fixture` | Line 998 in test body | **Unaffected** (Passes in both arms; pinned to prevent future regression) | `table_judge_ladder=False,  # docs/log/2026-09-03_p1-prep-latch-and-audit.md` |
| `tests/test_gh259_flagged_model_table_wins.py` | `test_document_status_audit_event_and_cli_surface_the_kept_page` | Line 316 in test body | **Unaffected** (Passes in both arms) | `table_judge_ladder=False,  # docs/log/2026-09-03_p1-prep-latch-and-audit.md` |
| `tests/test_gh259_flagged_model_table_wins.py` | `test_drift_reaches_document_metadata_and_cli` | Line 752 in test body | **Unaffected** (Passes in both arms) | `table_judge_ladder=False,  # docs/log/2026-09-03_p1-prep-latch-and-audit.md` |
| `tests/test_ladder_e2e.py` | `TestRejectedTerminal::test_flag_off_vs_on` | Line 196 in `_make_config` helper | **Unaffected** (Explicit dual-arm fixture) | Refactored `_make_config(table_judge_ladder: bool = False)` with explicit named keyword |
| `tests/test_ladder_e2e.py` | `TestPlainUnverifiedTerminal::test_flag_off_vs_on` | Line 196 in `_make_config` helper | **Unaffected** (Explicit dual-arm fixture) | Forwarded explicit named keyword |
| `tests/test_ladder_e2e.py` | `TestClampedUnverifiedTerminal::test_flag_off_vs_on` | Line 196 in `_make_config` helper | **Unaffected** (Explicit dual-arm fixture) | Forwarded explicit named keyword |
| `tests/test_p3_judged_bytes_ship.py` | `TestJudgedBytes::test_accepted_text_is_captured_and_nontrivial` | Line 120 in `_run_pipeline` | **Unaffected** (Pre-pinned) | Added audit citation comment |
| `tests/test_p3_judged_bytes_ship.py` | `TestJudgedBytes::test_provisional_flush_matches_accepted_text` | Line 120 in `_run_pipeline` | **Unaffected** (Pre-pinned) | Added audit citation comment |
| `tests/test_p3_judged_bytes_ship.py` | `TestJudgedBytes::test_page_fragment_on_disk_matches_accepted_text` | Line 120 in `_run_pipeline` | **Unaffected** (Pre-pinned) | Added audit citation comment |
| `tests/test_p3_judged_bytes_ship.py` | `TestJudgedBytes::test_final_markdown_page_body_equals_accepted_text_exactly` | Line 120 in `_run_pipeline` | **Unaffected** (Pre-pinned) | Added audit citation comment |
| `tests/test_p35_cold_review_round1.py` | `TestJudgedBytesOnTheSignalPath::test_clean_path_ships_the_judge_accepted_text` | Line 124 in `_run_pipeline` | **Unaffected** (Pre-pinned) | Added audit citation comment |
| `tests/test_p35_cold_review_round1.py` | `TestJudgedBytesOnTheSignalPath::test_fired_signal_ships_the_judge_accepted_text[refused]` | Line 124 in `_run_pipeline` | **Unaffected** (Pre-pinned) | Added audit citation comment |
| `tests/test_p35_cold_review_round1.py` | `TestJudgedBytesOnTheSignalPath::test_fired_signal_ships_the_judge_accepted_text[accepted]` | Line 124 in `_run_pipeline` | **Unaffected** (Pre-pinned) | Added audit citation comment |
| `tests/test_p35_cold_review_round1.py` | `TestJudgedBytesOnTheSignalPath::test_refused_patch_keeps_the_previously_accepted_bytes_and_says_so` | Line 124 in `_run_pipeline` | **Unaffected** (Pre-pinned) | Added audit citation comment |
| `tests/test_p35_cold_review_round1.py` | `TestJudgedBytesOnTheSignalPath::test_accepted_patch_is_what_ships` | Line 124 in `_run_pipeline` | **Unaffected** (Pre-pinned) | Added audit citation comment |
| `tests/test_p35_cold_review_round2.py` | `TestAcceptedRejudgeIsPromoted::test_exhausted_ladder_recovered_by_the_crop_matches_a_direct_acceptance` | Line 115 in `_run_pipeline` | **Unaffected** (Pre-pinned) | Added audit citation comment; pinned line 278 |
| `tests/test_p35_cold_review_round2.py` | `TestRefusedRejudgeLeavesNoTrace::test_refused_rejudge_events_match_the_no_crop_run` | Line 115 in `_run_pipeline` | **Unaffected** (Pre-pinned) | Added audit citation comment |
| `tests/test_p35_cold_review_round2.py` | `TestRefusedRejudgeLeavesNoTrace::test_refused_rejudge_does_not_change_the_trust_result` | Line 115 in `_run_pipeline` | **Unaffected** (Pre-pinned) | Added audit citation comment |
| `tests/test_p35_cold_review_round4.py` | `TestPageSpendSurvivesResume::test_real_exhausted_ladder_promotion_persists_all_spend` | Uses `_run_pipeline` from round 2 (line 115) | **Unaffected** | Pinned lines 114 and 198 |
| `tests/test_p35_cold_review_round5.py` | `TestRealRecoveryPathSpendIsStable::test_two_resumes_of_the_real_exhausted_ladder_page` | Uses `_run_pipeline` from round 2 (line 115) | **Unaffected** | Pinned lines 113, 158, 186, 231 |

### Summary of Audit Counts
- **Total In-Scope Modules Audited**: 10
- **Total Test Node IDs Evaluated**: 22
- **Moved Test Nodes**: 2 (`test_paired_pipeline_run_empty_vs_populated_differs_at_all_surfaces` in GH-190; `test_paired_process_regression` in GH-317)
- **Unaffected Test Nodes**: 20
- **AST Guard Invariant**: `tests/test_p1_golden_flag_pinned.py` scans every `PipelineConfig` call in the 10 modules and fails if any constructor omits `table_judge_ladder=` or conceals it in `**kwargs`.

---

## 7. Strict-Local Startup Diagnostic (`t13`)

A shared CLI diagnostic helper `_report_strict_local_ladder_diagnostic` was added to `src/socr/cli.py`.
- **Trigger**: Emitted by `process` and `batch` commands on the `UnifiedPipeline` lane when `table_judge_ladder=True`, `strict_local=True`, and `quiet=False`.
- **Diagnostic Content**:
  1. Informs the user that both table-judge rungs (ollama-cloud and gemini CLI) require external connectivity.
  2. Notes that all table pages will be demoted to `TABLE_UNVERIFIED`, preventing table-bearing documents from completing cleanly.
  3. Notes that table-free documents can still complete cleanly.
  4. Names unsetting either flag (`--strict-local` or `--table-judge-ladder`) as the resolution.
- **Hermeticity**: Emitted purely from configuration inspect; performs zero subprocess, filesystem, or network lookups. Bypassed under `--dry-run`, `--quiet`, or the HPC lane.

---

## 8. Verification Results & Pre-Flip Invariants (`t1`, `t14`)

### Verification Commands and Resolved Environment
- **Single-Source Import Resolution**:
  ```bash
  PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-p1b/src uv run python -c "import socr; print(socr.__file__)"
  ```
  **Output**: `/Users/rubenffuertes/repos/tools/socr-p1b/src/socr/__init__.py`
- **Code Formatter**:
  ```bash
  uvx ruff@0.16.0 format --check .
  ```
  **Result**: `522 files already formatted` (clean exit code 0).
- **Targeted Test Execution**:
  ```bash
  PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-p1b/src ~/venvs/socr/bin/pytest tests/test_table_verdict.py tests/test_table_rung_gemini.py tests/test_table_rung_ollama.py tests/test_table_ladder.py tests/test_table_judge_gate.py tests/test_table_latch_sidecar.py tests/test_ladder_resume.py tests/test_p1_ladder_retry_latch.py tests/test_p1_golden_flag_pinned.py tests/test_ladder_cli_strict_local_diagnostic.py
  ```
  **Result**: 252 passed in 10.15s.
- **Full Repository Test Suite**:
  ```bash
  PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-p1b/src ~/venvs/socr/bin/pytest tests -q
  ```
  **Result**: `3478 passed, 0 failed, 4 xfailed in 124.81s`.
  *(All 4 xfails are pre-existing test markings in `test_agentic.py`, `test_binding.py`, `test_structural_gate_b1_gh151.py`, and `test_table_repair_parity.py`)*.

### Pre-Flip Architectural Invariants
1. **`table_judge_ladder` Default**: Remains `False` in `src/socr/core/config.py`.
2. **Byte Identity for Unset Latch**: Runs with `table_judge_ladder=False` produce identical page sidecars, manifest records, and markdown outputs.
3. **Distinction from Future Flip**: The future default flip will intentionally alter the run fingerprint and require corpus reprocessing. The preparation ensures that this flip does not corrupt test suites, break resume atomicity, or produce unpredictable CI behavior.
4. **Winner Selection & Audit Log**: `audit_passed` assignments across the pipeline were untouched.
5. **Ledger Authority**: `RootIndex.record()` remains the sole author of `<root>/metadata.json`.
6. **Journal Accounting**: Engine run accounting routes solely through `DocumentState.record_engine_run`.
7. **Module Boundary Preservation**: `reconstruct.py`, `native_verifier.py`, chart lane, equation lane, and P6 disposition contracts were not modified.

---

## 9. Outstanding Decisions for the Future Default Flip

This engineering preparation task does **not** flip the default and does **not** resolve outstanding product and policy rulings. The future default flip requires resolving the following open items:

1. **Owner Ruling Q1 (Two-Low Quorum)**:
   - *Question*: If Rung 1 (Ollama) and Rung 2 (Gemini) both return `PASS` with low confidence, does this combined agreement satisfy consensus and promote to `TABLE_ACCEPTED`, or does low confidence on both rungs remain `TABLE_UNVERIFIED`?
   - *Status*: Pending owner ruling on the POLICY. The earlier claim here that the code is fail-closed on this path was wrong (cold review round 2, finding 3). Current behaviour: two low-confidence PASSes **ACCEPT**, under GH-359 ruling 1's quorum ("two witnesses in agreement"), implemented at `src/socr/judge/table_ladder.py` and pinned by `tests/test_table_judge_gate.py`. A lone low PASS with no corroborating preceding PASS stays `UNVERIFIED`. The open question is whether that quorum is the policy the owner wants at flip time, not what the code does today.
2. **Owner Ruling Q2 (REJECTED Shipping vs Floor-Style Withholding)**:
   - *Question*: When a table reaches a definitive `TABLE_REJECTED` terminal, should the pipeline ship the heuristic/native text with warning metadata in the sidecar, or should it withhold/blank the table block from the markdown output?
   - *Status*: Pending owner ruling. Current code ships the best-effort heuristic text with `TABLE_REJECTED` status metadata.
3. **Owner Ruling Q3 (Fail-Closed Default Posture)**:
   - *Question*: When `table_judge_ladder=True` is the default in environments with zero configured/reachable rungs (e.g., standard offline user install), should all table-bearing documents fail-closed to `TABLE_UNVERIFIED`, or should there be an automatic fallback to heuristic acceptance?
   - *Status*: Pending owner ruling. Current code enforces strict fail-closed posture.
4. **Live Two-Rung Smoke Validation**:
   - *Requirement*: Execute an end-to-end integration test against live Ollama daemon and live Gemini CLI endpoints with real credentials and network connectivity to measure latency, quota consumption, and real prompt responses.
   - *Status*: Pending live environment execution. All tests in this task were executed strictly hermetically.

*None of the four items above was resolved by this task; all remain prerequisite gates before flipping the default.*


---

## 10. Cold Review Round 1

Cold review of the uncommitted branch: `.troupe/runs/20260903-004826-standard-feature/outbox/review.md`. Three findings, all reproduced before any fix, all fixed.

### Finding 1 (bug) — `_resume_skippable` refused the skip unconditionally

**Reproduced: yes.** Three new two-run `process()` / `process_batch()` difference tests in `tests/test_p1_ladder_retry_latch.py` failed against the reviewed tree:
- `TestSingleFileRetryLatch::test_unavailable_then_still_unavailable_skips_and_does_not_re_run_the_ladder`
- `TestSingleFileRetryLatch::test_latch_survives_a_still_unavailable_skip_and_still_reopens_on_recovery`
- `TestBatchRetryLatch::test_still_unavailable_leaves_the_latched_file_skipped_at_the_pre_gate`

A run that latched `table_judge_retry_pending` and then resumed with the rung *still* unreachable was not SKIPPED: pages were re-routed and the rung was called again. On a persistent outage that re-pays timeout × tables × rungs on every single resume, purely to rediscover that the rung is still down. It also contradicted `_resume_skippable`'s own docstring, which says the skip is refused only when a rung is reachable now, and the spec item 1, which conditions the refusal on the latch AND the ladder being enabled AND a rung being reachable NOW.

**Cause.** The PARTIAL branch of `_resume_skippable` carried two unconditional latch checks (`equation_lane_retry_pending`, then `table_judge_retry_pending`) that returned `False` regardless of reachability. Both were introduced by this branch; neither exists at the branch point. The reachability-gated checks at the top of the function are the whole rule.

**Changed.** `src/socr/pipeline/orchestrator.py` — both unconditional PARTIAL-branch latch checks removed, replaced by a comment recording why no unconditional lane-latch check belongs there. Removing the equation one restores PR #518's semantics exactly as they stand at the branch point, satisfying the "do not change the equation lane's own semantics" constraint; `tests/test_equation_lane_pipeline_p4r.py` and `tests/test_ladder_resume.py` pass unchanged.

`tests/test_table_latch_sidecar.py::TestResumeSkippableTableLatch::test_root_gate_decisions_for_latched_entry` had pinned the defect in its parametrisation: the `(PARTIAL, reachable=False)` arm expected `skippable=False`. Corrected to `True`, matching the `(COMPLETED, reachable=False)` arm, with a comment naming this review. A latch records "an outage happened", not "reprocess forever"; only reachability-now may refuse the skip.

Both directions are now pinned through the real entry paths, single-file and batch: unavailable-then-unavailable skips with zero rung calls and byte-identical output; unavailable-then-available re-judges only the pending page; and a still-unavailable skip does not discard the latch, so a later recovery still reopens the document.

### Finding 2 (bug) — the P6 corpus fixture did not pin the flag

**Reproduced: yes.** Running `tests/p6_corpus_fixture.py::capture` with `table_judge_ladder=False` and again with `True` and no reachable rung moves five captured surfaces: `result_error`, `events`, `cli`, `sidecars`, `manifest`. Pages 3, 4 and 5 gain `table_ladder_unverified` events and a `table_unverified` line in the CLI summary. Independently confirmed at suite level: with the config default temporarily flipped to `True`, all five `tests/test_p6_stage_ab_difference.py::TestPreChangeByteIdentity` nodes fail against the stored `tests/fixtures/p6/prechange_assemble.json` baseline. That is exactly the #253/#257 machine-dependence trap this audit exists to prevent.

The guard's module list was `test_*.py`-shaped in practice, so it could never have caught a shared helper that pytest does not collect.

**Changed.**
- `tests/p6_corpus_fixture.py:183` — pinned `table_judge_ladder=False` with the audit comment. Classification: **moved** (five surfaces).
- `tests/test_p6_cold_review_round2.py` (2 remaining unpinned sites) and `tests/test_p6_disposition_finalization.py` (1 site) — pinned `table_judge_ladder=False` with the audit comment. Classification: **unaffected** (both modules pass in both arms); pinned to stop a future default flip from moving them silently.
- `tests/test_p1_golden_flag_pinned.py` — `_AUDITED_GOLDEN_MODULES` widened from 10 to 13 paths with the three above, and its comment now records that the tuple is *paths*, not test-module names.

**Enumeration of non-`test_*.py` helpers.** Every `.py` file under `tests/` that is not named `test_*.py` was scanned mechanically for `PipelineConfig(` constructor sites. `tests/p6_corpus_fixture.py` is the only one. There is no `conftest.py` constructor site and no subdirectory helper. If another such helper appears, it must be audited and added to the tuple.

### Finding 3 (suggestion) — `batch --dry-run` printed the startup diagnostic

**Reproduced: yes.** New test `tests/test_ladder_cli_strict_local_diagnostic.py::TestBatchDiagnostic::test_dry_run_does_not_print_the_diagnostic` failed: `socr batch --dry-run --strict-local --table-judge-ladder` emitted the notice. `process` returns at its own dry-run gate before reaching the helper, so the two commands disagreed and §7 of this log was wrong for `batch`.

**Changed.** `src/socr/cli.py` — `_report_strict_local_ladder_diagnostic` now returns immediately when `config.dry_run` is set, so the rule holds for both commands at one place rather than depending on call-site ordering. `tests/test_gh368_dry_run_single_file.py` passes unchanged.

### Unchanged by this round

The flag default stays `False` (`src/socr/core/config.py:328`). The latch's causal classification, the sparse sidecar key, `_LatchedDocMetadata`, `RootIndex.record` sole authorship, `audit_passed`, and the equation lane's behaviour are all untouched. The four items in §9 remain open.


---

## 11. Cold Review Round 2

Independent cold review of the round-1 tree (`NOT MERGEABLE`, two blocking). All four findings reproduced before any fix.

### Finding 1 (BLOCKING) — the gate's reachability probe and the rungs' "unavailable" were different notions

**Reproduced: yes**, through the real seam. New test `tests/test_p1_ladder_retry_latch.py::TestSingleFileRetryLatch::test_a_broken_but_installed_cli_does_not_read_as_recovered` drives two `process()` runs with a rung-2 binary that is present on PATH and exits nonzero on every invocation, and rung 1 unreachable. Against the round-1 gate the document was reopened and the ladder ran again; re-injecting the old `shutil.which`-only rule as a pytest plugin reproduces that failure on demand, so the test is causal rather than merely passing.

The gate answered True on `shutil.which` alone, or on a bare `/api/tags` liveness ping. Both say yes to a rung that is guaranteed to fail again: a CLI that is installed but broken, or a healthy daemon that never pulled the judge model. The gate was therefore authorizing a state transition its own evidence could not support, and a permanent outage re-paid the whole ladder on every resume.

**Changed.** One cheap, no-model reachability function per rung kind, owned by the rung that owns the transport, and used by the gate:
- `src/socr/judge/table_rung_ollama.py` — `ollama_rung_reachable(model, host)`: daemon up **and** the configured judge model actually listed by `/api/tags`, matched on the full `name:tag` (#133's rule, so a family prefix cannot satisfy an exact tag).
- `src/socr/judge/table_rung_gemini.py` — `gemini_rung_reachable(binary)`: on PATH **and** a trivial no-model health handshake (`--version`) exits zero, through its own module-local subprocess seam so the rung tests' no-real-subprocess guard still holds.
- `src/socr/pipeline/orchestrator.py` — `_table_judge_rung_available_now` now asks those two and caches the answer for the life of the pipeline. The resume gate asks once per file and the batch pre-gate once per candidate; a subprocess health check plus an HTTP model listing must not be paid per document, and reachability cannot usefully change inside one run.

The rung side now uses the same notion where it classifies a failure: a nonzero exit from a CLI that **passes** its health handshake is no longer automatically an outage (see finding 2).

**Known residue, deliberate.** A health handshake spends no quota, which also bounds what it can see: a CLI whose credentials or quota are exhausted still prints its version. That case costs one re-judge per resume, which is exactly what the latch is for when the quota is genuinely transient. What is now closed is the CLI that will never work at all. Pinning it tighter would require a real authenticated call at the resume gate, which the "cheap, no model call" constraint rules out.

Both directions are pinned through the unpatched seam, with the binary path held constant across the two runs (it is bound into the run fingerprint, so swapping `/usr/bin/false` for `/usr/bin/true` would reprocess for that reason alone and prove nothing).

### Finding 2 (BLOCKING) — every exception latched, so the latch was not causally limited to outages

**Reproduced: yes.** The round-1 tree turned any exception escaping a rung into `unavailable=True`, and `tests/test_table_judge_gate.py` explicitly pinned that for a `RuntimeError`. A `TypeError` in our own code therefore latched, and every resume re-ran the ladder to reproduce the same crash.

**Changed.** Unavailability is now a typed classification with one reference used everywhere:
- `src/socr/judge/table_verdict.py` — `is_availability_exception(exc)`, over an explicit set: `httpx.TransportError`, `ConnectionError`, `TimeoutError`, `subprocess.TimeoutExpired`. Bare `OSError` is deliberately excluded, because a missing or unreadable local crop is a deterministic local defect and the rungs already classify it as one.
- `src/socr/judge/table_ladder.py` — the per-rung guard classifies with it and logs the traceback either way.
- `src/socr/pipeline/orchestrator.py` — the whole-ladder guard uses the same function.
- `src/socr/judge/table_rung_ollama.py` — HTTP **status** errors are split: a response means the daemon answered, so 5xx, 429, 408 and 404 (model not pulled, which is exactly what `ollama_rung_reachable` refuses) are outages, and every other 4xx, 400 above all, is our own malformed request and does not latch.
- `src/socr/judge/table_rung_gemini.py` — a nonzero exit latches when the CLI fails its health handshake or when stderr names an external refusal (quota, auth, host down); a usage or configuration error from a healthy CLI does not.

The terminal is `TABLE_UNVERIFIED` in every one of these cases. The classification decides only whether the outcome is worth retrying, never whether the table is trusted, so no content can be silently promoted by it.

Pinned by a parametrised test over both classes (`ConnectionError`, `TimeoutError`, `httpx.ConnectError` latch; `TypeError`, `AssertionError`, `KeyError`, `ValueError`, `RuntimeError` do not), plus per-rung tests for the status-code and exit-code splits that isolate one variable at a time.

### Finding 3 (SHOULD-FIX) — the log misstated two live terminals

**Reproduced: yes**, by running the ladder directly on both sequences. Two low-confidence PASSes **ACCEPT** (ruling 1's quorum), where §9 claimed the code was fail-closed and left them `UNVERIFIED`. And the `B-then-C` row described "Low PASS -> Unavailable", although B is FAIL in this vocabulary; FAIL-then-unavailable and low-PASS-then-unavailable are two different paths that both end `UNVERIFIED`.

**Changed.** The transition table in §2 now names both paths separately, adds the two-low-PASS quorum row, and splits the old "Unexpected Crash" row into the typed outage and defect rows. Q1 in §9 now states what the code does today and keeps the policy question separate, which is the distinction that matters: the owner was being asked to rule on a premise that did not hold.

### Finding 4 (NIT) — stale "not yet implemented" blocks

**Reproduced: yes**, by grep. Six modules described shipped behaviour as a future assumption expected to fail.

**Changed.** `tests/test_p1_ladder_retry_latch.py`, `tests/test_table_latch_sidecar.py`, `tests/test_table_judge_gate.py`, `tests/test_ladder_cli_strict_local_diagnostic.py`, `tests/test_table_rung_gemini.py` and `tests/test_table_rung_ollama.py` now state the contract they hold the code to. The latch module's block also records which tests patch the reachability seam and which deliberately do not, since that distinction is what finding 1 turned on.

### Unchanged by this round

The flag default stays `False`. The latch's persistence (sparse sidecar key, `_LatchedDocMetadata`, `RootIndex.record` sole authorship), `audit_passed`, the equation lane's behaviour, and the P6 disposition contract are untouched. The four items in §9 remain open, with Q1's premise corrected.


---

## 12. Cold Review Round 3

Second independent cold review (`NOT MERGEABLE`; findings 1 and 2 still open and blocking, 3 partly open, 4 closed, plus two new should-fix). The reviewer supplied five adversarial reproducers; all five failed on the reviewed tree before any change here.

### Finding 1 (BLOCKING, still open) — the gate could not tell WHICH rung recovered

**Reproduced: yes.** Rung 1 answers and stays healthy; rung 2 is unavailable and stays unavailable. Round 2's fix asked "is ANY rung reachable", so the healthy rung 1 reopened a document whose latch was entirely about rung 2, and both rungs ran again on every resume. The round-2 repair therefore closed only the special case where every other rung is also down.

**Changed.** The latch records rung IDENTITY, not a bare bit.
- `src/socr/core/state.py` — `PageState.table_judge_retry_rungs: list[str]`, the rung KINDS whose unavailability caused the terminal.
- `src/socr/judge/table_verdict.py` — `RUNG_KIND_OLLAMA` / `RUNG_KIND_GEMINI` and `rung_kind()`, so the kind is derived from `RungResult.rung` in one place rather than re-parsed at each site.
- `src/socr/pipeline/orchestrator.py` — the gate derives the kind set alongside the latch; `_LatchedDocMetadata` carries VALUES so `table_judge_retry_rungs` rides the root entry through `RootIndex.record` in the same single save; the sidecar carries it sparsely; `_resume_skippable` reads it from the entry and passes it to the predicate; `_table_judge_rung_available_now(rung_kinds)` asks only about those kinds, per kind.

An entry or sidecar written before the list existed carries no kinds. That is "unknown", not "no rungs", so the question widens to any rung, which is the conservative reading for an old record. Pinned, along with both directions of the ruling: rung 1 unavailable with rung 2 healthy stays skipped; the rung the latch names coming back re-judges.

### Finding 2 (BLOCKING, still open) — the concrete rung classification was neither shared nor causally correct

**Reproduced: yes**, on three of the reviewer's five reproducers: an `OSError(E2BIG)` from an oversized argv latched as an outage; a quota signature printed past the 500-character audit excerpt was discarded before classification and did not latch; an HTTP 401 did not latch.

**Changed.** One shared table in `src/socr/judge/table_verdict.py`, imported by both rungs and both guards, replacing the per-module rules. It is documented in full in section 2 above; the rows this round settled:

- **Spawn errors** classify by errno. `ENOENT`/`ENOEXEC`/`EACCES`/`EPERM` describe the environment and are outages; `E2BIG` and every other errno describe this call and are defects. An `OSError` with no errno at all is a defect: nothing identifies an external cause, and the terminal is `TABLE_UNVERIFIED` either way, so the conservative choice is not to re-run the ladder forever for a cause we cannot name.
- **CLI exits** classify over the FULL captured stdout AND stderr, bounded by a named constant (`CLASSIFY_CAPTURE_CHARS`) that is deliberately much larger than the audit excerpt. The refusal marker list was also narrowed to whole phrases: the loose markers (`login`, `connection`, `permission denied`) were classifying ordinary local usage and file-permission errors as outages, which is the direction that re-judges forever.
- **HTTP statuses** — 401/403/407 are outages, because credentials, a token or a proxy can be restored and until they are the rung cannot succeed. 404 is the one status that needs the body: a missing model is an outage (pulling it fixes the identical call, and the reachability probe refuses for the same reason), a wrong route is a defect. Everything else 4xx is a defect; 408, 429 and every 5xx are outages.
- **httpx exceptions** — `TransportError` is an outage except `UnsupportedProtocol`, which means the URL we built names a scheme httpx cannot speak: our own configuration, identical forever. `DecodingError` and `TooManyRedirects` are defects; the daemon answered and we could not handle the answer.

Every row is pinned as a table-driven test, and every row still ends the table `TABLE_UNVERIFIED`. Only the latch bit and the rung id differ, so no classification can promote content.

### Finding 3 (SHOULD-FIX, partly open) — two log tables still described the round-1 defects

**Reproduced: yes**, by reading. The transition table and Q1 were corrected in round 2, but the cause matrix in section 2 still said every nonzero CLI exit, every HTTP status and every unexpected exception was an outage, and section 4 still documented the removed `shutil.which` / `probe_ollama_idle` gate with config attribute names that no longer exist.

**Changed.** The cause matrix is rewritten from the shared table and now states the rule it encodes rather than listing sites. Section 4 is rewritten to the current seam and gains the breaker. Section 3's wrapper and sidecar descriptions are updated for the rung list. The obsolete descriptions were removed rather than annotated: a log that documents a fixed defect as the current design is worse than no log.

### New finding 1 (SHOULD-FIX) — the reachability cache outlived the run

**Reproduced: yes.** The cache was initialized in `__init__` and never reset, so a reused pipeline that cached "unreachable" never observed recovery.

**Changed.** `_reset_table_judge_rung_probes()` starts a fresh epoch, called at the top of `process()` and `process_batch()`. `process_batch` is ONE run: it resets on entry, sets `_in_batch_run` for the duration of its per-file loop (in a `finally`, so an exception cannot leave the flag set), and the nested `process` calls do not reset again. That keeps the pre-gate's admission decision and the per-file resume decision consistent inside one batch while still dropping the answer at the run boundary. Both halves pinned: the cache holds within a run, and a second run on the same pipeline object sees the rung come back.

The reviewer's fifth reproducer asserted this by calling the internal probe twice on one object, which no longer maps to the API; the pin here crosses real `process()` boundaries instead, which is what the ruling asked for.

### New finding 2 (SHOULD-FIX) — a recognised refusal had no cooldown

**Reproduced: yes.** A `--version` handshake stays green under quota exhaustion, so every resume reopened, and inside a batch the pre-gate could admit every latched document and every table could pay the same refused call.

**Changed.** `RungResult.refusal` marks a recognised external refusal on a real call. `_note_table_rung_refusals` (called from the gate before latch derivation, so a refusal on an ACCEPTED page still spares the rest of the run) adds that rung kind to a per-run set, and the reachability seam reports it unreachable for the remainder of the run. The page latch still persists, so a later run retries. Both halves are pinned, including that the breaker does not leak across the run boundary.

This does not claim to make the residue zero. A health handshake spends no quota and therefore cannot see one; what the breaker removes is the amplification, so a quota outage now costs one refused call per run instead of one per table per document.

### Finding 4 — CLOSED

Confirmed closed by the reviewer; no further change.

### Unchanged by this round

The flag default stays `False`. `audit_passed`, `RootIndex.record` sole authorship, `DocumentState.record_engine_run` journalling, the equation lane's behaviour and the P6 disposition contract are untouched. The four items in section 9 remain open.


---

## 13. Cold Review Round 4

Third independent cold review. Closed by it: the rung-kind latch and all its probes, the cache lifetime, the log matrices, the docstrings, the cross-call breaker reset, and the P6 default-off byte-identity pin. Three items remained; all three reproduced on the reviewed tree before any change here, on the reviewer's own probe files.

### Finding 2 (BLOCKING, still open) — two classifier rows were still false-positive outages

**Reproduced: yes**, on both of the reviewer's classifier probes.

- A 404 body of `route /api/model-info not found; use /api/chat` classified as a missing model, because the rule was a bare `"model"` substring. That is the wrong-route row: a defect, identical forever.
- A healthy CLI exiting 2 with `unknown option --quota-project; see usage` classified as an external refusal, because the phrase list still carried the bare marker `quota` under unrestricted substring matching. It latched AND tripped the run breaker on an ordinary usage error.

**Changed** (`src/socr/judge/table_verdict.py`):
- 404 is matched against the daemon's actual error shape (`model '<name>' not found`, with or without a name, or a pull hint), read from the documented `error` field when the body is the JSON ollama returns and from the raw text otherwise. Requiring whitespace after `model` is what separates it from `model-info`. Reading the field rather than the whole payload also stops an unrelated string elsewhere in the document from deciding the classification.
- Refusal markers are the whole phrases the CLIs actually emit, compiled with non-word guards on both ends, so no phrase is recognised inside a longer token. The bare `quota` marker is gone. The tier-error signature is spelled out in full for the same reason.
- The server-error row is bounded at both ends, 500 to 599. `>= 500` also swept in codes no HTTP server issues, which a broken proxy or a test double can produce.

Pinned: the reviewer's two probes, the 600 and 999 cases, three more marker-shaped flag names, and the error-field precedence.

### Item 6 (SHOULD-FIX, still open) — the breaker did not spare pages 2..N

**Reproduced: yes.** A three-page document made the same refused call three times after page 1 returned `refusal=True`. The round-3 breaker lived only in the reachability seam, which is a resume decision; the rungs are built once per run and every later page was handed the unchanged list. The round-3 test masked this by queueing one refusal per page and never asserting the call count.

**Changed** (`src/socr/pipeline/orchestrator.py`): the gate filters its rung list per table through `_live_table_judge_rungs`, re-asked for every table because a refusal on the previous one must spare this one. Refusals are recorded by callable identity as well as by kind, mapped positionally from `run_table_ladder`'s results, which is the only reliable way back when the gate receives opaque callables and only the result names a kind.

When the breaker empties the ladder entirely the terminal is `_refused_ladder_result`, NOT the empty-rung terminal: the empty-rung case is settled by configuration and must not latch, while a refused ladder is transient, so the synthesized results name the refused kinds, the page latches, and a later run retries. Sparing a page must never settle it.

Pinned: exactly one call across three pages; the spared pages still latch the refused kind and are retried by a later run; and the same holds when it is the EQUATION lane's latch that reopens the document, which is the "partly open" interaction the reviewer named.

### New finding 1 (BLOCKING) — an unknown page lost its meaning in the document union

**Reproduced: yes**, on the reviewer's probe. With one latched page carrying no kinds (a record written before the field existed, conservatively any rung) and another carrying `["gemini"]`, the document persisted `["gemini"]`. The next run stopped asking about ollama for the old page, so an ollama recovery could be missed permanently.

**Changed.** UNKNOWN is the top element of the union, not the empty set. If any latched page carries no kinds, the document entry omits the key entirely and the gate widens to any rung. Pinned both as the entry shape and as the consequence that makes it matter: such a document reopens on a rung no individual page ever named.

### Unchanged by this round

The flag default stays `False`. `audit_passed`, `RootIndex.record` sole authorship, `DocumentState.record_engine_run` journalling, the equation lane's behaviour and the P6 disposition contract are untouched. The four items in section 9 remain open.


---

## 14. Cold Review Round 5

Fourth independent cold review. Every blocker closed, every probe passing; two should-fix items remained. Both reproduced on the reviewer's own probe file before any change here, and both have the same root cause: **the rung callables carried no identity**, so the only route from a rung to "which rung is this" ran through the position of its result in a list.

### Item 2 (SHOULD-FIX, still open) — the breaker was per-file, not per-run, inside a batch

**Reproduced: yes.** A two-document `process_batch` called a fresh gemini-kind rung twice after document 1 refused. Cross-document filtering could only recognise a refused rung by callable identity, and `_build_table_judge_rungs` rebuilds the closures for every document while `process_batch` holds one epoch across them all. Page-level and equation-forced-reopen behaviour were already correct; the run-wide bound was not.

**Changed.**
- `src/socr/judge/table_rung_ollama.py`, `src/socr/judge/table_rung_gemini.py` — each builder tags its closure with `rung_kind`, `rung_id` and `executing`. The gemini rung's executing identity is the BINARY, which is what that rung actually controls: `agy` has no per-call model selector, so the model is unconfirmed and must not be claimed.
- `src/socr/pipeline/orchestrator.py` — `_table_rung_callable_refused` now asks identity first, then "have we called this object without a refusal", and only then kind. The middle question is what makes the kind rule safe: without it, a healthy same-kind sibling would be dropped along with the refuser.

The test fixture advertises its kind the same way, derived from the results it is queued with rather than passed separately, so a fixture cannot claim one kind while answering as another.

Pinned: exactly one call across a two-file batch; the next `process_batch` retries; and the healthy-sibling case, reordered, still keeps the sibling.

### New finding 1 (SHOULD-FIX) — filtering a rung corrupted the audit trail's executing identity

**Reproduced: yes.** With rung 1 already refused and rung 2 the sole live callable returning a high PASS, the audit row recorded rung 1's model as the executor. Re-injecting the old positional rule as a pytest plugin reproduces it on demand, so the pin is causal.

The old rule mapped result index 0 to `table_judge_rung1_model` and index 1 to `table_judge_rung2_binary`. That held only while every run called the configured ladder in order. Once the breaker could hand the ladder a filtered sublist it was false, and synthesized refused results — which have no executor at all and are sorted by kind — never had ladder positions either.

**Changed.** `_executing_identity` resolves from identity, never from a configured-ladder position: the tag on the callable that actually produced this result, then the rung KIND mapped to its configured identity (which is what covers a synthesized result), and only then the callable's position in the list handed to THIS gate call, which stays correct for an unfiltered ladder of rungs that advertise nothing. An unrecognised rung with no tag and no position resolves to the empty string rather than borrowing an identity it never had.

False provenance in a citation corpus's audit trail is exactly the failure the trail exists to prevent, so this is the one row where guessing is worse than saying nothing.

Pinned: a filtered ladder whose sole executor is rung 2 names rung 2's binary; a synthesized refused result names the configured identity for the kind it carries; and the existing rung-trail expectations for real and for anonymous rungs are unchanged.

### A note on one probe

The reviewer's provenance probe queues a refusal naming `gemini` and then offers a live rung that also answers as `gemini`. Under the round-5 batch rule that live rung is now correctly filtered as the same kind rebuilt, so the probe reaches the refused-ladder terminal rather than an acceptance. Run against the narrative the finding describes — rung 1 (ollama) refused, rung 2 (gemini) live — the assertion passes, and fails under the old positional rule. That narrative is what the committed pin encodes.

### Unchanged by this round

The flag default stays `False`. `audit_passed`, `RootIndex.record` sole authorship, `DocumentState.record_engine_run` journalling, the equation lane's behaviour and the P6 disposition contract are untouched. The four items in section 9 remain open.
