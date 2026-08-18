```markdown
# GH-155 ticket graph

Rules for every ticket: one PR; work in the main checkout (the editable install makes a
separate worktree test the main tree's source); branch `feat/155-<slice>`; wait for CI
green before merging. Shared done-when for every slice, in addition to the ticket's own:

- full suite green in CI (CI has no ollama: any new test driving agentic flow patches
  `_available_engines_for_agentic`)
- `uvx ruff@0.16.0 format --check .` clean (the venv ruff is older and lies about
  Markdown code blocks; use uvx exactly)
- goldens/replay byte-identical for final `.md` and fragments; sidecar comparison
  excludes `run_fingerprint`/`socr_source_digest` (source digest changes every slice
  by design)
- `grep -rn "socr.pipeline.orchestrator.<moved-global>" tests/` empty for globals moved
  in this slice; negative canary proves the new lookup is hit (patch with raising
  sentinel or assert mock called)
- stdlib-ast boundary test green; facade delegate signatures unchanged;
  package `__init__.py` files empty

---

## T0 — lazy public export + boundary gates (Quick)

- **Problem:** `pipeline/__init__.py:3` eagerly imports `UnifiedPipeline`, so importing
  any future `socr.pipeline.*` submodule loads the 6k-line orchestrator, defeating the
  issue AC "unit tests that do not import the whole orchestrator". No CI gate enforces
  the module DAG.
- **Do:** replace the eager import with a PEP 562 module `__getattr__` preserving
  `from socr.pipeline import UnifiedPipeline`. Add (a) a clean-subprocess test that
  imports `socr.pipeline` and asserts `socr.pipeline.orchestrator` not in
  `sys.modules`; (b) a stdlib-`ast` boundary test over `src/socr/pipeline/` encoding
  the forbidden-import list from README (initially trivially green); (c) a documented
  canary helper/pattern for seam-liveness assertions used by later slices.
- **Files:** `src/socr/pipeline/__init__.py`; new `tests/test_pipeline_boundaries.py`.
- **Test-seam migration:** none (no test patches `socr.pipeline.UnifiedPipeline`;
  cli.py:437,526 import orchestrator directly).
- **Done when:** subprocess test proves lazy load; boundary test wired into the suite;
  full suite green.
- **Depends on:** nothing.

## T1 — persistence/identity.py (Short)

- **Problem:** run/source identity helpers are buried in the god-module; gh214 seams
  (module setattr + cache resets) pin their lookup site.
- **Do:** move `_socr_version`, `_socr_source_digest` + `_SOURCE_DIGEST_CACHE`,
  `_manifest_versions`, `_page_blob_key`, `_resume_skippable` to
  `persistence/identity.py` (public names may drop the underscore). Facade call sites
  (`_run_fingerprint`, sidecar/manifest writers, `_resume_skip`) switch to
  module-qualified calls `identity.socr_source_digest()` etc. Never
  `from identity import x`: a bound name would make later patches of the identity
  module dead, and an alias of the cache would make resets dead.
- **Files:** new `src/socr/pipeline/persistence/{__init__.py,identity.py}`;
  `orchestrator.py`; `tests/test_resume_source_version_gh214.py` (12 seam lines:
  setattr x4, cache resets x3, direct calls x5 -> all repoint to the identity module);
  `tests/test_equation_detection.py:369` (`_manifest_versions`);
  `tests/test_pp1_fragment_flush.py:32` (top-level `_page_blob_key` import);
  `tests/test_silent_content_destruction.py:620,626` (`_resume_skippable` imports).
- **Test-seam migration:** as listed; canary: patch `identity.socr_source_digest` with
  a sentinel and assert `_run_fingerprint()` output changes.
- **Done when:** gh214 suite green against the identity module; boundary test green
  (identity imports core/contract only).
- **Depends on:** T0.

## T2 — preflight.py (Short)

- **Problem:** born-digital assessment + its audit-event emission (:699-810) are
  entangled with the facade.
- **Do:** move `_phase_analyze` body to
  `analyze_document(state, detector, config, *, console) -> DocumentAssessment`;
  facade delegate passes `self.bd_detector`, stores `self._last_assessment`.
  `_sparse_page_ok` does NOT move (quality policy, not source analysis).
- **Files:** new `src/socr/pipeline/preflight.py`; `orchestrator.py`.
- **Test-seam migration:** none moved (no string patches exist; ~135 `bd_detector`
  references and instance replacement in 4 files ride the delegate; 8 files assigning
  `_last_assessment` unaffected).
- **Done when:** test_landscape_refusal_a2_gh147, test_structural_gate_b1_gh151,
  bd_detector-replacement suites green; preflight imports core only.
- **Depends on:** nothing (T0 nominal).

## T3 — characterization C1 + C2 (Short)

- **Problem:** the sidecar-vs-fragment contract and fingerprint stability are relied on
  but not pinned; moving page I/O without pins risks silent drift.
- **Do:** C1: after a full agentic run with figures, assert `pages/NNN.json`
  `winning_output.text` equals pre-transform text, `pages/NNN.md` equals final
  per-page bytes, and resumed `PageOutput.text` equals the fragment. C2: assert
  `_run_fingerprint()` is stable across repeated calls within one run.
- **Files:** new test module (e.g. `tests/test_page_persistence_contract.py`).
- **Test-seam migration:** none.
- **Done when:** both tests green at baseline behavior.
- **Depends on:** nothing; must land before T4.

## T4 — persistence/pages.py + DocPaths (Medium)

- **Problem:** fragment/sidecar/ledger I/O and the doc-address derivation are repeated
  facade internals (`_page_fragment_path` :4355).
- **Do:** add frozen `DocPaths` + `doc_paths_for(output_dir, pdf_path, scan_root)`;
  move `flush_page_fragment`, `flush_page_sidecar(..., *, run_fingerprint,
  input_checksum, terminal)`, `load_terminal_page(..., *, run_fingerprint)`, and
  sidecar-flag readback into `persistence/pages.py` as functions taking values.
  Facade delegates keep today's exact signatures and compute `self._run_fingerprint()`
  per call (no behavior change; per-document precompute is parked P7).
  `_restore_terminal_page_state` stays on the facade in this slice. Atomic tmp-rename
  semantics preserved verbatim.
- **Files:** new `src/socr/pipeline/persistence/pages.py`; `orchestrator.py`.
- **Test-seam migration:** none: `patch.object` sidecar spies
  (test_pp2_agentic_fuse.py:341,617) and all callers hit the unchanged delegates.
  Canary: patch `pages.flush_page_sidecar` with a sentinel, drive the delegate, assert hit.
- **Done when:** test_pp5_resume_ledger, test_resume_source_version_gh214,
  test_pp1_fragment_flush, T3 tests green; pages imports core/contract/identity only.
- **Depends on:** T1, T3.

## T5 — lanes/equations.py (Quick-Short)

- **Problem:** GH-36a/b orchestration (:5697,:5779) sits in the facade.
- **Do:** move `detect_and_crop_equations` and `attach_equation_latex_sidecars`;
  facade delegates preserved (both modes call through them).
- **Files:** new `src/socr/pipeline/lanes/{__init__.py,equations.py}`; `orchestrator.py`.
- **Test-seam migration:** none (spy at test_pp2_agentic_fuse.py:720 is `patch.object`
  on the delegate).
- **Done when:** equation suites green; the `equation_region_detected` state.events
  registry contract unchanged.
- **Depends on:** nothing.

## T6 — lanes/figures.py (Medium)

- **Problem:** figure embed (:5411), chart/D3 renders (:1418,:1459,:1558), and
  image-ref sanitize live in the facade; `FigureExtractor` (29 patches) and the figure
  console prints (3 patches) pin their lookup sites.
- **Do:** move `describe_and_embed_figures` (sole `FigureExtractor` call :5461),
  `sanitize_page_image_refs`, `_get_page_context`, `_build_figure_blocks`, and the
  three render helpers. `FigureExtractor` and a lane-local `console` become
  figures.py globals. `describe_and_embed_figures` calls
  `deps._record_figure_recoverable_labels` (production dispatch :5559) and
  `deps._get_vision_engine`; both stay facade methods with delegates (direct-call and
  patch.object sites). `_try_gemini` stays nested inside the facade's
  `_get_vision_engine`. Fix the stale docstring/comment claiming in-method fragment
  updates (superseded by `_rewrite_all_fragments`) — wording only.
- **Files:** new `src/socr/pipeline/lanes/figures.py`; `orchestrator.py`;
  `tests/test_orchestrator.py` (13 FigureExtractor + 2 console at :1606,:1644);
  `tests/test_pp4_inline_figures.py` (14 FigureExtractor + 1 console at :264);
  `tests/test_agentic_figures.py` (2 FigureExtractor).
- **Test-seam migration:** 29 FigureExtractor + 3 console patches repoint to
  `socr.pipeline.lanes.figures.*`; canary on FigureExtractor (raising sentinel).
  `patch.object` seams on `_record_figure_recoverable_labels` (:3562) and
  `_get_vision_engine` unchanged.
- **Done when:** figure suites green; canary proves lane lookup; scanned-page skip via
  `deps._last_assessment` behaves identically.
- **Depends on:** nothing.

## T7 — lanes/tables.py + DEFAULT_PROVIDER_TIMEOUTS relocation (Medium)

- **Problem:** table helpers (:1805 region, :3223, :3404) live in the facade;
  `DEFAULT_PROVIDER_TIMEOUTS` (agentic.py:43) is imported by both the agentic loop
  (:2152) and the dual-pass driver (:3447), and after the split a
  `lanes -> modes` import would be the one forbidden edge.
- **Do:** move `reread_page_tables` (verbatim; docstring pins its exception boundary to
  065e5dd), `run_dual_pass` (invoked from facade `process()`; the :566 gate stays in
  `process()`), the escalation trio + `clear_fail_closed_flags`,
  `guard_table_repetition`, `repetition_truncated_note`. Move
  `DEFAULT_PROVIDER_TIMEOUTS` to `socr/core/providers.py` (provider-calibration data;
  `core.providers` already owns `ProviderProfile`/`DEFAULT_PROVIDERS`); repoint
  agentic.py, orchestrator :2152/:3447, and `tests/test_agentic.py:21`. Keep benchmark
  imports function-local exactly as today (real lazy-order cycle with `socr.benchmark`).
  Note: `_phase_dual_pass_tables` never calls `get_engine`; no get_engine seam moves here.
- **Files:** new `src/socr/pipeline/lanes/tables.py`; `src/socr/core/providers.py`;
  `src/socr/pipeline/agentic.py`; `orchestrator.py`; `tests/test_agentic.py`.
- **Test-seam migration:** `tests/test_agentic.py:21` import repoints. Escalation tests
  constructed via `object.__new__` keep working (stateless functions, lazy deps).
- **Done when:** test_dual_pass_tables, test_gh95_tables_trust,
  test_gh96_escalation_lane, test_table_repair_parity green; boundary test confirms no
  `lanes -> modes` import.
- **Depends on:** nothing.

## T8 — modes/agentic/routing.py + judging.py (Medium, single PR)

- **Problem:** `agentic.py` mixes the route ladder (lines <= 315) with judge classes
  (317+); judge construction (`_build_page_judge` :3517, `_make_page_renderer` :3643,
  `_TimeoutJudge` :2072) is facade-embedded. Splitting routing and judging across two
  PRs would repoint `patch("socr.pipeline.agentic.VLMPageJudge")` twice.
- **Do:** in one PR: git mv `agentic.py` -> `modes/agentic/routing.py`; extract the four
  judge classes into `modes/agentic/judging.py`; move `build_page_judge` (with its
  nested `get_fitz_page`/`is_table_page`/`record_event` closures), `TimeoutJudge`,
  `make_page_renderer`, `JUDGE_IDENTITY_HEURISTIC` (:58), and
  `JUDGE_MODEL_CANDIDATES` (class attr :3078 -> judging module constant) out of the
  facade. Facade keeps: `_resolve_judge_model` (+ `_judge_model_cache`), delegates for
  `_build_page_judge` and `_make_page_renderer`, class-attr aliases
  `_TimeoutJudge = judging.TimeoutJudge` and
  `_JUDGE_MODEL_CANDIDATES = judging.JUDGE_MODEL_CANDIDATES`. `build_page_judge` must
  call `deps._make_page_renderer(state)` (never the judging-local function) — two
  patch sites depend on it. `JudgeDeps`: `config`, `heuristics`, `_sparse_page_ok`,
  `_resolve_judge_model`, `_page_has_tables`, `_make_page_renderer`.
- **Files:** `git mv src/socr/pipeline/agentic.py src/socr/pipeline/modes/agentic/routing.py`;
  new `modes/{__init__.py}`, `modes/agentic/{__init__.py,judging.py}`; `orchestrator.py`
  (imports at :45,:2091,:3135,:3526); 12 test files with 23
  `from socr.pipeline.agentic import` lines; `tests/test_silent_content_destruction.py:336`.
- **Test-seam migration:** 23 import lines repoint (`routing` for
  route_page/types/`_best_effort`/`_error_output`; `judging` for judge classes);
  `patch("socr.pipeline.agentic.VLMPageJudge")` -> `socr.pipeline.modes.agentic.judging.VLMPageJudge`
  with a canary (sentinel class, assert instantiated); `_make_page_renderer` patch
  sites (test_judge_hard_pages.py:69, test_silent_content_destruction.py:338) unchanged
  but canaried (patch and assert the built judge uses the patched renderer).
- **Done when:** test_judge_wiring_gh133, judge portion of test_p1_cascade_economics
  (bound-method identity — survives because `_sparse_page_ok` never moved),
  test_source_evidence_table_judge, test_silent_content_destruction, pp2 `_TimeoutJudge`
  constructions green; `socr.pipeline.agentic` no longer exists;
  `grep -rn "socr.pipeline.agentic" src tests` empty.
- **Depends on:** T7 (constant already out of agentic.py).

## T9 — modes/legacy/quality.py (Short)

- **Problem:** scoring phases and hard-page judging are facade-embedded.
- **Do:** move `_phase_score` (+ `_score_whole_doc`, `_score_per_page` with the
  B1/GH-200/TR-3 fail-closed short-circuit), `_phase_score_multi`,
  `_score_repair_result`, `_native_table_structure_gate_applies`,
  `_phase_judge_hard_pages` (:3135 region). VLM judge lookup is call-time
  module-qualified `judging.VLMPageJudge` (the one legal cross-mode edge); never a
  top-level `from ... import VLMPageJudge`.
- **Files:** new `src/socr/pipeline/modes/legacy/{__init__.py,quality.py}`;
  `orchestrator.py`.
- **Test-seam migration:** none expected (score tests drive facade delegates); U3
  census: check capsys-based console assertions against moved prints.
- **Done when:** scoring + judge-hard-pages suites green; boundary test confirms the
  only `legacy -> agentic` edge is quality -> judging.
- **Depends on:** T8.

## T10 — modes/legacy/workflow.py (Medium; may split backbone vs repair+consensus)

- **Problem:** legacy phases (:816,:887,:3728, repair/consensus callers) are
  facade-embedded; their `get_engine` reads pin the lookup site.
- **Do:** move `run_backbone`, `run_native_first` (with the two
  `deps._run_engine_on_pages` calls :1102/:1160, `_sparse_page_ok`-gated local-tier
  scoring :1132, corrupt-math recovery :915-1033 with its function-local
  `socr.math.recover` import, equation attach :1079-1095 via deps),
  `run_multi_engine` (:3728 region incl. the :3755 `get_engine` loop), `run_repair`
  (via `deps.repair_router`), `run_consensus`. `get_engine` becomes this module's
  global for reads at :835,:1112,:3755,:4101,:4159,:4224.
- **Files:** new `src/socr/pipeline/modes/legacy/workflow.py`; `orchestrator.py`;
  `tests/test_orchestrator.py` (subset of 49 `get_engine` patches).
- **Test-seam migration:** repoint ONLY the test_orchestrator `get_engine` patches whose
  production reader moved (backbone/native-first/multi-engine/repair bodies), using the
  empirical triage protocol (run unrepointed -> failures are the repoint list -> then
  prove each still-passing patch non-vacuous by making its engine raise). The five
  test_b2_routing patches (:269-:319) stay on `socr.pipeline.orchestrator.get_engine`:
  all wrap `_available_engines_for_agentic`, which never moves. pp2's 7 patches also
  stay (they intercept `_run_engine_on_pages`/`_available_engines_for_agentic`,
  both facade-resident).
- **Done when:** legacy-path suites green; canary: patch
  `modes.legacy.workflow.get_engine` with a raiser, drive backbone, assert hit.
- **Depends on:** T4, T5.

## T11 — characterization C4: resume cost fold (Short)

- **Problem:** no test proves resumed per-page `cost_usd` folds into
  `state.engine_runs`/budget gating: the two direct-call tests of
  `_restore_terminal_page_state` (test_structural_gate_b1_gh151.py:262,
  test_tr3_d3_floor.py:712) check flag restore only, and test_p1_cascade_economics
  never calls it.
- **Do:** add a characterization: terminal page carrying nonzero cost + a later page
  under a finite document budget; assert restored cost contributes to
  `state.total_cost` and prevents a later paid rung from overspending.
- **Files:** new test (or extend `tests/test_p1_cascade_economics.py`).
- **Test-seam migration:** none.
- **Done when:** green at baseline behavior.
- **Depends on:** nothing; must land before T12.

## T12 — modes/agentic/workflow.py (Large; atomic single PR)

- **Problem:** the fused page-major loop (:2116-3029) is the widest-blast move; 16
  `route_page` + 6 `probe_ollama_idle` patches and a source-inspection test pin it.
- **Do:** move the loop verbatim to `run_agentic_document(state, output_dir, deps)`
  (including the guarded no-op prune :2257-2283 — it is a self-enforcing invariant, not
  dead code) and `restore_terminal_page_state` (facade delegate KEPT: two tests call it
  directly). `route_page`/`probe_ollama_idle` become workflow globals.
  `provider_ladder` stays a function-local import (patched at source,
  test_b2_routing.py:118). `state._pp2_halt_reason` write (:3029) and cascade-halt
  latch (:2445,:2854-2878) move untouched. `AgenticDeps` carries every facade
  collaborator (`_run_engine_on_pages`, `_available_engines_for_agentic`, lane
  delegates, persistence delegates, `_build_page_judge`, predicates, `config`,
  `console` if witnessed).
- **Files:** new `src/socr/pipeline/modes/agentic/workflow.py`; `orchestrator.py`;
  test files listed below.
- **Test-seam migration (same commit):** 16 `route_page` patches (test_chart_lane 6,
  test_pp2_agentic_fuse 4, test_pp5_resume_ledger 3, test_landscape_refusal_a2_gh147 1,
  test_structural_gate_b1_gh151 1, test_table_repair_parity 1) and 6
  `probe_ollama_idle` patches (4 files) repoint to
  `socr.pipeline.modes.agentic.workflow.*` with canaries. Migrate the
  source-inspection test (test_p1_cascade_economics.py:256-263) to
  `inspect.getsource` of the moved function: on a delegate body,
  `"max_attempts" not in src` passes vacuously while
  `"remaining_budget=remaining" in src` fails red — either way it must move with the
  loop. pp2's 7 `get_engine` patches stay on the orchestrator (facade-resident readers).
- **Done when:** pp2/pp5/chart-lane/landscape/structural-gate/repair-parity suites and
  T11 green; hermeticity rule observed by any new test.
- **Depends on:** T4, T5, T6, T7, T8, T11.

## T13 — characterization C3: assembly order (Short)

- **Problem:** the error-chain append order and provisional->final metadata replacement
  are unpinned before the assembly move.
- **Do:** pin `PARTIAL_SAVE_VLM_TIMEOUT; <repetition note>; <trust note>` append order
  with all three present, and the `:pre-figures` metadata suffix disappearing after
  figures.
- **Files:** new test.
- **Done when:** green at baseline behavior. **Depends on:** nothing; before T14.

## T14 — assembly.py (Large; atomic single PR)

- **Problem:** finalization order (:4871-5270) and writers are facade-embedded; seven
  writer spies + two figure spies pin instance dispatch during `_phase_assemble`.
- **Do:** move `_phase_assemble` body to `run_assemble(state, output_dir, deps)` plus
  the writer bodies (`_write_metadata`; `_write_manifest` + `_fingerprint_inputs`
  keeping its function-local `socr.engines.registry.get_engine` import — it has never
  read the orchestrator global, so no patch seam is affected; `_write_audit_log`, which
  calls `_write_tables_trust` :5365 inside the never-lose-output try/except;
  `_tables_trust_note` (read :5196); `_save_markdown`; `_stitch_fragments`;
  `_canonical_body`; `_rewrite_all_fragments`). `run_assemble` dispatches through
  `AssembleDeps` for every spied operation: `_save_markdown`, `_write_metadata`,
  `_write_manifest`, `_rewrite_all_fragments`, `_flush_page_fragment`,
  `_flush_page_sidecar`, `_stitch_fragments`, `_describe_and_embed_figures`. Assembly
  imports `persistence/{pages,identity}` but NOT `lanes/figures` (figures go through
  deps). Order moved verbatim: terminal sidecars mid-assemble before figures
  (:4904-4906); manifest default-on in agentic (:5264); audit log last.
- **Files:** new `src/socr/pipeline/assembly.py`; `orchestrator.py`.
- **Test-seam migration:** none repointed (all assembly seams are `patch.object` on the
  pipeline instance; test_tr3_d3_floor.py:322-328,407-413,
  test_silent_content_destruction.py:700, test_pp5_resume_ledger.py:860). Canary: with
  `patch.object(pipeline, "_save_markdown")` raising, `run_assemble` must raise.
- **Done when:** golden byte-identity, test_canonical_readback, test_manifest_replay,
  T13, tr3 suites green.
- **Depends on:** T4, T6, T13.

## T15 — facade trim + docs + acceptance (Quick)

- **Problem:** orphaned code and stale docs after the moves; issue ACs need closing.
- **Do:** delete dead facade code; rewrite `docs/ARCHITECTURE.md` pipeline section
  (currently :84-91 names `orchestrator.py`/`agentic.py`) with the canonical-path
  (agentic) vs legacy section per the issue AC and the new module map; measure facade
  LOC: hard gate < 1.5k (estimate ~950), report the < 800 ideal; raise the D4 proposal
  to the owner.
- **Files:** `orchestrator.py`; `docs/ARCHITECTURE.md`.
- **Done when:** all issue ACs checked off or explicitly owner-waived; boundary test
  final; full suite green.
- **Depends on:** all previous.

---

## Separate defect tickets (outside the slice sequence)

File as individual issues; each fingerprint ticket forces corpus reprocessing — never
batch with structural slices, or the "bytes unchanged" evidence is destroyed.

- **D1a** fingerprint `auto_patch_tables` (:3290)
- **D1b** fingerprint `clean_equation_model` (:5829) + declare-or-delete
  `math_model_host` (:5854-5855; not a `PipelineConfig` field)
- **D1c** Qwen resolver: `resolved_model_version`/`fingerprint_determinants` via
  `resolve_qwen_intent` (engines/qwen.py:73-112; base.py:76-104 gap on
  vllm/sglang/api backends)
- **D1d** fingerprint `gemini_model` under custom `enabled_engines` excluding GEMINI
  (:5929-5930; defaults are covered via core/config.py:107)
- **D1e** fingerprint `recover_corrupt_math` + `math_model` (:915,:1017,:1027-1031)
- **D2** declare `DocumentState._pp2_halt_reason` (:3029 write, :5172 read)
- **D5** `_page_blob_key` canonicalization mismatch vs `core.cache.blob_hash`
  (:60-71 vs core/cache.py:24-37) + non-ASCII regression test

## Sequencing

- T0 -> T1; T2, T3, T5, T6, T7 are mutually independent and can interleave with
  ongoing correctness work.
- T8 after T7; T9 after T8; T10 after T4+T5; T11 anytime before T12.
- T12 and T14 are the wide-blast moves: schedule in a quiet window with a freeze on
  other orchestrator-touching PRs; if a correctness fix must touch `_phase_agentic`
  mid-sequence, land it before T12 or rebase T12 on it, never in parallel.
- One branch at a time in the main checkout (editable-install worktree trap).
```
