```markdown
# GH-155: split pipeline/orchestrator.py into owned modules

Design baseline: commit `0c5bbd11007d1aa038f040242c006a64d1a4483c`. Every file:line
citation in this packet refers to that commit. Verify the working tree still matches it
for `src/socr` and `tests/` before starting any slice (see STATUS.md).

## Rationale

`src/socr/pipeline/orchestrator.py` is 6,050 lines (`agentic.py` adds 728). It owns
legacy tiers, the agentic page-major loop, table/figure/equation orchestration, resume
identity, page persistence, and assembly. Issue #155 needs ownership boundaries for the
structure-lane roadmap and the #142 flag audit.

The governing constraint is the test suite's interface with this module. Verified seam
census at baseline:

- 61 string patches of `socr.pipeline.orchestrator.get_engine`
  (test_orchestrator 49, test_b2_routing 5, test_pp2_agentic_fuse 7)
- 29 of `orchestrator.FigureExtractor` (test_orchestrator 13, test_pp4_inline_figures 14,
  test_agentic_figures 2)
- 16 of `orchestrator.route_page` (test_chart_lane 6, test_pp2_agentic_fuse 4,
  test_pp5_resume_ledger 3, test_landscape_refusal_a2_gh147 / test_structural_gate_b1_gh151 /
  test_table_repair_parity 1 each)
- 6 of `orchestrator.probe_ollama_idle` (4 files); 3 of `orchestrator.console`
  (test_orchestrator.py:1606,1644; test_pp4_inline_figures.py:264)
- gh214: `monkeypatch.setattr(orch, "_socr_source_digest", ...)` x4, direct
  `orch._SOURCE_DIGEST_CACHE = None` x3, direct calls x5
  (tests/test_resume_source_version_gh214.py:27-175);
  1 patch of `orchestrator._manifest_versions` (tests/test_equation_detection.py:369)
- top-level import of `_page_blob_key` (tests/test_pp1_fragment_flush.py:32); in-test
  imports of `_resume_skippable` (tests/test_silent_content_destruction.py:620,626)
- 23 `from socr.pipeline.agentic import` lines across 12 test files, plus
  `patch("socr.pipeline.agentic.VLMPageJudge")` (tests/test_silent_content_destruction.py:336)
- instance seams: `object.__new__(UnifiedPipeline)` in 3 files (test_gemini_api,
  test_gh95_tables_trust, test_gh96_escalation_lane); `_last_assessment` assigned in
  8 files; `patch.object` spies on `_flush_page_sidecar` (test_pp2_agentic_fuse.py:341,617),
  `_detect_and_crop_equations` (:720), `_make_page_renderer`
  (test_judge_hard_pages.py:69, test_silent_content_destruction.py:338),
  `_record_figure_recoverable_labels` (test_orchestrator.py:3562),
  `_describe_and_embed_figures` (test_silent_content_destruction.py:700,
  test_pp5_resume_ledger.py:860), and the seven assembly writers
  (test_tr3_d3_floor.py:322-328, 407-413)
- direct calls: `pipeline._restore_terminal_page_state`
  (test_structural_gate_b1_gh151.py:262, test_tr3_d3_floor.py:712);
  `pipeline._TimeoutJudge(...)` constructions (test_pp2_agentic_fuse.py, 3 sites);
  bound-method identity `inner._sparse_ok == pipeline._sparse_page_ok`
  (test_p1_cascade_economics.py:252); source inspection of
  `UnifiedPipeline._phase_agentic` (test_p1_cascade_economics.py:256-263)

A moved global or a facade wrapper that production bypasses makes these patches silently
dead (the MagicMock-vacuous failure mode). Every slice therefore moves a symbol together
with its last production reader and repoints the tests in the same commit, with a
liveness canary (see Dependency and seam rules).

## Final tree

```
src/socr/pipeline/
├── __init__.py                 # lazy PEP 562 export of UnifiedPipeline
├── orchestrator.py             # facade + composition root (residual ~950 LOC; hard AC < 1.5k)
├── preflight.py                # analyze_document(...)
├── assembly.py                 # finalization order + writers
├── consensus.py  repair.py  reconciler.py  hpc_pipeline.py   # unchanged siblings
├── modes/
│   ├── __init__.py             # empty (no barrel imports)
│   ├── agentic/
│   │   ├── __init__.py         # empty
│   │   ├── routing.py          # git mv of agentic.py minus judge classes
│   │   ├── judging.py          # judge classes + build_page_judge + TimeoutJudge + renderer
│   │   └── workflow.py         # page-major fused loop (_phase_agentic)
│   └── legacy/
│       ├── __init__.py         # empty
│       ├── workflow.py         # backbone, native-first, multi-engine, repair, consensus
│       └── quality.py          # score phases, judge_hard_pages, repair re-score, table gate
├── lanes/
│   ├── __init__.py             # empty
│   ├── tables.py               # reread, dual-pass driver, escalation trio, repetition guard
│   ├── figures.py              # figure embed, chart/D3 renders, image-ref sanitize
│   └── equations.py            # GH-36a detect+crop, GH-36b latex sidecars
└── persistence/
    ├── __init__.py             # empty
    ├── identity.py             # source digest+cache, versions, blob key, resume_skippable
    └── pages.py                # DocPaths + atomic fragment/sidecar I/O + terminal load gate
```

Deviations from the pre-review target tree (both need owner sign-off, recorded in
STATUS.md):

1. `modes/legacy/extraction.py` is dropped. Its only candidate tenant,
   `_run_engine_on_pages` (orchestrator.py:1198), is cross-mode: legacy native-first
   calls it at 1102/1160 and the agentic `run_provider` closure at 2355. Housing it in
   `legacy/` would force an `agentic -> legacy` import. It stays a facade method.
2. `consensus.py`, `repair.py`, `reconciler.py`, `hpc_pipeline.py` are named as
   unchanged siblings (the target tree silently omitted them; hpc_pipeline.py:35
   imports reconciler, cli imports hpc_pipeline, tests import `socr.pipeline.repair`).

`socr.pipeline.agentic` ceases to exist when routing/judging land. There is no
compatibility shim: a re-export module would keep
`patch("socr.pipeline.agentic.VLMPageJudge")` importable while production reads the new
home, turning the patch silently dead. No non-test consumer exists in `src/` outside
orchestrator.py (verified); any out-of-repo import must repoint at that release.

## The deps-protocol pattern (used by every extracted module)

Extracted functions take `(state, ..., deps)` where `deps` is a structural
`typing.Protocol` defined in the consuming module and satisfied by `UnifiedPipeline`
without inheritance; production passes `self`. Members are named exactly after the
facade methods/attrs they carry. This preserves:

- `patch.object(pipeline, ...)` instance seams (the deps object is the pipeline)
- bound-method identity (`inner._sparse_ok == pipeline._sparse_page_ok`)
- `object.__new__(UnifiedPipeline)` construction (nothing captured eagerly; adapters lazy)
- unit tests that satisfy the Protocol with small fakes and never import the
  orchestrator (issue AC), enabled by the lazy `__init__` export

The facade keeps one-line delegates with frozen current signatures for every moved
method that tests call or patch. Module globals (`get_engine`, `route_page`,
`probe_ollama_idle`, `FigureExtractor`, `console`) move only in the slice that moves
their last production reader, with tests repointed in the same commit.

## Module responsibilities and APIs

| Module | Public entry | Owns | Notes |
|---|---|---|---|
| `orchestrator.py` | `UnifiedPipeline.process`, `process_batch` | composition root; dual-pass gate (orchestrator.py:566: dual-pass runs in agentic multi mode, so the driver is invoked from `process()`, not from either mode); `_resume_skip`; `_resolve_output_root`; `_print_summary`; shared predicates (`_assessment_for_page`, `_page_has_tables`, `_is_native_eligible_without_ocr`, `_is_trusted_native_without_ocr`, `_is_agentic_trusted_native`, `_is_chart_asset_page`, `_sparse_page_ok`); `_run_engine_on_pages`; resolvers (`_resolve_primary_engine`, `_resolve_judge_model` + `_judge_model_cache`, `_resolve_crop_vlm_model`, `_resolve_table_escalation_provider`, `_available_engines_for_agentic`); `_engine_determinants`; `_run_fingerprint`; `_get_vision_engine` (with nested `_try_gemini`); attributes `_last_assessment`, `bd_detector`, `_scan_root`, `repair_router`; delegates | `_available_engines_for_agentic` reads module-global `get_engine` (:3064): both stay put forever (CI hermeticity seam, CLAUDE.md). `_engine_determinants` keeps its deliberate function-local `get_engine` import (:302-305); it has never been patch-interceptable, do not change that. Class attr alias `_TimeoutJudge = judging.TimeoutJudge` after the judging slice. |
| `pipeline/__init__.py` | `from socr.pipeline import UnifiedPipeline` | lazy PEP 562 `__getattr__` export | Today's eager import (:3) loads orchestrator on any `socr.pipeline.*` submodule import, defeating the "unit tests that do not import the whole orchestrator" AC. cli imports orchestrator directly (cli.py:437,526), unaffected. |
| `preflight.py` | `analyze_document(state, detector, config, *, console) -> DocumentAssessment` | `detector.detect` + `state.apply_born_digital` + audit-event emission (landscape refusal via `native_table_lane_refused`, table-structure-defect and `native_only`-gated branches, orchestrator.py:699-810) | Delegate passes `self.bd_detector`, stores `self._last_assessment`. `_sparse_page_ok` is quality policy (defined :3851, consumed :1132/3593/3951/4294) and does not move here. Imports core only. |
| `modes/agentic/routing.py` | `route_page`, `PageJudge`, `AcceptDecision`, `ProviderAttempt`, `PageDecision`, `_best_effort`, `_error_output` | git mv of `agentic.py` lines <= 315 | File-level imports are core-only (config/providers/result). `DEFAULT_PROVIDER_TIMEOUTS` (agentic.py:43) moves to `core.providers` first (see tables slice). |
| `modes/agentic/judging.py` | `HeuristicPageJudge`, `VLMPageJudge`, `NativeTableVerifierJudge`, `SourceEvidenceTableJudge`, `TimeoutJudge`, `build_page_judge(state, deps)`, `make_page_renderer(state, deps)`, `JUDGE_MODEL_CANDIDATES`, re-hosted `JUDGE_IDENTITY_HEURISTIC` | judge construction and composition; emits `judge_degraded_to_heuristic`; sets `state.agentic_judge_model` | `JudgeDeps`: `config`, `heuristics`, `_sparse_page_ok`, `_resolve_judge_model`, `_page_has_tables`, `_make_page_renderer`. Production must call `deps._make_page_renderer(state)` (patched at test_judge_hard_pages.py:69, test_silent_content_destruction.py:338; dispatched at :3159,:3551). `_JUDGE_MODEL_CANDIDATES` (class attr :3078, self-reads :3096,:3581) becomes a judging module constant with a facade class-attr alias (nothing patches it). judging.py owns its own `console`. Imports routing (types), plus function-local `socr.judge`, `socr.audit`, `socr.tables` as today. |
| `modes/agentic/workflow.py` | `run_agentic_document(state, output_dir, deps) -> None`; `restore_terminal_page_state(...)` | the fused loop (orchestrator.py:2116-3029) verbatim: resume pre-pass (:2185-2187,:2481-2483), chart/table arbitration incl. the guarded self-enforcing no-op prune (:2257-2283, move verbatim), ladder setup, `run_provider` closure, cascade-halt latch (top-of-loop :2445, trip :2854-2878), per-page lifecycle (route -> header repair/structural recheck/D3 floor :2705-2797 -> reread :2896 -> escalate/surface :2925-2954 -> equations :2960-2974 -> sanitize -> repetition guard :2988 -> blob put -> provisional flush `terminal=False`), `state.pp2_halt_reason` write (:3029) | `route_page` and `probe_ollama_idle` become this module's globals (all 16+6 patches repoint here). `provider_ladder` stays a function-local import (:2151); test_b2_routing.py:118 patches it at source `socr.core.providers.provider_ladder`. The facade keeps a `_restore_terminal_page_state` delegate: two tests call it directly (test_structural_gate_b1_gh151.py:262, test_tr3_d3_floor.py:712). |
| `modes/legacy/workflow.py` | `run_backbone`, `run_native_first`, `run_multi_engine`, `run_repair`, `run_consensus` | legacy phase sequencing; native-first includes the two `deps._run_engine_on_pages` calls (:1102,:1160), `_sparse_page_ok`-gated local-tier scoring (:1132), corrupt-math recovery (:915-1033), equation attach (:1079-1095 via deps) | `get_engine` reads at :835,:1112,:4101,:4159,:4224 (and :3755 in `_backbone_multi_engine`) become this module's global. `LegacyDeps` includes `repair_router` (constructed :210). |
| `modes/legacy/quality.py` | `run_score`, `run_judge_hard_pages` | `_phase_score`, `_score_whole_doc`, `_score_per_page` incl. the B1/GH-200/TR-3 fail-closed short-circuit, `_phase_score_multi`, `_score_repair_result`, `_native_table_structure_gate_applies`; `_phase_judge_hard_pages` (:3135 region) | The single legal cross-mode edge: looks up `judging.VLMPageJudge` module-qualified at call time (never `from judging import VLMPageJudge`), so class patches keep intercepting. |
| `lanes/tables.py` | `reread_page_tables`, `run_dual_pass`, `escalate_table_page`, `surface_table_scoring`, `table_page_needs_escalation`, `clear_fail_closed_flags`, `guard_table_repetition`, `repetition_truncated_note` | discrete table state-mutating helpers + their beside-mutation audit events (:1805 region, :3223, :3404) | Table policy boundary: the route-result decision sequence (header repair, structural recheck, D3 floor, rejected-ladder marking) stays in agentic workflow; shipping/status policy stays in assembly and `core.manifest`. `reread_page_tables` docstring pins its exception boundary to pre-refactor 065e5dd; move verbatim. Benchmark imports stay function-local exactly as today (`tables/escalation_decision.py:72`, `tables/native_rows.py:19` vs `benchmark/table_exactness.py:93,102` is a real lazy-order cycle). `_phase_dual_pass_tables` never calls `get_engine`. |
| `lanes/figures.py` | `describe_and_embed_figures`, `sanitize_page_image_refs`, `render_chart_page_png`, `render_chart_region_pngs`, `render_d3_floor_png`, `_get_page_context`, `_build_figure_blocks` | figure embed (:5411; sole `FigureExtractor` production call :5461), chart/D3 renders (:1418,:1459,:1558, all DocumentState-free) | `FigureExtractor` and a lane-local `console` become this module's globals (29 + 3 patches repoint here). `FigureLaneDeps`: `config`, `_last_assessment` (scanned-page skip), `_get_vision_engine` (stays a facade method: object.__new__ + patch.object sites at :1522,:1571,:1696), `_record_figure_recoverable_labels` (dispatch :5559 must go through deps; patch.object at test_orchestrator.py:3562; facade delegate kept for direct calls :3467,:3513,:3594). `_try_gemini` is nested inside `_get_vision_engine` (:5924-5931) and stays there. |
| `lanes/equations.py` | `detect_and_crop_equations` (:5697), `attach_equation_latex_sidecars` (:5779) | GH-36a/b orchestration | Inter-step registry stays `state.events` kind `equation_region_detected`. Both modes call through facade delegates, preserving the spy at test_pp2_agentic_fuse.py:720. |
| `persistence/identity.py` | `socr_version`, `socr_source_digest` + `_SOURCE_DIGEST_CACHE`, `manifest_versions`, `page_blob_key`, `resume_skippable` | run/source identity, pure | No engine/judge resolution; imports core/contract only. Facade call sites become module-qualified (`identity.socr_source_digest()`), never `from identity import x`, so gh214-style patches on the identity module intercept. |
| `persistence/pages.py` | `@dataclass(frozen=True) DocPaths`, `doc_paths_for(output_dir, pdf_path, scan_root)`, `flush_page_fragment`, `flush_page_sidecar(..., *, run_fingerprint, input_checksum, terminal)`, `load_terminal_page(..., *, run_fingerprint)`, sidecar-flag readback | atomic tmp-rename page I/O; canonicalizes the `doc_dir_for`/`relative_key` derivation repeated in `_page_fragment_path` (:4355) | `run_fingerprint`/`input_checksum` travel as explicit string values; only the facade computes them (persistence never imports engine resolution). Facade delegates keep today's exact signatures and compute `self._run_fingerprint()` per call; `_judge_model_cache` memoization (:193-204) makes per-call and per-document forms equal, so precompute is a later optimization, not part of this refactor. Mutates files only, never DocumentState. |
| `assembly.py` | `run_assemble(state, output_dir, deps) -> EngineResult`, plus moved bodies of `_write_metadata`, `_write_manifest` + `_fingerprint_inputs`, `_write_audit_log` (which calls `_write_tables_trust`, :5365), `_tables_trust_note`, `_save_markdown`, `_stitch_fragments`, `_canonical_body`, `_rewrite_all_fragments` | the finalization order it alone owns (:4871-5270): canonical body -> PP-1 terminal flush + stitch byte-verify -> loss buckets (`failed`/`d3_floor`/`native_only_distrust`/`native_fallback`, exclusion predicates verbatim) -> status/events -> `strip_phantom_images` -> error chain in append order (`pp2_halt_reason` :5172 -> repetition note -> trust note) -> save md -> provisional metadata (`:pre-figures`, :5343) -> figures -> conditional re-save + final metadata -> `_rewrite_all_fragments` (sole authoritative fragment writer, :5257) -> manifest (default-on in agentic, :5264) -> audit log (+ tables_trust.json) | `run_assemble` dispatches every spied operation through `AssembleDeps` facade delegates: `_save_markdown`, `_write_metadata`, `_write_manifest`, `_rewrite_all_fragments`, `_flush_page_fragment`, `_flush_page_sidecar`, `_stitch_fragments` (test_tr3_d3_floor.py:322-328,407-413) and `_describe_and_embed_figures` (test_silent_content_destruction.py:700, test_pp5_resume_ledger.py:860). Assembly therefore does NOT import `lanes/figures`. `_fingerprint_inputs` keeps its function-local `from socr.engines.registry import get_engine` (:3668); it has never read the orchestrator module global, so no `get_engine` patch seam moves with assembly. |

## Dependency and seam rules

Legal graph (imports; "via deps" edges carry no import):

```
cli -> pipeline.orchestrator (direct, unchanged)
orchestrator -> preflight, modes/*, lanes/*, assembly, persistence/*, siblings, core, contract
modes/agentic/workflow -> modes/agentic/{routing,judging}, core;
                          function-local: socr.tables.header_repair, socr.tables.structure_check,
                          socr.core.audit_log, socr.core.providers.provider_ladder, fitz
modes/agentic/judging  -> modes/agentic/routing (types), core;
                          function-local: socr.judge, socr.audit, socr.tables
modes/agentic/routing  -> core only
modes/legacy/workflow  -> pipeline.{repair,consensus}, socr.engines.registry, core;
                          function-local: socr.math.recover
modes/legacy/quality   -> modes/agentic/judging (call-time qualified lookup ONLY), core
lanes/*     -> socr.{tables,figures,math}, core, contract; socr.benchmark function-local only
assembly    -> persistence/{pages,identity}, core, contract;
               function-local: socr.engines.registry (inside _fingerprint_inputs)
persistence -> core, contract only (pages may import identity)
preflight   -> core only
```

Forbidden (enforced by a stdlib-`ast` boundary test added in the first slice; no new
dependency): any `socr.pipeline` submodule importing `orchestrator` (cli and the lazy
package `__getattr__` are the only orchestrator importers); `lanes -> modes`;
`modes <-> assembly`; `lanes -> assembly`; `assembly -> lanes`; `persistence -> `any
pipeline module except identity; `modes/agentic <-> modes/legacy` except
`legacy/quality -> agentic/judging`; module-level `socr.benchmark` imports in lanes;
any barrel import in a package `__init__.py` (all stay empty; a barrel
`modes.agentic.__init__` importing workflow would make the legal quality->judging edge
load the whole loop).

Seam-liveness rule (every slice): the commit that moves a patched global or method adds
or keeps a negative canary proving production still hits the patched lookup, i.e. patch
the new location with a raising sentinel (or assert the mock was called) and drive the
production path. A grep for the old patch path proves removal, never liveness. For
ambiguous `get_engine` patches in test_orchestrator (49 sites), triage empirically:
run the slice with the move and no repoint; the failure list is the repoint list; then
for each still-passing patch, make the patched engine raise and confirm the test fails
(rules out vacuous passes). The five test_b2_routing `get_engine` patches
(:269,:285,:296,:309,:319) all wrap `_available_engines_for_agentic` calls and stay on
`socr.pipeline.orchestrator.get_engine` permanently.

## State and mutation ownership

`DocumentState` remains the sole blackboard; no second run-state object, no event bus,
no DI container, no phase graph. `_last_assessment` stays a facade attribute (8 test
files assign it); extracted code reads it via deps. Mutation ownership: preflight
mutates page assessments and appends its own events; agentic workflow mutates routing
flags, attempts, halt latch, provisional artifacts; lanes mutate page text/table
flags/figure refs beside their own audit events; assembly alone mutates final status,
error chain, terminal sidecars, authoritative fragments, metadata/manifest/audit files;
persistence mutates only files. Audit events remain emitted beside witnessed mutations;
there is deliberately no audit module. Domain packages (`socr.tables`, `socr.figures`,
`socr.math`) stay DocumentState-free (verified zero references at baseline).

## Invariants (every slice must keep all of these green)

| Invariant | Mechanism | Guard |
|---|---|---|
| Final `.md` bytes | phase order verbatim; `_canonical_body`/stitch/rewrite moved without edits | golden byte-identity + replay fixtures |
| Fragment identity | `_rewrite_all_fragments` sole authoritative writer, in assembly; loop flush stays `terminal=False` crash-copy | test_pp1_fragment_flush, test_pp4 |
| Sidecar-vs-fragment contract | sidecar freezes pre-transform winning text; `_load_terminal_page` prefers fragment bytes (:4676-4679) | characterization C1 + pp5 suite |
| Status / loss surfacing | loss buckets + error-chain append order (halt -> repetition -> trust) move verbatim inside one function | characterization C3 + test_gh95_tables_trust, test_native_only_table_status_gh211 |
| Audit proximity | events beside witnessed mutations; no centralization | review rule + existing event tests |
| Resume conservatism | `_load_terminal_page` gate unchanged: skip only on terminal + fingerprint + checksum + SUCCESS + non-marker | test_pp5_resume_ledger, test_resume_source_version_gh214 |
| Engine-cost accounting | `restore_terminal_page_state` cost fold moves untouched; facade delegate kept | characterization C4 + direct-call tests |
| Provider hermeticity | `_available_engines_for_agentic` + orchestrator `get_engine` global never move | pp2/pp5/pp7 hermetic patches; CLAUDE.md rule |
| Patch-seam liveness | globals move with their last production reader; same-commit repoint; no re-export shims; negative canaries | per-slice seam inventory |
| Public imports | `socr.pipeline` still exports `UnifiedPipeline` (lazily); sibling module paths unchanged | subprocess import test |
| Writer authority asymmetry | loop provisional vs assembly terminal timing unchanged; terminal sidecars written mid-assemble before figures (:4904-4906) | C1/C3 + pp2 fuse suite |

Byte-identity caveat: every slice edits `src/socr/**/*.py`, so `socr_source_digest`
(GH-214) changes and existing corpora reprocess on next run. That is the fail-safe
working as designed, not slice failure. "Bytes unchanged" applies to final `.md`,
fragments, and sidecar fields excluding `run_fingerprint`/`socr_source_digest`.

## Non-goals and declined alternatives

- No behavior change in any extraction slice; fingerprint fixes are separate tickets (below).
- No event bus, DI framework, service container, generic phase graph, second run-state object.
- No relocation of `consensus.py`, `repair.py`, `reconciler.py`, `hpc_pipeline.py`; no Rust.
- No compatibility shim for `socr.pipeline.agentic`; no `pipeline/predicates.py`;
  no `pipeline/classify.py`; no shared BlobStore owner (two independent fail-open
  constructions, :2392-2404 and :3701-3710, are correct for a content-addressed store).
- LOC is not the goal: issue AC hard target < 1.5k LOC for the facade is retained as a
  final gate (residual estimate ~950); the < 800 ideal is reported, not enforced.

## Known separate defects (never folded into extraction slices)

Each fingerprint ticket invalidates cached corpora; coordinate with running corpus jobs.

- **D1a** `auto_patch_tables` read at :3290, absent from `_run_fingerprint` extras;
  flips whether dual-pass patches ship.
- **D1b** `clean_equation_model` read at :5829; only the boolean
  `recover_clean_equations` is fingerprinted; sidecar bytes change silently. Includes a
  decision on `math_model_host`: read via `hasattr` at :5854-5855 but NOT a declared
  `PipelineConfig` field; declare it or delete the read.
- **D1c** `qwen_vllm_model`: crop reader reads it at :3117-3118 when
  `qwen_backend in ("vllm","sglang","api")`, but `BaseEngine.resolved_model_version`
  reads only `config.qwen_model` and `fingerprint_determinants` covers only
  `qwen_backend` (engines/base.py:76-104; no qwen override). Fix at the resolver
  boundary: make `QwenEngine.resolved_model_version`/`fingerprint_determinants` use
  `resolve_qwen_intent` (engines/qwen.py:73-112).
- **D1d** `gemini_model` figure captions: `_try_gemini` uses `config.gemini_model`
  (:5929-5930) regardless of `enabled_engines`. Covered on defaults
  (`enabled_engines` defaults to `list(EngineType)`, core/config.py:107, folded via
  `enabled_engine_determinants`); real omission only under a custom `enabled_engines`
  excluding GEMINI while captions still fall back to Gemini.
- **D1e** `recover_corrupt_math` + `math_model`: alter native-first bytes via
  `recover_math_regions`/`splice_math` (:915,:1017,:1027-1031); absent from extras.
- **D2** ~~promote `state._pp2_halt_reason` (written :3029 under
  `type: ignore[attr-defined]`, read :5172 via `getattr`) to a declared
  `DocumentState` field.~~ **DONE** (#234, PR #236): declared as the public
  field `DocumentState.pp2_halt_reason` (`core/state.py`), with the
  `type: ignore` and the `getattr` both dropped. Search for the name
  **without** the leading underscore.
- **D5** `_page_blob_key` (:60-71) does not mirror the BlobStore key as its docstring
  claims: default JSON ASCII escaping + `"sha256:"` prefix vs `core.cache._canonical_bytes`
  `ensure_ascii=False` + bare hex (core/cache.py:24-37). Non-ASCII winning text cannot
  cross-reference the manifest. Separate behavior ticket: delegate to `core.cache.blob_hash`,
  define the prefix format, add a non-ASCII regression test.
- **D4** (documentation) propose to the owner demoting the < 800 ideal to a report
  metric; the < 1.5k hard AC stays.

## Parked follow-ups

- **P5** reduce `_last_assessment` reach by copying `word_count` /
  `has_corrupt_math` / `text_is_rotated` onto `PageState` in `apply_born_digital`
  (touches 8 test files' seams; after the split).
- **P6** optionally move the DocumentState-free chart render primitives from
  `lanes/figures.py` into a new `socr/figures/` domain module; requires explicit
  approval for a new domain file.
- **P7** precompute `run_fingerprint` once per document (pure optimization; equal by
  `_judge_model_cache` memoization).
```
