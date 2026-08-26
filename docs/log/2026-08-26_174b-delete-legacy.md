# R174b legacy deletion proof and pre-edit report

## Ruling source

The governing ruling is `docs/log/2026-08-25_174-legacy-fork.md`, read from the
current branch at authoritative commit `fbb1a2f` before this proof was changed:

```text
git show fbb1a2f:docs/log/2026-08-25_174-legacy-fork.md
```

The ruling is DELETE. This report does not reopen or revise that decision. The
`--legacy-routing` kill-list addition remains attributed to `5288e5a`; the
current branch carries that ruling file through `fbb1a2f`. This report records
the current-tree proof needed to execute it: 88 `UnifiedPipeline` methods and a
13-method, 1,153-AST-line legacy-only set.

## Reachability proof

The disposable probe is
`.troupe/runs/20260826-110542-standard-feature/workers/execute-t1/probe_r174b.py`.
It parses `UnifiedPipeline` with `ast`, treats every `self.<method>` attribute
reference as a graph edge (including callback references), and derives route
heads from the `process()` control-flow AST. It does not use the ruling's kill
list as a root set. Nested function and class bodies are opaque only while
deriving direct `process()` roots; method graph construction includes their
bound-method references.

Command and result:

```text
uv run pytest -s .troupe/runs/20260826-110542-standard-feature/workers/execute-t1/probe_r174b.py
1 passed in 0.22s
```

The direct calls found in `process()` are:

- Agentic region (`if self.config.agentic and not is_multi`): `_phase_agentic`.
- Multi-engine region (`elif is_multi`): `_backbone_multi_engine`,
  `_phase_score_multi`, `_phase_consensus`.
- Deterministic region (`else`): `_phase_backbone`, `_phase_score`,
  `_phase_judge_hard_pages`, `_phase_repair`, `_phase_consensus`.
- Post-fork dual-pass region (`if self.config.dual_pass_tables and not
  (self.config.agentic and not is_multi)`): `_phase_dual_pass_tables`.
- All direct process roots, including common bookkeeping/presentation calls,
  are `_resolve_output_root`, `_resume_skip`, `_phase_analyze`, `_phase_assemble`,
  `_print_summary`, and the route references above. For the processing-lane
  reachability partition, common roots are `_resolve_output_root`,
  `_phase_analyze`, and `_phase_assemble`; `_resume_skip` and `_print_summary`
  remain reported but are not route roots.

For the processing-lane proof, route roots are the agentic branch head plus the
common processing heads `_resolve_output_root`, `_phase_analyze`, and
`_phase_assemble`. Resume bookkeeping and final presentation are reported but
not treated as processing-lane roots because they still contain fields under
deletion review. The derived counts are:

| Set | Count |
| --- | ---: |
| UnifiedPipeline methods | 88 |
| Agentic-reachable | 70 |
| Legacy-reachable | 29 |
| Legacy-minus-agentic | 13 |

The current-tree legacy-only methods and `ast` `lineno`/`end_lineno` spans are:

| Method | `lineno` | `end_lineno` | AST lines |
| --- | ---: | ---: | ---: |
| `_backbone_native_first` | 1081 | 1373 | 293 |
| `_phase_repair` | 4893 | 5101 | 209 |
| `_phase_dual_pass_tables` | 4224 | 4335 | 112 |
| `_score_per_page` | 4719 | 4818 | 100 |
| `_phase_judge_hard_pages` | 3949 | 4041 | 93 |
| `_phase_backbone` | 1010 | 1079 | 70 |
| `_phase_score_multi` | 4820 | 4887 | 68 |
| `_backbone_multi_engine` | 4566 | 4632 | 67 |
| `_phase_consensus` | 5149 | 5192 | 44 |
| `_score_repair_result` | 5103 | 5143 | 41 |
| `_score_whole_doc` | 4655 | 4687 | 33 |
| `_phase_score` | 4638 | 4653 | 16 |
| `_native_table_structure_gate_applies` | 4711 | 4717 | 7 |
| **Total** |  |  | **1,153** |

The corrected AST-derived set is exactly the ruling's 13 methods, with no
derived-minus-ruling or ruling-minus-derived discrepancy. `_sparse_page_ok` is
not legacy-only: it is reachable from `_build_page_judge` through the callback
reference `sparse_ok=self._sparse_page_ok` at current
`src/socr/pipeline/orchestrator.py:4413`, so it remains in the agentic graph and
is excluded from deletion.

The prior probe had a call-only edge bug. It recognized `self.foo` only when the
attribute was the function of an `ast.Call`, so it recorded `self.foo(...)` but
missed `self._sparse_page_ok` passed as a keyword callback. Because the judge
invokes that callback later, the missing edge `_build_page_judge ->
_sparse_page_ok` incorrectly reduced agentic reachability to 69 and inflated
the legacy-minus-agentic set to 14 methods and 1,174 span lines. The corrected
probe treats every bound-method attribute reference as an edge.

## Config-consumer proof

The same AST probe mapped every read of the six fields before deletion:

| Field | Containing methods and lines |
| --- | --- |
| `multi_engine` | `_run_fingerprint` (368, 373); `process` (562); `_backbone_multi_engine` (4577); `_print_summary` (7584, 7585) |
| `consensus_enabled` | `_run_fingerprint` (443); `process` (605) |
| `consensus_use_llm` | `_run_fingerprint` (444); `_phase_consensus` (5179) |
| `consensus_ollama_model` | `_run_fingerprint` (445); `_phase_consensus` (5180) |
| `max_retries` | `_run_fingerprint` (441); `_phase_repair` (4983, 4999, 5042) |
| `truncation_retries` | `_run_fingerprint` (440); `_phase_repair` (4922, 4936, 4941) |

The agentic route-processing set (`_phase_agentic`, `_phase_analyze`, and
`_resolve_output_root`) consumes none of these fields. The full 70-method
agentic production graph also reaches shared `_run_fingerprint` through
assembly bookkeeping; those reads are explicitly visible in the map and are
scheduled for removal as dead fingerprint material. The `process` branch
condition and summary/resume bookkeeping are likewise visible, not silently
omitted consumers. No agentic route method itself consumes a deleted field.

An AST import walk over every `src/socr/**/*.py` module, including literal
`import_module()` and `__import__()` calls, found:

- `src/socr/pipeline/consensus.py`: consumer is only
  `src/socr/pipeline/orchestrator.py`.
- `src/socr/pipeline/repair.py`: consumer is only
  `src/socr/pipeline/orchestrator.py`.
- `src/socr/pipeline/reconciler.py`: imported by
  `src/socr/pipeline/hpc_pipeline.py`.

Therefore `consensus.py` and `repair.py` are removable only with the
orchestrator legacy lane, while `reconciler.py` must survive for the HPC path.
This is an AST import proof, not a grep-only conclusion.

## Test triage

The current-tree inventory found references in:

- Legacy symbols: `tests/test_canon_round3.py`,
  `tests/test_cli_flag_agentic_status_gh142.py`,
  `tests/test_config_from_file.py`, `tests/test_consensus.py`,
  `tests/test_orchestrator.py`, `tests/test_p1_cascade_economics.py`,
  `tests/test_pp5_resume_ledger.py`, `tests/test_r174b_ast_reachability_contract.py`,
  `tests/test_r174b_config_schema.py`,
  `tests/test_r174b_dual_pass_and_table_contracts.py`,
  `tests/test_r174b_orchestrator_agentic_lane.py`,
  `tests/test_repair_router.py`, and
  `tests/test_silent_content_destruction.py`.
- Legacy flags: `tests/test_b2_routing.py`, `tests/test_canon_round3.py`,
  `tests/test_orchestrator.py`, and `tests/test_r174b_cli_guards.py`.

The ruling's original nine-file list is fully inventoried and classified:

| File | Classification | Reason |
| --- | --- | --- |
| `tests/test_consensus.py` | DELETE | The legacy consensus implementation is the subject. |
| `tests/test_orchestrator.py` | REWRITE/RETAIN | Remove legacy lane cases; retain surviving orchestrator behavior. |
| `tests/test_config_from_file.py` | REWRITE | Remove deleted-key expectations; retain live config loading. |
| `tests/test_cli_flag_agentic_status_gh142.py` | REWRITE | Remove legacy-flag contract; retain agentic status coverage. |
| `tests/test_canon_remediation.py` | KEEP | Historical `consensus(<engine>)` manifest compatibility. |
| `tests/test_canon_round3.py` | REWRITE | Remove dead consensus/multi-engine fingerprint cases; retain live fingerprint coverage. |
| `tests/test_pp5_resume_ledger.py` | REWRITE/RETAIN | Remove dead routing references; retain resume-ledger behavior. |
| `tests/test_binding.py` | KEEP | Its prose use of the ordinary word “consensus” is unrelated to the deleted implementation. |
| `tests/test_gh225_fabricated_image_urls.py` | KEEP/REVIEW | Fabricated-image guard behavior is live agentic behavior; legacy wording is descriptive. |

`test_canon_remediation.py` and `test_binding.py` are intentional KEEP items;
neither is silently omitted from triage.

## Intentional historical compatibility

Historical manifests may contain engine labels such as
`consensus(gemini)`. The manifest/readback compatibility behavior that resolves
the wrapper to its underlying engine remains a live compatibility contract.
The consensus implementation and routing lane can be deleted without deleting
that string-normalization/readback behavior. The dedicated historical test in
`test_canon_remediation.py` is therefore KEEP.

## Cross-repository edits

No cross-repository edits were made. The exact read-only migration inventory was:

- Sibling OCR CLI `/Users/rubenffuertes/repos/tools/qwen-ocr-cli`: no matches for
  the legacy flags, six config fields, or `socr.pipeline.consensus`/
  `socr.pipeline.repair`.
- Other source repositories under `/Users/rubenffuertes/repos/tools/*`,
  `/Users/rubenffuertes/repos/databases/*`, and
  `/Users/rubenffuertes/repos/open-source/*` (excluding the socr checkout and
  qwen-ocr-cli): no matches for the migration identifiers.
- `/Users/rubenffuertes/repos/tools/socr` is a separate checkout of the same
  `r-uben/socr` repository on `fix/181-iterative-cluster`; it was not edited.

No external flag/config consumer was found that requires a migration edit.

## Verification

Fresh pre-edit results for the required focused checks:

```text
uv run pytest tests/test_r174b_*.py -q
20 failed, 7 passed, 5 warnings in 48.27s. The failures are the pre-deletion
R174b contracts for the not-yet-removed legacy methods, modules, flags, and
config fields; this proof task did not modify `src/` or `tests/`.

uv run pytest tests/test_pp2_agentic_fuse.py::TestByteIdentity -q
2 passed, 5 warnings in 0.92s

uv run pytest --collect-only -q
2267 tests collected in 2.18s

uvx ruff@0.16.0 format --check \
  .troupe/runs/20260826-110542-standard-feature/workers/execute-t1/probe_r174b.py
1 file already formatted
```

The already-recorded full-suite pre-edit baseline remains `2257 passed, 7
failed, 3 xfailed` with `2267 collected`; no fresh full-suite command was run
because the required focused checks are the scope of this worker and no source
or test file changed.

The retained full-suite baseline's seven failures are the seven pre-existing
`tests/test_r174b_cli_guards.py` assertions that expect the not-yet-deleted
`--legacy-routing`, `--multi-engine`, and `--consensus-llm` entry points to be
rejected. In the pre-edit tree those flags still execute or appear in help, so
the failures are expected evidence of the pending deletion, not test edits. The
fresh focused R174b run has 20 failures because it also includes the pending
method, module, and config-surface deletion contracts.

The byte-identity baseline passed. No repository file was staged or committed.

### Worker t2 pre-edit node ids

Before editing the R174b acceptance tests, this worker ran the mandated
`uv run pytest tests/test_r174b_*.py -q`. The actual 20 failing node ids were:

```text
tests/test_r174b_ast_reachability_contract.py::TestUnifiedPipelineASTReachability::test_legacy_13_methods_reachability_partition
tests/test_r174b_ast_reachability_contract.py::TestUnifiedPipelineASTReachability::test_agentic_processing_methods_read_no_dead_config_fields
tests/test_r174b_ast_reachability_contract.py::TestModuleImportContracts::test_consensus_and_repair_import_boundaries
tests/test_r174b_cli_guards.py::TestCLINonexistenceGuards::test_process_rejects_deleted_flags[flag_args0]
tests/test_r174b_cli_guards.py::TestCLINonexistenceGuards::test_process_rejects_deleted_flags[flag_args1]
tests/test_r174b_cli_guards.py::TestCLINonexistenceGuards::test_process_rejects_deleted_flags[flag_args2]
tests/test_r174b_cli_guards.py::TestCLINonexistenceGuards::test_batch_rejects_deleted_flags[flag_args0]
tests/test_r174b_cli_guards.py::TestCLINonexistenceGuards::test_batch_rejects_deleted_flags[flag_args1]
tests/test_r174b_cli_guards.py::TestCLINonexistenceGuards::test_process_help_omits_deleted_flags
tests/test_r174b_cli_guards.py::TestCLINonexistenceGuards::test_batch_help_omits_deleted_flags
tests/test_r174b_config_schema.py::TestPipelineConfigSchema::test_deleted_fields_are_absent_from_dataclass
tests/test_r174b_config_schema.py::TestPipelineConfigSchema::test_from_file_rejects_deleted_keys[multi_engine-bad_value0]
tests/test_r174b_config_schema.py::TestPipelineConfigSchema::test_from_file_rejects_deleted_keys[consensus_enabled-True]
tests/test_r174b_config_schema.py::TestPipelineConfigSchema::test_from_file_rejects_deleted_keys[consensus_use_llm-True]
tests/test_r174b_config_schema.py::TestPipelineConfigSchema::test_from_file_rejects_deleted_keys[consensus_ollama_model-qwen3.5:cloud]
tests/test_r174b_config_schema.py::TestPipelineConfigSchema::test_from_file_rejects_deleted_keys[max_retries-3]
tests/test_r174b_config_schema.py::TestPipelineConfigSchema::test_from_file_rejects_deleted_keys[truncation_retries-2]
tests/test_r174b_orchestrator_agentic_lane.py::TestOrchestratorAgenticLane::test_process_unconditionally_executes_agentic_phase[True]
tests/test_r174b_orchestrator_agentic_lane.py::TestOrchestratorAgenticLane::test_process_unconditionally_executes_agentic_phase[False]
tests/test_r174b_orchestrator_agentic_lane.py::TestOrchestratorAgenticLane::test_fingerprint_excludes_dead_legacy_keys
```

Each failure was an acceptance property for a still-present legacy method,
module, flag/config surface, conditional routing branch, or dead fingerprint
key; no guessed failure count was used.

## Bugs found but not fixed

- The pre-edit `process()` still exposes the deterministic/multi-engine fork and
  its CLI/config entry points; this is the subject of the subsequent deletion
  work, not a redesign performed by this proof task.
- `_run_fingerprint` (reached from resume bookkeeping and assembly sidecars)
  and `_print_summary` still read legacy fields in the pre-edit tree. They were
  deliberately shown in the config-consumer map; the route-processing methods
  themselves consume none of them. The deletion implementation must remove the
  shared bookkeeping reads rather than rely on a post-deletion grep.
- The obsolete call-only probe edge omitted `_build_page_judge`'s callback
  reference `sparse_ok=self._sparse_page_ok` at
  `src/socr/pipeline/orchestrator.py:4413`. The corrected bound-method-aware
  graph makes `_sparse_page_ok` agentic-reachable, so it is not legacy-only and
  must survive deletion.

## Known coverage gap, deliberately left open

**The `not is_native` gate on the in-loop table re-read has no test.**

`origin/main:tests/test_dual_pass_tables.py:421` asserted `reader.calls == 0` — native text
is character-exact, so its tables must never be re-read. That test drove the deleted
`_phase_dual_pass_tables`, so it went with the phase. The gate it guarded **survives**, in
the agentic loop's in-loop re-read call site.

Four attempts to re-drive it through `_phase_agentic` all failed the same way: the page was
re-read anyway. In order — no detector mock (page routed to OCR, where re-reading is
correct), then a born-digital assessment with `native_text=""` (no text layer, so still
OCR), then a full assessment carrying a real text layer. It was still re-read.

**Hypothesis, NOT verified:** a born-digital page *with tables* is routed to OCR enhancement
because native table extraction is not trusted, so a native winner that also has tables may
be unreachable. If that is right, the `not is_native` gate is close to dead code and the
question is whether to delete it or widen it — not how to test it.

A first attempt at a replacement test evaluated the `not is_native` condition inside the test
body. That asserted nothing about production and was removed; recording it because it is
exactly the vacuous-guard pattern this repo keeps rediscovering.

Whoever picks this up: settle the hypothesis first. Testing a gate that cannot fire is wasted
work, and if it genuinely cannot fire that is the more interesting finding.
