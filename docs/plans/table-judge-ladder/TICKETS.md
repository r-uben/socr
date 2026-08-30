# TICKETS — table judge ladder (GH-353)

Implements `docs/log/2026-08-30_table-judge-ladder.md` (design) with the CLI₁ seat from
`docs/log/2026-08-30_gh356-bakeoff.md` (glm-5.3-flash:cloud via ollama HTTP API; CLI₂ =
gemini CLI). Panel-reviewed decomposition (codex/gemini/grok advisory, 2026-08-30).

Status keys: `TODO` · `WIP` · `DONE` · `BLOCKED`. `depends-on` gates dispatch.
Parallelizable = no shared files / no dep. Each ticket = one implementer agent,
then one reviewer pass before commit.

Standing constraints (every ticket):
- Feature flag `table_judge_ladder` **defaults OFF**. Golden/byte-identity tests must
  stay byte-identical with the flag off.
- CI has no ollama and no `gemini` binary. Tests mock at the module-local seam (rung
  callables / transport helper), never patch httpx inside the orchestrator, and patch
  `_available_engines_for_agentic` + `_resolve_judge_model` → `""` in any `process()`
  test. Pin flag-on vs flag-off DIFFERENCES in one process; never absolute outcome
  tuples (the #253/#257 trap).
- Lint gate: `uvx ruff@0.16.0 format --check .`

## Stream A — judge core (new files, no orchestrator surgery)

### TICKET-A0 — judge prompt policy file · DONE · depends-on: none · wave 1
**Problem:** A2/A3 have nothing to send; the design left prompt wording open.
**Do:** Write `src/socr/prompts/table_judge.md` (schema instructions, empty-cell rule,
"judge only the table region", tiebreak-findings injection slot) + a loader mirroring
`load_judge_prompt`. Start from the bake-off prompt (results log, Method).
**Files:** `src/socr/prompts/table_judge.md`, `src/socr/judge/table_prompt.py`
**Done when:** `~/venvs/socr/bin/pytest tests/test_table_prompt.py -q` exits 0; loader
returns the template with and without a findings payload; prompt file contains all six
finding codes verbatim.

### TICKET-A1 — verdict schema, rung contract, event kinds · DONE · depends-on: none · wave 1
**Problem:** The ladder needs one owned contract: verdict schema, an S1-failure
representation distinct from FAIL, a standard rung signature, and audit-event kinds.
**Do:** New `src/socr/judge/table_verdict.py`: `TableJudgeVerdict` (`verdict: PASS|FAIL`,
`confidence: high|low`, `findings[{code,where,detail}]`, closed 6-code enum
MISSING_VALUE / FABRICATED_VALUE / WRONG_BINDING / HEADER_MANGLED / STRUCTURE_MERGED /
NOT_A_TABLE); `RungResult` (S1 ok/failed + verdict + latency + rung id); rung callable
protocol `(crop_path, markdown, prior_findings) -> RungResult`; parser that strips
markdown fences (reuse `_extract_json`-style extraction — fenced JSON is NOT ¬S1; the
gemini CLI fences routinely) and treats missing required fields / unknown `code` /
non-JSON as S1 failure; audit-event kind constants for ladder verdicts.
**Files:** `src/socr/judge/table_verdict.py`, `tests/test_table_verdict.py`
**Done when:** `~/venvs/socr/bin/pytest tests/test_table_verdict.py -q` exits 0; tests
cover: exact JSON, fenced JSON (accepted), PASS⇔empty findings, missing `verdict`,
unknown `code`, prose-wrapped JSON, empty output.

### TICKET-A2 — CLI₁ rung: ollama HTTP judge · DONE · depends-on: A0, A1, G1 · wave 2
**Problem:** Rung 1 (glm-5.3-flash:cloud) needs a client. `OllamaVisionJudge` POSTs
`/api/generate`; the bake-off integration path is `/api/chat` + `format=json`
(`ollama run --images` does not exist on 0.32.15).
**Do:** New `src/socr/judge/table_rung_ollama.py`: POST `/api/chat` with base64 crop,
`format="json"`, model/host/timeout from config (G1); timeout / connection error /
HTTP error → `RungResult(S1 failed)`. Transport isolated in a module-local helper so
tests mock the seam, not httpx globally. Do NOT reuse `OllamaVisionJudge.is_available()`
(live `/api/tags`).
**Files:** `src/socr/judge/table_rung_ollama.py`, `tests/test_table_rung_ollama.py`
**Done when:** `~/venvs/socr/bin/pytest tests/test_table_rung_ollama.py -q` exits 0;
tests assert the exact JSON payload (model, format, images, stream=False); timeout
exception → ¬S1 without sleeping; a network-must-not-run sentinel guards every test.

### TICKET-A3 — CLI₂ rung: gemini CLI invoker · TODO · depends-on: A0, A1, G1 · wave 2
**Problem:** Rung 2 needs a per-crop subprocess invoker; `GeminiEngine` is a
document-level engine, not reusable.
**Do:** New `src/socr/judge/table_rung_gemini.py`: invoke the configured gemini binary
(G1) with the crop as a file path (crop is already a temp file from B0; do not delete
another module's file), prompt via A0 with `prior_findings` injected in tiebreak mode;
strict-parse stdout via A1; timeout / missing binary / non-zero exit → ¬S1.
**Files:** `src/socr/judge/table_rung_gemini.py`, `tests/test_table_rung_gemini.py`
**Done when:** `~/venvs/socr/bin/pytest tests/test_table_rung_gemini.py -q` exits 0;
tests patch the module-local subprocess helper (not PATH / not `shutil.which`), pin the
exact argv, and prove timeout → ¬S1 without sleeping.

### TICKET-A4 — ladder state machine + page reducer · DONE · depends-on: A1 · wave 2
**Problem:** The S1/S2 transition logic and the multi-table→page reduction are unowned.
**Do:** New `src/socr/judge/table_ladder.py`, pure functions over injected rung
callables: per-table ladder — A(high-conf PASS)→accept; A(low-conf PASS)→confirm at
next rung; B(FAIL)→tiebreak with findings; C(¬S1)→substitute without findings;
B-exhaustion→REJECTED; C-exhaustion→UNVERIFIED — plus the page reducer: any
REJECTED table ⇒ page REJECTED; else any UNVERIFIED ⇒ page UNVERIFIED; else accepted
(per-table results all kept for the sidecar/audit events).
**Files:** `src/socr/judge/table_ladder.py`, `tests/test_table_ladder.py`
**Done when:** `~/venvs/socr/bin/pytest tests/test_table_ladder.py -q` exits 0 against
the explicit transition table in the ticket (A-high, A-low→A, A-low→B, A-low→¬S1,
B→A, B→B, B→¬S1, C→A, C→B, C→¬S1, mixed multi-table pages incl. one-PASS+one-FAIL and
one-PASS+one-¬S1).

## Stream B — witnesses and trust

### TICKET-B0 — table witness preparation · TODO · depends-on: A1 · wave 2
**Problem:** Nothing maps emitted markdown tables to located regions: the locator
over-merges stacked tables (`tables/locate.py:132`), borderless tables can yield no
box, `binding.parse_grid` reads only the first grid, and `_render_crop` is private
with caller-owned cleanup.
**Do:** New `src/socr/tables/witness.py`: enumerate emitted table blocks
(`find_table_blocks`), associate each with a `locate_tables` region (or an explicit
missing/ambiguous witness state), render crops with guaranteed cleanup (context
manager), assign stable per-page table ids. Missing witness is representable, never an
exception.
**Files:** `src/socr/tables/witness.py`, `tests/test_table_witness.py`
**Done when:** `~/venvs/socr/bin/pytest tests/test_table_witness.py -q` exits 0; cases:
1 block/1 box, 2 blocks/1 merged box (ambiguous), 1 block/0 boxes (missing witness),
temp files removed after the context exits.

### TICKET-B2 — table-scoped trust events · TODO · depends-on: A1 · wave 2
**Problem:** `tables_trust` resolves by page number (`core/tables_trust.py:216`): one
resolving event erases every distrust event on the page, so one PASS could erase
another table's REJECTED.
**Do:** Register ladder event kinds in `TABLE_DISTRUST_KINDS`/`RESOLVING_KINDS` with
table-scoped resolution (or a page-level resolving event emitted only when the reducer
accepts every table); trust-note wording for both terminals.
**Files:** `src/socr/core/tables_trust.py`, `tests/test_gh95_tables_trust.py` (extend)
**Done when:** `~/venvs/socr/bin/pytest tests/test_gh95_tables_trust.py -q` exits 0
including new cases "one PASS + one FAIL" and "one PASS + one ¬S1" — the FAIL/¬S1
event survives in `tables_trust`.

## Stream C — status plumbing

### TICKET-C1 — terminal enums + serialization · DONE · depends-on: none · wave 1
**Problem:** The two terminals don't exist as states.
**Do:** `FailureMode.TABLE_REJECTED` / `FailureMode.TABLE_UNVERIFIED` in
`core/result.py` with serialization round-trip. Nothing sets them yet.
**Files:** `src/socr/core/result.py`, `tests/test_result_table_terminals.py`
**Done when:** `~/venvs/socr/bin/pytest tests/test_result_table_terminals.py -q` exits
0; both modes round-trip `PageOutput.to_dict`/`from_dict`.

### TICKET-C3 — manifest: disposition survives winner selection · TODO · depends-on: C1 · wave 2
**Problem:** Final winner selection can substitute native text or another attempt after
the gate (`core/manifest.py:937/996/1156`), and native-only can reconstruct a demoted
page as clean SUCCESS (`:1271`) — a REJECTED verdict could silently vanish.
**Do:** A durable page-level ladder disposition on the page state consumed by
`_winning_page_output`; every selection/reconstruction path preserves
REJECTED/UNVERIFIED (a rejected candidate can lose selection, but the page cannot
regain SUCCESS while its disposition is REJECTED).
**Files:** `src/socr/core/manifest.py`, `tests/test_manifest_ladder_disposition.py`
**Done when:** `~/venvs/socr/bin/pytest tests/test_manifest_ladder_disposition.py -q`
exits 0; injection goes through `_winning_page_output` (not merely `best_output`), and
the native-only reconstruction path (`:1271`) preserves the demotion.

### TICKET-C2 — document aggregation + CLI surfacing · TODO · depends-on: C1, C3, G1 · wave 3
**Problem:** The terminals must surface at document status, metadata, and CLI (the
no-silent-loss rule), through the hand-maintained `pages_ok` chain.
**Do:** `pages_ok` terms + document-status handling in `_phase_assemble`
(`orchestrator.py:5138-5191`); metadata note text (contract `Status` stays
COMPLETED/PARTIAL/FAILED — terminals map to PARTIAL + explicit note; contract change is
out of scope); `_print_summary` and batch-level surfacing.
**Files:** `src/socr/pipeline/orchestrator.py` (assemble + summary), `src/socr/cli.py`,
`tests/test_ladder_status_surfacing.py`
**Done when:** `~/venvs/socr/bin/pytest tests/test_ladder_status_surfacing.py -q` exits
0: injecting each mode into assemble (chart-lane test pattern) flips `pages_ok`, yields
`DocumentStatus.AUDIT_FAILED`, metadata `Status.PARTIAL` with a note naming the mode,
and `_print_summary` output contains both terminal names; flag-off control unchanged.

## Stream G — config

### TICKET-G1 — flag, config, fingerprint prep · DONE · depends-on: none · wave 1
**Problem:** The ladder needs a default-off switch and all knobs config-visible so tests
and H1 can inject dummies (no bare constants).
**Do:** `PipelineConfig` fields: `table_judge_ladder: bool = False`, rung-1 model
(default `glm-5.3-flash:cloud`), rung-1 host/endpoint, rung-2 binary name (default
`gemini`), `table_judge_timeout_sec` (default 600 — named, documented from the bake-off
measurement); CLI flag `--table-judge-ladder` in `common_options` + `build_config`
(watch the unconditional-override pattern at `cli.py:371` — don't clobber YAML with
Click defaults); document in G1's Do: `--strict-local` + ladder ⇒ both rungs
unavailable ⇒ every table page UNVERIFIED (B1 implements exactly that, no guessing).
Fingerprint extras land in B1, not here.
**Files:** `src/socr/core/config.py`, `src/socr/cli.py`, `tests/test_ladder_config.py`
**Done when:** `~/venvs/socr/bin/pytest tests/test_ladder_config.py -q` exits 0:
`socr process --help` lists the flag; `PipelineConfig().table_judge_ladder is False`;
YAML round-trip persists the new fields (generic sweep in `test_config_from_file.py`).

## Stream D/E/B — the gate and its sequels (orchestrator.py, serialized)

### TICKET-B1 — the gate · TODO · depends-on: A2, A3, A4, B0, B2, C1, C3, G1 · wave 4
**Problem:** The single choke point does not exist; five sites finalize a table today.
**Do:** In `_phase_agentic`, AFTER `_guard_agentic_page_table_repetition` (~`:3099`,
so the judged table is the shipped table), behind the flag: B0 witnesses → A4 ladder
with A2/A3 rung callables injected (constructed once, injectable for tests) → audit
events (A1 kinds) → page disposition (C3) + failure modes (C1). Skip rules:
`engine == "chart_asset"` skips; missing witness / render failure / any infra error ⇒
¬S1 → UNVERIFIED, never a silent pass and never an exception; ladder ¬S1 must NOT
write `"timeout"` into `route_page` attempt fields (`:2970` substring-arms
cascade-halt). Add fingerprint extras: flag, both rung model/binary identities,
`table_judge_timeout_sec`, prompt file digest (extend `live_keys` in
`test_r174b_orchestrator_agentic_lane.py`).
**Files:** `src/socr/pipeline/orchestrator.py`, `tests/test_table_judge_gate.py`
**Done when:** `~/venvs/socr/bin/pytest tests/test_table_judge_gate.py tests/test_pp2_agentic_fuse.py -q`
exits 0. Gate test: two `process()` runs in one test (flag on vs off, rung callables
injected, `_available_engines_for_agentic` patched, `_resolve_judge_model` → `""`),
asserting the DIFFERENCE (flag-off unchanged; flag-on carries the injected verdict's
disposition + audit events). Includes a native-only defective-table case (the former
F1). Fail-open probes: missing crop, render error, parser error, HTTP error, missing
binary ⇒ UNVERIFIED. Flag-off run makes zero crop/HTTP/subprocess calls (sentinel).

### TICKET-D1a — sidecar persistence + restore · TODO · depends-on: B1 · wave 5
**Problem:** `_restore_terminal_page_state` (`:4605`) does not restore audit events —
a skipped page would lose its ladder events, trust entry, and metadata note on resume.
**Do:** Persist a `table_ladder` record (per-table results + disposition) in
`_flush_page_sidecar`; restore it (events + disposition) in
`_restore_terminal_page_state`.
**Files:** `src/socr/pipeline/orchestrator.py`, `tests/test_ladder_sidecar.py`
**Done when:** `~/venvs/socr/bin/pytest tests/test_ladder_sidecar.py -q` exits 0:
sidecar round-trips byte-stable; restored state reproduces `tables_trust` and the
metadata note without re-judging.

### TICKET-D1b — resume skip policy · TODO · depends-on: D1a · wave 6
**Problem:** `_load_terminal_page` (`:4315`) requires SUCCESS to skip; REJECTED must
skip-and-keep (deliberate exception to doubt⇒reprocess) while UNVERIFIED reprocesses.
**Do:** Positive early-return for a terminal REJECTED page after fingerprint + checksum
validation but before the SUCCESS/audit checks; UNVERIFIED never skips.
**Files:** `src/socr/pipeline/orchestrator.py`, `tests/test_ladder_resume.py`
**Done when:** `~/venvs/socr/bin/pytest tests/test_ladder_resume.py -q` exits 0:
REJECTED + matching fingerprint → skipped, sidecar bytes unchanged; REJECTED + changed
rung identity → reprocessed; UNVERIFIED + matching fingerprint → reprocessed.

### TICKET-E1 — mechanical binding evidence · TODO · depends-on: B1 · wave 7
**Problem:** Judges provably miss binding-only shifts under adverse conditions (kimi in
the bake-off; two frontier judges in GH-273); `bind()` is benchmark-only.
**Do:** At the gate, run `tables/binding.py:bind()` per witnessed table;
`no_known_contradiction == False` forces the FAIL path with contradictions attached as
findings evidence. Absence of coverage stays NEUTRAL — do not use
`structural_agreement` (most real tables are incompletely measurable) and do not force
FAIL when there are no native words.
**Files:** `src/socr/pipeline/orchestrator.py`, `tests/test_ladder_binding_evidence.py`
**Done when:** `~/venvs/socr/bin/pytest tests/test_ladder_binding_evidence.py -q` exits
0: a fixture whose value multiset is identical but whose row labels are shifted by one
(row-label contradiction from `bind()`) demotes with the flag on and ships with it off;
a no-native-words fixture is NOT demoted by binding alone.

### TICKET-H1 — end-to-end + committed fixture · TODO · depends-on: D1b, E1 · wave 8
**Problem:** No committed fixture reproduces the GH-273 shape (bake-off artifacts live
in scratch), and no test walks both terminals through every surface.
**Do:** A deterministic generated fixture (generator alongside
`tests/fixtures/table_repair/` precedent) with a binding-shift page and a clean page;
e2e tests (mocked rungs) driving one REJECTED and one UNVERIFIED document.
**Files:** `tests/fixtures/table_ladder/` (+ generator), `tests/test_ladder_e2e.py`
**Done when:** `~/venvs/socr/bin/pytest tests/test_ladder_e2e.py -q` exits 0: for each
terminal, flag on vs off in one process, asserting the DELTA at page status, document
status, `metadata.json` note, and CLI summary; full suite + format gate green.
