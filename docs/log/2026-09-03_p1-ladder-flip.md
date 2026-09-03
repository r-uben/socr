# 2026-09-03 — P1: the table-judge ladder flip, the ruled tiebreak chain, and the withhold path

Branch: `feat/p1-ladder-flip` (worktree `/Users/rubenffuertes/repos/tools/socr-flip`).
Left UNCOMMITTED for the conductor. Implements the three owner rulings in
`docs/log/2026-09-02_gh359-ladder-terminals-design.md`, section
"Owner rulings on the ladder flip — 2026-09-03".

Grounding canary:

```
$ PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-flip/src \
    ~/venvs/socr/bin/python -c "import socr; print(socr.__file__)"
/Users/rubenffuertes/repos/tools/socr-flip/src/socr/__init__.py
```

## What shipped, per ruling

### Q1 — two low-confidence PASSes are not a quorum

`run_table_ladder` no longer accepts a low PASS after a low PASS. That ending is
`TableLadderOutcome.UNVERIFIED` with a new typed field
`TableLadderResult.pending = TableLadderPending.TWO_LOW_PASS` and
`final_verdict=None`. Every other transition is unchanged, and a caller that
ignores `pending` sees UNVERIFIED — fail-closed by construction. The ladder
stays pure: it has no geometry oracle and no adjudicator, so it names the
situation and the gate decides.

The gate then runs the ruled chain, cheapest first:

1. **Native geometry.** `bind()` over the page's own word layer, classified by
   the new `socr.tables.binding.classify_binding_evidence` into `PASS` /
   `CONTRADICT` / `ABSTAIN`. `PASS` requires `structural_agreement` — rows AND
   columns fully checked with nothing disagreeing. A matching numeric multiset
   can never reach `PASS`; that is the GH-273 shape the ruling names.
   `CONTRADICT` is terminal on this path (the E1 clamp's own rule).
2. **Blind cell transcription.** The union of the two readers' `doubts` is
   resolved against the emitted table and handed to a third-vendor adjudicator
   with the crop and nothing else. All doubted cells agreeing clears the table;
   anything less does not.
3. Otherwise UNVERIFIED, with the P1 latch iff the adjudicator was unavailable.

### Q2 — a rejection neither guard clears is WITHHELD

A `REJECTED` ladder result runs the **same** chain in the same order, with the
doubted cells taken from the FAIL verdicts' `findings[].where`. The one
difference is deliberate and follows the ruling's wording ("binding check
passes … else the adjudicator"): on this path a geometry `CONTRADICT` is
non-clearing evidence, not a stop, because a contradiction can be the native
layer's fault (GH-334) while the readers are still wrong.

Cleared ⇒ the readers are overruled and the table ships. Not cleared ⇒
`TableLadderOutcome.WITHHELD`. A withheld page ships **no table bytes**: the
table region is replaced by the standard failed-table marker plus the page
image reference through the GH-520 regional splice, and prose outside the
region survives. When GH-520's four coverage conditions do not all hold, the
whole page floors — the same rule, unweakened, as the structure-class floor.

### Q3 — default ON, fail-closed

`PipelineConfig.table_judge_ladder` defaults to `True`. `--table-judge-ladder`
became the tri-state pair `--table-judge-ladder/--no-table-judge-ladder` with
`default=None`, so an omitted flag no longer clobbers a YAML value in either
direction. A new startup diagnostic fires when the ladder is on and neither
READER rung is reachable, naming the cause and the exact opt-out; it is
suppressed by `--quiet`, `--dry-run`, an explicit disable, and by
`strict_local` (which has its own, more specific line). `docs/ARCHITECTURE.md`
gained a section stating the default, what fail-closed means, and the four
terminals.

## The exact new values, and why each is new

| Value | Where | Why not a reuse |
|---|---|---|
| `FailureMode.TABLE_WITHHELD` (`"table_withheld"`) | `core/result.py` | `TABLE_REJECTED` ships the table text demoted under a warning. `TABLE_WITHHELD` ships none of it. Conflating them would make every historical rejected page look, on replay, like one whose content was withheld. |
| `TableLadderOutcome.WITHHELD` | `judge/table_ladder.py` | The page reducer needs a fourth precedence step: WITHHELD > REJECTED > UNVERIFIED > ACCEPTED. Produced by the gate only; the pure ladder can still only reach REJECTED. |
| `TABLE_LADDER_WITHHELD_KIND` (`"table_ladder_withheld"`) | `judge/table_verdict.py` | A consumer reading the audit log or `tables_trust.json` must be able to tell "shipped but distrusted" from "not shipped". Added to `TABLE_LADDER_EVENT_KINDS` — a deliberate widening of GH-359's three-terminal drift guard. |
| `REASON_VERIFIED_BY_GEOMETRY` (`"verified_by_geometry"`) | `data["reason"]` on `table_ladder_accepted` | An ordinary ladder acceptance carries no reason. These two name the cases where the READERS were overruled, which is a different fact and must be legible as one. |
| `REASON_VERIFIED_BY_BLIND_CELL_TRANSCRIPTION` | same | as above |
| `PagePrimaryReason.TABLE_JUDGE_WITHHELD` | `core/manifest.py` | Without it a whole-page withhold falls back to `SHIPPED_FAILURE_MARKER` ("socr cannot attribute this marker") and a REGIONAL withhold keeps selection's own base, publishing a page whose table was deliberately removed as an ordinary accepted model output. socr knows this cause exactly. |
| `table_withheld_pages` | assemble bucket | Mutually exclusive with `table_rejected_pages` / `table_unverified_pages` by construction: `_table_ladder_terminal` returns exactly one mode per page. |
| `TableLadderPending.TWO_LOW_PASS` | `judge/table_ladder.py` | The pure ladder cannot run the tiebreak; it names the situation for the gate. |
| `RUNG_KIND_CELL_ADJUDICATOR` (`"adjudicator"`) | `judge/table_verdict.py` | Latchable AND probeable in its own right, but deliberately NOT in `TABLE_JUDGE_RUNG_KINDS` — see role separation below. Renamed from `RUNG_KIND_KIMI` in cold review round 1: it names the ROLE, so a vendor swap does not rewrite every historical journal entry's meaning. |

## The adjudicator: identity, role, and unavailability

New module `src/socr/judge/table_rung_kimi.py`, built on `table_rung_gemini.py`'s
template: module-local subprocess seams (`_run_kimi_cli`, `_run_health_check`),
a `--version` health handshake, an identity tag, and the shared typed
availability classifiers.

- **Identity:** `cursor-agent --model kimi-k3-max`, recorded as
  `cursor-agent:kimi-k3-max`. Unlike rung 2's `agy`, the model IS pinned per
  call, so claiming it in the audit trail is honest.
- **Role separation (load-bearing):** it is an ADJUDICATOR, not reader rung 3.
  It never enters `run_table_ladder`, produces no PASS/FAIL, and its kind is
  kept out of `TABLE_JUDGE_RUNG_KINDS`. Kimi being reachable must never satisfy
  "someone can read this table", or it would suppress the no-reader startup
  diagnostic and stand in for a reader latch.
- **Blindness:** the public API takes a crop path and canonical cell references
  and nothing else. There is no parameter through which the emitted markdown, a
  reader verdict, a finding, or an extraction token could reach the model, and
  `tests/test_table_rung_kimi.py` pins that by introspecting the signature.

Unavailability rows, appended to the prep log's table:

| Outcome | `ok` | `unavailable` | `refusal` |
|---|---|---|---|
| binary absent (`FileNotFoundError`) | False | True | False |
| timeout | False | True | False |
| spawn errno ENOEXEC / EACCES / EPERM | False | True | False |
| spawn errno E2BIG / EPIPE / other | False | False | False |
| nonzero exit, CLI healthy | False | False | False |
| nonzero exit, CLI unreachable | False | True | False |
| recognised refusal on either stream | False | True | True |
| malformed body, or a cell key set that is not exactly what was asked | False | False | False |

The last row is the one worth stating: a partial answer is a DEFECT, never a
partial clearance. Clearing a table on fewer cells than the guard asked about
would answer a question nobody posed.

## Metering

Every executed adjudicator call creates exactly one `EngineResult` (engine
`table_blind_cell_adjudicator`, `model_version` the rung identity) and is passed
once to `DocumentState.record_engine_run(..., page_nums=[page_num])` — which is
itself both the sole journal author and the sole page-spend charger, so it is
NOT also passed to `_add_page_cost`; doing both would double the page's spend
and make a resumed run's arithmetic disagree with a live one. The cost is
always a KNOWN float, including a known zero: `None` means "unmetered" and
would poison every later budget decision for the document.

Before the call, in this order: the per-page cap must cover the page's CURRENT
spend plus this call's rate, and the remaining document budget must cover the
call — with an unknown prior total treated as ZERO remaining, the same
fail-closed rule the equation lane uses. A budget refusal makes no call and
never sets the latch: it reproduces identically on every rerun, so latching it
would make the document permanently unskippable and change nothing.

`RootIndex.record` remains the sole author of `<root>/metadata.json`.

## Resume semantics for WITHHELD

`_load_terminal_page` treats `TABLE_WITHHELD` as a CONTENT terminal, exactly as
it treats D1b's `TABLE_REJECTED`: skip-and-keep under a matching fingerprint and
checksum, forfeited by `table_ladder_incomplete` and by a shipped
structure-class floor. Two things needed widening beyond a label change:

1. **The marker refusal.** A withheld page's fragment may BE a whole-page
   failure marker — that is what withholding produces when GH-520's coverage
   proof does not hold. The unconditional `is_page_failed_marker` rejection
   would have defeated the exception silently and re-judged the page forever.
2. **The completeness backfill.** `_backfill_missing_table_ladder_terminals`
   enumerates markdown blocks from the SHIPPED text. On a withheld page those
   bytes are gone by design, so it would have found no tables, concluded every
   table was witnessed, and lost the completeness signal for a SIBLING table
   nobody looked at. It now enumerates from what the page EMITTED when the
   disposition is WITHHELD, and `table_ladder_withheld` counts as an observed
   terminal.

The P1 latch still wins over the exception: a withheld page whose sidecar
carries `table_judge_retry_pending` with a rung reachable now is reprocessed.

**One consequence worth stating plainly.** A page whose only table is withheld
whole ships no content, so the DOCUMENT is ERROR under the pre-existing
no-content rule, and the document-level resume gate never skips an error
document. The page itself is still restored by the per-page ledger, so no rung
is called again — but the document is re-opened on every resume. That is a
convergence cost the ruling accepts, not a latch leak.
`tests/test_p1_ladder_retry_latch.py::test_content_only_rejected_does_not_reopen_when_reachability_changes`
was rewritten to pin the rung-call count rather than the document status for
exactly this reason.

## Test audit — what moved and why

Every failure under the new default was triaged as one of two things.

**(a) A real behaviour change this ticket intends.** All of these were updated
deliberately with a comment naming the ruling, never weakened:

- `tests/test_table_ladder.py` — the two-low quorum test inverted (Q1).
- `tests/test_table_judge_gate.py` — six tests: reader rejections now reach
  `TABLE_WITHHELD` and the `table_ladder_withheld` event; the two-low control
  now ends UNVERIFIED; two `process()` difference tests now assert ERROR with
  the withhold named, and one of them additionally pins `result_on.status !=
  result_off.status` so the flag difference is still a difference.
- `tests/test_gh359_ladder_terminals.py` — the `_rejected` predicate widened to
  both labels (these tests pin WHERE the ladder ends, not which label the
  ending carries); ruling 7's "keeps the markdown" half is explicitly
  superseded by Q2 and the surviving property (not a figure reroute) is pinned
  instead; the two GH-381 completeness tests take the new disposition.
- `tests/test_ladder_e2e.py`, `tests/test_ladder_binding_evidence.py`,
  `tests/test_gh373_degraded_witness.py` — same label move.
- `tests/test_p6_disposition_contract.py` — the brittle `< 15` member count
  replaced by the semantic invariant (fewer normalized causes than
  `SelectionProvenance` rows) and the new reason added to the required set.
  This is an extension: the guard still forbids one member per selector row.
- `tests/test_gh95_tables_trust.py`, `tests/test_table_verdict.py` — the
  three-terminal drift guards widened to four, deliberately, with the ruling
  cited. That is what those guards are for: they catch a terminal added by
  accident, and this one is added by ruling.
- `tests/conftest.py`, `tests/test_p6_orthogonal_bucket_helper.py` — the
  pre-refactor bucket oracles gained `table_withheld_pages`. A terminal that
  did not exist when the oracle was written is a deliberate vocabulary
  extension, not P6 drift.
- `tests/test_ladder_config.py` — `TestDefaults`/`TestCLIFlag`'s default-off
  pins superseded by `TestP1DefaultFlip`; the "unset flag does not clobber
  YAML" test now passes `None` rather than `False`, because under the tri-state
  `False` is an explicit `--no-table-judge-ladder` and MUST win.
- `tests/test_cli_flag_agentic_status_gh142.py` — the three new config fields
  classified as inert-but-fingerprinted, same row and same reason as the rung
  fields.

**(b) Silently relied on the default.** None found: the golden-flag audit guard
already on main (`tests/test_p1_golden_flag_pinned.py`) forces every golden /
byte-identity / replay module to pin `table_judge_ladder` explicitly, and that
guard passed unchanged throughout. `tests/test_p6_stage_ab_difference.py`'s
stored byte-identity baselines were not touched and still pass, which is the
proof that no golden moved.

**Hermeticity.** The flip exposed a real trap: with the ladder on by default,
any test that builds a pipeline without overriding `_build_table_judge_rungs`
constructs REAL rungs. On this machine `ollama` is up and both `agy` and
`cursor-agent` are on PATH, so the suite began making live model calls — slow,
quota-spending, and machine-dependent in exactly the #253/#257 way, since CI
has none of the three. `tests/conftest.py` gained an autouse fixture that pins
the suite to CI's environment (no daemon, no binaries) by stubbing the modules'
own subprocess/HTTP seams. It weakens no assertion and does not touch the
ladder flag; a test that wants a rung still injects one.

**Follow-up (this session): the autouse fixture blinded the two tests it was
never meant to touch.** `tests/test_p1_ladder_retry_latch.py` has two tests
built to drive the REAL gemini/`agy` reachability seam
(`test_a_broken_but_installed_cli_does_not_read_as_recovered` and
`test_a_healthy_cli_appearing_does_reopen_through_the_real_seam`) against a
controlled local stub binary (`_health_cli`), deliberately leaving
`_table_judge_rung_available_now` unpatched. The new autouse fixture
monkeypatches the SHARED `shutil` module's `which` attribute (not just the
name inside `table_rung_gemini`'s namespace) to always return `None` and
forces `_run_health_check` to raise, so the stub's exit code was never
consulted — the "recovered" test could not pass no matter what the seam did
(4 failures surfaced from a full-suite run: this test plus three others whose
failures traced back to it once isolated). Fix: `_real_gemini_reachability_seam()`,
a local context manager that captures the real `shutil.which` and
`table_rung_gemini._run_health_check` at collection time (before any
monkeypatching) and restores them for the duration of just these two tests,
leaving every other test in the file under the hermetic default. Not a
default-reliance pin — the ladder flag was already pinned `True` explicitly in
this file's `_make_config`; this was test infrastructure fighting itself.

## Corrections made to the tests-1 acceptance tests

Treated as acceptance tests, corrected only where they contradicted a ruling or
could not have passed as written. Each correction carries a comment in place:

- `tests/test_p1_tiebreak_and_withhold.py`, `tests/test_p1_false_reject_fixtures.py`
  pointed the gate at a PDF path that was never created, so witness preparation
  failed for every fixture and the ladder never ran — none of them could reach
  the guard chain they exist to pin. They now build the same ruled PDF
  `test_table_judge_gate.py` uses.
- The latch attribute those files read (`table_judge_rung_unavailable`) does not
  exist; the real one is `table_judge_retry_pending`. The tests-1 report
  anticipated this and asked the implementer to align, which is what was done.
- `tests/test_p1_withheld_resume.py` read `<out>/<stem>.md` twice and compared
  it to itself — a byte-identity assertion that could not fail, at a path the
  writer does not use. It now compares the page FRAGMENT across two runs, which
  is the surface that carries a withheld page's shipped bytes (a fully withheld
  single-table document writes no assembled body at all).
- `tests/test_ladder_config.py` patched
  `socr.pipeline.orchestrator.gemini_rung_reachable`, which does not exist —
  the orchestrator imports it as `table_judge_gemini_rung_reachable`, so the
  patch would have silently probed the real binary. Its help-text assertion
  searched the WHOLE `--help` output for "default off" and failed on unrelated,
  legitimately default-off flags; it now reads the ladder flag's own help block.
- `tests/test_table_cell_guard.py` passes `crop_path=None` throughout, so the
  guard service does not refuse on a missing crop; a missing crop reaches the
  adjudicator and fails closed there as a defect.

## Assumptions that survived

1. **The `kimi-k3-cursor` roster entry named in the spec does not exist.** The
   build uses `cursor-agent --model kimi-k3-max`, which `cursor-agent models`
   does list, with the binary and model as config fields.
2. **A low-confidence PASS carried no findings**, so Q1 step (b) had no cell
   list to work from. Closed by the PASS-legal, value-free `doubts` field added
   to the verdict schema and prompt (task t1). This moves the table-judge
   prompt digest, so the corpus reprocesses — already expected at the flip.
3. **Structural findings cannot be cleared by cell transcription.**
   `NOT_A_TABLE`, and a `HEADER_MANGLED`/`STRUCTURE_MERGED` that names a region
   rather than a cell, contribute no canonical reference, so the guard abstains
   and the table stays UNVERIFIED (Q1) or becomes WITHHELD (Q2). Fail-closed on
   purpose.
4. **No `table_judge_tiebreak` / `table_judge_withhold_rejected` feature
   switches** were added, per the critique's explicit constraint: they would be
   unruled product states existing only to serve tests. The difference tests
   vary the guard's own evidence step instead, which is the thing under test.
5. The pre-existing GH-367 `_transcribe_cell_token` calls remain unmetered.
   Out of scope here (the spec's cost clause is about the new adjudicator);
   recorded, not silently bundled.

## Not done, and why

**The `cursor-agent` image-attachment live smoke.** `cursor-agent --help`
documents `--print`, `--mode ask`, `--model`, `--output-format`, `--workspace`
and `--trust`, but documents no image-attachment flag. The `@<path>` reference
this module uses is the mechanism proven for `agy` by the 2026-08-30 smoke and
has NOT been proven live for `cursor-agent`. A text-only handoff would return
schema-valid JSON that saw no image — the worst failure shape. Every terminal
this module can produce is fail-closed, so an unproven transport costs
clearances and never false ones, but **the transport is not live-validated and
this log does not claim it is.** It is the first thing the corpus run must
settle.

## What the corpus run must measure

Run the corpus with the flag on and report:

1. The **rate of two-low endings** — how often Q1's chain fires at all.
2. The **geometry-clear rate.** GH-330 measured the binder as fully-checking
   0/15 real pages, and `structural_agreement` requires `fully_checked`, so
   guard (a) is expected to abstain on nearly everything. If it does, the
   flip's practical behaviour rests entirely on guard (b) and the fail-closed
   terminal, and that is the number that says so.
3. The **blind-transcription clear rate**, split by Q1 and Q2 paths — and,
   before any of it is believed, whether `cursor-agent` actually showed the
   model the crop (item above).
4. The **withhold rate**, and how much prose the regional splice keeps versus
   how often the whole page floors.
5. The **adjudicator's unavailable / refusal rate** and its quota behaviour
   over a long document, plus per-table ladder latency against the 600 s
   timeout.
6. The **document-status distribution** under the new default, especially how
   many documents land ERROR because a single-table page withheld whole.


---

# Cold review round 1

Nine findings, all reproduced before any fix, each with a failing test first.
Two of them changed the build's shape rather than its details: the adjudicator's
transport was replaced outright (finding 4) and the Q2 terminal chain was
rewritten to GH-575 literally (finding 1). The rest fall out of, or ride on,
those two.

## The live image-transport proof (finding 4)

Settled by measurement, not argument. One synthetic crop, one cell: the value
printed next to `Gamma` is **0.058**. Three paths were asked to read it.

| path | model | answered |
| --- | --- | --- |
| `cursor-agent -p` (crop named as `@<path>` in the prompt) | `kimi-k3-max` | **0.92** |
| ollama `/api/generate` with `images` | `kimi-k2.6:cloud` | **0.058** |
| ollama `/api/generate` with `images` | `glm-5.3-flash:cloud` | **0.058** |

`0.92` is not a misreading. It is the answer of a model that never received an
image and produced a plausible number anyway — which is precisely the failure
this rung must not have, because a guessable token (`0`, `N/A`, a round
number) can normalize equal to the extraction and clear a table nobody looked
at. The prior build's claim that an unproven transport "costs clearances,
never false ones" was wrong for exactly that reason.

**Ruling implemented:** the `cursor-agent` rung is DELETED
(`src/socr/judge/table_rung_kimi.py` and `tests/test_table_rung_kimi.py`).
The blind-cell adjudicator is now `kimi-k2.6:cloud` reached through the
existing ollama rung builder in `src/socr/judge/table_rung_ollama.py`.

Independence now lives where it actually belongs — in the MODEL and the
VENDOR, not the transport. Reader rung 1 is Zhipu (`glm-5.3-flash:cloud`),
reader rung 2 is Google (`agy`), the adjudicator is Moonshot
(`kimi-k2.6:cloud`). Sharing rung 1's `/api/chat` path buys the health probe,
the typed unavailability classification, the refusal signal, the identity tag
and the metering for free, and it is the path proven to carry pixels.

Config: `table_judge_kimi_binary` / `table_judge_kimi_model` /
`table_judge_kimi_cost_per_call_usd` are replaced by
`table_judge_adjudicator_model` (default the named constant
`TABLE_JUDGE_ADJUDICATOR_MODEL_DEFAULT = "kimi-k2.6:cloud"`),
`table_judge_adjudicator_host` and
`table_judge_adjudicator_cost_per_call_usd`. All three stay
inert-but-fingerprinted when the ladder flag is off.

## Per finding

| # | reproduced | what changed |
| --- | --- | --- |
| 1 | yes | GH-575 implemented literally. `continue_on_contradict` is GONE from `evaluate_cell_guard`: an ACTIVE binding contradiction is terminal on BOTH paths and the adjudicator is never asked. A new `GuardDisposition.MISMATCHED` separates "a blind reader looked and disagreed" from every other non-clearing cause, and it is the ONLY route to `WITHHELD` — and only when the readers had already rejected the table. Empty/unresolvable doubt set, no adjudicator, outage, refusal, malformed answer, budget refusal and internal error all end `UNVERIFIED`; only the outage and refusal latch. Every cell of that table is pinned for both paths in `tests/test_p1_tiebreak_and_withhold.py`. |
| 2 | yes | `TABLE_JUDGE_PROBEABLE_KINDS` (readers **plus** the adjudicator) is what the persisted latch is filtered through; `TABLE_JUDGE_RUNG_KINDS` stays reader-only for "can anyone READ this table". `_probe_table_judge_rung_kind` gained an adjudicator branch asking the daemon about the ADJUDICATOR's model. `tests/test_p1_withheld_resume.py` no longer replaces `_table_judge_rung_available_now`; it controls the three leaf reachability functions and pins both directions (adjudicator down + reader up ⇒ skip; adjudicator back + readers down ⇒ reopen). Reverting the one-line filter fails 3 of its 5 tests. |
| 3 | yes | The adjudicator goes through the same per-run breaker as the readers: a `refusal` adds `RUNG_KIND_CELL_ADJUDICATOR` to `_table_rung_refused_this_run`, and every later table in the run (and later document in the batch) is passed no adjudicator at all while still latching. Pinned: one refusal ⇒ exactly one call across three subsequent tables, plus a control proving a NON-refusal outage does not trip it. |
| 4 | yes | See above. |
| 5 | yes | The metered `EngineResult.model_version` and the synthesized error `rung` both read the callable's advertised `rung_id`, never a module constant. The test doubles advertise a NON-default model precisely so a default would be visibly wrong. |
| 6 | yes | `tests/test_p1_false_reject_fixtures.py` rebuilt: three fixtures BUILT to their named shapes (a spanning `Panel A / Panel B` header over two column groups with a notes row; a page rotated 90°; a 12-row near-identical dense grid), each asserted by a grounding canary, each driven through the REAL binding check with only `table_rung_ollama._post_chat` mocked. The three cover all three geometry branches — dense PASSes (a false reject overruled for free), spanning CONTRADICTs (terminal `UNVERIFIED`, adjudicator never asked), rotated ABSTAINs (the blind reader decides, and only there can bytes be withheld). Provider-state difference now runs `rungs=[]` versus injected rungs, which is the parameter `_run_table_judge_gate` actually consults. |
| 7 | yes | The list is below. |
| 8 | yes | New `_latched_rung_kinds(state, pages)` reads `table_judge_retry_rungs`, and both the document note and the CLI line name the kind(s) recorded. A page latched on `gemini` and a page latched on `adjudicator` now produce different text; before, both said "an adjudicator rung was unavailable". |
| 9 | yes | `PipelineConfig.validate_costs()` rejects a negative or non-finite adjudicator rate, called from `__post_init__` AND at the end of `from_file` (which assigns onto an already-constructed object and would otherwise never re-enter `__post_init__`). |

Also fixed: the trailing blank line at EOF in `src/socr/prompts/table_judge.md`.
`git diff --check` is clean.

## Finding 7: tests pinned to the OLD default in this diff

This corrects "None found" in the section above, which was contradicted by the
working tree. These are tests that would otherwise have depended on the ladder
default, and which this diff pins to `table_judge_ladder=False` explicitly:

- `tests/test_chart_lane.py::_run_chart_detection_case` (used by
  `test_gh318_chart_detection_failure_is_visible_at_page_and_document_status`)
  and `test_assemble_reports_non_success_for_render_failed_chart_page`.
- `tests/test_gh262_d3_marker_over_cached_grid.py::test_document_audit_metadata_and_trust_sidecar_surface_the_superseded_floor`.
- The five tests using `tests/test_gh539_provisional_fragment_matches_sidecar.py::_flush`.
- The two tests using `tests/test_native_only_table_status_gh211.py::_run_native_only`.
- The tests using `tests/test_tr3_d3_floor.py::TestAuditEventDistinctness::_run_assemble`
  and `TestDocumentStatusDemotion::_run_and_get_result`.

This is a DIFFERENT list from the 13-module golden/byte-identity/replay
inventory that `tests/test_p1_golden_flag_pinned.py` enforces. That guard was
already on main, it is unchanged by this diff, and it passed throughout. The
list above is the set of ordinary tests that were silently relying on the old
default and are now explicit.

## Label moves the GH-575 chain forces, and why they are not weakenings

A reader rejection no longer becomes `TABLE_WITHHELD` on its own. It needs a
blind reader to have looked and disagreed. Several existing files therefore
moved, and each moved in the direction the ruling dictates:

- `tests/test_gh359_ladder_terminals.py`, `tests/test_gh373_degraded_witness.py`
  — the FAIL verdicts' `where` was `"cell"` / `"header"` / `"r1c1"`, none of
  which parse as a canonical cell reference, so the readers localized nothing
  and every rejection would have ended on the no-doubt-set path. They now name
  a real cell and inject a mismatching adjudicator, which keeps every GH-359
  ruling about WHERE THE READERS end the ladder. Their `_process` helpers also
  isolate the binding EVIDENCE, not just the clamp: GH-575 made the evidence a
  terminal in its own right, so isolating the clamp alone no longer isolates
  `bind()` from the readers' verdict.
- `tests/test_ladder_binding_evidence.py`, `tests/test_ladder_e2e.py` — these
  drive a genuine row-label shift, where native geometry ACTIVELY contradicts.
  Under GH-575 that is terminal `UNVERIFIED` and the adjudicator is never
  asked, so the pages ship labelled rather than withheld. GH-359 ruling 5's
  "the clamp never claims a rejected page" is superseded for the contradiction
  case, in one direction only: mechanical evidence still cannot turn anything
  into a content REJECTION.
- `tests/test_table_judge_gate.py` — `_make_pipeline` now pins the binding
  evidence to ABSTAIN and injects a mismatching adjudicator by default, with
  `adjudicator=None` available for the "nothing configured" path. Incidentally
  this took the file from 69 s to 5 s: the real geometry pass on every fixture
  was the cost.

## Verification

- Full suite: `3938 passed, 4 xfailed` — exit 0.
- `uvx ruff@0.16.0 format --check .`: `551 files already formatted` — exit 0.
- `git diff --check`: clean.

---

# Cold review round 2

Round 1's nine findings: eight closed, one (the false-reject fixtures) still
open. Three new: two blocking, one should-fix. All four reproduced before any
fix.

## N1 — one coordinate contract, defined once (BLOCKING)

The grammar was prose, written out twice, and the two copies disagreed. The
reader prompt and `resolve_cell_refs` both counted the row-label (stub) column
as body column 1; the blind-transcription prompt told the model to start
counting to the RIGHT of the stub. For

```
| Region | Alpha | Beta |
| North  |    11 |   12 |
```

the gate compared `R1C1` against `North` while the model it asked had been
instructed to read `11`. Every body coordinate was shifted by one physical
column between the reader that raised a doubt and the blind reader asked to
check it — which can withhold a correct rejected table, or clear a wrong one
whenever the adjacent tokens happen to normalize equal.

Ruled and implemented: `R1C1` is the stub cell, as the reader prompt and the
resolver already said. The grammar now lives in ONE file,
`src/socr/prompts/cell_ref_grammar.md`, loaded by
`table_verdict.load_cell_ref_grammar()` and spliced into BOTH prompts through
a `{{CELL_REF_GRAMMAR}}` placeholder. It carries its own worked example, and
`tests/test_cell_ref_grammar.py::TestOneCoordinateContract` closes the loop:
the fragment must state that `R1C1` is `North`, and `resolve_cell_refs` must
return exactly that for the same table. Body and header refs remain
deliberately asymmetric — a body `col` counts the stub, a header `col` does
not — because that is the existing resolver contract; the fragment now says so
in one place instead of nowhere.

Both prompt digests hash the fragment, so a wording-only edit to the shared
grammar reprocesses, as it must.

## N2 — "unreadable" is now its own wire state (BLOCKING)

Prompt rules 2 and 3 both said "return the empty string", so *I looked and the
cell is blank* and *I could not read this cell at all* arrived as the same
token, and the guard had no bit to tell them apart. Two consequences, both
violating the ruling's own bar:

- `""` against a non-empty extraction was `MISMATCHED`, which WITHHOLDS a
  rejected table's bytes — on the strength of a reading nobody made.
- `""` against an empty extraction was agreement, and could CLEAR a rejected
  `MISSING_VALUE` case with no visual evidence behind the clearance.

The schema now has three states. A JSON string is a reading; the EMPTY string
is still a reading ("I looked, it is blank"); JSON `null` is the typed
non-reading. `parse_blind_cell_output` returns `(tokens, unreadable)`,
`BlindCellResult` carries `unreadable`, and the guard checks it FIRST, before
any comparison: an unreadable cell ends the chain `NOT_CLEARED`, never
`MISMATCHED` and never a clearance, and never latches (the adjudicator
answered; it is not an outage). `null` remains subject to the same strictness
as any other value — it must be a requested key, and a non-string non-null
value is still a defect.

Both positive controls are pinned: a visibly empty cell still clears an empty
extraction, and still mismatches a printed one.

## N3 — a tripped breaker latches only the pages it actually cost (SHOULD-FIX)

The caller added `adjudicator` to `guard_unavailable_kinds` before asking
whether this table needed an adjudicator at all. Geometry PASS and CONTRADICT
return without a call, and so does an empty or unresolvable doubt set — but
the page was latched anyway, promising a retry for a call that was never going
to happen, and one such table is enough to reopen the whole page on every
later run.

The suppression is now passed DOWN (`adjudicator_suppressed=True`) and
recorded inside the guard at the single point where the chain has established
that this table needed the adjudicator: an ABSTAIN verdict with a resolvable,
non-empty doubt set. Reverting that one line fails 6 of the new pins.

## 6 — the false-reject fixtures, closed

Round 2 was right that the round-1 rebuild still did not establish the claim.
Two defects, both fixed:

- **The spanning fixture's rejection was CORRECT, not false.** The page drew
  `Panel A` / `Panel B`, and `SPANNING_MD` omitted that spanning header
  entirely, which makes `HEADER_MANGLED` the right answer and manufactured the
  binding contradiction the test then pinned. The markdown now carries the
  spanning row as a real second header row (`parse_grid` supports multi-row
  headers), and the period sub-labels are non-numeric — with `2019`/`2020` the
  binder reads the header cells as unbindable numeric data and contradicts on
  a shape that has nothing to do with spanning.
- **The mocked blind answer is no longer written by hand.** `truthful_tokens()`
  derives it from `resolve_cell_refs` on the emitted markdown, so a mock can
  never again be green on an answer the shipped prompt told the model not to
  give.

Two new canaries make the "false" in false-reject load-bearing:
`TestTheRejectionsAreFalse` asserts every emitted cell appears on the page, and
specifically that each flagged cell holds a token the page really shows. The
old `SPANNING_MD` fails both.

The three fixtures now cover both routes by which the chain clears a false
reject, and both causes of an abstaining geometry:

| fixture | geometry | how the false reject is cleared |
| --- | --- | --- |
| spanning header + notes row | PASS | free, by geometry; no call is made |
| rotated crop | ABSTAIN (rotation) | by the blind reader's agreement |
| dense scan | ABSTAIN (no text layer) | by the blind reader's agreement |

The dense fixture is now a genuine SCAN — the text grid rasterised to pixels —
which is both the ruling's own reason for geometry to abstain and the
faithfulness proof for its markdown, since the scan is a picture of the very
page that markdown was checked against. Each shape additionally pins that only
an actual blind disagreement withholds, and that an unreadable answer never
does.

## Verification

- Full suite: `3978 passed, 4 xfailed` — exit 0.
- `uvx ruff@0.16.0 format --check .`: `552 files already formatted` — exit 0.
- `git diff --check`: clean.
- Reversion checks: the N3 pin fails 6/38 with the latch moved back to the call
  site; the round-1 finding-2 pin still fails 3/5 with the kind filter reverted.

---

# Cold review round 3

Round 2's N2 and N3 closed. Two blocking findings remained or were newly
opened, plus one should-fix. All three reproduced before any fix.

## NEW A — the coordinate contract is now PHYSICAL and stub-agnostic (BLOCKING)

Round 2 made the grammar a single spliced fragment, which fixed the wording
drift. It did not fix the rule, which assumed every table has a row-label
column: body coordinates counted that column, header coordinates skipped it,
and `resolve_cell_refs` implemented exactly that asymmetry
(`header_row[col]` against `body_row[col - 1]`).

Nothing in socr detects a row-label column. `parse_grid` accepts any
equal-width table of two or more columns, and a stubless table is explicitly
supported and pinned in `tests/test_binding.py`. So on

```
| Alpha | Beta |
| --- | --- |
| 11 | 12 |
```

both prompts described a row label that does not exist, and `H1C1` resolved to
`Beta` while a model reading the same rule would report `Alpha`. A reader
finding on `H1C1` could then compare a *correct* blind reading against the
wrong extraction cell and withhold a correct table. The off-by-one also made
the rightmost header cell unaddressable on every table.

Ruled and implemented: a reference is physical. `C<k>` is the k-th column
counting from the leftmost, for header and body rows alike, whatever that
column holds. `R<n>` is the n-th body row, `H<n>` the n-th header row.
`resolve_cell_refs` maps `(row, k)` to physical column `k - 1` in both
families, and the words "stub" and "row label" appear nowhere in the grammar,
the resolver or either prompt — a test enforces their absence, because a rule
that names a stub is a rule that is wrong on every table without one.

The fragment now carries two worked examples, a stubbed table and a stubless
one, and `TestOneCoordinateContract` checks the resolver against the rule over
three shapes: stubbed, stubless, and a multi-row spanning header.

This is a second move of the reader prompt's grammar block, so the table-judge
prompt digest moves again and the corpus reprocesses. Pages judged under
either earlier build carry a different coordinate convention and must not be
trusted on resume.

## NEW B / item 6 — the PDF builder is the fixture oracle (BLOCKING)

The round-2 fixtures derived the "visual truth" the mocked blind reader
answered from out of the markdown under test, and checked faithfulness only by
asking whether each token appeared *somewhere* in the page text. Round 3
swapped two flagged values in the dense extraction — leaving the page-wide
token multiset unchanged — and all 21 tests stayed green while the chain
published a wrong table as `verified_by_blind_cell_transcription`.

`_grid_pdf` now returns, alongside the PDF, the map of cells it physically
drew, keyed by canonical reference and recorded at draw time from the tokens
actually handed to `insert_text` and the columns they actually occupy.
Everything follows from that map:

- the mocked blind reader answers ONLY from it, so it reports the page rather
  than the markdown;
- "the rejection is false" is an equality, cell for cell, between the
  extraction and the drawn map, not a membership test;
- and the mutation is pinned: `test_a_two_cell_swap_in_the_extraction_is_not_cleared`
  exchanges two body values and requires that the chain does not clear the
  table. Rewiring the reader back to the markdown fails that test on both
  blind-reader shapes.

The shapes are also rendered as claimed. The spanning fixture draws ONE merged
header cell over two columns — a single centred token with the vertical rules
broken around it, asserted by counting vertical segments per row band — rather
than the same word twice in two ordinary cells. The rotated fixture applies
rotation exactly once, to the page, so what renders is the described table
turned on its side, and its drawn map stays in the pre-rotation frame the
extraction uses.

The three fixtures still cover both clearing routes and both causes of an
abstaining geometry: spanning PASSes, rotated abstains through rotation, dense
abstains as a scan.

## NEW C — a malformed response body never escapes as an exception (SHOULD-FIX)

`_post_chat` returned `message.content` untyped, and both callers reach for
string methods on it, so `{"message": {"content": 1}}` escaped as an
`AttributeError` out of two functions that promise a typed failure result. The
body is type-checked at the seam now — the response, the `message` object and
`content` — and the wrong shape is raised as a `ValueError`, which both the
reader rung and the adjudicator classify as a DEFECT. It reproduces on the
next call, so it never latches. Five malformed bodies are pinned for both
callers, with a well-formed control.

## Verification

- Full suite: `3997 passed, 4 xfailed` — exit 0.
- `uvx ruff@0.16.0 format --check .`: `552 files already formatted` — exit 0.
- `git diff --check`: clean.
- Counterfactual checks: rewiring the fixture oracle back to the markdown fails
  the two-cell-swap test on both blind-reader shapes; the round-2 and round-1
  reversion checks still bite as recorded above.

---

# Cold review round 4

Round 3's three findings closed; the reviewer rendered all three fixture PDFs
and confirmed the shapes. One new blocking finding and two low ones. All three
reproduced before any fix.

## NEW 1 — the blind prompt now carries no cell contents at all (BLOCKING)

The shared coordinate fragment ended with a worked example that stated literal
answers: ``R1C2 is 11``, ``R1C3 is 12``. `build_blind_cell_prompt` spliced that
fragment in, so a table whose extraction claims `R1C2 = 11` sent the
adjudicator that exact coordinate AND that exact token before it looked at
anything. The reviewer built a raster fixture whose drawn `R1C2` is `99` while
the markdown wrongly said `11`, had a mock ignore the pixels and copy the
example, and the real gate cleared the wrong table as
`verified_by_blind_cell_transcription`.

The round-3 no-leak test could not catch this, because it deliberately removed
the whole fragment before checking, on the premise that fixed policy text
corroborates nothing. That premise fails the moment the policy text names a
coordinate and a value that a real table happens to share.

Ruled and implemented: the rule and the examples are now two files.

| file | contents | goes to |
| --- | --- | --- |
| `prompts/cell_ref_grammar.md` | the coordinate RULE, value-free | both prompts |
| `prompts/cell_ref_examples.md` | the worked examples | the reader prompt only |

The rule contains no cell contents and, structurally, **no digits at all** —
pinned by a test, because every plausible numeric cell value is a digit string,
so a digit-free rule cannot name one by accident. The examples are reader-only
and use no plausible values either: every cell of every example table holds its
own reference (`| H1C1 | H1C2 |` over `| R1C1 | R1C2 |`), which also makes the
contract test exact — resolving every reference in every example must return
the reference itself, with no second copy of expected values to drift.

The no-leak test now checks the WHOLE blind prompt as sent, with nothing
removed: no token the page holds, no line of the emitted markdown, and no
reader finding detail may appear in it. Matching is whole-token, so a
two-letter heading is not reported as leaked because a longer word contains it.

The reviewer's reproducer is a standing test.
`_PromptCopyingReader` answers each requested coordinate with whatever the
prompt states it holds, and reports a non-reading when the prompt states
nothing. Against a corrupted extraction it must not clear the table. Appending
a single value-bearing example line to the rule file flips that test to a
clearance, which is the check that it bites.

Both prompt digests already covered the shared fragment; the reader digest now
covers the examples file too.

## NEW 2 — the banned-vocabulary claim now matches its coverage (LOW)

Round 3 asserted that "stub" and "row label" appear nowhere, but tested only
the loaded fragment while the resolver's own comments still used the terms.
The test now checks five surfaces: the rule file, the examples file, the
resolver's source, and BOTH generated prompts as sent. The resolver's comments
are rewritten accordingly. Runtime semantics were already physical and correct;
this closes a claim/coverage gap, not a defect.

## NEW 3 — the merged-header canary is positional (LOW)

It compared vertical-segment COUNTS between the spanning row and the row below,
which a builder omitting some other edge — the table's right border, say —
would also satisfy. It now compares POSITIONS: the sub-header row carries every
column boundary, and the spanning row must be missing exactly two of them, the
interior rule inside `Panel A` and the interior rule inside `Panel B`.

## Verification

- Full suite: `4004 passed, 4 xfailed` — exit 0.
- `uvx ruff@0.16.0 format --check .`: `553 files already formatted` — exit 0.
- `git diff --check`: clean.
- Counterfactual: restoring a value-bearing worked example to the shared rule
  fails six tests across the grammar contract and the fixture no-leak guard,
  and makes the prompt-copying reader clear a corrupted table.

---

# Cold review round 5

Round 4's NEW 2 and NEW 3 closed; the reviewer also verified independently that
both prompt digests moved and that a page carrying an earlier fingerprint is
refused on resume. NEW 1 stayed open through one residual path, reproduced
before any fix.

## NEW 1 — the blind prompt now binds no coordinate to any value (BLOCKING)

Round 4 removed the value-bearing worked example from the shared coordinate
rule. It did not remove the other answer key, which was in the blind prompt's
own template all along — the output-format example:

```json
{"R1C2": "1.24", "R2C3": "", "R3C1": null}
```

Three real-looking coordinates, each bound to a value, one of them numeric. The
reviewer reproduced the same publication failure through the real gate: a
raster table whose drawn `R1C2` reads `99`, an extraction wrongly claiming
`1.24`, and a reader that ignored the image and copied the JSON. The table
cleared as `verified_by_blind_cell_transcription`.

The round-4 guards could not see it. They were written against the syntax of
the round-4 leak — backtick prose shaped like `` `R1C2` is `11` `` — and the
fixture-level check was token-specific, so it only fired if one of these three
fixtures happened to contain the leaked value. None does.

**The prompt.** The format example is now
`{"<ref>": "<text as printed>", "<ref>": "", "<ref>": null}`, with a sentence
saying the three values stand for the three cases in the order the rules give
them: a cell read, a cell seen to be blank, a cell not read. The rule list is
bulleted rather than numbered, so no digit remains anywhere. `""` is still the
blank-cell protocol and `null` still the non-reading — what may never happen is
binding either, or anything else, to a reference.

**The guards are syntax-agnostic now**, which is the actual lesson of two
rounds finding the same defect in two spellings. They state properties no
spelling can satisfy rather than hunting for a spelling:

- every coordinate appearing anywhere in the blind prompt is one the caller
  actually asked about;
- with those coordinates stripped out, NO DIGIT remains — so no numeric value
  can be in the prompt by any means, in prose, in JSON, or in a form nobody has
  thought of yet;
- no JSON key shaped like a coordinate is bound to any value;
- and the round-4 prose check stays, as a named regression.

The fixture-level whole-prompt check gained the same digit-strip property on
top of its token check, so it no longer depends on a fixture happening to
contain the leaked value.

**Both reproducers are standing tests.** `_PromptCopyingReader` now reads both
representations, and `TestTheStandingPromptEchoReproducers` runs the scanned
`R1C2 = 99` fixture against an extraction claiming `11` and against one
claiming `1.24`, parametrised, so a third spelling of the same mistake fails
here too. Each has a control on the identical fixture with a correct
extraction, which still clears — so the failure is about the wrong value and
not about an unclearable fixture.

Counterfactuals, run both ways: restoring the JSON example fails six tests
including the `1.24` reproducer; restoring the prose example fails six tests
including the `11` one.

## Verification

- Full suite: `4011 passed, 4 xfailed` — exit 0.
- `uvx ruff@0.16.0 format --check .`: `553 files already formatted` — exit 0.
- `git diff --check`: clean.

---

# Cold review round 6

Round 5's numeric residual closed. One new blocking finding, reproduced before
any fix — and the third time the same defect has been found in a new spelling,
which is what the fix is really about.

## NEW 1 — the no-binding invariant is now STRUCTURAL (BLOCKING)

Rounds 4 and 5 each closed one spelling of a coordinate-to-value binding in the
blind prompt: prose (``R1C2 is 11``) and JSON (``{"R1C2": "1.24"}``). Both
fixes came with guards written against the syntax that had just been found. A
NON-numeric binding in a fourth spelling passed all four of them:

```text
for R1C2 write N/A
R1C2: N/A
| R1C2 | N/A |
```

The coordinate-set check permits any coordinate that was requested; the
digit-strip check only sees numbers; the JSON and prose checks recognise their
own spellings. With `for R1C2 write N/A` appended, every guard stayed green, a
prompt-copying reader returned `N/A` for a scanned table whose `R1C2` visibly
reads `99` against an extraction wrongly claiming `N/A`, and the gate published
it as `verified_by_blind_cell_transcription`.

Enumerating syntaxes was always going to lose. The invariant is structural now.
The blind prompt is built from two parts:

| part | contents | invariant |
| --- | --- | --- |
| policy | template plus the spliced coordinate rule; identical for every table | contains NO concrete coordinate at all — `<ref>` placeholders are fine |
| request list | generated from the requested references | carries coordinates and nothing else |

A binding has to NAME a concrete coordinate, so it cannot exist without
breaking the first row — in any syntax, including one nobody has thought of.

**Known limit, accepted.** "Concrete coordinate" means a coordinate TOKEN, and
the digit check means a numeric value, so a coordinate spelled out in words —
"row one, column two is ALPHA" — sits outside both. That is accepted: this
guard exists to catch a future edit to OUR OWN template, not an adversary who
controls it, and anyone able to write that sentence into the prompt file can
defeat any check we put beside it. An ordinal-word check was considered and
rejected as not worth its cost: the rule's own text legitimately says "column
number one" and "so `n` is usually one", so the check fires twice on correct
wording, and silencing it would mean rewording what the model is told — moving
the prompt digest and reprocessing the corpus — to guard against ourselves.

`build_blind_cell_prompt_parts()` returns the two halves and
`build_blind_cell_prompt()` joins them, but the guard does not trust the
builder: `split_blind_cell_prompt()` recovers the halves from the text at the
`REQUEST_LIST_HEADING` boundary, so the property is asserted on the prompt AS
SENT, taken off the wire payload. It raises unless the boundary occurs exactly
once. The digit-strip check stays as belt and braces, and the prose and JSON
checks stay as named regressions.

**The echo reader is generic now.** `tokens_near_reference()` collects every
token the prompt places within a window of a coordinate, whatever punctuation
sits between them, and the standing reproducer runs the gate once for EVERY
candidate the prompt offers, plus the nothing-to-copy case. Parametrising the
leaked value was never the missing piece — the reader's syntax coverage was —
so the values (`11`, `1.24`, `N/A`) and the candidate sweep now vary
independently, each with a correct-extraction control on the identical fixture.

Counterfactuals, one per spelling: appending any of the three non-numeric
bindings above fails six tests, including the structural policy check, the
any-spelling table, and the `N/A` sweep.

## Verification

- Full suite: `4019 passed, 4 xfailed` — exit 0.
- `uvx ruff@0.16.0 format --check .`: `553 files already formatted` — exit 0.
- `git diff --check`: clean.
