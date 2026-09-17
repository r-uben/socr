# GH-674 — an in-scope equation-sidecar skip must not ship SUCCESS with orphan crops

## What changed

`#664` made an in-scope equation-region skip AUDITED (the terminal
`equation_sidecar_skipped_no_page_output` event, on the resume allowlist so it
survives), but the event had no consumer: nothing demoted the document on it,
so a genuine in-scope miss could ship a clean `SUCCESS` while the crop PNG sat
on disk with no sidecar attached.

This ticket closes it with **demotion**, not fabrication — `#664` already
ruled out inventing a `PageOutput` ("a back-door SUCCESS path"), and the
ticket reaffirmed that.

The mechanism mirrors `#682`'s fresh precedent for the three `chart_region_*`
flags (`0cfaf12`, merged same day into `91901ab`):

1. **`PageState.equation_sidecar_skipped: bool`** (`src/socr/core/state.py`) —
   a new field next to the three `chart_region_*` flags.
2. **Set at the emit site** (`_attach_equation_latex_sidecars`,
   `orchestrator.py:~15599`): once per page, in the same branch that already
   appends the audit event, regardless of how many regions were skipped or
   already recorded (a disposition, not a count).
3. **Persisted** in the sidecar meta write block (`_flush_page_sidecar`,
   `orchestrator.py:~11176`) alongside the chart-region flags.
4. **OR-restored** in `_restore_terminal_page_state` (`orchestrator.py:~12100`)
   — never a plain assignment, so a flag this run already set survives a
   sidecar written before this fix, or one from a run that never hit the miss.
5. **Demotes** in `_phase_assemble`'s `pages_ok` reduction
   (`orchestrator.py:~13277`): a new `equation_sidecar_skipped_pages` bucket,
   same `AUDIT_FAILED` "completed with warnings, output written" path as every
   sibling content-doubt bucket around it — the page's own text is not wrong,
   but an unattached region's crop cannot leave the run reporting a clean
   `SUCCESS`.
6. **Surfaced** via a new `_equation_sidecar_skipped_note` static method
   (mirrors `_chart_region_note`), appended to `final_result.error` so a
   consumer reading `metadata.json` or the CLI sees the debt without opening
   `audit_log.json`.

## Files changed

- `src/socr/core/state.py` — new `PageState.equation_sidecar_skipped` field.
- `src/socr/pipeline/orchestrator.py` — flag-set at the emit site, persist,
  OR-restore, `pages_ok` bucket, `_equation_sidecar_skipped_note` + its call
  site in `_phase_assemble`.
- `tests/test_gh674_equation_skip_demotion.py` (new) — five tests, all pinning
  the OUTCOME (`DocumentState`/`EngineResult.status`), not the event:
  in-scope miss demotes with content still shipping; a clean run (no miss)
  still succeeds (reverse regression); an out-of-scope page that already
  attached is NOT demoted while the missing page is (regression guard for
  `#664`'s `page_nums` scoping); the flag survives a real flush/restore round
  trip; a flag set this run is not cleared by an older sidecar missing the key
  (the OR case).
- `tests/test_p6_disposition_persistence.py` — added
  `"equation_sidecar_skipped"` to the frozen pre-disposition sidecar key-set
  (`test_sidecar_only_additive_key_is_disposition`), same treatment `#682`
  gave its three keys.
- `tests/fixtures/p6/prechange_assemble.json` — purely additive:
  `"equation_sidecar_skipped": false` added to all 12 sidecar entries.

## A trap in regenerating the P6 golden fixture (worth flagging up)

`socr-regenerate-p6-prechange --rev <rev>` runs `git archive <rev> src` into a
temp tree and imports `p6_corpus_fixture` from **the launching checkout**
(deliberately, per its own docstring: "the corpus fixture always comes from
the checkout that launches the command"). Running it against `--rev 91901ab`
(main, unmodified) from this worktree produced a fixture that did **not**
match the checked-in one — not just my additive key, but page 9 losing a
`structure_class_model_table_kept` event and page 12 losing a
`corrupt_math_hybrid_shipped` event, neither touched by this ticket.

Measured, not assumed: two independent regenerations at the same rev were
byte-identical to each other (deterministic), and a from-scratch `git archive`
of `91901ab` (both `src/` and `tests/`) run through plain `pytest` reproduced
the checked-in fixture exactly (0 failures across the full suite, including
the P6 fixture tests). So the divergence is an artifact of the regenerate
tool's isolated subprocess environment (`_clean_environment()` strips most of
`os.environ`), not a real behavioural difference at `91901ab` — but I did not
chase why, since it is out of scope for this ticket and my own pytest-driven
capture (the same path CI exercises) showed zero difference beyond the one key
this ticket adds.

**Practical consequence:** I did not use the tool's own output. I restored the
original checked-in fixture and added `"equation_sidecar_skipped": false` to
each `pages/*.json` sidecar entry directly (matching exactly what
`test_p6_stage_ab_difference.py`'s real pytest-driven `current` fixture
reported as the only diff), then confirmed both P6 stage-A/B and stage-C
difference suites pass clean.

## Mutation proof (each guard separately, `/tmp/gh674-mutant`, deleted after)

Harness: `cp -R src tests pyproject.toml` outside the repo; canary asserted
`socr.__file__` resolves under `/private/tmp/gh674-mutant` before every run
(`pytest`'s `pythonpath=["src"]` in `pyproject.toml` would otherwise shadow an
external `PYTHONPATH`, per the standing repo trap — the copy includes its own
`pyproject.toml` so this canary is meaningful). Each mutation asserted its
anchor's uncapped `count == 1` and asserted the string actually changed before
writing, aborting otherwise.

| Guard | Mutation | Failing test(s) | Count |
|---|---|---|---|
| 1. Flag-set at emit site | Deleted the `ps.equation_sidecar_skipped = True` assignment | `test_document_status_is_not_success`, `test_only_the_missing_page_is_flagged`, `test_flag_is_persisted_and_restored` | 3 failed, 2 passed |
| 2. `pages_ok` demotion bucket | Replaced `pages_ok = pages_ok and not equation_sidecar_skipped_pages` with a no-op | `test_document_status_is_not_success`, `test_only_the_missing_page_is_flagged` | 2 failed, 3 passed |
| 3. Sidecar persistence | Deleted the `"equation_sidecar_skipped": (...)` meta-dict entry | `test_flag_is_persisted_and_restored` | 1 failed, 4 passed |
| 4. OR-restore | Replaced the OR-restore with a plain `meta.get(...)` assignment | `test_a_flag_set_this_run_is_not_cleared_by_an_older_sidecar` | 1 failed, 4 passed |

Each mutation was caught by a distinct, non-overlapping subset of the five
tests, and reverted before the next mutation (verified via `cp` from the
worktree, not `git checkout`, since the worktree's own git state is shared
with other agents).

## Test results (measured)

- `tests/test_gh674_equation_skip_demotion.py`: 5 passed.
- `tests/test_equation_latex.py` (the `#664`/GH-157 suite this ticket
  extends): 43 passed, unchanged.
- `tests/test_p6_stage_ab_difference.py` + `tests/test_p6_stage_c_difference.py`:
  49 passed (was 6 failed before the fixture update above).
- `tests/test_gh189_mixed_chart_preservation.py`, `tests/test_chart_lane.py`,
  `tests/test_p6_disposition_persistence.py`, `tests/test_gh225_fabricated_image_urls.py`
  (the `#682`/`#252` demotion-idiom neighbours): all pass.
- Full suite, baseline vs branch, both measured:
  - Baseline (`git archive 91901ab`, fresh copy, `pytest tests -q`):
    **5645 passed, 1 skipped, 4 xfailed** (the skip is
    `test_gh592_scoped_positional_emission.py` — "origin/main ref not present",
    the documented archive artifact, confirmed reproduced independently on a
    second archive run).
  - Branch (this worktree, real git history so `gh592` runs instead of
    skipping): **5651 passed, 4 xfailed**, 0 skipped.
  - Reconciles exactly: +6 passed = +1 (`gh592` moving from skip → pass, the
    archive artifact) + 5 (this ticket's new tests). Zero regressions, zero
    unexplained deltas.
- `uvx ruff@0.16.0 format --check .`: clean (one file needed reformatting
  after first draft, fixed with `uvx ruff@0.16.0 format` on that file only).

## Framing check

Nothing in the orchestrator's framing needed correction. The one addition
beyond the ticket's explicit ask is the `_equation_sidecar_skipped_note` — the
ticket's required evidence only asked to pin document status, but every
sibling demotion bucket in `_phase_assemble` also surfaces a free-text note at
the document level (the repo's stated no-silent-loss rule: "Failures must
surface at every level ... not just one"), so I added the matching note for
consistency rather than leaving this one bucket silent at that layer.
