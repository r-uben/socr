# #1129 — judge-rejected best effort demoted and marked

## Problem

On corpus run `23117187`, 385 pages shipped `status=success`, `failure_mode=none`, no
in-text mark, with `audit_passed=false` and `disposition.primary_reason=unaccepted_output_kept`.
`_select_page_output_tagged` returned `p.best_output` verbatim on `BEST_OUTPUT_UNVERIFIED`.

## Fix

- `_finalize_best_output_unverified`: on a copy, `SUCCESS` → `WARNING` when `audit_passed` is
  false; failure mode `audit_failed`, or `page_judge_timeout` when `judge_outcome` is a timeout.
  Does not flip `audit_passed`.
- `_kept_judge_rejection_page`: skips demotion/mark when `table_ladder_disposition` is a GH-353
  terminal or `unchecked_table_kind` is set — resume re-selection otherwise mis-tags ladder pages
  as `BEST_OUTPUT_UNVERIFIED`.
- `_apply_unaccepted_output_mark` at the `_select_and_finalize_page` seam (#1048 pattern): first-line
  `> [page unaccepted: …]` with trimmed `judge_reason`.

## Failure mode choice

Reused `FailureMode.AUDIT_FAILED` for completed judge refusals/crashes (the sibling
`BEST_ATTEMPT_FLAGGED` path already forwards attempt modes). `PAGE_JUDGE_TIMEOUT` when the typed
outcome is a timeout — distinct from provider `TIMEOUT`.

## Tests

`tests/test_gh1129_unaccepted_output_demoted.py` — unit demotion/mark, hermetic `process()` on a
scanned page (reject vs raise), resume gate.

## Measurement

Script: `scratch/routing-baseline/exec/1129/measure.py` on `runs_23117187/` sidecars.

## Round 2 (Astra review)

- **P0:** Replaced blanket ladder-terminal exemption with `_skip_unaccepted_mark_ladder_resume`
  (only when the winner is already `WARNING` with a ladder `failure_mode`). Demotion always runs
  for `audit_passed=false`.
- **P1:** Disclosure provenances are `BEST_OUTPUT_UNVERIFIED`, `BEST_ATTEMPT_FLAGGED`, and
  `WHOLE_DOC_SECTION`; mark requires `judge_reason` / `judge_outcome` (or `error`). `BEST_ATTEMPT_FLAGGED`
  forwards judge fields from the source attempt.
- **P1:** `_split_page_top_banners` / `_with_page_top_banners` make page-level banners idempotent.
- **P1:** Restored gh161 legacy SUCCESS + `audit_passed=false` sidecar test; fixed resume test
  `doc_dir` / fingerprint pipeline reuse; `autospec` on judge stubs.
- **P1:** `measure.py` replays `_select_and_finalize_page` and reports changes by original status.
- **P2:** `_neutralise_unaccepted_mark_reason` strips markdown/HTML from embedded judge reasons.

## Round 3 (Astra r2)

- **P1 disclosure:** Ladder-rejected pages get the judge-reason mark; skip only when an
  unaccepted or (#1048) page-unchecked banner is already at the top (ladder + page-unchecked
  on resume). `_WholeDoc` / `WHOLE_DOC_SECTION` forward judge fields.
- **P1 resume:** `test_resume_never_treats_demoted_page_as_clean_success` passes pipeline
  `output_dir` (not `doc/`); positive control proves accepted terminal pages still skip.
- **P1 measurement:** Full replay over all **17,091** winners of run 23117187 (Astra's
  round-3 replay of this final code): `success` + `audit_passed=false` **385 -> 0**;
  **757** pages change (385 SUCCESS, 367 WARNING, 5 ERROR by original status) and
  **16,334** are unchanged on text/status/failure_mode. The round-2 figures (759 changed,
  7 ERROR) are historical: two ERROR pages already carry a recognised banner and no longer
  get a duplicate. Every changed page keeps its full original text. A sidecar-state replay
  does not prove full artifact byte identity.
- **P2:** Judge reason embedded as Markdown inline code (reference-style `![x][y]` inert).

## Round 4 (PR #1142 reviewer)

- **Soft 1:** Removed `_skip_unaccepted_mark_ladder_resume` entirely; idempotency is the
  existing unaccepted-mark check only. Ladder + `unchecked_table_kind` pages get both #1048
  and judge-reason banners on the first finalize pass.
- **Soft 2:** `audit_passed=false` with no judge signal still ships the unaccepted mark using
  ``the page judge did not accept this reading; no reason was recorded``.
- **Nit:** Dropped `_trim_unaccepted_mark_reason` alias.
- **Mutants:** Reviewer's ``_skip`` always-false and ladder-branch-only mutants are **moot**
  (helper deleted). New r4 mutants: ``restored-ladder-skip``, ``skip-no-mark-without-reason``.

## Round 5 (PR #1142 reviewer r2)

- **Blocker:** Restored the early return in ``_apply_unaccepted_output_mark`` when there is
  no judge evidence (``judge_reason``, ``judge_outcome``, or ``error``). Round 4's
  ``audit_passed=false``-alone fallback stamped a false refusal on ladder-terminal pages
  whose C3 guard demoted a judge-accepted winner; resume then diverged from the first flush.
  The round-4 ``resumed_page.audit_passed = True`` overrides in two tests **hid** that
  regression and were **removed**.
- **P2:** Typed missing-verdict outcomes use ``page_judge_timeout`` / ``page_judge_exception``
  failure modes and "did not finish" banner wording (never "rejected").
- **Docs:** ``OUTPUT.md`` lists ``BEST_ATTEMPT_FLAGGED`` and ``WHOLE_DOC_SECTION``; ERROR
  winners stay ERROR.
- **Mutants (r5):** ``fallback-without-judge-evidence``, ``exception-as-refusal``.
