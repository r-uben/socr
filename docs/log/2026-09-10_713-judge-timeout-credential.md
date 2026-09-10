# #713 — a ladder-accepted candidate whose page judge timed out

**Branch** `fix/713-ladder-accept-vs-judge-timeout` · **Base** `origin/main` @ 6918114

## The defect

Measured during #703 on the third-institution census (`boe-meetings-2018-scan-p28-30.pdf`
p1). The audit log records `table_ladder_accepted` for the qwen candidate; the cached
`PageOutput` carries `audit_passed=false` with `judge_reason='judge raised: timed out'`.
`_grid_authored_attempt` admits an attempt only on `audit_passed` or the
`ambiguous_deferred` allowlist, so the strict pool is empty, `structure_class_grid_winner`
returns `None`, and S1 case (iii) ships the fail-closed floor under
`structure_class_ladder_exhausted`.

Two losses in one: a verified 3,539-character reading is discarded, and the reason claims
every candidate was refused when in fact none was judged.

## The ruling implemented

Astra, 2026-09-10: option (a) as a **narrowly typed, flagged exception**, with (c) whenever
its evidence is missing. A bounded retry may improve recovery but cannot be the correctness
mechanism, because a second timeout must still have a defined outcome.

## What was built

1. **Typed outcome.** `PageOutput.judge_outcome`, written only by the judge-exception guard
   in `route_page`, from the exception's *type* (`judge.is_page_judge_timeout`). A judge
   reason string is built from an arbitrary `str(exc)`, so any provider can put "timed out"
   in it; a gate keyed on that substring is a gate any upstream can open. A completed
   rejection never reaches the guard and carries no outcome at all.

2. **Credential.** `core/page_credential.py`. Minted at the end of `_run_table_judge_gate`
   — the last stage of the page loop, so it binds the page's settled bytes — and only for a
   candidate carrying the typed timeout (a credential on every accepted page would rewrite
   the content-addressed serialization of every table page in every corpus). Bound fields:
   schema, page number, document checksum, SHA-256 of the complete canonical candidate text,
   attempt engine + provider id/model/backend, judge model, typed judge outcome, run
   fingerprint, and one entry per emitted table carrying that table's markdown digest, the
   **witness image's** digest and scope, and the rung identities that executed.
   `finalized_sha256` is stamped later, in `_select_and_finalize_page`, after every guard.

3. **Admission.** `manifest.credentialed_judge_timeout_winner`, a new S1 ending placed after
   the strict pool and A1b's corroboration fallback. Requires the typed outcome, a credential
   that verifies against the bytes about to ship and covers **every** emitted table, no
   adverse `table_ladder_disposition`, and no event in `CREDENTIAL_BLOCKING_EVENT_KINDS` (a
   deliberate, drift-guarded subset of `tables_trust.TABLE_DISTRUST_KINDS`). `audit_passed`
   is never flipped on the stored attempt.

4. **Finalization.** A demoted COPY: `WARNING` / `audit_passed=False` /
   `FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED`, plus an in-body note stating that the table
   evidence passed while the page-level check remained incomplete and the prose is
   unverified.

5. **Without the credential**, the floor still ships but under
   `FailureMode.PAGE_JUDGE_TIMEOUT` and its own selection tag, with its own audit event
   (`page_judge_timeout_floor`) and CLI line. The page keeps every surface it already had:
   both new provenance members map to the *existing* public dispositions, so the floor stays
   in `structure_class_floor_pages` and the credentialed page stays a structure-class model
   page. #713 adds a reason; it removes no surfacing.

6. **Resume.** A second deliberate exception in `_load_terminal_page`, narrower than the
   GH-353 REJECTED one: the persisted record must be WARNING / `audit_passed=False` /
   `JUDGE_TIMEOUT_LADDER_ACCEPTED`, its credential must re-verify against the live document
   checksum, the attempt identity, the typed outcome and the **current run fingerprint**, and
   the fragment on disk must hash to `finalized_sha256`. Any mismatch revalidates.

## Deliberate residuals

- **Witness identity on resume is bound transitively.** The crops are temp files that no
  longer exist, so the run fingerprint (which binds render DPI, rung identities and prompts)
  is what stands in. The per-table witness digests are persisted, so an exact re-check is
  possible for a caller that can re-render.
- **Selection does not check the run fingerprint** — `core.manifest` cannot see the run's
  identity. The resume gate does, which is where a changed config must invalidate.
- **The existing BoE cache is not promoted.** It carries no typed outcome and no credential,
  so replaying it yields the marker, pinned by test. Whether a fresh BoE run now ships the
  page was **not run** here.

## Round 2 (2026-09-11) — Astra REQUEST_CHANGES at `20d3df8`

Four P1 and one P2, all one hop past the fresh-admission path round 1 pinned.

**P1-1, restore → reselection.** A restored credentialed page failed selection
again: the ledger body carries the disclosure note, and
`credentialed_judge_timeout_winner` re-verified it against `candidate_sha256`,
which describes bytes that predate the note. The page collapsed to
`PAGE_JUDGE_TIMEOUT` plus the marker, so a resume destroyed a reading the first
run had shipped. Fixed with a separate ending,
`restored_credentialed_judge_timeout_winner` /
`STRUCTURE_CLASS_JUDGE_TIMEOUT_RESTORED`, which verifies the body against
`finalized_sha256` and returns it VERBATIM. Not a weaker check: same refusals,
same coverage requirement, and an unfinalized credential still never verifies.
Pinned as restore → reselection → reassembly byte identity, plus a three-resume
note-count pin and a one-byte mutation control.

**P1-2, a completed rejection retires the authority.** Two shapes, two fixes.
At the judge boundary `route_page` now records `JUDGE_OUTCOME_COMPLETED` on the
attempt's own output and drops the credential on a refusal, so a rung that timed
out and then refused the same bytes can no longer ship under the timeout
exception. In selection, `superseding_rejection` refuses admission when a LATER
attempt completed a verdict against the SAME candidate digest. Keyed on bytes,
never on position or provider: a rejected re-cropped candidate is a verdict
about different text and leaves an unchanged incumbent alone (pinned both ways),
and `REJECTION_VERIFIER_ERROR` is excluded because a broken verifier is another
missing verdict.

**P1-3, the production adapter.** `_TimeoutJudge.assess` turned its own deadline
into `AcceptDecision(False, "judge timeout")`, a completed refusal as far as
`route_page` could tell, so the real loop never entered the exception branch and
the typed field round 1 added could not be set on the path it exists for. It now
raises `judge.PageJudgeTimeoutError` (a `TimeoutError` subclass, so
`is_page_judge_timeout` needs no special case), and `future.done()` separates our
deadline from a timeout the inner judge raised. The message keeps the word
"timeout" because `_phase_agentic`'s cascade-halt probe scans attempt reasons for
that substring. Tested through the real wrapper, three inner judges, one process.

**P1-4, `table_unexplained_lanes`.** It is a reported OMISSION (native lanes with
values in matched rows mapping to no emitted column), not an unscorable page, so
it now blocks admission. The exclusions are pinned explicitly as
`CREDENTIAL_NON_BLOCKING_EVENT_KINDS`: the old subset assertion is satisfied by
an empty policy and stays green when a new distrust kind is silently omitted.

**P2, document metadata.** `_structure_class_floor_note` now takes the finalized
records and derives document reporting from what each page SHIPPED. The
credentialed page no longer reports "ladder exhausted; fail-closed floor
shipped", and the two timeout endings get their own sentences.

Also: selection's docstring claimed a run-fingerprint check it does not perform.
Corrected rather than added — `core.manifest` cannot see the run's identity, and
a caller that cannot compute it must not satisfy the check by passing `None`.
The fingerprint is checked in the resume gate, the only path that admits a
credential minted by a different run.

### Residuals

- **No fresh BoE run.** Everything here is local credential/state fixtures. The
  historical `table_ladder_accepted` log event on the BoE 2018 p1 cache still
  cannot promote that page, by design, and that remains pinned.
- **Witness bytes are not re-rendered on resume.** Binding stays transitive
  through the run fingerprint; the per-table witness digests are persisted, so an
  exact re-check remains possible for a caller that can re-render.
- **Blocking events are not applicability-resolved.** Any historical blocking
  event on the page refuses admission, which is conservative beyond "outstanding
  applicable". `table_unexplained_lanes` has no retirement path today; when one
  exists it belongs there, not in the exempt list.
- **The cascade-halt probe still reads a substring.** Typing it is out of scope;
  the raised message preserves the existing signal deliberately.
