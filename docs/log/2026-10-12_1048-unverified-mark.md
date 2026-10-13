# #1048: tables socr could not check ship, marked unverified

## Ruling (owner, 2026-10-08, `docs/plans/done/GOAL.md`)

A table on a scanned page that socr COULD NOT CHECK ships its text, marked "unverified",
visibly in the page text. A table the check CONTRADICTS stays withheld as before.

## Change 1: split the reject in `SourceEvidenceTableJudge.assess`

`verify_scanned_table` returns `verifiable=False` for three endings: empty evidence bundle
("no local content evidence"), no OCR witness (`no_witness_backend`), and cells with no
verifiable tokens. These no longer fall to the fail-closed floor.

- The judge sets `PageOutput.table_unverifiable = <reason>` on the candidate and hands it to
  the inner judge. Inner accepts: the table ships. Inner refuses: today's floor, unchanged
  (the field is cleared and the old status/failure-mode/reason are written).
- `verifiable=True, passed=False` (the numbers or labels are absent from a reading that
  exists) never reaches the new branch: floor, unchanged. This is the contradicted case.
- The same audit events are emitted as before (`source_evidence_table_reject`, plus
  `source_evidence_no_witness_backend`), so `tables_trust.json` still lists the page.
- `_phase_agentic` records `ps.table_ladder_disposition = TABLE_UNVERIFIED` when an accepted
  scanned candidate carries `table_unverifiable` and no ladder terminal exists yet. That is the
  state every surface already reads for a table that shipped unchecked (the same state
  `_mark_escalation_not_received` reuses): `_apply_ladder_disposition_guard` demotes the
  finalized copy to `warning` / `table_unverified`, the document status, the `tables` counts
  (`unverified_text`), the CLI buckets and the sidecar follow. A stronger ladder terminal
  (rejected, withheld) set later wins. `audit_passed` on the chosen attempt is never touched.

Status chosen: `warning` + `table_unverified`, because that is what a table shipped without a
verdict already gets from the ladder; no new status.

## Change 2: the mark

Exact line, written above every table of a `table_unverified` page, followed by a blank line:

```
> [table unverified: socr could not check this table against the page; verify it against the page image before citing]
```

`UNVERIFIED_TABLE_MARK` / `UNVERIFIED_TABLE_MARK_RE` in `core/manifest.py`. Written by
`_apply_unverified_table_mark`, the last text-touching step of `_select_and_finalize_page`
(after `_apply_native_paragraphs`, before the #713 digest stamp), the seam every writer
shares: the crash-recovery fragment, the terminal fragment, the stitched `.md`, the manifest
and the resume path all see the same bytes. Idempotent, so a restored page whose saved text
carries the mark does not get a second one. It applies to:

- the new could-not-check scanned tables, and
- the existing ladder `TABLE_UNVERIFIED` pages that ship table text (a page whose unverified
  table was floored has no table block, so nothing is marked).

Not changed: verified pages, pages without a table, contradicted tables, `table_rejected`.

## Behaviour change to be aware of

The could-not-check candidate is now ACCEPTED at the rung that produced it. Before, the reject
made the router escalate to the next rung (cloud), which could have produced a checkable table.
The owner ruling ("ship them ... do not withhold them"; retrying until a check passes waits for
the run's counts) is read as accepting at the first rung. Re-escalating on could-not-check is a
separate decision.

The verifier-raised path (`_reject_unverified`, `REJECTION_VERIFIER_ERROR`) is untouched: it
carries no `source_evidence_table` token, never reached the floor, and keeps its current
ending. The brief lists it under "not verifiable"; it is not part of this change.

## Gleason-Lee 2003 "2 withheld"

Checked on `~/.local/state/socr-housekeeping/archive-scan/redo-out/2003__Gleason_Lee_...`
(metadata `tables: 4 text (0 verified, 4 unverified), 5 withheld`; the issue's run had 6 / 2).
The audit log has zero `source_evidence_table_reject` events: the withheld regions come from
`structure_class_ladder_exhausted` / `invisible_scan_unread` / `native_table_structure_failed`
pages, not from this gate. So those withheld tables did NOT come through this path, and this
change does not alter them. The shipped-unverified ones (pages 11, 21, 29 and others) are the
ladder `table_unverified` pages, which now carry the mark. The issue's original job output is
not on this machine.

## Measurement (`scratch/routing-baseline/step-1048/measure.py`, the 181 stored pages)

- 5 not verifiable: would now ship marked (3 "table cells contain no verifiable tokens",
  2 "no local content evidence available"). Dougherty p3, Levy p30, Liu et al. p25 and p27,
  Hong-Stein p59.
- 157 verifiable and failed: still withheld.
- 19 verifiable and passed: unchanged.

Caution: Dougherty p3's stored model text is a generic invented product-feature list, i.e. a
fabricated table. It would ship, marked. That is the owner ruling's consequence for a page
where nothing could check.

## Tests

`tests/test_gh1048_unverified_table_mark.py`. End to end through `process()`, same process,
only the OCR witness differs: silent (ships, marked, fragment and final `.md`, status warning,
`unverified_text` 1), contradicting (failed-table marker, no table, no mark, no disposition),
supporting (table, no mark; the unverified page equals the verified one minus the mark).
Resume of a marked page keeps exactly one mark. Unit pins for the guard (placement,
idempotence, unverified-only, not a failure marker) and the field's serialization.

Updated for the new contract: `test_gh658_no_witness_backend_reason.py` (inner judge refuses,
so the floor ending is still what it pins), `test_source_evidence_table_judge.py` (contradiction
now needs a reading; empty evidence has its own test), `test_gh163_scanned_native_trust.py`.

## Owner ruling 2026-10-12: disputed tables ship too (supersedes "contradicted stays withheld")

Evidence from the owner: on 12 hand-checked failing pages the model was right on 90 of 92
disputed numbers, 11 of 12 pages fully correct; Forney 1988 p22 had 2 wrong numbers (row shift).
Scope: ONLY the scanned-table source-evidence check. Ladder rejections, native-verifier
contradictions and everything else are unchanged.

- `verifiable and not passed` now takes the same path as could-not-check: the candidate is
  handed to the inner judge; accepted, it ships with `PageOutput.table_disputed` set to the
  EVERY unsupported token (`SourceEvidenceResult.disputed_values`, no cap; the reason text keeps its own 5-token excerpt). Refused by the
  inner judge: the old fail-closed floor, unchanged.
- Second mark, written by the same seam (`_apply_unverified_table_mark`, idempotent), exact form:
  `> [table disputed: a second reading of the page (Tesseract) disagrees with N values: V1, ..., VN; check them against the page image before citing. [page image](figures/failed_table_pN.png)]`
  `disputed_table_mark()` / `DISPUTED_TABLE_MARK_RE` in `core/manifest.py`. The link reuses the
  floor's PNG (`_render_d3_floor_png`, `d3_floor_png_ref`); no new asset path. Omitted if the
  render failed.
- Status: `warning` + `table_rejected`, via `ps.table_ladder_disposition = TABLE_REJECTED`.
  Why: it is the existing "checked and found wanting" state, at least as severe as
  `table_unverified` (both warning; rejected has its own bucket, CLI line and metadata note, so
  disputed is distinguishable from could-not-check), `audit_passed` semantics unchanged, and it
  stops `_withhold_contradicted_unverified_tables` from turning the page into a withhold. One
  consequence: `table_counts` counts these as shipped, not verified, and not `unverified_text`.
  A resumed run skips such a page (the D1b REJECTED exception), mark intact.
- Tests: the same-process difference test now runs silent / contradicting / supporting witness:
  silent = unverified mark, contradicting = text + disputed mark naming 41.5 and 17.2 + image
  link + `table_rejected` (before: floor marker), supporting = no mark. Mutant
  `drop_disputed_mark` and `disputed_back_on_the_floor` are killed.

### Measurement on the 181 stored pages (`step-1048/measure.py`)

- 157 verifiable-and-failed: now ship with the DISPUTED mark (before: withheld).
- 5 not verifiable: ship with the UNVERIFIED mark.
- 19 verified: unchanged.
- Withheld by this check: 0. (The page judge can still refuse a reading; the stored data has no
  judge verdicts, so that remainder is not measurable here.)
- Forney 1988 p22: verifiable, failed, ships with the disputed mark naming all 6 unsupported
  values: `4.52, 72, 4.01, 124, 4.31, 4.76` (an earlier 5-value cap dropped `4.76`, one of the two
  genuinely wrong numbers; the cap is removed).

## Revision (team lead, 2026-10-12): keep escalation; mark only where the floor would apply

The earlier build accepted the candidate at the rung that produced it, which stopped the router
from trying a later rung. Replaced:

- `SourceEvidenceTableJudge` REJECTS exactly as before (no inner-judge call, ERROR/HALLUCINATION or
  NO_WITNESS on the attempt) and only records `table_unverifiable` / `table_disputed` on the
  candidate. The router escalates as today: Gemini after local qwen on a default Mac run; nothing
  after the only rung on a `--strict-local` qwen-only cluster.
- A later rung that verifies is accepted and ships unmarked.
- `promote_unchecked_table_attempt` (`pipeline/agentic.py`) runs when the ladder ended with NO
  accepted rung, i.e. where the floor applied. It picks the LAST attempt carrying either field with
  text (the most escalated rung that produced a table; no constant, deterministic), restores it to
  SUCCESS / no failure mode, marks it accepted, strips the `source_evidence_table` token from its
  reason, and makes the decision accepted. The disposition (`TABLE_REJECTED` for disputed,
  `TABLE_UNVERIFIED` for could-not-check), the PNG for the link and the mark follow as before.
- Pre-existing bug found and fixed on the way: the floor latched on ANY attempt whose reason carried
  `source_evidence_table`, so a table rejected at rung 1 and VERIFIED at rung 2 was floored anyway.
  The floor now applies only when `not decision.accepted`. Without this, "a later rung verifies,
  ships unmarked" is false.
- Zero-support pages are unchanged (owner has not answered): same marks as above.

Tests: a two-rung ladder whose rung 2 verifies ships rung 2's text unmarked and calls rung 2,
pinned as a difference against the one-rung ladder (disputed mark on rung 1's table); both rungs
disputed ships rung 2's table with the disputed mark. Mutants killed: `accept_at_first_rung`
(10 tests fail), `exhausted_ladder_floors_again`, `drop_the_mark`, `drop_disputed_mark`,
`mark_only_in_saved_body_path`.

## Astra review fixes (2026-10-12)

- P0-1: marks cover every table form: pipe blocks and the outermost HTML `<table>` per line
  (`_table_start_lines`). Other forms (aligned text columns) are covered only by the page mark.
- P0-2: the mark follows the page. `PageState.unchecked_table_kind` / `unchecked_table_values`
  are set at promotion, persisted sparsely in the sidecar, restored on resume, and win over the
  candidate's own fields. A replacement from `_escalate_table_page` that carries neither field
  stays marked. I chose "keeps the mark" over "re-check and clear": that path has no
  source-evidence verdict to clear on. Only a passing crop re-judge clears.
- P0-3: promoted pages open with the page-level mark (first line, with the page-image link when
  rendered; the PNG is now rendered for could-not-check promotions too). Which pages are promoted
  is unchanged (zero-confirmation pages still ship; the owner has not decided).
- P1-4: `_rejudge_crop_patched_page`, accepted branch: clears `unchecked_table_kind/values`,
  the candidate fields and the disposition this path set. Refused branch untouched, link kept.
- P2-5: the resume test now uses the disputed (`table_rejected`) path, which IS restored from its
  sidecar, and asserts zero provider calls plus exactly one mark of each kind. A separate test
  pins that an unverified page is re-run and still marked once. A sidecar round-trip test covers
  the new state.
- Mutants killed: html_tables_unmarked, mark_ignores_page_state, no_page_level_mark,
  crop_rejudge_keeps_stale_claims, page_state_not_restored_on_resume (plus the earlier five).

## Corrected measurement (after rebase onto main with #1098/#1099/#1100)

Earlier figures in this log (157 disputed / 5 unverified / 19 verified) replayed the stored
Tesseract readings through an older evidence path, and an interim note called the corpus "scans".
It is not: by `pagetype/labels.json` field `layer` the 181 pages are 139 born-digital-layer pages
(no page-covering raster), 16 scan + OCR layer, 24 scan without text layer and 2 blank. The corpus run floored the born-digital-layer pages because the router's
`ps.is_born_digital` was False for them, so they ARE eligible for promotion and marks. The
guard that suppresses marks when `ps.is_born_digital` is True stays right and matches main's
floor condition.

Real `verify_scanned_table(page, model_text, native_trusted=False)` (main's evidence: #1099 text
layer merged with Tesseract by per-token max, #1085, #1087, #1098) on the PDFs in
`witness-pdfs/`, one subprocess per page (`step-1048/replay_real.py`, `page_one.py`; needs the
full environment, incl. TMPDIR, for pytesseract):

| layer | verified (no mark) | disputed | unverified | no table block |
|---|---|---|---|---|
| born-digital layer | 70 | 62 | 2 | 5 |
| scan + OCR layer | 0 | 16 | 0 | 0 |
| scan, no text layer | 2 | 18 | 1 | 3 |
| blank | 0 | 0 | 2 | 0 |
| total | 72 | 96 | 5 | 8 |

- The 2 blank pages would not reach a model after #1100; they never ship a mark.
- So after #1099 the pages reaching marks are 96 disputed + 3 unverified (non-blank) = 99,
  not 162 (157 + 5); 72 ship verified and unmarked.
- Forney 1988 p22 (scan + OCR layer) is still disputed, naming `4.52, 72, 4.01, 124, 4.31, 4.76`.
- The earlier `replay.py` (stored witness texts only, no text layer) is superseded by this one.
