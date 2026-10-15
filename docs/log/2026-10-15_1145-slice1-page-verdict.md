# #1145 slice 1 — the page verdict and the canonical unchecked mark

2026-10-15. Design: `scratch/routing-baseline/leads/1145/design.md`. Slices 2 and 3 follow
from the same design; this log records slice 1's decisions and what it deliberately leaves
open.

## What shipped

`core/verdict.py` (new) holds the per-page verdict: `(check, result, scope, reason)` clauses,
`CHECKED ⇔ no clause ⇔ no mark`. `core/marks.py` (new) is the one place that knows what a
visible mark is — a pure move of the grammar, the legacy families and the first-block
CommonMark reader out of `benchmark/acceptance.py`, which now imports them. The seam is
`_apply_verdict_marks`, step 9f of `_select_and_finalize_page`, last, so the verdict sees
the final bytes and every mark an earlier step already put at the top. The sidecar persists
`verdict`, and both resume rules are wired. `docs/OUTPUT.md` documents the grammar, the
coverage table and the resume rules.

Measured on all 17,091 terminal sidecars of run 23192118, replayed in two arms in one
process (the seam live, and the seam with #1145's emission stubbed out, so every other guard
cancels): `warning_unmarked` 4,170 → 0, `error_unmarked` 550 → 0, `mark_not_at_top`
2,064 → 0, with 0 CHECKED pages whose bytes changed, 0 SUCCESS pages demoted and 0 pages
whose status moved.

## Four bugs found in the implementation itself

Each was found by refusing to accept a diff or a failing test that could not be explained.

1. **`AUDIT_FAILED` was read as a judge refusal.** The mode is a catch-all:
   `_select_page_output_tagged`'s `CORRUPT_MATH_HYBRID` branch stamps it with no judge
   involved at all. Measured on the P6 corpus, page 11: `audit_passed=False`,
   `rejection_class=""`, no `judge_accepted`, no `judge_outcome` — and #1129's own banner
   stays silent on it for that reason. The mark told the reader the page judge had rejected
   bytes it never saw, on the design's largest class (1,118 corpus pages). Now
   `_corrupt_math_hybrid_clause` gives `equation_fidelity=FAIL`, and `_judge_refusal_clause`
   CALLS `_positive_judge_refusal_evidence` rather than restating it, so the
   `judge_accepted is True` veto over a stale `rejection_class` cannot drift.
2. **Both resume rules were unimplemented.** The sidecar was written but never read back:
   `PageState.published_verdict` had no writer, so rule (b) was inert, and rule (a) did not
   exist. A pre-#1145 sidecar is SUCCESS with `audit_passed=True`, so every gate passed and
   the page would have been skipped for ever, shipping unmarked — the defect surviving its
   own fix.
3. **A canonical line above the prose-recovery banner destroyed the page's text.**
   `_is_restored_prose_recovery` vouched for a page by `text.startswith(banner)`. That lane
   ships real prose, and the `scanned_flag` family covers only the four table checks, so an
   uncovered judge-refusal clause writes a line above the banner — the page then stops
   reading as a restored recovery and the NEXT run replaces its whole recovered prose with
   the bare failed-table marker. The same resume loss #649 round 2 closed, reached through a
   new door, and it is content loss. Fixed by skipping only this module's OWN canonical
   lines (the grammar's regex, with the page number in it) before testing the banner — not
   "any leading line", which is the forgeable rule #649's docstring rejects.
4. **The `native_glyphs` clause had been removed on a conflation.** The removal cited #140's
   report-only ruling and its 36.1% trigger rate; that ruling is about
   `has_math_font_typesetting`. The design names `has_unmapped_math_glyphs`, the PUA class,
   measured at 2.4% in the same document, which `_apply_unresolved_math_guard` already
   demotes on its own account. Restored as `FAIL` (the check RAN and found damage in the
   shipped bytes), with a dedupe so two sources for one check give one line, and a negative
   test pinning that the math-font signal still puts no line anywhere.

A fifth, PRE-EXISTING asymmetry became visible rather than being introduced: the
corrupt-math lane flip records `audit_failed` with the lane on and `none` with it off,
so the two arms now disclose different clauses (`equation_fidelity=FAIL` versus the
fail-closed `unknown=UNKNOWN`). The flip already set different modes and
`test_agentic_corrupt_math_default_flip_no_provider_is_additive_only` never compared
them, so nothing caught it. Pinned there as the exact known difference rather than
stripped away, and filed as **#1163**.

After (3) I swept every other first-line / `startswith` predicate over page text in
`core/manifest.py` and `pipeline/orchestrator.py`. Three remain; each is either guarded by
`is_page_failed_marker` — which the seam never prepends above, now a killed mutant — or
reads pre-finalization attempt text. No further site.

## Decisions

**`verdict` is NOT in the P6 oracle's `VOLATILE_KEYS`.** The #1027/#1030 precedent put
additive sidecar keys there, but this key is deterministic, it is the field the ticket is
about, and excising it would make the oracle blind to a wrong verdict on any page. It gets
28 real `ExpectedDiff` entries generated from the measured delta instead — 12 per-page
verdict entries and 16 text/digest entries — so the oracle is a readable table of the whole
clause derivation on that corpus. The frozen pre-change fixture was not regenerated.
`ADDITIVE_SIDECAR_KEYS` is a separate set, and the "no key moved" pin now also requires
every enumerated addition to be present on every page, so the allowance cannot become a
hole.

**The recovery-banner families are tuples, iterated, never named.** `PROSE_RECOVERED_FLAGS`
/ `NATIVE_TEXT_FLAGS` / `PROSE_RECOVERY_FLAGS` in `core/manifest.py`; both `marks.py`'s
`scanned_flag` family and `_is_restored_prose_recovery` iterate them. #1144 (PR #1158) adds
a born-digital pair, and a form either site fails to recognise is silent in both directions:
the meter counts a marked page unmarked, and the restore replaces a recovered page's prose
with the bare marker. Import-based so that reconcile is a one-line tuple change.

## Deferred, deliberately

**The `scanned_flag` family does not check the typed `scanned_prose_recovered` field**
(PR #1158, Fable round 6, P2). Filed as **#1164**, which carries the general fix below. The meter is bytes-only, so a model line byte-identical to a
recovery banner at the top of a WARNING/ERROR page counts as MARKED. Three reasons to leave
it in slice 1:

* The content-loss path is already closed. `_is_restored_prose_recovery` requires the typed
  field AND the banner, so a forged banner cannot get a model's whole output shipped past
  the floor. What remains is the meter *counting* such a page as marked — a measurement
  accuracy question, not reader harm.
* The class already exists on `main` for `page_failed` and is narrow: matching is whole-line
  equality against a template carrying this page's own number.
* Fixing it means widening `MARK_FAMILIES` from `Callable[[int, str], bool]` to something
  that sees the sidecar. That is exactly the claim slice 1 rests on — the move is PURE,
  pinned by `tests/test_acceptance_meter.py`'s 82 tests passing untouched. Widening the
  signature retires that pin and puts the meter's whole test file back in review scope, for
  a class whose worst outcome is one mis-counted page.

The right shape when it is done: a family contract that takes the page record rather than
`(page_num, line)`, so `scanned_flag`, `unaccepted_output` and the canonical family can each
require their own typed evidence, not just their own bytes. That is #1164, with a forged-banner page
(banner bytes, no typed field) pinned to count UNMARKED.

Also not measured in slice 1: whole-artifact byte identity (a replay sees page text, not the
assembled document or the figures directory — one fresh run plus a resume of it closes
this), and `table_label_unverified`, which is not serialized, so the replay cannot rebuild
that clause for ~15 pages; they land on the fail-closed fallback there.

**Resume rule (b) is not load-bearing today, and the design is wrong on why it is.** Every
clause-bearing page is WARNING or ERROR, and the ledger gate requires SUCCESS plus
`audit_passed` except on the D1b-withheld and credentialed-timeout paths — so the rule is
only REACHABLE through those two. The design argued from the escalated never-judged page
resuming as SUCCESS; slice 3 demotes that page, which makes it non-resumable. The rule is
kept because it makes the invariant true by construction and costs nothing, but its test is
the seam honouring a published verdict and the reader setting one, not a D1b page end to
end.

## Tests

`tests/test_gh1145_page_verdict_marks.py`: T1–T7 of design §9, the fail-closed fallback with
a SUCCESS control, the two glyph signals and their dedupe, the prose-recovery guard over both
banner forms, and `is_page_failed_marker` holding after the seam on every failed-marker
ending. T8/T9 need slice 3's SUCCESS-demoting clauses and T10 needs slice 2's document stub;
they ship with those slices.

Everything pins a DIFFERENCE, never a measured tuple (#253, reverted by #257). The
end-to-end lever is `chart_region_placement_unresolved`: a real signal on 260 corpus pages
whose guard is status-only, so it moves no byte of the body on its own, and one of the 322
`failure_mode=none` WARNING pages the ticket is about. Every `_phase_agentic` test is
parametrised over both provider states, pins `primary_engine` / `local_engine` /
`enabled_engines`, patches `_available_engines_for_agentic`, sets `_resolve_judge_model` to
`""` and asserts `route_page` un-called.

16 mutants, each killed by a named test. External copy of `src`, `tests` and
`pyproject.toml`; a canary inside the pytest process asserts `socr.__file__` resolves in the
copy (`pyproject.toml`'s `pythonpath = ["src"]` shadows an external `PYTHONPATH`, so a
canary beside the process cannot catch it); `src.count(anchor) == 1` asserted uncapped before
each edit. Two mutants survived first time and both were real gaps:
`prepend-above-failed-marker` survived because a page-scoped clause is always suppressed on
a marker page, so the guard was unreachable and had never been shown to guard anything — the
test now reaches it with a region-scoped published verdict, the one input that escapes
suppression; and the first `strip-helper-hides-body-change` mutant was an equivalent
implementation rather than a hiding one.

## Known readability cost

On a page like P6 page 3 the fact "the native table region failed verification" is now stated
three times: #1129's banner, the canonical line, and the pre-existing mid-body D3 note.
Suppression reads the leading block only, by design §0.2/§2, because the meter asks that the
first rendered BLOCK be all marks. Accepted for slice 1; to be put to the owner as a polish
item.
