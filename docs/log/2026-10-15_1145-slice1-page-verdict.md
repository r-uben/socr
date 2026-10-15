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

## Review round 2 (PR #1165, head a9f6ef5c)

The repo reviewer said merge once CI was green, with three softs. Fable said do not merge,
and independently replayed all 17,091 sidecars to the same 4,170/550/2,064 → 0/0/0 with
10,731 CHECKED pages byte-identical. The core held; **two marks could state something
false**, and one gate fixes both.

Both reproduced before being accepted:

1. **`page_judge=FAIL` on bytes the judge never saw.** The #649/#652 prose-recovery endings
   REBUILD the shipped body from the page's own text layer and then copy the refused MODEL
   attempt's `judge_accepted` / `rejection_class` onto it, so the refusal evidence is real
   and it is about bytes that did not ship. #1142 skips its banner on exactly this
   provenance set for exactly this reason; `_judge_refusal_clause` had no provenance input
   and wrote "the page judge did not accept the bytes that ship" above both recovery
   banners. `FAIL` asserts the check RAN on these bytes. It did not.
2. **`native_glyphs=FAIL` on a model-reading winner.** `_native_glyph_clause` read
   `has_unmapped_math_glyphs` with no engine gate, while the guard it mirrors
   (`_apply_unresolved_math_guard`) gates on `native_math_damage_ships(output, provenance)`.
   The review reported seven corpus pages. **Measured over all 17,091 terminal sidecars:
   259**, every one with a `qwen` winner, and on every one of them the false line was
   VISIBLE — the two counts (clause set changed, shipped bytes changed) came out identical,
   so none of the 259 was a page where another mark already covered the line. The review's
   finding was right and its magnitude was an undercount by a factor of thirty-seven; the
   number in this log is the replayed one, not the reported one.

**One fix: the ENDING is threaded into the seam.** `provenance` now reaches
`_apply_verdict_marks` → `build_page_verdict` → `derive_clauses`. The judge clause returns
`page_judge=UNKNOWN; reason=the bytes that ship were rebuilt after the judge ran and were
never judged` for an excluded ending — UNKNOWN rather than silence, because something true
is left to say, and it is the same FAIL-versus-UNKNOWN distinction this module carries
everywhere else. The glyph clause CALLS `native_math_damage_ships` rather than restating it;
that predicate is not a one-liner (an unknown ending and every ending that may carry native
bytes answer True, and a model body still carrying the codepoints answers True too, so it
errs toward keeping the warning).

The judge case is a reachability fix: 0 pages on this run, reachable on any fresh run where
the ladder refuses the model table. The glyph case is not — it was live on 259 pages.
Measured by running the seam twice per page in one process, once live and once with both
gates disabled (which is exactly the pre-fix behaviour), and diffing the clause set and the
shipped bytes:

```
pages_whose_clause_set_the_gate_changed = 259
clauses_withdrawn                       = {native_glyphs=FAIL: 259}
clauses_added                           = {}
pages_whose_SHIPPED_BYTES_changed       = 259   (by winning engine: qwen 259)
```

`clauses_added` being empty is the other half: the gate only ever WITHDRAWS a false
statement here, and the `page_judge=UNKNOWN` replacement fired on no page of this run, which
is consistent with the judge case being reachable rather than live.

### The three softs

* **Page-strict canonical match.** `_is_restored_prose_recovery` used `CANONICAL_MARK_RE`,
  which matches any page number, so a line about page 9 above page 4's banner would have
  been skipped and the banner trusted. Now `is_canonical_mark(head, page_num)`, which is
  the rule the comment already claimed.
* **A canonical line now covers only its own `(check, scope)`.** It was mapped to
  `frozenset(Check)`, so the first line written to a two-clause page silenced the second:
  the page kept line one and lost line two for ever. `parse_canonical_mark` reads the pair
  off the line; the family is out of the blanket map, and an unparseable or wrong-page line
  covers nothing, which is the fail-closed direction. Idempotence is unchanged because
  `leading_mark_blocks` returns EVERY leading mark line, so once both are present each
  covers its own pair.
* **A published verdict that says CHECKED on a non-SUCCESS page falls back.** "Not SUCCESS
  means at least one clause" is the invariant the ticket rests on, and it must hold at the
  point of use rather than depend on resume rule (a) — a rule in another module that a
  future skip path could widen without touching `core/verdict.py`. The page now falls
  through to the fail-closed clause with a warning logged.

### Fable's P2s

The orphaned rule-(b) comment block in `pipeline/orchestrator.py` is removed (its content
lives at the assignment). The three surviving test-helper mutants — a whitespace-normalising
compare, `required` disabled, the clean-arm assertion dropped — are each pinned by a named
test and each now dies. The whitespace one matters because `_with_page_top_banners` does
`body.lstrip("\n")`, so a body that began with a blank line comes back without it: that is
a genuine byte difference the helper must REPORT rather than absorb, and the test pins both
directions. The gh317 fixture now says in the test that its fragment/verdict pair is not
one production writes, and why deriving the fragment through the seam would be the fixture
edit that hid a regression on #1142.

### Round-2 measurement

Suite **7,727 passed, 2 skipped, 4 xfailed, 0 failed**. ruff 0.16.0 format clean, 947 files.
**23 mutants, 23 killed**, each by a named test — the 16 from round 1 plus eight for the
round-2 changes, with several round-1 anchors refreshed where the new code moved them.
Corpus replay unchanged: 4,170/550/2,064 → 0/0/0, 0 CHECKED pages whose bytes changed, 0
SUCCESS pages demoted, 0 pages whose status moved. The replay now also reports the clause
mix, which is the first per-check picture of the corpus:

| check | pages |
| --- | --- |
| `native_table_verifier` | 1,153 |
| `equation_fidelity` | 1,118 |
| `native_coverage` | 1,080 |
| `table_structure` | 972 |
| `native_glyphs` | 883 (was 1,142 before the gate) |
| `page_judge` | 438 |
| `chart_region` | 410 |
| `unknown` (fail-closed) | 398 |
| `figure_text` | 388 |
| `source_evidence` | 37 |
| `table_label` | 8 |

`equation_fidelity` at 1,118 matches the design's measured count for the corrupt-math
hybrid class exactly, which is the independent check that round 1's `AUDIT_FAILED` fix
landed on the right population. The 398 fail-closed pages are the ones where socr still
cannot name its own reason; narrowing that number is future work, not a slice-1 goal.

## Merging main in: #1144 (PR #1158) and #1050 step 2 (PR #1161)

Two conflicts, both in files this ticket rewrites. #1144's and #1050's work is kept whole in
both.

**`benchmark/acceptance.py`.** #1144 landed on the pre-move copy of the mark grammar. All of
it is kept — in `core/marks.py`, where the grammar now lives. #1144's substantive change was
that `_scanned_flag_mark` must read the EMITTERS' own enumerations rather than listing the
two `SCANNED_*` templates by name: the lane now serves structure-class floors on
born-digital pages, whose banner says "unverified extraction", and all 138 pages #1144
recovers are born-digital, so the by-name version counted every one of them
`error_unmarked`. The moved family reads `PROSE_RECOVERY_FLAGS` and so covers all four
forms.

**`core/manifest.py`.** #1144 factored `_is_prose_recovery_body` out of
`_is_restored_prose_recovery` so the structure-class ending records a recovery from the same
evidence the restore path reads. #1145's skip-own-canonical-lines fix moved INSIDE that
shared helper rather than staying in the caller, so both paths get it over all four banner
forms. #1145 had declared its own single-member `PROSE_RECOVERED_FLAGS` /
`NATIVE_TEXT_FLAGS` for the same reason; #1144's two-member versions supersede them and the
duplicates are gone.

**#1050 step 2, checked rather than assumed.** Its recovered `(OCR, not verified)` block is
NOT in the grammar, so it cannot make a page count as MARKED — correct, since it says what
socr did to a block, not that a check did not pass. Its pages ship WARNING
`text_ink_unread`, which the clause table maps to `native_coverage=FAIL`, so they are
disclosed by a clause. And #1050's own mark, `[page N: uncovered text — see image]`, IS in
the grammar and covers `native_coverage`, so a page carrying it is suppressed and keeps its
bytes. Both halves are pinned, because getting either backwards is silent.

### Two things the merge broke, and what they were

* **A layering violation of mine.** To keep #1144's pin passing I re-exported
  `_scanned_flag_mark` from `benchmark/acceptance.py`, and
  `tests/test_package_layering.py::test_no_private_symbol_imported_across_packages` rightly
  refused it: a private name must not cross `socr.*` packages. The shortcut is gone, and
  #1144's pin now asks the same question through the meter's public surface
  (`families_of` / `line_is_mark`), which is what the meter itself calls.
* **#1050's leak test counted this ticket's mark as recovered content.** It collected every
  `>`-prefixed line as the recovered block, and the canonical unchecked line is quoted — so
  its own wording ("this page", `native_coverage`) was compared against the page's native
  vocabulary and flagged as text the reader saw. Marks are now excluded through PRODUCTION's
  `line_is_mark`, so a mark that stops being recognised fails the test rather than silently
  widening the exclusion. **Proved still load-bearing** by the `1050-mask-disabled` mutant:
  with #1050's native-text mask off, the echoing reader transcribes the page's own sentence
  and the test still fails.

Post-merge: suite **7,879 passed, 2 skipped, 4 xfailed, 0 failed**; ruff clean (953 files);
**25 mutants, 25 killed**; corpus replay unchanged at 4,170/550/2,064 → 0/0/0 with 0
CHECKED-byte changes, 0 demotions and 0 status moves.

## Known readability cost

On a page like P6 page 3 the fact "the native table region failed verification" is now stated
three times: #1129's banner, the canonical line, and the pre-existing mid-body D3 note.
Suppression reads the leading block only, by design §0.2/§2, because the meter asks that the
first rendered BLOCK be all marks. Accepted for slice 1; to be put to the owner as a polish
item.
