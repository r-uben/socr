# P2 — an exhausted ladder on a structure-class page ships the fail-closed floor

2026-09-01. Programme item **P2** of `docs/log/2026-09-01_conceptual-revision.md`.
Closes the substantive half of **#317**. Built by a multi-model troupe run
(`.troupe/runs/20260901-211136-standard-feature/`, tasks t0-t9) in the worktree
`socr-revision` on branch `docs/conceptual-revision-2026-09`.

## What shipped

Before: a born-digital page carrying a table is excluded from the free native lane
and enters the cost-ordered provider ladder. If every rung was refused, case (iii)
of `_select_page_output_tagged` shipped the **native reconstruction** — the grid
built from PDF word geometry — as `WARNING` / `audit_passed=False`. That grid is
the reading socr already measured to be the worst of the three
(`docs/log/2026-08-30_model-vs-native-table-rows.md`: invented+missing native 21,
qwen 14, gemini 7), authored by the same geometry the verifier used to refuse the
models.

After: that page ships the **fail-closed floor** — the standard
`[page N failed: unverifiable table — see image]` marker plus the rendered page
PNG, with native prose outside the table regions retained by the existing
`splice_all_table_regions`. The native grid is not in the shipped bytes.

The pieces:

| Where | Change |
| --- | --- |
| `core/result.py` | `FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED` added |
| `core/manifest.py` | `structure_class_floor_text(p, page_num)`; case (iii) returns the floor; `WinnerKind.STRUCTURE_CLASS_NO_GRID` → `STRUCTURE_CLASS_FLOOR`; `structure_class_native_fallback_applies` → `structure_class_floor_applies` |
| `pipeline/orchestrator.py` | floor PNG rendered in `_phase_agentic` before the flush; provisional fragment derives from the floor text; assemble bucket renamed; new audit event; new document-level note; CLI line turned red |
| `core/audit_log.py` | event kind `structure_class_native_fallback` → `structure_class_ladder_exhausted_floor`, same rank 6 |
| `core/tables_trust.py` | same rename inside `TABLE_DISTRUST_KINDS` |

The marker string is **reused**, not minted: `is_page_failed_marker`,
`_PAGE_FAILED_ANY_RE` and every downstream consumer already match it. What
distinguishes this floor from TR-3's is the failure mode, the audit event and the
bucket, not the text.

The PNG is rendered at the in-loop seam, before the per-page fragment flush.
Rendering it in `_phase_assemble` would have made the `terminal:false` fragment
(no ref) differ from `_rewrite_all_fragments` (ref present) and tripped the PP-1
byte-identity guard.

## The ERROR-vs-WARNING choice

The floor page ships `PageStatus.ERROR`, `audit_passed=False`,
`FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED` — **the same page status the
pre-existing D3 floor uses**. WARNING was available and was rejected: it would
have made this the only fail-closed path in the codebase that is not ERROR, and
the whole point of P2 is that this page failed, it did not degrade.

`audit_passed` was **not** touched on any stored attempt. It is the
winner-selection flag (#252); flipping it on `p.best_output` discards the page.
The demotion is carried by status and failure mode, exactly as the D3 floor does.

Document status follows the standing rule already in `_phase_assemble`:
**AUDIT_FAILED when some page still carries text, ERROR when none does.** Both
outcomes are real and correct:

- The CE-like parity fixture floors its table regions but keeps its prose, so the
  document is `AUDIT_FAILED`.
- The GH-147 rotated-grid fixture is one page whose native layer holds no
  parseable table block, so the floor falls back to the whole-page marker, no page
  carries text, and the document is `ERROR`.

This required *not* excluding floor pages from `failed_pages`. The task plan's t4
asked for that exclusion to avoid a double count; it was deliberately not taken,
because a whole-page floor genuinely produced no usable output and the document
status rule above depends on it being counted. A floor page that keeps prose is
not a `page_failed` page; a whole-page floor is. The two buckets answer different
questions and are allowed to overlap on that one shape.

## Endings and buckets deleted

- The case-(iii) native `PageOutput` construction from `_native_text_with_appends`
  is gone. That ending now builds the floor.
- The `legacy_table_defect` re-derivation inside case (iii), which chose between
  `NATIVE_TABLE_STRUCTURE_FAILED` and `STRUCTURE_CLASS_NO_MODEL_ATTEMPT`, is
  deleted. `grep -n 'legacy_table_defect' src/` is empty. Every page taking this
  ending now carries exactly one failure mode.
- `structure_class_native_fallback` is gone as a name: zero hits across `src/` and
  `tests/`. The `docs/log/` history keeps it, as history.
- The cascade is still 15 returns and zero loops.
  `test_tags_and_endings_are_in_bijection` (`tests/test_r7_winner_kind_tags.py`)
  pins that every `WinnerKind` member is produced by exactly one ending, which is
  a stronger reachability guarantee than enumerating fixtures: no mirror bucket
  can survive that no ending reaches.

## `STRUCTURE_CLASS_NO_MODEL_ATTEMPT` — kept, deprecated

**Kept.** `result.py` parses `FailureMode(d.get(...))` with no fallback, so
deleting the member would raise `ValueError` the moment an older sidecar is read
from a cache written before P2. It is marked deprecated / deserialisation-only in
place, and `test_pre_p2_compatibility_failure_mode_deserializes` pins that a
sidecar carrying the old string still round-trips. No modern run can produce it.

## Resume

Confirmed by test, not assumed. `_load_terminal_page`'s gate is SUCCESS-only, so a
floored page returns `None` and is reprocessed. Two further pins:

- The floor does **not** qualify for the one deliberate skip exception (GH-353
  TICKET-D1b), which is keyed off `table_ladder_disposition == TABLE_REJECTED`,
  not off the failure mode.
- A **pre-P2** sidecar (WARNING / `audit_passed=False` /
  `native_table_structure_failed` or `structure_class_no_model_attempt`) also
  returns `None`, so a page that previously shipped the native grid is re-OCR'd
  rather than restored.
- `structure_class_model_kept_on_resume` is not set by the floor path, so a
  resumed floored page cannot re-enter the branch as a model-kept page.

## Retargeted tests, and the one deliberate trade

`tests/test_s1_structure_class_winner_gh_reachability.py`,
`tests/test_tr3_d3_floor.py`, `tests/test_manifest_agentic.py`,
`tests/test_gh259_flagged_model_table_wins.py` and `tests/test_orchestrator.py`
were retargeted to the floor, each keeping its original guarantee and gaining an
assertion that the native grid bytes are absent. Two are worth naming:

**`tests/test_table_repair_parity.py::TestEndToEndParity`.** This measured TR-2
cell parity on the assembled markdown of a fixture whose only rung is a stub that
accepts nothing — precisely the ladder-exhausted shape. Under P2 the grid is
withheld, so parity on the shipped bytes is gone by design. The measurement was
moved, not weakened: the same `assert_table_parity` over the same ground truth now
runs against the page's own native reconstruction, captured from the live
`DocumentState`, and still passes cell for cell including separate grids and
reading order. Added on top: the marker must be in the shipped document and the
row labels must not be.

**`tests/test_landscape_refusal_a2_gh147.py::TestHermeticProcessRefusal`** (renamed
`..._emits_refusal_event_and_fails_closed`). Its document status moved
`AUDIT_FAILED` → `ERROR` per the rule above.

The deliberate trade, recorded so it is not later mistaken for a regression: this
fixture's rotated page yields a native layer with **no parseable markdown table** —
the rotated cells come out as bare reading-order lines. `splice_all_table_regions`
therefore cannot isolate a table region and the floor falls back to the whole-page
marker, exactly as the pre-existing D3 and GH-90 floors already do on that shape.
The three prose lines GH-147 originally pinned do **not** survive on this fixture.
That is content withheld, not content lost silently: the page ships ERROR with a
`page_failed` event, the floor event, a document-level note in `metadata.json`, a
red CLI line and a rendered page PNG from which a human recovers the page. Shipping
the raw shredded cell sequence instead would have shipped exactly the kind of wrong
reading a citation corpus must never carry. The spec permits this: point 3 says the
floor keeps prose *if the existing floor path already does so*, and here it does
not, and inventing a new region-splicing mechanism was ruled out.

## Gates

Full suite and the blocking format gate were run in this worktree with
`PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-revision/src` and
`uvx ruff@0.16.0 format --check .`. Exit codes checked directly, never through a
pipe. Results are in the run report accompanying this change.

## Left for later

- A floored page pays the full ladder again on every resume, because
  reprocess-on-doubt is the load-bearing rule and it wins. Worth a follow-up
  ticket alongside GH-353's `TABLE_REJECTED` exception, not worth weakening the
  gate for.
- `_render_d3_floor_png` renders a full page. On a page whose table is a small
  fraction of it, the floor's PNG is coarser than a region crop — but the region
  bboxes here were never verified, the same reasoning TR-3 used. Noted so it is
  not mistaken for an oversight.
- P2 changed only what case (iii) *ships*, not which pages reach it. Widening the
  free-lane exclusion from `has_tables` to any structure signal is P4.

## Cold review round 1

An independent cold review returned NOT MERGEABLE with two blocking findings and
one should-fix. Both blocking findings were **reproduced with a failing test
before any production line was touched**; neither was accepted on the reviewer's
description alone.

### Finding 1 — a mixed-validity multi-table page still shipped an unverified native grid (round-1 fix SUPERSEDED, see round 2)

**Reproduced: yes.** `manifest.py:827`. `structure_class_floor_text` accepted any
non-empty result from `splice_all_table_regions`, which proves only that it
replaced every block *its own parser could find*. That is not coverage.

`test_mixed_validity_page_does_not_ship_the_collapsed_native_region` builds a page
with two recorded native table regions where reconstruction emitted one as valid
GFM and collapsed the other to ragged lines, then exhausts the ladder. Against the
pre-fix code the splice succeeds on the parseable region and the collapsed one
rides out with it: the assertion fails on the token `0.62`, a number that would
have entered the corpus unverified. That is exactly what P2 exists to prevent.

**Fixed** in the direction the reviewer named. The floor now proves coverage
against the per-region verifier's own enumeration, `native_table_region_count`
and `native_table_region_identities` (GH-371/GH-375), by calling
`splice_failed_table_regions` with every ordinal failed and the recorded
identities. That helper already fails closed to `None` on any count or identity
divergence. A short or absent identity list is not evidence of coverage either,
so it takes the whole-page floor. This is the same evidence and the same
fail-closed direction the D3 ending already uses, and no new mechanism was
invented.

Blast radius checked before committing to it: a real born-digital page carries
the metadata. The CE-like parity fixture records two regions whose identities
match the two parsed blocks 1:1, so it still takes the regional splice and still
keeps its prose. `test_fully_parseable_page_still_keeps_its_prose` pins that
difference, so the fix cannot silently degrade into flooring every page. The
GH-317 fixture builder was updated to populate the same metadata, because a page
state without it was never a realistic page.

Six existing tests then failed, in `test_s1_structure_class_winner_gh_reachability.py`
and `test_gh259_flagged_model_table_wins.py`, all for the same reason: their
hand-built page states set `native_text` and `has_tables` but recorded no
regions, so the floor could not prove coverage and took the whole-page marker,
dropping the prose those tests assert. The three fixture builders now derive the
region enumeration from their own fixture text, exactly as the GH-317 builder
does. That is fixture realism, not an assertion relaxed: every prose and
grid-absence assertion in those tests is unchanged, and the CE-like parity
fixture confirms independently that a real page carries this metadata.

The trade this fix makes explicit: a structure-class page whose per-region
verifier never ran carries no region enumeration, so its floor is whole-page and
its prose does not survive. The GH-147 rotated fixture is one such page. That is
the fail-closed direction, and it is the same one the D3 ending already takes
when isolation is unprovable.

### Finding 2 — the D1b `TABLE_REJECTED` resume exception could restore a floored page

**Reproduced: yes.** `orchestrator.py:5330`. GH-353 TICKET-D1b lets a page whose
table ladder ended in `TABLE_REJECTED` bypass resume's SUCCESS and `audit_passed`
gates. The only guard left after that bypass is `is_page_failed_marker`, which
deliberately returns `False` for a marker surrounded by preserved prose.

`test_floored_page_is_reprocessed_even_when_the_ladder_said_table_rejected` builds
the exact combination the previous coverage never did: a
`STRUCTURE_CLASS_LADDER_EXHAUSTED` winner, a `TABLE_REJECTED` disposition,
`table_ladder_incomplete=False`, and a prose-preserving regional floor body. It
asserts its own premises, including that the body is *not* a whole-page failure
marker, so there is no doubt about which guard is being tested. Against the
pre-fix code `_load_terminal_page` returns the floored page verbatim: it is never
re-OCR'd, and P2's reprocess-on-doubt promise was void for precisely the pages an
enabled table-judge ladder produces.

**Fixed.** The `winning_output` dict is now read before the exception is decided,
and the exception no longer applies when the shipped failure mode is the floor.
D1b is a judgment about a *table*; the floor is a page-level fail-closed
disposition and is never skippable. The fix is narrow, and
`test_d1b_exception_still_works_for_a_genuinely_rejected_table` pins the
difference: same disposition, same completeness, only the winner's failure mode
differs, and a genuine `TABLE_REJECTED` winner still skips and is kept.

### Finding 3 — retargeted `process()` tests missed the hermeticity contract

**Not a defect to reproduce; contract compliance.** Added
`_resolve_judge_model` → `""` to the landscape refusal and parity tests, and both
`_available_engines_for_agentic` and `_resolve_judge_model` to the retargeted
orchestrator test, which previously patched only `get_engine`. They passed
locally without these because they select the heuristic judge, but
`_phase_judge_hard_pages` builds an `OllamaVisionJudge` and POSTs to it
regardless of `judge_backend`, so the omission was a live CI trap rather than a
style point.

## Cold review round 2

Findings 2 and 3 closed. Finding 1 still open, and the reviewer was right: the
round-1 fix was circular.

### Why the round-1 coverage check did not work

It proved coverage against `native_table_region_count` /
`native_table_region_identities`. Those are not independent evidence:

- `extract_structured` builds `table_regions` **only from successful
  reconstructions** — a region enters the list only when `_table_to_markdown`
  returns something truthy, or when the rowizer produced a grid
  (`born_digital.py:1939-2003`).
- `_verify_regions` then walks that list, **skips any region with no Markdown
  separator**, and only afterwards increments the count and records the identity
  (`born_digital.py:2186-2192`, published at `:2215-2217`).

So a detected table whose reconstruction failed never enters the enumeration at
all. The recorded count equals what `find_table_blocks` sees in the assembled
text, the check agrees with the very parser it was meant to audit, and the
collapsed sibling ships. The round-1 reproducer only appeared to prove otherwise
because it injected a count of 2 and a synthetic second identity by hand, which
production never records for a non-GFM sibling.

### The search for an independent signal — result: none exists

The ruling was that coverage must be proven against a signal recorded **before**
reconstruction, or the splice goes. What is actually there:

| Candidate | Where | Verdict |
| --- | --- | --- |
| `page.find_tables()` in `_detect_tables` | `born_digital.py:1766-1804` | Reduced to a bool, count discarded, never stored |
| `page.find_tables()` in `extract_structured` | `born_digital.py:1926-1959` | Consumed inline, filtered by reconstruction success |
| `PageAssessment.has_tables` | `born_digital.py:815` | A bool |
| `native_table_region_count` / `_identities` | `born_digital.py:840-845` | Post-reconstruction, the circular signal |
| Any bbox / rect / region list on `PageAssessment` or `PageState` | swept both dataclasses | None exists |

No detection-level table count or bbox list is recorded anywhere. Adding one
means changing the detector, which is outside P2's scope (the spec confines P2 to
control flow in `manifest.py` / `orchestrator.py`).

### What shipped: branch 3, the splice is gone

`structure_class_floor_text` now returns the marker plus the PNG ref and nothing
else. It does not read `native_text` at all.

The property that replaces coverage is stronger precisely because it needs no
enumeration to be trusted: **no byte of the native layer can reach a floored
page**, whatever reconstruction did or did not manage to parse. There is nothing
left for a future parser change to invalidate.
`test_floor_text_is_independent_of_the_native_layer` pins it across five native
layers including the mixed-validity one, and
`test_region_metadata_cannot_reopen_the_splice` fails if anyone reinstates the
circular check.

### Known limitation, and the follow-up it needs

**Native prose on a floored page is withheld.** Before P2 that prose shipped;
in P2 round 1 it survived the regional splice; now it does not. The page ships
ERROR with the marker, the page PNG, a `page_failed` event, the
`structure_class_ladder_exhausted_floor` event, a document-level note in
`metadata.json` and a red CLI line, so the loss is surfaced at every level and a
human recovers the page from the image. It is a real cost and it is accepted
deliberately, because the alternative on the evidence available is shipping a
collapsed grid inside text labelled "preserved prose".

**Follow-up:** record a detection-level table count (and ideally the bboxes) on
`PageAssessment` at `_detect_tables`, before reconstruction runs. With that
signal the regional splice can return: allow it only when the number of parsed
GFM blocks equals the detection-level count. That is a detector change and wants
its own ticket.

**Closed 2026-09-03 (GH-520).** `PageAssessment.detected_table_count` /
`detected_table_bboxes` landed in #570, taken from `find_tables()` at detection
time and pinned as independent of reconstruction. `structure_class_floor_text`
splices again behind a three-part guard: at least one table detected, every
detected table carrying a usable bbox, and the parser finding exactly that many
blocks. Anything else floors whole, so the default in this document is still
what a page without the signal gets.

Two things the guard does NOT claim, stated here because the limitation above
was believed for a day longer than it should have been:

- The correspondence between block *i* and bbox *i* is by document order. A
  markdown block carries no geometry, so nothing verifies that block *i* is the
  text of bbox *i*. What equal counts establish is the property round 1 lacked
  -- that no detected table is missing from the parser's block list, so the
  collapsed sibling cannot be the one left behind.
- A borderless table seen only by the lane-cooccupancy pass contributes no bbox
  and is not counted at all, so `detected_table_count == 0` is common on
  exactly the pages where the parser is least trustworthy. Zero is treated as
  no evidence, never as "no tables to cover".

### Second-order consequence, pinned rather than papered over

A document whose every page floors carries no text, so `_phase_assemble` writes
no `<stem>.md` and `_rewrite_all_fragments` (which splits the final text) writes
no fragments. That is pre-existing behaviour for an all-failed document, but the
whole-page floor makes it reachable for a document of table pages. The
byte-identity and paired-process fixtures therefore carry a **second, clean prose
page**, which is both the realistic shape and the only one in which the
byte-identity guard has anything to compare. Page two's prose surviving intact is
also what pins the floor's blast radius to its own page.

### Round-1 fixture changes: reverted

The three fixture builders that derived `native_table_region_count` /
`native_table_region_identities` from fixture text are reverted. The floor no
longer consults that metadata, so they were doing nothing, and the reviewer's
objection stands that deriving them with the same parser that later validates
them is not evidence of anything.

### Tests retargeted, none relaxed

Twelve tests across four files moved from "prose survives the regional splice" to
"nothing native ships". Each keeps its original guarantee and states the new one:

- `test_gh317_structure_class_floor.py` — prose-survival tests inverted; the
  mixed-validity reproducer rebuilt with no injected metadata; the byte-identity
  and paired-process fixtures given a clean second page.
- `test_s1_structure_class_winner_gh_reachability.py` — the case-(iii) prose and
  appended-sidecar tests inverted, with the section header rewritten to say why.
  The GH-211 prose guarantee on every OTHER ending is untouched, which is the
  difference that file now pins.
- `test_gh259_flagged_model_table_wins.py` — the three "must fall back to native"
  tests keep their real subject, that the model's empty, prose-only or
  proved-wrong output does not win the selection, and now assert the floor's
  bytes rather than native's.

The finding-2 resume test needed care to stay honest: with a whole-page floor its
body would be a page-failure marker, and `_load_terminal_page`'s marker check
would refuse the resume for a reason unrelated to the D1b guard, making the test
pass vacuously. It now builds the artefact a round-1 build would have written —
the floor's failure mode with a prose-bearing body — and asserts that body is not
a whole-page marker, so only the D1b guard can refuse it. Verified live by
removing the guard and watching the test fail.
