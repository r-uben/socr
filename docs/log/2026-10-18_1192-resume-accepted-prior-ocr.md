# #1192 — an ACCEPTED earlier-OCR page is not re-read on every resume

Scope given by the owner: the half of #1192 that needs no ruling. The REJECTED half (the
fail-closed floor: page image plus mark) keeps re-reading on every resume, exactly as
today, pending his decision. The number he asked for is in "What is left" below.

## The defect

#1156 Part A flags a page whose text layer is a PREVIOUS OCR run's reading rather than the
document's own characters. Such a page is never native-trusted and is routed to a model.
When the judge ACCEPTS that reading it ships, and the page is demoted to WARNING carrying
one verdict clause, `prior_ocr_layer=UNKNOWN` ("earlier OCR, not re-read") — not because a
check found anything wrong with it, but because one reading of a scan is one reading and
the page must be MARKED rather than CHECKED.

The ledger (`_load_terminal_page`) admits only a terminal SUCCESS page with
`audit_passed=True`. A flagged page is ALWAYS WARNING. So every flagged page was re-OCR'd
on every resume, for ever — the precise non-convergence #713's and #1145's exceptions were
written to prevent ("re-OCR'd on every resume forever … so the work never converges"),
reached through a third door.

Measured at `process()` on the Part A fixture before the change: a run, a crash that loses
page 2, a resume — and page 1's accepted reading is read again.

## What ships

A FOURTH deliberate exception, `_resume_prior_ocr_reading_was_accepted`, admitting a page
only on positive evidence of that one outcome. Each clause and the single thing it rules
out (all nine are pinned by a mutant):

| clause | what it rules out |
|---|---|
| `prior_ocr_layer_suspect is True` in the sidecar | a page outside this family |
| …and THIS run's analysis agrees | a STALE suspicion locked in for ever (below) |
| `prior_ocr_layer_text` present, non-empty | an incomplete sidecar; Part B re-finalizes from exactly this key |
| `audit_passed is True` | the REJECTED half — the floor is `audit_passed=False` by construction |
| engine not in `_NATIVE_TEXT_LANES` | a `native`/`chart_asset` winner, whose text IS the suspect layer |
| `status` exactly WARNING | anything worse than the demotion |
| `failure_mode` exactly `PRIOR_OCR_LAYER_UNVERIFIED` | a page demoted for some other reason as well |
| verdict clauses == exactly `[(PRIOR_OCR_LAYER, UNKNOWN)]` | a table left unjudged or a figure left unread riding along |
| `table_ladder_incomplete` false | GH-359's forfeiture: a table assemble had to backfill was never witnessed |

The floor is excluded by TWO independent guards: `audit_passed` fails, and the
failed-marker-body refusal fires (bypassed only for the table-WITHHELD exception, not this
one).

The restore side needed nothing: #1156 round 1 had already restored
`prior_ocr_layer_suspect` and `prior_ocr_layer_text` for a resume it described as
unreachable ("the page ledger admits only a terminal SUCCESS page and a flagged page is
always WARNING"). This ticket makes that path live; both comments were corrected to say so,
and the guarantee is now pinned on the live path at `process()` as well as at the seam.

### The stale-suspicion clause, which came out of the design review

`prior_ocr_layer_suspect` is a UNION that folds the detector's FAILED SCANS in beside its
positive findings (`state.py`: `prior_ocr_layer_scan_failed`), and the flag is not in the
run fingerprint. So a run that could not decide demotes the page, and every other condition
still matches on the next run — which CAN decide and may find nothing wrong with it. The
three sibling guards a few hundred lines below (`minus_as_digit_suspect`,
`garbled_math_suspect`, `table_not_reconstructed_suspect`) all cross-check the FRESH
`PageState` for exactly this reason. This exception needs that check more than they do, not
less: the flag is what CREATED the demotion being carried forward. Pinned as a difference
on the analysis alone — one admissible sidecar shape, two documents, one flagged and one
not.

## A pre-existing defect this made reachable, fixed here

`_describe_and_embed_figures` appended each page's figure blocks UNCONDITIONALLY. A page
restored from the ledger arrives at assemble with those blocks already in its body (the
previous run wrote them into the fragment, and the fragment is what assemble reads back),
so every partial resume added a SECOND copy of each block, and of its description.

Not a defect of #1192, and measured that way before being fixed: it reproduces on the
ORDINARY terminal-SUCCESS gate with no flagged page involved — 1 figure ref before the
resume, 2 after, growing by one per resume. The sibling phase in the same method already
had this guard (`insert_descriptions` "skips refs that are already described"), which is
why the duplication showed in the blocks and not in the descriptions.

Fixed here rather than filed because #1192 widens the set of pages that reach it from "a
partial resume on a page that happens to be SUCCESS" to every accepted earlier-OCR page —
~102,000 on the Fed corpus — and shipping a duplicated figure block at that scale is
content loss of the kind the citation corpus cannot take. The guard keys on the IMAGE REF,
the one part of a block that identifies the asset (the header carries a description a later
run can legitimately change), and filters per figure, so a page that gained a NEW figure
still gets that one embedded beside the ones already there. One shared helper
(`_figure_ref_for`) now computes the ref for both writing and recognising it, so the two
cannot drift. **It is a separable change and can be split into its own commit.**

## The saving, measured through the predicate itself

The first version of this section computed the saving from the pages the JUDGE accepted.
r-uben's round-1 review (should-fix 2) pointed out that is not the same set: the predicate
also requires EXACTLY ONE verdict clause, and he showed at `process()` that an accepted
reading on a flagged page carrying a markdown table ends `table_unverified`,
`audit_passed=False`, with clauses `[prior_ocr_layer, source_evidence]` — so it keeps
re-reading, exactly as on main.

So the count was redone by calling `_resume_prior_ocr_reading_was_accepted` on the 18 Oct
pilot's real sidecars (`fed-repilot-1018`, 185 pages over 8 documents;
`scratch/1192/measure_pilot.py`), with `flagged_by_this_run=True` — what a resume of the
same input under this build sees. No reimplementation of the rule:

| | pages |
|---|---|
| pilot pages with a sidecar | 185 |
| flagged (prior OCR layer) | 76 |
| **ADMITTED by the predicate** | **55 (72.4% of flagged)** |
| refused | 21 |

All 21 refusals are fail-closed FLOORS — 20 by #1156's own route
(`fail_closed_marker`/`prior_ocr_layer_unread`), 1 by #1027's `invisible_scan_unread` — so
every refusal is a page whose reading was refused, which is the half held back on purpose.
**Zero pages are lost to the one-clause requirement:** 75 of the 76 flagged pages carry
exactly one clause, and the single page with two (`native_coverage=FAIL` beside
`prior_ocr_layer=UNKNOWN`) is already among the floors.

| | pages | GPU-s each | per resume |
|---|---|---|---|
| all flagged pages (today) | 141,000 | 3.39 | **132.8 H100-h** |
| admitted — no longer re-read | 102,039 | 3.39 | **96.1 H100-h saved** |
| refused (floors) — still re-read | 38,961 | 3.39 | 36.7 H100-h |

So 96.1 H100-h is the figure the predicate actually buys on this pilot, not an upper bound
on it. **But the pilot is a thin basis for the table claim specifically.** 3 of the 55
admitted pages do carry a pipe table and were admitted anyway — on a suspect layer Part A
makes the table judges ABSTAIN rather than fail, so no second clause is published — while
r-uben's `process()` fixture did produce `source_evidence=UNKNOWN`. Three pages cannot
settle which of those two behaviours the Fed corpus will show, and the pilot is 1951–1957
material, lighter in tables than later decades. Read 96.1 as measured on this slice, and
expect it to fall if later decades publish a second clause on their table pages.

**This is a caveat, not an open ticket (owner, 2026-10-19).** The full Fed run's
calibration — 70 documents across all decades, running as this lands — measures it: rerun
`scratch/1192/measure_pilot.py` against its sidecars for the corpus-wide admitted share.

Inputs not measured in this ticket:

* **141,000** prior-OCR pages: 52.8% of a reconciled 4,248-page stratified sample,
  extrapolated by decade over a ≈218,000-page Fed corpus (#1156's measurement).
* **3.39 GPU-s** per model-read page: measured earlier, over 111 documents.

The admitted SHARE (55/76) is now measured rather than assumed, but on 76 pages; it scales
the saving linearly.

The saving is per RESUME and only for pages already read; it does not reduce the first
run's cost (≈47 H100-h once, measured over 111 documents).

## What is left — the owner's pending ruling

**The rejected half: 21 of 76 flagged pages (28%) in the 18 Oct pilot, ≈38,961 pages and
≈36.7 H100-h per resume at Fed scale.** Those pages keep re-reading. The argument for
leaving them is in #1192: their reading was REFUSED, and another rung, model or prompt
might yet read them, which is not true of a reading already accepted. The argument against
is that nothing has ever been measured to change that verdict. What would settle it is the
first full Fed run: how many rejected pages a different rung rescues. If none ever does,
`PRIOR_OCR_LAYER_UNVERIFIED` can join the #1145 treatment and this exception's
`audit_passed` clause relaxes; if some do, the re-read is the price of the rescue.

## Round 1 (PR #1210: Grok MERGE, Argon MERGE-AFTER-FIX, r-uben MERGE once should-fix 1)

Third commit on the branch. r-uben could not get the exception to admit a page that should
be re-read; all three seats' remaining findings were either fixed here or filed.

* **The per-figure filter is now pinned** (r-uben should-fix 1). The guard's comment claimed
  it filters per FIGURE so a page that gained a new one still gets it; nothing tested that,
  and collapsing it to a per-PAGE skip left all 644 tests in the 30 related files passing
  while DROPPING the second figure — silent loss, worse than the duplicate the guard fixes.
  His refs for (figure 1, figure 2): main `3bcc43d` (2, 1); this PR (1, 1); per-page mutant
  (1, **0**). Pinned by a two-figure arm that cuts one block out of the restored fragment,
  and the per-page mutant is now one of the twelve.
* **The saving is measured** (should-fix 2) — see above.
* **The clause comparison keeps its RESULT** (r-uben surviving mutant): comparing the check
  alone admitted `prior_ocr_layer=FAIL`, and FAIL means a check DID decide and found
  something wrong, which a later run may fix. Grid row added.
* **A record naming no engine is refused** (r-uben nit, Grok P2): `PageOutput` defaults to
  `engine=""` beside `audit_passed=True`, so an empty string read as "a model read this".
  Two grid rows (empty string, missing key).
* **Nits:** D1b is no longer called "the ONE deliberate exception"; the three sibling
  suspect guards in the same function now ask `engine_ships_native_text` rather than
  respelling the lane tuple (each of the three proved by its own existing tests — a mutant
  reverting each one is killed by `test_gh913_minus_as_digit.py`,
  `test_gh990_control_byte.py`, `test_gh960_garbled_math.py`,
  `test_gh994_flattened_table.py`); and the figure test now reads the ledger's own decision
  through a `_load_terminal_page` spy instead of `assert 1 not in pages_read`, which could
  not fail on a fixture whose pages take a native lane.
* **#1207 coordination** (Argon §5): a comment at the restore line records that whichever
  of the two PRs merges SECOND takes #1207's `ps.prior_ocr_layer_text` version. No
  behaviour change here.
* **Filed, not fixed** (`leads/1192/issue-followups.md`): the ditto guard blocking resume
  for ever and naming `unknown=UNKNOWN` (Argon §2, mechanism confirmed by reading the guard
  chain); the doc-global figure counter shift (Argon §4a) and its general case, a changed
  ref (Grok); `describe_figure` called on figures whose blocks are dropped (Argon §4b);
  #1145's equation route missing the fresh-analysis cross-check (Argon §1); and GH-189's
  duplicate suppression looking inert on one image (r-uben). Every one is a missed saving or
  a duplicated block — none is a wrong or a lost number.

## Not measured

* No OCR run, no Ollama OCR, no GPU time: the laptop rule. Every number above is quoted
  from an earlier measurement, not produced here. (Four figure-DESCRIPTION calls did reach
  local Ollama from an early diagnostic probe, before descriptions were switched off in it;
  no OCR reading was ever requested.)
* The byte-identity of the resumed page (body AND whole sidecar) was measured on the Part A
  fixture under one config — no dual-pass tables, no escalation, no equations, descriptions
  off. It is not a claim about every config.
* The 141,000 and the 3.39 GPU-s are inherited numbers. The admitted share (55/76) is
  measured, on 76 pages of 1951-57 material; whether a flagged TABLE page publishes a
  second clause on table-heavier decades is not settled by the 3 such pages this pilot has,
  and the full Fed run's 70-document calibration is what will settle it.
* Whether a different rung would change a REJECTED page's verdict: unmeasured, and the
  reason the other half stays out of scope.

## Design review (Gemini 4 Argon, before and after the build)

One real finding, taken: the fresh-analysis cross-check above. It also made several
specific claims that are FALSE against this tree and are recorded here so they are not
re-raised — `dualpass_crop_timeout`, `dualpass_crop_failed` and
`table_row_repetition_truncated` ARE in `_RESUME_REPLAYED` (it claimed they are dropped on
resume, flipping `tables_trust.json` from distrusted to trusted); `figure_crop_unreadable`,
`fabricated_url_stripped` and `src/socr/tables/trust.py` do not exist anywhere in the
source. The advisor has no file tools, so its line citations are reconstructions; the
substance was checked independently of the quote, which is what found the one that held.

Its remaining findings (a transient dual-pass or escalation failure leaving a page accepted
with no clause and no retry latch) are REAL shapes but not holes this exception opens: a
non-flagged page in that state is `SUCCESS` with `audit_passed=True` and no clause, so the
ORDINARY gate already resumes it today. This exception admits a page under a strict
SUPERSET of the ordinary gate's protections — it additionally requires the single clause,
the mode, the acceptance, the retention and the fresh analysis — so every such shape it
admits is one main already admits for every other page. Worth its own ticket; not this one.
