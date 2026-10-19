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

## The saving, from measured numbers only

| | pages | GPU-s each | per resume |
|---|---|---|---|
| all flagged pages (today) | 141,000 | 3.39 | **132.8 H100-h** |
| accepted — no longer re-read | 102,039 | 3.39 | **96.1 H100-h saved** |
| rejected — still re-read | 38,961 | 3.39 | 36.7 H100-h |

Inputs, none of them measured in this ticket:

* **141,000** prior-OCR pages: 52.8% of a reconciled 4,248-page stratified sample,
  extrapolated by decade over a ≈218,000-page Fed corpus (#1156's measurement).
* **3.39 GPU-s** per model-read page: measured earlier, over 111 documents.
* **55 of 76 accepted**: the 18 Oct pilot measured 21 of 76 flagged pages REJECTED (28%).
  76 pages is a small sample and the split scales the saving linearly, so read 96 H100-h as
  "the accepted share of 133", not as a number with two significant figures.

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

## Not measured

* No OCR run, no Ollama OCR, no GPU time: the laptop rule. Every number above is quoted
  from an earlier measurement, not produced here. (Four figure-DESCRIPTION calls did reach
  local Ollama from an early diagnostic probe, before descriptions were switched off in it;
  no OCR reading was ever requested.)
* The byte-identity of the resumed page (body AND whole sidecar) was measured on the Part A
  fixture under one config — no dual-pass tables, no escalation, no equations, descriptions
  off. It is not a claim about every config.
* The accepted/rejected split, the 141,000 and the 3.39 GPU-s are all inherited numbers.
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
