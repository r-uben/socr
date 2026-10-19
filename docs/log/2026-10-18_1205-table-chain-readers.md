# #1205 — the table chain's remaining readers of a suspect text layer

Built on `origin/main` `3bcc43db` (#1202 / Part A merged). Processor only: no OCR run, no
Ollama, no model call anywhere in this ticket, so nothing here is a claim about what a
model does. The design note and its probes are at
`scratch/routing-baseline/leads/1205/{design.md,notes-1205.md,*.py,*.txt}`; this ticket's
own probes are in `scratch/1205/`.

## What ships

Every reader in the table chain that decided something from the page's own text layer now
abstains when that layer may not serve as evidence, on ONE predicate:
`UnifiedPipeline._table_layer_is_evidence`, which is `layer_is_admissible_evidence` — the
UNION, scan-failed included.

| reader | site | what it decided with the layer | now |
|---|---|---|---|
| `_table_page_needs_escalation` (both callers: `_surface_table_scoring`, `_escalate_table_page`) | `orchestrator.py:6811` | `native_rows_from_page` → `score_page`: escalate or not, and the `table_not_scorable` / `table_unexplained_lanes` events that drive `tables_trust` | abstains BEFORE the parse; same ending as a page with no scorable grid, and the absence is SURFACED as `table_not_scorable` |
| `decide_escalation` → `score_page` ×2 and `judge_escalation`'s value oracle | `escalation_decision.py:121`, guarded at `orchestrator.py:7212` | the layer as GROUND TRUTH for the exactness gate and as the canary's oracle: the path where a correct `-350` loses to the layer's `350` | the candidate is refused on the record (`gate=none`), the incumbent ships, and the page is marked as not having received its escalation |
| `page_needs_crop_repair_fallback`, `crop_patch_improves_verification` → `verify_native_table` | `crop_repair.py:93,144`, guarded at `orchestrator.py:14254` | the trigger for a crop repair and the credential for a patch that CHANGES numbers already accepted by the judge | the fallback does not run; the accepted bytes ship and the page records `table_crop_repair_unverifiable`, a new kind registered in `TABLE_DISTRUST_KINDS` |
| `prepare_table_witnesses` → `_corroboration_contradicted_indices` → `_native_numeric_multisets` | `witness.py:212,399,436` | demotes a count-matched pairing to AMBIGUOUS with NO crop, so nothing can look at that table and it reaches the ladder UNVERIFIED | the page is not opened; empty evidence is NEUTRAL by construction and the index pairing ships with its crops |
| `NativeTableVerifierJudge.assess` → `verify_native_table`, `_maybe_repair_collapsed_headers` | `agentic.py:1301,1626` | rejects a reading that disagrees with the layer, and ACCEPTS one that matches it through `EXACT_PASS`, an exit that consults no judge | split by the two keys — see below |
| `_record_unresolved_binding_boundary`'s re-read | `orchestrator.py:7994` | "resolved" proof for a standing boundary distrust | unchanged (#1202 gated it); **pinned here**, which it was not |

## The two #1156 keys, and the direction round 1 got wrong

Part A's split is by ACTION, not by site, and the first version of this ticket keyed every
reader on the wrong half of it:

* **EVIDENCE takes the UNION** (`layer_is_admissible_evidence`, `prior_ocr_layer_suspect`,
  scan-failed included). Refusing evidence is never destructive — the reader behaves as it
  does on a page with no text layer, a state every one of them already handles — and
  "unknown" is no reason to trust the words either.
* **WITHHOLDING the layer's own text takes the POSITIVE finding** (`prior_ocr_layer_found`).
  That is #1156's withholding floor and it is not this ticket's business.

Round 1 gated all four sites on the positive finding. Argon's review (P1) found the
consequence and the owner ruled on it: a page whose DETECTION SCAN FAILED still graded a
correct reading down, promoted the candidate that matched its own layer, credentialled a
crop patch, and withheld a crop from the visual ladder. Every one of those is an evidence
decision, so every one takes the union. Pinned in all three flag states.

The one place the narrower key is still needed is `NativeTableVerifierJudge.assess`, because
the two halves of that function need opposite answers (below).

## The P0 at the judge, and why my own reason for leaving it was wrong

This site was deliberately left alone in round 1, on the ground that the layer could only
REJECT there — so the worst case was a missing number, which this corpus prefers to a wrong
one. That was wrong, and Argon found it. `VerifierState.EXACT_PASS` is an ACCEPTING exit
that passes `inner_consulted=False` and stamps confidence 1.0: on a page whose layer is an
earlier OCR run, a model reading that reproduces the layer's own error can be waved through
by that error.

The site's previous gate narrowed the union down to `page_has_opaque_page_covering_raster`,
over two review rounds whose finding was that suspicion must never make a check weaker —
dropping row-level verification is only safe when something INDEPENDENT read the pixels.
Sound about the rejection half, blind to the accepting one. So the halves are now split:

* a **POSITIVE finding** bypasses native verification altogether. The layer is established
  as a previous reading, and #1202 already gives the page the pixel lane
  (`SourceEvidenceTableJudge`'s raster witness), the visual table ladder and the blind cell
  guard, so this is not an absence of checking;
* a **FAILED scan** keeps row-level verification, because that page may be genuinely
  born-digital and rejection cannot ship a wrong number — but it may never take the
  `EXACT_PASS` short-circuit. The inner judge is consulted instead, and the withholding is
  recorded.

#1202's `page_has_opaque_page_covering_raster` bypass is KEPT and the new one JOINS it.
Round 3 of this ticket replaced it, and #1202's own test caught the consequence
(`test_the_suspect_layer_cannot_veto_the_reading_that_replaced_it`): a page that is suspect
but NOT established, and that carries an opaque page-covering raster, has had its pixels read
independently by the outer `SourceEvidenceTableJudge` — so verifying it against its own
unknown layer can only veto a reading that already has a better witness. The two bypasses
answer different questions and either is sufficient: the positive finding says the layer IS
an earlier OCR run (true on every render-mode-3 page, where the raster proxy answers False
because there is no single page-covering image), and the raster says something independent
read these pixels.

**What is NOT established, recorded rather than claimed.** Argon's second direction — that
such a reading SHIPS at confidence 1.0 — could not be reproduced. On every synthetic ruled
table tried here the header-attribution chain abstains (`native_header_row` returns `None`),
and #245 makes an abstain on that path delegate to the inner judge, so the layer-matching
reading reached a judge anyway. The exit exists in code and the guard closes it; its firing
is pinned by the event it emits, not by a confidence of 1.0, and no claim is made about how
often the 1.0 ship is reachable on real pages.

## The new distrust kind, and exactly what registering it buys

The crop-repair refusal emits `table_crop_repair_unverifiable`, which is registered in
`tables_trust.TABLE_DISTRUST_KINDS`. The first version emitted it without registering, which
is the defect this very file fixed once before: GH-166's crop TIMEOUT kind was emitted and
not listed, so `tables_trust.json` reported no untrusted pages after an incomplete
verification and the page read as verified.

Measured rather than claimed, because the first pin for it was inert: on this page
`dualpass_flagged` also fires and is already registered, so the page appears in
`untrusted_pages` either way. What the registration buys is the REASON — a consumer reading
`counts_by_kind()` is told the crop reading went unverified, rather than only that a crop
disagreed. The pin is on `counts_by_kind()` for that reason; pinned on `untrusted_pages` it
survived the mutant that deletes the registration.

## The design: per-site gating, and the handle that was built and dropped

The brief chose Option B — a text-free page HANDLE for the chain's evidence reads — on the
grounds that it covers present and future readers with no per-site knowledge. It was built
(`TextFreePage`, delegating everything but `get_text`), measured, and then removed. Three
things, in order:

1. **There is no single place the table phase opens the page.** Counted, not assumed:
   `orchestrator.py` opens it for this chain at 6970 (scoring), 7058 (escalation), 14305
   (crop repair) and 14565 (the judge's own getter, which is shared with the PIXEL judge and
   therefore must not be wrapped); `witness.py` opens it four more times, three of them for
   boxes and crops. `prepare_table_witnesses` takes a PATH, not a page, so no handle can
   reach it at all — it has to be handed the answer.
2. **So the handle was three wraps, not one entry point**, and the owner's condition for
   keeping it (one entry point, pinned there directly) could not be met.
3. **The mutants showed all three wraps inert.** With the union guards in place, deleting
   any one of them changed no outcome: the guard abstains first. A guard never seen to fail
   has not been shown to guard anything, so it went.

**The cost, stated rather than smoothed over:** per-site gating means every future reader of
this page's text layer is a new hole, which is exactly the complaint #1205 was filed about —
#1202 needed four review rounds to find four such sites, and this ticket found five more.
The mitigation is only partial: there is ONE predicate, its docstring names the log, and the
log's table above lists every site, so a new reader has one place to look. Anyone adding a
reader to this chain has to consult it. If that proves not to be enough, the structural fix
is the refactor this ticket declined — the table phase opening each page once and passing one
handle down — and it should be done as its own change, not smuggled in as three wraps.

What the handle's own measurement DID establish, and worth keeping for that refactor: a
stripped-copy version of it is not sound. The design note measured a text-stripped page as
pixel-identical on 60 of 60 page/DPI cells, but only because those pages' layers are
entirely hidden; the detector's rule admits a page whose VISIBLE words all sit at repeated
furniture slots, and stripping such a page removes its running header. A VIEW does not have
that failure. Either way nothing in what ships touches pixels: the witness crops are
byte-identical across both arms, with the canary that the fixture's visible header does
contribute pixels at 72/150/300 dpi.

## Readers found by the sweep, OUTSIDE the table chain

Named so the next round does not have to find them again:

* `escalation_canary.region_presence_verdict` at `orchestrator.py:4355` and `:20799` — the
  EQUATION lane's numeric-presence guard (`--detect-equations`, default off). Takes
  `native_text` as a string, and is rejection-only: a spurious `PRESENCE_INVENTED` from a
  prior-OCR layer refuses a LaTeX sidecar and the page keeps its native prose. Content lost
  on the layer's say-so, in a lane the Fed run does not enable.
* `manifest.py:765` `presence_verdict_from_text`, the D3-superseding path. Same shape.
* `born_digital.py:4763` `verify_native_table_region`, which checks socr's own native
  reconstruction against the same layer. It runs during ASSESSMENT, before the #1156 flag
  exists, so it cannot consult a predicate derived from its own output.

Not holes, checked: `_read_page_words` (`orchestrator.py:12888`) sits behind
`_is_native_table_first_candidate`, which is False whenever `needs_ocr_enhancement` is set —
every flagged page — and requires `is_born_digital`. `_record_figure_recoverable_labels` is
log-only. `locate_tables` / `_horizontal_rules` / `find_tables` read drawings.
`core/page_credential._table_blocks_of` works from markdown alone. `benchmark/` is offline
tooling.

## Tests and mutants

`tests/test_1205_table_chain_evidence_readers.py`, 14 tests. Every pin is a DIFFERENCE run
twice in one process, changing only the flag, with a control arm in which the layer DOES
count and a third arm for a FAILED scan. Nothing is pinned by grepping the source — that is
how #1202's round-3 guard shipped a crash that read as green.

One fixture carries most of it: a ruled two-row table whose native layer reads `350` where
the page says `-350` (the measured #1156 round-3 defect, a dropped minus sign). On the
control arm, measured:

* the correct reading scores **75%** against the layer and the layer-matching one **100%**,
  so the page is marked as needing escalation;
* a forced escalation whose candidate reproduces the layer is ACCEPTED (`gate=exactness`,
  75.0% → 100.0%) and the correct `-350` is replaced — Argon's #1207 round-3 case, observed
  rather than argued;
* `verify_native_table` CERTAIN_FAILs the correct reading, and at the judge it does so with
  the inner judge never consulted — one deterministic check, answered from an earlier OCR
  run, discarding the right numbers on its own.

On the flagged and scan-failed arms each of those reverses. The forced-escalation arm is the
one that matters: the provider is still called and the candidate still produced, so what is
pinned is that the layer no longer GRADES, not merely that the page is no longer escalated.

The witness pin uses a two-table side-by-side page (so `locate_tables` returns exactly two
boxes and corroboration runs at all) with the blocks emitted in the opposite order: control
demotes both tables to AMBIGUOUS with no crop, the flagged arm ships the index pairing with
its crops, and a second control with the blocks in the right order is LOCATED on both arms —
so the demotion is the swap verdict and not an artefact. The orchestrator's own derivation
of the flag is observed by capturing what `_run_table_judge_gate` actually passes, over all
three flag states.

**A correction to Argon's crop-repair finding, measured.** It said the layer's verifier would
have REFUSED a row-swapped patch, so round 1 had removed a real check. It would not:
`verify_native_table` returns EXACT_PASS for the swapped crop, because the numeric multiset
is identical and only the row binding differs — the GH-273 shape this verifier is blind to at
page scope. So this guard is a STRENGTHENING, not the restoration of a lost check, and the
clean arm pins that measurement so it cannot quietly change underneath the claim.

Thirteen external mutants (`scratch/1205/mutants_1205.py`), each on a copy of `src` + `tests` +
`pyproject.toml` outside the repo — `pyproject` because its `pythonpath = ["src"]` shadows an
external `PYTHONPATH`, so without it the mutant would test this checkout. Every anchor is
counted UNCAPPED and asserted `== 1` before any replacement, a stale anchor reports
`ANCHOR-STALE` rather than reading as a kill, and the copy's own
`test_loaded_source_is_this_checkout` resolves `socr.__file__` against the copy's tree. It
passed on all thirteen. **All thirteen killed**, each by the pin written for it:

| mutant | killed by |
|---|---|
| the predicate neutered to `True` | 5 tests |
| the predicate narrowed back to the positive finding | 5 tests |
| the escalation scorer's guard removed | 3 |
| the forced-escalation guard removed | 2 |
| the crop-repair fallback guard removed | 2 |
| the witness caller hard-codes `True` | the witness-caller pin |
| the witness reader ignores the answer | the pairing-swap pin |
| the judge's positive-finding bypass removed | the judge pin |
| the unjudged-accept guard removed | the judge pin |
| #1202's boundary guard deleted | the boundary pin |
| the new distrust kind unregistered | the crop pin, via `counts_by_kind()` |
| #1202's opaque-raster bypass removed | #1202's own veto test |
| the credential policy exempts the new kind | the credential pin, and #713's drift guard |

The previous round's 15 mutants are recorded for the record: 12 killed, and the 3 that
survived were the three page wraps, which is what settled their removal.

## Two test stand-ins updated, and why that is not an expectation change

`prepare_table_witnesses` gained a `layer_is_evidence` keyword, and four tests in two files
stood in for it with a fixed argument list. They now forward `**kwargs`. A double that
rejects the real signature does not stand in for the function any more, and the way it fails
is the #1202 round-3 shape: witness preparation raises, every table on the page reads
UNVERIFIED, and the suite blames the gate.

`test_gh696_review_probes.py` asserted the call as a single source STRING, which this change
reformatted across several lines. The property it pins is unchanged and is checked the same
way — the witness is built from `bo.text`, the object `assess` rewrote, not from a cached or
pre-repair copy — by reading the call's arguments instead of matching one line of source. No
expectation was loosened.

## Two defects of my own, caught after round 3 and recorded

Both were found by the full suite and by a mutant, not by review, and both are the shape
this ticket exists to prevent — a change that removes a check while looking like it adds one:

1. **The judge's new bypass REPLACED #1202's instead of joining it.** Consequence above; it
   is now additive and #1202's test is what pins it.
2. **The new distrust kind was emitted but not registered**, so its reason never reached the
   sidecar. Fixed, and the pin was then found to be inert and re-pinned on the reason.

## The credential decision #713's drift guard demanded

Adding a kind to `TABLE_DISTRUST_KINDS` fails
`test_the_blocking_policy_pins_its_exclusions_not_only_its_members` until someone decides in
writing which side of the credential policy it belongs on. It BLOCKS, for the same reason
`dualpass_crop_timeout` and `dualpass_crop_failed` do: a crop reading of this page's table
exists and nothing verified it, so admitting a stand-in that says every table was accepted
would admit it over an outstanding "verification never completed". Nothing retires it today.
Pinned at `_credential_admission_refusal`, the predicate the credential actually consults,
with an unrelated kind as the control.

## Gates

* **Full suite, frozen tree, under the shared lead lock: `8247 passed, 2 skipped, 4 xfailed`
  in 6m09s, zero failures.** The tree was not touched during the run (`scratch/1205/suite_tree_r5.txt`
  records what was modified); the three earlier runs are kept beside it, including the two
  that were red and why.
* `uvx ruff@0.16.0 format --check .` -- 981 files already formatted.
* `ruff check` on the two large touched files is unchanged from `3bcc43db` (746 findings
  before and after, all pre-existing); the new test file is clean. Only the format step is a
  CI gate.

## Round 1 of PR #1211

Grok MERGE-AFTER-FIX, Argon BLOCK, r-uben MERGE once two should-fixes land. Each finding
was reproduced before anything was changed; the probes are in `scratch/1205r1/`.

**Four more readers, all gated on the union.**

* `verify_scanned_table` -> `build_born_digital_evidence` (`source_evidence.py:1354`).
  On a suspect page with NO page-covering raster -- a scan-failed page, a cover under
  50%, a `get_image_info` error -- the PDF text layer was unioned into the numeric
  evidence, so the layer passed its own `350`. It now takes `layer_is_evidence`, and
  excludes the layer on that arm exactly as GH-163's raster arm already did. The
  orchestrator answers from the union; the judge's exception arm excludes the layer.
  Measured on a visible-layer page with an injected pixel witness: the layer's `350`
  goes from PASS to REFUSED while the pixels' `-350` passes. Stated so the claim does
  not outrun it: with the layer included BOTH readings pass (the bundle is a union), so
  what the exclusion removes is the layer passing its own error, not a refusal of the
  correct reading -- that refusal needs a page with no pixel witness, and there real
  tesseract reads the same ink the layer holds.
* `_truncated_grid_reading_ids` (`manifest.py:1320`). Term (b) of `table_truncated`
  scored the candidate's row count against the layer's own bands, so a judge-accepted
  18-row reading carrying the correct `-100.0` was called truncated against 20 bands of
  an earlier OCR run -- and the manifest then prefers a lower-confidence sibling or
  drops the page to the structure-class floor. The layer is dropped, not the predicate:
  term (a) reads the candidate's markdown alone and keeps firing, which is pinned.
* `_maybe_repair_collapsed_headers` (`agentic.py`). It rewrites `output.text` -- the
  bytes that SHIP -- from `fitz_page.get_text("words")`. That is the layer authoring
  content, a stronger act than grading it, and round 1 left it open on a scan-failed
  page. Now gated on the union, which is kept for the whole method as `union_suspect`
  and never flipped.
* The judge's exception arm **failed open**: `layer_found = True` meant "this layer is
  established", which SKIPS native verification -- a raising trust source bought the
  page a bypass. Now `False`: the page is unknown, it is verified, and it gets no
  EXACT_PASS.

**The unjudged EXACT_PASS ship is CONFIRMED, correcting this log's earlier entry.** Last
round it was recorded as unreproducible because header attribution abstained on every
fixture tried. On `_make_probability_table_page` it does not abstain: the clean arm
accepts at confidence 1.0 with `reason='native_table_verifier: EXACT_PASS'` and the
inner judge never consulted. The guard added last round is therefore load-bearing, not
precautionary.

**Which floor a scan-failed table page gets once every rung is refused** (r-uben's nit,
measured at `_winning_page_output` rather than argued): the fail-closed
`[page N failed: unverifiable table -- see image]` marker, on all three flag arms, with
none of the layer's numbers in the shipped text. So "refusing cannot ship a wrong
number" holds on that page -- its floor is a marker plus a page image, not its layer.

**Two directions corrected from round 2/3 of this ticket.** The crop refusal fired
whenever a crop existed, so a crop that AGREES with the accepted table -- the best
evidence the table is right -- listed the page in `tables_trust.json` and voided its
credential; on the Fed scans that is every table page with a crop. It now runs the same
`auto_patch=True` reconcile the fallback would have run and emits only when that changed
something. And the escalation backstop moved ABOVE `submit_daemon`, so a refused
escalation is never billed (r-uben M4/M5).

## What was NOT measured

* No model, no OCR, no provider: every reversal above is measured on deterministic PyMuPDF +
  arithmetic fixtures, not on a run that caught a wrong number in the corpus.
* The corpus EXPOSURE is the design note's, not re-measured here: 720 census pages (~9,300
  corpus pages, ×12.9) reach the scoring/promotion class on a local-only run, 1,283 (~16,600)
  reach the verify/witness class as an upper bound.
* Whether the judge's unjudged `EXACT_PASS` ship at confidence 1.0 is reachable on real
  pages, and how often. See above.
* Cost: the guards are one boolean per page. Not profiled.
