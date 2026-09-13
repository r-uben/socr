# 2026-09-13 — #734 Stage B: the filled-grid reconciler is wired into the pipeline

Branch `fix/734b-wire-filled-grid-reconciliation` off `main@65133a2` (the merge of #740,
which shipped Stage A). Stage A built the pure reconciler and **nothing called it**; this
branch closes that. From this commit on, a model's filled chart grid is compared against
the counts read from the page's own vector geometry, and a cell the two disagree about no
longer ships.

## The gap

`_suppress_chart_table_skeletons` returns at its structural pre-check when
`find_empty_skeletons(text)` is false, so #635 Stage 0 and Stage 1 act only where the model
left a grid whose every data cell is EMPTY. A model that FILLS the chart region's grid with
numbers bypassed the geometric reader entirely.

## The binding, and the proof that is deliberately absent

A grid is bound to a region by #635 Stage 0's own anchor rules, now shared rather than
duplicated: `chart_data.region_anchors` was extracted and Stage 0 rewired to call it, so the
two lanes cannot drift apart about WHICH panel a grid sits under. Stage B adds
`bind_filled_grids`, which applies the same three rules — nearest preceding anchor, refusal
when another panel's anchor intervenes, refusal of the page's whole set when the bindings run
backwards against source order.

**Measured: the anchor rules bind all 37 filled grids on the SEP corpus, 1:1, in source
order, across all 8 pages that ship one.** No positional fallback is needed and none exists.

**One Stage 0 proof is absent, and it is the stronger one.** Stage 0 also requires
`_axis_attested` — every data column key drawn IN FULL, in order, along the region's axis.
Measured over the corpus, `_axis_attested` attests **neither axis of any of the 37 grids**
(0 on the header axis, 0 on the row-label axis). The cause is structural and visible in
`region_axis_rows`: it groups words into horizontal rows, and a dot plot draws its bin
labels ROTATED on the x-axis, so no horizontal word-row ever carries them — what the axis
rows hold is the y-axis ticks and the legend. Requiring attestation here would refuse all 37
grids and the lane would check nothing. The proof that replaces it is supplied by the
reconciler, not by the binder: `reconcile_grid` refuses unless one axis of the grid names
strictly more of the panel's bins than the other. The two are independent on purpose —
anchors decide WHICH panel, identity decides whether the cells are the same cells. That
split matters because dot-plot panels share their bins, so bin identity could never have
chosen between panels of one figure.

## What ships and what does not

* **A contradicted cell publishes nothing** — not the model's number, not geometry's. The
  cell becomes `WITHHELD` (`CONTRADICTED_MARKER`, the sibling of the reader's
  `UNRESOLVED_MARKER`) and BOTH readings are kept on the audit event. Nothing adjudicates
  between them, and the log's own corpus finding is why: 9 of 10 contradictions cluster on
  one page, which is as consistent with a reader defect as with a model one.
* **Every other cell keeps the model's number.** Geometry with no opinion is not a
  contradiction. Withholding unknown cells would drop **424 readings** on this corpus on the
  strength of a documented abstention (#739).
* **Nothing touches `audit_passed`** (#252): it selects the winner, so flipping it would make
  assemble discard the page's text and turn a withheld cell into a lost page. Demotion is by
  page STATUS, at the flush site beside GH-318's — deliberately not inside the reconciler,
  because the reconciler runs at the CANDIDATE boundary, on bytes that may never become the
  page's winner, so a status set there is an ATTEMPT's status and not the page's. An earlier
  draft justified the placement by saying a non-SUCCESS status would make the escalation gate
  discard the candidate; the reviewer showed that gate runs BEFORE the reconciler at that seam
  (5649 against 5669), so it could not. The placement is right; the reason was overstated.

Five surfaces: the page note (`bo.audit_notes`), per-grid and per-cell audit events, the page
sidecar, the CLI, and the page status plus `PageState.chart_grids_reconciled` /
`chart_grid_cells_contradicted`.

## Disclosure rules the note enforces

Each is pinned by a guard and killed by a mutant.

* unknown and uncovered are reported on **separate lines and never summed** — a cell geometry
  could not resolve is BOTH, and adding them double-counts one cell;
* `uncovered_beside_published` is reported **before** pure recall loss, ranked by
  co-occurrence and never by volume: a grid that published nothing can leave a whole chart
  uncovered while shipping no number, whereas uncovered geometry beside cells that DID agree
  is a table that reads as checked and is half a chart;
* coverage is reported **per series**, not only per page;
* geometry **ABSTAINED** — never "failed", never "disagreed".

## Measurement: predicted, then measured

The team lead's prediction, and the wired result through
`UnifiedPipeline._reconcile_chart_table_grids` over the 8 SEP pages that ship a filled grid:

    grids bound        37   (33 reach a verdict, 4 refused by the reconciler)
    agreed            106
    unknown_to_geom   424
    contradicted       10
    uncovered         720   (308 of them carrying a count)
    beside_published    0
    verified            0

Every figure matches the prediction. Ten cells are withheld, across two demoted pages.

**Zero verified is the CORRECT answer and must not be loosened to move it.** On every panel
the prior meeting is a dashed staircase the reader refuses with a stated reason, so a grid
naming both series cannot reach complete coverage whatever this lane does. The cause is #739
and the remedy is #739.

## A defect this branch's own probe found

The first wired run was **not idempotent**, and the extra record was worse than noise.
Withholding CHANGES the grid's bytes by design, so its sha256 is a different sha256 and no
dedup key built from it can match. On the second crossing the withheld cell no longer holds
a count, so the reconciler reached `not_a_count` and filed a **clean** reconciliation —
`contradicted: 0` — on top of a real contradiction. A consumer reading the latest event for
that grid would have concluded nothing was ever disputed. Measured on `sep-20201216-p09`,
whose grid 5 contradicts once.

The fix is that a grid already carrying the marker is socr's own withheld output and is not
re-judged. Pinned by a guard and by mutant M3. The same exclusion covers socr's own Stage 1
derivation block, which is a filled grid on every later crossing and would otherwise have
reconciled geometry against itself and reported the tautology as a check.

## Mutant battery — 21 mutants, no survivors

`src` AND `tests` copied to `/tmp/mut734`, run from that rootdir, with the three traps from
`docs/log/2026-09-13_734-filled-grid-reconciliation.md` each checked separately:

1. **loaded** — a `conftest.py` canary asserts `socr.__file__` resolves inside the mutant. It
   **fired for real** on the first run (macOS resolves `/tmp` to `/private/tmp`), which is
   the only reason the battery is known to be testing the mutant at all;
2. **applied** — the substitution COUNT is asserted, never printed. Two anchors
   (`if ordered != sorted(ordered):` and the intervening-anchor condition) appear **twice** in
   `chart_data.py` — Stage 0's copy and Stage B's — so those mutants assert `count == 2` and
   mutate the last occurrence only;
3. **load-bearing** — each mutant must turn its named guard RED. Control run green first.

| Mutant | Guard |
| --- | --- |
| M1 publish the contradicted value instead of withholding | the withheld-cell guard |
| M2 withhold unknown cells too | the unknown-cell-keeps-its-number guard |
| M3 re-judge already-withheld bytes | the second-crossing guard |
| M4 reconcile socr's own derivation | the socr-authored-derivation guard |
| M5 flip `audit_passed` | `test_audit_passed_is_never_touched_by_the_reconciler` |
| M6 drop the status demotion | the demote-by-status guard |
| M7 drop the source-order refusal | backwards-order + label-permutation guards |
| M8 drop the intervening-anchor refusal | the intervening-label guard |
| M9 drop the unbound-grid refusal | unbound-grid guards |
| M10 sum unknown and uncovered | `test_the_note_never_sums_unknown_and_uncovered` |
| M11 drop the per-series coverage | the per-series coverage guard |
| M12 beside-published on a grid that published nothing | the recall-loss split guard |
| M13 remove the CLI line | the sidecar-and-CLI guard |
| M14 drop the `failure_mode` assignment | the names-its-failure-mode guard |
| M15 drop the cell from the dedup key | the per-cell-record guard |
| M16 recognise socr's output by substring | the model-writes-the-marker guard |
| M17 drop the three kinds from the replay set | the resumed-run guard |
| M18 stop rebuilding the counters on restore | the resumed-run guard |
| M19 count CLI figures by event | the count-by-identity guard |
| M20 drop the unreadable-geometry record | the unreadable-page guard |
| M21 drop the 1:1 pairing rule | the two P5 binding guards |

**A fourteenth mutant, and the trap that hid it.** At `e1a0340` the battery was 13 and the
reviewer found an unguarded line: deleting `bo.failure_mode = FailureMode.CHART_GRID_CONTRADICTED`
left all 23 tests green while the page reported `FailureMode.NONE`. The status demotion beside
it was genuinely guarded; the mode was not.

What hid it is the load-bearing trap one level up, and it is worth stating in its own words:
**a guard that only fails once a DIFFERENT failure has already occurred has not been shown to
guard its own subject.** `test_the_page_is_demoted_by_status_and_keeps_its_text` went red under
the `withhold_contradicted` mutant, which made the whole demotion look covered — but it went red
because withholding fails UPSTREAM of the demotion and nothing downstream can run, not because
it tests the demotion. The remedy is the discrimination check the battery now runs: under M14 the
status guard must still PASS, proving the new guard is what dies rather than collateral from
another break. `failure_mode` is one of the five surfaces this ticket exists to populate, so an
unguarded line there would have let the metadata surface be hollowed out with no signal at all.

A guard that had to be corrected in the writing: the first version of the label-binding test
asserted that swapping two panel labels swaps the pairing. It does not — a permutation makes
the candidate's panel order run backwards against source order, which the rule refuses by
design. The assertion conflated two rules; the honest pin is the stronger one, since a
positional binder would have returned the same pairing for both arms and this one returns
nothing for the permuted arm.

## What the reviewer's rounds found — one failure at four ranges

Both were live on the corpus, and neither was visible to any guard that existed.

**P1 — the audit trail dropped every contradiction after the first in a grid.** The dedup
identity was `(kind, table_index, sha256, region_index)`, but `GRID_CONTRADICTED` is recorded
per CELL, so two contradicted cells of one grid collided and the second and every later one
was discarded. The surfaces then disagreed with each other: measured, a grid with three
contradicted cells withheld three values in the body and the note said three, while the event
list and the page counter both said one.

This invalidated a figure this log published. The ten contradictions sit in five grids, so the
lane filed **5 events for 10 withheld numbers**, and the CLI — which prints the length of that
event list — would have reported 5 CONTRADICTED on a run that removed 10. The "10" was right
as a statement about `result.contradicted`, which is the one field the wired surfaces never
read. Re-measured after the fix: 10 events, counter 10, 10 markers in the bodies.

Why no guard saw it: all nine pipeline guards used a single bad cell, and the corpus test sums
the PURE reconciler rather than the events. A guard that never exercises the SECOND instance of
a repeated record cannot witness deduplication.

**P2 — a model writing `WITHHELD` anywhere in a grid silently opted out of the check.** The
exclusion was `CONTRADICTED_MARKER not in grid.text`, a substring test over the whole grid
applied before any event was recorded. Measured on two arms identical but for one unrelated
cell: the ordinary arm reconciled and withheld; the marked arm produced no reconciliation, no
refusal, no counter and an unchanged body — the only unrecorded route in this lane.

It also contradicted this branch's own stated rule. `_grid_authored_by_socr` recognises Stage
1's block by the head line socr stamps, never by contents, precisely because a model is
entitled to write a table that looks like one. A model is equally entitled to write `WITHHELD`:
central-bank releases redact values, and a model re-transcribing an earlier socr output carries
the marker straight back in. socr's own withheld output is now recognised by the DIGEST of what
socr itself wrote, recorded on the event as `withheld_sha256`, which keeps the idempotency the
substring test provided — verified on the corpus, where a second crossing of all 8 pages still
records nothing new.

**P3 — a resumed run lost the lane's entire record.** `resume_restore_kinds` omitted all
three of this lane's kinds while both #635 precedents were present, and nothing rebuilt the two
counters. This is the worst shape the lane can produce and strictly worse than P1 and P2: a
resumed page is terminal and never re-processed, so the reconciler does not run, while the
restored body still carries `WITHHELD` where numbers were removed. The CLI printed nothing, the
counter read zero, the sidecar carried no reconciliation — **the document shipped with content
removed and no surface anywhere saying why.**

The code already contained the argument against itself: three lines above the omission, the
#635 entry says dropping those kinds would "silently zero the document's CLI count, which is
computed from these events", citing #563. This lane's counters are computed from its events in
exactly the same way and did not get the same treatment.

Closed on the sibling's terms: the three kinds added with the reason, and BOTH counters rebuilt
in the flag-restore block the way `chart_derivations` is. Pinned as a RESUMED RUN rather than as
a membership test over the kind set — a membership test passes the moment the kinds are added
and stays green if the counters never come back, which is the half it cannot reach. The battery
proves the guard covers both halves: **M17 (drop the kinds) and M18 (stop rebuilding the
counters) both kill it**, and a membership test would have survived M18. That set's own
docstring already records a guard which rebuilt the union from the same sources and so could not
see its own subject; this is the same lesson applied one layer out.

**P4 — the CLI counted raw events.** It was the only surface reading neither the deduped
counters nor the reconciler's results, so P1 landed there too: it under-reported withheld cells
(5 for 10) and over-reported grids checked — a page re-emitted with a DIFFERENT wrong number is
correctly re-judged and files a second reconciliation, and the line then said "2 filled chart
grid(s) checked" for one grid. Every CLI figure is now counted by grid and cell IDENTITY, the
same identities the PageState counters already used.

Kept on the events rather than repointed at the counters, which is a deliberate deviation from
the suggested fix: the counters carry only two of the line's five figures, and sourcing one line
from two places is how surfaces drift apart in the first place. The resume property that
justified using events survives because P3 replays them, and the ordering dependency the lead
identified is real — repointing before P3 would have reported never-rebuilt zeros through the
surface just repointed. P3 landed first.

**The unreadable-geometry route, closed rather than declared.** Three routes left a chart page
unchecked with no record: unreadable geometry, a model-written marker, and a resumed run. Two
were being fixed; leaving the third declared as a residual is the inconsistency a later reader
trips on, and Stage 0 records its equivalent (`SKELETON_UNBOUND`) for one event's cost. It now
records one refusal per filled grid. The operator's question is the same however the checking
failed.

**`audit_notes` accumulation: kept, deliberately.** A page re-emitted with a different wrong
number carries two notes describing the same grid with different numbers. That is accurate as
history and a reader meets two disclosures for one cell. Collapsing them would mean a later
record silently replacing the truth of an earlier one — the exact shape that produced P1's
false clean reconciliation on this very branch — so the confusion is preferred to the
overwrite, and the CLI and counters (which DO dedup) are the surfaces that answer "how many".

**The through-line, which is the best description anyone gave this lane:** *its disclosure was
strong wherever a verdict was reached and silent wherever one was not.* All the blocking
findings are instances of it — the dedup key lost 5 of today's 10 withheld numbers, the marker
exclusion lost all of them for one word in one cell, and the resume gap lost all of them on the
second run. Each surfaces as a page whose body is missing content while the record says it is
not.

**P5 — a wrong-panel binding deleted correct numbers, and the branch stated the mechanism
against itself.** This is the only finding on the branch that MANUFACTURES a loss rather than
under-reporting one, and the justification that hid it was written in this branch's own
docstring.

The binder was derived from Stage 0's by removing `_axis_attested`, and the docstring claimed
the reconciler's bin identity supplied the replacement proof. It does not, and the same
docstring says so ten lines earlier: a grid bound to the wrong panel of one figure shares that
panel's bins, because dot-plot panels draw the same bins. Identity proves the grid and the panel
are about the same KIND of cell; it cannot tell two panels apart. **Two proofs were claimed and
one existed.** That is not a coding error — it is a justification that was never checked against
the code sitting above it.

Two routes produced a wrong binding with no refusal, both reproduced through real code:

    route A  region 2 draws no unique label   ->  bound {1: grid 1}, refusals []
    route B  caption drawn BELOW its figure   ->  bound {1: grid 2}, refusal names grid 1

Route A works because a region resolving no anchor is simply ABSENT from `anchors`, so the
intervening-label guard has nothing to fire on. Route B needs no unresolved label at all: it
shifts every pairing by one while leaving the bindings ascending, which the whole-set order
check cannot see.

The consequence, measured end to end: a grid holding panel 2's TRUE counts reconciled against
panel 1's geometry gives no refusal, 2 agreed, 2 contradicted — and two CORRECT numbers replaced
by `WITHHELD`, filed as a successful reconciliation.

**Why it blocks although it is latent.** Verified on the corpus: 8 pages, 37 regions, 37 anchors
resolved, 37 filled grids, zero pages where they disagree. It ships no wrong number today. It
blocks because *Stage B is the lane that deletes* — the same mis-binding under Stage 0 withholds
an empty grid, and here it removes published numbers. A latent defect in a lane that only
withholds is a different proposition from one in a lane that removes content.

**Closure, in the module's own idiom and cheaper than what it replaces:** the page-level
whole-set refusal already existed for backwards order, and it is extended to require that the
page's regions and the candidate's filled grids pair 1:1. Any region resolving no anchor, or any
filled grid left unbound, refuses the page's whole set. Both routes need exactly that
disagreement, so both close, and `_axis_attested` — which cannot fire on these pages anyway —
is not reinstated. On the corpus it costs nothing: 37 and 37 on every page, and the measurement
is unchanged after the fix.

Its real cost, stated rather than buried: a chart page that ALSO carries an ordinary filled
table is now refused wholesale rather than partly checked. That is a recall loss. It is recorded
per grid rather than silent, and it is the deliberate trade for never deleting a correct number.

**Why the rule is not narrowed to regions alone, which was asked and is a fair question.** Both
P5 routes are about a REGION that fails to bind, so requiring only "every region binds to
exactly one grid, no two sharing" closes both while permitting extra unclaimed grids — which
would keep mixed pages checked instead of refused. The objection to the stronger rule is also
fair on its own terms: all 8 shipped corpus pages are chart-only, 37 regions against 37 grids,
so the corpus is structurally incapable of showing this rule's cost. "Costs nothing here" is
true and unfalsifiable, which is weaker than it reads.

The narrower rule is nonetheless unsafe, and the counterexample is the ordinary case it was
meant to permit. Put a data table BETWEEN a panel's label and that panel's own chart grid —
label, table, chart grid, next label, next chart grid — and measure the binding with the pairing
rule disabled (a mutant tree, `socr.__file__` asserted inside it):

    bound = {1: grid 1, 2: grid 3}      unbound: grid 2
    every region bound: True      no two regions share a grid: True
    => the narrower rule ACCEPTS

Region 1 is bound to the ORDINARY TABLE, because the binder takes the first grid after the
anchor; the panel's real chart grid is grid 2, left unbound. Every region binds, uniquely, and
the binding is wrong. Stage B would then reconcile the panel's geometry against a data table and
replace that table's numbers with the withheld marker — the deletion shape again, on a page
where nothing was ambiguous to a human.

So "an unbound grid" is not merely noise to be tolerated: on a chart page it is evidence that
the anchor-to-first-grid rule has been displaced, and it is the ONLY signal of that displacement
when every region still finds something to bind.

**A second counterexample settles it, and its control is the part that matters.** The
caption-below layout with ONE ordinary filled table at the foot of the page, measured the same
way (pairing rule disabled in a mutant tree, ``socr.__file__`` asserted inside it):

    2 panels + trailing table   bound {1: 2, 2: 3}        all regions bound  => ACCEPTS
    3 panels + trailing table   bound {1: 2, 2: 3, 3: 4}  all regions bound  => ACCEPTS
    control: no trailing table  bound {1: 2}              region 2 unbound   => refuses

Every panel binds to the NEXT panel's grid and the last panel binds to the unrelated data table.
The page satisfies "every region binds exactly one grid, no two regions share a grid" perfectly,
and the only trace is a refusal against grid 1 — which the narrower rule is built to ignore as
an extra grid nobody claimed.

**The control is what makes it conclusive.** Strip the trailing table and the last region fails
to bind, which the narrower rule DOES catch. So the extra grid is not incidental: it is what
converts a detectable shift into an undetectable one, by giving the final region something to
absorb. Any extra filled grid after the last panel does it. The region-side proposal would have
exempted precisely the condition that creates the vulnerability.

**Three things a future reader needs, because they will otherwise propose what was proposed
here.** First, the 1:1 rule DOES narrow what gets checked: a mixed chart-and-table page is
refused wholesale rather than partly checked. Second, that narrowing is the deliberate trade for
never deleting a correct number, and it is justified by the counterexamples above rather than by
the corpus — "costs nothing on the corpus" has been withdrawn as support, because all 8 shipped
pages are chart-only and the failing shape cannot occur in that evidence. Third, a weaker
region-side rule was proposed, built, tested and defeated; it is not an unexplored option.

**An untested idea, recorded as an idea and not as a plan.** Keep the region-side rule and
additionally require each region's grid to be the FIRST grid following that region's anchor with
no other grid between. That would refuse the caption-below route on the grid whose binding is
actually wrong, and leave a trailing table alone. **Nobody has built or measured it**, and it is
recorded here only as the next thing to try if the narrowing proves expensive in practice. It
must not be read as a decision, and it carries no evidence whatever at this commit.

**The trap behind all three findings, including the `failure_mode` one.** A guard that only
fails once a DIFFERENT failure has already occurred has not been shown to guard its own
subject. The demotion test went red under the `withhold_contradicted` mutant — because
withholding fails UPSTREAM of the demotion and nothing downstream runs — which made the whole
demotion look covered while the `failure_mode` line beneath it was pinned by nothing. The
battery now runs an explicit discrimination check for exactly this: under M14 the status guard
must still PASS, and under M16 the idempotency guard must still PASS, proving each new guard
dies on its own subject rather than on collateral from another break.

## Residuals

1. **Hermetic tests use a ONE-panel fixture.** A synthetic two-panel page is detected as two
   regions but the reader refuses both ("no legend on this figure binds a series name to a
   drawing style") — figure-level legend binding across regions is a Stage 1 concern
   (DESIGN.md 1.1), not this ticket's. Multi-panel binding is therefore pinned against the
   PURE `bind_filled_grids` with synthetic `interiors`, which needs no PDF, and the
   corpus-gated test covers the real multi-panel pages.
2. **`published` still does not consult the acceptance hook** (Stage A residual 2), so
   `agreed` is not Stage 2's full bar.
3. **Refused grids hide reader identities no field reports** (Stage A residual 4), unchanged.
4. **Whether a panel whose grid is unverifiable may ship the reader's OWN numbers** is still
   filed rather than settled. The 308 uncovered readings that carry a count make it concrete:
   geometry proved those counts and nobody publishes them.
5. **The 1:1 pairing rule narrows what is checked on a mixed chart-and-table page**, which is
   refused wholesale rather than partly checked (see above). Not closed, and deliberately so.
   The untested candidate for relaxing it — region-side pairing PLUS "no other grid between the
   anchor and its grid" — is described above and has been neither built nor measured.
6. **A page whose chart geometry cannot be read is left alone**, silently from this lane's
   point of view — `_chart_page_geometry` logs and returns `None`. Stage 0 records a
   `SKELETON_UNBOUND` event in the same situation; Stage B does not, because it has no grid
   it can name as the thing that went unchecked until after the geometry is read.
