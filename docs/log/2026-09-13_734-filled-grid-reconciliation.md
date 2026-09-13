# 2026-09-13 — #734: a FILLED chart grid is reconciled against the page's own geometry

Branch `fix/734-filled-grids-bypass-reader` off `main@a850841`. Two commits: `30b0f2e`
(Stage A — the pure functions) and `807e060` (the reviewer's requested changes).

**None of this is in effect on any output.** The reconciler is not wired into the
pipeline: nothing calls it, no page is checked by it, and no shipped number today has
been compared against geometry. Wiring is Stage B and is deliberately not in this branch.
A reader finding this log later should not conclude that the checking described here is
live.

## What was wrong

`_suppress_chart_table_skeletons` (`pipeline/orchestrator.py`) opens with a structural
pre-check — `if not find_empty_skeletons(text): return 0` — so #635's Stage 0 and Stage 1
act only where the model left a grid whose every data cell is EMPTY. A model that fills
the chart region's grid with numbers bypasses the geometric reader entirely, and nothing
in the pipeline compares those numbers to what the page draws.

Not hypothetical. On `sep-20201216-p09` the shipped page carries
`| 0.13-0.37 | 17 | 17 |` from engine `gemini`, while the Stage 1 reader, called directly
on the same PDF, independently derives `December projections = [17, 0, 0, …]` against the
bins `0.13-0.37, 0.38-0.62, …` for all five panels. The reader half of #734 was fixed on
the #735 branch; the reconciliation half was not, and was recorded as item 14 of
`docs/plans/chart-data/STATUS.md`.

## What was built

`figures/chart_data.find_filled_grids` — the exact complement of `find_empty_skeletons`
over one shared `_parse_grids`. No grid is both; a grid with no data position is neither;
no run/separator/body parsing is duplicated. It says only that the page carries a grid
with values in it — not that the values are wrong, invented, or a chart's.

`figures/chart_reconcile.reconcile_grid` — pure, no I/O. It takes a parsed grid and the
`PanelReading` for the region the caller already bound it to, and returns a verdict per
cell: `agreed`, `contradicted`, `unknown_to_geometry`, `not_a_count`.

Matching is by cell identity and never by position, since position is the model's layout
and the model's layout is what is in question. Bin labels reduce to their atoms (Stage 0's
`_key_atoms`, the form Stage 1's `_join_atoms` emits), series names go through the repo's
own `decode_label_cell` — the binder's view of a label — so no new string comparison was
written for this.

The three prohibitions from `docs/plans/chart-data/DESIGN.md` (Stage 2) are structural
rather than conventions a caller must remember:

- **a contradicted number is never published.** `CellVerdict.published` returns a value
  only on agreement, and the value is the one both sides independently hold;
- **competing counts are never averaged.** There is no arithmetic in the module at all;
- **totals never force agreement.** Nothing sums anything.

**Geometry with no opinion is not a contradiction.** A reader cell that is UNRESOLVED, or
absent, says nothing about the model's number, which survives as unverified. Refusing it
would delete a reading on the strength of a refusal to read.

### Two things that are NOT the design's, stated so neither is taken for ratified

**The orientation rule is this module's own.** DESIGN.md says to reconcile by cell
identity and says nothing about which axis of a grid carries the bins. The reader
publishes series as rows and bins as columns; a model commonly writes the transpose. The
axis matching more of the panel's bins carries them, and a tie refuses. That rests on a
precondition the corpus satisfies and no rule enforces: that the grid's other axis does
not also carry the panel's bin labels. Where a chart prints single-value bins that could
head either axis, the tie refuses the grid — the safe direction, but a refusal caused by
this rule rather than by the page.

**`published` does not check what Stage 2 requires.** The design says agreement must
satisfy calibration and constraints. `published` checks only that both sides hold the same
integer. The calibration half was already enforced when the reader emitted the cell; the
constraint half — the caller's acceptance hook, `chart_reader.verify_panel` — is not
consulted here at all, so a cell can be `agreed` inside a panel whose derivation a caller
would reject.

## What the reviewer's first round changed

Five findings, all in `807e060`. Two of them changed the design rather than the code.

1. **The `unknown_to_geometry` detail string was false.** It read "geometry has no such
   cell" on cells whose bin matched and for which geometry held a number under another
   series. Five causes now separate geometry's own limit (`reader_unresolved`) from a
   reading that exists and was never compared (`series_unmatched`, `bin_unmatched`,
   `cell_absent`, `neither_matched`). Reporting the second as the first states that the
   page was checked where it was not.
2. **Every coverage field ran model→reader.** "Geometry read something the model never
   wrote" did not exist at any level of the output. The first attempt at a fix was a
   series-level field, `unpublished_series`, and the reviewer showed it provably too
   narrow: a grid keeping both series and dropping one printed bin row produces four
   agreed cells, no unmatched bin, no unmatched series, no refusal — and two reader
   identities never addressed. Nothing dropped there *is* a series. The committed field is
   cell-granularity — the identities in `_reader_index(panel)` that no grid cell addressed
   — so series drops, bin-row drops and partial drops all fall out of one set, and the
   series summary is **derived** from it rather than computed beside it, so the two cannot
   disagree.
3. **Severity runs opposite to volume.** A grid whose every column is caption-headed
   matches nothing, publishes nothing, and can leave a whole chart uncovered while
   shipping no number. The dangerous shape is the small one: a grid whose cells did agree
   and publish, beside geometry never consulted — a table that reads as checked and is
   half a chart. `uncovered_beside_published` reports the second and is zero for the first.
4. **Grid-level verification is withheld when identity coverage is incomplete**, so a
   model cannot earn a clean bill by renaming precisely the series that would have
   contradicted it: the rename removes the cells rather than the disagreement. Withholding
   only ever removes agreement — it changes no cell verdict and no `published` value, and
   cannot manufacture a contradiction.
5. **The panel's bin axis was the one identity axis of four that silently overwrote.** The
   grid's bins, the grid's series and the panel's series all refused on a repeat; a bin
   repeated within one read series kept whichever cell was indexed last and judged the
   model against it. Measured before the fix: a panel holding `1.0 → 5` and `1.0 → 9`
   against a grid saying `5` produced `contradicted`, `reader_count=9`.

## What the reviewer's second round changed

Two findings at `807e060`. The first is a route around the gate the first round built.

**An uncompared cell counted as covered.** `addressed` recorded an identity before the
cell's content was examined, so a cell left blank, a cell carrying prose, and a cell whose
reader counterpart resolved nothing all bought coverage for a reading none of them
checked. Measured on a panel holding four proven counts across two series:

    control, both columns filled   agreed 4  not_a_count 0  unknown 0  uncovered 0  verified True
    one column left blank          agreed 2  not_a_count 2  unknown 0  uncovered 0  verified True
    one column carrying prose      agreed 2  not_a_count 2  unknown 0  uncovered 0  verified True
    every cell unknown             agreed 0  not_a_count 0  unknown 2  uncovered 0  verified True

Rows 2 and 3 are the rename route one move cheaper: the first round stopped a clean bill
being earned by REMOVING a column, and leaving the column in place with empty cells bought
the same clean bill. Geometry proved 4 and 5, nothing compared them, and the grid reported
as verified. Row 4 is the worse claim — a grid that corroborated nothing at all was
`verified`, which contradicted the property's own docstring.

The blank-cell route measures **zero occurrences on this corpus** — of 37 grids none
carries a blank cell and none is mixed — so it is a real and cheaply reachable defect
rather than a live corpus problem. It is recorded here as the former.

`published` was unaffected in every row, so no fabricated number could ship; what shipped
wrong was the disclosure, which is the whole purpose of the commit. The fix is one line of
rule: an identity counts as addressed only when a verdict actually COMPARED it — reached
`agreed` or `contradicted`. Naming a reading is not examining it. The invariant holds:
withholding still only ever removes agreement and cannot manufacture a contradiction,
because nothing in it touches a cell's own verdict. The six shapes above are unchanged
under it; only cells that compared nothing lost their coverage.

**The P4 property was not protected, and the earlier answer that it was carried by the
derivation mutant was wrong.** The reviewer removed `unpublished_series` from the dataclass
and computed it from the PANEL instead — the panel series no grid entry names — and the
whole file passed, 40 of 40. The divergence is real and narrow: on a panel series geometry
read but which holds no cell at all, deriving from the identity set returns `()` because
that series contributed no identity to lose, while computing from the panel returns that
series. Both implementations agree on every other shape, which is exactly why the existing
consistency test could not witness it — both of its cases hold under either. Now pinned by
a guard whose two cases differ only in whether the absent series has a reading.

## Measurement

All corpus figures are the team lead's, over the SEP dot-plot corpus at `807e060`. They
are stated with their bound: **8 of 23 pages had shipped model output to compare against**,
from one model and one run.

    grids 37, refused 4
    verdicts:  agreed 106   unknown_to_geometry 424   contradicted 10
    reader identities 836; refused grids hide a further 80

Reader-side coverage was then measured under two instruments, both from `git archive`
trees with `socr.__file__` asserted inside them — the commit as it stood, and a prototype
of the coverage fix below:

    807e060    uncovered 604   with a number 308   without 296   beside_published 0   verified 4
    prototype  uncovered 720   with a number 308   without 412   beside_published 0   verified 0

**The increment is 116, and all of it is cells where the model published a number and
geometry ABSTAINED from reading the mark.** The with-a-number column is flat at 308; the
without column rises 296 → 412. The fix reveals no hidden readings that carry a count — it
reclassifies cells that had been buying coverage without ever being compared. The word
matters: the reader did not disagree with those numbers and did not fail silently on them.
It declined, with a stated reason, and "unchecked" is the accurate description.

**`verified` falls from 4 to 0.** The honest statement is not "nothing is verified" as a
flat fact about the corpus, but that the fix removes verification from four grids that
should never have earned it, leaving none verified here. That is the floor Stage B has to
improve on rather than inherit — but see the reader limit below before reading the zero
as a statement about reconciliation.

**The zero is gated upstream of reconciliation, is the CORRECT answer here, and must
never be reported bare.** Measured independently on all 23 SEP pages at `807e060` — 94
panels, 188 series rows — **95 series resolve every cell, 90 hold cells and resolve none,
3 hold no cells at all, and none resolves partly.** On every panel the CURRENT meeting is
drawn as filled bars and resolves 12/12, while the PRIOR meeting is a dashed staircase and
resolves 0/12. Zero partials is what makes this a limit rather than a reader struggling.

**All 90 abstain explicitly**: every one is `dashed_stroke`, and every one carries the
same stated reason — the outline is drawn as a path the reader cannot decompose, naming
the segment that is neither a horizontal run nor a riser. That is the compound-staircase
limit already recorded in `STATUS.md`, filed as **#739**. None is silent. So the 412
without a number are a documented refusal, not a hole: fail-closed behaviour working as
this repo requires.

A grid naming both series therefore cannot reach complete coverage whatever the
reconciler does, because half its identities were honestly refused. **Zero verified grids
is the expected and correct output on this corpus, and the gate must not be loosened to
make the number move.** It will stay zero until #739's dashed-staircase decomposition is
fixed, and a gate that is never satisfiable on the only corpus anyone has is precisely the
kind of thing a later maintainer relaxes believing it broken. The cause is #739 and the
remedy is #739 — not a weaker definition of coverage.

That also accounts for the 308/412 split above: the identities carrying a number are
approximately the current-meeting half. "0 of 37 grids verified" is a fact about the
reader first and about reconciliation second.

**Two counts that must never be added together.** Under the fix, a cell geometry could not
resolve is BOTH `unknown_to_geometry` and uncovered: the model wrote a number nobody could
check, and a reading existed that nothing compared. They answer different questions and
belong in the disclosure separately; summing them double-counts one cell.

Three figures reported earlier on this branch were wrong and are recorded as such rather
than quietly replaced. "604, of which geometry had a number: 604" was a script testing
index values instead of resolved counts — the field says 308. "Zero of 37 grids earn
verified" was true of the prototype only, not of `807e060`, where four did. And an earlier
604-vs-720 comparison was invalid because one side was measured against the live working
tree rather than an archive.

**The ten contradictions are the model undercounting, corroborated.** Summing each series
per panel on `sep-20220316-p09` — a dot plot column totals the number of participants,
since every participant places one dot per horizon — the reader is self-consistent at 16
across three independently calibrated panels while the model reads 14, 14, 15. The page's
axis tops at 16 and March 2022 had 16 FOMC participants; the fourth panel at 15 on both
sides is consistent with one participant not projecting the longer run. A reader error
would have to coincide identically across three separately calibrated panels.

This is corroboration and not proof: there is no external ground-truth table for these
pages, and the argument is cross-panel self-consistency plus the axis maximum. The totals
were used as evidence about which side to trust when deciding what to build. They are
**not** used by the code, and must not be — a matching total that can flip a cell to
agreed would certify a wrong reader and a wrong model that happen to sum alike. A mutant
that does exactly that is in the battery.

### The six shapes the coverage field distinguishes

One panel, two reader series, three bins, six counts known to geometry; one thing changed
per grid.

| grid | agreed | uncovered (with a number) | `unpublished_series` | verified |
| --- | --- | --- | --- | --- |
| both series published | 6 | 0 | () | yes |
| one column, real series name | 3 | 3 (3) | (December,) | no |
| one column, caption-headed | 0 | 6 (6) | (Sep, Dec) | no |
| one printed bin row dropped | 4 | 2 (2) | **()** | no |
| part of one series dropped | 2 | 1 (1) | (December,) | no |
| dropped column was UNRESOLVED | 1 | 1 (**0**) | (December,) | no |

Row 2 is the shape that was a clean sheet on every field before this round. Row 4 is the
one no axis-level field can express — two readings lost, and correctly no series dropped.
Row 6 is the split: unaddressed, but nothing was lost.

### What the corpus numbers do and do not support

**The reproduction corroborates the set arithmetic, not the identity model.** The lead computed index keys minus addressed keys in a throwaway script
before the field existed; the reconciler computes it internally; the two agree. But both
build keys with the same `_series_key` / `_bin_key`, so a fold that merged two distinct
bins or split one would fool both identically. What is confirmed is that the subtraction
is right and that nothing in the loop silently drops identities.

**The corpus-wide zero for `uncovered_beside_published` survives the fix, and only now is
it worth asserting.** At `807e060` it was produced by an instrument blind to the
blank-column form of the dangerous shape, so "the dangerous configuration does not occur
here" was not a claim that instrument could support. Under the fixed instrument it is, and
the zero holds. **It is structural, not lucky, and that is the sharper half.** On the affected pages the collapse is total — every panel a
single caption-headed column — so nothing matches, nothing publishes, and the zero follows
from the shape of the failure rather than from the model being careful. **The safe bucket
is one rename away from the dangerous one:** a model changing nothing except heading that
column `December projections` instead of `Number of participants` would move those
readings — 720 uncovered, of which 308 carry a number — from recall loss to uncovered
geometry riding beside published agreed cells. That case is not
merely constructible; it is the immediate neighbour of what the corpus already does.

**Both defect counts are floors, for one structural reason.** 258 model-side and 720
reader-side each key on a name geometry never read, so neither reaches the case where a
collapse keeps a real series name and produces a clean sheet in every field that existed
at the time. That is precisely why the committed field is the reader-side one.

## Method traps — three, and they are distinct

`docs/log/2026-09-12_735-sep-reader.md` established that a guard which has never been seen
to fail has not been shown to guard anything. Running that rule on new work produced three
separate silent failures, and each needs its own check. **Loaded, applied, and
load-bearing are three different properties.**

1. **The mutant must be loaded.** #735's trap A: pytest's rootdir re-injects the real
   source ahead of the mutant, and the run is green for the wrong reason. Remedy: copy
   `src` AND `tests` outside the repo and assert `socr.__file__` resolves inside the
   mutant, in a `conftest.py` canary.
2. **The mutation must have applied.** A correctly loaded mutant running unmutated bytes
   is green in exactly the same way. The lead's first spot-check of this battery had a
   regex that matched nothing; 36 tests passed against untouched source, and it was caught
   only because the script happened to print `mutated: False`. Remedy: assert the
   substitution count — `assert n == 1` before running pytest — which cannot be read past.
   This battery does that, and the assertion fired for real: `N5 … anchor appears 0x`
   aborted the run when ruff's reformatting wrapped a return across lines and a literal
   anchor stopped matching. Three mutants had not yet run; they were re-run with regex
   anchors. As a print, that would have been sixteen green mutants with three measuring
   nothing.
3. **The guard must be load-bearing.** A mutation can apply to correctly loaded source and
   the guard still not care. **This one caught the author twice on the same branch**, which
   is the reason it is stated as its own property rather than folded into the rule about
   running the battery. `verified` required `bool(self.cells) and not self.refusal and
   …`; deleting `bool(self.cells)` left every test green, because a refused result also has
   no cells, so the clause the guard named was never the clause under test. Fixing that
   exposed the mirror case — deleting `not self.refusal` then survived for the same reason.
   Both are now pinned by guards that construct `GridReconciliation` directly, one with no
   cells and no refusal, one with cells AND a refusal. Neither guard would exist had the
   battery been assumed rather than run.

   The second occurrence was subtler and was found by the reviewer rather than by the
   battery: the guard for "the series summary cannot disagree with the cell-level set"
   passed against a genuinely different implementation of that summary, because both of
   its cases agreed under either one. A mutant that a guard survives is not always a dead
   mutant — sometimes it is a guard whose cases do not reach the divergence. The remedy is
   the same discipline one level up: choose the case where the two implementations must
   differ, and pin that, rather than a case where they happen to coincide.

4. **The instrument must not be the tree under edit.** A comparison of the two coverage
   instruments was invalid because one side was measured against the live working
   checkout, which already carried the in-progress fix, while the other was an archive.
   Two agents share this checkout, so any measurement taken from `$PWD/src` is a
   measurement of whatever someone was mid-way through writing. Remedy: measure both
   sides from `git archive` trees, and assert `socr.__file__` inside each.

5. **Throwaway probes are still code the rules apply to.** Both the measuring here and
   the measuring alongside it invoked the interpreter on a script path, which the global
   instructions forbid; a heredoc or `-c` is the workable form for a one-off probe. The
   measurements stand — the pattern is what should not be repeated, and it is recorded
   because this work brushed against the rule twice in one evening.

Final battery: **24 mutants, no survivors**, each killed by the guard named for it. Four
were added in the second round: reverting the coverage rule so that naming an identity
counts as examining it; counting a blank or prose cell as coverage; counting an unknown
cell as coverage; and the reviewer's own variant of the series summary, computed from the
panel rather than derived from the identity set. Two mutants from the first round were
retired because the stored fields they targeted were deleted in the rewrite.

## Residuals

1. **Not wired.** Stage B. Nothing above affects any output today.
2. **`published` does not consult the acceptance hook** (see above), so `agreed` is not
   Stage 2's full bar.
3. **Laundering is reachable in general, and unobserved here.** A paraphrased bin label
   falls through `_bin_key`'s fallback into `unknown_to_geometry` rather than being
   contradicted; an orientation tie refuses a whole grid, which launders more thoroughly
   than an unknown. Measured on this corpus: zero identity mismatches, zero unmatched
   bins — every bin label the model wrote met a bin the reader read. Tightening identity
   matching would currently buy nothing and could only manufacture false contradictions,
   so the defence is disclosure rather than stricter matching.
4. **Refused grids hide 80 reader identities** that no field reports, because a refusal
   reaches no verdict at all.
5. **Whether a panel whose grid is unverifiable may ship the reader's own numbers** is a
   separate decision, filed rather than settled here. The 308 uncovered readings that
   carry a number make it concrete: geometry proved those counts and nobody publishes
   them.
