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

## What the reviewer's round changed

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

## Measurement

All corpus figures are the team lead's, over the SEP dot-plot corpus at `807e060`. They
are stated with their bound: **8 of 23 pages had shipped model output to compare against**,
from one model and one run.

    grids 37, refused 4
    verdicts:  agreed 106   unknown_to_geometry 424   contradicted 10
    reader identities 836; never addressed by any grid cell 604, on 5 pages,
      all carrying a number; refused grids hide a further 80
    uncovered_beside_published: 0 — zero grids, zero pages

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

**The 604 was reproduced independently, and that corroborates the set arithmetic, not the
identity model.** The lead computed index keys minus addressed keys in a throwaway script
before the field existed; the reconciler computes it internally; the two agree. But both
build keys with the same `_series_key` / `_bin_key`, so a fold that merged two distinct
bins or split one would fool both identically. What is confirmed is that the subtraction
is right and that nothing in the loop silently drops identities.

**The corpus-wide zero for `uncovered_beside_published` is structural, not lucky, and
that is the sharper half.** On the affected pages the collapse is total — every panel a
single caption-headed column — so nothing matches, nothing publishes, and the zero follows
from the shape of the failure rather than from the model being careful. **The safe bucket
is one rename away from the dangerous one:** a model changing nothing except heading that
column `December projections` instead of `Number of participants` would move all 604 from
recall loss to uncovered geometry riding beside published agreed cells. That case is not
merely constructible; it is the immediate neighbour of what the corpus already does.

**Both defect counts are floors, for one structural reason.** 258 model-side and 604
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
   the guard still not care. `verified` required `bool(self.cells) and not self.refusal and
   …`; deleting `bool(self.cells)` left every test green, because a refused result also has
   no cells, so the clause the guard named was never the clause under test. Fixing that
   exposed the mirror case — deleting `not self.refusal` then survived for the same reason.
   Both are now pinned by guards that construct `GridReconciliation` directly, one with no
   cells and no refusal, one with cells AND a refusal. Neither guard would exist had the
   battery been assumed rather than run; this is the #735 rule's first catch on new work,
   and it caught the author.

Final battery: **20 mutants, no survivors**, each killed by the guard named for it. Two
mutants from the previous round were retired because the stored fields they targeted were
deleted in the rewrite — the property they protected ("the series summary cannot disagree
with the cell-level set") is now carried by the mutant that computes the summary beside
the set instead of deriving it, which dies on two tests.

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
   separate decision, filed rather than settled here. The 604 make it concrete: geometry
   proved those counts and nobody publishes them.
