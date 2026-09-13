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
  because the candidate boundary is crossed mid-ladder where a non-SUCCESS status makes the
  escalation gate discard the candidate outright.

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

## Mutant battery — 13 mutants, no survivors

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

A guard that had to be corrected in the writing: the first version of the label-binding test
asserted that swapping two panel labels swaps the pairing. It does not — a permutation makes
the candidate's panel order run backwards against source order, which the rule refuses by
design. The assertion conflated two rules; the honest pin is the stronger one, since a
positional binder would have returned the same pairing for both arms and this one returns
nothing for the permuted arm.

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
5. **A page whose chart geometry cannot be read is left alone**, silently from this lane's
   point of view — `_chart_page_geometry` logs and returns `None`. Stage 0 records a
   `SKELETON_UNBOUND` event in the same situation; Stage B does not, because it has no grid
   it can name as the thing that went unchecked until after the geometry is read.
