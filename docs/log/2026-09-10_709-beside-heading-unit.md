# GH-709 — a heading printed beside a pair joins it as one emission unit

Branch `fix/709-beside-heading`, on `cf28858` (which carries #704, #706 and #710).

## The defect

`#704` adopts one declined label/value pair at each aligned-run boundary and leaves every
other line of that band where block order puts it. Print the section heading BESIDE the pair
and that separation prints the member above the heading that introduces it. Every token
survives and the section affiliation does not, which is the class of loss GH-592 exists to
prevent.

Astra remeasured 1977-11-15 p1 and the measurement changes the diagnosis. `PRESENT:` (x0
142.0), `Mr.` (214.0) and `Burns, Chairman` (243.0) are ALL in baseline band 8; the accepted
run starts at band 9. So `PRESENT:` is beside Burns, **not one band above him**. The order this
page has always produced is block order by accident, not a rule — and a fix that only refused
to adopt heading-bearing bands would lose that legitimate recovery.

## The rule (Astra design note, option (b) scoped to one boundary band)

After the existing `_adoptable_pair` checks pass, every extra line in the boundary band must
qualify as a standalone heading printed beside the pair. Then the band is adopted as ONE
emission unit — heading, label, value, left to right — placed where the band sits relative to
the run: before it for an upper boundary, after it for a lower one. Each line is consumed
exactly once. If any extra line fails, the adoption **abstains**: the pair keeps block order
rather than being separated from content whose relationship to it could not be established.
Either way a heading-bearing band ends the continuation walk. `#706`'s rules are unchanged.

An extra line qualifies on three conditions (`_beside_heading_lines`):

1. wholly LEFT of the label, so its reading position in the row is unambiguous and it overlaps
   neither the label nor the value;
2. baseline-aligned with the label (vertical extents overlap);
3. not being detached from a multi-line column.

### Why condition 3 is not a block test

The first implementation asked whether every other line of the heading's block was already in
the unit. That is wrong in **both** directions, and both were observed:

- 1977-11-15's `PRESENT:` shares one **13-line block** with the whole roster's `Mr.` labels,
  because that block IS the label column. A block test happens to pass here.
- On the synthetic `STAFF:` fixture PyMuPDF lumps the heading together with two unrelated
  columns into one **6-line block**, and the block test abstained on a genuine heading.

This module already treats a block as a segmentation accident. What answers the question is the
LEFT EDGE: a paragraph or a column is a stack of lines sharing a starting x. The test is
whether any other line of the heading's block starts within **one word space** — a measured
quantity, not a threshold — of it and is not already in the unit. `PRESENT:` and `STAFF:` each
stand alone at their x; a prose column's line, and a numbered marker column's, do not.

Disclosed limit: a genuinely unrelated SINGLE-line block printed left of the pair on the same
baseline is indistinguishable from a heading, and is adopted. No evidence on the page separates
them.

## Measurements

Fed sweep, 6 minutes documents x first 4 pages, against `cf28858`:

- **0 of 24 pages differ.** 1977-11-15 p1 is byte-identical, as required — its correct order is
  now produced by the unit's own left-to-right ordering rather than by block order, and the
  bytes are the same. No other page has a boundary band with extra content, so nothing else
  moves.

Real-page controls, all as tests:

- 1977-11-15 p1 emits `PRESENT:` -> `Mr.` -> `Burns, Chairman`, then `Mr. Volcker, Vice
  Chairman` and the rest of the roster, each line exactly once. The band composition above is
  asserted, not assumed.
- 1990-11-13 p1 keeps Kohn, Bernard and Gillum adjacent to their labels and in printed order,
  and the 24.457pt gap above Kohn still stops the walk against the staff run's 12.336pt pitch.

Synthetic:

- the `STAFF:` fixture from #709 emits the roster first, then `STAFF:` before BOTH `Burns` and
  `Gillum`;
- a heading printed one band ABOVE the staff rows is left in its own band — the unit never
  reaches into a neighbouring band;
- a multi-line prose column sharing the boundary band's baseline makes the adoption abstain,
  and the paragraph stays whole and in sequence;
- a `[note]` printed right of the value also makes it abstain.

## Witnesses

Deleted from a copy of the source and re-run with `-o pythonpath=<copy>/src` (NOT the
`PYTHONPATH` env var — `pyproject.toml`'s `pythonpath = ["src"]` is inserted ahead of it and
silently tests the worktree's own source):

| clause deleted | test that fails |
| --- | --- |
| wholly left of the label | `test_a_marker_printed_right_of_the_value_makes_the_adoption_abstain` |
| same-column sibling test | `test_a_band_whose_marker_belongs_to_a_column_is_not_adopted_at_all` |
| abstain on failure (fall back to adopting the bare pair) | that test and `test_a_line_of_an_unrelated_column_makes_the_adoption_abstain` |
| heading joins the unit | `test_the_staff_heading_precedes_both_staff_members` |

## Residuals

- **The baseline-alignment condition has no witness.** Bands are clustered by vertical centre
  within half the page's median line height, so a band member's extent is essentially always
  overlapping the label's. Making the tolerance large enough to separate them also merges the
  whole roster into one band and destroys the run. It is a defensive check, identical to the
  one `_adoptable_pair` already applies between label and value, and it is recorded here as
  unwitnessed rather than described as tested.
- **A recovery #704 made is now refused.** `test_a_band_whose_marker_belongs_to_a_column_is_not_adopted_at_all`
  (formerly the walk-stop pin) has a two-line marker column `1` / `2` beside its leading bands.
  Moving `2` with its pair would detach it from `1`, so the whole adoption abstains and that
  page keeps block order. This is the design note's ambiguity fallback working as written; it
  costs nothing on the Fed corpus.
- A single-line block left of the pair on the same baseline is adopted as a heading whatever it
  actually is (above).
- Everything GH-592/704/706 lists as residual still stands.

---

## Round 2 — the left-edge test was wrong, and no wider tolerance repairs it

Astra's review of `545de02` reproduced a P1 with real `fitz` geometry: a **centred** two-line
heading, `ALTERNATE` over `MEMBERS`. `ALTERNATE` starts at x0 19.55 and `MEMBERS` at x0 23.72,
a 4.17pt difference against the page's own 2.78pt word space, so the same-left-edge test called
`ALTERNATE` standalone, moved it into the pair's unit and left `MEMBERS` behind in block order.
An indented `STAFF:` / `    advisers` heading fails the same way by construction, and PyMuPDF
splits both of those into two separate blocks, so block membership does not catch them either.

Raising the word-space multiplier only moves the indentation at which the tear happens. The
test is replaced, not widened.

### The rule now

A continuation cannot avoid being printed over the same horizontal ground as the line it
continues: however it is aligned inside its column — flush, indented, centred, hanging — its
x-extent **intersects** the first line's. So an extra qualifies as a standalone heading only
when nothing in the **immediately adjacent baseline bands, above and below, across every
block**, intersects it horizontally. Same-block sibling and adjacent-band overlapper are one
test now, not two, and it carries no tolerance of its own.

An intersecting neighbour is dismissed only when BOTH of these fail:

- it too lies **wholly left of the label**, i.e. occupies the ground left of the pair that a
  wrapped heading would occupy;
- its band is **no further away than the run's own row pitch** (`_run_row_pitch`), so it is
  printed at this material's own leading rather than at some unrelated distance.

Both clauses are needed, and each is measured on a real page rather than chosen.

`1977-11-15` is what forces the pair. The line immediately above `PRESENT:` is the last line of
the meeting's opening paragraph, `1977, at 9:30 a.m.`, and it **does** intersect `PRESENT:`
horizontally — a bare intersection test would abstain and lose the recovery. It runs past the
label lane (x1 235.12 against the lane's x0 214.0) and its band sits 23.8pt away against the
run's 12.4pt pitch, so it is neither heading-shaped ground nor the roster's leading. Below,
`Mr.` / `Volcker` are in the run's lanes, to the right of `PRESENT:`, intersecting nothing.

The `settled` exemption drafted for run-owned lines was dropped: with it the intersection test
had no witness at all, and without it every measurement still holds. No class of line is exempt.

### Witnesses at this commit

| clause deleted | test that fails |
| --- | --- |
| wholly left of the label | `test_a_marker_printed_right_of_the_value_makes_the_adoption_abstain` |
| horizontal intersection (dismiss a non-overlapping neighbour) | real `test_1977_present_row_is_emitted_as_one_unit_without_duplicates`, `test_a_boundary_band_carrying_a_heading_is_still_adopted` |
| neighbour within the run's row pitch | `test_a_continuation_that_crosses_the_label_lane_still_refuses_the_adoption` |
| neighbour wholly left of the label | `test_a_centered_headings_first_line_is_refused_whatever_the_run_pitch` |
| abstain on failure | all three wrapped-heading variants, the unrelated-column and marker-column tests |

Astra's three reproducers are in the suite: the centred, indented and explicitly positioned
wrapped headings as `test_a_wrapped_headings_first_line_is_not_torn_off_its_continuation`, and
the focused helper probe as `test_a_centered_headings_first_line_is_refused_whatever_the_run_pitch`
(rewritten to the new signature; it pins the refusal at a run pitch of zero, so it depends on
the lane clause alone).

Fed 6 documents × 4 pages against `cf28858`: **0 of 24** pages differ.

### Residuals after round 2

- The baseline-alignment condition still has no witness (above).
- The previously logged "single-line block left of the pair is adopted whatever it is" residual
  is narrowed but not gone: such a block is now also required to have nothing intersecting it in
  the adjacent bands. A genuinely unrelated one-line note printed alone beside the pair is still
  adopted as a heading.
- A page whose section heading wraps and whose continuation is **two or more** bands away — a
  blank line's worth of leading between them — is not seen: only the immediately adjacent bands
  are examined. Bands are built from lines with measurable word extents, so a whitespace-only
  line between them does not itself create a band, but a large leading does.
- The marker-column recovery #704 made is still refused (above).
