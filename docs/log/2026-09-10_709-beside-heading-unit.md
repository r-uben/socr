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

---

## Round 3 — the two sides are not symmetric, and one measurement in the ruling was wrong

Astra's review of `d5fcd15` reproduced a P1 with real geometry: `STAFF AND OTHER` beside the
boundary pair, continued by `ATTENDEES AT THE MEETING OF THE COMMITTEE` in the **immediately
adjacent band below**, crossing the label lane at 1.5x the roster's 14.77pt pitch. Round 2's
conjunction dismissed it — neither wholly left of the label nor within the pitch — and split the
heading around its first member. Neither of those two facts says anything about whose line it is.

### The rule now

* **Below the extra**, an intersecting line is ALWAYS a possible continuation, because that is
  the direction a heading is read in. There is no dismissal clause on that side at all.
* **Above the extra**, an intersecting line is dismissed only on independent evidence that it is
  a paragraph's line rather than the extra's own heading: it must continue a **left-aligned
  stack** of its own that the extra does **not** belong to.

The pitch clause and the lane-crossing clause are both gone. Nothing here is a threshold except
the page's own measured word space, which is the same measurement the run guards use.

### The ruling said "its own PyMuPDF block"; that block does not exist

The dispatched direction was to read the stack from the intersector's own block: `>= 2` lines
sharing a left edge. Measured on 1977-11-15 page 1, the opening paragraph is **four separate
one-line blocks** — blocks 4, 5, 6 and 7, at x0 178.00, 107.00, 107.00 and 108.00 — so a
block-based test finds a single line, produces no evidence, and abstains on the one real page
the dismissal exists to keep.

The evidence is there in the **bands**: 107.00, 107.00, 108.00 in consecutive bands, against
that page's own 8.28pt word space, with `PRESENT:` starting at 142.00, 34pt right of that edge.
So `_continues_a_left_aligned_stack` reads the band immediately above the intersector, not its
block. This is the third time block membership has been the wrong instrument on this ticket, and
it is now recorded in the helper's own docstring alongside the shared-left-edge attempt.

### One fixture changed, and why

`test_a_boundary_band_carrying_a_heading_is_still_adopted` (GH-706 suite) modelled 1977-11-15's
opening paragraph as a **single** floating prose line directly above the heading. Under the new
rule that is not a stack, so the fixture asked for an adoption the real page's geometry never
asks for. It now prints two prose lines, as the real page does. The one-line shape is kept as an
abstention pin in the GH-709 suite
(`test_a_lone_line_above_the_heading_with_nothing_behind_it_refuses_the_adoption`).

### What this recovers

Round 2 vetoed the adoption whenever the line above ended before the label lane, however clearly
it belonged to a paragraph. Astra measured all six Fed minutes: the line preceding `PRESENT:`
ends at x1 235.12 (1977) to 481.04 (1970), and only 1977 exposes `PRESENT:` as a line of its own
beside the pair at all, so no real page exercised that veto either way. It is now driven by
paragraph evidence instead, pinned by
`test_a_short_paragraph_line_above_the_heading_no_longer_refuses_the_adoption`.

### Witnesses at this commit

| clause deleted | test that fails |
| --- | --- |
| wholly left of the label | `test_a_marker_printed_right_of_the_value_makes_the_adoption_abstain` |
| the whole BELOW branch | all three wrapped-heading variants, the helper probe, all four of Astra's paired cases |
| the whole ABOVE branch | `test_a_lone_line_above_the_heading_with_nothing_behind_it_refuses_the_adoption`, `test_a_band_whose_marker_belongs_to_a_column_is_not_adopted_at_all` |
| left-aligned stack required | `test_a_display_headings_last_line_is_not_torn_off_the_lines_above_it`, the lone-line test |
| a stack needs a line above it (band 0 returns False) | the lone-line test |
| the extra must not share the stack's edge | `test_a_left_aligned_headings_last_line_is_not_torn_off_the_lines_above_it` |
| abstain on failure | all ten abstention tests |

Astra's four paired cases are in the suite as
`test_a_continuation_printed_below_the_heading_always_refuses_the_adoption`.

Fed 6 documents x 4 pages against `cf28858`: **0 of 24** pages differ.

### Residuals after round 3

- The baseline-alignment condition still has no witness (round 1).
- **"Two or more bands away" means beyond an intervening OCCUPIED band, not a distance.** Bands
  are built from lines with measurable word extents, so leading and whitespace do not manufacture
  empty bands: a continuation set far below its heading is still in the immediately adjacent band
  and IS seen. What is not seen is a continuation with another occupied band between it and the
  heading — a heading interleaved with something else.
- A lone line above the heading, with nothing above it, refuses the adoption whatever it is. It
  cannot be told from a heading's own first line. That is a recall loss on a shape no measured
  Fed page has.
- A genuinely unrelated isolated single line printed beside the pair, with nothing intersecting
  it in either adjacent band, is still adopted as a heading.
- The marker-column recovery #704 made is still refused (round 1).

## Round 4 — the dismissing stack must be full-measure prose

Astra's review of `052a749` (`astra-rev-709c-out.md`, reproducer
`/private/tmp/test_astra_709c.py`) reproduced a third tear. A display heading
reading `STAFF` (x0 30) / `AND` (30) / `OTHERS` (40, beside `Mr.`/`Bernard`)
satisfied round 3's two conditions: `STAFF` and `AND` share a left edge within
the page's 2.78pt word space, `OTHERS` does not share it, so the intersecting
`AND` was dismissed as a paragraph's line and `OTHERS` was pulled ahead of the
two lines it belongs under. The paired hanging-indent half (30/40/40) was
already correct. Two equal left edges followed by a different one are not
evidence of independence.

The third condition is what the real page has and the heading fixtures do not:
**every line of the dismissing stack must cross the label lane.** Prose fills
its measure, so a paragraph's lines run past the column the roster's labels
start in. A narrow heading block stops short of it. Dismissal above now needs
all three: a stack of at least two consecutive bands sharing a left edge within
the page's word space; an extra that does not share that edge; and every line
of that stack ending at or past `label["x0"]`. Round 3's
`_continues_a_left_aligned_stack` is now `_left_aligned_stack` and returns the
stack's lines rather than a boolean, because the third condition has to read
them.

Measured on 1977-11-15 page 1:

| Quantity | Value |
| --- | ---: |
| `PRESENT:` extent | 142.00–198.64 |
| label lane (`Mr.`) x0 | 214.00 |
| value lane (`Burns, Chairman`) x0 | 243.00 |
| stack line `1977, at 9:30 a.m.` x1 | 235.12 |
| stack line above it x1 | 543.08 |

Both stack lines cross the label lane, so 1977-11-15 is still dismissed and
`PRESENT:` still recovered.

### The counterexample, built and measured

The rule was not shipped on the reviewer's fixtures alone. A wide display
heading was constructed to defeat it: `STAFF AND OTHER ATTENDEES AT THE`
(x0 30, x1 224.47) / `NOVEMBER MEETING OF THE FEDERAL` (30, 223.92) /
`OPEN MARKET COMMITTEE` (40, 175.57) beside a pair whose label lane starts at
x0 200, with the roster objects written first so block order cannot mask the
result. **It detaches.** Its first two lines supply every piece of evidence the
real page supplies, and the last line is adopted and torn off them.

Both candidate discriminators were measured and both fail:

* **Stack crosses the VALUE lane too.** 1977's last paragraph line stops at
  235.12 and the value column starts at 243.00, so this refuses the one real
  page in the Fed set that exposes the heading at all. It does not even
  separate the counterexample, whose stack (x1 223.92) clears its value lane at
  217.78. Rejected on both counts.
* **The pair's label sits directly under the stack's last line.** True on
  1977-11-15 (label 214.00 inside 108.00–235.12) and equally true on the
  counterexample (label 200.00 inside 30.00–223.92). No separation.

No discriminator with real-page support separates them, so the three-condition
rule ships and the counterexample is pinned as a strict xfail
(`test_a_wide_display_headings_last_line_is_not_torn_off_the_lines_above_it`)
for Astra to rule on. Its enabling geometry is pinned separately so it cannot
drift into passing for the wrong reason.

### Recall given back

Round 3's recovered case is withdrawn. A stack whose lines stop short of the
label lane is now refused, which is exactly Astra's `STAFF`/`AND`/`OTHERS`
shape, so the short-paragraph fixture abstains again and its test pins that.
Nothing measured is lost: Astra measured all six Fed opening paragraphs and
every line of every one is full measure. The abstention costs recall only on a
shape the corpus does not contain.

### Deletion witnesses

| Clause | Tests that fail without it |
| --- | ---: |
| extra wholly left of the label | 1 |
| baseline overlap with the label | **0** |
| below-branch intersection | 8 |
| stack evidence | 2 |
| extra shares the stack's edge | 1 |
| **stack crosses the label lane** | **2** |
| stack needs a line above | **0** |
| whole above-branch | 7 |
| abstain rather than adopt | 16 |

### Residuals

* The wide-display-heading counterexample above, pinned as a strict xfail.
* The baseline-overlap clause still has no deletion witness, unchanged since
  round 1.
* `_left_aligned_stack`'s `above < 0` guard has no individual witness: on every
  fixture the wrap-around lookup it prevents returns no aligned line, so the
  stack-evidence clause refuses first. It is kept as a correctness guard.
* Unchanged from round 3: an unrelated isolated line beside the pair is still
  adopted; the marker column still abstains; "two or more bands away" means
  beyond an intervening OCCUPIED band, not a distance.

## Round 5 — invert what moves

Astra rejected round 4's residual (`astra-rev-709d-out.md`, reproducer
`/private/tmp/test_astra_709d.py`). Crossing the label lane filters out the
narrow counterexample without establishing that the text above the extra is
independent, and a strict xfail documents a wrong emission order rather than
fixing it. Astra also measured the alternative and showed the cheap escape does
not exist: `break` at the boundary abandons the whole unit, there is no
bare-pair fallback, and refusing every intersecting line above keeps the
synthetic heading whole but moves `Burns, Chairman` from emitted index 10 to
index 21 on the real 1977-11-15 page, past `Mr. Roos` and `Mr. Wallich`, away
from his own label.

Rounds 1 to 4 all shared one premise: the extra is pulled to the pair, and the
unit is placed at the boundary band's position. That is what made the extra's
independence load-bearing. To move a heading's last line safely you must prove
the lines above it are not part of it, and no test does. Three were tried and
each was defeated by an ordinary layout -- shared block membership, a shared
left edge, and a left-aligned stack crossing the label lane.

**Round 5 inverts it. The extra never moves.** It stays where the caller's
block order puts it, and so does everything above it, which is never inspected.
The PAIR travels to the extra and is emitted immediately after it. A heading of
any line count therefore stays intact and still precedes its member, and no
claim about the lines above is needed.

Mechanically, the pair leaves the run's emitted group. It is recorded in
`beside_units`, keyed by the block-order key of the last extra, and printed
directly after that line; the pair's own keys go in `relocated`, so they neither
print in block order nor drag the run's group to their position. The run keeps
its own position. The ordinary GH-704 case, a boundary band with no extras, is
untouched: the pair still joins the run's group.

The above-branch machinery is gone -- `_left_aligned_stack`, the shared-edge
test and the lane-crossing test all deleted, and `_beside_heading_lines` no
longer takes `word_space_width`. The below-branch abstention stays and is now
the only neighbour test: the pair is inserted directly after the extra, so a
heading that carries on downward would have the label and value pushed between
its own two lines.

### The correction the direction did not anticipate

Moving the pair is safe only when the boundary band is the ONLY band of its
kind. The GH-592 marker fixture is not: two declined rows carry markers `1` and
`2` in one left-margin column, and the walk adopts only the boundary one.
Relocating its pair to sit after `2` emitted

```text
1  2  Mr.  Gillum  Mr.  Bernard  ...
```

reversing two roster rows -- the loss GH-592 exists to prevent. Neither the
below-branch test nor the unique-pair test refuses it, so the directed rule set
was not sufficient as given.

The added condition is that the band on the FAR side of the boundary, away from
the run, must not look like another band of the same series. It does when both
hold: a candidate in the run's own LABEL lane, so it is a row of this kind that
is staying put; and a line outside both lanes that horizontally intersects one
of our extras, so its marker is in the same column as ours. Both halves are
needed and both are witnessed. The #706 staff fixture has a plain Gillum row on
the far side, a label in the lane with no marker, and must still adopt.
1977-11-15 has `1977, at 9:30 a.m.` there, out of the label lane at x0 108.00
against 214.00, and must still adopt. Refusing is the safe direction, so this
needs no proof that the far band IS a series member, only that it looks like
one.

### Measured emissions

| Page | Result |
| --- | --- |
| real 1977-11-15 p1 | `PRESENT:` 8, `Mr.` 9, `Burns, Chairman` 10, `Mr. Volcker` 11 |
| real 1990-11-13 p1 | unchanged |
| Astra's wide heading | roster, then all three heading lines in order, then `Mr.` / `Bernard` |
| #706 staff fixture | byte-identical to round 4 |
| marker column | abstains, whole page in block order |
| Fed 6 documents x 4 pages vs `cf28858` | 0 of 24 differ |

### Two pins were vacuous and are now exact

Both were found by a deletion witness returning nothing, not by a failure.

* The staff fixture asserted `Mr.` immediately precedes `Burns`. Block order
  prints the two labels together and then the two values, so that is true when
  the adoption abstains and nothing is recovered. It now pins the exact adopted
  sequence.
* The right-marker fixture asserted `[note]` precedes `Burns`. That is true
  whether the pair abstains or is relocated to sit behind the note, which is
  the defect. It now pins the whole tail.

### Deletion witnesses

| Clause | Tests that fail without it |
| --- | ---: |
| extra wholly left of the label | 1 |
| baseline overlap with the label | **0** |
| below-branch continuation | 5 |
| far band holds a label-lane candidate | 12 |
| far band's marker shares the extra's column | 1 |
| the far-band rule as a whole | 2 |
| abstain rather than adopt | 4 |
| relocate the pair rather than the extra | 4 |

### Residuals

* The baseline-overlap clause still has no deletion witness, unchanged since
  round 1.
* Astra's `test_measure_conservative_cost` fails on the reviewer's own
  monkeypatch, which wraps the five-argument helper; the signature is now four.
  Its substantive half, `test_wide_heading_is_not_reversed`, passes. The same
  stale-signature failure has stood in `test_astra_709.py` since round 2.
* Only the band immediately at a run boundary is ever adopted, so a sub-list of
  several consecutive declined rows is still recovered one row deep. Unchanged.
* "Two or more bands away" still means beyond an intervening OCCUPIED band.
