# #1262: the prior-OCR mark names which reading shipped

## Why

The #1156 clause marks a scan whose text layer is an earlier OCR run UNKNOWN, and never
CHECKED. Round 3 (Argon) made that deliberate: "one reading of a scan, whoever produced it,
is one reading."

The reason, however, always read "earlier OCR, not re-read". Task 0 of the Fed post-2000
Rorqual run (array 25587868, socr `aebc444c`) had 103 marked pages. On 71 of them qwen HAD
re-read the page from its image, and its accepted reading shipped. The mark described the
wrong reading.

## Change (fix 1 only; the status is unchanged)

- **`verdict.py`** gets two named reasons:
  - `PRIOR_OCR_NATIVE_REASON`, "earlier OCR, not re-read";
  - `PRIOR_OCR_MODEL_REASON`, "read once from the image, no second reading agreed".

  The clause and the mode table both choose by the winner: a model reading (#1185's engine
  set) gets the model reason. The mode table needs the choice too. When only the failure
  mode carries the demotion (rewrite and resume passes), its row is what the page prints,
  and the merge prefers the mode's wording over the clause's.
- **`manifest.py`**: the figure-extraction skip (`page_ships_its_own_page_image`) is keyed on
  the mark's own text. It now recognises both marks. Otherwise a model-read scan would have
  its raster extracted and captioned as a figure, the bug Part B round 3 (Argon P1-4)
  closed.
- **No `;` in a reason.** The mark renderer turns `;` into `,`, so a reason containing one
  never matches its own shipped form. A test caught this on the first draft.

## Tests

`tests/test_gh1262_prior_ocr_reason.py`, 9 tests. Each one pins a difference: prior layer
vs model reading.
- the reason names what shipped;
- the status and failure mode are identical for both;
- the figure skip still fires for both;
- both marks survive rendering byte for byte;
- the mode-table row names what shipped.

One existing expectation changed, in `test_gh1156_part_a_interim.py::test_an_accepted_reading_is_demoted_with_its_own_reason_named`.
It asserted the old wording on an accepted qwen reading, which is the label this issue
corrects. Its status, mode and no-second-banner assertions are unchanged and still pass.

## Mutants

Each ran in an external copy, with `socr.__file__` inside it and `count(anchor) == 1`:

| mutant | failing tests |
| --- | --- |
| figure skip blind to the model mark | 1 |
| mode table ignores the winner | 1 |
| clause always native | 3 |
| `;` back in the reason | 3 |

The mode-table mutant survived the first test set, so `test_the_mode_table_names_what_shipped`
was added to cover it.

## Not in scope

Fix 2: the Part B escape, an independent second reading of a scan that can make it
CHECKED. That goes with the repair-cascade work. See #1262, #1245.
