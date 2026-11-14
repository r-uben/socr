# #1305: the corrupt-math lane deleted prose next to printed fractions

2026-11-13. Plan A4 (external working notes; not committed). Review round on PR #1310.

## Defect

`line_has_corrupt_math` treated the printed glyphs ¼ ½ ¾ as font damage. Fed text prints
"2¾ percent" for real, so 403 of the 408 pages in the 156-doc run `51270eb5_words` that
went through the lane were there only because of a printed fraction. The lane cropped the
whole line, the reader returned `2^{3/4}` (¾ read as an exponent), and `splice_math`
replaced the whole native line with the crop and the candidate. The sentence was deleted
and the mixed number shipped as an exponent (about 216 `d^{n/m}` on 134 pages).

## Design rationale

Two independent guards, because either alone leaves a hole.

* The trigger decides which lines enter the lane. A printed fraction must not enter it
  (cost: a crop and a model call per page, and a wrong exponent shipped).
* The splice decides what ships once a line is in the lane. The owner rule is that source
  text is never deleted. A candidate is unverified model output; the first review round
  tried to keep the native line only when the source "carried prose" (three consecutive
  plain words). That test is a guess about the content of a line and failed both ways
  (`Real GDP ðx` has two words; `The state-of-the-art model PðAÞ ¼ 1` hid its words behind
  hyphens). A guess that decides deletion is the wrong shape, so deletion is gone.

## Change

1. **Splice: no region replaces its native line** (`recover.splice_math`,
   `recover._region_replacement`). Every aligned region with a retained crop keeps the whole
   native slice, followed by the crop. A resolved region adds the visible label
   "[equation candidate, unverified: the line above is the native text, the crop is ground
   truth]" and then the candidate; an unresolved one adds the unresolved marker as before.
   Pure equation lines are included: `ðxÞ ¼ y` keeps its native text with the candidate
   beside it. `keeps_native_line`, `source_carries_prose`, `line_has_prose_run` and
   `PROSE_RUN_WORDS` are removed; nothing decides deletion any more.
   - The page keeps its existing math mark. Coverage accounting is unchanged
     (`accounting.corrupt_region_evidence`): the native private-use codepoints now ship next
     to the reading, so `unresolved_math_detail` finds residual PUA in the body and a page
     with damaged glyphs is no longer cleared by a recovery. The two
     `test_gh165_unresolved_math_outcome` tests that pinned "a fully covered page clears
     the warning" now pin the opposite (the mark stays, with the residual-glyph reason, and
     it restores identically on resume). Two `test_math_recover` tests that asserted the
     native glyphs were gone now assert they are kept, before the crop.
   - Event kinds and record keys are unchanged (no `native_line_kept` key; it would be
     constant), so nothing new to restore on resume
     (`tests/test_resume_restore_kinds.py` passes).
   - The 5 real-mojibake pages (2020-07-29 transcript-presentation-materials p23, p26-p29)
     used to get the replace-with-candidate treatment; their expectation is now native line
     kept, then crop, label, candidate.
2. **Trigger** (`born_digital._is_printed_fraction`, one predicate shared by the page
   detector, the line selector and `_numbered_math_groups`). Rule by neighbours, L the token
   before, R the token after (brackets, dashes, a sign and `%` skipped; a neighbouring
   fraction of the same series skipped; for a dash compound, the piece against the glyph):
   - glued after a digit: the digit is L and the glyph must not touch an operand on its right
     (`2¾`, `2¾%`, `2¾ percent` printed; `10¼5`, `10¼x`, `10¼ 5+5`, `2¼x` operator).
   - glued before a hyphenated word after a space or line start (`¼-point`): printed.
     `x¼-y` is the '=' of `x = -y`: operator.
   - R is a number: operator (`x ¼ 0`, `2 ¼ 4`, `log likelihood ¼ 0`). Prose does not put a
     bare number after a fraction.
   - L is a number (an optional leading sign, `-` or U+2212, is stripped: `−2 ½ percent`,
     `-2½`): printed iff `%`, a word or the end of the line follows (`1 ¼ percent`,
     `8 ½%`, a wrapped `is 2 ½`). A line never ends on the '=' of `2 ¼`: mojibake '=' sits
     between two operands.
   - L is the article "a": printed iff R is a word and the token after R is a word or absent
     (`a ¼ point cut`). `a ¼ sin x` ends on a variable: operator.
   - both sides compact (at most two letters, or no vowel) with one side vowel-free: operator
     (`yy ¼ ax`, `xy ¼ z`). Two short words each carry a vowel (`at ½] to`) and stay prose.
   - both sides plain words, or missing: printed (`the ½ share`, `GDP ½ percentage point`).
   - everything else is operator: a variable, closing bracket or math glyph on a side
     (`x ¼ y`, `y ¼ αx`, `P ¼ NP`, `f(x) ¼ 1`).
   A plain word is letters only with hyphen compounds allowed (`state-of-the-art`), at least
   two letters in some piece, and Latin script (`αx` is a symbol). Two constants, both fixed
   and not corpus-tuned: `_PLAIN_WORD_MIN_LETTERS = 2` (one letter is a variable; the only
   one-letter word, "a", is handled by name) and `_VARIABLE_MAX_LETTERS = 2` (two letters is
   a product of single-letter variables). Everything else counts "a number", "a word" and "the
   next token". Ties go to operator: a false operator costs a crop, a false fraction ships
   '=' as '¼' silently.
   Every verdict is a row of `_VERDICTS` in `tests/test_gh1305_fraction_splice.py`.
   ð Þ þ Ð and C0 controls still count wherever they stand.
3. No font test: `is_math_font` does not cover AdvP4C4E74/AdvOT*/MTSYN/Symbol, and a TeX
   math font prints real fractions too, so the font proves neither way.

## Measured, cached data, no OCR

Fed run `51270eb5_words` (156 docs), text layer re-read from `inputs/fed/pdfs`, final
predicate against the 408 pages that carried `corrupt_math_region_recovery`:

| | pages |
|---|---|
| in the lane before | 408 (81 docs) |
| leave the lane (plain native text) | 403 |
| stay in the lane | 5 |
| the 5 staying pages | the real-mojibake pages (2020-07-29 transcript-presentation-materials p23, p26-p29) |

Intermediate versions of the rule, on the same pages: the first review-round rule kept 17
(12 printed fractions: `GDP ½ percentage point`, `4½ percent ... mid-2018—about ½` (the table is representative, not exhaustive; the exact `4½ percent` case is the row `inflation of 4½ percent`), a wrapped
`2 ½ `, `-¼ percent`, `percent—is`); each is now a row in `_VERDICTS`. A literal `^{n/m}`
count over the leaving pages is 219 on the same regex family as before (216 in the first
count; the pattern differs slightly), none of which ships from the lane any more.

Not measured: a fresh shipped-page comparison (needs the reader, i.e. OCR). The leaving
pages ship the native text layer by construction.

### Recall check: '¼' standing for '=' (References library)

`~/papers` (the 2026-09-03 366-PDF corpus) is not on this machine. Substitute: the
965-PDF library `/Volumes/Main/Library/References`, text layer only, old trigger (any
¼½¾ðÞþÐ or C0 glyph, page count >= `MIN_MATH_CORRUPTION_GLYPHS`) against the new one:

- pages with a text layer: 209,519
- old-flagged pages: 1751
- still flagged: 1750
- left the lane: 1 (a Spanish text, "7½ Hz", "8½ Hz", "9½ Hz": printed mixed numbers)

So the real corrupt-math pages (the 14.5% trigger) are all kept. This does not exercise the
Elsevier-font population itself beyond what that library holds; the unit tests pin
`y ¼ αx`, `x ¼ y`, `P ¼ NP`, `10¼5`, `log likelihood ¼ 0` as detected.

## Residual risks

- `by ½ of` (two short words, one vowel-free because `y` is not a vowel here) reads as an
  operator: a false operator costs a crop, no content. `total cost ¼ price quantity` below
  is the opposite error.
- `total cost ¼ price quantity` (words on both sides of a spaced '=') reads as printed.
- A mojibake '=' that ends a wrapped line after a number (`... 2 ¼`) reads as printed.

## Guards seen to fail (mutant copies of src and tests outside the repo)

Each run asserted `socr.__file__` inside the copy and `src.count(anchor) == 1` before
editing, then ran `tests/test_gh1305_fraction_splice.py`.

| mutant | result |
|---|---|
| a resolved region replaces its native line again (deletion restored) | 9 failed |
| glued digit then a digit counted as a fraction (`return True` for a digit-glued glyph) | 1 failed (`10¼5`) |
| operator-shape guard removed (round 3) | 3 failed (`yy ¼ ax`, its sentence form, `xyz ¼ xy`) |
| the glued-digit skip of the glue check removed (the astra recheck fix) | 6+ failed (`4½ percent`, `8 ½%`, ...) |
| hyphen shortcut no longer requires a space before the glyph | 1 failed (`x¼-y`) |
| leading sign no longer stripped in `_is_number` (round 3) | 2 failed (`−2 ½ percent` rows) |

Tests: `tests/test_gh1305_fraction_splice.py` (the `_VERDICTS` table; splice over
`Real GDP ðx`, `The state-of-the-art model PðAÞ ¼ 1`, `ðxÞ ¼ y`, the mixed line, and the
pure mojibake line; real-page recover-then-splice; shared page/line/numbered agreement;
`process()` pair fraction line vs mojibake line in both provider states and flag on/off;
resume of a fraction-only page).


Round 3 (cubic + astra recheck): Fed lane re-measured on the final rule: 403 leave, 5 stay
(same 5). References recall on the final rule: 209,519 pages, 1751 old-flagged, 1750 still flagged (same single Spanish "7½ Hz" page leaves).
