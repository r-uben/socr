# #1305: the corrupt-math lane deleted prose next to printed fractions

2026-11-13. Plan A4 (`scratch/army/phaseA/BRIEF-A4.md`, `DECISIONS.md`).

## Defect

`line_has_corrupt_math` treated the printed glyphs ¼ ½ ¾ as font damage. Fed text prints
"2¾ percent" for real, so 403 of the 408 pages in the 156-doc run `51270eb5_words` that
went through the lane were there only because of a printed fraction. The lane cropped the
whole line, the reader returned `2^{3/4}` (¾ read as an exponent), and `splice_math`
replaced the whole native line with the crop and the candidate. The sentence was deleted
and the mixed number shipped as an exponent (216 `d^{n/m}` on 134 pages).

## Change

1. **One predicate** (`born_digital.corrupt_math_glyph_count`, `line_has_corrupt_math`).
   The page detector (`_count_math_corruption`), the recovery line selector and
   `_numbered_math_groups` all use it, so no page is flagged without a region (an orphan
   `garbled_math_native`/hybrid banner) or the reverse.
   - ð Þ þ Ð and C0 controls count wherever they stand (unchanged; the page-level
     threshold `MIN_MATH_CORRUPTION_GLYPHS` stays the context evidence for them).
   - ¼ ½ ¾ count only in operator position. A fraction glyph is PRINTED when it is
     digit-attached (`2¾`), hyphen-compounded (`¼-point`), or a stand-alone token inside a
     sentence: the line has a run of `PROSE_RUN_WORDS` (3) plain words and the neighbours
     are plain words, or a number then a word or `%` (`1 ¼ percent`, `8 ½% by`), with
     brackets and neighbouring fractions of the same series skipped (`[¼ to] ½{ to ¾}`).
     Everything else is operator: `y ¼ αx`, `x¼1`, `P ¼ NP`, `y ¼ sin x and ...`.
     Ties go to operator: a false operator costs a crop, a false fraction ships `=` as `¼`.
   - No font test was added. `is_math_font` does not cover AdvP4C4E74/AdvOT*/MTSYN/Symbol
     (Carbon), and a TeX math font prints real fractions too (Kimi/Astra), so the font
     proves neither way. The shape of the token does the work.
2. **No-delete splice** (`recover.CorruptMathRegion.keeps_native_line`, `splice_math`).
   A resolved region whose source has a prose run keeps the WHOLE native line; crop and
   candidate go after it, with a visible line "[equation candidate, unverified: the line
   above is the native text, the crop is ground truth]" so a native `2¾` never sits beside
   an unqualified `2^{3/4}`. Pure equation lines (no prose run, the 5 real mojibake
   pages) are replaced as before. The region record gains `native_line_kept`
   (additive; event kinds and the existing keys are unchanged).
3. `accounting.corrupt_region_evidence` does not count a kept-native region as covering
   its glyphs: the damaged glyphs still ship, so the page must not clear as recovered.
4. Pages that leave the lane are not in it any more, so their math banners and the
   `corrupt_math_hybrid_shipped`/`native+math` disposition are never produced; other
   marks (words check, tables, ...) are computed independently and stay.

No new durable event kind, so nothing new to restore on resume
(`tests/test_resume_restore_kinds.py` passes; the lane's existing events are unchanged).

## Measured, cached data, no OCR

Fed run `51270eb5_words` (156 docs), text layer re-read from
`inputs/fed/pdfs`, new predicate against the 408 pages that carried
`corrupt_math_region_recovery`:

| | pages |
|---|---|
| in the lane before | 408 (81 docs) |
| leave the lane (now plain native text) | 403 |
| stay in the lane | 5 |
| the 5 staying pages | exactly the real-mojibake pages (2020-07-29 transcript-presentation-materials p23, p26-p29), all with ð/Þ/þ glyphs |
| `\d^{n/m}` before | 216 on 134 pages |
| `\d^{n/m}` on the pages that leave | 216 on 134 pages (so 0 on the pages that stay) |

The 5 staying pages hold 11 recovery groups, none with a prose run, so they still get the
replace-with-candidate treatment exactly as before. Two intermediate predicate versions
kept 9 and 6 pages; the extra pages were "8 ½%", "[at ½] to 3/4" and "[¼ to] ½{ to ¾}"
(real fractions), which is why digit-then-`%` and bracket/series handling exist.

Not measured: a fresh shipped-page comparison (needs the reader, i.e. OCR). The leaving
pages ship the native text layer by construction.

### Recall check: '¼' standing for '=' (Papers library)

The 2026-09-03 corpus (366 PDFs, `~/papers`) is not on this machine (`~/papers` is absent).
Substitute: the 965-PDF reference library `/Volumes/Main/Library/References` (209,519 pages
with a text layer, no OCR), old trigger (any ¼½¾ðÞþÐ or C0, page count >= 4) against the new
one:

- old-flagged pages: 1751
- still flagged: 1750
- left the lane: 1 (a Spanish text, "8½ Hz", "7½": printed fractions)

So the real corrupt-math pages (the 14.5% trigger) are all kept. This does not exercise
the Elsevier-font population itself beyond what that library holds; the unit tests pin
`y ¼ αx`, `y ¼ ax`, `x¼1`, `P ¼ NP`, `y ¼ sin x and ...` as detected.

## Residual risks

- `where yy ¼ ax and the model holds` (two-letter variables inside a sentence, `¼` between
  them) reads as a printed fraction and is not detected. Single-letter variables, which is
  the normal case, are detected.
- `10¼5` style (digit directly before `¼` with no spaces) reads as a mixed number.
- `PROSE_RUN_WORDS = 3` is a named, reasoned constant, not fitted to the corpus.

## Guards seen to fail (mutant copies of src and tests outside the repo)

Each run asserted `socr.__file__` inside the copy and `src.count(anchor) == 1` before
editing, then ran `tests/test_gh1305_fraction_splice.py`.

| mutant | result |
|---|---|
| digit-attached exemption removed | fails |
| `_is_printed_fraction` always False (old trigger) | fails |
| `_is_printed_fraction` always True (operator recall lost) | fails |
| `splice_math` replaces the resolved mixed line again | fails (`test_resolved_mixed_line_keeps_the_whole_native_line`) |

Tests: `tests/test_gh1305_fraction_splice.py` (predicate on real Fed lines and operator
lines, shared page/line/numbered agreement, exact mixed-line splice, pure-equation line
still replaced, `process()` pair fraction-line vs mojibake-line in both provider states and
flag on/off, resume of a fraction-only page).
