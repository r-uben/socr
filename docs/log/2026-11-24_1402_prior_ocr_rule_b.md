# #1402: scanned-page confirmation, rule B (PROPOSED, awaits the owner's ruling)

## What changes
`verdict.prior_ocr_layer_agreement` (#1381) keeps every gate and loosens two conditions.

- Completeness. Besides the strict fact (`text_hidden_layer_ink_accounted`), the page may carry
  `text_hidden_layer_ink_below_letter`
  (`text_coverage.scan_residual_ink_holds_no_character`). Dark ink outside the layer's
  character boxes is excused in two cases: a speck (under the page's letter height both ways,
  not beside a layer word that holds a digit), and a dash beside a page number alone on its row
  when the layer's TEXT for that number already has a dash on that side. Letter-height ink, a
  dash anywhere else, a mark beside a number and any drawing that is not a horizontal rule fail.
- Text. `_reading_matches_layer_normalised` sets aside the reading's Markdown marks (heading,
  bold, italic, table pipes, delimiter rows) and the form of a word's line-break hyphen. The
  layer is never stripped. Digits are never joined across a line break.

No new threshold: the letter height is the page's median covered-glyph height times
`LETTER_H_PX / GLYPH_H_P50_PX`; the reach of "beside" is that median height.

## Measurement (pilot: 18 scanned Fed documents, 1990-1999, socr fd496b25, free readers)
519 pages ship `prior_ocr_layer_unverified`; 515 are model readings. Pages confirmed, of 515:

| rule | pass | errors found |
|---|---|---|
| #1381 as merged | 0 | none |
| normalised text, no ink check | 149 | dropped numbers on 13 pages |
| A: normalised text + strict ink fact | 19 | 0 |
| B: normalised text + ink check above | 74 | 0 across 451 numbers |
| B2: B, ink touching a layer word excused | 119 | 0 found; unread minus not tested |
| number tokens identical, no ink check | 210 | dropped numbers on 22 pages |

Errors were looked for with tesseract at 300 dpi as a third reading on the 210 pages passing any
rule (1,084 numbers), every disagreement viewed as a crop, plus the leftover-ink crops of rule
B's 40 number-heaviest pages. The 22 real drops are page numbers and page labels missing from
both the reading and the layer (#1403); none passes A, B or B2.

## Two holes in the rule as first measured, closed here
The measured rule excused every component under letter height and any dash beside a lone page
number. On a digit-heavy page the letter height is about half a digit, and a 10 pt hyphen is
narrower than that: an unread minus sign is a "speck". A dash before a lone number over a layer
that reads only the digits is a lost sign, not a page-number dash. The code refuses both. On the
pilot the narrowing changes no outcome: the nine dashes it excuses all sit beside a layer word
that reads `-N-`.

## Replay through the real predicate
`socr-refinalize-prior-ocr <out> --agreement-report --pdf-dir <pdfs>` (read-only) rebuilds each
marked page's state from its sidecar and the input PDF and calls `prior_ocr_layer_agreement`.
Pilot: 74 pages would be confirmed, the same 74 pages as the measurement, of 483 pages with a
finished page record (36 of the 519 belong to two documents the pilot job did not finish).

## Not done
- Rewriting a finished run's pages. The report counts; it writes nothing.
- Light-on-dark ink is not classified (requiring it leaves 40 of the 74).
- Ink under a layer box (#1383) is still unchecked.
- A lone pair of asterisks around a phrase in the reading is read as emphasis; if the page
  prints them and the old layer lacks them, they are lost under a confirmation.
- 18 documents, one corpus; no page was read whole by eye.
- Table pages: none of the 74 is one.
