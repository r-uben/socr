# #1399: scanned tables verified against the stored layer (round 2, after three reviews)

Stage 1 of the design panel. Round 1 (PR #1400 head 2764145) was reviewed by Neon, Astra and
Fable: DO NOT MERGE. This round narrows the scope and hardens the checks. Audit data:
`/Volumes/Main/Library/socr-work/audit_scan_tables/`. Hand-read material (renders and the shipped
table text of every page that now ships WARNING): `.../round2/`.

## Scope (what the layer may do)

The layer acts ONLY on a scanned table on a page whose layer is excluded as suspect
(`layer_is_evidence=False`, the suspect union) that the existing pixel (tesseract) check has
REJECTED (`verifiable` and not `passed`, a table with numbers). A table the pixel check accepts or
could not read, a page where the layer is evidence, born-digital pages, and any call with
`layer_witness=False` give byte-identical gate results; pinned as differences in
`tests/test_gh1399_layer_table_witness.py`. The best outcome is WARNING with the table. Nothing
ships SUCCESS from this path (`LAYER_CONFIRMS_ALONE` is removed).
`tests/test_1211_r2_layer_decides_nothing.py` passes unedited.

## Rules (all must hold for WARNING; otherwise the rejection stands)

* Pre-gates, first: cut-off reading (`structure_check.table_truncated`), chart page
  (`has_chart_marks`, `has_figure_caption`).
* A: every reading table token is in the layer tokens inside the located crop (the whole-page layer
  when no crop was located), multiset, signs and points verbatim.
* B, two-sided, all reading tokens: the layer side is carried by D (an unconsumed layer occurrence
  is a D entry); for each tesseract reading (base 150 dpi and each extra rung, never pooled) an
  occurrence no exact page-reading token consumes that holds a reading token's digits with a mark
  the reading lacks disputes that token. Tesseract holding FEWER marks is the ordinary lost mark
  and is not a conflict. Layer dash-led words (collected inside the crop only) veto a reading
  token sharing their digits.
* C: an integer cell in a row whose other numeric cells (decimal columns only) share one decimal
  count of at least one. Header rows and all-integer columns excluded; a zero is exempt (dropping
  the point of 0.0 changes no value).
* D always runs: layer tokens (crop, else page) absent from the whole-page reading. Non-empty means
  not accepted. The HTML reading is converted to markdown first.

The mark (verdict clause, `source_evidence=UNKNOWN`): the numbers agree with the old stored layer,
which is not an independent reading; row and column placement was not verified; every value the
pixel check could not confirm is listed, uncapped. The candidate is never recorded in
`_pixel_confirmed_candidates`, so the inner verifier's row check still runs for it.
Event `source_evidence_table_layer_unproven` is in `TABLE_DISTRUST_KINDS`,
`CREDENTIAL_BLOCKING_EVENT_KINDS` and the resume replay set; `tests/test_resume_restore_kinds.py`
carries it as a ledger line with its reason (expected size 81).

## Constants

None numeric. Fixed reason strings only.

## Replay of the 30 audited pages (saved qwen readings, real tesseract, real gate)

`a` = D fails closed (shipped design). `b` = D allowed and named in the mark (measured, not
shipped). Old = today's gate on the same reading.

| page | ground truth | old | a | b |
|---|---|---|---|---|
| a04 a10 a11 a15 a16 | not deserved | ERROR | WARNING | WARNING |
| a02 a03 | not deserved | ERROR | ERROR | WARNING (D names 1 and 5 numbers) |
| a00 a01 | not deserved | ERROR | ERROR (tesseract veto, see doubts) | ERROR |
| a13 a22 a23 a24 | not deserved (layer wrong) | ERROR | ERROR | ERROR |
| a14 | misbound | ERROR | WARNING | WARNING |
| a05 a06 | misbound | ERROR | ERROR | ERROR |
| a08 | deserved (dropped row) | ERROR | ERROR | WARNING (D names the missing numbers) |
| a07 a09 a12 a17 a18 a20 | deserved | ERROR | ERROR | ERROR |
| a21 a25 a26 | chart | ERROR | ERROR | ERROR |

None of the 7 deserved, 3 misbound or 3 chart pages reaches SUCCESS in either variant.
The 13 not-deserved: (a) 5 WARNING, 8 ERROR; (b) 7 WARNING, 6 ERROR. Under (b) a08, a real dropped
row, ships. That is why (a) is the shipped design.

## C on every saved reading

Fires on none of the 13 correct readings (about 2,400 decimal cells). Flags on other readings:
a12 `-2 -2 -7`, a20 `-1 -8 -8` (both audited dropped points), p20 `-2 -5` (the layer holds
`-0.2 -0.5`), p25 `-2 -4`. No `0` or `100` in a decimal row occurs in the corpus readings; a zero
is exempt by argument (value unchanged), `100` is flagged (same shape as a dropped point of 10.0).

## Mutation test (13 correct readings, cell by cell; counts are mutants)

| class | reading only | reading and layer |
|---|---|---|
| drop decimal point | ERROR 1091 | ERROR 1091 |
| drop sign | ERROR 194, SUCCESS 1 | ERROR 190, WARNING 4, SUCCESS 1 |
| last digit +1 | ERROR 1226 | ERROR 970, WARNING 256 |
| last digit -1 | ERROR 1226 | ERROR 972, WARNING 254 |
| confusable 3/8 5/6 1/7 | ERROR 1034 | ERROR 824, WARNING 210 |
| delete row | ERROR 186, SUCCESS 1 | ERROR 131, WARNING 55, SUCCESS 1 |
| delete column | ERROR 84, SUCCESS 1 | ERROR 66, WARNING 18, SUCCESS 1 |
| swap two cells | ERROR 741, WARNING 184 | same |

The 3 SUCCESS are the unchanged pixel gate accepting a mutated reading on a04 (tesseract lost the
sign on `-3` itself): the layer path did not touch them. Reading-and-layer sign drops: 162 had the
signed token in some tesseract reading, 0 shipped; 33 did not, 4 ship WARNING and 1 SUCCESS
(pixel gate). Swaps ship WARNING with the placement mark, because only the layer can place
numbers and it has no positions.

## Doubts and not done

* B's tesseract side vetoes correct tables when tesseract adds a mark: a00 (`-268.8` at 150 dpi
  from a rule line) and a01 (`-8`, `-9`) stay ERROR although every number is right. 2 of 9
  not-deserved pages that A passes.
* A page-level number consumes a layer or tesseract occurrence, so a number repeated in prose can
  hide a table omission of the same value.
* Strict row binding does not run on suspect-layer pages; the mark says placement is unverified.
* a26-type chart pages with no marks and no caption are not stopped by the pre-gate (A, B and the
  pixel check reject a26).
* A detector that raises in the chart pre-gate is read as "not a chart".
* The "table disputed" mark text on the rejection path still names Tesseract.
* No blind out-of-sample hand-read of the WARNING pages has been done; material is in `round2/`.
