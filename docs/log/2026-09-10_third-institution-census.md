# Third-institution census — BoE (+1 Banxico scan) on main@52532c0

2026-09-10. Question from the owner: are the merged rules and their constants general, or
fitted to the two institutions they were measured on? Sample: 9 three-page excerpts, 27 pages,
drawn at random from `/Volumes/Main/Library/Databases/central_banks/boe/` (4 born-digital
table pages, 2 pure 1997 MPC-minutes scans with no text layer, 2 decorative/slide "scans", and
1 Banxico scanned excerpt as a Spanish-language scanned witness; BCB minutes turned out to have
text layers and added nothing). Pinned checkout `~/repos/.worktrees/socr-census-boe` at
`52532c0`; inputs, outputs, `sample.json`, `run.log`/`run2.log` at
`~/Data/socr/census-boe-2026-09-10/`. Scorer as in the Fed/ECB census (numeric multiset of the
shipped page body vs `pdftotext -layout`; for the text-layer-less 1997 scans, tesseract 5.5.3
page OCR at 200 dpi is the reference instead).

## Numbers shipped by class

| class | pages | numbers shipped / source | notes |
|---|---:|---:|---|
| born-digital table pages (2003/2006/2019/2023) | 12 | 490 / 504 (97%) | 2006 p2 qwen 143/143 shipped `table_unverified` (flag); 2019 p2 gemini 73/87 `model_output_flagged` |
| pure scans, 1997 MPC minutes (no text layer) | 6 | 65 / 66 vs tesseract; word overlap 0.99 on every page | prose scans ship correctly through qwen |
| Banxico 2018 scan (text layer present) | 3 | 65 / 65 | chart_asset lane, native text kept |
| BoE 2024 speech "scan" (slide export) | 3 | 7 / 7 | chart_asset; decorative raster (#511/E1 shape) |
| BoE 2018 Inflation-Report box page | 3 | 19 / 132 (14%) | **the one generality gap, below** |

## The gap: text tables vs the row-shape family (#703)

`boe-meetings-2018-scan-p28-30` p1 is a two-column comparison box ("Developments anticipated
in February" vs "Developments now anticipated"): cells are sentences with zero or one number.
The cached qwen candidate carries 23/23 of the page's numbers with 0 extras (10 pipe rows) and
the judge ladder ACCEPTED it. Shipped: the fail-closed marker, 0/23. Trail: `candidate_truncated`
(A2, #647) fired on qwen and gemini alike → `structure_class_ladder_exhausted_floor`.

Mechanism: A2's shortfall term derives `row_shape_min` from the candidate's own body rows
(`numeric_body_rows` → 2 rows of width 1, so `row_shape_min = 1`) and counts every native
baseline band with ≥ 1 numeric token as a "table row"; on a prose-heavy page that is dozens of
rows, the candidate's 2 look like a truncation, and a complete, accepted table is discarded.
This is #643's narrow-table blind spot amplified into total loss. The row-shape constants
(`ROW_CORROBORATION_MIN`, `SKIPPED_ROWS_MAX`, the A2 allowance) were all measured on
numeric-dominant statistical tables (ECB annex, Fed swap lines); they do not describe text
tables. Rule fix, general: apply numeric row-shape reconciliation only when the candidate's rows
are numeric-dominant; text tables use the word-overlap witness. Filed as #703.

p2 of the same excerpt (chart + table page) floors with a prose-only qwen candidate (30/45):
#189-class, not new. p3 ships qwen 19/64: chart axis ticks (gist, acceptable).

## Verdict on generality

- Rules: institution-agnostic by construction; nothing in the diff knows what a Fed or an ECB is.
- Constants: hold on BoE numeric tables (97%) and on pure scans (99% words) with no re-fitting.
  They fail on a table SHAPE that neither calibration corpus contained (text tables), not on an
  institution. The remaining risk is shape coverage, and the census now has three shapes on
  record that need their own rule: narrow numeric tables beside numeric footnotes (#643), text
  tables (#703), and dense-label raster charts (#653).
- Attendee lists (#592) could not be tested here: BoE MPC minutes list attendees in one column.
