# A2 counter (issue #1282 denominator)

Run: `cd` here, `~/venvs/socr/bin/python -c "import counter; counter.main()"` (8 workers, ~2-3 min). Independent of socr repair/row-binding; imports only `find_documents`/`_read_json`.
Reads run `fed_rerun_51270eb5_words/unpacked` and PDFs `inputs/fed/pdfs/<doc>.pdf` (page N = sidecar page_num, 1-based).

## Pages
Included: sidecar engine not in {native, blank, chart_asset}, status success|warning, md has a pipe table, md has no `[page N failed` marker, PDF page has words. Skip reasons are in summary.json (`pages_skipped_by_reason`; engine/status skips are counted over all sidecars, the rest over eligible ones).

## PDF lines
`page.get_text("words")`, sorted by top; a word joins a line when vertical overlap with the line's band >= 0.5 of the smaller height (a super/subscript shift is < half a glyph; adjacent text rows overlap ~0; the 0.5 is the midpoint between those two cases, not fitted). Lone minus glyph merged into the next number if the gap < glyph height. Line = label tokens (leading non-numeric; a leading `1.` enumerator stays in the label) + numeric tokens. Lines with a non-numeric token after the first number are unusable.

## Normalisation (equal, not contradictions)
U+2212/en/figure dash/hyphens -> `-`; `(x)` -> `-x`; thousands commas removed; `$`, `%`, trailing superscripts/`*`/dagger stripped; leading `0.` -> `.`; trailing `.` dropped; zero carries no sign (`-0.0` = `0.0`); placeholders (empty, `-`, em dash, n.a., ..., NM) are dropped from both sides (so a row with an empty cell is compared on the remaining cells); footnote glue (shipped ends in a superscript mark, PDF value = shipped + one extra digit) is equal. Labels: lowercase alnum only.

## Attribution
Shipped row = first cell (label) + numeric cells. Attributed iff: label non-empty/non-numeric; all non-empty cells numeric; exactly one PDF line with same label and same numeric count; at least 2 positions equal. ADDED (not in the brief): the (label, count) must also be unique among shipped rows on the page, because repeated labels (panels of a Tealbook table: `2018:Q4`) otherwise attributed one PDF line to several shipped rows (first run: 206 cells, mostly such false matches). Unattributable reasons in summary.json.

## Classes (per differing cell, vs PDF text layer)
- point_drop: PDF has a leading point (`-.7`), shipped lacks it, same digits and sign. point_and_sign: same plus sign differs.
- sign_only: same digits and point, sign differs.
- point_other (extension): point missing but not leading (`12.5` vs `125`). None found.
- cell_shift: >=2 differing cells in the row that are all explained by shifting by +1 or -1 positions. None found.
- digit_change: anything else.

## Files
cells.jsonl (one differing cell; `layer` = PDF text-layer value, `column` = markdown column index, `numeric_pos` = position among numeric cells, `row_n`/`row_diff` = row width / differing cells), summary.json, sample.jsonl (30, stratified by class: 7 per class cap of available, seed 0, rest random fill) with PNGs at 150 dpi rendered from the PDF in png/.
