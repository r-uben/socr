# #1294 — the words check (fix-plan A1), 2026-11-12

## Why

The 2026-11-11 independent audit found SUCCESS pages that had lost words their own PDF
prints ("Core PCE inflation" → "PCE inflation"; "seasonally adjusted annual rates",
"projected" gone). No check compared shipped words with the text layer:
`text_layer_numbers` and `text_layer_drops` compare numbers, on model readings only.

## Data

- Runs aebc444c and 51270eb5, the same 156 Fed documents (12,413 pages each).
- Gold pages: `gdrive:databases/socr/gold/fed-2026-11-11.jsonl` (auditor verdicts, carried
  only to byte-identical pages). "Lost text" = a defective verdict with a defect type in
  {lost_content, lost_text, lost_qualifier, splice_math_prose_deletion,
  dropped_ligature_glyphs, lost_table, dropped_redaction_notice}.
- Only unmarked SUCCESS pages are scored (the pages a reader takes as good).

## How the rules were chosen

Each rule removed a false-alarm class seen on gold-clean pages. Prototype counts (scratch
script, raw layer unless stated):

| version | gold lost-text pages caught (/111) | gold clean flagged (/445) | 51270eb5 silent pages flagged |
|---|---|---|---|
| raw multiset diff, words ≥ 4 letters | 80 | 71 | 1,303 / 8,076 |
| + repaired layer, fused-word rescue, all repeated text excused | 38 | 9 | 96 |
| repeated text excused only at the page edge (owner's ruling) | 53 | 12 | 107 |
| + hyphen breaks, edge rule ignores folios, figure words, ligatures | 47 | 1 | 46 |

Findings behind the rules:

- Excusing **all** repeated text hid real losses: repeated table headings ("percentage
  point contribution to growth"). Many raw "catches" were only the release stamp or the
  classification banner, not the loss the auditor found. Owner: excuse repeated text only
  at the page edge.
- The raw layer keeps broken ligatures ("infation", "frst"); socr's repaired layer does not.
- Words split across lines: the layer prints `Home-`/`builders`; pages ship
  `Homebuilders`, `infla-`/`tion` (native lane, no witness, #1074), or `head winds`.
- Some generators write each line as its own block (Beige Books 2024–2026), so a hyphen
  break crosses blocks.
- Words inside a chart are owed only when no picture of the chart ships.
- Math-italic letters (U+1D400–1D7FF) are formula variables, not words.

## Built-in check, measured

`page_missing_words` / `text_layer_words_clause` applied with socr's own page analysis
(`BornDigitalDetector().detect`) to every page of both runs, no OCR:

| | value |
|---|---|
| gold lost-text pages caught | 47 / 111 |
| gold other-defect pages flagged | 10 / 96 |
| gold clean pages flagged | 2 / 445 |
| aebc444c unmarked SUCCESS pages flagged | 80 / 8,297 (all qwen) |
| 51270eb5 unmarked SUCCESS pages flagged | 16 / 8,125 (all qwen; native 67 → 0 after the split-word rules) |

Most of the 16 are qwen readings of chart pages that leave out the chart's labels while
no chart picture ships.

## Not caught here

- Pages that lost only numbers or table cells (fix-plan A2; `text_layer_drops`).
- Pages with no text layer of their own (scans).
- A word lost while the same word survives elsewhere on the page, beyond the multiset count.

## Side test: a stronger reader (not part of this change)

Claude Sonnet re-read the 92 pages the auditors marked as qwen losses (qwen's exact prompt,
page image only). Against the text layer: words missing 719 → 98, pages missing numbers
68 → 31, pages matching exactly 18 → 40 of 90; data values the layer does not print about
equal (qwen 14 on 2 pages, Sonnet 13 on 6), and Sonnet guessed unclear digits despite
being told not to. Biased toward the challenger (pages chosen because qwen failed); a fair
test needs random pages.
