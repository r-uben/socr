# #1294 — the words check (fix-plan A1), 2026-11-12

## Why

The 2026-11-11 independent audit found SUCCESS pages that had lost words their own PDF
prints ("Core PCE inflation" → "PCE inflation"; "seasonally adjusted annual rates",
"projected" gone). No check compared shipped words with the text layer:
`text_layer_numbers` and `text_layer_drops` compare numbers, on model readings only.

## Data

- Runs aebc444c and 51270eb5, the same 156 Fed documents (12,413 pages each).
- Gold pages: `gdrive:databases/socr/gold/fed-2026-11-11.jsonl` (local mirror
  `/Volumes/Main/Library/Databases/socr/gold/`), 1,155 lines, sha256
  `27b2e25db7cf101fcf8f849907af9c75419c2ce3cc3b0c61d9fa86c26f44e633`. Auditor verdicts,
  carried only to byte-identical pages. Built by the audit's `opus/scripts/build_gold.py`. "Lost text" = a defective verdict with a defect type in
  {lost_content, lost_text, lost_qualifier, splice_math_prose_deletion,
  dropped_ligature_glyphs, lost_table, dropped_redaction_notice}.
- Only unmarked SUCCESS pages are scored (the pages a reader takes as good).

## How the rules were chosen

Each rule removed a false-alarm class seen on gold-clean pages (counts below are
page-versions over both runs, as in the next section). These are PROTOTYPE counts
(a scratch script, raw layer unless stated), kept to show why each rule exists; the numbers
that describe the shipped check are in the next section.

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
(`BornDigitalDetector().detect`) to every page of both runs, no OCR
(`2026-11-12_1294-measure.py`). Final rules, after the PR #1295 review:

Gold counts are page-versions: a page judged in one run is counted in that run, and a page
byte-identical in both runs is counted in each.

| | aebc444c | 51270eb5 | both runs | distinct pages |
|---|---|---|---|---|
| gold lost-text pages caught | 42 / 90 | 6 / 21 | 48 / 111 | 90 |
| gold other-defect pages flagged | 8 / 55 | 2 / 41 | 10 / 96 | 57 |
| gold clean pages flagged | 3 / 235 | 0 / 210 | 3 / 445 | 235 |

| | value |
|---|---|
| aebc444c unmarked SUCCESS pages flagged | 86 / 8,297 (all qwen) |
| 51270eb5 unmarked SUCCESS pages flagged | 16 / 8,125 (all qwen; native 67 → 0 after the split-word rules) |

Most of the 16 are qwen readings of chart pages that leave out the chart's labels while no
chart picture ships. The three gold-clean flags: "period" and "rate, prices" (qwen dropped a
running word) and "durables" (qwen wrote "Durable"); all three are real differences on pages
the auditors judged clean overall.

Review fix (owner, PR #1295): the first version rescued a missing word found anywhere in the
page's letters, so "core" lost from "Core PCE inflation" passed because "underscore" was on
the page. Every rescue is now word-aligned (a join of consecutive shipped words, or a ligature
collapsed to its `f`). Effect on the gold pages: 47 → 48 caught, 2 → 3 clean flagged (the
"durables" page above, previously hidden inside "nondurables"). Soft hyphens inside words
("guaran\u00adteed") are not splits.

## Not caught here

- Pages that lost only numbers or table cells (fix-plan A2; `text_layer_drops`).
- Pages with no text layer of their own (scans).
- A word lost while the same word survives elsewhere on the page, beyond the multiset count.
- One shipped picture excuses the words of every chart box on the page, including charts
  that ship no picture (figure boxes are not matched to pictures one by one).
- A page where socr withheld a table is not checked at all; words lost elsewhere on it pass
  (follow-up: excuse only the withheld table's box).
- Known noise: three Tealbook equation pages whose layer prints the token "math" dozens of
  times (tagged formulas) are flagged for it.

## Side test: a stronger reader (not part of this change)

Claude Sonnet re-read the 92 pages the auditors marked as qwen losses (qwen's exact prompt,
page image only). Two of the 92 shipped socr's failed-page marker instead of a qwen reading,
so the comparison covers 90. Against the text layer: words missing 719 → 98, pages missing numbers
68 → 31, pages matching exactly 18 → 40 of 90; data values the layer does not print about
equal (qwen 14 on 2 pages, Sonnet 13 on 6), and Sonnet guessed unclear digits despite
being told not to. Biased toward the challenger (pages chosen because qwen failed); a fair
test needs random pages.
