# #1097 — a currency `$` must not switch off paragraph reflow

Worktree `fix/issue-1097`. Caller is the native-text path (`get_text("text")` +
`clean_native_text` + `page_paragraphs` + `reflow_native_prose`), same as #1079.

## Rule (round 3)

A single `$` is currency, and is hidden from `_MARKER` / `_markers_unbalanced`,
only when every one of these holds. Any doubt returns false and the `$` stays a
marker (fail-closed).

1. It is not one half of `$$` and is not escaped by `\`.
2. No other `$` appears anywhere on the same printed line, before or after
   (so `$1$`, `$2$`, `$x_t$`, `$5 + x$`, `$5 per share and $10` stay maths).
3. Either
   - **prefix:** a number follows it on the same printed line (optional spaces,
     digits, `,` thousands, decimal) and that line then continues with an
     optional comma, whitespace, and a prose word of 2+ letters not preceded
     by `\` (`$218 million`, `$29.99 apiece`). `$5` at EOL, `$5.`, `$5 x`,
     `$5%`, `$5x` stay markers; or
   - **suffix:** a digit immediately precedes it, the `$` ends the token
     (`1987$`, `1987$.`, `1987$ `), and those digits are a standalone token
     (preceded by whitespace, line start, or `(`). `b+2$` / `b+3$` stay
     maths. `US$5 each` is prefix, not suffix.

The #1079 sample forced both arms: Carhart is prefix `$217.8` / `$ 95.6`
followed by a word on the line; Christiano–Eichenbaum–Evans 1994 is suffix
`1987$`. A “`$` then a digit” constant would have left the CEE pages
protected and would have hidden numeric maths openers. Stephens–Donnelly `$`
next to letters or controls is not currency and stays protected.

## What changed

`_is_currency_dollar` / `_hide_currency_dollars` in
`src/socr/core/native_paragraphs.py`. `_protected_lines` scans hidden lines;
`_markers_unbalanced` counts `$` after the same hide. Real maths, including an
unbalanced `$x`, still opens the first-to-last (or to-end) span.

## Measurement

Same 30 documents / 204 native `$` pages as #1079. Before = `_is_currency_dollar`
forced false (pre-#1097). Overprotected = currency-shaped page, ≥50% of nonempty
lines in the span, and reflow differs from “`$` is not a marker”.

- Overprotected before: **35 / 204** (17.2%), matching #1079.
- Round 1: 25 / 35 recovered.
- Round 2: 22 / 35 recovered.
- Round 3 (this rule): **20 / 35** recovered; 15 still overprotected; 54 `$`
  pages changed text. Demers p36 and Andonov p17 drop out of the round-2
  22 (1-letter / bare-punct / glued-suffix `$`). CEE `1987$` and
  `$78 million` / `$1 invested` / `$5 at` pages still recover. Leftovers
  include Stephens–Donnelly OCR `$` and Forney mathish `$0`.

Ten changed lines (old printed break → new joined line):

1. CEE 1994 p40: `…billions of 1987$. With the exception of CNET and` → `…1987$. With the exception of CNET and CNET, all data pertains to the total nonfinancial business sector. …`
2. CEE 1994 p41: `…business sector (BNET and` → `…(BNET and BNET*) and on net funds raised by the nonfinancial business corporate sector (CNET and CNET*).`
3. Carhart 1997 p3: `…I obtain data on surviving` → `…surviving funds, and for funds that have disappeared since 1989, from Micropal/Investment Company Data, Inc. (ICDI). …`
4. Cohen–Frazzini 2008 p12: `At the beginning of every calendar` → `…calendar month, stocks are ranked in ascending order on the basis of the return of a portfolio of its principal cus`
5. Pástor–Veronesi 2009 p22: `…NASDAQ and NYSE/AMEX, namely,` → `…namely, the value of $1 invested in these indices in January 1992, with dividend reinvestment. …`
6. Heston–Korajczyk–Sadka 2010 p20: `…13 disjoint half-hour return intervals. We analyze` → `…We analyze strategies with holding periods of one half-hour (i.e., one interval). …`
7. Andonov 2017 p11: `Panel A shows that the majority of` → `…majority of observations are U.S. pension funds, followed by Canadian funds …`
8. Snowberg–Wolfers–Zitzewitz 2013 p19: `…to predict Google’s` → `…Google’s market cap at the end of the first day of trading. …` (page also has `$225 million`)
9. Snowberg–Wolfers–Zitzewitz 2013 p21: `…in the buildup to the` → `…the second Iraq war. … every 10% increase in the probability of war lead to a $1 increase in the spot price of oil.`
10. Eichengreen–Viswanath-Natraj 2025 p15: `…generated $78 million in cumulative` → `FOMC markets are among the most liquid on Polymarket. For example, the March 2025 contract attracted over 27,000 unique users and generated $78 million in cumulative trading volume. …`

## Tests

`tests/test_gh1097_currency_dollar.py`: currency reflow (`$29.99`, `$1,200 million`,
`US$5`, `$ 95.6`, `1987$` — each followed by a 2+ letter word); `$29.99 apiece`,
`$1,200 million`, `US$5 each`, `1987$.` / `1987$ ` stay currency; `$5.` and
`$5 x` stay markers; P0-1 ``$5 x\na-\nb\n$``; P0-2
``$x+\na-\nb+2$\n$y+\na-\nb+3$``; mixed page uses `apiece`. `test_gh1074` no
longer treats `costs $5 here` as an unbalanced marker.

Round-3 mutants (copy of `src` + `tests` under `/tmp/socr-1097-mutants-r3`,
`socr.__file__` inside the copy, each anchor count == 1):
`_PROSE_WORD_MIN = 2` → `1` → P0-1 fails (`$5 x\nab\n$`); drop
`_suffix_number_is_standalone` → P0-2 fails (`$y+\nab+3$`).
