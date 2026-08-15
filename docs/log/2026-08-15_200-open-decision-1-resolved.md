# OPEN DECISION 1 resolved — SOFT stays advisory, not promoted to a reject

The ratified #200 shipping spec (`headerCheckFalsifier`) named a blocking pre-merge
measurement: re-open the four damaged pages hand-judged in
`docs/log/2026-08-15_tr3-hand-judgement.md`, staged outside the repo at
`~/.local/share/socr/tr3-judge/pages/`, and classify each header defect as ABSENT
(header words entirely missing from the emitted header row) or PRESENT-but-mis-columned
(header words present in the emitted header row, but under the wrong column index).
Resolution rule: `<=2` mis-columned → ship as specified, `SOFT` stays advisory;
`>=3` mis-columned → promote `SOFT` to a reject term before merge.

## Classification (read from the staged `.md` fragments, not the corpus PDFs)

- `021_2017__ozdagli_p38.md` — the emitted table's first row (the true header row,
  line 9) is entirely blank (`|  |  |  |  |  |  |  |`). The actual header tokens
  (`SAR:`, `1997`, `2002`, `time-varying`, `(1)`, `(2)`, `(3)`) landed in two BODY
  rows (lines 10-11) instead, not in the header row at all. **ABSENT** from the
  emitted header row — `header_attribution` sees these tokens nowhere in `grid[0]`
  and fires HARD.
- `028_2018__herskovic__JF_p25.md` — the header words (`L`, `H`, `H−L`, `t-Stat.`,
  `(1)`..`(5)`) are emitted as loose text ABOVE the table block entirely (lines
  13-21), never inside the markdown table at all. The table's own header row
  (line 23) is blank. **ABSENT** — same HARD outcome.
- `037_2020__tsukioka_yamasaki..._p25.md` — the emitted header row (line 1) DOES
  contain `1994-2017` and `1997-2017`, but shifted one column against the derived
  data lanes (`without` stranded in its own cell on line 4, offset from the sample-
  period row on line 3). Tokens are **PRESENT, mis-columned** — this is the `SOFT`
  case, not `HARD`.
- `015_2015__Hameed_Morck_Shen_Yeung..._p49` — the hand judgement recorded this
  page's verdict directly from the viewer without an independent page-by-page
  defect breakdown ("same failure family as 021/028/037" — see
  `~/.local/share/socr/tr3-judge/verdicts.json`). Not independently classifiable
  from the artifacts available; excluded from the count rather than guessed.

## Count

1 of 3 independently-classifiable damaged pages is mis-columned (037); 2 are
outright absent (021, 028). `1 <= 2` — **ship as specified**: `SOFT` remains a
recorded, advisory-only signal (`table_header_verdicts`, emitted as a
`table_header_unverifiable`-style audit event only when `UNVERIFIABLE`, never a
reject); `HARD` is the sole header-attribution reject term wired into
`table_output_defect`. Even in the pessimistic reading that counts the
unclassified `015` as mis-columned too, the total is `2 <= 2` — the resolution
rule still does not cross the promotion threshold.

This does not claim `SOFT` is unimportant — the code and tests already record it
as a measured, deliberately-unpromoted limit (see
`src/socr/tables/header_attribution.py` module docstring and
`tests/test_header_attribution.py::test_misplaced_header_token_is_soft_not_hard`).
Revisit if a larger, independently re-inspected sample shifts the count past the
threshold.
