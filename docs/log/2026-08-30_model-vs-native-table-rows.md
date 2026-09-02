# Model vs native on tables — the referee is the thing that is wrong

2026-08-30. Settles the question raised by the owner: *"why don't we check quality and
go to an LLM?"* The answer is that socr already escalates every table page to a model.
What was never measured is whether the thing grading that model is fit to grade it.

It is not. On 13 blind-transcribed rows, both models beat the native text layer on the
**pre-registered falsifier** (invented + missing): native 21, qwen 14, gemini 7.

~~both models beat the native text layer on every measure, and the free local model wins
outright~~ — **corrected (GH-338).** "Every measure" overstates: row labels wrong is
0/0/0, a tie. And "wins outright" inverts the registered metric — the falsifier was
invented+missing, on which **gemini wins (7 vs qwen's 14)**. Qwen leads only on exact
rows (12 vs 11). Those are different questions and this log conflated them.

Content-free per the convention of `2026-08-22_binding-oracle-corpus-measurement.md`:
method, identifiers, aggregate counts only. No page content, no transcribed values, no
candidate markdown is recorded here or committed anywhere.

## Method

The question needs ground truth, and neither candidate can supply it — a broken text
layer cannot grade itself, and model-model agreement is not truth. So:

1. **Row selection was mechanical.** One row per table page of the 9-paper manifest
   (`2026-08-20_lane-comparison-manifest.json`), chosen by `sha256(paper|page)` modulo
   the count of eligible rows. Neither the owner nor the agent picked them. Eligibility:
   a y-band carrying >= 3 numeric tokens with >= 2 neighbouring numeric bands within
   twice the page's own median band pitch — which excludes prose lines that happen to
   contain numbers.
2. **Transcription was blind.** Each row was rendered at 190 dpi with red rules
   bracketing the target, and transcribed from the image alone, before any tool output
   was produced or seen. Blank cells were recorded explicitly, because whether a reader
   puts a number into a cell that is empty on the page is the measurement.
3. **Three readings of the same pages**: the native lane
   (`BornDigitalDetector.extract_structured`), `qwen3-vl:30b-a3b-instruct` local via
   ollama (free), and `gemini-3-flash-preview` (~$0.003 total). Engines were called
   directly, so orchestrator routing could not influence the comparison.
4. **Rows were matched by numeric multiset, never by row index.** Index matching is
   exactly what makes the existing drift detector convict correct output; a scorer that
   repeated it would have reproduced the bug it was built to test for.

The falsifier was written down before any data existed: *if the local model's
invented-plus-missing count is not lower than native's, the inversion does not hold on
this corpus and native stays the referee.*

## Result

| | native | qwen (local, free) | gemini |
|---|---|---|---|
| rows found | 12 / 13 | **13** | **13** |
| **rows exactly right** | 8 | **12** | 11 |
| values missing | 10 | 7 | **4** |
| values invented | 11 | 7 | **3** |
| row labels lost | 2 | **0** | **0** |
| row labels wrong | 0 | 0 | 0 |

invented + missing: native **21**, qwen **14**, gemini **7**. The falsifier is not met;
the inversion holds.

Failures are concentrated, not diffuse. ~~Native is clean on 11 rows and catastrophic on
two.~~ **Corrected (GH-338): the arithmetic does not close.** Native found 12 of 13 and
was exactly right on 8, leaving 4 found-but-wrong plus 1 never found. "Clean on 11" plus
"catastrophic on two" sums to 13 and silently counts the unfound row as clean.

What the aggregates DO support, over the 12 found: 8 exact, 4 found-but-wrong, and those
4 carry all 21 invented+missing between them. How many of the 4 are "catastrophic" is
**not determined** by the aggregate -- it is between 2 and 4, depending on whether the 2
label-lost rows also carry invented or missing values, which this log does not record.
Clean is correspondingly at most 10.

Stating an exact 2 here would repeat the very error being corrected: a distributional
claim the aggregate cannot carry (GH-421 review). Qwen is exact on 12 and bad on exactly one, with an invented count equal to its missing
count.

That equality was read here as a mis-matched row, and therefore as evidence qwen's true
score is better than recorded. **GH-338: do not spend that tell twice.** Leftover
invented ≈ missing is the same signature used to catch the two earlier scorer bugs. After
the switch to numeric-multiset matching it is either a REMAINING scorer bug or a real
wrong reading — it cannot be assumed to be the benign one without re-running the grader,
which this log does not make possible (see the caveat below).

Sample size is 13 rows. This is enough to reject "native is the trustworthy baseline";
it is not enough to rank qwen against gemini.

## Two scorer bugs that nearly produced the opposite conclusion

Recorded because both are the same class of defect socr itself has, and because an
earlier draft of this measurement had qwen scoring *worse* than native:

1. **LaTeX delimiters not stripped.** `$-5.78$` never compared equal to `-5.78`, so
   every cell counted as both missing AND invented.
2. **Markdown outer pipes not stripped.** `| 5 | -5.78 |` splits to
   `['', '5', '-5.78', '']`, so the row's real label was read as blank and counted as an
   *invented value* — manufacturing "1 invented, 1 label lost" on nearly every row.

Both were caught by the same tell: invented ≈ missing on a row that, inspected by hand,
matched the transcription character for character. That tell is worth keeping.

## Consequences

- **Native cannot referee model output on tables.** It loses row labels (2 here; #331
  measures 18/18, 37/51 and 23/28 orphaned-stub rows on three corpus pages) and it
  relocates values across rows. Grading a model against it produces false convictions —
  10 of them on a page verified correct cell by cell
  (`docs/log/2026-08-29_gh326-binder-gate-dry-run.md`). ~~that file is still not in the
  repo, which is the same hole #331 names~~ — **corrected (GH-422).** The earlier
  citation here was a placeholder filename, and replacing it with the real one was the
  fix; carrying the "not in the repo" clause across undid that. The dry-run log is
  tracked on main (added by #332, `4a2ea70`) and is the artefact this bullet cites. The
  reproducibility hole that IS still open is the one below — the 13 identifiers and the
  scorer, tracked on #338 — not this file.
- ~~**#326 cannot be the gate as written.**~~ **Retracted (GH-338).** This experiment
  asks whether a reconstructed ROW matches a blind transcription. `binding.py` asks
  whether a native WORD sits inside a cell bbox (`model_unbound`). Different instruments,
  different questions — and #331 already establishes that the native grid poisons
  index-matched value drift, which is the finding that actually holds. It does not show
  that word geometry cannot assert "a numeric token is present or absent on the page",
  which is the remaining role this log itself assigns the text layer and is what a scoped
  #326 would be. **Order of record stays #330 → #326 → #322.** Do not close or re-scope
  #326 from this log.
- The binder's own coverage number cited here as "#330" is the 13-bindable scoreboard
  from **#332** (`fully_checked` 0/13). #330 as filed is **0/15**. Different denominators;
  see the sample caveat below.
- **The cheap option is competitive, not a winner on the registered metric.** The free
  local model scored best on EXACT ROWS (12 vs 11). On invented+missing — the
  pre-registered falsifier — gemini won (7 vs 14). Escalating to paid cloud is not what
  fixes tables; trusting the model over the text layer is. Which model to trust is not
  settled by 13 rows.

## Sample and reproducibility caveats (GH-338)

Two acceptance items on GH-338 need artefacts this log did not record, and they are
**not** supplied here:

- **The 13 identifiers.** ~~never listed~~ — **supplied below (GH-338).** The scorer is
  still missing, so this is the reproducibility half that now lands.
- **The scorer.** Still not in the repo, and no content-free result hash was recorded.
  An earlier draft of this same measurement concluded the OPPOSITE (qwen worse than
  native) because of two stripping defects. A measurement that once inverted cannot be a
  decision of record until someone else can re-run the grader. Identifiers and the scorer
  are not page content, so the content-free convention does not forbid committing them.

Until the scorer lands, treat the direction as evidence and the numbers as unreproduced.

## The 13, and the 2 dropped (GH-338)

The original selection script was never committed, so the 13 cannot be RECOVERED, only
recomputed. `2026-08-30_row-eligibility.py` (committed alongside this log) applies the
rule exactly as this log states it -- a y-band with >= 3 numeric tokens and >= 2
neighbouring numeric bands within twice the page's own median band pitch -- reusing
socr's own `_is_numeric_word` and `round(y0)` banding rather than reimplementing either.

It returns **13 of the manifest's 15** `kind: table` pages, which is the count this log
reported. That is corroboration, not proof: a second implementation of the same written
rule agreeing on the split is evidence the rule was applied as described, and nothing
more.

| paper | page | eligible bands |
|---|---|---|
| cochrane_piazzesi 2002 | 7 | 6 |
| cochrane_piazzesi 2002 | 10 | 4 |
| cochrane_piazzesi 2002 | 12 | 6 |
| cochrane_piazzesi 2002 | 15 | 1 |
| gertler_karadi 2015 | 14 | 8 |
| gertler_karadi 2015 | 15 | 3 |
| gertler_karadi 2015 | 16 | 2 |
| nakamura_steinsson 2018 | 13 | 11 |
| nakamura_steinsson 2018 | 43 | 4 |
| nakamura_steinsson 2018 | 44 | 11 |
| pflueger_rinaldi 2020 | 34 | 3 |
| bauer_swanson 2022 | 21 | 13 |
| bauer_swanson 2023 | 20 | 13 |
| **dropped** — nakamura_steinsson 2018 | **42** | **0** |
| **dropped** — kaminska_mumtaz_sustek 2021 | **39** | **0** |

**Under the recomputation, Nakamura p42 is excluded** -- it has no band anywhere on the
page carrying three numeric tokens with two numeric neighbours. p42 is #326's named
fixture and one of the three no-grid pages, so the sample cannot speak to that fixture.

~~it drops the pages where native banding fails~~ -- **narrowed (GH-505).** One page
failing the band rule does not show the rule drops every page where native banding or
grid reconstruction fails; those are different diagnostics and this recompute measured
only the first. What it establishes is the specific fact: the page #326 is named for
fails the measured band condition, so a sample built by following the written rule
excludes it. The general worry -- that a numeric-band eligibility test preferentially
drops the pages a native-vs-model comparison most needs -- is plausible and stays
UNMEASURED here.

~~This settles the sample question~~ -- **corrected (GH-501).** It does not. The original
selection script and its selected rows are gone, so agreeing on the COUNT cannot show
that the historical 13 were these 13, and therefore cannot prove the original sample
excluded p42. What is established is narrower and still enough to matter: the rule as
written excludes p42, so a sample built by following it could not have spoken to #326.
The retraction already recorded above rests on its own reasoning (different instruments)
and does not depend on this.

In the RECOMPUTED set (GH-505: this distribution is a property of the table above, not a
recovered fact about the historical 13), two of nine papers contribute nothing and one
contributes four of the thirteen rows, so those 13 are not thirteen independent documents
either. That is a further reason 13 rows
cannot rank qwen against gemini. ~~native's failures here are not concentrated in one
paper~~ -- **dropped (GH-501):** the content-free record carries no per-paper
native-failure map, so that clause asserted a distribution nothing here measures.

## What this does NOT settle

Misplacement of a value that genuinely appears on the page is invisible to every
content-free automatic check available today. #270's fabrication evidence stands — it
was grounded in page images judged by two independent vendors, not in the native
rowizer, so it is not an artefact of the broken baseline.
