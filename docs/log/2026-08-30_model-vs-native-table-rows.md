# Model vs native on tables — the referee is the thing that is wrong

2026-08-30. Settles the question raised by the owner: *"why don't we check quality and
go to an LLM?"* The answer is that socr already escalates every table page to a model.
What was never measured is whether the thing grading that model is fit to grade it.

It is not. On 13 blind-transcribed rows, **both models beat the native text layer on
every measure**, and the free local model wins outright.

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

Failures are concentrated, not diffuse. Native is clean on 11 rows and catastrophic on
two. Qwen is exact on 12 and bad on exactly one, with an invented count equal to its
missing count — the signature of a mis-matched row rather than a wrong reading, so its
true score is probably better than recorded here.

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
  10 of them on a page verified correct cell by cell (`2026-08-29_gh326-...`).
- **#326 cannot be the gate as written.** A native-geometric binder cannot authorize a
  model when its reference is bent, and the binder independently fully-checks 0 of 13
  real tables (#330).
- **The cheap option wins.** The free local model scored best on exact rows. Escalating
  to paid cloud is not what fixes tables; trusting the model over the text layer is.

## What this does NOT settle

Misplacement of a value that genuinely appears on the page is invisible to every
content-free automatic check available today. #270's fabrication evidence stands — it
was grounded in page images judged by two independent vendors, not in the native
rowizer, so it is not an artefact of the broken baseline.
