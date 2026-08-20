# Native vs model: what socr actually ships when it has a choice

2026-08-20. 21 pages across 9 economics papers, run so that a discarded attempt could
survive in socr's cache. On 8 of them two candidates survived and socr had to choose — see
"Not every page yielded two candidates" below before quoting that as coverage.

This is the measurement that produced issues #259, #262 and #263, and the case for
replacing multiset comparison as the winner-side verification oracle. **That replacement
has not landed.** `main` still verifies by multiset today; the binding oracle (#266) and
the selection change (#269) are both unmerged. Read this record as the argument, not as a
description of current behaviour.

**The page content is not here and cannot be.** The corpus is copyrighted and this repo
is public. What is committed is the method, the per-page routing verdicts, and the
statistics — enough to re-run it and to check the arithmetic, not enough to redistribute
anyone's paper.

## The number, stated precisely

On the **8 pages where socr held two or more candidate outputs and had to choose, it
shipped the worse one 7 times.** One was a genuine tie. Native won **zero** outright.

Do not restate that as 7 of 21. Across all 21 pages the shipped engine was the local
model on 12, native on 8, and the cloud model on 1 — the model already wins most pages.
The 7/8 is specifically about the contested subset, which is where the routing decision
is actually exercised.

## Method

`2026-08-20_lane-comparison-select.py` builds the manifest;
`2026-08-20_lane-comparison-runner.py` consumes it. The manifest itself is committed as
`2026-08-20_lane-comparison-manifest.json`.

The selection script is committed because this record claims the pages were chosen
deterministically rather than by eye, and that claim was previously unverifiable — the
runner takes the manifest as input and contains no selection code at all.

- Pages chosen deterministically, not by eye: parenthesised-standard-error density for
  tables, math-glyph density for equations, image presence with low text for figures.
- Each document excerpted to its own PDF, then `socr process --agentic --no-native-first`
  so the model is forced on every page and a discarded attempt can still survive in socr's
  own cache. That is the whole trick: #259 is about a correct model answer being thrown
  away, so the thrown-away answer had to be recoverable.
- **Not every page yielded two candidates.** 13 of the 21 have only one surviving cached
  attempt — either the other lane produced nothing cacheable, or it produced something that
  was not cached. Which of the two is not established by this record. The 8 pages with two
  or more candidates are the contested set, and they are the only pages where the routing
  decision was actually exercised. Do not read this as "both lanes captured on every page".
- The cache is then mined per `(page_num, engine)`. Cache entries carry `page_num`; an
  earlier version of this runner mined document-wide and attributed one document's output
  to every page in it. That bug is why the runner is committed rather than described.
- Per-document subprocess timeout. socr can hang before it ever reaches the model — one
  observed run sat 12 minutes at 1.96 s of CPU with no output directory.

## Judging

Two models judged each contested page independently against the page image: which output
matches what the page prints. They agreed on 7 of 8.

**On the eighth they flatly contradicted each other**, each describing the same character
fragments as coming from the other lane. That was resolved by reading the two candidate
files directly: native produced the 176-character shredded output, the model returned the
caption verbatim. One judge was right, one misattributed. **That resolution is a hand
check, not a third judge**, and is recorded as such.

## The finding that invalidates the old oracle

The failure mode is **structure, not values**. Native rarely drops a number; it drops the
binding between numbers and their rows and columns. A flattened regression table keeps
every one of its 152 decimals and tells you nothing about which coefficient belongs to
which specification.

Therefore: **a scrambled table has an identical numeric multiset to a correct one.**

Stated precisely, after a peer review corrected an earlier overstatement in this file: it
is **the winner-side verification chain** that was multiset-blind — `native_verifier`,
`source_evidence`, the header anchors — plus the old benchmark scorer, which falls back to
a numeric multiset on shape mismatch (`benchmark/scorer.py:462-498`). It is **not** true
that every check in the codebase compared multisets.

`benchmark/table_exactness.py` already does the right thing and has since #123: row-label
paths, native lanes, and a global monotone injective lane-to-column map (lines 222-412),
explicitly built because "that fallback is blind to the failure". Anyone implementing the
binding oracle should read it first — the prior art is in this repo and was not referenced
when the replacement was designed.

The blindness was in the chain that decides what ships. That is why months of CI, tests
and review never caught it, and a person looking at one page caught it in a minute.

The `decimals` counts in the verdict file are descriptive statistics. **They are not an
oracle and must not be used as one.** That is the point of the whole exercise.

## Against the model

Two defects across all eight pages: one transcription slip (`1.11` where the page prints
`1.10`) and one row-label shift. Both judges found the same two independently and neither
could produce a third. Recorded because a measurement that only indicts one side deserves
suspicion.

Neither lane captured any plotted chart data — axes, dates, series values. Both recover
only the text scaffolding around a figure. That is a gap in both.

## What came out of it

- **#259** — a flagged-but-correct model table discarded, native shipped in its place.
- **#262** — the failed-table placeholder shipping while a faithful 36-value extraction
  sat in the same run's cache, taking the page's prose and equations with it.
- **#263** — the rotation refusal gated on a table being detected, so a rotated figure
  page shipped `SUCCESS` with 176 characters of reversed fragments. (PR #265, merged.)
- The ratified replacement for multiset comparison: bind each value to its row-label path
  and column-header path from page geometry, and compare those.

## Re-running it

The runner needs a local corpus and a working Ollama with
`qwen3-vl:30b-a3b-instruct`. Build the manifest with
`2026-08-20_lane-comparison-select.py` (pass it a glob of your PDFs), then point
`LANE_CAMPAIGN_DIR` at the directory holding it and run
`2026-08-20_lane-comparison-runner.py`.

`select.py` holds the selection heuristics and is deterministic; the runner consumes the
manifest and contains no selection code. Budget roughly 14 minutes per dense table page on
a local 30B model — a session observation on one machine, not a benchmark. The wall-clock,
not the money, is the cost.

## What in this record cannot be checked from the repo

Marked explicitly, because a durable record should not contain claims a future reader
cannot confirm.

The **7-worse / 1-tie / 0-native-win** split is the central number and it is **not in the
verdict JSON** — that file records which engine shipped, not which output was better. The
8-page contested denominator IS derivable (count the rows with two or more candidates).
The quality judgement on those 8 exists only as prose here, because it came from two model
judges reading page images that cannot be committed.

Also session record rather than repo-checkable: the 176-character shredded output, the
`1.11`-for-`1.10` slip, the row-label shift, the judges' 7-of-8 agreement, and the hang
observed at 12 minutes elapsed against 1.96 s of CPU.

Four more claims in this file are assertion rather than evidence. Marked here rather than
softened away, because a reader deserves to know which is which:

- **"the ratified replacement"** for multiset comparison — the ratification happened in a
  working session. No record of it exists in this repo, and the work is unmerged.
- **"multiset comparison was abandoned"** — it has NOT been, and the opening now says so.
  `main` verifies by multiset today. The replacement is proposed, partly built, unmerged.
- **"months of CI, tests and review never caught it"** — rhetorically true, historically
  uncheckable from the repo.
- **"roughly 14 minutes per dense table page"** — one machine, one session, not a benchmark.
- **"Neither lane captured any plotted chart data — axes, dates, series values"** — a
  content-level observation from reading the page images. The images are not committed, so
  a reader cannot confirm it here. Found on the third fact-check pass of this file, after
  two earlier passes claimed completeness.

Everything else — the arithmetic, the per-page dispositions, the flags, the decimal counts,
the selection code, the method — is in the committed files.

## Fields in the verdict file

`doc`, `page`, `kind` (table/equation/figure), `shipped_engine`, `page_status`, `flags`
(the `native_table_*` flags that fired), `candidates` (which engines produced an attempt
that survived in cache), `decimals` (count per engine — descriptive only).

A page with two or more `candidates` is one where socr had a choice. Those are the eight.
