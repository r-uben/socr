# 2026-09-20 — GH-831: measuring the inverted-line-completeness check

READ-ONLY measurement. No `src/` change. Worktree `/tmp/wt-831m`, detached at
`fc33c92` (= `origin/main` at dispatch time). Corpus: the 380 PDFs listed in
`/tmp/gh64_pdfs.txt` (local-only file, not part of this repo — copyrighted,
not committed; only counts and basenames are recorded here).

## The invariant measured

> For every native line a markdown table row anchors to, every numeric token
> on that line must land in that row.

Zero tolerance, no new threshold — see the ticket
(`/tmp/ticket_831m.md`, issue #831) for the full background.

## What was built

A read-only probe, entirely outside `src/`, built from existing helpers only:

- `baseline_bands`, `table_blocks`, `numeric_body_rows`, `match_rows_monotonic`,
  `words_in_region` (all `socr/tables/row_corroboration.py`) — the inverted
  check is: for every candidate row `match_rows_monotonic` binds to a native
  band, does `Counter(band.tokens) - Counter(row_tokens)` have any leftover?
  Each leftover is one evicted token.
- The native `(rect, markdown)` table pairs `extract_structured` would ship
  were obtained by patching `BornDigitalDetector._verify_regions` (the last
  step before the regions are spliced into page text) to raise with the list
  it was called with, instead of re-deriving `find_tables` / the lane-stacked
  rowizer / `reconstruct_table_regions` / the chart-aware rowizer from
  scratch. This reuses the real production control flow unmodified.
- The existing presence oracle for (c) is
  `socr.tables.escalation_canary.presence_verdict_from_text`, called exactly
  as `manifest.py`'s D3 substitution calls it
  (`native_text=page.get_text("text")`,
  `candidate_markdown=extract_structured(page)`); "fires" =
  `verdict.blocks_success` (i.e. `PRESENCE_INVENTED`).

**No new geometry code was needed.** Everything above is an existing,
reused helper.

Scripts (not committed, `/tmp`-only, read-only):
`/tmp/probe_831_lib.py`, `/tmp/probe_831_controls.py`, `/tmp/probe_831_corpus.py`.

## Commands run

```
cd /tmp/wt-831m && git log --oneline -1   # fc33c92

PYTHONPATH=/tmp/wt-831m/src:/tmp ~/venvs/socr/bin/python /tmp/probe_831_controls.py

PYTHONPATH=/tmp/wt-831m/src ~/venvs/socr/bin/python /tmp/probe_831_corpus.py \
    /tmp/gh64_pdfs.txt > /tmp/probe_831_full_out.json 2> /tmp/probe_831_full_err.log
```

Both scripts assert `os.path.realpath(socr.__file__).startswith("/private/tmp/wt-831m")`
before doing anything else (the mandated canary).

The corpus run's stderr log contained real extracted numeric values from the
copyrighted corpus (the pipeline's own `reconstruct_table_regions` debug logs
list destroyed tokens at `logger.debug`/warning level on rejection). That log
was `/tmp`-only, never committed, and has been deleted after this log was
written; no value from it appears anywhere in this file.

## The funnel

| Step | Count |
| --- | --- |
| Documents listed / opened | 380 / 380 (0 failed) |
| Pages total | 4,419 |
| Pages classified born-digital (`PageAssessment.is_born_digital`) | 4,121 |
| **(a)** pages with >= 1 native markdown table region | 2,756 |
| **(b)** pages the inverted check would newly demote | 15 |
| **(c)** pages the EXISTING presence oracle already fires on | 99 |
| **(d)** overlap of (b) and (c) | 1 |
| Genuine coverage gap = (b) − (d) | 14 |

Every step above (a) has a visible, non-trivial denominator, so a small (b)
is not an artefact of a starved funnel: 2,756 pages actually carried a table
the check ran against.

## (e) — top 10 pages by evicted-token count (counts and basenames only)

| Basename : page | evicted tokens | violated rows |
| --- | --- | --- |
| mpr-2008-07.pdf : p46 | 32 | 4 |
| mpr-2021-02.pdf : p55 | 15 | 5 |
| mpr-2022-06.pdf : p69 | 15 | 1 |
| ecb-meetings-2021-economic_bulletin-p127-129.pdf : p1 | 13 | 13 |
| ecb-meetings-2021-economic_bulletin-p127-129.pdf : p2 | 13 | 13 |
| mpr-2019-07.pdf : p52 | 13 | 6 |
| ecb-meetings-2021-economic_bulletin-p127-129.pdf : p3 | 12 | 12 |
| mpr-2020-06.pdf : p60 | 10 | 5 |
| doc01.pdf : p2 | 9 | 4 |
| mpr-2021-02.pdf : p18 | 8 | 8 |

No values or page text are recorded — counts and basenames only, per the
ticket's hard boundary.

## The control (three synthetic fixtures)

Built in-process with PyMuPDF (`fitz`), each a small ruled (top/mid/bottom
rule) booktabs-style table with 1 label + 3 numeric columns (4 data rows),
run through the real `extract_structured`/`reconstruct_table_regions` path.
Ground truth for each is `page.get_text("text")` on the REOPENED PDF, never
the string passed to `insert_text` (per the ticket's PyMuPDF-edge-drop
warning — not triggered here since every mark was placed well inside the
page, but honoured as specified).

| Control | Fires? |
| --- | --- |
| Page number on the same baseline as the last data row | No |
| Footnote marker (right margin) level with a data row | No |
| Running header sharing a band with the first data row | No |

**0/3 false fires.** The presence oracle (`presence_verdict_from_text`) also
did not fire on any of the 3.

**Caveat, not a clean pass.** At the geometries tested (the foreign token
placed inside the reconstructed table's own bbox, on a row's baseline), the
production grid-builder (`reconstruct_table_regions` / the text-strategy
`find_tables` cell assignment) absorbed the foreign token INTO the
neighbouring cell's text rather than leaving it outside the row — e.g. a
page-number `"7"` planted in a column gutter was concatenated into the
adjacent cell as `"6.6 7"`, both tokens then legitimately present in the row.
So the inverted check correctly abstained, but not because it distinguished
"belongs to this row" from "shares this baseline" — the grid-builder had
already folded the foreign token into the row before the check ever ran.
I could not, with the real pipeline's own cell-building geometry, construct
a case where a legitimate same-baseline neighbour stays outside the row's
own token set while remaining inside the table's baseline-band scope. Three
fixtures do not rule out such a case existing elsewhere in the corpus; this
narrows but does not close the false-positive question.

## Is containment implementable under zero tolerance?

Yes, in the narrow sense asked: the check itself needs no new geometry code
and reuses vetted helpers, and 0/3 controls did not false-fire. But the
corpus numbers argue against calling this "done": (b)=15 out of (a)=2,756 is
a 0.5% would-demote rate, and (d)=1 means 14 of those 15 pages are a genuine
NEW coverage gap the existing presence oracle misses today — small in
absolute count, non-zero, and every one of the top-10 (e) pages is from a
real statistical-table-heavy document (MPR, ECB bulletin, a replay-binding
fixture). Whether 15 demotions (with unmeasured false-positive risk beyond
3 controls) is an acceptable trade for closing a 14-page-wide gap is a
policy call, not a mechanical one — this log stops at the measurement.

## What this log does NOT claim (per the ticket's guardrails)

- No frequency for the eviction bug itself is quoted; (b) is "pages the
  check would demote," which includes any false positives, not a measured
  defect rate.
- No claim about which grid builder is responsible.
- No claim about `_run_column_lanes` or any other mechanism — untested here.
