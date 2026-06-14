# Dev log — CE summary-row validation (TICKET-D1) (2026-06-14)

Branch: `fix/46-ce-summary-rows` (off `main`). Closes the open half of the
2026-06-14 D1 field note: the per-forecaster body of the Consensus Forecasts US
table was confirmed earlier; the bottom **summary** and **comparison** rows were
"not yet inspected." This session inspected them.

## Method
- Image: `scratch/bench/ce/202606_p4.png` (US forecaster page, June 2026; 1653×2339).
- Model: `qwen3-vl:30b-a3b-instruct` via Ollama (`/api/generate`, `temperature=0`),
  exact production prompt `src/socr/prompts/table_extract.md`. One pass, ~1m45s on Metal.
- Ground truth: read directly from high-DPI crops of the table (`sips` crops of the
  summary band and a right-edge strip isolating the two rightmost column groups,
  Employment Costs vs Auto & Light Truck Sales). Cell-by-cell compare, not eyeball.

## Result

**Statistical summary rows — exact (120/120 cells):** every value in
`Consensus (Mean)`, `Last Month's Mean`, `3 Months Ago`, `High`, `Low`, and
`Standard Deviation` matched the image, all on the correct 2026/2027 lane, across
all 10 variable groups. The D1 core concern — the VLM un-pairing the year columns
on summary rows — **does not occur** with the instruct MoE. These rows are as clean
as the forecaster body.

**Comparison Forecasts — one residual misalignment.**
- `IMF (Apr. '26)` and `OECD (Jun. '26)`: correct. Sparse rows (only GDP, Personal
  Consumption, Consumer Prices populated) landed on the right lanes.
- `CBO (Feb. '26)`: digits all correct, but the `Employment Costs` pair `3.4 3.4`
  was placed one lane to the right, into the rightmost `Auto & Light Truck Sales`
  column (which is blank in the source); `Employment Costs` was left blank. Verified
  against the right-edge strip: source has Core PCE `2.9 2.4` | Producer Prices blank
  | **Employment Costs `3.4 3.4`** | Auto blank.

## Reading
The paired-column failure D1 was written for is **not** a summary-row problem — it is
a **sparse-row** problem. On dense rows (every cell filled, including the statistical
summaries) the model tracks lanes perfectly. It slides a value into the wrong lane only
when a row has many blank cells and few anchors, as in the institutional comparison
rows. CBO is the worst case here: a long run of blanks (Industrial Production, Producer
Prices) before the lone Employment Costs value, with no Auto value to "stop" against.

So D1's prompt language ("summary rows must use the same number of columns") is already
satisfied for the statistical block. The remaining gap is lane-tracking across blank
runs in sparse comparison rows — a different, narrower fix.

## Status / follow-up
- D1 statistical-summary objective: **met** (validated, no code change needed).
- New, narrower issue: sparse-row lane drift (CBO). Options, cheapest first:
  1. Prompt nudge in `table_extract.md`: for rows with blank cells, count column
     positions from the header and keep each value under its own header lane; never
     left-pack or right-pack values across blanks.
  2. If the prompt nudge doesn't hold, this is a candidate for the structural
     reconcile path (column-count is fixed by the header; a value with an ambiguous
     lane could be checked against the per-column data type).
- This was a single-page check. Firing-rate across the 27-PDF CE set (TICKET-Z1)
  would tell us how often sparse-row drift actually occurs; one CBO row is not a rate.
