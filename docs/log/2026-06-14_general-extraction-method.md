# Dev log — General extraction method: extract / verify / escalate (2026-06-14)

Branch: `feat/49-general-extraction-method` (off `main`). Issue: #49. Architecture
decision following the #46 table validation (`2026-06-14_ce-summary-row-validation.md`).

## The question
Do we need **agentic crop-and-reconcile every time** to extract tables (and figures)
reliably, or is there a **general method**? The recurring worry is that high-fidelity
extraction implies an expensive per-element VLM loop on every page.

## The evidence that settled it
The CE validation ran a **single vision-model pass** — `qwen3-vl:30b-a3b-instruct`,
full page, one `/api/generate` call, the production prompt `prompts/table_extract.md`,
no crop, no reconcile, no agentic ladder. Result on a dense forecaster table: 120/120
statistical-summary cells exact, full forecaster body correct, **one** residual error
(CBO comparison row: `Employment Costs 3.4 3.4` slid one column lane into blank `Auto
sales`).

Conclusion: a single VLM pass **is** the general extraction method. Agentic is
escalation, not the default.

## The decision — three separable layers
The "agentic every time?" confusion comes from collapsing three distinct concerns:

| Layer | Question it answers | Cost |
|-------|--------------------|------|
| **Extract** | how do I get the element out? | single VLM pass (qwen local / Gemini), or native text where the layer is trustworthy |
| **Verify** | how do I know it's right? | **free** native cross-check — text-layer geometry, header column count — NOT a second model |
| **Escalate** | what do I do when verify fails? | agentic crop-reconcile / second VLM — **only on a fired signal** |

### Why verify should be free, not a second model
The single error we observed (lane drift) is invisible to a lone VLM pass but trivially
catchable without spending a second call: on a born-digital page, PyMuPDF already knows
the column x-positions, and the table header fixes the column count. A value that lands
outside its header lane is a deterministic, zero-cost red flag.

This is the genuinely new idea relative to today's pipeline. The current judge
(`judge/ollama_judge.py`) is itself a VLM call — the verify layer is paid. The decision
is to put a **free deterministic verifier in front of the VLM judge** for table/figure
pages on born-digital PDFs, so the paid judge (and any agentic escalation) only fires
when the cheap check actually disagrees.

## Scope boundary — born-digital vs scans
This holds for **born-digital** PDFs (≈the entire corpus per the 2026-06-06 log: one
scanned, table-less paper in the pre-1996 set). **Pure scans** have no native text layer
to verify against; there the only checks are a second VLM pass or self-consistency
voting, which is closer to agentic. So the precise default is:

> single-pass VLM + free native verification as the general method; agentic reserved
> for scans-with-disagreement, never every page.

## How this maps onto the existing architecture
This is a refinement of the agentic mode (`docs/ARCHITECTURE.md` "Agentic, cost-aware"),
not a new pipeline. The provider ladder already does extract→judge→escalate; this
decision (a) names the layers, (b) inserts a free deterministic verifier ahead of the
VLM judge for structured content on born-digital pages, and (c) records that single-pass
is sufficient as the default extract step — agentic crop-reconcile is the tail, not the
trunk.

## Follow-ups (separate tickets, not this doc)
- Implement the free native verifier for table pages (column-lane / column-count check
  against the text layer); wire it ahead of the VLM judge on born-digital pages.
- Apply the same extract/verify/escalate lens to **figures** (#47): is figure
  description single-pass-VLM with a free verify, or does it need agentic? Investigate.
- D2 (sparse-row lane drift, #46): the first concrete case the free verifier should catch.
- Quantify firing rates on a real batch (CE set, Z1) before tuning thresholds.
