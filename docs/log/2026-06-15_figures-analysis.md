# Dev log — Figure extraction pipeline analysis (#47) (2026-06-15)

Branch: `feat/47-figures-analysis` (off `main`). Investigative per issue #47 — assess
`--save-figures`, localization, and the local-first (TICKET-C2) figure descriptions.
Read alongside the general extraction method (#49): the extract/verify/escalate lens
applies here, with one important asymmetry (figures have no free verifier — see below).

## Method
- Corpus: the OCR'd papers library (`.../Library/Papers/ocr/`), which already holds
  extracted figure PNGs per paper. Probe paper: Ferrari-Minesso & Siena, *Private money
  and public debt* (ECB, 2026) — 25 extracted figures.
- Localization: inspected extracted PNGs directly (vector + embedded-image strategies
  already covered by `tests/test_figure_pass.py`: false-positive prose, dedup, scanned
  page, rotated-vector crop, dashboard split).
- Description quality: ran the **production** path on real figures —
  `qwen3-vl:30b-a3b-instruct` via Ollama, the exact `_build_figure_prompt` base prompt
  (`engines/gemini_api.py`), `temperature=0`, no page context (the floor; orchestrator
  adds context, which can only help). Two figures chosen to contrast: a dense multi-panel
  data chart and a label-rich schematic. Each description compared cell-by-claim against
  the image.
- **Sample is small (n=1 paper, 2 descriptions + 2 localization spot-checks).** Findings
  are directional, not rates. Firing rates need a batch.

## Finding 1 — Localization: two concrete defects
1. **Letterhead/logo false positive.** `figure_1_page1.png` is the ECB letterhead banner
   (logo + "EUROPEAN CENTRAL BANK / EUROSYSTEM"), extracted as a "figure." The
   `HEADER_FOOTER_MARGIN` (0.1) gate did not catch a centered title-page logo. A logo is
   a placed raster passing the embedded-image strategy; it needs a content/position
   filter (title-page band, low ink-density, or known-logo aspect) to suppress.
2. **Silent truncation at `max_total=25`.** At least five library papers have *exactly*
   25 figure PNGs — i.e. they hit the cap and the rest were dropped with no record. For a
   research corpus this is a silent-coverage hole (cf. the table work's rule: no silent
   caps — log what was dropped). The cap should at minimum be logged, ideally raised or
   made per-document.

## Finding 2 — Description quality: gist reliable, specifics hallucinated
The headline result. Across both figures the model is **strong on structure/gist and
unreliable on verbatim specifics** — and the prompt's instruction to "be specific about
numbers, labels, and relationships" actively *induces confident fabrication*.

**Figure 6 (stability map, 5x5 red/blue phase grid):**
- Correct: it is a 5x5 stability map; columns vary the Taylor-rule inflation coefficient
  theta_pi (0.2..0.4 by 0.05); per-panel x-axis = "Stablecoin share in total US debt";
  blue = stable / red = unstable; the qualitative story (higher theta_pi enlarges the
  stable region; higher risk premium shrinks it).
- Hallucinated: "a series of **ten** charts" (then says 5x5 = 25 — internal
  contradiction); an **invented per-panel y-axis** ("Inflation reaction theta_pi, 0 to 6")
  — theta_pi is the per-column constant, not a per-panel axis; **fabricated numeric
  thresholds** ("stable for x < 3", "up to x ~ 15") it cannot read at this resolution.

**Figure (model schematic, agent-flow diagram):**
- Correct: the full cast of agents (Central bank, Government, Households, Firms, Retailers,
  Stablecoin issuer), the Domestic / International split, the "Exchange rate determination"
  bar, the three international channels, and the green-vs-black arrow semantics.
- Wrong (swapped arrow labels — exactly the "relationships" the prompt asks for): claims
  Central bank -> Government is labeled "Bonds" (actual: "Sets the risk-free rate"); claims
  Government -> Households "Cash" (actual: "Cash" is Central bank -> Households; Gov ->
  Households is "Bonds"); claims the "Stablecoin" arrow goes issuer -> Central bank (actual:
  issuer -> Households; the green left arrow is "Bonds").

**Pattern:** reliable as a *searchable abstract* of what a figure is about; **not**
reliable as a source of verbatim figure facts (axis ranges, thresholds, which arrow is
which). Latency ~57s (chart) / ~103s (schematic) local on Metal.

## Finding 3 — Architectural: the #49 free verifier does NOT generalize to figures
For tables, the verify layer is free because the born-digital text layer holds the ground
truth (digits, column geometry). **Figures break this.** A data chart's ground truth lives
in *rendered pixels*, not text — PyMuPDF can recover a figure's **text labels** (axis
titles, legend, panel headers) but not its **data values, curves, or arrow directions**.
So:
- **Partially verifiable for free:** the *labels* the VLM claims can be checked against the
  page's embedded text — this alone would catch the invented "theta_pi 0..6" axis and some
  swapped labels.
- **Not verifiable for free:** data values, trends, thresholds, arrow semantics — no cheap
  oracle exists. Verifying these means a second VLM pass / self-consistency voting (agentic).

So figures sit on the **weak-verify** side of #49: single-pass VLM for the gist + a cheap
label cross-check; treat data-value claims as unverified unless escalated. This is the
opposite end from clean born-digital tables (strong free verify).

## Finding 4 — Prompt induces fabrication
`_build_figure_prompt` says "Be specific about numbers, labels, and relationships shown."
On figures whose numbers can't be read, the model obliges by inventing them. The table
prompt already solved the analog ("transcribe what you can see rather than guessing a
plausible number"). The figure prompt needs the same anti-fabrication clause and a
honest-uncertainty instruction (describe what is visible; do not guess values/labels you
cannot read; say when a value is unreadable).

## Follow-up tickets (proposed)
- **F1 — Anti-fabrication figure prompt.** Add the table prompt's no-guessing discipline
  to `_build_figure_prompt`; separate "what the figure is" (reliable) from "specific
  values" (only if clearly legible). Re-run the two probe figures to confirm the invented
  axis/thresholds disappear.
- **F2 — Suppress letterhead/logo false positives.** Position/content filter for title-page
  logos in the embedded-image strategy. Add a regression fixture (logo banner -> 0 figures).
- **F3 — No silent figure cap.** Log when `max_total` truncates; consider per-document or
  raised cap. (Mirrors the corpus's no-silent-caps rule.)
- **F4 — Free label cross-check (the #49 verify layer for figures).** Compare VLM-claimed
  axis/legend labels against the page's embedded text; flag mismatches. Cheapest partial
  verifier; does not touch data values.
- **F5 (bigger, later) — figure data-value verification.** Only if verbatim chart accuracy
  is ever required: second-pass / self-consistency. Likely not worth it if descriptions are
  scoped as gist. Decide after F1.
