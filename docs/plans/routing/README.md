# Routing design — OCR only what needs it, check everything that ships

Opened 2026-10-09. Diagram: [`diagram.html`](diagram.html) (current routing on top, target below;
open it in a browser). Issues: #1050 (coverage check), #1051 (agreement check + Jev).

## Goal

Cut OCR time and cost by sending a model only the parts of a page that need one, **without**
shipping any text that nothing checked. A wrong or dropped number is worse than a missing one.

## Today (read from `main@c4937472`)

Each page is classified once by text-layer rules (`core/born_digital.py`) and takes one lane:

| Lane | Model sees | Checked by |
|---|---|---|
| Trusted native (`_is_agentic_trusted_native`) | nothing | **nothing** |
| Tables / equations / corrupt math | crops | table verifiers; LaTeX validation |
| Figures / charts | page or region image | chart reader where it applies |
| Scanned or defect-flagged | **whole page** | page judge, then cost ladder |

Three problems:

1. **Local OCR is priced at `0.0`** in `core/providers.py`. The ladder sorts by price, so GPU time
   and long-input hallucination never enter the decision.
2. **The free lane is the unchecked lane.** 245 of 423 pages took it in one run (#317).
3. **Whole page or nothing.** One bad paragraph sends the full page to the VLM, and the model
   output replaces the good native prose around it.

## Target

Native text ships only after two model-free checks; models see gaps and bad lines, not pages.

1. **Coverage check (#1050).** Render the page, find ink, subtract text-layer word boxes. Text-like
   ink with no words over it → OCR that crop only. Figure-lane boxes mask the check (ink inside a
   figure is expected to have no text). Finds **missing** text.
2. **Agreement check (#1051).** Re-read the rendered page with a cheap CPU OCR, align line by
   line with the native text. Disagreeing lines → crop to the VLM. Finds **wrong** text.
3. **Jev as the text-only judge (#1051).** Jev cannot see images. It judges text pairs: whether a
   word difference is real or an OCR misread, and the per-region route. **Digit disagreements
   never go through Jev**; they always go to the VLM.
4. **Fail closed.** Any check that cannot run or cannot decide sends the page to today's
   whole-page route.
5. **Honest local price.** GPU seconds enter the ladder's cost. Not filed yet (see Open).

Lanes that already work (tables, equations, figures) are unchanged.

## Prior evidence that constrains this

- **`docs/plans/fake-native-pages`**: 72 of 2972 pages (2.4%) are old scans with a baked-in OCR
  layer, 71 of them from two documents; raster coverage catches them. Since then #961 routes an
  invisible text layer over a page raster to OCR. The coverage check must not re-solve that.
- **Same plan, ticket B2 — lexical quality signal: CLOSED, NOT BUILT.** A1 measured it as noise,
  not coverage. This is direct evidence against Jev's garble-classification use in #1051; that
  use needs a new measurement that beats B2's finding, or it is dropped.
- **Judge agreement is not corroboration.** Two vendor judges agreed on a page and both missed a
  shifted row. Hence the digit rule.

## Order of work

Measurements gate builds. Each step names what would stop the plan.

1. **#1050 measurement.** Coverage check alone over trusted-native pages: how many pages carry
   text-like uncovered regions, with crops inspected by hand. **Stop condition:** about zero real
   hits → #1050 shrinks to recording the coverage pass as a witness; no crop route.
2. **#1051 measurement.** CPU re-read vs native text on the same pages: disagreeing line pairs,
   split digit / non-digit; hand-label a sample and score Jev against a plain string-distance rule.
   **Stop condition:** Jev no better than the rule → use the rule, drop Jev.
3. **Price local by GPU time.** Needs per-page GPU seconds, which step 1 and 2 runs can record.
4. Build what survived, coverage first (it supplies the crop and splice path the agreement
   check reuses).

## Open

- **Which CPU OCR.** Tesseract is the obvious candidate; not checked whether it is installed or
  how it reads the corpus's math.
- **Splicing.** Gap and line crops must land at their reading-order position. Equation P4-R and
  the corrupt-math hybrid already splice crops into native prose; reuse that, don't build a second.
- **Figure masking interface.** How the figure lane hands its boxes to the coverage check, given
  that today they run at different points in the page loop.
- **GPU-time pricing issue.** Not filed.
- **Jev key and data handling.** Jev is a cloud service; no call has been made from socr yet.
