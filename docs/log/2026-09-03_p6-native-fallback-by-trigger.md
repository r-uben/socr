# NATIVE_FALLBACK by trigger — the P6 stage-D input (no model calls)

2026-09-03. The P6 panel (design note §8 Q1) kept the demoted-native ending as a
fourth ending on the condition that its triggers be enumerated over the corpus and
assigned one by one to native prose or the floor. This is that enumeration, produced
by `src/socr/benchmark/native_fallback_rates.py` over the Papers library (366 PDFs on
disk read cleanly, 23,190 pages, 21,651 born-digital with native text), socr at
`39dd656`. Analysis-time triggers only; `chart_asset_render_failed`,
`native_table_structure_failed` and the `p.attempts` gate are runtime and not
observable here, so these are upper bounds on pages that CAN demote, not shipped rates.
Content-free: counts only.

## The table

- pages: 23190; born-digital with native text: 21651
- not observable here (runtime): chart_asset_render_failed, native_table_structure_failed, the p.attempts gate
| trigger | pages | share of born-digital pages |
|---|---|---|
| needs_ocr_enhancement (of which corrupt math) | 3378 | 3378 (15.6%) |
|   corrupt math alone | 3130 | 3130 (14.5%) |
| unverifiable table region (TR-3 geometry hard-fail) | 420 | 420 (1.9%) |
| native table structure defective (GH-151 B1) | 606 | 606 (2.8%) |
| native table header unattributed | 0 | 0 (0.0%) |
| text-strategy grid rejected (GH-195) | 687 | 687 (3.2%) |
| ANY analysis-time trigger | 4486 | 4486 (20.7%) |
| any trigger on a page WITHOUT a table signal | 2893 | 2893 (13.4%) |
- documents with at least one triggered page: 313
- max triggered pages in one document: 495

## What it says

- **The fourth ending is, in practice, the corrupt-math ending.** `needs_ocr_enhancement`
  fires on 15.6% of born-digital pages and 14.5 points of that is corrupt math (font-map
  mojibake in maths). Every table trigger together is under 8%, and they overlap.
- **Two thirds of the triggered pages have no table signal** (13.4% of 20.7%). Those pages
  are not structure-class; under today's routing they leave the free lane for the ladder
  because the native layer is known-damaged, and if the ladder is refused they ship
  demoted native prose with corrupt maths inside it.
- **The table triggers are small and already fail closed elsewhere**: an unverifiable
  region or a structurally defective grid on a routed page ends in the D3 floor or the P2
  floor; their `NATIVE_FALLBACK` residue is the case where the page never reached a
  model rung. That residue is at most 2–3% and shrinks as the ladder flips on.
- `native_table_header_unattributed` fired on zero pages: the HARD term abstains on this
  corpus (as #245's design predicted), so it contributes nothing to the ending today.

## Assignment, trigger by trigger (recommendation for stage D)

| trigger | assign to | reason |
|---|---|---|
| corrupt math (`needs_ocr_enhancement` via `has_corrupt_math`) | **route to the corrupt-math region lane by default; if the lane is off or refuses, ship native prose with the maths regions marked, not demoted whole-page** | the prose on these pages is intact; only the maths is mojibake. Whole-page WARNING punishes 14% of pages for a region-sized defect. The region lane exists (`recover_corrupt_math`) and is opt-in today. |
| `needs_ocr_enhancement` for the two rotation cases | floor (rotated shred already has its own floor) | the native layer is confetti, not prose |
| unverifiable table region / structure defective / text-grid rejected | **floor when the page reached a rung and every rung was refused (that is P2, already shipped); native prose WARNING only when no rung ran** | consistent with the P2 ruling: never ship the grid the verifier refused |
| `chart_asset_render_failed` (runtime) | native prose, WARNING, with the render failure surfaced | nothing about the text is wrong |
| `native_table_header_unattributed` | delete the trigger from the ending's predicate once #245's delegation has a corpus measurement; today it is inert | zero firings |

The consequence for stage D: the fourth ending does not merge into one of the three; it
splits. Its corrupt-math majority becomes a region-lane question (the same shape as the
equation lane PR #518), its table minority is already owned by the P2 floor, and the
render-failure case is plain native prose with an alert. The panel's fear that flooring it
would delete readable prose was right for the majority and moot for the minority.

## Follow-ups

- A ticket for the corrupt-math default: enabling the region lane by default is a cost
  decision (a local model call on 14.5% of born-digital pages) and needs the owner, in
  the same form as the P4 trigger ruling.
- Re-run this measurement after the ladder flips (P1) to get the shipped rate, not the
  upper bound.
