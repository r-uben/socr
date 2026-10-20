# Native cell golden verification log

Goldens are **transcribed from born-digital PDF text** (official RESF PLF 2025 and PLFSS 2025 Annexe 3), not from OCR model output.

## Method (each fixture)

1. Open the single-page fixture PDF with PyMuPDF (`page.get_text("text")` and `page.get_text("words")`).
2. Build or correct the markdown table row-by-row against the PDF text layer (labels, organisme codes, numeric cells, year headers).
3. For wide native grids, use `markdown_from_native_words()` only as a **draft**, then fix column headers and row labels against the PDF until numeric tokens align with word bboxes (manual spot-check of at least three numeric cells per table via word x/y positions).
4. Run `~/venvs/socr/bin/python tools/native_cell_golden_eval.py` — correct goldens must be `verified` or `ambiguous`, never `contradicted`; every mutant in the matrix must not `verify`.
5. Record page-specific notes below.

## Page notes

| Fixture | PDF source | Verification |
| --- | --- | --- |
| `p59_tableau1.md`, `p59_tableau2.md` | `resf_p59.pdf` | Hand-checked against RESF p.59 Tableaux 1–2 text layer |
| `p67_tableau8.md` | `resf_p67.pdf` | Hand-checked Tableau 8 |
| `p38_tableau18.md` | `plfss_p38.pdf` | Hand-checked Tableau 18 year grid |
| `p44_native_draft.md` | `plfss_p44.pdf` | Native-word draft, spot-checked € / space-thousands cells |
| `p65_full.md` | `plfss_p65.pdf` | Full Annexe 3 p.65 grid: organ + montants + 11-year band from PDF headers |
| `p60_tableau2bis.md`, `p60_tableau3.md` | `resf_p60.pdf` | Hand-transcribed from RESF p.60 Tableaux 2bis and 3 |
| `p63_tableau7.md` | `resf_p63.pdf` | Hand-transcribed Tableau 7 (PLF vs LPFP columns) |
| `p42_*`, `p61_*`, `p39_*`, `p40_*`, `p41_*`, `p42_plfss_*` | matching `*.pdf` | Native draft + PDF text spot-check on header years and first/last data rows |

Mutations are generated mechanically in `tools/native_cell_golden_eval.py` (`mut_number`, `mut_sign`, `mut_swap_labels`, `mut_drop_row`, `mut_swap_years`, `mut_add_col`, `mut_add_row`).
