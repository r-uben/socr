# Native cell golden verification

Goldens represent **model markdown** (what OCR would emit). They must **not** be copied from `page.get_text()` / `markdown_from_native_words()`. Each table was checked against a **rasterized page image** (PyMuPDF `page.get_pixmap(matrix=fitz.Matrix(2,2))`, saved under `/opt/cursor/artifacts/golden-raster/` during verification).

## Method (every fixture)

1. Render the fixture PDF page to PNG at 2× zoom.
2. Transcribe the table **from the image** (labels, years, numeric cells, minus signs, `%`, spaced thousands).
3. Optionally note disagreements with the PDF text layer (born-digital extraction); goldens follow the **image**.
4. Run `tools/native_cell_golden_eval.py`; correct rows must not be `contradicted`; no mutant (including `mut_model_*`) may `verify`.

## Image vs text layer (disagreements recorded)

| Page | Image vs text layer |
| --- | --- |
| RESF p63 T7 | Section bands **Administrations publiques centrales/locales/sécurité sociale** are visible section headers in the image; included as label-only rows (empty data cells). Text stream omits them as separate rows. |
| PLFSS p65 full | Image shows **70 / 71** on row *Taxe annuelle sur les inscriptions de médicaments*, not on the grey section banner; montant cells use spaced thousands (**1 285**, **6 354**). Text layer merges multi-line CMU/CNAM prose in year columns — golden keeps montant columns only for lane binding. |
| PLFSS p44 | Image confirms € decimals and minus placement; golden predates this pass and was re-spot-checked on raster (dense merged label cells remain ambiguous). |

## Per-page source

| Fixture | PDF | Verification |
| --- | --- | --- |
| `p59_tableau1.md`, `p59_tableau2.md` | `resf_p59.pdf` | Image read (RESF p.59 annex tables); independent of text export |
| `p67_tableau8.md` | `resf_p67.pdf` | Image + prior hand pass (round 7); complex wrap rows |
| `p38_tableau18.md` | `plfss_p38.pdf` | Image read Tableau 18 (2014–2023 grid) |
| `p44_native_draft.md` | `plfss_p44.pdf` | Image spot-check on € cells; row labels simplified vs PDF merge |
| `p65_pharma_slice.md`, `p65_full.md` | `plfss_p65.pdf` | Image read (slice + full montant columns) |
| `p60_tableau2bis.md`, `p60_tableau3.md` | `resf_p60.pdf` | Image read Tableaux 2bis and 3 |
| `p63_tableau7.md` | `resf_p63.pdf` | Image read Tableau 7; section headers added from image |

Mechanical mutants: `tools/native_cell_golden_eval.py`. Realistic OCR mutants: `mut_model_digit`, `mut_model_column_shift`, `mut_model_minus_drop`.

## Batch 2 (cursor/native-cell-goldens-batch2)

Raster PNGs under `/opt/cursor/artifacts/batch2-raster/` at verification time. Sources are **PLFSS 2025 Annexe 3** and **RESF PLF 2025** single-page fixture PDFs (not VMM / évaluations préalables — see below).

| Fixture | Image check notes |
| --- | --- |
| `p31_tableau15.md` | Tableau 15 branch split; image uses spaced thousands, golden uses digits without interior spaces (text layer uses spaces — documented disagreement). |
| `p42_prestations_slice.md` | Tableau situation État/SS; € columns and programme codes from image. |
| `p43_exonerations_slice.md` | Same layout family as p44 (€ exonerations); Apprentissage / aide domicile rows from image. |
| `p68_csg_activite_montants.md` | p65-like organ + montant + year band; montant row only (year cells hold multi-line pt splits in PDF). |
| `p67_annexe_csg_placement.md` | CSG/CRDS placement rows; montant columns from image. |
| `p61_tableau5_slice.md` | RESF p.61 Tableau 5; dash cells for blank years; minus signs from image. |

## Census documents not in repo (wanted, not invented)

No source PDF for these in `tests/fixtures/` or the workspace uploads. To extend batch 3 from census-heavy corpora, add:

- **VMM tome II** — table pages with multi-year numeric grids and merged row labels (exact page numbers from the Shepherd census manifest once available).
- **Évaluations préalables** — draft budget table pages with organism/amount + year bands and dash-filled model cells.

Until those PDFs are checked in, batch 2 uses RESF/PLFSS annexe pages that match the same layout failure modes (p65/p67/p44 families).
