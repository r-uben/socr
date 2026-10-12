# GH-1099: born-digital source evidence (witness replay)

## Round 3 (Astra P0s)

**P0-1** Text layer evidence uses `get_text("words")` with words dropped when ≥
50% of their bbox overlaps an invisible span from `get_text("dict")` (`alpha==0`,
#961). Not `get_texttrace()`: ~71 witness calls in one process tripped a PyMuPDF
refcount crash in later `find_tables`.
`page_has_page_covering_raster` treats `_raster_coverage` exceptions as scan (fail
closed).

**P0-2** `_merge_source_bundles` max-merges numeric counters per token; crop/crop
merges inside `build_scanned_evidence` stay additive.

**P0-3** Removed `recover_control_byte_minus_for_evidence`; control-byte-before-digit
tokens are ambiguous readings (#1085-shaped), sign from pixel OCR when present.

**Measure** (181 witness pages, subprocess per page): born-digital **71/139** pass
(vs r2 **89**, r1 **87**); scans **2/40** unchanged in intent; **0**
`model_wrong` pages newly pass; Bochkay p15 still fails (OCR lacks minus on
negatives). Result: `scratch/routing-baseline/exec/1099/result_r3.json`.

**Tests** `tests/test_gh1099_born_digital_evidence.py` (10). Suite: 7250 passed.

## Round 4 (Astra: alpha-visible hidden text)

**Fix:** Text-layer evidence from ``get_text("rawdict")`` chars: drop span
``alpha==0``; drop chars with no non-white RGB in a per-glyph ``DETECT_DPI``
colour clip (rotation-aware matrix); regroup chars into words per line. Catches
white-on-white and text under a blank/white image; ``mixed`` → ``99`` only;
``hiddenminus`` → ``5`` only.

**Residual:** Visible text under a **non-blank** photograph is still read from
the layer if alpha is 255 and the glyph pixels differ from white — acceptable
for citation tables (numbers are printed on paper, not under photos); a forged
layer under a chart photo is a separate, rarer shape than baked-in OCR.

**Measure** (181 witness, subprocess): born-digital **61/139** pass (r3 **71**);
**0** `model_wrong` newly pass; 181-page in-process loop + hermetic multi-page
test green.

**Tests** `tests/test_gh1099_born_digital_evidence.py` (20); gh304 prerotate site
table +1 (`_text_layer_ink_matrix`).
