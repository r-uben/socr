# Decision Log: GH-304b — De-rotate Table-Crop and Source-Evidence Rasters

**Date:** 2026-08-27  
**Grounded in:** Commit `f2424e8` ("fix(routing): de-rotate landscape pages before OCR and before judging")  
**Scope:** Extending page-level de-rotation (PR #305) to table-crop and source-evidence raster lanes  
**Status:** IMPLEMENTED AND TESTED. `extract.py` and both `source_evidence.py` raster
sites de-rotate; `image_locate.py` deliberately unchanged (see below); 16 tests in
`tests/test_gh304b_crop_derotation.py`; full suite 2,243 passed / 3 xfailed.
(This line said 'pending' after the work had landed — corrected 2026-08-27.)

---

## Summary

De-rotate table-crop and source-evidence raster pixels only. All table bboxes, clip regions, figure anchors, audit bboxes, and emitted `CropTable` bboxes remain in PDF page space unchanged. Only the rendering matrix—the mathematical transform applied to rasterize an already-located clip—rotates to match the page's dominant text direction.

---

## Design Decision: Localization Stays in PDF Page Space

**The constraint:** `TableBox.bbox`, padded/clamped crop clip boundaries, `CropTable.bbox` (emitted output), figure anchors, and audit bboxes are **never transformed**. They stay in their original PDF page coordinates.

**Why:**
- Figure and chart anchors (references to specific regions by page-space bboxes) break if coordinates transform.
- Audit records that cite regions by bbox must stay stable across rendering changes.
- Page-space coordinates are the authoritative reference for every downstream consumer (assembly, indexing, cross-document citations).

**What rotates:**
- Only the **rendering matrix**—the 2×3 affine transform passed to `page.get_pixmap(matrix=...)` when rasterizing a crop or full page.
- The clip region (in page-space PDF points) is **not transformed**.
- The prerotated matrix applies the rotation (90, 180 or 270 degrees) as pixel samples are rendered from the PDF operators.

**Consequence:** A crop in a rotated region is rendered sideways (matching its text direction) but still anchored to its original page-space bbox.

---

## Image Localization Stays Unrotated

**File:** `src/socr/tables/image_locate.py` — **no change**

**Why this design is locked:**

The function `_raster_rules()` renders the page raster at a fixed DPI and performs connected-component analysis to find horizontal rule pixels. The pixel coordinates are then converted back to PDF points:

```python
scale = 72.0 / dpi  # px → PDF points
ymid = (y + ch / 2.0) * scale
rules.append((ymid, x * scale, (x + cw) * scale))
```

If the rendered raster were prerotated:
- Pixel (x, y) from the rotated image would correspond to a *different* region in page space.
- The inverse coordinate mapping would need to account for:
  - The swap of width/height axes under 90/270-degree rotation.
  - The translation of the rotated bounds back into page coordinates.
  - The pixel-to-PDF-point scale applied *before* the rotation.

This inverse mapping is **untested, algebraically risky, and would couple image localization to rotation logic**. Scanned pages (the primary use case for image-based localization) have no text direction evidence anyway—`upright_rotation_for(page)` returns 0 for textless rasters—so rotating the image provides no benefit. Therefore, the image locator stays unrotated.

**Implication:** Raster-based table detection always works in page-space pixel coordinates, matching the vector-based detection path (`find_tables()` on unrotated PDF drawings).

---

## Angle Derivation Rule

All rotation angles are **derived from page content**, never guessed or hard-coded.

**For extract/source_evidence crops:**

1. **Clip-local inspection:** The crop calls `page.get_text('dict', clip=clip)` and inspects the `dir` field of text lines within that clip region only.
   - If the clip contains text lines with a non-horizontal direction, that direction determines the angle.
   - If the clip contains text lines but all are horizontal, angle = 0.

2. **Fallback to page-level:** If the clip contains no text lines (image-only region, blank margin, etc.), fall back to the page's dominant text direction.
   - Derived from `dominant_text_direction()` across the whole page.
   - If the page has no directional text evidence (scanned, blank), return 0.

3. **Exception during clipped inspection:** If `page.get_text('dict', clip=clip)` raises an exception:
   - **Fail open:** Return 0 immediately.
   - **No fallback to page-level:** Do not retry with full-page inspection.
   - Reasoning: Inspection failure is transient/environmental; rotating on its absence is worse than not rotating.

4. **Full-page evidence render (source_evidence.py):** When no crop tokens are available and the fallback renders the whole page for classical OCR:
   - Use the page's dominant text direction.
   - If page inspection fails, return 0 and render unrotated.

**Angle values:**
- `upright_rotation_degrees(direction)` maps a unit-vector direction to 0, 90, **180**, 270, or 360 ≡ 0.
  (GH-426: earlier wording here omitted 180; the measured table below shows `(-1, 0)` -> 180.)
- Snapped to right angles: text at slight skew (e.g., 7 degrees) maps to the nearest quadrant.
- 0 for horizontal `(1.0, 0.0)`.
- 90 for `(0.0, -1.0)`.
- 270 for `(0.0, 1.0)`.
- ~~**Never 180:** Text running upside-down ... returns 0 (absence-of-evidence rule).~~
  **Corrected (GH-311): the code does NOT do this.** Measured against
  `upright_rotation_degrees`:

  | direction vector | returns |
  |---|---|
  | `(1, 0)` | 0 |
  | `(0, -1)` | 90 |
  | `(-1, 0)` | **180** |
  | `(0, 1)` | 270 |
  | `(0, 0)` | 0 |

  Upside-down dominant text yields 180 (bearing 180 → `round(-2) * 90 % 360`), and a
  180° `prerotate` is applied.

  0 is returned for **horizontal text** `(1.0, 0.0)`, near-horizontal skew such as
  `(0.99, 0.05)`, the all-zero vector, and a failed or empty inspection. Returning 180 is
  the signature of upside-down text — it is not one of the 0-returning
  absence-of-evidence cases, which is exactly what the old wording got backwards.

  A maintainer reading the old wording would believe upside-down text is suppressed. It is
  not. This corrects the RECORD to match the code; it does not change the angle rule,
  which would need its own pin (GH-311 is explicit on that point). GH-426: that pin now
  exists -- `test_angle_is_derived_from_direction`.

**Do not claim scanned pages lack text.** The scanned-page predicate is based on *structural* evidence (embedded image coverage, drawn pixel content, absence of proper fonts), not absence of text direction. A scanned page may carry an OCR text layer with directional metadata. Image localization stays unrotated regardless; the angle rule applies only to crop/evidence raster rendering.

---

## PyMuPDF Implementation: Verified Under Current Lock

**PyMuPDF version:** 1.28.2 (locked in `uv.lock`)

**Key behavior under 1.28.2:**

1. **Matrix.prerotate() mutates in place AND returns the matrix:**
   ```python
   mat = fitz.Matrix(dpi / 72, dpi / 72)
   mat.prerotate(90)  # the 304b lanes: mutate-only, the return is discarded
   mat = mat.prerotate(90)  # the page-level lanes: assignment, same effect
   pix = page.get_pixmap(matrix=mat, clip=clip)
   ```
   - Do NOT reuse the matrix after calling `prerotate()` without reinitializing.
   - GH-311: "returns None" was wrong. Verified on PyMuPDF 1.28.2 -- `prerotate` returns a
     `Matrix`.
   - GH-426: the correction then overshot in the other direction, claiming production
     writes `mat = mat.prerotate(...)` and that the assignment is what makes it work.
     Both forms are in the tree, and **this ADR's own 304b lanes are the mutate-only
     form**. Measured on main:

     | form | sites |
     |---|---|
     | mutate-only `mat.prerotate(rotation)` | `tables/extract.py`, `tables/source_evidence.py` (x2), `tables/witness.py`, `pipeline/orchestrator.py` (D3 floor render) |
     | assignment `mat = mat.prerotate(rotation)` | `engines/base.py`, `core/document.py` (x2), `review/html.py`, `pipeline/orchestrator.py` (chart page + chart region) |

     Because `prerotate` mutates AND returns the same matrix, the two forms are
     equivalent here -- which is precisely why neither can be cited as "what production
     does". The mutation is what carries the rotation in the 304b lanes; the assignment
     is redundant where it appears.

2. **Dimension swap on 90/270 rotation:**
   - Unrotated clip `(x0, y0, x1, y1)` with page space width `w = x1 - x0` and height `h = y1 - y0`.
   - Under `prerotate(90)`, the rendered pixmap dimensions are swapped: `pix.width ≈ h * scale` and `pix.height ≈ w * scale`.
   - Exact dimensions depend on subpixel rounding in the PDF render engine; tests use fixture-based byte-comparison, not dimension assertions.

3. **Coordinates stay in page space:**
   - `clip=fitz.Rect(x0, y0, x1, y1)` is **not transformed** by `prerotate()`.
   - The clip bounds are applied in page-space units *before* the rotation.
   - Example: A 170-by-310-point clip (width × height) becomes approximately 310 px wide and 170 px tall at scale 2 (144 DPI) after `prerotate(90)`.

4. **Determinism and reproducibility:**
   - Same matrix configuration + same clip + same PyMuPDF version = byte-identical pixmap.
   - Tests pin this byte-identity under `prerotate(90)` and `prerotate(270)` to verify correct quadrant.

---

## Resume Audit: Fingerprint Accuracy

**`_run_fingerprint()` in `src/socr/pipeline/orchestrator.py`:**

- Accepts **no page number or bbox parameters**.
- Computes a fingerprint over *resolved run configuration* only: engine models, backends, routing flags, etc.
- **No geometry component:** page count, bbox dimensions, rotation angles are not in the fingerprint.

**`_socr_source_digest()` in `src/socr/pipeline/orchestrator.py`:**

- Hashes the **shipped socr Python source** (every `.py` file in the package).
- Already included in the run fingerprint (see GH-214).
- Guarantees that any correctness fix (including rotation logic) invalidates the fingerprint and reprocesses already-terminal pages.

**Consequence for GH-304b:**
- **No change needed** to `_run_fingerprint()` or `_socr_source_digest()`.
- When the crop-rotation code ships, the source digest changes; resume checks invalidate cached output on pages that had already finished.
- No geometry-specific fingerprinting is required because the source change is the invalidation signal.

**Audit gate:** `RootIndex.is_completed()` compares the stored fingerprint against the recomputed one. On mismatch (including source changes), the page reprocesses.

---

## Evasion Matrix: E1–E9 Test Coverage

All tests are **hermetic** (no Ollama, no VLM provider, no GPU required) and live in `tests/test_gh304b_crop_derotation.py`.

### Helper-Level Tests (Task t2)

**E0: Helper Parity**
- `test_legacy_rotation_wrappers_match_shared_helper`: Verify that `base._upright_rotation_for()` and `document._upright_rotation_for()` match the shared `upright_rotation_for()` entry point across all scenarios (rotated, horizontal, blank, uninspectable).

**E0a: Clip-Local vs Page Direction**
- `test_upright_rotation_for_clip_local_vs_page_direction`: Mixed-direction page (dominant horizontal + rotated table block). Clip over table region → 90. Clip over prose → 0. Page-level → 0.

**E0b: Fallback to Page**
- `test_upright_rotation_for_clip_no_lines_falls_back_to_page`: Empty clip (no text lines) on a 90-degree rotated page → falls back to page 90, not 0.

**E0c: Image-Only / No Evidence**
- `test_upright_rotation_for_image_only_or_no_lines_yields_zero`: Blank page + clip. Scanned image page + clip. Both → 0.

**E0d: Exception Fail-Open**
- `test_upright_rotation_for_exploding_get_text_yields_zero`: Uninspectable page raises → returns 0.
- `test_upright_rotation_for_uninspectable_clip_yields_zero`: Page inspects fine at full level (90°), but clipped inspection raises → returns 0, not fallback 90.

### Raster Rendering Tests (Tasks t5–t7)

**E1: Rotation Sign (90 vs 270)**
- **Test:** `test_crop_rotation_sign_90_vs_270`
- **Verifies:** 90-degree rotation renders a distinct quadrant from 270-degree rotation.
- **Fixture:** Asymmetric PDF (F-shaped mark at top-left) so all four rotations (0, 90, 180, 270) are visually distinct.
- **Assertions:**
  - Rendered crop under direction `(0, -1)` matches `prerotate(90)` reference.
  - Rendered crop under direction `(0, 1)` matches `prerotate(270)` reference.
  - Both differ from unrotated baseline.
  - 90 and 270 do not produce the same raster.

**E2: Mixed-Page Clip-Local Direction**
- **Test:** `test_mixed_page_crop_uses_clip_local_direction`
- **Verifies:** Page with horizontal prose + rotated table. Table crop rotates (clip-local 90). Prose crop stays unrotated (clip-local 0).
- **Assertions:**
  - Table crop PNG matches `prerotate(90)` of its clip.
  - Prose crop PNG matches legacy unrotated pipeline byte-for-byte.

**E3: Origin-Clamped Crop**
- **Test:** `test_origin_clamped_crop_rotates_the_same_region`
- **Verifies:** A bbox near (0, 0) with padding that crosses the origin clamps correctly and rotates the exact clamped region.
- **Fixture:** Markers inside and outside the near-origin clip so an accidental clip expansion is detectable.
- **Assertions:**
  - Rendered dimensions reflect 90° rotation (width/height swapped).
  - Byte content matches prerotated reference of the exact clamped clip.
  - Outside marker is absent (clip was not accidentally expanded).

**E4: Horizontal Page Byte-Identity**
- **Test (extract):** `test_horizontal_extract_crop_png_is_byte_identical`
- **Test (source_evidence):** `test_horizontal_evidence_pixmaps_are_byte_identical`
- **Verifies:** Horizontal pages (angle 0) render byte-identical to legacy unrotated pipeline.
- **Assertions:**
  - `_render_crop()` PNG matches legacy unrotated render for extract.
  - `_render_crop_pixmap()` pixmap matches legacy unrotated render for evidence.
  - Full-page evidence fallback also matches legacy.

**E5: Page-Space Round Trip**
- **Test:** `test_located_crop_bbox_round_trips_in_page_space`
- **Verifies:** Bboxes located in page space, rendered with rotation, then emitted stay in page space.
- **Fixture:** Ruled table with rotated text content.
- **Assertions:**
  - Emitted `CropTable.bbox` equals the located `TableBox.bbox`.
  - Bbox stays inside page boundaries.
  - Raster handed to reader matches `prerotate(90)` of the clip region in page space.

**E6: Fail-Open Byte-Identity**
- **Test (extract):** `test_uninspectable_extract_crop_png_is_byte_identical`
- **Test (source_evidence):** `test_uninspectable_evidence_pixmaps_are_byte_identical`
- **Verifies:** When page inspection fails, fall back to angle 0 and produce legacy-identical unrotated rasters.
- **Fixture:** Pages that raise during `get_text('dict')` or `get_text('dict', clip=...)`.
- **Assertions:**
  - Extract crop PNG matches legacy unrotated.
  - Evidence pixmaps match legacy unrotated.

**E7: Source Evidence Full-Page Fallback Rotation**
- **Test:** `test_source_evidence_crop_and_full_page_follow_rotation`
- **Verifies:** Both `_render_crop_pixmap()` and `build_scanned_evidence()` full-page fallback apply the same rotation.
- **Fixture:** Rotated PDF page (no born-digital words).
- **Assertions:**
  - Crop pixmap matches `prerotate(90)` reference.
  - Full-page OCR pixmap (when crop evidence is empty) also matches `prerotate(90)`.
  - Both differ from unrotated baseline.

**E8: Image Locator Stays Unrotated**
- **Test:** `test_image_locator_textless_scan_stays_unrotated`
- **Verifies:** Scanned image pages use unrotated matrix in `locate_tables_image()`.
- **Fixture:** Synthetic scanned page (raster image, no vectors, no text direction).
- **Assertions:**
  - Page rotation angle is 0 (no text direction evidence).
  - Locator uses a scale-only matrix (no rotation).
  - Returned bboxes are in page space.
  - Pixmap consumed by locator matches direct unrotated render.

### Resume Validation (E9)

**Test:** `tests/test_resume_source_version_gh214.py::test_source_digest_is_in_the_run_fingerprint`

**Verifies:** Changes to socr source code (including new rotation logic) invalidate the run fingerprint, forcing re-OCR on already-terminal pages.

**Mechanism:**
- `_socr_source_digest()` hashes all shipped `.py` files.
- `_run_fingerprint()` includes the source digest.
- When crop-rotation code ships, the digest changes → fingerprint changes → resume gate fails → page reprocesses.

**No new test required:** E9 is already covered by the existing resume validation suite; no geometry-specific fingerprinting is needed.

---

## Verification Path

All tests are invoked via:
```bash
uv run --extra dev pytest tests/test_gh304b_crop_derotation.py -v
```

The session's `pyproject.toml` configures `pythonpath = ['src']`, routing `import socr` to the local source tree (not a stale system install).

---

## Known Limitations and Out-of-Scope Items

1. **Judge timeout on reference page (OBSERVED, NOT FIXED):**
   - The reference page (IJCB Table A.2, landscape results table) still logs a 300-second timeout during judging, even though de-rotation fixed the table read itself.
   - Investigation and remediation are separate; the timeout is a symptom of the judge's polling or cancellation logic, not the rotation fix.
   - Recorded as-is; do not include transient branch state or unrelated checkout paths in this decision record.

2. **Raster-localized tables on rotated pages:**
   - Scanned pages (textless) have no rotation angle; `image_locate.py` returns 0.
   - Born-digital rotated pages with ruled/booktabs tables are detected by `find_tables()` on unrotated vector drawings; the crop lane then applies rotation to the raster.
   - No mixed mode (vector detect + raster localize on the same page) is currently tested; such pages would use vector bboxes and leave image localization unused.

3. **Multi-degree rotations and skew:**
   - Angles are snapped to 0/90/**180**/270. Text set at slight skew maps to the nearest quadrant; sub-degree variations are tolerated by models. (GH-426: 180 was omitted here.)
   - No sub-angle rotation is applied; it would resample glyphs with no gain.

---

## References

- **Commit:** `f2424e8` (core page-level de-rotation before OCR and judging)
- **Issue:** GH-304 (de-rotate landscape pages before OCR and judging)
- **PR:** #305 (implemented page-level fix; deferred table-crop lane)
- **Test suite:** `tests/test_gh304b_crop_derotation.py` (E1–E9 evasion matrix)
- **Resume audit:** GH-214 (`_socr_source_digest` invalidates cache on source changes)
- **PyMuPDF:** 1.28.2 (locked; `Matrix.prerotate()` mutates in place **and returns the same matrix**; dimensions swap on 90/270, not on 180)
