# socr output reference

This page explains what socr writes and how to tell whether a page can be trusted.
Every table names the file that defines it. If the code and this page disagree, the
code is right; fix this page.

## 1. What is on disk

One run writes one folder per input document: `<output>/<stem>/`. If the input sits
in a subfolder, the folder structure is mirrored (`<output>/<sub/dir>/<stem>/`). A
non-PDF input uses `<stem>_<ext>` so it cannot collide with a PDF of the same name
(`doc_dir_for` in the `ocr_output_contract` package).

```
<output>/
├── metadata.json              # root index: one entry per document (status, checksum, fingerprint)
└── <stem>/
    ├── <stem>.md              # final text, stitched from pages/
    ├── metadata.json          # this document: status, model, pages, error note
    ├── pages/
    │   ├── 00001.md           # body text of page 1, no "## Page" header
    │   ├── 00001.json         # page sidecar: status, failure_mode, provenance, resume ledger
    │   └── ...
    ├── manifest.json          # per-page fingerprints and blob pointers, used by `socr replay`
    ├── cache/                 # content-addressed blobs the manifest points to
    │   └── ab/abcdef....json
    ├── audit_log.json         # every notable event of the run
    ├── tables_trust.json      # pages whose tables should not be trusted
    ├── figures/               # PNGs the text links to
    └── equations/             # equation crop PNGs
```

| Path | Purpose | Written by |
| --- | --- | --- |
| `<stem>.md` | The document text. Byte-identical to the stitched `pages/` fragments. | `_phase_assemble` |
| `metadata.json` (document) | Document status (`completed` / `partial` / `failed`), model, page count, run fingerprint, an `error` note that names the document-level debts, and a `tables` block of readable-table counts (section 2). | `_write_metadata` |
| `metadata.json` (root) | Index over all documents in the run. The resume gate reads it. | `RootIndex.record` |
| `pages/NNNNN.md` | One page body, five-digit page number. Written as soon as the page finishes. | `_flush_page_fragment` |
| `pages/NNNNN.json` | Page sidecar. Section 2 says which fields to read. | `_flush_page_sidecar` |
| `manifest.json` | Frozen record of which output won each page, with fingerprint, journal of attempts, disposition and asset hashes. Written when the run is agentic (the default) or `write_manifest` is set, and only if the document produced text. | `_write_manifest`, `Manifest` in `core/manifest.py` |
| `cache/` | Blob store: the winning `PageOutput` of each page, named by the SHA-256 of its content, sharded by the first two hex characters. Replay reads these and never calls an engine. | `BlobStore` in `core/cache.py` |
| `audit_log.json` | `{pdf_filename, event_count, counts, events[]}`. Each event is `{page_num, kind, engine, detail, data}`. Written only when there is at least one event. Page 0 means a document-level event. | `core/audit_log.py` |
| `tables_trust.json` | Small summary derived from the audit events: which pages carry a table flag and why. Absent means no table is flagged. A clean re-run deletes a stale file. | `core/tables_trust.py` |
| `figures/` | Images the text links to: `figure_N_pageP.png` (embedded figures), `chart_page_N.png` (chart pages shipped as an image), `chart_region_pP_I.png` (chart crops on mixed pages), `failed_table_pN.png` and `shredded_rotated_page_pN.png` (full-page renders that replace content socr would not ship). | `figures/extractor.py`, `figures/chart_regions.py`, orchestrator |
| `equations/` | Equation crop PNGs: `equation_N_pageP.png` (equation detection) and `corrupt_math_pNNNNN_rNNN.png` (corrupt-equation recovery). A crop stays on disk even when its LaTeX is refused. | `math/detect_equations.py`, `math/recover.py` |

README's earlier tree omitted `manifest.json`, `tables_trust.json`, `cache/` and
`equations/`. They are real outputs.

## 2. Can I trust this page?

Open `pages/NNNNN.json` and read three fields, in this order.

1. `failure_mode`. Anything other than `"none"` means socr has a stated reason to
   doubt the page. Look it up in section 4.
2. `status`. `"success"` or `"warning"` or `"error"`. See section 3. `"missing"`
   means no output exists for the page.
3. `audit_passed`. `false` on a `success` page means the audit rejected the page
   text but nothing better replaced it.

A page is clean only when `status` is `success`, `failure_mode` is `none` and
`audit_passed` is `true`. Those three fields are not the whole answer:

- A page can pass all three and still be listed in `tables_trust.json`, which
  exists because table flags in the audit log were invisible to readers of the
  Markdown.
- Some non-table warnings leave all three fields clean. Unrecovered symbol glyphs
  (audit event `native_unrecovered_symbol_glyphs`, raised in `_agentic_native_page`;
  no code demotes the page for it) and math-font damage (event
  `native_math_font_unrecovered` plus an audit note; `_apply_math_font_unrecovered_guard`
  in `core/manifest.py` never touches `status`) are reported only that way. Also
  read the page's `audit_events` and `audit_notes`.

Then read the details:

| Question | Where to look |
| --- | --- |
| What shipped on this page, and why? | `disposition.ending` and `disposition.primary_reason` in the sidecar (also in `manifest.json`). Endings: `native_prose`, `model_output`, `fail_closed_marker`, `demoted_native`. Defined by `PageEnding` and `PagePrimaryReason` in `core/manifest.py`. |
| What happened during the run? | `audit_events` in the sidecar (this page's events from the run state) or `audit_log.json` (whole document). The log also adds events derived at write time: `escalation`, `recitation_escalation` and `structure_floor_overrode_ladder`, which the sidecar does not carry. Kinds are in section 5. |
| Which tables are in doubt? | `tables_trust.json`: `untrusted_pages`, and per page `reasons`, `details`, `patch_eligible`. `resolved_by_escalation` lists pages whose table was replaced by a measured better one. |
| Which engine and what cost? | `engine`, `provider`, `cost_usd` (the winner), `page_cost_usd` (everything the page spent). |
| Which socr produced it? | `socr_version`, `socr_source_digest`, `run_fingerprint`, `input_checksum`. |
| Was it a final result? | `terminal`. `false` is a mid-run crash-recovery copy. |
| What does the document say overall? | `metadata.json`: `status` and `error`. |
| How many tables can a reader use? | `metadata.json` `tables`: `shipped_text`, `verified_text`, `unverified_text`, `withheld` (definitions below). The CLI prints `tables: N as text (V verified, U unverified), W withheld` once per document; `socr library` copies the block into each `manifest.json` entry and prints the corpus total. |

Other sidecar fields (`native_table_*`, `chart_*`, `d3_floor_png_ref`,
`table_ladder_disposition`, `figure_refs`, `winning_output`) are the page decision
flags the resume gate restores. They are listed in `_flush_page_sidecar`.

### Readable tables (`tables` in `metadata.json`)

Counts derived from what is already recorded: the shipped text of each page, its
status and failure mode, and the table-distrust index. Nothing is detected anew.
The unit is the markdown pipe-table block (`find_table_blocks`), so a table the
producer fragmented counts once per fragment. Code: `core/table_counts.py`.

| Key | Meaning |
| --- | --- |
| `shipped_text` | Table blocks in the shipped page text, whatever the page status: verified pages, WARNING pages whose text was kept, and ERROR pages that still carry a table block (a regional splice keeps tables beside a withheld one). |
| `verified_text` | The part of `shipped_text` on a `success` page that carries no entry in `tables_trust.json`. |
| `unverified_text` | The part of `shipped_text` on a page whose failure mode is `table_unverified` or that carries a live `table_ladder_unverified` flag. |
| `withheld` | Table regions shipped only as a marker (usually with a page image): one per `[page N failed: unverifiable table ...]` or `[page N failed: invalid table emission ...]` marker. On a prose-recovery page (`[page N: unverified scan ...]`) the markers are per withheld run, not per table, so the page counts as one. |

Not counted: tables flattened to prose (socr does not record them yet, #994) and
chart pages routed to an image asset. `shipped_text - verified_text` includes
rejected and flagged text, not only unverified text. The block is absent when it
could not be derived: absent means not recorded, never zero. Output written before
this block existed has none; `socr library` derives the same counts from the page
sidecars and `tables_trust.json` for such a document.
Derivation from sidecars never returns a confident number from incomplete evidence: an
unreadable page sidecar makes the whole document's counts `null` in `manifest.json`, and an
unreadable `tables_trust.json` makes `verified_text` and `unverified_text` `null`. The
library total sums only documents whose four counts are all known and reports the rest as
`tables_unknown_documents`.

A document can be `partial` or `audit_failed` while most pages are clean. Use the
page sidecars to find which pages carry the debt.

## 3. Statuses

### Page status

Defined by `PageStatus` in `core/result.py`. The sidecar `status` mirrors the
winning output's status.

| Value | What it means for a reader |
| --- | --- |
| `success` | The page produced text. Not proof it is right: check `failure_mode` and `audit_passed`. |
| `warning` | The page shipped with a flag. The text is kept, but it is not clean. Read `failure_mode`. |
| `error` | The page failed. It usually ships a failure marker, but surviving native prose can ship beside the marker (`core/manifest.py`, no-output ending). Read the text, not just the status. |
| `pending` | Default value of a page output that has no outcome yet. |
| `skipped` | Defined but not set anywhere in `src/socr` at the time of writing. |
| `missing` | Sidecar-only value: no winning output exists for the page. |

### Document status

Defined by `DocumentStatus` in `core/result.py`. The orchestrator decides it in
`_phase_assemble`.

| Value | What it means for a reader | `metadata.json` status |
| --- | --- | --- |
| `success` | Text exists and no page or document bucket carries a debt. | `completed` |
| `audit_failed` | Text exists but at least one page or document-level debt remains (disputed table value, unverified label, rejected or unverified table, withheld table, text grid rejected, lost or unplaced chart region, orphaned equation crop, fabricated image link, and others). "Completed with warnings, output written." | `partial` |
| `error` | No usable text. | `failed` |
| `skipped` | The resume gate found the document already processed and did nothing. | n/a |
| `pending` | Initial value; not a final state. | n/a |

The mapping to `metadata.json` is in `_write_metadata`. The `error` string in
`metadata.json` carries the document-level notes, so a reader can see the debt
without opening `audit_log.json`.

## 4. Failure modes

Defined by `FailureMode` in `src/socr/core/result.py` (34 members). The sidecar
carries the value as `failure_mode`. "Ships" below describes what the reader gets
in the Markdown.

### Engine and audit failures

| Value | What happened | What the reader should do |
| --- | --- | --- |
| `none` | No failure recorded. | Nothing from this field. |
| `timeout` | An engine timed out and produced no text. | Re-run the page. If it is the final state, the page has no content. |
| `cli_error` | An engine subprocess failed. | Re-run; read `winning_output.error` in the sidecar. |
| `empty_output` | An engine returned no text. | Treat the page as empty unless another engine's text shipped. |
| `api_error` | A provider API call failed. | Re-run later. |
| `model_unavailable` | The needed model or engine was not reachable. | Start the model or fix the provider, then re-run. |
| `audit_failed` | The text audit rejected the output. | Read `audit_notes` in `winning_output`. |
| `hallucination` | The output had content its evidence did not support: a table whose numbers a witness did not find on the page, or image links with no source. For image links, the links are removed, the cleaned text ships, and the page is marked `error` with this mode. | Do not use the flagged numbers. Check the page against the PDF. |
| `refusal` | The model refused the input. | Another engine should have taken over; if none did, the page is empty. |
| `recitation` | Gemini's recitation filter blocked verbatim output. | Another engine should have taken over. Check `audit_log.json` for a `recitation_escalation`. |
| `garbage` | Output had a high share of non-text characters. | Do not use the text. |
| `low_word_count` | Too few words for the page. | Check whether the page is really sparse. |
| `truncated` | Output stopped early. | The text is incomplete. |
| `unreadable_input` | A page could not be loaded. No engine ran for that page, which ships an `error` marker. If no page loads, the document is recorded failed and the next run retries it. If other pages produced text, the document is `audit_failed` (`partial` in `metadata.json`). | Repair or replace the PDF. |

### Native text and table structure

| Value | What happened | What the reader should do |
| --- | --- | --- |
| `native_table_structure_failed` | The native text layer lost the table's grid. | Treat table numbers as unverified. |
| `table_emission_invalid` | The chosen page text still held table syntax that cannot be valid GFM, or its delimiter row disagreed with the grid. For a malformed-markup defect, final validation replaced the page with a failure marker. For a content defect (such as an empty table), the original text or table is kept and the page is demoted to `error` (`_apply_table_emission_guard` in `core/manifest.py`). | Check the page text. The table may be absent (marker) or present but defective (kept). Use the PDF. |
| `native_text_shredded` | A rotated page whose native text came back as one glyph run per line. The fragments are not a reading of the page. | The page ships a marker and an image of the page. Read the image. |
| `native_minus_as_digit` | A sign or symbol in the native layer is unreliable: a minus sign is encoded as the digit `2` (#913), or an undecoded control character sits directly before a number (#990), or the scan for either could not run. A negative number can read as a different or a positive one. The text is kept and ships `warning`. | Check every signed number on the page against the PDF. Audit kind `minus_extracted_as_digit` (the `2` case) or `control_byte_before_digit` (the control-character case) says how many hits. |
| `native_invisible_text_scan` | The page is a scan whose invisible baked-in OCR text layer (render mode 3 over a page-sized raster) is what ships, or the scan for that failed. The text is kept and ships `warning`. Added in #961. | The text is an old OCR layer that nothing verified. Check the page against the PDF. Audit kind `invisible_text_scan` has `data.error`: true means the scan failed, not that the layer was found. |
| `invisible_scan_unread` | A scan whose native text is an invisible baked-in OCR layer (known unreliable: one letter per line, stray axis ticks) and whose model ladder ran but accepted no reading (#1027). Neither the layer nor the rejected reading ships. The page ships `warning` with a marker, `audit_passed` false. The marker points at an image of the page only when one was written (it needs `--save-figures` and a successful render); otherwise it reads "not transcribed, see PDF page N". The document is never `success`. | Read the image, or the PDF page when no image was written. A re-run re-OCRs the page. Per-rung detail is in the page sidecar `attempts_summary`. |
| `native_garbled_math` | The native text layer garbled the page's mathematics: private-use glyphs, math-alphanumeric codepoints, letters of a script the corpus is not written in (`MISDECODED_MATH_SCRIPTS` in `born_digital.py`: Syriac, Tamil and 21 others), or a span in a math font the region lane does not list (`_MATH_FAMILY_FONT_RE`, matched at the start of the font name after any subset prefix: MathTime (MTMI, MTSY, MTSYN, MTEX, RMTMI and bold forms), MathematicalPi, UniMath, MnSymbol, Universal-GreekwithMath (e.g. Universal-GreekwithMathPi), Libertine/Libertinus ... Math, EuclidMath, XCharterMath, Fourier-Math, MathDesign, "Cambria Math" (with a space), MathTechnical, LucidaMath, AdvMathPack, TeX-math), or the scan for that failed (#960). By default such a page is routed to a whole-page OCR read, and this mode means no read replaced it (no provider, the ladder never ran, or every rung failed). Under `--native-only`, and on the chart-asset lane, the page is not routed: its native text ships with this mode directly. In every case the native text is kept and ships `warning`. Added in #960. | Equations, symbols and sub/superscripts on the page are unreliable. Check them against the PDF. Audit kind `garbled_math_native` lists which signals fired; `data.error` true means the scan failed. |
| `table_not_reconstructed` | A page has a `Table N` caption line and table structure (at least three horizontal rules of one width, recurring numeric columns, or the label-and-value shape), but table detection found no table. Native extraction flattened the grid to prose. The text is kept unchanged and ships `warning`; nothing is re-routed (#994). The document is not `success`. | Treat table numbers as unverified and read the table from the PDF. About one in five fires is a false positive (a figure page or a prose page that starts a line with `Table N`). |
| `figure_words_unread` | A raster figure on a single-column chart page was cropped and placed inline, and no native word lies inside its box, so any words in its pixels are unread (#1053). The page keeps its exact native prose and the crop; it ships `warning` and the document is not `success`. Only the crop is a candidate for a later read, never the page. | Read the words in the crop yourself; the page prose is exact. A photograph with no text also fires. |

### Table judging and fail-closed floors

| Value | What happened | What the reader should do |
| --- | --- | --- |
| `model_output_flagged` | A model produced the page, no ladder rung accepted it, and the native table carries a distrust flag. The model output ships, flagged. | Verify table numbers. |
| `model_table_over_failed_floor` | The native table failed its geometry check and a model attempt authored a grid. The model's grid ships instead of the failed-table marker. A stronger warning than `model_output_flagged`. | Verify the whole table against the PDF. |
| `table_rejected` | The table judge ladder ran out of rungs on a corroborated FAIL. The table text ships, demoted under a warning. | Do not quote its numbers without checking. |
| `table_unverified` | Every ladder rung failed to give a verdict (timeout, transport, missing binary, unparseable). Nothing said the table is wrong, and nothing confirmed it. The bytes are kept. The audit event `table_ladder_unverified` has `cause` and `latched` in `data`. | Re-run when the judge is available; otherwise verify by hand. |
| `table_withheld` | Readers rejected the table and a blind transcription of the flagged cells read different tokens. The table bytes are not shipped. A fail-closed marker and a page image replace the table region. Prose outside the region survives. | Read the table from the image or the PDF. |
| `header_binding_unverified` | The table shipped because its rows match the native rows. Which header belongs to which column was never checked. | Check column headers. |
| `no_witness_backend` | A scanned table failed the source-evidence gate because the host has no classical OCR backend (for example tesseract). Nothing read the pixels, so there is no evidence about the table either way. The page still fails closed. | Install the OCR backend and re-run. Do not read this as a model fabrication. |
| `page_judge_timeout` | The page judge timed out on the page's only grid candidate. Nothing rejected it and nothing confirmed it. The page fails closed. | Re-run when the judge is available. |
| `judge_timeout_ladder_accepted` | The page judge timed out but the table ladder accepted every table, with a stored credential bound to these exact bytes. The page ships demoted. The tables passed; the surrounding prose was not verified. | Tables are supported. Read the prose with normal care. |
| `row_shape_not_reconcilable_text_table` | The only grid candidate is a text table (prose cells, a few numbers). Matching a few numeric rows is not evidence for prose cells. The candidate is withheld. | Read the table from the PDF. A completed page-judge acceptance would clear it. |
| `structure_class_ladder_exhausted` | A table page reached selection with no attempt that authored a usable grid. The native grid is dropped. The page ships a whole-page marker and a page image. Status `error`, `audit_passed` false. | Read the table from the image or the PDF. |
| `structure_class_no_model_attempt` | Deprecated. Only old sidecars carry it. New runs use `structure_class_ladder_exhausted`. | Treat as `structure_class_ladder_exhausted`. |
| `chart_grid_contradicted` | A filled chart grid was compared with counts read from the page's own vector geometry and at least one cell disagreed. Both readings are withheld and the page ships demoted. This does not say which reading is wrong. | Read the chart values from the PDF. |

### Table-gate DEFERs are not failure modes

The native ship gate (`src/socr/tables/ship_gate.py`) checks a native grid that
passed the verifier against the PDF's own words. It only DEFERs and never refuses.
A DEFER sends the page to normal model routing and the judges. The DEFER leaves
two traces and no failure mode of its own:

- the audit event `native_ship_gate_deferred`, with the fault predicate names in
  `data.predicates` and the fault details in `data.faults`;
- a plan reason that starts with `ship_gate`.

The page then ends in one of the modes above, depending on what the model route
produced. Rotated pages that pass the verifier get the sibling event
`rotated_native_table_quarantined`. The list of predicates is in the module
docstring and in the `*_faults` functions of `ship_gate.py`. It is not repeated
here, so it cannot drift.

The newest gate additions (#958) are `header_over_empty_column` and a tightening of
`text_in_numeric_column`, both DEFER-only, so they show up as that event and not
as a new failure mode.

## 5. Audit event kinds

Defined at the emit sites listed per group. Kinds are stable strings in
`audit_log.json` (`events[].kind`, `counts`) and in each sidecar's `audit_events`.
Kinds in `TABLE_DISTRUST_KINDS` (`core/tables_trust.py`) feed `tables_trust.json`.
This list is the kinds present at the time of writing; the code is the authority.

### Routing and run control (`pipeline/orchestrator.py`, `core/audit_log.py`)

- `escalation`: an engine attempt failed and a later one took over.
- `recitation_escalation`: same, for a Gemini recitation block.
- `native_fallback`: OCR did not ship a passing result; flagged native text shipped.
- `native_only_table_distrusted`: `--native-only` and the table region is unverifiable; OCR was never tried.
- `page_failed`: the page shipped a failure marker.
- `page_unloadable`: the page could not be loaded and was not processed.
- `partial_save_vlm_timeout`: the local backend wedged; finished pages were saved and the run stopped.
- `local_rung_excluded_after_rescue`: a local rung timed out, a later rung rescued the page, and the local rung is skipped for the rest.
- `judge_degraded_to_heuristic`: the configured page judge was unusable; the heuristic judge ran.
- `qwen_cloud_pin_unavailable`, `qwen_cloud_pin_refused`: a pinned cloud qwen model was unreachable or refused by policy (page 0).
- `resume_ledger_audit_reject`: a stored SUCCESS page had `audit_passed` false and was reprocessed, not reused.

### Native text layer

- `native_encoding_hygiene_suspect`: cosmetic text-layer damage (fused words); content kept.
- `native_unrecovered_symbol_glyphs`: a symbol font had no ToUnicode map and some glyphs have no verified recovery.
- `minus_extracted_as_digit`: minus signs extracted as the digit 2 (`data.hits` is the count). If `data.error` is true, the scan itself failed and the page is treated as affected; the count is not a finding.
- `control_byte_before_digit`: a control character (C0 other than tab, newline, CR) directly before a digit or `.digit` in the native text, where the PDF prints a minus or another symbol (`data.hits` is the count). The page is routed to OCR; under `--native-only` it is retained and ships `warning`. If `data.error` is true, the scan itself failed and the page is treated as affected.
- `native_minus_as_digit_retained`: the page shipped `native_minus_as_digit`.
- `invisible_text_scan`: the page is a full-page raster carrying invisible text (an old baked-in OCR layer), or the scan failed (`data.error`). The page is routed to OCR unless `--native-only`.
- `native_invisible_text_retained`: no OCR read replaced the invisible layer, so it shipped `native_invisible_text_scan` (`warning`).
- `invisible_scan_unread`: the invisible layer was not shipped because the ladder ran and accepted nothing; the page shipped the marker (plus the page image when one was written) (`invisible_scan_unread`, `warning`). Emitted once per page, instead of `page_failed`.
- `garbled_math_native`: the native text layer garbled the page's mathematics; `data.signals` holds the non-zero counts (`private_use`, `math_alphanumeric`, `misdecoded_script_letters`, `unlisted_math_font_chars`), `data.error` is true if the scan failed. The page is routed to a whole-page OCR read (also off the corrupt-math region lane) unless `--native-only`; on the chart-asset lane under `--native-only` it ships the chart lane's native text with `native_garbled_math`. Recomputed from the PDF every run.
- `native_garbled_math_retained`: no OCR read replaced that text, so it shipped `native_garbled_math` (`warning`).
- `native_math_unrecovered`: math-glyph damage survived into the shipped page.
- `native_math_font_unrecovered`: math-font typesetting that extracts unreliably, not covered by an equation lane.
- `rotated_text_shredded`: rotated page whose native lines are fragments; native text refused.
- `landscape_page_refused`: native table reconstruction refused on a rotated page; prose kept, page routed to OCR.

### Native table checks (`pipeline/agentic.py`, `tables/`)

- `native_table_verifier_exact_pass`, `native_table_exact_pass`: the grid matched the native words.
- `native_table_verifier_warn`, `native_table_verifier_hard_fail`: the verifier found a mismatch.
- `table_region_geometry_hard_fail`: region geometry check failed at analyze time (detection only).
- `table_region_unverifiable`: the same failure acted on: OCR also failed, so a marker and page image shipped.
- `table_structure_failed`: grid-shape or emission defect found.
- `table_header_unverifiable`, `table_header_repair`: header attribution abstained, or a collapsed header was repaired.
- `table_furniture_removed`: a site menu printed on every page, written by the model as a table, was removed from the shipped text (#988).
- `text_grid_rejected`: a lane boundary split a native numeric token; the page is demoted.
- `orphan_word_dropped`: words the rowizer dropped far from every column lane.
- `native_table_cell_repaired`, `native_table_cell_unresolved`: failing cells were re-read and fixed, or the table was not shipped.
- `native_ship_gate_deferred`, `rotated_native_table_quarantined`: see section 4.
- `table_value_drift_unadjudicated`, `value_guard_row_count_warning`: a numeric mismatch was seen but not adjudicated.
- `table_verifier_error`: the verifier raised; the table is rejected fail-closed.
- `table_not_scorable`, `table_unexplained_lanes`: the native table could not be scored, or has lanes with no column.
- `table_row_repetition_truncated`: consecutive duplicate rows were dropped.
- `table_ditto_unresolved`: a ditto mark was kept verbatim, not expanded.
- `possible_table_structure_not_reconstructed`: a borderless label|value shape was seen and not rebuilt. Report only on its own; with a `Table N` caption it also raises `table_not_reconstructed`.
- `table_not_reconstructed`: a caption plus table structure on a page where detection found no table (#994). The page ships `warning` / `table_not_reconstructed`. Recomputed from the PDF every run.
- `table_not_reconstructed_retained`: the document-level mirror, emitted at assemble when the native text of such a page is what shipped.
- `figure_crops`: the page took the per-figure crop route (#1053); `data` carries the figure count, the figures with no native word inside, and the word count per owner (prose, caption, `figure:N`).
- `figure_words_unread_retained`: the document-level mirror of `figure_words_unread`, emitted at assemble.

### Scanned-table evidence (`pipeline/agentic.py`, `tables/source_evidence.py`)

- `source_evidence_table_reject`: the evidence gate failed the table. Either a witness read the page and the model's numbers were not there, or no witness backend existed (then `source_evidence_no_witness_backend` is also emitted). Read `data.cause`.
- `source_evidence_no_witness_backend`: no OCR backend, so no witness.
- `source_evidence_table_label_unverified`: numbers are supported, at least one label is not.

### Re-reads, escalation and judge ladder (`pipeline/orchestrator.py`, `judge/table_verdict.py`)

- `dualpass_patched`, `dualpass_flagged`: a dual-pass reread changed cells, or only flagged a disagreement.
- `dualpass_crop_timeout`, `dualpass_crop_failed`: the crop reread did not complete; the table stays unverified.
- `table_reread_rejudged`: a reread patch was judged; accepted, or refused and the earlier bytes kept.
- `table_escalation_accepted`, `table_escalation_rejected`, `table_escalation_refused`, `table_escalation_timeout`, `table_escalation_withheld`: outcome of the stronger-engine retry for a table.
- `table_escalation_recovered_fail_closed`: an accepted escalation cleared a fail-closed state.
- `table_ladder_accepted`, `table_ladder_rejected`, `table_ladder_unverified`, `table_ladder_withheld`: the four per-table judge ladder terminals.
- `table_binding_adjudicated`, `table_binding_boundary_unresolved`, `table_binding_boundary_resolved`: cell-to-word binding evidence.
- `table_spacer_rows_dropped`, `table_wrapped_label_merged`: layout rows dropped, or a wrapped label merged.
- `page_judge_timeout_credential`, `judge_timeout_ladder_accepted`: the page judge timed out and a ladder credential let the page ship.
- `page_judge_timeout_floor`: the page judge timed out with no credential; the page failed closed.

### What shipped for a table page

- `flagged_model_table_kept`: model table kept, flagged.
- `d3_floor_model_table_kept`: model table shipped over a failed floor.
- `structure_class_model_table_kept`: model table kept for a table page where native may not author the grid.
- `structure_class_row_corroborated`: rows matched native; header binding unchecked.
- `structure_class_ladder_exhausted_floor`: marker and page image shipped; every candidate was refused or absent.
- `row_shape_not_reconcilable_text_table_floor`: text-table candidate withheld.
- `structure_floor_overrode_ladder`: the ladder accepted a table and the floor still shipped the marker.
- `candidate_truncated`: a candidate that ends mid-emission was dropped.

### Charts and figures

- `chart_asset_page`: chart page shipped as an image; data values not transcribed.
- `chart_asset_render_failed`, `chart_asset_detection_failed`: the chart PNG could not be rendered, or chart detection raised.
- `chart_table_arbitration`, `chart_math_arbitration`: a page with both chart and table, or chart and equation, signals.
- `chart_region_preserved`, `chart_region_not_preserved`, `chart_region_inventory_failed`, `chart_region_source_unreadable`, `chart_region_duplicate_suppressed`: chart crops on mixed pages.
- `chart_table_skeleton_suppressed`, `chart_table_skeleton_unbound`: an empty grid derived from a chart was withheld, or could not be proven so.
- `chart_counts_derived`, `chart_counts_not_derived`: counts read from vector geometry and published, or refused.
- `chart_grid_reconciled`, `chart_grid_contradicted`, `chart_grid_not_reconciled`: a filled grid compared with geometry: agreed, disagreed, or not comparable.
- `visual_values_not_transcribed`: a figure's in-image text and values are not in the Markdown.
- `scanned_figure_asset`: a scanned page (invisible OCR layer over a raster) whose layer carries a figure caption line shipped its page image beside the page text (#1030). Per page, not per region: a scan has no vector marks and nothing isolates the figure box. `data.png_saved` false means the image could not be written (the page is `warning`). Sidecar key `scanned_figure_png_ref`. When the text that ships IS the layer, runs of 7 or more one-character lines (an axis title spelled down the page) are fenced IN PLACE as a visible text code block under the note "[unreadable figure text from scan, kept verbatim]" (every line kept; not inside tables, math or lists). `png_saved` is the render; the document note and CLI say whether the finalised page references the image.
- `figure_placeholder_unresolved`, `figure_phase_failed`, `figure_cap_reached`, `figure_recoverable_labels`: figure phase outcomes.
- `fabricated_image_ref`: image links with no source were removed; the document is demoted.

### Equations

- `corrupt_math_region_recovery`, `corrupt_math_hybrid_shipped`: corrupt-equation lane evidence, and the shipped hybrid (WARNING).
- `equation_lane_detection_failed`, `equation_lane_no_region`: no region could be found; native prose ships.
- `equation_region_detected`: a display-equation region was cropped.
- `equation_region_reading_attached`, `_rejected`, `_unaligned`, `_unsafe_markup`, `_unvalidated`, `_unverifiable`: what happened to a region's LaTeX reading (all prefixed `equation_region_reading`).
- `equation_latex_accepted`, `equation_latex_rejected_kept_crop`: the legacy detect/recover path attached LaTeX, or kept only the crop.
- `equation_sidecar_refused`, `equation_sidecar_skipped_no_page_output`: LaTeX refused by a guard, or a crop left without a page output (demotes the document).

## 6. Ship-gate predicates

The full list of ship-gate predicates, with what each one checks, is in the module
docstring of `src/socr/tables/ship_gate.py`. Predicate names are the module
constants there (`SIGN_DETACHED`, `ROW_ORDER`, and so on) and appear in
`native_ship_gate_deferred` events under `data.predicates`.
