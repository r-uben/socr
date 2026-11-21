# #1399: scanned tables verified against the stored layer in both directions

Stage 1 of the design panel (Fable checks A-D, Neon's two pre-gates). Cell alignment and
arithmetic on totals are stage 2. Audit data: `/Volumes/Main/Library/socr-work/audit_scan_tables/`.

## What changed

* `tables/layer_witness.py` (new): the layer witness (`table_layer_witness`), checks A (forward
  multiset against the layer tokens inside the located crop, signs and points verbatim), B (mark
  conflict), C (format outlier, witness-free), D (reverse, whole-page reading), the two pre-gates.
* `tables/source_evidence.py`: `verify_scanned_table(..., layer_witness=False)`; when set on a page
  whose layer is excluded (`layer_is_evidence=False`) the verdict goes through `_apply_layer_witness`.
  New result fields `unproven`, `unproven_values`.
* `pipeline/agentic.py`: the judge passes `layer_witness=True` and records `unproven` on the
  output (`table_layer_unproven`, event `source_evidence_table_layer_unproven`).
* `core/result.py`, `core/manifest.py`, `core/verdict.py`: the field (serialised, omitted when
  empty), the existing post-selection guard demotes SUCCESS to WARNING, and a
  `source_evidence=UNKNOWN` clause is the visible mark.

## Decision rules

1. Pre-gates (cut-off reading: existing `structure_check.table_truncated`; chart page: existing
   `has_chart_marks` or `has_figure_caption`): not accepted, `verifiable=False`.
2. B or C non-empty: not accepted, `disputed_values` = B and C tokens. Tesseract never rescues.
3. A passes, D empty, a crop located: SUCCESS only if the existing image check also passed
   (`LAYER_CONFIRMS_ALONE = False`), else WARNING with the table.
4. A passes, D non-empty or no crop: WARNING with the table.
5. A fails: the existing verdict stands; if the image check had passed, WARNING (never SUCCESS).
6. Layer with no numbers, or the image check never read the page (no backend, crash): unchanged.
7. Layer absent or `layer_witness=False`: byte-identical to before.

B consumes with exact layer tokens first; a leftover reading token is disputed when the layer
(crop) holds the same digits with other marks, a dash-led layer word holds them, or any
tesseract rung (150 plus the extra rungs the existing check read) holds them with other marks.

C: header rows (row 0 and rows after it with an empty stub cell, the existing header-region rule)
are skipped; a column whose numeric cells are all integers (label, year, line number) is excluded;
in the rest, an integer cell is an outlier when the other numeric cells of its row share one
decimal count of at least one. No threshold: everything is derived from the table.

## Constants

None numeric. `layer_witness.LAYER_CONFIRMS_ALONE = False`: an open owner ruling, a policy switch
(origin: this audit, see below). Fixed reason strings `REASON_TRUNCATED`, `REASON_CHART`.

## The open ruling (needs the owner before merge)

Under `LAYER_CONFIRMS_ALONE = True` (the 2026-11-18 ruling read widely) 11 of the 58 pages ship
SUCCESS, among them a14, whose numbers are right and whose column headers are misbound. The layer
multiset has no positions and strict row binding does not run on a suspect layer
(`row_binding_check.layer_can_judge_tables` requires an admissible layer), so nothing sees it.
The mutation test also shows the switch is unsafe: a layer that shares the model's error yields
SUCCESS (below). Default is `False`; `True` needs the hand-read of the list in the replay section.

## Replay of the 30 audited pages (saved qwen readings, real tesseract, real gate)

Old = today's gate on the same reading. Default = `LAYER_CONFIRMS_ALONE False`; Alone = `True`.
Downstream checks (page judge, structure) are not part of this replay: they can only demote.

| page | ground truth | old | default | alone |
|---|---|---|---|---|
| a00 a01 a04 a10 a11 a15 a16 | not deserved | ERROR | WARNING | SUCCESS |
| a02 a03 | not deserved | ERROR | WARNING | WARNING |
| a13 a22 a23 a24 | not deserved (layer itself wrong) | ERROR | ERROR | ERROR |
| a05 | misbound | ERROR | WARNING | WARNING |
| a06 | misbound | ERROR | ERROR | ERROR |
| a14 | misbound | ERROR | WARNING | SUCCESS |
| a07 a09 a12 a17 a18 a20 | deserved | ERROR | ERROR | ERROR |
| a08 | deserved (dropped row) | ERROR | WARNING (no crop located) | WARNING |
| a21 a25 a26 | chart | ERROR | ERROR | ERROR |
| a19 a27 a28 a29 | undecided (no reading, truncated) | no table | no table | no table |

The 13 not-deserved pages: default 0 SUCCESS, 9 WARNING, 4 ERROR; alone 7 SUCCESS, 2 WARNING,
4 ERROR. The charts stay ERROR through the existing content and numeric checks (a21, a25 by the
chart detector as well; a26 has no chart marks and no caption in its layer, so the pre-gate does
not fire on it, A and the image check reject it). a08's dropped row ships WARNING with the table
because no crop was located; that is the specified outcome for "no crop found".

## Mutation test (13 correct readings, cell by cell)

Mutants: reading only, and the identical edit also applied to the layer witness. Counts are
mutants, over 13 pages. Default switch:

| class | reading only | reading and layer |
|---|---|---|
| drop decimal point | ERROR 1091 | ERROR 1091 |
| drop sign | ERROR 195 | WARNING 101, ERROR 93, SUCCESS 1 |
| last digit +1 | ERROR 1224, WARNING 2 | WARNING 830, ERROR 396 |
| last digit -1 | ERROR 1225, WARNING 1 | WARNING 830, ERROR 396 |
| confusable 3/8 5/6 1/7 | ERROR 1034 | WARNING 713, ERROR 321 |
| delete row | WARNING 126, ERROR 61 | WARNING 125, ERROR 61, SUCCESS 1 |
| delete column | WARNING 61, ERROR 24 | WARNING 60, ERROR 24, SUCCESS 1 |
| swap two cells | WARNING 619, ERROR 306 | same |

The 3 SUCCESS are all a04, where the image check had lost a sign on `-3`, so the mutated reading,
the mutated layer and tesseract agree. Swapped cells never reach SUCCESS on these pages only
because their baseline is WARNING; strict row binding does not run on them (above).

With `LAYER_CONFIRMS_ALONE = True` reading-only mutants stay at zero SUCCESS except swaps (486),
but reading-and-layer mutants reach SUCCESS in large numbers (digit +1: 640, drop sign: 102,
delete row: 89). A layer that shares the error is the one thing no layer-only rule can see.

## Hand-read list (SUCCESS under `True`; the default ships SUCCESS only for p24, which passes today)

Renders (150 dpi): `/Volumes/Main/Library/socr-work/audit_scan_tables/img_pop/{a00,a01,a04,a10,a11,a14,a15,a16,p08,p19,p24}.png`.
Population readings of the 28 unsampled pages were taken from the cluster endpoint with
`qwen-ocr` (`p00..p27`; 7 produced no table reading). Ground truth is known for a-pages only.

## Existing test that fails (reported, not edited)

`tests/test_1211_r2_layer_decides_nothing.py::test_the_layer_does_not_convict_a_pixel_confirmed_reading`
pins that the layer's own reading is refused by the pixel lane. With the layer as the table witness
that reading now passes the source-evidence gate marked (the inner judge refuses it for another
reason). This is the behaviour change itself. The full suite otherwise passes (11,341 passed).

## Not done / doubts

* Strict row binding does not run on suspect-layer pages, so label misbinding is unseen there.
* A shared error between reading, layer and tesseract is invisible (3 mutants above).
* a26-type chart pages with no marks and no caption are not caught by the pre-gate.
* A detector that raises in the chart pre-gate is read as "not a chart".
* The disputed mark text names "a second reading (Tesseract)"; for B or C from the layer or the
  format check that attribution is imprecise.
* The `source_evidence_table_layer_unproven` event is not in `tables_trust`, the credential
  blocking sets or the resume replay list (`tests/test_resume_restore_kinds.py` pins that union
  exactly). The serialised `table_layer_unproven` field is what survives a resume.
