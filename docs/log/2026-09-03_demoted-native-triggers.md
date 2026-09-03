# GH-548 — what DEMOTED_NATIVE actually claims, per trigger

**Date:** 2026-09-03
**Script:** `docs/log/2026-09-03_demoted-native-triggers.py`
**Corpus:** 277 PDFs, 15250 pages (14356 born-digital, 894 scanned). 0 unreadable files.

`PageEnding.DEMOTED_NATIVE` is a panel-approved temporary fourth ending, and its exit
criterion is per-trigger: enumerate the corpus by trigger, hand-check fidelity per trigger,
then assign each independently to `NATIVE_PROSE` or `FAIL_CLOSED_MARKER`. This is the
enumeration. **No trigger is assigned here**, and two findings below say why the assignment
could not have been made from the ticket's own list.

## The count

**2975 born-digital pages — 20.72% — fire at least one trigger.** A page can fire more
than one.

| trigger | pages | % of born-digital | papers |
| --- | ---: | ---: | ---: |
| `needs_ocr_enhancement` | 2205 | 15.36% | 158 |
| `text_grid_rejected` | 479 | 3.34% | 138 |
| native table defect (**lower bound**) | 600 | 4.18% | 167 |
| `chart_asset_render_failed` | — | not measurable | — |

Two of those cells need reading carefully.

**The defect column is a lower bound, not a count.** One member of that union —
`native_table_structure_failed` — is an orchestrator PageState flag set during the
pipeline's own native ship, so no corpus sweep can see it. The other three
(`native_table_unverifiable` via the TR-3 hard fail and the GH-371 ordinals,
`native_table_structure_defective`, `native_table_header_unattributed`) are all on
`PageAssessment` and all counted here.

**`chart_asset_render_failed` is not measurable this way at all.** It is a PNG render/save
failure; nothing about a PDF predicts it. Reported as "not measurable" rather than as zero,
because a structural zero and a measured zero are not the same claim.

## Finding 1 — there are six triggers, not four

`needs_ocr_enhancement` is not one condition. The detector sets it from three unrelated
causes, and they do not get the same answer:

| sub-cause | pages | % of the trigger | papers |
| --- | ---: | ---: | ---: |
| `has_corrupt_math` | 2030 | 92.06% | 141 |
| `native_table_lane_refused` | 178 | 8.07% | 43 |
| `native_rotated_text_shredded` | 1 | 0.05% | 1 |
| unattributed | 0 | — | — |

They are three different questions:

- **`has_corrupt_math`** — font-map mojibake in the math spans. The prose around it may be
  perfectly sound; the math is not. This is 92% of the trigger and 14% of the whole
  corpus, so whatever answer it gets is effectively the answer for `DEMOTED_NATIVE`.
- **`native_table_lane_refused`** — a rotated page where table reconstruction was refused
  and the prose was **deliberately retained**. The prose is intact by construction; the
  tables are absent from the output. `NATIVE_PROSE` would call that page clean.
- **`native_rotated_text_shredded`** — the extracted lines are pieces of one text run. One
  page in the corpus. Nothing about it is trustworthy prose.

So the trigger cannot be assigned an ending as a unit. Splitting it is a prerequisite for
the ticket, not an optimisation of it.

## Finding 2 — `text_grid_rejected` is not a fidelity signal, and reassigning it would undo GH-195

`PageState.text_grid_rejected` says it outright: *"The word-geometry rebuild is lossless,
so the page keeps its text — but it is demoted to WARNING and the document to
AUDIT_FAILED, because the issue requires the rejection to surface at page and document
status, not only in a log."* `_select_page_output_tagged` agrees in its own comment: *"the text
is the lossless word-geometry rebuild and is unchanged"*.

So this trigger demotes for **visibility**, not because anything is wrong with the text.
On fidelity grounds it is the clearest `NATIVE_PROSE` candidate of the six.

That is exactly why moving it needs care rather than a patch: GH-195 required the rejection
to reach page and document status, and today the demotion is *how* it reaches them. A
migration to `NATIVE_PROSE` that does not first give the rejection another durable
surface — a disposition reason, an audit kind — silently repeals GH-195 for 479 pages
across 138 papers. The ticket's difference-pin ("a silent upgrade that makes suspect pages
resume-skippable without a pin must red") catches the resume half of that; it does not
catch the surfacing half.

### A contradiction fixed on the way

`PageAssessment.text_grid_rejections` ended with *"the page is not demoted on it"*. That is
false, and it is the docstring a reader of the detector meets first. PageState's docstring
and the selector both said the opposite. Corrected in this change.

## What is still needed before any assignment

- **A fidelity measurement for `has_corrupt_math`**, which decides 92% of the trigger. The
  useful next number is the distribution of corrupt-glyph counts per page: a page with one
  mangled subscript and a page with two hundred are not the same decision, and the
  detector already counts them -- in `_count_math_corruption`, which returns the
  number; `_detect_corrupt_math` only thresholds it to a bool and discards the count.
- **A surfacing plan for `text_grid_rejected`** before it can move, per finding 2.
- **`chart_asset_render_failed` needs a different instrument** — an induced-failure run,
  not a corpus sweep.

`native_rotated_text_shredded` (1 page) and `native_table_lane_refused` (178) are small
enough to hand-check directly, and their fidelity story is already legible from the code
paths that set them.

## Reproduce

```
PYTHONPATH=src ~/venvs/socr/bin/python docs/log/2026-09-03_demoted-native-triggers.py \
    <papers dir> [--limit N]
```

Content-free: counts and paper counts only, never page text.
