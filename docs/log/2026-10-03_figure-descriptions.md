# 2026-10-03 number-free figure descriptions (#1032)

Owner ruling: every figure gets a short model-written description, labelled model-written,
containing NO numbers (kind of chart, axes, series, what is compared). A wrong number is worse
than a missing one, so a description that cannot be validated is dropped, never shipped.

## Phase 1: measurement (counts only; the corpus is copyrighted, samples stay local)

Local model qwen3-vl:30b-a3b-instruct, one call at a time on the shared Ollama. Full numbers in
`~/.local/state/socr-housekeeping/figure-desc/MEASURE.md`.

Sample 1, 20 images of mixed kind (crops, chart-page assets, scanned-page images):
- existing `--describe-figures` prompt: a digit in 20/20 outputs;
- number-free prompt: first answer passes 9/20, one retry fixes 3/11, shippable 12/20;
- faithfulness (every image inspected): 3 of the 12 shippable were unfaithful. All three were
  WHOLE TEXT PAGES (a journal first page, a page of notes and references, a text page that
  mentions a table) that the model described as a chart or a table. Real figures: 0 unfaithful.
  This exceeded the stop limit (2) and phase 2 was held.

Ruling: describe ONLY genuine crops, decided by asset kind (filename the pipeline gave the asset),
never by pixel size; keep the validator strict (a dropped description is a missing description).

Sample 2, crops only (20 runs: 8 re-run + 12 new; charts, event-study plots, tables-as-image, a
word cloud, a diagram, code, a flag, a text snippet):
- first answer passes 9/20 (45%); one retry fixes 2/11 (18%); shippable 11/20 (55%), 9/20 dropped;
- shipped and unfaithful: 0/11 (one borderline: an illustration of a flag called a photograph);
- one rejected candidate was unfaithful (a series name misread), caught only because it also
  carried a digit: the validator is not a faithfulness check;
- latency, model warm: median 2.7 s first answer, 2.1 s retry. Old prompt: median 13.9 s.
- Dominant drop reason: spelled counts ("two panels", "four series") and numerals in labels.

## What changed

- `src/socr/figures/crop_descriptions.py`: prompt, validator (every Unicode numeric character plus
  spelled numbers two..ninety, hundred..trillion, dozen, zero, double, triple, half, quarter, percent;
  "one" allowed), one-retry policy, crop allow-list (`chart_region_pP_N.png`, `figure_N_pageP.png`),
  idempotent insertion under the image ref (skips code fences and table rows).
- Format, directly under the ref: `> *Figure description [model-generated, non-authoritative gist, no
  values]:* text` (the existing caption marker wording plus "no values").
- `UnifiedPipeline._describe_crop_refs`, called in `process()` after the figure phase and before the
  final-body guard, fragment rewrite and sidecar flush, so the final `.md`, `pages/NNN.md`,
  sidecars and manifest all carry the same bytes. Per page body; a document with no crop ref is
  returned unchanged.
- Config `describe_figure_crops` (default True), CLI `--figure-descriptions/--no-figure-descriptions`.
  Off under `--native-only`. Local Ollama model only (never a cloud rung), so `--strict-local` and a
  zero cost cap need no extra branch. Model unreachable: dropped, reason `model_unavailable`.
- Events per figure: `figure_description_described`, `figure_description_retried`,
  `figure_description_dropped` (data carries the asset name, reason and the offending TOKENS, never the
  rejected text). CLI summary line: `Figure descriptions: N described (M retried), K dropped`. No
  failure mode: optional enrichment cannot move page status, document status, table counts or
  `audit_passed` (pinned by a two-run difference test).
- Resume: results are cached per document in `figures/figure_descriptions.json`, keyed by image
  sha256 + prompt version + model; definitive outcomes (described, validator-dropped) are cached,
  "model unavailable" is not. A cached text is re-validated on read. Pages restored from a terminal
  fragment already carry their description and are skipped by the idempotence check.
- The run fingerprint records the prompt version when descriptions are effectively on, so a run
  with them and one without never share a resume gate. Consequence: the first run after this merges
  re-processes documents that resume from older output (default is now ON).

## Deviations from the brief

- Not inside the page loop. Extracted figures (`figure_N_pageP`) do not exist until the document-level
  figure phase after the loop, and chart-region refs are placed at assemble. A pass over each page's
  final body just before the single authoritative fragment/sidecar writers gives fragments, sidecars
  and resume the same bytes without moving figure extraction into the loop.
- Legacy `--describe-figures` is kept (its `**Figure N**` caption lines). When it is on, it owns the
  extracted `figure_N_pageP` crops and the new pass skips them, so no figure is described twice. It
  still uses the old prompt and can print numbers; retiring it is a follow-up.
- Golden tests: none changed. No existing byte-identity or golden test contained a crop ref and the model
  is absent for every test (conftest autouse stub), so figure pages are untouched there.

## Verification

- `tests/test_figure_descriptions.py` (58 tests): validator, pass / digit-then-retry / digit-twice-dropped
  / model-unavailable, allow-list, idempotence, cache, tampered cache, gate (flag, native-only), fingerprint,
  and four end-to-end `process()` runs with `_available_engines_for_agentic` and `_resolve_judge_model`
  patched (enabled vs disabled differ only by the description lines; native-only; dropped equals disabled;
  resume with a different model reproduces the same bytes with zero calls).
- Mutants in an external copy (src, tests and pyproject copied to /tmp; canary asserts the loaded
  `socr.__file__` is the copy; uncapped anchor count of exactly 1 asserted before editing): validator off
  (18 tests fail), ships on second failure (4), page-sized assets described (3), native-only ignored (2),
  no retry (5). All killed.
- Full suite: 6968 passed, 1 failed on the first run (`test_gh974_review_pins::test_budget_exhaustion_...`,
  a table-ladder disposition assertion), which passes alone and in three re-runs; that run overlapped the
  mutant runs. `uvx ruff@0.16.0 format --check .` clean.
