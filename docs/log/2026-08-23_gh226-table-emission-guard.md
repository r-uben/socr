# GH-226 table-emission guard

## Scope

GH-226 has two live, deterministic final-output defects after the narrow
GH-276 header repair:

1. A literal LaTeX table command can survive inside an otherwise rectangular
   Markdown grid.
2. A delimiter row can disagree with the header/body width because the current
   parser drops the delimiter before `table_output_defect` compares widths.

Both reproduce on `origin/main@855a121` in
`tests/test_gh226_table_emission_guard.py`: two tests fail because
`table_output_defect` returns no defect.

The historic `\\multicolumn{...}{c|}{...}` spelling is accidentally caught by
the current ragged-grid check because the parser mistakes the alignment pipe
for a Markdown cell boundary. The equivalent `{c}` spelling proves there is no
lexical guard; accidental parser behavior is not the invariant.

## Wave 0 decision boundary

- Detect and fail closed on residual table-only LaTeX commands.
- Compare raw header, delimiter, and body widths before the delimiter is lost.
- Ignore examples inside fenced/indented code and HTML comments.
- Cover agentic, non-agentic, whole-document, fragment, manifest, and replay
  output through one final winner seam.
- Do not add a generalized padding/span repair or change numeric-lane
  derivation in this wave.

## Current checkpoint

- Branch: `fix/226-table-emission-guard`
- Reproduction: complete; both focused tests failed on the baseline and pass
  after the change.
- Implementation:
  - `table_emission_defect` retains raw delimiter rows, ignores non-content
    regions, and splits escaped pipes correctly.
  - `table_output_defect` emits distinct `table_latex_leak` and
    `table_width_mismatch` codes before separator-free grid checks.
  - born-digital extraction and the post-route repair recheck consume the same
    predicate.
  - `_winning_page_output` is the universal hard-fail backstop for per-page,
    whole-document, fragment, manifest, and replay output.
  - `ASSEMBLY_VERSION` is now `3`, invalidating cached pages assembled without
    the new final guard.
- Policy boundary: a GH-226 emission defect remains a hard page failure even if
  the selected attempt was already flagged. This is intentionally different
  from GH-259's keep-and-flag treatment of a generic ragged body. It can discard
  valid prose on the same page; partial-page salvage or attaching a newly
  rendered page image is follow-up work, not a syntax repair inferred here.
- Verification:
  - 18 focused GH-226 tests pass.
  - 261 compatibility tests pass with 2 expected xfails.
  - Full suite: 2201 passed, 3 expected xfails.
  - 40 preserved TR-3 page artifacts: zero emission-guard firings.
  - Changed-file formatting, lint, and `git diff --check` pass. Whole-file lint
    on `born_digital.py` / `orchestrator.py` still reports unrelated pre-existing
    findings outside this diff.
- Independent review: Kimi found no parser/seam regression but surfaced the
  explicit hard-fail collateral above; Gemini 3.7 Flash returned no actionable
  findings on the current diff.
- Initial checkpoint action: show the first-commit plan before pushing PR #281.

## PR #281 review-fix round

Cubic and Qodo found four distinct roots after the first commit: post-figure
text could bypass the final guard through `saved_body`; literal command names
inside inline code, escaped backslashes, and raw `<pre>` blocks could false
fire; `PageOutput.text = None` could crash the guard; and born-digital audit
provenance collapsed GH-226 defects into GH-151 `grid_shape`.

Each finding was reproduced before the fix. The review round now:

- masks emission-only literal regions without changing GH-268 grid-selection
  policy;
- normalizes missing text;
- runs one shared guard over selected and exact post-transform page outputs;
- keeps final Markdown, fragments, sidecars, manifest blobs, replay, document
  status, and audit events consistent;
- persists the exact native emission-defect code through sidecars and resume.

The first full verification exposed two PP-2 tests whose spies pinned the
established `_flush_page_sidecar` signature. After a second Kimi review, the
signature was restored and exact final outputs are supplied through a scoped
mapping cleared in `finally`.

Final verification for this round: 25 focused tests; 243 compatibility tests
with 2 expected xfails; full suite 2209 passed with 3 expected xfails; CI's
`uvx ruff@0.16.0 format --check .` clean across 344 files; 40 preserved TR-3
pages with zero guard firings. Gemini 3.7 Flash independently reviewed the
corrected diff and returned no findings.

Current live state: review-fix commit `d03c5bd` is pushed to PR #281. CI,
typecheck, and Cubic are green; the PR is mergeable and awaiting merge. Issue
#226 remains open until the PR lands.

Next action: merge PR #281, then close GH-226 and remeasure/select the next
ticket (GH-271 is the current measured follow-up candidate).
