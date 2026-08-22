# GH-271 corrupt-equation region guardrail

2026-08-22. Implementation record for `fix/271-equation-winner-selection`, based on
`main@dc6e6c6`.

**Decision: keep corrupt-equation recovery opt-in and region-only.** A born-digital page with
trustworthy prose and positively detected equation corruption may re-read complete numbered
equation rows through the configured vision model. The pipeline never promotes a rejected
whole-page candidate. It preserves surrounding native text byte-for-byte, retains each crop as
source evidence, and ships every adopted candidate as unverified under `WARNING` /
`AUDIT_FAILED`.

The copyrighted source page, crops, native text, and generated Markdown remain outside the
repository. This record contains only architecture, aggregate results, and content-free
provenance.

## Why the original line detector was insufficient

The initial implementation cropped only native lines that visibly contained font-map corruption.
On the measured page, one equation was spread across several extraction-order fragments, so the
crop captured only a small part of the row and could not align to the native page text.

The existing display-equation detector was extended rather than duplicated:

- a terminal right-margin equation label anchors a complete numbered row;
- vertically connected math-font fragments are assigned to the nearest label by geometry, never
  by extraction-order range;
- a label with no math-font member is refused as a replacement target;
- only assigned member text enters the source target, and exact cleaned-native substring alignment
  remains mandatory before replacement;
- numbered rows and uncovered unnumbered corrupt lines coexist, but fragments owned by an
  unalignable numbered row cannot fall back to partial replacement;
- crop padding is bounded by whitespace between adjacent rows;
- right-margin prose ending in an equation reference is not an anchor unless its prefix is
  equation-like;
- fallback lines merge only when both geometry and cleaned-native contiguity agree.

A candidate counts as recovered only when its crop exists, its LaTeX is non-empty and structurally
valid, and its source slice aligns exactly. Failure at any boundary keeps the native source text
and appends explicit evidence instead of silently deleting content.

## Trust and policy boundaries

- Structural LaTeX validation establishes syntax only, never mathematical fidelity.
- Even a structurally valid candidate remains visibly non-authoritative and keeps its crop.
- The equation label comes from the source detector. One exact duplicate written by the model is
  removed before one canonical source-controlled `\\tag{...}` is appended.
- `--strict-local` blocks a model name marked as cloud.
- An active page or document cost cap blocks an unpriced remote equation call.
- A policy-blocked call still retains the crop, records zero attempts and zero cost, and exposes
  the skip reason.
- An allowed direct call has unknown cost rather than falsely reporting it as free.
- If an unmetered call precedes later pages under an active document budget, the remaining paid
  budget is unknowable; later paid rungs are suppressed while free rungs remain eligible.
- Native-only mode, table pages, and shredded rotated native text cannot enter the hybrid lane.
  Winner selection independently preserves the rotated-text hard floor if restored state is
  contradictory.

## Provenance and lifecycle

The hybrid is a first-class `native+math` page output with provider ID
`corrupt-math-region`, the configured model, and the `ollama-compatible` backend. The page,
document engine run, audit log, sidecar, metadata, and manifest retain that identity and the
prompt fingerprint. The page remains non-passing, so clean terminal resume does not accept it.
The final whole-document assembly and canonical page fragment remain identical after normal page
splitting.

## Content-safe real-page check after independent review

A strict-local run used `qwen3-vl:30b-a3b-instruct` on the locally retained source page:

- 2 positively corrupt regions retained as crops;
- 1 exact source alignment and 1 unalignable complete numbered row;
- both model candidates passed the structural LaTeX gate;
- 1 candidate adopted and 1 region left unresolved with its native bytes retained;
- surrounding native bytes were preserved;
- page status remained `warning` and document metadata remained `partial`;
- semantic fidelity remained explicitly unverified;
- direct-call cost remained unknown.

The earlier pre-review run reported 10 adopted numbered rows. That result is withdrawn: its
extraction-order source ranges could include unrelated bytes, and page-level corruption was being
used to reread clean numbered rows. The conservative result above has lower coverage and is the
intended safety trade-off. Visual comparison still found that structural validity does not imply
symbol fidelity, so the output is not citation-ready without checking every candidate against its
crop or source page.

## Independent review findings and remediation

Four independent read-only reviews completed through Grok 4.6, Claude Sonnet, Composer 2.5, and
Gemini 3.7 Flash. Grok, Sonnet, and Composer blocked the first diff; Flash passed it with minor
recommendations. Every accepted finding below was reproduced before repair:

- extraction-order ranges crossed numbered rows and could delete unrelated native bytes;
- label-only anchors could target an earlier prose citation;
- `numbered or fallback` suppressed separate unnumbered corruption on mixed pages;
- fallback geometry could merge non-contiguous native lines;
- clean numbered rows were reread merely because another row made the page corrupt;
- the direct call ignored the configured Ollama host;
- page cost was unknown while document cost falsely reported zero;
- `DocumentState.text` preferred deficient native text over the explicit hybrid;
- the legacy path recorded and named the hybrid engine twice;
- model output containing `\\tag{(A8)}` could survive beside the canonical source tag.

The repairs use geometry-connected ownership, exact cleaned-native alignment, numbered-row
ownership of its fragments even on alignment failure, union with uncovered fallback regions,
configured-host propagation, explicit unknown-cost propagation, authoritative hybrid selection,
single engine-run accounting, and source-controlled tag normalization.

On the second review pass, Grok returned `PASS`, Sonnet reported no substantive issue, Composer
reported no merge blocker, and Flash reported no remaining finding. Composer's non-blocking
docstring and audit-payload observations were also addressed; unknown prior spend intentionally
suppresses later paid rungs under an active budget.

PR #280's CodeRabbit pass found three further defects, each reproduced before repair: unresolved
evidence used `rstrip()` and changed trailing native bytes; trailing math-span whitespace could
inflate the detector ratio above one; and a contradictory chart+math page could enter the region
lane without retaining its mandatory chart PNG. The final implementation preserves the native
prefix exactly, counts only retained characters, and embeds the chart asset alongside the math
hybrid with an explicit arbitration event.

Cubic's PR pass additionally found that combined legacy engine names could lose the math prompt
fingerprint, a restored hybrid could outrank a hard table floor, and numbered artifacts bypassed
the height floor. Each was reproduced and fixed. Its claim that identical source strings made the
second region unalignable was refuted by a passing two-occurrence regression; `replace(..., 1)`
leaves the second native occurrence available for the next region.

## Verification

- Focused equation/recovery/state/orchestrator regressions: 293 passed.
- Full suite after all PR review regressions: 2,181 passed, 3 expected failures.
- Exact Continuous Integration formatting gate: 342 files formatted.
- Whitespace check: clean.
- Load-bearing sabotage: forcing the region eligibility route off made the paired regression fail
  because whole-page optical character recognition ran; restoring the source made it pass.
- Repository-wide Ruff remains advisory and reports pre-existing debt; the changed focused modules
  pass Ruff. The configured `mypy` executable is absent from the development environment, matching
  the advisory CI job's current inability to run it.
