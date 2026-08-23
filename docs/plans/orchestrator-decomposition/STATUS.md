# STATUS — orchestrator decomposition

Last updated: 2026-08-23
Stage: **measured, not designed.** No code written. No tickets dispatchable yet.
Next action: dispatch the design ticket in `TICKETS.md` (D1) — read the two large
functions and propose seams. Nothing else starts before D1 lands.

## Why this plan exists

`src/socr/pipeline/orchestrator.py` is the file almost every open bug touches. Before
ordering the 51 live issues, we measured it. The measurement changed the framing, so it
is recorded here rather than left in a chat log.

## Measured facts (main@bc194e9, 2026-08-23)

| Fact | Value |
|---|---|
| `orchestrator.py` | **7,520 lines** |
| Functions in it | 100 |
| Functions > 100 lines | 18 |
| Functions > 200 lines | 5 |
| `UnifiedPipeline._phase_agentic` | **1,101 lines** |
| `UnifiedPipeline._phase_assemble` | **895 lines** |
| Lines inside function bodies | 7,342 (97%) |
| Whole `src/socr` tree | 36,796 lines |

Next four by size: `_backbone_native_first` 293, `_describe_and_embed_figures` 211,
`_phase_repair` 209, `_phase_analyze` 200.

**`CLAUDE.md` and issue #155 both describe this file as ~5.5k LOC. It is 7,520.** It has
grown ~36% since that number was written, and it is still growing — the GH-226 work merged
on 2026-08-23 added a further +585 lines to it. Any plan that treats #155's figure as
current is planning against a stale target.

## The correction that shapes this plan

The intuition going in was "the file is too long, split it into modules." The measurement
does not support that as the primary move.

- The mass is **not** spread across 100 functions. Two functions are 2,000 lines — over a
  quarter of the file on their own.
- Carving the file into modules would relocate `_phase_agentic` intact into a new file. The
  1,101-line function would still be a 1,101-line function, now with an import.
- 97% of lines live inside function bodies, so there is almost no module-level structure to
  redistribute. The complexity is *intra-function*, not *inter-module*.

**Therefore: the unit of work is decomposing those two functions, and the module boundaries
should fall out of the seams that decomposition exposes — not be chosen up front.**

This is the open question D1 must answer. It is deliberately not answered here.

## Sequencing rule (decided 2026-08-23)

**Structure moves land behaviour-identical. Bug fixes land separately, afterwards.**

The tempting alternative — fix issues while carving each module, "designing each module the
right way" — was considered and rejected for this codebase specifically:

- When a golden/byte-identity test breaks mid-refactor, you must be able to conclude "the
  move was wrong." If a behaviour fix rode along in the same commit, you cannot.
- This repo already has byte-identity guarantees on the assembled `.md` output and a
  documented history of provider-dependent outcomes differing between local and CI. Those
  make a mixed commit especially hard to bisect.

The bugs still get a vote: **they inform where the seams go.** A seam that makes #144, #151,
#215 or #270 easier to fix is a better seam than one that does not. D1 should read the live
issue list with that lens.

## Relationship to the open-issue backlog

A two-model evidence-gated triage of the 56 unboarded open issues ran 2026-08-23
(112 agents, 0 errors). Outcome:

- **5 closed as verified-fixed**: #147, #195, #198, #205, #268. Each was confirmed by
  checking the fixing commit is an ancestor of `main` *and* running the cited regression
  tests (176 passed).
- **5 rescoped, kept open**: #144, #146, #151, #158, plus a correction on #144 — its
  strict-xfail canary `test_no_table_value_is_lost` now passes, so the "still drops four
  values" claim does not reproduce on current main.
- **51 remain live.** By area: tables 10, routing 6, architecture 5, equations 5, audit 5,
  cli 4, proposals 4, figures 3, born-digital 2, agentic 2, one each of pipeline, docs,
  review, structure, fabrication.

**Ordering those 51 is deliberately deferred until D1 lands**, because the decomposition
determines which of them can run in parallel. Today almost all of them serialize on one
file; that is the actual reason the backlog cannot be parallelized.

## Known-stale planning artifacts (do not schedule from these)

- `docs/plans/TICKETS.md` / `STATUS.md` — last touched 2026-08-10. Covers 4 of the
  55 open issues. Issue #156 tracks this drift.
- `docs/plans/gh144-rowizer-destroys-values/STATUS.md` — dated 2026-08-11, still reads
  "Scaffolded, not dispatched" with A1/A2/A2b/A3 all TODO, while its fixes (d645b24,
  be2c3e4) have landed.
