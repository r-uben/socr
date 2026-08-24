# STATUS — orchestrator decomposition

Last updated: 2026-08-24
Stage: **pass one in progress.** R1–R4 landed; R7 part one landed (#290). R5 `HELD`.
R7 part two is **`NEEDS-DESIGN`, not ready** — its scope was falsified today, see below.
Next action: rewrite R7 part two's scope in `TICKETS.md` before writing any more of its
code. The mapping it assumed does not exist.

## 2026-08-24

**R7's design fork was ruled: option B — tag the cascade** (#288). Each of the 15 returns
in `_select_page_output` carries an internal disposition tag; the public function drops it,
so output stays byte-identical. Decided on Kimi's finding, not diff size:
`corrupt_math_hybrid`'s disjointness from the model-kept buckets is only *implicit*
(`manifest.py:411`, `:608`), so a re-deriving classifier would have to INVENT a priority
order nothing has ever needed — a new unrecorded decision, made blind, inside a change
contracted to change nothing. Tagging inherits the cascade's own order.

**R7 part one shipped** (#290, merged). `WinnerKind` names all 15 endings; AST-verified as
15 returns / 0 loops, so exactly one ending runs per page and exclusivity is structural.
Two review findings fixed before merge: `NATIVE_FALLBACK` was covering two dispositions
(GPT — it also served the ordinary clean native page, which would have made every healthy
`--native-only` page count as a fallback), and the bijection test compared sets, so
transposed tags passed silently (peer session). The owner also retired the `D3_` prefix on
the new members: it names Option D3 of a 2026-06-17 design menu and carries no meaning.

**R7 part two's premise is false, and this is the day's real finding.** The ticket implies
the buckets can be swapped to read the tag. They cannot, because they do not map one-to-one:

- Four buckets ARE exactly equivalent, proven over 4,096 synthetic states and pinned in
  `tests/test_r7_bucket_tag_equivalence.py`: `flagged_model_pages`,
  `structure_class_model_pages`, `structure_class_native_fallback_pages`,
  `d3_model_table_pages`.
- One tag can cover SEVERAL buckets. `NATIVE_FALLBACK` covers both `native_fallback_pages`
  and `text_grid_rejected_pages`, which are deliberately separate lists.
- Two buckets OVER-CLAIM — they count pages the manifest ships as something else:
  - `corrupt_math_hybrid_pages` (#292): diverges in 1,984/4,096 states. Hides a real
    native-fallback page behind a false exclusion, and drives a document to AUDIT_FAILED
    over a page that ships `NATIVE_CLEAN`.
  - `native_fallback_pages` (**unfiled**): 78 states where it claims a page that actually
    ships `ROTATED_TEXT_SHREDDED`. Same shape as #292, distinct cause.

So part two is not a swap; it is a mapping layer that must reproduce today's behaviour
**including both bugs**, since R7 is contracted behaviour-identical. Both fixes are
deferred: #292 filed, the second still to file.

**Correction to the fan-out claim below.** Decision 2 says implementation cannot fan out
across worktrees "as a property of the repo". That is too strong. It is a property of the
single SHARED venv: `~/venvs/socr` carries a `.pth` pointing at this checkout, so any
worktree using it tests main's source. A per-worktree venv (`uv venv .venv && uv pip
install -e .`) defeats it. Two residues make it fragile rather than free: `CLAUDE.md`
hardcodes `~/venvs/socr/bin/pytest`, so an agent following repo instructions silently tests
the wrong source with NO detectable symptom (tests pass, against the wrong code); and
anything driving the local VLM still serialises on one GPU. Recorded because the owner's
parallel-sessions plan was parked on the stronger claim.

D1 landed 2026-08-23 (`docs/log/2026-08-23_orchestrator-seams.md`, PR #284). Two decisions
were ratified on the back of it:

1. **Two passes.** Pass one names the pieces in place as methods; pass two moves out only
   the disposition group to its own module. The five per-page lanes stay methods
   permanently — they are `self.config`-heavy and moving them buys an import.
   Honest expected shrink: **~470 lines, not 2,000.** Issue #155 overpromises and is
   rescoped by R11.
2. **The fleet cannot fan out on implementation.** socr is installed editable, so
   `import socr` resolves to this checkout regardless of worktree — four models in four
   worktrees would all test the same source. Design and review run in parallel (read-only);
   implementation is strictly serial, one branch at a time. `agent-fanout` is unavailable
   for this work as a property of the repo, not a preference.

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

D1 answered this on 2026-08-23. The answer: the disposition group in `_phase_assemble`
does fall out as a module; the five lanes in `_phase_agentic` do not, and stay methods.

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
#215 or #270 easier to fix is a better seam than one that does not. D1 did read the live
issue list with that lens; the per-seam issue mapping is section 4 of its note.

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

### Correction (2026-08-23, same day): the block is partial, not total

An earlier draft of this file said ordering all 51 was deferred until D1. That was too
strong, and the measurement contradicts it:

- **30 of the 51 cite `orchestrator.py` in their triage evidence.** These genuinely
  serialize on the file and wait for D1.
- **21 do not.** They are orderable and workable today, independent of the refactor.

The ordered 21 are in `TICKETS.md` under "W — work available now". The remaining 30 are
D2's job; D2 is unblocked as of 2026-08-23.

## Known-stale planning artifacts (do not schedule from these)

- `docs/plans/TICKETS.md` / `STATUS.md` — last touched 2026-08-10. Covers 4 of the
  55 open issues. Issue #156 tracks this drift.
- `docs/plans/gh144-rowizer-destroys-values/STATUS.md` — dated 2026-08-11, still reads
  "Scaffolded, not dispatched" with A1/A2/A2b/A3 all TODO, while its fixes (d645b24,
  be2c3e4) have landed.
