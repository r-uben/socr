# STATUS — orchestrator decomposition

Last updated: 2026-08-27
Stage: **the decomposition's goal was achieved by DELETION, not by carving.**
`#174` shipped: the legacy deterministic pipeline is gone from `main`, and with it
~6,400 lines. `process()` is now analyze → agentic → assemble.

Next action: **`#293`** — see "Where this actually stands" below.
The analysis and a verified reproduction are on issue #293. **There is no branch** — an
earlier version of this file sent readers to `fix/293-native-fallback-overclaim`, which was
never pushed and 404s.

> **This file was two days and six merged PRs out of date until 2026-08-27.** It read
> "designed, not started. No source code written yet." while R1–R4, R7a and the entire
> `#174` deletion had already shipped. The updates existed — on branches that were never
> merged (`docs/plan-174-sequencing`, 8 commits; `refactor/r7-part2-disposition-classifier`,
> 3). Plan state written on a feature branch is invisible to everyone reading `main`.
> That is the failure `docs/adr/0002-claims-that-matter-get-a-test.md` describes, in the
> plan file that is supposed to prevent it.

## Where this actually stands (2026-08-27)

### Shipped to `main`

| PR | What | Ticket / issue |
|---|---|---|
| #286 | R1 — extract the trusted-native lane | R1 |
| #287 | R2–R4 | R2–R4 |
| #290 | R7a — tag the 15 cascade endings (`WinnerKind`) | R7a |
| #295 | broke the `tables`↔`benchmark` import cycle; layering guard covers relative, dynamic and private-symbol evasions | closed #175 |
| #296 | `ARCHITECTURE.md` and CLI help state agentic as the sole default | R174a |
| #298 | **deleted the legacy deterministic pipeline** — 13 orchestrator methods, `consensus.py`, `repair.py`, 6 config fields, and `--legacy-routing` / `--multi-engine` / `--consensus-llm` | #174 (still OPEN — see below) |
| #299 | empty-table gate — a grammar, not a threshold | closed #190 |
| #305 | de-rotate pages before OCR **and before judging** | closed #304 |
| #309 | de-rotate table crops; boxes stay in page space | the deferred half of #304, not an issue |

**"Shipped" here means the LEVER moved, not that GitHub closed the issue.** `#174` and `#155`
are both still **open**: the code is gone, but `ARCHITECTURE.md`, `README.md` and `CLAUDE.md`
still advertise the corpse. Closing them needs that documentation pass — tracked by `#156`.

`#309` left its own residue: **#310, #311, #312** — including one where a `docs/log/` note
contradicts the code it describes.

`main` at `f086f4a`. ADR 0001 and 0002 and the `#174` ruling all reached `main` via #298.

### The `#155` question is settled

`#155` asked to split the god-module. The measurement said the module split yields ~470
lines while the deletion yields ~6,400 — so **`#174` delivered `#155`'s goal**, and the
R-tickets are the remainder. Do not schedule R-tickets expecting them to shrink the file
much further.

### Cross-repo fallout from #298 — deleting a flag reaches outside this repo

`--multi-engine` and friends were named in two other repos, and nothing here could detect
it. `ai-skills#39` (merged) fixed the `/ocr` skill *before* #298 landed, so nothing broke.
`disputatio#62` fixes the last reference. Tracked as `#300`.

**Before deleting any user-facing flag, grep the skills repo and `disputatio`.**

### `#293` — the next action, and what it needs

`native_fallback_pages` claims pages that ship a failure marker rather than native text.
78 of 4,096 synthetic states diverge. A verified reproduction is on the issue.

**There is no failing test for #293 today, and the sweep is not one.**
`tests/test_r7_bucket_tag_equivalence.py` (on `refactor/r7-part2-disposition-classifier`,
unmerged) proves four buckets equivalent to their tags and pins #292 as a strict superset.
It does **not** assert the 78-state `native_fallback_pages` / `ROTATED_TEXT_SHREDDED`
overclaim. Cherry-pick the harness for its shape — but expect to *write* the #293 case, not
to find it already failing. An earlier version of this file called that sweep "the oracle"
for #293; that was wrong.

**Line pins on `main`@3d8522f** (`orchestrator.py`, 6,331 lines):

| | line |
|---|---|
| `failed_pages` | 4744 |
| `corrupt_math_hybrid_pages` (#292) | 4859 |
| `native_fallback_pages` (#293) | **4864** |
| `native_fallback_pages` (routing-time, in `_phase_agentic` — NOT the bug) | 2298 |

An earlier version pinned "4744-4800", which is `failed_pages`, not the #293 predicate.
Every one of these carries a comment about a past double-counting bug caused by drift from
what `_winning_page_output` actually ships; the exclusion must match the ship disposition
**exactly**.

Then `#292` (same shape, `corrupt_math_hybrid_pages`, 1,984 divergent states), then R7b
becomes the trivial tag swap it was always meant to be, then R8/R9/R10 — which finally
unblocks `#176`.

### Known open, not scheduled

- `#297` figures clustering — another session's PR
- The VLM judge logs a timeout on rotated pages even when they now succeed. Bound is
  `max(DEFAULT_PROVIDER_TIMEOUTS)` = 300s; judging one page should not approach it.
  Observed, never diagnosed.
- `--fallback`, `--no-audit`, `--no-judge-hard-pages` still advertise behaviour they can
  no longer change. `#142`'s `_INERT_BUT_FINGERPRINTED` list is now *dead*, not inert.

### Running socr at scale

`docs/hpc/` is **gitignored** by repo convention, so the verified vLLM runbook lives in
`ai-skills#40` instead — corrected partitions (the ones this skill named do not exist),
the 175 G/180 G home quota, and the trap where a missing `PATH` entry makes socr fall off
the VLM and write `model: none` **at exit 0**.

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

## Measured facts

> **Restated against `main`@3d8522f, 2026-08-28.** The table below the divider is the
> ORIGINAL 2026-08-23 measurement at `bc194e9`, kept because the decomposition argument was
> built on it. It is history, not current state — `#298` has since removed ~1,150 lines and
> `_phase_repair` no longer exists. Do not schedule from it.

| Fact | 2026-08-23 (`bc194e9`) | now (`3d8522f`) |
|---|---|---|
| `orchestrator.py` | 7,520 lines | **6,331** |
| `_phase_repair` | 209 lines | **deleted** (#298) |
| `_phase_backbone`, `_phase_consensus`, `_phase_score*`, `_backbone_*` | present | **deleted** (#298) |

### Original measurement (main@bc194e9, 2026-08-23) — historical

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

- `docs/plans/TICKETS.md` — last touched 2026-08-10. Covers 4 of the
  55 open issues. Issue #156 tracks this drift.
- `docs/plans/gh144-rowizer-destroys-values/STATUS.md` — dated 2026-08-11, still reads
  "Scaffolded, not dispatched" with A1/A2/A2b/A3 all TODO, while its fixes (d645b24,
  be2c3e4) have landed.
