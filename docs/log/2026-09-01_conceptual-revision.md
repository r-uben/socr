# Conceptual revision — what socr is, and the shortest path to it

2026-09-01. Owner asked for a conceptual and code-level revision of the whole tool.
Five independent reviewers (Codex gpt-5.6-sol, Grok 4.6, Gemini 3.1 Pro, Kimi K3,
Claude Opus 5) worked the same brief read-only against `main` at `75a82ce`, each
with a grounding canary. Claude Fable synthesised and verified every claim below
against the code before writing it down. Reviewer reports are not committed; the
brief and this log are the record.

## The owner's intent, in one sentence

Page by page: try the free native text layer, check its quality, and call a vision
model only when needed — which, for tables, figures and formulas, is most of the time.

## What the code does instead — verified

Unanimous across all five reviewers, and confirmed line by line:

1. **A flag decorates; it never routes.** A trusted-native page ships `SUCCESS`, or
   `WARNING` if a table-distrust flag is already set, and no flag ever sends it to a
   model (`_agentic_native_page`, `orchestrator.py:4213-4259`). Post-route, a defect
   found after header repair demotes status in place and explicitly does not reroute
   (`orchestrator.py:3616-3652`). Open as #317; symptoms #151, #140, #271, #223.
2. **Native geometry is extractor, verifier and fallback at once.** `reconstruct.py`
   (2,059 lines) authors grids from word geometry; `native_verifier.py` (1,170) grades
   model grids against the same words; when every ladder rung is refused on a
   structure-class page, the grid that ships is the native one, as `WARNING`
   (`structure_class_native_fallback_applies`, `manifest.py:787-812`). The referee is
   the thing already measured to be the worst reader
   (`2026-08-30_model-vs-native-table-rows.md`: invented+missing native 21, qwen 14,
   gemini 7). Not filed as a single issue.
3. **An abstaining verifier is treated as certainty.** `EXACT_PASS` accepts at
   confidence 1.0 without the inner judge; a header-attribution `UNVERIFIABLE` only
   emits an event and returns that acceptance (`agentic.py:736-754`, `:793-811`).
   Open as #245.
4. **Judged bytes are not shipped bytes.** `repair_table_headers_on_page` mutates the
   accepted text after the judge ran; the compensating recheck is string-only
   (`orchestrator.py:3582-3652`). Four of five reviewers; not filed on its own.
5. **Assemble is a second policy engine.** `_select_page_output_tagged` has 15
   endings; `_phase_assemble` mirrors them in buckets that must agree with it by hand
   (`manifest.py:902-1383`, `orchestrator.py:5930-6018`). Open as #176.

Two facts the reviewers missed and this log adds:

- The escalation the owner is asking for is **already built and switched off**. The
  GH-353 table-judge ladder board is closed (16 of 16 tickets DONE,
  `docs/plans/table-judge-ladder/STATUS.md`) and `table_judge_ladder` defaults to
  `False` (`config.py:294`), gated on #359 pinning three terminals.
- The structure lanes the owner considers the point of the tool default off:
  `detect_equations`, `recover_clean_equations`, `save_figures`, `describe_figures`
  are all `False` (`config.py:189-233`). An equation page with no table signal takes
  the free bypass (`_is_trusted_native_without_ocr` excludes tables only,
  `orchestrator.py:1336`).

Discarded reviewer claims, with the check that discarded them: "the legacy
deterministic stack is still present" (Kimi) — `pipeline/` holds four modules,
`consensus.py` and `repair.py` are gone, `--legacy-routing` has zero hits. Only the
docs are stale (fixed in this commit).

## Ruling — the target shape

Native geometry is a **reference and a verifier**. It authors shipped bytes only for
prose. It never authors a grid that ships without a model having tried first, and it
is never the consolation prize when the ladder is exhausted.

Per page, one procedure:

1. Read the native layer. Always, free. Record facts (tables, figures, equations,
   glyph/encoding hygiene), not decisions.
2. Free lane: born-digital prose with **no structure signal and no hygiene flag**.
   Ship native. This lane stays unwitnessed because there is nothing to witness.
3. Everything else enters the cost ladder. The free deterministic verifier runs
   ahead of the VLM judge, as `ARCHITECTURE.md` already says. Its verdict is
   `PASS | FAIL | ABSTAIN`. `FAIL` escalates. `ABSTAIN` goes to the judge, never to
   confidence 1.0.
4. Header repair, crop rereads and the table-judge ladder are escalation tools fired
   by a signal, run before the verdict, never after an accept.
5. Ladder exhausted on a structure-class page: ship the fail-closed floor (marker plus
   page PNG, the existing D3 path) — never the native grid the verifier already
   refused. A ragged native table is not "better than nothing" in a citation corpus;
   the PNG is how a human recovers the cells.
6. One winner selector with three endings: native prose, accepted model output,
   fail-closed marker. Assemble stitches; it does not retry policy.

Rejected alternative: make `native` a $0 rung inside `route_page` and judge every
page (Gemini, Kimi, Opus). Rejected because it spends a judge on the 58% prose lane
where the measured losses are not, and it conflates "did the model invent a number"
with "did our geometry pick the right column". Grok and Codex argued the same; the
free lane survives, narrowed.

## Programme — in order, smallest blast radius first

| # | Change | Deletes | Closes | Size |
|---|--------|---------|--------|------|
| P0 | Abstain is not a pass: `UNVERIFIABLE` on the `EXACT_PASS` path delegates to the inner judge | the confidence-1.0 short-circuit on abstain | #245 | ~15 LOC |
| P1 | Flip `table_judge_ladder` on, after #359 pins the terminals | the flag, eventually | #359, then #353 flip | design + ~30 LOC |
| P2 | Exhausted ladder on a structure page ships the D3 floor, not the native grid | `structure_class_native_fallback` as a ship path (~100 LOC of endings) | the substantive half of #317 | ~150 LOC |
| P3 | Judged bytes are shipped bytes: header repair moves inside the verifier before its verdict; the post-route recheck goes | `orchestrator.py:3582-3652` (71 LOC) | unfiled defect 4 | ~100 LOC |
| P4 | Widen the free-lane exclusion from `has_tables` to any structure signal or hygiene flag; decide the structure-lane defaults | — | #140, #271 partly | owner decision, see below |
| P5 | Dual-pass reread becomes escalate-on-signal (`dual_pass_tables` default off, kept for the escalation path) | in-loop block `:3775-3807` as trunk | — | ~40 LOC |
| P6 | Collapse the 15-ending selector to three; assemble derives buckets from the tag | ~400-500 LOC in `manifest.py` and `_phase_assemble` | #176, the real lever of #155 | large, last |

Every ticket pins a **difference**, never a value (CLAUDE.md, #257): the same fixture
with the flag off and on, both provider states, asserting that only the intended
outcome moves.

P0 and P5 need no design. P1 is already designed and blocked on #359. P2 and P3 get
one design note each before dispatch. P6 waits for P2–P5 to land, because each of
them removes endings it would otherwise have to preserve.

## Open for the owner — one decision

P4 turns on lanes that cost money or time. Which default does the owner want?

- **(a) Owner intent, literally:** `detect_equations` and `save_figures` default on;
  equation and figure pages leave the free lane. Every equation page pays a local
  VLM call; the corpus rate is unmeasured.
- **(b) Route, don't extract:** equation and figure pages leave the free lane and go
  to the ladder, but LaTeX splicing and figure description stay opt-in. Cheaper;
  formulas still ship as whatever the model's page read gives.

Recommendation: (b) now, (a) after one corpus run measures the equation-page rate.

## Process notes

- Cross-vendor subagents through the gateway fail from a plain `claude` session;
  the vendor CLIs ran headless in herdr panes instead and wrote to disk. Artifacts on
  disk were the only completion signal used.
- A git worktree tests its own source with `PYTHONPATH=<worktree>/src`; the
  editable install's path entry loses to it. This log was written that way while
  another session held the main checkout.
