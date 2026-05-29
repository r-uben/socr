# Architecture Decision: Hybrid (deterministic backbone + agentic repair), not full agentic rewrite (2026-05-29)

## Context

The owner was frustrated ("can't get it running, it's a nightmare") and proposed
**replacing socr wholesale** with a fully agentic procedure: a set of `.md` +
`.claude/` configs where Claude itself reads each PDF page, decides per-page
which engine/tool to call, judges quality, and falls back as needed.

Two distinct problems were fused under "nightmare":

1. **"Can't run it"** — turned out to be largely *environmental*, not
   architectural: the `.venv` lived inside iCloud Drive, which evicted/
   conflict-copied `socr-2.4.0.dist-info/RECORD`, so `uv` failed with
   `RECORD file is invalid ... Operation timed out (os error 60)`. This is the
   exact iCloud-venv corruption documented in the global CLAUDE.md. Fixed by
   moving the venv off iCloud (`UV_PROJECT_ENVIRONMENT=~/venvs/socr`).

2. **"It's a nightmare"** — genuinely architectural, but *not* "deterministic
   vs agentic." The real rot is **five overlapping orchestrators** doing
   variations of the same job:

   | File | LOC | Role |
   |------|-----|------|
   | `pipeline/orchestrator.py` (UnifiedPipeline) | 1496 | 5-phase analyze→backbone→score→repair→assemble |
   | `pipeline/consensus.py` | 528 | cross-engine consensus |
   | `pipeline/hpc_pipeline.py` | 482 | HPC variant |
   | `pipeline/reconciler.py` | 332 | reconcile outputs |
   | `pipeline/repair.py` | 281 | repair pass |
   | `pipeline/processor.py` | 269 | StandardPipeline |

   Plus a heuristic stack: `audit/heuristics.py`, `audit/scorer.py`,
   `core/difficulty.py`, failure-mode scorer, repair router.

## Decision

**Do NOT do the full agentic rewrite. Consolidate to ONE deterministic
orchestrator built on the existing `DocumentState` blackboard, and make only
the repair/judgment stage agentic, on the small flagged subset of pages.**

This was independently recommended by Claude, Codex (gpt-5.4, high reasoning),
and Gemini — three-way convergence.

### Why not full agentic

- **Cost/latency.** ~350 papers × ~30 pages ≈ 10,500 pages. Most are clean
  born-digital text needing zero intelligence. Running an LLM judge on every
  page pays frontier-model tokens to rubber-stamp pages a 5-line check handles
  for free — a large, compounding tax.
- **Reproducibility.** This is a *citable research corpus*. Re-running OCR
  months later must yield comparable text. A fully agentic loop is
  non-deterministic: routing varies run to run. Poison for a versioned dataset.
- **HPC ergonomics.** SLURM array jobs want a deterministic, headless binary
  per shard that maximizes local GPU use. An agent loop blocking on external
  API rate limits idles compute nodes and resists clean resume.

### What is salvageable (most of it)

- **Engines as standalone CLIs + subprocess adapters** — keep untouched. This
  is exactly the clean tool surface an agent would want anyway.
- **`core/result.py` `PageOutput` / `EngineResult` contract — ALREADY EXISTS.**
  The docstring already states "Canonical engine contract: all engines return
  EngineResult with structured PageOutput list."
- **`core/state.py` `DocumentState` blackboard — ALREADY EXISTS.** Per-page
  `PageState` with `attempts`, `best_output`, `needs_repair`. This is the
  correct core; the competing orchestrators bypass or duplicate it.

So Codex's "highest-leverage change = make per-page `PageOutput` the canonical
contract everywhere" is *partially already done*. The real gap: whole-doc CLI
engines still collapse into a single `PageOutput(page_num=0)` holding the entire
document, and four orchestrators route around the blackboard.

## Second-opinion synthesis

- **Codex (gpt-5.4):** Hybrid, not full agentic. "Just consolidate the 5
  orchestrators" is too conservative on its own — pair consolidation with
  agentic judgment for the inconclusive hard pages. Per-page output as the
  universal internal contract is the key seam (explains the benchmark: whole-doc
  Gemini CLI is fast on short born-digital papers, but per-page gemini-api is
  the safe backbone on long papers because it eliminates truncation risk).
  Per-page LLM judgment over the full corpus is "not acceptable as the default";
  reserve for single-digit-percent flagged pages, persist every decision.

- **Gemini:** Agrees. Adds two critical guardrails:
  1. **Do NOT slice the *input* PDF page-by-page before OCR.** Whole-document
     engines (Marker, Nougat) rely on full-doc context (font dictionaries,
     header/footer patterns, bibliography structure) to resolve local
     ambiguities. Feed the *entire* document to the engine, then slice its
     *output* into the `PageOutput` contract for evaluation. Slice outputs, not
     inputs.
  2. **Heuristic calibration trap.** If the triage heuristics are naive they
     either silently pass mangled equations/tables (poisoning the corpus) or
     over-trigger the LLM fallback (degrading back into the agentic bottleneck
     we're avoiding).
  - Refinement: ground the LLM trigger in *hard data*, not brittle heuristics —
    invoke the LLM for **diff reconciliation** when two deterministic engines
    run the same page and produce structurally divergent output (edit distance
    over threshold), rather than as a pure pass/fail heuristic fallback.

## Resulting design

One deterministic orchestrator over `DocumentState`:

1. **analyze** — born-digital detection (deterministic, free). Clean text PDFs
   skip OCR entirely.
2. **backbone** — one primary engine per document. Whole-doc engines get the
   *full* PDF (no input slicing); their monolithic output is split into
   per-page `PageOutput`s afterward. Per-page API engines (gemini-api) preferred
   for long papers to avoid truncation.
3. **triage** — cheap deterministic heuristics as a *gate only*: flag suspect
   pages. Most pass and never see an LLM. Calibrate the gate against the
   benchmark to avoid both failure modes Gemini named.
4. **repair (agentic, flagged pages only)** — LLM/VLM looks at the page image +
   candidate OCR(s), judges quality, picks the next engine, re-runs. Triggered
   by hard signals (audit failure OR engine-disagreement diff), not vibes.
   **Decisions cached/persisted keyed by page hash, temperature 0** → re-runs
   replay the cache → reproducible.
5. **assemble** — deterministic stitch.

Net: deletes/absorbs `consensus.py`, `reconciler.py`, `repair.py`,
`processor.py`, `hpc_pipeline.py` and most of the scorer stack into one
orchestrator — roughly 3–4k LOC removed — without losing batchability or
reproducibility, and puts intelligence exactly where heuristics fail.

## Immediate action taken

- Moved venv off iCloud: `uv venv ~/venvs/socr --python 3.11`,
  `export UV_PROJECT_ENVIRONMENT=~/venvs/socr`, `uv pip install -e .`.
  (Fixes the RECORD timeout that was 50% of "can't run it".)

## Next

- See `TICKETS.md` (TICKET-12…) for the consolidation work.
- Open a GitHub issue to track the refactor before starting implementation.

---

## Revision after go-team panel (2026-05-29, later same day)

A full design session iterated the above toward "agentic, with Python parts for
reproducibility." A `/go-team` panel (Codex gpt-5.4 high + Gemini, independent)
pressure-tested the resulting architecture and **converged on two corrections
that overturn load-bearing assumptions.** Both adopted.

### Correction 1 — Python-on-top, NOT agent-on-top

The `.md`/`.claude` agent must **not** be the top-level driver of a multi-hour,
10.5k-page loop.

- Gemini: *"Agent-on-top is a fatal operational flaw... Control flow is for code.
  LLMs are terrible orchestrators but excellent single-turn decision engines."*
- Codex: *"Python-on-top. The agent should be an optional bounded policy module
  that proposes the next action... the agent is just one policy backend."*

Reframe (Claude): the real axis is **where durable, inspectable, crash-recoverable
state lives** — not agent-vs-Python ideology. A conversational context degrades,
wanders, and can't resume mid-batch. So **Python owns the loop, the retry budget,
the manifest writes, and error handling, and checkpoints state to disk per page.**
The LLM is consulted per page as a **stateless** function:
`decide_next_action(page_image, current_ocr) -> Action`. The `.md` survives as the
**judge/decide prompt** (policy text), never as the orchestrator. Entry points
stay Python: `socr agent`, `socr replay`, `socr batch`.

### Correction 2 — Manifest is an artifact cache, NOT a re-execution recipe

Earlier framing ("`socr replay` re-executes the manifest in pure Python →
bit-identical") was **wrong** and is retracted. VLM/LLM OCR is non-deterministic
*even at temperature 0*, plus silent provider model-version drift. You cannot
reproduce bytes by re-running.

- Gemini: *"`socr replay` should never invoke an engine; it should just fetch the
  cached PageOutput blob corresponding to that manifest entry."*
- Codex: manifest as *"execution journal + invalidation metadata + optional frozen
  artifacts"* is sound; manifest as "sequence of calls keyed by page_hash" is not.

So the manifest holds BOTH: (a) a content-addressed **cache of the winning output
blob** (replay serves these, zero model calls), and (b) an **execution journal +
input fingerprint** for provenance and invalidation.

**page_hash / fingerprint must cover the rendered image, not the PDF bytes.**
Codex's full fingerprint: `pdf_file_hash + page_num + rendered_image_hash +
render_params(DPI etc.) + prompt/template hash + engine id + engine version/model
id + decoding params + normalizer version + assembly version`. Gemini: hash the
**rendered PNG bytes** so a PyMuPDF render-drift invalidates the cache entry.

### Claude's added refinement — don't assume the loop; measure it

The existing code already does *single-step* escalation (lite local → cloud on
failure). It is plausible that **one** escalation fixes ~95% of flagged pages and a
multi-iteration self-correcting loop is over-engineering. So the judge benchmark
must also measure **iterations-to-fix**; set the repair depth from that data, not
by assumption. If the answer is ~1, build a gate + one escalation (mostly already
present), not a 5-engine loop.

### What is actually NEW work (scope shrinks)

1. The **cache / manifest / `socr replay`** layer (missing today) — the foundation.
2. Replacing **heuristic triage with a VLM judge**, benchmarked on labeled pages.

Everything else is consolidation of code that already exists.

### De-risked build order (committed)

1. **Cache + manifest + `socr replay`** (pure Python, zero models). Prove: OCR once
   → cache → `replay` rebuilds identical markdown with no API calls. If this
   plumbing is flaky, the rest is built on sand.
2. **Judge benchmark** — labeled good/mangled pages; measure false-pos/false-neg
   AND iterations-to-fix of the lite-model judge.
3. **Python-owned repair step** calling judge + engines as stateless tools; depth
   set by step 2.
4. Delete the 5 orchestrators + heuristic stack as their function is absorbed.

Building step 1 now. Tickets updated: TICKET-15 (cache/manifest/replay),
TICKET-16 (judge benchmark) supersede the earlier 11–14 framing.
