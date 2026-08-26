# ADR 0001 — Stay Python; optional native kernels only after profiling

Status: **Accepted** · 2026-08-25 · Closes [#178](https://github.com/r-uben/socr/issues/178)

## Decision

socr stays Python for orchestration and policy. There is no Rust rewrite.

Native code (Rust or C++) is admissible **only** for pure, deterministic geometry or
token kernels, and **only after** profiling names that specific function on the critical
path. "It feels slow" is not profiling.

## Context

The file this decision keeps getting raised about is `src/socr/pipeline/orchestrator.py`,
7,593 lines with 97% of them inside function bodies. Large enough that "rewrite it in a
fast language" recurs as a proposal, and large enough that acting on it would be
expensive to reverse.

## Why not Rust

**The bottleneck is not the interpreter.** Runtime is dominated by page rendering
(PyMuPDF), OCR engines invoked as subprocesses, and VLM calls over HTTP. Each is either
already native or is network latency. A rewrite moves the orchestration layer — the part
that spends most of its wall-clock waiting on the other three.

**The product risk is structural correctness, not speed.** What actually costs this
project is content silently lost or mis-shipped: table structure, equations, figures, and
the dual-path divergence between agentic and legacy routing. None of that gets better in
a faster language, and a rewrite would relitigate every correctness gate currently held
by ~1,070 tests.

**Orchestration policy is the product bet.** socr's value is deciding what to run, when
to escalate, and when to distrust a result. That is branch-heavy, config-heavy, rapidly
changing code — the profile Python suits and a compiled language taxes.

**This claim is reasoned, not measured.** No profiling run backs the "rendering and HTTP
dominate" statement. If someone wants to overturn this ADR, a profile is the way; do not
overturn it on intuition either.

## What is admissible

Pure functions with no I/O, no config, and no policy, where profiling has shown them hot.
Candidates named in #178, to be treated as candidates and not as a plan:

- `src/socr/tables/native_rows.py`
- `src/socr/tables/reconstruct.py`
- `src/socr/tables/native_verifier.py`
- figure drawing / clustering geometry

Any such extraction must sit behind a stable interface and be pinned by golden vectors
taken from the existing tests, so the native and Python implementations are provably
interchangeable.

## What is not admissible

Rewriting the CLI, the engine adapters, or the orchestrator in a native language without
a measured bottleneck naming them.

## Consequences

- Issue #155 (splitting the orchestrator) is a **Python** refactor. It was never a
  language question.
- Performance complaints route to a profiler first, and to this ADR second.
- If a profile ever does name a hot pure function, this ADR permits the work — it sets a
  gate, not a ban.
