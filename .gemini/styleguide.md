# socr — Gemini Code Assist style guide

Multi-engine document OCR for citation corpora. Build, test, lint, and architecture details
are in **`CLAUDE.md`**. This file tells you what to flag on `/gemini review` and what to
leave alone.

## When you review

Reviews are **mention-only**. gitty triggers `/gemini review` when it wants a second opinion.
Do not assume every pull request needs comments. When invoked, focus on durable correctness
issues (P0/P1), not style nits CI already catches.

## Flag: silent data loss (P0)

Citation OCR treats a wrong or dropped number as worse than a missing one.

- Flag paths that drop tables, figures, equations, or numeric content without failing loudly.
- Failures must surface at page status, document status, metadata, and CLI — not just one layer.
- Flag restamping SUCCESS when assembly, routing, or verification still shows a structural miss.

## Flag: tests that would not catch a production revert

- Agentic end-to-end tests must patch `_available_engines_for_agentic` (CI has no ollama).
- Assert at shipping boundaries: `process`, `_phase_assemble`, or the terminal page flush — not
  isolated helpers that stay green while the real path regresses.
- Prefer pinning an in-process behavioral difference over outcomes that depend on a live provider.

## Flag: resume/audit events missing from replay allowlists

New event kinds that must survive resume belong on the right allowlist (`EQUATION_LANE_EVENT_KINDS`,
`TABLE_DISTRUST_KINDS`, and similar). Flag additions that emit a kind but omit it from the set, or
that lack a test proving omission would fail.

## Flag: wrong local VLM identity

Local OCR must use **`qwen3-vl:30b-a3b-instruct`**. Flag `qwen3-vl:30b` (thinking) or `:8b`
(table collapse).

## Safe to skip

- Formatting and import order — CI runs `uvx ruff@0.16.0 format --check .` (blocking).
- Generated or vendor paths matched by `.gemini/config.yaml` `ignore_patterns`.
- Subjective refactors with no behavioral change unless they touch the areas above.

## Helpful review tone

Be specific: file, mechanism, and the user-visible failure mode. Prefer one high-severity finding
over many low-severity nits.
