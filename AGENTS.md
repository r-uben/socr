# AGENTS.md — socr

Multi-engine document OCR (`socr` routes pages, audits quality, falls back). Build, test,
lint, architecture, and branch conventions live in **`CLAUDE.md`** — read that first.

**GitHub advisory reviewers are mention-only.** gitty (the PR desk) triggers a second opinion
with `@codex review` or `/gemini review`. Do not auto-review every pull request.

## Code Review Rules

### Silent data loss is P0

This repo serves citation OCR: a wrong or dropped number is worse than a missing one. Flag any
change that can lose or corrupt content without surfacing failure at every level that matters —
page status, document status, metadata, and CLI — not just one layer. Do not restamp SUCCESS
over a structural miss (tables, figures, equations, routing, assembly).

### Tests must fail if production behavior regresses

Agentic tests that drive `process()` or `_phase_agentic` must patch `_available_engines_for_agentic`
(CI has no ollama). Assertions belong at real shipping boundaries — `process`, `_phase_assemble`,
or the flush that writes terminal page artifacts — not a helper that can stay green while production
reverts. Pin an in-process behavioral difference, not a local absolute outcome tied to a live provider.

### Resume and audit events must survive replay

When a diff introduces a new audit/resume event kind that must persist across resume, it must appear
on the appropriate replay allowlist (`EQUATION_LANE_EVENT_KINDS`, `TABLE_DISTRUST_KINDS`, and peers)
with a test that goes red if the kind is omitted from that set.

### Local VLM model identity

Local OCR must reference **`qwen3-vl:30b-a3b-instruct`** (instruct, non-thinking). Flag
`qwen3-vl:30b` (thinking; runs away on dense tables) or `:8b` (collapses tables).

### Leave mechanical format and lint to CI

Do not nitpick formatting in review comments. CI enforces `uvx ruff@0.16.0 format --check .`
(blocking). `ruff check` is advisory.
