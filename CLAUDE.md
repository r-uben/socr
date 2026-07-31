# CLAUDE.md — socr

Multi-engine document OCR. Each engine is a standalone CLI subprocess; `socr` routes
pages, audits quality, and falls back. Two modes: deterministic (default) and
**agentic** (`--agentic`, the primary path).

## Build / test / lint (use these exact tools)

- **Python:** `uv run …` or the direct binaries `~/venvs/socr/bin/{python,pytest,ruff,socr}`.
  Never `python script.py`. The editable install resolves `import socr` to **this repo's
  `src/socr`**, so edits are picked up — but that means a *separate git worktree* would test
  the **main** tree's source, not its own. Do code work in the main checkout, one branch at a time.
- **Tests:** `~/venvs/socr/bin/pytest <paths> -q`. Full suite is ~1070 tests.
- **Lint (BLOCKING CI gate):** run it the way CI does — `uvx ruff@0.16.0 format --check .`
  Only the **format** step blocks; `ruff check` runs with `continue-on-error`.
  **Do NOT use `~/venvs/socr/bin/ruff` for the format gate.** That venv has an older
  ruff which refuses to format Python code blocks inside Markdown ("experimental,
  enable preview mode"), so it reports clean on files CI rejects — this exact gap
  turned `main` red and blocked four PRs (#102, #107). Ruff is pinned exactly in
  `pyproject.toml`; bump it in a deliberate commit that fixes any new findings.
- **CI has NO ollama / no provider.** Any test that drives `_phase_agentic` / `process()` in
  agentic mode **must patch `_available_engines_for_agentic`** (e.g. return `[PROFILE_QWEN_LOCAL]`
  from `socr.core.providers`) or it passes locally (ollama installed) and **fails in CI** (the
  provider ladder is empty → the loop bails before routing). This trap bit PP-2/PP-7 — always
  confirm hermeticity, and **wait for CI green before merging.**

## Architecture (the agentic path is page-major)

- **Agentic mode is progressive / page-major** (`_phase_agentic` in `pipeline/orchestrator.py`):
  one loop over `sorted(state.pages)` does route → extract → tables → figures → equations → flush.
  Each finished page is written to `<doc_dir>/pages/NNN.md` + a `NNN.json` sidecar **immediately**.
- **`_phase_assemble` stitches the fragments** into the final `<stem>.md`. `_rewrite_all_fragments`
  (end of assemble) is the **sole authoritative fragment writer**; the in-loop flush is a
  `terminal:false` crash-recovery copy. The final `.md` is **byte-identical** to the whole-doc
  assembly — there are golden/byte-identity tests guarding this; do not break it.
- **Resume:** before processing a page, a per-page ledger gate (`_load_terminal_page`) skips it iff a
  terminal `NNN.json` + `NNN.md` exist with a matching `_run_fingerprint` + input checksum + SUCCESS.
  Conservative — reprocess on ANY doubt; never skip an unfinished page.
- **Cascade-halt:** a wedged local VLM → flush done pages, mark `PARTIAL_SAVE_VLM_TIMEOUT`, stop
  (don't fire the next page into a stuck GPU). The halt latch is checked at the top of the loop.
- **Tables** verified per-page (`_reread_page_tables`); **figures** embedded inline per page;
  **chart pages** routed to image assets (`has_chart_marks` / `_is_chart_asset_page`); **equations**
  behind `--detect-equations` / `--recover-clean-equations` (default off).
- Non-agentic paths (single-engine / multi-engine / consensus / repair) stay phase-major.

## Conventions / gotchas

- Local OCR model is **`qwen3-vl:30b-a3b-instruct`** (instruct, non-thinking). NEVER `qwen3-vl:30b`
  (thinking; runs away on dense tables) or `:8b` (collapses tables).
- **No magic thresholds** in code or prompts — derive from data or use a named, documented constant.
- **No silent content loss** (citation corpus): a wrong/dropped number is worse than a missing one.
  Failures must surface at *every* level (page status, document status, metadata, CLI) — not just one.
- Branch per change (`feat/NN-…` / `fix/NN-…`); stage by name (never `git add -A`); one commit per
  ticket; **wait for CI green before merging**.

## Plan / history

The progressive-pages initiative (the page-by-page rewrite) is recorded in
`docs/plans/progressive-pages/` (tickets PP-0…PP-7, design notes, the `/consilium` decisions).
Per-ticket decision logs are in `docs/log/`.
