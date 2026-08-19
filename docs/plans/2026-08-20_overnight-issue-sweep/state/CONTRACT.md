# CONTRACT — every agent dispatched by the overnight sweep receives this verbatim

Plan: `docs/plans/2026-08-20_overnight-issue-sweep/`. Repo: socr (multi-engine
document OCR). You are running unattended. **Nobody will answer a question.** If
you are blocked, record the blocker in your output file and exit — never guess,
never invent a result.

## The seven hard facts

1. **Editable-install trap.** `import socr` resolves to the MAIN checkout's
   `src/socr`, so a git worktree does NOT isolate the code under test. Run
   everything as `PYTHONPATH=<your-worktree>/src ~/venvs/socr/bin/pytest …` and
   run `<plan>/bin/isolation_canary.sh` first — it exits 1 if `socr.__file__` is
   not inside your worktree. A test result obtained without this is void.
2. **The main checkout is owned by another session.** Never `cd` into it to
   write, never switch its branch, never commit there. Your worktree only.
3. **CI has no ollama and no provider.** Any test that drives `_phase_agentic`
   or `process()` in agentic mode MUST patch `_available_engines_for_agentic`
   (e.g. return `[PROFILE_QWEN_LOCAL]` from `socr.core.providers`) or it passes
   locally and fails in CI.
4. **Lint gate is exactly `uvx ruff@0.16.0 format --check .`** — not
   `~/venvs/socr/bin/ruff`, which is older and reports clean on files CI rejects.
   Only `format` blocks; `ruff check` is advisory.
5. **No `Co-Authored-By` trailer. No `git add -A`.** Stage files by name. Branch
   from the pinned baseline SHA (below), never from whatever is checked out.
6. **Forge is GitHub** (`git@github.com:r-uben/socr.git`). `gh` is correct here.
   Never `git push --force`. Never `gh pr merge`. PRs are proposed, not merged.
7. **Cardinal rule.** No silent content loss: a wrong or dropped number is worse
   than a missing one, and failures must surface at page, document, metadata and
   CLI level — not just one of them.

## The pinned baseline

Every citation, every branch, and every claim in this run refers to the single
commit recorded in `baseline.json` as `main_sha` = `53b0637b928c486e9ff3023fa9527905fec017b2`. `origin/main` may move while
you work; you do not follow it. Read historical file content with
`git show <main_sha>:<path>`.

## Evidence rules

- A citation is `{path, line, snippet}` and must resolve at `main_sha`, with
  `snippet` actually present on that line. A script checks this; a citation that
  fails the check voids the verdict that carried it.
- **A citation proves code exists. It never proves a bug is fixed.** To claim
  `ALREADY-FIXED` you must additionally supply `fixed_by_commit` (proved an
  ancestor of `main_sha`), the issue's acceptance criteria mapped one by one,
  and a reproducer or regression test that actually passes. Without all three,
  the verdict is `NEEDS-MEASUREMENT`.
- You never adjudicate your own verdict. You never review your own diff.

## Two failure modes you are being guarded against

- **Silent fabrication.** An agent denied a tool permission will produce
  confident invented output and exit 0. That is why evidence is machine-checked
  rather than trusted.
- **Confident wrong measurement.** Issue #249 took three owner revisions before
  its diagnosis held. PR #250's own fix reintroduced the bug it was fixing, and
  a test in that PR asserted the defective behaviour and passed. Assume your
  first reading is wrong until something mechanical agrees with it.

## Abort latch

`<plan>/state/ABORT` — if this file exists, stop all tracker writes and all
pushes immediately, finish your current write to disk, and exit. Check it before
every mutating action. `gh auth status` failing creates it.

---

## Substituted at A1 (2026-08-20)

- `main_sha` = `53b0637b928c486e9ff3023fa9527905fec017b2` (`53b0637`, "fix(glyph): gate on the Monotype H<number> namespace, not table membership")
- base worktree (read-only reference at that SHA): `/Users/rubenffuertes/repos/.worktrees/socr-night-base`
- canary: `docs/plans/2026-08-20_overnight-issue-sweep/bin/isolation_canary.sh`
- `gh` authenticated as `r-uben`; abort latch NOT set at A1.

### Correction to fact 1, proved in `logs/2026-08-20_A1-sentinel-transcript.md`

`pyproject.toml` sets `[tool.pytest.ini_options] pythonpath = ["src"]`, resolved relative
to **rootdir**. So a pytest run whose targets are inside your worktree is ALREADY
isolated, even with `PYTHONPATH` unset. The trap is narrower than stated but real: the
`socr` console script, `python -c` probes, reproducer scripts and any pytest run rooted
outside the worktree all silently execute MAIN CHECKOUT code. Since those are exactly
the tools you reach for when building a reproducer, the mandate is unchanged — export
`PYTHONPATH=<worktree>/src` and pass the canary before any measurement.
