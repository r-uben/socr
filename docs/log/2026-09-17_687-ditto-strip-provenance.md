# 2026-09-17 — #687 the ditto scan must not see fenced or commented tables

Branch `fix/687-ditto-strip-provenance` off `main@6fb8e67`. Worktree `/tmp/wt-687`.

## The defect

`detect_ditto_columns` (`socr/tables/ditto.py:61`) scanned RAW shipped text with a bare
`find_table_blocks(markdown)`. Every other raw-table shipping predicate in `reconcile.py`
strips provenance first via `_markdown_content_lines`, precisely so a code sample or a
commented-out grid cannot manufacture a page-level artifact. Ditto detection did not, so a
ditto mark inside a fenced code block or an HTML comment produced the #625 distrust signal
(`table_ditto_columns` populated, SUCCESS demoted to WARNING, `DITTO_UNRESOLVED_KIND` into
`tables_trust.json` / the audit log / the CLI / the metadata note) on a page whose only real
content is clean. Opposite direction from most defects here: a false positive, not a loss.

## The fix

`detect_ditto_columns` now runs `find_table_blocks` against
`"\n".join(_markdown_content_lines(markdown))` instead of the raw text. Line positions are
preserved (both stripping passes blank matched regions without changing line count), and
`detect_ditto_columns` never uses positions — only the parsed grid — so the join is safe.

## Which stripping contract: `_markdown_content_lines` alone, NOT
## `_strip_emission_literal_blocks`

Four call sites in `reconcile.py` compose provenance stripping; three (`:350`, `:414`,
`:475` — `table_emission_defect`, `table_content_defect`, `raw_table_block_lines`) also run
`_strip_emission_literal_blocks`, one (`_has_table_grid`, `:580`, backing
`has_strict_table_grid` / `has_authored_table_grid` / `markdown_table_identity`) does not.

The three that add the literal-block strip all read RAW ROWS looking for a formatting
DEFECT in the shipped text itself (a LaTeX leak, a width mismatch, an empty body) — and the
comment above `_RAW_LITERAL_BLOCK_OPEN` names it explicitly: "Keep this emission-only:
`_markdown_content_lines` is also the provenance policy for GH-268's grid-selection
decisions" — i.e. `_strip_emission_literal_blocks` is scoped to that one family of raw-row
emission checks, not to the grid-existence family.

`detect_ditto_columns` does not read raw rows for a defect; it calls `find_table_blocks` to
get a parsed grid and asks whether a cell in that grid is a ditto mark — structurally the
same operation `_has_table_grid` performs (also via a `find_table_blocks`-shaped scan) to
answer "does a real GFM table exist here". `_has_table_grid`'s own docstring enumerates the
three provenance failures it defends against — a grid inside a code fence, inside an HTML
comment, inside indented code — and does not mention raw HTML literal blocks
(`<pre>`/`<script>`/`<style>`/`<textarea>`); none of the grid-existence predicates strip
those. Ditto detection follows that contract rather than the emission-defect one: it is
answering the same kind of question, with the same evidence source
(`find_table_blocks`-parsed grid, not raw rows), so it inherits the same provenance floor.

## Evidence, both directions

`tests/test_gh625_ditto_unresolved.py`, new section 7 (5 tests):
- a ditto table wrapped in a fence detects zero columns
- a ditto table wrapped in an HTML comment detects zero columns
- a REAL ditto table sitting beside a fenced one is still detected (the strip does not
  swallow genuine content merely because the page also carries a code sample)
- `_apply_ditto_guard` on a fenced-table page: `guarded is out` (identity, same as any clean
  page), status stays SUCCESS, `text` byte-identical, `table_ditto_columns == []` — the
  issue's specific ask, pinned on the demotion/audit-event absence, not just the count
- same for an HTML-commented page

All 22 tests in the file pass (17 pre-existing + 5 new).

## Full suite, measured

Baseline: `git archive HEAD` (pre-fix, `main@6fb8e67`) into `/tmp/gh687-baseline`, canary
(`os.path.realpath(socr.__file__)` under the archived `/private/tmp/...` path) confirmed
before running. `PYTHONPATH=.../src ~/venvs/socr/bin/pytest tests -q -p no:cacheprovider`:
**5650 passed, 1 skipped, 4 xfailed** (the 1 skip is `test_gh592_scoped_positional_emission.py`,
a known archive artifact per the repo's own note, not a regression).

After (this worktree, fix + new tests applied): **5656 passed, 4 xfailed**, 0 skipped.
Reconciles exactly: `5650 + 1 (skip resolves to a real pass outside the archive) + 5 (new
GH-687 tests) = 5656`. No unexplained delta.

Scoped table/reconcile suite (`tests/tables/` + 30 named table-owning files including
`test_gh625_ditto_unresolved.py`): 1014 passed, 1 xfailed — unchanged shape from before the
change (the 1 xfailed is a pre-existing, unrelated marker).

`uvx ruff@0.16.0 format --check .` — clean, whole repo.

## Mutation proof

Copied `src`, `tests`, `pyproject.toml` to `/tmp/gh687-mutant` (outside the repo). Canary
(`os.path.realpath(socr.__file__)` starts with the mutant path) passed before mutating.
Asserted the uncapped anchor (`stripped = "\n".join(_markdown_content_lines(markdown))` ...
`find_table_blocks(stripped)`) occurs exactly once in `ditto.py` before editing, and that the
post-replace source differs from the pre-replace source (mutation actually applied) — both
checked in the same script that performs the edit, so a failed match raises instead of
silently no-opping.

Reverted to the bare `find_table_blocks(markdown)` (pre-fix behaviour) and ran
`tests/test_gh625_ditto_unresolved.py` against the mutant:

```
FAILED tests/test_gh625_ditto_unresolved.py::test_fenced_ditto_table_is_not_a_page_reading
FAILED tests/test_gh625_ditto_unresolved.py::test_html_commented_ditto_table_is_not_a_page_reading
FAILED tests/test_gh625_ditto_unresolved.py::test_a_real_shipped_ditto_table_is_still_detected_alongside_a_fenced_one
FAILED tests/test_gh625_ditto_unresolved.py::test_guard_leaves_a_fenced_ditto_table_page_success_and_byte_identical
FAILED tests/test_gh625_ditto_unresolved.py::test_guard_leaves_a_commented_ditto_table_page_success_and_byte_identical
5 failed, 17 passed, 5 warnings
```

The 5 new GH-687 tests all redden under the reverted mutation; the 17 pre-existing tests
(real-table detection, guard behaviour, trust/resume plumbing) stay green — the guard fails
exactly where it should and nowhere else.

## Framing check

The orchestrator's framing (bare scan on raw text, false positive on fenced/commented
tables, fix = compose `_markdown_content_lines` same as the sibling predicates) measured
correctly; nothing to correct. The one open question the ticket posed — which stripping
contract — resolved to `_markdown_content_lines` alone, for the reason above (ditto reads a
parsed grid like the grid-existence family, not raw rows like the emission-defect family).

## Files changed

- `src/socr/tables/ditto.py` — strip provenance before `find_table_blocks`; docstring
  records the contract decision.
- `tests/test_gh625_ditto_unresolved.py` — 5 new tests (section 7).
- `docs/log/2026-09-17_687-ditto-strip-provenance.md` — this file.
