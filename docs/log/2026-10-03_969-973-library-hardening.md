# 2026-10-03 library hardening (#969-#973)

Five follow-ups from PR #966, one branch `fix/969-973-library-hardening`.

- **#969** `library._stem` rejects empty, `/`, `\`, NUL, `.`, `..` and absolute stems with
  `LibraryError` before any path join; called from `check_promote` / `check_rerun`.
- **#970** `doc_status` counts a page as bad only when its `status` is a `str` in
  `{warning, error}`; list/object statuses no longer raise `TypeError`.
- **#971** New read-only `check_promote` / `check_rerun` hold the refusals of the live
  commands (promote/rerun now call them). `socr library --dry-run --promote/--rerun`
  calls them too. This needed a two-line edit in `src/socr/cli.py` (the dry-run branch), the
  only file outside library.py. The existing dry-run test promoted a stem with nothing
  staged and expected exit 0; it now stages `b` first (that was the footgun).
- **#972** `refresh_index` preflights all four index paths for symlinks before the first
  write, and `_read_entries` refuses a symlinked curated list. Absent file is still empty.
- **#973** `recover_promotion` roll-forward fsyncs `archive_dir` (when `archived` is set and
  the dir exists) before unlinking the journal.

Tests: `tests/test_gh964_library.py` (107 passed with the new ones; plus gh993: 139).
Mutants killed in an external copy (socr.__file__ canary checked, uncapped anchor
occurrences == 1): drop `_stem` call, drop the `isinstance` gate, drop the CLI
`check_promote` call, disable both symlink checks, disable the archive fsync. Each failed
exactly its own tests; baseline copy 107 passed.
Mutant for #972 needs both checks disabled: the read-side refusal alone also blocks the
partial write, the preflight is the belt for symlinked documents/missing_text/manifest.
