# GH-964: `socr library`

## What

`socr library [--config PATH] [--dry-run] [--rerun STEM | --promote STEM] [--primary] [--profile]`.
Logic in `src/socr/library.py`; thin command in `src/socr/cli.py`; tests in
`tests/test_gh964_library.py` (43). README section added.

## Design

- Loader validates every key listed in the issue; missing key, absolute or `..` path outside
  `root`, or a symlink leading out of `root`, raises `LibraryConfigError`. Staging/pdf/text/archive
  may not coincide, and staging may not sit inside pdf/text/archive.
- `output.document.{markdown,figures,metadata}` are authoritative but the pipeline writes fixed
  names (`{stem}.md`, `figures`, `metadata.json`, from `ocr_output_contract`). A config that
  disagrees is rejected at load instead of being silently ignored by the writer.
- Processing calls `UnifiedPipeline.process(pdf, out_root, scan_root=pdf.parent)`, the per-file
  call `batch` makes, with `out_root = output.text` so the pipeline's own `<root>/<stem>/<stem>.md`
  layout lands in the library layout. (`process_batch` was not used: it treats the output root as a
  mirror of the input subtree and would write its resume index per batch.) The pipeline also
  writes its root `metadata.json` index into `output.text` / the staging dir; it is a file, so it
  is not mistaken for a stem.
- PDFs: top-level `*.pdf` under `input.pdf` (the text layout is flat by stem).
- Never overwrite: work list excludes stems with any text dir; `process_new` re-checks
  immediately before each write. A text dir that exists without its markdown is "blocked":
  reported and skipped, handled via `--rerun`.
- `--rerun`: staging is the optional top-level `staging` key, default `<index.dir>/staging`.
  Runs with `reprocess=True`. Refuses if the stem is already staged. "Awaiting approval" is
  derived from disk (staged dir with markdown) and recorded as `awaiting_approval` in the manifest.
- `--promote`: `os.rename` old text dir to `<archive.dir>/<stem>.<YYYY-MM-DD>[.N]`, then rename
  staged into place; rolls the archive step back if the install fails. Refuses if nothing is staged.
- Index (atomic temp+`os.replace`): `documents.txt` = absolute PDF paths (my reading of
  "document"; the config comment says one absolute path per document), `missing_text.txt`,
  `unverified.txt`, `manifest.json` (`{documents: {stem: {status, verified, bad_pages,
  awaiting_approval}}}`). Unverified = metadata status != `completed`, or any page sidecar status
  outside {`success`, `skipped`} (so warning, error, unreadable, or missing metadata all count).
  Refreshed in a `finally`, so a partial/failed run still reports what is on disk.
- `backup.rclone_remote` is only echoed in the summary.

## Deviations / follow-ups

- The config comment says each unverified document "also carries UNVERIFIED.txt in its text dir".
  Not done: that would write into existing text dirs, which contradicts never-overwrite. Needs a
  decision if wanted.
- Pipeline settings are limited to `--primary` and `--profile`; no other batch options.
- `--dry-run` combined with `--rerun`/`--promote` prints the intent only.

## Verification

- 43 new tests, hermetic (`tmp_path` libraries, engine pinned, `_available_engines_for_agentic`,
  `_resolve_judge_model`, `_resolve_crop_vlm_model`, `_run_engine_on_pages` patched; real
  `~/papers` never touched). A canary test asserts `socr.__file__` is inside the tree under test.
- Mutations in an external copy of `src` + tests (canary passes there), each must fail the suite:
  never-overwrite guard in `process_new` (1 failed), work-list filter (1), archive rename ->
  `rmtree` (3), promote refusal (2), symlink/escape check (1; the lexical check alone is
  redundant with the resolved check, mutating only it survives by design), non-atomic index
  write (1). Unmutated: 43 passed.
- Full suite and format result: see the commit report.

## Amendment: three states, curated unverified list (coordinator review of the real library)

The real library has 342/362 legacy `metadata.json` with no `status`, a curated `unverified.txt`,
and 19 hand-placed `UNVERIFIED.txt`. The first version (non-completed => unverified) would have
listed ~355 papers. Now:
- state is `verified` / `unverified` / `unknown`. Unverified only on explicit evidence: status
  present and not `completed`, a page `warning`/`error`, or an `UNVERIFIED.txt` marker. Missing
  status => `unknown` (recorded in the manifest, never in `unverified.txt`). Pages with unreadable
  or absent status no longer count.
- `unverified.txt` = existing entries union computed ones; an entry leaves only for a stem socr
  processed this run (new PDF or promote; not rerun, which leaves text untouched) that is `verified`.
- markers are never written or deleted.
- tests: legacy fixture (no status + curated entry + marker), clear-only-when-processed-clean,
  processed-but-still-partial. Mutants in an external copy (all fail the suite): drop the union
  (2 failed), clear every curated entry (2), legacy->unverified (2), ignore marker (1).

## Amendment 2: data-safety review (PR #966)

- Lock: `flock` (LOCK_EX|LOCK_NB) on `<index.dir>/.library.lock` for the whole non-dry run; released by the OS on a crash, so no stale-lock cleanup. Dry-run takes no lock and writes nothing.
- Index writes: `mkstemp` in the same dir, fsync, `os.replace`; refuse a symlinked target or temp.
- `unverified.txt` unreadable (I/O or non-UTF-8) raises before any index file is written. Absent is empty.
- Config: index file names (plus the lock and journal names) distinct case-insensitively. pdf/text/index/archive/staging pairwise equal-or-nested is rejected after `resolve()` (symlink aliases count). This forced the default staging out of `index.dir` (nesting is now rejected): default is `<root>/.socr-staging` (named constant `DEFAULT_STAGING_NAME`), validated like every other dir.
- New papers: process into staging, then `install_staged` (no-replace rename, requires `{stem}.md`). Exception, missing markdown or existing leftovers => FAILED/BLOCKED, text/ untouched, exit 1. A completed run with status partial/failed IS installed and shows up in unverified.txt via its metadata status (a run whose metadata carries no status would be `unknown`; the pipeline always writes one).
- Promotion: journal written before the two renames, removed after; `recover_promotion` runs first under the lock and rolls forward (never leaves text absent); unreadable or inconsistent journal aborts. Archive name is chosen under the lock.
- Stem collisions: `check_stem_collisions` (casefold) refuses before any work, including dry-run.
- Tests: 72 in `tests/test_gh964_library.py`. Mutants in an external copy, each fails the suite: lock, curated-read abort, symlink checks (individually redundant, both removed: 1 failed), unique temp, distinct names, case-insensitive names, dir overlap, resolve-before-compare, staging-first, install requires markdown, journal recovery, journal written, no-replace rename, collision preflight, never-overwrite, archive rmtree, promote refusal, curated union.

## Amendment 3: atomic rename, journal validation, durability (PR #966 round 3)

- `_rename_noreplace` now calls the kernel primitive via ctypes: macOS `renamex_np(RENAME_EXCL)`, Linux `renameat2(RENAME_NOREPLACE)`. EEXIST/ENOTEMPTY becomes "refusing to replace". Fallback to check-then-rename ONLY when neither symbol exists or the filesystem returns ENOTSUP/EINVAL/ENOSYS; in that case it is safe only against other socr runs (the lock), not other programs. Used for install, archive and recovery (all via `_rename_durable`).
- `recover_promotion` validates the journal before acting: target/staged/archived must be directly inside the CURRENT text/staging/archive dirs after `resolve()`, none may be a symlink, names must match the stem, and a staged dir must hold `{stem}.md`. Otherwise it raises naming the journal and touches nothing (journal kept for a human).
- Durability: `_atomic_write` fsyncs the file, renames, then fsyncs the directory (so the journal and its dir entry are durable before the first rename). `_rename_durable` fsyncs the parent dir(s) after each rename. The journal is unlinked only after both renames are durable, then its dir is fsynced. Helper `_fsync_dir`.
- Documentation: the library must be on a local filesystem; locks and atomic renames are not guaranteed on iCloud or network mounts (`~/papers` is local by policy).
- Tests: 81. Mutants in an external copy (all fail the suite): native primitive disabled, plain os.rename in recovery / install / archive, journal containment, journal symlink check (needed an in-library symlink test; the first test matched its own tmp path name), staged-markdown check, missing fsyncs (journal dir, after renames, after journal delete), journal deleted before the second rename.
