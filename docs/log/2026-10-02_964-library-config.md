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
