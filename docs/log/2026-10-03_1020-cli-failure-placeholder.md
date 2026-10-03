# 2026-10-03 GH-1020: a CLI failure placeholder is CLI_ERROR, not SUCCESS

## Bug

`BaseEngine.process_pages` (`src/socr/engines/base.py`) treated any non-empty page text as
SUCCESS with `audit_passed=True`. qwen-ocr-cli, on a backend error (Ollama 500), exits 1
and writes `*[OCR failed for page N]*` into the page slot (`qwen_ocr/processor.py::_ocr_pages`,
reason in `DocResult.page_errors`). socr read the marker as text, judged it, and could ship it.

## Engine survey (sibling repos read only)

- qwen-ocr-cli: `*[OCR failed for page {idx}]*` per page (idx = position in the image dir);
  `*[OCR Failed]*` for a document with no pages (`processor.py` `_write_document`).
- mistral-ocr-cli: `*[OCR Failed]*` for a document with no pages (`processor.py`).
- gemini, marker, glm, deepseek, nougat: no sibling source on this machine writes a
  placeholder; they skip the page file and exit non-zero, already CLI_ERROR/EMPTY_OUTPUT.
  Not verified beyond socr's own wrappers.

## Change

- `CLI_FAILURE_PLACEHOLDERS` + `is_cli_failure_placeholder()` in `engines/base.py`: two exact
  regexes with sources cited, matched against the WHOLE stripped text (never a substring, so
  a page quoting the marker is untouched).
- `process_pages`: a placeholder page (checked raw and after `_clean_output`, which covers the
  aggregated qwen read-back) becomes `ERROR` / `CLI_ERROR` / `audit_passed=False`, carrying the
  CLI exit note. Applies whatever the exit code.
- `process_document`: an exit-0 whole-document placeholder becomes an `ERROR` / `CLI_ERROR` result.

The structured signal (`page_errors` in the CLI's metadata.json) was not used: it is keyed by the
CLI's own image-dir index and is only written for qwen; the marker is the one signal common to
both CLIs that emit one. The placeholder-to-ERROR mapping reuses the existing CLI_ERROR path, so
page, document, metadata and CLI surfacing are the existing ones.

"No judge call": the shipped judges refuse a non-SUCCESS/empty output without a model call
(`agentic.py` "empty/error output"), and `route_page` then escalates to the next rung.

## Tests

`tests/test_gh1020_cli_failure_placeholder.py` (12): marker recognition and non-recognition
(substring, embedded, near-miss); real `process_pages` with a faked subprocess, success vs
placeholder+exit 1 vs placeholder+exit 0; agentic loop run twice, differing only in what rung one
wrote: the judge saw the real text in one leg and never the marker in the other, the next rung ran
only in the failure leg, and the marker is absent from written markdown. Hermetic (ladder patched,
judge model "", `get_engine` stubbed, subprocess faked); no absolute status pinned.

Mutant (external copy of src+tests, `socr.__file__` canary, anchor count 1): neutering the matcher
(`return False and any(...)`) fails 6 of the 12 tests.

## Follow-up (not done, other repo)

qwen-ocr-cli `backends/base.py`: include `resp.text[:300]` in the 5xx error and retry once.
