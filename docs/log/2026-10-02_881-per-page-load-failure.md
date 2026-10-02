# GH-881 -- one unloadable page costs that page, not the document (2026-10-02)

Branch `fix/881-per-page-load-failure`, from `origin/main` d8dc9b13. Follow-up named in
`docs/log/2026-09-23_871-unreadable-input.md` ("Out of scope: partially damaged PDFs").
No prior or withdrawn attempt exists (`grep` of `docs/log` for 871/881: only the #871 log,
which defers this).

## Before

`BornDigitalDetector.detect` did `doc[page_idx]` for every page unguarded. A partially
damaged PDF (some pages load, some raise MuPDF `FzErrorFormat`) died with a raw traceback
out of `_phase_analyze`: nothing written, every readable page lost with the bad one. The
#871 probe only refuses when ZERO pages load. Corpus rate measured by the issue author
(2026-09-23): 0 of 400 partially damaged, so there is no real fixture.

## Design

A page that fails to load becomes a placeholder, is carried through state as data, and is
failed at the top of the page loop. Nothing downstream is asked to cope with it.

- `PageAssessment.load_error` / `PageState.load_error` (empty for every loadable page).
  `detect()` guards ONLY `doc[page_idx]`; an error inside `_assess_page` stays loud.
- `_phase_agentic`: `unloadable_pages` is held out of the corrupt-math, equation-region,
  OCR, chart-scan sets. At the top of the loop (after the halt guard, before the resume
  gate) `_fail_unloadable_page` records an audit event and flushes fragment + sidecar.
  The page is never routed, rendered, planned or judged, so none of the ~15 later
  `doc[page_num - 1]` sites is reachable for it (each is called with a page the loop is
  processing; checked by reading every site's caller).
- Winner selection: the existing no-text ending (`NO_TEXT_MARKER`) already ships
  `[page N failed: no usable OCR output]` with status ERROR. It now carries
  `failure_mode=UNREADABLE_INPUT` and the load error text when `load_error` is set.
  Reused the #871 mode: "no engine ran, the input could not be read" fits a single page
  exactly as it fits a whole document. No new enum member.
- `_phase_analyze` native-words cache skips unloadable pages. Before, the one raise was
  swallowed by the surrounding `try` and every LATER page lost its cached words, logged only.
- `_document_handle_for`: `DocumentHandle` counts via `open_pdf(repair=True)`, which on a
  damaged tree can report fewer pages than declared (measured 64 -> 0 in #871). For a
  partial document that would drop the bad pages from `state.pages`, so they could never be
  recorded. When `0 < loadable < declared` the declared count is used; an undamaged
  document takes the unchanged `from_path` route.

## Surfacing

| level | where |
|---|---|
| page sidecar | `pages/NNNNN.json`: `status=error`, `failure_mode=unreadable_input`, error text |
| page fragment / final .md | `[page N failed: ...]` marker, page header kept (header count == page count) |
| audit log | `page_unloadable` (at the loop) and `page_failed` with `unreadable_input` detail |
| document status | existing `failed_pages` path: PARTIAL / AUDIT_FAILED, never SUCCESS |
| metadata.json | `status: partial`, `error` names `unreadable_input: page(s) N could not be loaded` |
| CLI | extra line `unreadable_input: K page(s) could not be loaded from the PDF: [N]` |

## Resume

A FAILED page is ERROR, and `_load_terminal_page` accepts only SUCCESS, so it is never
terminal-skipped (pinned with a control: a good page of the same run IS restored). One
consequence worth stating: the DOCUMENT gate (`_resume_skippable`) skips a PARTIAL document
whose checksum and fingerprint match. Damage is deterministic for identical bytes, so a
re-run could not improve it anyway; a repaired file has a new checksum, and `--reprocess`
forces the re-read (tested).

## Tests

`tests/test_gh881_per_page_load_failure.py`, 9 tests, hermetic (ladder, judge model, crop
model, engine runner and primary/local/enabled engines pinned). `fitz.Document.load_page` /
`__getitem__` raise `FzErrorFormat` for one index on a real 4-page PDF. Pins are
differences: the same document damaged vs undamaged. Undamaged pages' fragments are
byte-identical between the two runs; the undamaged control has no `unreadable_input` and no
failure marker anywhere.

Mutations (external copy of src + tests + pyproject, `socr.__file__` canary asserted per
copy, uncapped anchor count == 1, baseline 9 passed):

| mutant | result |
|---|---|
| catch in `detect` removed | 6 failed |
| loop-top skip removed | 4 failed |
| `failure_mode` dropped in manifest | 1 failed |
| declared page count dropped | 1 failed |
| native-words filter removed | 1 failed |
| document `error` stops naming the cause | 1 failed |
| bad page not excluded from `ocr_pages` | **survives** (9 passed) |

The surviving mutant is equivalent: the loop-top skip handles the page before the OCR list
is consulted. The exclusion is kept only so a document whose sole OCR-bound page is the bad
one does not print "No OCR providers available". It is not claimed as guarded.

## Not covered

- No real partially damaged file; the fixture simulates the exact MuPDF exception.
- `detect_page` (single-page assessment) still raises on a bad page: it is asked for that
  page by number and has no document to survive.
- Errors from `get_text` / `find_tables` AFTER a successful load are not caught; a page
  that loads but cannot be read is a different failure and stays loud.

## Byte identity and suite

Undamaged 4-page fixture run under origin/main's `src` and under this branch's `src`
(separate copies, `socr.__file__` printed for each): final `.md` sha256
`347cb5f3...e9b83b` and all four page fragments identical.

Full suite, default `OLLAMA_HOST`, nohup, one complete run: 6103 passed, 2 skipped,
4 xfailed (1157 s). `uvx ruff@0.16.0 format --check .` clean (807 files).

## Review round 1 (Astra, PR #947)

- P1a: `FigureExtractor.extract` loaded the page before the `skip_pages` check, and its
  document-wide catch ended the loop, so every later page lost its figures with
  `figure_phase_failed` unset. The skip now precedes the load and the load is guarded per
  page. Test: bad middle page, figure on the last page, damaged vs undamaged runs agree.
- P1b: glyph recovery runs before `range(len(doc))` and can shrink the count. `detect` now
  fixes the declared count first and records every page missing after repair as
  `unreadable_input` ("missing after repair"). Test shrinks the doc inside a patched recovery.
- New tests: empty provider ladder vs full ladder (difference pin on bad-page status, mode and
  document status `partial`); committed sha256 of the undamaged fixture's final markdown
  (measured identical under origin/main and the change), replacing the "no markers" proxy.
- New mutants, all killed: extractor load before skip (1), extractor load unguarded (1),
  declared count read after repair (1). Earlier mutants re-run: all still killed except the
  documented equivalent one.

## Review round 2 (Astra, 9c3f6b2)

- Skip test records `__getitem__`/`load_page` requests: the skipped page is never requested
  (control: unskipped, it is). Mutant "guarded load before the skip" killed.
- Shrink test now removes the last AND a middle page. The middle case exposed a real defect in
  my round-1 fix: indexing after the shrink read page 3's text under page 2's number. `detect`
  now records each declared page's xref before recovery and looks pages up by it afterwards
  (only when the count changed), so survivors keep their identity and exactly the vanished
  page is FAILED. Mutant "index instead of identity" killed by the middle case.
- Golden sha256 replaced by a same-process pin: the undamaged fixture through `detect` and
  through a copy of the pre-#881 `detect`; final .md and every fragment byte-equal. Mutant
  "catch fires on a healthy page" killed (7 tests). A first mutant (`is_born_digital=False`)
  survived because the fixture's pages are all OCR-routed: noted, not used.
