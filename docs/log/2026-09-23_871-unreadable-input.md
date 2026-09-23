# GH-871 — refuse and record a PDF none of whose pages load (2026-09-23)

Branch `fix/871-unloadable-page-refusal`.

## Measured before the fix (current main, real damaged file)

A paper whose page tree MuPDF rejects (`format error: non-page object in page tree`):

| open path | page_count | loadable |
|---|---|---|
| `open_pdf(p, repair=False)` | 64 | 0 (first `FzErrorFormat`, then `IndexError: page 63 not in document`) |
| `open_pdf(p, repair=True)` (default) | **0** | 0 — no exception |

`repair` here is symbol-font glyph recovery, not structure repair; touching the pages
lets MuPDF run its own xref repair, which collapses the count. `socr process` on the file:
exit 1, a raw traceback out of `_phase_analyze` → `BornDigitalDetector.detect` →
`doc[page_idx]`, and nothing written — no `metadata.json`, no root-index entry. A batch
already survived it (`process_batch` catches per file) but still recorded nothing.

## What

- `FailureMode.UNREADABLE_INPUT`. No existing member fit: `EMPTY_OUTPUT` / `CLI_ERROR`
  describe what an engine produced, and here no engine ran. Only one test enumerates the
  enum, for uniqueness; nothing maps on it exhaustively.
- `socr.core.pdf.probe_page_loads` — opens with plain `fitz.open`, loads every page, reports
  declared vs loadable. Deliberately bypasses glyph recovery, which would erase the evidence.
- `UnifiedPipeline._refuse_unreadable_input`, called in `process()` before the document
  handle is built. Zero loadable pages → a FAILED record through the ordinary
  `_write_metadata` path (per-doc `metadata.json` AND the root index), an ERROR result
  naming the cause, and a one-line console message. The handle gets the declared count
  explicitly, because counting it through `open_pdf` would record the 64-page file as 0.

After the fix, the same file: exit 1, one line (`unreadable_input: 0 of 64 declared page(s)
could be loaded (FzErrorFormat: code=7: malformed page tree)`), no traceback, per-doc and
root records both `failed` with that cause.

## Design input

Fable (read-only consult) recommended the document-level probe plus a per-page catch, a new
failure mode, and recording through `_write_metadata`; its code references checked out.
Its resume point held: a FAILED entry is refused by the gate, so the file is retried.

## Tests

`tests/test_gh871_unreadable_input.py`, 9 tests. The real file is copyrighted, and MuPDF
repairs simple hand-authored page-tree damage (differently across versions), so page loads
are made to raise the exact exception the real file produced. The resume test carries a
control — the same record flipped to `completed` with its markdown present must be skipped —
so "not skipped" cannot pass vacuously. Mutations seen to fail, out-of-repo copy with the
import canary: removing the call in `process()` (1 failure), and making `unreadable` always
false (6 failures).

`tests/test_orchestrator.py::TestFullPipeline` mocks the document layer with a path that
does not exist; it now tells the probe the input is readable, since those tests are about
the agentic loop, not the file on disk.

## Out of scope

- **Partially damaged PDFs** (some pages load, some raise) are not handled here — the
  refusal fires only at zero loadable pages. A per-page catch in `BornDigitalDetector.detect`
  plus an explicit per-page status is the follow-up; it touches page-status plumbing and
  the fragment/marker balance checks, so it gets its own ticket.
- `DocumentHandle._count_pages` still counts through `open_pdf(repair=True)` for every
  other document.
