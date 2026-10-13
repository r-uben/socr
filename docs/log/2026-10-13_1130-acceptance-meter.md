# 2026-10-13 #1130: the corpus acceptance meter

Code: `src/socr/benchmark/acceptance.py`, entry point `socr-acceptance`, tests
`tests/test_acceptance_meter.py`. Spec: the thinkers' consolidated meter spec (not committed),
including "Corrections after round 3".

## Rules as built

- Denominator: `corpus/manifest.csv` `n_pages`. Document dirs are found by a
  `metadata.json` carrying `key` (the batch dir `out/wN/metadata.json` has `files`, no `key`).
- Accounted: a terminal `pages/NNNNN.json` + `.md`. Missing, extra, non-terminal, duplicate
  document, unlisted document: `unaccounted`. Failed doc with `unreadable_input`: MARKED only
  if its document `.md` has a mark, else `document_failed_unmarked`.
- SUCCESS pages: `success_common` (top level AND winning_output: status, failure_mode,
  audit_passed, empty rejection_class) plus one function per lane in `LANE_RULES`.
  `success_audit_false` is the audit_passed==false case; any other non-passing common field is
  `success_unchecked`.
- WARNING/ERROR: MARKED iff a recognised mark is in the page md and the same line in the
  document section; else `warning_unmarked` (split by failure mode) / `error_unmarked`.
- Marks: visible lines only (HTML comments and fenced code stripped). `MARK_FAMILIES`.
- Surface: disposition vs manifest entry, sidecar failure_mode vs `journal[-1].failure_mode`,
  page md vs document slice (assembler regex `PAGE_MARKER_RE`, trailing whitespace per line
  ignored), metadata per-status counts when present. `metadata_counts_missing` is
  document-level, outside METER and exit status.
- METER = size of the union of violating page ids.

## Spec ambiguities, my choice

- Stricter: a SUCCESS page whose lane has no rule, or `native+math`, is `success_unchecked`.
- `[socr: ...]` notes other than the blank receipt (fabricated-image receipt) are NOT counted
  as marks (they report what socr removed, not an unrun check). Not stricter; no such SUCCESS
  page needed a decision on this run but the choice is open.
- Evidence events for native coverage, figure text and chart values do not exist; names are
  provisional constants (`EVIDENCE_*`), `data.result == "pass"`. No such event => violation.
- `table_only_acceptance` requires `audit_passed == true` (spec text); without it 9 pages,
  with it 3. The other 6 are already `success_audit_false`.
- `equation_unverified_success`: any `equation_region_reading_attached` with
  `semantic_fidelity_verified != true`.

## Mark emitters found (src/)

`page_failed_marker` (no usable OCR output), D3/TR-3 floors `[page N failed: unverifiable table`
and its siblings, rotated-shredded, invisible-OCR-layer-unread, `invalid table emission` (manifest.py),
`UNVERIFIED_TABLE_MARK` (`> [table unverified:`), `disputed_table_mark`, `page_unchecked_mark`,
`kept_table_flag_note` (`[page N: flagged table kept`), the "unverifiable table — kept a model
reading" note (manifest.py ~704, via `d3_superseded_note`), `SCANNED_PROSE_RECOVERED_FLAG` and
`SCANNED_NATIVE_TEXT_FLAG`, `judge_timeout_note` (`[socr: page N: ...]`), `blank_page_marker`
(receipt), `FABRICATED_IMAGE_MARKER` (receipt). Planned and recognised: the #1130 canonical
grammar and `[page N: uncovered text — see image]` (#1050). Caveat: emitters are many; I
matched the ones above by reading `grep` output, not by running every code path.

## Measurement (CPU, run 23117187 and the previous run)

Run 23117187: METER = 12,165 of 17,155 (spec: 12,165; Fable 12,098 + 64 + 3). CHECKED 2,930
qwen + 8 blank; MARKED 1,605 error + 447 warning. native_unchecked 7,571, chart 919,
success_audit_false 385, equation_unverified 400, warning_unmarked 3,221 (322 none),
error_unmarked 2, document_failed_unmarked 64, table_only 3, success_unchecked 67 (all are
also audit_false). Every class anchor matches. Differences from Fable/Astra: none over 1%.

Run main (previous run): METER = 12,146. native 7,571, chart 1,305, audit_false 353,
warning_unmarked 2,848, error_unmarked 2, document failed 64, table_only 3, equation 0.
CHECKED 2,884 qwen (0 blank: no blank lane then).

Surface: 3,102 (23117187) / 3,000 (main) `failure_mode_mismatch`, all but one because
`journal[-1].failure_mode` is `none` while the sidecar carries a post-selection mode
(garbled_math, ladder_exhausted, ...). The journal records engine attempts, so this is a
real gap between surfaces, but the spec's comparison cannot pass until the journal (or the
verdict record) carries the final mode. 0 disposition and 0 page/document mismatches.
`metadata_counts_missing`: 394 documents (the failed one has no pages).

## Mutants (external copies of src, tests and pyproject; not committed)

(a) HTML comments counted as marks: fails the multi-line-comment case only. The one-line
comment case passes under the mutant because `<!--` prefixes the line and no family matches
it; keep the multi-line case. (b) native success without witness treated as CHECKED: 3 tests
fail. (c) document-slice comparison skipped: 2 tests fail. Unmutated copy: 32 pass.

## Not measured

Per-page verdict correctness (the meter reads sidecar fields, not shipped bytes against
source); marks in document .md for pages with no sidecar; nothing run against a run that has
the new witnesses; the `uncovered text` and canonical-grammar families are only unit-tested.

## Round 2 (Astra's review, 2026-10-13)

Seven items, each with a test and an external mutant (12 mutants in external copies,
canary `socr.__file__` inside the copy, anchor `count == 1`; every one killed, control 50 pass).

1. P0 shipped bytes: CHECKED needs `winning_output.text` == page `.md` with generated mark
   lines removed (trailing whitespace per line ignored). Class `shipped_bytes_differ`. A missing
   `winning_output.text` is a violation. Only SUCCESS pages carry it (WARNING/ERROR pages are
   judged by marks, not by a check on bytes).
2. P0 hidden marks: `scan_lines` keeps comment, fence and bracket state across lines. Unclosed
   comment hides the rest; a fence closes only with the same char, >= length, nothing after;
   an unclosed `[` / `![` hides lines until a blank line. The document is scanned once so state
   opened before a `## Page N` carries into the section. Doubt means NOT marked.
3. P0 identity: input rows keyed by `out_file` path; two rows with one stem are
   `duplicate_input_stem` (never merged, never matched to an output). Only canonical
   `NNNNN.json|md` page files are read; anything else in `pages/` is `noncanonical_page_file`.
4. P0 `unlisted_document` (one id `stem:*`, so a document with no pages still counts) and
   `nonterminal_page`.
5. P0 a failed document's pages are MARKED only by a document-wide mark. No emitter exists;
   provisional grammar `[document failed: <reason>]` (`DOCUMENT_FAILED_MARK_RE`). The old test
   that let a single page mark excuse the document was rewritten.
6. P1 a disposition missing on either side is a violation (`None == None` no longer agrees).
7. P1 `journal[-1].failure_mode` vs the sidecar is DIAGNOSTIC only (3,102 / 3,000 pages): the
   journal holds the last engine attempt, the sidecar the final page mode after post-selection
   guards, so they differ by design. Reported and in the JSON, outside the exit status.

Re-measurement: METER did not move: 12,165 (23117187) and 12,146 (main). One page now also
carries `shipped_bytes_differ` (Adamek et al. p5, a native+equations page already counted);
it is the only page whose shipped text differs from the judged text. Surface section is 0 on
both runs (was 3,102 / 3,000, all from the now-diagnostic comparison).
Not measured: the document-level hidden-mark scan on real documents beyond the two runs.

## Round 3 (Astra's second review; orchestrator decision: stop chasing parser states)

Astra reproduced METER 0 / exit 0 with a mark in a blockquoted fence, in a `<script>` block,
behind a comment closed and reopened on one line; with `<!--` before `## Page 1` in the
document; and with an unlisted output that has no `metadata.json`. All five are now tests
that fail on the round 2 code (checked against the round 2 control copy) and pass now.

Rules changed:
1. MARKED is position-fixed: the page `.md` STARTS (after blank lines) with one or more
   consecutive mark lines. A mark anywhere else is not a mark. Diagnostic `mark_not_at_top`
   counts unmarked WARNING/ERROR pages that do have a mark lower down (step 3 of the plan
   moves every mark to the top). `success_with_mark` still fires for a mark anywhere on a
   SUCCESS page (stricter, needs no parsing). A failed document's mark must be the first line
   of its document `.md`.
2. The document must be the assembly of its pages: `ocr_output_contract.assemble_pages`
   (the pure function behind `_stitch_fragments` and `_canonical_body`) over the terminal
   page `.md` files in page order with explicit page numbers, compared to the shipped `.md`
   with trailing whitespace per line normalised. A difference is `document_not_assembly` for
   the document's pages. `_stitch_fragments` itself is a method that needs pipeline state, so
   the meter calls the shared pure function and re-implements nothing.
3. Documents are found by structure: a directory holding `pages/NNNNN.json`; a
   `metadata.json` with `key` still finds a document that failed before any page.
4. Dead code removed: the comment/fence/link scanner, document slicing, page-vs-slice surface
   class. The module is simpler.

Re-measurement. METER 23117187: 12,165 -> 12,815 (+650). `runs_main`: 12,146 -> 12,800 (+654).
The rise is exactly `mark_not_at_top` (650 / 654): pages whose mark exists but sits below the
first line, now unmarked by rule. `document_not_assembly` is 0 on both runs: every shipped
document equals the assembly of its pages. Surface 0; journal diagnostic unchanged.
Mutants: 15 external, all killed, control 56 pass.

## Round 4 (Astra's third review: our own visibility rules were still wrong)

Two bypasses: a top line `> [table unverified: ...]: /hidden` is a CommonMark reference
definition (renders nothing, for four of the six mark forms; `[page N failed: x]: /hidden`
is one too); and page 1 ending with `<!--` hides page 2 and its mark although the assembly
is exact. Both are tests that fail on the round 3 code (4 failures there, run on a copy).

Change: visibility is decided by the parser socr already depends on (`markdown-it-py`, set up
as in `figures/crop_descriptions._parse_blocks`: `MarkdownIt("commonmark").enable("table")`).
- Page MARKED: the first rendered block is a paragraph (or a quote opening with one) whose
  every line is a mark. Reference definitions produce no tokens; HTML blocks, fences and
  headings are not marks. (A mark in a paragraph with prose is not marked: stricter than
  round 3, and it moved nothing on the corpus.)
- `check_document`: one parse of the document. Each page heading must be a rendered `h2`
  token, in page order; otherwise `document_page_hidden` for that page and each following
  page until the next visible heading. A MARKED page's mark must be the first rendered block
  under its heading (`document_mark_not_first`). A hidden or misplaced page is not counted MARKED.
- `page_leaks_raw_html`: the fragment's last HTML block is a comment, `<?`, CDATA, declaration
  or script/pre/style/textarea block that never closes. Detected from the parser's html_block
  token. An unclosed code fence also swallows the next heading but is caught only by
  `document_page_hidden` (it is not raw HTML).
- Exact assembly comparison kept.

Re-measurement: 23117187 METER 12,815 (unchanged from round 3), `runs_main` 12,800
(unchanged). `document_page_hidden`, `document_mark_not_first`, `page_leaks_raw_html` and
`document_not_assembly` are all 0 on both runs; `mark_not_at_top` 650 / 654. So no real page
trips a new class. Measured: the parser path takes about 40 s (23117187) and 60 s (main).
Not measured: pages whose first block is a table or list that starts with a mark.
Mutants: 17 external, all killed, control 74 pass. One mutant (dropping a
"block has text" guard) survived because the guard was dead: reference definitions yield no
tokens at all. I deleted the guard instead of keeping dead code.

## Round 5 (cubic review of the pull request)

Five findings; each was reproduced on the code as it stood before the fix.

1. P1, real. The text normaliser stripped leading whitespace, so a fragment indented four
   spaces (a code block when rendered) compared equal to the unindented judged text. Now only
   trailing whitespace per line and surrounding blank lines are ignored. Test and mutant added.
2. P2, real. `docs/OUTPUT.md` now states the position rule: the first rendered block of the
   page, and the first rendered block under the page's heading in the document.
3. P2, real. The wrapped-mark tests had prose before the wrapper, so a parser that rendered
   the wrapped mark would not have been caught. The wrapper now opens the page. A mutant
   that renders the content of HTML blocks and fences is killed by six of those cases (the
   inline and multi-line comment, the quoted fence, both fence styles, the unclosed comment).
   The link-text and image-alt cases are paragraphs, not wrappers; they are killed by a mutant
   that accepts a block when any one line is a mark.
4. P2, real. `stem:*` was a document-level id counted inside the page union, and a non-canonical
   file name was too. Both are now document-level findings: listed in the report and the text
   output, they fail the exit status, and they are not in the METER union. The real page ids of
   an unlisted document stay `unaccounted`.
5. P3, real. Scratch paths and the elided mutant heading are gone from this log; evidence is
   described in words and counts.

Re-measurement: no change. METER 12,815 (23117187) and 12,800 (previous run);
`shipped_bytes_differ` still 1 page on each, `unlisted_document` and `noncanonical_page_file`
0 on both. Five external mutants, all killed, control 79 tests pass.
