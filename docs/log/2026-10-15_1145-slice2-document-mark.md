# #1145 slice 2 — a failed document says so where a reader looks

2026-10-15. Design: `scratch/routing-baseline/leads/1145/design.md` §4 and §7, test T10.
Branch `feat/issue-1145-slice2` off `origin/main` (80f36827, slice 1 merged).

## What shipped

`_refuse_unreadable_input` (#871) records a PDF whose pages cannot be loaded in
`metadata.json` and the root index — which is what that ticket fixed — but it wrote no
`.md` at all and left `output_path` empty. A reader or a downstream tool that opened the
document's markdown found **nothing**, and "no file" is indistinguishable from "not
processed yet". This is the document-level analogue of the page rule: for a document socr
could not read, the whole artifact is the mark.

It now writes `<stem>.md` holding one `[document failed: <detail>]` line, through the
contract's own `doc_dir_for` / `markdown_path_for`, so the file is at the exact path
`_write_metadata` records and a normal run would have used — not a second convention that
could drift from it. **Verified empirically rather than assumed:** a fresh run over an
unreadable PDF puts the file at `broken/broken.md`, and the meter reads
`<doc_dir>/<stem>.md`, which is the same path.

### The status must not soften, and that needed a parameter

`_write_metadata` derived BOTH the record status and `output_path` from one `has_text`
argument. Writing a markdown file is exactly the kind of change that quietly turns FAILED
into PARTIAL, and if it did, the resume gate would stop refusing the entry and the next run
would never retry the file: a document socr cannot read would be recorded as done, for ever.
So the two questions are separated. `markdown_written: bool | None = None` defaults to
`has_text`, which leaves every other caller byte-identical (pinned by a test that asserts
the default and drives both arms), and the refusal passes `has_text=False` with
`markdown_written=<whether the write landed>`.

**Fail closed on the write.** The refusal must survive a disk it cannot write to — a
document that cannot be written must still be REFUSED and recorded rather than crashing the
batch — but it must not then record a path to a file that does not exist, which would send
every downstream reader to a missing file and look like corruption rather than a failure. So
`_write_document_failed_mark` returns whether the file landed, and `output_path` names it
only then.

**The `]` terminator is escaped**, because the detail interpolates a PyMuPDF error string
that socr does not author. A `]` in it would end the mark early and the remainder would read
as prose. Driven through the real writer with a detail containing one, not asserted against
the regex alone.

## Measurement

`document_failed_unmarked`: **64 → 0**. Run 23192118 has one document with
`status=failed` and an unreadable-input error, declaring 64 pages, carrying no document
mark. The "after" cannot come from a replay — a replay reconstructs pages, and this path has
no page loop — so it is demonstrated by a fresh run over an unreadable PDF, which is T10.

Suite: **7,884 passed, 2 skipped, 4 xfailed, 0 failed**. ruff 0.16.0 format clean (954
files).
**29 mutants, 29 killed**, each by a named test; four are new here (`no-document-mark`,
`document-mark-softens-the-status`, `document-mark-path-recorded-without-the-file`,
`document-mark-terminator-unescaped`).

## #1135 part 2 is already closed, and the design's premise about it was wrong

The design assigned `page_judge=UNKNOWN` for #1135 part 2 to this slice, on the stated
premise that "#1135's page ships `audit_passed=true`, so `page_judge=UNKNOWN` is what
catches it". Measured on run 23192118 before building anything:

* **11 pages** carry a timeout/crash judge outcome (all `page_judge_exception`), all
  shipping SUCCESS with no mark in the run as it was produced;
* **none of them has `audit_passed=true`** — all eleven have `audit_passed=false` while
  shipping SUCCESS, which is #1129's defect class, not a new one;
* replayed through today's seam, **all 11 come out WARNING carrying
  `page_judge=UNKNOWN`**, and 0 are still unmarked.

So slice 1 plus #1129 already close the measured population, and adding a clause for it
would have been inert. No clause was added.

**The hypothesised shape is a real hole, though, and it is slice 3's.** Constructed
directly at the seam: a page with `judge_outcome=page_judge_exception` (or `_timeout`),
`audit_passed=True` and no failure mode ships SUCCESS, with an empty clause list and no
mark. It is reachable in code — `pipeline/agentic.py`'s judge-exception handler sets
`judge_outcome` and never touches `audit_passed`, which defaults `True` — but whether
selection can deliver such an output as the winner is unresolved, and it is 0 pages on this
corpus. That is the same shape, the same evidence status and the same fix as the escalated
never-judged winner: a SUCCESS demotion plus a mark, witnessable only by a constructed test.
It therefore ships with slice 3, beside it, rather than being half-built here.

## Not done here

- Slice 3's two SUCCESS-demoting populations (399 `native+equations`, 210 chart) go out as
  one PR each (orchestrator, 2026-10-15), with the escalation `page_judge` clause attached
  to whichever lands first.
- The judge-crash-with-`audit_passed=true` clause, per above.
