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

Suite: **7,887 passed, 2 skipped, 4 xfailed, 0 failed**. ruff 0.16.0 format clean (955
files).
**32 mutants, 32 killed**, each by a named test; seven are new here (`no-document-mark`,
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

## Review round 1 on PR #1168 — the stub reached further than it should have

All three reproduced before being accepted. Each is the same shape: a one-line file added
to a directory is read by more things than the reader it was written for.

**P1, and it is a regression this change introduced.** `install_staged` gated on "a markdown
file is here", which until slice 2 doubled as "socr got something out of this document": an
unreadable PDF produced no markdown, so install raised, the stem stayed in `missing_text`,
and the next run re-queued it. The stub satisfied that file check — so the document
INSTALLED, was reported as `Installed`, left `pending_pdfs` for ever, and landed in
`unverified` beside documents that do have text. A failure made invisible by the very change
that exists to make failures visible.

Fixed by reading the status the pipeline RECORDED rather than inferring it from what exists
on disk: a `failed` record is refused exactly as a missing file was. Tested through
`process_new`, with the positive control in the same test — a `partial` run that produced
real text must still install, because that is the library's own rule (best available text,
listed unverified), and without that half the test would pass on an install that refuses
everything. Mutant `install-gates-on-file-existence-only`.

**P2, the newline.** Only `]` was escaped. `DOCUMENT_FAILED_MARK_RE` excludes `\r` and
`\n`, so a multi-line detail produced a file with no recognised mark in it at all and the
document counted `document_failed_unmarked` exactly as before — the defect surviving its own
fix, for the second time in this ticket. The detail interpolates a PyMuPDF error string socr
does not author and which is free to contain either. Now flattened the way the page grammar
flattens a reason (`core/marks.clean_reason`): collapse all whitespace, then neutralise the
terminator. The test asserts the words survive the flattening, not merely that a mark
matches.

**P2, the overwrite window.** `probe_page_loads` can fail transiently, and this path runs
before anything else looks at the input, so the stub replaced an earlier successful run's
real text with a one-line mark while that run's `pages/` stayed on disk — a stale MIX the
meter reports as `document_not_assembly`, restored on the next run, so a window rather than
a permanent loss, but silent, and of the only copy in that directory.

**Chosen behaviour: refuse to write when markdown already exists**, and say so at WARNING.
Why this and not the alternatives: writing the stub alongside under another name would
invent a second convention no reader or tool knows, and reconciling the stale `pages/` is a
larger problem that PREDATES this change — the FAILED metadata record already overwrites the
earlier COMPLETED one on this path (#871), and redesigning that is not this ticket's. With
the refusal, the failure stays recorded in `metadata.json` exactly as before slice 2 and
`output_path` stays empty, which is also exactly as before: the only thing slice 2 changes
for such a document is that it no longer destroys the text. The writer returns `False`, so
`output_path` cannot name a stub that was not written. Mutant
`document-mark-overwrites-prior-text`.

## Not done here

- Slice 3's two SUCCESS-demoting populations (399 `native+equations`, 210 chart) go out as
  one PR each (orchestrator, 2026-10-15), with the escalation `page_judge` clause attached
  to whichever lands first.
- The judge-crash-with-`audit_passed=true` clause, per above.
