# #213 — book indexes routed to table reconstruction

**Read this first. It is written to be picked up cold, in a fresh session, with no memory of
the conversation that produced it.**

Last updated: 2026-08-15
Issue: <https://github.com/r-uben/socr/issues/213>

---

## The defect, in one paragraph

A book's back-matter **index** — the alphabetical list at the end, where each entry ends in a
page number — is not a table. socr classifies it as one, hands it to table reconstruction, and
the rowizer reshapes prose lines into a fabricated grid. Every model call and every downstream
structural check on that page is wasted at best, and produces a confidently wrong "table" at
worst.

## Evidence it is real

Two independent hand judgements, on two different signals, both landed on the **same
document**:

- `docs/log/2026-08-15_tr3-hand-judgement.md` — 3 of 7 pages judged were not tables; two were
  book indexes (`2003__woodford` p799, `039_2021__nagel__ml_ap`).
- `docs/log/2026-08-15_b1-hand-judgement.md` — 1 of 5 pages judged was not a table:
  `2003__woodford` p798.

So this is not an artefact of one detector's quirk. Something upstream, at the table
*detector*, admits index pages.

## What is NOT known — and this is the whole point of the first task

**The mechanism is unidentified.** The issue originally asserted one, it was checked, and it
was wrong. Do not re-assert it.

`BornDigitalDetector._detect_tables` (`src/socr/core/born_digital.py`, ~line 1006) has exactly
two ways to answer "yes, table":

```python
tables = page.find_tables()  # BRANCH A — PyMuPDF's own table finder
if len(tables.tables) > 0:
    return True
return has_numeric_columns(page)  # BRANCH B — socr's numeric-lane gate
```

**Branch B is ruled out on the code.** `has_numeric_columns`
(`src/socr/tables/reconstruct.py`, ~line 432) requires each qualifying row to occupy at least
`_MIN_LANES_PER_ROW = 3` distinct numeric lanes, over at least `_MIN_TABLE_ROWS = 3` rows
(constants at `reconstruct.py:86-88`). An index row has exactly **one** numeric token, its page
number, so it occupies one lane. It cannot satisfy a three-lane rule. `_detect_tables`'s own
docstring also records that this gate *replaced* an older single-token heuristic precisely
because that one false-fired on this shape.

**Branch A is therefore the leading candidate, and is unverified.** That is a deduction, not a
measurement, and the last deduction on this ticket was wrong.

---

## Task 1 — Identify the branch (do this first, nothing else)

**Goal:** a recorded fact about which branch returns True on a real index page.

Use the **real** page. It is diagnosis, run locally, and nothing about it gets committed:

- File: `2003__woodford.pdf`, pages **798 and 799**
- Location: `~/Library/Mobile Documents/com~apple~CloudDocs/library/Papers/papers/`
- If it is a 0-byte file, iCloud has evicted it — see "iCloud" below.

Call the two branches **separately** on that page and record which fires:

```python
import fitz

doc = fitz.open(path)
page = doc[797]  # p798, zero-indexed
print("A find_tables :", len(page.find_tables().tables))
from socr.tables.reconstruct import has_numeric_columns

print("B lane gate   :", has_numeric_columns(page))
```

Then repeat on `2021__nagel` (the other judged index) so the answer is not one page's accident.

**Record the result in `docs/log/`** even if it is boring. A negative result — "neither
branch fires, so the page is reaching reconstruction some third way" — is the most valuable
possible outcome, because it would mean the whole framing above is wrong.

**Stop here and report.** Do not start Task 2 until Task 1's answer is written down.

## Task 2 — Fix, chosen by Task 1's answer

Do not pre-commit to any of these. The branch decides.

- **If Branch A (`find_tables`) fires:** the remedy is arbitration or a negative index-shape
  signal — something that says "this looks like an index, do not treat it as a table" and wins
  against PyMuPDF's finder. Do **not** touch `has_numeric_columns`; it is behaving correctly.
- **If Branch B (the lane gate) fires:** the code reading above is wrong somewhere. Find out
  why before changing constants. Do not tune `_MIN_LANES_PER_ROW` to make a symptom go away.
- **If neither fires:** the page is entering reconstruction by a path not yet mapped. Trace it
  from the assessment to the reconstruction call and write down the real route. This
  supersedes everything above.

Whatever the shape, an index-detection signal should key on what an index actually *is* —
many short lines, one trailing integer each, no ruled lines, no header band — rather than on a
threshold tuned to these two pages.

## Task 3 — Regression test

**This one must use a generated fixture, not corpus content.** The repository is public and
the papers are copyrighted. Never commit a corpus PDF or a page extract.

Build a synthetic page that mimics the *shape*: short prose lines, each followed by a
right-aligned integer, no ruled lines, no header row. `tests/test_header_repair.py` shows the
in-process pattern (`fitz.open()` + `page.insert_text()`).

The test must assert the page is classified as prose, **not** routed to table reconstruction.

---

## Rules that will bite you

- **Never work on `main`.** Branch first: `fix/213-index-routing`.
- **Stage files by name.** Never `git add -A`.
- **Python:** `~/venvs/socr/bin/{python,pytest}` or `uv run`. Never `python file.py`.
- **Lint is a blocking CI gate, and the venv's ruff lies.** Run it exactly as CI does:
  `uvx ruff@0.16.0 format --check .`. `~/venvs/socr/bin/ruff` is older and reports clean on
  files CI rejects — this exact gap turned `main` red and blocked four pull requests.
- **CI has no ollama and no model provider.** Any test that drives agentic mode must patch
  `_available_engines_for_agentic`, or it passes locally and fails in CI.
- **No magic numbers.** Derive from the page's own geometry, or use a named constant whose
  justification is in its docstring.
- **No git worktree.** The editable install resolves `import socr` to the main checkout, so a
  worktree tests the wrong source and produces false green tests.
- **iCloud:** the papers library lives in iCloud. An evicted file is a **0-byte placeholder**
  that opens as an empty or broken PDF. Always check `st_size > 0` before trusting a read; a
  0-byte file must be reported, never silently treated as a clean result. 45 of 407 papers
  were evicted as of 2026-08-15.

## Useful existing tooling

- `~/.local/share/socr/tr3-judge/build_b1_review_set.py` — stages pages for side-by-side
  human judging (page render + what socr produced). Its docstring states its sampling rule;
  copy that discipline.
- `~/.local/share/socr/tr3-judge/scan_sign_corruption.py` — an example of a corpus-wide scan
  that handles iCloud eviction correctly.

## Related tickets — context only, do not fix them here

- **#150** — figures extracted as tables. The *figure* half of the original #213 was moved
  there; #213 is indexes only now.
- **#205** — the TR-3 signal's firing rate; index false positives are part of why its
  precision as a table signal is only ~57% in the hand-judged sample.
- **#113 (closed)** — a different false-positive class (degenerate 2-cell pseudo-tables),
  fixed at the escalation-trigger layer. This ticket is upstream of that, at the detector.
