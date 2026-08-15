"""Header-attribution check for emitted table output (GH-200).

TR-3 (``native_verifier.py``) is a numeric-token multiset check: it proves the
numbers on a page and the numbers in the emitted table are the same bag of
values, and it is blind by construction to whether each number is still bound
to the right column — a destroyed, detached, or re-lettered header changes
zero numerals. The 2026-08-15 hand judgement
(``docs/log/2026-08-15_tr3-hand-judgement.md``) found this exact defect on
4 of 4 damaged tables it inspected. This module is the third, disjunctive
term the ratified spec adds alongside TR-3 and the B1 grid-shape gate
(``structure_check.structural_gate_fires``); it never replaces either.

The check reuses the SAME anchor -> lane -> header-band geometry chain
``header_repair.py`` already uses to repair collapsed headers
(``header_repair.native_header_row``), run on the words the source PDF page
actually carries. A native header cell is "owed" a match in the emitted
header row iff the native page carries non-blank header words over that data
lane; an empty native cell (an implied/spacer column) owes nothing — that is
Gemini's named falsifier, and it is the reason this check needs BOTH the
emitted grid and the native words, never the grid alone.

Verdicts, in order of how much they can do to the page:

- ``HARD``  — some data lane has non-blank native header words, and NONE of
  that cell's whitespace-split tokens appear anywhere in the emitted header
  row (``grid[0]``). Header content was LOST, not merely moved. Shift- and
  skew-tolerant by construction: token membership is tested against the
  whole emitted header row, never a fixed column index. Callers reject on
  this verdict.
- ``SOFT``  — every native header cell's tokens appear SOMEWHERE in the
  emitted header row, but not all under the matching column index. Evaluated
  only when the emitted header row is the same width as the derived native
  header row (label + one cell per lane) — a documented, deliberately
  unpromoted advisory signal (see the hand-judgement falsifier note: whether
  this should ever reject needs a measurement that does not exist yet).
  Callers record it, never reject on it alone.
- ``UNVERIFIABLE`` — the geometry chain abstained (no anchor row with an
  exact numeric-multiset match, fewer than 2 derived data lanes, or no
  lane-aligned header band above the anchor). This is the deliberate answer
  for headerless two-column indexes (book back-matter) and any table whose
  values are typeset in a notation ``_NUM_TOKEN_RE`` does not recognise
  (leading-decimal coefficients, star-only significance marks) — abstain
  instead of buying a model call or a false reject.
- ``OK``    — every native header cell's tokens are attributable to the
  matching emitted column, or there was nothing to check (fewer than 2
  emitted header cells, or no data lanes with native header words at all).

Pure; no I/O, no model calls, no mutation of its inputs.
"""

from __future__ import annotations

from enum import Enum

from socr.tables.header_repair import native_header_row


class HeaderVerdict(Enum):
    HARD = "hard"
    SOFT = "soft"
    UNVERIFIABLE = "unverifiable"
    OK = "ok"


def header_attribution(grid: list[list[str]], words: list) -> HeaderVerdict:
    """Classify one emitted table block's header against native word geometry.

    ``grid`` is the emitted, separator-free table grid (``grid[0]`` is the
    emitted header row). ``words`` is ``page.get_text("words")`` from the
    SOURCE pdf page (or an empty/None value on a page with no fitz access,
    which abstains).
    """
    if not grid or not grid[0]:
        return HeaderVerdict.UNVERIFIABLE

    native_header = native_header_row(grid, words)
    if native_header is None:
        return HeaderVerdict.UNVERIFIABLE

    lanes = native_header[1:]  # native_header[0] is the label cell, never checked
    if not any(cell.strip() for cell in lanes):
        # No lane carries native header words at all -- nothing owed, nothing to check.
        return HeaderVerdict.OK

    emitted_header = grid[0]
    emitted_tokens = {tok for cell in emitted_header for tok in cell.split() if tok.strip()}

    for cell in lanes:
        text = cell.strip()
        if not text:
            continue  # spacer/implied column: no native header owed for this lane
        tokens = [t for t in text.split() if t]
        if not any(t in emitted_tokens for t in tokens):
            return HeaderVerdict.HARD

    if len(emitted_header) == len(native_header):
        for i, cell in enumerate(lanes, start=1):
            text = cell.strip()
            if not text:
                continue
            emitted_cell_tokens = {t for t in emitted_header[i].split() if t.strip()}
            tokens = {t for t in text.split() if t}
            if not (tokens & emitted_cell_tokens):
                return HeaderVerdict.SOFT

    return HeaderVerdict.OK
