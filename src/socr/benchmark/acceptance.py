"""#1130: the corpus acceptance meter (CPU only, no model).

    socr-acceptance <run_dir> --inputs corpus/manifest.csv [--report acceptance.json]

Prints ``METER = <violations> of <input pages>``: the number of input pages that are
neither CHECKED nor MARKED, or that carry a contradiction. Target 0. Exit status 1 when
the meter, the surface-agreement section or a document-level finding (an unlisted document,
a non-canonical file in ``pages/``) is non-zero.

Rule
----
Every input page is exactly one of CHECKED or MARKED, and every surface agrees. The
meter is the size of the UNION of violating page ids; classes overlap (a
``native+equations`` page is both ``native_unchecked`` and ``equation_unverified_success``).

Accounting. The denominator is the input manifest (``out_file``, ``n_pages``); page
identity is (document stem, 1-based page). A page is accounted for when a terminal
``pages/NNNNN.json`` + ``NNNNN.md`` exists. A document recorded as ``failed`` with an
``unreadable_input`` reason accounts for its pages only if its document ``.md`` carries a
visible mark. Missing, extra, duplicate and non-terminal pages are ``unaccounted``.

CHECKED needs positive evidence; a missing or failed check is never a pass. Every lane
requires, on the sidecar top level AND ``winning_output``: ``status == success``,
``failure_mode == none``, ``audit_passed == true``, and no mark line in the page ``.md``.
Per lane (one function each, so a later per-page verdict record replaces them locally):

- ``qwen``: ``judge_outcome == page_judge_completed``; no disputed, unverifiable or
  label-unverified table; and not a table-only acceptance (``judge_reason`` starting
  ``native_table_verifier:`` means the page judge was bypassed).
- ``native``: a native coverage witness event with ``data.result == pass``.
- ``native+equations``: the native rule, and ``semantic_fidelity_verified`` true on every
  ``equation_region_reading_attached`` event.
- ``native+math``: never CHECKED; it must be MARKED.
- ``chart_asset``: a figure-text coverage pass; and a chart-values pass when values were
  transcribed (no ``visual_values_not_transcribed`` event).
- ``blank``: a ``blank_page`` event and a page ``.md`` equal to the blank note.

MARKED is position-fixed and judged by the CommonMark parser socr already depends on
(markdown-it-py): the FIRST rendered block of the page ``.md`` is a paragraph (or a quote
opening with one) whose every line is a recognised mark (see ``MARK_FAMILIES``). A
reference definition, HTML block, fence or heading renders no text and is no mark. A mark
elsewhere does not count; a page whose only mark sits lower is reported in the diagnostic
``mark_not_at_top``. The blank receipt is not a mark. A SUCCESS page carrying a mark
anywhere is its own class (``success_with_mark``).

The document ``.md`` is parsed once: every page heading must be a rendered heading token in
page order (else ``document_page_hidden``: a heading swallowed by a comment or raw HTML,
even one opened by an earlier page), and a MARKED page's mark must be the first rendered
block under its heading (``document_mark_not_first``). A page fragment that leaves an HTML
comment or raw block open is ``page_leaks_raw_html``. The document ``.md`` must also be the
assembly of its pages: rebuilt with the contract's ``assemble_pages`` from the page
fragments and compared (trailing whitespace per line aside); a difference is
``document_not_assembly`` for the document's pages. Documents are found by structure (a
directory holding ``pages/NNNNN.json``), with or without ``metadata.json``.

Surface agreement (own section; must also be 0): the sidecar disposition equals the
manifest entry disposition; the sidecar failure mode equals ``journal[-1].failure_mode``;
``metadata.json`` per-status counts, when present, equal the sidecar counts. A
missing surface is UNKNOWN, which is a violation. ``metadata.json`` carries no per-status
page counts today: that is the document-level class ``metadata_counts_missing``, kept out
of the meter and the exit status.

Evidence events are provisional names (``EVIDENCE_*``): no native or chart coverage
witness exists in socr yet (#1050 step 1), so today every such page is a violation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token
from ocr_output_contract import assemble_pages

from socr.core.blank_page import blank_page_marker
from socr.core.manifest import (
    DISPUTED_TABLE_MARK_RE,
    PAGE_UNCHECKED_MARK_RE,
    SCANNED_NATIVE_TEXT_FLAG,
    SCANNED_PROSE_RECOVERED_FLAG,
    SOCR_MARKER_RE,
    UNVERIFIED_TABLE_MARK_RE,
    d3_superseded_note,
    judge_timeout_note,
)
from socr.core.result import JUDGE_OUTCOME_COMPLETED, FailureMode, PageStatus

# --- violation classes -----------------------------------------------------------------
UNACCOUNTED = "unaccounted"
DOCUMENT_FAILED_UNMARKED = "document_failed_unmarked"
NATIVE_UNCHECKED = "native_unchecked"
CHART_UNCHECKED = "chart_unchecked"
SUCCESS_AUDIT_FALSE = "success_audit_false"
EQUATION_UNVERIFIED_SUCCESS = "equation_unverified_success"
WARNING_UNMARKED = "warning_unmarked"
ERROR_UNMARKED = "error_unmarked"
SUCCESS_WITH_MARK = "success_with_mark"
TABLE_ONLY_ACCEPTANCE = "table_only_acceptance"
SUCCESS_UNCHECKED = "success_unchecked"  # a SUCCESS page no lane rule can vouch for
SHIPPED_BYTES_DIFFER = "shipped_bytes_differ"  # the page .md is not the text that was judged
UNLISTED_DOCUMENT = "unlisted_document"
NONTERMINAL_PAGE = "nonterminal_page"
DUPLICATE_INPUT_STEM = "duplicate_input_stem"
NONCANONICAL_PAGE_FILE = "noncanonical_page_file"
DOCUMENT_NOT_ASSEMBLY = "document_not_assembly"
DOCUMENT_PAGE_HIDDEN = "document_page_hidden"
DOCUMENT_MARK_NOT_FIRST = "document_mark_not_first"
PAGE_LEAKS_RAW_HTML = "page_leaks_raw_html"

#: Findings about a document or a file, not about a page: reported, and they fail the gate,
#: but they are not page ids and must not inflate METER.
DOCUMENT_LEVEL_VIOLATIONS = (UNLISTED_DOCUMENT, NONCANONICAL_PAGE_FILE)

VIOLATION_CLASSES = (
    UNACCOUNTED,
    DOCUMENT_FAILED_UNMARKED,
    NATIVE_UNCHECKED,
    CHART_UNCHECKED,
    SUCCESS_AUDIT_FALSE,
    EQUATION_UNVERIFIED_SUCCESS,
    WARNING_UNMARKED,
    ERROR_UNMARKED,
    SUCCESS_WITH_MARK,
    TABLE_ONLY_ACCEPTANCE,
    SUCCESS_UNCHECKED,
    SHIPPED_BYTES_DIFFER,
    UNLISTED_DOCUMENT,
    NONTERMINAL_PAGE,
    DUPLICATE_INPUT_STEM,
    NONCANONICAL_PAGE_FILE,
    DOCUMENT_NOT_ASSEMBLY,
    DOCUMENT_PAGE_HIDDEN,
    DOCUMENT_MARK_NOT_FIRST,
    PAGE_LEAKS_RAW_HTML,
)

# --- surface classes -------------------------------------------------------------------
SURFACE_DISPOSITION = "disposition_mismatch"
SURFACE_FAILURE_MODE = "failure_mode_mismatch"
SURFACE_METADATA = "metadata_counts_mismatch"
SURFACE_CLASSES = (SURFACE_DISPOSITION, SURFACE_METADATA)
#: Diagnostic only, outside the exit status: ``journal[-1].failure_mode`` is the last engine
#: ATTEMPT, the sidecar's is the page's FINAL mode after post-selection guards. They differ by
#: design on ~3,100 pages, so the comparison cannot be right until a final verdict record exists.
MARK_NOT_AT_TOP = "mark_not_at_top"
DIAGNOSTIC_CLASSES = (SURFACE_FAILURE_MODE, MARK_NOT_AT_TOP)
METADATA_COUNTS_MISSING = "metadata_counts_missing"
#: The metadata.json field that should carry per-status page counts (a later slice adds it).
METADATA_COUNTS_FIELD = "pages_by_status"

# --- engines and evidence --------------------------------------------------------------
ENGINE_QWEN = "qwen"
ENGINE_NATIVE = "native"
ENGINE_NATIVE_EQUATIONS = "native+equations"
ENGINE_NATIVE_MATH = "native+math"
ENGINE_CHART = "chart_asset"
ENGINE_BLANK = "blank"

EVENT_BLANK_PAGE = "blank_page"
EVENT_EQUATION_ATTACHED = "equation_region_reading_attached"
EVENT_VALUES_NOT_TRANSCRIBED = "visual_values_not_transcribed"
#: Provisional: #1050 step 1 and the verdict record have not landed. Absent today.
EVIDENCE_NATIVE_COVERAGE = "native_coverage_witness"
EVIDENCE_FIGURE_TEXT = "figure_text_coverage"
EVIDENCE_CHART_VALUES = "chart_values_check"
EVIDENCE_PASS = "pass"
TABLE_ONLY_JUDGE_REASON_PREFIX = "native_table_verifier:"

SUCCESS = PageStatus.SUCCESS.value
WARNING = PageStatus.WARNING.value
ERROR = PageStatus.ERROR.value
FAILURE_NONE = FailureMode.NONE.value
UNREADABLE_INPUT_PREFIX = "unreadable_input"
DOC_FAILED = "failed"


# --- mark recognition ------------------------------------------------------------------
def _canonical_mark(n: int, line: str) -> bool:
    """The #1130 grammar: ``> [page N unchecked: <check>=FAIL|UNKNOWN; scope=S; reason=R]``."""
    pattern = rf"> \[page {n} unchecked: [^\];=]+=(?:FAIL|UNKNOWN); scope=[^\];]+; reason=.+\]"
    return re.fullmatch(pattern, line) is not None


def _failed_mark(n: int, line: str) -> bool:
    """``[page N failed: …]`` (page_failed_marker and the D3/TR-3/invisible-scan floors)."""
    m = SOCR_MARKER_RE.fullmatch(line)
    return bool(m) and line.startswith(f"[page {n} failed:")


def _table_mark(_n: int, line: str) -> bool:
    return bool(DISPUTED_TABLE_MARK_RE.match(line) or UNVERIFIED_TABLE_MARK_RE.match(line))


def _page_unchecked_mark(_n: int, line: str) -> bool:
    return bool(PAGE_UNCHECKED_MARK_RE.match(line))


#: ``[page N: flagged table kept — …]`` (kept_table_flag_note, #259): the variable tail has
#: no constant, so the fixed prefix is the recogniser.
FLAGGED_TABLE_PREFIX = "flagged table kept — "


def _flagged_table_mark(n: int, line: str) -> bool:
    return line.startswith(f"[page {n}: {FLAGGED_TABLE_PREFIX}") and line.endswith("]")


def _fixed_note_mark(n: int, line: str) -> bool:
    """Notes whose whole line is a function of the page number."""
    return line in {judge_timeout_note(n), d3_superseded_note(n)}


def _scanned_flag_mark(n: int, line: str) -> bool:
    """``[page N: unverified scan — …]`` banners (SCANNED_*_FLAG templates)."""
    return line in {
        SCANNED_PROSE_RECOVERED_FLAG.format(page_num=n),
        SCANNED_NATIVE_TEXT_FLAG.format(page_num=n),
    }


#: #1050 step 2 (not emitted yet): recognised so the meter keeps working once it ships.
UNCOVERED_TEXT_NOTE = "uncovered text — see image"


def _uncovered_text_mark(n: int, line: str) -> bool:
    return line.startswith(f"[page {n}: {UNCOVERED_TEXT_NOTE}") and line.endswith("]")


#: Every visible mark family, with its emitter. The blank receipt (``[socr: blank page]``,
#: core/blank_page.py) and the fabricated-image receipt (core/url_provenance.py) are
#: receipts of what socr did, not statements that a check did not run: not marks.
MARK_FAMILIES: dict[str, Callable[[int, str], bool]] = {
    "canonical_unchecked": _canonical_mark,  # #1130 grammar, not emitted yet
    "page_failed": _failed_mark,  # core/manifest.py page_failed_marker + floors
    "table_disputed_or_unverified": _table_mark,  # core/manifest.py #1048
    "page_unchecked": _page_unchecked_mark,  # core/manifest.py page_unchecked_mark
    "flagged_table_kept": _flagged_table_mark,  # core/manifest.py kept_table_flag_note
    "fixed_note": _fixed_note_mark,  # judge_timeout_note (#713), d3_superseded_note
    "scanned_flag": _scanned_flag_mark,  # SCANNED_*_FLAG banners
    "uncovered_text": _uncovered_text_mark,  # #1050 step 2
}


def line_is_mark(line: str, page_num: int) -> bool:
    return any(fam(page_num, line) for fam in MARK_FAMILIES.values())


# Visibility is settled by the CommonMark reference parser socr already uses (GH-189's chart
# reconciler, crop descriptions), not by rules of ours: reference definitions produce no tokens,
# HTML blocks and code are not text. Same setup as ``figures/crop_descriptions._parse_blocks``.
_PARSER = MarkdownIt("commonmark").enable("table")


def parse_blocks(text: str) -> list[Token]:
    return _PARSER.parse(text)


def first_block_lines(tokens: list[Token]) -> list[str]:
    """The visible text lines of the FIRST rendered block, or [].

    Only a paragraph, or a block quote opening with a paragraph, counts. Quote lines are given
    back with their ``> `` prefix (the mark grammars include it). An HTML block, a fence, a
    heading yields []; a reference definition produces no tokens at all, so it never gets here.
    """
    if not tokens:
        return []
    if tokens[0].type == "paragraph_open" and len(tokens) > 1:
        prefix, inline = "", tokens[1]
    elif (
        tokens[0].type == "blockquote_open"
        and len(tokens) > 2
        and tokens[1].type == "paragraph_open"
    ):
        prefix, inline = "> ", tokens[2]
    else:
        return []
    return [prefix + line.rstrip() for line in inline.content.split("\n")]


def first_block_marks(tokens: list[Token], page_num: int) -> list[str]:
    """The mark lines of the first rendered block when EVERY line of it is a mark, else []."""
    lines = first_block_lines(tokens)
    return lines if lines and all(line_is_mark(ln, page_num) for ln in lines) else []


def leading_marks(text: str, page_num: int) -> list[str]:
    """MARKED is position-fixed: the first rendered block of *text* is made of mark lines."""
    return first_block_marks(parse_blocks(text), page_num)


_RAW_HTML_ENDS = (
    (re.compile(r"<!--"), "-->"),
    (re.compile(r"<\?"), "?>"),
    (re.compile(r"<!\[CDATA\["), "]]>"),
    (re.compile(r"<![A-Za-z]"), ">"),
)
_RAW_HTML_TAG_RE = re.compile(r"<(script|pre|style|textarea)(?:\s|>|$)", re.IGNORECASE)


def leaks_raw_html(tokens: list[Token]) -> bool:
    """True when the last HTML block is a comment or raw block that never closes: it would
    swallow whatever follows the fragment in the assembled document."""
    html = [t for t in tokens if t.type == "html_block"]
    if not html:
        return False
    body = html[-1].content.lstrip()
    for start, end in _RAW_HTML_ENDS:
        if start.match(body):
            return end not in body
    m = _RAW_HTML_TAG_RE.match(body)
    return bool(m) and f"</{m.group(1).lower()}>" not in body.lower()


def has_any_mark_line(text: str, page_num: int) -> bool:
    """A mark-shaped line anywhere (used for the contradiction and mark_not_at_top checks)."""
    return any(line_is_mark(raw.rstrip(), page_num) for raw in text.split("\n"))


#: Provisional document-wide failure mark (no emitter exists yet): a failed document's pages
#: are MARKED only by this at the top of its document .md, never by a page or table mark.
DOCUMENT_FAILED_MARK_RE = re.compile(r"\[document failed: [^\]\r\n]+\]")


def has_document_failure_mark(text: str) -> bool:
    lines = first_block_lines(parse_blocks(text))
    return bool(lines) and all(DOCUMENT_FAILED_MARK_RE.fullmatch(ln) for ln in lines)


def strip_marks(text: str, page_num: int) -> str:
    """*text* without its leading mark block (the generated part of a page body)."""
    tokens = parse_blocks(text)
    if not first_block_marks(tokens, page_num) or not tokens[0].map:
        return text
    start, end = tokens[0].map
    return "\n".join(text.split("\n")[:start] + text.split("\n")[end:])


_PAGE_HEADING_RE = re.compile(r"\s*Page\s+(\d+)\s*", re.IGNORECASE)


def check_document(doc_md: str, terminal: dict[int, tuple[dict, str]]) -> tuple[set[int], set[int]]:
    """Parse the whole document once. Returns (hidden pages, pages whose mark is not the
    first visible block under their heading).

    Every page heading must be a rendered heading token, in page order. A heading swallowed by
    a comment or raw HTML (even one opened by an earlier page) hides that page.
    """
    tokens = parse_blocks(doc_md)
    heads: list[tuple[int, int]] = []  # (token index, page number)
    for i, t in enumerate(tokens):
        if t.type == "heading_open" and t.level == 0 and t.tag == "h2" and i + 1 < len(tokens):
            m = _PAGE_HEADING_RE.fullmatch(tokens[i + 1].content)
            if m:
                heads.append((i, int(m.group(1))))
    hidden: set[int] = set()
    misplaced: set[int] = set()
    cursor = 0
    for n in sorted(terminal):
        if cursor < len(heads) and heads[cursor][1] == n:
            end = heads[cursor + 1][0] if cursor + 1 < len(heads) else len(tokens)
            wanted = leading_marks(terminal[n][1], n)
            if wanted and first_block_marks(tokens[heads[cursor][0] + 3 : end], n) != wanted:
                misplaced.add(n)
            cursor += 1
        else:
            hidden.add(n)
    return hidden, misplaced


def _norm(text: str) -> str:
    """Trailing whitespace per line, and surrounding blank lines, are ignored. Leading
    indentation is kept: four spaces turn a paragraph into a code block."""
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


# --- page record -----------------------------------------------------------------------
@dataclass
class Page:
    doc: str
    num: int
    sidecar: dict
    md: str
    manifest_entry: dict | None

    @property
    def pid(self) -> str:
        return f"{self.doc}:{self.num}"

    @property
    def wo(self) -> dict:
        return self.sidecar.get("winning_output") or {}

    @property
    def events(self) -> list[dict]:
        return [e for e in self.sidecar.get("audit_events") or [] if isinstance(e, dict)]

    def events_of(self, kind: str) -> list[dict]:
        return [e for e in self.events if e.get("kind") == kind]

    @property
    def marks(self) -> list[str]:
        return leading_marks(self.md, self.num)

    @property
    def marked(self) -> bool:
        return bool(self.marks)

    @property
    def mark_not_at_top(self) -> bool:
        return not self.marks and has_any_mark_line(self.md, self.num)


def _evidence_passed(page: Page, kind: str) -> bool:
    return any(
        str((e.get("data") or {}).get("result", "")).lower() == EVIDENCE_PASS
        for e in page.events_of(kind)
    )


# --- lane rules: each returns the violation classes it finds (empty = CHECKED) ----------
def success_common(page: Page) -> list[str]:
    """Rules every lane shares, on the top level AND winning_output."""
    j, wo = page.sidecar, page.wo
    out: list[str] = []
    if j.get("audit_passed") is False or wo.get("audit_passed") is False:
        out.append(SUCCESS_AUDIT_FALSE)
    elif j.get("audit_passed") is not True or wo.get("audit_passed") is not True:
        out.append(SUCCESS_UNCHECKED)
    for name, want in (("status", SUCCESS), ("failure_mode", FAILURE_NONE)):
        if j.get(name) != want or wo.get(name) != want:
            out.append(SUCCESS_UNCHECKED)
    if wo.get("rejection_class"):
        out.append(SUCCESS_UNCHECKED)
    judged = wo.get("text")
    if not isinstance(judged, str) or _norm(strip_marks(page.md, page.num)) != _norm(judged):
        out.append(SHIPPED_BYTES_DIFFER)
    return out


def rule_qwen(page: Page) -> list[str]:
    out: list[str] = []
    if page.wo.get("judge_outcome") != JUDGE_OUTCOME_COMPLETED:
        out.append(SUCCESS_UNCHECKED)
    bypassed = str(page.wo.get("judge_reason") or "").startswith(TABLE_ONLY_JUDGE_REASON_PREFIX)
    if bypassed and page.sidecar.get("audit_passed") is True:
        out.append(TABLE_ONLY_ACCEPTANCE)
    for name in ("table_disputed", "table_unverifiable", "table_label_unverified"):
        if page.wo.get(name):
            out.append(SUCCESS_UNCHECKED)
    return out


def rule_native(page: Page) -> list[str]:
    return [] if _evidence_passed(page, EVIDENCE_NATIVE_COVERAGE) else [NATIVE_UNCHECKED]


def rule_native_equations(page: Page) -> list[str]:
    out = rule_native(page)
    readings = page.events_of(EVENT_EQUATION_ATTACHED)
    if any((e.get("data") or {}).get("semantic_fidelity_verified") is not True for e in readings):
        out.append(EQUATION_UNVERIFIED_SUCCESS)
    return out


def rule_native_math(page: Page) -> list[str]:
    return [SUCCESS_UNCHECKED]  # never CHECKED: it must be MARKED


def rule_chart(page: Page) -> list[str]:
    ok = _evidence_passed(page, EVIDENCE_FIGURE_TEXT)
    if ok and not page.events_of(EVENT_VALUES_NOT_TRANSCRIBED):
        ok = _evidence_passed(page, EVIDENCE_CHART_VALUES)
    return [] if ok else [CHART_UNCHECKED]


def rule_blank(page: Page) -> list[str]:
    ok = bool(page.events_of(EVENT_BLANK_PAGE)) and page.md.strip() == blank_page_marker()
    return [] if ok else [SUCCESS_UNCHECKED]


LANE_RULES: dict[str, Callable[[Page], list[str]]] = {
    ENGINE_QWEN: rule_qwen,
    ENGINE_NATIVE: rule_native,
    ENGINE_NATIVE_EQUATIONS: rule_native_equations,
    ENGINE_NATIVE_MATH: rule_native_math,
    ENGINE_CHART: rule_chart,
    ENGINE_BLANK: rule_blank,
}


@dataclass
class Verdict:
    checked: bool = False
    marked: bool = False
    violations: list[str] = field(default_factory=list)
    detail: str = ""  # failure mode of an unmarked warning
    not_at_top: bool = False  # unmarked only because its mark sits below the top


def judge_page(page: Page) -> Verdict:
    """Classify one terminal page: CHECKED, MARKED, or a list of violation classes."""
    j = page.sidecar
    status = j.get("status")
    v = Verdict()
    if status == SUCCESS:
        found = success_common(page)
        rule = LANE_RULES.get(j.get("engine") or "")
        found += rule(page) if rule else [SUCCESS_UNCHECKED]
        if has_any_mark_line(page.md, page.num):  # a mark on a SUCCESS page is a contradiction
            found.append(SUCCESS_WITH_MARK)
        v.violations = list(dict.fromkeys(found))
        v.checked = not v.violations
        return v
    if status in (WARNING, ERROR):
        if page.marked:
            v.marked = True
        else:
            v.violations = [WARNING_UNMARKED if status == WARNING else ERROR_UNMARKED]
            v.detail = str(j.get("failure_mode") or "missing")
            v.not_at_top = page.mark_not_at_top
        return v
    v.violations = [UNACCOUNTED]  # pending / skipped / unknown: no finished verdict
    return v


# --- reading a run ---------------------------------------------------------------------
def read_inputs(path: Path) -> list[tuple[str, int]]:
    """(out_file path, declared page count) per input row, in manifest order.

    Identity is the ``out_file`` path; the stem alone is how outputs are named on disk, so
    two rows sharing a stem are reported (``duplicate_input_stem``), never merged.
    """
    with path.open(newline="") as fh:
        return [(r["out_file"], int(r["n_pages"])) for r in csv.DictReader(fh)]


def _has_canonical_sidecar(root: str) -> bool:
    pages = os.path.join(root, "pages")
    try:
        return any(
            CANONICAL_PAGE_FILE_RE.fullmatch(n) and n.endswith(".json") for n in os.listdir(pages)
        )
    except OSError:
        return False


def find_doc_dirs(run_dir: Path) -> list[Path]:
    """Every document directory, found by structure: one with ``pages/NNNNN.json``, or a
    ``metadata.json`` carrying a per-document ``key`` (a document that failed before any page).

    The batch directory above them holds a ``metadata.json`` too (a ``files`` map, no
    ``key``); it is walked through, not counted as a document.
    """
    found: list[Path] = []
    for root, dirs, files in os.walk(run_dir):
        is_doc = _has_canonical_sidecar(root)
        if not is_doc and "metadata.json" in files:
            is_doc = "key" in (_load_json(Path(root) / "metadata.json") or {})
        if is_doc:
            found.append(Path(root))
            dirs[:] = []
    return sorted(found)


def _load_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text()
    except (OSError, UnicodeDecodeError):
        return None


@dataclass
class Report:
    input_pages: int = 0
    checked: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    marked: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    violations: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    warning_unmarked_by_failure_mode: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    surface: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    diagnostic: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    metadata_counts_missing: list[str] = field(default_factory=list)
    doc_level_bad: set[str] = field(default_factory=set)  # hidden or misplaced in the document

    def violating_ids(self) -> set[str]:
        """Page ids only: document-level findings are not pages and stay out of the union."""
        return {
            pid
            for cls, ids in self.violations.items()
            if cls not in DOCUMENT_LEVEL_VIOLATIONS
            for pid in ids
        }

    def document_level_count(self) -> int:
        return sum(len(self.violations.get(cls, [])) for cls in DOCUMENT_LEVEL_VIOLATIONS)

    def surface_count(self) -> int:
        return sum(len(ids) for ids in self.surface.values())

    def to_json(self) -> dict:
        return {
            "meter": len(self.violating_ids()),
            "input_pages": self.input_pages,
            "checked": dict(self.checked),
            "marked": dict(self.marked),
            "violations": {k: self.violations.get(k, []) for k in VIOLATION_CLASSES},
            "warning_unmarked_by_failure_mode": dict(self.warning_unmarked_by_failure_mode),
            "surface": {k: self.surface.get(k, []) for k in SURFACE_CLASSES},
            "diagnostic_not_gating": {k: self.diagnostic.get(k, []) for k in DIAGNOSTIC_CLASSES},
            "document_level": {METADATA_COUNTS_MISSING: self.metadata_counts_missing},
        }


def _surface_checks(page: Page, rep: Report) -> None:
    j = page.sidecar
    entry = page.manifest_entry
    side_disp = j.get("disposition")
    # A disposition missing on either side is UNKNOWN, never agreement (None == None).
    if entry is None or side_disp is None or entry.get("disposition") != side_disp:
        rep.surface[SURFACE_DISPOSITION].append(page.pid)
    journal = (entry or {}).get("journal") or []
    last = journal[-1].get("failure_mode") if journal and isinstance(journal[-1], dict) else None
    if last is None or last != j.get("failure_mode"):
        rep.diagnostic[SURFACE_FAILURE_MODE].append(page.pid)


def _record_verdict(page: Page, v: Verdict, rep: Report) -> None:
    for cls in v.violations:
        rep.violations[cls].append(page.pid)
        if cls == WARNING_UNMARKED:
            rep.warning_unmarked_by_failure_mode[v.detail].append(page.pid)
    if v.not_at_top:
        rep.diagnostic[MARK_NOT_AT_TOP].append(page.pid)
    if v.checked:
        rep.checked[page.sidecar.get("engine") or "unknown"].append(page.pid)
    elif v.marked and page.pid not in rep.doc_level_bad:
        rep.marked[str(page.sidecar.get("status"))].append(page.pid)


CANONICAL_PAGE_FILE_RE = re.compile(r"(\d{5})\.(json|md)")


@dataclass
class DocPages:
    terminal: dict[int, tuple[dict, str]] = field(default_factory=dict)
    nonterminal: list[int] = field(default_factory=list)
    noncanonical: list[str] = field(default_factory=list)


def _terminal_pages(doc_dir: Path) -> DocPages:
    """Terminal pages by canonical ``NNNNN.json`` + ``NNNNN.md`` name only.

    Any other file in ``pages/`` is reported, never read (a ``1.json`` must not overwrite
    ``00001.json``). A canonical pair that is not terminal or cannot be read is reported
    as non-terminal.
    """
    out = DocPages()
    pages_dir = doc_dir / "pages"
    if not pages_dir.is_dir():
        return out
    nums: set[int] = set()
    for f in sorted(pages_dir.iterdir()):
        m = CANONICAL_PAGE_FILE_RE.fullmatch(f.name)
        if m:
            nums.add(int(m.group(1)))
        else:
            out.noncanonical.append(f.name)
    for num in sorted(nums):
        stem = pages_dir / f"{num:05d}"
        j = _load_json(stem.with_suffix(".json"))
        md = _read_text(stem.with_suffix(".md"))
        if j is None or md is None or j.get("terminal") is not True or j.get("page_num") != num:
            out.nonterminal.append(num)
        else:
            out.terminal[num] = (j, md)
    return out


def _is_assembly(doc_md: str, terminal: dict[int, tuple[dict, str]]) -> bool:
    """The document .md is the contract's ``assemble_pages`` of its page fragments.

    Page order, explicit page numbers: what ``_stitch_fragments`` does. Trailing whitespace
    per line is the only normalisation."""
    nums = sorted(terminal)
    rebuilt = assemble_pages([terminal[n][1] for n in nums], page_numbers=nums)
    return _norm(rebuilt) == _norm(doc_md)


def _measure_document(doc_dir: Path, expected: int, rep: Report) -> None:
    stem = doc_dir.name
    meta = _load_json(doc_dir / "metadata.json") or {}
    doc_md = _read_text(doc_dir / f"{stem}.md")
    found = _terminal_pages(doc_dir)
    terminal = found.terminal
    rep.violations[NONCANONICAL_PAGE_FILE].extend(f"{stem}:{name}" for name in found.noncanonical)
    rep.violations[NONTERMINAL_PAGE].extend(f"{stem}:{n}" for n in found.nonterminal)
    wanted = range(1, expected + 1)
    if not terminal and meta.get("status") == DOC_FAILED:
        ids = [f"{stem}:{n}" for n in wanted]
        if not str(meta.get("error") or "").startswith(UNREADABLE_INPUT_PREFIX):
            rep.violations[UNACCOUNTED].extend(ids)
        elif doc_md and has_document_failure_mark(doc_md):
            rep.marked["document_failed"].extend(ids)
        else:
            rep.violations[DOCUMENT_FAILED_UNMARKED].extend(ids)
        return
    entries = (_load_json(doc_dir / "manifest.json") or {}).get("entries") or {}
    if doc_md is None or not _is_assembly(doc_md, terminal):
        rep.violations[DOCUMENT_NOT_ASSEMBLY].extend(f"{stem}:{n}" for n in wanted if n in terminal)
    if doc_md is not None:
        hidden, misplaced = check_document(doc_md, terminal)
        rep.violations[DOCUMENT_PAGE_HIDDEN].extend(f"{stem}:{n}" for n in sorted(hidden))
        rep.violations[DOCUMENT_MARK_NOT_FIRST].extend(f"{stem}:{n}" for n in sorted(misplaced))
        rep.doc_level_bad.update(f"{stem}:{n}" for n in hidden | misplaced)
    rep.violations[UNACCOUNTED].extend(f"{stem}:{n}" for n in wanted if n not in terminal)
    rep.violations[UNACCOUNTED].extend(f"{stem}:{n}" for n in sorted(terminal) if n not in wanted)
    status_counts: Counter[str] = Counter()
    for n in (n for n in wanted if n in terminal):
        j, md = terminal[n]
        page = Page(stem, n, j, md, entries.get(str(n)))
        if leaks_raw_html(parse_blocks(md)):
            rep.violations[PAGE_LEAKS_RAW_HTML].append(page.pid)
        _record_verdict(page, judge_page(page), rep)
        _surface_checks(page, rep)
        status_counts[str(j.get("status"))] += 1
    declared = meta.get(METADATA_COUNTS_FIELD)
    if not isinstance(declared, dict):
        rep.metadata_counts_missing.append(stem)
    elif {k: v for k, v in declared.items() if v} != dict(status_counts):
        rep.surface[SURFACE_METADATA].append(stem)


def measure(run_dir: Path, inputs: list[tuple[str, int]]) -> Report:
    rep = Report(input_pages=sum(n for _, n in inputs))
    by_stem: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for out_file, n in inputs:
        by_stem[Path(out_file).stem].append((out_file, n))
    duplicates = {stem for stem, rows in by_stem.items() if len(rows) > 1}
    for stem in sorted(duplicates):  # ambiguous on disk: reported, never merged
        for out_file, n in by_stem[stem]:
            rep.violations[DUPLICATE_INPUT_STEM].extend(f"{out_file}:{p}" for p in range(1, n + 1))
    expected = {stem: rows[0][1] for stem, rows in by_stem.items() if stem not in duplicates}
    seen: set[str] = set()
    for doc_dir in find_doc_dirs(run_dir):
        stem = doc_dir.name
        if stem in duplicates:
            continue
        if stem not in expected:  # in the run, not in the manifest
            rep.violations[UNLISTED_DOCUMENT].append(f"{stem}:*")
            rep.violations[UNACCOUNTED].extend(
                f"{stem}:{n}" for n in _terminal_pages(doc_dir).terminal
            )
        elif stem in seen:  # a second output for the same input
            rep.violations[UNACCOUNTED].append(f"{stem}:duplicate_output:{doc_dir}")
        else:
            seen.add(stem)
            _measure_document(doc_dir, expected[stem], rep)
    for stem, n in expected.items():
        if stem not in seen:
            rep.violations[UNACCOUNTED].extend(f"{stem}:{p}" for p in range(1, n + 1))
    return rep


# --- output ----------------------------------------------------------------------------
def format_report(rep: Report) -> str:
    lines = [f"METER = {len(rep.violating_ids())} of {rep.input_pages}", "", "CHECKED (by lane)"]
    lines += [f"  {len(v):7d}  {k}" for k, v in sorted(rep.checked.items())] or ["        0"]
    lines.append("MARKED (by status)")
    lines += [f"  {len(v):7d}  {k}" for k, v in sorted(rep.marked.items())] or ["        0"]
    lines.append("VIOLATION CLASSES (page ids; classes overlap, METER is their union)")
    for cls in VIOLATION_CLASSES:
        if cls in DOCUMENT_LEVEL_VIOLATIONS:
            continue
        lines.append(f"  {len(rep.violations.get(cls, [])):7d}  {cls}")
        if cls == WARNING_UNMARKED:
            for mode, v in sorted(rep.warning_unmarked_by_failure_mode.items()):
                lines.append(f"  {len(v):7d}    failure_mode={mode}")
    lines += ["", "DOCUMENT-LEVEL FINDINGS (fail the exit status, not page ids, not in METER)"]
    lines += [f"  {len(rep.violations.get(c, [])):7d}  {c}" for c in DOCUMENT_LEVEL_VIOLATIONS]
    lines += ["", f"SURFACE AGREEMENT = {rep.surface_count()} disagreements"]
    lines += [f"  {len(rep.surface.get(c, [])):7d}  {c}" for c in SURFACE_CLASSES]
    lines += ["", "DIAGNOSTIC (not in METER, not in the exit status)"]
    lines += [f"  {len(rep.diagnostic.get(c, [])):7d}  {c}" for c in DIAGNOSTIC_CLASSES]
    lines += [
        "",
        "DOCUMENT-LEVEL (not in METER, not in the exit status)",
        f"  {len(rep.metadata_counts_missing):7d}  {METADATA_COUNTS_MISSING}"
        f" (metadata.json lacks `{METADATA_COUNTS_FIELD}`)",
    ]
    return "\n".join(lines)


def main_exit_code(rep: Report) -> int:
    return 1 if rep.violating_ids() or rep.surface_count() or rep.document_level_count() else 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="socr-acceptance",
        description="Count input pages that are neither CHECKED nor visibly MARKED (target 0).",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--inputs", type=Path, required=True, help="manifest.csv (out_file, n_pages)"
    )
    parser.add_argument("--report", type=Path, default=Path("acceptance.json"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    rep = measure(args.run_dir.expanduser(), read_inputs(args.inputs.expanduser()))
    args.report.write_text(json.dumps(rep.to_json(), indent=1))
    print(format_report(rep))
    return main_exit_code(rep)


if __name__ == "__main__":
    sys.exit(main())
