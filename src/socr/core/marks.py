"""#1145: the one place that knows what a visible MARK looks like.

A page socr did not CHECK must say so where the reader is looking: as the FIRST rendered
block of its page ``.md``, which ``assemble_pages`` then places directly under that page's
``## Page N`` heading.

This module is deliberately leaf-level. The mark-shape regexes are defined here and
``core/manifest.py`` imports them back under the names it used to define, so no caller's
import changes. The two note families compare against prose that ``manifest`` BUILDS, so
those two predicates import it lazily -- the same edge-thinning trick
``manifest._credential_sha256_text`` already uses, and the only way to keep equality
semantics without a cycle.

Provenance of the code below
----------------------------
Everything under "mark recognition" and "where a mark sits" is a **pure move** from
``benchmark/acceptance.py`` (#1130, merged in #1131 at ``2fb89d4d``), which now imports it
from here so there is exactly one copy. The move is behaviour-preserving by construction:
the predicates, the grammar and the CommonMark setup are unchanged, including
``_fixed_note_mark``/``_scanned_flag_mark`` comparing by EQUALITY against the rendered
note rather than by prefix. ``tests/test_acceptance_meter.py`` passing unchanged is the
pin that the move is pure.

The ONE deliberate behaviour change, requested with #1145: the ``unaccepted_output``
family is added, so #1142's ``> [page unaccepted: …]`` banner counts as a mark. Without it
the meter reads #1142's own marked pages as unmarked.

The #1145 additions -- the verdict vocabularies, the canonical-line builder, and the
coverage map -- are below, in their own section, and are new rather than moved.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from enum import Enum

from markdown_it import MarkdownIt
from markdown_it.token import Token

# --- mark shapes ------------------------------------------------------------------------
# Moved here from ``core/manifest.py``, which imports them back. Regex literals with no
# dependencies, so they keep this module leaf-level.

#: Every span socr itself authors INSIDE a page body: the fail-closed page markers and the
#: ``[socr: …]`` notes a sanitizer leaves where it removed something.
SOCR_MARKER_RE = re.compile(r"\[socr:[^\]\r\n]*\]|\[page \d+ failed:[^\]\r\n]+\]")
#: #1048: the one-line mark written above every table that ships unverified.
UNVERIFIED_TABLE_MARK: str = (
    "> [table unverified: socr could not check this table against the page; "
    "verify it against the page image before citing]"
)
UNVERIFIED_TABLE_MARK_RE = re.compile(r"^> \[table unverified: ", re.MULTILINE)
#: #1048 (owner ruling 2026-10-12): the stronger mark for a scanned table whose numbers or
#: labels a second reading disputes.
DISPUTED_TABLE_MARK_RE = re.compile(r"^> \[table disputed: ", re.MULTILINE)
#: #1048: the first line of a page that shipped a rejected, unchecked table.
PAGE_UNCHECKED_MARK_RE = re.compile(r"^> \[page unchecked: ")
#: #1129: the first line of a page whose kept model reading the page judge did not accept.
UNACCEPTED_OUTPUT_MARK_RE = re.compile(r"^> \[page unaccepted: ")
#: Provisional document-wide failure mark: a failed document's pages are MARKED only by
#: this at the top of its document ``.md``, never by a page or table mark.
DOCUMENT_FAILED_MARK_RE = re.compile(r"\[document failed: [^\]\r\n]+\]")


# --- mark recognition (pure move from benchmark/acceptance.py) ---------------------------
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


def _unaccepted_output_mark(_n: int, line: str) -> bool:
    """#1129's banner. The one family ADDED rather than moved (see the module docstring)."""
    return bool(UNACCEPTED_OUTPUT_MARK_RE.match(line))


#: ``[page N: flagged table kept — …]`` (kept_table_flag_note, #259): the variable tail has
#: no constant, so the fixed prefix is the recogniser.
FLAGGED_TABLE_PREFIX = "flagged table kept — "


def _flagged_table_mark(n: int, line: str) -> bool:
    return line.startswith(f"[page {n}: {FLAGGED_TABLE_PREFIX}") and line.endswith("]")


def _fixed_note_mark(n: int, line: str) -> bool:
    """Notes whose whole line is a function of the page number.

    Lazy import: these compare against prose ``manifest`` builds, and ``manifest`` imports
    this module. Compared by EQUALITY, exactly as the meter did -- a prefix match would be
    strictly more permissive and so would not be a pure move.
    """
    from socr.core.manifest import d3_superseded_note, judge_timeout_note

    return line in {judge_timeout_note(n), d3_superseded_note(n)}


def _scanned_flag_mark(n: int, line: str) -> bool:
    """This lane's recovery banners, EVERY form of them. Lazy import, as above.

    Iterates ``PROSE_RECOVERY_FLAGS`` rather than naming the two scan forms: #1144 adds a
    born-digital pair, and a form this family does not recognise makes a page that IS marked
    count as unmarked in the meter.
    """
    from socr.core.manifest import PROSE_RECOVERY_FLAGS

    return line in {flag.format(page_num=n) for flag in PROSE_RECOVERY_FLAGS}


#: #1050 step 2 (not emitted yet): recognised so the meter keeps working once it ships.
UNCOVERED_TEXT_NOTE = "uncovered text — see image"


def _uncovered_text_mark(n: int, line: str) -> bool:
    return line.startswith(f"[page {n}: {UNCOVERED_TEXT_NOTE}") and line.endswith("]")


#: Every visible mark family, with its emitter. The blank receipt (``[socr: blank page]``,
#: core/blank_page.py) and the fabricated-image receipt (core/url_provenance.py) are
#: receipts of what socr did, not statements that a check did not run: not marks.
MARK_FAMILIES: dict[str, Callable[[int, str], bool]] = {
    "canonical_unchecked": _canonical_mark,  # #1130 grammar, emitted from #1145
    "page_failed": _failed_mark,  # core/manifest.py page_failed_marker + floors
    "table_disputed_or_unverified": _table_mark,  # core/manifest.py #1048
    "page_unchecked": _page_unchecked_mark,  # core/manifest.py page_unchecked_mark
    "unaccepted_output": _unaccepted_output_mark,  # core/manifest.py #1129
    "flagged_table_kept": _flagged_table_mark,  # core/manifest.py kept_table_flag_note
    "fixed_note": _fixed_note_mark,  # judge_timeout_note (#713), d3_superseded_note
    "scanned_flag": _scanned_flag_mark,  # SCANNED_*_FLAG banners
    "uncovered_text": _uncovered_text_mark,  # #1050 step 2
}


def line_is_mark(line: str, page_num: int) -> bool:
    return any(fam(page_num, line) for fam in MARK_FAMILIES.values())


def families_of(line: str, page_num: int) -> frozenset[str]:
    """#1145: which families recognise this line. A line may match more than one."""
    return frozenset(name for name, fam in MARK_FAMILIES.items() if fam(page_num, line))


# --- where a mark sits (pure move from benchmark/acceptance.py) --------------------------
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
    """MARKED is position-fixed: the first rendered block of *text* is made of mark lines.

    A blockquote and a paragraph are different CommonMark blocks, so a leading block is
    homogeneous: a canonical (quoted) line above a legacy unquoted marker leaves that
    marker in its own block below -- which is where a per-table marker belongs anyway.
    """
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


# ========================================================================================
# #1145 additions below: new code, not part of the move.
# ========================================================================================

# --- the verdict vocabularies -----------------------------------------------------------


class Check(str, Enum):
    """Which check did not pass. One member per thing socr can actually check.

    Rendered into ``<check>``, so a value must contain no ``]``, ``;`` or ``=`` -- which
    ``_canonical_mark`` enforces, and ``test_every_check_renders`` pins for every member.
    """

    PAGE_JUDGE = "page_judge"
    SOURCE_EVIDENCE = "source_evidence"
    NATIVE_TABLE_VERIFIER = "native_table_verifier"
    NATIVE_COVERAGE = "native_coverage"
    FIGURE_TEXT = "figure_text"
    CHART_REGION = "chart_region"
    TABLE_STRUCTURE = "table_structure"
    NATIVE_GLYPHS = "native_glyphs"
    EQUATION_FIDELITY = "equation_fidelity"
    TABLE_LABEL = "table_label"
    PAGE_PROCESSED = "page_processed"
    UNREADABLE_INPUT = "unreadable_input"
    #: The fail-closed fallback: the page is not CHECKED and nothing more specific named
    #: the reason. Never silence it -- a page with no clause at all ships unmarked, which
    #: is the defect this ticket exists to remove.
    UNKNOWN_CHECK = "unknown"


class CheckResult(str, Enum):
    """PASS carries no mark; FAIL and UNKNOWN each carry one.

    FAIL means the check ran and the page did not pass it. UNKNOWN means it could not run
    or could not decide. The distinction is the reader's, not bookkeeping: UNKNOWN on a
    page whose glyphs are measurably damaged would say socr never looked, which is false.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


#: Rendered into ``<scope>``: ``page``, ``document``, or a region with its ordinal
#: (``table:3``). The ordinal is load-bearing: a page may hold one withheld table and one
#: rendered, corrupted table, and a bare ``scope=table`` would impugn whichever the reader
#: happened to look at.
SCOPE_PAGE = "page"
SCOPE_DOCUMENT = "document"


def region_scope(kind: str, ordinal: int) -> str:
    """``table:3`` -- a region-identified scope. ``kind`` is table/figure/equation."""
    return f"{kind}:{ordinal}"


#: Position-anchored canonical line, for finding one without knowing the page number.
CANONICAL_MARK_RE = re.compile(r"^> \[page \d+ unchecked: ")


def is_canonical_mark(line: str, page_num: int) -> bool:
    """Public accessor for the canonical grammar, argument order as a caller reads it.

    ``_canonical_mark`` keeps the ``(n, line)`` signature every ``MARK_FAMILIES`` predicate
    shares. Callers outside this module use this name instead of the private one, so a
    cross-package private import never has to exist.
    """
    return _canonical_mark(page_num, line)


_REASON_CLEAN_RE = re.compile(r"\s+")


def clean_reason(reason: str) -> str:
    """Flatten a reason to one line and neutralise the characters the grammar reserves.

    ``]`` would close the span early, ``;`` would read as another field, a newline would
    end the line. Mirrors ``manifest.socr_marker``'s guarantee: the builder cannot emit a
    line its own recogniser would miss.
    """
    flattened = _REASON_CLEAN_RE.sub(" ", str(reason or "")).strip()
    flattened = flattened.replace("]", ")").replace(";", ",")
    return flattened or "no reason was recorded"


def page_unchecked_clause_mark(
    page_num: int, check: Check | str, result: CheckResult | str, scope: str, reason: str
) -> str:
    """Render one canonical line. Guaranteed to satisfy :func:`_canonical_mark`."""
    check_val = check.value if isinstance(check, Check) else str(check)
    result_val = result.value if isinstance(result, CheckResult) else str(result)
    if result_val == CheckResult.PASS.value:
        raise ValueError("a PASS check carries no mark")
    return (
        f"> [page {page_num} unchecked: {check_val}={result_val}; "
        f"scope={scope}; reason={clean_reason(reason)}]"
    )


#: The canonical line's own fields, for reading back what one already on the page STATES.
_CANONICAL_FIELDS_RE = re.compile(
    r"^> \[page (?P<page>\d+) unchecked: (?P<check>[a-z_]+)=(?P<result>[A-Z]+); "
    r"scope=(?P<scope>[^;\]]+); reason="
)


def parse_canonical_mark(line: str, page_num: int) -> tuple[Check, str] | None:
    """The ``(check, scope)`` a canonical line names, or ``None`` if it is not one.

    Why parse rather than treat the family as covering everything: a canonical line states
    exactly ONE check about ONE region, and that is all it may suppress. Covering
    ``frozenset(Check)`` meant the first line written to a two-clause page silenced the
    second -- on a re-finalization the page would keep line one and lose line two, and the
    reader would never learn the second fact.

    Idempotence is unchanged, and this is why: ``leading_mark_blocks`` returns EVERY leading
    mark line, so on a second pass both lines are present and each covers its own pair.

    An unparseable or wrong-page line covers nothing, which is the fail-closed direction:
    the clause gets its line rather than being silently swallowed.
    """
    m = _CANONICAL_FIELDS_RE.match(line.strip())
    if m is None or int(m.group("page")) != page_num:
        return None
    try:
        check = Check(m.group("check"))
    except ValueError:
        return None
    return check, m.group("scope").strip()


# --- what a legacy mark already states --------------------------------------------------
#: Which checks each family's line actually DISCLOSES, so a clause it already states is not
#: repeated. FAIL-CLOSED on purpose, in the shape of #1129's
#: ``_UNACCEPTED_OUTPUT_MARK_EXCLUDED_PROVENANCES``: a family ABSENT from this map
#: suppresses NOTHING, so a mark shape added later still gets its canonical line rather
#: than silently swallowing one.
#:
#: Suppression is matched by the caller on ``(check, region)``, never per page: a page may
#: ship one withheld table and one rendered, corrupted table, and a per-page rule would let
#: the first silence the second.
_TABLE_CHECKS: frozenset[Check] = frozenset(
    {
        Check.SOURCE_EVIDENCE,
        Check.NATIVE_TABLE_VERIFIER,
        Check.TABLE_STRUCTURE,
        Check.TABLE_LABEL,
    }
)

FAMILY_COVERAGE: dict[str, frozenset[Check]] = {
    # "canonical_unchecked" is deliberately ABSENT: a canonical line states one check about
    # one region and may suppress only that pair, which ``parse_canonical_mark`` reads off
    # the line itself. It was ``frozenset(Check)``, which let the first line written to a
    # two-clause page silence the second.
    "table_disputed_or_unverified": _TABLE_CHECKS,
    "page_unchecked": _TABLE_CHECKS,
    "flagged_table_kept": _TABLE_CHECKS,
    "scanned_flag": _TABLE_CHECKS,
    "fixed_note": _TABLE_CHECKS,
    "unaccepted_output": frozenset({Check.PAGE_JUDGE}),
    "uncovered_text": frozenset({Check.NATIVE_COVERAGE}),
    # "page_failed" is deliberately ABSENT: what it covers depends on whether the page
    # shipped anything else, which only the caller knows. See ``page_failed_coverage``.
}


def page_failed_coverage(ships_nothing_else: bool) -> frozenset[Check]:
    """What a ``[page N failed: …]`` line states.

    When the body IS the marker plus at most an image, no content shipped and there is
    nothing left for another check to be wrong about: it covers everything. Otherwise it is
    a per-table D3 marker standing where one table was withheld, and the page's prose ships
    beside it -- measured on run 23117187, 192 of 1,290 such pages ship real prose, many
    carrying unrelated native-table defects. Treating those as fully covered would silence a
    live clause on a page a reader is about to cite.
    """
    return frozenset(Check) if ships_nothing_else else _TABLE_CHECKS


def leading_mark_blocks(text: str, page_num: int) -> list[str]:
    """Every mark line in the page's leading run of all-mark blocks, in order.

    ``leading_marks`` answers the meter's question -- is the FIRST rendered block made of
    marks -- and that is all MARKED requires. But the seam separates banners with a blank
    line, so a page with two clauses carries two consecutive single-line blockquote blocks,
    and reading only the first would:

    * let the second clause be re-emitted on every re-finalize, duplicating it; and
    * leave a test helper stripping one mark and comparing a body that still holds another.

    So this walks forward while each successive rendered block is entirely marks, and
    returns their lines. Idempotence of the written bytes depends on it.
    """
    tokens = parse_blocks(text)
    lines: list[str] = []
    idx = 0
    while idx < len(tokens):
        if tokens[idx].type == "paragraph_open":
            end = idx + 3
        elif tokens[idx].type == "blockquote_open":
            end = next(
                (j + 1 for j in range(idx, len(tokens)) if tokens[j].type == "blockquote_close"),
                len(tokens),
            )
        else:
            break
        block = first_block_marks(tokens[idx:end], page_num)
        if not block:
            break
        lines.extend(block)
        idx = end
    return lines
