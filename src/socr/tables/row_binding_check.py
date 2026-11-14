"""Row-binding check for model-read tables on born-digital pages (#1267).

Question answered: does every shipped table row carry the label that the page's own
text layer prints with those values?  The model can empty a parent row and slide its
values one label down (2003-09-16 greenbook p22), blank a label (2010-08-10 tealbook
p68) or drop the values of a labelled row (2003-08-12 greenbook p42).  Value-only
checks (multiset, row corroboration) and the per-cell binder miss these because they
never compare the LABEL with the line the values sit on, and the binder sees a
rotated page in the wrong frame.

This module is pure: words in the READING frame (``native_first.upright_words_for_page``)
plus the shipped markdown in, per-row verdicts out.  The pipeline calls it at the agentic flush,
on the finalized record (``UnifiedPipeline._run_row_binding_check``).

Verdicts use positive geometry only.  A row is convicted (DISPLACED / LABEL_LOST /
EMPTIED) only when the text layer shows a specific contradiction; anything the layer
cannot settle (no line, ambiguous assignment, label unmatched) is UNVERIFIABLE and
never convicts.

Each row's values are assigned to native lines by value-sequence equality with a
ONE-TO-ONE, ORDER-PRESERVING assignment.  A row that stays compatible with several
lines whose verdicts differ is UNVERIFIABLE (identical rows, all-zero rows).
"""

from __future__ import annotations

import html
import re
import statistics
import unicodedata
from dataclasses import dataclass
from enum import Enum

from socr.tables.native_verifier import _normalize_binding_numeric_token, is_numeric_token
from socr.tables.reconcile import _is_separator_row, _parse_grid, _split_row, find_table_blocks

# ---------------------------------------------------------------------------
# Named constants.  Every one is taken from the frame-1 comparator measurements in
# scratch/army/precision/scripts/compare.py (same words, same pages), cited per line.
# ---------------------------------------------------------------------------

# Faux-bold / double-painted text repeats a word within ~1 pt of the original (comparator
# DUP_OVERPRINT_TOL_HEIGHTS); a repeat within this fraction of the word height is one word.
DUP_OVERPRINT_TOL_HEIGHTS = 0.25
# A word joins a line when its centre is within this fraction of the smaller word height of
# the line's mean centre (comparator LINE_CENTRE_TOL; growing the union extent instead merges
# tightly set Greenbook rows, measured there).
LINE_CENTRE_TOL = 0.5
# A gap wider than this many median word widths (along the reading axis) splits a line into
# panels/segments (comparator SEGMENT_GAP_HEIGHTS, which also measures along the reading axis;
# a normal space is well under one, twin panels exceed 3).
SEGMENT_GAP_WIDTHS = 3.0
# An EMPTIED verdict needs a native label line carrying at least this many values: a single number
# on a label line is a footnote mark, year or list number; a dropped data row has a full set
# (comparator EMPTIED_MIN_NUMBERS: 252 raw hits with 1 number were footnotes and group headers).
EMPTIED_MIN_VALUES = 2
# A row is convicted only when it has at least this many values (numbers, or categorical cells when it
# has no numbers).  One value matches too many lines to say which line the row sits on: the single-value
# header bands (`2010`, `H2`, `Four-quarter changes`) were the only control convictions left on frame 1
# after change 1 (measure/CHANGELOG.md, change 2).  Same floor as EMPTIED_MIN_VALUES, same reason.
MIN_VALUES_TO_CONVICT = 2
# A footnote mark glued to a label ('Receipts2' vs 'Receipts') is at most this many trailing digits
# (frame 1: 'Receipts2', 'Previous2', 'Other3', 'Gross saving less gross investment4': one digit).
MAX_FOOTNOTE_DIGITS = 1

_SUPERSCRIPT_TRANSLATE = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_UNICODE_MINUS = str.maketrans({"−": "-", "–": "-", "—": "-", "‒": "-"})
_MD_DECORATION = re.compile(r"[*_`]|<br\s*/?>")  # value cells only (`**1.2**` is 1.2)
# Structural markdown in a SHIPPED label (round 5, R1): paired emphasis and code spans, line breaks,
# HTML entities. An unpaired `*` or backtick is a literal character of the label (`A*` != `A**`).
_PAIRED_MARKUP = (
    re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*"),
    re.compile(r"(?<![*\w])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![*\w])"),
    re.compile(r"`([^`]+)`"),
)


def _strip_markup(text: str) -> str:
    """Remove structural markdown from a shipped label; literal characters stay."""
    s = re.sub(r"<br\s*/?>", " ", text)
    for pat in _PAIRED_MARKUP:
        s = pat.sub(r"\1", s)
    return html.unescape(s)


# A footnote mark after a label: in parentheses ('United Kingdom (4)'), superscript ('Contributions¹')
# or LaTeX ('Consumer prices$^{2}$'). Its digit is glued to the label word before it ('kingdom4'), on
# both sides, so ``_token_eq``'s trailing-footnote rule applies: a side that omits the mark still
# matches, and when BOTH sides carry one the digits must agree (`Previous¹` / `Previous²`, `Model (1)` /
# `Model (2)` are different rows). A bare digit stays a token of its own ('Tier 1' / 'Tier 2').
_PAREN_FOOTNOTE = re.compile(
    rf"\(\d{{1,{MAX_FOOTNOTE_DIGITS}}}\)"
    rf"|\$\^\{{?\d{{1,{MAX_FOOTNOTE_DIGITS}}}\}}?\$"
    rf"|[\u00b9\u00b2\u00b3\u2070\u2074-\u2079]{{1,{MAX_FOOTNOTE_DIGITS}}}"
)


class RowVerdict(str, Enum):
    BOUND = "bound"
    DISPLACED = "displaced"
    LABEL_LOST = "label_lost"
    EMPTIED = "emptied"
    UNVERIFIABLE = "unverifiable"


class TableOutcome(str, Enum):
    BOUND = "bound"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


CONVICTING = frozenset({RowVerdict.DISPLACED, RowVerdict.LABEL_LOST, RowVerdict.EMPTIED})


@dataclass(frozen=True)
class RowEvidence:
    """One shipped data row and what the text layer says about it."""

    line_index: int  # line index of the row in the table markdown given to the check
    verdict: RowVerdict
    reason: str
    shipped_label: str
    native_label: str  # label of the native line the row was assigned to ("" when none)
    values: tuple[str, ...]  # shipped value tokens (numeric) or cell texts (categorical)
    native_line: int | None  # index of the assigned native line, None when unassigned
    evidence: str = ""  # the positive evidence kind that bound the row, "" for the same-line rule


@dataclass(frozen=True)
class RowBindingResult:
    outcome: TableOutcome
    rows: tuple[RowEvidence, ...]

    def count(self, verdict: RowVerdict) -> int:
        return sum(1 for r in self.rows if r.verdict is verdict)

    def evidence_counts(self) -> dict[str, int]:
        """BOUND rows per positive evidence kind (#1257); the same-line rule is not listed."""
        out: dict[str, int] = {}
        for r in self.rows:
            if r.verdict is RowVerdict.BOUND and r.evidence:
                out[r.evidence] = out.get(r.evidence, 0) + 1
        return out

    def bound_count(self) -> int:
        return self.count(RowVerdict.BOUND)


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def _alnum(text: str) -> str:
    """Lower-cased letters, digits and combining marks of any script; everything else (spaces,
    punctuation, symbols) is dropped.  NFKC turns '²' into '2' and expands ligature glyphs.

    Label identity rule (merge gate round 3, Astra): two labels that differ in any letter, digit or
    combining mark never normalise to the same token.  Categories L*, N*, M* are kept: a filter on
    ``\\w`` dropped Devanagari vowel signs (`कुल` and `काल` both became `कल`).  An earlier rule that
    collapsed fi/fl/ff on both sides merged labels differing in a letter and is gone."""
    s = unicodedata.normalize("NFKC", text.translate(_SUPERSCRIPT_TRANSLATE)).lower()
    return "".join(c for c in s if unicodedata.category(c)[0] in "LNM")


_FOOTNOTE_AFTER_WORD = re.compile(rf"(?<=[^\s|])[\s*_`]*({_PAREN_FOOTNOTE.pattern})")


def _footnote_digits(mark: str) -> str:
    return re.sub(r"\D", "", mark.translate(_SUPERSCRIPT_TRANSLATE))


def _label_tokens(text: str) -> list[str]:
    text = _FOOTNOTE_AFTER_WORD.sub(lambda m: _footnote_digits(m.group(1)), _strip_markup(text))
    # Same per-word normalisation as a text-layer word (`Series_A` is one word on both sides).
    return [t for t in (_alnum(w) for w in text.split()) if t]


def _has_letter(tokens: list[str]) -> bool:
    return any(c.isalpha() for t in tokens for c in t)


def _token_eq(a: str, b: str) -> bool:
    if a == b:
        return True
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    rest = long_[len(short) :]
    return (
        bool(short)
        and long_.startswith(short)
        and rest.isdigit()
        and len(rest) <= MAX_FOOTNOTE_DIGITS
    )


def _tokens_in_pool(tokens: list[str], pool: list[str]) -> bool:
    """Every label token occurs in *pool*, short ones included.

    Merge gate round 2 (#1271, Astra P0): short tokens are often the whole identity of a row
    (`US government` / `UK government`); filtering them out bound swapped rows."""
    return bool(tokens) and all(any(_token_eq(t, p) for p in pool) for t in tokens)


def _canon_number(text: str) -> str | None:
    """Canonical numeric value of one whitespace-free token, None when it is not a number.

    ``is_numeric_token`` / ``_normalize_binding_numeric_token`` already accept leading-dot values
    ('.6' -> '0.6', '-.4' -> '-0.4'); the trailing '%' and '.' are presentation only."""
    t = _MD_DECORATION.sub("", text.translate(_UNICODE_MINUS)).strip()
    if not t or not is_numeric_token(t):
        return None
    c = _normalize_binding_numeric_token(t).lstrip("+").rstrip("%")
    c = c.rstrip(".")
    return c or None


# ---------------------------------------------------------------------------
# Native lines
# ---------------------------------------------------------------------------


@dataclass
class _Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    num: str | None
    tok: str  # alnum of text


@dataclass
class _Line:
    index: int
    words: list[_Word]

    @property
    def nums(self) -> list[str]:
        return [w.num for w in self.words if w.num is not None]


@dataclass
class _Cand:
    """A native line, or one gap-separated panel of it, offered to a shipped row."""

    line: _Line
    start: int  # index of the first word of this panel in line.words
    stop: int  # one past the last
    key: tuple[int, float]  # reading order: (line index, x of the panel's first word)

    @property
    def words(self) -> list[_Word]:
        return self.line.words[self.start : self.stop]


def _dedupe_overprint(words: list) -> list:
    kept: list = []
    for w in words:
        tol = DUP_OVERPRINT_TOL_HEIGHTS * (w[3] - w[1])
        if not any(
            k[4] == w[4] and abs(k[0] - w[0]) <= tol and abs(k[1] - w[1]) <= tol for k in kept
        ):
            kept.append(w)
    return kept


def _in_region(w, region) -> bool:
    if region is None:
        return True
    cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def build_native_lines(
    words: list, region: tuple[float, float, float, float] | None
) -> list[_Line]:
    """Group reading-frame *words* inside *region* into lines, top to bottom, words left to right."""
    kept = [
        w for w in _dedupe_overprint(list(words)) if str(w[4]).strip() and _in_region(w, region)
    ]
    order = sorted(kept, key=lambda w: (w[1] + w[3]) / 2)
    groups: list[dict] = []
    for w in order:
        c, h = (w[1] + w[3]) / 2, w[3] - w[1]
        g = groups[-1] if groups else None
        if g is not None:
            mean_c = sum(g["cs"]) / len(g["cs"])
            if abs(c - mean_c) <= LINE_CENTRE_TOL * min(statistics.median(g["hs"]), h):
                g["words"].append(w)
                g["cs"].append(c)
                g["hs"].append(h)
                continue
        groups.append({"words": [w], "cs": [c], "hs": [h]})
    widths = [w[2] - w[0] for w in kept]
    med = statistics.median(widths) if widths else 1.0
    lines: list[_Line] = []
    for i, g in enumerate(groups):
        ws = sorted(g["words"], key=lambda w: w[0])
        out: list[_Word] = []
        for w in ws:
            text = str(w[4])
            prev = out[-1] if out else None
            # Round 3 (Fable P2): a footnote mark printed as its own word right after a label word
            # (`Model` `(1)`; no panel gap between them) is not a value (`(1)` reads as -1): its digit
            # joins the label word, exactly as on the shipped side.
            if (
                prev is not None
                and _PAREN_FOOTNOTE.fullmatch(text.strip())
                and prev.num is None
                and _has_letter([prev.tok])
                and w[0] - prev.x1 <= SEGMENT_GAP_WIDTHS * med
            ):
                prev.tok += _footnote_digits(text.strip())
                out.append(_Word(w[0], w[1], w[2], w[3], text, None, ""))
                continue
            out.append(_Word(w[0], w[1], w[2], w[3], text, _canon_number(text), _alnum(text)))
        lines.append(_Line(index=i, words=out))
    return lines


def _candidates(lines: list[_Line]) -> list[_Cand]:
    widths = [w.x1 - w.x0 for ln in lines for w in ln.words]
    med = statistics.median(widths) if widths else 1.0
    out: list[_Cand] = []
    for ln in lines:
        n = len(ln.words)
        if not n:
            continue
        out.append(_Cand(ln, 0, n, (ln.index, ln.words[0].x0)))
        start = 0
        segs: list[tuple[int, int]] = []
        for i in range(1, n + 1):
            if i == n or ln.words[i].x0 - ln.words[i - 1].x1 > SEGMENT_GAP_WIDTHS * med:
                segs.append((start, i))
                start = i
        # A gap-separated stretch without a value is a label, it belongs to the panel that follows it.
        merged: list[tuple[int, int]] = []
        carry: int | None = None
        for a, b in segs:
            if all(w.num is None for w in ln.words[a:b]) and (a, b) != segs[-1]:
                carry = a if carry is None else carry
                continue
            merged.append((a if carry is None else carry, b))
            carry = None
        if len(merged) > 1:
            out.extend(_Cand(ln, a, b, (ln.index, ln.words[a].x0)) for a, b in merged)
    return out


# ---------------------------------------------------------------------------
# Shipped rows
# ---------------------------------------------------------------------------


@dataclass
class _Row:
    line_index: int
    label: str
    label_tokens: list[str]
    numbers: tuple[str, ...]  # canonical numeric values of the data cells
    categories: tuple[str, ...]  # alnum of non-numeric data cells (used when there are no numbers)
    cell_texts: tuple[str, ...]
    is_header: bool = False  # the markdown header row (judged apart, see check_row_binding)
    values_only: bool = True  # every value cell holds numbers only (no words beside them)
    cells_raw: tuple[str, ...] = ()  # full text of every non-empty data cell, in order
    cells_all: tuple[str, ...] = ()  # every data cell in column order, empty ones included

    @property
    def has_content(self) -> bool:
        return bool(self.numbers or self.categories)


def _shipped_rows(markdown: str) -> list[_Row]:
    lines = markdown.splitlines()
    rows: list[_Row] = []
    for block in find_table_blocks(markdown):
        for li in range(block.start, block.end + 1):
            grid = _parse_grid([lines[li]])
            if not grid:
                # The parser reads any dash-only line as a separator, so this keeps only the real
                # ones out (Astra P1, rounds 12-13): an all-empty line, a strict `---` delimiter
                # anywhere, the block's own second line, and a leading line with no dash-only line
                # under it. Any other dash-only line is content and is judged.
                raw = _split_row(lines[li])
                if (
                    not any(raw)
                    or _is_separator_row(raw)
                    or li == block.start + 1
                    or (
                        li == block.start
                        and (
                            li == block.end
                            or not any(_split_row(lines[li + 1]))
                            or bool(_parse_grid([lines[li + 1]]))
                        )
                    )
                ):
                    continue
                grid = [raw]
            cells = grid[0]
            # The header is the block's first line by position: an all-empty heading line parses to no
            # grid, and the first DATA row must not be taken for the header in its place.
            is_header = li == block.start
            label = cells[0] if cells else ""
            numbers: list[str] = []
            cats: list[str] = []
            texts: list[str] = []
            values_only = True
            for cell in cells[1:]:
                toks = _MD_DECORATION.sub(" ", cell.translate(_UNICODE_MINUS)).split()
                nums = [n for n in (_canon_number(t) for t in toks) if n is not None]
                if nums:  # '± 0.8' is the value 0.8
                    if any(_canon_number(t) is None and t != "±" for t in toks):
                        values_only = False  # `(March 24 -`: a number inside heading words
                    numbers.extend(nums)
                    texts.extend(t for t in toks if _canon_number(t) is not None)
                else:
                    a = _alnum(cell)
                    if a and _has_letter([a]):
                        cats.append(a)
                        texts.append(cell.strip())
            rows.append(
                _Row(
                    line_index=li,
                    label=label,
                    label_tokens=_label_tokens(label),
                    numbers=tuple(numbers),
                    categories=tuple(cats),
                    cell_texts=tuple(texts),
                    is_header=is_header,
                    values_only=values_only,
                    cells_raw=tuple(c.strip() for c in cells[1:] if c.strip()),
                    cells_all=tuple(c.strip() for c in cells[1:]),
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Matching a row to native candidates
# ---------------------------------------------------------------------------


def _label_words_before(cand: _Cand, first_value_word: int) -> list[_Word]:
    return cand.words[: first_value_word - cand.start]


def _numeric_options(row: _Row, cand: _Cand) -> list[int]:
    """Absolute word index of the first value word, for each way the row's numbers equal the panel's."""
    idx = [i for i in range(cand.start, cand.stop) if cand.line.words[i].num is not None]
    nums = tuple(cand.line.words[i].num for i in idx)
    hits: list[int] = []
    if nums == row.numbers:
        hits.append(idx[0])
    # numeric first word is the row label (year, quarter, participant id): drop it from the values.
    if idx and idx[0] == cand.start and nums[1:] == row.numbers and len(nums) > 1:
        hits.append(idx[1])
    return hits


def _category_option(row: _Row, cand: _Cand) -> int | None:
    """Absolute word index where the categorical cells start, if they appear in order on the panel."""
    ws = cand.words
    offsets: list[int] = []
    pos = 0
    text = ""
    for w in ws:
        offsets.append(pos)
        text += w.tok
        pos += len(w.tok)
    at = 0
    first_at = None
    for c in row.categories:
        j = text.find(c, at)
        if j < 0:
            return None
        first_at = j if first_at is None else first_at
        at = j + len(c)
    wi = max(i for i, o in enumerate(offsets) if o <= (first_at or 0))
    return cand.start + wi


def _options(row: _Row, cands: list[_Cand]) -> list[tuple[_Cand, int]]:
    out: list[tuple[_Cand, int]] = []
    for c in cands:
        if row.numbers:
            out.extend((c, wi) for wi in _numeric_options(row, c))
        elif row.categories:
            wi = _category_option(row, c)
            if wi is not None:
                out.append((c, wi))
    return out


def _feasible(opts: list[list[tuple[_Cand, int]]]) -> list[list[tuple[_Cand, int]]] | None:
    """Options of each optioned row that lie on some strictly increasing, one-to-one assignment of
    ALL optioned rows.  None when no such assignment exists (the markdown order differs from reading order)."""
    n = len(opts)
    if not n:
        return []
    fwd = [[False] * len(o) for o in opts]
    bwd = [[False] * len(o) for o in opts]
    fwd[0] = [True] * len(opts[0])
    for k in range(1, n):
        for j, (c, _) in enumerate(opts[k]):
            fwd[k][j] = any(fwd[k - 1][i] and p.key < c.key for i, (p, _) in enumerate(opts[k - 1]))
    bwd[n - 1] = [True] * len(opts[n - 1])
    for k in range(n - 2, -1, -1):
        for j, (c, _) in enumerate(opts[k]):
            bwd[k][j] = any(bwd[k + 1][i] and c.key < p.key for i, (p, _) in enumerate(opts[k + 1]))
    kept = [[o for j, o in enumerate(opts[k]) if fwd[k][j] and bwd[k][j]] for k in range(n)]
    return None if any(not k for k in kept) else kept


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def _label_only(line: _Line) -> bool:
    return not line.nums


def _pool(lines: list[_Line], line: _Line, own: list[_Word]) -> list[str]:
    """Tokens that count as the label of *line*: its own label words plus label-only neighbour bands
    (wrapped labels, 'Previous' / 'Receipts2' sub-bands printed on their own baseline)."""
    toks = [w.tok for w in own if w.tok]
    for d in (-1, 1):
        j = line.index + d
        if 0 <= j < len(lines) and _label_only(lines[j]):
            toks.extend(w.tok for w in lines[j].words if w.tok)
    return toks


def _same_label(a: list[str], b: list[str]) -> bool:
    """The same label: the same tokens in the same order (each pair equal under ``_token_eq``, which
    allows only a one-sided trailing footnote digit).

    Merge gate round 3: set equality merged labels that differ in word order or repetition
    (`Exports less imports` / `Imports less exports`), found by the corpus label property check."""
    return bool(a) and len(a) == len(b) and all(_token_eq(x, y) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# BOUND needs exact evidence (merge gate round 4, burden of proof inverted)
# ---------------------------------------------------------------------------
# Every rule above may convict (with positive geometry) or leave a row UNVERIFIABLE. None of them may
# grant BOUND on its own: four review rounds each found a new pair of distinct labels that a tolerant
# rule (token pools, adjacency bands, subsets, trailing digits, dropped symbols, header skips) declared
# the same. A row is BOUND only when its shipped label IS the label printed before its values on the
# same text-layer line, under the minimal normalisation of ``_exact_label``.
_FOOTNOTE_TAIL = re.compile(rf"\s*(?:{_PAREN_FOOTNOTE.pattern}|\*+)$")


def _exact_label(text: str, *, shipped: bool) -> str:
    """NFC, casefold, whitespace collapsed; on the SHIPPED side structural markdown is removed first
    (round 5, R1: asymmetric; the text layer has no markdown, so its `*` is a character). Every letter,
    digit and symbol (%, $, &, (), -, /, a literal *) is kept."""
    s = _strip_markup(text) if shipped else text
    return " ".join(unicodedata.normalize("NFC", s).casefold().split())


def _labels_exact(shipped: str, native: str, *, strict: bool = False) -> bool:
    """The same label string. A trailing footnote mark (incl. a run of `*`) is dropped only from the
    side that has one when the other has none, and never when it is the whole label; then one
    trailing period or colon is dropped from each side."""
    a, b = _exact_label(shipped, shipped=True), _exact_label(native, shipped=False)
    fa, fb = (None, None) if strict else (_FOOTNOTE_TAIL.search(a), _FOOTNOTE_TAIL.search(b))
    if fa and not fb and a[: fa.start()].strip():
        a = a[: fa.start()]
    elif fb and not fa and b[: fb.start()].strip():
        b = b[: fb.start()]
    a = a[:-1].rstrip() if a[-1:] in ".:" else a
    b = b[:-1].rstrip() if b[-1:] in ".:" else b
    return a == b


def _bound_exactly(label: str, printed: list[str], lines: list[_Line]) -> bool:
    """BOUND gate: *label* equals the label printed on every candidate line, footnote marks included.

    Round 6 (top-orchestrator decision): no footnote tolerance on the BOUND path at all. Any footnote
    difference (`Total` / `Total*`, `2020 Total` / `2020 Total*`, `Receipts` / `Receipts¹`) leaves the
    row UNVERIFIABLE. The conviction rules never used this gate, so convictions are unaffected."""
    del lines  # kept in the signature: callers pass the page lines
    return bool(printed) and all(_labels_exact(label, t, strict=True) for t in printed)


def _label_cell_empty(row: _Row) -> bool:
    """Round 7 (Astra P1): a label cell is empty only when nothing but whitespace or markup is in it. A
    symbol label (`%`, `$`) has no letter or digit tokens but is a label, judged by the exact gate."""
    return not _strip_markup(row.label).strip()


def _symbol_cells_only(row: _Row) -> bool:
    """Round 10 (Astra P1): a row with no numbers or words can still print `$` or `%` in its data
    cells. Those cells count as neither, so such a row must not pass as blank or valueless."""
    return not row.has_content and any(_strip_markup(c).strip() for c in row.cells_raw)


def _same_word(native: str, shipped: str) -> bool:
    n = " ".join(unicodedata.normalize("NFC", native).casefold().split())
    t = " ".join(unicodedata.normalize("NFC", _strip_markup(shipped)).casefold().split())
    if n == t:
        return True
    cn, ct = _canon_number(native), _canon_number(shipped)
    return cn is not None and cn == ct


def _cells_cover_line(row: _Row, cand: _Cand, first_value_word: int) -> bool:
    """Round 8 positional rule (BOUND path only). The shipped row's data cells, word by word and in
    order, must match the candidate line's words from the row's first value onward, with NO native
    word between the first and the last matched one that a shipped cell does not match (a second
    stub, a symbol, a suffix, `n.a.`, a carried label); native text after the last matched word
    (running heads, page numbers) is allowed. Cells are compared on their full text (`1.8 to 2.1`)."""
    shipped = [t for c in row.cells_raw for t in _strip_markup(c).translate(_UNICODE_MINUS).split()]
    native = cand.line.words[first_value_word : cand.stop]
    j = 0
    for w in native:
        if j == len(shipped):
            return True  # trailing native text after the last matched word
        if not _same_word(w.text.translate(_UNICODE_MINUS), shipped[j]):
            return False
        j += 1
    return j == len(shipped)


def _printed_label(cand: _Cand, first_value_word: int) -> str:
    """The words printed before the row's first value on its candidate line (panel)."""
    return " ".join(w.text for w in cand.words[: first_value_word - cand.start])


def _band_tokens(lines: list[_Line], line: _Line) -> list[str]:
    """Tokens of the label-only bands directly above and below *line* (wrapped-label continuations)."""
    toks: list[str] = []
    for d in (-1, 1):
        j = line.index + d
        if 0 <= j < len(lines) and _label_only(lines[j]):
            toks.extend(w.tok for w in lines[j].words if w.tok)
    return toks


def _own_label(line: _Line) -> list[_Word]:
    out = []
    for w in line.words:
        if w.num is not None:
            break
        out.append(w)
    return out


def _adjacent_valued(lines: list[_Line], line: _Line, categorical: bool) -> list[_Line]:
    """Nearest value-carrying line above and below (the adjacent line when the row is categorical,
    because its neighbour's cells are words, not numbers)."""
    out = []
    for step in (-1, 1):
        j = line.index + step
        while 0 <= j < len(lines):
            if lines[j].nums or categorical:
                out.append(lines[j])
                break
            j += step
    return out


def _judge(row: _Row, cand: _Cand, first_value_word: int, lines: list[_Line], prev: _Row | None):
    line = cand.line
    own = _label_words_before(cand, first_value_word)
    native_label = " ".join(w.text for w in own)
    native_tokens = [w.tok for w in own if w.tok]
    # Round 8 (Fable P2): a symbol-only label (`•`, `%`) on a line whose own label has words is judged
    # exactly like an empty label (LABEL_LOST when the line prints a label).
    if _label_cell_empty(row) or (not row.label_tokens and _has_letter(native_tokens)):
        # The label text may sit in another cell of the row (bullet rows) or the row may be a
        # header continuation (Q1 Q2 Q3, H1 H2): words in a data cell are not a lost label.
        if _has_letter(native_tokens) and not row.categories:
            # post-held-out (2004-12-14 greenbook p28): markdown split a wrapped label over two rows,
            # the labelled all-empty row above holds the native label of this line.
            if (
                prev is not None
                and not prev.has_content
                and prev.label_tokens
                and _tokens_in_pool(native_tokens, prev.label_tokens)
            ):
                # 1B: a real wrap continuation adds something the adjoining band does not carry
                # (p28: `investment3` is not on the `Gross saving less gross` band). A line whose own
                # label sits wholly inside the band is a data line under a header, not a wrapped label.
                if _tokens_in_pool(native_tokens, _band_tokens(lines, line)):
                    return (
                        RowVerdict.UNVERIFIABLE,
                        "unlabelled row under a labelled empty row; the line's label is the header's own",
                        native_label,
                    )
                return RowVerdict.BOUND, "label shipped on the all-empty row above", native_label
            return (
                RowVerdict.LABEL_LOST,
                "shipped label empty; native line prints a label",
                native_label,
            )
        if _has_letter(native_tokens) and not row.numbers:
            # post-held-out (Argon c2): a categorical row with no label; BOUND only when the native
            # label text sits in one of its word cells, otherwise it cannot be told from a lost label.
            # Merge gate (#1271): token equality, never substring (`us` is inside `business`), and
            # never an empty token list.
            cell_tokens = [t for c in row.cell_texts for t in _label_tokens(c)]
            if _tokens_in_pool(native_tokens, cell_tokens):
                return RowVerdict.BOUND, "native label sits in a word cell of the row", native_label
            return (
                RowVerdict.UNVERIFIABLE,
                "categorical row without a label; native line prints one",
                native_label,
            )
        return RowVerdict.BOUND, "no label on either side", native_label
    if not row.label_tokens:
        # Round 7 (Astra P1): a symbol-only label (`%`, `$`) has no tokens to compare; only the exact
        # printed label decides, and it never convicts.
        if _labels_exact(row.label, native_label, strict=True):
            return (
                RowVerdict.BOUND,
                "symbol label printed exactly on the matched line",
                native_label,
            )
        return RowVerdict.UNVERIFIABLE, "symbol label not printed on the matched line", native_label
    on_line = [w.tok for w in own if w.tok] + [
        w.tok for w in cand.words[first_value_word - cand.start :] if w.tok
    ]
    if _tokens_in_pool(row.label_tokens, on_line):
        # Containment is ONE way ('Exports' on 'Net exports'). When the shipped label is not the
        # line's own label, the page may print the shipped label as a line of its own: that line
        # is the label's home, and the values sitting here belong to the line printing this label.
        if _has_letter(native_tokens) and not _same_label(native_tokens, row.label_tokens):
            homes = [
                ln
                for ln in lines
                if ln.index != line.index
                and ln.nums
                and _same_label([w.tok for w in _own_label(ln) if w.tok], row.label_tokens)
            ]
            if len(homes) == 1:
                if homes[0] in _adjacent_valued(lines, line, not row.numbers):
                    return (
                        RowVerdict.DISPLACED,
                        f"label belongs to adjacent native line {homes[0].index}",
                        native_label,
                    )
                return (
                    RowVerdict.UNVERIFIABLE,
                    "label is the own label of a non-adjacent native line",
                    native_label,
                )
            if len(homes) > 1:
                return (
                    RowVerdict.UNVERIFIABLE,
                    "label is the own label of several native lines",
                    native_label,
                )
            # Merge gate P1 (#1271): a partial label with no exact home elsewhere proves nothing
            # either way (`Government` on `Federal government`): UNVERIFIABLE, never BOUND.
            return (
                RowVerdict.UNVERIFIABLE,
                "shipped label is part of the line's label; no line prints it exactly",
                native_label,
            )
        return RowVerdict.BOUND, "label on the matched line", native_label
    pool = _pool(lines, line, own) + on_line
    if _tokens_in_pool(row.label_tokens, pool):
        # post-held-out (Argon b): a label from an adjoining band only completes the line's own label;
        # when the line prints a label of its own that the shipped label does not contain, the band's
        # text was moved onto a row it does not belong to.
        if not _has_letter(native_tokens) or _tokens_in_pool(native_tokens, row.label_tokens):
            return (
                RowVerdict.BOUND,
                "label on the matched line and its adjoining label band",
                native_label,
            )
        # Round 5 (Fable P2): the values ARE on this line; a band label above it may be the first line
        # of a wrapped label continued here (`Exports` / `of goods 1.1 2.2 3.3`). Not a conviction.
        return (
            RowVerdict.UNVERIFIABLE,
            "shipped label is the adjoining band's; the line prints another label",
            native_label,
        )
    for adj in _adjacent_valued(lines, line, not row.numbers):
        adj_label = _own_label(adj)
        if _tokens_in_pool(row.label_tokens, _pool(lines, adj, adj_label)):
            return (
                RowVerdict.DISPLACED,
                f"label belongs to adjacent native line {adj.index}",
                native_label,
            )
    return (
        RowVerdict.UNVERIFIABLE,
        "label matches neither the line nor an adjacent value line",
        native_label,
    )


def _judge_empty(
    row: _Row,
    lines: list[_Line],
    lo: int,
    hi: int,
    neighbours: tuple[tuple[int, list[str]], ...] = (),
):
    """Row with a label and no values: BOUND when a native line with that label carries no values
    (group header), EMPTIED when exactly one carries a full set of values.  Only native lines between
    the lines of the neighbouring uniquely-assigned rows (*lo*, *hi*, inclusive: a shifted row sits on its
    parent's line) are candidates: the shipped
    order is the reading order, and repeated section labels ('Industrial Countries' in two blocks)
    are told apart by it."""
    header_like = valued = single = split = 0
    exact_header = exact_valued = False
    for ln in lines:
        if not lo <= ln.index <= hi:
            continue
        own = _own_label(ln)
        if not own:
            continue
        line_pool = _pool(lines, ln, own)
        if not _tokens_in_pool(row.label_tokens, line_pool):
            continue
        own_tokens = [w.tok for w in own if w.tok]
        if len(ln.nums) >= EMPTIED_MIN_VALUES and any(
            # Adversarial pass (F3): an adjacent shipped row already carries this line's values, and
            # the two labels together are the line's label: the model split one printed row over two
            # markdown rows (`Gross saving less gross` + `investment`) or repeated the label. The
            # values were not dropped. The neighbour must ADD a word or repeat the label exactly: a
            # neighbour whose label is a shorter piece of this one (`Exports` under `Net exports`) is
            # the shift defect, measured on frame 1, and stays EMPTIED.
            anchor == ln.index
            and (
                # an UNLABELLED neighbour carries the line's values: the whole label went onto the
                # empty row (2004-12-14 greenbook p28 shape; round 4 found it convicted)
                not toks
                or (
                    _tokens_in_pool(toks, line_pool)
                    and _tokens_in_pool(own_tokens, row.label_tokens + toks)
                    and (
                        not _tokens_in_pool(toks, row.label_tokens)
                        or _same_label(toks, row.label_tokens)
                    )
                )
            )
            for anchor, toks in neighbours
        ):
            split += 1
        elif len(ln.nums) >= EMPTIED_MIN_VALUES:
            valued += 1
            exact_valued = exact_valued or _labels_exact(row.label, " ".join(w.text for w in own))
        elif ln.nums:
            single += (
                1  # post-held-out (Argon d): one number is a value or a footnote, not a header
            )
        else:
            header_like += 1
            exact_header = exact_header or _labels_exact(
                row.label, " ".join(w.text for w in ln.words), strict=True
            )
    if header_like and exact_valued:
        # Round 5 (Fable P2): a line printing exactly this label WITH values is also in range, so the
        # row may be a data row whose values were dropped; an exact valueless line does not settle it.
        return (
            RowVerdict.UNVERIFIABLE,
            "label matches a valueless line and a line with values",
            None,
        )
    if header_like:
        if exact_header:
            return (
                RowVerdict.BOUND,
                "label is printed exactly on a native line without values",
                None,
            )
        return (
            RowVerdict.UNVERIFIABLE,
            "label matches a valueless native line, but not exactly",
            None,
        )
    if split:
        return (
            RowVerdict.UNVERIFIABLE,
            "label split over two shipped rows; an adjacent row carries the line's values",
            None,
        )
    if valued == 1:
        return RowVerdict.EMPTIED, "label ships without values; native line prints them", None
    if valued > 1:
        return RowVerdict.UNVERIFIABLE, "label matches several native value lines", None
    if single:
        return RowVerdict.UNVERIFIABLE, "label matches a native line with a single value", None
    return RowVerdict.UNVERIFIABLE, "no native line carries this label", None


# ---------------------------------------------------------------------------
# Positive evidence beyond the same-line rule (#1257, plan B1)
# ---------------------------------------------------------------------------
# The same-line rule binds a row only when its whole label is printed before its values on one text
# line.  Three layouts are correct and still abstained: a label wrapped over several lines (E1), a
# heading stacked over several lines (E2) and a row whose label and cells reproduce the printed line
# word for word, though the label is not the stretch before the first number (E3: a label spread over
# two cells, ranges and ellipses).  Each may only move an UNVERIFIABLE row to BOUND: the pending rows
# are the ones whose verdict is already final and not a conviction, and nothing here convicts or
# removes a conviction.  Every check is exact (footnote marks and symbols included, as on the
# same-line path); a check that cannot be exact leaves the row UNVERIFIABLE.  None of them binds a
# table that left a row-shaped line of its region without a shipped row (``_unclaimed_row_lines``):
# the labels of such a table may be exact while the table has lost content, and the mark it carries is
# then the only one it has.

EVIDENCE_WRAPPED_LABEL = "wrapped_label_exact"
EVIDENCE_STACKED_HEADING = "stacked_heading_geometry"
EVIDENCE_WHOLE_LINE = "whole_line_exact"
REASON_WHOLE_LINE = "whole line: the row's label and cells are the printed line, word for word"
REASON_WRAPPED_LABEL = "wrapped label: the label is the exact join of consecutive printed fragments"
REASON_STACKED_HEADING = (
    "stacked heading: every heading cell is exactly the words stacked over its data column"
)


def _line_centre(line: _Line) -> float:
    return statistics.fmean((w.y0 + w.y1) / 2 for w in line.words)


def _line_height(line: _Line) -> float:
    return statistics.median(w.y1 - w.y0 for w in line.words)


def _pitch_bound(lines: list[_Line]) -> float | None:
    """The largest centre-to-centre distance at which two lines still count as consecutive: the
    median distance between consecutive native lines of the region (the page's own line pitch) plus
    the jitter a word is allowed on its own line (``LINE_CENTRE_TOL`` of the line height, the rule
    that put the words on that line).  None when the region has fewer than two lines."""
    if len(lines) < 2:
        return None
    centres = [_line_centre(ln) for ln in lines]
    gaps = [b - a for a, b in zip(centres, centres[1:])]
    return statistics.median(gaps) + LINE_CENTRE_TOL * statistics.median(
        _line_height(ln) for ln in lines
    )


def _consecutive(upper: _Line, lower: _Line, bound: float | None) -> bool:
    return bound is not None and 0 < _line_centre(lower) - _line_centre(upper) <= bound


def _has_run(seq: list[str], run: list[str]) -> bool:
    n = len(run)
    return bool(run) and any(seq[i : i + n] == run for i in range(len(seq) - n + 1))


def _fragment_text(line: _Line) -> str:
    return " ".join(w.text for w in line.words)


def _claimed_elsewhere(line: _Line, others: list[_Row]) -> bool:
    """Another shipped row (group label, categorical row, heading, any row) carries this line's words
    as a run of its own label or cells: the line may be that row's, so it is nobody's continuation."""
    run = [w.tok for w in line.words if w.tok]
    if not run:
        return True
    for q in others:
        if _has_run(q.label_tokens, run):
            return True
        if any(_has_run(_label_tokens(c), run) for c in q.cells_raw):
            return True
    return False


def _single_home(opts: list[tuple[_Cand, int]]) -> tuple[_Cand, int] | None:
    """The one place the row's values sit: its only option, or several options that are one and the
    same stretch of one line (a whole line and the gap-separated panel carrying the values, both
    starting at the same first value): the widest of them.  Options in different panels of a line
    (twin tables with equal values) have no single home."""
    if len(opts) == 1:
        return opts[0]
    widest = max(opts, key=lambda o: o[0].stop - o[0].start)
    wide, wi = widest
    if all(
        c.line.index == wide.line.index
        and w == wi
        and c.start >= wide.start
        and c.stop <= wide.stop
        for c, w in opts
    ):
        return widest
    return None


def _wrapped_label_fragments(
    row: _Row,
    cand: _Cand,
    first_value_word: int,
    lines: list[_Line],
    bound: float | None,
    anchored: set[int],
    others: list[_Row],
) -> frozenset[int] | None:
    """E1.  The native line indices of the valueless fragment lines whose exact join with the
    same-line fragment is the shipped label, or None.

    All must hold: the fragments are valueless lines directly above and/or below the row's line,
    each consecutive with the next (``_pitch_bound``); each lies in the label column (every word ends
    before the row's first value, and no line starts left of the first line of the label by more than
    the overprint tolerance); none is the line of another shipped row nor a run of another row's label
    or cells (``_claimed_elsewhere``); the join equals the shipped label under the strict gate
    (footnote marks included); and exactly one such split exists."""
    line = cand.line
    value_x0 = line.words[first_value_word].x0
    printed = _printed_label(cand, first_value_word)
    height = _line_height(line)
    tol = DUP_OVERPRINT_TOL_HEIGHTS * height

    def free(ln: _Line) -> bool:
        return (
            _label_only(ln)
            and ln.index not in anchored
            and all(w.x1 <= value_x0 for w in ln.words)
            and not _claimed_elsewhere(ln, others)
        )

    above: list[_Line] = []
    j = line.index - 1
    nxt = line
    while j >= 0 and free(lines[j]) and _consecutive(lines[j], nxt, bound):
        above.append(lines[j])
        nxt = lines[j]
        j -= 1
    below: list[_Line] = []
    j = line.index + 1
    prv = line
    while j < len(lines) and free(lines[j]) and _consecutive(prv, lines[j], bound):
        below.append(lines[j])
        prv = lines[j]
        j += 1
    hits: list[frozenset[int]] = []
    for a in range(len(above) + 1):
        for b in range(len(below) + 1):
            if a + b == 0 or (not printed and not (a and b)):
                continue
            top = [*reversed(above[:a])]
            parts = [_fragment_text(ln) for ln in top]
            if printed:
                parts.append(printed)
            parts.extend(_fragment_text(ln) for ln in below[:b])
            if not _labels_exact(row.label, " ".join(parts), strict=True):
                continue
            head_x0 = top[0].words[0].x0 if top else cand.words[0].x0
            starts = [ln.words[0].x0 for ln in (*top[1:], *below[:b])]
            if printed and top:
                starts.append(cand.words[0].x0)
            if any(x < head_x0 - tol for x in starts):
                continue
            hits.append(frozenset(ln.index for ln in (*top, *below[:b])))
    return hits[0] if len(hits) == 1 else None


def _expand_dots(tokens: list[str]) -> list[str]:
    """A run of full stops (`...`) is the same text whether the layer spaces it (`. . .`) or not."""
    out: list[str] = []
    for t in tokens:
        out.extend("." * len(t) if len(t) > 1 and set(t) == {"."} else [t])
    return out


def _row_words(row: _Row) -> list[str]:
    """Every word the shipped row prints: label cell, then every data cell, in order."""
    parts = [row.label, *row.cells_raw]
    return _expand_dots(
        [t for p in parts for t in _strip_markup(p).translate(_UNICODE_MINUS).split()]
    )


def _whole_line_matches(row: _Row, cand: _Cand) -> bool:
    """E3.  The shipped row reproduces the whole candidate line, word for word and in order: its
    label cell and data cells are the line's words, none missing and none left over.  A label spread
    over two cells (`1` | `LR`), a range or ellipsis cell (`.0-3.0`, `...`) and a row with an empty
    label cell all print exactly what the page prints on that line.  Words compare as in
    ``_same_word`` (footnote marks and symbols included, no tolerance)."""
    shipped = _row_words(row)
    native = _expand_dots([w.text.translate(_UNICODE_MINUS) for w in cand.words])
    return (
        bool(shipped)
        and len(shipped) == len(native)
        and all(_same_word(n, t) for n, t in zip(native, shipped))
    )


def _wrapped_group_label(
    row: _Row,
    lines: list[_Line],
    bound: float | None,
    anchored: set[int],
    others: list[_Row],
) -> frozenset[int] | None:
    """E1 for a row with a label and no values (a group label): the label is the exact join of two or
    more consecutive valueless lines, each free (no row owns it), consecutive (``_pitch_bound``) and
    none starting left of the first by more than the overprint tolerance.  Exactly one run may do it."""
    hits: list[frozenset[int]] = []
    for i, first in enumerate(lines):
        run: list[_Line] = []
        for j in range(i, len(lines)):
            ln = lines[j]
            if (
                not _label_only(ln)
                or ln.index in anchored
                or _claimed_elsewhere(ln, others)
                or (run and not _consecutive(run[-1], ln, bound))
            ):
                break
            run.append(ln)
            tol = DUP_OVERPRINT_TOL_HEIGHTS * _line_height(first)
            if (
                len(run) >= 2
                and all(r.words[0].x0 >= first.words[0].x0 - tol for r in run[1:])
                and _labels_exact(row.label, " ".join(_fragment_text(r) for r in run), strict=True)
            ):
                hits.append(frozenset(r.index for r in run))
    return hits[0] if len(hits) == 1 else None


def _stacked_heading_binds(
    header: _Row,
    body: list[tuple[_Row, _Cand, int]],
    lines: list[_Line],
    bound: float | None,
    group_rows: list[_Row] = (),  # type: ignore[assignment]
) -> frozenset[int] | None:
    """E2.  Every cell of the heading row is exactly the words stacked over its data column.

    *body* lists the body rows that have ONE assigned line (row, candidate, first value word).  The
    data columns are the x-extents of the j-th value of every body row that has exactly one value
    per heading cell (they must be ordered and disjoint).  The heading lines are the N lines directly
    above the first body line (or above a group label printed right under the heading), consecutive
    (``_pitch_bound``).  Each word of those lines belongs to the column slot its centre lies in (slots
    are cut halfway between neighbouring columns; the label column ends where the printed labels
    end).  The words of a slot, read top to bottom and left to right, must be the cell's words one
    by one, the stub words the label's, and no word may be left over (so at most one N can fit).
    Membership comes from the geometry of the columns, not from the flattened word sequence: the
    same words under the wrong column fail, and a spanning heading, whose words fall in several
    slots, can only match cells that split it the way the page does.  Returns the heading's native
    line indices, or None."""
    cells = header.cells_all
    if not cells or any(not _strip_markup(c).strip() for c in cells) or not body:
        return None
    wanted = [_strip_markup(c).translate(_UNICODE_MINUS).split() for c in cells]
    stub_words = _strip_markup(header.label).translate(_UNICODE_MINUS).split()
    n_cols = len(cells)
    extents: list[list[float]] = [[float("inf"), float("-inf")] for _ in range(n_cols)]
    full = 0
    stub_right = float("-inf")  # right edge of the label column, from the labels the rows print
    for r, c, wi in body:
        if r.categories or len(r.numbers) != n_cols:
            continue
        vals = [w for w in c.line.words[wi : c.stop] if w.num is not None][:n_cols]
        if len(vals) != n_cols:
            continue
        full += 1
        stub_right = max([stub_right, *(w.x1 for w in c.words[: wi - c.start])])
        for k, w in enumerate(vals):
            extents[k][0] = min(extents[k][0], w.x0)
            extents[k][1] = max(extents[k][1], w.x1)
    if not full or any(
        a >= b for a, b in zip((e[1] for e in extents), (e[0] for e in extents[1:]))
    ):
        return None
    if stub_right >= extents[0][0]:
        return None  # the label column reaches the first data column: no room to tell them apart
    # Slot borders: halfway between two neighbouring data columns; the label column ends where the
    # labels the rows print end (minus infinity when no row prints one).  A heading word belongs to
    # the slot its centre lies in, so a heading wider than its numbers still sits over its column.
    edges = [(extents[k][1] + extents[k + 1][0]) / 2 for k in range(n_cols - 1)]
    edges = [stub_right, *edges]  # edges[k] is the left border of data column k
    first = min(c.line.index for _, c, _ in body)
    # A group label printed between the heading and the first data line (`Annual`, `Monthly Growth
    # Rates`) is a shipped row of its own: the heading ends above the line that prints it exactly.
    for g in group_rows:
        homes = [
            ln.index
            for ln in lines[:first]
            if _label_only(ln) and _labels_exact(g.label, _fragment_text(ln), strict=True)
        ]
        if len(homes) == 1:
            first = min(first, homes[0])
    for n in range(1, first + 1):
        block = lines[first - n : first]
        if any(not _consecutive(a, b, bound) for a, b in zip(block, block[1:])):
            break  # a longer block would only add a gap
        groups: list[list[tuple[int, float, str]]] = [[] for _ in range(n_cols + 1)]  # [0] = stub
        for li, ln in enumerate(block):
            for w in ln.words:
                centre = (w.x0 + w.x1) / 2
                slot = sum(1 for e in edges if centre > e)  # 0 = label column
                groups[slot].append((li, w.x0, w.text))
        reads = [[t.translate(_UNICODE_MINUS) for _, _, t in sorted(g)] for g in groups]
        expected = [stub_words, *wanted]
        if all(
            len(got) == len(want) and all(_same_word(g, w) for g, w in zip(got, want))
            for got, want in zip(reads, expected)
        ):
            return frozenset(ln.index for ln in lines[first - n : first])
    return None


def _value_columns(chosen: dict[int, list[tuple[_Cand, int]]]) -> list[tuple[float, float]]:
    """The x-extents of the table's value columns: the value words of every assigned line, merged
    where they overlap."""
    spans = sorted(
        (w.x0, w.x1)
        for opts in chosen.values()
        for c, wi in opts[:1]
        for w in c.line.words[wi : c.stop]
        if w.num is not None
    )
    merged: list[list[float]] = []
    for x0, x1 in spans:
        if merged and x0 < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])
    return [(a, b) for a, b in merged]


def _unclaimed_row_lines(
    lines: list[_Line], claimed: set[int], columns: list[tuple[float, float]]
) -> list[int]:
    """Native lines of the region that look like a table row and that no shipped row or heading
    accounts for: a first number, then only numbers and symbols to the end of the line, with numbers
    in at least ``MIN_VALUES_TO_CONVICT`` different value columns of the table (prose with a date,
    or a running head, sits in one column at most)."""
    out: list[int] = []
    for ln in lines:
        if ln.index in claimed:
            continue
        at = next((k for k, w in enumerate(ln.words) if w.num is not None), None)
        if at is None:
            continue
        tail = ln.words[at:]
        if not all(w.num is not None or not _has_letter([w.tok]) for w in tail):
            continue
        hit = {
            k
            for w in tail
            if w.num is not None
            for k, (x0, x1) in enumerate(columns)
            if w.x0 < x1 and w.x1 > x0
        }
        if len(hit) >= MIN_VALUES_TO_CONVICT:
            out.append(ln.index)
    return out


def check_row_binding(
    words: list,
    region: tuple[float, float, float, float] | None,
    table_markdown: str,
    *,
    extra_evidence: bool = True,
) -> RowBindingResult:
    """Compare every shipped data row of *table_markdown* with the text layer.

    *words* are fitz word tuples in the reading frame; *region* is the table box in the same frame
    (None = whole page).  *extra_evidence* lets the wrapped-label and stacked-heading evidence
    (E1, E2) move UNVERIFIABLE rows to BOUND; the point-repair gate passes False, so a repair is
    judged by the same-line rule it was built on."""
    lines = build_native_lines(words, region)
    shipped = _shipped_rows(table_markdown)
    rows = [r for r in shipped if not r.is_header]
    cands = _candidates(lines)
    optioned = [(r, _options(r, cands)) for r in rows if r.has_content]
    optioned = [(r, o) for r, o in optioned if o]
    feas = _feasible([o for _, o in optioned])
    # No one-to-one order-preserving assignment (feas is None) means no row's verdict may rest on one:
    # every row that found a line is UNVERIFIABLE, and the table cannot be BOUND.
    infeasible = {r.line_index for r, _ in optioned} if feas is None else set()
    chosen: dict[int, list[tuple[_Cand, int]]] = (
        {r.line_index: feas[k] for k, (r, _) in enumerate(optioned)} if feas is not None else {}
    )
    anchors: list[int | None] = []
    for r in rows:
        opts = chosen.get(r.line_index) or []
        idx = {c.line.index for c, _ in opts}
        anchors.append(idx.pop() if len(idx) == 1 else None)
    evidence: list[RowEvidence] = []
    # evidence index -> (row position, assigned candidate and first value word; None for a row without values)
    wrap_pending: dict[int, tuple[int, _Cand | None, int]] = {}
    for pos, r in enumerate(rows):
        values = r.cell_texts or r.cells_raw  # the evidence records what shipped, symbols included
        if not r.has_content:
            _symbols = _symbol_cells_only(r)
            if _label_cell_empty(r):
                if _symbols:
                    evidence.append(
                        RowEvidence(
                            r.line_index,
                            RowVerdict.UNVERIFIABLE,
                            "cells hold only symbols, which the check cannot bind",
                            r.label,
                            "",
                            r.cells_raw,
                            None,
                        )
                    )
                continue  # blank row
            lo = next((a for a in reversed(anchors[:pos]) if a is not None), -1)
            hi = next((a for a in anchors[pos + 1 :] if a is not None), len(lines))
            neighbours = tuple(
                (anchors[q], rows[q].label_tokens)
                for q in (pos - 1, pos + 1)
                if 0 <= q < len(rows) and anchors[q] is not None
            )
            v, why, _ = _judge_empty(r, lines, lo, hi, neighbours)
            # Symbols must not let a row pass as valueless, and `—` where the page prints numbers
            # stays the EMPTIED conviction it is.
            if _symbols and v is RowVerdict.BOUND:
                v, why = (
                    RowVerdict.UNVERIFIABLE,
                    "cells hold only symbols, which the check cannot bind",
                )
            if extra_evidence and v is RowVerdict.UNVERIFIABLE and r.label_tokens:
                wrap_pending[len(evidence)] = (pos, None, -1)
            evidence.append(RowEvidence(r.line_index, v, why, r.label, "", values, None))
            continue
        if r.line_index in infeasible:
            evidence.append(
                RowEvidence(
                    r.line_index,
                    RowVerdict.UNVERIFIABLE,
                    "no one-to-one order-preserving assignment",
                    r.label,
                    "",
                    values,
                    None,
                )
            )
            continue
        opts = chosen.get(r.line_index)
        if not opts:
            evidence.append(
                RowEvidence(
                    r.line_index,
                    RowVerdict.UNVERIFIABLE,
                    "no native line carries these values",
                    r.label,
                    "",
                    values,
                    None,
                )
            )
            continue
        prev_row = rows[pos - 1] if pos else None
        verdicts = [(_judge(r, c, wi, lines, prev_row), c) for c, wi in opts]
        if len(r.numbers or r.categories) < MIN_VALUES_TO_CONVICT:
            verdicts = [
                (
                    (RowVerdict.UNVERIFIABLE, "a single value does not identify its line", v[1])
                    if v[0] in CONVICTING
                    else v,
                    c,
                )
                for v, c in verdicts
            ]
        classes = {v[0] for v, _ in verdicts}
        (v, why, nat), c = verdicts[0]
        if len(opts) > 1 and len(classes) > 1:
            evidence.append(
                RowEvidence(
                    r.line_index,
                    RowVerdict.UNVERIFIABLE,
                    f"ambiguous assignment ({len(opts)} lines, verdicts differ)",
                    r.label,
                    "",
                    values,
                    None,
                )
            )
            continue
        if v is RowVerdict.BOUND and (
            not _bound_exactly(r.label, [_printed_label(c2, wi2) for c2, wi2 in opts], lines)
            or not all(_cells_cover_line(r, c2, wi2) for c2, wi2 in opts)
        ):
            v, why = (
                RowVerdict.UNVERIFIABLE,
                "label not printed exactly before these values, or their line carries another stub",
            )
        home = _single_home(opts) if extra_evidence else None
        if home is not None and v is not RowVerdict.BOUND and v not in CONVICTING:
            wrap_pending[len(evidence)] = (pos, home[0], home[1])
        evidence.append(RowEvidence(r.line_index, v, why, r.label, nat, values, c.line.index))
    # E1 / E3 (#1257): wrapped labels and whole-line rows, resolved over the whole table so that no
    # line serves two rows and no line another shipped row may own is taken.
    bound_gap = _pitch_bound(lines)
    proposals: list[tuple[RowEvidence, RowEvidence]] = []  # (as judged, as it would bind)
    anchored_lines = {c.line.index for opts in chosen.values() for c, _ in opts}
    if wrap_pending:
        anchored_lines = {c.line.index for opts in chosen.values() for c, _ in opts}
        found: dict[int, tuple[frozenset[int], str, str]] = {}
        for at, (pos, c, wi) in wrap_pending.items():
            row = rows[pos]
            others = [q for q in shipped if q is not row]
            if c is None:
                if not row.cells_raw:
                    frags = _wrapped_group_label(row, lines, bound_gap, anchored_lines, others)
                    if frags is not None:
                        found[at] = (frags, EVIDENCE_WRAPPED_LABEL, REASON_WRAPPED_LABEL)
                    continue
                homes = [k for k in cands if not k.line.nums and _whole_line_matches(row, k)]
                if len(homes) == 1 and homes[0].line.index not in anchored_lines:
                    found[at] = (
                        frozenset({homes[0].line.index}),
                        EVIDENCE_WHOLE_LINE,
                        REASON_WHOLE_LINE,
                    )
                continue
            if _whole_line_matches(row, c):
                found[at] = (frozenset(), EVIDENCE_WHOLE_LINE, REASON_WHOLE_LINE)
                continue
            frags = (
                _wrapped_label_fragments(row, c, wi, lines, bound_gap, anchored_lines, others)
                if row.label_tokens
                else None
            )
            if frags is not None and _cells_cover_line(row, c, wi):
                found[at] = (frags, EVIDENCE_WRAPPED_LABEL, REASON_WRAPPED_LABEL)
        for at, (frags, kind, reason) in found.items():
            if frags and any(
                frags & other[0] for k, other in found.items() if k != at and other[0]
            ):
                continue  # a shared line: nobody's
            old = evidence[at]
            proposals.append(
                (
                    old,
                    RowEvidence(
                        old.line_index,
                        RowVerdict.BOUND,
                        reason,
                        old.shipped_label,
                        old.native_label,
                        old.values,
                        old.native_line,
                        kind,
                    ),
                )
            )
    heading_lines: set[int] = set()  # native lines the heading rows account for
    # Header-position rows (adversarial pass F1; rounds 3-5). Round 5 (R2, COMPLETE JUDGEMENT): a table
    # is BOUND only if every markdown row was judged by the exact gate, so EVERY header row with a
    # non-empty data cell (numbers, mixed text and numbers, or words) is judged like a body row, outside
    # the assignment (a heading line cannot reorder the data rows), and binds only on the exact rule. A
    # header row that cannot be located is UNVERIFIABLE. It may CONVICT only when it is a data row by its
    # own content: numbers-only cells with a decimal value, or a label word over values that are not all
    # four-digit years (frame-1 live boxes convicted a `(March 24 - April 24)` heading before the
    # numbers-only condition). Any other conviction is downgraded to UNVERIFIABLE.
    for h in shipped:
        if not h.is_header:
            continue
        if not h.has_content:
            if _symbol_cells_only(h):
                evidence.insert(
                    0,
                    RowEvidence(
                        h.line_index,
                        RowVerdict.UNVERIFIABLE,
                        "header-position row: cells hold only symbols, which the check cannot bind",
                        h.label,
                        "",
                        h.cells_raw,
                        None,
                    ),
                )
                continue
            # Round 6: a header-position row with a label and no values (`| Exports | | | |`) is judged
            # like a valueless body row; it binds only on an exact valueless line and never convicts.
            if not _label_cell_empty(h):
                v, why, _ = _judge_empty(h, lines, -1, len(lines))
                if v is not RowVerdict.BOUND:
                    v, why = RowVerdict.UNVERIFIABLE, f"header-position row without values: {why}"
                evidence.insert(
                    0, RowEvidence(h.line_index, v, why, h.label, "", h.cell_texts, None)
                )
            continue
        has_decimal = any("." in n for n in h.numbers)
        all_years = bool(h.numbers) and all(re.fullmatch(r"\d{4}", n) for n in h.numbers)
        may_convict = (
            bool(h.numbers)
            and h.values_only
            and (has_decimal or (_has_letter(h.label_tokens) and not all_years))
        )
        opts = _options(h, cands)
        heading_lines.update(c.line.index for c, _ in opts)
        if not opts:
            v, why, nat, line_no = (
                RowVerdict.UNVERIFIABLE,
                "header-position row; no native line carries these cells",
                "",
                None,
            )
        else:
            judged = [(_judge(h, c, wi, lines, None), c) for c, wi in opts]
            classes = {j[0] for j, _ in judged}
            (v, why, nat), c = judged[0]
            line_no = c.line.index
            why = f"header-position row: {why}"
            if len(classes) > 1:
                v, why = (
                    RowVerdict.UNVERIFIABLE,
                    f"header-position row: ambiguous ({len(opts)} lines)",
                )
            elif v in CONVICTING and not (may_convict and len(h.numbers) >= MIN_VALUES_TO_CONVICT):
                v, why = (
                    RowVerdict.UNVERIFIABLE,
                    "header-position row: a heading cannot be convicted",
                )
            elif v is RowVerdict.BOUND and (
                not _bound_exactly(h.label, [_printed_label(c2, wi2) for c2, wi2 in opts], lines)
                or not all(_cells_cover_line(h, c2, wi2) for c2, wi2 in opts)
            ):
                v, why = (
                    RowVerdict.UNVERIFIABLE,
                    "header-position row: label not printed exactly before these cells",
                )
        if extra_evidence and v is RowVerdict.UNVERIFIABLE and "ambiguous" not in why:
            body = [
                (rows[k], chosen[rows[k].line_index][0][0], chosen[rows[k].line_index][0][1])
                for k in range(len(rows))
                if len(chosen.get(rows[k].line_index, ())) == 1
            ]
            lead = next(
                (k for k in range(len(rows)) if len(chosen.get(rows[k].line_index, ())) == 1),
                len(rows),
            )
            group_rows = [q for q in rows[:lead] if not q.has_content and not _label_cell_empty(q)]
            block = _stacked_heading_binds(h, body, lines, bound_gap, group_rows)
            if block is not None:
                heading_lines |= block
                judged = RowEvidence(h.line_index, v, why, h.label, nat, h.cell_texts, line_no)
                evidence.insert(0, judged)
                proposals.append(
                    (
                        judged,
                        RowEvidence(
                            h.line_index,
                            RowVerdict.BOUND,
                            REASON_STACKED_HEADING,
                            h.label,
                            "",
                            h.cell_texts,
                            None,
                            EVIDENCE_STACKED_HEADING,
                        ),
                    )
                )
                continue
        evidence.insert(0, RowEvidence(h.line_index, v, why, h.label, nat, h.cell_texts, line_no))
    # The new evidence binds only a COMPLETE table: no row-shaped value line of the region is left
    # without a shipped row (a dropped row is not a binding fault, but a table that lost one must not
    # lose its mark because its remaining labels are exact).  The same-line rule is unchanged.
    if proposals and not _unclaimed_row_lines(
        lines, anchored_lines | heading_lines, _value_columns(chosen)
    ):
        for old, new in proposals:
            at = next(k for k, e in enumerate(evidence) if e is old)
            evidence[at] = new
    if any(e.verdict in CONVICTING for e in evidence):
        outcome = TableOutcome.CONTRADICTED
    # (a table with no data rows has nothing shipped to bind; the empty-table checks own it)
    elif any(e.verdict is RowVerdict.UNVERIFIABLE for e in evidence):
        outcome = TableOutcome.UNVERIFIABLE
    else:
        outcome = TableOutcome.BOUND
    return RowBindingResult(outcome, tuple(evidence))


# ---------------------------------------------------------------------------
# Shared by the pipeline step and the offline re-finalize tool
# ---------------------------------------------------------------------------


def layer_can_judge_tables(ps, out, page=None, *, native_only: bool = False) -> bool:
    """The ONE predicate for "the text layer may judge this page's model-read tables" (#1267, #1282).

    Called by ``UnifiedPipeline._run_row_binding_check`` and ``_repair_dropped_points`` and by the
    offline ``socr-refinalize-point-drops``, so none of them can touch a page another would not.
    Without *page*: the cheap guards (text present, not a chart asset / failed candidate / native
    engine, not ``native_only``, and an admissible layer: an old OCR layer never votes, #1156).
    With *page* (a fitz page): also the layer must be born-digital, or refused for spacing alone and
    admitted by ``BornDigitalDetector.refused_layer_can_judge`` (the classifier also refuses real
    layers as excessively spaced: 9 of the 35 convicted held-out pages).  Every other refusal
    doubts the layer's characters."""
    from socr.core.born_digital import BornDigitalDetector
    from socr.core.state import layer_is_admissible_evidence
    from socr.pipeline.agentic import is_failed_candidate

    if out is None or not out.text or out.engine == "chart_asset" or is_failed_candidate(out):
        return False
    if native_only or (out.engine or "") == "native":
        return False
    if not layer_is_admissible_evidence(ps):
        return False
    if page is None:
        return True
    return bool(getattr(ps, "is_born_digital", False)) or bool(
        getattr(ps, "refused_for_spacing_alone", False)
        and BornDigitalDetector().refused_layer_can_judge(page)
    )


def row_evidence_dicts(result: RowBindingResult) -> list[dict]:
    """The sidecar/event rows of one table's result: every row that is not BOUND."""
    return [
        {
            "md_line": e.line_index,
            "verdict": e.verdict.value,
            "reason": e.reason,
            "shipped_label": e.shipped_label,
            "native_label": e.native_label,
            "values": list(e.values),
            "native_line": e.native_line,
        }
        for e in result.rows
        if e.verdict is not RowVerdict.BOUND
    ]


def table_payload(
    page_num: int,
    table_index: int,
    outcome: str,
    rows: list[dict],
    error: str = "",
    result: RowBindingResult | None = None,
) -> dict:
    """One table's ``ps.row_binding`` record (before ``withheld_markdown`` / ``repairs``).

    *result* adds what a pass leaves behind (#1257): ``rows_bound`` / ``rows_total`` and, when a row
    was bound by evidence beyond the same-line rule, ``evidence`` (rows per kind).  ``rows`` still
    lists only the rows that are not BOUND."""
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    payload = {
        "table_id": f"p{page_num}-t{table_index}",
        "table_index": table_index,
        "outcome": outcome,
        "counts": counts,
        "rows": rows,
    }
    if result is not None:
        payload["rows_bound"] = result.bound_count()
        payload["rows_total"] = len(result.rows)
        if evidence := result.evidence_counts():
            payload["evidence"] = evidence
    if error:
        payload["error"] = error
    return payload


# ---------------------------------------------------------------------------
# Decimal-point repair (#1282)
# ---------------------------------------------------------------------------

# A repair needs this many OTHER values of the row to agree exactly (same position) with the
# candidate line.  One agreeing value identifies no line (the same argument as
# MIN_VALUES_TO_CONVICT), so two are required: with one, a two-value row `-7 | 1.5` would be
# repaired on the strength of a single `1.5` (review of #1282, case A).  Measured on the 308-page
# corpus (scratch/army/decimals, 175 repairs): the smallest count of agreeing values on any
# repaired row was 3, so this floor loses no measured repair; it only refuses rows too short to
# corroborate themselves.
MIN_AGREEING_VALUES = 2

# A page line that prints the row's exact label, with the same number of values, and differs from
# the shipped row in at most this many values may be the row's OWN line (condition 7).  The row's
# own line fails condition 1 only because the model slipped somewhere in the row; when it is not
# the hit, at least one of those slips is not a point drop (a detached minus `- 7`, a misread
# `-7.1`).  One such slip plus the dropped point being repaired in the same row is two
# differences.  A hit with such a rival anywhere on the page could be another section's line
# (`Previous`, a country name) whose value would be written into this row (review of #1282,
# findings 2a and 3).  A named judgement, not a measured optimum: TB-34's cached pages lose 2 of
# 152 proposed repairs at 2 (docs/log/2026-11-07_1282-point-repair.md); requiring the label to be
# unique on the page instead loses 74.  The same tolerance names a labelled line as "near" in the
# refusal reasons.
MAX_RIVAL_MISMATCHES = 2

# Refusal reasons (named, so a test and a sidecar can quote them).
REFUSE_CONTRADICTED = "table_contradicted"
REFUSE_AMBIGUOUS = "ambiguous_lines"
REFUSE_CLAIMED = "line_claimed_by_another_row"
REFUSE_ORDER = "out_of_reading_order"
REFUSE_DECORATED = "decorated_cell"
REFUSE_TOKEN_MAP = "token_map_failed"
REFUSE_WORSENS = "repair_worsens_table"
REFUSE_NOT_BOUND = "gate_does_not_bind_repaired_row"
REFUSE_LABEL = "label_not_exact"
REFUSE_NO_AGREEING = "no_agreeing_value"
REFUSE_AFFIX = "token_affix_differs"
REFUSE_SHARED_REGION = "region_shared_with_another_block"
REFUSE_RIVAL_LINE = "another_line_prints_the_label_close_to_the_row"
# Sign repair (#1282, plan A2 step 2): one refusal per extra condition.
REFUSE_SIGN_GEOMETRY = "minus_not_attached_to_its_number_by_geometry"
REFUSE_SIGN_NO_MINUS_EVIDENCE = "layer_shows_no_minus_in_this_table"
REFUSE_SIGN_MIRROR = "mirror_row_prints_the_same_absolute_values"

# The kinds of differing pair a repair may fix, one per cell (``_classify_pair``).  Both at once in
# one cell is ``point_and_sign``: refused, never repaired.
REPAIR_POINT = "point"
REPAIR_SIGN = "sign"
_REPAIRABLE_PAIRS = frozenset({"point_dropped", "sign_flipped"})
_PAIR_TO_REPAIR_KIND = {"point_dropped": REPAIR_POINT, "sign_flipped": REPAIR_SIGN}

# A numeric token split into what stands before the number, the number and what follows it
# (`-.7%` -> "", "-.7", "%").  The affixes of the shipped token and of the native one must be
# equal: a repair changes the point and nothing else (review of #1282, case E: `-7` for `-.7%`).
# The number ends on a digit, so a trailing `,` or `.` (`-.7,`) falls into the affix and refuses.
_NUMBER_AFFIX = re.compile(
    r"(?P<pre>[^\d.\-+\u2212\u2013\u2014\u2012]*)"
    r"(?P<num>[-+\u2212\u2013\u2014\u2012]?(?:\d(?:[\d.,]*\d)?|\.\d(?:[\d.,]*\d)?))"
    r"(?P<post>.*)",
    re.S,
)


def _affixes(token: str) -> tuple[str, str] | None:
    m = _NUMBER_AFFIX.fullmatch(token)
    return (m.group("pre"), m.group("post")) if m else None


def effective_outcome(outcome: str, repaired: bool) -> str:
    """The outcome a table is RECORDED with.  A table with a repaired row is never recorded BOUND:
    the table ladder judged the unrepaired bytes (it runs before the repair's text ships), so the
    shipped text was not the judged text, and a repair lifts no mark (#1282 ruling 2).  The check
    itself says BOUND (every row sits on its line); the record says UNVERIFIABLE, with no rows."""
    return (
        TableOutcome.UNVERIFIABLE.value
        if repaired and outcome == TableOutcome.BOUND.value
        else outcome
    )


def _digit_run(canon: str) -> str:
    return canon.lstrip("-").replace(".", "").lstrip("0") or "0"


def _digits_apart_from_point(canon: str) -> str:
    """The digits of canonical *canon* with the sign and the decimal point removed, and the one
    ``0`` a canonical form writes before a leading point (``-.7`` -> ``-0.7``) dropped again."""
    unsigned = canon.lstrip("-")
    if unsigned.startswith("0."):
        unsigned = unsigned[1:]
    return unsigned.replace(".", "")


def point_dropped(shipped: str, native: str) -> bool:
    """Canonical *shipped* is canonical *native* with its decimal point (and the zero before it)
    removed: ``-7`` for ``-0.7``, ``4`` for ``0.4``, ``12`` for ``1.2``, ``70`` for ``7.0``.  Same
    sign, and the SAME digit string apart from the point: no digit is added, dropped or moved.
    Never a different digit run (``-0.1`` for ``-1.1``), a zero lost after the point (``7`` for
    ``.07``, ``70`` for ``.070``: review of #1282, finding 2b), a point on both sides, or no point
    on the native side."""
    if shipped.startswith("-") != native.startswith("-"):
        return False
    if "." in shipped or "." not in native:
        return False
    return _digits_apart_from_point(shipped) == _digits_apart_from_point(native)


def _is_zero(canon: str) -> bool:
    try:
        return float(canon.lstrip("-")) == 0.0
    except ValueError:
        return False


def sign_flipped(shipped: str, native: str) -> bool:
    """Canonical *shipped* and *native* print the SAME digits, point included, and only the sign
    differs (``-3.0`` / ``3.0``), and the value is not zero (``0``, ``0.0``, ``-0`` carry no sign).
    A cell that is off in the point as well (``-3`` / ``.3``) is not a sign flip: one difference
    per cell (#1282, plan A2 step 2)."""
    if shipped.startswith("-") == native.startswith("-"):
        return False
    unsigned = shipped.lstrip("-")
    return unsigned == native.lstrip("-") and not _is_zero(unsigned)


def _classify_pair(shipped: str, native: str) -> str:
    """Why a differing pair is or is not a point drop or a sign flip (for the refusal reasons)."""
    if point_dropped(shipped, native):
        return "point_dropped"
    if shipped.startswith("-") != native.startswith("-"):
        if sign_flipped(shipped, native):
            return "sign_flipped"
        if shipped.lstrip("-") == native.lstrip("-"):
            return "sign_of_zero"
        if _digits_apart_from_point(shipped) == _digits_apart_from_point(native):
            return "point_and_sign"
        return "sign_differs"
    if "." in shipped and "." in native:
        ds, dn = _digit_run(shipped), _digit_run(native)
        if ds == dn:
            return "point_moved"
        if dn.endswith(ds) or dn.startswith(ds):
            return "digit_dropped"
        return "digits_differ"
    if "." not in native:
        return "native_has_no_point"
    return "digits_differ"


@dataclass(frozen=True)
class PointRepair:
    """One cell changed by :func:`propose_point_repairs`."""

    md_line: int  # line of the row in the table markdown given
    label: str
    col: int  # index of the numeric token within the row
    old: str
    new: str  # the text layer's own string
    native_line: int
    printed_label: str
    agreeing_values: int
    n_values: int
    kind: str = REPAIR_POINT  # REPAIR_POINT (a dropped point) or REPAIR_SIGN (a flipped minus)


def repair_row_dict(r: PointRepair, row_after: str) -> dict:
    """One repaired cell as the event and the sidecar record it (live step and offline tool)."""
    return {
        "table_line": r.md_line,  # within the table block: stable across the marks above the page
        "label": r.label,
        "col": r.col,
        "old": r.old,
        "new": r.new,
        "kind": r.kind,
        "native_line": r.native_line,
        "printed_label": r.printed_label,
        "agreeing_values": r.agreeing_values,
        "n_values": r.n_values,
        "row_after": row_after,
    }


def repair_event_detail(table_id: str, rows: list[dict]) -> str:
    """The audit-event text of one table's repairs, naming what the model got wrong."""
    kinds = {r["kind"] for r in rows}
    what = (
        "the model dropped the decimal point"
        if kinds == {REPAIR_POINT}
        else "the model flipped the minus sign"
        if kinds == {REPAIR_SIGN}
        else "the model dropped a decimal point or flipped a minus sign"
    )
    return (
        f"table {table_id}: {len(rows)} cell(s) restored from the page's text layer ({what}): "
        + ", ".join(f"{r['old']} -> {r['new']}" for r in rows)
    )


@dataclass(frozen=True)
class PointRefusal:
    md_line: int  # -1 for a refusal of the whole table
    label: str
    why: tuple[str, ...]
    detail: str = ""


def _alignments(c: _Cand, n_values: int) -> list[tuple[list[int], list[str]]]:
    """The ways candidate *c*'s numbers line up with a row of *n_values* values: all of them, or
    all but a leading one (a numeric row label leads the line)."""
    idx = [i for i in range(c.start, c.stop) if c.line.words[i].num is not None]
    nums = [c.line.words[i].num for i in idx]
    out = []
    if len(nums) == n_values:
        out.append((idx, nums))
    if idx and idx[0] == c.start and len(nums) - 1 == n_values and len(nums) > 1:
        out.append((idx[1:], nums[1:]))
    return out


def _value_word_keys(c: _Cand, aidx: list[int]) -> frozenset:
    """The text-layer words under *aidx*, by position: the same printed words whichever region
    the lines were built from."""
    return frozenset((w.x0, w.y0, w.x1, w.y1, w.text) for w in (c.line.words[i] for i in aidx))


def _is_detached_sign(text: str) -> bool:
    return text.translate(_UNICODE_MINUS).strip() in ("-", "+")


def _rival_lines(r: _Row, hit: dict, page_cands: list[_Cand]) -> list[str]:
    """Condition 7: page lines other than *hit* that could be row *r*'s own line.  A line counts
    when it prints the row's label exactly (the hit's rule; a sign printed as a word of its own
    before the first value is read as the value's, not the label's), carries the same number of values,
    shares no value word with the hit, and either differs from the row in at most
    ``MAX_RIVAL_MISMATCHES`` values or is itself a point-only hit (every difference a point drop,
    ``MIN_AGREEING_VALUES`` agreeing)."""
    out: list[str] = []
    for c in page_cands:
        for aidx, anums in _alignments(c, len(r.numbers)):
            if _value_word_keys(c, aidx) & hit["words"]:
                continue  # the hit's own printed words
            # a sign printed as its own word before the first value (`Previous - 7`) belongs to
            # the value, not to the label: that line prints the label `Previous`
            lead = aidx[0]
            while lead > c.start and _is_detached_sign(c.line.words[lead - 1].text):
                lead -= 1
            printed = _printed_label(c, lead)
            if not _labels_exact(r.label, printed, strict=True):
                continue
            mism = [(s, n) for s, n in zip(r.numbers, anums) if s != n]
            point_only = (
                bool(mism)
                and all(point_dropped(s, n) for s, n in mism)
                and len(anums) - len(mism) >= MIN_AGREEING_VALUES
            )
            if len(mism) <= MAX_RIVAL_MISMATCHES or point_only:
                out.append(f"{printed!r}: {len(mism)} of {len(anums)} values differ")
    return out


def _carries_minus(text: str) -> bool:
    """The printed word itself holds a minus glyph (hyphen-minus, U+2212, en dash, figure dash).
    A parenthesised negative ``(0.3)`` does not: that is notation, never a sign evidence."""
    return "-" in text.translate(_UNICODE_MINUS)


# A detached glyph counts as sitting in a number's line band when its vertical extent, widened by this
# fraction of the NUMBER's height, meets the number's extent.  Same constant and same reading as the line
# builder's rule (a word joins a line when its centre is within LINE_CENTRE_TOL of a height of the line
# centre): the builder asks about centres, this asks the same slack of the glyph's whole extent, because
# a minus glyph is a thin box that the builder may place anywhere in the number's band or just outside it.
# A glyph the builder put on a line of its own is exactly the case this check exists for.
SIGN_BAND_TOL = LINE_CENTRE_TOL

_RANGE_DASHES = "\u2013\u2014\u2012"  # en dash, em dash, figure dash: a minus or a range separator


def _is_word_box(w, t) -> bool:
    return w[0] == t.x0 and w[2] == t.x1 and str(w[4]) == t.text


def _band_neighbours(words: list, t) -> list:
    """The raw page *words* whose vertical extent, widened by ``SIGN_BAND_TOL`` of the height of *t*,
    meets the vertical extent of word *t* (the same printed line band, or just outside it), other than
    *t*, sorted by x.  Geometry only: word order is never used."""
    slack = SIGN_BAND_TOL * (t.y1 - t.y0)
    out = [
        w
        for w in words
        if str(w[4]).strip()
        and w[3] + slack >= t.y0
        and w[1] - slack <= t.y1
        and not _is_word_box(w, t)
    ]
    return sorted(out, key=lambda w: w[0])


def _in_own_line(w, t) -> bool:
    """The centre of raw word *w* lies inside the vertical extent of *t* (the strict reading)."""
    return t.y0 <= (w[1] + w[3]) / 2 <= t.y1


# A minus DRAWN in front of a number (a vector segment, absent from the page words) is no longer than the
# number it signs: a table rule spans a column, a minus does not.  Length is measured in widths of the
# number's own box, so no point size is assumed; "roughly horizontal" is a segment no taller than long.
DRAWN_MINUS_MAX_NUMBER_WIDTHS = 1.0


def _drawn_minus_candidates(drawn, t) -> list:
    """The drawn boxes shaped like a minus before *t* (no longer than ``DRAWN_MINUS_MAX_NUMBER_WIDTHS``
    of its width, no taller than long), as pseudo word boxes (a glyph ``-``)."""
    limit = DRAWN_MINUS_MAX_NUMBER_WIDTHS * (t.x1 - t.x0)
    return [
        (b[0], b[1], b[2], b[3], "-")
        for b in drawn
        if 0 < b[2] - b[0] <= limit and (b[3] - b[1]) <= (b[2] - b[0])
    ]


def _detached_minus_attached(words: list, t, drawn=()) -> bool:
    """Is a detached minus glyph (a word that is only ``-``) attached to the number word *t*, or
    impossible to show NOT attached?  Every minus-only word in the (widened) band of *t* is tested; the
    answer is True if ANY is attached.  A glyph is attached when it overlaps the number's box, when it
    has no word on its left in the number's own line, or when it lies closer to *t* than to the word on
    its own left (a glyph exactly midway cannot be shown NOT to be attached, so it counts as attached
    and the caller refuses).  The distance rule reads x positions only, never word order.

    *drawn* are vector boxes (reading frame): those shaped like a minus and inside the same band are
    candidates exactly like glyph words, but they are never a "word on its left"."""
    band = _band_neighbours(words, t)
    slack = SIGN_BAND_TOL * (t.y1 - t.y0)
    marks = [
        m
        for m in band
        if _is_detached_sign(str(m[4])) and "-" in str(m[4]).translate(_UNICODE_MINUS)
    ] + [
        d
        for d in _drawn_minus_candidates(drawn or (), t)
        if d[3] + slack >= t.y0 and d[1] - slack <= t.y1
    ]
    for m in marks:
        if m[2] > t.x0 and m[0] < t.x1:  # the boxes overlap: ambiguous, so attached
            return True
        if m[0] >= t.x0:  # to the right of the number
            continue
        before = [w for w in band if w is not m and w[2] <= m[0] and _in_own_line(w, t)]
        if not before:
            return True
        gap_to_number = t.x0 - m[2]
        gap_to_previous = m[0] - max(w[2] for w in before)
        if gap_to_number <= gap_to_previous:
            return True
    return False


def _unary_dash(words: list, t) -> bool:
    """Is the minus glyph of the printed word *t* UNARY: the word starts with the dash (any dash:
    hyphen-minus, en, em, figure, U+2212), so it is not glued to a number on its left, and the nearest
    word on its left in the number's own band is not a number?  ``2.3 -2.6`` and ``2.3 \u20132.6`` are
    range-shaped: the dash may separate the two numbers instead of signing the second, and nothing in
    the row or the table can say which.  Refused, whatever the dash glyph."""
    if "-" not in t.text.lstrip()[:1].translate(_UNICODE_MINUS):
        return False
    left = [w for w in _band_neighbours(words, t) if w[2] <= t.x0 and _in_own_line(w, t)]
    if not left:
        return True
    m = max(left, key=lambda w: w[2])
    return _canon_number(str(m[4]).translate(_UNICODE_MINUS)) is None


def _sign_attached_by_geometry(
    words: list, c: _Cand, aidx: list[int], cell: int, native: str, drawn: list | None = ()
) -> bool:
    """Plan A2 step 2: is the minus (or its absence) of text-layer value *cell* settled by the
    geometry of the layer?  The layer's value is negative (a minus would be ADDED): its own word must
    carry the minus glyph (a ``(0.3)`` is notation, not evidence) and the dash must be unary
    (:func:`_unary_dash`); a range-shaped token is refused.  The layer's value is positive (a minus
    would be REMOVED): no detached minus glyph may be attached to it (the layer would then print
    ``- 3.0`` and the model's ``-3.0`` would be right), and no drawn minus either: a short, roughly
    horizontal vector segment (*drawn*, reading frame) attached the same way is a minus the text
    layer cannot show.  ``drawn=None`` means the drawings could not be read: removal is refused.
    Never by word order."""
    w = c.line.words[aidx[cell]]
    if native.startswith("-"):
        return _carries_minus(w.text) and _unary_dash(words, w)
    if drawn is None:
        return False
    return not _detached_minus_attached(words, w, drawn)


def _layer_encodes_minus(
    c: _Cand,
    aidx: list[int],
    mism: list,
    lines: list[_Line],
    bound_rows: list[tuple[int, set[str]]],
) -> bool:
    """Plan A2 step 2: the layer provably encodes minus signs in this table: at least one OTHER
    value, negative in the layer as a glyph on its word, agrees with the shipped one.  Looked for
    on the row's own agreeing values and on every BOUND row of the table (``bound_rows``: native
    line index and the canonical values the row shipped)."""
    off = {k for k, _, _ in mism}
    for k, i in enumerate(aidx):
        w = c.line.words[i]
        if k not in off and w.num is not None and w.num.startswith("-") and _carries_minus(w.text):
            return True
    for line_index, shipped in bound_rows:
        for w in lines[line_index].words:
            if (
                w.num is not None
                and w.num.startswith("-")
                and w.num in shipped
                and _carries_minus(w.text)
            ):
                return True
    return False


def _mirror_rows(r: _Row, hit: dict, page_cands: list[_Cand]) -> list[str]:
    """Plan A2 step 2: page lines other than *hit* that carry the row's ABSOLUTE values at every
    position, whatever their label and however far away (a ``Previous`` / ``Revision`` / ``Change``
    row mirrors the row above it with the opposite sign: sign is one bit, and a mirror is common).
    Label-agnostic on purpose: it is a superset of "the same label, or a Previous / Revision /
    Change-type row", so no vocabulary of row names is needed."""
    absolute = [n.lstrip("-") for n in r.numbers]
    out: list[str] = []
    for c in page_cands:
        for aidx, anums in _alignments(c, len(r.numbers)):
            if _value_word_keys(c, aidx) & hit["words"]:
                continue
            if [n.lstrip("-") for n in anums] == absolute:
                lead = aidx[0]
                out.append(f"{_printed_label(c, lead)!r}: {' '.join(anums)}")
    return out


def propose_point_repairs(
    words: list,
    region: tuple[float, float, float, float] | None,
    table_markdown: str,
    drawn: list | None = (),
) -> tuple[str, list[PointRepair], list[PointRefusal]]:
    """``(repaired_markdown, repairs, refusals)`` for one table block.  *drawn* are the page's vector
    boxes in the reading frame (``None``: unreadable, so no minus is removed); see
    :func:`_sign_attached_by_geometry`.

    A model that drops a printed decimal point ships ``-7`` for ``-.7``.  The row then equals no
    native line, so the binder calls it UNVERIFIABLE.  A shipped data row is repaired only when
    ALL of these hold (the numbers are quoted by the tests and the mutation record):

    0. the table is not CONTRADICTED: those are withheld, and nothing derived from the text layer
       may reach the re-read of a withheld table (#1267 follow-up);
    1. no native line carries its values exactly (``_options`` empty): a number the page prints is
       never changed;
    2. exactly one native line/panel has the same number of values, prints the row's label EXACTLY
       before them (``_labels_exact(strict=True)``, the BOUND gate's rule), and every differing
       value pair is a :func:`point_dropped`;
    3. at least ``MIN_AGREEING_VALUES`` other values on that line equal the row's, by position;
    4. no other shipped row has an exact option on that line, and the line lies strictly between
       the lines of the nearest BOUND rows above and below (reading order);
    5. the data cells carry no markdown decoration (the substitution is token-exact);
    6. after the substitution the production check binds that row BOUND, no row that was BOUND
       loses it, and the table does not turn CONTRADICTED;
    7. no OTHER line anywhere on the page (not just in the region) prints the row's exact label
       with the same number of values and is within ``MAX_RIVAL_MISMATCHES`` values of the row, or
       is itself a point-only hit for the row (:func:`_rival_lines`).

    The number written is the text layer's printed string.  Pure and idempotent."""
    base = check_row_binding(words, region, table_markdown, extra_evidence=False)
    if base.outcome is TableOutcome.CONTRADICTED:  # condition 0
        return table_markdown, [], [PointRefusal(-1, "", (REFUSE_CONTRADICTED,))]
    lines = build_native_lines(words, region)
    cands = _candidates(lines)
    # condition 1 looks at the WHOLE page: the table's box may miss the row's own line (and hold
    # another table's line with the same label), so "no exact line" is asked of every line.
    page_cands = _candidates(build_native_lines(words, None))
    rows = [r for r in _shipped_rows(table_markdown) if not r.is_header]
    claimed: set[int] = set()
    row_opts: dict[int, list] = {}
    for r in rows:
        if not r.has_content:
            continue
        o = _options(r, cands)
        row_opts[r.line_index] = o
        claimed.update(c.line.index for c, _ in o)
    bound_anchor = {
        e.line_index: e.native_line
        for e in base.rows
        if e.verdict is RowVerdict.BOUND and e.native_line is not None
    }
    md = table_markdown.splitlines()
    repairs: list[PointRepair] = []
    refusals: list[PointRefusal] = []
    for pos, r in enumerate(rows):
        if not r.numbers or row_opts.get(r.line_index) or _options(r, page_cands):  # condition 1
            continue
        hits: list[dict] = []
        near: list[dict] = []
        for c in cands:
            for aidx, anums in _alignments(c, len(r.numbers)):
                pairs = list(zip(r.numbers, anums))
                mism = [(k, s, n) for k, (s, n) in enumerate(pairs) if s != n]
                if not mism:
                    continue
                pair_kind = {k: _classify_pair(s, n) for k, s, n in mism}
                kinds = set(pair_kind.values())
                printed = _printed_label(c, aidx[0])
                label_ok = bool(r.label.strip()) and _labels_exact(r.label, printed, strict=True)
                agree = len(pairs) - len(mism)
                rec = {
                    "line": c.line.index,
                    "first_word": aidx[0],
                    "printed": printed,
                    "label_ok": label_ok,
                    "agree": agree,
                    "mism": [(k, s, n, c.line.words[aidx[k]].text) for k, s, n in mism],
                    "kinds": kinds,
                    "pair_kind": pair_kind,
                    "cand": c,
                    "aidx": aidx,
                    "mismatches": mism,
                    "words": _value_word_keys(c, aidx),
                }
                if (
                    kinds
                    <= _REPAIRABLE_PAIRS  # condition 2: every difference is a point drop OR a sign flip
                    and label_ok  # condition 2 (label)
                    and agree >= MIN_AGREEING_VALUES  # condition 3
                ):
                    hits.append(rec)
                elif kinds <= _REPAIRABLE_PAIRS or (label_ok and len(mism) <= MAX_RIVAL_MISMATCHES):
                    near.append(rec)
        if not hits:
            if near:
                why: set[str] = set()
                for n in near:
                    if not n["kinds"] <= _REPAIRABLE_PAIRS:
                        why.add("pair:" + ",".join(sorted(n["kinds"])))
                    elif not n["label_ok"]:
                        why.add(REFUSE_LABEL)
                    else:
                        why.add(REFUSE_NO_AGREEING)
                refusals.append(
                    PointRefusal(
                        r.line_index, r.label, tuple(sorted(why)), repr(near[0]["mism"][:3])
                    )
                )
            continue
        if len({(h["line"], h["first_word"]) for h in hits}) > 1:
            refusals.append(PointRefusal(r.line_index, r.label, (REFUSE_AMBIGUOUS,)))
            continue
        h = hits[0]
        rivals = _rival_lines(r, h, page_cands)
        if rivals:  # condition 7
            refusals.append(PointRefusal(r.line_index, r.label, (REFUSE_RIVAL_LINE,), rivals[0]))
            continue
        if "sign_flipped" in h["kinds"]:  # the sign conditions (plan A2 step 2)
            flipped = [m for m in h["mismatches"] if h["pair_kind"][m[0]] == "sign_flipped"]
            if not all(
                _sign_attached_by_geometry(words, h["cand"], h["aidx"], k, n, drawn)
                for k, _, n in flipped
            ):
                refusals.append(PointRefusal(r.line_index, r.label, (REFUSE_SIGN_GEOMETRY,)))
                continue
            bound_rows = [
                (e.native_line, {_canon_number(v) for v in e.values})
                for e in base.rows
                if e.verdict is RowVerdict.BOUND and e.native_line is not None
            ]
            if not _layer_encodes_minus(h["cand"], h["aidx"], h["mismatches"], lines, bound_rows):
                refusals.append(
                    PointRefusal(r.line_index, r.label, (REFUSE_SIGN_NO_MINUS_EVIDENCE,))
                )
                continue
            mirrors = _mirror_rows(r, h, page_cands)
            if mirrors:
                refusals.append(
                    PointRefusal(r.line_index, r.label, (REFUSE_SIGN_MIRROR,), mirrors[0])
                )
                continue
        if h["line"] in claimed:  # condition 4 (one-to-one)
            refusals.append(PointRefusal(r.line_index, r.label, (REFUSE_CLAIMED,)))
            continue
        above = [
            bound_anchor[rows[q].line_index]
            for q in range(pos)
            if rows[q].line_index in bound_anchor
        ]
        below = [
            bound_anchor[rows[q].line_index]
            for q in range(pos + 1, len(rows))
            if rows[q].line_index in bound_anchor
        ]
        if not (
            max(above, default=-1) < h["line"] < min(below, default=len(lines))
        ):  # condition 4 (reading order)
            refusals.append(PointRefusal(r.line_index, r.label, (REFUSE_ORDER,)))
            continue
        cells = _split_row(md[r.line_index])
        if any(_MD_DECORATION.search(c) for c in cells[1:]):  # condition 5
            refusals.append(PointRefusal(r.line_index, r.label, (REFUSE_DECORATED,)))
            continue
        targets = {m[0]: m for m in h["mism"]}
        k, ok, affix_ok, subs, new_cells = 0, True, True, [], [cells[0]]
        for cell in cells[1:]:
            out_toks = []
            for t in cell.split():
                if _canon_number(t.translate(_UNICODE_MINUS)) is None:
                    out_toks.append(t)
                    continue
                if k in targets:
                    _, _, n, printed_text = targets[k]
                    if _canon_number(printed_text.translate(_UNICODE_MINUS)) != n:
                        ok = False
                    # A sign is one glyph of the cell: a cell holding other tokens beside the number
                    # (a detached `-`, `n.a.`) is not edited by token (`- 7` would become `- -7`).
                    if h["pair_kind"][k] == "sign_flipped" and len(cell.split()) != 1:
                        ok = False
                    # The shipped trailing-point form (`-2.` for `-.2`) is the dropped point itself,
                    # not an affix (ONE terminal period only: `-7..` still refuses); the PRINTED token has no such licence (`-.7,` refuses).
                    if _affixes(t.removesuffix(".")) != _affixes(printed_text):
                        affix_ok = False
                    subs.append((t, printed_text, k, _PAIR_TO_REPAIR_KIND[h["pair_kind"][k]]))
                    out_toks.append(printed_text)
                else:
                    out_toks.append(t)
                k += 1
            new_cells.append(" ".join(out_toks))
        if not affix_ok:
            refusals.append(PointRefusal(r.line_index, r.label, (REFUSE_AFFIX,)))
            continue
        if not ok or k != len(r.numbers):
            refusals.append(PointRefusal(r.line_index, r.label, (REFUSE_TOKEN_MAP,)))
            continue
        trial = list(md)
        trial[r.line_index] = "| " + " | ".join(new_cells) + " |"
        after = check_row_binding(words, region, "\n".join(trial), extra_evidence=False)
        after_by = {e.line_index: e for e in after.rows}
        mine = after_by.get(r.line_index)
        worse = any(
            e.verdict is RowVerdict.BOUND
            and (a := after_by.get(e.line_index)) is not None
            and a.verdict is not RowVerdict.BOUND
            for e in base.rows
        ) or (after.outcome is TableOutcome.CONTRADICTED)
        if mine is None or mine.verdict is not RowVerdict.BOUND or worse:  # condition 6
            refusals.append(
                PointRefusal(
                    r.line_index,
                    r.label,
                    (REFUSE_WORSENS,) if worse else (REFUSE_NOT_BOUND,),
                    mine.reason if mine else "row absent after",
                )
            )
            continue
        md = trial
        base = after  # later rows are judged against the repaired table
        for old, new, col, kind in subs:
            repairs.append(
                PointRepair(
                    r.line_index,
                    r.label,
                    col,
                    old,
                    new,
                    h["line"],
                    h["printed"],
                    h["agree"],
                    len(r.numbers),
                    kind,
                )
            )
    return "\n".join(md), repairs, refusals


# ---------------------------------------------------------------------------
# Page helper
# ---------------------------------------------------------------------------


def _page_frame(page, regions_raw):
    """``(words, per-block-box source, rotation)``: the page's words in the reading frame and its
    detected table boxes (raw page frame) rotated with the same pivot and rotation the rowizer
    uses, as ``(boxes, union)``.  Shared by the check and the repair so they see one geometry."""
    from socr.tables.native_first import upright_words_for_page
    from socr.tables.reconstruct import _rotate_point, _words_center

    raw = list(page.get_text("words"))
    uw, rotation = upright_words_for_page(page)
    words = uw if rotation else raw
    cx, cy = _words_center(raw) if raw else (0.0, 0.0)

    def rotate_box(b):
        if not rotation:
            return tuple(b)
        pts = [_rotate_point(x, y, cx, cy, rotation) for x in (b[0], b[2]) for y in (b[1], b[3])]
        return (
            min(p[0] for p in pts),
            min(p[1] for p in pts),
            max(p[0] for p in pts),
            max(p[1] for p in pts),
        )

    boxes = [rotate_box(b) for b in (regions_raw or [])]
    union = (
        (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )
        if boxes
        else None
    )
    return words, (boxes, union), rotation


def check_page_row_binding(
    page, markdown: str, regions_raw: list[tuple[float, float, float, float]] | None = None
) -> tuple[list[RowBindingResult], int]:
    """Run :func:`check_row_binding` for every table block of *markdown* on fitz *page*.

    Words are rotated into the reading frame with ``native_first.upright_words_for_page`` and the
    detected table boxes (raw page frame) with the same pivot and rotation the rowizer uses
    (``reconstruct._words_center`` / ``_rotate_point``).  Block *i* takes box *i* when there is one box
    per block; otherwise every block takes the union of the boxes (a superset: it can only add
    ambiguity, never a false line).  Returns ``(results per block, rotation)``."""
    words, (boxes, union), rotation = _page_frame(page, regions_raw)
    md_lines = markdown.splitlines()
    blocks = find_table_blocks(markdown)
    results = []
    for i, b in enumerate(blocks):
        region = boxes[i] if len(boxes) == len(blocks) else union
        results.append(check_row_binding(words, region, "\n".join(md_lines[b.start : b.end + 1])))
    return results, rotation


def _drawn_boxes(page, rotation: int) -> list | None:
    """The boxes of every drawn path item of *page* in the reading frame (same pivot and rotation as
    the words), or ``None`` when the drawings cannot be read."""
    from socr.tables.reconstruct import _drawing_item_bbox, _rotate_point, _words_center

    try:
        drawings = page.get_drawings()
        cx, cy = _words_center(list(page.get_text("words"))) or (0.0, 0.0)
    except Exception:
        return None
    out = []
    for d in drawings:
        for item in d.get("items") or []:
            b = _drawing_item_bbox(item)
            if b is None:
                continue
            if rotation:
                pts = [
                    _rotate_point(x, y, cx, cy, rotation)
                    for x in (b[0], b[2])
                    for y in (b[1], b[3])
                ]
                b = (
                    min(p[0] for p in pts),
                    min(p[1] for p in pts),
                    max(p[0] for p in pts),
                    max(p[1] for p in pts),
                )
            out.append(b)
    return out


def propose_page_point_repairs(
    page, markdown: str, regions_raw: list[tuple[float, float, float, float]] | None = None
) -> tuple[str, list[tuple[int, list[PointRepair]]], list[tuple[int, list[PointRefusal]]]]:
    """:func:`propose_point_repairs` for every table block of *markdown* on fitz *page*, with the
    geometry of :func:`check_page_row_binding`.  Returns ``(repaired page markdown, [(block index,
    repairs)], [(block index, refusals)])``; only blocks with something to report are listed.
    Repairs change cell text only, so line numbers (``md_line`` is relative to the block) and the
    block boundaries are unchanged.

    A block that would scan a region shared with another block (no box per block: no boxes at
    all, or a box count that differs from the block count, with more than one block) gets NO
    repair and a ``REFUSE_SHARED_REGION`` refusal of the whole table.  The check may judge such a
    block against the union (a superset only adds ambiguity), but a repair WRITES a number: with
    two blocks over one region, the other table's line with the same label can be the only hit and
    its value lands in this table (review of #1282, finding 1)."""
    words, (boxes, union), rotation = _page_frame(page, regions_raw)
    drawn = _drawn_boxes(page, rotation)
    md_lines = markdown.splitlines()
    kept = markdown.splitlines(keepends=True)  # the page's own line endings survive a repair
    blocks = find_table_blocks(markdown)
    repaired: list[tuple[int, list[PointRepair]]] = []
    refused: list[tuple[int, list[PointRefusal]]] = []
    if len(blocks) > 1 and len(boxes) != len(blocks):
        shared = [PointRefusal(-1, "", (REFUSE_SHARED_REGION,), f"{len(boxes)} boxes")]
        return markdown, [], [(i, list(shared)) for i in range(len(blocks))]
    for i, b in enumerate(blocks):
        region = boxes[i] if len(boxes) == len(blocks) else union
        new, reps, refs = propose_point_repairs(
            words, region, "\n".join(md_lines[b.start : b.end + 1]), drawn
        )
        if refs:
            refused.append((i, refs))
        if reps:
            repaired.append((i, reps))
            for at, line in zip(range(b.start, b.end + 1), new.splitlines()):
                ending = kept[at][len(md_lines[at]) :]
                kept[at] = line + ending
    return "".join(kept), repaired, refused
