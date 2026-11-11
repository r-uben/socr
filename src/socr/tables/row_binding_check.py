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


@dataclass(frozen=True)
class RowBindingResult:
    outcome: TableOutcome
    rows: tuple[RowEvidence, ...]

    def count(self, verdict: RowVerdict) -> int:
        return sum(1 for r in self.rows if r.verdict is verdict)


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


def check_row_binding(
    words: list,
    region: tuple[float, float, float, float] | None,
    table_markdown: str,
) -> RowBindingResult:
    """Compare every shipped data row of *table_markdown* with the text layer.

    *words* are fitz word tuples in the reading frame; *region* is the table box in the same frame
    (None = whole page)."""
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
        evidence.append(RowEvidence(r.line_index, v, why, r.label, nat, values, c.line.index))
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
        evidence.insert(0, RowEvidence(h.line_index, v, why, h.label, nat, h.cell_texts, line_no))
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
    page_num: int, table_index: int, outcome: str, rows: list[dict], error: str = ""
) -> dict:
    """One table's ``ps.row_binding`` record (before ``withheld_markdown`` / ``repairs``)."""
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

# A numeric token split into what stands before the number, the number and what follows it
# (`-.7%` -> "", "-.7", "%").  The affixes of the shipped token and of the native one must be
# equal: a repair changes the point and nothing else (review of #1282, case E: `-7` for `-.7%`).
_NUMBER_AFFIX = re.compile(
    r"(?P<pre>[^\d.\-+\u2212\u2013\u2014\u2012]*)"
    r"(?P<num>[-+\u2212\u2013\u2014\u2012]?(?:\d[\d.,]*|\.\d[\d.,]*))"
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


def _classify_pair(shipped: str, native: str) -> str:
    """Why a differing pair is or is not a point drop (for the refusal reasons)."""
    if point_dropped(shipped, native):
        return "point_dropped"
    if shipped.startswith("-") != native.startswith("-"):
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


def propose_point_repairs(
    words: list,
    region: tuple[float, float, float, float] | None,
    table_markdown: str,
) -> tuple[str, list[PointRepair], list[PointRefusal]]:
    """``(repaired_markdown, repairs, refusals)`` for one table block.

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
    base = check_row_binding(words, region, table_markdown)
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
                kinds = {_classify_pair(s, n) for _, s, n in mism}
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
                    "words": _value_word_keys(c, aidx),
                }
                if (
                    kinds == {"point_dropped"}
                    and label_ok  # condition 2 (label)
                    and agree >= MIN_AGREEING_VALUES  # condition 3
                ):
                    hits.append(rec)
                elif kinds == {"point_dropped"} or (label_ok and len(mism) <= MAX_RIVAL_MISMATCHES):
                    near.append(rec)
        if not hits:
            if near:
                why: set[str] = set()
                for n in near:
                    if n["kinds"] != {"point_dropped"}:
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
                    if _affixes(t) != _affixes(printed_text):
                        affix_ok = False
                    subs.append((t, printed_text, k))
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
        after = check_row_binding(words, region, "\n".join(trial))
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
        for old, new, col in subs:
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
    words, (boxes, union), _rotation = _page_frame(page, regions_raw)
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
            words, region, "\n".join(md_lines[b.start : b.end + 1])
        )
        if refs:
            refused.append((i, refs))
        if reps:
            repaired.append((i, reps))
            for at, line in zip(range(b.start, b.end + 1), new.splitlines()):
                ending = kept[at][len(md_lines[at]) :]
                kept[at] = line + ending
    return "".join(kept), repaired, refused
