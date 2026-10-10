"""Paragraphs and split words in shipped native prose (#1074).

``page.get_text("text")`` returns one printed line per output line, so native prose ships
with the PDF's line endings: no paragraphs, and words split at a line-end hyphen. This
module restores both without changing a character except a removed line-end hyphen.

Two halves, joined by exact line content:

* **Analyze time** (:func:`page_paragraphs`, :func:`native_vocabulary`): paragraph
  boundaries are read from page geometry, and the document's own unhyphenated words and
  hyphenated compounds are collected. Both live on the in-memory state only.
* **Emit time** (:func:`reflow_native_prose`): called from the one finalize seam, it finds
  each analyze-time paragraph in the shipped text by exact whole-line match and joins it.
  It needs no geometry to be a no-op, so replay and resume (whose text is already
  reflowed) pass through unchanged.

The paragraph rule (the "B-prime" of ``docs/log/2026-10-10_1074-native-paragraphs.md``):

1. **Printed line first.** MuPDF lines that follow each other, move right, and sit on one
   baseline (within the smaller font size) are fragments of ONE printed line. Wide gaps and
   super/subscripts split a printed line into several MuPDF lines; fragments join with a
   space and are never a boundary.
2. **Boundary before printed line ``b``, previous line ``a``**, within one column:

   * a different column or size class is always a boundary, and so is a different font
     when ``a`` ends short (a heading); a font change on a full-width line is emphasis;
   * *indent test*: ``a`` ends short of its local environment's right edge ``R_env(a)`` by
     more than a word space AND ``b`` starts more than a word space right of the column's
     left edge. ``R_env`` is the right edge of the lines that share ``a``'s left edge, so
     an abstract or block quote (narrow, full-width inside itself) does not split on every
     line; it is the column's edge only when ``a``'s left edge is shared by no other line;
   * lines in one MuPDF block: the indent test alone decides;
   * lines in different blocks: the indent test, OR a baseline gap that is not within a
     word space of the modal pitch of ``a``'s size class. No pitch evidence keeps the
     printed (block) boundary.

Every yardstick (word space, pitch, edges) is measured on the page itself; the only fixed
number is :data:`SIZE_CLASS_PT`, the bucket that makes a size comparable across the
jitter of OCR text layers.

Residual failures, accepted because each is visible and none loses content: display
equation rows split; a hanging-indent reference list with no item spacing merges; an
equation row followed by prose at exactly one pitch merges; a paragraph continuing across
a column or page break stays split.

Hyphen rule (no dictionary; a library-wide wordlist would make one document's bytes depend
on the other PDFs, which breaks replay): ``x-`` + newline + lowercase loses its hyphen
only when the joined word appears unhyphenated elsewhere in the document's analyze-time
native text (the split occurrences themselves excluded). The hyphenated form seen anywhere
keeps the hyphen, ties keep it, and an unwitnessed split keeps it as printed (line break
removed, hyphen kept). A hyphen or dash glued to its word before anything else joins with
no space and removes nothing.
"""

from __future__ import annotations

import logging
import re
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Bucket, in points, that makes font sizes comparable. Born-digital sizes are exact;
#: the baked-in OCR layers of scans jitter by fractions of a point per line, and an
#: exact comparison would put a boundary on every line of such a page.
SIZE_CLASS_PT = 2.0

#: ``paragraph -> printed line -> MuPDF-line fragments``, all stripped. Tuples: this is
#: held on state across a run and must not be mutated by a consumer.
NativeParagraphs = tuple[tuple[tuple[str, ...], ...], ...]

_LETTER = r"[^\W\d_]"
_SPLIT = re.compile(rf"{_LETTER}+-\n{_LETTER}+")
_TAIL = re.compile(rf"({_LETTER}+)-$")
_HEAD = re.compile(rf"^({_LETTER}+)")
_WORD = re.compile(rf"{_LETTER}+")
_COMPOUND = re.compile(rf"{_LETTER}+(?:-{_LETTER}+)+")
_DASHES = "-–—"


@dataclass(frozen=True)
class NativeVocabulary:
    """What the document itself witnesses about a line-end split."""

    words: frozenset[str]
    #: ``left-right`` for every adjacent pair inside a hyphenated chain.
    compounds: frozenset[str]


@dataclass
class _Frag:
    block: int
    x0: float
    x1: float
    base: float
    size: float
    font: str
    text: str


@dataclass
class _Line:
    """One printed line: one or more same-baseline MuPDF lines."""

    block: int
    x0: float
    x1: float
    base: float
    size: float
    font: str
    frags: list[str]
    run: int = 0
    group: int = 0

    @property
    def size_class(self) -> int:
        return round(self.size / SIZE_CLASS_PT)


# --------------------------------------------------------------------------- analyze time


def native_vocabulary(texts: list[str]) -> NativeVocabulary:
    """Witnesses for the hyphen rule, from the document's analyze-time native text.

    The split occurrences are masked before counting: a word that appears ONLY as a
    line-end split witnesses nothing about itself.
    """
    joined = "\n".join(t for t in texts if t)
    masked = _SPLIT.sub(
        lambda m: " " if m.group(0).split("\n")[1][0].islower() else m.group(0), joined
    )
    masked = masked.lower()
    words = frozenset(_WORD.findall(masked))
    compounds: set[str] = set()
    for chain in _COMPOUND.findall(masked):
        parts = chain.split("-")
        compounds.update(f"{a}-{b}" for a, b in zip(parts, parts[1:]))
    return NativeVocabulary(words=words, compounds=frozenset(compounds))


def page_paragraphs(page) -> NativeParagraphs:
    """The page's paragraphs, from its geometry; ``()`` when the page cannot be measured."""
    import fitz

    from socr.core.born_digital import _median_word_space_width

    try:
        frags = _page_fragments(page, fitz)
        ws = _median_word_space_width(page.get_text("words"))
    except Exception as exc:  # noqa: BLE001 - geometry is advisory; printed lines stand
        logger.debug("native paragraphs: page not measurable (%s: %s)", type(exc).__name__, exc)
        return ()
    if not frags or not ws or ws <= 0:
        return ()
    return _segment(_printed_lines(frags), ws)


def _page_fragments(page, fitz) -> list[_Frag]:
    from socr.core.born_digital import (
        block_is_page_furniture,
        clean_native_text,
        dominant_text_direction,
    )

    blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT).get("blocks", [])
    dominant = dominant_text_direction(blocks)
    out: list[_Frag] = []
    block_idx = -1
    for block in blocks:
        if block.get("type", 0) != 0 or block_is_page_furniture(block, dominant):
            continue
        block_idx += 1
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            main = max(spans, key=lambda s: len(s["text"]))
            text = clean_native_text("".join(s["text"] for s in line["spans"]))[0].strip()
            if not text:
                continue
            x0, _y0, x1, _y1 = line["bbox"]
            out.append(
                _Frag(
                    block=block_idx,
                    x0=float(x0),
                    x1=float(x1),
                    base=float(main["origin"][1]),
                    size=float(main["size"]),
                    font=str(main.get("font", "")),
                    text=text,
                )
            )
    return out


def _printed_lines(frags: list[_Frag]) -> list[_Line]:
    """Merge same-baseline MuPDF lines into printed lines (step 1)."""
    groups: list[list[_Frag]] = []
    for f in frags:
        if groups:
            anchor, prev = groups[-1][0], groups[-1][-1]
            if (
                abs(f.base - anchor.base) < min(f.size, anchor.size)
                and f.x0 >= prev.x0
                and f.x0 >= prev.x1 - min(f.size, anchor.size)
            ):
                groups[-1].append(f)
                continue
        groups.append([f])
    lines = []
    for g in groups:
        main = max(g, key=lambda f: len(f.text))
        lines.append(
            _Line(
                block=g[0].block,
                x0=min(f.x0 for f in g),
                x1=max(f.x1 for f in g),
                base=main.base,
                size=main.size,
                font=main.font,
                frags=[f.text for f in g],
            )
        )
    return lines


def _clusters(values: list[float], tol: float) -> list[list[float]]:
    """Single-linkage clusters of ``values`` with neighbour gap <= ``tol``, ascending."""
    out: list[list[float]] = []
    for v in sorted(values):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return out


def _edge(values: list[float], tol: float, right: bool) -> float:
    """The page's own margin: the outermost member of the best-populated cluster.

    A tie in population goes to the outer cluster, so a margin is never read from the
    sparser side of a draw.
    """
    clusters = _clusters(values, tol)
    best = max(clusters, key=lambda c: (len(c), c[-1] if right else -c[0]))
    return best[-1] if right else best[0]


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def _segment(lines: list[_Line], ws: float) -> NativeParagraphs:
    if not lines:
        return ()
    # Runs: reading order restarts (baseline does not advance) or the next line shares no
    # horizontal extent with the previous one -> a new column run.
    run = 0
    lines[0].run = 0
    for prev, cur in zip(lines, lines[1:]):
        if cur.base <= prev.base or min(prev.x1, cur.x1) <= max(prev.x0, cur.x0):
            run += 1
        cur.run = run
    # Groups: runs whose own left edge agrees are one column.
    by_run: dict[int, list[_Line]] = defaultdict(list)
    for ln in lines:
        by_run[ln.run].append(ln)
    run_left = {r: _edge([ln.x0 for ln in ls], ws, right=False) for r, ls in by_run.items()}
    group_of: dict[int, int] = {}
    gid = -1
    last = None
    for r in sorted(run_left, key=lambda r: run_left[r]):
        if last is None or run_left[r] - last > ws:
            gid += 1
        group_of[r] = gid
        last = run_left[r]
    members: dict[int, list[_Line]] = defaultdict(list)
    for ln in lines:
        ln.group = group_of[ln.run]
        members[ln.group].append(ln)

    col_left = {g: _edge([ln.x0 for ln in ls], ws, right=False) for g, ls in members.items()}
    col_right = {g: _edge([ln.x1 for ln in ls], ws, right=True) for g, ls in members.items()}
    diffs: dict[tuple[int, int], list[float]] = defaultdict(list)
    for prev, cur in zip(lines, lines[1:]):
        if prev.run == cur.run and prev.size_class == cur.size_class:
            diffs[(prev.group, prev.size_class)].append(cur.base - prev.base)
    # One observation cannot disagree with itself: a pitch needs two.
    pitch = {k: _median(v) for k, v in diffs.items() if len(v) >= 2}

    def r_env(a: _Line) -> float:
        sharing = [ln.x1 for ln in members[a.group] if ln is not a and abs(ln.x0 - a.x0) <= ws]
        return _edge(sharing, ws, right=True) if sharing else col_right[a.group]

    def boundary(a: _Line, b: _Line) -> bool:
        if a.run != b.run or a.size_class != b.size_class:
            return True
        ends_short = a.x1 < r_env(a) - ws
        # A font change alone is not a boundary: an italic title or emphasis flips the
        # longest span of a line in the middle of a sentence. A heading is a font change
        # that ENDS SHORT; a full-width line cannot end a paragraph by font alone.
        if a.font != b.font and ends_short:
            return True
        indent = ends_short and b.x0 > col_left[b.group] + ws
        if a.block == b.block:
            return indent
        p = pitch.get((a.group, a.size_class))
        return indent or p is None or abs((b.base - a.base) - p) > ws

    paragraphs: list[list[_Line]] = [[lines[0]]]
    for a, b in zip(lines, lines[1:]):
        if boundary(a, b):
            paragraphs.append([b])
        else:
            paragraphs[-1].append(b)
    return tuple(tuple(tuple(ln.frags) for ln in para) for para in paragraphs)


# --------------------------------------------------------------------------- emit time


def _join_lines(a: str, b: str, vocabulary: NativeVocabulary) -> str:
    """Join printed line ``a`` to the one after it, ``b``."""
    tail, head = _TAIL.search(a), _HEAD.match(b)
    if tail and head and b[0].islower():
        left, right = tail.group(1).lower(), head.group(1).lower()
        joined_seen = (left + right) in vocabulary.words
        hyphenated_seen = f"{left}-{right}" in vocabulary.compounds
        if joined_seen and not hyphenated_seen:
            return a[:-1] + b
        return a + b
    if len(a) >= 2 and a[-1] in _DASHES and a[-2].isalnum():
        return a + b
    return a + " " + b


def _join_paragraph(paragraph, vocabulary: NativeVocabulary) -> str:
    printed = [" ".join(frags) for frags in paragraph]
    out = printed[0]
    for nxt in printed[1:]:
        out = _join_lines(out, nxt, vocabulary)
    return out


#: Anything that opens, closes or hides math, a comment or code. Both ends of every such
#: construct are in this set, so the first and last hit bracket whatever lies between.
_MARKER = re.compile(r"\$|\\\(|\\\)|\\\[|\\\]|\\begin\{|\\end\{|<!--|-->|^\s*(?:```|~~~)")


_FENCE_LINE = re.compile(r"^\s*(?:```|~~~)", re.MULTILINE)
_ENV = re.compile(r"\\(begin|end)\{([^}]*)\}")


def _markers_unbalanced(text: str) -> bool:
    """Whether the text opens something it never closes (or closes what it never opened).

    Single ``$`` is counted after removing ``$$``; currency makes it over-fire, which is
    the intended trade: a lost reflow is visible, a deleted minus is not.
    """
    envs: dict[str, int] = defaultdict(int)
    for kind, name in _ENV.findall(text):
        envs[name] += 1 if kind == "begin" else -1
    return (
        text.count("$$") % 2 == 1
        or text.replace("$$", "").count("$") % 2 == 1
        or text.count("\\[") != text.count("\\]")
        or text.count("\\(") != text.count("\\)")
        or text.count("\\begin{") != text.count("\\end{")
        or any(envs.values())
        or text.count("<!--") != text.count("-->")
        or len(_FENCE_LINE.findall(text)) % 2 == 1
    )


def _protected_lines(lines: list[str]) -> set[int]:
    """Lines a paragraph must never swallow, join across, or be joined into.

    Fail-closed: every line from the FIRST line carrying any math, comment or code marker
    through the LAST such line is protected, interior included (to the END of the text when
    the markers are unbalanced). Tracking open/close pairs
    leaked on every unbalanced, nested or multi-line-inline shape tried (`$x+` ... `+c$`,
    `\\[` inside `\\[`, `\\begin{a}` closed by `\\end{b}`), and a deleted minus there is a
    changed formula. Reflow still runs above the first marker and below the last. A LaTeX
    ``%`` comment joined onto the next line would comment it out, so a protected line is
    never joined in either direction. Tables (`|`), figure placeholders and CommonMark code
    / HTML blocks are protected line by line as before.
    """
    from socr.figures.chart_regions import markdown_literal_context

    protected = set(markdown_literal_context(lines)[0])
    marked = [i for i, line in enumerate(lines) if _MARKER.search(line)]
    if marked:
        # Unbalanced markers leave an environment open: its extent is unknowable, so the
        # protected span runs to the end of the text.
        end = len(lines) if _markers_unbalanced("\n".join(lines)) else marked[-1] + 1
        protected.update(range(marked[0], end))
    for i, line in enumerate(lines):
        st = line.strip()
        if "|" in st or st.startswith("!["):
            protected.add(i)
    return protected


def _reflow_once(
    text: str, paragraphs: NativeParagraphs, vocabulary: NativeVocabulary | None
) -> str:
    """Join each analyze-time paragraph found in ``text``; every other line is untouched.

    A paragraph is found only if ALL its lines are present, contiguous, in order, and
    unprotected; otherwise it stays as printed. A blank line is inserted only between two
    paragraphs that were adjacent in ``text``, so lines spliced in by another lane keep
    their adjacency.
    """
    lines = text.split("\n")
    keys = [ln.strip() for ln in lines]
    protected = _protected_lines(lines)
    at: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        at[k].append(i)

    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for paragraph in paragraphs:
        flat = [frag for printed in paragraph for frag in printed]
        n = len(flat)
        candidates = at.get(flat[0], [])
        for j in range(bisect_left(candidates, cursor), len(candidates)):
            start = candidates[j]
            if keys[start : start + n] == flat and not protected.intersection(
                range(start, start + n)
            ):
                spans.append((start, start + n, _join_paragraph(paragraph, vocabulary)))
                cursor = start + n
                break
    if not spans:
        return text

    out: list[str] = []
    prev_end = 0
    last_end = None
    for start, end, joined in spans:
        out.extend(lines[prev_end:start])
        if last_end == start and out and out[-1].strip():
            out.append("")
        out.append(joined)
        last_end = end
        prev_end = end
    out.extend(lines[prev_end:])
    return "\n".join(out)


def reflow_native_prose(
    text: str, paragraphs: NativeParagraphs, vocabulary: NativeVocabulary | None
) -> str:
    """Reflow ``text`` once, and only if the result is a fixed point of the reflow.

    Resume re-finalizes shipped text WITH regenerated geometry while replay re-reads it
    WITHOUT, so ``reflow(reflow(x))`` must equal ``reflow(x)``. A joined line can happen to
    equal a different geometry line (paragraphs ``alpha|beta`` and ``alpha beta|gamma`` over
    ``alpha beta gamma``); rather than argue that such a collision cannot occur, the result
    is verified and, if a second pass would still change it, the text ships as printed. The
    fallback is itself stable: ``reflow(x)`` is then ``x``, and ``x`` falls back again.
    """
    if not text or not paragraphs or vocabulary is None:
        return text
    once = _reflow_once(text, paragraphs, vocabulary)
    if once == text or _reflow_once(once, paragraphs, vocabulary) == once:
        return once
    return text
