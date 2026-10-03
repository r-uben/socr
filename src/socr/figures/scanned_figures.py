"""#1030: figure pages of a SCANNED document.

A scan carries no vector marks, so ``has_chart_marks`` / ``_is_chart_asset_page`` never claim
its figures, and the figure extractor skips every scanned page ("a scan has no localizable
figures"). A chart on such a page therefore ships as whatever the text layer or a model made of
it, and with no model and no ``--save-figures`` as nothing at all.

What a scan DOES carry is an invisible OCR text layer, and a figure's caption survives in it.
``docs/log/2026-10-03_scanned-figures.md`` scores that signal on 98 raster pages of three
documents (19 figure pages): a caption line in the layer fired on 17 pages, every one a figure
page, with 0 false fires on 29 table pages and 50 prose pages. Neighbouring signals do not hold
up alone -- a run of one-character lines is also how a table sets a vertical column header.

The detector identifies a PAGE, not a figure box: nothing a scan has (a page-sized image block,
no layout boxes recorded, no vector marks) isolates the region. The consumer therefore decides
per page and never removes text from it.
"""

from __future__ import annotations

import re

# A figure caption STARTS a line: ``Figure 7.--Market ...``, ``FIGURE 2``, ``Fig. 3:``. The label
# is followed by punctuation or the end of the line, which is what separates a caption from an
# in-text reference ("Figure 5 shows ..."). Measured on the 98-page sample: the same shape
# without the terminator also fires on 2 prose pages (wrapped lines that begin with a reference).
_CAPTION_LINE_RE = re.compile(
    r"^[ \t]*(?:FIGURE|Figure|FIG\.?|Fig\.?)[ \t]*"
    r"(?:[0-9]+[A-Za-z]?|[IVX]+|A[0-9]+)"
    r"([ \t]*(?:[.:\u2014\u2013\-]|$))",
    re.MULTILINE,
)
_BARE_NUMBER_RE = re.compile(r"^[\s\-\u2212\d.,()%$]+$")

#: The longest caption line seen on a caption-shaped line in the 98-page sample (50 characters,
#: Forsythe-Lundholm p24; the 27 caption lines run 8 to 50). Prose that wraps at the page's column
#: width is 79 to 87 characters, so a label that opens a LONG line is a sentence, not a caption --
#: unless the line after it is figure furniture (see ``_is_figure_junk``).
MAX_CAPTION_LINE = 50

#: The shortest run of one-character lines that is an axis title spelled down the page.
#: Data, not a guess: 7 is the smallest run found on a page whose vertical text was confirmed by
#: eye to be an axis title (Forsythe-Lundholm p16); the longest run on any page that carries both a
#: caption and a table was 4 (Hansen p13), and table headers set vertically elsewhere in the sample
#: reach 25 but never share a page with a caption.
MIN_SPELLED_RUN = 7

SPELLED_FENCE_OPEN = "<!-- socr:spelled-axis-residue"
SPELLED_FENCE_CLOSE = "socr:end-spelled-axis-residue -->"
SPELLED_FENCE_NOTE = (
    "axis-title characters the scan's text layer printed one per line; the figure is in "
    "the page image on this page."
)


def _is_figure_junk(line: str) -> bool:
    """A lone character or a bare number: what a chart's tick and axis labels look like."""
    s = line.strip()
    return len(s) == 1 or bool(_BARE_NUMBER_RE.match(s) and any(c.isdigit() for c in s))


def has_figure_caption(layer_text: str) -> bool:
    """Whether *layer_text* carries a line shaped like a figure caption.

    A label at the start of a line is necessary, not sufficient: it must be short
    (``MAX_CAPTION_LINE``) or be followed by figure furniture, and a label that ENDS its line
    must not be followed by a lowercase continuation ("Figure 3" / "shows that ...").
    """
    if not layer_text:
        return False
    lines = layer_text.split("\n")
    for idx, line in enumerate(lines):
        m = _CAPTION_LINE_RE.match(line)
        if m is None:
            continue
        nxt = next((x.strip() for x in lines[idx + 1 :] if x.strip()), "")
        if not line[m.end() :].strip() and len(nxt) > 1 and nxt[0].islower():
            continue
        if len(line.strip()) <= MAX_CAPTION_LINE or (nxt and _is_figure_junk(nxt)):
            return True
    return False


_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")
_BULLET_MARKERS = frozenset("-*+")


def _math_lines(lines: list[str]) -> set[int]:
    """Indices of lines inside, opening or closing a ``$$`` block or an inline ``$...$`` span."""
    inside: set[int] = set()
    display = False
    inline = False
    for i, line in enumerate(lines):
        starts_in = display or inline
        rest = line
        dd = rest.count("$$")
        rest = rest.replace("$$", "")
        singles = rest.count("$")
        if dd % 2:
            display = not display
        if singles % 2 and not display:
            inline = not inline
        if starts_in or display or inline or dd or singles:
            inside.add(i)
    return inside


def fence_spelled_runs(text: str) -> tuple[str, int]:
    """Fence, IN PLACE, runs of >= ``MIN_SPELLED_RUN`` one-character lines.

    Returns ``(text, lines_fenced)``. Each run keeps its position and every one of its lines
    (blank lines between the characters included) verbatim; the fence is two lines before it
    and one after, so deleting those three lines gives back the input byte for byte. Nothing
    moves and nothing merges. A page with no such run is returned unchanged.

    Abstains on a run that is part of something else: inside or beside a markdown table, inside
    ``$...$`` / ``$$`` math, or in a list (a neighbouring list item, or a run of bare bullet
    markers). A vertical table header, an equation and a list of single characters all look like
    an axis title to a line counter and are not one.
    """
    if not text:
        return text, 0
    lines = text.split("\n")
    n = len(lines)
    math = _math_lines(lines)

    def neighbour(i: int, step: int) -> str:
        j = i + step
        while 0 <= j < n and not lines[j].strip():
            j += step
        return lines[j] if 0 <= j < n else ""

    out: list[str] = []
    fenced = 0
    i = 0
    while i < n:
        if len(lines[i].strip()) != 1:
            out.append(lines[i])
            i += 1
            continue
        j = i
        last = i
        while j < n and (len(lines[j].strip()) == 1 or not lines[j].strip()):
            if len(lines[j].strip()) == 1:
                last = j
            j += 1
        run = lines[i : last + 1]
        chars = [ln.strip() for ln in run if ln.strip()]
        before, after = neighbour(i, -1), neighbour(last, 1)
        eligible = (
            len(chars) >= MIN_SPELLED_RUN
            and not any(k in math for k in range(i, last + 1))
            and not before.lstrip().startswith("|")
            and not after.lstrip().startswith("|")
            and not _LIST_ITEM_RE.match(before)
            and not _LIST_ITEM_RE.match(after)
            and not all(c in _BULLET_MARKERS for c in chars)
        )
        if eligible:
            out.extend([SPELLED_FENCE_OPEN, SPELLED_FENCE_NOTE, *run, SPELLED_FENCE_CLOSE])
            fenced += len(chars)
        else:
            out.extend(run)
        i = last + 1
    if not fenced:
        return text, 0
    return "\n".join(out), fenced
