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

#: The fence is VISIBLE: a note line and a ``text`` code block, never an HTML comment (a comment
#: vanishes in every rendered view, which turns "kept verbatim" into silent loss for a reader).
SPELLED_FENCE_NOTE = "[unreadable figure text from scan, kept verbatim]"
#: The delimiter is three backticks unless the page already holds a longer backtick run, in
#: which case it is one longer than the longest (CommonMark: a closing fence must be at least as
#: long as the opening one, so a shorter delimiter could be closed early by page content).
SPELLED_FENCE_OPEN = "```text"
SPELLED_FENCE_CLOSE = "```"


def _fence_delimiters(lines: list[str]) -> tuple[str, str]:
    longest = max((len(m) for ln in lines for m in re.findall(r"`+", ln)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}text", ticks


def _scan_code_fences(lines: list[str]) -> tuple[set[int], str]:
    """(indices of lines in an existing fenced code block, the delimiter still open at the end).

    The open delimiter is "" when every block is closed. A block left open runs to the end of the
    document in CommonMark, so anything appended after it renders as code.
    """
    inside: set[int] = set()
    marker = ""
    for i, ln in enumerate(lines):
        stripped = ln.lstrip()
        char = stripped[:1]
        run = len(stripped) - len(stripped.lstrip(char)) if char in "`~" else 0
        if marker:
            inside.add(i)
            if char == marker[0] and run >= len(marker) and not stripped[run:].strip():
                marker = ""
        elif run >= 3:
            inside.add(i)
            marker = char * run
    return inside, marker


def _code_fence_lines(lines: list[str]) -> set[int]:
    """Indices of lines inside, opening or closing an existing fenced code block."""
    return _scan_code_fences(lines)[0]


def close_open_code_fence(text: str) -> str:
    """Return *text* ready to have markdown appended: an unclosed fence gets its closer.

    Unchanged when no fenced block is open at the end. The closer is the matching delimiter on
    its own line, so an image reference appended after it renders as an image, not as code.
    """
    marker = _scan_code_fences(text.split("\n"))[1]
    return f"{text.rstrip()}\n{marker}" if marker else text


def _is_figure_junk(line: str) -> bool:
    """A lone character or a bare number: what a chart's tick and axis labels look like."""
    s = line.strip()
    return len(s) == 1 or bool(_BARE_NUMBER_RE.match(s) and any(c.isdigit() for c in s))


def has_figure_caption(layer_text: str) -> bool:
    """Whether *layer_text* carries a line shaped like a figure caption.

    A label at the start of a line is necessary, not sufficient: it must be short
    (``MAX_CAPTION_LINE``) or be followed by figure furniture, and a label that ENDS its line
    must not be followed by a lowercase continuation ("Figure 3" / "shows that ...").

    Deliberately NOT a sentence classifier. A prose line that opens with ``Figure 3. We find ...``
    and is short can still fire, and no word list can tell it from a short title without
    misreading real captions ("Figure 2. Annual reports."). That is harm-bounded and accepted: a
    false fire only ADDS one page-image reference beside unchanged text (the fence is a separate
    gate, on runs of ``MIN_SPELLED_RUN`` or more one-character lines outside tables, math and
    lists, so a prose page without such runs is otherwise byte-identical). Precision beyond the
    measured 17/17 is not worth more heuristics.
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


#: Dashes a native text layer uses as a list marker, beyond the glyphs ``born_digital`` already
#: names (``_LIST_MARKER_GLYPHS``: bullet, triangular bullet, white bullet, square, ...).
_EXTRA_BULLETS = "-*+\u2013\u2014\u2212\u2043"


def _bullet_markers() -> frozenset[str]:
    from socr.core.born_digital import LIST_MARKER_GLYPHS

    return frozenset(LIST_MARKER_GLYPHS) | frozenset(_EXTRA_BULLETS)


def _is_list_item(line: str, markers: frozenset[str]) -> bool:
    stripped = line.lstrip()
    if len(stripped) > 1 and stripped[0] in markers and stripped[1].isspace():
        return bool(stripped[1:].strip())
    return re.match(r"\d+[.)]\s+\S", stripped) is not None


_OPENERS = (("\\[", "\\]"), ("\\(", "\\)"), ("\\begin{", "\\end{"))


def _math_lines(lines: list[str]) -> set[int]:
    """Indices of lines inside, opening or closing math.

    Dollar math (``$$`` blocks, inline ``$...$`` spans) and LaTeX delimiters: ``\\[ \\]``,
    ``\\( \\)`` and ``\\begin{..} \\end{..}``, each tracked as a depth across lines.
    """
    inside: set[int] = set()
    display = False
    inline = False
    depth = [0] * len(_OPENERS)
    for i, line in enumerate(lines):
        starts_in = display or inline or any(depth)
        rest = line
        dd = rest.count("$$")
        rest = rest.replace("$$", "")
        singles = rest.count("$")
        if dd % 2:
            display = not display
        if singles % 2 and not display:
            inline = not inline
        latex = False
        for k, (op, cl) in enumerate(_OPENERS):
            opened, closed = line.count(op), line.count(cl)
            if opened or closed:
                latex = True
            depth[k] = max(0, depth[k] + opened - closed)
        if starts_in or display or inline or dd or singles or latex:
            inside.add(i)
    return inside


def fence_spelled_runs(text: str) -> tuple[str, int]:
    """Fence, IN PLACE, runs of >= ``MIN_SPELLED_RUN`` one-character lines.

    Returns ``(text, lines_fenced)``. Each run keeps its position and every one of its lines
    (blank lines between the characters included) verbatim, inside a visible ``text`` code block
    under a one-line note; deleting those three wrapper lines gives back the input byte for byte.
    Nothing moves and nothing merges. A page with no such run is returned unchanged.

    Abstains on a run that is part of something else: inside or beside a markdown table, inside
    ``$...$`` / ``$$`` / ``\\[ \\]`` / ``\\( \\)`` / ``\\begin..\\end`` math, or in a list (a neighbouring list item, or a run of bare bullet
    markers). A vertical table header, an equation and a list of single characters all look like
    an axis title to a line counter and are not one.
    """
    if not text:
        return text, 0
    lines = text.split("\n")
    n = len(lines)
    math = _math_lines(lines)
    code = _code_fence_lines(lines)
    fence_open, fence_close = _fence_delimiters(lines)
    markers = _bullet_markers()

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
            and not any(k in math or k in code for k in range(i, last + 1))
            and not before.lstrip().startswith("|")
            and not after.lstrip().startswith("|")
            and not _is_list_item(before, markers)
            and not _is_list_item(after, markers)
            and not all(c in markers for c in chars)
        )
        if eligible:
            out.extend([SPELLED_FENCE_NOTE, fence_open, *run, fence_close])
            fenced += len(chars)
        else:
            out.extend(run)
        i = last + 1
    if not fenced:
        return text, 0
    return "\n".join(out), fenced
