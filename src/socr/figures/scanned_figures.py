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
    r"[ \t]*(?:[.:—–\-]|$)",
    re.MULTILINE,
)

#: The shortest run of one-character lines that is an axis title spelled down the page.
#: Data, not a guess: 7 is the smallest run found on a page whose vertical text was confirmed by
#: eye to be an axis title (Forsythe-Lundholm p16); the longest run on any page that carries both a
#: caption and a table was 4 (Hansen p13), and table headers set vertically elsewhere in the sample
#: reach 25 but never share a page with a caption.
MIN_SPELLED_RUN = 7

SPELLED_FENCE_OPEN = "<!-- socr:spelled-axis-residue"
SPELLED_FENCE_CLOSE = "socr:end-spelled-axis-residue -->"


def has_figure_caption(layer_text: str) -> bool:
    """Whether *layer_text* carries a line that is a figure caption."""
    return bool(layer_text) and _CAPTION_LINE_RE.search(layer_text) is not None


def fence_spelled_runs(text: str) -> tuple[str, int]:
    """Move runs of >= ``MIN_SPELLED_RUN`` one-character lines into a fenced comment.

    Returns ``(text, lines_fenced)``. SEPARATED, never dropped: every line goes into the fence
    verbatim, so a wrong reading is never turned into a missing one without a trace, and a page
    with no such run is returned unchanged (``lines_fenced == 0``). Blank lines between the
    one-character lines belong to the run (a layer prints them that way) and travel with it.
    """
    if not text:
        return text, 0
    lines = text.split("\n")
    keep: list[str] = []
    fenced: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if len(lines[i].strip()) != 1:
            keep.append(lines[i])
            i += 1
            continue
        j = i
        last_char = i
        while j < n and (len(lines[j].strip()) == 1 or lines[j].strip() == ""):
            if len(lines[j].strip()) == 1:
                last_char = j
            j += 1
        run = [ln for ln in lines[i : last_char + 1] if ln.strip()]
        if len(run) >= MIN_SPELLED_RUN:
            fenced.extend(run)
            i = last_char + 1
        else:
            keep.extend(lines[i : last_char + 1])
            i = last_char + 1
    if not fenced:
        return text, 0
    # A literal ``-->`` cannot be in a run of one-character lines, but a lone ``>`` or ``-`` can
    # be a line; the fence is closed by its own sentinel, which no such line can spell.
    body = "\n".join(keep).rstrip()
    block = "\n".join(
        [
            SPELLED_FENCE_OPEN,
            "axis-title characters the scan's text layer printed one per line; the figure is in "
            "the page image on this page.",
            *fenced,
            SPELLED_FENCE_CLOSE,
        ]
    )
    return (f"{body}\n\n{block}" if body else block), len(fenced)
