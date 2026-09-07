"""GH-189: mandatory preservation of chart regions on mixed chart+table pages.

A page carrying BOTH chart marks and a table signal is deliberately held out of
the page-level chart-asset lane (GH-150 TICKET-B1) and keeps its normal route.
On that route the chart is represented by an inline placeholder that
``rowize_from_words_chart_aware`` emits into the page's NATIVE text -- and the
native text is only what ships when the judge rejects every ladder rung.  When
the judge ACCEPTS a rung, the accepted candidate governs and nothing ever
transferred the chart into it, so the chart left the document without a trace
(GH-189).

This module owns the representation-level answer to that: an inventory of the
page's detected chart regions that is independent of any candidate's text, and
a pure reconciliation pass that guarantees

    for each detected chart region on a mixed chart/table page, the final page
    markdown references its successfully rendered crop exactly once, in source
    order, regardless of which candidate won -- and every failed render or
    unresolved placement has a visible, durable disposition.

The denominator is DETECTED regions, not successful renders: "render nothing"
must not satisfy the invariant vacuously.

Placement honesty.  Arbitrary model markdown does not carry enough geometry to
guarantee source-relative placement universally.  Two source-grounded bindings
are attempted -- a unique surrounding native text line, then an unambiguous
1:1 table binding -- and when neither holds the crop is preserved in an
explicitly LABELLED unresolved block rather than appended as if its position
were known.  A silent append presented as success is the failure this module
exists to prevent, not an acceptable fallback.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Disposition of one detected chart region after reconciliation.
PLACED_ANCHOR = "placed_anchor"
PLACED_TABLE_BOUND = "placed_table_bound"
UNRESOLVED_PLACEMENT = "unresolved_placement"
RENDER_FAILED = "render_failed"

#: Dispositions that mean "this region's crop is referenced from the page body
#: at a position derived from the source geometry".
_PLACED = frozenset({PLACED_ANCHOR, PLACED_TABLE_BOUND})

#: Dispositions under which the crop IS referenced from the page body, whether
#: or not its position could be established.
PRESERVED_DISPOSITIONS = frozenset({PLACED_ANCHOR, PLACED_TABLE_BOUND, UNRESOLVED_PLACEMENT})

_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


@dataclass(frozen=True)
class ChartRegionAsset:
    """One detected chart region and the outcome of rendering its crop.

    ``region_index`` is 1-based and is the detector's own ordinal, which is what
    the crop filename encodes.  It is never renumbered: a later sort by source
    geometry produces a separate VIEW, so a filename always names the region the
    detector found, and two runs agree.
    """

    page_num: int
    region_index: int
    bbox: tuple[float, float, float, float]
    rel_path: str = ""
    rendered: bool = False
    error: str = ""

    @property
    def filename(self) -> str:
        return chart_region_filename(self.page_num, self.region_index)

    def to_dict(self) -> dict:
        return {
            "page_num": self.page_num,
            "region_index": self.region_index,
            "bbox": list(self.bbox),
            "rel_path": self.rel_path,
            "rendered": self.rendered,
            "error": self.error,
        }


@dataclass(frozen=True)
class ChartRegionOutcome:
    """What reconciliation did with one region."""

    page_num: int
    region_index: int
    disposition: str
    rel_path: str = ""
    detail: str = ""

    @property
    def placed(self) -> bool:
        return self.disposition in _PLACED

    @property
    def preserved(self) -> bool:
        return self.disposition in PRESERVED_DISPOSITIONS


def chart_region_filename(page_num: int, region_index: int) -> str:
    """The canonical crop filename for a region.

    Shared with ``rowize_from_words_chart_aware``'s placeholder and with
    ``_render_chart_region_pngs`` so all three name the same file.
    """
    return f"chart_region_p{page_num}_{region_index}.png"


# ---------------------------------------------------------------------------
# Source anchors
# ---------------------------------------------------------------------------


def _page_text_lines(page, exclude_bboxes) -> list[tuple[float, float, str]]:
    """Return ``(y0, y1, text)`` for each native word-row, chart words removed.

    Rows are built from ``get_text("words")`` rather than ``get_text("text")``
    so the same geometry that produced the chart bboxes decides which words are
    inside a chart.  Never raises; returns ``[]`` when word geometry is absent.
    """
    try:
        words = page.get_text("words") or []
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("chart_regions: get_text('words') failed: %s", exc)
        return []

    rows: dict[int, list[tuple]] = {}
    for w in words:
        wx0, wy0, _wx1, wy1 = w[0], w[1], w[2], w[3]
        if any(b.x0 <= wx0 <= b.x1 and b.y0 <= wy0 <= b.y1 for b in exclude_bboxes):
            continue
        rows.setdefault(round(wy0), []).append((wx0, wy0, wy1, w[4]))

    out: list[tuple[float, float, str]] = []
    for _key, row in sorted(rows.items()):
        row.sort(key=lambda t: t[0])
        text = " ".join(t[3] for t in row).strip()
        if not text:
            continue
        out.append((min(t[1] for t in row), max(t[2] for t in row), text))
    return out


def chart_region_anchors(page, bboxes) -> dict[int, tuple[str, str]]:
    """Return ``{region_index: (line_above, line_below)}`` source anchors.

    The anchors are the nearest native word-rows immediately above and below the
    region's (already label-expanded) bbox.  They are the only source-grounded
    handle a model's markdown can be matched against: the model rewrites layout
    freely but reproduces the page's own prose.

    Never raises.
    """
    try:
        lines = _page_text_lines(page, bboxes)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("chart_regions: anchor extraction failed: %s", exc)
        return {}

    anchors: dict[int, tuple[str, str]] = {}
    for idx, box in enumerate(bboxes, start=1):
        above = [ln for ln in lines if ln[1] <= box.y0]
        below = [ln for ln in lines if ln[0] >= box.y1]
        anchors[idx] = (
            above[-1][2] if above else "",
            below[0][2] if below else "",
        )
    return anchors


def table_bindings(
    bboxes,
    table_bboxes: list[tuple[float, float, float, float]],
) -> dict[int, str]:
    """Return ``{region_index: "before"|"after"}`` for an unambiguous binding.

    Only a 1:1 correspondence qualifies: exactly one detected source table, and
    the chart region wholly above or wholly below it.  Ordinal pairing across
    several tables is deliberately refused -- when counts or correspondence
    disagree the region is an unresolved placement, not a guess.
    """
    if len(table_bboxes) != 1:
        return {}
    _tx0, ty0, _tx1, ty1 = table_bboxes[0]
    out: dict[int, str] = {}
    for idx, box in enumerate(bboxes, start=1):
        if box.y1 <= ty0:
            out[idx] = "before"
        elif box.y0 >= ty1:
            out[idx] = "after"
    return out


# ---------------------------------------------------------------------------
# Owned artifacts: the exact strings this module writes, and how to find them
# ---------------------------------------------------------------------------


def image_ref(asset: ChartRegionAsset) -> str:
    return f"![chart region {asset.region_index}]({asset.rel_path})"


def render_failure_prefix(page_num: int, region_index: int) -> str:
    return f"> **Chart region {region_index} on page {page_num} was NOT preserved**"


def render_failure_marker(asset: ChartRegionAsset) -> str:
    """Visible, greppable marker for a region whose crop could not be rendered.

    Deliberately NOT a markdown image link: a link to a file that was never
    written renders as a broken image and is stripped downstream by
    ``strip_phantom_images``, which is precisely how this loss stayed silent.
    """
    reason = _normalize(asset.error) or "unknown error"
    return (
        f"{render_failure_prefix(asset.page_num, asset.region_index)} "
        f"— crop `{asset.filename}` failed to render ({reason}). "
        "The chart is present in the source PDF; no image was produced for it."
    )


def unresolved_placement_prefix(page_num: int) -> str:
    return f"> **Unresolved chart placement on page {page_num}**"


def unresolved_placement_note(page_num: int, indices: list[int]) -> str:
    listed = ", ".join(str(i) for i in indices)
    return (
        f"{unresolved_placement_prefix(page_num)} — the source position of "
        f"chart region(s) {listed} could not be located in the accepted text. The crop(s) "
        "below are preserved in source order; their position relative to the surrounding "
        "text and tables is NOT established."
    )


def _parse_image_token(line: str, start: int) -> tuple[int, str] | None:
    """Parse one complete ``![alt](dest "title")`` at *start*; ``(end, dest)`` or None.

    A real scan, not a prefix regex. ``[^)]*`` stops at the first ``)``, so an
    image whose title carries one -- ``![c](x.png "model (A)")`` -- was removed
    only as far as that paren and left ``") `` sitting in the prose. Handles
    escapes, a bracketed ``<dest>``, balanced parentheses in a bare destination,
    and a quoted or parenthesised title.
    """
    n = len(line)
    if not line.startswith("![", start):
        return None
    i, depth = start + 2, 1
    while i < n:
        c = line[i]
        if c == "\\":
            i += 2
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                break
        i += 1
    if i >= n or line[i] != "]":
        return None
    i += 1
    if i >= n or line[i] != "(":
        return None
    i += 1
    while i < n and line[i] in " \t":
        i += 1

    dest: list[str] = []
    if i < n and line[i] == "<":
        i += 1
        while i < n and line[i] != ">":
            if line[i] == "\\" and i + 1 < n:
                dest.append(line[i + 1])
                i += 2
                continue
            dest.append(line[i])
            i += 1
        if i >= n:
            return None
        i += 1
    else:
        depth = 0
        while i < n:
            c = line[i]
            if c == "\\" and i + 1 < n:
                dest.append(line[i + 1])
                i += 2
                continue
            if c in " \t":
                break
            if c == "(":
                depth += 1
            elif c == ")":
                if depth == 0:
                    break
                depth -= 1
            dest.append(c)
            i += 1

    while i < n and line[i] in " \t":
        i += 1
    if i < n and line[i] in "\"'(":
        closer = ")" if line[i] == "(" else line[i]
        i += 1
        while i < n and line[i] != closer:
            if line[i] == "\\" and i + 1 < n:
                i += 2
                continue
            i += 1
        if i >= n:
            return None
        i += 1
        while i < n and line[i] in " \t":
            i += 1
    if i >= n or line[i] != ")":
        return None
    return i + 1, "".join(dest)


def _in_ranges(index: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= index < end for start, end in ranges)


def live_image_tokens(line: str, literal: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
    """``(start, end, destination)`` for each LIVE image on *line*, in order.

    A token whose ``!`` falls inside a literal range is markdown the page is
    SHOWING, not a reference the page is making, so it is skipped -- it is
    neither removed nor counted toward the exactly-once check.
    """
    out: list[tuple[int, int, str]] = []
    i, n = 0, len(line)
    while i < n:
        if line[i] == "!" and not _in_ranges(i, literal):
            parsed = _parse_image_token(line, i)
            if parsed is not None:
                end, dest = parsed
                out.append((i, end, dest))
                i = end
                continue
        i += 1
    return out


def _block_literal_lines(lines: list[str]) -> set[int] | None:
    """Line numbers inside fenced/indented code or an HTML block, per CommonMark.

    Delegated to ``markdown-it-py`` -- already in the tree behind ``rich``, and
    declared for this use. Block context is where a hand-rolled scanner keeps
    being wrong in ways that EDIT accepted content: a boolean toggled on any
    fence-looking line closes a four-backtick block at an inner three-backtick
    example, and a multi-line HTML comment whose closing line carries text is
    left unprotected exactly where an anchor can match it. Both are settled
    rules with a reference implementation; reusing it beats re-deriving it.

    ``None`` when the tokenizer is unavailable, so the caller falls back to the
    scanner below rather than failing open on a document it cannot classify.
    """
    try:
        from markdown_it import MarkdownIt
    except Exception as exc:  # pragma: no cover - dependency is declared
        logger.debug("chart_regions: markdown-it unavailable (%s); using the line scanner", exc)
        return None
    try:
        tokens = MarkdownIt("commonmark").parse("\n".join(lines))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("chart_regions: markdown-it parse failed (%s); using the line scanner", exc)
        return None
    out: set[int] = set()
    for token in tokens:
        if token.type in ("fence", "code_block", "html_block") and token.map:
            out.update(range(token.map[0], min(token.map[1], len(lines))))
    return out


def _scanned_literal_lines(lines: list[str]) -> set[int]:
    """Fallback block scan: fences by delimiter CHARACTER and LENGTH, comments.

    Used only when the tokenizer is unavailable. A fence closes on the same
    character at least as long as the opening run and nothing else on the line,
    so a shorter inner fence is content and a tilde run cannot close a backtick
    block.
    """
    out: set[int] = set()
    open_char, open_len = "", 0
    in_comment = False
    for idx, line in enumerate(lines):
        if in_comment:
            out.add(idx)
            if "-->" in line:
                in_comment = False
            continue
        match = _FENCE_RE.match(line)
        if match:
            delim = match.group(1)
            char, length = delim[0], len(delim)
            if not open_char:
                open_char, open_len = char, length
                out.add(idx)
                continue
            body = line.strip()
            if char == open_char and length >= open_len and body == char * len(body):
                open_char, open_len = "", 0
                out.add(idx)
                continue
        if open_char:
            out.add(idx)
            continue
        if line.lstrip().startswith("<!--") and "-->" not in line:
            out.add(idx)
            in_comment = True
    return out


def markdown_literal_context(lines: list[str]) -> tuple[set[int], dict[int, list[tuple[int, int]]]]:
    """Where markdown is SHOWING image syntax rather than using it.

    Returns the wholly-literal line numbers (fenced or indented code, HTML
    blocks) and, per remaining line, the character ranges covered by inline code
    spans and inline HTML comments.

    One context pass, computed on the ORIGINAL body and shared by removal,
    anchor matching, insertion and the final liveness check. Independent regex
    exceptions per markdown construct would disagree about which bytes form an
    image, which is how a fence-only fix left inline code and comments open, and
    an inline-only fix left block fences and comment blocks open.

    Inline spans are resolved per line, with an unterminated comment carried
    forward; a backtick span split across a line break is rare enough, and
    failing to see one only means an owned reference inside it is treated as
    live.
    """
    block = _block_literal_lines(lines)
    literal_lines = set(block) if block is not None else _scanned_literal_lines(lines)

    ranges: dict[int, list[tuple[int, int]]] = {}
    in_comment = False
    for idx, line in enumerate(lines):
        if idx in literal_lines:
            ranges[idx] = []
            in_comment = False
            continue
        found: list[tuple[int, int]] = []
        i, n = 0, len(line)
        while i < n:
            if in_comment:
                k = line.find("-->", i)
                if k == -1:
                    found.append((i, n))
                    i = n
                else:
                    found.append((i, k + 3))
                    in_comment = False
                    i = k + 3
                continue
            if line.startswith("<!--", i):
                k = line.find("-->", i + 4)
                if k == -1:
                    found.append((i, n))
                    in_comment = True
                    i = n
                else:
                    found.append((i, k + 3))
                    i = k + 3
                continue
            if line[i] == "`":
                run = 1
                while i + run < n and line[i + run] == "`":
                    run += 1
                fence = "`" * run
                close = line.find(fence, i + run)
                while close != -1 and close + run < n and line[close + run] == "`":
                    close = line.find(fence, close + 1)
                if close == -1:
                    i += run
                    continue
                found.append((i, close + run))
                i = close + run
                continue
            i += 1
        ranges[idx] = found
        if in_comment:
            literal_lines.add(idx)
    return literal_lines, ranges


def code_fence_spans(lines: list[str]) -> list[tuple[int, int]]:
    """Inclusive line spans that are wholly literal, for insertion protection."""
    fenced, _ranges = markdown_literal_context(lines)
    spans: list[tuple[int, int]] = []
    for idx in sorted(fenced):
        if spans and spans[-1][1] == idx - 1:
            spans[-1] = (spans[-1][0], idx)
        else:
            spans.append((idx, idx))
    return spans


def _owns(target: str, filenames: set[str]) -> bool:
    return target.rsplit("/", 1)[-1] in filenames


def _strip_owned_tokens(line: str, filenames: set[str], literal: list[tuple[int, int]]) -> str:
    """Remove owned LIVE image tokens from *line*, leaving everything else intact.

    Token-level, never line-level: an owned reference embedded in a sentence must
    not take the sentence with it, and a line mixing an owned crop with an
    unrelated figure is not an all-or-nothing unit. Only the whitespace the
    removal itself collapsed is normalised, and only on a line actually edited.
    """
    hits = [t for t in live_image_tokens(line, literal) if _owns(t[2], filenames)]
    if not hits:
        return line
    out = line
    for start, end, _dest in reversed(hits):
        out = out[:start] + out[end:]
    return re.sub(r"[ \t]{2,}", " ", out).rstrip()


def _strip_owned(
    lines: list[str],
    filenames: set[str],
    prefixes: tuple[str, ...],
    context: tuple[set[int], dict[int, list[tuple[int, int]]]],
) -> list[str]:
    """Remove every owned artifact and the ONE blank separator it brought.

    Three cases, in order: a wholly-literal line (fenced code, or inside an HTML
    comment) is content and is never touched; a line that is one of this
    module's own markers is dropped whole; any other line has its owned LIVE
    image tokens removed -- skipping any that sit inside an inline code span or
    a comment on that line -- and is dropped only if that emptied it.

    Dropping a line exactly inverts ``_insert_block``, so reconciling this
    module's own output reproduces it byte-for-byte instead of accumulating
    blank lines.
    """
    literal_lines, literal_ranges = context
    rewritten: list[str | None] = []
    for i, line in enumerate(lines):
        if i in literal_lines:
            rewritten.append(line)
            continue
        stripped = line.strip()
        if stripped and any(stripped.startswith(prefix) for prefix in prefixes):
            rewritten.append(None)
            continue
        new = _strip_owned_tokens(line, filenames, literal_ranges.get(i, []))
        if new != line and not new.strip() and stripped:
            rewritten.append(None)  # the line was nothing but owned references
        else:
            rewritten.append(new)

    if all(v is not None for v in rewritten):
        # Nothing was dropped, so no blank separator has to be reclaimed -- but
        # a line may still have had an inline token removed from it.
        return [v for v in rewritten if v is not None]

    out: list[str] = []
    i, n = 0, len(rewritten)
    while i < n:
        value = rewritten[i]
        if value is not None:
            out.append(value)
            i += 1
            continue
        i += 1
        if out and not out[-1].strip():
            if i < n and rewritten[i] is not None and not rewritten[i].strip():
                i += 1  # drop the blank that followed
            elif i >= n:
                out.pop()  # end of body: drop the blank that preceded
        elif not out:
            if i < n and rewritten[i] is not None and not rewritten[i].strip():
                i += 1
    return out


# ---------------------------------------------------------------------------
# Protected blocks and slot resolution
# ---------------------------------------------------------------------------


def protected_spans(lines: list[str]) -> list[tuple[int, int]]:
    """Inclusive line spans no insertion may land inside: tables and code fences.

    Splitting an accepted table with an image and two blank lines destroys the
    structure the page was selected for (finding 1). Placement matching decides
    POSITION only; it may never rewrite what the winner authored.
    """
    from socr.tables.reconcile import find_table_blocks

    spans = [(b.start, b.end) for b in find_table_blocks("\n".join(lines))]
    spans.extend(code_fence_spans(lines))
    return spans


def _inside(index: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= index <= end for start, end in spans)


def _splits(slot: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < slot <= end for start, end in spans)


def _normalize(s: str) -> str:
    return " ".join(s.split())


def _unique_line_index(
    lines: list[str],
    anchor: str,
    literal: dict[int, list[tuple[int, int]]] | None = None,
) -> int | None:
    """Index of the ONLY line containing *anchor*, or ``None``.

    Containment rather than equality because a model legitimately re-marks the
    same source line (``GDP Growth`` -> ``# GDP Growth``). Uniqueness, not
    length, is the admission test -- a short line that occurs once is a better
    anchor than a long one that occurs twice, and a length cutoff would be a
    magic threshold.
    """
    needle = _normalize(anchor)
    if not needle:
        return None
    hits = []
    for i, line in enumerate(lines):
        if needle not in _normalize(line):
            continue
        # A hit whose bytes sit inside an inline code span or a comment is text
        # the page is SHOWING. Anchoring to it would bind the crop's position to
        # an example rather than to the page's own prose.
        offset = line.find(anchor)
        if offset >= 0 and _in_ranges(offset, (literal or {}).get(i, [])):
            continue
        hits.append(i)
    return hits[0] if len(hits) == 1 else None


def _anchor_slot(
    lines: list[str],
    spans: list[tuple[int, int]],
    anchors: tuple[str, str],
    literal: dict[int, list[tuple[int, int]]] | None = None,
) -> tuple[int | None, str]:
    """Resolve an insertion slot from the region's source anchors.

    Both anchors are looked up, not just the first that matches: when they are
    present and CONTRADICT each other (the line that is below the chart in the
    source sits above the line that is over it), the winner's layout does not
    agree with the page's and neither anchor establishes anything.

    An anchor line inside a table or code block is refused outright. A word-row
    that the model folded into a table cell is not prose, and using it would put
    the crop inside the block or at a boundary the source never described.
    """
    above, below = anchors
    ia = _unique_line_index(lines, above, literal)
    ib = _unique_line_index(lines, below, literal)
    if ia is not None and _inside(ia, spans):
        ia = None
    if ib is not None and _inside(ib, spans):
        ib = None
    if ia is not None and ib is not None and ia >= ib:
        return None, "the source anchors above and below the region contradict the winner's order"
    if ia is not None:
        return ia + 1, "bound to the unique source line above the region"
    if ib is not None:
        return ib, "bound to the unique source line below the region"
    return None, "no unique source anchor"


def _table_slot(
    lines: list[str],
    side: str,
) -> tuple[int | None, str]:
    """Resolve a slot at the boundary of the winner's ONLY table block."""
    from socr.tables.reconcile import find_table_blocks

    if not side:
        return None, "no unambiguous table binding"
    blocks = find_table_blocks("\n".join(lines))
    if len(blocks) != 1:
        return None, "the winner's table blocks do not correspond 1:1 with the source"
    block = blocks[0]
    return (
        block.start if side == "before" else block.end + 1,
        f"bound {side} the page's only table block",
    )


def _insert_block(lines: list[str], idx: int, block: list[str]) -> list[str]:
    """Insert *block* at line *idx*, keeping blank-line separation."""
    payload: list[str] = []
    if idx > 0 and lines[idx - 1].strip():
        payload.append("")
    payload.extend(block)
    if idx < len(lines) and lines[idx].strip():
        payload.append("")
    return lines[:idx] + payload + lines[idx:]


def _interleave(refs: list[str]) -> list[str]:
    """Blank-separated block, so ``_strip_owned`` can invert it exactly."""
    out: list[str] = []
    for ref in refs:
        if out:
            out.append("")
        out.append(ref)
    return out


# ---------------------------------------------------------------------------
# Reconciliation (pure)
# ---------------------------------------------------------------------------


def _refuse_all(
    placements: list[tuple[ChartRegionAsset, int, str, str]],
    unresolved: list[ChartRegionAsset],
    reasons: dict[int, tuple[str, str]],
    detail: str,
) -> tuple[list, list[ChartRegionAsset]]:
    """Withdraw every position CLAIM on the page, keeping every crop."""
    out = list(unresolved)
    for asset, _slot, _disposition, _detail in placements:
        out.append(asset)
        reasons[asset.region_index] = (UNRESOLVED_PLACEMENT, detail)
    out.sort(key=lambda a: (a.bbox[1], a.region_index))
    return [], out


def _build_body(
    base: list[str],
    page_num: int,
    ordered: list[ChartRegionAsset],
    placements: list[tuple[ChartRegionAsset, int, str, str]],
    unresolved: list[ChartRegionAsset],
) -> list[str]:
    """Apply slots bottom-up, then the unresolved block, then failure markers."""
    out = list(base)
    slots: dict[int, list[ChartRegionAsset]] = {}
    for asset, slot, _disposition, _detail in placements:
        slots.setdefault(slot, []).append(asset)
    for slot in sorted(slots, reverse=True):
        out = _insert_block(out, slot, _interleave([image_ref(a) for a in slots[slot]]))
    if unresolved:
        block = [unresolved_placement_note(page_num, [a.region_index for a in unresolved]), ""]
        block.extend(_interleave([image_ref(a) for a in unresolved]))
        out = _insert_block(out, len(out), block)
    for asset in ordered:
        if not asset.rendered:
            out = _insert_block(out, len(out), [render_failure_marker(asset)])
    return out


def _all_refs_live(body: list[str], assets: list[ChartRegionAsset]) -> bool:
    """True when each asset appears exactly once as a LIVE image in *body*.

    Keyed on the path actually EMITTED, not on the canonical filename: the check
    has to ask about the bytes that were written, or it answers a question about
    a reference the body does not contain.
    """
    literal_lines, ranges = markdown_literal_context(body)
    for asset in assets:
        wanted = {asset.rel_path.rsplit("/", 1)[-1] or asset.filename}
        seen = 0
        for idx, line in enumerate(body):
            if idx in literal_lines:
                continue
            for _start, _end, dest in live_image_tokens(line, ranges.get(idx, [])):
                if _owns(dest, wanted):
                    seen += 1
        if seen != 1:
            return False
    return True


def reconcile_chart_region_refs(
    text: str,
    assets: list[ChartRegionAsset],
    anchors: dict[int, tuple[str, str]],
    bindings: dict[int, str],
) -> tuple[str, list[ChartRegionOutcome]]:
    """Guarantee exactly one reference per detected chart region in *text*.

    Pure: no I/O, no state. An empty *assets* returns *text* byte-for-byte and
    no outcomes, which is what keeps chart-free documents byte-identical.

    Method, in three passes:

    1. **Strip every owned artifact.** References to these regions' canonical
       crops, and the markers this module writes, are removed wherever they sit.
       Counting an existing reference as a finished placement cannot enforce
       "exactly once, in source order": two references to one crop stay two, and
       references the winner emitted in the wrong order stay wrong. Ownership is
       decided on the image TARGET, never on alt text a model could invent.
    2. **Resolve a slot per region against the STRIPPED body**, in source order.
       Slots are line indices in that one body, so two regions sharing an anchor
       collect at the same slot instead of each prepending to a body the previous
       one just grew -- which reverses them.
    3. **Apply the slots bottom-up**, one source-ordered group per slot, so
       earlier indices stay valid.

    Nothing the winner authored is rewritten: no insertion may land inside a
    table or code block, and a region whose position cannot be established from
    the source goes to a labelled unresolved block rather than being appended as
    if it were known. Placement matching decides position only and must never be
    mistaken for table verification.

    Idempotent by construction: pass 1 exactly inverts what pass 3 wrote.
    """
    if not assets:
        return text, []

    page_num = assets[0].page_num
    # Both the canonical crop name and whatever path was actually rendered: the
    # two agree in production, and keying on only one of them would leave a
    # reference this pass wrote invisible to the pass that has to remove it.
    filenames = {a.filename for a in assets}
    filenames.update(a.rel_path.rsplit("/", 1)[-1] for a in assets if a.rel_path)
    prefixes = tuple(
        [render_failure_prefix(a.page_num, a.region_index) for a in assets]
        + [unresolved_placement_prefix(page_num)]
    )
    original = text.split("\n")
    lines = _strip_owned(original, filenames, prefixes, markdown_literal_context(original))
    _literal_lines, literal_ranges = markdown_literal_context(lines)
    spans = protected_spans(lines)

    ordered = sorted(assets, key=lambda a: (a.bbox[1], a.region_index))
    rendered = [a for a in ordered if a.rendered]
    reasons: dict[int, tuple[str, str]] = {
        a.region_index: (RENDER_FAILED, a.error) for a in ordered if not a.rendered
    }
    placements: list[tuple[ChartRegionAsset, int, str, str]] = []
    unresolved: list[ChartRegionAsset] = []

    for asset in rendered:
        slot, detail = _anchor_slot(
            lines, spans, anchors.get(asset.region_index, ("", "")), literal_ranges
        )
        disposition = PLACED_ANCHOR
        if slot is None or _splits(slot, spans):
            slot, detail = _table_slot(lines, bindings.get(asset.region_index, ""))
            disposition = PLACED_TABLE_BOUND
        if slot is None or _splits(slot, spans):
            unresolved.append(asset)
            reasons[asset.region_index] = (UNRESOLVED_PLACEMENT, detail)
            continue
        placements.append((asset, slot, disposition, detail))

    # Cross-slot monotonicity. Each region resolves its slot independently, so
    # nothing above notices that the winner's layout runs BACKWARDS against the
    # source: two charts whose surviving anchors appear in the opposite order
    # each bind happily and ship reversed, both labelled placed. The per-region
    # above/below contradiction check cannot see it, because only one anchor
    # survives for each. So compare the resolved slots against source order
    # here, once, and when they disagree refuse EVERY placement on the page:
    # a body whose geometry contradicts the source cannot position any of them,
    # and the alternative -- keeping the ones that happen to fit -- would pick
    # which charts to believe on no evidence. They still ship, source-ordered,
    # in the labelled unresolved block; what is withheld is only the CLAIM.
    if [slot for _a, slot, _d, _t in placements] != sorted(slot for _a, slot, _d, _t in placements):
        placements, unresolved = _refuse_all(
            placements,
            unresolved,
            reasons,
            "the winner's anchor order runs backwards against the source order",
        )

    body = _build_body(lines, page_num, ordered, placements, unresolved)

    # Liveness. A slot can be inside a construct that the span check reads as
    # safe and markdown still swallows -- the reference is present as BYTES and
    # renders as nothing. Reporting that as a placement is the same silent loss
    # in a new costume, so the answer is checked, not assumed: every rendered
    # region must appear exactly once as a LIVE image token in the final body.
    # On any failure every placement is refused, the same way a non-monotone
    # layout is. The fallback body is not re-checked: its block is appended at
    # the very end, which is the last position left to try.
    if placements and not _all_refs_live(body, rendered):
        placements, unresolved = _refuse_all(
            placements,
            unresolved,
            reasons,
            "the reference could not be placed anywhere markdown renders it",
        )
        body = _build_body(lines, page_num, ordered, placements, unresolved)

    for asset, _slot, disposition, detail in placements:
        reasons[asset.region_index] = (disposition, detail)

    outcomes = [
        ChartRegionOutcome(
            asset.page_num,
            asset.region_index,
            reasons[asset.region_index][0],
            "" if not asset.rendered else asset.rel_path,
            reasons[asset.region_index][1],
        )
        for asset in ordered
    ]
    return "\n".join(body), outcomes
