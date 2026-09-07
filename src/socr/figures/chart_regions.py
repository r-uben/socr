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

_IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\(\s*([^)\s]+)")
_FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")
_IMAGE_REF_FULL_RE = re.compile(r"!\[[^\]]*\]\(\s*([^)\s]*)[^)]*\)")


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


def _image_targets(line: str) -> list[str]:
    """Every image TARGET on a line, in order. Alt text is never consulted."""
    return [m.group(1) for m in _IMAGE_REF_RE.finditer(line)]


def code_fence_spans(lines: list[str]) -> list[tuple[int, int]]:
    """Inclusive line spans of fenced code blocks, fence lines included.

    Computed on the ORIGINAL body, before anything is removed. Literal image
    syntax inside a fence is CONTENT -- a markdown example the page is showing
    the reader -- not a live reference to one of our crops. Deleting it would
    edit the accepted text and move the example's image outside its own block,
    so block protection has to cover removal, not only insertion.
    """
    spans: list[tuple[int, int]] = []
    open_at: int | None = None
    for i, line in enumerate(lines):
        if not _FENCE_RE.match(line):
            continue
        if open_at is None:
            open_at = i
        else:
            spans.append((open_at, i))
            open_at = None
    if open_at is not None:
        spans.append((open_at, len(lines) - 1))
    return spans


def _owns(target: str, filenames: set[str]) -> bool:
    return target.rsplit("/", 1)[-1] in filenames


def _strip_owned_tokens(line: str, filenames: set[str]) -> str:
    """Remove owned image TOKENS from *line*, leaving everything else intact.

    Token-level, never line-level: an owned reference embedded in a sentence
    must not take the sentence with it, and a line mixing an owned crop with an
    unrelated figure is not an all-or-nothing unit. Only the whitespace the
    removal itself collapsed is normalised, and only on a line actually edited.
    """
    out = _IMAGE_REF_FULL_RE.sub(lambda m: "" if _owns(m.group(1), filenames) else m.group(0), line)
    if out == line:
        return line
    return re.sub(r"[ \t]{2,}", " ", out).rstrip()


def _strip_owned(
    lines: list[str],
    filenames: set[str],
    prefixes: tuple[str, ...],
    code_spans: list[tuple[int, int]],
) -> list[str]:
    """Remove every owned artifact and the ONE blank separator it brought.

    Three cases, in order: a line inside a code fence is content and is never
    touched; a line that is one of this module's own markers is dropped whole;
    any other line has its owned image tokens removed, and is dropped only if
    that emptied it.

    Dropping a line exactly inverts ``_insert_block``, so reconciling this
    module's own output reproduces it byte-for-byte instead of accumulating
    blank lines.
    """
    protected = [any(a <= i <= b for a, b in code_spans) for i in range(len(lines))]
    rewritten: list[str | None] = []
    for i, line in enumerate(lines):
        if protected[i]:
            rewritten.append(line)
            continue
        stripped = line.strip()
        if stripped and any(stripped.startswith(prefix) for prefix in prefixes):
            rewritten.append(None)
            continue
        new = _strip_owned_tokens(line, filenames)
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


def _unique_line_index(lines: list[str], anchor: str) -> int | None:
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
    hits = [i for i, ln in enumerate(lines) if needle in _normalize(ln)]
    return hits[0] if len(hits) == 1 else None


def _anchor_slot(
    lines: list[str],
    spans: list[tuple[int, int]],
    anchors: tuple[str, str],
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
    ia = _unique_line_index(lines, above)
    ib = _unique_line_index(lines, below)
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
    filenames = {a.filename for a in assets}
    prefixes = tuple(
        [render_failure_prefix(a.page_num, a.region_index) for a in assets]
        + [unresolved_placement_prefix(page_num)]
    )
    original = text.split("\n")
    lines = _strip_owned(original, filenames, prefixes, code_fence_spans(original))
    spans = protected_spans(lines)

    ordered = sorted(assets, key=lambda a: (a.bbox[1], a.region_index))
    outcomes: list[ChartRegionOutcome] = []
    placements: list[tuple[ChartRegionAsset, int, str, str]] = []
    unresolved: list[ChartRegionAsset] = []

    for asset in ordered:
        if not asset.rendered:
            outcomes.append(
                ChartRegionOutcome(
                    asset.page_num, asset.region_index, RENDER_FAILED, "", asset.error
                )
            )
            continue

        slot, detail = _anchor_slot(lines, spans, anchors.get(asset.region_index, ("", "")))
        disposition = PLACED_ANCHOR
        if slot is None or _splits(slot, spans):
            slot, detail = _table_slot(lines, bindings.get(asset.region_index, ""))
            disposition = PLACED_TABLE_BOUND
        if slot is None or _splits(slot, spans):
            unresolved.append(asset)
            outcomes.append(
                ChartRegionOutcome(
                    asset.page_num,
                    asset.region_index,
                    UNRESOLVED_PLACEMENT,
                    asset.rel_path,
                    detail,
                )
            )
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
    slot_order = [slot for _asset, slot, _d, _t in placements]
    if slot_order != sorted(slot_order):
        for asset, _slot, _d, _t in placements:
            unresolved.append(asset)
            outcomes.append(
                ChartRegionOutcome(
                    asset.page_num,
                    asset.region_index,
                    UNRESOLVED_PLACEMENT,
                    asset.rel_path,
                    "the winner's anchor order runs backwards against the source order",
                )
            )
        unresolved.sort(key=lambda a: (a.bbox[1], a.region_index))
        placements = []
    else:
        slots: dict[int, list[ChartRegionAsset]] = {}
        for asset, slot, disposition, detail in placements:
            slots.setdefault(slot, []).append(asset)
            outcomes.append(
                ChartRegionOutcome(
                    asset.page_num, asset.region_index, disposition, asset.rel_path, detail
                )
            )
        for slot in sorted(slots, reverse=True):
            lines = _insert_block(lines, slot, _interleave([image_ref(a) for a in slots[slot]]))

    if unresolved:
        block = [unresolved_placement_note(page_num, [a.region_index for a in unresolved]), ""]
        block.extend(_interleave([image_ref(a) for a in unresolved]))
        lines = _insert_block(lines, len(lines), block)

    for asset in ordered:
        if not asset.rendered:
            lines = _insert_block(lines, len(lines), [render_failure_marker(asset)])

    return "\n".join(lines), outcomes
