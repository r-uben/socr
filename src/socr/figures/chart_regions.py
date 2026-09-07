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
ALREADY_PRESENT = "already_present"
UNRESOLVED_PLACEMENT = "unresolved_placement"
RENDER_FAILED = "render_failed"

#: Dispositions that mean "this region's crop is referenced from the page body
#: at a position derived from the source geometry".
_PLACED = frozenset({PLACED_ANCHOR, PLACED_TABLE_BOUND, ALREADY_PRESENT})

_IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")


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
    tx0, ty0, tx1, ty1 = table_bboxes[0]
    del tx0, tx1
    out: dict[int, str] = {}
    for idx, box in enumerate(bboxes, start=1):
        if box.y1 <= ty0:
            out[idx] = "before"
        elif box.y0 >= ty1:
            out[idx] = "after"
    return out


# ---------------------------------------------------------------------------
# Reconciliation (pure)
# ---------------------------------------------------------------------------


def _normalize(s: str) -> str:
    return " ".join(s.split())


def _unique_line_index(lines: list[str], anchor: str) -> int | None:
    """Index of the ONLY line containing *anchor*, or ``None``.

    Containment rather than equality because a model legitimately re-marks the
    same source line (``GDP Growth`` -> ``# GDP Growth``).  Uniqueness, not
    length, is the admission test -- a short line that occurs once is a better
    anchor than a long one that occurs twice, and a length cutoff would be a
    magic threshold.
    """
    needle = _normalize(anchor)
    if not needle:
        return None
    hits = [i for i, ln in enumerate(lines) if needle in _normalize(ln)]
    return hits[0] if len(hits) == 1 else None


def _insert_block(lines: list[str], idx: int, block: list[str]) -> list[str]:
    """Insert *block* at line *idx*, keeping blank-line separation."""
    payload: list[str] = []
    if idx > 0 and lines[idx - 1].strip():
        payload.append("")
    payload.extend(block)
    if idx < len(lines) and lines[idx].strip():
        payload.append("")
    return lines[:idx] + payload + lines[idx:]


def _image_targets(text: str) -> set[str]:
    return {m.group(1) for m in _IMAGE_REF_RE.finditer(text)}


def image_ref(asset: ChartRegionAsset) -> str:
    return f"![chart region {asset.region_index}]({asset.rel_path})"


def render_failure_marker(asset: ChartRegionAsset) -> str:
    """Visible, greppable marker for a region whose crop could not be rendered.

    Deliberately NOT a markdown image link: a link to a file that was never
    written renders as a broken image and is stripped downstream by
    ``strip_phantom_images``, which is precisely how this loss stayed silent.
    """
    reason = _normalize(asset.error) or "unknown error"
    return (
        f"> **Chart region {asset.region_index} on page {asset.page_num} was NOT preserved** "
        f"— crop `{asset.filename}` failed to render ({reason}). "
        "The chart is present in the source PDF; no image was produced for it."
    )


def unresolved_placement_note(page_num: int, indices: list[int]) -> str:
    listed = ", ".join(str(i) for i in indices)
    return (
        f"> **Unresolved chart placement on page {page_num}** — the source position of "
        f"chart region(s) {listed} could not be located in the accepted text. The crop(s) "
        "below are preserved in source order; their position relative to the surrounding "
        "text and tables is NOT established."
    )


def reconcile_chart_region_refs(
    text: str,
    assets: list[ChartRegionAsset],
    anchors: dict[int, tuple[str, str]],
    bindings: dict[int, str],
) -> tuple[str, list[ChartRegionOutcome]]:
    """Guarantee one reference per detected chart region in *text*.

    Pure: no I/O, no state. An empty *assets* returns *text* byte-for-byte and
    no outcomes, which is what keeps chart-free documents byte-identical.

    References are counted by TARGET, never by alt text -- a model that invented
    the words "chart region 1" has not preserved anything. Nothing already in
    the text is stripped or rewritten; the accepted table cells and prose are
    left exactly as the winner authored them, because placement matching here
    decides position only and must never be mistaken for table verification.

    Idempotent: running it on its own output changes nothing.
    """
    if not assets:
        return text, []

    lines = text.split("\n")
    outcomes: list[ChartRegionOutcome] = []
    unresolved: list[ChartRegionAsset] = []
    failed: list[ChartRegionAsset] = []

    for asset in sorted(assets, key=lambda a: (a.bbox[1], a.region_index)):
        current = "\n".join(lines)
        if not asset.rendered:
            if asset.filename in current:
                outcomes.append(
                    ChartRegionOutcome(
                        asset.page_num,
                        asset.region_index,
                        RENDER_FAILED,
                        "",
                        "marker already present",
                    )
                )
            else:
                failed.append(asset)
                outcomes.append(
                    ChartRegionOutcome(
                        asset.page_num, asset.region_index, RENDER_FAILED, "", asset.error
                    )
                )
            continue

        if asset.rel_path in _image_targets(current):
            outcomes.append(
                ChartRegionOutcome(
                    asset.page_num, asset.region_index, ALREADY_PRESENT, asset.rel_path
                )
            )
            continue

        above, below = anchors.get(asset.region_index, ("", ""))
        idx = _unique_line_index(lines, above)
        if idx is not None:
            lines = _insert_block(lines, idx + 1, [image_ref(asset)])
            outcomes.append(
                ChartRegionOutcome(
                    asset.page_num,
                    asset.region_index,
                    PLACED_ANCHOR,
                    asset.rel_path,
                    "bound to the unique source line above the region",
                )
            )
            continue

        idx = _unique_line_index(lines, below)
        if idx is not None:
            lines = _insert_block(lines, idx, [image_ref(asset)])
            outcomes.append(
                ChartRegionOutcome(
                    asset.page_num,
                    asset.region_index,
                    PLACED_ANCHOR,
                    asset.rel_path,
                    "bound to the unique source line below the region",
                )
            )
            continue

        side = bindings.get(asset.region_index, "")
        if side:
            from socr.tables.reconcile import find_table_blocks

            blocks = find_table_blocks("\n".join(lines))
            if len(blocks) == 1:
                at = blocks[0].start if side == "before" else blocks[0].end + 1
                lines = _insert_block(lines, at, [image_ref(asset)])
                outcomes.append(
                    ChartRegionOutcome(
                        asset.page_num,
                        asset.region_index,
                        PLACED_TABLE_BOUND,
                        asset.rel_path,
                        f"bound {side} the page's only table block",
                    )
                )
                continue

        unresolved.append(asset)
        outcomes.append(
            ChartRegionOutcome(
                asset.page_num,
                asset.region_index,
                UNRESOLVED_PLACEMENT,
                asset.rel_path,
                "no unique source anchor and no unambiguous table binding",
            )
        )

    if unresolved:
        page_num = unresolved[0].page_num
        block = [unresolved_placement_note(page_num, [a.region_index for a in unresolved]), ""]
        block.extend(image_ref(a) for a in unresolved)
        lines = _insert_block(lines, len(lines), block)

    for asset in failed:
        lines = _insert_block(lines, len(lines), [render_failure_marker(asset)])

    return "\n".join(lines), outcomes
