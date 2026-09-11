"""#635 Stage 0: withhold the EMPTY table skeleton a chart page derives, keep the chart.

A page whose charts are held out of the page-level chart lane (GH-150 B1) keeps
its normal route, and a model reading it commonly emits, per panel, a markdown
grid whose header row is the chart's own axis bins and whose single body row
carries no observation at all:

    | Percent range | 1.88-2.12 | 2.13-2.37 | ... |
    | :--- | :---: | :---: | ... |
    | **Participants** |  |  | ... |

Nothing was read. The grid is a derivation of the chart that contains no data,
and publishing it presents the reader with a table shaped exactly like an
extraction that succeeded. The crop of the same chart IS preserved (GH-189), so
the evidence is not at risk -- what is at risk is the reader believing the empty
grid is the chart's content.

This module answers that, and only that. It does NOT read a chart (Stage 1) and
it does not touch a grid carrying any value, a literal zero included.

Two independent source proofs are required before a grid is withheld, and both
come from the PDF's own geometry, never from words in the model's text:

1. **A unique in-region label.** Each chart region's INTERIOR word-rows -- the
   rows ``chart_region_anchors`` deliberately excludes from its anchors, because
   they are inside the chart -- are collected, and only those unique across the
   page's regions survive. A surviving label that matches exactly one line of
   the candidate (as a heading/list item in its own right, not by containment)
   ties that region to one position in the candidate.
2. **Axis attestation.** Every DATA column header of the grid must share a token
   with the region's interior text. The column keys of a grid derived from a
   chart are that chart's axis labels; a grid whose keys the chart never drew is
   not that chart's derivation.

Anything short of both, for every skeleton on the page, is a refusal: the text
is left byte-identical and the refusal is recorded. A quarantined or unbound
empty grid is never deleted -- an empty form that is not a chart derivation is
the page's content, and this module has no opinion about it.

Placement. The mutation belongs at the CANDIDATE seam (#688's
``canonicalize_candidate`` neighbourhood), once, before judging and identity
creation -- not inside ``reconcile_chart_region_refs``, whose contract is to
preserve model-authored text and place crop references against it. #189 then
reconciles the crops normally over the already-mutated body, so fragments, the
stitched document and the sidecars all serialize the same bytes.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Audit event kind recorded for each withheld skeleton.
SKELETON_SUPPRESSED = "chart_table_skeleton_suppressed"
#: Audit event kind recorded when a skeleton COULD be a chart derivation but no
#: unambiguous source proof binds it to one. The text is untouched.
SKELETON_UNBOUND = "chart_table_skeleton_unbound"

_SEP_CELL_RE = re.compile(r"^:?-+:?$")
_TOKEN_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z.]*")
#: Leading markdown a line may carry while still BEING that label: an ATX
#: heading marker, a bullet, or an ordered-list marker.
_LEAD_RE = re.compile(r"^\s*(?:#{1,6}\s*|[-*+]\s+|\d+[.)]\s+)?")
_EMPH_RE = re.compile(r"^[*_`~\s]+|[*_`~\s:.]+$")
_FOLD = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "−": "-",
}


@dataclass(frozen=True)
class Skeleton:
    """One markdown grid on the page whose every DATA cell is empty.

    ``start``/``end`` are inclusive line indices into the caller's own line
    split. ``table_index`` is the grid's 1-based reading-order ordinal among
    the page's markdown tables, and is the id carried in provenance.
    """

    table_index: int
    start: int
    end: int
    header: list[str]
    label_header: str
    data_headers: list[str]
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SkeletonSuppression:
    """A grid proven to be a chart region's empty derivation, and withheld."""

    page_num: int
    table_index: int
    region_index: int
    crop_filename: str
    label: str
    original_text: str
    sha256: str

    def to_dict(self) -> dict:
        return {
            "page_num": self.page_num,
            "table_index": self.table_index,
            "region_index": self.region_index,
            "crop": self.crop_filename,
            "label": self.label,
            "sha256": self.sha256,
            "original_text": self.original_text,
        }


@dataclass(frozen=True)
class SkeletonRefusal:
    """A grid that could be a chart derivation, left byte-identical instead."""

    page_num: int
    table_index: int
    reason: str
    sha256: str

    def to_dict(self) -> dict:
        return {
            "page_num": self.page_num,
            "table_index": self.table_index,
            "reason": self.reason,
            "sha256": self.sha256,
        }


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def _fold(text: str) -> str:
    """Compare-ready text: NFKC, typographic punctuation folded, case removed.

    A PDF draws a curly apostrophe and an en dash where a model writes the ASCII
    forms; comparing the two without folding makes a real match look like a
    miss, and a missed match here means a refusal, not a wrong binding.
    """
    out = unicodedata.normalize("NFKC", text)
    out = "".join(_FOLD.get(ch, ch) for ch in out)
    return " ".join(out.split()).casefold()


def _tokens(text: str) -> set[str]:
    return {t.casefold().rstrip(".") for t in _TOKEN_RE.findall(_fold(text))}


def _bare_label(line: str) -> str:
    """The line's own label text, with one layer of markdown marking removed."""
    return _EMPH_RE.sub("", _LEAD_RE.sub("", line, count=1))


def _label_of_line(line: str) -> str:
    """Compare-ready form of the line's own label text.

    Containment is deliberately NOT used. ``2018`` is contained in a figure
    caption that reads "2018-21 and over the longer run", and binding a panel to
    a caption because the caption mentions its year is exactly the kind of guess
    this module exists to refuse.
    """
    return _fold(_bare_label(line))


# ---------------------------------------------------------------------------
# Structural skeleton detection
# ---------------------------------------------------------------------------


def _split_row(row: str) -> list[str]:
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(_SEP_CELL_RE.match(c) for c in cells)


def _table_runs(lines: list[str]) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` for each genuine markdown table on the page.

    Delegated to ``table_syntax_line_indices``, the strict separator-anchored
    grammar #649 settled, so a pipe-carrying sentence beside a table is prose
    here exactly as it is everywhere else. Contiguous runs of its indices are
    the tables.
    """
    from socr.tables.reconcile import table_syntax_line_indices

    indices = sorted(table_syntax_line_indices(lines))
    runs: list[tuple[int, int]] = []
    for idx in indices:
        if runs and idx == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], idx)
        else:
            runs.append((idx, idx))
    return runs


def find_empty_skeletons(text: str) -> list[Skeleton]:
    """Every grid in *text* whose DATA cells are, structurally, all empty.

    The check is structural and lexical rules are refused outright: no cell is
    matched against a word, and the header row is never an observation -- it is
    the column key, and a key carrying numeric bin limits is a label for data,
    not data. Column 0 of a body row is the ROW LABEL, which is the convention
    ``binding.parse_grid`` and ``label_canonical`` already hold the page to; the
    data positions are the remaining columns.

    A cell is empty when it carries no non-whitespace character. That single
    rule is what makes a literal ``0`` data, a text-valued cell data, and an
    unresolved-value token (``--``, ``n/a``, a socr marker) data: each of them
    is something the reading says about the cell, and only a cell that says
    nothing at all is empty.
    """
    if not text or "|" not in text:
        return []
    lines = text.split("\n")
    out: list[Skeleton] = []
    for ordinal, (start, end) in enumerate(_table_runs(lines), start=1):
        rows = [_split_row(lines[i]) for i in range(start, end + 1)]
        separators = [i for i, cells in enumerate(rows) if _is_separator(cells)]
        # One header band, one separator, one body. A run with two separators is
        # two stacked grids or a malformed one; either way this pass does not
        # know which header keys which body, so it does not act.
        if len(separators) != 1:
            continue
        sep = separators[0]
        header = rows[sep - 1] if sep >= 1 else []
        body = rows[sep + 1 :]
        if not header or len(header) < 2 or not body:
            continue
        data_cells = [cell for row in body for cell in row[1:]]
        if not data_cells or any(cell for cell in data_cells):
            continue
        out.append(
            Skeleton(
                table_index=ordinal,
                start=start,
                end=end,
                header=header,
                label_header=header[0],
                data_headers=header[1:],
                text="\n".join(lines[start : end + 1]),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Source proofs
# ---------------------------------------------------------------------------


def region_interior_rows(page, bboxes) -> dict[int, list[str]]:
    """``{region_index: [word-row text, ...]}`` for text INSIDE each region.

    The complement of ``chart_region_anchors``, which reads the rows outside the
    regions. A chart's panel heading, legend and axis labels are drawn inside its
    own bbox, and they are the only source evidence that distinguishes one panel
    of a repeated figure from the next -- every anchor around them is identical.

    Never raises; returns ``{}`` when word geometry is absent.
    """
    try:
        words = page.get_text("words") or []
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("chart_data: get_text('words') failed: %s", exc)
        return {}

    out: dict[int, list[str]] = {}
    for idx, box in enumerate(bboxes, start=1):
        rows: dict[int, list[tuple[float, str]]] = {}
        for w in words:
            if box.x0 <= w[0] <= box.x1 and box.y0 <= w[1] <= box.y1:
                rows.setdefault(round(w[1]), []).append((w[0], w[4]))
        texts = []
        for key in sorted(rows):
            row = sorted(rows[key], key=lambda t: t[0])
            joined = " ".join(t[1] for t in row).strip()
            if joined:
                texts.append(joined)
        out[idx] = texts
    return out


def unique_region_labels(interiors: dict[int, list[str]]) -> dict[int, list[tuple[str, str]]]:
    """Interior rows that occur inside exactly ONE of the page's regions.

    A row every panel draws ("Number of participants", an axis tick "16") names
    no panel in particular; only a row unique to one region can bind it. Each
    entry is ``(compare-ready, as drawn)`` and the list keeps the region's own
    top-to-bottom order, which is what lets the caller tell a panel's heading
    from its legend without a lexical rule about either.
    """
    seen: dict[str, set[int]] = {}
    for idx, rows in interiors.items():
        for row in rows:
            folded = _fold(row)
            if folded:
                seen.setdefault(folded, set()).add(idx)
    out: dict[int, list[tuple[str, str]]] = {idx: [] for idx in interiors}
    for idx, rows in interiors.items():
        for row in rows:
            folded = _fold(row)
            if folded and len(seen.get(folded, ())) == 1:
                out[idx].append((folded, " ".join(row.split())))
    return out


def _resolve_anchor(
    lines: list[str],
    table_lines: set[int],
    labels: list[tuple[str, str]],
) -> tuple[int, str] | None:
    """``(anchor line, panel name)`` from this region's unique labels, or None.

    A label matching several candidate lines proves nothing and is dropped. The
    two answers come from different matches on purpose:

    * the anchor is the LAST surviving match, because a panel's heading, its
      axis title and its legend all precede the grid derived from it and the
      nearest one bounds the grid most tightly;
    * the panel name is the TOPMOST label inside the region, which is where a
      panel heading is drawn. Taking the anchor's own text instead would name
      the panel after its legend ("June projections").
    """
    hits: list[tuple[int, str]] = []
    for folded, drawn in labels:
        matched = [
            i
            for i, line in enumerate(lines)
            if i not in table_lines and _label_of_line(line) == folded
        ]
        if len(matched) == 1:
            hits.append((matched[0], drawn))
    if not hits:
        return None
    return max(i for i, _d in hits), hits[0][1]


def _axis_attested(skeleton: Skeleton, interior: list[str]) -> bool:
    """Every DATA column key of the grid shares a token with the chart's interior.

    The column keys of a grid derived from a chart are that chart's own axis
    labels. A key the chart never drew means the grid is a derivation of
    something else, whatever else lines up.
    """
    available = set()
    for row in interior:
        available |= _tokens(row)
    if not available:
        return False
    for cell in skeleton.data_headers:
        cell_tokens = _tokens(cell)
        if not cell_tokens or not (cell_tokens & available):
            return False
    return True


# ---------------------------------------------------------------------------
# Owned artifact
# ---------------------------------------------------------------------------


def suppression_prefix(page_num: int, region_index: int) -> str:
    return f"> **Chart region {region_index} on page {page_num} — counts not extracted**"


def suppression_note(
    page_num: int,
    region_index: int,
    crop_filename: str,
    label: str,
    skeleton: Skeleton,
) -> str:
    """The line that ships where the empty grid was.

    Every specific in it is read off the source: the panel's own unique label,
    the grid's label-column key and its column count. There is no chart-class
    classifier in the tree, so the description is specific to THIS chart rather
    than to a chart class -- it names what the chart drew, which is the part a
    reader needs to know the crop below is the panel this note is about.
    """
    axis = skeleton.label_header.strip("* _`") or "category"
    bins = ""
    if skeleton.data_headers:
        first, last = skeleton.data_headers[0], skeleton.data_headers[-1]
        bins = f" ({first} … {last})" if first != last else f" ({first})"
    return (
        f"{suppression_prefix(page_num, region_index)} — the reading of this chart"
        f"{f' ({label})' if label else ''} was an empty grid over its {len(skeleton.data_headers)} "
        f"“{axis}” column(s){bins} with no observation in any cell. That grid is withheld "
        f"rather than published as chart data. The chart image `{crop_filename}` is preserved "
        "in this document; its values have NOT been read."
    )


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


def suppress_chart_table_skeletons(
    text: str,
    *,
    page_num: int,
    interiors: dict[int, list[str]],
    crop_names: dict[int, str],
) -> tuple[str, list[SkeletonSuppression], list[SkeletonRefusal]]:
    """Withhold each empty grid PROVEN to derive from one of the page's charts.

    Pure: no I/O, no state. Returns *text* byte-for-byte when nothing is proven,
    which is what keeps every page without an empty chart grid unchanged.

    Idempotent: a withheld grid is gone, so a second pass finds no skeleton and
    returns the same bytes again. That is what makes running this at a seam the
    pipeline crosses more than once, and on a resumed page, free.

    Refusals leave the text alone. Deleting an empty form the page happens to
    contain, because the page also has a chart, would be content loss dressed up
    as a fix.
    """
    if not text:
        return text, [], []
    skeletons = find_empty_skeletons(text)
    if not skeletons:
        return text, [], []
    if not interiors:
        return (
            text,
            [],
            [
                SkeletonRefusal(page_num, s.table_index, "no chart region on the page", s.sha256)
                for s in skeletons
            ],
        )

    from socr.tables.reconcile import table_syntax_line_indices

    lines = text.split("\n")
    table_lines = table_syntax_line_indices(lines)
    labels = unique_region_labels(interiors)
    # region -> (anchor line, the label that matched)
    anchors: dict[int, tuple[int, str]] = {}
    for region in sorted(interiors):
        resolved = _resolve_anchor(lines, table_lines, labels.get(region, []))
        if resolved is not None:
            anchors[region] = resolved

    by_index = {s.table_index: s for s in skeletons}
    chosen: dict[int, int] = {}  # region -> table_index
    refusals: list[SkeletonRefusal] = []

    for region, (anchor, _label) in anchors.items():
        following = [s for s in skeletons if s.start > anchor]
        if not following:
            continue
        target = following[0]
        # An intervening anchor means another panel sits between this one and the
        # grid, so the grid describes that panel, not this one.
        if any(
            other != region and anchor < oa < target.start for other, (oa, _l) in anchors.items()
        ):
            refusals.append(
                SkeletonRefusal(
                    page_num,
                    target.table_index,
                    f"another chart region's label separates region {region} from the grid",
                    target.sha256,
                )
            )
            continue
        if not _axis_attested(target, interiors.get(region, [])):
            refusals.append(
                SkeletonRefusal(
                    page_num,
                    target.table_index,
                    f"the grid's column keys are not attested by chart region {region}'s axis",
                    target.sha256,
                )
            )
            continue
        chosen[region] = target.table_index

    # Source order. Regions are keyed in detector order, which is source order
    # for the bboxes ``chart_region_bboxes`` returns; the grids they bind must
    # run the same way. When they do not, the candidate's layout contradicts the
    # page and no binding on it is evidence of anything -- refuse the page's
    # whole set rather than keep the ones that happen to fit.
    ordered = [chosen[r] for r in sorted(chosen)]
    if ordered != sorted(ordered):
        for region in sorted(chosen):
            refusals.append(
                SkeletonRefusal(
                    page_num,
                    chosen[region],
                    "the candidate's panel order runs backwards against the source order",
                    by_index[chosen[region]].sha256,
                )
            )
        chosen = {}

    bound_tables = set(chosen.values())
    for skeleton in skeletons:
        if skeleton.table_index not in bound_tables and not any(
            r.table_index == skeleton.table_index for r in refusals
        ):
            refusals.append(
                SkeletonRefusal(
                    page_num,
                    skeleton.table_index,
                    "no chart region binds this empty grid; left as the page's own content",
                    skeleton.sha256,
                )
            )

    if not chosen:
        return text, [], refusals

    suppressions: list[SkeletonSuppression] = []
    # Bottom-up, so earlier line indices stay valid.
    for region in sorted(chosen, key=lambda r: by_index[chosen[r]].start, reverse=True):
        skeleton = by_index[chosen[region]]
        crop = crop_names.get(region, "")
        label = anchors[region][1]
        note = suppression_note(page_num, region, crop, label, skeleton)
        lines[skeleton.start : skeleton.end + 1] = [note]
        suppressions.append(
            SkeletonSuppression(
                page_num=page_num,
                table_index=skeleton.table_index,
                region_index=region,
                crop_filename=crop,
                label=label,
                original_text=skeleton.text,
                sha256=skeleton.sha256,
            )
        )
    suppressions.reverse()
    return "\n".join(lines), suppressions, refusals
