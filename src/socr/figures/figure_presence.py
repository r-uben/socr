"""#1276: a chart a page DRAWS either ships as a picture or the page says it did not.

Two defects, one invariant. On run ``fed_rerun_51270eb5_words`` (156 docs, 12,413 pages):

* **Native lane.** 163 native pages ship SUCCESS with no picture and no mark although
  ``chart_region_bboxes`` (the production box finder) finds a chart box on them, and about 43 of
  those are real charts (all 17 of the auditors' pages). ``_preserve_chart_regions`` only looks
  at pages with a table signal, and ``_is_chart_asset_page`` routes on a different predicate
  (``has_chart_marks``), so a chart page with no table signal reached no lane that ships a
  picture and nothing recorded the outcome.
* **Model lane.** A model that does not transcribe a figure writes ``<!-- Image (x, y, w, h) -->``
  and moves on. No code read that string.

The evidence rule, and why it is regional
-----------------------------------------
A box alone is ~1/8 precise (slide footers, sidebar tabs, rule-only tables, title slides: #1268).
A tick run alone fires on a table with no delimiters and a header row of consecutive years
(``tick_run``'s own docstring says so). So a crop ships only when BOTH hold in the SAME region:
a chart box, and a run of evenly spaced numbers among the words INSIDE that box, with the words
of any detected table removed and the region not itself a table (recurring numeric columns).
A page-level evidence pair ("a box here, a tick run elsewhere") is not enough, and neither is a
nearby image: an image excuses a region only if it is that region's own crop or the page's own
image. A logo, or one surviving panel, never excuses a different region.

No new numeric constant. The tick definition is ``unshipped_figures.MIN_TICK_RUN``; the table
test is ``tables.reconstruct.has_recurring_numeric_columns``; a word belongs to a region by the
same top-left-corner convention ``chart_region_bboxes``' own callers use; lines are the text
layer's own (block, line) lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from socr.figures.unshipped_figures import MIN_TICK_RUN, tick_run

#: The page ships a crop of a chart region it carries tick evidence for. Marked through
#: ``visual_values_not_transcribed`` (the #1174 line), not through this kind.
PRESENCE_SHIPPED_KIND = "figure_presence_shipped"
#: A chart region with evidence whose crop could not be saved. Marked.
PRESENCE_RENDER_FAILED_KIND = "figure_presence_render_failed"
#: The check ran and owes the page nothing. Never marks. The reason says why, and it never
#: claims the page's figures are complete: "no chart evidence under this check" is not "no chart".
PRESENCE_PASS_KIND = "figure_presence_pass"
#: A model reading wrote more ``<!-- Image`` placeholders than images ship. Marked.
PLACEHOLDER_KIND = "unshipped_figure_placeholder"

PRESENCE_KINDS: frozenset[str] = frozenset(
    {PRESENCE_SHIPPED_KIND, PRESENCE_RENDER_FAILED_KIND, PRESENCE_PASS_KIND, PLACEHOLDER_KIND}
)
#: The subset that marks the page.
PRESENCE_MARKED_KINDS: frozenset[str] = frozenset({PRESENCE_RENDER_FAILED_KIND, PLACEHOLDER_KIND})

#: The crop's filename stem: distinct from ``chart_region_p{n}_{i}.png`` (GH-189) and
#: ``unshipped_figure_p{n}_{i}.png`` (#1187) so no lane overwrites another's asset. The index is
#: the detector's own region ordinal, so a name always says WHICH region it is the crop of.
CROP_STEM = "figure_presence"

PASS_NO_EVIDENCE = (
    "a chart box was found but no chart evidence under this check "
    "(no run of evenly spaced axis numbers inside the box, or the box is a table region); "
    "this is not a statement that the page carries no figure"
)
PASS_ALL_SHIPPED = "every chart box on this page already ships its own picture"

#: What the mark of a lost region says. Figure-specific, never the generic page sentence.
RENDER_FAILED_REASON = (
    "a chart region with axis ticks was located on this page and no crop of it could be saved; "
    "it is preserved nowhere; read it from the page image"
)
PLACEHOLDER_REASON = (
    "the model reading marked a figure here (an image placeholder comment) and no picture "
    "of it ships; read it from the page image"
)

_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_PLACEHOLDER_RE = re.compile(r"<!--\s*Image\b", re.IGNORECASE)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)[^)]*\)")


def crop_filename(page_num: int, index: int) -> str:
    return f"{CROP_STEM}_p{page_num}_{index}.png"


def _outside_fences(body: str) -> list[str]:
    """The lines of *body* that are not inside a fenced code block (same fence character
    closes it, as ``tick_run`` does)."""
    out: list[str] = []
    fence = ""
    for line in (body or "").split("\n"):
        marker = _FENCE_RE.match(line)
        if fence:
            if marker and marker.group(1)[0] == fence:
                fence = ""
            continue
        if marker:
            fence = marker.group(1)[0]
            continue
        out.append(line)
    return out


def placeholder_count(body: str) -> int:
    """``<!-- Image`` placeholders in *body*, ignoring fenced code blocks."""
    return sum(len(_PLACEHOLDER_RE.findall(line)) for line in _outside_fences(body))


#: The filename stems of the pictures that can match a model's figure placeholder: the chart
#: lanes' crops and page image, this lane's, #1187's, and the figure extractor's. Everything
#: else socr writes beside a page is NOT a figure: ``failed_table`` (a table crop),
#: ``uncovered_text`` (a text-coverage crop). Whole-page floor images are handled by the
#: caller (``page_ships_its_own_page_image``). Measured on the 156-doc run: the image stems are
#: chart_page, chart_region, failed_table, invisible_scan_page, prior_ocr_layer_page,
#: uncovered_text, unshipped_figure.
FIGURE_IMAGE_STEMS = (
    "chart_region_",
    "chart_page_",
    "unshipped_figure_",
    f"{CROP_STEM}_",
    "figure_",
)


def _is_figure_image(ref: str) -> bool:
    return ref.rsplit("/", 1)[-1].startswith(FIGURE_IMAGE_STEMS)


def shipped_image_count(body: str) -> int:
    """Figure images in *body*, ignoring fenced code blocks. A placeholder is matched
    one-to-one by a shipped image: figure 1 shipping never excuses figure 2, and a table crop
    is not a figure."""
    return sum(
        1
        for line in _outside_fences(body)
        for ref in _IMAGE_RE.findall(line)
        if _is_figure_image(ref)
    )


def own_region_refs(body: str, page_num: int) -> set[int]:
    """Region ordinals whose OWN crop (GH-189 or this lane) the body links."""
    pattern = re.compile(rf"(?:chart_region|{CROP_STEM})_p{int(page_num)}_(\d+)\.png")
    found: set[int] = set()
    for line in _outside_fences(body):
        for ref in _IMAGE_RE.findall(line):
            match = pattern.search(ref)
            if match:
                found.add(int(match.group(1)))
    return found


def words_in_box(words: list, box) -> list:
    """Words whose top-left corner lies in *box* (``tables.reconstruct._word_in_any_bbox``)."""
    return [w for w in words if box.x0 <= w[0] <= box.x1 and box.y0 <= w[1] <= box.y1]


def _in_table(word, table_bboxes) -> bool:
    cx, cy = (word[0] + word[2]) / 2.0, (word[1] + word[3]) / 2.0
    return any(
        float(t[0]) <= cx <= float(t[2]) and float(t[1]) <= cy <= float(t[3]) for t in table_bboxes
    )


def region_lines(words: list) -> list[str]:
    """Text lines of *words*, as the PDF's own text layer has them.

    A word is ``(x0, y0, x1, y1, text, block, line, word)``; words sharing ``(block, line)`` are
    one line of the layer, which is the line the shipped native text carries. Grouping by
    vertical overlap instead merged an axis tick with the legend label beside it
    (``400 ABS 80``) on Fed charts, so a tick column the shipped text has one-per-line vanished
    (mpr 2011-07 p35, FSR 2020-05 p46: measured). Blocks and lines keep the layer's order.
    """
    lines: dict[tuple[int, int], list] = {}
    for w in words:
        lines.setdefault((int(w[5]), int(w[6])), []).append(w)
    return [
        " ".join(str(w[4]) for w in sorted(line, key=lambda w: w[0]))
        for _key, line in sorted(lines.items())
    ]


@dataclass(frozen=True)
class RegionAssessment:
    index: int
    covered: bool
    ticks: int
    table_region: bool

    @property
    def fires(self) -> bool:
        """Evidence in this region: uncovered, tick run, not a table."""
        return not self.covered and not self.table_region and self.ticks >= MIN_TICK_RUN

    def to_dict(self) -> dict:
        return {
            "region": self.index,
            "covered": self.covered,
            "ticks": self.ticks,
            "table_region": self.table_region,
            "fires": self.fires,
        }


def assess_regions(
    words: list,
    boxes: list,
    covered: set[int],
    table_bboxes,
    page_shipped: bool,
) -> list[RegionAssessment]:
    """One assessment per chart box, the evidence tied to that box's own words."""
    from socr.tables.reconstruct import has_recurring_numeric_columns

    tables = [t for t in (table_bboxes or ()) if len(t) == 4]
    out: list[RegionAssessment] = []
    for index, box in enumerate(boxes, start=1):
        inside = [w for w in words_in_box(words, box) if not _in_table(w, tables)]
        is_table = bool(inside) and has_recurring_numeric_columns(inside)
        ticks = 0 if is_table else tick_run("\n".join(region_lines(inside)))
        out.append(
            RegionAssessment(
                index=index,
                covered=page_shipped or index in covered,
                ticks=ticks,
                table_region=is_table,
            )
        )
    return out


@dataclass(frozen=True)
class PresenceDecision:
    kind: str
    detail: str
    regions: list[RegionAssessment] = field(default_factory=list)
    refs: tuple[str, ...] = ()
    failed: tuple[int, ...] = ()

    @property
    def marks(self) -> bool:
        return self.kind in PRESENCE_MARKED_KINDS

    def event_data(self) -> dict:
        return {
            "regions": [r.to_dict() for r in self.regions],
            "refs": list(self.refs),
            "render_failed_regions": list(self.failed),
        }


def decide_presence(assessments: list[RegionAssessment], render) -> PresenceDecision | None:
    """The whole native rule. ``None`` when the page has no chart box.

    *render(indices)* returns ``{index: ref}`` for every crop it managed to save.
    """
    if not assessments:
        return None
    firing = [a for a in assessments if a.fires]
    if not firing:
        everything_shipped = all(a.covered for a in assessments)
        return PresenceDecision(
            PRESENCE_PASS_KIND,
            PASS_ALL_SHIPPED if everything_shipped else PASS_NO_EVIDENCE,
            assessments,
        )
    saved = render([a.index for a in firing])
    refs = tuple(saved[a.index] for a in firing if saved.get(a.index))
    failed = tuple(a.index for a in firing if not saved.get(a.index))
    if failed:
        return PresenceDecision(
            PRESENCE_RENDER_FAILED_KIND,
            f"chart region(s) {', '.join(map(str, failed))} carry axis ticks and no crop could be "
            "saved: the chart is preserved nowhere",
            assessments,
            refs,
            failed,
        )
    return PresenceDecision(
        PRESENCE_SHIPPED_KIND,
        f"{len(refs)} chart region(s) with axis ticks shipped as image crops; their values are "
        "not transcribed",
        assessments,
        refs,
    )


def clause_records(page_num: int, labels: tuple[str, ...], kind: str) -> tuple[dict, ...]:
    """One figure-specific, machine-readable mark clause per lost region / unmatched
    placeholder. UNKNOWN: socr did not establish the figure is absent; it failed to show it."""
    from socr.core.marks import Check, CheckResult, region_scope

    reason = PLACEHOLDER_REASON if kind == PLACEHOLDER_KIND else RENDER_FAILED_REASON
    return tuple(
        {
            "check": Check.FIGURE_TEXT.value,
            "result": CheckResult.UNKNOWN.value,
            "scope": region_scope("figure", index),
            "reason": f"{label}: {reason}",
        }
        for index, label in enumerate(labels, start=1)
    )
