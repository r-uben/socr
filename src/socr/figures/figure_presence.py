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
a chart box, and a run of evenly spaced numbers among the words INSIDE that box that lie on a
common axis line (one x-column, or one baseline that is not a table's header: ``tick_axis``),
with the words of any detected table removed and the region not itself a table (recurring
numeric columns).
A page-level evidence pair ("a box here, a tick run elsewhere") is not enough, and neither is a
nearby image: an image excuses a region only if it is that region's own crop or the page's own
image. A logo, or one surviving panel, never excuses a different region.

What this guarantees, and what it does not
-------------------------------------------
A chart this check can SEE ships as a picture or the page says it did not, for a page whose
shipped body is the reading the lane judged. The limit: the lane is chosen from the best
reading the page loop holds (``best_output``), before ``finalized_page_record`` selects what
ships. When the selector replaces a refused model reading by the native text (``NATIVE_FALLBACK``)
or a floor, the native chart rule never ran on the text that ships, and a chart on such a page
can still ship unmarked. The model lane's placeholder rule is not exposed to this: it is decided
again at assembly from the final record's engine and the final text
(``UnifiedPipeline._reconcile_model_placeholders``). Charts with no numeric ticks, ticks drawn
as shapes, and pages with no chart box are outside the check. So "never disappears silently"
holds for the pages and charts above, not for every chart.

No new numeric constant. The tick definition is ``unshipped_figures.MIN_TICK_RUN``; the table
test is ``tables.reconstruct.has_recurring_numeric_columns``; a word belongs to a region by the
same top-left-corner convention ``chart_region_bboxes``' own callers use; lines are the text
layer's own (block, line) lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from socr.figures.unshipped_figures import _TICK_NUMBER_RE, MIN_TICK_RUN, tick_run

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
#: A page with no chart box is recorded too: every page this lane looked at carries a record, so
#: a count of records against pages checked has no gap (Fable 5).
PASS_NO_BOX = (
    "no chart box was found on this page; this is not a statement that the page carries no figure"
)
#: A page with a table signal is GH-189's: ``_preserve_chart_regions`` crops every box of exactly
#: that candidate set at assembly, so it is not decided here twice.
PASS_LEFT_TO_CHART_REGION_PASS = (
    "the page has a table signal; its chart boxes are left to the chart-region pass "
    "(_preserve_chart_regions), which crops them at assembly where it can (it skips a floor "
    "page, and a crop can fail: each outcome is its own record)"
)

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


def _fence_scanned_lines(body: str) -> list[str]:
    """Fallback when the markdown tokenizer is unavailable: lines outside fenced code (the
    same fence character closes it)."""
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


_INLINE_CODE_RE = re.compile(r"(`+)(?:(?!\1).)+?\1")


def _outside_fences(body: str, *, drop_tables: bool = False) -> list[str]:
    """The lines of *body* that are not code, with inline code spans blanked out. With
    *drop_tables* the lines of a markdown TABLE are dropped too (CommonMark plus the GFM table
    rule, the parser's own structure: a prose line holding two ``|`` is not a table).

    Block context is CommonMark's (``markdown-it-py``, the tokenizer ``figures.chart_regions``
    already uses): a four-backtick block is not closed by an inner three-backtick line, and
    indented code is code. An HTML block is NOT excluded: a model's ``<!-- Image -->``
    placeholder is one. Inline code (a prose line quoting ``<!-- Image (x) -->``) is blanked, so
    a placeholder shown rather than used is not counted (carbon2 5).
    """
    lines = (body or "").split("\n")
    try:
        from markdown_it import MarkdownIt

        tokens = MarkdownIt("commonmark").enable("table").parse("\n".join(lines))
    except Exception:  # noqa: BLE001 -- the scanner below is the fallback, never a silent pass
        kept = _fence_scanned_lines(body)
        if drop_tables:
            kept = [line for line in kept if not line.lstrip().startswith("|")]
    else:
        literal: set[int] = set()
        for token in tokens:
            if (
                token.type in ("fence", "code_block")
                or (drop_tables and token.type == "table_open")
            ) and token.map:
                literal.update(range(token.map[0], min(token.map[1], len(lines))))
        kept = [line for i, line in enumerate(lines) if i not in literal]
    return [_INLINE_CODE_RE.sub("", line) for line in kept]


def placeholder_count(body: str) -> int:
    """``<!-- Image`` placeholders in *body*, ignoring code (block and inline) and table cells.
    A placeholder in a cell is the cell's text, not a figure the reading marked on the page."""
    return sum(
        len(_PLACEHOLDER_RE.findall(line)) for line in _outside_fences(body, drop_tables=True)
    )


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


def shipped_image_count(
    body: str,
    doc_dir=None,
    *,
    page_num: int | None = None,
    credited_chart_regions: set[int] | None = None,
) -> int:
    """Figure images in *body*, ignoring fenced code blocks. A placeholder is matched
    one-to-one by a shipped image: figure 1 shipping never excuses figure 2, a table crop is not
    a figure, and one file linked twice is one picture.

    With *doc_dir* a link counts only if it resolves to an asset that exists on disk
    (``manifest.page_asset_refs``: the manifest's own scanner, which skips literal context,
    URLs, anchors, missing files and paths that escape *doc_dir*). A link a model wrote to a file
    nobody saved is not a shipped picture (Fable 1). Without *doc_dir* the link text decides.

    *credited_chart_regions*, with *page_num*: when given, a GH-189 ``chart_region_p{page}_{i}``
    crop of this page counts only if ``i`` is in the set (the boxes that carry regional tick
    evidence); the crop of a decorative box is not the figure a model marked. ``None`` credits
    every one.
    """
    if doc_dir is not None:
        from socr.core.manifest import page_asset_refs

        refs = [
            ref.logical_path for ref in page_asset_refs("\n".join(_outside_fences(body)), doc_dir)
        ]
    else:
        refs = [ref for line in _outside_fences(body) for ref in _IMAGE_RE.findall(line)]
    counted: set[str] = set()
    for ref in refs:
        if not _is_figure_image(ref):
            continue
        if credited_chart_regions is not None and page_num is not None:
            match = re.search(rf"chart_region_p{int(page_num)}_(\d+)\.png$", ref)
            if match and int(match.group(1)) not in credited_chart_regions:
                continue
        counted.add(ref)
    return len(counted)


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


def _alike(values: list[float], tol: float) -> list[list[int]]:
    """Indices of *values* chained into groups whose consecutive sorted members differ by at
    most *tol*."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    groups: list[list[int]] = []
    for i in order:
        if groups and values[i] - values[groups[-1][-1]] <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _is_table_header(row: list, words: list) -> bool:
    """A row of numbers with other words beneath it, in its own columns.

    A table's header introduces rows: beneath it, at least :data:`MIN_TICK_RUN` of its numbers
    have a word under them (their horizontal extents overlap). A chart's x-axis row is the last
    line of the drawing's box: what sits under it is an axis title, a legend entry, or the
    figure's own ``NOTE:`` / ``SOURCE:`` line, and those are not table rows, so a line that
    opens with a note lead is not counted (a note runs the figure's whole width and would
    otherwise overlap every column).
    """
    from socr.figures.figure_crop import _NOTE_LEAD_RE

    bottom = max(float(w[3]) for w in row)
    lines: dict[tuple[int, int], list] = {}
    for w in words:
        if (float(w[1]) + float(w[3])) / 2.0 > bottom:
            lines.setdefault((int(w[5]), int(w[6])), []).append(w)
    below = [
        w
        for line in lines.values()
        if not _NOTE_LEAD_RE.match(" ".join(str(v[4]) for v in sorted(line, key=lambda v: v[0])))
        for w in line
    ]
    under = sum(
        1
        for r in row
        if any(float(r[0]) < float(w[2]) and float(w[0]) < float(r[2]) for w in below)
    )
    return under >= MIN_TICK_RUN


def tick_axis(words: list) -> str:
    """Where the chart's scale is, from where the numbers ARE on the page: ``"y"`` for a column,
    ``"x"`` for a row, ``""`` for neither.

    Text-layer lines are the extractor's grouping and not a geometry: a table's header row of
    years can be stored as one layer line per year (Fable 3, PR #1309), so counting lines
    called it a column. Here a scale needs a common axis line:

    * ``"y"``: at least :data:`MIN_TICK_RUN` numbers sharing one x-column (right edge, left edge
      or centre within one body line *pitch*), one per line, that form an evenly spaced run;
    * ``"x"``: at least :data:`MIN_TICK_RUN` numbers sharing one baseline (within half a body
      line pitch: closer to this line than to the next) that form an evenly spaced run, and
      that are not the header of a table (:func:`_is_table_header`).

    The pitch is the body line pitch of the words given (``figure_crop.line_pitch``): one number
    for the page, not a constant.
    """
    from socr.figures.figure_crop import line_pitch

    pitch = line_pitch(words)
    if pitch <= 0:
        return ""
    numbers = [w for w in words if _TICK_NUMBER_RE.match(str(w[4]))]
    if len(numbers) < MIN_TICK_RUN:
        return ""
    for edge in (lambda w: float(w[0]), lambda w: float(w[2]), lambda w: (w[0] + w[2]) / 2.0):
        for group in _alike([edge(w) for w in numbers], pitch):
            column = sorted((numbers[i] for i in group), key=lambda w: float(w[1]))
            if len(column) >= MIN_TICK_RUN and tick_run("\n".join(str(w[4]) for w in column)) >= (
                MIN_TICK_RUN
            ):
                return "y"
    for group in _alike([(float(w[1]) + float(w[3])) / 2.0 for w in numbers], pitch / 2.0):
        row = sorted((numbers[i] for i in group), key=lambda w: float(w[0]))
        if (
            len(row) >= MIN_TICK_RUN
            and tick_run(" ".join(str(w[4]) for w in row)) >= MIN_TICK_RUN
            and not _is_table_header(row, words)
        ):
            return "x"
    return ""


@dataclass(frozen=True)
class RegionAssessment:
    index: int
    covered: bool
    ticks: int
    table_region: bool
    axis: str = ""

    @property
    def has_evidence(self) -> bool:
        """A chart's scale inside this region: a run of evenly spaced numbers on a common axis
        line (:func:`tick_axis`), not a table's header, not a table. The one test the native
        lane and the model lane's table-page count share."""
        return not self.table_region and self.ticks >= MIN_TICK_RUN and self.axis != ""

    @property
    def fires(self) -> bool:
        """Evidence in this region and nothing of its own ships it."""
        return not self.covered and self.has_evidence

    def to_dict(self) -> dict:
        return {
            "region": self.index,
            "covered": self.covered,
            "ticks": self.ticks,
            "axis": self.axis,
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
        layer = [] if is_table else region_lines(inside)
        ticks = 0 if is_table else tick_run("\n".join(layer))
        out.append(
            RegionAssessment(
                index=index,
                covered=page_shipped or index in covered,
                ticks=ticks,
                table_region=is_table,
                axis="" if is_table else tick_axis(inside),
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
    """The whole native rule. ``None`` when the page has no chart box (the caller records
    that with :func:`no_box_decision`).

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
    # Panels of one figure share a crop: one ref, however many regions it covers.
    refs = tuple(dict.fromkeys(saved[a.index] for a in firing if saved.get(a.index)))
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


_TRAILING_ORDINAL_RE = re.compile(r"(\d+)\s*$")


def label_ordinal(label: str, position: int) -> int:
    """The region / placeholder ordinal a label names (``chart region 2``, ``image placeholder
    3``), which is what the scope must carry: failed region 2 alone is ``figure:2``, not
    ``figure:1`` by its position in the list. A label with no trailing number falls back to its
    list position."""
    match = _TRAILING_ORDINAL_RE.search(label or "")
    return int(match.group(1)) if match else position


def clause_records(page_num: int, labels: tuple[str, ...], kind: str) -> tuple[dict, ...]:
    """One figure-specific, machine-readable mark clause per lost region / unmatched
    placeholder. UNKNOWN: socr did not establish the figure is absent; it failed to show it."""
    from socr.core.marks import Check, CheckResult, region_scope

    reason = PLACEHOLDER_REASON if kind == PLACEHOLDER_KIND else RENDER_FAILED_REASON
    return tuple(
        {
            "check": Check.FIGURE_TEXT.value,
            "result": CheckResult.UNKNOWN.value,
            "scope": region_scope("figure", label_ordinal(label, position)),
            "reason": f"{label}: {reason}",
        }
        for position, label in enumerate(labels, start=1)
    )


def no_box_decision() -> PresenceDecision:
    """The record of a page the lane looked at and found no chart box on. Never marks."""
    return PresenceDecision(PRESENCE_PASS_KIND, PASS_NO_BOX)


def left_to_chart_region_pass_decision() -> PresenceDecision:
    """The record of a table-signal page: GH-189's, so not decided here. Never marks."""
    return PresenceDecision(PRESENCE_PASS_KIND, PASS_LEFT_TO_CHART_REGION_PASS)
