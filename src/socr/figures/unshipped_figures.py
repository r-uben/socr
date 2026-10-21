"""#1187: a figure a page NAMES never ships as a caption with nothing under it.

The defect, measured on finished run 23319732 (395 PDFs, 17,091 pages): of the 3,082 model-read
pages the meter counts CHECKED, 2,751 ship no picture at all, and on 147 of those the shipped
body itself names a figure -- ``Figure 7: Bias and MSE ...`` with nothing under it,
``figure_refs=[]``, and no mark. 147/3,082 is 4.8%, inside the issue's estimate of 5.1%
(95% CI 1.9-12.8%).

Why the existing lanes miss them, measured rather than assumed:

* The issue's own hypothesis -- ``has_chart_marks`` scoring 0 data marks on a hairline
  monochrome plot -- is FALSE for most of the adjudicated pages: ``has_chart_marks`` returns
  True on 7 of the 12, and ``_vector_chart_found`` on 6. The loss is downstream of detection.
* ``_is_chart_asset_page`` opens with ``if not self._is_native_eligible_without_ocr(...)``, on
  the stated assumption that "the page would go to the OCR ladder anyway; there is nothing to
  intercept". 97 of the 113 first-pass hits carry ``garbled_math_native`` or a sibling, which is
  exactly what makes a page NOT native-eligible. The assumption is that a model reading covers
  the chart. It does not: a VLM transcribes the caption and drops the plot.
* A chart+table page is held out of the page-level lane by ``chart_table_arbitration``, whose
  recorded promise is that ``rowize_from_words_chart_aware`` will emit an inline placeholder
  that ``_render_chart_region_pngs`` resolves. That path runs on ``ps.native_text`` and only
  under ``if not decision.accepted``; when the model's reading IS accepted the shipped text is
  the model's, carries no placeholder, and no crop is ever rendered.
* The general figure phase was not running at all: ``runs_figures = (save_figures or
  describe_figures) and has_text`` and ``save_figures`` defaults to False. No worker log of run
  23319732 carries an "Extracted N figures" or "No figures detected" line. Every picture that
  shipped on that corpus came from a chart lane or a fail-closed floor.

So this lane does not add a detector. It asks one question the other lanes never ask -- does the
body name a figure that the body does not show? -- and answers it with the locator the repo
already has (``FigureExtractor.extract_page``, which found a region on all 12 adjudicated
pages). It ships the crop where the locator is positive, and marks where it is not.

One decision function, two callers
----------------------------------
:func:`decide` holds the whole rule. The live lane
(``UnifiedPipeline._agentic_unshipped_figure_page``) and the re-finalize tool
(``socr-refinalize-unshipped-figures``) differ only in where the body and the PDF come from, so
neither owns a copy of the rule. An earlier draft duplicated it and the two copies had already
drifted in three places by the first review -- the event kind on a render failure, the ``regions``
count, and which captions the event recorded.

Two matchers, not one, and deliberately
---------------------------------------
``scanned_figures._CAPTION_LINE_RE`` (#1030) is measured at 17/17 on a scanned OCR layer, but it
parses the label number as ``[0-9]+``, so on ``Figure 4.5 shows contour maps ...`` it reads the
label as ``Figure 4`` and the DECIMAL POINT as the caption's terminator. Measured here: that is
3 of the 17 pages a mark branch built on it would fire on (``nagel`` p96/p121/p126, all three
in-text cross-references on prose pages). It also caps a caption line at 50 characters, which on
born-digital Markdown rejects 116 of 139 real caption lines -- a cap derived from a scan's
wrapped OCR layer does not describe an unwrapped Markdown line.

So this module parses the whole dotted number, atomically, and splits on what the branch COSTS:

``LOOSE_CAPTION_RE`` also accepts a label followed by whitespace and a non-lowercase character,
which is how a caption with no terminator opens (``Fig. 4.15 2D Gaussians with ...``, dougherty
p74). It gates the branch that SHIPS A PICTURE, where a false fire adds one crop to a page that
already has a located figure region -- the same harm bound #1030 accepts for its own lane ("a
false fire only ADDS one page-image reference beside unchanged text").

``STRICT_CAPTION_RE`` requires a real terminator or the end of the line. It gates the branch that
MARKS on the caption alone, where a false fire demotes a page from CHECKED to MARKED. Measured on
run 23319732: 15 pages, and by eye against the page image 14 of the 15 are genuine captions with
a real figure above them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: The label: a figure word, then the whole number, parsed ATOMICALLY.
#:
#: Both halves are load-bearing and both were found by running this on real pages. Parsing
#: ``4.5`` as ``[0-9]+`` reads the label as ``Figure 4`` and the decimal point as the caption's
#: terminator, so ``Figure 4.5 shows contour maps ...`` is a caption (nagel p96, p121, p126).
#: Parsing it as ``[0-9]+(?:\.[0-9A-Za-z]+)*`` without ``(?>...)`` fixes nothing, because the
#: regex simply BACKTRACKS to ``Figure 3`` when the longer parse fails -- which is how
#: ``Figure 3.E provides the distribution of participants' judgments ...`` (Fed 2016-06-15
#: minutes p17) marked a prose page in the first run of the re-finalize tool on the Fed
#: calibration copy. An atomic group cannot be backtracked into, so a label that parses whole is
#: never re-read as a shorter one with a terminator after it.
#:
#: Leading Markdown emphasis or a heading marker is consumed because a VLM writes
#: ``**Figure 7:** ...`` and ``# Fig 3. ...`` as often as a bare line.
_LABEL = (
    r"^[ \t]*(?:[*_]{1,3}|#{1,6}[ \t]*)?"
    r"(?P<label>(?:FIGURE|Figure|FIG\.?|Fig\.?|CHART|Chart|EXHIBIT|Exhibit)[ \t]*"
    r"(?>"
    r"[0-9]+(?:[.\-][0-9A-Za-z]+)*[a-zA-Z]?"
    r"|[A-Z]{1,2}(?:[.\-][0-9A-Za-z]+)+"
    r"|[A-Z]{1,2}[0-9]+(?:[.\-][0-9A-Za-z]+)*"
    # No tail of its own: the two-letter alternative above consumes ``II-3`` and ``I.2``
    # first, so a tail here could never fail and the mutation check found it unkillable.
    r"|[IVXLC]+"
    r"))"
    r"[*_]{0,3}"
)
#: A terminator, or the end of the line. No lookahead guards the ``.`` here, deliberately: the
#: atomic label above has already consumed the whole number, so nothing can reach this
#: alternative with a decimal point still in front of it. A ``(?![0-9])`` was written here first
#: and the mutation check found it unkillable -- a guard that has never been seen to fail has not
#: been shown to guard anything (CLAUDE.md).
#: A dash of any width followed by a DIGIT is a RANGE, not a terminator: ``Figure 1-3 show
#: …``, ``Figure 1–3 present …``, ``Figure 2—4 show …`` are cross-references, and the
#: label alternatives only consume the ASCII form, so the wider dashes reached this and
#: marked prose pages (owner item 3, both rounds). ``Fig. 1—Probability of sample
#: inclusion.`` must keep matching, so the rule is what FOLLOWS the dash, not the dash.
_TERMINATOR = r"(?:[.:]|[—–\-](?![0-9])|$)"

STRICT_CAPTION_RE = re.compile(_LABEL + _TERMINATOR)
#: Also a label followed by whitespace and a non-lowercase character, which is how a caption with
#: no terminator opens. The module docstring says which branch may use which.
LOOSE_CAPTION_RE = re.compile(_LABEL + r"(?:[.:]|[—–\-](?![0-9])|[ \t]+(?=[^a-z])|$)")

#: Characters a Markdown image's alt text may not carry: ``]`` closes it early, and a newline
#: ends the line. Same guarantee ``marks.clean_reason`` gives the canonical mark grammar -- a
#: builder must not be able to emit a span its own reader would misparse.
_ALT_UNSAFE_RE = re.compile(r"[\[\]\r\n]+")
_WHITESPACE_RE = re.compile(r"\s+")

#: The crop's filename stem. Distinct from ``chart_region_p{n}_{i}.png`` so a page that also
#: carries a chart-region crop cannot have one lane's asset overwritten by the other's.
CROP_STEM = "unshipped_figure"

#: The outcomes this lane records, as audit-event kinds.
#:
#: ``PARTIAL`` was a fifth kind in round 1, for a page naming more figures than it shipped
#: crops. It is gone, because round 2 showed the distinction it drew was not the one that
#: matters: on Fed calibration p3 the page names ONE figure ("Exhibit 1") and carries SIX
#: numbered panels, two of which the locator found. ``len(refs) >= len(captions)`` therefore
#: read as "all of them are shown" while three charts were shipped by nothing (Argon P0-1,
#: reproduced). **A caption count can never establish how many figures a page carries**, so
#: no outcome of this lane claims it does.
RENDERED_KIND = "unshipped_figure_rendered"
RENDER_FAILED_KIND = "unshipped_figure_render_failed"
UNLOCATED_KIND = "unshipped_figure_unlocated"
NOT_FOUND_KIND = "unshipped_figure_not_found"
EVENT_KINDS: frozenset[str] = frozenset(
    {RENDERED_KIND, RENDER_FAILED_KIND, UNLOCATED_KIND, NOT_FOUND_KIND}
)
#: The kinds that MARK the page. Everything this lane acts on is in here except the
#: abstention: a crop is a picture, not a reading, and this lane cannot say its crops cover
#: every figure the page carries. The page-level chart lane already marks exactly this case
#: (``verdict._chart_values_clause``), and shipping a crop is strictly weaker evidence than
#: shipping the whole page image, so claiming CHECKED on it was the stronger claim from the
#: weaker evidence.
MARKED_KINDS: frozenset[str] = frozenset({RENDERED_KIND, RENDER_FAILED_KIND, UNLOCATED_KIND})
#: The subset that ships a picture. The document note and the CLI need the two apart.
SHIPPED_KINDS: frozenset[str] = frozenset({RENDERED_KIND})


def crop_filename(page_num: int, index: int) -> str:
    """The canonical crop filename for one located region. Deterministic: a second run of the
    re-finalize tool must write the same bytes to the same name and change nothing."""
    return f"{CROP_STEM}_p{page_num}_{index}.png"


def figure_captions(body: str, *, strict: bool = False) -> tuple[str, ...]:
    """The LABEL of every figure this body's own captions name, in document order.

    Reads the SHIPPED body, not the page's text layer, and that is the point: the defect is a
    caption the READER can see with nothing under it. A figure named nowhere in the shipped text
    leaves no dangling caption, whatever the layer says.

    The label (``Exhibit 1``, ``Fig. 4.15``), never the whole caption line. The line is unbounded
    -- a Fed minutes cross-reference runs to 900 characters, and one landed inside a mark in the
    first run of the re-finalize tool -- and the label is all a mark or an alt text needs: the
    ruling asks the page to NAME the figure that did not ship, and the caption's own prose is
    already on the page, two lines above.
    """
    pattern = STRICT_CAPTION_RE if strict else LOOSE_CAPTION_RE
    labels = []
    for line in (body or "").split("\n"):
        match = pattern.match(line)
        if match is not None:
            labels.append(" ".join(match.group("label").split()))
    return tuple(labels)


def caption_alt(caption: str) -> str:
    """One caption label, safe to place inside a Markdown image's alt text."""
    flattened = _WHITESPACE_RE.sub(" ", _ALT_UNSAFE_RE.sub(" ", caption or "")).strip()
    return flattened or "figure"


#: Engines whose body is the PDF's own text layer rather than a reading of the page.
#: ``""`` is a page with no engine recorded, which is the same case.
_NATIVE_ENGINES = frozenset({"", "native", "native+equations"})


def page_was_read_by_a_model(engine: str) -> bool:
    """Whether this page's shipped body is a MODEL's reading of it.

    This lane fires only on those, and that is the scope every number it has was measured on:
    the 154/3,082 came from a crosstab filtered to exactly this predicate, and the Fed tool run
    read finalized sidecars.

    A native body is the PDF's text layer, and the page loop sees it BEFORE
    ``_apply_native_paragraphs`` reflows it at finalisation -- so a sentence that wraps with
    ``Figure 4.`` at the start of a raw line reads as a strict caption on a page where the
    shipped text has it mid-paragraph. The owner reproduced that on four real prose pages
    (correia_luck_verner p14, rollet p23, rollet p33, cengiz p80) and measured ~7 such
    cross-reference fires in 70 pages of raw layer across 70 econ papers; since round 2 the
    false mark also takes the whole document from ``completed`` to ``partial``.

    Judging the reflowed text instead would be the other way to close it, and it is NOT taken
    here: the lane would then be deciding on a body no measurement covers. Native pages are
    out of scope until they are measured -- see the follow-up issue.
    """
    return str(engine or "").strip().lower() not in _NATIVE_ENGINES


def body_shows_a_picture(body: str) -> bool:
    """Whether *body* already shows a picture of its own. The lane's ONE abstention gate."""
    return any(line.lstrip().startswith("![") for line in (body or "").splitlines())


def locate_figure_regions(pdf_path, page_num: int, max_per_page: int) -> list:
    """The figure regions on ONE page, as fitz rects. THE locator, for both callers.

    ``FigureExtractor.extract_page`` -- the locator the repo already has, which found a region
    on all 12 pages a human adjudicated for #1187 -- run on this page alone, so the doc-wide
    ``figures_max_total`` never applies: this lane answers a question about this page, and a
    document's 26th figure is not less lost than its first. ``figures_max_per_page`` is kept,
    because it is the repo's existing answer to "how many assets may one page carry".

    A box covering essentially the whole page is DROPPED. On a scan the extractor's only region
    is the page raster itself, and a whole-page "crop" is not a figure crop: it is the page, and
    the page image is #1030's artifact and the floors', under their own names and their own
    claims. Shipping one here gave a #1030 control page a second picture of itself under a
    figure caption (its end-to-end test caught it). The bar is
    ``SCAN_RASTER_PAGE_COVERAGE_MIN``, the repo's existing, measured "this raster IS the page"
    floor -- no new number. Such a page then takes the mark branch, which still satisfies the
    ruling: it names the figure that did not ship.

    ``[]`` on any failure, which is the fail-closed direction -- a page socr could not look at
    is not a page with no figure.
    """
    import fitz

    from socr.core.pdf import open_pdf
    from socr.figures.extractor import SCAN_RASTER_PAGE_COVERAGE_MIN, FigureExtractor

    try:
        with open_pdf(pdf_path) as doc:
            page = doc[page_num - 1]
            page_area = max(page.rect.width * page.rect.height, 1.0)
            result = FigureExtractor(
                save_dir=None, max_total=max_per_page, max_per_page=max_per_page
            ).extract_page(page, page_num, doc, 1)
            boxes = []
            for figure in result.figures:
                if figure.bbox is None:
                    continue
                x0, y0, x1, y1 = figure.bbox
                if (x1 - x0) * (y1 - y0) / page_area >= SCAN_RASTER_PAGE_COVERAGE_MIN:
                    continue
                boxes.append(fitz.Rect(x0, y0, x1, y1))
            return boxes
    except Exception:  # noqa: BLE001 -- the caller marks the page
        return []


@dataclass(frozen=True)
class FigureDecision:
    """What this lane does about one page. Built by :func:`decide`, applied by both callers."""

    captions: tuple[str, ...]
    strict: tuple[str, ...]
    regions: int
    refs: tuple[str, ...]
    #: The labels the page must MARK: figures it names and does not show. Empty when every
    #: figure it names is shown, or when nothing is owed.
    unlocated: tuple[str, ...]
    kind: str
    detail: str

    @property
    def marks(self) -> bool:
        return bool(self.unlocated)

    @property
    def state_fields(self) -> dict:
        """The three ``PageState`` fields / sidecar keys this lane owns."""
        return {
            "unshipped_figure_refs": list(self.refs),
            "unshipped_figure_captions": list(self.captions),
            "unshipped_figure_unlocated": list(self.unlocated),
            "unshipped_figure_kind": self.kind,
        }

    def event_data(self) -> dict:
        return {
            "captions": list(self.captions),
            "regions": self.regions,
            "rendered": len(self.refs),
            "unlocated": list(self.unlocated),
        }


def decide(body: str, locate, render) -> FigureDecision | None:
    """The whole rule, in one place. ``None`` when this page is not this lane's.

    ``locate()`` returns the figure regions on the page (fitz rects); ``render(bboxes)`` returns
    one ref per crop it managed to save, and may return ``[]`` without writing anything (the
    ``--dry-run`` path renders nothing and the caller supplies the names it WOULD write).

    **Every page this lane acts on is MARKED.** Round 1 marked only the pages it could give no
    picture, on the reasoning that a shipped crop discharges the debt. Fed calibration p3
    disproved it: the page names "Exhibit 1", carries six numbered panels, and the locator finds
    two regions — both in the left column — so three charts shipped nowhere while the page
    counted CHECKED (Argon P0-1). A caption count cannot establish how many figures a page has,
    and nothing else here can either, so the lane states what it actually knows: a region was
    located and cropped, its values were not transcribed, and socr cannot say the crops cover
    every figure the page carries.

    The branches, and what makes each of them positive evidence:

    * **No caption at all** -> ``None``. Nothing is dangling.
    * **No region located** -> the page may be marked only on the STRICT matcher, because the
      only evidence here is the caption's own shape and the loose one also fires on a
      cross-reference that opens a line. On the loose matcher alone the lane records that it
      looked and says nothing (``NOT_FOUND_KIND``).
    * **A region located** -> geometry is now the positive evidence that a figure is on this
      page, so the loose matcher is enough to name it.
    """
    captions = figure_captions(body)
    if not captions:
        return None
    strict = figure_captions(body, strict=True)
    bboxes = locate()
    regions = len(bboxes)
    refs = tuple(render(bboxes)) if bboxes else ()

    if not regions:
        if not strict:
            return FigureDecision(
                captions,
                strict,
                0,
                (),
                (),
                NOT_FOUND_KIND,
                "a figure label opens a line on this page but no figure region was located "
                "and the label is not caption-shaped; nothing is owed",
            )
        return FigureDecision(
            captions,
            strict,
            0,
            (),
            strict,
            UNLOCATED_KIND,
            "this page names a figure and shows none; socr could not locate a figure region "
            "on the page, so the page is marked",
        )

    named = strict or captions
    if not refs:
        return FigureDecision(
            captions,
            strict,
            regions,
            (),
            named,
            RENDER_FAILED_KIND,
            f"this page names a figure, {regions} region(s) were located, and NO crop could "
            "be saved: the figure is preserved nowhere",
        )
    return FigureDecision(
        captions,
        strict,
        regions,
        refs,
        named,
        RENDERED_KIND,
        f"this page names {len(named)} figure(s) and ships {len(refs)} crop(s) of the "
        f"{regions} figure region(s) located on it; the crops are pictures, not readings, so "
        "their values are not transcribed, and socr cannot say they cover every figure the "
        "page carries",
    )


def figure_block(page_num: int, refs: list[str], captions: tuple[str, ...]) -> str:
    """The Markdown this lane appends: one image per located region.

    Appended rather than spliced under the caption. Every sibling lane in this repo appends its
    asset (``_apply_scanned_figure_guard``, the page-level chart lane), the alt text names the
    figures so a reader can bind them, and splicing would need a caption-to-region binding this
    lane does not have -- see the decision log for what measuring one would take.

    EVERY caption is named, not the first. A page can carry a wrapped cross-reference above a
    real caption (``Figure 1 (panel a) shows ...`` over ``Figure 2: ...``), and labelling the
    second figure's crop with the first label would state a binding this lane does not have
    (Argon, diff review).
    """
    lead = ", ".join(caption_alt(c) for c in captions) or f"Figure on page {page_num}"
    total = len(refs)
    lines = []
    for index, ref in enumerate(refs, start=1):
        where = f"page {page_num}" if total == 1 else f"page {page_num}, region {index} of {total}"
        lines.append(f"![{lead} — {where}]({ref})")
    return "\n\n".join(lines)


#: What the page's MARK says, per outcome. One sentence each, and each is the truth about that
#: branch: round 1 hard-coded the "could not locate" wording and used it on every branch,
#: including the one where crops DID ship.
_CLAUSE_REASON: dict[str, str] = {
    UNLOCATED_KIND: (
        "this page names it and shows no picture of it; socr could not locate a figure "
        "region on the page; read it from the page image"
    ),
    RENDER_FAILED_KIND: (
        "this page names it, a figure region was located, and no crop of it could be saved; "
        "it is preserved nowhere; read it from the page image"
    ),
    RENDERED_KIND: (
        "this page names it and ships a crop of the figure region(s) located on the page; a "
        "crop is a picture, not a reading, so its values are not transcribed, and socr cannot "
        "say the crops cover every figure this page carries; read it from the page image"
    ),
}


def clause_records(page_num: int, captions: tuple[str, ...], kind: str) -> tuple[dict, ...]:
    """One machine-readable clause per figure this page names and does not show whole.

    Records first, line second (#1155's lesson: "a rendered mark string is not the
    machine-readable evidence"). ``core.verdict`` reads these back and ``_apply_verdict_marks``
    renders the page's lines from the same records, so the line a reader sees and the clause a
    consumer parses cannot disagree.

    ``UNKNOWN`` rather than ``FAIL``: the check could not run. socr did not establish that the
    page has no figure -- it failed to show one the page names, or shipped a crop it cannot
    prove complete.
    """
    from socr.core.marks import Check, CheckResult, region_scope

    reason = _CLAUSE_REASON.get(kind, _CLAUSE_REASON[UNLOCATED_KIND])
    return tuple(
        {
            "check": Check.FIGURE_TEXT.value,
            "result": CheckResult.UNKNOWN.value,
            "scope": region_scope("figure", index),
            "reason": f"{caption_alt(caption)}: {reason}",
        }
        for index, caption in enumerate(captions, start=1)
    )


#: Opens every note :func:`document_note` builds, so a re-finalize run can find and replace the
#: note it wrote last time rather than concatenating a second copy (Argon, round-1 diff review).
NOTE_PREFIX = "unshipped figures: "


def document_note(shipped: list[int], marked: list[int]) -> str:
    """The lane's ``metadata.json`` note, built from the pages in each class.

    One builder, used by the live run's ``_unshipped_figure_note`` and by the re-finalize tool,
    so a re-finalized document says exactly what a fresh one would. The leading
    :data:`NOTE_PREFIX` is what lets the tool REPLACE its own earlier note instead of appending
    a second copy when a run is interrupted and redone.
    """
    parts = []
    if shipped:
        parts.append(
            f"page(s) {', '.join(str(n) for n in sorted(set(shipped)))} ship a crop of the "
            "located figure region rather than a reading of it, so their values are not "
            "transcribed"
        )
    if marked:
        parts.append(
            f"page(s) {', '.join(str(n) for n in sorted(set(marked)))} name a figure socr "
            "could not locate, could not save, or could not tell apart from the others the "
            "page names, and are marked"
        )
    # No ``;`` ANYWHERE inside, including between the two clauses: ``;`` is what separates one
    # lane's note from another's in ``metadata.json``'s ``error``, so a note carrying one
    # cannot be found and replaced as a unit -- which is how the re-finalize tool concatenated
    # a second copy of it when a run was interrupted and redone (Argon, diff review).
    return f"{NOTE_PREFIX}{' — '.join(parts)}" if parts else ""
