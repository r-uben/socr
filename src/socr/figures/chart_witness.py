"""#1155: the document's own projection table as an independent witness for a
chart reading.

A deterministic chart reading needs something to be checked against before its
numbers ship, and the alternatives available on the page are weak. The
participant-count sum is blind to a one-bin shift, to a bin swap, to offsetting
errors, and to binding the previous meeting's series as the current one's --
both the September and June 2015 FOMC meetings had 17 participants, so June's
bars bound as September's sum to 17 just as happily. The figure's own
``count_unit`` ("Number of participants") declares a unit, not a number.

The Fed SEP release prints, on another page of the SAME document, a table that
publishes each variable's median, central tendency and RANGE for each year, as
exact PDF characters. That range is independent of the drawing, so it can
witness the drawing. Measured for 2015-09-17 against the Fed's own published
HTML (``fomcprojtabl20150917.htm``): all 210 published numeric values appear on
the PDF page exactly, recall 1.0000.

What this module provides:

* ``parse_table1_witness`` -- the published ranges, keyed by variable, series
  row and year, for rows that pass their OWN self-check;
* ``gate_series`` -- whether a series' bins are consistent with one such range.

Two rules this module exists to enforce, both of them the owner's:

* **a wrong number is worse than a missing one.** Every failure path here
  returns a refusal with a named reason, never a guess and never a silent pass.
  No witness, an unrecognised layout, a row that fails its own self-check, or a
  reading that contradicts the range: all withhold.
* **the witness must itself be proven.** A parsed row is only a witness when
  its own arithmetic holds -- median inside central tendency inside range, and
  a value count equal to the header's own column count. A table whose parse is
  unchecked would launder a parsing bug into a published number.

Scope, deliberately narrow (see #1184). Only the **upright single-block**
layout is parsed: one row per series carrying the row label and then one cell
per column, group by group. The 2008-10-29 release splits the same table into
two stacked blocks, and the 2015-09-17 release draws it rotated 90 degrees;
both are REFUSED here with a named reason rather than guessed at, so the gate
withholds on them until #1184 lands. The Fed minutes corpus runs to 1,021
documents across more decades than these three, so the layout set is not
closed, and an unrecognised layout must always refuse rather than improvise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: A range separator as the releases print it: en dash, em dash, hyphen, or the
#: word "to". NOT a minus sign -- a bin label's "1.8-1.9" is two positive
#: numbers joined by a hyphen, and reading that hyphen as a sign yields
#: ``(1.8, -1.9)``, which then fails every comparison silently. This cost one
#: wrong measurement run during #1155 before it was caught.
_MINUS = "-−"
#: A figure, which MAY be negative. The Fed prints negative projections
#: ("-0.3 to 0.5", 2008-10-29 Table 1) and negative federal-funds bins
#: ("−0.37-−0.13", with a Unicode MINUS, not a hyphen). An unsigned pattern
#: dropped those signs silently: Argon measured
#: ``bin_interval("−0.37-−0.13")`` returning ``(0.37, 0.13)`` -- an interval
#: whose low exceeds its high -- and ``-0.3 to 0.5`` parsed as ``(0.3, 0.5)``.
_NUM = rf"[{_MINUS}]?\d+\.\d+"
#: The separator between two figures. A hyphen is BOTH the separator and a
#: possible sign, so the two are told apart by what follows: a separating
#: hyphen always has a figure after it, and that figure may carry its own
#: sign. Matching the sign first would eat the separator; requiring a digit or
#: a sign after the separator keeps "−0.37-−0.13" splitting on the bare "-".
_SEP = rf"(?:\s*(?:to|–|—)\s*|\s*[-−]\s*(?=[{_MINUS}]?\d))"
_RANGE_RE = re.compile(rf"({_NUM}){_SEP}({_NUM})")
_CELL_RE = re.compile(rf"({_NUM}){_SEP}({_NUM})|({_NUM})|(n\.a\.)")
_YEAR_RE = re.compile(r"\b(?:19|20)\d\d\b")
#: A sub-row of a variable, naming the meeting whose projections it repeats.
_PRIOR_ROW_RE = re.compile(r"^([A-Z][a-z]+)\s+projection")

#: The row labels the SEP projection table prints. Order matters: the longer
#: "Core PCE inflation" must be tested before "PCE inflation" or it is never
#: matched. The ligature in "inflation" differs between releases, so the test
#: stops at the stem.
_VARIABLES: tuple[str, ...] = (
    "Change in real GDP",
    "Unemployment rate",
    "Core PCE in",
    "PCE in",
    "Federal funds rate",
)

#: ``layout`` / ``refusal`` values. Named so a caller's mark can quote them.
LAYOUT_UPRIGHT_SINGLE_BLOCK = "upright_single_block"
REFUSED_ROTATED = "table drawn rotated; layout not parsed here (see #1184)"
REFUSED_SPLIT_BLOCKS = "table split into stacked blocks; layout not parsed here (see #1184)"
REFUSED_NO_HEADER = "no year header row found in this table"

#: ``gate_series`` refusal reasons.
GATE_PASSED = ""
GATE_NO_FILLED_BINS = "the series has no filled bin to check"
GATE_UNPARSEABLE_BIN = "a bin label could not be read as a numeric interval"
GATE_OUT_OF_RANGE = "the filled bins do not span the published range"


@dataclass(frozen=True)
class WitnessTable:
    """Published ranges that passed their own self-check, plus what was refused.

    ``rows`` maps ``(variable, series_row, year)`` to ``(lo, hi)``.
    ``series_row`` is ``"current"`` for the variable's own row and ``"prior"``
    for its ``"<Month> projection"`` sub-row -- the previous meeting's figures,
    which a series must be matched to by ITS OWN month, never by position.
    """

    rows: dict[tuple[str, str, str], tuple[float, float]] = field(default_factory=dict)
    rejected: tuple[tuple[str, str, str], ...] = ()
    prior_months: frozenset[str] = frozenset()
    layout: str = ""
    refusal: str = ""

    def range_for(self, variable: str, series_row: str, year: str) -> tuple[float, float] | None:
        return self.rows.get((variable, series_row, year))

    def series_row_for(self, series_name: str) -> str:
        """Which row of the table a series belongs to, by its OWN month.

        Every series on these figures is month-prefixed ("September
        projections", "April projections"), so testing merely for a month name
        classes all of them as the prior meeting's -- a bug that checked every
        current-meeting series against the previous meeting's range during
        #1155. A series is the prior meeting's only when its month is one the
        table's own sub-rows name.
        """
        head = re.match(r"([A-Z][a-z]+)\b", series_name or "")
        return "prior" if (head and head.group(1) in self.prior_months) else "current"


def _page_is_rotated(page) -> bool:
    """Whether this page's text is drawn turned away from horizontal.

    Delegates to ``born_digital.upright_rotation_for``, which reads the text
    direction the page itself records. Two cheaper tests were tried and are
    wrong here: ``page.rotation`` reads 0 on these releases because they draw
    rotated CONTENT inside an unrotated page, and "a word box taller than it is
    wide" fails on exactly this table, because a rotated ONE-CHARACTER token --
    which is most of a numeric table -- is wider than it is tall.
    """
    try:
        from socr.core.born_digital import upright_rotation_for

        return upright_rotation_for(page) % 360 != 0
    except Exception:
        return False


def _upright_rows(page) -> list[str]:
    """The page's lines, left to right, banded by their baseline."""
    try:
        words = page.get_text("words")
    except Exception:
        return []
    bands: dict[float, list[tuple[float, str]]] = {}
    for x0, y0, x1, y1, text, *_ in words:
        bands.setdefault(round(y0), []).append(((x0 + x1) / 2, text))
    return [" ".join(t for _, t in sorted(v)) for _, v in sorted(bands.items())]


def _cells(line: str) -> list[tuple[float, float] | None]:
    """The line's value cells, left to right.

    A cell is a range, or a bare value read as the degenerate range ``[x, x]``
    -- the releases print a collapsed range as one figure ("2.0", not
    "2.0 - 2.0"), and dropping those silently loses whole variables. ``n.a.``
    is a hole: present, but carrying no value.
    """
    out: list[tuple[float, float] | None] = []
    for m in _CELL_RE.finditer(line):
        if m.group(1):
            out.append((_as_float(m.group(1)), _as_float(m.group(2))))
        elif m.group(3):
            value = _as_float(m.group(3))
            out.append((value, value))
        else:
            out.append(None)
    return out


def _header_years(rows: list[str]) -> list[str]:
    """The year labels the table's own header prints, in printed order."""
    best: list[str] = []
    for line in rows:
        if line.startswith("Table"):
            continue
        years = _YEAR_RE.findall(line)
        if len(years) > len(best):
            best = years
    seen: set[str] = set()
    ordered: list[str] = []
    for year in best:
        if year not in seen:
            seen.add(year)
            ordered.append(year)
    return ordered


def parse_table1_witness(page) -> WitnessTable:
    """Parse the projection table on *page* into self-checked witness ranges.

    Returns a ``WitnessTable`` whose ``rows`` are only those that passed the
    self-check. A layout this function does not parse returns an EMPTY table
    carrying a named ``refusal``; the caller must withhold rather than proceed.
    Never raises.
    """
    if _page_is_rotated(page):
        return WitnessTable(refusal=REFUSED_ROTATED)

    rows = _upright_rows(page)
    years = _header_years(rows)
    if not years:
        return WitnessTable(refusal=REFUSED_NO_HEADER)

    page_text = " ".join(rows)
    if "Longer" in page_text:
        years = [*years, "Longer run"]
    # The table declares its own groups: a median column group exists only from
    # the 2012 format onward. Nothing here assumes a count.
    groups = 3 if "Median" in page_text else 2
    expected = len(years) * groups

    found: dict[tuple[str, str, str], tuple[float, float]] = {}
    rejected: list[tuple[str, str, str]] = []
    prior_months: set[str] = set()
    current_variable: str | None = None
    short_rows = 0
    data_rows = 0

    for line in rows:
        variable = next((v for v in _VARIABLES if line.startswith(v)), None)
        prior = _PRIOR_ROW_RE.match(line)
        if variable is not None:
            current_variable = variable
            series_row = "current"
        elif prior is not None and current_variable is not None:
            prior_months.add(prior.group(1))
            series_row = "prior"
        else:
            continue

        data_rows += 1
        cells = _cells(line)
        if len(cells) != expected:
            # A row carrying a whole-number multiple less than expected is the
            # split-block layout, where each block holds one column group.
            if cells and len(cells) < expected:
                short_rows += 1
            rejected.append(
                (current_variable, series_row, f"{len(cells)} cells, header declares {expected}")
            )
            continue

        n = len(years)
        if groups == 3:
            median, central, full = cells[:n], cells[n : 2 * n], cells[2 * n :]
        else:
            median, central, full = [None] * n, cells[:n], cells[n:]

        broken = ""
        for i in range(n):
            if central[i] is None or full[i] is None:
                continue
            if not (full[i][0] - 1e-9 <= central[i][0] and central[i][1] <= full[i][1] + 1e-9):
                broken = "central tendency is not inside the range"
                break
            if median[i] is not None and not (
                central[i][0] - 1e-9 <= median[i][0] <= central[i][1] + 1e-9
            ):
                broken = "median is not inside the central tendency"
                break
        if broken:
            rejected.append((current_variable, series_row, broken))
            continue

        for i, year in enumerate(years):
            if full[i] is not None:
                found[(current_variable, series_row, year)] = full[i]

    # Every data row short of the header's own column count is the stacked-block
    # layout, not a table of broken rows. Say so, so the caller's mark names a
    # layout rather than implying the document is corrupt.
    if data_rows and short_rows == data_rows:
        return WitnessTable(refusal=REFUSED_SPLIT_BLOCKS, rejected=tuple(rejected))

    return WitnessTable(
        rows=found,
        rejected=tuple(rejected),
        prior_months=frozenset(prior_months),
        layout=LAYOUT_UPRIGHT_SINGLE_BLOCK,
    )


def _as_float(token: str) -> float:
    """A figure's value, with the Unicode MINUS (U+2212) normalised to ASCII."""
    return float(token.replace("\u2212", "-"))


def bin_interval(bin_label: str) -> tuple[float, float] | None:
    """A bin label's own numeric interval, or None when it carries no figures.

    A hyphen in these labels is BOTH the separator and a possible sign, so the
    label is matched as a RANGE rather than scanned for figures: scanning
    "1.8-1.9" with a sign-bearing pattern finds ``1.8`` and ``-1.9``, which is
    the same silent corruption in the opposite direction from the unsigned
    pattern that preceded it (that one read "\u22120.37-\u22120.13" as
    ``(0.37, 0.13)``, low above high). The range pattern knows where the
    separator is and keeps each figure's own sign.
    """
    text = bin_label or ""
    match = _RANGE_RE.search(text)
    if match:
        return (_as_float(match.group(1)), _as_float(match.group(2)))
    single = re.search(_NUM, text)
    if not single:
        return None
    value = _as_float(single.group(0))
    return (value, value)


def gate_series(filled_bin_labels, published_range) -> tuple[bool, str]:
    """Whether a series' filled bins are consistent with its published range.

    The check: the LOWEST filled bin must contain the published minimum and the
    HIGHEST filled bin must contain the published maximum. That is the sharpest
    form the witness supports -- the range is by definition the lowest and
    highest projection any participant made, so the bins holding them are
    determined, and shifting the reading by one bin in either direction moves
    at least one of them off its endpoint.

    Its limit, stated because it is real and was measured rather than argued:
    a shift that leaves both extremes inside their own bins still passes. This
    gate narrows what can ship; it does not prove a reading correct.

    ``filled_bin_labels`` is the labels of the bins the reader resolved to a
    count above zero, in bin order.
    """
    if not filled_bin_labels:
        return False, GATE_NO_FILLED_BINS
    low = bin_interval(filled_bin_labels[0])
    high = bin_interval(filled_bin_labels[-1])
    if low is None or high is None:
        return False, GATE_UNPARSEABLE_BIN
    published_low, published_high = published_range
    if not (low[0] - 1e-9 <= published_low <= low[1] + 1e-9):
        return False, GATE_OUT_OF_RANGE
    if not (high[0] - 1e-9 <= published_high <= high[1] + 1e-9):
        return False, GATE_OUT_OF_RANGE
    return True, GATE_PASSED


#: What a panel line says when the numbers DO ship. The owner's standing rule is that a
#: model reading ships "(OCR, not verified)" and a mark never claims a value was verified
#: (#1181). A gated chart reading is in the same position: the gate NARROWS what can ship,
#: it does not prove a reading correct, and only the two extreme bars were compared at
#: all. Saying so where the reader is looking is the point -- the limitation is in this
#: module's docstring too, but a docstring is not read by whoever cites the number.
SHIPPED_REASON = (
    "counts read from the drawing, not verified; only the lowest and highest filled bars "
    "were checked against this document's own projection-table range"
)


def panel_mark_line(
    page_num: int,
    region_index: int,
    region_count: int,
    reason: str,
    *,
    shipped: bool,
    decided: bool = True,
) -> str:
    """One canonical #1145 line for one chart region of a page.

    ``region_index`` is 1-based in the order ``chart_region_bboxes`` returns, which is
    the page's DRAWING order, not a geometric one. That is the same index the region's
    crop is named with (``chart_region_p{page}_{index}.png``), so a reader can always
    match this line to the picture it is about. No claim is made here about where the
    region sits on the page, because nothing sorts that list.

    ``decided`` is False when the check could not run or could not decide (no witness,
    nothing resolved); it selects UNKNOWN over FAIL, the distinction #1145 draws.
    """
    from socr.core.marks import Check, CheckResult, page_unchecked_clause_mark, region_scope

    return page_unchecked_clause_mark(
        page_num,
        Check.CHART_REGION,
        CheckResult.UNKNOWN if (shipped or not decided) else CheckResult.FAIL,
        region_scope("chart", region_index),
        f"region {region_index} of {region_count}: {reason}",
    )


#: What a region's line says when its picture could not be drawn. The page still ships its
#: prose and its other regions; a render failure must never cost the page its text, and
#: must never pass silently either.
PICTURE_NOT_RENDERED = "picture not rendered ({error}); values not shipped"
#: What a region's line says in this version, where the gate is built but no count is
#: wired to an output path yet. Distinct from a gate REFUSAL on purpose: saying the
#: numbers were judged and rejected, when nothing judged them, would be a false record.
COUNTS_NOT_WIRED = (
    "the reader resolved this panel, but counts are not shipped in this version; "
    "the witness gate is not yet wired to an output path"
)


def region_clause_records(
    page_num: int, *, region_pictures: dict, reading, region_count: int
) -> tuple[dict, ...]:
    """#1155 round 3: one MACHINE-READABLE clause per chart region, from the results.

    The first version of this lane wrote a per-region line into the page and recorded
    nothing: all 12 finalized verdicts carried zero ``chart_region`` clauses while the
    Markdown showed a mark for every region (#1155 PR 1 re-review, Astra and Argon both).
    A rendered mark string is not a record -- nothing downstream parses the page back --
    so the per-region outcome is built HERE, as the clause dicts
    ``core.verdict.VerdictClause.from_dict`` reads, and the page's lines are rendered
    FROM those same records by :func:`region_clause_mark`. One source, so the line a
    reader sees and the clause a consumer reads cannot drift apart.

    ``result`` is UNKNOWN when the check could not run or could not decide (no picture,
    nothing resolved, counts not wired) and FAIL only where a region's own outcome was
    established -- the distinction #1145 draws, applied per region rather than per page.
    """
    from socr.core.marks import Check, CheckResult, region_scope

    records: list[dict] = []
    for index in range(1, region_count + 1):
        picture = region_pictures.get(index)
        panel = reading.panels.get(index) if getattr(reading, "panels", None) else None
        if not picture:
            reason = PICTURE_NOT_RENDERED.format(
                error=region_pictures.get(f"error:{index}", "no picture was produced")
            )
            decided = False
        elif panel is None:
            reason = (
                "the reader resolved no panel here"
                f" ({reading.refusals.get(index, 'no reason recorded')}); values not shipped"
            )
            decided = False
        elif not panel.label:
            reason = (
                "the panel carries no printed label, so its values cannot be bound to a "
                "year; values not shipped"
            )
            decided = True
        else:
            reason = f"panel {panel.label}: {COUNTS_NOT_WIRED}"
            decided = False
        records.append(
            {
                "check": Check.CHART_REGION.value,
                "result": (CheckResult.FAIL if decided else CheckResult.UNKNOWN).value,
                "scope": region_scope("chart", index),
                "reason": f"region {index} of {region_count}: {reason}",
            }
        )
    return tuple(records)


def clause_mark(page_num: int, record: dict) -> str:
    """The canonical line one clause record of this lane states.

    Round 4: every mark this lane writes is rendered through here, from the record that
    is also handed to the verdict -- the per-region records above and the two
    page-scoped ones below. A mark rendered from anything else could say something the
    record does not, which is the defect round 3 fixed for regions and round 4 for the
    page-scoped lines (Astra P2).
    """
    from socr.core.marks import page_unchecked_clause_mark

    return page_unchecked_clause_mark(
        page_num,
        record["check"],
        record["result"],
        record["scope"],
        record["reason"],
    )


#: What one line says when the reading-order gate refused this page's recovered prose.
#: #1144 round 5 holds back text whose bands are not in the page's own reading order, and
#: that refusal used to be SILENT here: the builder set ``recovered = ""`` and the page
#: shipped crops with no statement that its text had been withheld and -- because the
#: full-page image was embedded in the string being discarded -- no way to read it either
#: (#1155 PR 1 re-review, Astra).
ORDER_REFUSED_REASON = (
    "this page's text was withheld whole: the bands recovered from it are not in the "
    "page's own reading order (#1144), and words in the wrong order rank below missing "
    "ones; read the page's text from the page image"
)


def _page_coverage_record(reason: str) -> dict:
    """One page-scoped ``native_coverage=FAIL`` record: text of the page did not ship."""
    from socr.core.marks import SCOPE_PAGE, Check, CheckResult

    return {
        "check": Check.NATIVE_COVERAGE.value,
        "result": CheckResult.FAIL.value,
        "scope": SCOPE_PAGE,
        "reason": reason,
    }


def order_refused_record() -> dict:
    """The record for a reading-order refusal of the whole page's prose.

    Round 4 (Astra P2): round 3 produced only a STRING here, so the page said
    ``native_coverage=FAIL`` in its Markdown while its terminal JSON carried no such
    clause -- a machine-readable audit could not tell that the page's whole prose had
    been withheld for ordering. The record is now what exists, and the line is rendered
    from it (:func:`clause_mark`).
    """
    return _page_coverage_record(ORDER_REFUSED_REASON)


def floor_replacement_body(
    page,
    page_num: int,
    recovered_prose: str,
    *,
    region_pictures: dict,
    reading,
    region_count: int,
    extra_marks: list | None = None,
    page_image: str | None = None,
    region_records: tuple[dict, ...] | list[dict] | None = None,
) -> str | None:
    """#1155: what a chart page ships INSTEAD of a bare whole-page failure marker.

    The structure-class floor withholds a page whole when it finds no usable table
    box to splice prose around. On a page that is a CHART, that withheld everything
    -- prose, pictures and the chart reading alike -- and 9 Fed SEP pages shipped
    103 characters of marker and nothing else.

    ``recovered_prose`` is what #1144's own lane positively identified as prose
    OUTSIDE every table box AND every chart region; this function never sees the
    page's raw text. The first version of it did, and shipped the chart's own
    y-axis ticks and shredded bin labels as loose numbers beside a mark saying
    values were not shipped -- 149 numeric lines on one page (#1155 PR 1 review,
    Astra 2 / Argon 2). Nothing inside a chart region can reach the page now,
    because the geometry withheld it before this function was called.

    ``region_pictures`` maps a region index to a picture reference ALREADY
    rendered, and only references not already present in ``recovered_prose`` are
    appended: the native layer embeds its own inline region placeholders, and
    appending unconditionally emitted ten links for five regions.

    ``region_records`` are the per-region clause records the caller has already built
    and RECORDED on the page (``region_clause_records``); the lines below are rendered
    from them, so the page's text and the page's verdict state the same thing by
    construction. Omitted, they are computed here, which is what the probes and the
    isolated tests do -- the bytes are identical either way.

    Returns ``None`` when the reading resolves nothing, leaving the floor's
    existing behaviour untouched: this may only ever ADD to a page that was
    going to ship nothing.
    """
    if not getattr(reading, "panels", None):
        return None

    records = (
        tuple(region_records)
        if region_records is not None
        else region_clause_records(
            page_num,
            region_pictures=region_pictures,
            reading=reading,
            region_count=region_count,
        )
    )
    lines = [clause_mark(page_num, record) for record in records]
    pictures: list[str] = []
    for index in range(1, region_count + 1):
        picture = region_pictures.get(index)
        if picture and picture not in (recovered_prose or ""):
            pictures.append(picture)

    parts = ["\n".join([*lines, *(extra_marks or [])])]
    prose = (recovered_prose or "").strip()
    if prose:
        parts.append(prose)
    parts.extend(pictures)
    # The page image ONCE, after the crops, when the lane withheld page text.
    if page_image and page_image not in (recovered_prose or ""):
        parts.append(page_image)
    return "\n\n".join(parts)


#: What one line says in place of the per-run failure markers the prose lane interleaves.
#: The lane marks EVERY withheld run, which on a Fed SEP page is 3-6 identical
#: "[page N failed: no usable OCR output]" blocks plus the page image each time, because
#: the digit-band rule withholds every figure title and note (they carry "2.A", "table 1").
#: That caption behaviour is #1154's and is NOT changed here; what is changed is saying it
#: once, in the canonical grammar, with the picture once.
WITHHELD_RUNS_REASON = (
    "{runs} run(s) of this page's text were withheld: the prose rule holds back any line "
    "carrying a figure, which on this page includes the figure titles and notes (#1154); "
    "read them from the page image"
)


def collapse_withheld_runs(recovered: str, marker_line: str, png_ref: str, page_num: int):
    """Replace the lane's repeated per-run markers with ONE canonical line.

    Returns ``(prose_without_markers, clause_record_or_None, image_or_None)``. The
    caller renders the record with :func:`clause_mark`, places that line with the
    per-region marks and carries the image once, so a page carries one statement of what
    was withheld instead of one per run (#1155 PR 1 re-review, Argon 4). Round 4 changed
    the middle element from the rendered LINE to the record it is rendered from, so the
    same statement reaches the page and the verdict.
    """
    text = recovered or ""
    runs = text.count(marker_line) if marker_line else 0
    if marker_line:
        text = text.replace(marker_line, "")
    had_image = bool(png_ref) and png_ref in text
    if png_ref:
        text = text.replace(png_ref, "")
    # The lane writes its own "[page image](...)" beside each withheld run, which is a
    # different spelling from the caller's ``png_ref``. Both have to go, or the page
    # carries one picture link per withheld run (4 on 2010-04-28 p15) on top of the
    # single image appended at the end.
    lane_images = re.findall(r"\[page image\]\([^)]*\)", text)
    for ref in set(lane_images):
        text = text.replace(ref, "")
    had_image = had_image or bool(lane_images)
    lines = [ln.rstrip() for ln in text.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        if not line.strip() and (not cleaned or not cleaned[-1].strip()):
            continue
        cleaned.append(line)
    # PAGE scope and the native-coverage check: this is not about a chart region, it
    # is about text of the page that did not ship. A region-scoped chart_region clause
    # here would impugn a region that is in fact represented.
    #
    # Round 4: a RECORD, not a rendered line. Astra's P2 was that the order-refusal line
    # existed only as a string; this line had the same gap and fires on all 12 real Fed
    # pages where that one fires on none, so fixing only the one that was named would
    # leave the commoner case unrecorded. The caller renders it with ``clause_mark`` and
    # hands the same record to the verdict.
    record = _page_coverage_record(WITHHELD_RUNS_REASON.format(runs=runs)) if runs else None
    return (
        "\n".join(cleaned).strip(),
        record,
        (png_ref if (had_image or runs) and png_ref else None),
    )
