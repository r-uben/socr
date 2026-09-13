"""#747: the Fed's own published per-bin counts for the SEP dot-plot corpus.

Until this module existed, this repo operated on a recorded assumption that no
external ground truth exists for the SEP dot-plot pages (`~/Data/socr/sep-dotplots`).
That assumption was never checked and is false: every SEP release publishes an
accessible companion page at
``https://www.federalreserve.gov/monetarypolicy/fomcprojtabl<YYYYMMDD>.htm`` (one
release used ``fomcprojtable`` instead -- see ``_HTM_NAME_OVERRIDES``), and that page
carries, as an ordinary HTML table, **the same histogram the corpus PDF page draws**:
"Figure 3.E. Distribution of participants' judgments of the midpoint of the
appropriate target range for the federal funds rate...". Every corpus page in
`~/Data/socr/sep-dotplots/in/sep-<YYYYMMDD>-p09.pdf` is a render of exactly that
figure, at the same 0.25-point bin width the HTML table already uses.

**This is the one place this module's design departs from the brief that asked for
it, and the departure is a measured finding, not a guess.** The brief assumed the
Fed's public data was a list of raw per-participant values (e.g. ``0.625``, one dot)
requiring a many-to-one "fold onto socr's 0.25-wide bin" step before it could be
compared to a reader or model reading. That data DOES exist (Figure 2's individual
dots), but it is not what the corpus pages draw or what this ticket needs to check:
the corpus pages draw Figure 3.E, whose own published HTML table is *already*
binned at 0.25 points, with column headers that are already the socr bin labels
(``0.13 - 0.37`` etc, comparable via the SAME normaliser `chart_reconcile._bin_key`
already uses for the reader/model comparison -- no separate mapping rule is needed
here at all).

The brief's second assumption -- that a page's *own* release table cannot supply
its own prior-meeting comparison, and a second release must be fetched -- is also
false for every corpus page but one. Figure 3.E prints TWO columns per year panel,
literally labelled by projection month ("September projections", "December
projections", ...), and both are populated from the CURRENT release's own table:
December 2021's own page carries the September 2021 counts already. The one
exception measured is the corpus's own edge case: the September 2020 release
(the prior meeting for `sep-20201216-p09`) never published a Figure 3 series at
all -- 2020 Q3 stopped at Figure 2 -- so there is no standalone September 2020
Figure 3.E to fetch. December 2020's own page carries the September 2020 counts
regardless, which is what this module relies on; a caller does not need a second
release fetch for any corpus page, including the first one.
"""

from __future__ import annotations

import html as html_lib
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from socr.figures.chart_reconcile import _bin_key, _series_key

logger = logging.getLogger(__name__)

#: Base URL every release but one is served from.
_BASE_URL = "https://www.federalreserve.gov/monetarypolicy/fomcprojtabl{date}.htm"
#: federalreserve.gov returns a 403 to urllib's default User-Agent
#: (`Python-urllib/x.y`) on a cold cache -- measured on a fresh checkout with
#: no cache directory, see #747's decision log. An honest, identifying
#: User-Agent (not a browser impersonation) is what a real client sends.
_USER_AGENT = "socr-figures-ground-truth/1 (+https://github.com/r-uben/socr; issue #747)"
#: Releases whose accessible page is served from a differently spelled path.
#: Measured, not guessed: 2022-03-16 404s at ``fomcprojtabl`` and 200s at
#: ``fomcprojtable`` (with an extra "e") -- the Fed's own naming is inconsistent
#: and a caller trusting one spelling for every date silently drops this release.
_HTM_NAME_OVERRIDES = {
    "20220316": "https://www.federalreserve.gov/monetarypolicy/fomcprojtable{date}.htm"
}

#: The figure this module reads. Every corpus page in `sep-dotplots/in/` renders
#: this exact figure -- see the module docstring.
_FIGURE_HEADING = "Figure 3.E"
#: Where the figure's table ends: the next numbered figure, or (if none is found
#: within a generous window) a hard cap so a malformed page cannot make the
#: regex scan run away.
_NEXT_FIGURE_RE = re.compile(r"Figure\s+\d")
_MAX_CHUNK = 40_000

_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.S)
_THEAD_RE = re.compile(r"<thead[^>]*>(.*?)</thead>", re.S)
_TBODY_RE = re.compile(r"<tbody[^>]*>(.*?)</tbody>", re.S)
_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
#: One header/data cell, its opening tag's attributes captured separately from
#: its text so ``id``/``headers``/``class`` survive stripping the tag.
_CELL_RE = re.compile(r"<(t[dh])([^>]*)>(.*?)</t[dh]>", re.S)
_ATTR_RE = re.compile(r'(\w[\w-]*)="([^"]*)"')
_TAG_RE = re.compile(r"<[^>]+>")


class GroundTruthUnavailable(Exception):
    """No Figure 3.E table could be found or fetched for a release.

    Raised rather than returning an empty table, because an empty
    :class:`ReleaseTable` and "this release does not publish this figure" mean
    different things to a caller scoring corpus pages: the first is silently
    scoreless, and the second explains why (`sep-20200916` is the one release
    among 24 fetched for #747 that raises this, and its corpus is not this
    module's problem to solve).
    """


@dataclass(frozen=True)
class GroundTruthCell:
    """One (bin, series) count as the Fed's own table publishes it.

    ``count`` is ``0`` for a blank cell: Figure 3.E is a complete histogram over
    every printed bin, and a bin with no dot in it is drawn as an empty cell in
    the same table, not omitted from it. There is therefore no "cell absent"
    case for this ground truth the way there is for a reader or a model that
    only writes what it resolved.
    """

    bin_label: str
    count: int


@dataclass(frozen=True)
class GroundTruthPanel:
    """One year (or "Longer run") panel of Figure 3.E, both projection columns.

    ``columns`` is keyed by the column's own printed label ("September
    projections", "December projections", ...) -- the same literal label socr's
    reader names its series after (`chart_reader.read_dashed_series` /
    `read_solid_series`) and a model-authored grid's own header carries. Keying
    by that label, rather than by a "prior"/"current" role this module would
    have to infer, is what lets a caller match a reader series or a model
    column to ground truth by nothing but the label both sides already print.
    """

    year_label: str
    columns: dict[str, tuple[GroundTruthCell, ...]]

    def count_for(self, column_label: str, bin_label: str) -> int | None:
        """The published count for *bin_label* under *column_label*, or ``None``
        if this panel names no such column or bin.

        Matching is by the corpus's own normalised keys
        (`chart_reconcile._series_key` / `_bin_key`), the same normalisation the
        reader-vs-model reconciler already uses, so a caller comparing three-way
        (ground truth, reader, model) is comparing the same identity everywhere.
        """
        target_series = _series_key(column_label)
        for label, cells in self.columns.items():
            if _series_key(label) != target_series:
                continue
            target_bin = _bin_key(bin_label)
            for cell in cells:
                if _bin_key(cell.bin_label) == target_bin:
                    return cell.count
            return None
        return None


@dataclass(frozen=True)
class ReleaseTable:
    """Figure 3.E for one SEP release, every panel it draws."""

    release_date: str
    panels: dict[str, GroundTruthPanel]

    def panel(self, year_label: str) -> GroundTruthPanel | None:
        """The panel matching *year_label*, case- and space-insensitive.

        A reader's `PanelReading.label` is `"Longer run"`; some releases' own
        HTML prints `"Longer Run"`. Neither is a bin or series identity, so this
        is a plain casefold rather than `chart_reconcile`'s key functions.
        """
        target = " ".join(year_label.split()).casefold()
        for label, panel in self.panels.items():
            if " ".join(label.split()).casefold() == target:
                return panel
        return None


def parse_release_html(html: str, *, release_date: str = "") -> ReleaseTable:
    """Parse Figure 3.E's own table out of one release's accessible HTML page.

    Raises :class:`GroundTruthUnavailable` when the page carries no Figure 3.E
    at all -- measured on the corpus's own September 2020 release, which stops
    at Figure 2 (see the module docstring). Never guesses a partial table: a
    header row this parser cannot make sense of is the same failure as no
    heading found, because a caller trusting a malformed header would silently
    misname every column beneath it.
    """
    idx = html.find(_FIGURE_HEADING)
    if idx < 0:
        raise GroundTruthUnavailable(
            f"no {_FIGURE_HEADING!r} heading in this release's accessible page"
            + (f" ({release_date})" if release_date else "")
        )
    tail = html[idx + len(_FIGURE_HEADING) : idx + len(_FIGURE_HEADING) + _MAX_CHUNK]
    next_figure = _NEXT_FIGURE_RE.search(tail)
    end = idx + len(_FIGURE_HEADING) + (next_figure.start() if next_figure else _MAX_CHUNK)
    chunk = html[idx:end]

    table_match = _TABLE_RE.search(chunk)
    thead_match = table_match and _THEAD_RE.search(table_match.group(1))
    tbody_match = table_match and _TBODY_RE.search(table_match.group(1))
    if not (table_match and thead_match and tbody_match):
        raise GroundTruthUnavailable(
            f"{_FIGURE_HEADING!r} heading found but no <table><thead><tbody> follows it"
            + (f" ({release_date})" if release_date else "")
        )

    head_rows = [_parse_cells(m) for m in _TR_RE.findall(thead_match.group(1))]
    if len(head_rows) < 2:
        raise GroundTruthUnavailable(
            f"{_FIGURE_HEADING!r} table's <thead> carries {len(head_rows)} row(s), not "
            "the 2 (year labels, then projection labels) every corpus release has"
            + (f" ({release_date})" if release_date else "")
        )
    year_row, series_row = head_rows[0], head_rows[1]
    # Year labels: every <th> in the first header row EXCEPT the row-label
    # stub ("Percent Range", identified by carrying no ``headers`` attribute
    # itself and none of the OTHER cells naming it as their header either --
    # in practice its own ``id`` is simply not referenced anywhere).
    year_ids = {c.attrs["id"]: c.text for c in year_row if "id" in c.attrs}
    # Series columns, in table (left-to-right) order: each one's ``headers``
    # names the year id it belongs under. A year id this row never mentions
    # names no data column and is dropped rather than guessed at -- the
    # asymmetric case measured on every corpus September release, where one
    # year has only a single ("September projections") column because the
    # prior meeting did not project that far out.
    columns: list[tuple[str, str, str]] = []  # (series_id, year_label, series_label)
    for cell in series_row:
        year_id = cell.attrs.get("headers", "")
        year_label = year_ids.get(year_id)
        if year_label is None or "id" not in cell.attrs:
            continue
        columns.append((cell.attrs["id"], year_label, cell.text))
    if not columns:
        raise GroundTruthUnavailable(
            f"{_FIGURE_HEADING!r} table's series header row names no column whose "
            "own `headers` attribute resolves to a year label from the row above it"
            + (f" ({release_date})" if release_date else "")
        )
    panel_cells: dict[str, dict[str, list[GroundTruthCell]]] = {}
    for row_html in _TR_RE.findall(tbody_match.group(1)):
        cells = _parse_cells(row_html)
        if not cells:
            continue
        stub, data_cells = cells[0], cells[1:]
        bin_label = stub.text
        # Every row carries exactly one <td> per table-wide column, in the same
        # left-to-right order as `columns` -- an `emptystub` cell (no `headers`
        # attribute) is still a column-shaped placeholder, not a gap, so this
        # zip is positional rather than a `headers`-id lookup. That is what
        # lets a blank cell become an explicit ``count=0`` instead of "absent":
        # scoring a reader/model value against a bin the Fed shows nothing in
        # (a fabrication) requires knowing the Fed's own count there IS zero,
        # not that this parser never looked.
        if len(data_cells) != len(columns):
            raise GroundTruthUnavailable(
                f"{_FIGURE_HEADING!r} row {bin_label!r} carries {len(data_cells)} "
                f"cell(s), not the {len(columns)} column(s) the header row named"
                + (f" ({release_date})" if release_date else "")
            )
        for cell, (series_id, year_label, series_label) in zip(data_cells, columns):
            if cell.tag != "td":
                continue
            if cell.attrs.get("class") == "emptystub":
                count = 0
            else:
                header_ids = set(cell.attrs.get("headers", "").split())
                if series_id not in header_ids:
                    raise GroundTruthUnavailable(
                        f"{_FIGURE_HEADING!r} row {bin_label!r} column {series_label!r} "
                        f"cell's own headers={header_ids!r} does not name {series_id!r} "
                        "-- positional and attribute-driven column order disagree"
                        + (f" ({release_date})" if release_date else "")
                    )
                count = _as_int(cell.text)
                if count is None:
                    continue  # not a count (a dash, a footnote mark)
            by_series = panel_cells.setdefault(year_label, {})
            by_series.setdefault(series_label, []).append(GroundTruthCell(bin_label, count))

    panels = {
        year: GroundTruthPanel(
            year_label=year,
            columns={series: tuple(cells) for series, cells in by_series.items()},
        )
        for year, by_series in panel_cells.items()
    }
    return ReleaseTable(release_date=release_date, panels=panels)


@dataclass(frozen=True)
class _Cell:
    tag: str
    attrs: dict[str, str]
    text: str


def _parse_cells(row_html: str) -> list[_Cell]:
    out = []
    for tag, attr_text, inner in _CELL_RE.findall(row_html):
        attrs = dict(_ATTR_RE.findall(attr_text))
        text = html_lib.unescape(_TAG_RE.sub("", inner))
        text = " ".join(text.split())
        out.append(_Cell(tag=tag, attrs=attrs, text=text))
    return out


def _as_int(text: str) -> int | None:
    stripped = text.strip()
    if not stripped or stripped == "\xa0":
        return 0  # an emptystub's own text, when one slips through with a headers attribute
    return int(stripped) if re.fullmatch(r"-?\d+", stripped) else None


def _htm_url(release_date: str) -> str:
    template = _HTM_NAME_OVERRIDES.get(release_date, _BASE_URL)
    return template.format(date=release_date)


def fetch_release_html(release_date: str, *, timeout: float = 30.0) -> str:
    """Fetch one release's accessible page over the network. No caching here --
    see :func:`cached_release_table` for the cached, parsed entry point a
    scoring caller should actually use."""
    url = _htm_url(release_date)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        status = getattr(exc, "code", None)
        raise GroundTruthUnavailable(
            f"could not fetch {url} for release {release_date!r} ({exc}). "
            "If this is a network-restricted environment, supply a cache_dir "
            "whose raw/<release_date>.htm is already populated instead of "
            "fetching -- see cached_release_table()."
            + (f" HTTP status {status}." if status is not None else "")
        ) from exc


def cached_release_table(
    release_date: str, cache_dir: Path, *, timeout: float = 30.0
) -> ReleaseTable:
    """The parsed :class:`ReleaseTable` for *release_date*, fetching once.

    Raw HTML is cached at ``<cache_dir>/raw/<release_date>.htm`` so a rerun of
    the scoring harness never re-fetches the Fed's site. The cache holds the
    RAW page, not the parsed table: `parse_release_html` can change (a bug fix,
    a new field) without invalidating a byte a network call already paid for.
    """
    raw_path = cache_dir / "raw" / f"{release_date}.htm"
    if raw_path.exists():
        html = raw_path.read_text(encoding="utf-8", errors="replace")
    else:
        html = fetch_release_html(release_date, timeout=timeout)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(html, encoding="utf-8")
    return parse_release_html(html, release_date=release_date)
