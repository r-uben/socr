"""Content-free native self-binding coverage measurements.

The sweep mirrors :meth:`BornDigitalDetector.extract_structured`'s exclusive
region-discovery chain.  It keeps every strict region in ``regions`` and one
selected primary grid per bindable page in ``pages``.  Neither collection
contains PDF text, markdown, labels, coordinates, or rejection payloads.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import fitz

from socr.core.born_digital import BornDigitalDetector, _is_lane_stacked
from socr.core.pdf import open_pdf
from socr.tables.binding import BindingResult, Grid, bind, parse_grid


class NativeExtractionRegion(NamedTuple):
    """One native table region: its extent, its markdown, and what produced it.

    Every discovery stage returns ``(rect, markdown)`` pairs, so the candidate
    arrives paired with its own extent. This wrapper just names the pair for the
    sweep; it deliberately does not live in ``born_digital`` so the harness adds
    no surface to the extraction path it measures.
    """

    rect: Any
    content: str
    provenance: str


_RECONSTRUCT_LOGGER_NAME = "socr.tables.reconstruct"
_TABLE_KIND = "table"


@dataclass(frozen=True)
class ManifestPage:
    """One one-based page selected by the measurement manifest."""

    paper: str
    page: int
    pdf_name: str


@dataclass(frozen=True)
class CoverageRecord:
    """The content-free fields permitted in one coverage result record."""

    paper: str
    page: int
    source_stage: str
    region_ordinal: int
    selected_primary: bool
    grid_rows: int
    grid_columns: int
    fully_checked: bool
    structural_agreement: bool
    row_binding_unverifiable: bool
    column_binding_unverifiable: bool
    row_label_unverifiable: bool
    row_labels_checked: int
    ambiguous_count: int
    candidate_valueless_unbound: int
    native_valueless_unbound: int
    native_unbound_count: int
    model_unbound_count: int
    model_unbound_nonempty: bool
    cell_contradiction_count: int = 0
    row_label_contradiction_count: int = 0
    contradiction_count: int = 0

    def __post_init__(self) -> None:
        """Keep the old combined record count as a derived compatibility view."""
        split_count = self.cell_contradiction_count + self.row_label_contradiction_count
        if split_count or self.contradiction_count == 0:
            object.__setattr__(self, "contradiction_count", split_count)


@dataclass(frozen=True)
class CoverageReport:
    """A stable report containing region and selected page populations."""

    summary: dict[str, Any]
    regions: tuple[CoverageRecord, ...]
    pages: tuple[CoverageRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "regions": [asdict(record) for record in self.regions],
            "pages": [asdict(record) for record in self.pages],
        }

    def to_json(self) -> str:
        """Serialize with stable key ordering and a final newline."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def _page_record(paper: Any, page: Any, pdf_name: Any) -> ManifestPage:
    if not isinstance(paper, str) or not paper:
        raise ValueError("coverage manifest page is missing a paper identifier")
    if not isinstance(pdf_name, str) or not pdf_name:
        raise ValueError(f"coverage manifest page for {paper!r} is missing a PDF name")
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ValueError(f"coverage manifest page for {paper!r} is not a positive integer")
    return ManifestPage(paper=paper, page=page, pdf_name=pdf_name)


def _manifest_pages(data: Any) -> list[ManifestPage]:
    """Read the two manifest shapes used by the lane-comparison tests/corpus."""
    if not isinstance(data, list):
        raise ValueError("coverage manifest must contain a JSON list")

    pages: list[ManifestPage] = []
    for paper_entry in data:
        if not isinstance(paper_entry, dict):
            raise ValueError("coverage manifest entries must be JSON objects")

        # Synthetic/early measurement shape: one object per page.
        if "page" in paper_entry:
            if paper_entry.get("kind", _TABLE_KIND) != _TABLE_KIND:
                continue
            pages.append(
                _page_record(
                    paper_entry.get("paper") or paper_entry.get("name"),
                    paper_entry.get("page"),
                    paper_entry.get("file") or paper_entry.get("pdf"),
                )
            )
            continue

        # Committed lane-comparison shape: one object per paper.
        paper = paper_entry.get("name") or paper_entry.get("paper")
        pdf_name = paper_entry.get("pdf") or paper_entry.get("file")
        selected_pages = paper_entry.get("pages")
        if not isinstance(selected_pages, list):
            raise ValueError("coverage manifest paper entries need a pages list")
        for page_entry in selected_pages:
            if not isinstance(page_entry, dict):
                raise ValueError("coverage manifest page entries must be JSON objects")
            if page_entry.get("kind", _TABLE_KIND) != _TABLE_KIND:
                continue
            pages.append(_page_record(paper, page_entry.get("page"), pdf_name))

    return sorted(pages, key=lambda item: (item.paper, item.page, item.pdf_name))


def load_manifest(path: Path) -> list[ManifestPage]:
    """Load and validate a coverage manifest without opening any PDF."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read coverage manifest {path}: {exc}") from exc
    pages = _manifest_pages(data)
    if not pages:
        raise ValueError("coverage manifest contains no table pages")
    return pages


def resolve_pdf_paths(pages: list[ManifestPage], pdf_root: Path) -> dict[str, Path]:
    """Resolve and preflight every referenced PDF before PyMuPDF is called."""
    paths: dict[str, Path] = {}
    for page in pages:
        path = Path(page.pdf_name)
        if not path.is_absolute():
            path = pdf_root / path
        key = str(path)
        if key in paths:
            continue
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if not path.is_file():
            raise ValueError(f"PDF is not a file: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"PDF is 0-byte (empty): {path}")
        paths[key] = path
    return paths


@contextmanager
def suppress_reconstruct_logger() -> Iterator[None]:
    """Suppress only reconstruct diagnostics that may interpolate token text."""
    logger = logging.getLogger(_RECONSTRUCT_LOGGER_NAME)
    old_level = logger.level
    old_disabled = logger.disabled
    logger.disabled = True
    try:
        yield
    finally:
        logger.level = old_level
        logger.disabled = old_disabled


@contextmanager
def _suppress_pymupdf_layout_advisory() -> Iterator[None]:
    """Keep PyMuPDF's optional-layout advice out of machine-readable output."""
    variable = "PYMUPDF_SUGGEST_LAYOUT_ANALYZER"
    old_value = os.environ.get(variable)
    os.environ[variable] = "0"
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = old_value


def select_primary_grid(
    regions: list[tuple[int, NativeExtractionRegion, Grid]],
) -> tuple[int, NativeExtractionRegion, Grid] | None:
    """Select the dry-run primary grid deterministically.

    The dry run selected the largest strict pipe block by raw markdown length.
    A tie is resolved by the lowest extraction-region ordinal (top-to-bottom
    order), making selection independent of incidental list ordering.
    """
    if not regions:
        return None
    return min(regions, key=lambda item: (-len(item[1].content), item[0]))


def _is_chart_placeholder(content: str) -> bool:
    """Return whether *content* is the chart-aware rowizer's image marker."""
    return content.lstrip().startswith("![")


def _discover_native_regions(
    page: fitz.Page, detector: BornDigitalDetector
) -> list[NativeExtractionRegion] | None:
    """Mirror ``extract_structured``'s exclusive table discovery chain.

    ``None`` means the initial ``find_tables`` call failed.  That distinction
    matters: production returns plain text immediately for that failure, so
    the benchmark must not continue into either fallback stage.
    """
    try:
        tables_result = page.find_tables()
    except Exception:
        return None

    table_regions: list[NativeExtractionRegion] = []
    lane_stacked_regions: list[NativeExtractionRegion] = []
    from socr.tables.reconstruct import rowize_from_word_list

    for table in tables_result.tables:
        if _is_lane_stacked(table):
            bbox = fitz.Rect(table.bbox)
            region_words = [
                word
                for word in page.get_text("words")
                if bbox.contains(fitz.Point(word[0], word[1]))
            ]
            # GH-351: NO rotation kwargs. Production's lane-stacked path
            # (born_digital.extract_structured ~1953) calls
            # ``rowize_from_word_list(region_words)`` bare. The harness passed a
            # clip-scoped rotation and page_rect, so on a rotated lane-stacked
            # page the scoreboard described a DIFFERENT native candidate than
            # the one that ships.
            #
            # The _is_lane_stacked allowlist exists precisely so the instrument
            # cannot drift from the thing it measures; the call beside it had
            # drifted. rowize_from_words (the reconstruct fallback) applies a
            # page-wide rotation, which is a second policy -- the harness must
            # not invent a third.
            for rect, content in rowize_from_word_list(region_words) or []:
                lane_stacked_regions.append(NativeExtractionRegion(rect, content, "lane_stacked"))
        else:
            content = detector._table_to_markdown(table)
            if content:
                table_regions.append(
                    NativeExtractionRegion(fitz.Rect(table.bbox), content, "find_tables_lines")
                )

    # This is intentionally a fall-through, not a union: one successful stage
    # suppresses every later stage exactly as in extract_structured.
    table_regions.extend(lane_stacked_regions)
    if not table_regions:
        from socr.tables.reconstruct import reconstruct_table_regions

        table_regions = [
            NativeExtractionRegion(rect, content, "reconstruct_table_regions")
            for rect, content in (reconstruct_table_regions(page) or [])
        ]

    if not table_regions:
        from socr.tables.reconstruct import rowize_from_words_chart_aware

        page_num = getattr(page, "number", 0) + 1
        table_regions = [
            NativeExtractionRegion(rect, content, "rowize_chart_aware")
            for rect, content in (rowize_from_words_chart_aware(page, page_num=page_num) or [])
        ]

    # Production uses a stable top-to-bottom sort.  Python's stable sort keeps
    # same-y regions in the order emitted by the successful stage.
    table_regions.sort(key=lambda region: region.rect.y0)
    return table_regions


def _record(
    page: ManifestPage,
    region_ordinal: int,
    extraction: NativeExtractionRegion | None,
    grid: Grid | None,
    result: BindingResult,
    selected_primary: bool,
) -> CoverageRecord:
    return CoverageRecord(
        paper=page.paper,
        page=page.page,
        source_stage=extraction.provenance if extraction is not None else "none",
        region_ordinal=region_ordinal,
        selected_primary=selected_primary,
        grid_rows=len(grid.rows) if grid is not None else 0,
        grid_columns=grid.n_cols if grid is not None else 0,
        fully_checked=bool(result.fully_checked),
        structural_agreement=bool(result.structural_agreement),
        row_binding_unverifiable=bool(result.row_binding_unverifiable),
        column_binding_unverifiable=bool(result.column_binding_unverifiable),
        row_label_unverifiable=bool(result.row_label_unverifiable),
        row_labels_checked=result.row_labels_checked,
        ambiguous_count=int(result.ambiguous_count),
        candidate_valueless_unbound=result.candidate_valueless_unbound,
        native_valueless_unbound=result.native_valueless_unbound,
        contradiction_count=len(result.contradicted_cells) + len(result.row_label_contradictions),
        native_unbound_count=len(result.native_unbound),
        model_unbound_count=len(result.model_unbound),
        model_unbound_nonempty=bool(result.model_unbound),
        cell_contradiction_count=len(result.contradicted_cells),
        row_label_contradiction_count=len(result.row_label_contradictions),
    )


def _aggregate(records: tuple[CoverageRecord, ...], denominator: int) -> dict[str, int]:
    """Aggregate only counts; booleans are represented as passing-record counts."""
    return {
        "denominator": denominator,
        "fully_checked": sum(record.fully_checked for record in records),
        "structural_agreement": sum(record.structural_agreement for record in records),
        "row_binding_unverifiable": sum(record.row_binding_unverifiable for record in records),
        "column_binding_unverifiable": sum(
            record.column_binding_unverifiable for record in records
        ),
        "row_label_unverifiable": sum(record.row_label_unverifiable for record in records),
        "row_labels_checked_positive": sum(record.row_labels_checked > 0 for record in records),
        "ambiguity_nonempty": sum(record.ambiguous_count > 0 for record in records),
        "candidate_valueless_unbound": sum(
            record.candidate_valueless_unbound for record in records
        ),
        "native_valueless_unbound": sum(record.native_valueless_unbound for record in records),
        "cell_contradiction_nonempty": sum(
            record.cell_contradiction_count > 0 for record in records
        ),
        "row_label_contradiction_nonempty": sum(
            record.row_label_contradiction_count > 0 for record in records
        ),
        "contradiction_nonempty": sum(record.contradiction_count > 0 for record in records),
        "native_unbound_nonempty": sum(record.native_unbound_count > 0 for record in records),
        "model_unbound_nonempty": sum(record.model_unbound_nonempty for record in records),
    }


def measure_manifest(manifest: Path, pdf_root: Path) -> CoverageReport:
    """Run the deterministic native self-bind sweep for *manifest*."""
    pages = load_manifest(manifest)
    pdf_paths = resolve_pdf_paths(pages, pdf_root)
    detector = BornDigitalDetector()
    region_records: list[CoverageRecord] = []
    page_records: list[CoverageRecord] = []
    placeholder_region_count = 0

    with suppress_reconstruct_logger():
        documents: dict[str, fitz.Document] = {}
        try:
            for page_ref in pages:
                pdf_path = Path(page_ref.pdf_name)
                if not pdf_path.is_absolute():
                    pdf_path = pdf_root / pdf_path
                key = str(pdf_path)
                if key not in documents:
                    # GH-330 review: MUST be socr.core.pdf.open_pdf, not fitz.open.
                    # The helper applies symbol-font glyph recovery; reading raw
                    # means the harness measures different text than the native lane
                    # produces, so its coverage numbers would not describe the
                    # pipeline being fixed (tests/test_pdf_open.py enforces this).
                    documents[key] = open_pdf(pdf_paths[key])
                document = documents[key]
                if page_ref.page > document.page_count:
                    raise ValueError(f"PDF has no page {page_ref.page}: {pdf_path}")
                page = document[page_ref.page - 1]
                # PyMuPDF emits this one-time advisory from ``find_tables`` on
                # the first call. It is not a measurement result and would
                # make repeated CliRunner captures differ, so filter only this
                # known advisory while preserving unrelated warnings.
                with _suppress_pymupdf_layout_advisory(), warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Consider using the pymupdf_layout package.*",
                        category=UserWarning,
                    )
                    extracted = _discover_native_regions(page, detector)
                # ``None`` is the production early fallback for a failed
                # initial find_tables call.  It must not be mistaken for an
                # empty successful probe, which is what enables reconstruction.
                if extracted is None:
                    extracted = []
                strict_regions: list[tuple[int, NativeExtractionRegion, Grid]] = []
                placeholder_region_count += sum(
                    _is_chart_placeholder(region.content) for region in extracted
                )
                for ordinal, region in enumerate(extracted, start=1):
                    if _is_chart_placeholder(region.content):
                        continue
                    grid = parse_grid(region.content)
                    if grid is not None:
                        strict_regions.append((ordinal, region, grid))

                primary = select_primary_grid(strict_regions)
                for ordinal, region, grid in strict_regions:
                    result = bind(page.get_text("words"), region.content, region=tuple(region.rect))
                    region_records.append(
                        _record(
                            page_ref,
                            ordinal,
                            region,
                            grid,
                            result,
                            primary is not None and ordinal == primary[0],
                        )
                    )

                if primary is not None:
                    ordinal, region, grid = primary
                    result = bind(page.get_text("words"), region.content)
                    page_records.append(_record(page_ref, ordinal, region, grid, result, True))
                else:
                    # The committed dry run called bind with no selected pipe
                    # block on these pages. Preserve its default unverifiable
                    # page-level result explicitly, without pretending it is a
                    # strict grid or a successful zero-model-unbound case.
                    page_records.append(_record(page_ref, 0, None, None, BindingResult(), False))
        finally:
            for document in documents.values():
                document.close()

    regions = tuple(
        sorted(
            region_records, key=lambda record: (record.paper, record.page, record.region_ordinal)
        )
    )
    selected_pages = tuple(sorted(page_records, key=lambda record: (record.paper, record.page)))
    total_pages = len(pages)
    bindable_pages = sum(record.selected_primary for record in selected_pages)
    stage_pages: dict[str, set[tuple[str, int]]] = {}
    for record in regions:
        stage_pages.setdefault(record.source_stage, set()).add((record.paper, record.page))
    summary = {
        "total_pages": total_pages,
        "bindable_pages": bindable_pages,
        "strict_grids": len(regions),
        "placeholder_regions": placeholder_region_count,
        "no_grid_pages": total_pages - bindable_pages,
        "bindable_pages_by_stage": {
            stage: len(stage_pages[stage]) for stage in sorted(stage_pages)
        },
        "region_scoped": _aggregate(regions, len(regions)),
        "whole_page": _aggregate(selected_pages, total_pages),
    }
    return CoverageReport(summary=summary, regions=regions, pages=selected_pages)


def run_binding_coverage(manifest: Path, pdf_root: Path) -> CoverageReport:
    """Compatibility alias for callers that prefer an imperative name."""
    return measure_manifest(manifest, pdf_root)


def summary_text(report: CoverageReport) -> str:
    """Render the human format separately from JSON."""
    summary = report.summary
    region = summary["region_scoped"]
    whole = summary["whole_page"]
    lines = [
        f"Total pages: {summary['total_pages']}",
        f"Bindable pages: {summary['bindable_pages']}",
        f"Strict grids: {summary['strict_grids']}",
        f"Chart placeholders: {summary['placeholder_regions']}",
        f"No-grid pages: {summary['no_grid_pages']}",
        "Bindable pages by stage:",
        *[f"  {stage}: {count}" for stage, count in summary["bindable_pages_by_stage"].items()],
        "Region-scoped strict-grid results:",
        f"  fully checked: {region['fully_checked']}/{region['denominator']}",
        f"  structural agreement: {region['structural_agreement']}/{region['denominator']}",
        f"  row binding unverifiable: {region['row_binding_unverifiable']}",
        f"  column binding unverifiable: {region['column_binding_unverifiable']}",
        f"  ambiguity non-empty: {region['ambiguity_nonempty']}",
        f"  model unbound non-empty: {region['model_unbound_nonempty']}",
        f"  native unbound non-empty: {region['native_unbound_nonempty']}",
        f"  cell contradiction non-empty: {region['cell_contradiction_nonempty']}",
        f"  row-label contradiction non-empty: {region['row_label_contradiction_nonempty']}",
        "Selected whole-page results:",
        f"  fully checked: {whole['fully_checked']}/{whole['denominator']}",
        f"  structural agreement: {whole['structural_agreement']}/{whole['denominator']}",
        f"  model unbound non-empty: {whole['model_unbound_nonempty']}/{whole['denominator']}",
    ]
    return "\n".join(lines) + "\n"
