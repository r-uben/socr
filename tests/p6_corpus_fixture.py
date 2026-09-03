"""A multi-page assemble corpus, and a capture of every surface stage A/B must not move.

Imported by both the current tree's difference tests and the pre-change baseline
capture tools. Because the baseline is stage A/B, PageDisposition, SelectionProvenance,
and finalized_page_records exist on both sides and may be imported lazily.

The corpus covers the 7 historical stage-A/B cases and the 5 stage-C cases:
  * Clean born-digital prose (1)
  * D3 fail-closed floor (2)
  * D3 model-kept (#262) (3)
  * Cold review shape one (4)
  * Cold review shape two (5)
  * Explicit failure marker for no text (6)
  * Ordinary passing model page (7)
  * Passing structure-class grid control (8)
  * Structure-class grid candidate rewritten by emission guard (9)
  * Genuine structure-class floor control (10)
  * Clean corrupt-math hybrid control (11)
  * Corrupt-math hybrid rewritten by emission guard (12)

It also captures ONE non-assemble shape, :func:`legacy_resume_capture`: a page
restored from a pre-stage-A sidecar that carries the older
``structure_class_model_kept`` flag but no ``disposition`` key. Cold review round 2
found a stage-C edit that silently moved that page's public disposition and bucket
membership while every corpus page stayed identical -- the 12-page assemble corpus
cannot reach the shape, because a live run never sets the flag. It is captured here
so the ordinary difference oracle covers it, on both sides, like anything else.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState

# ---------------------------------------------------------------------------
# Named page constants (no raw page numbers scattered through tests)
# ---------------------------------------------------------------------------

CLEAN_BORN_DIGITAL_PAGE = 1
D3_FLOOR_PAGE = 2
D3_MODEL_KEPT_PAGE = 3
COLD_REVIEW_SHAPE_ONE_PAGE = 4
COLD_REVIEW_SHAPE_TWO_PAGE = 5
NO_TEXT_FAILURE_PAGE = 6
PASSING_MODEL_PAGE = 7

STRUCT_MODEL_PASSING_PAGE = 8
STRUCT_MODEL_REWRITTEN_PAGE = 9
STRUCT_FLOOR_PAGE = 10
HYBRID_CLEAN_PAGE = 11
HYBRID_REWRITTEN_PAGE = 12

PAGE_LABELS: dict[int, str] = {
    CLEAN_BORN_DIGITAL_PAGE: "clean_born_digital",
    D3_FLOOR_PAGE: "d3_floor",
    D3_MODEL_KEPT_PAGE: "d3_model_kept",
    COLD_REVIEW_SHAPE_ONE_PAGE: "cold_review_shape_one",
    COLD_REVIEW_SHAPE_TWO_PAGE: "cold_review_shape_two",
    NO_TEXT_FAILURE_PAGE: "no_text_failure",
    PASSING_MODEL_PAGE: "passing_model",
    STRUCT_MODEL_PASSING_PAGE: "struct_model_passing",
    STRUCT_MODEL_REWRITTEN_PAGE: "struct_model_rewritten",
    STRUCT_FLOOR_PAGE: "struct_floor",
    HYBRID_CLEAN_PAGE: "hybrid_clean",
    HYBRID_REWRITTEN_PAGE: "hybrid_rewritten",
}

PAGE_COUNT: int = len(PAGE_LABELS)

STAGE_C_PAGE_LABELS: dict[int, str] = {
    STRUCT_MODEL_PASSING_PAGE: "struct_model_passing",
    STRUCT_MODEL_REWRITTEN_PAGE: "struct_model_rewritten",
    STRUCT_FLOOR_PAGE: "struct_floor",
    HYBRID_CLEAN_PAGE: "hybrid_clean",
    HYBRID_REWRITTEN_PAGE: "hybrid_rewritten",
}

STAGE_C_PAGE_COUNT: int = len(STAGE_C_PAGE_LABELS)

#: A model reading with a strict grid, prose and an equation.
GRID_PAGE = (
    "The regressions are reported below.\n\n"
    "$$y_{t+1} = \\alpha + \\beta x_t + \\varepsilon_{t+1}$$\n\n"
    "| | const. | slope | $R^2$ |\n"
    "|---|---|---|---|\n"
    "| one | -0.04 | 0.06 | 0.00 |\n"
    "| two | 1.96 | -0.06 | 0.23 |\n\n"
    "The estimates are reported to two decimals.\n"
)
NATIVE_COLLAPSED = "one | -0.04 0.06 0.00 two 1.96\n"
CLEAN_NATIVE = "Ordinary prose with no table on it at all.\n"

#: The provably-soft refusal disposition, written as a literal so this module
#: does not depend on a constant name (it is a literal in the GH-262 tests too).
SOFT_JUDGE_ONLY = "judge_only"

#: A structurally uniform markdown grid -- passes ``has_strict_table_grid`` and is
#: therefore SELECTED as a grid winner -- that embeds a live LaTeX table command
#: (``\\hline``) in a cell. That is invisible to the pre-selection shape check but
#: is exactly what ``table_emission_defect`` (``TABLE_EMISSION_LATEX_LEAK``) rejects,
#: so the post-selection emission guard rewrites the page to a failure marker AFTER
#: selection already chose it. This is the shape stage C's migration is about: the
#: selection tag still says "a grid was chosen", but the disposition that ships is
#: ``(FAIL_CLOSED_MARKER, INVALID_TABLE_EMISSION)``.
GUARD_REWRITTEN_GRID_TEXT = "| n | slope | R2 |\n|---|---|---|\n| 1 \\hline | 0.03 | 0.91 |"

#: A malformed markdown table (column-count mismatch between header/separator/body)
#: -- the same shape ``test_emission_guard_rewrites_and_demotions_in_bucket_derivation``
#: in ``tests/test_p6_disposition_buckets.py`` uses. The hybrid branch has no
#: pre-selection grid-shape gate (unlike the structure-class branch above), so this
#: simpler malformed shape is enough to trip the guard once selected.
MALFORMED_TABLE_TEXT = "| a | b |\n| --- |\n| 1 | 2 |"

STRUCTURE_NATIVE_TEXT = "0.03 0.91 0.44\nn slope R2\n"
STRUCTURE_GRID_TEXT = "| n | slope | R2 |\n|---|---|---|\n| 1 | 0.03 | 0.91 |"


def make_pdf(tmp_dir: Path) -> Path:
    import fitz

    path = tmp_dir / "p6_corpus.pdf"
    doc = fitz.open()
    for n in range(PAGE_COUNT):
        doc.new_page().insert_text((72, 72), f"corpus page {n + 1}")
    doc.save(str(path))
    doc.close()
    return path


def _native_attempt(page_num: int) -> PageOutput:
    return PageOutput(
        page_num=page_num,
        text=NATIVE_COLLAPSED,
        status=PageStatus.WARNING,
        engine="native",
        audit_passed=False,
    )


def _refused_grid_attempt(page_num: int) -> PageOutput:
    out = PageOutput(
        page_num=page_num,
        text=GRID_PAGE,
        status=PageStatus.SUCCESS,
        engine="gemini",
        audit_passed=False,
        failure_mode=FailureMode.NONE,
    )
    out.rejection_class = SOFT_JUDGE_ONLY
    return out


def _passing_model_output(page_num: int, text: str = GRID_PAGE) -> PageOutput:
    return PageOutput(
        page_num=page_num,
        text=text,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
        failure_mode=FailureMode.NONE,
    )


def _structure_native_attempt(page_num: int) -> PageOutput:
    return PageOutput(
        page_num=page_num,
        text=STRUCTURE_NATIVE_TEXT,
        status=PageStatus.SUCCESS,
        engine="native",
        audit_passed=True,
    )


def _d3_flags(ps) -> None:
    ps.is_born_digital = True
    ps.has_tables = True
    ps.native_text = NATIVE_COLLAPSED
    ps.native_table_structure_failed = True
    ps.native_table_unverifiable = True
    ps.d3_floor_png_ref = "![p](figures/p.png)"


def build_corpus_state(pdf_path: Path) -> DocumentState:
    """Twelve pages: the 7 stage-A/B cases plus 5 stage-C cases, built field by field."""
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))

    # 1 -- clean born-digital prose. No distrust flag anywhere.
    p1 = state.pages[CLEAN_BORN_DIGITAL_PAGE]
    p1.is_born_digital = True
    p1.native_text = CLEAN_NATIVE

    # 2 -- the ordinary D3 fail-closed floor: flags, an attempt, nothing that
    #      qualifies to supersede the marker.
    p2 = state.pages[D3_FLOOR_PAGE]
    _d3_flags(p2)
    p2.attempts.append(_native_attempt(D3_FLOOR_PAGE))

    # 3 -- D3 model-kept (#262): a refused grid qualifies and supersedes the floor.
    p3 = state.pages[D3_MODEL_KEPT_PAGE]
    _d3_flags(p3)
    p3.attempts.append(_native_attempt(D3_MODEL_KEPT_PAGE))
    p3.attempts.append(_refused_grid_attempt(D3_MODEL_KEPT_PAGE))

    # 4 -- COLD REVIEW SHAPE ONE: D3 flags plus a PASSING non-native winner.
    #      Selection returns the passing winner before the D3 branch; the
    #      pre-change bucket predicate still claimed the page for d3_floor_pages.
    p4 = state.pages[COLD_REVIEW_SHAPE_ONE_PAGE]
    _d3_flags(p4)
    p4.attempts.append(_native_attempt(COLD_REVIEW_SHAPE_ONE_PAGE))
    p4.attempts.append(_passing_model_output(COLD_REVIEW_SHAPE_ONE_PAGE))
    p4.best_output = _passing_model_output(COLD_REVIEW_SHAPE_ONE_PAGE)

    # 5 -- COLD REVIEW SHAPE TWO: the same, plus a qualifying refused grid, so
    #      the pre-change predicate claimed it for d3_model_table_pages instead.
    p5 = state.pages[COLD_REVIEW_SHAPE_TWO_PAGE]
    _d3_flags(p5)
    p5.attempts.append(_native_attempt(COLD_REVIEW_SHAPE_TWO_PAGE))
    p5.attempts.append(_refused_grid_attempt(COLD_REVIEW_SHAPE_TWO_PAGE))
    p5.attempts.append(_passing_model_output(COLD_REVIEW_SHAPE_TWO_PAGE))
    p5.best_output = _passing_model_output(COLD_REVIEW_SHAPE_TWO_PAGE)

    # 6 -- a page nothing produced text for: explicit failure marker.
    state.pages[NO_TEXT_FAILURE_PAGE].is_born_digital = False

    # 7 -- an ordinary passing model page.
    p7 = state.pages[PASSING_MODEL_PAGE]
    p7.attempts.append(
        _passing_model_output(PASSING_MODEL_PAGE, text="Plain model prose for page seven.\n")
    )
    p7.best_output = _passing_model_output(
        PASSING_MODEL_PAGE, text="Plain model prose for page seven.\n"
    )

    # 8 -- STRUCT_MODEL_PASSING_PAGE: clean structure-class grid winner.
    #      Finalizes as (MODEL_OUTPUT, STRUCTURE_CLASS).
    p8 = state.pages[STRUCT_MODEL_PASSING_PAGE]
    p8.is_born_digital = True
    p8.native_text = STRUCTURE_NATIVE_TEXT
    p8.has_tables = True
    nat8 = _structure_native_attempt(STRUCT_MODEL_PASSING_PAGE)
    grid8 = PageOutput(
        page_num=STRUCT_MODEL_PASSING_PAGE,
        text=STRUCTURE_GRID_TEXT,
        status=PageStatus.SUCCESS,
        engine="gemini",
        audit_passed=False,
    )
    grid8.rejection_class = "ambiguous_deferred"
    p8.attempts.extend([nat8, grid8])
    p8.best_output = nat8

    # 9 -- STRUCT_MODEL_REWRITTEN_PAGE: structure-class candidate rewritten by emission guard.
    #      Finalizes as (FAIL_CLOSED_MARKER, INVALID_TABLE_EMISSION) with selection
    #      provenance STRUCTURE_CLASS_GRID_FLAGGED.
    p9 = state.pages[STRUCT_MODEL_REWRITTEN_PAGE]
    p9.is_born_digital = True
    p9.native_text = STRUCTURE_NATIVE_TEXT
    p9.has_tables = True
    nat9 = _structure_native_attempt(STRUCT_MODEL_REWRITTEN_PAGE)
    grid9 = PageOutput(
        page_num=STRUCT_MODEL_REWRITTEN_PAGE,
        text=GUARD_REWRITTEN_GRID_TEXT,
        status=PageStatus.SUCCESS,
        engine="gemini",
        audit_passed=False,
    )
    grid9.rejection_class = "ambiguous_deferred"
    p9.attempts.extend([nat9, grid9])
    p9.best_output = nat9

    # 10 -- STRUCT_FLOOR_PAGE: genuine structure-class floor (no attempt authors a grid).
    #       Finalizes as (FAIL_CLOSED_MARKER, STRUCTURE_CLASS) and remains in structure_class_floor_pages.
    p10 = state.pages[STRUCT_FLOOR_PAGE]
    p10.is_born_digital = True
    p10.native_text = STRUCTURE_NATIVE_TEXT
    p10.has_tables = True
    nat10 = _structure_native_attempt(STRUCT_FLOOR_PAGE)
    model10 = PageOutput(
        page_num=STRUCT_FLOOR_PAGE,
        text="prose with no markdown table",
        status=PageStatus.ERROR,
        engine="qwen",
        audit_passed=False,
        failure_mode=FailureMode.AUDIT_FAILED,
    )
    p10.attempts.extend([nat10, model10])
    p10.best_output = nat10

    # 11 -- HYBRID_CLEAN_PAGE: clean corrupt-math hybrid control.
    #       Finalizes as (MODEL_OUTPUT, CORRUPT_MATH_HYBRID).
    p11 = state.pages[HYBRID_CLEAN_PAGE]
    p11.is_born_digital = True
    p11.native_text = "native prose with a corrupt equation"
    hybrid11 = PageOutput(
        page_num=HYBRID_CLEAN_PAGE,
        text="native prose plus crop-backed region candidate",
        status=PageStatus.WARNING,
        engine="native+math",
        audit_passed=False,
        failure_mode=FailureMode.AUDIT_FAILED,
    )
    p11.attempts.append(hybrid11)
    p11.corrupt_math_hybrid = hybrid11

    # 12 -- HYBRID_REWRITTEN_PAGE: corrupt-math hybrid rewritten by emission guard.
    #       Finalizes as (FAIL_CLOSED_MARKER, INVALID_TABLE_EMISSION) with selection
    #       provenance CORRUPT_MATH_HYBRID.
    p12 = state.pages[HYBRID_REWRITTEN_PAGE]
    p12.is_born_digital = True
    p12.native_text = "native prose with a corrupt equation"
    hybrid12 = PageOutput(
        page_num=HYBRID_REWRITTEN_PAGE,
        text=MALFORMED_TABLE_TEXT,
        status=PageStatus.WARNING,
        engine="native+math",
        audit_passed=False,
        failure_mode=FailureMode.AUDIT_FAILED,
    )
    p12.attempts.append(hybrid12)
    p12.corrupt_math_hybrid = hybrid12

    return state


build_stage_c_corpus_state = build_corpus_state
make_stage_c_pdf = make_pdf


def _events(state) -> list[list]:
    """Page, kind, ENGINE and detail.

    ``engine`` is named by the acceptance bar and was missing from an earlier
    capture (cold review round 2, finding 7). Page-local engines happen to be
    repeated in the page sidecars for this corpus, but that covers neither
    document-level (page 0) events nor the direct event assertion.
    """
    return sorted(
        [
            [
                getattr(e, "page_num", 0),
                getattr(e, "kind", ""),
                getattr(e, "engine", ""),
                getattr(e, "detail", ""),
            ]
            for e in state.events
        ]
    )


# ---------------------------------------------------------------------------
# The legacy-resume shape (cold review round 2)
# ---------------------------------------------------------------------------

#: A model reading whose grid keeps each row label paired with its own numbers.
LEGACY_RESUME_MODEL_GRID = (
    "Table 2. Yield regressions by maturity\n\n"
    "| $n$ | const. | slope | $R^2$ |\n"
    "|---|---|---|---|\n"
    "| 2 | 0.03 | 0.91 | 0.44 |\n"
    "| 5 | 0.07 | 0.85 | 0.51 |\n"
)

#: The same numbers as a flattened native layer: the row labels have fallen out
#: of the grid. The numeric multiset is identical, which is the measured defect.
LEGACY_RESUME_NATIVE_FLATTENED = "0.03 0.91 0.44 0.07 0.85 0.51\nn const. slope R2\n2 5\n"


def _legacy_resume_pdf(tmp_dir: Path) -> Path:
    import fitz

    path = tmp_dir / "legacy_resume.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Table 2. Yield regressions by maturity")
    doc.save(str(path))
    doc.close()
    return path


def legacy_resume_capture(tmp_dir: Path) -> dict:
    """Resume a pre-stage-A structure-class sidecar and capture what is REPORTED.

    Run 1 is a clean-pass case-(i) page: native is ``best_output``, but a model
    attempt authored the grid outright, so the model's bytes ship. Its real
    sidecar therefore carries ``structure_class_model_kept: true``. The sidecar is
    then stripped of its ``disposition`` key, which is what a sidecar written
    before stage A looks like, and run 2 resumes from it.

    Resume collapses ``attempts`` to the single frozen winner, so selection can no
    longer see that a model grid was kept over native: it returns
    ``PASSING_BEST_OUTPUT``, and the page is reported as an ordinary passing model
    page. The shipped BYTES are the model grid either way -- only the reporting is
    at stake -- and stage C must not move the reporting, because this page was
    never guard-rewritten.
    """
    from socr.core.manifest import finalized_page_records
    from socr.pipeline import orchestrator as orch

    pdf_path = _legacy_resume_pdf(tmp_dir)
    output_dir = tmp_dir / "legacy_resume_out"
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = orch.UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            agentic=False,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            save_figures=False,
            write_manifest=False,
            # docs/log/2026-09-03_p1-prep-latch-and-audit.md: this capture is part
            # of the P6 byte-identity fixture, so the flag is pinned rather than
            # defaulted -- a future default flip must not move it silently.
            table_judge_ladder=False,
        )
    )

    def _fresh() -> DocumentState:
        state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
        page = state.pages[1]
        page.is_born_digital = True
        page.has_tables = True
        page.native_text = LEGACY_RESUME_NATIVE_FLATTENED
        return state

    native_attempt = PageOutput(
        page_num=1,
        text=LEGACY_RESUME_NATIVE_FLATTENED,
        status=PageStatus.SUCCESS,
        engine="native",
        audit_passed=True,
    )
    model_attempt = PageOutput(
        page_num=1,
        text=LEGACY_RESUME_MODEL_GRID,
        status=PageStatus.SUCCESS,
        engine="gemini",
        audit_passed=True,
    )

    live_state = _fresh()
    live_page = live_state.pages[1]
    live_page.attempts.extend([native_attempt, model_attempt])
    live_page.best_output = native_attempt
    pipeline._flush_page_sidecar(live_state, 1, output_dir)

    sidecar_path = next(output_dir.rglob("pages/00001.json"))
    sidecar = json.loads(sidecar_path.read_text())
    sidecar.pop("disposition", None)
    sidecar_path.write_text(json.dumps(sidecar))

    resumed_state = _fresh()
    resumed_page = resumed_state.pages[1]
    pipeline._restore_terminal_page_state(resumed_state, 1, model_attempt, output_dir)

    records = finalized_page_records(resumed_state)
    derive_fn = getattr(
        orch._derive_disposition_buckets, "__wrapped__", orch._derive_disposition_buckets
    )
    buckets = derive_fn(resumed_state, records)
    record = records[0]

    return {
        # The fixture is only meaningful if the legacy flag actually came back and
        # no disposition did. Captured so a broken fixture shows as a difference
        # rather than as a silently trivial comparison.
        "flag_restored": bool(getattr(resumed_page, "structure_class_model_kept_on_resume", False)),
        "resumed_disposition": getattr(resumed_page, "resumed_disposition", None),
        "disposition": [
            _enum_value(record.disposition.ending),
            _enum_value(record.disposition.primary_reason),
        ],
        "selection_provenance": _enum_value(record.selection_provenance),
        "winning_output": {
            "engine": record.output.engine,
            "status": _enum_value(record.output.status),
            "audit_passed": bool(record.output.audit_passed),
            "text_is_the_model_grid": record.output.text == LEGACY_RESUME_MODEL_GRID,
        },
        "buckets": {name: sorted(pages) for name, pages in sorted(buckets.items())},
    }


def _enum_value(value) -> str:
    return value.value if hasattr(value, "value") else str(value)


def capture(tmp_dir: Path) -> dict:
    """Run `_phase_assemble` over the corpus and capture every surface that must not move."""
    from rich.console import Console

    from socr.core.manifest import finalized_page_records, is_page_failed_marker
    from socr.pipeline import orchestrator as orch

    pdf_path = make_pdf(tmp_dir)
    state = build_corpus_state(pdf_path)

    output_dir = tmp_dir / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        primary_engine=EngineType.QWEN,
        agentic=False,
        enabled_engines=[EngineType.QWEN],
        quiet=False,
        save_figures=False,
        write_manifest=True,
        # docs/log/2026-09-03_p1-prep-latch-and-audit.md (cold review round 1):
        # this corpus is the P6 assemble byte-identity fixture; the flag MOVES it
        # (events, CLI, sidecars, result_error), so it is pinned, not defaulted.
        table_judge_ladder=False,
    )
    pipeline = orch.UnifiedPipeline(config)

    buf = io.StringIO()
    real_console = orch.console
    # Width is deliberately huge: rich wraps on the ABSOLUTE output path, whose
    # length is environmental, so a narrow console would make the captured CLI
    # depend on where the temp dir happened to land.
    orch.console = Console(file=buf, force_terminal=False, width=100_000, no_color=True)
    try:
        result = pipeline._phase_assemble(state, output_dir)
    finally:
        orch.console = real_console

    def _scrub(text: str) -> str:
        """Absolute tmp paths are environmental, not behaviour: normalize them."""
        return text.replace(str(tmp_dir), "<TMP>").replace(str(tmp_dir.resolve()), "<TMP>")

    md_files = sorted(output_dir.rglob("*.md"))
    sidecars = sorted(output_dir.rglob("pages/*.json"))
    manifests = sorted(output_dir.rglob("manifest.json"))
    audit_logs = sorted(output_dir.rglob("audit_log.json"))
    tables_trust_files = sorted(output_dir.rglob("tables_trust.json"))
    metadata_files = sorted(output_dir.rglob("*metadata.json"))

    records = finalized_page_records(state)
    page_contract = []
    for r in records:
        num = r.output.page_num
        label = PAGE_LABELS.get(num, f"page_{num}")
        ending = (
            r.disposition.ending.value
            if hasattr(r.disposition.ending, "value")
            else str(r.disposition.ending)
        )
        primary_reason = (
            r.disposition.primary_reason.value
            if hasattr(r.disposition.primary_reason, "value")
            else str(r.disposition.primary_reason)
        )
        prov = (
            r.selection_provenance.value
            if hasattr(r.selection_provenance, "value")
            else str(r.selection_provenance)
        )
        is_marker = is_page_failed_marker(r.output.text or "")
        status_str = (
            r.output.status.value if hasattr(r.output.status, "value") else str(r.output.status)
        )
        failure_mode_str = (
            r.output.failure_mode.value
            if hasattr(r.output.failure_mode, "value")
            else str(r.output.failure_mode)
        )
        page_contract.append(
            {
                "page_label": label,
                "page_num": num,
                "disposition": [ending, primary_reason],
                "selection_provenance": prov,
                "is_failure_marker": is_marker,
                "winning_output": {
                    "engine": r.output.engine,
                    "status": status_str,
                    "audit_passed": bool(r.output.audit_passed),
                    "failure_mode": failure_mode_str,
                    "text": _scrub(r.output.text or ""),
                },
            }
        )

    # Call unwrapped derivation if wrapped by conftest autouse guard, so capture() itself
    # does not add an extra entry to GUARD_CALL_LOG.
    derive_fn = getattr(
        orch._derive_disposition_buckets, "__wrapped__", orch._derive_disposition_buckets
    )
    raw_buckets = derive_fn(state, records)
    buckets = {k: sorted(list(v)) for k, v in sorted(raw_buckets.items())}

    metadata = {}
    for f in metadata_files:
        rel = f.relative_to(output_dir).as_posix()
        raw_meta = json.loads(_scrub(f.read_text()))
        if isinstance(raw_meta, dict):
            item = {
                k: _scrub(v) if isinstance(v, str) else v
                for k, v in raw_meta.items()
                if k in ("status", "error", "pages", "key", "version", "model", "backend")
            }
            if "files" in raw_meta and isinstance(raw_meta["files"], dict):
                item["files"] = {
                    fname: {
                        k: _scrub(v) if isinstance(v, str) else v
                        for k, v in finfo.items()
                        if k in ("status", "error", "pages", "model", "backend")
                    }
                    for fname, finfo in raw_meta["files"].items()
                    if isinstance(finfo, dict)
                }
            metadata[rel] = item

    return {
        "doc_status": str(state.status),
        "result_status": str(result.status),
        "result_error": _scrub(result.error or ""),
        "result_audit_passed": bool(result.audit_passed),
        "events": [[n, k, eng, _scrub(d)] for n, k, eng, d in _events(state)],
        "cli": _scrub(buf.getvalue()),
        "markdown": {f.relative_to(output_dir).as_posix(): _scrub(f.read_text()) for f in md_files},
        "sidecars": {
            f.relative_to(output_dir).as_posix(): json.loads(_scrub(f.read_text()))
            for f in sidecars
        },
        "manifest": [json.loads(_scrub(f.read_text())) for f in manifests],
        "audit_log": [json.loads(_scrub(f.read_text())) for f in audit_logs],
        "tables_trust": [json.loads(_scrub(f.read_text())) for f in tables_trust_files],
        "metadata": metadata,
        "page_contract": page_contract,
        "buckets": buckets,
        "legacy_resume": legacy_resume_capture(tmp_dir),
    }
