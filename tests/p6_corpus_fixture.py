"""A multi-page assemble corpus, and a capture of every surface stage A/B must not move.

Imported by BOTH the current tree's difference test and the capture script that
runs it against the pre-change sources (``git show HEAD:...`` exported to a temp
tree). It must therefore use ONLY symbols that exist on both sides: no
``PageDisposition``, no ``SelectionProvenance``, no ``finalized_page_records``.

The corpus deliberately includes the two shapes the cold review used to prove the
D3 divergence -- D3 flags with a passing non-native winner, with and without a
qualifying refused grid -- alongside the ordinary D3 floor, D3 model-kept, clean
native prose, an ordinary passing model page, and a page nothing produced text for.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState

PAGE_COUNT = 7

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


def _d3_flags(ps) -> None:
    ps.is_born_digital = True
    ps.has_tables = True
    ps.native_text = NATIVE_COLLAPSED
    ps.native_table_structure_failed = True
    ps.native_table_unverifiable = True
    ps.d3_floor_png_ref = "![p](figures/p.png)"


def build_corpus_state(pdf_path: Path) -> DocumentState:
    """Seven pages, one ending each, built field by field."""
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))

    # 1 -- clean born-digital prose. No distrust flag anywhere.
    p1 = state.pages[1]
    p1.is_born_digital = True
    p1.native_text = CLEAN_NATIVE

    # 2 -- the ordinary D3 fail-closed floor: flags, an attempt, nothing that
    #      qualifies to supersede the marker.
    p2 = state.pages[2]
    _d3_flags(p2)
    p2.attempts.append(_native_attempt(2))

    # 3 -- D3 model-kept (#262): a refused grid qualifies and supersedes the floor.
    p3 = state.pages[3]
    _d3_flags(p3)
    p3.attempts.append(_native_attempt(3))
    p3.attempts.append(_refused_grid_attempt(3))

    # 4 -- COLD REVIEW SHAPE ONE: D3 flags plus a PASSING non-native winner.
    #      Selection returns the passing winner before the D3 branch; the
    #      pre-change bucket predicate still claimed the page for d3_floor_pages.
    p4 = state.pages[4]
    _d3_flags(p4)
    p4.attempts.append(_native_attempt(4))
    p4.attempts.append(_passing_model_output(4))
    p4.best_output = _passing_model_output(4)

    # 5 -- COLD REVIEW SHAPE TWO: the same, plus a qualifying refused grid, so
    #      the pre-change predicate claimed it for d3_model_table_pages instead.
    p5 = state.pages[5]
    _d3_flags(p5)
    p5.attempts.append(_native_attempt(5))
    p5.attempts.append(_refused_grid_attempt(5))
    p5.attempts.append(_passing_model_output(5))
    p5.best_output = _passing_model_output(5)

    # 6 -- a page nothing produced text for: explicit failure marker.
    state.pages[6].is_born_digital = False

    # 7 -- an ordinary passing model page.
    p7 = state.pages[7]
    p7.attempts.append(_passing_model_output(7, text="Plain model prose for page seven.\n"))
    p7.best_output = _passing_model_output(7, text="Plain model prose for page seven.\n")

    return state


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


def capture(tmp_dir: Path) -> dict:
    """Run `_phase_assemble` over the corpus and capture every surface that must not move."""
    from rich.console import Console

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

    md_files = sorted(output_dir.rglob("*.md"))
    sidecars = sorted(output_dir.rglob("pages/*.json"))
    manifests = sorted(output_dir.rglob("manifest.json"))

    def _scrub(text: str) -> str:
        """Absolute tmp paths are environmental, not behaviour: normalize them."""
        return text.replace(str(tmp_dir), "<TMP>").replace(str(tmp_dir.resolve()), "<TMP>")

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
    }
