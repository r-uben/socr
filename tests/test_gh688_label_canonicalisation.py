"""#688: the decoded label must reach the shipped ``.md``, not just the binder.

#624a taught ``binding.parse_grid`` to decode entities and strip leading
whitespace from a row-label cell, so ``bind()`` and ``resolve_cell_refs`` both
saw ``Swiss francs``. The shipped bytes did not change. The reviewer's
unmocked reproduction: a real bind of ``&nbsp;&nbsp;Swiss francs`` reports
``candidate_row_labels == ('Swiss francs',)`` while ``finalized_page_records``
plus ``_phase_assemble`` still write ``| &nbsp;&nbsp;Swiss francs | 600.0 |``.

Astra's ruling: one shared canonicalisation boundary at the point a proposed
``PageOutput.text`` becomes the candidate used for judging and selection --
``tables.label_canonical.canonicalize_candidate`` -- crossed by extraction
(``route_page``), header repair, crop reconciliation and the escalation lane,
and NOT by the final selector.

Hermetic by construction: ``route_page`` is a pure function with the provider
and the judge injected, and the finalization/assembly tests drive
``finalized_page_records`` / ``_phase_assemble`` on hand-built page state. No
ollama, no provider ladder, no real tesseract; the resume tests pin a
DIFFERENCE (canonical fragment versus legacy fragment, same setup otherwise),
never an absolute outcome.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.judge.table_verdict import resolve_cell_refs
from socr.pipeline import orchestrator
from socr.pipeline.agentic import AcceptDecision, route_page
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.binding import bind, parse_grid
from socr.tables.label_canonical import (
    canonicalize_candidate,
    canonicalize_label_cell,
    canonicalize_table_labels,
    decode_label_cell,
)
from socr.tables.reconcile import find_table_blocks, markdown_table_identity

fitz = pytest.importorskip("fitz")

#: The reviewer's fixture, verbatim in shape: entity indentation on the label,
#: a value the transform must not touch.
RAW_PAGE = (
    "Reserves held at the end of the year.\n"
    "\n"
    "| Item | Amount |\n"
    "| --- | --- |\n"
    "| &nbsp;&nbsp;Swiss francs | 600.0 |\n"
    "| &nbsp;&nbsp;Pounds sterling | 1,204.5 |\n"
    "| Total | 1,804.5 |\n"
)
CANONICAL_ROW = "| Swiss francs | 600.0 |"


def _pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
    path = tmp_path / name
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Reserves held at the end of the year.")
    doc.save(str(path))
    doc.close()
    return path


def _pipeline() -> UnifiedPipeline:
    return UnifiedPipeline(
        PipelineConfig(
            agentic=True,
            judge_backend="heuristic",
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            save_figures=False,
            write_manifest=False,
            dual_pass_tables=False,
            detect_equations=False,
            recover_clean_equations=False,
            table_judge_ladder=False,
        )
    )


def _state_with(pdf_path: Path, text: str) -> DocumentState:
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps = state.pages[1]
    ps.is_born_digital = False
    ps.native_text = ""
    output = PageOutput(
        page_num=1,
        text=text,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    ps.attempts.append(output)
    ps.best_output = output
    return state


class _RecordingJudge:
    """Accepts, and remembers exactly what it was asked to judge."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def assess(self, output, provider) -> AcceptDecision:
        self.seen.append(output.text)
        return AcceptDecision(accept=True, reason="ok")


# ---------------------------------------------------------------------------
# The reproduction.
# ---------------------------------------------------------------------------


def test_gh688_shipped_markdown_carries_the_plain_label(tmp_path: Path) -> None:
    """The whole reviewer reproduction, end to end: what ``_phase_assemble``
    writes is the plain label, not the entity run."""
    judge = _RecordingJudge()
    decision = route_page(
        1,
        [PROFILE_QWEN_LOCAL],
        lambda prof, page: PageOutput(
            page_num=page, text=RAW_PAGE, status=PageStatus.SUCCESS, engine=prof.engine.value
        ),
        judge,
    )
    assert decision.accepted is True

    # The boundary precedes JUDGING, not just shipping.
    assert judge.seen == [decision.final_output.text]
    assert "&nbsp;" not in judge.seen[0], judge.seen[0]

    state = _state_with(_pdf(tmp_path), decision.final_output.text)
    result = _pipeline()._phase_assemble(state, tmp_path / "out")

    assert CANONICAL_ROW in result.markdown, result.markdown
    assert "&nbsp;" not in result.markdown, result.markdown
    # The value cell and the untouched label are byte-identical.
    assert "| Total | 1,804.5 |" in result.markdown, result.markdown

    # Falsification: the SAME assembly, differing only in whether the candidate
    # crossed the boundary, still ships the entity run. Without this the test
    # could pass on a fixture that never reproduced the bug.
    unbounded = _pipeline()._phase_assemble(
        _state_with(_pdf(tmp_path, "raw.pdf"), RAW_PAGE), tmp_path / "out_raw"
    )
    assert "| &nbsp;&nbsp;Swiss francs | 600.0 |" in unbounded.markdown, unbounded.markdown


def test_gh688_binder_and_shipped_bytes_now_agree(tmp_path: Path) -> None:
    """The two halves of the reviewer's finding meet: the binder's label and
    the corpus text's label are the same string."""
    state = _state_with(_pdf(tmp_path), canonicalize_table_labels(RAW_PAGE)[0])
    markdown = _pipeline()._phase_assemble(state, tmp_path / "out").markdown

    words = [
        _word(100, 60, 200, 70, "Swiss"),
        _word(205, 60, 260, 70, "francs"),
        _word(300, 60, 340, 70, "600.0"),
    ]
    block = "| Item | Amount |\n| --- | --- |\n| Swiss francs | 600.0 |\n"
    assert block.splitlines()[2] in markdown, markdown
    result = bind(words, block)
    assert result.candidate_row_labels == ("Swiss francs",)
    assert result.row_label_contradictions == []
    assert result.candidate_row_labels[0] in markdown


def _word(x0: float, y0: float, x1: float, y1: float, text: str) -> tuple:
    """One ``page.get_text("words")`` tuple, the shape ``bind()`` consumes."""
    return (x0, y0, x1, y1, text, 0, 0, 0)


# ---------------------------------------------------------------------------
# The transform's own contract.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "markdown",
    [
        RAW_PAGE,
        "| Item | A |\n| --- | --- |\n| &nbsp;X | 1 |\n",
        "| Item | A |\n| --- | --- |\n|   X | 1 |\n",
        "| Item | A |\n| --- | --- |\n| A&amp;B | 1 |\n",
        "| Item | A |\n| --- | --- |\n| A&#124;B | 1 |\n",
        "| Item | A |\n| --- | --- |\n|  padded  | 1 |\n",
        "| Item | A |\n| --- | --- |\n| &nbsp; | |\n| X | 1 |\n",
        "no table here, just a | pipe in prose\n",
        "",
    ],
)
def test_gh688_a_second_normalisation_changes_nothing(markdown: str) -> None:
    """Idempotence, the invariant fresh-versus-resume agreement rests on."""
    once, _ = canonicalize_table_labels(markdown)
    twice, changed = canonicalize_table_labels(once)
    assert changed == 0
    assert twice == once


def test_gh688_only_label_cells_change() -> None:
    """Delimiters, row order, row count, column count, header cells and every
    non-label cell are byte-identical; only column 0 of a body row moves."""
    canonical, changed = canonicalize_table_labels(RAW_PAGE)
    assert changed == 2

    raw_blocks = find_table_blocks(RAW_PAGE)
    new_blocks = find_table_blocks(canonical)
    assert len(raw_blocks) == len(new_blocks) == 1
    raw_grid, new_grid = raw_blocks[0].grid, new_blocks[0].grid
    assert len(raw_grid) == len(new_grid)
    for raw_row, new_row in zip(raw_grid, new_grid, strict=True):
        assert len(raw_row) == len(new_row)
        assert raw_row[1:] == new_row[1:]
    # Header row untouched, exactly as ``parse_grid`` leaves it.
    assert raw_grid[0] == new_grid[0]
    # Line count and every non-table line untouched.
    assert RAW_PAGE.count("\n") == canonical.count("\n")
    assert canonical.splitlines()[0] == RAW_PAGE.splitlines()[0]


def test_gh688_decoding_cannot_manufacture_a_cell_or_a_row() -> None:
    """A decoded pipe or newline is re-encoded, so the column count cannot
    move -- and ``decode_label_cell`` still resolves it to the real
    character, so the binder reads what the model meant."""
    markdown = "| Item | A |\n| --- | --- |\n| &#124;split&#10;here | 1 |\n"
    canonical, changed = canonicalize_table_labels(markdown)
    assert changed == 0, canonical  # already canonical: nothing to gain
    grid = parse_grid(canonical)
    assert grid is not None
    assert len(grid.rows) == 1
    assert len(grid.rows[0]) == 2
    assert grid.rows[0][0] == "|split\nhere"

    # And the entity NAME form collapses onto the numeric one, once.
    named = "| Item | A |\n| --- | --- |\n| &vert;split | 1 |\n"
    once, _ = canonicalize_table_labels(named)
    assert once == "| Item | A |\n| --- | --- |\n| &#124;split | 1 |\n"
    assert canonicalize_table_labels(once)[1] == 0
    assert len(parse_grid(once).rows[0]) == 2


def test_gh688_canonical_cell_decodes_to_the_same_label_as_the_raw_cell() -> None:
    """The boundary is label-preserving in the binder's own vocabulary."""
    for raw in ("&nbsp;&nbsp;Swiss francs", " X", "A&amp;B", "&#124;x", "&vert;x", "plain"):
        assert decode_label_cell(canonicalize_label_cell(raw)) == decode_label_cell(raw)


def test_gh688_every_cell_ref_resolves_to_the_same_physical_cell() -> None:
    """No physical coordinate moves: every ``RnCm`` the judge can name
    resolves identically before and after the rewrite."""
    canonical, _ = canonicalize_table_labels(RAW_PAGE)
    grid = parse_grid(RAW_PAGE)
    assert grid is not None
    refs = [f"R{r}C{c}" for r in range(1, len(grid.rows) + 1) for c in range(1, grid.n_cols + 1)]
    before = resolve_cell_refs(RAW_PAGE, refs)
    after = resolve_cell_refs(canonical, refs)
    assert before is not None and after is not None
    assert before == after
    # refs[0] is R1C1 -- the label the reproduction is about.
    assert after[next(iter(after))] == "Swiss francs"


def test_gh688_table_identity_is_computed_over_canonical_bytes() -> None:
    """The identity the caching/adjudication layer keys on is the canonical
    one -- which is why a verdict keyed to pre-rewrite bytes must never be
    carried forward (see the resume tests below)."""
    canonical, _ = canonicalize_table_labels(RAW_PAGE)
    assert markdown_table_identity(RAW_PAGE) != markdown_table_identity(canonical)
    assert markdown_table_identity(canonical) == markdown_table_identity(
        canonicalize_table_labels(canonical)[0]
    )


# ---------------------------------------------------------------------------
# The boundary's call sites.
# ---------------------------------------------------------------------------


def test_gh688_a_replacement_candidate_crosses_the_boundary_again() -> None:
    """``canonicalize_candidate`` is the shared entry point every site uses;
    it is in-place, reports how many label cells moved, and is free to call
    twice."""
    output = PageOutput(page_num=1, text=RAW_PAGE)
    assert canonicalize_candidate(output) == 2
    first = output.text
    assert canonicalize_candidate(output) == 0
    assert output.text == first

    # A later text replacement is a NEW candidate, and crossing again fixes it.
    output.text = RAW_PAGE
    assert canonicalize_candidate(output) == 2
    assert output.text == first


def test_gh688_the_native_lane_crosses_the_boundary_in_the_page_loop(tmp_path: Path) -> None:
    """A page that reaches the per-page lifecycle by a door other than
    ``route_page`` is still canonical before anything persists it."""
    pipeline = _pipeline()
    state = _state_with(_pdf(tmp_path), RAW_PAGE)
    out_dir = tmp_path / "out"
    # The lifecycle backstop, exercised through the same helper the loop calls.
    canonicalize_candidate(state.pages[1].best_output)
    pipeline._flush_page_fragment(state, 1, state.pages[1].best_output.text, out_dir)
    fragment = next(out_dir.rglob("pages/00001.md")).read_text()
    assert CANONICAL_ROW in fragment
    assert "&nbsp;" not in fragment


# ---------------------------------------------------------------------------
# Resume.
# ---------------------------------------------------------------------------


def _flush_terminal(pipeline: UnifiedPipeline, tmp_path: Path, body: str, tag: str):
    pdf_path = _pdf(tmp_path, f"{tag}.pdf")
    state = _state_with(pdf_path, body)
    out_dir = tmp_path / f"out_{tag}"
    pipeline._flush_page_fragment(state, 1, body, out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)
    sidecar = next(out_dir.rglob("pages/00001.json"))
    return pdf_path, out_dir, sidecar


def test_gh688_fresh_and_resume_agree_byte_for_byte(tmp_path: Path) -> None:
    """The resume invariant, pinned as a DIFFERENCE between two runs in one
    process that change only whether the page came from the ledger."""
    pipeline = _pipeline()
    canonical, _ = canonicalize_table_labels(RAW_PAGE)

    pdf_path, out_dir, sidecar = _flush_terminal(pipeline, tmp_path, canonical, "fresh")
    fresh_state = _state_with(pdf_path, canonical)
    fresh_md = pipeline._phase_assemble(fresh_state, tmp_path / "assemble_fresh").markdown

    resumed = pipeline._load_terminal_page(
        DocumentState(handle=DocumentHandle.from_path(pdf_path)), 1, out_dir
    )
    assert resumed is not None, "a canonical terminal page must still resume"
    assert resumed.text == canonical

    resume_state = _state_with(pdf_path, resumed.text)
    resume_md = pipeline._phase_assemble(resume_state, tmp_path / "assemble_resume").markdown

    assert resume_md == fresh_md
    assert markdown_table_identity(resumed.text) == markdown_table_identity(canonical)
    assert (
        json.loads(sidecar.read_text())["run_fingerprint"]
        == json.loads(next((tmp_path / "assemble_fresh").rglob("pages/00001.json")).read_text())[
            "run_fingerprint"
        ]
    )


def test_gh688_a_pre_rewrite_fragment_is_never_lifted(tmp_path: Path) -> None:
    """Stale derived evidence is INVALIDATED, not dual-keyed: a ledger body
    that is not label-canonical carries a verdict keyed to bytes that would
    not ship, so the page reprocesses. The only difference between the two
    halves of this test is the fragment's label markup."""
    pipeline = _pipeline()
    canonical, _ = canonicalize_table_labels(RAW_PAGE)

    _, clean_dir, _ = _flush_terminal(pipeline, tmp_path, canonical, "clean")
    legacy_pdf, legacy_dir, _ = _flush_terminal(pipeline, tmp_path, canonical, "legacy")
    # Rewrite ONLY the fragment back to the pre-#688 bytes.
    legacy_fragment = next(legacy_dir.rglob("pages/00001.md"))
    legacy_fragment.write_text(RAW_PAGE, encoding="utf-8")

    clean_pdf = tmp_path / "clean.pdf"
    accepted = pipeline._load_terminal_page(
        DocumentState(handle=DocumentHandle.from_path(clean_pdf)), 1, clean_dir
    )
    refused = pipeline._load_terminal_page(
        DocumentState(handle=DocumentHandle.from_path(legacy_pdf)), 1, legacy_dir
    )
    assert accepted is not None
    assert refused is None, "a non-canonical ledger body must reprocess, not resume"


def test_gh688_the_run_fingerprint_carries_the_source_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Why the refusal above is inert in practice: every shipped ``socr``
    ``.py`` file is hashed into the fingerprint, so a page made terminal by a
    socr that predates this change fails the fingerprint gate before the
    fragment is ever read."""
    pipeline = _pipeline()
    before = pipeline._run_fingerprint(EngineType.QWEN)
    monkeypatch.setattr(orchestrator, "_socr_source_digest", lambda: "a-different-socr")
    after = pipeline._run_fingerprint(EngineType.QWEN)
    assert before != after


# ---------------------------------------------------------------------------
# #624b's font-evidence merge stays binder-internal.
# ---------------------------------------------------------------------------


_TWO_BASELINE_WORDS = [
    (50, 60, 100, 70, "Other", 0, 0, 0),
    (105, 60, 165, 70, "authorized", 0, 0, 0),
    (50, 90, 110, 100, "European", 0, 0, 0),
    (115, 90, 175, 100, "currencies", 0, 0, 0),
    (300, 90, 340, 100, "1250.0", 0, 0, 0),
]


def _span(x0, y0, x1, y1, text, *, bold: bool = False) -> dict:
    return {
        "bbox": (x0, y0, x1, y1),
        "text": text,
        "font": "Helvetica" + ("-Bold" if bold else ""),
        "size": 10.0,
        "flags": (2**4) if bold else 0,
    }


@pytest.mark.parametrize("bold_first", [False, True])
def test_gh688_font_merge_controls_keep_their_outcomes(bold_first: bool) -> None:
    """#624b's wrapped-label merge changes a ROW COUNT, so it stays inside
    ``bind()``. Supplying its fixture with entity-indented labels -- the
    candidate as it would arrive before #688, and as it ships after -- must
    leave both control outcomes exactly where the #624b tests put them: the
    same-font pair merges, the bold-first pair does not."""
    entity_markdown = (
        "| Item | A |\n"
        "| --- | --- |\n"
        "| &nbsp;&nbsp;Other authorized | |\n"
        "| &nbsp;&nbsp;European currencies | 1250.0 |\n"
    )
    canonical, changed = canonicalize_table_labels(entity_markdown)
    assert changed == 2
    spans = [
        _span(50, 60, 165, 70, "Other authorized", bold=bold_first),
        _span(50, 90, 175, 100, "European currencies"),
    ]
    raw_result = bind(_TWO_BASELINE_WORDS, entity_markdown, spans=spans)
    canonical_result = bind(_TWO_BASELINE_WORDS, canonical, spans=spans)

    assert raw_result.candidate_wrapped_label_merges == (
        canonical_result.candidate_wrapped_label_merges
    )
    assert raw_result.candidate_row_labels == canonical_result.candidate_row_labels
    expected = () if bold_first else ("Other authorized European currencies",)
    assert canonical_result.candidate_wrapped_label_merges == expected
    # The row count the merge operates on is the binder's own working copy --
    # the shipped bytes still carry both rows.
    assert len(parse_grid(canonical).rows) == 2


def test_gh688_a_table_inside_a_fence_is_a_code_sample_not_a_label() -> None:
    """A grid the model echoed inside a fence is not a reading of the page,
    so its cells are left exactly as written."""
    fenced = (
        "Here is the shape:\n"
        "\n"
        "```\n"
        "| Item | A |\n"
        "| --- | --- |\n"
        "| &nbsp;&nbsp;Swiss francs | 600.0 |\n"
        "```\n"
    )
    canonical, changed = canonicalize_table_labels(fenced)
    assert changed == 0
    assert canonical == fenced
