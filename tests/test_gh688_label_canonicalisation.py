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
import shutil
from pathlib import Path

import pytest

from socr.core.born_digital import DocumentAssessment, PageAssessment
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.manifest import _winning_page_output
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.judge.table_verdict import resolve_cell_refs
from socr.pipeline import orchestrator
from socr.pipeline.agentic import AcceptDecision, route_page
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.binding import bind, parse_grid
from socr.tables.label_canonical import (
    _LINE_BOUNDARIES,
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

    # The entity NAME form is kept as WRITTEN (#688 round 3: an entity that
    # spells structure or active syntax is never respelled, only preserved).
    named = "| Item | A |\n| --- | --- |\n| &vert;split | 1 |\n"
    once, changed_named = canonicalize_table_labels(named)
    assert (once, changed_named) == (named, 0)
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


# ---------------------------------------------------------------------------
# Round 2, finding 1: EVERY boundary the real parsers recognise.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("char", ("|",) + _LINE_BOUNDARIES)
def test_gh688_no_decoded_boundary_can_split_a_row(char: str) -> None:
    """``parse_grid`` and ``_markdown_content_lines`` split with
    ``str.splitlines``, which honours far more than LF and CR. Round 1
    re-encoded only pipe, CR and LF, so a decoded U+2028 became a physical row
    boundary and the grid parsed to ``None``. Pinned per character rather than
    on one example, because the enumeration IS the fix."""
    raw = f"| Item | A |\n| --- | --- |\n| A&#{ord(char)};B | 1 |\n"
    before = parse_grid(raw)
    assert before is not None, "the raw fixture must itself be a grid"

    canonical, _ = canonicalize_table_labels(raw)
    after = parse_grid(canonical)

    assert after is not None, "canonicalisation must not destroy the table"
    assert len(after.rows) == len(before.rows)
    assert [len(r) for r in after.rows] == [len(r) for r in before.rows]
    assert after.rows == before.rows, "the binder must read the same labels"
    assert canonical.split("\n")[2].count(char) == raw.split("\n")[2].count(char), (
        "a structural character was emitted raw"
    )
    assert resolve_cell_refs(canonical, ["R1C1"]) == resolve_cell_refs(raw, ["R1C1"])


def test_gh688_a_literal_boundary_character_leaves_the_cell_alone() -> None:
    """A LITERAL line boundary in the decoded remainder has no serialised form
    that keeps the row where it is, so the cell is returned unchanged. (An
    ENTITY that spells one is kept encoded instead -- see the round-3 rule.)"""
    for char in ("\v", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"):
        cell = f"&nbsp;A{char}B"
        assert canonicalize_label_cell(cell) == cell


# ---------------------------------------------------------------------------
# Round 2, finding 2: nested escapes, and a serialised fixed point.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cell",
    (
        "&amp;nbsp;Swiss francs",
        "&amp;amp;nbsp;Swiss francs",
        "&nbsp;&nbsp;Swiss francs",
        "R&D",
        "A&#124;B",
        "A&#8232;B",
        "Total",
    ),
)
def test_gh688_the_serialised_label_is_a_fixed_point(cell: str) -> None:
    """Round 1 decoded to a *value* and re-encoded only structure, so a
    literal entity text such as ``&amp;nbsp;X`` lost one level of escaping on
    every pass: ``route_page`` judged one string and the lifecycle hook then
    mutated it again. The transform now escapes the decoded ampersand, and
    accepts a serialisation only when ``decode_label_cell`` maps it straight
    back -- which makes a second pass a byte-for-byte no-op."""
    once = canonicalize_label_cell(cell)
    assert canonicalize_label_cell(once) == once
    assert decode_label_cell(once) == decode_label_cell(cell)


def test_gh688_a_nested_entity_label_survives_judging_unchanged() -> None:
    """The same defect at page level, and the invariant that matters: the
    binder's reading of the shipped cell equals its reading of the raw cell,
    and the hook that runs after the judge changes nothing."""
    raw = "| Item | A |\n| --- | --- |\n| &amp;nbsp;Swiss francs | 1 |\n"
    once, _ = canonicalize_table_labels(raw)
    twice, changes = canonicalize_table_labels(once)
    assert twice == once
    assert changes == 0
    assert parse_grid(once).rows[0][0] == parse_grid(raw).rows[0][0] == "&nbsp;Swiss francs"


def test_gh688_the_lifecycle_hook_cannot_mutate_judged_bytes() -> None:
    """``route_page`` judges, then the per-page lifecycle crosses the boundary
    again. Pinned as a difference of zero on the nested-escape fixture that
    broke round 1."""
    body = "| Item | A |\n| --- | --- |\n| &amp;nbsp;Swiss francs | 1 |\n"
    output = PageOutput(page_num=1, text=body, status=PageStatus.SUCCESS, engine="qwen")
    canonicalize_candidate(output)
    judged = output.text
    assert canonicalize_candidate(output) == 0
    assert output.text == judged


def test_gh688_an_encoded_pipe_stays_one_cell() -> None:
    """Astra's control: the encoded pipe is stable and still resolves."""
    raw = "| Item | A |\n| --- | --- |\n| A&#124;B | 1 |\n"
    assert canonicalize_table_labels(raw) == (raw, 0)
    assert "A|B" in resolve_cell_refs(raw, ["R1C1"]).values()


# ---------------------------------------------------------------------------
# Round 2, finding 3: the native fallback candidate.
# ---------------------------------------------------------------------------


def _born_digital_state(pdf_path: Path, native: str, *, via_ingestion: bool) -> DocumentState:
    """A page whose only engine attempt was REJECTED, so the manifest falls
    back to the native reading. ``via_ingestion`` selects the producer:
    ``apply_born_digital`` (the real door) or a direct write to
    ``PageState.native_text`` (the pre-fix shape, kept as the falsification
    arm)."""
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    if via_ingestion:
        state.apply_born_digital(
            DocumentAssessment(
                path=pdf_path,
                pages=[
                    PageAssessment(
                        page_num=1,
                        is_born_digital=True,
                        native_text=native,
                        confidence=1.0,
                    )
                ],
            )
        )
    else:
        ps = state.pages[1]
        ps.is_born_digital = True
        ps.native_text = native
    ps = state.pages[1]
    rejected = PageOutput(
        page_num=1,
        text=canonicalize_table_labels(native)[0],
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
    )
    ps.attempts.append(rejected)
    ps.best_output = None
    return state


def test_gh688_the_native_fallback_ships_canonical_rows_and_resumes(tmp_path: Path) -> None:
    """The reachable hole round 1 left open. The per-page lifecycle skips a
    page with no ``best_output``, but ``_winning_page_output`` then builds the
    winner from the raw native reading -- so the page shipped ``&nbsp;`` rows
    AND ``_load_terminal_page`` refused them on every resume, a page that
    never becomes resumable under its own fingerprint.

    Pinned as a difference between two producers of the same page, and across
    two reconstructed runs under one fingerprint."""
    pipeline = _pipeline()
    pdf_path = _pdf(tmp_path)

    ingested = _born_digital_state(pdf_path, RAW_PAGE, via_ingestion=True)
    assert ingested.pages[1].native_text_raw == RAW_PAGE, "raw provenance is kept"
    assert "&nbsp;" not in (ingested.pages[1].native_text or "")

    winner = _winning_page_output(ingested, 1)
    assert winner is not None
    assert CANONICAL_ROW in winner.text
    assert "&nbsp;" not in winner.text

    first = pipeline._phase_assemble(ingested, tmp_path / "run1").markdown
    assert CANONICAL_ROW in first
    assert "&nbsp;" not in first

    restored = pipeline._load_terminal_page(
        DocumentState(handle=DocumentHandle.from_path(pdf_path)), 1, tmp_path / "run1"
    )
    assert restored is not None, "the fallback page must be resumable"

    # The same document reconstructed a second time, same fingerprint.
    second = pipeline._phase_assemble(
        _born_digital_state(pdf_path, RAW_PAGE, via_ingestion=True), tmp_path / "run2"
    ).markdown
    assert second == first

    # Falsification: the pre-fix producer still reproduces both halves of the
    # defect, so neither assertion above is vacuous.
    legacy = _born_digital_state(pdf_path, RAW_PAGE, via_ingestion=False)
    assert "&nbsp;" in _winning_page_output(legacy, 1).text
    legacy_md = pipeline._phase_assemble(legacy, tmp_path / "legacy").markdown
    assert "&nbsp;" in legacy_md
    assert (
        pipeline._load_terminal_page(
            DocumentState(handle=DocumentHandle.from_path(pdf_path)), 1, tmp_path / "legacy"
        )
        is None
    )


def test_gh688_the_d3_regional_floor_splices_around_canonical_native_text(
    tmp_path: Path,
) -> None:
    """D3's floor reads ``p.native_text`` directly (``manifest`` region
    splice), bypassing every candidate hook. It needs no hook of its own
    because the bytes it reads are canonical at ingestion -- pinned here so a
    future move of the canonicalisation site cannot silently un-fix it."""
    pdf_path = _pdf(tmp_path)
    state = _born_digital_state(pdf_path, RAW_PAGE, via_ingestion=True)
    native = state.pages[1].native_text or ""
    assert CANONICAL_ROW in native
    assert native == canonicalize_table_labels(native)[0]


# ---------------------------------------------------------------------------
# Round 2, finding 4: the fence mask must be index-aligned.
# ---------------------------------------------------------------------------


def test_gh688_a_unicode_separator_does_not_unmask_a_fence() -> None:
    """``_markdown_content_lines`` re-splits with ``str.splitlines``; the
    caller holds a ``split("\\n")`` list. A U+2028 in prose above the fence
    made the two lengths differ, and round 1's fallback then read the raw,
    unmasked lines and rewrote the code sample."""
    raw = "Prose continued again\n```\n| Item | A |\n| --- | --- |\n| &nbsp;X | 1 |\n```\n"
    assert canonicalize_table_labels(raw) == (raw, 0)


def test_gh688_an_unalignable_mask_abstains() -> None:
    """An unclosed HTML comment truncates the masked text. Nothing below it is
    mapped, so nothing below it is rewritten."""
    raw = "<!-- unterminated\n| Item | A |\n| --- | --- |\n| &nbsp;X | 1 |\n"
    assert canonicalize_table_labels(raw) == (raw, 0)


def test_gh688_the_mapped_prefix_is_kept_when_the_tail_is_unmappable() -> None:
    """Round 3 narrows that abstention: a table ABOVE the malformed comment is
    still canonicalised, while the unmapped tail stays untouched. Pinned as the
    difference between the two halves of one document."""
    above = "| Item | A |\n| --- | --- |\n| &nbsp;Above | 1 |\n"
    below = "| Item | A |\n| --- | --- |\n| &nbsp;Below | 1 |\n"
    canonical, changed = canonicalize_table_labels(above + "\n<!-- unterminated\n" + below)
    assert changed == 1
    assert "| Above | 1 |" in canonical
    assert "| &nbsp;Below | 1 |" in canonical


# ---------------------------------------------------------------------------
# Round 3, finding 1: derived region identities and D3's regional splice.
# ---------------------------------------------------------------------------

_D3_MARKER = "[page 1 failed: unverifiable table — see image]"
_BAD_REGION = "| Label | Value |\n| --- | --- |\n| &nbsp;Bad | 1 |\n"
_HEALTHY_REGION = "| Label | Value |\n| --- | --- |\n| &nbsp;Healthy | 2 |\n"


def _d3_page(pdf_path: Path, *, identities: list[str]) -> DocumentState:
    """A born-digital page with two native table regions, the first of which
    the per-region geometry verifier hard-failed, ingested through the real
    ``apply_born_digital`` door with extractor-supplied identities."""
    raw = _BAD_REGION + "\n" + _HEALTHY_REGION
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    state.apply_born_digital(
        DocumentAssessment(
            path=pdf_path,
            pages=[
                PageAssessment(
                    page_num=1,
                    is_born_digital=True,
                    native_text=raw,
                    confidence=1.0,
                    native_table_region_count=2,
                    native_table_region_identities=list(identities),
                    native_table_unverifiable_ordinals=[0],
                    has_unverifiable_table_region=True,
                )
            ],
        )
    )
    ps = state.pages[1]
    ps.native_table_structure_failed = True
    ps.attempts.append(
        PageOutput(page_num=1, text="", engine="qwen", status=PageStatus.ERROR, audit_passed=False)
    )
    ps.best_output = None
    return state


def test_gh688_the_d3_regional_splice_still_retains_a_healthy_sibling(tmp_path: Path) -> None:
    """Round 2 canonicalised the page text at ingestion but copied the
    extractor's region identities unchanged. ``_verify_regions`` computes those
    identities during extraction, so after ingestion they no longer matched the
    canonical regions, ``splice_failed_table_regions`` refused every region,
    and D3's floor shipped only the marker -- dropping a healthy sibling table
    it used to retain. That is content loss, not a resume nuisance.

    Pinned through the real selector: the failed region becomes the marker, the
    healthy one survives, and its label is canonical."""
    pdf_path = _pdf(tmp_path)
    identities = [markdown_table_identity(_BAD_REGION), markdown_table_identity(_HEALTHY_REGION)]

    shipped = _winning_page_output(_d3_page(pdf_path, identities=identities), 1).text

    assert _D3_MARKER in shipped
    assert "| Healthy | 2 |" in shipped, "the healthy sibling table must survive the floor"
    assert "| Bad | 1 |" not in shipped, "the failed region must still be replaced"
    assert "&nbsp;" not in shipped


def test_gh688_identities_and_text_are_rebuilt_together_or_not_at_all(tmp_path: Path) -> None:
    """The rule that makes the above hold: ingestion rebuilds the identities
    through a PROVEN mapping (same block count, and the recorded identities are
    exactly this text's blocks in order) or leaves the text alone. Pinned as a
    difference between provable and unprovable evidence, never as an absolute
    identity value."""
    pdf_path = _pdf(tmp_path)
    raw = _BAD_REGION + "\n" + _HEALTHY_REGION

    provable = _d3_page(
        pdf_path,
        identities=[markdown_table_identity(_BAD_REGION), markdown_table_identity(_HEALTHY_REGION)],
    ).pages[1]
    assert "&nbsp;" not in (provable.native_text or "")
    assert provable.native_table_region_identities == [
        markdown_table_identity(canonicalize_table_labels(_BAD_REGION)[0]),
        markdown_table_identity(canonicalize_table_labels(_HEALTHY_REGION)[0]),
    ]

    # Evidence that does not describe this page's blocks proves no mapping, so
    # the text is left exactly as the extractor wrote it and the two sides stay
    # consistent with each other.
    unprovable = _d3_page(pdf_path, identities=["not-this-page", "nor-this-one"]).pages[1]
    assert unprovable.native_text == raw
    assert unprovable.native_table_region_identities == ["not-this-page", "nor-this-one"]
    assert _winning_page_output(_d3_page(pdf_path, identities=["x", "y"]), 1).text == _D3_MARKER


def test_gh688_the_regional_floor_page_is_reproducible_and_not_blocked_by_688(
    tmp_path: Path,
) -> None:
    """Two reconstructed runs of the same regional-floor page under one
    fingerprint write byte-identical markdown, and the fragment they leave is
    already label-canonical -- so #688's resume gate is not what refuses it.

    It IS refused: a D3 floor page ships ``PageStatus.ERROR``, and the ledger
    only ever restores an exactly-SUCCESS page. That rule predates this ticket
    and is pinned here as a difference against a SUCCESS page written into the
    same directory shape, so a future reader does not mistake it for the
    non-canonical-body refusal."""
    pipeline = _pipeline()
    pdf_path = _pdf(tmp_path)
    identities = [markdown_table_identity(_BAD_REGION), markdown_table_identity(_HEALTHY_REGION)]

    first = pipeline._phase_assemble(
        _d3_page(pdf_path, identities=identities), tmp_path / "d3run1"
    ).markdown
    second = pipeline._phase_assemble(
        _d3_page(pdf_path, identities=identities), tmp_path / "d3run2"
    ).markdown
    assert "| Healthy | 2 |" in first
    assert second == first

    body = next((tmp_path / "d3run1").rglob("pages/00001.md")).read_text(encoding="utf-8")
    assert canonicalize_table_labels(body)[1] == 0, "the ledger body is already canonical"
    assert (
        pipeline._load_terminal_page(
            DocumentState(handle=DocumentHandle.from_path(pdf_path)), 1, tmp_path / "d3run1"
        )
        is None
    )

    # The control: same pipeline, same directory shape, a SUCCESS page -- which
    # does restore. The floor page's refusal is its status, not its bytes.
    canonical_success, _ = canonicalize_table_labels(RAW_PAGE)
    _, success_dir, _ = _flush_terminal(pipeline, tmp_path, canonical_success, "d3control")
    assert (
        pipeline._load_terminal_page(
            DocumentState(handle=DocumentHandle.from_path(tmp_path / "d3control.pdf")),
            1,
            success_dir,
        )
        is not None
    )


def test_gh688_native_regions_are_canonical_before_their_identities_exist(
    tmp_path: Path,
) -> None:
    """Where the fix actually lives: the regions cross the boundary during
    extraction, so ``_verify_regions`` hashes canonical bytes and the page text
    it interleaves is the same bytes. Ingestion's rebuild is the backstop for a
    page arriving by another door, not the primary repair."""
    import fitz as _fitz

    from socr.core.born_digital import BornDigitalDetector

    path = tmp_path / "regions.pdf"
    doc = _fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Item")
    page.insert_text((260, 100), "Amount")
    page.insert_text((72, 120), "Swiss francs")
    page.insert_text((260, 120), "600.0")
    doc.save(str(path))
    doc.close()

    detector = BornDigitalDetector()
    reopened = _fitz.open(str(path))
    try:
        text = detector.extract_structured(reopened[0])
    finally:
        reopened.close()

    identities = list(getattr(detector, "_last_extraction_region_identities", []) or [])
    assert canonicalize_table_labels(text) == (text, 0), (
        "the extractor's page text must already be label-canonical"
    )
    for identity in identities:
        assert identity, "a region identity must still be computable"


# ---------------------------------------------------------------------------
# Round 3, finding 2: decoding is not meaning-preserving.
# ---------------------------------------------------------------------------


def test_gh688_an_entity_that_spells_emphasis_is_kept_encoded() -> None:
    """``&ast;important&ast;`` decodes to ``*important*``, which every Markdown
    renderer reads as italics: the label's own asterisks vanish from the
    visible text. The entity is doing real work, so it is kept as written."""
    raw = "| Label | Value |\n| --- | --- |\n| &ast;important&ast; | 1 |\n"
    assert canonicalize_table_labels(raw) == (raw, 0)


def test_gh688_an_entity_that_spells_a_tag_is_kept_encoded() -> None:
    """``&lt;b&gt;X&lt;/b&gt;`` decodes to a live CommonMark tag."""
    markdown_it = pytest.importorskip("markdown_it")
    raw = "| Label | Value |\n| --- | --- |\n| &lt;b&gt;X&lt;/b&gt; | 1 |\n"
    canonical, changed = canonicalize_table_labels(raw)
    assert (canonical, changed) == (raw, 0)

    renderer = markdown_it.MarkdownIt("commonmark").enable("table")
    assert "<b>X</b>" not in renderer.render(canonical)
    assert renderer.render(canonical) == renderer.render(raw)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_gh688_the_review_renderer_shows_the_same_label_before_and_after() -> None:
    """Astra's control, run through the review viewer's own ``renderMd`` under
    Node: what a reader sees must not change, and no tag may go live.

    The skip is decided at IMPORT time on purpose: ``conftest`` neuters
    ``shutil.which`` for every test, so a call-time probe always reports node
    missing."""
    from test_gh652_review_renderer_literal_escapes import _render, _visible

    for raw in (
        "| Label | Value |\n| --- | --- |\n| &lt;b&gt;X&lt;/b&gt; | 1 |\n",
        "| Label | Value |\n| --- | --- |\n| &ast;important&ast; | 1 |\n",
    ):
        canonical, _ = canonicalize_table_labels(raw)
        rendered = _render(canonical)
        assert _visible(rendered) == _visible(_render(raw))
        assert "<b>X</b>" not in rendered
        assert "<i>important</i>" not in rendered


def test_gh688_the_indentation_target_still_decodes() -> None:
    """The keep-encoded rule must not swallow the ticket: ``&nbsp;`` is
    whitespace, not syntax, and the label it indents still ships plain."""
    canonical, changed = canonicalize_table_labels(RAW_PAGE)
    assert changed == 2
    assert CANONICAL_ROW in canonical
    assert "&nbsp;" not in canonical


def test_gh688_a_literal_ampersand_is_never_newly_encoded() -> None:
    """Nothing is ever newly encoded, so a clean label is never churned."""
    raw = "| Label | Value |\n| --- | --- |\n| R&D | 1 |\n"
    assert canonicalize_table_labels(raw) == (raw, 0)
    assert canonicalize_label_cell("&nbsp;R&D") == "R&D"
