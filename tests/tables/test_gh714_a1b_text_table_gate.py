"""GH-714: A1b's row-shape reconciliation and the TEXT table it cannot verify.

``manifest._row_shape_reconciliation`` (TICKET-A1b, #640) is the twin of A2's
term (b) at a different call site, and carried the same defect #703 fixed
there: ``row_shape_min`` derived from the candidate's own numeric body rows
collapses to 1 on a text table, at which every native prose band mentioning a
figure counts as a table row and a complete candidate reads as a massive row
shortfall.

**Round 1 fixed that by returning True, and Astra falsified the remedy.** True
at this call site means ADMIT, and the only evidence behind the admission is
A1a's numeric-row corroboration. Replace one prose row of the real BoE 2018 p1
candidate with a fabricated sentence and that corroboration is byte-identical
(bound=2, total=2, no extras), because the fabricated row contributes nothing
to the denominator. Two matching numeric rows cannot validate arbitrary prose
cells, and A1c is disclosure, not verification.

Round 2 keeps the diagnosis and reverses the remedy. Where the native page has
no recurring numeric column lanes the reconciliation is NOT APPLICABLE, so the
numeric-row corroboration route is not available for that candidate: A1b
declines it with ``RowShapeOutcome.NOT_RECONCILABLE_TEXT_TABLE``, and the page
falls through to the routes that CAN carry authority for prose -- a completed
page-judge acceptance, or #713's table-acceptance credential. The invalid
row-count comparison is not restored as an accidental defence; the decline
carries its own reason, visible at page ``failure_mode``, in the sidecar, in
the document-level note and on the CLI, so it never reads as the old false
refusal.

Every pin is a DIFFERENCE measured twice in one process, changing only whether
the lane gate is consulted. Fixtures are #703's, reused rather than reinvented.
"""

from __future__ import annotations

import pytest

from socr.core.manifest import (
    RowShapeOutcome,
    SelectionProvenance,
    _row_shape_reconciliation,
    _select_page_output_tagged,
    structure_class_text_table_declined,
)
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import PageState
from socr.tables import structure_check

from test_gh703_text_table_dominance import (  # noqa: I001  (pytest rootdir import)
    BOE_2018_P1_QWEN,
    BOE_2018_PDF,
    TEXT_TABLE_MD,
    TEXT_TABLE_WORDS,
    _boe_p1,
    _sparse_prefix_fixture,
)
from test_structure_check_truncated import (  # noqa: I001  (pytest rootdir import)
    BULLETIN_P2_COMPLETE,
    BULLETIN_P2_TRUNCATED,
    BULLETIN_P3_COMPLETE,
    BULLETIN_P3_TRUNCATED,
    _fixture_words,
)


def _gated_vs_open(
    monkeypatch: pytest.MonkeyPatch, words: list, markdown: str
) -> tuple[RowShapeOutcome, RowShapeOutcome]:
    """``(gate_open, gate_real)`` outcomes for one candidate, in one process.

    ``gate_open`` forces ``_native_page_has_column_lanes`` to True, which is
    exactly the pre-#714 predicate: the lane gate is the only change.
    """
    real = _row_shape_reconciliation(words, markdown)
    with monkeypatch.context() as m:
        m.setattr(structure_check, "_native_page_has_column_lanes", lambda words: True)
        forced = _row_shape_reconciliation(words, markdown)
    return forced, real


# ---------------------------------------------------------------------------
# The three-valued outcome
# ---------------------------------------------------------------------------


def test_hermetic_text_table_declines_the_route_with_its_own_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#703's synthetic BoE-shaped comparison box, at A1b's call site.

    Ungated the candidate is refused as a SHORTFALL, which is the false refusal
    #714 was filed for. Gated it is declined as NOT_RECONCILABLE_TEXT_TABLE,
    which is a different fact with a different remedy. Neither admits it, and
    that is the round-2 correction: neither outcome is RECONCILED.
    """
    assert structure_check._native_page_has_column_lanes(TEXT_TABLE_WORDS) is False

    assert _gated_vs_open(monkeypatch, TEXT_TABLE_WORDS, TEXT_TABLE_MD) == (
        RowShapeOutcome.SHORTFALL,
        RowShapeOutcome.NOT_RECONCILABLE_TEXT_TABLE,
    )
    assert (
        _row_shape_reconciliation(TEXT_TABLE_WORDS, TEXT_TABLE_MD) is not RowShapeOutcome.RECONCILED
    )


def test_only_reconciled_admits() -> None:
    """Admission is ``RECONCILED`` and nothing else -- the round-1 defect,
    pinned on real fixtures rather than on the enum.

    Round 1 made the text-table case admit, which is what shipped the
    candidate on numeric evidence alone.
    """
    complete, truncated, numeric_words = _sparse_prefix_fixture()

    assert _row_shape_reconciliation(numeric_words, complete) is RowShapeOutcome.RECONCILED
    assert _row_shape_reconciliation(numeric_words, truncated) is not RowShapeOutcome.RECONCILED
    assert (
        _row_shape_reconciliation(TEXT_TABLE_WORDS, TEXT_TABLE_MD) is not RowShapeOutcome.RECONCILED
    )

    # and the three cases really are three, not two wearing one name
    assert {
        _row_shape_reconciliation(numeric_words, complete),
        _row_shape_reconciliation(numeric_words, truncated),
        _row_shape_reconciliation(TEXT_TABLE_WORDS, TEXT_TABLE_MD),
    } == set(RowShapeOutcome)


# ---------------------------------------------------------------------------
# Numeric pages: every existing outcome is preserved
# ---------------------------------------------------------------------------


def test_sparse_prefix_outcomes_are_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Astra's #703 counterexample at A1b's call site: a numeric table
    truncated to two legitimately sparse rows. The native page has lanes, so
    the gate is inert and both outcomes are what they were.
    """
    complete, truncated, words = _sparse_prefix_fixture()
    assert structure_check._native_page_has_column_lanes(words) is True

    assert _gated_vs_open(monkeypatch, words, truncated) == (
        RowShapeOutcome.SHORTFALL,
        RowShapeOutcome.SHORTFALL,
    )
    assert _gated_vs_open(monkeypatch, words, complete) == (
        RowShapeOutcome.RECONCILED,
        RowShapeOutcome.RECONCILED,
    )


# The row values below are the ones the two existing ECB fixture tests in
# ``test_structure_check_truncated`` build their native words from, verbatim --
# the real values the COMPLETE candidate's table contains.
_P2_ROWS = [
    [
        "2018",
        "4,404.9",
        "4,489.0",
        "991.4",
        "844.2",
        "2,569.4",
        "5,741.9",
        "6,024.9",
        "682.6",
        "4,356.4",
        "702.9",
    ],
    [
        "2019",
        "4,475.8",
        "4,577.9",
        "967.4",
        "878.0",
        "2,630.4",
        "5,931.1",
        "6,224.0",
        "720.1",
        "4,524.6",
        "686.4",
    ],
    [
        "2020",
        "4,723.6",
        "4,841.3",
        "898.9",
        "1,012.0",
        "2,812.7",
        "6,119.9",
        "6,390.1",
        "700.2",
        "4,725.1",
        "694.6",
    ],
]
_P3_ROWS = [
    [
        "2018",
        "389.2",
        "6,817.4",
        "1,940.0",
        "56.1",
        "2,099.7",
        "2,721.6",
        "1,030.0",
        "460.2",
        "187.0",
        "194.9",
    ],
    [
        "2019",
        "364.2",
        "7,058.9",
        "1,946.1",
        "50.1",
        "2,156.5",
        "2,906.1",
        "1,455.5",
        "452.3",
        "178.9",
        "187.2",
    ],
    [
        "2020",
        "749.0",
        "6,967.4",
        "1,916.7",
        "42.1",
        "1,994.9",
        "3,013.7",
        "1,432.7",
        "539.6",
        "130.1",
        "139.2",
    ],
]


@pytest.mark.parametrize(
    ("truncated_md", "complete_md", "rows"),
    [
        (BULLETIN_P2_TRUNCATED, BULLETIN_P2_COMPLETE, _P2_ROWS),
        (BULLETIN_P3_TRUNCATED, BULLETIN_P3_COMPLETE, _P3_ROWS),
    ],
    ids=["bulletin_p2", "bulletin_p3"],
)
def test_real_ecb_truncation_fixtures_keep_their_a1b_outcome(
    monkeypatch: pytest.MonkeyPatch,
    truncated_md: str,
    complete_md: str,
    rows: list[list[str]],
) -> None:
    """The two real ECB bulletin truncation fixtures: aligned numeric columns,
    so the gate opens and A1b's verdicts are identical to the pre-#714 ones --
    the truncated reading refused as a shortfall, the complete one reconciled.
    """
    words = _fixture_words(rows)
    assert structure_check._native_page_has_column_lanes(words) is True

    assert _gated_vs_open(monkeypatch, words, truncated_md) == (
        RowShapeOutcome.SHORTFALL,
        RowShapeOutcome.SHORTFALL,
    )
    assert _gated_vs_open(monkeypatch, words, complete_md) == (
        RowShapeOutcome.RECONCILED,
        RowShapeOutcome.RECONCILED,
    )


def test_deleted_row_refusals_are_unchanged_by_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The A1b selection suite's own reproducers (a 20-row grid with two rows
    deleted from either end) reach A1b through an aligned native page, so the
    gate is inert on both edges and both stay SHORTFALL.
    """
    rows = [(2000 + i, float(100 + i), float(200 + i)) for i in range(20)]

    def _md(subset: list[tuple[int, float, float]]) -> str:
        lines = ["| Year | A | B |", "|---|---|---|"]
        lines += [f"| {y} | {a} | {b} |" for y, a, b in subset]
        return "\n".join(lines) + "\n"

    words: list[tuple] = []
    for i, (year, a, b) in enumerate(rows):
        x = 0.0
        for tok in (str(year), str(a), str(b)):
            words.append((x, 10.0 + i * 20.0, x + 8.0, 20.0 + i * 20.0, tok))
            x += 12.0

    assert structure_check._native_page_has_column_lanes(words) is True
    assert _gated_vs_open(monkeypatch, words, _md(rows)) == (
        RowShapeOutcome.RECONCILED,
        RowShapeOutcome.RECONCILED,
    )
    for subset in (rows[:-2], rows[2:]):
        assert _gated_vs_open(monkeypatch, words, _md(subset)) == (
            RowShapeOutcome.SHORTFALL,
            RowShapeOutcome.SHORTFALL,
        )


# ---------------------------------------------------------------------------
# Selection: the declined route, and the two routes that can carry a text table
# ---------------------------------------------------------------------------

_TEXT_TABLE_BBOX = (
    min(w[0] for w in TEXT_TABLE_WORDS) - 5.0,
    min(w[1] for w in TEXT_TABLE_WORDS) - 5.0,
    max(w[2] for w in TEXT_TABLE_WORDS) + 5.0,
    max(w[3] for w in TEXT_TABLE_WORDS) + 5.0,
)

# One prose cell of the fixture, replaced. Every number in the candidate is
# untouched, so A1a's corroboration is identical -- which is the whole point.
FABRICATED_TEXT_TABLE_MD = TEXT_TABLE_MD.replace(
    "| Quarterly hourly labour productivity growth to average just over a quarter of a "
    "percent. | Unchanged from February. |",
    "| The Bank guarantees permanent prosperity without any risk. | Unconditional guarantee. |",
)


def _text_table_page(
    markdown: str = TEXT_TABLE_MD,
    *,
    audit_passed: bool = False,
) -> PageState:
    """A born-digital structure-class page whose one grid candidate is the
    hermetic text table, over native words that show no column lanes.
    """
    out = PageOutput(
        page_num=1,
        text=markdown,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=audit_passed,
    )
    p = PageState(page_num=1)
    p.is_born_digital = True
    p.has_tables = True
    p.native_text = "Inflation Report May 2018 Section 3"
    p.native_words = list(TEXT_TABLE_WORDS)
    p.detected_table_bboxes = [_TEXT_TABLE_BBOX]
    p.attempts = [out]
    p.best_output = out
    return p


def test_fixture_would_have_been_admitted_on_numeric_evidence_alone() -> None:
    """Grounding for everything below: A1a DOES clear on this fixture, and
    clears identically on the fabricated variant.

    Without this the decline below would prove nothing -- a candidate that
    fails A1a never reaches the row-shape check at all.
    """
    from socr.tables.row_corroboration import corroborate_rows

    honest = corroborate_rows(TEXT_TABLE_WORDS, TEXT_TABLE_MD, _TEXT_TABLE_BBOX)
    fabricated = corroborate_rows(TEXT_TABLE_WORDS, FABRICATED_TEXT_TABLE_MD, _TEXT_TABLE_BBOX)

    assert honest.clears is True
    assert (fabricated.bound, fabricated.total, fabricated.extra_numbers) == (
        honest.bound,
        honest.total,
        honest.extra_numbers,
    )
    assert FABRICATED_TEXT_TABLE_MD != TEXT_TABLE_MD


def test_declined_text_table_floors_under_its_own_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """The difference the round-2 design makes, at selection.

    Gate real: the route declines and the page fails closed under
    ``ROW_SHAPE_NOT_RECONCILABLE_TEXT_TABLE`` / its own selection tag. Gate
    forced open: the pre-#714 shortfall refusal, which floors the same bytes
    under the reason that says every candidate was refused. Same marker, two
    different explanations -- which is exactly what #713 established a floor
    reason is for.
    """
    p = _text_table_page()
    assert structure_class_text_table_declined(p) is True

    out, tag = _select_page_output_tagged(_DocStateStub(p), 1)
    assert (out.failure_mode, tag) == (
        FailureMode.ROW_SHAPE_NOT_RECONCILABLE_TEXT_TABLE,
        SelectionProvenance.STRUCTURE_CLASS_TEXT_TABLE_FLOOR,
    )

    with monkeypatch.context() as m:
        m.setattr(structure_check, "_native_page_has_column_lanes", lambda words: True)
        ungated_out, ungated_tag = _select_page_output_tagged(_DocStateStub(_text_table_page()), 1)
    assert (ungated_out.failure_mode, ungated_tag) == (
        FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED,
        SelectionProvenance.STRUCTURE_CLASS_FLOOR,
    )
    # both withhold, and neither reason is the other's
    assert ungated_out.status is out.status is PageStatus.ERROR


def test_fabricated_prose_never_ships_on_a_declined_text_table() -> None:
    """Astra's P1, hermetic: one prose cell replaced, every number identical.

    The corroboration cannot tell the two candidates apart (pinned above), so
    the guard cannot be the corroboration. It is the decline.
    """
    p = _text_table_page(FABRICATED_TEXT_TABLE_MD)
    out, tag = _select_page_output_tagged(_DocStateStub(p), 1)

    assert "guarantees permanent prosperity" not in (out.text or "")
    assert (out.failure_mode, tag) == (
        FailureMode.ROW_SHAPE_NOT_RECONCILABLE_TEXT_TABLE,
        SelectionProvenance.STRUCTURE_CLASS_TEXT_TABLE_FLOOR,
    )


def test_a_completed_page_acceptance_ships_the_text_table() -> None:
    """The positive: the route that CAN speak for prose cells.

    Same page, same words, same declined numeric route -- the only thing added
    is a completed page-judge acceptance (``audit_passed``), and the text table
    ships in full. The decline withholds a candidate from ONE route; it is not
    a verdict on the page.
    """
    declined_out, _tag = _select_page_output_tagged(_DocStateStub(_text_table_page()), 1)
    accepted_out, accepted_tag = _select_page_output_tagged(
        _DocStateStub(_text_table_page(audit_passed=True)), 1
    )

    assert declined_out.failure_mode is FailureMode.ROW_SHAPE_NOT_RECONCILABLE_TEXT_TABLE
    assert "Unemployment rate to fall to 4%" not in (declined_out.text or "")

    assert "Unemployment rate to fall to 4%" in (accepted_out.text or "")
    assert accepted_tag is not SelectionProvenance.STRUCTURE_CLASS_TEXT_TABLE_FLOOR
    assert accepted_out.status is PageStatus.SUCCESS


class _DocStateStub:
    """The two attributes ``_select_page_output_tagged`` reads for a
    single-page selection: the page map and the event list.

    A stub rather than a real ``DocumentState`` because these fixtures have no
    PDF -- the hermetic native words ARE the page. ``MagicMock`` is deliberately
    not used: a bare mock makes every negative assertion in this file vacuous.
    """

    def __init__(self, page: PageState) -> None:
        self.pages = {page.page_num: page}
        self.events: list = []
        self.whole_doc_attempts: list = []
        self.handle = None


# ---------------------------------------------------------------------------
# #713's credential: the other route that can carry a text table
# ---------------------------------------------------------------------------


def _credentialed_text_table_state(tmp_path, *, credential: bool):
    """A real one-page ``DocumentState`` (a credential is bound to the document
    checksum, so this one needs a PDF) whose single grid candidate is the
    hermetic text table, carrying #713's TYPED page-judge timeout.
    """
    pymupdf = pytest.importorskip("pymupdf")
    from socr.core.document import DocumentHandle
    from socr.core.page_credential import (
        TableAcceptance,
        TableAcceptanceCredential,
        sha256_text,
    )
    from socr.core.result import JUDGE_OUTCOME_TIMEOUT
    from socr.core.state import DocumentState
    from socr.tables.reconcile import find_table_blocks

    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "gh714.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Monitoring the MPC's key judgements")
    doc.save(str(path))
    doc.close()

    state = DocumentState(handle=DocumentHandle.from_path(path))
    ps = state.pages[1]
    page = _text_table_page()
    for attr in ("is_born_digital", "has_tables", "native_text", "native_words"):
        setattr(ps, attr, getattr(page, attr))
    ps.detected_table_bboxes = list(page.detected_table_bboxes)

    out = PageOutput(
        page_num=1,
        text=TEXT_TABLE_MD,
        status=PageStatus.SUCCESS,
        engine="qwen",
        provider_id="qwen-local",
        provider_model="qwen3-vl:30b-a3b-instruct",
        provider_backend="ollama",
        audit_passed=False,
        judge_reason="judge raised: timed out",
        judge_outcome=JUDGE_OUTCOME_TIMEOUT,
    )
    if credential:
        lines = TEXT_TABLE_MD.splitlines()
        entries = [
            TableAcceptance(
                table_id=f"p1-t{idx}",
                markdown_sha256=sha256_text("\n".join(lines[b.start : b.end + 1])),
                witness_sha256=sha256_text(f"witness-bytes-{idx}"),
                witness_scope="page",
                rungs=("glm-5.3-flash:cloud",),
            )
            for idx, b in enumerate(find_table_blocks(TEXT_TABLE_MD))
        ]
        assert entries, "the fixture must emit at least one table for a credential to cover"
        out.table_acceptance_credential = TableAcceptanceCredential(
            page_num=1,
            document_checksum=state.handle.file_hash or "",
            candidate_sha256=sha256_text(TEXT_TABLE_MD),
            attempt_engine="qwen",
            attempt_provider_id="qwen-local",
            attempt_provider_model="qwen3-vl:30b-a3b-instruct",
            attempt_provider_backend="ollama",
            judge_model="qwen3-vl:30b-a3b-instruct",
            judge_outcome=JUDGE_OUTCOME_TIMEOUT,
            run_fingerprint="fp:test",
            tables=tuple(entries),
        ).to_dict()

    ps.attempts = [out]
    ps.best_output = out
    return state


def test_a_713_credential_ships_the_text_table_flagged(tmp_path) -> None:
    """The second positive, and the difference is the credential alone.

    Without it the page fails closed; with it the same bytes ship DEMOTED under
    #713's own mode. Nothing here claims the prose was verified -- what the
    credential proves is that the table judge ladder accepted every table these
    exact bytes emit.
    """
    without = _select_page_output_tagged(
        _credentialed_text_table_state(tmp_path / "a", credential=False), 1
    )
    with_cred = _select_page_output_tagged(
        _credentialed_text_table_state(tmp_path / "b", credential=True), 1
    )

    assert "Unemployment rate to fall to 4%" not in (without[0].text or "")
    assert without[0].status is PageStatus.ERROR

    assert "Unemployment rate to fall to 4%" in (with_cred[0].text or "")
    assert (with_cred[0].status, with_cred[0].failure_mode) == (
        PageStatus.WARNING,
        FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED,
    )
    assert with_cred[1] is SelectionProvenance.STRUCTURE_CLASS_JUDGE_TIMEOUT_CREDENTIALED


def test_a_typed_judge_timeout_outranks_the_text_table_reason(tmp_path) -> None:
    """Disclosed precedence: when the page judge TIMED OUT and no credential
    cleared it, the floor reports the timeout, not the text-table decline.

    Both facts hold on that page. The timeout is reported because its remedy --
    re-run the page judge -- IS the remedy for the declined text table, which
    ships on a completed page acceptance.
    """
    out, tag = _select_page_output_tagged(
        _credentialed_text_table_state(tmp_path / "c", credential=False), 1
    )
    assert (out.failure_mode, tag) == (
        FailureMode.PAGE_JUDGE_TIMEOUT,
        SelectionProvenance.STRUCTURE_CLASS_PAGE_JUDGE_TIMEOUT_FLOOR,
    )


# ---------------------------------------------------------------------------
# Document level: the reason survives assemble
# ---------------------------------------------------------------------------


def _pipeline():
    from socr.core.config import EngineType, PipelineConfig
    from socr.pipeline.orchestrator import UnifiedPipeline

    return UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            agentic=True,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            save_figures=False,
            write_manifest=False,
            table_judge_ladder=False,
        )
    )


def test_assemble_surfaces_the_text_table_reason_at_document_level(tmp_path) -> None:
    """DIFFERENCE through the real assemble phase: the lane gate real vs forced
    open, on the same page.

    The declined page must emit its OWN event kind and its own sentence in the
    floor note, and must KEEP the structure-class floor surfacing it already had
    -- #713's rule, applied again: a reason is added, never traded for one a
    consumer already reads.
    """
    state = _credentialed_text_table_state(tmp_path / "declined", credential=False)
    # a completed (not timed-out) judge outcome, so the text-table reason is the
    # one under test rather than #713's
    attempt = state.pages[1].attempts[0]
    attempt.judge_outcome = ""
    attempt.judge_reason = ""

    pipeline = _pipeline()
    result = pipeline._phase_assemble(state, tmp_path / "declined_out")
    kinds = {getattr(e, "kind", "") for e in state.events}

    assert "row_shape_not_reconcilable_text_table_floor" in kinds
    assert "structure_class_ladder_exhausted_floor" in kinds
    assert "page_judge_timeout_floor" not in kinds

    from socr.core.manifest import finalized_page_records
    from socr.core.result import DocumentStatus
    from socr.pipeline.orchestrator import UnifiedPipeline

    assert result.status is not DocumentStatus.SUCCESS

    note = UnifiedPipeline._structure_class_floor_note(state, finalized_page_records(state))
    assert note is not None
    assert "TEXT table" in note
    assert "table-acceptance credential" in note
    # and the page is NOT reported as a ladder it never exhausted
    assert "structure-class ladder exhausted" not in note


# ---------------------------------------------------------------------------
# The real BoE page (corpus-skipped)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (BOE_2018_PDF.exists() and BOE_2018_P1_QWEN.exists()),
    reason="BoE census corpus not present on this machine",
)
def test_real_boe_p1_outcome_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ticket's page, on the real PDF and the real cached qwen attempt.

    Its two numeric body rows (``('4%',)``, ``('32.',)``) set
    ``row_shape_min = 1``, at which the native page shows 19 "table-shaped
    rows" -- the shortfall the ungated predicate reports. Gated, the page's
    absent column lanes make the comparison inapplicable instead.
    """
    markdown, words = _boe_p1()

    # grounding: this IS the complete, ladder-accepted candidate
    assert "Table 3.B Monitoring the MPC's key judgements" in markdown
    assert "Unemployment rate to fall to 4% by the end of the year." in markdown
    assert structure_check._native_page_has_column_lanes(words) is False

    assert _gated_vs_open(monkeypatch, words, markdown) == (
        RowShapeOutcome.SHORTFALL,
        RowShapeOutcome.NOT_RECONCILABLE_TEXT_TABLE,
    )
