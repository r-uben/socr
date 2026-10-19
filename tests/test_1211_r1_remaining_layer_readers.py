"""PR #1211 round 1: the readers the first round missed, and its two wrong directions.

Reviews of `1b3007d3`: Argon BLOCK (three more readers, plus a fail-open), Grok
MERGE-AFTER-FIX (`verify_scanned_table`), r-uben MERGE once two should-fixes land.

The rule is unchanged and is the one every pin here reads: on the UNION
(``state.layer_is_admissible_evidence``, so a FAILED detection scan is included), the
suspect layer is NEVER evidence for or against a reading. Round 1 of #1205 gated the
table chain; these are the readers outside it, plus the two places the gating itself
pointed the wrong way.

Every pin is a difference run in one process with only the flag changing, and a control
arm in which the layer DOES count. Hermetic: no provider, no model, and no tesseract --
the pixel witness is injected, which is what the production caller passes anyway
(``ocr_image_fn``); CI has no tesseract, so a test that relied on the real one would
read as "no witness" there and pass vacuously.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import fitz
import pytest
from test_1205_table_chain_evidence_readers import (
    _LAYER_MD,
    _TRUE_MD,
    _bo,
    _flag,
    _make_config,
    _make_pipeline,
    _make_state,
    _prior_ocr_table_pdf,
)

from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# ---------------------------------------------------------------------------
# 1. Argon P0 1.1 / Grok P1 -- verify_scanned_table's born-digital bundle
# ---------------------------------------------------------------------------


#: A pixel witness that reads what the PAGE shows, disagreeing with the layer on the
#: sign of one value. Injected rather than real: this stands for tesseract on the crop,
#: and the disagreement is the whole point -- a page where the layer and the pixels
#: agree cannot show whether the layer was consulted.
def _pixel_witness_true(_pix) -> str:
    return "Item OLS IV\nRowA -350 200\nRowB 300 400"


def _visible_layer_table_pdf(tmp_path: Path, name: str = "visible_layer.pdf") -> Path:
    """A ruled table whose VISIBLE text layer reads ``350``, with no covering raster.

    This is the shape the hole needs: ``page_has_page_covering_raster`` is False (a
    scan-failed page, a cover under 50%, or a ``get_image_info`` error), so
    ``verify_scanned_table`` took the ``build_born_digital_evidence`` branch and unioned
    ``page.get_text`` into the numeric evidence.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    cols = [100, 180, 250]
    rows = [120, 142, 164]
    kw = {"fontsize": 9}
    for i, head in enumerate(("Item", "OLS", "IV")):
        page.insert_text((cols[i] + 4, rows[0] + 12), head, **kw)
    for r, (label, a, b) in enumerate((("RowA", "350", "200"), ("RowB", "300", "400"))):
        page.insert_text((cols[0] + 4, rows[r + 1] + 12), label, **kw)
        page.insert_text((cols[1] + 4, rows[r + 1] + 12), a, **kw)
        page.insert_text((cols[2] + 4, rows[r + 1] + 12), b, **kw)
    for yy in [*rows, rows[-1] + 22]:
        page.draw_line((100, yy), (330, yy))
    for xx in [*cols, 330]:
        page.draw_line((xx, rows[0]), (xx, rows[-1] + 22))
    pdf_path = tmp_path / name
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_the_source_evidence_bundle_excludes_a_suspect_layer(tmp_path: Path) -> None:
    """Argon's own repro, both directions.

    Control (``layer_is_evidence=True``, today's behaviour on an admissible page): the
    layer's ``350`` PASSES and the pixel witness's ``-350`` is REFUSED as unsupported --
    an earlier OCR run's reading of the page passing itself and vetoing the pixels.
    With the layer excluded the two swap, which is the whole claim.

    ``page_has_page_covering_raster`` is asserted False first: with a covering raster
    GH-163 already excluded the layer, so this fixture would prove nothing.
    """
    from socr.tables.source_evidence import page_has_page_covering_raster, verify_scanned_table

    doc = fitz.open(_visible_layer_table_pdf(tmp_path))
    try:
        page = doc[0]
        assert not page_has_page_covering_raster(page), (
            "the fixture has a covering raster, so GH-163 already excluded the layer"
        )

        def verdicts(layer_is_evidence: bool):
            return {
                arm: verify_scanned_table(
                    page,
                    text,
                    ocr_image_fn=_pixel_witness_true,
                    native_trusted=False,
                    layer_is_evidence=layer_is_evidence,
                ).passed
                for arm, text in (("layer_350", _LAYER_MD), ("pixels_-350", _TRUE_MD))
            }

        control = verdicts(True)
        assert control["layer_350"] is True, (
            "control: the layer is not passing its own reading here, so the arm below "
            f"proves nothing: {control}"
        )

        excluded = verdicts(False)
        assert excluded == {"layer_350": False, "pixels_-350": True}, (
            f"the suspect layer still decided the table: {excluded}"
        )
        # Measured, and stated so the claim does not outrun it: with the layer INCLUDED
        # both readings pass, because the bundle is the union of the layer's tokens and
        # the pixels'. So what the exclusion changes on this fixture is that the layer
        # can no longer pass its OWN error -- the refusal of the correct reading needs a
        # page with no usable pixel witness, and there the real tesseract reads the same
        # ink the layer holds, so the two agree and nothing can be shown.
        assert control["pixels_-350"] is True
    finally:
        doc.close()


def test_the_source_evidence_judge_asks_the_union_at_the_real_caller(tmp_path: Path) -> None:
    """The wiring, observed rather than read: what the judge actually passes.

    Built through ``pipeline._build_page_judge(state)`` so a hard-coded ``True``, or a
    key taken from the wrong predicate, fails here. All three flag states, and the
    exception arm, which must exclude the layer rather than trust it.
    """
    import socr.tables.source_evidence as se

    captured: list = []
    real = se.verify_scanned_table

    def spy(page, text, **kwargs):
        captured.append(kwargs.get("layer_is_evidence"))
        return real(page, text, **kwargs)

    pdf_path = _visible_layer_table_pdf(tmp_path / "wiring")
    for positive, scan_failed in ((False, False), (False, True), (True, False)):
        pipeline = _make_pipeline()
        state = _make_state(pdf_path)
        ps = state.pages[1]
        ps.has_tables = True
        ps.is_born_digital = True
        _flag(ps, positive=positive, scan_failed=scan_failed)
        judge = pipeline._build_page_judge(state)
        with patch.object(se, "verify_scanned_table", spy):
            judge.assess(
                PageOutput(
                    page_num=1,
                    text=_LAYER_MD,
                    status=PageStatus.SUCCESS,
                    engine="qwen",
                    audit_passed=True,
                ),
                PROFILE_QWEN_LOCAL,
            )

    assert captured == [True, False, False], (
        f"the judge does not ask the UNION, so an unknown layer still decides: {captured}"
    )


# ---------------------------------------------------------------------------
# 2. Argon P0 1.2 -- the truncation scorer
# ---------------------------------------------------------------------------


def _bands(n: int) -> list:
    """``get_text("words")``-shaped tuples: ``n`` table-shaped bands, 10pt apart.

    The spacing matters: at 1pt the bands overlap and the native row counter collapses
    them, which made an earlier version of this fixture inert.
    """
    out = []
    for i in range(n):
        y = 100.0 + i * 10
        out.append((100.0, y, 140.0, y + 8.0, f"Row{i}", 0, i, 0))
        out.append((160.0, y, 200.0, y + 8.0, f"{100 + i}.0", 0, i, 1))
        out.append((240.0, y, 280.0, y + 8.0, f"{200 + i}.0", 0, i, 2))
    return out


def _reading(n_rows: int, first_value: str) -> str:
    rows = [
        f"| Row{i} | {first_value if i == 0 else f'{100 + i}.0'} | {200 + i}.0 |"
        for i in range(n_rows)
    ]
    return "| Item | A | B |\n| --- | --- | --- |\n" + "\n".join(rows) + "\n"


def test_a_reading_is_not_called_truncated_against_a_suspect_layer() -> None:
    """``_truncated_grid_reading_ids`` scored the candidate against the layer's bands.

    Consequence on a flagged page: a judge-accepted 18-row reading carrying the correct
    ``-100.0`` is flagged truncated against 20 bands of an earlier OCR run, and the
    manifest then prefers its lower-confidence sibling or drops the page to the
    structure-class floor.

    Term (a) of ``table_truncated`` reads the candidate's own markdown and is NOT
    disabled -- pinned here by a reading whose final row breaks the block's pipe style,
    which is still flagged on a flagged page.
    """
    from socr.core.manifest import _truncated_grid_reading_ids

    words = _bands(20)
    accepted = _bo(_reading(18, "-100.0"))
    sibling = _bo(_reading(20, "100.0"))

    def ids(*, positive: bool, scan_failed: bool = False, outs=(accepted, sibling)):
        ps = _make_state(Path("x.pdf")).pages[1]
        ps.best_output = outs[0]
        ps.attempts = list(outs)
        ps.native_words = words
        _flag(ps, positive=positive, scan_failed=scan_failed)
        return _truncated_grid_reading_ids(ps)

    assert id(accepted) in ids(positive=False), (
        "control: the layer does not call the accepted reading truncated, so nothing "
        "below is a difference"
    )
    assert ids(positive=True) == frozenset(), "a positive finding still scored on the layer"
    assert ids(positive=False, scan_failed=True) == frozenset(), (
        "an unknown layer still scored the reading"
    )

    # Term (a) survives: a final row that breaks the block's own pipe style.
    ragged = _bo(_reading(18, "-100.0").rstrip("\n").rsplit("\n", 1)[0] + "\n| Row17 | 117.0")
    flagged_ids = ids(positive=True, outs=(ragged, sibling))
    assert id(ragged) in flagged_ids, (
        "the markdown-only truncation term was disabled along with the layer; a flag "
        "must never remove a check that does not read the layer"
    )


def test_no_floor_loses_the_reading_when_the_scorer_abstains() -> None:
    """The task's second half: show that abstaining does not cost the page its text.

    Abstention returns an EMPTY set, which is the same value the scorer returns on a
    page with fewer than two candidates and on a page where every candidate truncates
    (TICKET-A2's own "if it is the only candidate, it still ships, flagged" clause). An
    empty set is what every consumer treats as "nothing to prefer away from", so the
    accepted reading stays the winner and no floor is reached.
    """
    from socr.core.manifest import _truncated_grid_reading_ids

    words = _bands(20)
    accepted = _bo(_reading(18, "-100.0"))
    sibling = _bo(_reading(20, "100.0"))

    def page(*, positive: bool):
        ps = _make_state(Path("x.pdf")).pages[1]
        ps.best_output = accepted
        ps.attempts = [accepted, sibling]
        ps.native_words = words
        _flag(ps, positive=positive)
        return ps

    clean, flagged = page(positive=False), page(positive=True)

    # The clause every consumer reads is "is the winner's id in this set". On the clean
    # page it is, which is what makes the page reach the S1 branch at all; on the
    # flagged page the set is empty, the clause is False, and there is nothing for the
    # floor or for the prefer-the-sibling path to act on.
    assert id(accepted) in _truncated_grid_reading_ids(clean)
    assert _truncated_grid_reading_ids(flagged) == frozenset()
    assert id(accepted) not in _truncated_grid_reading_ids(flagged)

    # And the abstention's value is not a new state: it is the SAME empty set the
    # scorer already returns when fewer than two candidates exist, which every consumer
    # has always treated as "nothing to prefer away from".
    solo = page(positive=False)
    solo.attempts = [accepted]
    solo.best_output = accepted
    assert _truncated_grid_reading_ids(solo) == frozenset()


# ---------------------------------------------------------------------------
# 3. Argon P0 1.3 -- header repair rewrites the SHIPPED bytes from the layer
# ---------------------------------------------------------------------------


class _RecordingInner:
    def __init__(self) -> None:
        self.calls = 0

    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        self.calls += 1
        return AcceptDecision(accept=True, reason="inner judge accepted", confidence=0.6)


def _judge_arm(page, markdown: str, *, suspect: bool, found: bool, layer_raises: bool = False):
    from socr.pipeline.agentic import NativeTableVerifierJudge

    output = PageOutput(
        page_num=1, text=markdown, status=PageStatus.SUCCESS, engine="qwen", audit_passed=True
    )
    inner = _RecordingInner()

    def found_fn(_n):
        if layer_raises:
            raise RuntimeError("trust source unavailable")
        return found

    decision = NativeTableVerifierJudge(
        inner=inner,
        get_fitz_page=lambda _n: page,
        is_table_page=lambda _n: True,
        native_evidence_suspect=lambda _n: suspect,
        native_layer_found=found_fn,
    ).assess(output, PROFILE_QWEN_LOCAL)
    return decision, output, inner


def test_the_layer_never_rewrites_the_shipped_header() -> None:
    """``_maybe_repair_collapsed_headers`` mutates ``output.text`` -- the shipped bytes.

    It takes the replacement header words from ``fitz_page.get_text("words")``. On a
    page whose layer may not serve as evidence that is the layer AUTHORING shipped
    content, which is a stronger act than grading one, and round 1 left it open on a
    scan-failed page. Argon's own fixture, all three flag states.

    The clean arm is also the measurement last round could not make: there the repair
    runs AND the page is accepted at confidence 1.0 with no judge consulted, so the
    unjudged ``EXACT_PASS`` ship is reachable after all. That is recorded in the log.
    """
    from test_header_repair import _collapsed_exchange_rate_md, _make_probability_table_page

    from socr.core.verdict import EXACT_PASS_ACCEPT_REASON

    page = _make_probability_table_page()
    markdown = _collapsed_exchange_rate_md()

    clean, clean_out, clean_inner = _judge_arm(page, markdown, suspect=False, found=False)
    assert clean_out.text != markdown, (
        "control: the repair does not fire on this fixture, so the arms below prove nothing"
    )
    assert clean.reason == EXACT_PASS_ACCEPT_REASON and clean.confidence == 1.0
    assert clean_inner.calls == 0

    for label, suspect, found in (
        ("failed scan", True, False),
        ("positive finding", True, True),
    ):
        _d, out, _inner = _judge_arm(page, markdown, suspect=suspect, found=found)
        assert out.text == markdown, f"{label}: the suspect layer rewrote the shipped header"


def test_the_judge_fails_closed_when_the_trust_source_raises() -> None:
    """Argon P0 2 / r-uben M13: ``layer_found = True`` on an exception failed OPEN.

    ``True`` means "this layer IS established as an earlier OCR run", which SKIPS native
    verification -- so a raising trust source bought the page a bypass. Fail closed
    instead: the page is unknown, it is verified, its header is not rewritten, and it
    gets no ``EXACT_PASS``. Pinned against the ``found=True`` arm, which does bypass.
    """
    doc = fitz.open(_prior_ocr_table_pdf(Path("scratch/1205r1/tmp_failclosed")))
    try:
        page = doc[0]
        # The layer reads ``350`` where the page says ``-350``, so the verifier
        # CONTRADICTS this correct reading. That contradiction is the only thing that
        # distinguishes "verification ran" from "verification was skipped".
        raised, _raised_out, raised_inner = _judge_arm(
            page, _TRUE_MD, suspect=True, found=False, layer_raises=True
        )
        bypassed, _bypassed_out, bypassed_inner = _judge_arm(
            page, _TRUE_MD, suspect=True, found=True
        )

        assert bypassed_inner.calls == 1 and not bypassed.reason.startswith(
            "native_table_verifier:"
        ), "control: an ESTABLISHED layer does bypass verification, which is the arm above"

        assert raised.reason.startswith("native_table_verifier:"), (
            "a raising trust source still bought the page a bypass: it was treated as "
            "ESTABLISHED on the strength of a failure, which is the fail-open direction"
        )
        assert raised_inner.calls == 0
    finally:
        doc.close()


def test_the_withheld_exact_pass_exit_still_runs_the_structural_gate() -> None:
    """r-uben M20: nothing pinned that the new deferral passes through the gate.

    The exit #1205 added returns the inner judge's decision. Every other accepting path
    out of ``assess`` runs ``_apply_structural_gate`` on the way, and a path that does
    not would reopen GH-162/GH-200: an accepting inner judge ships a structurally
    defective grid with no deterministic term having run.

    Pinned by OBSERVING the call rather than by its effect, and the reason is recorded
    rather than hidden: a candidate that both reaches this exit and carries a defect
    could not be constructed. The exit needs ``EXACT_PASS``, which requires the row
    counts to match and the numeric multiset to be clean against the layer, and every
    candidate that satisfies that was defect-free on both fixtures tried -- the #1205
    sign-flip page (header attribution ``unverifiable``, no defect) and
    ``_make_probability_table_page`` (an all-empty table gives
    ``table_content_empty`` but ``AMBIGUOUS``, never ``EXACT_PASS``). So the pin is
    that the gate RUNS, with the arguments that let it do its work, and the control is
    the neighbouring exits.
    """
    import fitz

    from socr.pipeline.agentic import NativeTableVerifierJudge

    doc = fitz.open(_prior_ocr_table_pdf(Path("scratch/1205r1/tmp_m20_pin")))
    try:
        page = doc[0]
        calls: list = []
        real = NativeTableVerifierJudge._apply_structural_gate

        def spy(self, decision, output, page_num, words, rules, **kwargs):
            calls.append({"words_given": words is not None, "rules_given": rules is not None})
            return real(self, decision, output, page_num, words, rules, **kwargs)

        with patch.object(NativeTableVerifierJudge, "_apply_structural_gate", spy):
            # The withheld-EXACT_PASS exit: scan-failed, and a reading the layer
            # EXACT_PASSes. Asserted to be that exit by the event it emits.
            _d, _out, inner = _judge_arm(page, _LAYER_MD, suspect=True, found=False)
            assert inner.calls == 1, "this is not the withheld-EXACT_PASS exit"
            assert calls and calls[-1]["words_given"], (
                "the withheld-EXACT_PASS exit returned without the structural gate, or "
                "without the page's words, so no deterministic term could run"
            )

            before = len(calls)
            _judge_arm(page, _LAYER_MD, suspect=False, found=False)
            assert len(calls) > before, "control: the gate is not reached on the clean exit"
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# 6. r-uben should-fix 2 -- the native_layer_found wiring at the real caller
# ---------------------------------------------------------------------------


def test_the_native_layer_found_wiring_is_pinned_at_the_real_caller(tmp_path: Path) -> None:
    """r-uben should-fix 2 / mutant M15: deleting the kwarg passed 650 tests.

    Every judge test builds ``NativeTableVerifierJudge`` directly with a lambda, so none
    went through ``_build_page_judge``. Without the wiring, a positive-finding page with
    an invisible render-mode-3 layer falls through to the opaque-raster proxy, which
    answers False, and the page is verified natively exactly as on main: the correct
    ``-350`` is rejected by the layer's ``350``, with no judge consulted.

    The OUTER ``SourceEvidenceTableJudge`` has to be made to defer first, or it answers
    the page itself and the native judge is never reached -- measured: on this fixture
    it rejects with ``source_evidence_table: numeric tokens unsupported``, because the
    layer is now excluded (fix 1) and an invisible-text page offers no pixel evidence.
    Deferral is one of its real endings (a page with native words it is willing to hand
    inward), so it is patched to that ending rather than to something it never returns.
    """
    import socr.tables.source_evidence as se

    from socr.tables.source_evidence import SourceEvidenceResult

    def _defer(_page, _text, **_kwargs):
        return SourceEvidenceResult(
            verifiable=True, passed=True, reason="deferred for this test", deferred=True
        )

    def decide(*, flagged: bool):
        pipeline = _make_pipeline()
        state = _make_state(_prior_ocr_table_pdf(tmp_path / f"wiring-{flagged}"))
        ps = state.pages[1]
        ps.has_tables = True
        _flag(ps, positive=flagged)
        judge = pipeline._build_page_judge(state)
        with patch.object(se, "verify_scanned_table", _defer):
            return judge.assess(
                PageOutput(
                    page_num=1,
                    text=_TRUE_MD,
                    status=PageStatus.SUCCESS,
                    engine="qwen",
                    audit_passed=True,
                ),
                PROFILE_QWEN_LOCAL,
            )

    unflagged = decide(flagged=False)
    assert unflagged.reason.startswith("native_table_verifier:"), (
        "control: the native verifier is not reached through the real stack, so the "
        f"assertion below proves nothing: {unflagged.reason!r}"
    )

    flagged = decide(flagged=True)
    assert not flagged.reason.startswith("native_table_verifier:"), (
        "the suspect layer decided the page through the real judge stack: the "
        "``native_layer_found`` wiring is not reaching it"
    )


def test_the_source_evidence_judge_excludes_the_layer_when_its_source_raises() -> None:
    """The exception arm of fix 1's wiring, which a mutant found unpinned.

    A trust source that RAISES is not a licence to use the layer. Observed by capturing
    what reaches ``verify_scanned_table``, with the non-raising arm as the control.
    """
    import socr.tables.source_evidence as se

    from socr.pipeline.agentic import SourceEvidenceTableJudge
    from socr.tables.source_evidence import SourceEvidenceResult

    captured: list = []

    def spy(_page, _text, **kwargs):
        captured.append(kwargs.get("layer_is_evidence"))
        return SourceEvidenceResult(verifiable=True, passed=True, reason="stub", deferred=True)

    def raises(_n):
        raise RuntimeError("trust source unavailable")

    for layer_fn in (lambda _n: True, raises):
        judge = SourceEvidenceTableJudge(
            inner=_RecordingInner(),
            get_fitz_page=lambda _n: object(),
            layer_is_evidence=layer_fn,
        )
        with patch.object(se, "verify_scanned_table", spy):
            judge.assess(
                PageOutput(
                    page_num=1,
                    text=_LAYER_MD,
                    status=PageStatus.SUCCESS,
                    engine="qwen",
                    audit_passed=True,
                ),
                PROFILE_QWEN_LOCAL,
            )

    assert captured == [True, False], (
        f"a raising trust source did not exclude the layer: {captured}"
    )
