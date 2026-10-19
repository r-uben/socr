"""#1205: the table chain's remaining readers of a suspect text layer.

#1156 Part A ruled that a page whose text layer is an EARLIER OCR run's reading is
never EVIDENCE: nothing may pass, corroborate, grade or credential a reading on the
strength of that layer. #1202 gated four readers on ``layer_is_admissible_evidence``;
the reviews named several more, in three decision classes, and this file pins the rest.

The design is PER-SITE gating on that one predicate, and the route not taken is recorded
in ``docs/log/2026-10-18_1205-table-chain-readers.md``: a text-free page handle was built
and dropped, because the table phase opens the page in four separate places (and the
witness path takes a PATH, not a page), so the handle was three wraps rather than one
entry point -- and with the per-site guards in place, deleting any of the three changed
no outcome at all.

Every pin here is a DIFFERENCE run twice in one process, changing only the flag, with
a control arm in which the layer DOES count: no absolute outcome measured on this
machine is pinned (CLAUDE.md, #253/#257). Nothing here is pinned by grepping the
source -- that is how #1202's round-3 guard shipped a crash that read as green.

Hermetic: no provider, no model, no ollama. Every reader driven is deterministic
PyMuPDF + arithmetic, and ``process()`` is never entered, so neither
``_available_engines_for_agentic`` nor ``resolve_auto_engine`` is reachable from here.
"""

from __future__ import annotations

import glob
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fitz
import pytest
from test_ladder_binding_evidence import (  # noqa: E402
    _bo,
    _events_of_kind,
    _fake_open_pdf,
    _make_config,
    _make_pipeline,
    _make_state,
)

from socr.core.audit_log import AuditEvent
from socr.core.result import PageOutput, PageStatus
from socr.core.tables_trust import build_tables_trust

_CORPUS = Path("/Users/rubenffuertes/repos/tools/socr/scratch/routing-baseline/corpus/pdfs")

# ---------------------------------------------------------------------------
# Fixtures
#
# One shape throughout: a fully ruled two-row table whose NATIVE layer carries
# ``350`` where the page's real value is ``-350``. That is the measured #1156
# round-3 defect (a prior OCR run dropped a minus sign) and Argon's #1207 round-3
# escalation case in one fixture: the layer grades the CORRECT reading down and the
# layer-matching one up.
# ---------------------------------------------------------------------------

#: What the earlier OCR run read: the minus sign is gone.
_LAYER_VALUES = ("350", "200", "300", "400")
#: What the page actually says, and what a correct model reading emits.
_TRUE_VALUES = ("-350", "200", "300", "400")


def _table_md(values: tuple[str, ...]) -> str:
    return (
        "|      | OLS | IV  |\n| ---- | --- | --- |\n"
        f"| RowA | {values[0]} | {values[1]} |\n"
        f"| RowB | {values[2]} | {values[3]} |\n"
    )


_LAYER_MD = _table_md(_LAYER_VALUES)
_TRUE_MD = _table_md(_TRUE_VALUES)


def _draw_table(page, x0: float, y0: float, values, *, render_mode: int = 0) -> None:
    cols = [x0, x0 + 60, x0 + 120]
    rows = [y0, y0 + 22, y0 + 44]
    right = x0 + 180
    kw = {"fontsize": 9, "render_mode": render_mode}
    page.insert_text((cols[1] + 4, rows[0] + 12), "OLS", **kw)
    page.insert_text((cols[2] + 4, rows[0] + 12), "IV", **kw)
    for i, label in enumerate(("RowA", "RowB")):
        page.insert_text((cols[0] + 4, rows[i + 1] + 12), label, **kw)
        page.insert_text((cols[1] + 4, rows[i + 1] + 12), values[2 * i], **kw)
        page.insert_text((cols[2] + 4, rows[i + 1] + 12), values[2 * i + 1], **kw)
    for yy in [*rows, rows[-1] + 22]:
        page.draw_line((x0, yy), (right, yy))
    for xx in [*cols, right]:
        page.draw_line((xx, rows[0]), (xx, rows[-1] + 22))


def _prior_ocr_table_pdf(tmp_path: Path, name: str = "prior_ocr.pdf") -> Path:
    """One ruled table whose native layer says ``350`` where the page says ``-350``.

    The layer is written render mode 3 (invisible), the shape the detector's #961
    route finds, and the page also carries a VISIBLE running header -- the furniture
    a flagged page is allowed to keep (#1156's detector rule). That header is what
    makes the rendering pins below non-vacuous.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 60), "FEDERAL RESERVE BULLETIN", fontsize=11)
    _draw_table(page, 100, 120, _LAYER_VALUES, render_mode=3)
    pdf_path = tmp_path / name
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _no_furniture_pdf(tmp_path: Path, name: str = "no_furniture.pdf") -> Path:
    """:func:`_prior_ocr_table_pdf` with the visible header omitted.

    The canary for the rendering pins: if the header contributed no pixels,
    "byte-identical" would be true of a blank page too.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    _draw_table(page, 100, 120, _LAYER_VALUES, render_mode=3)
    pdf_path = tmp_path / name
    doc.save(pdf_path)
    doc.close()
    return pdf_path


#: Two SIDE-BY-SIDE tables, so ``locate_tables`` returns exactly two boxes and the
#: witness pairing corroboration (n >= 2, count matched) actually runs.
_LEFT_VALUES = ("11", "12", "13", "14")
_RIGHT_VALUES = ("21", "22", "23", "24")


def _two_table_pdf(tmp_path: Path, name: str = "two_tables.pdf") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 60), "FEDERAL RESERVE BULLETIN", fontsize=11)
    _draw_table(page, 60, 120, _LEFT_VALUES, render_mode=3)
    _draw_table(page, 330, 120, _RIGHT_VALUES, render_mode=3)
    pdf_path = tmp_path / name
    doc.save(pdf_path)
    doc.close()
    return pdf_path


_SWAPPED_BLOCKS = _table_md(_RIGHT_VALUES) + "\nprose between\n\n" + _table_md(_LEFT_VALUES)
_CORRECT_BLOCKS = _table_md(_LEFT_VALUES) + "\nprose between\n\n" + _table_md(_RIGHT_VALUES)


def _flag(ps, *, positive: bool = False, scan_failed: bool = False) -> None:
    """Set the #1156 flags on *ps*: a POSITIVE finding, a FAILED scan, or neither."""
    ps.prior_ocr_layer = positive
    ps.prior_ocr_layer_scan_failed = scan_failed
    ps.invisible_text_over_raster = False
    ps.prior_ocr_layer_suspect = positive or scan_failed


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# ---------------------------------------------------------------------------
# 1. The view itself
# ---------------------------------------------------------------------------


def _corpus_pages(limit: int = 8):
    paths = sorted(glob.glob(str(_CORPUS / "*.pdf")))[:limit]
    if not paths:
        pytest.skip(f"corpus not present at {_CORPUS}")
    for path in paths:
        doc = fitz.open(path)
        try:
            yield Path(path).name, doc[0]
        finally:
            doc.close()


def test_the_one_predicate_takes_the_union(tmp_path: Path) -> None:
    """The single question every site asks, in all three flag states.

    An EVIDENCE decision takes the UNION (``layer_is_admissible_evidence``,
    scan-failed included) -- the owner's ruling of 2026-10-18. A page with no
    ``PageState`` keeps today's answer, which is what every caller outside a live run
    relies on.
    """
    pipeline = _make_pipeline()
    ps = _make_state(Path("x.pdf")).pages[1]

    _flag(ps)
    assert pipeline._table_layer_is_evidence(ps) is True
    _flag(ps, scan_failed=True)
    assert pipeline._table_layer_is_evidence(ps) is False, (
        "an 'unknown' layer may not serve as evidence either: refusing evidence is "
        "never destructive, and only WITHHOLDING the layer's own text needs a finding"
    )
    _flag(ps, positive=True)
    assert pipeline._table_layer_is_evidence(ps) is False
    assert pipeline._table_layer_is_evidence(None) is True


def test_the_flagged_page_still_renders_its_visible_furniture(tmp_path: Path) -> None:
    """Nothing in this ticket touches pixels, and that is pinned rather than asserted.

    The route NOT taken would have handed the chain a page copy with the text layer
    stripped out. The design note measured that pixel-identical on 60 of 60 page/DPI
    cells -- but only because those pages' layers are entirely hidden, and the
    detector's own rule admits a page whose VISIBLE words all sit at repeated
    furniture slots. Stripping such a page removes its running header.

    So the flagged fixture carries a visible running header, and the same page without
    it renders differently at every DPI. That is the canary for the crop byte-identity
    pin in :func:`test_the_layer_no_longer_supplies_the_pairing_swap_verdict`, which is
    the one place an abstention here could have reached a render.
    """
    doc = fitz.open(_prior_ocr_table_pdf(tmp_path / "render"))
    bare = fitz.open(_no_furniture_pdf(tmp_path / "render"))
    try:
        for dpi in (72, 150, 300):
            assert doc[0].get_pixmap(dpi=dpi).samples != bare[0].get_pixmap(dpi=dpi).samples, (
                f"canary at {dpi} dpi: the visible furniture contributes no pixels"
            )
        assert doc[0].get_text("words"), "and the flagged page does carry a text layer"
    finally:
        doc.close()
        bare.close()


# ---------------------------------------------------------------------------
# 2. Class (a): the layer GRADES a reading
# ---------------------------------------------------------------------------


def _scoring_arm(tmp_path: Path, *, positive: bool, scan_failed: bool = False):
    """Drive ``_surface_table_scoring`` -- the real caller of the escalation scorer.

    Returns ``(needs_escalation, state)``.
    """
    tag = f"{'pos' if positive else 'scanfail' if scan_failed else 'clean'}"
    pipeline = _make_pipeline()
    state = _make_state(_prior_ocr_table_pdf(tmp_path / tag))
    ps = state.pages[1]
    ps.has_tables = True
    # ``table_not_scorable`` is gated on the STRUCTURAL table signal, not on the
    # lane-cooccurrence heuristic (GH-655 E2), so both are set: the caller gates on
    # ``has_tables`` and the distrust surface on ``detected_table_count``.
    ps.detected_table_count = 1
    _flag(ps, positive=positive, scan_failed=scan_failed)
    bo = _bo(_TRUE_MD)
    return pipeline._surface_table_scoring(state, 1, ps, bo), state


def test_the_layer_no_longer_grades_a_correct_reading(tmp_path: Path) -> None:
    """The scorer's own direction, at ``_surface_table_scoring``.

    Control: the correct ``-350`` scores 75% against the layer's ``350``, so the page
    is marked as needing escalation -- the layer has graded a correct reading down.
    Flagged: there is no ground truth, the page is not graded, and the absence
    SURFACES as ``table_not_scorable`` rather than passing silently.
    """
    needs_clean, clean_state = _scoring_arm(tmp_path, positive=False)
    assert needs_clean, (
        "control: the layer did not grade the correct reading down, so the flagged arm "
        "below proves nothing"
    )
    assert _events_of_kind(clean_state, "table_not_scorable") == []

    needs_flagged, flagged_state = _scoring_arm(tmp_path, positive=True)
    assert not needs_flagged, "the suspect layer still graded the page"
    assert _events_of_kind(flagged_state, "table_not_scorable"), (
        "the page lost its ground truth silently; the distrust event is the surface"
    )


def test_a_failed_scan_abstains_as_well(tmp_path: Path) -> None:
    """Round 2, the owner's ruling of 2026-10-18, and it reverses round 1 here.

    #1156 Part A's split is by ACTION, not by site: an EVIDENCE decision takes the
    UNION (`layer_is_admissible_evidence`, scan-failed included), because refusing
    evidence is never destructive and "unknown" is no reason to trust the words
    either; only WITHHOLDING the layer's own text takes the positive finding. Round 1
    keyed every reader on the positive finding, so a page whose DETECTION SCAN FAILED
    still graded a correct reading down and promoted the candidate that matched its
    layer. Pinned here in all three states, against the clean control.
    """
    needs_clean, clean_state = _scoring_arm(tmp_path, positive=False)
    assert needs_clean, "control: a clean page must still be graded"
    assert _events_of_kind(clean_state, "table_not_scorable") == []

    for label, kwargs in (
        ("failed scan", {"positive": False, "scan_failed": True}),
        ("positive finding", {"positive": True}),
    ):
        needs, state = _scoring_arm(tmp_path, **kwargs)
        assert not needs, f"{label}: the layer still graded the page"
        assert _events_of_kind(state, "table_not_scorable"), f"{label}: the loss is silent"


# ---------------------------------------------------------------------------
# 3. Argon's #1207 round-3 escalation case
# ---------------------------------------------------------------------------


class _FixedProvider:
    """A ``run_provider`` that answers with the LAYER's reading, no model involved."""

    def __init__(self, engine: str) -> None:
        self._engine = engine
        self.calls = 0

    def __call__(self, profile, page_num):
        self.calls += 1
        return PageOutput(
            page_num=page_num,
            text=_LAYER_MD,
            status=PageStatus.SUCCESS,
            engine=self._engine,
            audit_passed=True,
        )


def _escalation_arm(tmp_path: Path, *, positive: bool, needs_escalation, scan_failed=False):
    from socr.core.providers import PROFILE_QWEN_LOCAL

    tag = f"esc-{'pos' if positive else 'scanfail' if scan_failed else 'clean'}-{needs_escalation}"
    pipeline = _make_pipeline(_make_config(cost_budget=0.0))
    state = _make_state(_prior_ocr_table_pdf(tmp_path / tag))
    ps = state.pages[1]
    ps.has_tables = True
    ps.detected_table_count = 1
    _flag(ps, positive=positive, scan_failed=scan_failed)
    bo = _bo(_TRUE_MD)
    ps.best_output = bo
    provider = _FixedProvider(PROFILE_QWEN_LOCAL.engine.value)
    _withheld, out = pipeline._escalate_table_page(
        state,
        1,
        ps,
        bo,
        PROFILE_QWEN_LOCAL,
        provider,
        state.handle.path,
        needs_escalation=needs_escalation,
        abandoned=[],
    )
    return out, state, provider


def test_a_second_reading_is_no_longer_promoted_by_scoring_100pct_against_the_layer(
    tmp_path: Path,
) -> None:
    """Argon's #1207 round-3 case, end to end at ``_escalate_table_page``.

    Control: the incumbent is correct (``-350``), the escalation candidate reproduces
    the layer's ``350``, it scores 100% against the layer where the incumbent scores
    75%, and it is ACCEPTED -- a wrong number replaces a right one, on the strength of
    an earlier OCR run's reading of the same pixels.

    Flagged, both arms of the gate:

    * ``needs_escalation=None`` -- the scorer is asked and abstains, so no provider
      call is made at all;
    * ``needs_escalation=True`` -- the escalation is FORCED past the scorer, the
      candidate is produced, and ``decide_escalation`` still refuses it: no ground
      truth for the exactness gate and no value oracle for the canary.

    The second arm is the one that matters: it pins that the layer no longer GRADES,
    rather than merely that this page is no longer escalated.
    """
    clean_out, clean_state, clean_provider = _escalation_arm(
        tmp_path, positive=False, needs_escalation=True
    )
    assert clean_provider.calls == 1
    assert _events_of_kind(clean_state, "table_escalation_accepted"), (
        "control: the layer-matching candidate was not promoted, so nothing below is a difference"
    )
    assert clean_out.text == _LAYER_MD
    assert "-350" not in clean_out.text, "control: the correct value survived; fixture is wrong"

    forced_out, forced_state, forced_provider = _escalation_arm(
        tmp_path, positive=True, needs_escalation=True
    )
    assert forced_provider.calls == 1, "the forced arm must still ASK, or it pins the wrong thing"
    assert _events_of_kind(forced_state, "table_escalation_accepted") == []
    rejected = _events_of_kind(forced_state, "table_escalation_rejected")
    assert rejected and (rejected[0].data or {}).get("gate") == "none", (
        f"the candidate was judged against the suspect layer: {[e.detail for e in rejected]}"
    )
    assert forced_out.text == _TRUE_MD

    asked_out, _asked_state, asked_provider = _escalation_arm(
        tmp_path, positive=True, needs_escalation=None
    )
    assert asked_provider.calls == 0
    assert asked_out.text == _TRUE_MD


def test_a_forced_escalation_is_not_judged_against_an_unknown_layer(tmp_path: Path) -> None:
    """Round 2: the same refusal on a page whose DETECTION SCAN merely failed.

    Such a page gets no text-free view (nothing has established there is an earlier
    OCR run to act on), so the refusal has to be its own guard before
    ``decide_escalation`` — otherwise the candidate is graded against a layer that
    was never cleared. The candidate is still produced and still billed; it is
    refused on the record and the incumbent ships.
    """
    out, state, provider = _escalation_arm(
        tmp_path, positive=False, needs_escalation=True, scan_failed=True
    )
    assert provider.calls == 1, "the provider must still be asked, or this pins nothing"
    assert _events_of_kind(state, "table_escalation_accepted") == []
    rejected = _events_of_kind(state, "table_escalation_rejected")
    assert rejected and (rejected[0].data or {}).get("gate") == "none"
    assert out.text == _TRUE_MD


# ---------------------------------------------------------------------------
# 4. Class (b): the layer CREDENTIALS a change of numbers (crop repair)
# ---------------------------------------------------------------------------


def _crop_arm(
    tmp_path: Path,
    *,
    positive: bool,
    scan_failed: bool = False,
    pdf=None,
    incumbent: str = None,
    crop_md: str = None,
    tag: str = "",
):
    from socr.tables.extract import CropTable

    tag = tag or f"crop-{'pos' if positive else 'scanfail' if scan_failed else 'clean'}"
    pipeline = _make_pipeline(_make_config(auto_patch_tables=False))
    state = _make_state((pdf or _prior_ocr_table_pdf)(tmp_path / tag))
    ps = state.pages[1]
    ps.has_tables = True
    _flag(ps, positive=positive, scan_failed=scan_failed)
    bo = _bo(_TRUE_MD if incumbent is None else incumbent)
    ps.best_output = bo
    crop = CropTable(
        markdown=_LAYER_MD if crop_md is None else crop_md,
        source="ruled",
        bbox=(100.0, 120.0, 280.0, 186.0),
    )
    patched, _flagged = pipeline._reread_page_tables(state, 1, [crop], extractor=object())
    return patched, ps.best_output.text, state


def test_a_crop_patch_is_no_longer_credentialled_by_the_layer(tmp_path: Path) -> None:
    """``crop_repair``'s two layer reads, at ``_reread_page_tables``.

    ``auto_patch_tables`` is OFF, so the only way the crop's numbers can replace the
    page's is the fallback: ``page_needs_crop_repair_fallback`` (the layer hard-fails
    the correct reading) and then ``crop_patch_improves_verification`` (the layer says
    the crop's reading verifies better). Control: both fire and the correct ``-350``
    is replaced by the layer's ``350``. Flagged: there is nothing to verify against,
    so neither fires and the page keeps its own reading.
    """
    clean_patched, clean_text, clean_state = _crop_arm(tmp_path, positive=False)
    assert clean_patched == 1, "control: the layer did not credential the patch"
    assert (
        build_tables_trust(clean_state.handle.filename, clean_state.events).untrusted_pages == []
    ), (
        "control: the clean page is already untrusted for another reason, so the "
        "sidecar assertions below are not evidence of this refusal"
    )
    assert "-350" not in clean_text

    for label, kwargs in (
        ("positive finding", {"positive": True}),
        ("failed scan", {"positive": False, "scan_failed": True}),
    ):
        patched, text, state = _crop_arm(tmp_path, **kwargs)
        assert patched == 0, f"{label}: the layer still credentialled a change of numbers"
        assert text == _TRUE_MD
        assert _events_of_kind(state, "table_crop_repair_unverifiable"), (
            f"{label}: the patch was refused silently"
        )
        # The event is not the surface: ``tables_trust.json`` is, and a kind emitted
        # but never registered in ``TABLE_DISTRUST_KINDS`` does not reach it. This
        # file's own history is the precedent -- GH-166's crop TIMEOUT kind shipped
        # that way and the page read as verified.
        #
        # Pinned on the REASON, not on the page being listed: ``dualpass_flagged``
        # also fires here and is already registered, so the page is untrusted either
        # way. What the registration buys is the consumer being told WHY -- the crop
        # reading went unverified, rather than only that a crop disagreed -- and
        # ``untrusted_pages`` cannot show that.
        trust = build_tables_trust(state.handle.filename, state.events)
        assert trust.untrusted_pages == [1], f"{label}: the page is not flagged at all"
        assert "table_crop_repair_unverifiable" in trust.counts_by_kind(), (
            f"{label}: the refusal's own reason never reached tables_trust.json: "
            f"{sorted(trust.counts_by_kind())}"
        )


def test_the_crop_refusal_withholds_a_table_acceptance_credential() -> None:
    """The written decision #713's drift guard demands for a new distrust kind.

    A table-acceptance credential stands in for a page judge that TIMED OUT: it
    asserts every table on the page was accepted by the ladder. This refusal says a
    crop reading of the page's table exists and nothing verified it, because the only
    thing that could have approved it is a layer that may not serve as evidence. Same
    family as ``dualpass_crop_timeout`` / ``dualpass_crop_failed``, which both block
    for the same reason -- partial or absent crop verification -- so it blocks too.
    Nothing retires it today.

    Pinned at the predicate the credential actually consults, with the control that an
    unrelated kind on the same page does NOT block.
    """
    from socr.core.manifest import (
        CREDENTIAL_BLOCKING_EVENT_KINDS,
        CREDENTIAL_NON_BLOCKING_EVENT_KINDS,
        _credential_admission_refusal,
    )

    assert "table_crop_repair_unverifiable" in CREDENTIAL_BLOCKING_EVENT_KINDS
    assert "table_crop_repair_unverifiable" not in CREDENTIAL_NON_BLOCKING_EVENT_KINDS

    ps = _make_state(Path("x.pdf")).pages[1]
    ps.table_ladder_disposition = None

    def refusal(kind: str) -> str:
        state = SimpleNamespace(events=[AuditEvent(page_num=1, kind=kind)])
        return _credential_admission_refusal(state, ps, 1, _bo(_TRUE_MD))

    assert "table_crop_repair_unverifiable" in refusal("table_crop_repair_unverifiable"), (
        "the refusal does not withhold the credential"
    )
    assert refusal("dualpass_patched") == "", (
        "control: every kind blocks, so the assertion above is not about this one"
    )


#: A table whose native layer AGREES with the correct reading, so the only thing that
#: can make a crop patch look like an improvement is the MARKDOWN's header shape.
_PLAIN_VALUES = ("350", "200", "11", "300", "400", "12")
_PLAIN_COLLAPSED_MD = "| OLS |\n| --- |\n| RowA | 350 | 200 | 11 |\n| RowB | 300 | 400 | 12 |\n"
_PLAIN_FULL_MD = (
    "|      | OLS | IV | SE |\n| ---- | --- | --- | --- |\n"
    "| RowA | 350 | 200 | 11 |\n| RowB | 300 | 400 | 12 |\n"
)
_PLAIN_SWAPPED_MD = (
    "|      | OLS | IV | SE |\n| ---- | --- | --- | --- |\n"
    "| RowA | 300 | 400 | 12 |\n| RowB | 350 | 200 | 11 |\n"
)


def _plain_table_pdf(tmp_path: Path, name: str = "plain.pdf") -> Path:
    """Six-value ruled table, layer == the correct reading, text layer invisible."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 60), "FEDERAL RESERVE BULLETIN", fontsize=11)
    cols = [100, 180, 250, 320]
    rows = [120, 142, 164]
    kw = {"fontsize": 9, "render_mode": 3}
    for i, head in enumerate(("OLS", "IV", "SE")):
        page.insert_text((cols[i + 1] + 4, rows[0] + 12), head, **kw)
    for r, label in enumerate(("RowA", "RowB")):
        page.insert_text((cols[0] + 4, rows[r + 1] + 12), label, **kw)
        for c in range(3):
            page.insert_text((cols[c + 1] + 4, rows[r + 1] + 12), _PLAIN_VALUES[3 * r + c], **kw)
    for yy in [*rows, rows[-1] + 22]:
        page.draw_line((100, yy), (400, yy))
    for xx in [*cols, 400]:
        page.draw_line((xx, rows[0]), (xx, rows[-1] + 22))
    pdf_path = tmp_path / name
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_a_crop_patch_is_not_approved_on_header_shape_alone(tmp_path: Path) -> None:
    """Argon's P1 on round 1, and the owner's ruling on it.

    Three of ``crop_patch_improves_verification``'s five severity terms come from the
    markdown alone (header collapse, header gap) and two from the page's text layer
    (hard fail, label-binding failure, plus the lane gap). With the layer gone a crop
    that merely fills in header cells still scores as an improvement — so anything
    else it changed, a row swap included, rides in on the header fix. The ruling: with
    the layer gone the patch needs independent evidence or it stays unverified, and
    never a fall back to the layer.

    Argon's version of this finding said the layer's verifier would have REFUSED the
    row-swapped patch, so round 1 had removed a real check. Measured here rather than
    taken: it would not. On the clean arm below the swapped crop is approved too —
    ``verify_native_table`` returns EXACT_PASS for it, because the numeric multiset is
    identical and only the row binding differs, which is the GH-273 shape this
    verifier is blind to at page scope. So this fix is a STRENGTHENING, not the
    restoration of a lost check, and the third arm is what it is pinned on.

    Three arms on a page whose layer AGREES with the correct reading, so the header
    shape is the only thing that can differ:

    * clean + a crop that only fixes the header — patched. The lane works.
    * clean + a crop that fixes the header AND swaps two rows — patched as well, which
      is the measurement above, pinned so it cannot quietly change under us.
    * flagged + the benign crop — refused, and recorded. Not patched on header shape.
    """
    benign = dict(pdf=_plain_table_pdf, incumbent=_PLAIN_COLLAPSED_MD, crop_md=_PLAIN_FULL_MD)
    swapped = dict(pdf=_plain_table_pdf, incumbent=_PLAIN_COLLAPSED_MD, crop_md=_PLAIN_SWAPPED_MD)

    patched, text, _state = _crop_arm(tmp_path, positive=False, tag="hdr-clean", **benign)
    assert patched == 1, "control: a header-only crop repair does not happen at all here"
    assert text.strip() == _PLAIN_FULL_MD.strip()

    patched, text, _state = _crop_arm(tmp_path, positive=False, tag="hdr-swap", **swapped)
    assert patched == 1, (
        "the layer's verifier has started refusing a row-swapped patch; it did not when "
        "this was measured, and the claim that round 1 removed that check rests on it"
    )
    assert text.strip() == _PLAIN_SWAPPED_MD.strip()

    patched, text, state = _crop_arm(tmp_path, positive=True, tag="hdr-flagged", **benign)
    assert patched == 0, "a patch was approved on header shape alone"
    assert text.strip() == _PLAIN_COLLAPSED_MD.strip()
    assert _events_of_kind(state, "table_crop_repair_unverifiable")


# ---------------------------------------------------------------------------
# 5. Class (b): the layer WITHHOLDS a table (witness pairing corroboration)
# ---------------------------------------------------------------------------


def _witness_statuses(pdf_path: Path, markdown: str, *, layer_is_evidence: bool):
    from socr.tables.witness import prepare_table_witnesses

    with prepare_table_witnesses(
        pdf_path, 1, markdown, layer_is_evidence=layer_is_evidence
    ) as witnesses:
        return [
            (w.status.value, w.scope.value, w.crop_path.read_bytes() if w.crop_path else None)
            for w in witnesses
        ]


def test_the_layer_no_longer_supplies_the_pairing_swap_verdict(tmp_path: Path) -> None:
    """``prepare_table_witnesses`` on a count-matched two-table page.

    Control: the emitted blocks are in the opposite order to the page's boxes, the
    layer's per-box numeric multisets prove the swap, and BOTH tables demote to
    AMBIGUOUS with no crop -- nothing can look at either table, so each reaches the
    ladder UNVERIFIED. Flagged: the evidence is empty, which is NEUTRAL by
    construction, and the index pairing ships with its crops.

    The second control is the point of the first: with the blocks in the RIGHT order
    the page is LOCATED on both arms, so the demotion above is the swap verdict and
    not an artefact of the fixture.
    """
    pdf_path = _two_table_pdf(tmp_path)

    swapped_clean = _witness_statuses(pdf_path, _SWAPPED_BLOCKS, layer_is_evidence=True)
    assert [(s, sc) for s, sc, _ in swapped_clean] == [("ambiguous", "none")] * 2, (
        f"control: the layer did not contradict the pairing: {swapped_clean}"
    )

    swapped_flagged = _witness_statuses(pdf_path, _SWAPPED_BLOCKS, layer_is_evidence=False)
    assert [(s, sc) for s, sc, _ in swapped_flagged] == [("located", "located")] * 2, (
        f"the suspect layer still withheld both tables: {swapped_flagged}"
    )

    for flag in (True, False):
        correct = _witness_statuses(pdf_path, _CORRECT_BLOCKS, layer_is_evidence=flag)
        assert [(s, sc) for s, sc, _ in correct] == [("located", "located")] * 2, (
            f"control: a correctly paired page is not LOCATED at layer_is_evidence={flag}"
        )

    # The rendering pin for the witness lane: the crops the judge is shown are the
    # same bytes on both arms of a page that is LOCATED either way, visible furniture
    # included.
    crops_clean = [
        c for _s, _sc, c in _witness_statuses(pdf_path, _CORRECT_BLOCKS, layer_is_evidence=True)
    ]
    crops_flagged = [
        c for _s, _sc, c in _witness_statuses(pdf_path, _CORRECT_BLOCKS, layer_is_evidence=False)
    ]
    assert all(c for c in crops_clean) and crops_clean == crops_flagged, (
        "the witness crops are not byte-identical; the view reached a RENDER"
    )


def test_the_witness_caller_derives_the_flag_from_the_page_state(tmp_path: Path) -> None:
    """The real caller, ``_run_table_judge_gate``, in both directions.

    Observed by capturing what it actually passes, not by reading the source: a
    hard-coded ``True`` or a flag taken from the wrong predicate fails here.
    """
    captured: list = []

    from socr.tables import witness as witness_mod

    original = witness_mod.prepare_table_witnesses

    def _spy(pdf_path, page_num, markdown, **kwargs):
        captured.append(kwargs.get("layer_is_evidence"))
        return original(pdf_path, page_num, markdown, **kwargs)

    for positive, scan_failed in ((False, False), (False, True), (True, False)):
        pipeline = _make_pipeline()
        state = _make_state(_two_table_pdf(tmp_path / f"spy-{positive}-{scan_failed}"))
        ps = state.pages[1]
        ps.has_tables = True
        _flag(ps, positive=positive, scan_failed=scan_failed)
        bo = _bo(_CORRECT_BLOCKS)
        with patch.object(witness_mod, "prepare_table_witnesses", _spy):
            with patch.object(pipeline, "_transcribe_cell_token", return_value=None):
                pipeline._run_table_judge_gate(state, 1, ps, bo, [])

    assert captured == [True, False, False], (
        f"the caller does not read the UNION, so an unknown layer still decides: {captured}"
    )


# ---------------------------------------------------------------------------
# 6. #1202's own unpinned line: the boundary re-read abstention
# ---------------------------------------------------------------------------


def test_a_boundary_resolution_cannot_be_proven_from_a_suspect_layer(tmp_path: Path) -> None:
    """#1205's last ask: pin the one #1202 guard that shipped with no behavioural test.

    A standing boundary distrust is cleared when a later pass can PROVE the word is
    present in the page's own native words and positively disposed of. On a flagged
    page that proof is the earlier OCR run's, so the guard abstains and the distrust
    stands. Driven through ``_record_unresolved_binding_boundary`` with the same
    known-geometry harness ``test_ladder_binding_evidence`` uses, so the resolution
    does fire on the control arm.
    """
    from socr.judge.table_verdict import (
        TABLE_BINDING_BOUNDARY_RESOLVED_KIND,
        TABLE_BINDING_BOUNDARY_UNRESOLVED_KIND,
    )
    from socr.tables.binding import BindingResult

    word = (390.0, 100.0, 440.0, 110.0, "0.51", 0, 1, 2)
    resolved_counts = {}
    for label, positive, scan_failed in (
        ("clean", False, False),
        ("scan_failed", False, True),
        ("positive", True, False),
    ):
        pipeline = _make_pipeline()
        state = _make_state(_prior_ocr_table_pdf(tmp_path / f"boundary-{label}"))
        witness = SimpleNamespace(table_id="t1")
        _flag(state.pages[1], positive=positive, scan_failed=scan_failed)

        with patch("socr.core.pdf.open_pdf", _fake_open_pdf([word])):
            pipeline._record_unresolved_binding_boundary(
                state, 1, witness, BindingResult(unresolved_boundary_words=[word])
            )
        assert _events_of_kind(state, TABLE_BINDING_BOUNDARY_UNRESOLVED_KIND)

        with patch("socr.core.pdf.open_pdf", _fake_open_pdf([word])):
            pipeline._record_unresolved_binding_boundary(
                state,
                1,
                witness,
                BindingResult(unresolved_boundary_words=[], region_scoped_words=(word,)),
            )
        resolved_counts[label] = len(_events_of_kind(state, TABLE_BINDING_BOUNDARY_RESOLVED_KIND))

    assert resolved_counts["clean"] == 1, (
        "control: no resolution fires on a clean page, so the guard arms prove nothing"
    )
    assert resolved_counts["scan_failed"] == 0, (
        "the #1202 guard takes the UNION, so an 'unknown' layer may not prove a resolution"
    )
    assert resolved_counts["positive"] == 0


# ---------------------------------------------------------------------------
# 7. The P0 Argon found on round 1: the native table verifier judge
# ---------------------------------------------------------------------------


class _RecordingInnerJudge:
    """An accepting inner judge that records whether it was consulted at all."""

    def __init__(self) -> None:
        self.calls = 0

    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        self.calls += 1
        return AcceptDecision(accept=True, reason="inner judge accepted", confidence=0.6)


def _verifier_judge_arm(tmp_path: Path, *, model_text: str, positive: bool, scan_failed: bool):
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline.agentic import NativeTableVerifierJudge

    tag = f"njudge-{'pos' if positive else 'scanfail' if scan_failed else 'clean'}"
    doc = fitz.open(_prior_ocr_table_pdf(tmp_path / tag))
    inner = _RecordingInnerJudge()
    try:
        decision = NativeTableVerifierJudge(
            inner=inner,
            get_fitz_page=lambda _n: doc[0],
            is_table_page=lambda _n: True,
            native_evidence_suspect=lambda _n: positive or scan_failed,
            native_layer_found=lambda _n: positive,
        ).assess(
            PageOutput(
                page_num=1,
                text=model_text,
                status=PageStatus.SUCCESS,
                engine="qwen",
                audit_passed=True,
            ),
            PROFILE_QWEN_LOCAL,
        )
    finally:
        doc.close()
    return decision, inner


def _verifier_events(tmp_path: Path, *, model_text, positive, scan_failed):
    """``_verifier_judge_arm`` with the judge's own emitted events as well."""
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline.agentic import NativeTableVerifierJudge

    tag = f"njev-{'pos' if positive else 'scanfail' if scan_failed else 'clean'}-{len(model_text)}"
    doc = fitz.open(_prior_ocr_table_pdf(tmp_path / tag))
    inner = _RecordingInnerJudge()
    events: list = []
    try:
        decision = NativeTableVerifierJudge(
            inner=inner,
            get_fitz_page=lambda _n: doc[0],
            is_table_page=lambda _n: True,
            record_event=events.append,
            native_evidence_suspect=lambda _n: positive or scan_failed,
            native_layer_found=lambda _n: positive,
        ).assess(
            PageOutput(
                page_num=1,
                text=model_text,
                status=PageStatus.SUCCESS,
                engine="qwen",
                audit_passed=True,
            ),
            PROFILE_QWEN_LOCAL,
        )
    finally:
        doc.close()
    return decision, inner, [getattr(e, "kind", "") for e in events]


def test_the_native_verifier_judge_no_longer_decides_on_a_suspect_layer(tmp_path: Path) -> None:
    """Argon's P0 on round 1, and why my own reasoning for leaving this site was wrong.

    I left this site alone on the ground that the layer could only REJECT here, so the
    worst case was a missing number rather than a wrong one. Half the function does
    that. The other half is ``VerifierState.EXACT_PASS``, an ACCEPTING exit that
    consults no judge (``inner_consulted=False``) and stamps confidence 1.0.

    What is measured here, on a page whose layer reads ``350`` where the page says
    ``-350``:

    * clean + the CORRECT reading: the layer rejects it and the inner judge is never
      consulted. One deterministic check, answered from an earlier OCR run, discards
      the right numbers on its own. This is the control and the harm.
    * POSITIVE finding + either reading: the verifier does not run, so it neither
      rejects the right reading nor passes the wrong one, and the judge decides. #1202
      already gives this page the pixel lane, the visual table ladder and the blind cell
      guard, so this is not an absence of checking.
    * FAILED scan only ("unknown"): the REJECTION half is kept, because that page may
      be genuinely born-digital and refusing cannot ship a wrong number. Only the
      unjudged accept is closed, which is observable as the withheld-EXACT_PASS event.

    NOT measured, and recorded rather than claimed: Argon's second direction is that a
    model reading reproducing the layer's error SHIPS at confidence 1.0 with no judge.
    The exit exists, but on every synthetic ruled table tried here the header-attribution
    chain abstains (``native_header_row`` returns ``None``), and #245 makes an abstain on
    that path delegate to the inner judge — so the layer-matching reading reached a judge
    anyway and the 1.0 ship could not be reproduced. The guard added here closes the exit
    regardless; its firing is pinned by the event, not by a confidence of 1.0.
    """
    rejected, inner, _kinds = _verifier_events(
        tmp_path, model_text=_TRUE_MD, positive=False, scan_failed=False
    )
    assert not rejected.accept, (
        "control: the layer does not reject the correct reading on a clean page, so "
        "nothing below is a difference"
    )
    assert rejected.reason.startswith("native_table_verifier:")
    assert inner.calls == 0, (
        "control: the rejection is the layer's alone; if a judge was consulted this is "
        "not the unjudged decision the finding is about"
    )

    for label, model_text in (("right reading", _TRUE_MD), ("the layer's error", _LAYER_MD)):
        decision, inner, _kinds = _verifier_events(
            tmp_path, model_text=model_text, positive=True, scan_failed=False
        )
        assert inner.calls == 1, f"positive finding, {label}: the judge was not consulted"
        assert not decision.reason.startswith("native_table_verifier:"), (
            f"positive finding, {label}: the suspect layer still decided the page"
        )

    kept, inner, _kinds = _verifier_events(
        tmp_path, model_text=_TRUE_MD, positive=False, scan_failed=True
    )
    assert not kept.accept, (
        "an unknown layer must keep its REJECTION power: that page may be genuinely "
        "born-digital, and refusing cannot ship a wrong number"
    )

    _deferred, inner, kinds = _verifier_events(
        tmp_path, model_text=_LAYER_MD, positive=False, scan_failed=True
    )
    assert inner.calls == 1
    assert "table_value_drift_unadjudicated" in kinds, (
        "the unknown layer's agreement took the unjudged accepting exit"
    )
    _clean, _inner, clean_kinds = _verifier_events(
        tmp_path, model_text=_LAYER_MD, positive=False, scan_failed=False
    )
    assert "table_value_drift_unadjudicated" not in clean_kinds, (
        "control: the withheld-EXACT_PASS event fires on a clean page too, so it is not "
        "evidence of the guard"
    )
