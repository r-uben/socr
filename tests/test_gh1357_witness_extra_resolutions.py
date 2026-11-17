"""#1357: the scanned witness reads further resolutions only when the base pass leaves
numeric table tokens unsupported.

Hermetic: the witness is the injected ``ocr_image_fn`` (no tesseract, no ollama, no
provider). The reader is keyed on the pixmap width, which is the render resolution on a
page of known size. Assertions pin DIFFERENCES between two runs that share a page and a
table and differ in one thing (the ladder, or what one rung reads), never an outcome
measured on a host.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import fitz

import socr.tables.source_evidence as se
from socr.core.audit_log import AuditEvent
from socr.core.result import PageOutput, PageStatus
from socr.pipeline.agentic import AcceptDecision, SourceEvidenceTableJudge
from socr.tables.source_evidence import (
    EXTRA_RESOLUTION_KIND,
    verify_scanned_table,
)

PAGE_W_PT = 500
BASE = se._EVIDENCE_OCR_DPI
LADDER = se._EXTRA_EVIDENCE_OCR_DPIS


def _dpi(pix) -> int:
    return round(pix.width * 72 / PAGE_W_PT)


def _page() -> fitz.Page:
    return fitz.open().new_page(width=PAGE_W_PT, height=700)


def _md(*cells: str, label: str = "row") -> str:
    header = "| v | " + " | ".join(f"c{i}" for i in range(len(cells))) + " |"
    sep = "|" + "---|" * (len(cells) + 1)
    return "\n".join([header, sep, f"| {label} | " + " | ".join(cells) + " |"])


class Reader:
    """Injected witness: text per resolution; an Exception value is raised."""

    def __init__(self, by_dpi: dict[int, object]):
        self.by_dpi = by_dpi
        self.dpis: list[int] = []

    def __call__(self, pix) -> str:
        d = _dpi(pix)
        self.dpis.append(d)
        out = self.by_dpi.get(d, "")
        if isinstance(out, Exception):
            raise out
        return str(out)


def _verify(text: str, reader: Reader, *, ladder=LADDER, **kw):
    with patch.object(se, "_EXTRA_EVIDENCE_OCR_DPIS", ladder):
        return verify_scanned_table(
            _page(), text, ocr_image_fn=reader, native_trusted=False, layer_is_evidence=False, **kw
        )


TABLE = _md("1.5", "2.5", "4.1")
BASE_TEXT = "1.5 2.5"  # "4.1" missing at the base resolution


def test_pass_at_base_makes_no_extra_call_and_same_result():
    full = "1.5 2.5 4.1"
    with_ladder = Reader({BASE: full, LADDER[0]: "9.9"})
    without = Reader({BASE: full, LADDER[0]: "9.9"})
    a = _verify(TABLE, with_ladder)
    b = _verify(TABLE, without, ladder=())
    assert a.passed and b.passed
    assert a == b
    assert with_ladder.dpis == without.dpis == [BASE]
    assert a.extra_resolutions == ()


def test_missing_token_found_at_later_rung_passes_and_names_the_rung():
    r = Reader({BASE: BASE_TEXT, LADDER[0]: "1.5 2.5 4.1"})
    res = _verify(TABLE, r)
    base_only = _verify(TABLE, Reader({BASE: BASE_TEXT}), ladder=())
    assert not base_only.passed and base_only.disputed_values == ("4.1",)
    assert res.passed
    assert res.extra_resolutions == (LADDER[0],)
    assert res.extra_resolution_tokens == 1


def test_ladder_stops_at_first_rung_that_clears():
    r = Reader({BASE: BASE_TEXT, LADDER[0]: "4.1", LADDER[1]: "4.1", LADDER[2]: "4.1"})
    res = _verify(TABLE, r)
    assert res.passed
    assert r.dpis == [BASE, LADDER[0]]


def test_later_rung_used_when_first_does_not_clear():
    r = Reader({BASE: BASE_TEXT, LADDER[0]: "7.7", LADDER[1]: "4.1"})
    res = _verify(TABLE, r)
    assert res.passed
    assert res.extra_resolutions == LADDER[:2]
    assert r.dpis == [BASE, *LADDER[:2]]


def test_max_merge_does_not_sum_counts_across_rungs():
    table = _md("1.5", "4.1", "4.1")
    r = Reader({BASE: "1.5", LADDER[0]: "4.1", LADDER[1]: "4.1", LADDER[2]: "4.1"})
    res = _verify(table, r)
    # Each rung saw one "4.1"; the table emits two. Summing would pass.
    assert not res.passed
    assert res.disputed_values == ("4.1",)
    assert res.extra_resolutions == LADDER
    # A rung that really reads two supports both.
    r2 = Reader({BASE: "1.5", LADDER[0]: "4.1 4.1"})
    assert _verify(table, r2).passed


def test_token_no_rung_reads_stays_disputed_like_base_only():
    r = Reader({BASE: BASE_TEXT})
    res = _verify(TABLE, r)
    base_only = _verify(TABLE, Reader({BASE: BASE_TEXT}), ladder=())
    assert not res.passed
    assert res.disputed_values == base_only.disputed_values
    assert res.reason == base_only.reason
    assert res.verifiable == base_only.verifiable
    assert res.extra_resolutions == LADDER
    assert res.extra_resolution_tokens == 0


def test_erroring_extra_rung_leaves_the_base_outcome():
    err = {d: TimeoutError("tesseract timeout") for d in LADDER}
    res = _verify(TABLE, Reader({BASE: BASE_TEXT, **err}))
    base_only = _verify(TABLE, Reader({BASE: BASE_TEXT}), ladder=())
    # Not "no witness", and the base disputed values are still named.
    assert res.verifiable and not res.passed and res.cause == ""
    assert res.disputed_values == base_only.disputed_values
    assert res.witness_state == base_only.witness_state


def test_failed_rung_contributes_nothing_but_later_rung_still_counts():
    r = Reader({BASE: BASE_TEXT, LADDER[0]: RuntimeError("boom"), LADDER[1]: "4.1"})
    res = _verify(TABLE, r)
    assert res.passed
    assert res.extra_resolutions == LADDER[:2]


def test_empty_extra_rung_is_not_a_witness_failure():
    res = _verify(TABLE, Reader({BASE: BASE_TEXT, LADDER[0]: "   "}))
    assert res.verifiable and res.cause == ""


def test_labels_stay_at_base_reading():
    # Rung 1 reads the missing number AND the label "Counterparty"; the base read
    # neither label. The label verdict is the base reading's, so it is still unverified.
    table = _md("1.5", "4.1", label="Counterparty")
    r = Reader({BASE: "1.5", LADDER[0]: "1.5 4.1 Counterparty"})
    res = _verify(table, r)
    base_label = _verify(
        table, Reader({BASE: "1.5 4.1"}), ladder=()
    )  # base-pass path, label unread
    assert res.passed and res.extra_resolutions == (LADDER[0],)
    assert res.content_unverified
    assert res.content_unverified == base_label.content_unverified


def test_alpha_only_table_never_reads_extra_resolutions():
    table = "| a | b |\n|---|---|\n| Counterparty | Bundesbank |\n"
    r = Reader({BASE: "Other"})
    res = _verify(table, r)
    assert r.dpis == [BASE]
    assert res.extra_resolutions == ()


def test_empty_base_reading_does_not_trigger_rungs():
    r = Reader({BASE: "", LADDER[0]: "1.5 2.5 4.1"})
    res = _verify(TABLE, r)
    assert r.dpis == [BASE]
    assert not res.verifiable


def test_born_digital_text_layer_is_not_multiplied_by_rungs():
    # The layer holds one "4.1"; the table emits two. Pixel rungs each read one "4.1".
    # Layer + rung must be a max, never a sum, and rungs never re-read the layer.
    page = _page()
    page.insert_text((72, 100), "4.1 1.5", fontsize=12)
    table = _md("1.5", "4.1", "4.1")
    r = Reader({BASE: "1.5", **{d: "4.1" for d in LADDER}})
    with patch.object(se, "_EXTRA_EVIDENCE_OCR_DPIS", LADDER):
        res = verify_scanned_table(
            page, table, ocr_image_fn=r, native_trusted=False, layer_is_evidence=True
        )
    assert not res.passed
    assert res.disputed_values == ("4.1",)


def test_rungs_never_bring_in_a_text_layer_the_base_path_excluded():
    # The layer holds "7.7"; the layer is not admissible evidence, so base and rungs
    # read pixels only and a table emitting "7.7" stays unsupported at every rung.
    page = _page()
    page.insert_text((72, 100), "7.7 1.5", fontsize=12)
    r = Reader({BASE: "1.5"})
    with patch.object(se, "_EXTRA_EVIDENCE_OCR_DPIS", LADDER):
        res = verify_scanned_table(
            page, _md("1.5", "7.7"), ocr_image_fn=r, native_trusted=False, layer_is_evidence=False
        )
    assert not res.passed and res.disputed_values == ("7.7",)
    assert res.extra_resolutions == LADDER


def test_default_reader_uses_the_bound_derived_for_each_rung():
    assert set(LADDER) == set(se._EXTRA_EVIDENCE_TIMEOUT_SEC)
    assert all(
        t >= se.TESSERACT_WITNESS_TIMEOUT_SEC for t in se._EXTRA_EVIDENCE_TIMEOUT_SEC.values()
    )
    seen: list[tuple[int, float]] = []

    def fake(pix, timeout_sec=se.TESSERACT_WITNESS_TIMEOUT_SEC):
        seen.append((_dpi(pix), timeout_sec))
        return ("1.5 2.5", se.WITNESS_READING, "")

    with patch.object(se, "classical_ocr_with_state", fake):
        res = verify_scanned_table(_page(), TABLE, native_trusted=False, layer_is_evidence=False)
    assert not res.passed
    assert seen[0] == (BASE, se.TESSERACT_WITNESS_TIMEOUT_SEC)
    assert seen[1:] == [(d, se._EXTRA_EVIDENCE_TIMEOUT_SEC[d]) for d in LADDER]


# ---- route level: the judge, run twice in one process --------------------------


def _assess(reader: Reader):
    events: list[AuditEvent] = []
    inner = MagicMock()
    inner.assess.return_value = AcceptDecision(accept=True, reason="inner accepts")
    page = _page()
    judge = SourceEvidenceTableJudge(
        inner=inner,
        get_fitz_page=lambda pn: page,
        record_event=events.append,
        ocr_image_fn=reader,
        native_trusted=lambda pn: False,
        layer_is_evidence=lambda pn: False,
    )
    out = PageOutput(
        page_num=3,
        text=TABLE,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
        confidence=0.9,
    )
    return judge.assess(out, MagicMock()), out, events, inner


def test_route_level_runs_differ_only_in_what_the_second_rung_reads():
    base = {BASE: BASE_TEXT}
    yes = _assess(Reader({**base, LADDER[0]: "4.1"}))
    no = _assess(Reader({**base, LADDER[0]: "9.9"}))
    d_yes, _, ev_yes, inner_yes = yes
    d_no, out_no, ev_no, inner_no = no
    assert d_yes.accept and not d_no.accept
    kinds_yes = [e.kind for e in ev_yes]
    kinds_no = [e.kind for e in ev_no]
    assert EXTRA_RESOLUTION_KIND in kinds_yes
    assert EXTRA_RESOLUTION_KIND not in kinds_no
    assert "source_evidence_table_reject" not in kinds_yes
    assert "source_evidence_table_reject" in kinds_no
    ev = next(e for e in ev_yes if e.kind == EXTRA_RESOLUTION_KIND)
    assert ev.data == {"resolutions": [LADDER[0]], "tokens_resolved": 1}
    inner_yes.assess.assert_called_once()
    inner_no.assess.assert_not_called()
    assert out_no.table_disputed_values == ("4.1",)
    rej = next(e for e in ev_no if e.kind == "source_evidence_table_reject")
    assert rej.data["extra_resolutions"] == list(LADDER)


def test_route_level_base_pass_emits_no_extra_event():
    d, _, ev, _ = _assess(Reader({BASE: "1.5 2.5 4.1"}))
    assert d.accept
    assert all(e.kind != EXTRA_RESOLUTION_KIND for e in ev)
