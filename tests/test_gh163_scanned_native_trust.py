"""GH-163: word presence is not native trust.

`verify_scanned_table` deferred to the native verifier whenever
`page_has_native_words` found any non-empty word. A scanned page carrying a
baked-in or corrupt OCR layer has words -- so it deferred, and the native
verifier then graded the model's table against that same untrusted layer. The
fail-closed raster/classical evidence check was skipped for exactly the pages
that need it, and a hallucinated table could be corroborated by a hallucinated
text layer.

Deferral now hinges on the caller's born-digital classification:

- ``native_trusted=True``  -> defer, as before
- ``native_trusted=False`` -> run the evidence check, however many words exist
- ``native_trusted=None``  -> the caller cannot tell; keep the pre-GH-163
  behaviour, because an unknown classification must not silently start failing
  pages closed

All three are pinned, on ONE page with ONE text layer, so the only thing that
varies between them is the trust flag.

Not deferring is only half the fix (cubic P1 on #512). `build_scanned_evidence`
merged `page.get_text()` in FIRST, and the full-page raster branch only fires
when nothing else produced evidence -- so the suspect layer was still the
primary corroboration, and a table agreeing with a corrupt layer verified
against it. An untrusted page now excludes that layer, leaving only readings
taken from the pixels. That is the case `TestTheSuspectLayerCannotCorroborate`
covers, and it is the one that actually matters: a hallucination that agrees
with nothing was never going to pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from socr.tables.source_evidence import (  # noqa: E402
    page_has_native_words,
    verify_scanned_table,
)

# A table the page does not support: none of these numbers appear in its text
# layer. Against real evidence this must hard-reject.
HALLUCINATED = "\n".join(
    [
        "| Variable | Coefficient | SE |",
        "| --- | --- | --- |",
        "| growth | 0.8172 | 0.0413 |",
        "| inflation | -0.2946 | 0.0517 |",
        "| output gap | 1.4408 | 0.1122 |",
    ]
)


def _page_with_untrusted_ocr_layer(tmp_path: Path):
    """A page whose text layer has words, none of them the table's numbers.

    This stands in for a scanned page with a baked-in OCR layer: text is
    present, so `page_has_native_words` is True, but it is not a trustworthy
    reading of the page.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i in range(12):
        page.insert_text((72, 100 + i * 16), f"rn1 garbled ocr line {i} lll 000", fontsize=10)
    pdf = tmp_path / "scan.pdf"
    doc.save(pdf)
    doc.close()
    return fitz.open(pdf)


def _verify(doc, *, native_trusted):
    return verify_scanned_table(
        doc[0],
        HALLUCINATED,
        ocr_image_fn=lambda _img: "",
        native_trusted=native_trusted,
    )


def test_an_untrusted_layer_with_words_no_longer_defers(tmp_path: Path) -> None:
    """The acceptance case: non-empty untrusted OCR words + a hallucinated table."""
    doc = _page_with_untrusted_ocr_layer(tmp_path / "untrusted")
    try:
        assert page_has_native_words(doc[0]), (
            "fixture must have words, or it would not exercise the old defer path"
        )

        result = _verify(doc, native_trusted=False)

        assert not result.deferred, (
            "an untrusted text layer still deferred to the native verifier, which "
            "grades the table against that same layer"
        )
        assert not result.passed, (
            f"a table whose numbers appear nowhere on the page was accepted: {result.reason}"
        )
    finally:
        doc.close()


def test_a_trusted_page_still_defers(tmp_path: Path) -> None:
    """Control. Without it, a change that never deferred would satisfy the test
    above while sending every born-digital table down the scanned lane."""
    doc = _page_with_untrusted_ocr_layer(tmp_path / "trusted")
    try:
        result = _verify(doc, native_trusted=True)
        assert result.deferred, "a trusted-native page must still defer to the native verifier"
    finally:
        doc.close()


def test_an_unknown_classification_keeps_the_old_behaviour(tmp_path: Path) -> None:
    """`None` is not `False`.

    A caller that cannot classify the page must not have its pages start
    failing closed as a side effect of this change.
    """
    doc = _page_with_untrusted_ocr_layer(tmp_path / "unknown")
    try:
        result = _verify(doc, native_trusted=None)
        assert result.deferred, (
            "an unknown classification was treated as untrusted; that silently "
            "changes behaviour for every caller that cannot tell"
        )
    finally:
        doc.close()


def test_the_orchestrator_supplies_the_classification(tmp_path: Path) -> None:
    """The fix is inert unless the real judge is built with a trust source.

    Everything above passes `native_trusted` by hand. If `_build_page_judge`
    did not wire one, production would keep deferring on word presence and this
    file would be testing a parameter nobody sets.
    """
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState
    from socr.pipeline.orchestrator import UnifiedPipeline

    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "d.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "a text layer long enough to be a real one.")
    doc.save(pdf)
    doc.close()

    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            judge_backend="heuristic",
        )
    )
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    judge = pipeline._build_page_judge(state)

    supplier = getattr(judge, "_native_trusted", None)
    assert callable(supplier), (
        "the judge was built with no trust source, so the verifier still defers "
        "on word presence and the GH-163 fix never runs in production"
    )

    state.pages[1].is_born_digital = False
    assert supplier(1) is False, "the trust source does not follow the page's classification"
    state.pages[1].is_born_digital = True
    assert supplier(1) is True

    assert supplier(99) is None, (
        "an unknown page must report None (cannot tell), not False (untrusted) -- "
        "False would fail those pages closed on a missing record"
    )


class TestTheSuspectLayerCannotCorroborate:
    """cubic P1 on #512: non-deferral alone does not make the check independent.

    The dangerous page is not one whose OCR layer is garbage unrelated to the
    table -- that fails on any evidence. It is one where the model reproduced
    the corrupt layer faithfully. Then output and evidence agree perfectly, and
    the check passes on a reading nobody trusts.
    """

    TABLE = "\n".join(
        [
            "| Variable | Coefficient |",
            "| --- | --- |",
            "| growth | 0.8172 |",
            "| inflation | -0.2946 |",
        ]
    )

    def _page_whose_layer_matches(self, tmp_path: Path):
        """A text layer carrying exactly the table's tokens, and nothing on the
        pixels to corroborate them independently."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 100), "growth 0.8172", fontsize=10)
        page.insert_text((72, 120), "inflation -0.2946", fontsize=10)
        pdf = tmp_path / "scan.pdf"
        doc.save(pdf)
        doc.close()
        return fitz.open(pdf)

    def _verify(self, doc, *, native_trusted):
        # No classical OCR reading available: the pixels corroborate nothing,
        # which is what isolates the text layer's contribution.
        return verify_scanned_table(
            doc[0],
            self.TABLE,
            ocr_image_fn=lambda _pix: "",
            native_trusted=native_trusted,
        )

    def test_a_table_matching_an_untrusted_layer_is_not_verified_by_it(
        self, tmp_path: Path
    ) -> None:
        doc = self._page_whose_layer_matches(tmp_path / "match")
        try:
            result = self._verify(doc, native_trusted=False)
            assert not result.deferred
            assert not result.passed, (
                "the model's table was verified against the very text layer the "
                f"caller marked untrusted: {result.reason}"
            )
        finally:
            doc.close()

    def test_the_same_page_when_the_layer_IS_trusted_defers(self, tmp_path: Path) -> None:
        """Control: the page is unchanged; only the classification differs."""
        doc = self._page_whose_layer_matches(tmp_path / "match_trusted")
        try:
            assert self._verify(doc, native_trusted=True).deferred
        finally:
            doc.close()


def test_an_untrusted_page_reaches_the_classical_ocr_path(tmp_path: Path) -> None:
    """cubic P2 on #512: the earlier tests never ran the raster/classical branch.

    With the text layer excluded there is no evidence to start from, so the
    pixel readings are the only ones -- and this asserts the OCR function is
    actually invoked, rather than inferring it from a rejection that a missing
    OCR path would also produce.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "growth 0.8172", fontsize=10)
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "scan.pdf"
    doc.save(pdf)
    doc.close()

    calls: list[object] = []

    def _ocr(pix):
        calls.append(pix)
        return ""

    doc = fitz.open(pdf)
    try:
        verify_scanned_table(
            doc[0],
            TestTheSuspectLayerCannotCorroborate.TABLE,
            ocr_image_fn=_ocr,
            native_trusted=False,
        )
    finally:
        doc.close()

    assert calls, (
        "no pixel reading was attempted for an untrusted page; the evidence "
        "check ran on nothing, which is not verification"
    )


class TestTheJudgePassesTheClassificationThrough:
    """GH-513: `assess()` itself, not the helper and not the construction site.

    Everything above either calls `verify_scanned_table` by hand or checks that
    `_build_page_judge` attaches a supplier. Neither runs the line between
    them. Reverting `native_trusted=trusted` inside `assess()` left this whole
    file green while restoring word-presence deferral in production for every
    scanned page with a baked-in OCR layer -- the exact GH-163 failure -- and
    never reaching the `include_text_layer=False` path either.

    Same standard the wiring test applies at the other end.
    """

    class _AlwaysAccepts:
        """Stands in for the inner judge chain.

        It accepts unconditionally, so a deferral is visible as an ACCEPT: if
        the classification is not passed through, the page defers here and
        ships. That makes the difference between the two cases below entirely
        attributable to the pass-through.
        """

        def __init__(self) -> None:
            self.calls = 0

        def assess(self, output, provider):
            from socr.pipeline.agentic import AcceptDecision

            self.calls += 1
            return AcceptDecision(accept=True, reason="inner accepted", confidence=1.0)

    def _judge(self, doc, trusted):
        from socr.pipeline.agentic import SourceEvidenceTableJudge

        inner = self._AlwaysAccepts()
        judge = SourceEvidenceTableJudge(
            inner=inner,
            get_fitz_page=lambda _n: doc[0],
            record_event=None,
            ocr_image_fn=lambda _pix: "",
            native_trusted=lambda _n: trusted,
        )
        return judge, inner

    def _output(self):
        from socr.core.result import PageOutput, PageStatus

        return PageOutput(
            page_num=1,
            text=HALLUCINATED,
            status=PageStatus.SUCCESS,
            engine="qwen",
        )

    def _provider(self):
        from socr.core.providers import PROFILE_QWEN_LOCAL

        return PROFILE_QWEN_LOCAL

    def test_an_untrusted_page_is_rejected_by_the_real_judge(self, tmp_path: Path) -> None:
        doc = _page_with_untrusted_ocr_layer(tmp_path / "assess_untrusted")
        try:
            assert page_has_native_words(doc[0]), (
                "the page must have words, or the old code would not have deferred"
            )
            judge, inner = self._judge(doc, trusted=False)
            decision = judge.assess(self._output(), self._provider())

            assert inner.calls == 0, (
                "the judge deferred to the inner chain on an untrusted page, so "
                "the classification never reached verify_scanned_table"
            )
            assert not decision.accept, (
                f"a hallucinated table on an untrusted page was accepted: {decision.reason}"
            )
        finally:
            doc.close()

    def test_a_trusted_page_defers_to_the_inner_chain(self, tmp_path: Path) -> None:
        """Control: the same page, the same table, the opposite classification.

        Without it, a judge that rejected everything would satisfy the test
        above.
        """
        doc = _page_with_untrusted_ocr_layer(tmp_path / "assess_trusted")
        try:
            judge, inner = self._judge(doc, trusted=True)
            decision = judge.assess(self._output(), self._provider())

            assert inner.calls == 1, "a trusted page must reach the inner judge chain"
            assert decision.accept, "the inner judge accepted; the wrapper must not override it"
        finally:
            doc.close()
