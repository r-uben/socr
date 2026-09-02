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
