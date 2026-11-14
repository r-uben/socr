"""#1305: printed fractions are not corrupt math, and a resolved mixed line keeps its text.

The corrupt-math lane read the printed glyphs of "2¾ percent" as font damage, then
replaced the whole prose line with a candidate (``2^{3/4}``). Two guards:

* the trigger: a vulgar fraction counts as the '=' mojibake only in operator position;
* the splice: a resolved region never replaces its native line (owner rule: never
  delete source text); crop, label and candidate go after it.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import fitz
import pytest

from socr.core.born_digital import (
    BornDigitalDetector,
    corrupt_math_glyph_count,
    line_has_corrupt_math,
)
from socr.math import recover as recover_module
from socr.math.recover import (
    CorruptMathRegion,
    corrupt_math_line_rects,
    recover_math_regions,
    splice_math,
)

# The sentence the audit found deleted (tealbook-partb 2018-06-13 p13, F-A-054).
_FED_SENTENCE = (
    "widen through 2019. In this projection, real GDP growth is 2¾ percent this "
    "year and then gradually slows to 1¾ percent in 2020."
)

#: (line, is_printed_fraction). Every verdict of the trigger rule, pinned. The first
#: group is taken from the 156-doc Fed run.
_VERDICTS = [
    # printed: glued after a digit, not followed by a digit
    (_FED_SENTENCE, True),
    ("growth of 2¾", True),
    ("inflation of 4½ percent", True),
    ("an 8 ½% rate, up from 2¾% last year", True),
    # printed: stand-alone after a number, before a word or %
    ("We now expect the unemployment rate to be around 8 ½% by the end of 2011 and 7 ½ % by", True),
    ("deflator are now likely to top out in the 2 ½ % to 2 ¾% range in 2011Q3.", True),
    ("it grew 1 ¼ percent", True),
    ("1 ¼ percent", True),
    # printed: after an article or word, before a word
    ("the Committee raised rates by a ¼ percentage point cut in rates today", True),
    ("a ¼ point cut", True),
    ("8 ½%", True),
    ("2 ½ million", True),
    ("2 ½ to 3 percent", True),
    ("the ½ share of output", True),
    ("raised it ¼ percentage point", True),
    # printed: hyphen compound, bracketed series
    ("the ¼-point increase was widely expected", True),
    ("target range for the federal funds rate [at ½] to 3/4 {to 1} percent. The stance", True),
    ("range of [¼ to] ½{ to ¾}[¼ to] ½{ to ¾}percent, including overnight reverse", True),
    ("a state-of-the-art ¼ share", True),
    ("GDP ½ percentage point", True),
    ("up ¼ of a point", True),
    # printed: signed mixed numbers
    ("growth of −2 ½ percent", True),
    ("−2 ½ percent", True),
    ("it fell -2 ¼ percent", True),
    ("it fell -2½", True),
    # printed: dash and sign neighbours, a line that ends on the number (wrapped)
    ("a 3¼ percent in the third quarter—½ percentage point faster", True),
    ("plunging to -¼ percent in the third quarter", True),
    ("the move was −¼ percent", True),
    ("Potential GDP growth is 2 ½ ", True),
    ("real GDP ½ percentage point lower at the end", True),
    ("the ¾ percent—is well below its average", True),
    ("by mid-2018—about ½ percentage point", True),
    # operator: the '=' mojibake
    ("10¼5", False),
    ("10¼x", False),
    ("10¼ 5+5", False),
    ("x¼-y", False),
    ("2¼x", False),
    ("x¼1", False),
    ("2 ¼ 4", False),
    ("see 2 ¼ 4", False),
    ("x ¼ y", False),
    ("y ¼ αx", False),
    ("y ¼ ax", False),
    ("yy ¼ ax", False),
    ("where yy ¼ ax and the model holds", False),
    ("xy ¼ z", False),
    ("xyz ¼ xy", False),
    ("y₁ ¼ x₂", False),
    ("y ¼ x2", False),
    ("P ¼ NP", False),
    ("log likelihood ¼ 0", False),
    ("a ¼ sin x", False),
    ("a ¼ b", False),
    ("f(x) ¼ 1", False),
    ("y ¼ sin x and then the model holds up", False),
    ("the state-of-the-art model y ¼ x holds", False),
    ("rate ¼ 3", False),
    ("2 ¼ sin x", False),
    ("3 ¼ log y", False),
    ("1 ¼ x + y", False),
    ("2 ¼ xy + 1", False),
    # Unicode operators after a vowel-free variable (cubic P1)
    ("2 ¼ xy ≤ 1", False),
    ("2 ¼ xy ≥ 1", False),
    ("2 ¼ xy ≠ z", False),
    ("2 ¼ xy × 3", False),
    ("2 ¼ xy − 1", False),
    # math function: operand only with a math-shaped continuation (cubic P2)
    ("2 ¼ min x", False),
    ("2 ¼ min(x, y)", False),
    ("2 ¼ max 3", False),
    ("2 ¼ log α", False),
    ("2 ½ min", True),
    ("2 ½ min.", True),
    ("3 ¼ min later", True),
    ("2 ½ max of the two", True),
    # prose function words are not compact variables (cubic P3)
    ("2 ½ by 4", True),
    ("2 ½ by 4 inches", True),
    ("3 ¼ per 4", True),
]
_PRINTED = [line for line, printed in _VERDICTS if printed]
_OPERATOR = [line for line, printed in _VERDICTS if not printed]

_MOJIBAKE = "PðA or BÞ ¼ PðAÞ þ PðBÞ"


@pytest.mark.parametrize("line", _PRINTED)
def test_printed_fractions_are_not_corrupt_math(line):
    assert corrupt_math_glyph_count(line) == 0
    assert not line_has_corrupt_math(line)


@pytest.mark.parametrize("line", _OPERATOR)
def test_operator_fraction_glyph_stays_detected(line):
    """Carbon: 14.5% of the Papers library depends on '¼' standing in for '='."""
    assert corrupt_math_glyph_count(line) >= 1
    assert line_has_corrupt_math(line)


def test_hyphen_compounds_are_plain_words():
    from socr.core.born_digital import _is_plain_word

    assert _is_plain_word("state-of-the-art")
    assert not _is_plain_word("-x")
    assert _is_plain_word("GDP")
    assert not _is_plain_word("x")


def test_delimiter_mojibake_and_c0_still_count_everywhere():
    assert corrupt_math_glyph_count(_MOJIBAKE) == 8
    assert line_has_corrupt_math("a \x02 b")
    # prose beside the mojibake does not exempt it
    assert corrupt_math_glyph_count("the growth is 2¾ percent and PðAÞ holds") == 2


def test_detector_page_count_uses_the_shared_predicate():
    """No orphan banner: the page detector and the line selectors agree."""
    text = f"{_FED_SENTENCE}\n{_MOJIBAKE}\ny ¼ ax"
    per_line = sum(corrupt_math_glyph_count(line) for line in text.split("\n"))
    assert BornDigitalDetector._count_math_corruption(text) == per_line == 9
    assert not BornDigitalDetector()._detect_corrupt_math(_FED_SENTENCE * 4)


def _page(lines):
    doc = fitz.open()
    page = doc.new_page()
    for y, text in lines:
        page.insert_text((72, y), text, fontsize=10)
    return doc, page


def test_fraction_only_line_selects_no_recovery_region():
    _doc, page = _page([(100, "growth is 2¾ percent this year and then slows to 1¾ percent")])
    assert corrupt_math_line_rects(page) == []
    assert recover_math_regions(page, ocr=MagicMock()) == []


def test_numbered_equation_row_uses_the_same_predicate():
    """``_numbered_math_groups`` must not select a row only a printed fraction flags."""
    region = MagicMock(
        has_eq_number=True,
        source_text="growth is 2¾ percent this year (3)",
        source_bbox=(0, 0, 10, 10),
        bbox=(0, 0, 10, 10),
        equation_label="(3)",
    )
    detected = MagicMock(regions=[region])
    with patch("socr.math.detect_equations.detect_display_equations", return_value=detected):
        assert recover_module._numbered_math_groups(MagicMock()) == []
        region.source_text = "PðAÞ ¼ 1 (3)"
        assert len(recover_module._numbered_math_groups(MagicMock())) == 1


def test_standalone_operator_line_is_still_a_recovery_region():
    _doc, page = _page([(100, "y ¼ ax")])
    assert len(corrupt_math_line_rects(page)) == 1


def _resolved(source, latex, crop="equations/crop.png"):
    return CorruptMathRegion(
        rect=fitz.Rect(1, 1, 2, 2),
        source_text=source,
        crop_path=crop,
        raw_latex=latex,
        validation_ok=True,
        validation_reason="ok",
        model_id="fixture-model",
        attempts=1,
    )


_MIXED = "In the model PðAÞ ¼ 2¾ percent growth this year and then slows."
#: Sources whose prose is short or hyphenated: no prose-run test may decide deletion.
_SHORT_LABEL = "Real GDP ðx"
_HYPHENATED = "The state-of-the-art model PðAÞ ¼ 1"
_PURE = "ðxÞ ¼ y"


@pytest.mark.parametrize("source", [_MIXED, _SHORT_LABEL, _HYPHENATED, _PURE, _MOJIBAKE])
def test_resolved_region_keeps_the_whole_native_line(source):
    native = f"Before.\n{source}\nAfter."
    region = _resolved(source, r"P(A) = 2^{3/4}")
    out = splice_math(None, native, [region])

    assert out.startswith(f"Before.\n{source}\n")
    assert out.endswith("\nAfter.")
    assert region.source_aligned is True
    assert "![Corrupt equation crop](equations/crop.png)" in out
    assert "equation candidate, unverified" in out
    assert r"P(A) = 2^{3/4}" in out
    # the native text precedes the crop, and the label precedes the candidate
    assert out.index(source) < out.index("![Corrupt equation crop]")
    assert out.index("equation candidate, unverified") < out.index("2^{3/4}")


@pytest.mark.parametrize("source", [_MIXED, _SHORT_LABEL, _HYPHENATED, _PURE])
def test_recover_then_splice_on_a_real_page(tmp_path, source):
    _doc, page = _page([(100, source)])
    ocr = MagicMock(return_value=r"P(A) = 2^{3/4}")
    regions = recover_math_regions(page, ocr=ocr, crop_dir=tmp_path / "eq", page_num=1)
    native = page.get_text("text").rstrip("\n")
    out = splice_math(page, native, regions)
    assert len(regions) == 1 and regions[0].resolved
    assert source in out
    assert out.index(source) < out.index("equation candidate, unverified")


# --- process(): the pair, in both provider states -----------------------------------------


def _pdf(path, line):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Native prose survives on this page.", fontsize=10)
    page.insert_text((72, 140), line, fontsize=10)
    doc.save(str(path))
    doc.close()


@pytest.mark.parametrize("provider", [False, True], ids=["no-provider", "provider"])
def test_process_pair_fraction_line_vs_mojibake_line(tmp_path, provider):
    """Pin a DIFFERENCE: the fraction-only line ships native and unmarked by the lane;
    the mojibake line goes through it. Same in both provider states."""
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline.orchestrator import UnifiedPipeline

    sentence = "Real GDP growth is 2¾ percent this year and then slows to 1¾ percent."
    shipped = {}
    for arm, line in (("fraction", sentence), ("mojibake", _MOJIBAKE)):
        for recover in (False, True):
            pdf_dir = tmp_path / f"{arm}-{recover}"
            pdf_dir.mkdir()
            pdf = pdf_dir / "paper.pdf"
            _pdf(pdf, line)
            config = PipelineConfig(
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                agentic=True,
                native_first=True,
                recover_corrupt_math=recover,
                judge_backend="heuristic",
                ollama_host="http://127.0.0.1:1",
                save_figures=False,
                quiet=True,
                tiered=False,
            )
            pipeline = UnifiedPipeline(config)
            pipeline._available_engines_for_agentic = MagicMock(
                return_value=[PROFILE_QWEN_LOCAL] if provider else []
            )
            engine = MagicMock()
            engine.process_pages.side_effect = RuntimeError("whole-page engine must not run")
            with (
                patch("socr.pipeline.orchestrator.get_engine", return_value=engine),
                patch.object(pipeline, "_resolve_judge_model", return_value=""),
                patch.object(pipeline, "_resolve_crop_vlm_model", return_value=None),
            ):
                pipeline.process(pdf, pdf_dir / "out")
            fragment = (pdf_dir / "out" / "paper" / "pages" / "00001.md").read_text()
            sidecar = json.loads((pdf_dir / "out" / "paper" / "pages" / "00001.json").read_text())
            shipped[(arm, recover)] = (fragment, sidecar)

    for recover in (False, True):
        fragment, sidecar = shipped[("fraction", recover)]
        assert sentence in fragment
        assert "2^{3/4}" not in fragment
        assert "Corrupt equation crop" not in fragment
        assert sidecar["winning_output"]["engine"] != "native+math"
    # the flag changes nothing on the fraction-only page: no lane, same bytes
    assert shipped[("fraction", True)][0] == shipped[("fraction", False)][0]

    fragment, sidecar = shipped[("mojibake", True)]
    assert "Corrupt equation crop" in fragment
    assert _MOJIBAKE in fragment  # unresolved without a reachable reader: native kept
    assert sidecar["winning_output"]["engine"] == "native+math"
    assert "Corrupt equation crop" not in shipped[("mojibake", False)][0]


def test_resume_of_a_fraction_only_page_ships_the_same_bytes(tmp_path):
    """The page is outside the lane, so a second run over the same output dir resumes it
    and the shipped bytes (fragment and stitched document) do not move."""
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline.orchestrator import UnifiedPipeline

    sentence = "Real GDP growth is 2¾ percent this year and then slows to 1¾ percent."
    pdf = tmp_path / "paper.pdf"
    _pdf(pdf, sentence)
    shipped = []
    for _ in range(2):
        config = PipelineConfig(
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            agentic=True,
            native_first=True,
            judge_backend="heuristic",
            ollama_host="http://127.0.0.1:1",
            save_figures=False,
            quiet=True,
            tiered=False,
        )
        pipeline = UnifiedPipeline(config)
        pipeline._available_engines_for_agentic = MagicMock(return_value=[PROFILE_QWEN_LOCAL])
        with (
            patch("socr.pipeline.orchestrator.get_engine", return_value=MagicMock()),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch.object(pipeline, "_resolve_crop_vlm_model", return_value=None),
        ):
            result = pipeline.process(pdf, tmp_path / "out")
        doc_dir = tmp_path / "out" / "paper"
        assert result is not None
        shipped.append(
            ((doc_dir / "pages" / "00001.md").read_text(), (doc_dir / "paper.md").read_text())
        )
    assert shipped[0] == shipped[1]
    assert sentence in shipped[1][0] and sentence in shipped[1][1]
