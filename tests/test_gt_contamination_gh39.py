"""GH-39: the benchmark's ground truth cannot be the raw native text layer.

`GroundTruthExtractor` took `page.get_text("text")` verbatim. GH-136 established
that layer can be WRONG — a broken font map ships "(1997)" as "(/997)".

Scored against such a page, an engine that reads the raster correctly and emits
"1997" is counted as making an error. The benchmark would therefore penalise
correct OCR precisely where OCR is most valuable, and reward engines that
reproduce the corruption — flattering the native-trusting lane that agentic
routing depends on. The bias runs in the design's favour, which is the worst
direction for a bias to run.

Contaminated pages are marked rather than dropped, so page numbers stay aligned
with engine output and the exclusion count stays reportable.
"""

from __future__ import annotations

import json

import fitz
import pytest

from socr.benchmark.ground_truth import (
    GroundTruthExtractor,
    PageGroundTruth,
    exclusion_report,
    usable_truths,
)

_CLEAN = "The estimated coefficient on schooling is 0.082 and significant at the 1 percent level"


def _pdf(tmp_path, pages: list[list[str]], name="doc.pdf"):
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page()
        y = 80
        for ln in lines:
            page.insert_text((60, y), ln, fontsize=9)
            y += 16
    path = tmp_path / name
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------------------
# The contamination itself
# ---------------------------------------------------------------------------


def test_eaten_digit_page_is_not_usable_as_truth(tmp_path):
    """The motivating case: a wrong year cannot be the yardstick for a right one."""
    pdf = _pdf(
        tmp_path,
        [[_CLEAN] * 11 + ["Fama, E. and French, K. (/997). Multifactor explanations."]],
    )

    truths = GroundTruthExtractor().extract(pdf)

    assert truths[0].usable is False
    assert "eaten-digit" in truths[0].exclusion_reason


def test_clean_page_remains_usable(tmp_path):
    pdf = _pdf(tmp_path, [[_CLEAN] * 12])

    truths = GroundTruthExtractor().extract(pdf)

    assert truths[0].usable is True
    assert truths[0].exclusion_reason == ""


def test_mild_hygiene_corruption_is_still_usable(tmp_path):
    """Lost spaces make truth ugly, not wrong — excluding them would gut the corpus."""
    pdf = _pdf(
        tmp_path,
        [[_CLEAN] * 11 + ["FrenchfJoumal ofFinancial Economics 43 volumeNumberFortyThree"]],
    )

    truths = GroundTruthExtractor().extract(pdf)

    assert truths[0].usable is True


def test_pervasively_corrupted_page_is_not_usable(tmp_path):
    """If the pipeline will not trust the layer, the benchmark must not score on it."""
    junk = (
        "The coefficienton schoolingis significantAt the percentAcross "
        "firmsInTheSampleSet estimatedValueOfCoefficients robustnessCheckResults"
    )
    pdf = _pdf(tmp_path, [[junk] * 12])

    truths = GroundTruthExtractor().extract(pdf)

    assert truths[0].usable is False
    assert "encoding corruption" in truths[0].exclusion_reason


# ---------------------------------------------------------------------------
# Marked, not dropped — alignment and reporting
# ---------------------------------------------------------------------------


def test_every_page_is_returned_so_numbering_stays_aligned(tmp_path):
    """Dropping pages here would silently misalign truth against engine output."""
    pdf = _pdf(
        tmp_path,
        [
            [_CLEAN] * 12,
            [_CLEAN] * 11 + ["Shiller (/98/) argues"],
            [_CLEAN] * 12,
        ],
    )

    truths = GroundTruthExtractor().extract(pdf)

    assert [t.page_num for t in truths] == [1, 2, 3]
    assert [t.usable for t in truths] == [True, False, True]


def test_usable_truths_filters(tmp_path):
    pdf = _pdf(tmp_path, [[_CLEAN] * 12, [_CLEAN] * 11 + ["pp. /23-/45"]])

    truths = GroundTruthExtractor().extract(pdf)

    assert [t.page_num for t in usable_truths(truths)] == [1]


def test_save_writes_no_page_file_for_an_excluded_page(tmp_path):
    """A scorer reading this directory must not be able to find corrupted truth."""
    pdf = _pdf(tmp_path, [[_CLEAN] * 12, [_CLEAN] * 11 + ["Econometrica 47 (/979)"]])
    out = tmp_path / "gt"

    ex = GroundTruthExtractor()
    ex.save(ex.extract(pdf), out)

    assert (out / "page_1.txt").exists()
    assert not (out / "page_2.txt").exists()


def test_save_records_the_exclusion_count(tmp_path):
    pdf = _pdf(tmp_path, [[_CLEAN] * 12, [_CLEAN] * 11 + ["Econometrica 47 (/979)"]])
    out = tmp_path / "gt"

    ex = GroundTruthExtractor()
    ex.save(ex.extract(pdf), out)

    report = json.loads((out / "exclusions.json").read_text())
    assert report["pages_total"] == 2
    assert report["pages_usable"] == 1
    assert report["pages_excluded"] == 1
    assert report["excluded"][0]["page_num"] == 2


def test_excluded_text_never_reaches_full_txt(tmp_path):
    pdf = _pdf(tmp_path, [[_CLEAN] * 12, [_CLEAN] * 11 + ["Econometrica 47 (/979)"]])
    out = tmp_path / "gt"

    ex = GroundTruthExtractor()
    ex.save(ex.extract(pdf), out)

    assert "(/979)" not in (out / "full.txt").read_text()


# ---------------------------------------------------------------------------
# The breakdown that stops a quiet narrowing of the corpus
# ---------------------------------------------------------------------------


def test_exclusion_report_breaks_down_by_page_type():
    """Contaminated pages are not randomly distributed.

    Broken font maps cluster in older and denser documents — plausibly the same
    pages where escalation earns its keep. A scalar count hides whether the
    table stratum was gutted; the per-type breakdown is what makes that visible.
    """
    truths = [
        PageGroundTruth(1, "ok", 1, 2),
        PageGroundTruth(2, "bad", 1, 3, usable=False, exclusion_reason="eaten-digit"),
        PageGroundTruth(3, "bad", 1, 3, usable=False, exclusion_reason="eaten-digit"),
    ]

    report = exclusion_report(truths, page_types={1: "prose", 2: "table", 3: "table"})

    assert report["pages_excluded"] == 2
    assert report["excluded_by_page_type"] == {"table": 2}


def test_exclusion_report_handles_missing_page_types():
    truths = [PageGroundTruth(1, "bad", 1, 3, usable=False, exclusion_reason="x")]

    report = exclusion_report(truths)

    assert report["pages_excluded"] == 1
    assert report["excluded_by_page_type"] == {}
    assert report["excluded_fraction"] == pytest.approx(1.0)


def test_exclusion_report_on_empty_input_does_not_divide_by_zero():
    assert exclusion_report([])["excluded_fraction"] == 0.0
