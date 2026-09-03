#!/usr/bin/env python3
"""GH-548: how many corpus pages does each DEMOTED_NATIVE trigger claim?

`PageEnding.DEMOTED_NATIVE` is a panel-approved temporary fourth ending, and its
exit criterion is per-trigger: enumerate the corpus by trigger, hand-check
fidelity per trigger, then assign each independently to `NATIVE_PROSE` (intact
trustworthy prose) or `FAIL_CLOSED_MARKER` (cannot safely ship without OCR).
Neither blanket answer is a refactor -- both change shipped bytes and resume
skippability -- so the assignment needs the counts first.

The four triggers, and which of them this sweep can see:

* ``needs_ocr_enhancement``   -- detector flag. MEASURED here.
* ``text_grid_rejected``      -- set from ``PageAssessment.text_grid_rejections``
                                 during native extraction. MEASURED here.
* native table defect         -- the union of ``native_table_structure_failed``,
                                 ``native_table_unverifiable``,
                                 ``native_table_structure_defective`` and
                                 ``native_table_header_unattributed``. Every
                                 member the DETECTOR exposes is measured; see
                                 the caveat below for the one that is not.
* ``chart_asset_render_failed`` -- NOT measurable without a real run. It is a
                                 PNG render/save failure, so it needs the render
                                 to fail; nothing about a PDF predicts it. Its
                                 count here is structurally zero and is reported
                                 as "not measurable", never as "does not occur".

**The caveat that decides how to read the numbers.** ``native_table_structure_failed``
is an orchestrator PageState flag set during the pipeline's own native ship, not
detector output, so no corpus sweep can see it. The other three members are all
on ``PageAssessment`` and all counted here. So the defect column is a LOWER
BOUND on that trigger, and the sweep says so rather than presenting it as the
count.

Content-free per the 2026-08-22 measurement convention: counts and basenames
only, never page text.

Usage::

    PYTHONPATH=src ~/venvs/socr/bin/python docs/log/2026-09-03_demoted-native-triggers.py \\
        <dir containing the corpus PDFs> [--limit N]
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

import fitz

from socr.core.born_digital import BornDigitalDetector

#: ``needs_ocr_enhancement`` is not one condition. The detector sets it from
#: three unrelated causes, and they do not get the same answer:
#:
#: * ``has_corrupt_math``            -- font-map mojibake in the math spans.
#: * ``native_table_lane_refused``   -- rotated page, table reconstruction
#:                                      refused, prose deliberately RETAINED.
#: * ``native_rotated_text_shredded`` -- the extracted lines are pieces of one
#:                                      text run.
#:
#: So the ticket's four triggers are really six, and this one cannot be assigned
#: to NATIVE_PROSE or FAIL_CLOSED_MARKER until it is split. Measured separately
#: for that reason.
NEEDS_OCR_SUBCAUSES = (
    "has_corrupt_math",
    "native_table_lane_refused",
    "native_rotated_text_shredded",
)

#: The triggers, in the order the ticket lists them.
TRIGGERS = (
    "needs_ocr_enhancement",
    "text_grid_rejected",
    "native_table_defect_lower_bound",
    "chart_asset_render_failed",
)


def _flag(name: str) -> str | None:
    if name not in sys.argv:
        return None
    index = sys.argv.index(name) + 1
    if index >= len(sys.argv) or sys.argv[index].startswith("--"):
        raise SystemExit(f"{name} needs a value")
    return sys.argv[index]


def _int_flag(name: str) -> int | None:
    raw = _flag(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"{name} needs a whole number, got {raw!r}") from None
    if value <= 0:
        raise SystemExit(f"{name} needs a positive number, got {value}")
    return value


def _page_triggers(assessment) -> set[str]:
    """Which DEMOTED_NATIVE triggers this page would fire, as far as detection
    can tell. A page can fire more than one, which is exactly why the ticket
    asks for a per-trigger assignment and not a single verdict."""
    fired: set[str] = set()
    if getattr(assessment, "needs_ocr_enhancement", False):
        fired.add("needs_ocr_enhancement")
    if getattr(assessment, "text_grid_rejections", None):
        fired.add("text_grid_rejected")
    # Every member of the defect union the DETECTOR exposes (cubic P2 on #576).
    # ``native_table_structure_failed`` is the one that stays out: it is an
    # orchestrator PageState flag set during the native ship, which is why this
    # column is a lower bound rather than the count.
    if (
        getattr(assessment, "has_unverifiable_table_region", False)
        or getattr(assessment, "native_table_unverifiable_ordinals", None)
        or getattr(assessment, "native_table_structure_defective", False)
        or getattr(assessment, "native_table_header_unattributed", False)
    ):
        fired.add("native_table_defect_lower_bound")
    return fired


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: demoted-native-triggers.py <pdf dir> [--limit N]")
    pdf_dir = Path(sys.argv[1])
    limit = _int_flag("--limit")

    # The detector logs per-region diagnostics at WARNING; deafening across the
    # corpus, and it buried a traceback during the GH-511 sweep.
    logging.disable(logging.CRITICAL)

    detector = BornDigitalDetector()
    pdfs = sorted(pdf_dir.glob("*.pdf"))[:limit]

    tally: Counter[str] = Counter()
    per_trigger_papers: dict[str, set[str]] = {t: set() for t in TRIGGERS}
    combos: Counter[str] = Counter()
    subcauses: Counter[str] = Counter()
    subcause_papers: dict[str, set[str]] = {c: set() for c in NEEDS_OCR_SUBCAUSES}

    for path in pdfs:
        try:
            assessment = detector.detect(path)
        except Exception as exc:  # noqa: BLE001 - an unreadable file is not a measurement
            # Named, not swallowed (cubic P2 on #576). A silent skip lets the
            # sweep publish corpus totals with a hole in them and no way to see
            # which file made it -- the GH-511 sweep lost a whole run that way.
            print(f"  skipped {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            tally["unreadable_pdf"] += 1
            continue
        tally["paper"] += 1
        for pa in assessment.pages:
            tally["page"] += 1
            if not pa.is_born_digital:
                tally["scanned_page"] += 1
                continue
            tally["born_digital_page"] += 1

            fired = _page_triggers(pa)
            if not fired:
                tally["no_trigger"] += 1
                continue
            tally["demoted_native_page"] += 1
            for trigger in fired:
                tally[trigger] += 1
                per_trigger_papers[trigger].add(path.name)
            combos["+".join(sorted(fired))] += 1

            if "needs_ocr_enhancement" in fired:
                named = False
                for cause in NEEDS_OCR_SUBCAUSES:
                    if getattr(pa, cause, False):
                        subcauses[cause] += 1
                        subcause_papers[cause].add(path.name)
                        named = True
                if not named:
                    # The flag is set and none of the three known causes is
                    # recorded on the assessment. Counted rather than assumed
                    # away: an unnamed cause cannot be assigned an ending.
                    subcauses["unattributed"] += 1

    born = tally["born_digital_page"] or 1
    print(
        f"corpus: {tally['paper']} paper(s), {tally['unreadable_pdf']} unreadable; "
        f"{tally['page']} pages ({tally['born_digital_page']} born-digital, "
        f"{tally['scanned_page']} scanned)\n"
    )
    print(
        f"  born-digital pages firing at least one trigger: {tally['demoted_native_page']} "
        f"({100.0 * tally['demoted_native_page'] / born:.2f}%)\n"
    )

    print("  per trigger (a page can fire more than one):")
    for trigger in TRIGGERS:
        if trigger == "chart_asset_render_failed":
            print(f"    {trigger:34s}  not measurable without a real run -- see the docstring")
            continue
        count = tally[trigger]
        print(
            f"    {trigger:34s} {count:7d}  ({100.0 * count / born:.2f}% of born-digital, "
            f"{len(per_trigger_papers[trigger])} papers)"
        )

    if subcauses:
        total = tally["needs_ocr_enhancement"] or 1
        print("\n  needs_ocr_enhancement decomposed (a page can carry more than one):")
        for cause in (*NEEDS_OCR_SUBCAUSES, "unattributed"):
            count = subcauses.get(cause, 0)
            papers = len(subcause_papers.get(cause, ()))
            suffix = f", {papers} papers" if cause in subcause_papers else ""
            print(
                f"    {cause:34s} {count:7d}  ({100.0 * count / total:.2f}% of the trigger{suffix})"
            )

    if combos:
        print("\n  co-occurrence (which triggers fire together on one page):")
        for combo, count in combos.most_common():
            print(f"    {count:7d}  {combo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
