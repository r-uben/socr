"""#1282 golden: the measured cells reproduce on the real pages (CI has no PDFs).

The 29 pages that carried a dropped decimal point in the 308-page measurement
(``scratch/army/decimals/measure/repairs.tsv``).  The PDFs and the model readings live under
``scratch/`` (gitignored), so every test here SKIPS, with the reason, when they are absent; the
same behaviour is covered by synthetic fixtures in ``test_gh1282_point_repair.py`` and
``test_gh1282_point_repair_pipeline.py``, which CI runs.

Pinned as differences: with the repair the measured rows are BOUND on the line that prints the
point and carry the PDF's own string; without it they are UNVERIFIABLE with the reason the
design names; a CONTRADICTED table is repaired in the prototype's list and NOT here.
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

import pytest

ARMY = Path(os.environ.get("SOCR_ARMY_DIR", "/Users/rubenffuertes/repos/tools/socr/scratch/army"))
REPAIRS = ARMY / "decimals" / "measure" / "repairs.tsv"
NO_LINE = "no native line carries these values"

pytestmark = pytest.mark.skipif(
    not (REPAIRS.is_file() and (ARMY / "1267").is_dir()),
    reason=f"the measured pages are not on this machine ({ARMY}); gitignored, CI has none",
)


def _cases():
    """``{stem: (pdf, reading markdown path)}`` for the 308 pages."""
    out = {}
    for line in (ARMY / "1267/measure/frame1.jsonl").read_text().splitlines():
        r = json.loads(line)
        s, p = r["stem"], r["page"]
        out[f"{s}__p{p:03d}"] = (
            ARMY / f"1267/pdf171/{s}__p{p:03d}.pdf",
            ARMY / f"precision/data/out1/pages/{s}/{p:05d}.md",
        )
    for line in (ARMY / "1267/measure/heldout.jsonl").read_text().splitlines():
        r = json.loads(line)
        s, p = r["stem"], r["page"]
        out[f"{s}__p{p:03d}"] = (
            ARMY / f"1267/heldout/heldout/pdfs/{s}__p{p:03d}.pdf",
            ARMY / f"1267/heldout/heldout/pages/{s}/{p:05d}.md",
        )
    return out


def _expected():
    rows = list(csv.DictReader(REPAIRS.open(encoding="utf-8"), delimiter="\t"))
    by_page: dict[str, list[dict]] = {}
    for r in rows:
        by_page.setdefault(r["page_stem"], []).append(r)
    return by_page


def _reading(path: Path) -> str:
    text = re.sub(r"^## Page \d+\s*", "", path.read_text(encoding="utf-8"))
    return "\n".join(line for line in text.splitlines() if not line.startswith(">"))


def _page_repairs(stem: str):
    import fitz

    from socr.tables.locate import locate_tables
    from socr.tables.row_binding_check import propose_page_point_repairs

    pdf, md = _cases()[stem]
    doc = fitz.open(pdf)
    page = doc[0]
    regions = [tuple(t.bbox) for t in locate_tables(page)] or None
    text = _reading(md)
    new, repaired, refused = propose_page_point_repairs(page, text, regions)
    return doc, page, text, new, repaired, refused, regions


def _row_has_a_sign_repair(reps, r) -> bool:
    """A row with a flipped minus is repaired whole (its dropped points too): the point-only rule
    refused such a row, so the 308-page list never had its cells."""
    return any(o.md_line == r.md_line and o.kind == "sign" for o in reps)


def test_the_cells_reproduce_on_the_shipping_pages_and_the_contradicted_ones_do_not():
    shipping_cells = 0
    shipping_pages = 0
    skipped_cells = 0
    for stem, rows in sorted(_expected().items()):
        doc, _page, text, new, repaired, _refused, _regions = _page_repairs(stem)
        got = Counter(
            (r.label, r.col, r.old, r.new, r.native_line)
            for _i, reps in repaired
            for r in reps
            if not _row_has_a_sign_repair(reps, r)  # those rows are pinned by the test below
        )
        want = Counter(
            (r["row_label"], int(r["column"]), r["old"], r["new"], int(r["native_line"]))
            for r in rows
            if r["table_outcome_before"] != "contradicted"
        )
        assert got == want, stem
        contradicted = [r for r in rows if r["table_outcome_before"] == "contradicted"]
        skipped_cells += len(contradicted)
        if want:
            shipping_pages += 1
            shipping_cells += sum(want.values())
            assert new != text, stem
        else:
            assert new == text, stem  # a withheld table's reading is not altered
        doc.close()
    # the design's 175 cells on 29 pages = 152 on 24 shipping pages + 23 on 5 withheld ones
    assert (shipping_cells, shipping_pages, skipped_cells) == (152, 24, 23)


def test_without_the_repair_the_rows_have_no_line_and_with_it_they_are_bound():
    from socr.tables.reconcile import find_table_blocks
    from socr.tables.row_binding_check import (
        RowVerdict,
        _page_frame,
        check_row_binding,
    )

    checked = 0
    for stem, rows in sorted(_expected().items()):
        shipping = [r for r in rows if r["table_outcome_before"] != "contradicted"]
        if not shipping:
            continue
        doc, page, text, new, repaired, _refused, regions = _page_repairs(stem)
        words, (boxes, union), _rot = _page_frame(page, regions)
        blocks = find_table_blocks(text)
        old_lines, new_lines = text.splitlines(), new.splitlines()
        for idx, reps in repaired:
            b = blocks[idx]
            region = boxes[idx] if len(boxes) == len(blocks) else union
            before = check_row_binding(words, region, "\n".join(old_lines[b.start : b.end + 1]))
            after = check_row_binding(words, region, "\n".join(new_lines[b.start : b.end + 1]))
            b_by = {e.line_index: e for e in before.rows}
            a_by = {e.line_index: e for e in after.rows}
            for line in {r.md_line for r in reps}:
                assert b_by[line].verdict is RowVerdict.UNVERIFIABLE, stem
                assert b_by[line].reason == NO_LINE, (stem, b_by[line].reason)
                assert a_by[line].verdict is RowVerdict.BOUND, stem
                checked += 1
            # no row that was BOUND lost it, and the table did not turn CONTRADICTED
            assert all(
                a_by[l].verdict is RowVerdict.BOUND
                for l, e in b_by.items()
                if e.verdict is RowVerdict.BOUND
            ), stem
            assert after.outcome.value == before.outcome.value, stem
        doc.close()
    assert checked > 0


def test_the_repaired_cell_is_the_pdfs_own_string():
    """Run the code: every cell it writes is a word the PDF itself prints, character for
    character, and never a zero-padded form the model would have used."""
    seen = 0
    for stem in sorted(_expected()):
        doc, page, _text, new, repaired, _refused, _regions = _page_repairs(stem)
        printed = {w[4] for w in page.get_text("words")}
        for _i, reps in repaired:
            for r in reps:
                assert r.new in printed, (stem, r.new)
                assert r.new in new, (stem, r.new)
                assert not r.new.lstrip("-+").startswith("0."), (stem, r.new)
                seen += 1
        doc.close()
    assert seen > 0


# The sign repairs (#1282 step 2) the 308 pages yield, each checked by hand against the rendered
# page image (2026-11-13): `(page stem, label, column, old, new)`.  Every one is the printed value.
# p085's `Previous Tealbook` row also gets its two dropped points (cols 6 and 9): that row used to be
# refused whole because of its flipped minus, and its three cells are all printed as written.
SIGN_REPAIRS = {
    (
        "fed-meetings-2008-2008-12-2008-12-16-greenbook-part1__p034",
        "Unit labor costs",
        6,
        ".2",
        "-.2",
    ),
    (
        "fed-meetings-2010-2010-06-2010-06-23-tealbook-parta__p085",
        "Previous Tealbook",
        6,
        "-2.",
        "-.2",
    ),
    (
        "fed-meetings-2010-2010-06-2010-06-23-tealbook-parta__p085",
        "Previous Tealbook",
        7,
        "-.8",
        ".8",
    ),
    (
        "fed-meetings-2010-2010-06-2010-06-23-tealbook-parta__p085",
        "Previous Tealbook",
        9,
        "-5.",
        "-.5",
    ),
    ("fed-meetings-2017-2017-06-2017-06-14-tealbook-parta__p130", "Japan", 3, ".3", "-.3"),
    ("fed-meetings-2017-2017-12-2017-12-13-tealbook-parta__p125", "Japan", 0, ".1", "-.1"),
    ("fed-meetings-2017-2017-12-2017-12-13-tealbook-parta__p125", "Japan", 1, ".3", "-.3"),
    ("fed-meetings-2017-2017-12-2017-12-13-tealbook-parta__p125", "China", 0, ".6", "-.6"),
}


def test_the_sign_repairs_on_the_real_pages_are_the_hand_checked_ones():
    got = set()
    for stem in sorted(_cases()):
        doc, _page, _text, _new, repaired, _refused, _regions = _page_repairs(stem)
        got |= {
            (stem, r.label, r.col, r.old, r.new)
            for _i, reps in repaired
            for r in reps
            if _row_has_a_sign_repair(reps, r)
        }
        doc.close()
    assert got == SIGN_REPAIRS
