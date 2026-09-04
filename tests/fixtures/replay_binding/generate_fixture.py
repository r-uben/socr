"""Generate the TICKET-A1 replay-binding fixture corpus.

Mirrors ``tests/fixtures/table_ladder/generate_fixture.py``'s contract:
semantically idempotent (same words/geometry/JSON every run; raw PDF bytes
vary because PyMuPDF embeds a random document ID). Synthesized content
only — no real data.

Lays out a MINIATURE version of a frozen corpus directory
(``~/Data/socr/ladder-run2-2026-09-04``'s shape), just enough for
``socr.benchmark.replay_binding`` to exercise both paths TICKET-A1 needs:

- ``corpus/in/doc00.pdf`` — one page, one ruled table, one row whose native
  label ("Treasury yield") the candidate markdown over-specifies ("2Y
  Treasury yield") — the exact GH-331 row-label shape A2 will repair.
- ``corpus/out/doc00/doc00/pages/00001.json`` — a sidecar whose
  ``winning_output.text`` is the REAL candidate markdown, and whose
  ``binding_adjudication`` is the row_label contradiction ``bind()``
  actually produces for this PDF+markdown pair (computed here, not
  hand-typed, so the fixture cannot silently drift from ``bind()``'s own
  behaviour).
- ``corpus/out/doc00/doc00/pages/00002.json`` + matching cache entry — the
  D3 fail-closed-marker fallback path: ``winning_output.text`` is the
  marker; the real candidate text only exists in
  ``cache/aa/<hash>.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

LEFT = 100.0
COL_W = 140.0
ROW_H = 22.0
TOP = 100.0

#: page 1: a two-row ruled table. Row "Treasury yield" carries a native
#: label the candidate markdown over-specifies as "2Y Treasury yield" (the
#: row_label contradiction shape). The value cell is deliberately
#: consistent (candidate and native SAME numeral) so the row-label
#: contradiction is the only signal.
P1_ROWS = [
    {"label": "Treasury yield", "Value": "0.48"},
    {"label": "Term premium", "Value": "0.12"},
]
P1_CANDIDATE_LABELS = {"Treasury yield": "2Y Treasury yield", "Term premium": "Term premium"}

#: page 2: same shape, used only to exercise the D3 fail-closed-marker
#: fallback (winning_output.text is the marker; the cache holds the real
#: candidate text).
P2_ROWS = [
    {"label": "Inflation swap", "Value": "1.75"},
    {"label": "Real rate", "Value": "0.33"},
]
P2_CANDIDATE_LABELS = {"Inflation swap": "5Y Inflation swap", "Real rate": "Real rate"}


def _draw_ruled_table(page, rows: list[dict], top: float, label_col: str = "label") -> None:
    import fitz  # noqa: F401  (imported for side effect parity with table_ladder fixture)

    col_xs = [LEFT, LEFT + COL_W, LEFT + 2 * COL_W]
    row_ys = [top + i * ROW_H for i in range(len(rows) + 2)]

    page.insert_text((col_xs[0] + 4, row_ys[0] + 15), "Label", fontsize=9)
    page.insert_text((col_xs[1] + 4, row_ys[0] + 15), "Value", fontsize=9)

    for r, row in enumerate(rows):
        y = row_ys[r + 1]
        page.insert_text((col_xs[0] + 4, y + 15), row[label_col], fontsize=9)
        page.insert_text((col_xs[1] + 4, y + 15), row["Value"], fontsize=9)

    for yy in row_ys:
        page.draw_line((col_xs[0], yy), (col_xs[-1], yy))
    for xx in col_xs:
        page.draw_line((xx, row_ys[0]), (xx, row_ys[-1]))


def _markdown_for(rows: list[dict], candidate_labels: dict[str, str]) -> str:
    lines = ["| Label | Value |", "| :--- | :--- |"]
    for row in rows:
        lines.append(f"| {candidate_labels[row['label']]} | {row['Value']} |")
    return "\n".join(lines) + "\n"


def generate_pdf(out_path: Path) -> None:
    import fitz

    doc = fitz.open()
    page1 = doc.new_page()
    _draw_ruled_table(page1, P1_ROWS, TOP)
    page2 = doc.new_page()
    _draw_ruled_table(page2, P2_ROWS, TOP)
    doc.save(str(out_path))
    doc.close()


def _binding_adjudication_for(pdf_path: Path, page_num: int, markdown: str) -> dict:
    """Compute the REAL ``bind()`` contradiction for one page, so the fixture
    sidecar cannot drift from ``bind()``'s own behaviour."""
    import sys

    sys.path.insert(0, str(HERE.parents[2] / "src"))
    from socr.core.pdf import open_pdf
    from socr.tables.adjudication import ContradictionItem, items_from_binding
    from socr.tables.binding import bind
    from socr.tables.witness import WitnessStatus, prepare_table_witnesses

    with open_pdf(pdf_path) as doc:
        words = doc[page_num - 1].get_text("words")
    with prepare_table_witnesses(pdf_path, page_num, markdown) as witnesses:
        assert len(witnesses) == 1, witnesses
        witness = witnesses[0]
        assert witness.status is WitnessStatus.LOCATED, witness
        result = bind(words, witness.markdown, region=witness.box.bbox)
    items = items_from_binding(result)
    assert items, "fixture must produce at least one contradiction"
    return {
        "status": "held",
        "markdown_sha256": "unused-in-fixture",
        "signatures": [list(item.signature()) for item in items],
        "items": [item.to_record(disproof=None) for item in items],
    }, witness.table_id


D3_MARKER = "[page {page_num} failed: unverifiable table — see image]"


def main() -> None:
    corpus_dir = HERE / "corpus"
    pdf_dir = corpus_dir / "in"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / "doc00.pdf"
    generate_pdf(pdf_path)

    md1 = _markdown_for(P1_ROWS, P1_CANDIDATE_LABELS)
    md2 = _markdown_for(P2_ROWS, P2_CANDIDATE_LABELS)

    ba1, table_id1 = _binding_adjudication_for(pdf_path, 1, md1)
    ba2, table_id2 = _binding_adjudication_for(pdf_path, 2, md2)

    pages_dir = corpus_dir / "out" / "doc00" / "doc00" / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = corpus_dir / "out" / "doc00" / "doc00" / "cache" / "aa"
    cache_dir.mkdir(parents=True, exist_ok=True)

    sidecar1 = {
        "page_num": 1,
        "winning_output": {"text": md1},
        "binding_adjudication": {table_id1: ba1},
    }
    (pages_dir / "00001.json").write_text(json.dumps(sidecar1, indent=2, sort_keys=True) + "\n")

    marker = D3_MARKER.format(page_num=2)
    sidecar2 = {
        "page_num": 2,
        "winning_output": {"text": f"{marker}\n\n![Failed table page 2](figures/x.png)"},
        "binding_adjudication": {table_id2: ba2},
    }
    (pages_dir / "00002.json").write_text(json.dumps(sidecar2, indent=2, sort_keys=True) + "\n")

    cache_entry = {"page_num": 2, "text": md2}
    (cache_dir / "aa0000000000000000000000000000000000000000000000000000000000001.json").write_text(
        json.dumps(cache_entry, indent=2, sort_keys=True) + "\n"
    )

    print(f"Generated: {pdf_path}")
    print(f"Generated: {pages_dir / '00001.json'} (table_id={table_id1})")
    print(f"Generated: {pages_dir / '00002.json'} (table_id={table_id2}, fail-closed marker)")


if __name__ == "__main__":
    main()
