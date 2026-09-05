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
- ``corpus/out/doc00/doc00/pages/00002.json`` + matching cache entry +
  ``audit_events`` — the D3 fail-closed-marker fallback path:
  ``winning_output.text`` is the marker; the real candidate text only
  exists in ``cache/aa/<hash>.json``, identified BY PROVENANCE (the
  ``table_binding_adjudicated`` audit event's ``engine`` field), never by
  running ``bind()`` on candidates and picking the best-scoring one.
- ``corpus/out/doc00/doc00/pages/00003.json`` — same fail-closed-marker
  shape, but TWO distinct cache candidates share the recorded provenance
  engine: ambiguous, so the row must come back ``unreplayable`` and
  ``bind()`` must never be called for it.
- ``corpus/out/doc00/doc00/pages/00004.json`` — same fail-closed-marker
  shape, but the table has TWO ``table_binding_adjudicated`` events naming
  DIFFERENT engines. Each engine has exactly one matching cache candidate
  (so the cache side ALONE would look unambiguous) — provenance itself is
  what is ambiguous here, one level up from the cache-collision case.
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

#: page 3: same shape again, used for the AMBIGUOUS-provenance case (two
#: distinct cache candidates both claiming the recorded engine).
P3_ROWS = [
    {"label": "Credit spread", "Value": "0.61"},
    {"label": "Term spread", "Value": "0.27"},
]
P3_CANDIDATE_LABELS = {"Credit spread": "3Y Credit spread", "Term spread": "Term spread"}
#: A second, textually distinct candidate for page 3 that ALSO relocates
#: the table (different row-label defect) -- both claim engine "qwen".
P3_ALT_CANDIDATE_LABELS = {"Credit spread": "10Y Credit spread", "Term spread": "Term spread"}

#: page 4: same shape again, used for the CONFLICTING-PROVENANCE-ENGINES
#: case -- two table_binding_adjudicated events name different engines,
#: each with exactly one cache candidate.
P4_ROWS = [
    {"label": "Breakeven inflation", "Value": "0.83"},
    {"label": "Real yield", "Value": "0.19"},
]
P4_CANDIDATE_LABELS = {"Breakeven inflation": "5Y Breakeven inflation", "Real yield": "Real yield"}
#: The gemini-provenance candidate for page 4: textually distinct, ALSO
#: relocates the table (own row-label defect).
P4_GEMINI_CANDIDATE_LABELS = {
    "Breakeven inflation": "10Y Breakeven inflation",
    "Real yield": "Real yield",
}


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
    page3 = doc.new_page()
    _draw_ruled_table(page3, P3_ROWS, TOP)
    page4 = doc.new_page()
    _draw_ruled_table(page4, P4_ROWS, TOP)
    doc.save(str(out_path))
    doc.close()


def _binding_adjudication_for(pdf_path: Path, page_num: int, markdown: str) -> dict:
    """Compute the REAL ``bind()`` contradiction for one page, so the fixture
    sidecar cannot drift from ``bind()``'s own behaviour."""
    import sys

    sys.path.insert(0, str(HERE.parents[2] / "src"))
    from socr.core.pdf import open_pdf
    from socr.tables.adjudication import items_from_binding
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
    md3 = _markdown_for(P3_ROWS, P3_CANDIDATE_LABELS)
    md3_alt = _markdown_for(P3_ROWS, P3_ALT_CANDIDATE_LABELS)
    md4 = _markdown_for(P4_ROWS, P4_CANDIDATE_LABELS)
    md4_gemini = _markdown_for(P4_ROWS, P4_GEMINI_CANDIDATE_LABELS)

    ba1, table_id1 = _binding_adjudication_for(pdf_path, 1, md1)
    ba2, table_id2 = _binding_adjudication_for(pdf_path, 2, md2)
    ba3, table_id3 = _binding_adjudication_for(pdf_path, 3, md3)
    ba4, table_id4 = _binding_adjudication_for(pdf_path, 4, md4)

    pages_dir = corpus_dir / "out" / "doc00" / "doc00" / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = corpus_dir / "out" / "doc00" / "doc00" / "cache" / "aa"
    cache_dir.mkdir(parents=True, exist_ok=True)

    sidecar1 = {
        "page_num": 1,
        "winning_output": {"text": md1},
        "binding_adjudication": {table_id1: ba1},
        "audit_events": [],
    }
    (pages_dir / "00001.json").write_text(json.dumps(sidecar1, indent=2, sort_keys=True) + "\n")

    marker2 = D3_MARKER.format(page_num=2)
    sidecar2 = {
        "page_num": 2,
        "winning_output": {"text": f"{marker2}\n\n![Failed table page 2](figures/x.png)"},
        "binding_adjudication": {table_id2: ba2},
        "audit_events": [
            {
                "kind": "table_binding_adjudicated",
                "engine": "qwen",
                "data": {"table_id": table_id2},
            }
        ],
    }
    (pages_dir / "00002.json").write_text(json.dumps(sidecar2, indent=2, sort_keys=True) + "\n")

    cache_entry2 = {"page_num": 2, "engine": "qwen", "text": md2}
    (cache_dir / "aa0000000000000000000000000000000000000000000000000000000000001.json").write_text(
        json.dumps(cache_entry2, indent=2, sort_keys=True) + "\n"
    )

    # Page 3: same fail-closed-marker shape, but TWO cache candidates share
    # the recorded provenance engine -- ambiguous, must come back
    # unreplayable, bind() must never be called for it.
    marker3 = D3_MARKER.format(page_num=3)
    sidecar3 = {
        "page_num": 3,
        "winning_output": {"text": f"{marker3}\n\n![Failed table page 3](figures/x.png)"},
        "binding_adjudication": {table_id3: ba3},
        "audit_events": [
            {
                "kind": "table_binding_adjudicated",
                "engine": "qwen",
                "data": {"table_id": table_id3},
            }
        ],
    }
    (pages_dir / "00003.json").write_text(json.dumps(sidecar3, indent=2, sort_keys=True) + "\n")

    cache_entry3a = {"page_num": 3, "engine": "qwen", "text": md3}
    (cache_dir / "aa0000000000000000000000000000000000000000000000000000000000002.json").write_text(
        json.dumps(cache_entry3a, indent=2, sort_keys=True) + "\n"
    )
    cache_entry3b = {"page_num": 3, "engine": "qwen", "text": md3_alt}
    (cache_dir / "aa0000000000000000000000000000000000000000000000000000000000003.json").write_text(
        json.dumps(cache_entry3b, indent=2, sort_keys=True) + "\n"
    )

    # Page 4: same fail-closed-marker shape, but the table has TWO
    # table_binding_adjudicated events naming DIFFERENT engines. Each
    # engine has exactly one cache candidate -- the cache side alone would
    # look unambiguous; provenance itself is the ambiguity here.
    marker4 = D3_MARKER.format(page_num=4)
    sidecar4 = {
        "page_num": 4,
        "winning_output": {"text": f"{marker4}\n\n![Failed table page 4](figures/x.png)"},
        "binding_adjudication": {table_id4: ba4},
        "audit_events": [
            {
                "kind": "table_binding_adjudicated",
                "engine": "qwen",
                "data": {"table_id": table_id4},
            },
            {
                "kind": "table_binding_adjudicated",
                "engine": "gemini",
                "data": {"table_id": table_id4},
            },
        ],
    }
    (pages_dir / "00004.json").write_text(json.dumps(sidecar4, indent=2, sort_keys=True) + "\n")

    cache_entry4_qwen = {"page_num": 4, "engine": "qwen", "text": md4}
    (cache_dir / "aa0000000000000000000000000000000000000000000000000000000000004.json").write_text(
        json.dumps(cache_entry4_qwen, indent=2, sort_keys=True) + "\n"
    )
    cache_entry4_gemini = {"page_num": 4, "engine": "gemini", "text": md4_gemini}
    (cache_dir / "aa0000000000000000000000000000000000000000000000000000000000005.json").write_text(
        json.dumps(cache_entry4_gemini, indent=2, sort_keys=True) + "\n"
    )

    print(f"Generated: {pdf_path}")
    print(f"Generated: {pages_dir / '00001.json'} (table_id={table_id1})")
    print(f"Generated: {pages_dir / '00002.json'} (table_id={table_id2}, fail-closed marker)")
    print(
        f"Generated: {pages_dir / '00003.json'} "
        f"(table_id={table_id3}, fail-closed marker, ambiguous cache provenance)"
    )
    print(
        f"Generated: {pages_dir / '00004.json'} "
        f"(table_id={table_id4}, fail-closed marker, conflicting provenance engines)"
    )


if __name__ == "__main__":
    main()
