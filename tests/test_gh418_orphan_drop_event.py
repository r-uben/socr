"""GH-418 step 1: the word-geometry rowizer's orphan-word drop must surface.

``_rowize_segment`` (``reconstruct.py``) has always dropped a word that lands
further than the snap radius from every column lane: a bare ``elif`` with no
``else``, no log, no event. The word never reaches ``grid_row`` and the table
ships without it -- silent content loss, on a citation corpus, with no trace
anywhere.

Panel ruling (three-seat, unanimous, ``docs/log/2026-09-16_418-design.md``):
surface it first, repair later, as two tickets. This file covers step 1 only
-- the event. **No grid cell, column count or markdown byte changes here.**

Scoping (panel refinement 1): a segment the prose guard (``_looks_tabular``)
rejects sends its page down the plain-text path, so a word dropped from that
segment is not a table loss -- reporting it would be noise. The distinction is
only available inside ``_rowize_word_group``, which buffers each segment's
drops locally and merges them into the caller's list ONLY on the branch that
also accepts the segment's markdown (see that function's own docstring).

Surfaces asserted here, one per requirement in the ticket ("not a log line"):
  * page level    -- ``PageState.orphan_word_drops``, reaching the page
    sidecar through its ``audit_events`` list (an ``orphan_word_dropped``
    ``AuditEvent`` per page with drops) -- no new sidecar key, matching
    GH-195's ``text_grid_rejected`` precedent, which surfaces the same way
    and adds none either.
  * metadata      -- ``audit_log.json``'s per-kind ``counts``
  * CLI           -- the ``console.print`` summary line in ``_phase_analyze``

Deliberately NOT a document-level orthogonal-assemble bucket: that vocabulary
records assemble *decisions* (what shipped, what failed, which ladder
terminal was reached), and by design step 1 changes none of those -- no
grid, no status, no disposition. A diagnostic does not belong in that list
(team-lead review, 2026-09-16); the per-kind ``counts`` in ``audit_log.json``
already gives the document-level surface the ticket asked for.

Every test here exercises the new ``orphan_drops=`` keyword or a field that
did not exist before this ticket, so each fails on an ``AttributeError`` or
``TypeError`` at the pre-fix revision, not merely on a changed value.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402
from socr.tables.reconstruct import rowize_from_word_list  # noqa: E402

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "table_repair" / "ce_like_p4.pdf"


def _ce_like_with_orphan_word(tmp_path: Path) -> Path:
    """The TR-1 fixture (real PyMuPDF-rendered page, no synthetic tuples) with
    one extra ROW appended below the main table: a label, a single numeric
    lane (``1.5``), and an orphan word (``n.a.``) mid-gutter.

    GH-418 step 2 retarget: the original version of this fixture put the
    orphan on ``Ashford Capital``'s row, which populates all 4 numeric
    lanes -- under step 2 that word is now CAPTURED, not dropped, so it no
    longer exercises this test's purpose (the step-1 drop event still
    reaching every surface). A single-numeric-lane row stays below the
    ``>= 2`` capture gate, so its orphan is still deleted and still recorded
    -- this is the residual the panel ruling requires stay measurable.

    ``(240.0, 144.0)`` is the insertion point for the new row's own text
    origin; PyMuPDF's own reported word bbox for it is
    ``(240.0, 135.4, 253.3, 146.4, "n.a.")`` (measured off this fixture's own
    ``get_text("words")`` output), 40pt from the ``GDP 2024`` lane (200),
    past the 18pt snap radius and right of the label boundary -- the exact
    geometry ``_rowize_segment``'s orphan branch exists for. ``1.5`` at
    x=200 is the row's only numeric lane, so the row's numeric-lane count is
    1, under the ``>= 2`` capture gate.
    """
    doc = fitz.open(str(FIXTURE_PDF))
    page = doc[0]
    page.insert_text((36.0, 144.0), "Gullwing Fund", fontsize=8, fontname="helv")
    page.insert_text((200.0, 144.0), "1.5", fontsize=8, fontname="helv")
    page.insert_text((240.0, 144.0), "n.a.", fontsize=8, fontname="helv")
    out = tmp_path / "ce_like_p4_orphan.pdf"
    doc.save(str(out))
    doc.close()
    return out


# ---------------------------------------------------------------------------
# 1. The drop reaches every surface: page, document, metadata, CLI.
# ---------------------------------------------------------------------------


def test_orphan_drop_reaches_every_surface(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pdf_path = _ce_like_with_orphan_word(tmp_path)

    pipeline = UnifiedPipeline(
        PipelineConfig(
            agentic=True,
            judge_backend="heuristic",
            native_first=True,
            native_only=False,
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            tiered=False,
            dual_pass_tables=False,
            detect_equations=False,
            save_figures=False,
            write_manifest=True,
            quiet=False,
        )
    )
    # CI hermeticity: no ollama, no provider ladder, no judge backend probe.
    pipeline._available_engines_for_agentic = lambda: []
    pipeline._resolve_judge_model = lambda: ""

    capsys.readouterr()
    output_dir = tmp_path / "out"
    pipeline.process(pdf_path, output_dir)
    cli_output = capsys.readouterr().out

    # -- page level (via the audit_events channel, not a new sidecar key --
    # GH-195's text_grid_rejected precedent adds none either) -------------
    doc_dir = output_dir / "ce_like_p4_orphan"
    sidecar = json.loads((doc_dir / "pages" / "00001.json").read_text(encoding="utf-8"))

    events = [e for e in sidecar["audit_events"] if e["kind"] == "orphan_word_dropped"]
    assert len(events) == 1, sidecar["audit_events"]
    assert "n.a." in events[0]["detail"]
    assert events[0]["data"] == {"dropped_count": 1, "words": ["n.a."]}

    # -- metadata (audit_log.json counts) -----------------------------
    audit = json.loads((doc_dir / "audit_log.json").read_text(encoding="utf-8"))
    assert audit["counts"].get("orphan_word_dropped") == 1, audit["counts"]

    # -- CLI ------------------------------------------------------------
    assert "dropped by the table rowizer" in cli_output, cli_output
    assert "n.a." not in cli_output or "1 word(s) dropped" in cli_output


# ---------------------------------------------------------------------------
# 2. Scoping: a segment the prose guard rejects produces NO record.
# ---------------------------------------------------------------------------


def _w(x: float, y: float, text: str, width: float = 26.0, h: float = 10.0) -> tuple:
    return (x, y, x + width, y + h, text, 0, 0, 0)


def _shipping_segment_with_orphan() -> list:
    """Four numeric lanes, three clean data rows, one orphan mid-gutter word.

    Passes ``_looks_tabular`` (all three rows are 4-cell data rows) -- the
    positive control this scoping test is paired against.
    """
    words: list = []
    lanes = [100.0, 220.0, 340.0, 460.0]
    y = 100.0
    for r in range(3):
        words.append(_w(36.0, y, f"Row{r}"))
        for c, x in enumerate(lanes):
            words.append(_w(x, y, f"{r}{c}.5"))
        if r == 0:
            words.append(_w(160.0, y, "note"))  # 60pt from both flanking lanes
        y += 20.0
    return words


def _rejected_segment_with_orphan() -> list:
    """Same lanes, same orphan word, but diluted with single-numeric-cell rows
    so fewer than half the non-empty rows carry >=2 numeric cells --
    ``_looks_tabular``'s own majority gate, so the segment is rejected and the
    page falls back to plain text. Row/marker labels avoid embedded digits
    (``Note``/``Extra``, not ``Dil0``..``Dil4``): a digit in the label cell
    itself is a numeric token by ``_looks_tabular``'s own regex and would
    silently inflate the data-row count through the label, not the data.
    """
    words: list = []
    lanes = [100.0, 220.0, 340.0, 460.0]
    y = 100.0
    for r in range(3):
        words.append(_w(36.0, y, f"Row{r}"))
        for c, x in enumerate(lanes):
            words.append(_w(x, y, f"{r}{c}.5"))
        y += 20.0
    words.append(_w(36.0, y, "Note"))
    words.append(_w(160.0, y, "note"))  # the orphan word, on a diluting row
    y += 20.0
    for _ in range(4):
        words.append(_w(36.0, y, "Extra"))
        words.append(_w(100.0, y, "x.x"))  # non-numeric filler, only lane 0
        y += 20.0
    return words


def test_shipping_segment_orphan_is_recorded() -> None:
    """Positive control for the scoping test below: the SAME geometry, when
    the segment ships, does produce a record naming the word.

    GH-418 step 2 retarget: this row populates all 4 numeric lanes (>= 2), so
    the orphan is now CAPTURED into the trailing column instead of dropped --
    it must therefore no longer report as a drop. The scoping test right
    below this one is unaffected (its diluting rows keep it under the
    ``>= 2`` gate), and stays the positive control that a genuine drop still
    reaches ``orphan_drops``.
    """
    words = _shipping_segment_with_orphan()
    drops: list[dict] = []
    regions = rowize_from_word_list(words, orphan_drops=drops)

    assert regions, "fixture must produce a shipping table region"
    assert drops == [], f"a captured orphan must not also report as a drop: {drops}"
    grid = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in regions[0][1].splitlines()
        if line.lstrip().startswith("|") and "---" not in line
    ]
    row0 = [row for row in grid if row and row[0] == "Row0"][0]
    assert row0[-1] == "note", f"the orphan must be captured in the trailing column: {row0}"


def test_scoping_no_record_when_segment_is_rejected() -> None:
    """The load-bearing negative: the identical drop on a segment the prose
    guard rejects produces NO record -- reporting it would be noise for a
    page that never becomes a table."""
    words = _rejected_segment_with_orphan()
    drops: list[dict] = []
    regions = rowize_from_word_list(words, orphan_drops=drops)

    assert regions == [], "setup: this segment must be REJECTED by _looks_tabular"
    assert drops == [], (
        "a word dropped from a segment the prose guard rejected reached "
        f"orphan_drops anyway: {drops}"
    )


# ---------------------------------------------------------------------------
# 3. A page with no drops: no record, byte-identical output.
# ---------------------------------------------------------------------------


def test_clean_page_produces_no_record_and_is_byte_identical() -> None:
    """Pinned as a DIFFERENCE, not an absolute: run the pristine (unmodified)
    TR-1 fixture's own main-table words through ``rowize_from_word_list``
    twice in the same process -- once with ``orphan_drops`` wired, once
    without -- and assert the two calls are byte-identical AND the wired call
    records nothing. This is the page-major flush path's contract (the final
    assembled ``.md`` must be byte-identical to the whole-doc assembly): a
    page that carries no drop must be completely unaffected by this ticket.
    """
    doc = fitz.open(str(FIXTURE_PDF))
    page = doc[0]
    main_bbox = fitz.Rect(36.0, 40.0, 520.0, 150.0)
    words = [
        w for w in page.get_text("words") if fitz.Rect(w[0], w[1], w[2], w[3]).intersects(main_bbox)
    ]
    doc.close()

    without = rowize_from_word_list(words)
    drops: list[dict] = []
    with_wiring = rowize_from_word_list(words, orphan_drops=drops)

    assert drops == [], f"the clean fixture must not produce any drop record: {drops}"
    assert [md for _, md in with_wiring] == [md for _, md in without], (
        "wiring orphan_drops changed the shipped markdown on a page with no drops"
    )


# ---------------------------------------------------------------------------
# 4. Grid output is unchanged for fixtures that already have drops today.
# ---------------------------------------------------------------------------


def test_grid_unchanged_on_the_ce_like_fixture_with_an_orphan_word(tmp_path: Path) -> None:
    pdf_path = _ce_like_with_orphan_word(tmp_path)
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    words = page.get_text("words")
    doc.close()

    without = rowize_from_word_list(words)
    with_wiring = rowize_from_word_list(words, orphan_drops=[])

    assert [md for _, md in with_wiring] == [md for _, md in without], (
        "wiring orphan_drops changed shipped markdown on a fixture with a real drop"
    )


def test_grid_unchanged_on_gh342_gutter_marker_fixture() -> None:
    """GH-342's own fixture already exercises a word dropped by
    ``_rowize_segment`` (see that test file's ``test_no_data_value_is_lost_
    into_the_label``, which asserts ``"n.a." not in emitted`` as a
    pre-existing, deliberate placement rule). Reuse it here to confirm THIS
    ticket changes no cell on a fixture the repo already relies on for that
    exact behaviour.
    """
    from test_gh342_stub_promotion_runaway import _four_data_lanes

    words = _four_data_lanes(markers=True)
    without = rowize_from_word_list(words)
    with_wiring = rowize_from_word_list(words, orphan_drops=[])

    assert [md for _, md in with_wiring] == [md for _, md in without], (
        "wiring orphan_drops changed shipped markdown on the GH-342 fixture"
    )
