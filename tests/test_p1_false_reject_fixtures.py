"""P1 (task t12): the three false-reject shapes the owner ruling named, each a
GENUINE false reject that the ruled chain clears — with the PDF builder, not
the markdown, as the oracle for what is physically on the page.

The ruling recorded three ways a double reader rejection is wrong, so the
guards could be aimed at them:

1. a spanning header over column groups, plus a notes row, read as
   misalignment;
2. a bad crop -- cut or rotated -- fooling both readers identically;
3. decoy-tuned prompts over-triggering on a dense table.

Round 1 replaced a patched ``evaluate_cell_guard`` with the real chain. Round 2
made the markdown faithful to the page. Round 3 found the remaining hole, which
was the important one: the "truth" the blind reader answered from was DERIVED
FROM THE MARKDOWN UNDER TEST, and the faithfulness canary only checked that
each token appeared *somewhere* in the page text. Swapping two flagged values
in the dense extraction left the whole file green while the chain published a
wrong table as ``verified_by_blind_cell_transcription`` -- the exact scenario
these fixtures exist to exclude.

**The builder is now the oracle.** ``_grid_pdf`` returns, alongside the PDF,
the map of the cells it physically DREW, keyed by canonical reference and
recorded at draw time. From that map everything follows:

* the mocked blind reader answers ONLY from the drawn map -- it reports what
  is on the page, never what the markdown claims;
* "the rejection is FALSE" is asserted as an equality, not a membership test:
  the extraction under test must equal the drawn map cell for cell;
* and the mutation is pinned. Swap two values in the extraction and the chain
  must NOT clear it, which is the property the round-2 file lacked.

The shapes are also rendered as described. The spanning fixture draws ONE
header cell spanning two columns -- a single centred token, with the vertical
rules broken around it -- rather than the same word twice in two ordinary
cells. The rotated fixture applies rotation exactly once, to the page, so what
is rendered is the described table turned on its side, and its drawn map is
recorded in the pre-rotation frame the extraction uses.

Only the adjudicator's network seam (``table_rung_ollama._post_chat``) is
mocked. The real payload builder, the real base64 of the real crop, the real
strict parser, the real binding check, the real comparison and the real
terminal selection all execute.

Between them the three fixtures cover both routes by which the chain clears a
false reject, and both causes of an abstaining geometry:

======================  ==================  ==================================
fixture                 geometry            how the false reject is cleared
======================  ==================  ==================================
spanning header+notes   PASS                free, by geometry; no call is made
rotated crop            ABSTAIN (rotation)  by the blind reader's agreement
dense scan              ABSTAIN (no text)   by the blind reader's agreement
======================  ==================  ==================================
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.judge.table_verdict import (
    Finding,
    FindingCode,
    RungResult,
    TableJudgeVerdict,
    resolve_cell_refs,
)
from socr.pipeline.orchestrator import UnifiedPipeline

#: The canonical reference grammar, as a matcher (round 5).
_REF_PATTERN = re.compile(r"[RH]\d+C\d+")

#: One drawn cell: its token and how many physical columns it spans.
Cell = tuple[str, int]

_FONT_SIZE = 8


def _grid_pdf(
    path: Path,
    header_rows: list[list[Cell]],
    body_rows: list[list[Cell]],
    *,
    page_rotation: int = 0,
) -> tuple[Path, dict[str, str]]:
    """Draw a ruled grid and RETURN WHAT WAS DRAWN, keyed by canonical ref.

    The second return value is the oracle (round 3, NEW B). It is built in
    this loop, from the tokens actually handed to ``insert_text`` and the
    columns they actually occupy, so nothing downstream can claim a cell holds
    something the page does not show.

    A cell with ``span > 1`` is drawn ONCE, centred over the columns it
    covers, with the vertical rules broken around it -- a genuinely merged
    header cell rather than the same word repeated in two ordinary cells. It
    is recorded against every column it spans, which is the markdown
    convention for a merged cell and therefore also what a blind reader asked
    about any of those columns would correctly report.

    ``page_rotation`` is applied ONCE, to the page, after everything is drawn.
    The rendered page is then the described table rotated, and the drawn map
    stays in the pre-rotation frame -- the frame the extraction uses.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = header_rows + body_rows
    ncols = max(sum(span for _t, span in row) for row in rows)
    left, top, colw, rowh = 55, 90, 92, 24
    xs = [left + i * colw for i in range(ncols + 1)]
    ys = [top + i * rowh for i in range(len(rows) + 1)]

    doc = fitz.open()
    page = doc.new_page()
    drawn: dict[str, str] = {}
    for r, row in enumerate(rows):
        is_header = r < len(header_rows)
        prefix = "H" if is_header else "R"
        n = r + 1 if is_header else r - len(header_rows) + 1
        col = 0
        for token, span in row:
            x0, x1 = xs[col], xs[col + span]
            if token:
                width = fitz.get_text_length(token, fontsize=_FONT_SIZE)
                page.insert_text(
                    (max(x0 + 2, (x0 + x1 - width) / 2), ys[r] + 15), token, fontsize=_FONT_SIZE
                )
            for k in range(span):
                drawn[f"{prefix}{n}C{col + k + 1}"] = token
            # The left edge of THIS cell only: a spanning cell has no rule
            # running through the middle of it.
            page.draw_line((x0, ys[r]), (x0, ys[r + 1]))
            col += span
        page.draw_line((xs[-1], ys[r]), (xs[-1], ys[r + 1]))
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y))
    if page_rotation:
        page.set_rotation(page_rotation)
    doc.save(path)
    doc.close()
    return path, drawn


def _vertical_rules_by_band(page) -> list[list[int]]:
    """The x positions of the vertical rules in each row band, top band first.

    Used by the merged-header canary (round 4, NEW 3) to check WHICH rules are
    absent, not merely how many.
    """
    bands: dict[int, set[int]] = {}
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] != "l":
                continue
            start, end = item[1], item[2]
            if abs(start.x - end.x) >= 0.01:
                continue
            bands.setdefault(round(min(start.y, end.y)), set()).add(round(start.x))
    return [sorted(bands[y]) for y in sorted(bands)]


def _rasterise(src_path: Path, out_path: Path) -> Path:
    """Re-render a page as pure pixels: same image, no text layer."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src = fitz.open(src_path)
    pix = src[0].get_pixmap(dpi=150)
    out = fitz.open()
    page = out.new_page(width=src[0].rect.width, height=src[0].rect.height)
    page.insert_image(src[0].rect, pixmap=pix)
    out.save(out_path)
    out.close()
    src.close()
    return out_path


# ---------------------------------------------------------------------------
# The three shapes
# ---------------------------------------------------------------------------


def spanning_header_pdf(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    return _grid_pdf(
        tmp_path / "spanning" / "doc.pdf",
        [
            [("Region", 1), ("Panel A", 2), ("Panel B", 2)],
            [("Region", 1), ("Early", 1), ("Late", 1), ("Early", 1), ("Late", 1)],
        ],
        [
            [("North", 1), ("11", 1), ("12", 1), ("41", 1), ("42", 1)],
            [("South", 1), ("21", 1), ("22", 1), ("51", 1), ("52", 1)],
            [("Notes: revised series.", 1), ("", 1), ("", 1), ("", 1), ("", 1)],
        ],
    )


SPANNING_MD = (
    "| Region | Panel A | Panel A | Panel B | Panel B |\n"
    "| Region | Early | Late | Early | Late |\n"
    "| --- | --- | --- | --- | --- |\n"
    "| North | 11 | 12 | 41 | 42 |\n"
    "| South | 21 | 22 | 51 | 52 |\n"
    "| Notes: revised series. |  |  |  |  |\n"
)


def rotated_crop_pdf(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    return _grid_pdf(
        tmp_path / "rotated" / "doc.pdf",
        [[("Zone", 1), ("Gamma", 1), ("Delta", 1)]],
        [[("East", 1), ("31", 1), ("32", 1)], [("West", 1), ("41", 1), ("42", 1)]],
        page_rotation=90,
    )


ROTATED_MD = (
    "| Zone | Gamma | Delta |\n| --- | --- | --- |\n| East | 31 | 32 |\n| West | 41 | 42 |\n"
)

DENSE_ROWS = [(f"r{i}", f"{i}.01", f"{i}.02", f"{i}.03", f"{i}.04") for i in range(1, 13)]
DENSE_MD = "| id | q1 | q2 | q3 | q4 |\n| --- | --- | --- | --- | --- |\n" + "".join(
    "| " + " | ".join(row) + " |\n" for row in DENSE_ROWS
)


def dense_decoy_text_pdf(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """The dense grid WITH a text layer. The scan below is a picture of it."""
    return _grid_pdf(
        tmp_path / "dense_text" / "doc.pdf",
        [[(c, 1) for c in ("id", "q1", "q2", "q3", "q4")]],
        [[(c, 1) for c in row] for row in DENSE_ROWS],
    )


def dense_decoy_pdf(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    src, drawn = dense_decoy_text_pdf(tmp_path)
    return _rasterise(src, tmp_path / "dense_scan" / "doc.pdf"), drawn


#: Which branch of the ruled chain each shape's REAL native geometry takes, and
#: therefore how the ruling says its false reject gets cleared. Pinned rather
#: than skipped: ``bind()`` is pure local geometry over the PDF's own text
#: layer -- no daemon, no binary, no network -- so unlike a provider-dependent
#: outcome it is the same on every machine.
GEOMETRY_BRANCH = {
    "spanning_header_and_notes_row": "pass",
    "cut_or_rotated_crop": "abstain",
    "dense_decoy_scan": "abstain",
}

#: builder, emitted markdown, the cells the readers WRONGLY flag, and a pair of
#: body cells whose values the mutation test swaps.
SHAPES = {
    "spanning_header_and_notes_row": (
        spanning_header_pdf,
        SPANNING_MD,
        [(FindingCode.HEADER_MANGLED, "H1C2"), (FindingCode.STRUCTURE_MERGED, "R1C3")],
        ("R1C2", "R1C3"),
    ),
    "cut_or_rotated_crop": (
        rotated_crop_pdf,
        ROTATED_MD,
        [(FindingCode.FABRICATED_VALUE, "R1C2"), (FindingCode.FABRICATED_VALUE, "R2C3")],
        ("R1C2", "R2C3"),
    ),
    "dense_decoy_scan": (
        dense_decoy_pdf,
        DENSE_MD,
        [(FindingCode.FABRICATED_VALUE, "R7C2"), (FindingCode.FABRICATED_VALUE, "R11C4")],
        ("R7C2", "R11C4"),
    ),
}


def _swap(markdown: str, drawn: dict[str, str], a: str, b: str) -> str:
    """Return ``markdown`` with the tokens at refs ``a`` and ``b`` exchanged.

    A minimal, targeted corruption: the page-wide token multiset is unchanged,
    so nothing but an actual cell-level comparison can notice it. That is
    exactly the mutation the round-2 file survived.
    """
    token_a, token_b = drawn[a], drawn[b]
    assert token_a != token_b, "the mutation must actually change something"
    out = []
    for line in markdown.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if set("".join(cells)) <= {"-"} or cells == []:
            out.append(line)
            continue
        cells = [token_b if c == token_a else token_a if c == token_b else c for c in cells]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Gate harness
# ---------------------------------------------------------------------------


def _config(**overrides) -> PipelineConfig:
    kwargs = dict(
        primary_engine=EngineType.QWEN,
        agentic=True,
        judge_backend="heuristic",
        enabled_engines=[EngineType.QWEN],
        quiet=True,
        save_figures=False,
        write_manifest=False,
        table_judge_ladder=True,
    )
    kwargs.update(overrides)
    return PipelineConfig(**kwargs)


def _make_state(pdf_path: Path) -> DocumentState:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf_path, page_count=1)
    return DocumentState(handle=handle)


class _QueueRung:
    def __init__(self, results, rung_id="fake"):
        self._results = list(results)
        self.rung_id = rung_id
        self.calls = []

    def __call__(self, crop_path, markdown, prior_findings):
        self.calls.append((crop_path, markdown, prior_findings))
        return self._results.pop(0)


def _rejecting_rung(codes_and_wheres):
    findings = [Finding(code=c, where=w, detail="flagged") for c, w in codes_and_wheres]
    return _QueueRung(
        [
            RungResult(
                rung="r1",
                ok=True,
                verdict=TableJudgeVerdict(verdict="FAIL", confidence="high", findings=findings),
            )
        ],
        "r1",
    )


class _BlindReaderOfThePage:
    """The adjudicator's ONE network seam, answering from the DRAWN map.

    This is the whole point of round 3's ruling. The mock is a stand-in for a
    model looking at the crop, so it must answer what the page shows -- not
    what the markdown under test claims. ``distort`` lets a test make it read
    something else, which is how the disagreement arm is driven; refs the page
    has no cell for come back as the typed ``null`` non-reading.
    """

    def __init__(self, drawn: dict[str, str], distort=None):
        self.drawn = drawn
        self.distort = distort or (lambda ref, token: token)
        self.payloads: list[dict] = []

    def __call__(self, host, payload, timeout):
        self.payloads.append(payload)
        prompt = payload["messages"][0]["content"]
        asked = [ref for ref in self.drawn if f"{ref}," in prompt or prompt.rstrip().endswith(ref)]
        return json.dumps({ref: self.distort(ref, self.drawn[ref]) for ref in asked})


#: How far from a coordinate a copied token may sit, in characters. Wide
#: enough to span "for R1C2 write N/A", "R1C2: N/A" and a markdown row.
_ECHO_WINDOW = 48


def tokens_near_reference(prompt: str, ref: str, window: int = _ECHO_WINDOW) -> list[str]:
    """Every token sitting within ``window`` characters of ``ref`` in ``prompt``.

    Cold review round 6. The previous echo reader recognised two SPELLINGS of a
    coordinate-to-value binding, so a third one -- ``for R1C2 write N/A`` --
    slipped past it while publishing a wrong table. Proximity is spelling-free:
    whatever syntax a binding is written in, the value has to sit next to the
    coordinate it binds, or a reader could not act on it either.

    Other coordinates are excluded (they are the request list's own commas'
    neighbours, never an answer).
    """
    candidates: list[str] = []
    for match in re.finditer(re.escape(ref), prompt):
        start = max(0, match.start() - window)
        neighbourhood = prompt[start : match.end() + window]
        for token in re.findall(r"[^\s\"\'`,:;()\[\]{}|]+", neighbourhood):
            token = token.strip(".")
            if not token or _REF_PATTERN.fullmatch(token) or token in candidates:
                continue
            candidates.append(token)
    return candidates


class _PromptCopyingReader:
    """A blind reader that never looks at the image and echoes the prompt.

    Rounds 4, 5 and 6, NEW 1. It answers each requested coordinate with a token
    the prompt places NEXT TO that coordinate, and reports a non-reading when
    the prompt places none. This is the failure mode an answer key in the
    prompt enables: a text-only model with one in front of it returns a
    schema-valid match without seeing a single pixel.

    ``pick`` selects which neighbouring token it copies, so a test can sweep
    every candidate the prompt offers rather than trusting one guess about how
    a leak would be spelled.
    """

    def __init__(self, drawn: dict[str, str], distort=None, *, pick: int = 0):
        self.drawn = drawn
        self.pick = pick
        self.payloads: list[dict] = []

    def __call__(self, host, payload, timeout):
        self.payloads.append(payload)
        prompt = payload["messages"][0]["content"]
        asked = [ref for ref in self.drawn if f"{ref}," in prompt or prompt.rstrip().endswith(ref)]
        answer: dict[str, str | None] = {}
        for ref in asked:
            candidates = tokens_near_reference(prompt, ref)
            answer[ref] = candidates[self.pick] if self.pick < len(candidates) else None
        return json.dumps(answer)


def _run_shape(
    tmp_path: Path,
    shape: str,
    *,
    markdown: str | None = None,
    distort=None,
    reader=None,
) -> dict:
    """Run one shape through the gate with a blind reader that reads the page."""
    builder, default_md, flags, _swap_refs = SHAPES[shape]
    pdf_path, drawn = builder(tmp_path)
    pipeline = UnifiedPipeline(_config())
    state = _make_state(pdf_path)
    ps = state.pages[1]
    bo = PageOutput(
        page_num=1,
        text=markdown if markdown is not None else default_md,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    blind = (reader or _BlindReaderOfThePage)(drawn, distort)
    with patch("socr.judge.table_rung_ollama._post_chat", blind):
        pipeline._run_table_judge_gate(state, 1, ps, bo, [_rejecting_rung(flags)])
    return {
        "drawn": drawn,
        "disposition": ps.table_ladder_disposition,
        "reasons": [
            e.data.get("reason") for e in state.events if e.kind == "table_ladder_accepted"
        ],
        "payloads": blind.payloads,
        "latched": bool(getattr(ps, "table_judge_retry_pending", False)),
    }


# ---------------------------------------------------------------------------
# Canaries: each fixture is the named shape
# ---------------------------------------------------------------------------


class TestTheFixturesAreTheNamedShapes:
    def test_the_spanning_header_is_one_merged_cell_over_two_columns(self, tmp_path: Path):
        pdf_path, drawn = spanning_header_pdf(tmp_path)
        page = fitz.open(pdf_path)[0]
        text = page.get_text()

        # ONE drawn token per panel, covering two columns each.
        assert text.count("Panel A") == 1 and text.count("Panel B") == 1
        assert drawn["H1C2"] == drawn["H1C3"] == "Panel A"
        assert drawn["H1C4"] == drawn["H1C5"] == "Panel B"
        assert "Notes: revised series." in text

        # Round 4, NEW 3: POSITIONAL, not a segment count. A builder that
        # dropped some other vertical edge -- the table's own right border,
        # say -- would satisfy a count comparison while drawing something that
        # is not a merged cell at all. The sub-header row below carries every
        # column boundary, so it defines the full set; the spanning row must be
        # missing exactly the two interior rules, the one inside Panel A and
        # the one inside Panel B.
        bands = _vertical_rules_by_band(page)
        assert len(bands) >= 2
        spanning_row, sub_header_row = bands[0], bands[1]
        assert len(sub_header_row) == 6, "the sub-header row has every boundary"
        interior_of_panel_a, interior_of_panel_b = sub_header_row[2], sub_header_row[4]
        assert spanning_row == [
            x for x in sub_header_row if x not in (interior_of_panel_a, interior_of_panel_b)
        ]

    def test_the_bad_crop_fixture_is_rotated_exactly_once(self, tmp_path: Path):
        pdf_path, drawn = rotated_crop_pdf(tmp_path)
        page = fitz.open(pdf_path)[0]
        assert page.rotation == 90
        # The drawn map is in the PRE-rotation frame, which is the frame the
        # extraction uses: the page is the described table, turned.
        assert drawn["H1C1"] == "Zone"
        assert drawn["R1C1"] == "East"
        assert drawn["R2C3"] == "42"

    def test_the_decoy_fixture_is_dense_near_identical_and_a_scan(self, tmp_path: Path):
        pdf_path, drawn = dense_decoy_pdf(tmp_path)
        assert len([ref for ref in drawn if ref.startswith("R")]) >= 12 * 5
        # Every row differs from its neighbour only in the leading digit.
        assert {tuple(c.split(".")[1] for c in row[1:]) for row in DENSE_ROWS} == {
            ("01", "02", "03", "04")
        }
        page = fitz.open(pdf_path)[0]
        assert page.get_text().strip() == "", "a scan has no text layer"
        assert page.get_images(), "a scan is pixels"


# ---------------------------------------------------------------------------
# The rejections really are false: the extraction IS the page
# ---------------------------------------------------------------------------


class TestTheRejectionsAreFalse:
    """The load-bearing claim, asserted against the builder's own record.

    A fixture only tests a FALSE reject if the emitted markdown is right. Round
    2 checked that each token appeared *somewhere* on the page, which two
    swapped values survive. This is an equality, cell by cell.
    """

    @pytest.mark.parametrize("shape", sorted(SHAPES))
    def test_the_extraction_equals_the_drawn_page_cell_for_cell(self, tmp_path: Path, shape: str):
        builder, markdown, _flags, _swap_refs = SHAPES[shape]
        _pdf, drawn = builder(tmp_path)
        resolved = resolve_cell_refs(markdown, sorted(drawn))
        assert resolved is not None, f"{shape}: every drawn cell must be addressable"
        assert {str(ref): token for ref, token in resolved.items()} == drawn

    @pytest.mark.parametrize("shape", sorted(SHAPES))
    def test_the_flagged_cells_are_exactly_what_the_page_shows(self, tmp_path: Path, shape: str):
        """And specifically the cells the readers flag, so each finding is
        wrong about a cell the page settles."""
        builder, markdown, flags, _swap_refs = SHAPES[shape]
        _pdf, drawn = builder(tmp_path)
        resolved = resolve_cell_refs(markdown, [where for _c, where in flags])
        assert resolved is not None
        for ref, token in resolved.items():
            assert token == drawn[str(ref)]


# ---------------------------------------------------------------------------
# The chain, on each real shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", sorted(SHAPES))
class TestTheChainClearsEveryFalseReject:
    def test_a_correct_page_reading_clears_the_false_reject(self, tmp_path: Path, shape: str):
        """The claim the file exists to establish: on every named shape, a
        rejection of a CORRECT table does not cost the table its bytes."""
        run = _run_shape(tmp_path, shape)
        branch = GEOMETRY_BRANCH[shape]

        assert run["disposition"] is None, "a false reject must not demote the page"
        assert run["latched"] is False
        if branch == "pass":
            assert run["reasons"] == ["verified_by_geometry"]
            assert run["payloads"] == [], "geometry clears for free; no call is made"
        else:
            assert run["reasons"] == ["verified_by_blind_cell_transcription"]
            assert run["payloads"], "an abstaining geometry must consult the blind reader"

    def test_a_two_cell_swap_in_the_extraction_is_not_cleared(self, tmp_path: Path, shape: str):
        """The mutation round 3 asked for, and the one the previous oracle
        survived. Two body values are exchanged, so the page-wide token
        multiset is unchanged and only a real cell-level comparison can
        notice. The blind reader still reads the PAGE, so it now disagrees --
        and whichever guard sees it first, the table must not be cleared."""
        _b, markdown, _f, (ref_a, ref_b) = SHAPES[shape]
        _pdf, drawn = SHAPES[shape][0](tmp_path / "oracle")
        corrupted = _swap(markdown, drawn, ref_a, ref_b)
        assert corrupted != markdown

        run = _run_shape(tmp_path, shape, markdown=corrupted)

        assert run["disposition"] is not None, "a wrong table must never be cleared"
        assert run["reasons"] == []

    def test_only_an_actual_blind_disagreement_withholds(self, tmp_path: Path, shape: str):
        """Changing ONLY what the blind reader reports is what moves the
        terminal -- and only on the shapes where it was consulted at all."""
        clear = _run_shape(tmp_path / "clear", shape)
        differ = _run_shape(tmp_path / "differ", shape, distort=lambda ref, token: "≠")

        if GEOMETRY_BRANCH[shape] == "pass":
            assert differ["disposition"] == clear["disposition"]
            assert differ["payloads"] == []
        else:
            assert differ["disposition"] == FailureMode.TABLE_WITHHELD
            assert clear["disposition"] is None

    def test_an_unreadable_answer_never_withholds_the_table(self, tmp_path: Path, shape: str):
        """Round 2, N2, tied to the fixtures. A blind reader that reports it
        could not read the cells has produced no evidence, so it may neither
        clear the table nor hide it."""
        run = _run_shape(tmp_path, shape, distort=lambda ref, token: None)
        assert run["disposition"] != FailureMode.TABLE_WITHHELD

    def test_the_crop_pixels_go_on_the_wire_and_no_extraction_token_does(
        self, tmp_path: Path, shape: str
    ):
        """Round 1 finding 4 and round 4 NEW 1, as one standing guard.

        A blind reader that never received the image can still emit a
        schema-valid guessable token and clear a table nobody looked at. So the
        request must carry the crop's BYTES, and it must not carry a single
        value from this table -- otherwise the agreement it reports is
        agreement with itself.

        The whole prompt AS SENT is checked. The previous version removed the
        shared coordinate fragment first, on the premise that fixed policy text
        corroborates nothing; round 4 showed that premise is false the moment
        that policy text contains a literal cell value tied to a coordinate,
        which it did.
        """
        run = _run_shape(tmp_path, shape)
        if GEOMETRY_BRANCH[shape] != "abstain":
            assert run["payloads"] == []
            return

        payload = run["payloads"][0]
        images = payload["messages"][0]["images"]
        assert len(images) == 1
        assert len(base64.b64decode(images[0])) > 1000

        prompt = payload["messages"][0]["content"]
        _b, markdown, flags, _s = SHAPES[shape]
        requested = [ref for _code, ref in flags]
        for ref in requested:
            assert ref in prompt, "the coordinates must be asked for"

        # Round 6: STRUCTURAL, on the prompt AS SENT. The policy half is the
        # same for every table and may name no cell at all; the request list is
        # generated and carries coordinates only. A binding of a coordinate to
        # a value, in ANY spelling, has to name a concrete coordinate -- so it
        # cannot exist without breaking the first half of that. Two rounds of
        # syntax-specific guards each closed one spelling and left the next
        # open; this closes the shape.
        from socr.judge.table_rung_ollama import REQUEST_LIST_HEADING, split_blind_cell_prompt

        policy, request_list = split_blind_cell_prompt(prompt)
        assert _REF_PATTERN.findall(policy) == [], "the policy half of the prompt names cells"
        assert set(_REF_PATTERN.findall(request_list)) == set(requested)
        remainder = _REF_PATTERN.sub("", request_list.replace(REQUEST_LIST_HEADING, ""))
        assert remainder.strip(" ,\n") == ""

        # Round 5, kept as belt and braces: no digit beyond the coordinates the
        # caller handed over, so no numeric value can hide anywhere.
        digits = sorted({c for c in _REF_PATTERN.sub("", prompt) if c.isdigit()})
        assert digits == [], f"the blind prompt carries non-coordinate digits: {digits}"

        # Non-numeric tokens still have to be checked by name. Whole-token
        # matching, so a two-letter heading like ``id`` is not reported as
        # leaked because the word "grid" contains it.
        for ref, token in run["drawn"].items():
            if not token:
                continue
            assert not re.search(rf"(?<!\w){re.escape(token)}(?!\w)", prompt), (
                f"the token at {ref} ({token!r}) is in the blind prompt"
            )
        for line in markdown.splitlines():
            assert line not in prompt
        # And nothing the READERS said either: their finding details are the
        # other channel through which an expectation could reach a blind eye.
        assert "flagged" not in prompt

    def test_a_prompt_copying_reader_cannot_clear_a_wrong_table(self, tmp_path: Path, shape: str):
        """Round 4, NEW 1 — the reviewer's reproducer, as a standing test.

        The extraction is corrupted, so the table under test is WRONG. The
        blind reader ignores the image entirely and answers by echoing whatever
        the prompt associates with the coordinate it was asked about -- exactly
        what a text-only model does with an answer key in front of it. When the
        shared fragment ended with ``R1C2 is 11`` and the extraction claimed
        ``R1C2 = 11``, that reader returned a schema-valid match and the gate
        published the wrong table as verified. There is nothing left to copy,
        so it can produce no reading, and the table is not cleared.
        """
        _b, markdown, _f, (ref_a, ref_b) = SHAPES[shape]
        _pdf, drawn = SHAPES[shape][0](tmp_path / "oracle")
        corrupted = _swap(markdown, drawn, ref_a, ref_b)

        run = _run_shape(tmp_path, shape, markdown=corrupted, reader=_PromptCopyingReader)

        assert run["disposition"] is not None, "a wrong table must never be cleared"
        assert run["reasons"] == []


class TestTheStandingPromptEchoReproducers:
    """The two reviewer reproducers, kept as tests rather than as history.

    Each is the same shape: a scanned table whose cell `R1C2` visibly reads
    ``99``, an extraction that wrongly claims the value a leaky prompt happened
    to state, a reader that rejects that cell, and a blind reader that ignores
    the image and copies whatever the prompt associates with `R1C2`.

    * round 4 leaked ``11`` through the shared fragment's prose worked example;
    * round 5 leaked ``1.24`` through the blind prompt's own JSON
      output-format example.

    Both published a wrong table as ``verified_by_blind_cell_transcription``.
    They are parametrised over the leaked value so a THIRD spelling of the same
    mistake fails here too, whatever syntax it arrives in.
    """

    DRAWN_VALUE = "99"

    def _scanned_fixture(self, tmp_path: Path, claimed: str):
        text_pdf, drawn = _grid_pdf(
            tmp_path / "echo_text" / "doc.pdf",
            [[("id", 1), ("qa", 1), ("qb", 1)]],
            [
                [("aa", 1), (self.DRAWN_VALUE, 1), ("77", 1)],
                [("bb", 1), ("55", 1), ("66", 1)],
            ],
        )
        assert drawn["R1C2"] == self.DRAWN_VALUE
        scan = _rasterise(text_pdf, tmp_path / "echo_scan" / "doc.pdf")
        markdown = (
            f"| id | qa | qb |\n| --- | --- | --- |\n| aa | {claimed} | 77 |\n| bb | 55 | 66 |\n"
        )
        return scan, drawn, markdown

    @pytest.mark.parametrize("claimed", ["11", "1.24", "N/A"])
    def test_no_token_the_prompt_offers_can_clear_the_wrong_value(
        self, tmp_path: Path, claimed: str
    ):
        """The sweep, round 6.

        Parametrising the leaked VALUE was not enough: the reader recognised
        only two spellings of a binding, so ``for R1C2 write N/A`` published a
        wrong table while every guard stayed green. The reader is proximity-
        based now, and this runs the gate once for EVERY token the prompt
        places next to the flagged coordinate. If any of them clears the wrong
        table, the prompt is an answer key -- whatever syntax put it there.
        """
        from socr.judge.table_rung_ollama import build_blind_cell_prompt

        candidates = tokens_near_reference(build_blind_cell_prompt(["R1C2"]), "R1C2")
        assert claimed not in candidates, "the prompt offers the wrong value directly"

        # Plus one past the end, which is the "nothing to copy" case.
        for pick in range(len(candidates) + 1):
            pdf_path, drawn, markdown = self._scanned_fixture(tmp_path / f"pick{pick}", claimed)
            run = self._gate(pdf_path, drawn, markdown, reader_pick=pick)
            assert run["disposition"] is not None, (
                f"copying {candidates[pick : pick + 1]} cleared a wrong table"
            )
            assert run["reasons"] == []

    def _gate(self, pdf_path, drawn, markdown, *, reader_pick=None):
        pipeline = UnifiedPipeline(_config())
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = PageOutput(
            page_num=1,
            text=markdown,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        reader = (
            _PromptCopyingReader(drawn, pick=reader_pick)
            if reader_pick is not None
            else _BlindReaderOfThePage(drawn)
        )
        rung = _rejecting_rung([(FindingCode.FABRICATED_VALUE, "R1C2")])
        with patch("socr.judge.table_rung_ollama._post_chat", reader):
            pipeline._run_table_judge_gate(state, 1, ps, bo, [rung])
        assert reader.payloads, "the blind reader must actually have been consulted"
        return {
            "disposition": ps.table_ladder_disposition,
            "reasons": [
                e.data.get("reason") for e in state.events if e.kind == "table_ladder_accepted"
            ],
            "prompt": reader.payloads[0]["messages"][0]["content"],
        }

    @pytest.mark.parametrize("claimed", ["11", "1.24", "N/A"])
    def test_the_wrong_value_is_never_in_the_prompt(self, tmp_path: Path, claimed: str):
        """The direct statement of the leak, per value, on the prompt as sent."""
        pdf_path, drawn, markdown = self._scanned_fixture(tmp_path, claimed)
        run = self._gate(pdf_path, drawn, markdown, reader_pick=0)
        assert claimed not in run["prompt"]

    @pytest.mark.parametrize("claimed", ["11", "1.24", "N/A"])
    def test_the_same_fixture_still_clears_when_the_page_agrees(self, tmp_path: Path, claimed: str):
        """The control that keeps the sweep honest: with an extraction that
        MATCHES the page, a reader that actually reads the page clears the
        false reject. The tests differ only in whether the extraction is right,
        so the failures above are about the wrong value and not about the
        fixture being unclearable."""
        pdf_path, drawn, _wrong = self._scanned_fixture(tmp_path, claimed)
        correct = (
            "| id | qa | qb |\n"
            "| --- | --- | --- |\n"
            f"| aa | {self.DRAWN_VALUE} | 77 |\n"
            "| bb | 55 | 66 |\n"
        )
        run = self._gate(pdf_path, drawn, correct)
        assert run["disposition"] is None
        assert run["reasons"] == ["verified_by_blind_cell_transcription"]
