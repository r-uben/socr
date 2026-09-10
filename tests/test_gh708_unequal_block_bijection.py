"""GH-708: a CI-hermetic pin for the #592 round-2 UNEQUAL-BLOCK search change.

#704 (merged ``8a63646``) retargeted ``_find_aligned_runs`` from walking
PyMuPDF text BLOCKS to walking baseline BANDS (see
``docs/log/2026-09-10_592-line-level-bijection.md``), because the D3 remeasure
found that a scanned Fed minutes page's text layer can fragment a label
column into several small blocks against a value column held in a single
block (1968-10-29, 1977-11-15, 1990-11-13) -- and the old block-granularity
walk's fail-streak limit gives up before it ever reaches the value block.

Every positive synthetic already in this suite (``_build_attendee_list_page``
and friends in ``tests/test_born_digital_aligned_runs.py``) is one textbox
per column: an EQUAL two-block shape the old walk handles fine. Reverting
``_find_aligned_runs`` to the old block walk would leave every one of those
tests, and the rest of CI, green. This file is the missing pin: a synthetic
fixture with the same unequal-block shape as the real D3 pages, checked
against both implementations.
"""

from __future__ import annotations

import statistics

import fitz

from socr.core import born_digital as bd

#: The attendee roster this fixture builds, mirroring the real Fed 1989 /
#: 1968 / 1977 / 1990 shape: a "Mr."/"Ms." label column and a name column.
_ROWS = [
    ("Mr.", "Greenspan, Chairman"),
    ("Mr.", "Corrigan, Vice Chairman"),
    ("Mr.", "Angell"),
    ("Mr.", "Black"),
    ("Ms.", "Seger"),
    ("Mr.", "Kelley"),
]

#: Insertion order for the label column, deliberately out of visual (y) order.
#: PyMuPDF's block segmenter groups consecutive TEXT-INSERTION calls that are
#: already close in y into one block (verified empirically -- six sequential
#: same-y-gap ``insert_text`` calls land in ONE block); inserting them out of
#: order breaks that grouping and yields one block PER LABEL LINE instead,
#: the same fragmentation shape a scanned/reconstructed text layer produces.
#: The value column is written with a single ``insert_textbox`` call, which
#: stays one block. This gives 6 label blocks against 1 value block -- more
#: unequal than the real pages (which split into a handful of blocks each),
#: and enough consecutive block-growth failures (5) to exceed
#: ``_ALIGNED_RUN_FAIL_STREAK_LIMIT`` (4) under the old block walk.
_LABEL_INSERTION_ORDER = (0, 3, 1, 4, 2, 5)


def _build_unequal_block_attendee_page() -> fitz.Page:
    """A label column fragmented into 6 PyMuPDF blocks vs. a 1-block value column."""
    doc = fitz.open()
    page = doc.new_page()
    y0 = 72
    page.insert_text(
        (72, y0),
        "Minutes of the Federal Open Market Committee meeting held on",
        fontsize=10,
        fontname="helv",
    )
    page.insert_text((72, y0 + 14), "November 14, 1989", fontsize=10, fontname="helv")
    page.insert_text((72, y0 + 34), "PRESENT:", fontsize=10, fontname="helv")

    left_x = 90
    label_w = fitz.get_text_length("Mr.", fontname="helv", fontsize=10)
    space_w = fitz.get_text_length(" ", fontname="helv", fontsize=10)
    gap = 1.2 * space_w  # comparable to one word space -- the defect's shape
    right_x = left_x + label_w + gap
    row_start_y = y0 + 50
    row_h = 14
    row_ys = [row_start_y + i * row_h for i in range(len(_ROWS))]

    for i in _LABEL_INSERTION_ORDER:
        page.insert_text((left_x, row_ys[i]), _ROWS[i][0], fontsize=10, fontname="helv")

    right_rect = fitz.Rect(right_x, row_start_y - 8, 560, row_start_y + 140)
    page.insert_textbox(
        right_rect,
        "\n".join(name for _, name in _ROWS),
        fontsize=10,
        fontname="helv",
    )
    return page


def test_fixture_block_structure_is_genuinely_unequal():
    """The fixture must actually fragment into unequal blocks, not merge back.

    Guards the fixture itself: if a future PyMuPDF version's block segmenter
    changes and starts merging the out-of-order label insertions back into
    fewer blocks, this must fail loudly rather than let the rest of the file
    silently stop exercising the unequal-block shape.
    """
    page = _build_unequal_block_attendee_page()
    page_dict = page.get_text("dict")
    blocks = [b for b in page_dict.get("blocks", []) if b.get("type", 0) == 0]

    def block_text(b) -> str:
        return "".join(
            span.get("text", "") for line in b.get("lines", []) for span in line.get("spans", [])
        ).strip()

    label_blocks = [b for b in blocks if block_text(b) in ("Mr.", "Ms.")]
    value_blocks = [b for b in blocks if "Greenspan" in block_text(b)]

    assert len(label_blocks) == 6, (
        f"expected 6 separate one-line label blocks, got {len(label_blocks)}: "
        f"{[block_text(b) for b in blocks]}"
    )
    assert len(value_blocks) == 1, (
        f"expected the value column as a single block, got {len(value_blocks)}: "
        f"{[block_text(b) for b in blocks]}"
    )
    (value_block,) = value_blocks
    value_lines = [
        "".join(span.get("text", "") for span in line.get("spans", []))
        for line in value_block.get("lines", [])
    ]
    assert value_lines == [name for _, name in _ROWS], (
        f"value column must carry all 6 names in one block, got {value_lines}"
    )
    # The label blocks collectively carry every honorific in the roster, one
    # per block -- not merged, not dropped.
    assert sorted(block_text(b) for b in label_blocks) == sorted(label for label, _ in _ROWS)


def test_current_extractor_pairs_every_label_with_its_name():
    """(1) The current ``extract_structured`` pairs this unequal-block roster."""
    page = _build_unequal_block_attendee_page()
    out = bd.BornDigitalDetector().extract_structured(page)

    for label, name in _ROWS:
        assert f"{label} {name}" in out, f"expected {label!r} paired with {name!r} in:\n{out}"

    bare_honorifics = sum(1 for line in out.splitlines() if line.strip() in ("Mr.", "Ms."))
    assert bare_honorifics == 0, (
        f"expected zero bare honorific lines after merge, got {bare_honorifics}: {out!r}"
    )


# --- Reconstruction of the pre-#704 (commit 4ce7ecc) block-granularity search ---
#
# ``_find_aligned_runs`` / ``_try_aligned_run`` / ``_cluster_two_bands`` below
# are copied, structurally unchanged, from
# ``git show 4ce7ecc:src/socr/core/born_digital.py`` -- the C1 implementation
# immediately before #704's round 2 (``ba9e10d``) replaced block-granularity
# growth with baseline-band growth. Reproduced inline rather than loaded via
# ``git show <sha>`` at test time: a git-history lookup ties this pin's
# hermeticity to checkout depth (this repo's CI checkout is not guaranteed to
# reach a commit this far back in history), where a frozen copy of dead code
# has none of that risk. The guard constants below
# (``ALIGNED_RUN_GAP_MAX_WORD_SPACES``, ``LABEL_COLUMN_WIDTH_SHARE``,
# ``RIGHT_BLOCK_FILL_TOLERANCE_WORD_WIDTHS``, ``MEASURE_FILL_SHARE_MAX``,
# ``_ALIGNED_RUN_FAIL_STREAK_LIMIT``) are unchanged across every #592 round
# per the decision log, so they are read live off ``bd`` rather than
# reproduced a second time; only ``_ALIGNED_RUN_MIN_ROWS`` differed (2 at
# 4ce7ecc, 3 from round 2 on) and is hardcoded here as the historical value.

#: `_ALIGNED_RUN_MIN_ROWS` at commit 4ce7ecc, before round 2 raised it to 3.
_OLD_ALIGNED_RUN_MIN_ROWS = 2

#: `_COLUMN_SEED_TOLERANCE` at commit 4ce7ecc (unchanged constant, but the
#: helper that reads it, `_cluster_two_bands`, was deleted in round 4 when
#: `_split_two_columns` replaced it -- reproduced here alongside its helper).
_OLD_COLUMN_SEED_TOLERANCE = 3.0


def _old_cluster_two_bands(values: list[float]) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    ordered = sorted(values)
    groups: list[list[float]] = [[ordered[0]]]
    for v in ordered[1:]:
        if v - groups[-1][-1] <= _OLD_COLUMN_SEED_TOLERANCE:
            groups[-1].append(v)
        else:
            groups.append([v])
    if len(groups) < 2:
        return None
    groups.sort(key=len, reverse=True)
    seed_a = sum(groups[0]) / len(groups[0])
    seed_b = sum(groups[1]) / len(groups[1])
    if seed_a == seed_b:
        return None
    return (seed_a, seed_b) if seed_a < seed_b else (seed_b, seed_a)


def _old_try_aligned_run(
    items: list[dict],
    word_space_width: float,
    gap_max_word_spaces: float,
    word_width: float,
) -> list[str] | None:
    if len(items) < _OLD_ALIGNED_RUN_MIN_ROWS * 2:
        return None

    seeds = _old_cluster_two_bands([it["x0"] for it in items])
    if seeds is None:
        return None
    seed_left, seed_right = seeds

    left = [it for it in items if abs(it["x0"] - seed_left) <= abs(it["x0"] - seed_right)]
    right = [it for it in items if abs(it["x0"] - seed_left) > abs(it["x0"] - seed_right)]
    if len(left) < _OLD_ALIGNED_RUN_MIN_ROWS or len(left) != len(right):
        return None

    left_width = statistics.median([it["x1"] - it["x0"] for it in left])
    right_width = statistics.median([it["x1"] - it["x0"] for it in right])
    if right_width <= 0 or left_width > bd.LABEL_COLUMN_WIDTH_SHARE * right_width:
        return None

    if word_width > 0:
        right_widths = [it["x1"] - it["x0"] for it in right]
        max_right_width = max(right_widths)
        tol = bd.RIGHT_BLOCK_FILL_TOLERANCE_WORD_WIDTHS * word_width
        fill_share = sum(1 for w in right_widths if (max_right_width - w) <= tol) / len(
            right_widths
        )
        if fill_share > bd.MEASURE_FILL_SHARE_MAX:
            return None

    left.sort(key=lambda it: it["y0"])
    right.sort(key=lambda it: it["y0"])

    merged: list[str] = []
    for l, r in zip(left, right):
        if l["y1"] <= r["y0"] or r["y1"] <= l["y0"]:
            return None
        gap = r["x0"] - l["x1"]
        if gap <= 0 or gap > gap_max_word_spaces * word_space_width:
            return None
        merged.append(f"{l['text'].rstrip()} {r['text'].rstrip()}".rstrip())

    return merged


def _old_find_aligned_runs(
    block_lines: list[list[dict]],
    word_space_width: float,
    word_width: float,
) -> list[tuple[int, int, list[str]]]:
    n = len(block_lines)
    runs: list[tuple[int, int, list[str]]] = []
    pos = 0
    while pos < n:
        items = list(block_lines[pos])
        best: tuple[int, list[str]] | None = None
        fail_streak = 0
        end = pos
        while end + 1 < n and fail_streak < bd._ALIGNED_RUN_FAIL_STREAK_LIMIT:
            end += 1
            items = items + block_lines[end]
            candidate = _old_try_aligned_run(
                items, word_space_width, bd.ALIGNED_RUN_GAP_MAX_WORD_SPACES, word_width
            )
            if candidate is not None:
                best = (end, candidate)
                fail_streak = 0
            else:
                fail_streak += 1
        if best is not None:
            end_pos, merged_lines = best
            runs.append((pos, end_pos, merged_lines))
            pos = end_pos + 1
        else:
            pos += 1
    return runs


def _old_assemble_prose_with_aligned_runs(page: fitz.Page) -> str | None:
    """The pre-#704 (commit 4ce7ecc) block-granularity assembler."""
    try:
        words = page.get_text("words") or []
    except Exception:
        return None
    word_space_width = bd._median_word_space_width(words)
    if not word_space_width:
        return None
    word_width = bd._median_word_width(words) or 0.0

    try:
        page_dict = page.get_text("dict")
    except Exception:
        return None

    extents = bd._line_word_extents(words)
    blocks = [b for b in page_dict.get("blocks", []) if b.get("type", 0) == 0]

    block_lines: list[list[dict]] = []
    block_line_texts: list[list[str]] = []
    for bi, block in enumerate(blocks):
        lines = block.get("lines", []) or []
        items = []
        texts = []
        for li, line in enumerate(lines):
            text = "".join(s.get("text", "") for s in line.get("spans", []) or [])
            texts.append(text)
            ext = extents.get((bi, li))
            if ext is None:
                continue
            bbox = line.get("bbox")
            if not bbox:
                continue
            items.append({"y0": bbox[1], "y1": bbox[3], "x0": ext[0], "x1": ext[1], "text": text})
        block_lines.append(items)
        block_line_texts.append(texts)

    runs = _old_find_aligned_runs(block_lines, word_space_width, word_width)
    if not runs:
        return None

    run_by_start = {start: (end, merged) for start, end, merged in runs}
    consumed_until = -1
    out_lines: list[str] = []
    for bi in range(len(blocks)):
        if bi <= consumed_until:
            continue
        if bi in run_by_start:
            end, merged = run_by_start[bi]
            out_lines.extend(merged)
            consumed_until = end
        else:
            out_lines.extend(block_line_texts[bi])
    return "\n".join(out_lines).strip()


def test_old_block_walk_still_pairs_the_equal_block_fixture():
    """Sanity check on the reconstruction: it must still solve the EQUAL-block
    case (the existing positive synthetic), so a failure below is attributable
    to unequal blocks specifically, not to a broken reconstruction."""
    from test_born_digital_aligned_runs import _build_attendee_list_page

    page = _build_attendee_list_page()
    out = _old_assemble_prose_with_aligned_runs(page)
    assert out is not None
    assert "Mr. Greenspan, Chairman" in out


def test_old_block_walk_fails_the_unequal_block_fixture():
    """(2) The old C1 block walk goes red on this fixture.

    Confirms the fixture is a real differentiator: the fail-streak limit
    (``_ALIGNED_RUN_FAIL_STREAK_LIMIT`` = 4) is exhausted growing through the
    6 fragmented one-line label blocks before the search ever reaches the
    single value block, so no run is found and every "Mr."/"Ms." line stays
    bare. This is the exact defect shape #704 fixed by switching to
    baseline-band granularity -- a revert of ``_find_aligned_runs`` back to
    this block walk reproduces it.
    """
    page = _build_unequal_block_attendee_page()
    out = _old_assemble_prose_with_aligned_runs(page)
    assert out is None, (
        "the old block-granularity walk was expected to decline this "
        f"unequal-block fixture (no run found), but it produced: {out!r}"
    )


def test_current_search_recovers_what_the_old_block_walk_missed():
    """The difference, stated as one assertion: same fixture, two outcomes.

    Pinning a DIFFERENCE rather than an absolute value, per this repo's
    CLAUDE.md -- both implementations run in the same process against the
    identical fitz page.
    """
    page = _build_unequal_block_attendee_page()

    old_out = _old_assemble_prose_with_aligned_runs(page)
    new_out = bd._assemble_prose_with_aligned_runs(page)

    assert old_out is None, f"old block walk unexpectedly found a run: {old_out!r}"
    assert new_out is not None, "current band walk unexpectedly found no run"
    for label, name in _ROWS:
        assert f"{label} {name}" in new_out
