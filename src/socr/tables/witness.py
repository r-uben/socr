"""Table witness preparation — map emitted markdown tables to page crops.

The judge ladder (A2/A3) needs, per emitted table, a crop image of the page
region that table came from. Nothing today owns that mapping:

- ``reconcile.find_table_blocks`` enumerates the markdown table blocks the
  model emitted, in reading order.
- ``locate.locate_tables`` enumerates geometric boxes on the page, in reading
  order — but it over-merges vertically stacked tables into one band
  (``locate.py`` ``bands_from_rules``, documented limitation) and finds
  nothing at all for a borderless (whitespace-only) table.

Pairing them 1:1 by count is only sound when the counts agree. When they
don't, the honest answer is "I don't know which box is which table" (or "no
box exists"), not a guess — a wrong crop would judge the wrong table. This
module makes that uncertainty a first-class, representable state
(``WitnessStatus.AMBIGUOUS`` / ``MISSING``) instead of raising or silently
picking one.

**Residual assumption when counts match (LOCATED):** pairing is by index —
block ``i`` gets box ``i`` — after both lists are already in each source's own
"reading order" (``find_table_blocks`` walks the markdown top to bottom;
``locate_tables`` sorts by ``(y0, x0)``). This assumes the model emits tables
in the same top-to-bottom order the page geometry does, which holds for the
single-column layouts this has been exercised against. It is UNVERIFIED for
side-by-side (multi-column / multi-panel) layouts, where two tables can share
a ``y0`` band and ``(y0, x0)`` order need not match emission order.

## Pairing corroboration (consilium decision, 2026-08-30, 3 rounds)

The panel considered — and rejected — strict per-page verification of the
index pairing: E1's mechanical binding check (``tables/binding.py bind()``)
is deliberately CONTRADICTION-ONLY, NEUTRAL on missing coverage (binding
coverage was measured incomplete on 12/13 corpus table pages), so witness
pairing cannot lean on it as a general-purpose safety net; and a strict
"prove the assignment" gate would mass-demote count-matched pages under
ordinary matching noise (sparse native text, tables with few numeric values)
— the same failure mode E1's own NEUTRAL-on-no-coverage rule exists to avoid.

Instead, count-matched pairs get lightweight CORROBORATION: for every pair of
indices ``(i, j)`` (``n >= 2``), compare each box's native numeric-token
evidence against BOTH candidate blocks' emitted numeric tokens
(``_native_numeric_multisets`` / ``_numeric_multiset_from_tokens``, the same
primitives ``native_verifier.py`` and ``source_evidence.py`` already use).
Three outcomes, in order of how much evidence is required:

- **Neutral (default, ships)** — no matched-token evidence favors a swap
  (identical evidence, a tie, or no native words in one/both boxes at all).
  Neutral is NOT contradiction; the identity pairing ships ``LOCATED``. This
  is what prevents the mass-demotion failure mode above — most real pages
  have thin or absent native-numeric coverage in at least one table and must
  not be punished for it.
- **Corroborated** — box ``i``'s matched evidence favors block ``i`` (its own
  paired block) over block ``j``, or box ``i`` has no matched evidence at
  all. Ships ``LOCATED``.
- **Positively contradicted** — a *structural majority test*, not a
  score-difference threshold: swap ``(i, j)`` is a contradiction ONLY when
  BOTH box ``i``'s matched evidence strictly favors block ``j`` over its own
  block ``i``, AND box ``j``'s matched evidence strictly favors block ``i``
  over its own block ``j``. Two-sided, strict-inequality agreement, no float
  cutoff. Both ``i`` and ``j`` demote to ``AMBIGUOUS`` (no crop, no auto-swap
  — a demonstrated wrong pairing must never silently self-correct into
  another guess).

## Degraded scope (GH-373)

Count-mismatch ``AMBIGUOUS`` (``_classify`` only) is a mapping failure, not
an absence of pixels. Those witnesses get a full-page crop
(``WitnessScope.PAGE``) so the judge can look, with a policy-file scope
note telling it to match the emitted markdown. The union of located boxes
is NOT used: a spanning header can sit outside every located box, which is
the HEADER_MANGLED catch the first live run hid by abstaining.

Corroboration-contradicted ``AMBIGUOUS`` does **not** get a page crop.
Counts matched and the index pairing is known-wrong; showing any crop
(swapped, merged, or the page) would be another guess. ``MISSING`` stays
¬S1: no geometric evidence a table region exists.

Crops are rendered into temp files with caller-owned lifetime scoped to a
context manager (mirrors ``TableCropExtractor._render_crop``'s caller-owned
cleanup, but guarantees the unlink here so a gate that raises mid-ladder never
leaks table-crop PNGs).
"""

from __future__ import annotations

import logging
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

from socr.core.pdf import open_pdf
from socr.tables.extract import CROP_PADDING_PT, DEFAULT_CROP_DPI
from socr.tables.locate import TableBox, locate_tables
from socr.tables.native_verifier import _numeric_multiset_from_tokens, _numeric_tokens_from_text
from socr.tables.reconcile import find_table_blocks

logger = logging.getLogger(__name__)


class WitnessStatus(str, Enum):
    """How confidently an emitted table block maps to a page region."""

    #: Exactly one box paired with this block — the crop is trustworthy.
    LOCATED = "located"
    #: The page located zero boxes for its blocks — no geometric anchor at all
    #: (e.g. a borderless table). No crop is rendered.
    MISSING = "missing"
    #: Block count and box count on the page disagree (locator over-merge,
    #: an extra spurious box, ...) — no box can be confidently assigned to
    #: any one block. Count-mismatch AMBIGUOUS still renders a full-page
    #: crop (GH-373); corroboration-contradicted AMBIGUOUS does not.
    AMBIGUOUS = "ambiguous"


class WitnessScope(str, Enum):
    """What image, if any, the judge is shown for this witness.

    Distinct from ``WitnessStatus``: both AMBIGUOUS causes share a status,
    but only count mismatch ships a page image.
    """

    #: 1:1 box crop. Status is LOCATED.
    LOCATED = "located"
    #: Full-page image. Status is count-mismatch AMBIGUOUS.
    PAGE = "page"
    #: No image. MISSING, corroboration-contradicted AMBIGUOUS, or render failure.
    NONE = "none"


@dataclass(frozen=True)
class TableWitness:
    """One emitted table block, and (if confidently known) its page region."""

    table_id: str  # stable per-page id, e.g. "p3-t0"
    page_num: int  # 1-indexed
    block_index: int  # reading-order index among this page's emitted blocks
    markdown: str  # the emitted block's own markdown (header + separator + rows)
    status: WitnessStatus
    box: TableBox | None = None  # set only when status == LOCATED
    crop_path: Path | None = None  # set when a crop was rendered (LOCATED or
    # count-mismatch AMBIGUOUS); valid only for the lifetime of the
    # ``prepare_table_witnesses`` context.
    boxes_found_on_page: int = 0  # diagnostic: len(locate_tables(page))
    note: str = ""
    scope: WitnessScope = WitnessScope.NONE


def _block_markdown(markdown: str, start: int, end: int) -> str:
    lines = markdown.splitlines()
    return "\n".join(lines[start : end + 1])


def _classify(n_blocks: int, n_boxes: int) -> WitnessStatus:
    """Count-only classification. See module docstring for the residual
    index-pairing assumption this implies for the ``LOCATED`` case."""
    if n_boxes == 0:
        return WitnessStatus.MISSING
    if n_boxes != n_blocks:
        return WitnessStatus.AMBIGUOUS
    return WitnessStatus.LOCATED


@contextmanager
def prepare_table_witnesses(
    pdf_path: Path,
    page_num: int,
    markdown: str,
    *,
    crop_dpi: int = DEFAULT_CROP_DPI,
) -> Iterator[list[TableWitness]]:
    """Yield one ``TableWitness`` per emitted table block on ``page_num``.

    Never raises: any render/locate failure degrades a witness to ``MISSING``
    rather than aborting. Rendered crop files are guaranteed to be removed
    when the context exits, whether it exits normally or via exception —
    callers must not keep ``crop_path`` beyond this block.

    On a block/box count match, pairing is index-order, checked pairwise
    against native numeric-token evidence ("pairing corroboration" — see
    module docstring): a demonstrated swap demotes the pair to ``AMBIGUOUS``;
    absent or tied evidence is NEUTRAL and ships the index pairing.
    """
    blocks = find_table_blocks(markdown)
    witnesses: list[TableWitness] = []
    crop_paths: list[Path] = []
    try:
        if blocks:
            boxes, page_open_error = _locate_boxes(pdf_path, page_num)
            status = _classify(len(blocks), len(boxes))
            block_mds = [_block_markdown(markdown, b.start, b.end) for b in blocks]
            contradicted: set[int] = set()
            if status is WitnessStatus.LOCATED and len(blocks) >= 2:
                contradicted = _corroboration_contradicted_indices(
                    pdf_path, page_num, boxes, block_mds
                )
            # Count-mismatch AMBIGUOUS (from _classify, not corroboration)
            # gets one full-page crop shared by every block on the page.
            page_crop: Path | None = None
            if status is WitnessStatus.AMBIGUOUS:
                page_crop = _render_page_safe(pdf_path, page_num, crop_dpi)
                if page_crop is not None:
                    crop_paths.append(page_crop)

            for idx, block_md in enumerate(block_mds):
                table_id = f"p{page_num}-t{idx}"
                if status is WitnessStatus.LOCATED and idx in contradicted:
                    witnesses.append(
                        TableWitness(
                            table_id=table_id,
                            page_num=page_num,
                            block_index=idx,
                            markdown=block_md,
                            status=WitnessStatus.AMBIGUOUS,
                            boxes_found_on_page=len(boxes),
                            note="pairing corroboration contradicted the index pairing (swap evidence)",
                            scope=WitnessScope.NONE,
                        )
                    )
                elif status is WitnessStatus.LOCATED:
                    box = boxes[idx]
                    crop_path = _render_crop_safe(pdf_path, page_num, box, crop_dpi)
                    if crop_path is None:
                        witnesses.append(
                            TableWitness(
                                table_id=table_id,
                                page_num=page_num,
                                block_index=idx,
                                markdown=block_md,
                                status=WitnessStatus.MISSING,
                                boxes_found_on_page=len(boxes),
                                note="crop render failed",
                                scope=WitnessScope.NONE,
                            )
                        )
                        continue
                    crop_paths.append(crop_path)
                    witnesses.append(
                        TableWitness(
                            table_id=table_id,
                            page_num=page_num,
                            block_index=idx,
                            markdown=block_md,
                            status=WitnessStatus.LOCATED,
                            box=box,
                            crop_path=crop_path,
                            boxes_found_on_page=len(boxes),
                            scope=WitnessScope.LOCATED,
                        )
                    )
                else:
                    note = page_open_error or (
                        "no located box on page"
                        if status is WitnessStatus.MISSING
                        else f"{len(blocks)} table block(s) vs {len(boxes)} located box(es)"
                    )
                    crop_path = None
                    scope = WitnessScope.NONE
                    if status is WitnessStatus.AMBIGUOUS and page_crop is not None:
                        crop_path = page_crop
                        scope = WitnessScope.PAGE
                    witnesses.append(
                        TableWitness(
                            table_id=table_id,
                            page_num=page_num,
                            block_index=idx,
                            markdown=block_md,
                            status=status,
                            crop_path=crop_path,
                            boxes_found_on_page=len(boxes),
                            note=note,
                            scope=scope,
                        )
                    )
        yield witnesses
    finally:
        for path in crop_paths:
            path.unlink(missing_ok=True)


def _locate_boxes(pdf_path: Path, page_num: int) -> tuple[list[TableBox], str]:
    """Best-effort ``locate_tables`` on ``page_num``. Never raises.

    Returns ``([], "reason")`` on any failure to open the document or page so
    callers can fold the reason into the witness note rather than an exception
    propagating out of witness preparation.
    """
    try:
        doc = open_pdf(pdf_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("table witness: cannot open %s (%s)", pdf_path, exc)
        return [], f"cannot open pdf: {exc}"
    try:
        page = doc[page_num - 1]
        return locate_tables(page), ""
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("table witness: locate_tables failed p%d (%s)", page_num, exc)
        return [], f"locate failed: {exc}"
    finally:
        doc.close()


def _render_crop_safe(pdf_path: Path, page_num: int, box: TableBox, crop_dpi: int) -> Path | None:
    """Render ``box`` on ``page_num`` to a temp PNG. Never raises; ``None`` on failure."""
    import fitz
    from PIL import Image

    from socr.core.born_digital import upright_rotation_for

    try:
        doc = open_pdf(pdf_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("table witness: cannot open %s for crop (%s)", pdf_path, exc)
        return None
    try:
        page = doc[page_num - 1]
        page_rect = page.rect
        x0, y0, x1, y1 = box.bbox
        clip = fitz.Rect(
            max(page_rect.x0, x0 - CROP_PADDING_PT),
            max(page_rect.y0, y0 - CROP_PADDING_PT),
            min(page_rect.x1, x1 + CROP_PADDING_PT),
            min(page_rect.y1, y1 + CROP_PADDING_PT),
        )
        rotation = upright_rotation_for(page, clip=clip)
        mat = fitz.Matrix(crop_dpi / 72, crop_dpi / 72)
        if rotation != 0:
            mat.prerotate(rotation)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("table witness: crop render failed p%d (%s)", page_num, exc)
        return None
    finally:
        doc.close()

    fd, name = tempfile.mkstemp(prefix="socr_tablewitness_", suffix=".png")
    path = Path(name)
    try:
        import os

        os.close(fd)
        img.save(path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("table witness: crop save failed p%d (%s)", page_num, exc)
        path.unlink(missing_ok=True)
        return None
    return path


def _render_page_safe(pdf_path: Path, page_num: int, crop_dpi: int) -> Path | None:
    """Render the full page to a temp PNG. Never raises; ``None`` on failure.

    Reuses ``_render_crop_safe`` with a box equal to the page rect so padding
    clips to the page and rotation/DPI stay identical to a located crop.
    """
    try:
        doc = open_pdf(pdf_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("table witness: cannot open %s for page crop (%s)", pdf_path, exc)
        return None
    try:
        page = doc[page_num - 1]
        rect = page.rect
        box = TableBox(
            bbox=(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)),
            source="page",
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("table witness: page rect failed p%d (%s)", page_num, exc)
        return None
    finally:
        doc.close()
    return _render_crop_safe(pdf_path, page_num, box, crop_dpi)


def _corroboration_contradicted_indices(
    pdf_path: Path, page_num: int, boxes: list[TableBox], block_mds: list[str]
) -> set[int]:
    """Pairwise swap-contradiction check for a count-matched page (n >= 2).

    See "Pairing corroboration" in the module docstring for the panel
    rationale. Never raises: any failure to read native words yields empty
    evidence for every box, which is NEUTRAL by construction (no matched
    evidence can ever exceed 0), so the index pairing ships unchanged.
    """
    native = _native_numeric_multisets(pdf_path, page_num, boxes)
    output = [_numeric_multiset_from_tokens(_numeric_tokens_from_text(md)) for md in block_mds]
    n = len(boxes)
    overlap = [[_multiset_overlap(native[i], output[j]) for j in range(n)] for i in range(n)]
    contradicted: set[int] = set()
    for i in range(n):
        for j in range(i + 1, n):
            # Structural majority test: BOTH members' own matched evidence
            # must strictly prefer the OTHER block over their own paired
            # block. A tie, or evidence favoring only one side, is neutral.
            if overlap[i][j] > overlap[i][i] and overlap[j][i] > overlap[j][j]:
                contradicted.add(i)
                contradicted.add(j)
    return contradicted


def _multiset_overlap(a: Counter, b: Counter) -> int:
    return sum((a & b).values())


def _native_numeric_multisets(
    pdf_path: Path, page_num: int, boxes: list[TableBox]
) -> list[Counter]:
    """Best-effort per-box native numeric-token multiset. Never raises.

    Mirrors ``native_verifier.verify_native_table_region``'s word-clipping
    (top-left corner of each word inside the box), reading the page's native
    text layer once for all boxes rather than once per box.
    """
    empty = [Counter() for _ in boxes]
    try:
        doc = open_pdf(pdf_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("table witness: cannot open %s for corroboration (%s)", pdf_path, exc)
        return empty
    try:
        page = doc[page_num - 1]
        words = page.get_text("words")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("table witness: get_text failed p%d (%s)", page_num, exc)
        return empty
    finally:
        doc.close()

    out: list[Counter] = []
    for box in boxes:
        x0, y0, x1, y1 = box.bbox
        region_words = [w for w in words if x0 <= w[0] <= x1 and y0 <= w[1] <= y1]
        out.append(_numeric_multiset_from_tokens([w[4] for w in region_words]))
    return out
