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
a ``y0`` band and ``(y0, x0)`` order need not match emission order — that
misalignment would currently ship as a false ``LOCATED`` pairing rather than
degrading to ``AMBIGUOUS``. Tracked for the ladder design panel; not resolved
by this module today.

Crops are rendered into temp files with caller-owned lifetime scoped to a
context manager (mirrors ``TableCropExtractor._render_crop``'s caller-owned
cleanup, but guarantees the unlink here so a gate that raises mid-ladder never
leaks table-crop PNGs).
"""

from __future__ import annotations

import logging
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

from socr.core.pdf import open_pdf
from socr.tables.extract import _CROP_PADDING_PT, DEFAULT_CROP_DPI
from socr.tables.locate import TableBox, locate_tables
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
    #: any one block. No crop is rendered.
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class TableWitness:
    """One emitted table block, and (if confidently known) its page region."""

    table_id: str  # stable per-page id, e.g. "p3-t0"
    page_num: int  # 1-indexed
    block_index: int  # reading-order index among this page's emitted blocks
    markdown: str  # the emitted block's own markdown (header + separator + rows)
    status: WitnessStatus
    box: TableBox | None = None  # set only when status == LOCATED
    crop_path: Path | None = None  # set only when status == LOCATED; valid
    # only for the lifetime of the ``prepare_table_witnesses`` context.
    boxes_found_on_page: int = 0  # diagnostic: len(locate_tables(page))
    note: str = ""


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

    On a block/box count match, pairing is index-order (see module docstring
    for the residual "same reading order" assumption — unverified for
    side-by-side/multi-column table layouts).
    """
    blocks = find_table_blocks(markdown)
    witnesses: list[TableWitness] = []
    crop_paths: list[Path] = []
    try:
        if blocks:
            boxes, page_open_error = _locate_boxes(pdf_path, page_num)
            status = _classify(len(blocks), len(boxes))
            for idx, block in enumerate(blocks):
                table_id = f"p{page_num}-t{idx}"
                block_md = _block_markdown(markdown, block.start, block.end)
                if status is WitnessStatus.LOCATED:
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
                        )
                    )
                else:
                    note = page_open_error or (
                        "no located box on page"
                        if status is WitnessStatus.MISSING
                        else f"{len(blocks)} table block(s) vs {len(boxes)} located box(es)"
                    )
                    witnesses.append(
                        TableWitness(
                            table_id=table_id,
                            page_num=page_num,
                            block_index=idx,
                            markdown=block_md,
                            status=status,
                            boxes_found_on_page=len(boxes),
                            note=note,
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
            max(page_rect.x0, x0 - _CROP_PADDING_PT),
            max(page_rect.y0, y0 - _CROP_PADDING_PT),
            min(page_rect.x1, x1 + _CROP_PADDING_PT),
            min(page_rect.y1, y1 + _CROP_PADDING_PT),
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
