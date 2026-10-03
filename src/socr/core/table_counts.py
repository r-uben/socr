"""GH-993: how many tables can a reader actually read, per document.

Every metric before this counted defects. This one counts usefulness, derived only
from what socr already records: the shipped text of each page, its status and failure
mode, and the table-distrust index (``tables_trust``). No new detection.

Three counts, plus the verified subset of the first:

``shipped_text``
    Tables emitted as markdown text (``find_table_blocks`` over the page text that
    shipped), whether the page is verified or WARNING with the text kept.
``verified_text``
    The part of ``shipped_text`` on a page that is ``success`` and carries no table
    distrust event. A SUCCESS page flagged in ``tables_trust`` is not verified.
``unverified_text``
    The part of ``shipped_text`` on a page the ladder left UNVERIFIED
    (``failure_mode == table_unverified`` or a live ``table_ladder_unverified`` flag).
``withheld``
    Table regions shipped only as a marker (plus, usually, a page image): one per
    ``[page N failed: unverifiable table ...]`` or ``[page N failed: invalid table
    emission ...]`` marker in the shipped text. A regional splice stamps one marker per
    covered region; a whole-page floor carries one.

``flattened_to_prose`` is deliberately absent: socr does not record it yet (#994).

Counts are over BLOCKS, so a table the producer fragmented into several pipe blocks
counts several times. That is the unit ``find_table_blocks`` defines, and it is the
same unit the withheld-table analysis used.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from socr.core.manifest import SCANNED_PROSE_RECOVERED_FLAG
from socr.core.result import FailureMode, PageStatus
from socr.tables.reconcile import find_table_blocks

#: Markers socr authors where a table's bytes were withheld. Kept as one regex so a
#: new withholding marker is added in one place.
_WITHHELD_MARKER_RE = re.compile(
    r"\[page \d+ failed: (?:unverifiable table|invalid table emission)"
)

#: The banner of a prose-recovery page. Its markers are one per contiguous withheld RUN,
#: which the producer itself says is not a table count, so such a page withholds at
#: least one table and no more is claimed.
_PROSE_RECOVERY_RE = re.compile(
    r"\[page \d+" + re.escape(SCANNED_PROSE_RECOVERED_FLAG.split("{page_num}", 1)[1])
)

_UNVERIFIED_TRUST_KIND = "table_ladder_unverified"


@dataclass(frozen=True)
class TableCounts:
    shipped_text: int = 0
    verified_text: int = 0
    unverified_text: int = 0
    withheld: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    def summary_line(self) -> str:
        return (
            f"tables: {self.shipped_text} as text ({self.verified_text} verified), "
            f"{self.withheld} withheld"
        )


def count_page_tables(
    text: str,
    status: str,
    failure_mode: str,
    *,
    trust_reasons: Iterable[str] = (),
) -> TableCounts:
    """Counts for one page. ``status`` / ``failure_mode`` are the enum VALUES (strings)."""
    text = text or ""
    blocks = len(find_table_blocks(text))
    withheld = len(_WITHHELD_MARKER_RE.findall(text))
    if withheld and _PROSE_RECOVERY_RE.search(text):
        withheld = 1
    reasons = set(trust_reasons)
    unverified = failure_mode == FailureMode.TABLE_UNVERIFIED.value or (
        _UNVERIFIED_TRUST_KIND in reasons
    )
    verified = status == PageStatus.SUCCESS.value and not reasons
    return TableCounts(
        shipped_text=blocks,
        verified_text=blocks if verified else 0,
        unverified_text=blocks if unverified else 0,
        withheld=withheld,
    )


def sum_counts(counts: Iterable[TableCounts]) -> TableCounts:
    items = list(counts)
    return TableCounts(
        shipped_text=sum(c.shipped_text for c in items),
        verified_text=sum(c.verified_text for c in items),
        unverified_text=sum(c.unverified_text for c in items),
        withheld=sum(c.withheld for c in items),
    )


def _value(member) -> str:
    return getattr(member, "value", member) or ""


def count_document_tables(outputs: Iterable, trust_pages: dict[int, list[str]]) -> TableCounts:
    """Counts over finalized ``PageOutput``s; ``trust_pages`` maps page -> distrust kinds."""
    return sum_counts(
        count_page_tables(
            o.text,
            _value(o.status),
            _value(o.failure_mode),
            trust_reasons=trust_pages.get(o.page_num, ()),
        )
        for o in outputs
    )


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def count_from_sidecars(doc_dir: Path) -> TableCounts | None:
    """The same counts, read from a finished document directory.

    For output written before the ``tables`` metadata block existed. ``None`` when the
    directory has no page sidecars, which means "not recorded", never "zero".
    """
    pages_dir = Path(doc_dir) / "pages"
    sidecars = sorted(pages_dir.glob("*.json")) if pages_dir.is_dir() else []
    if not sidecars:
        return None
    trust = _read_json(Path(doc_dir) / "tables_trust.json")
    trust_pages: dict[int, list[str]] = {}
    if isinstance(trust, dict):
        for num, rec in (trust.get("pages") or {}).items():
            try:
                trust_pages[int(num)] = list((rec or {}).get("reasons") or ["unspecified"])
            except (TypeError, ValueError):
                continue
    per_page: list[TableCounts] = []
    for sidecar in sidecars:
        rec = _read_json(sidecar)
        if not isinstance(rec, dict):
            continue
        win = rec.get("winning_output")
        if not isinstance(win, dict):
            continue
        num = rec.get("page_num") or win.get("page_num") or 0
        per_page.append(
            count_page_tables(
                win.get("text") or "",
                str(rec.get("status") or win.get("status") or ""),
                str(rec.get("failure_mode") or win.get("failure_mode") or ""),
                trust_reasons=trust_pages.get(int(num), ()),
            )
        )
    return sum_counts(per_page) if per_page else None
