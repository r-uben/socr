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
from dataclasses import asdict, dataclass, replace
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
    #: ``None`` means unknown: the evidence for that count was incomplete or unreadable.
    #: Never a confident number built from partial evidence.
    shipped_text: int | None = 0
    verified_text: int | None = 0
    unverified_text: int | None = 0
    withheld: int | None = 0

    def to_dict(self) -> dict[str, int | None]:
        return asdict(self)

    def summary_line(self) -> str:
        return (
            f"tables: {self.shipped_text} as text ({self.verified_text} verified, "
            f"{self.unverified_text} unverified), {self.withheld} withheld"
        )


def count_page_tables(
    text: str,
    status: str,
    failure_mode: str,
    *,
    trust_reasons: Iterable[str] = (),
    withheld_events: int = 0,
) -> TableCounts:
    """Counts for one page. ``status`` / ``failure_mode`` are the enum VALUES (strings).

    ``withheld_events`` is the number of distinct tables the page's ``table_ladder_withheld``
    events name. A whole-page floor carries ONE marker however many tables it removed, so the
    markers alone undercount a page-granular withhold; each removed table has its own event.
    """
    text = text or ""
    blocks = len(find_table_blocks(text))
    withheld = len(_WITHHELD_MARKER_RE.findall(text))
    if withheld and _PROSE_RECOVERY_RE.search(text):
        withheld = 1
    if withheld:
        withheld = max(withheld, withheld_events)
    reasons = set(trust_reasons)
    unverified = failure_mode == FailureMode.TABLE_UNVERIFIED.value or (
        _UNVERIFIED_TRUST_KIND in reasons
    )
    verified = status == PageStatus.SUCCESS.value and not reasons and not unverified
    return TableCounts(
        shipped_text=blocks,
        verified_text=blocks if verified else 0,
        unverified_text=blocks if unverified else 0,
        withheld=withheld,
    )


def _sum_field(values: list[int | None]) -> int | None:
    return None if any(v is None for v in values) else sum(values)


def sum_counts(counts: Iterable[TableCounts]) -> TableCounts:
    """Field-wise sum; a field unknown on any input is unknown in the sum."""
    items = list(counts)
    return TableCounts(
        shipped_text=_sum_field([c.shipped_text for c in items]),
        verified_text=_sum_field([c.verified_text for c in items]),
        unverified_text=_sum_field([c.unverified_text for c in items]),
        withheld=_sum_field([c.withheld for c in items]),
    )


def _value(member) -> str:
    return getattr(member, "value", member) or ""


def withheld_table_events(events: Iterable[dict]) -> dict[int, int]:
    """``{page: distinct tables named by its table_ladder_withheld events}``.

    Events are plain dicts (``kind``, ``page_num``, ``data``) so the live audit events and a
    sidecar's ``audit_events`` read through one function.
    """
    tables: dict[int, set[str]] = {}
    for event in events:
        if event.get("kind") != "table_ladder_withheld":
            continue
        table_id = str((event.get("data") or {}).get("table_id") or "")
        if table_id:
            tables.setdefault(int(event.get("page_num") or 0), set()).add(table_id)
    return {page: len(ids) for page, ids in tables.items()}


def count_document_tables(
    outputs: Iterable,
    trust_pages: dict[int, list[str]],
    withheld_events: dict[int, int] | None = None,
) -> TableCounts:
    """Counts over finalized ``PageOutput``s; ``trust_pages`` maps page -> distrust kinds."""
    withheld_events = withheld_events or {}
    return sum_counts(
        count_page_tables(
            o.text,
            _value(o.status),
            _value(o.failure_mode),
            trust_reasons=trust_pages.get(o.page_num, ()),
            withheld_events=withheld_events.get(o.page_num, 0),
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
    directory has no page sidecars, or any sidecar is unreadable or lacks a winning output:
    unknown, never "zero" and never a total over the pages that happened to parse. If only
    ``tables_trust.json`` is unreadable, ``verified_text`` and ``unverified_text`` are
    ``None`` and the other two counts stay known.
    """
    pages_dir = Path(doc_dir) / "pages"
    sidecars = sorted(pages_dir.glob("*.json")) if pages_dir.is_dir() else []
    if not sidecars:
        return None
    # The document's recorded page set. Reprocessing does not delete sidecars of pages
    # that are gone, so the glob alone can include strangers; and a recorded page with no
    # sidecar would make the total partial. Either way the evidence is not the document.
    meta = _read_json(Path(doc_dir) / "metadata.json")
    page_count = meta.get("pages") if isinstance(meta, dict) else None
    if type(page_count) is not int or page_count < 1:
        return None
    wanted = set(range(1, page_count + 1))
    # tables_trust.json: absent means no table is flagged (the file's own contract), but a
    # file that is present and unreadable says nothing. Reading it as "no distrust" would
    # overstate verified, so the trust-dependent counts become unknown instead.
    trust_path = Path(doc_dir) / "tables_trust.json"
    trust_known = True
    trust_pages: dict[int, list[str]] = {}
    if trust_path.exists():
        trust = _read_json(trust_path)
        if not isinstance(trust, dict) or not isinstance(trust.get("pages") or {}, dict):
            trust_known = False
        else:
            for num, rec in (trust.get("pages") or {}).items():
                try:
                    trust_pages[int(num)] = list((rec or {}).get("reasons") or ["unspecified"])
                except (TypeError, ValueError, AttributeError):
                    trust_known = False
    per_page: list[TableCounts] = []
    seen: set[int] = set()
    for sidecar in sidecars:
        rec = _read_json(sidecar)
        win = rec.get("winning_output") if isinstance(rec, dict) else None
        if not isinstance(win, dict):
            # A skipped page would make every total a partial sum that reads as complete.
            return None
        try:
            num = int(rec.get("page_num") or win.get("page_num") or 0)
        except (TypeError, ValueError):
            return None
        if num not in wanted:
            continue
        seen.add(num)
        per_page.append(
            count_page_tables(
                win.get("text") or "",
                str(rec.get("status") or win.get("status") or ""),
                str(rec.get("failure_mode") or win.get("failure_mode") or ""),
                trust_reasons=trust_pages.get(num, ()),
                withheld_events=withheld_table_events(
                    {**e, "page_num": num}
                    for e in (rec.get("audit_events") or [])
                    if isinstance(e, dict)
                ).get(num, 0),
            )
        )
    if seen != wanted:
        return None
    total = sum_counts(per_page)
    if not trust_known:
        total = replace(total, verified_text=None, unverified_text=None)
    return total
