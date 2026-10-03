"""A total wall-clock budget for ONE page's table-judge ladder (GH-974).

Every ladder call already has its own total deadline (GH-968), but a page with
several tables runs rung 1, the adjudicator and the cell transcriber once per
table, so the per-call bounds multiply: a wedged peer that answers each call just
under its deadline can hold a single page for hours, silently. This budget caps
the SUM across all tables and all call kinds on the page.

Semantics:

* The budget is checked BEFORE each call. A call that starts inside the budget is
  allowed its own per-call deadline, so the page can overrun by at most one call's
  timeout (the call cannot be shortened without threading a deadline through every
  transport; that is deliberately not done here).
* Once spent, every remaining call is SKIPPED with a result the caller already
  treats as "no verdict" (``RungResult(ok=False)``, ``BlindCellResult(ok=False)``,
  ``None`` token), so the table ends UNVERIFIED through the existing terminals.
* ``on_exhausted`` fires exactly once per budget, with a message naming the
  budget, so the audit trail carries one event per page.
* ``report`` fires once per call that RAN (name, model, elapsed), the console
  line that makes a long page visible.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any


#: Audit-event kind emitted once per page when the budget runs out.
TABLE_LADDER_BUDGET_EXHAUSTED_KIND = "table_ladder_budget_exhausted"


class PageLadderBudget:
    def __init__(
        self,
        total_sec: float,
        *,
        on_exhausted: Callable[[str], None] | None = None,
        report: Callable[[str, str, float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.total_sec = float(total_sec)
        self._clock = clock
        self._started = clock()
        self._on_exhausted = on_exhausted
        self._report = report
        self.skipped = 0
        self._announced = False

    def remaining(self) -> float:
        return self.total_sec - (self._clock() - self._started)

    def message(self) -> str:
        return f"page table-ladder budget of {self.total_sec:g}s exhausted; remaining calls skipped"

    def call(
        self,
        name: str,
        model: str,
        fn: Callable[[], Any],
        skipped: Callable[[str], Any],
    ) -> Any:
        """Run ``fn`` if budget remains; else return ``skipped(message)``."""
        if self.remaining() <= 0:
            self.skipped += 1
            msg = self.message()
            if not self._announced:
                self._announced = True
                if self._on_exhausted is not None:
                    self._on_exhausted(msg)
            return skipped(msg)
        start = self._clock()
        try:
            return fn()
        finally:
            if self._report is not None:
                self._report(name, model, self._clock() - start)

    def wrap_rung(self, rung: Callable[..., Any]) -> Callable[..., Any]:
        """A reader rung that respects this budget; attributes are preserved."""
        from socr.judge.table_verdict import RungResult

        rung_id = getattr(rung, "rung_id", "") or getattr(rung, "__name__", "") or "rung"
        model = getattr(rung, "executing", "") or ""

        @functools.wraps(rung)
        def _budgeted(crop_path, markdown, prior_findings):
            return self.call(
                rung_id,
                model,
                lambda: rung(crop_path, markdown, prior_findings),
                lambda msg: RungResult(rung=rung_id, ok=False, error=msg),
            )

        return _budgeted

    def wrap_adjudicator(self, adjudicator: Callable[..., Any] | None) -> Callable[..., Any] | None:
        """The blind-cell adjudicator under this budget (``None`` stays ``None``)."""
        if adjudicator is None:
            return None
        from socr.judge.table_rung_ollama import BlindCellResult

        rung_id = getattr(adjudicator, "rung_id", "") or "adjudicator"
        model = getattr(adjudicator, "executing", "") or ""

        @functools.wraps(adjudicator)
        def _budgeted(crop_path, cell_refs):
            return self.call(
                rung_id,
                model,
                lambda: adjudicator(crop_path, cell_refs),
                lambda msg: BlindCellResult(rung=rung_id, ok=False, error=msg),
            )

        return _budgeted
