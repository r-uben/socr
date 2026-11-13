"""#1282: the point repair acts only on the reading that ships.

``_repair_dropped_points`` runs before ``finalized_page_record`` picks the shipped table (D3 floor
keep, structure-class grid winner, else ``best_output``).  When an OLDER attempt wins, a repair on
``best_output`` would record a ``table_point_repaired`` event for bytes that never ship.

Both arms run ``process()`` on the same synthetic PDF and fake model output, and differ in one
thing: which attempt the shipped-table predicate returns (the older, unrepaired one, or
``best_output`` itself).  The assertion is the DIFFERENCE: the repair event appears iff the repaired
text ships.  Run in both provider states, with ``primary_engine`` / ``local_engine`` /
``enabled_engines`` pinned and ``_available_engines_for_agentic`` patched by ``_run``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("fitz")

from test_gh1267_row_binding_check import md  # noqa: E402
from test_gh1267_row_binding_pipeline import _events, _fragment, _run  # noqa: E402
from test_gh1282_point_repair_pipeline import DROPPED, NATIVE  # noqa: E402

from socr.core.result import PageOutput, PageStatus  # noqa: E402

KIND = "table_point_repaired"


def _older(p):
    """An older attempt: the same table as the model read it, point still dropped."""
    return PageOutput(
        page_num=p.best_output.page_num,
        text=md(DROPPED),
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )


def _run_with_winner(tmp_path, name, older_wins, providers):
    older_holder = {}

    def _winner(p):
        if not older_wins or p.best_output is None:
            return p.best_output
        return older_holder.setdefault("o", _older(p))

    with (
        patch("socr.core.manifest._reaches_structure_class_branch", return_value=True),
        patch("socr.core.manifest.structure_class_grid_winner", side_effect=_winner),
    ):
        return _run(
            tmp_path, DROPPED, 0, check_runs=True, out_name=name, native=NATIVE, providers=providers
        )


@pytest.mark.parametrize("providers", [True, False], ids=["ladder", "no_ladder"])
def test_repair_event_agrees_with_the_shipped_reading(tmp_path, providers):
    _, _, st_best, best, pdf = _run_with_winner(tmp_path, "best", False, providers)
    _, _, st_old, old, _ = _run_with_winner(tmp_path, "old", True, providers)
    ev_best, ev_old = _events(best, KIND), _events(old, KIND)
    if not providers:
        # no model reading to repair: the arms cannot differ
        assert ev_best == ev_old == []
        return
    # control: best_output ships, so the repair ships and its event fires
    assert "| -.7 |" in _fragment(best)
    assert len(ev_best) == 1
    # older attempt ships: its bytes are unrepaired, and no event or repair describes others
    assert "| -.7 |" not in _fragment(old)
    assert ev_old == []
    assert all(not ps.point_repairs for ps in st_old.pages.values())
    assert any(ps.point_repairs for ps in st_best.pages.values())
