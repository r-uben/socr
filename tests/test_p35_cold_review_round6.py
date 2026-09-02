"""Cold review round 6 on the P3+P5 branch (`fix/p3-p5-judged-bytes-ship`).

Round 5 made per-page spend a recorded fact and wired the five sites it knew
about. The round-5 review found the class was still open, not the instance: the
corrupt-math recovery lane journals a page ``EngineResult`` with an UNKNOWN cost
and never records it, so the page kept the default known zero; and
``DocumentState.apply_result`` journals without touching the fact at all.

Ruling: journaling and recording become ONE call. Everything that appends an
``EngineResult`` goes through ``DocumentState.record_engine_run``, and a guard
test fails if a new site appears outside it -- so the next lane cannot bypass the
contract by being written the obvious way.

Hermetic: no provider, no network, no live model.
"""

from __future__ import annotations


import pytest

from socr.core.document import DocumentHandle
from socr.core.result import DocumentStatus, EngineResult, PageOutput, PageStatus
from socr.core.state import DocumentState

fitz = pytest.importorskip("fitz")


def _state(tmp_path):
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "source")
    doc.save(pdf)
    doc.close()
    return DocumentState(DocumentHandle(pdf))


# ---------------------------------------------------------------------------
# 1 — one call journals AND records
# ---------------------------------------------------------------------------


class TestJournalingAndRecordingAreOneCall:
    def test_recording_a_run_charges_the_page(self, tmp_path) -> None:
        state = _state(tmp_path)
        state.record_engine_run(
            EngineResult(
                document_path=state.handle.path,
                engine="gemini",
                status=DocumentStatus.SUCCESS,
                cost=0.0002,
            ),
            page_nums=[1],
        )
        assert state.total_cost == 0.0002
        assert state.pages[1].page_cost_usd == 0.0002

    def test_an_unknown_cost_makes_the_page_total_unknown(self, tmp_path) -> None:
        """The absorbing rule, which is the whole corrupt-math finding: a lane
        that cannot say what it spent must not leave the page reading 0.00."""
        state = _state(tmp_path)
        state.record_engine_run(
            EngineResult(
                document_path=state.handle.path,
                engine="native+math",
                status=DocumentStatus.AUDIT_FAILED,
                cost=None,
            ),
            page_nums=[1],
        )
        assert state.total_cost is None
        assert state.pages[1].page_cost_usd is None

    def test_unknown_never_decays_back_to_a_number(self, tmp_path) -> None:
        state = _state(tmp_path)
        state.record_engine_run(
            EngineResult(
                document_path=state.handle.path,
                engine="native+math",
                status=DocumentStatus.AUDIT_FAILED,
                cost=None,
            ),
            page_nums=[1],
        )
        state.record_engine_run(
            EngineResult(
                document_path=state.handle.path,
                engine="gemini",
                status=DocumentStatus.SUCCESS,
                cost=0.0002,
            ),
            page_nums=[1],
        )
        assert state.pages[1].page_cost_usd is None

    def test_apply_result_records_the_fact_too(self, tmp_path) -> None:
        """``apply_result`` is the module's stated orchestrator merge point. It
        has no in-tree caller today, which is exactly why it would have been the
        next lane to bypass the contract silently."""
        state = _state(tmp_path)
        state.apply_result(
            EngineResult(
                document_path=state.handle.path,
                engine="gemini",
                status=DocumentStatus.SUCCESS,
                cost=0.0002,
                pages=[
                    PageOutput(page_num=1, text="x", status=PageStatus.SUCCESS, engine="gemini")
                ],
            )
        )
        assert state.total_cost == 0.0002
        assert state.pages[1].page_cost_usd == 0.0002


# ---------------------------------------------------------------------------
# 2 — no site may bypass the helper
# ---------------------------------------------------------------------------


# The AST guard that stood here is retired. The round-6 review defeated it four
# ways -- an alias then ``append``, a ``list(...) + [...]`` reassignment, a
# ``getattr`` hop, and a subclass method that simply took the exempt name -- and
# a pattern-matcher cannot win that game. Round 7 replaced it with
# ENCAPSULATION: the journal is private and ``engine_runs`` is a read-only view,
# so every one of those shapes raises where it is written. What survives is a
# much smaller scoped guard on the private NAME, in
# ``tests/test_p35_cold_review_round7.py``, together with the reviewer's five
# probes. The one-call contract this file pins is unchanged and still tested
# above; only its enforcement mechanism moved.
