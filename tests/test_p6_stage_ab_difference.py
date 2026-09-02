"""P6 stage A/B is BEHAVIOUR-PRESERVING: the required difference test.

Cold review round 1, finding 2. The stage A/B acceptance bar is old-bucket
membership == new-bucket membership, and byte identity of document status,
metadata notes, CLI summary, manifest, sidecars and the final ``.md``. This module
pins that bar three ways:

1. :class:`TestColdReviewD3Shapes` -- the two fixtures the cold review used to
   prove the D3 divergence, asserted directly against the pre-change predicates
   held in ``tests/conftest.py``.
2. :class:`TestAssembleFixtureCoverage` -- proves the autouse guard in
   ``tests/conftest.py`` is armed, and that it is armed for every module in the
   suite that drives ``_phase_assemble`` (so no fixture has to opt in, and none
   can be forgotten). That guard is what covers the golden corpus fixture in
   ``tests/test_pp2_agentic_fuse.py`` and the other 33 modules.
3. :class:`TestPreChangeByteIdentity` -- a seven-page corpus run through
   ``_phase_assemble`` and compared, field by field, against a capture of the SAME
   corpus run on the pre-change sources (``git archive HEAD src`` into a temp tree,
   captured into ``tests/fixtures/p6/prechange_assemble.json``).

The comparison is a DIFFERENCE, never an absolute measured tuple: both sides are
the same fixture, and the only thing that varies is which source tree computed it.
No provider is consulted -- ``_phase_assemble`` never routes -- so the pin holds
identically with and without a local ladder.

Two families of field are excluded from the byte comparison, both for reasons that
are not behaviour:

* ``disposition`` -- stage A's ADDITIVE sidecar/manifest field. Its presence is the
  ticket. :meth:`TestPreChangeByteIdentity.test_disposition_is_the_only_added_key`
  pins that it is the ONLY key either surface gained.
* ``socr_source_digest`` / ``run_fingerprint`` -- these move whenever ANY source
  byte changes, which is what they are for; and ``input_checksum`` /
  ``pdf_file_hash``, which hash a PyMuPDF-written PDF that embeds a creation
  timestamp and is therefore not byte-stable between runs at all. Nothing else is
  excluded: the manifest entry's ``fingerprint`` object is compared in full, minus
  the PDF hash nested in it.

Regenerate the pinned capture with::

    PYTHONPATH=<worktree>/src ~/venvs/socr/bin/python tests/regenerate_p6_prechange.py

which exports the pre-change sources with ``git archive HEAD src`` and runs this same
corpus against them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import P6_BUCKET_NAMES, old_disposition_buckets
from p6_corpus_fixture import build_corpus_state, capture, make_pdf
from socr.core.manifest import finalized_page_records
from socr.pipeline.orchestrator import _derive_disposition_buckets

pytest.importorskip("fitz")

PRECHANGE = Path(__file__).parent / "fixtures" / "p6" / "prechange_assemble.json"

#: Fields excluded from the byte comparison. See the module docstring.
#:
#: Narrowed in cold review round 2 (finding 7): the manifest entry's whole
#: ``fingerprint`` object used to be dropped, which also hid its ``engine``,
#: ``model_version``, ``image_hash``, ``prompt_hash``, ``render_dpi`` and the two
#: version fields -- every one of them behaviour-relevant. Only the PDF hash NESTED
#: inside it is volatile, and it is named here on its own.
VOLATILE_KEYS = frozenset(
    {
        # Stage A's additive field. Its presence is pinned separately, by
        # test_disposition_is_the_only_added_key.
        "disposition",
        # Move whenever any source byte changes, which is what they are for.
        "socr_source_digest",
        "run_fingerprint",
        # Hash a PyMuPDF-written PDF, which embeds a creation timestamp and is not
        # byte-stable between runs at all.
        "input_checksum",
        "pdf_file_hash",
    }
)


def _normalize(obj):
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    return obj


def _new_buckets(state) -> dict[str, set[int]]:
    return _derive_disposition_buckets(state, finalized_page_records(state))


class TestColdReviewD3Shapes:
    """The two shapes the cold review measured, pinned against the old predicates.

    Both carry the whole D3 flag conjunction AND a passing non-native
    ``best_output``. Selection returns at the ``PASSING_BEST_OUTPUT`` ending before
    the D3 branch is reached, so the page's disposition is
    ``(MODEL_OUTPUT, ACCEPTED_OUTPUT)`` and no disposition pair can recover the D3
    membership. The buckets must still claim the page exactly as they did before.
    """

    def test_flags_plus_passing_winner_stays_a_d3_floor_page(self, tmp_path: Path) -> None:
        state = build_corpus_state(make_pdf(tmp_path))
        old, new = old_disposition_buckets(state), _new_buckets(state)
        assert 4 in old["d3_floor_pages"], "fixture no longer builds the reviewed shape"
        assert new["d3_floor_pages"] == old["d3_floor_pages"]

    def test_flags_plus_passing_winner_plus_refused_grid_stays_d3_model(
        self, tmp_path: Path
    ) -> None:
        state = build_corpus_state(make_pdf(tmp_path))
        old, new = old_disposition_buckets(state), _new_buckets(state)
        assert 5 in old["d3_model_table_pages"], "fixture no longer builds the reviewed shape"
        assert new["d3_model_table_pages"] == old["d3_model_table_pages"]

    def test_every_bucket_matches_the_pre_change_predicate(self, tmp_path: Path) -> None:
        state = build_corpus_state(make_pdf(tmp_path))
        assert _new_buckets(state) == old_disposition_buckets(state)

    def test_the_reviewed_pages_are_not_disposition_derivable(self, tmp_path: Path) -> None:
        """Why the two D3 buckets stay flag-derived, stated as a test rather than a claim.

        If this ever fails, the ending vocabulary has changed and the D3 buckets can
        be folded back into the disposition-derived set.
        """
        from socr.core.manifest import PageEnding, PagePrimaryReason

        state = build_corpus_state(make_pdf(tmp_path))
        records = finalized_page_records(state)
        for page_num in (4, 5):
            disposition = records[page_num - 1].disposition
            assert disposition.ending is PageEnding.MODEL_OUTPUT
            assert disposition.primary_reason is PagePrimaryReason.ACCEPTED_OUTPUT
        # ...which is the same pair page 7, an ordinary passing model page with no
        # D3 flag on it at all, carries. The disposition cannot separate them.
        assert records[6].disposition == records[3].disposition


class TestAssembleFixtureCoverage:
    """The autouse guard is armed, and it covers every assemble-driving module."""

    def test_the_guard_actually_compares_on_a_real_phase_assemble(self, tmp_path: Path) -> None:
        """Not "installed" -- REACHED. One real assemble, one recorded comparison.

        Cold review round 2, finding 8. If ``_phase_assemble`` stops calling the
        helper, calls an imported alias, or a later patch displaces the wrapper,
        this fails while every structural check would still pass.
        """
        import conftest

        assert conftest.GUARD_CALL_LOG == [], "the log is cleared per test"
        capture(tmp_path)
        assert len(conftest.GUARD_CALL_LOG) == 1, (
            "one _phase_assemble must produce exactly one bucket comparison"
        )
        compared = conftest.GUARD_CALL_LOG[0]
        assert set(compared) == set(P6_BUCKET_NAMES)
        # ...and it compared a real document, not an empty one.
        assert compared["d3_floor_pages"] == {4} or compared["d3_floor_pages"] == {2, 4}

    def test_removing_the_guard_leaves_nothing_compared(self, tmp_path: Path, monkeypatch) -> None:
        """The control that makes the assertion above load-bearing.

        With the wrapper replaced by the production function the same real assemble
        records no comparison at all, so a green run of the test above is evidence
        the monkeypatch is doing work rather than evidence that nothing checks.
        """
        import conftest
        from socr.pipeline import orchestrator as orch

        monkeypatch.setattr(
            orch, "_derive_disposition_buckets", orch._derive_disposition_buckets.__wrapped__
        )
        conftest.GUARD_CALL_LOG.clear()
        capture(tmp_path)
        assert conftest.GUARD_CALL_LOG == []

    def test_guard_is_installed(self) -> None:
        from socr.pipeline import orchestrator as orch

        assert orch._derive_disposition_buckets.__name__ == "_checked", (
            "the conftest difference guard is not wrapping the production derivation"
        )

    def test_guard_catches_a_membership_change(self, tmp_path: Path) -> None:
        """Negative control: the guard must actually fail when membership moves.

        Without it, a guard that silently stopped comparing would be
        indistinguishable from a guard that passes.
        """
        from conftest import assert_buckets_unchanged

        state = build_corpus_state(make_pdf(tmp_path))
        assert_buckets_unchanged(state, _new_buckets(state))

        moved = _new_buckets(state)
        moved["d3_floor_pages"] = moved["d3_floor_pages"] - {4}
        with pytest.raises(AssertionError, match="changed assemble bucket membership"):
            assert_buckets_unchanged(state, moved)

    def test_every_assemble_driving_module_runs_under_the_guard(self) -> None:
        """Enumerated, so a new assemble fixture cannot quietly escape the guard.

        The guard is autouse at the ``tests/`` conftest level, so coverage is
        structural: every module below lives under that conftest. The list is the
        `grep -rl _phase_assemble tests/` result at the time of the fix, kept here
        so a module MOVING out of ``tests/`` is caught.
        """
        here = Path(__file__).parent
        drivers = sorted(
            p.name for p in here.glob("test_*.py") if "_phase_assemble" in p.read_text()
        )
        assert drivers, "no module drives _phase_assemble any more"
        assert (here / "conftest.py").exists()
        for name in drivers:
            assert (here / name).exists()


class TestPreChangeByteIdentity:
    """Every surface the acceptance bar names, against the pre-change capture."""

    @pytest.fixture(scope="class")
    def prechange(self) -> dict:
        return json.loads(PRECHANGE.read_text())

    @pytest.fixture
    def current(self, tmp_path: Path) -> dict:
        return capture(tmp_path)

    def test_document_status(self, prechange, current) -> None:
        assert current["doc_status"] == prechange["doc_status"]
        assert current["result_status"] == prechange["result_status"]
        assert current["result_audit_passed"] == prechange["result_audit_passed"]

    def test_metadata_notes(self, prechange, current) -> None:
        assert current["result_error"] == prechange["result_error"]

    def test_audit_events(self, prechange, current) -> None:
        assert _normalize(current["events"]) == prechange["events"]

    def test_cli_summary_lines(self, prechange, current) -> None:
        assert current["cli"] == prechange["cli"]

    def test_final_markdown_and_fragments(self, prechange, current) -> None:
        assert current["markdown"] == prechange["markdown"]

    def test_sidecars(self, prechange, current) -> None:
        assert _normalize(current["sidecars"]) == prechange["sidecars"]

    def test_manifest(self, prechange, current) -> None:
        assert _normalize(current["manifest"]) == prechange["manifest"]

    def test_disposition_is_the_only_added_key(self, current) -> None:
        """Stage A's persistence is additive: nothing else appeared, nothing left."""
        raw = json.loads(PRECHANGE.read_text())
        for name, sidecar in current["sidecars"].items():
            gained = set(sidecar) - set(raw["sidecars"][name]) - VOLATILE_KEYS
            lost = set(raw["sidecars"][name]) - set(sidecar)
            assert not gained and not lost, (name, gained, lost)
            assert "disposition" in sidecar
        entries = current["manifest"][0]["entries"]
        old_entries = raw["manifest"][0]["entries"]
        for page_key, entry in entries.items():
            gained = set(entry) - set(old_entries[page_key]) - VOLATILE_KEYS
            lost = set(old_entries[page_key]) - set(entry)
            assert not gained and not lost, (page_key, gained, lost)
            assert "disposition" in entry
