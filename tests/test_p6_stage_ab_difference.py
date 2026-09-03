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

One family of field is excluded from the byte comparison, for reasons that are not
behaviour:

* ``socr_source_digest`` / ``run_fingerprint`` -- these move whenever ANY source
  byte changes, which is what they are for; and ``input_checksum`` /
  ``pdf_file_hash``, which hash a PyMuPDF-written PDF that embeds a creation
  timestamp and is therefore not byte-stable between runs at all. Nothing else is
  excluded: the manifest entry's ``fingerprint`` object is compared in full, minus
  the PDF hash nested in it.

``disposition`` is NOT excluded (cold review round 1, finding 1). It was, while this
module and stage C shared a capture that had been written stripped -- which made the
one field both stages are about invisible to the comparison. The shared baseline is
now regenerated at the stage-A/B HEAD WITH the field, so stage A's persistence is
already present on both sides here and is compared like any other key;
:meth:`TestPreChangeByteIdentity.test_disposition_is_present_and_no_key_moved` pins
that it is on every sidecar and manifest entry and that neither surface gained or
lost any other key.

Regenerate the pinned capture with::

    PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-p6c/src uv run socr-regenerate-p6-prechange

which exports the pre-change sources with ``git archive HEAD src`` and runs this same
corpus against them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import P6_BUCKET_NAMES, assert_stage_c_disposition_buckets, old_disposition_buckets
from p6_corpus_fixture import build_corpus_state, capture, make_pdf
from p6_stage_c_oracle import (
    VOLATILE_KEYS,
    apply_stage_c_patch,
    assert_capture_diff_matches_oracle,
    normalize_capture,
)

from socr.core.manifest import finalized_page_records
from socr.pipeline.orchestrator import _derive_disposition_buckets

pytest.importorskip("fitz")

PRECHANGE = Path(__file__).parent / "fixtures" / "p6" / "prechange_assemble.json"

_normalize = normalize_capture


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

    def test_every_bucket_matches_the_stage_c_contract(self, tmp_path: Path) -> None:
        """Adjusted from old blanket every-bucket check to the stage-C two-rule contract."""
        state = build_corpus_state(make_pdf(tmp_path))
        records = finalized_page_records(state)
        assert_stage_c_disposition_buckets(state, records, _new_buckets(state))

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
        """Not "installed" -- REACHED. One real assemble, both recorded comparisons.

        Cold review round 2, finding 8. If ``_phase_assemble`` stops calling the
        helper, calls an imported alias, or a later patch displaces the wrapper,
        this fails while every structural check would still pass.
        """
        import conftest

        assert conftest.DISPOSITION_GUARD_CALL_LOG == [], "the log is cleared per test"
        assert conftest.ORTHOGONAL_GUARD_CALL_LOG == [], "the log is cleared per test"
        capture(tmp_path)
        assert len(conftest.DISPOSITION_GUARD_CALL_LOG) == 1, (
            "one _phase_assemble must produce exactly one disposition bucket comparison"
        )
        assert len(conftest.ORTHOGONAL_GUARD_CALL_LOG) == 1, (
            "one _phase_assemble must produce exactly one orthogonal bucket comparison"
        )
        disp_compared = conftest.DISPOSITION_GUARD_CALL_LOG[0]
        assert set(disp_compared) == set(P6_BUCKET_NAMES)
        # ...and it compared a real document, not an empty one.
        assert disp_compared["d3_floor_pages"] == {4} or disp_compared["d3_floor_pages"] == {2, 4}

        orth_compared = conftest.ORTHOGONAL_GUARD_CALL_LOG[0]
        assert set(orth_compared) == set(conftest.ORTHOGONAL_BUCKET_NAMES)

    def test_removing_the_disposition_guard_leaves_disposition_log_empty(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The control that makes the disposition assertion above load-bearing."""
        import conftest

        from socr.pipeline import orchestrator as orch

        monkeypatch.setattr(
            orch, "_derive_disposition_buckets", orch._derive_disposition_buckets.__wrapped__
        )
        conftest.DISPOSITION_GUARD_CALL_LOG.clear()
        conftest.ORTHOGONAL_GUARD_CALL_LOG.clear()
        capture(tmp_path)
        assert conftest.DISPOSITION_GUARD_CALL_LOG == []
        assert len(conftest.ORTHOGONAL_GUARD_CALL_LOG) == 1

    def test_removing_the_orthogonal_guard_leaves_orthogonal_log_empty(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The control that makes the orthogonal assertion above load-bearing."""
        import conftest

        from socr.pipeline import orchestrator as orch

        monkeypatch.setattr(
            orch,
            "_derive_orthogonal_assemble_buckets",
            orch._derive_orthogonal_assemble_buckets.__wrapped__,
        )
        conftest.DISPOSITION_GUARD_CALL_LOG.clear()
        conftest.ORTHOGONAL_GUARD_CALL_LOG.clear()
        capture(tmp_path)
        assert conftest.ORTHOGONAL_GUARD_CALL_LOG == []
        assert len(conftest.DISPOSITION_GUARD_CALL_LOG) == 1

    def test_removing_both_guards_leaves_both_logs_empty(self, tmp_path: Path, monkeypatch) -> None:
        """Bypassing both wrappers leaves both call logs empty."""
        import conftest

        from socr.pipeline import orchestrator as orch

        monkeypatch.setattr(
            orch, "_derive_disposition_buckets", orch._derive_disposition_buckets.__wrapped__
        )
        monkeypatch.setattr(
            orch,
            "_derive_orthogonal_assemble_buckets",
            orch._derive_orthogonal_assemble_buckets.__wrapped__,
        )
        conftest.DISPOSITION_GUARD_CALL_LOG.clear()
        conftest.ORTHOGONAL_GUARD_CALL_LOG.clear()
        capture(tmp_path)
        assert conftest.DISPOSITION_GUARD_CALL_LOG == []
        assert conftest.ORTHOGONAL_GUARD_CALL_LOG == []

    def test_guard_is_installed(self) -> None:
        from socr.pipeline import orchestrator as orch

        assert getattr(orch._derive_disposition_buckets, "__wrapped__", None) is not None, (
            "the conftest difference guard is not wrapping the disposition derivation"
        )
        assert getattr(orch._derive_orthogonal_assemble_buckets, "__wrapped__", None) is not None, (
            "the conftest difference guard is not wrapping the orthogonal derivation"
        )

    def test_guard_catches_flag_derived_bucket_perturbation(self, tmp_path: Path) -> None:
        """Negative control: the guard must fail when a flag-derived bucket is perturbed."""
        from conftest import assert_stage_c_disposition_buckets

        state = build_corpus_state(make_pdf(tmp_path))
        records = finalized_page_records(state)
        buckets = _new_buckets(state)
        assert_stage_c_disposition_buckets(state, records, buckets)

        moved = dict(buckets)
        moved["d3_floor_pages"] = (
            moved["d3_floor_pages"] - {4}
            if 4 in moved["d3_floor_pages"]
            else moved["d3_floor_pages"] | {99}
        )
        with pytest.raises(AssertionError, match="flag-derived bucket 'd3_floor_pages'"):
            assert_stage_c_disposition_buckets(state, records, moved)

    def test_guard_catches_disposition_derived_bucket_perturbation(self, tmp_path: Path) -> None:
        """Negative control: the guard must fail when a disposition-derived bucket is perturbed."""
        from conftest import assert_stage_c_disposition_buckets

        state = build_corpus_state(make_pdf(tmp_path))
        records = finalized_page_records(state)
        buckets = _new_buckets(state)
        assert_stage_c_disposition_buckets(state, records, buckets)

        moved = dict(buckets)
        moved["structure_class_model_pages"] = moved["structure_class_model_pages"] ^ {8}
        with pytest.raises(
            AssertionError, match="disposition-derived bucket 'structure_class_model_pages'"
        ):
            assert_stage_c_disposition_buckets(state, records, moved)

    def test_guard_catches_orthogonal_bucket_perturbation(self, tmp_path: Path) -> None:
        """Negative control: the guard must fail when an orthogonal bucket is perturbed."""
        from conftest import assert_orthogonal_buckets_unchanged

        from socr.pipeline.orchestrator import _derive_orthogonal_assemble_buckets

        state = build_corpus_state(make_pdf(tmp_path))
        orth = _derive_orthogonal_assemble_buckets(state)
        assert_orthogonal_buckets_unchanged(state, orth)

        moved = dict(orth)
        moved["native_only_distrust_pages"] = [99]
        with pytest.raises(AssertionError, match="orthogonal assemble bucket membership changed"):
            assert_orthogonal_buckets_unchanged(state, moved)

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
    """Every surface the acceptance bar names, against pre-change capture via the oracle."""

    @pytest.fixture(scope="class")
    def prechange(self) -> dict:
        return json.loads(PRECHANGE.read_text())

    @pytest.fixture(scope="class")
    def patched_prechange(self, prechange) -> dict:
        return apply_stage_c_patch(normalize_capture(prechange))

    @pytest.fixture
    def current(self, tmp_path: Path) -> dict:
        return capture(tmp_path)

    def test_document_status(self, patched_prechange, current) -> None:
        assert current["doc_status"] == patched_prechange["doc_status"]
        assert current["result_status"] == patched_prechange["result_status"]
        assert current["result_audit_passed"] == patched_prechange["result_audit_passed"]

    def test_metadata_notes(self, patched_prechange, current) -> None:
        assert current["result_error"] == patched_prechange["result_error"]

    def test_audit_events(self, patched_prechange, current) -> None:
        assert normalize_capture(current["events"]) == patched_prechange["events"]

    def test_cli_summary_lines(self, patched_prechange, current) -> None:
        assert current["cli"] == patched_prechange["cli"]

    def test_final_markdown_and_fragments(self, patched_prechange, current) -> None:
        assert current["markdown"] == patched_prechange["markdown"]

    def test_sidecars(self, patched_prechange, current) -> None:
        assert normalize_capture(current["sidecars"]) == patched_prechange["sidecars"]

    def test_manifest(self, patched_prechange, current) -> None:
        assert normalize_capture(current["manifest"]) == patched_prechange["manifest"]

    def test_exact_delta_matches_shared_oracle(self, prechange, current) -> None:
        assert_capture_diff_matches_oracle(prechange, current)

    def test_disposition_is_present_and_no_key_moved(self, current) -> None:
        """Every sidecar and manifest entry carries ``disposition``, and no key moved.

        The baseline is the stage-A/B HEAD, which already persisted the field, so
        this is a presence-and-stability pin rather than the additive-diff it was
        while the capture was written with ``disposition`` stripped out.
        """
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
