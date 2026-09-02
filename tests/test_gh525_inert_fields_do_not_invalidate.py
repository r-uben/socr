"""GH-525: a setting that changes nothing must change nothing.

`--no-judge-hard-pages` and `--fallback` are rejected at the CLI (GH-142)
because neither gates any phase. YAML and `PipelineConfig` can still set
`judge_hard_pages` and `fallback_chain`, and those values reached
`_run_fingerprint` -- so a config-only toggle invalidated every terminal page
and forced a full reprocess producing byte-identical output. The same cost as
the rejected flags, without the error that explains it.

The fix records the DEFAULTS for those two fields instead of the configured
values. That choice is deliberate over the two obvious alternatives:

- DROPPING the keys would change the fingerprint for every existing run,
  imposing one global reprocess to fix a problem almost nobody has hit;
- REJECTING them at config load would break `benchmark calibrate
  --apply-config`, which writes a YAML containing `fallback_chain`.

Recording the default costs neither: the key stays, every existing fingerprint
is unchanged, and the toggle stops mattering.

Ignoring a setting SILENTLY is the failure this ticket family is about, so the
run also says which fields it ignored. Both halves are pinned here.
"""

from __future__ import annotations

import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.pipeline.orchestrator import UnifiedPipeline


def _fingerprint(**overrides) -> str:
    return UnifiedPipeline(PipelineConfig(quiet=True, **overrides))._run_fingerprint()


def test_toggling_an_inert_field_does_not_move_the_fingerprint() -> None:
    """The bug, as a difference that must NOT appear."""
    baseline = _fingerprint()

    assert _fingerprint(judge_hard_pages=False) == baseline, (
        "setting judge_hard_pages invalidated the run identity; it gates no "
        "phase, so every terminal page would be reprocessed for identical output"
    )
    assert _fingerprint(fallback_chain=[EngineType.QWEN]) == baseline, (
        "setting fallback_chain invalidated the run identity; no execution path reads it"
    )
    assert _fingerprint(judge_hard_pages=False, fallback_chain=[EngineType.QWEN]) == baseline


def test_a_live_field_still_moves_the_fingerprint() -> None:
    """The control, and the thing that could go wrong with this fix.

    Freezing the wrong field would silently make a REAL setting invisible to
    the resume ledger -- pages reused across a genuine change. Without this,
    a change that froze half the fingerprint would satisfy the test above.
    """
    baseline = _fingerprint()

    assert _fingerprint(strict_local=True) != baseline, "strict_local stopped being fingerprinted"
    assert _fingerprint(render_dpi=123) != baseline, "render_dpi stopped being fingerprinted"
    assert _fingerprint(native_only=True) != baseline, "native_only stopped being fingerprinted"


def test_the_ignored_fields_are_named_out_loud() -> None:
    """Ignoring a setting in silence is the failure being fixed, one layer down."""
    from socr.pipeline.orchestrator import _warn_inert_config

    assert _warn_inert_config(PipelineConfig()) == [], (
        "a default config reported ignored fields, so every run would warn and "
        "the warning would mean nothing"
    )

    cfg = PipelineConfig(judge_hard_pages=False, fallback_chain=[EngineType.QWEN])
    assert set(_warn_inert_config(cfg)) == {"judge_hard_pages", "fallback_chain"}


@pytest.mark.parametrize("field", ["judge_hard_pages", "fallback_chain"])
def test_the_frozen_value_is_the_config_default(field: str) -> None:
    """Taken from PipelineConfig, never written out beside it.

    A hand-copied default would drift the day someone changes the real one, and
    the fingerprint would then freeze a value the config no longer has.
    """
    from socr.pipeline.orchestrator import _INERT_FIELD_DEFAULTS

    assert _INERT_FIELD_DEFAULTS[field] == getattr(PipelineConfig(), field)


class TestTheWarningReachesTheRun:
    """cubic P3 on #529: the helper was pinned, its call site was not.

    Everything above asserts what `_warn_inert_config` RETURNS. A refactor that
    stopped calling it would leave all of that green while the setting went back
    to being ignored in silence -- which is the half of this fix that is about
    the operator rather than the cache.
    """

    def _run(self, tmp_path, caplog, *, inert: bool, documents: int = 1, batch: bool = False):
        import logging
        from unittest.mock import patch

        import fitz

        from socr.pipeline.orchestrator import UnifiedPipeline

        tmp_path.mkdir(parents=True, exist_ok=True)
        pdfs = []
        for i in range(documents):
            doc = fitz.open()
            doc.new_page().insert_text((72, 72), f"a text layer for document {i}.")
            path = tmp_path / f"d{i}.pdf"
            doc.save(path)
            doc.close()
            pdfs.append(path)

        overrides = {"judge_hard_pages": False} if inert else {}
        pipeline = UnifiedPipeline(PipelineConfig(quiet=True, **overrides))

        with (
            caplog.at_level(logging.WARNING),
            patch.object(UnifiedPipeline, "_phase_agentic", lambda self, *a, **k: None),
        ):
            if batch:
                # The REAL batch entry point (cubic P2 on #529). Looping
                # `process` myself simulated a batch and would have passed while
                # `process_batch` regressed -- the same "drive the real caller"
                # failure this suite keeps finding elsewhere.
                processed = len(pipeline.process_batch(tmp_path, tmp_path / "out"))
            else:
                for pdf in pdfs:
                    pipeline.process(pdf, tmp_path / "out")
                processed = len(pdfs)

        return processed, [
            r.getMessage() for r in caplog.records if "judge_hard_pages" in r.getMessage()
        ]

    def test_a_run_with_an_inert_field_set_says_so(self, tmp_path, caplog) -> None:
        _processed, messages = self._run(tmp_path / "warned", caplog, inert=True)
        assert messages, (
            "the run ignored judge_hard_pages without saying so; the call site "
            "is not wired, so only the helper is tested"
        )

    def test_a_default_run_stays_quiet(self, tmp_path, caplog) -> None:
        """Control: a warning on every run is a warning nobody reads."""
        _processed, messages = self._run(tmp_path / "quiet", caplog, inert=False)
        assert messages == []

    def test_a_batch_says_it_once_not_once_per_document(self, tmp_path, caplog) -> None:
        """Through `process_batch` itself, not a hand-rolled loop over `process`."""
        processed, messages = self._run(
            tmp_path / "batch", caplog, inert=True, documents=3, batch=True
        )

        # cubic P3 on #529: `_report_inert_config` runs BEFORE the PDF glob, so
        # one warning is emitted whether the batch found three documents or
        # none. Without this, a regression in document discovery would leave
        # this test green while the "once, not once-per-document" claim it makes
        # went unexercised -- the same false green the rest of this PR removed.
        assert processed == 3, (
            f"the batch processed {processed} documents, so the once-per-batch "
            "claim was not measured against a batch that did anything"
        )
        assert len(messages) == 1, (
            f"3 documents produced {len(messages)} identical warnings; the "
            "config cannot change between them"
        )

    def test_an_empty_batch_still_reports_what_it_is_ignoring(self, tmp_path, caplog) -> None:
        """cubic P2 on #529: nothing to process is not nothing to say.

        Warning only from `process` meant a batch that found no documents --
        an empty directory, or one where everything was already done -- reported
        nothing, while its fingerprint ignored the fields just the same. That is
        the silence this half of the fix exists to remove.
        """
        processed, messages = self._run(
            tmp_path / "empty", caplog, inert=True, documents=0, batch=True
        )

        assert processed == 0, "the control failed: this batch was supposed to find nothing"
        assert len(messages) == 1, (
            f"an empty batch produced {len(messages)} warnings; it ignores the "
            "same settings a non-empty one does"
        )
