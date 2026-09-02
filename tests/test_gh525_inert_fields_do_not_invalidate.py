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
