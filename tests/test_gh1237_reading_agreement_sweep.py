"""#1237: a change to ``tables.reading_agreement`` may not lose a dispute the baseline had.

#1227 needed three rounds, and two of them made a wrong number agree that ``main``
disputed (``$\\pi^* = (9 \\%)$``, then ``(9) \\%``); only a reviewer caught each. This
runs the comparator over the generated grid in ``socr.devtools.reading_agreement_sweep``
and compares it with the checked-in baseline:

* WRONG values (the layer prints another number): no reading may compare fewer numbers
  or raise fewer disputes than the baseline. A drop of a disputed value is a false
  agreement.
* RIGHT values (the layer prints the same value): a reading that now disputes more
  than the baseline must dispute exactly as its unspaced control does today, so a
  change can only move a spaced writing to where the unspaced one already is.

The baseline pins a DIFFERENCE, not a measured page outcome, and ``compare_numbers`` is
pure text, so this is the same with and without a provider. A change that is meant to
move these results regenerates the baseline (``socr-regenerate-reading-agreement-sweep``)
and the reviewed diff shows every reading it moved.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import socr
from socr.devtools import reading_agreement_sweep as sweep
from socr.tables.reading_agreement import PROVENANCE_OWN_CHARACTERS, compare_numbers

ROOT = Path(__file__).resolve().parents[1]
BASELINE = json.loads((ROOT / sweep.FIXTURE).read_text())
COMPARE = functools.partial(compare_numbers, provenance=PROVENANCE_OWN_CHARACTERS)


def test_loaded_source_is_this_checkout() -> None:
    assert Path(socr.__file__).resolve().is_relative_to(ROOT / "src")


def test_baseline_covers_exactly_the_grid() -> None:
    """A grid edit that is not regenerated would silently stop checking readings."""
    for side, numbers in (("wrong", sweep.WRONG_NUMBERS), ("right", sweep.RIGHT_NUMBERS)):
        grid = {reading for _key, reading in sweep.readings(numbers)}
        assert set(BASELINE[side]["outcomes"]) == grid, side
    assert BASELINE["wrong"]["layer"] == sweep.WRONG_LAYER
    assert BASELINE["right"]["layer"] == sweep.RIGHT_LAYER


def test_no_wrong_value_loses_a_dispute_or_a_compared_number() -> None:
    now = sweep.outcomes(COMPARE, sweep.WRONG_LAYER, sweep.WRONG_NUMBERS)
    base = BASELINE["wrong"]["outcomes"]
    lost = [
        (reading, base[reading], result)
        for reading, result in now.items()
        if result[0] < base[reading][0] or result[1] < base[reading][1]
    ]
    assert not lost, f"{len(lost)} readings lost a dispute or a compared number: {lost[:10]}"


def test_a_new_right_value_dispute_matches_its_unspaced_control() -> None:
    now = sweep.outcomes(COMPARE, sweep.RIGHT_LAYER, sweep.RIGHT_NUMBERS)
    control = {
        key: now[reading]
        for key, reading in sweep.readings(sweep.RIGHT_NUMBERS)
        if reading == key[2].format(sweep.cell(key[0], "", key[1], key[3]))
    }
    base = BASELINE["right"]["outcomes"]
    new = [
        (reading, base[reading], now[reading], control[key])
        for key, reading in sweep.readings(sweep.RIGHT_NUMBERS)
        if now[reading][1] > base[reading][1] and now[reading] != control[key]
    ]
    assert not new, f"{len(new)} new false disputes: {new[:10]}"
