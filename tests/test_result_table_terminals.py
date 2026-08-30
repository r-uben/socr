"""GH-353 TICKET-C1: the two ladder terminal enum members exist and round-trip.

Nothing sets ``TABLE_REJECTED`` / ``TABLE_UNVERIFIED`` yet -- this file only
guards the contract the later gate ticket (B1) will write into: the values
are stable strings, distinct from every existing ``FailureMode``, and
``PageOutput.to_dict`` / ``PageOutput.from_dict`` round-trip them losslessly.
"""

import pytest

from socr.core.result import FailureMode, PageOutput, PageStatus


def test_table_rejected_and_unverified_are_distinct_members() -> None:
    assert FailureMode.TABLE_REJECTED != FailureMode.TABLE_UNVERIFIED
    assert FailureMode.TABLE_REJECTED.value == "table_rejected"
    assert FailureMode.TABLE_UNVERIFIED.value == "table_unverified"


def test_table_terminal_values_are_unique_across_the_enum() -> None:
    values = [m.value for m in FailureMode]
    assert len(values) == len(set(values))


@pytest.mark.parametrize(
    "mode",
    [FailureMode.TABLE_REJECTED, FailureMode.TABLE_UNVERIFIED],
)
def test_page_output_round_trips_each_terminal(mode: FailureMode) -> None:
    original = PageOutput(
        page_num=3,
        text="| a | b |\n| --- | --- |\n| 1 | 2 |",
        status=PageStatus.SUCCESS,
        failure_mode=mode,
        audit_passed=False,
        audit_notes=["table judge ladder demoted this page"],
    )

    restored = PageOutput.from_dict(original.to_dict())

    assert restored.failure_mode == mode
    assert restored.failure_mode.value == mode.value
    assert restored == original


def test_from_dict_parses_raw_string_value_without_the_enum_member() -> None:
    """Sidecars on disk store the plain string; ``from_dict`` must accept it."""
    d = PageOutput(page_num=1, text="x").to_dict()
    d["failure_mode"] = "table_unverified"

    restored = PageOutput.from_dict(d)

    assert restored.failure_mode is FailureMode.TABLE_UNVERIFIED


def test_default_failure_mode_is_unaffected() -> None:
    """Flag-off / untouched pages keep NONE -- adding terminals must not shift it."""
    assert PageOutput(page_num=1).failure_mode == FailureMode.NONE
