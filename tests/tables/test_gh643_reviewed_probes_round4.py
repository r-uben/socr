from socr.core.manifest import _row_shape_reconciliation_ok
from socr.tables.row_corroboration import table_shaped_native_row_count
from socr.tables.structure_check import table_truncated


def w(x, y, text):
    return (x, y, x + len(text) * 6, y + 10, text, 0, 0, 0)


def words(n, labels=True):
    out = []
    for i in range(n):
        y = 20 + i * 20
        out += [w(10, y, str(i + 3) + ")")]
        if labels:
            out += [w(35, y, "Alpha")]
        out += [w(130, y, str(12 + i)), w(210, y, str(45 + i))]
    return out


def test_numbered_two_value_two_row_table_is_kept():
    assert table_shaped_native_row_count(words(2), 2) == 2


def test_numeric_marker_stub_truncation_not_excused():
    source = words(4, False)
    candidate = "| Item | A | B |\n|---|---|---|\n| 3) | 12 | 45 |"
    assert not _row_shape_reconciliation_ok(source, candidate)


def test_numeric_marker_stub_structure_check_detects_truncation():
    source = words(4, False)
    candidate = "| Item | A | B |\n|---|---|---|\n| 3) | 12 | 45 |"
    assert table_truncated(candidate, source)
