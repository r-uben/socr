from socr.core.manifest import _row_shape_reconciliation_ok
from socr.tables.row_corroboration import table_shaped_native_row_count


def w(x, y, text):
    return (x, y, x + len(text) * 6, y + 10, text, 0, 0, 0)


def test_numbered_label_rows_not_footnotes():
    words = []
    for i, (label, val) in enumerate([("Alpha", "80"), ("Beta", "90"), ("Gamma", "100")]):
        y = 20 + i * 20
        words += [w(10, y, str(i + 1) + ")"), w(35, y, label), w(130, y, val)]
    candidate = "| Item | Value |\n|---|---|\n| 1) Alpha | 80 |"
    assert table_shaped_native_row_count(words, 1) == 3
    assert not _row_shape_reconciliation_ok(words, candidate)


def test_one_row_second_table_not_erased_by_first_tables_lanes():
    words = []
    for i, (label, val) in enumerate([("Alpha", "80"), ("Beta", "90")]):
        words += [w(10, 20 + i * 20, label), w(100, 20 + i * 20, val)]
    words += [w(10, 200, "Gamma"), w(300, 200, "100")]
    candidate = "| Item | Value |\n|---|---|\n| Alpha | 80 |\n| Beta | 90 |"
    assert table_shaped_native_row_count(words, 1) == 3
    assert not _row_shape_reconciliation_ok(words, candidate)
