from socr.core.manifest import _row_shape_reconciliation_ok
from socr.tables.row_corroboration import table_shaped_native_row_count


def word(x0, x1, y, text):
    return (x0, y, x1, y + 10, text, 0, 0, 0)


def md(rows):
    return "| Item | Value |\n|---|---|\n" + "\n".join(
        "| " + label + " | " + value + " |" for label, value in rows
    )


def test_two_row_left_aligned_control():
    words = [
        word(10, 40, 20, "Alpha"),
        word(100, 110, 20, "8"),
        word(10, 40, 40, "Beta"),
        word(100, 130, 40, "888"),
    ]
    assert table_shaped_native_row_count(words, 1) == 2
    assert not _row_shape_reconciliation_ok(words, md([("Alpha", "8")]))


def test_right_aligned_two_row_truncation():
    words = [
        word(10, 40, 20, "Alpha"),
        word(120, 130, 20, "8"),
        word(10, 40, 40, "Beta"),
        word(100, 130, 40, "888"),
    ]
    assert not _row_shape_reconciliation_ok(words, md([("Alpha", "8")]))


def test_aligned_numeric_footnotes():
    words = []
    for i, (label, a, b) in enumerate(
        [("Alpha", "50", "60"), ("Beta", "70", "80"), ("Gamma", "90", "100")]
    ):
        y = 20 + i * 20
        words += [word(10, 40, y, label), word(100, 120, y, a), word(160, 180, y, b)]
    for i in range(2):
        y = 200 + i * 20
        words += [
            word(10, 25, y, str(i + 1) + ")"),
            word(30, 80, y, "See pages"),
            word(90, 110, y, "45"),
            word(120, 150, y, "and"),
            word(160, 180, y, "12"),
        ]
    complete = (
        "| Item | A | B |\n|---|---|---|\n"
        "| Alpha | 50 | 60 |\n| Beta | 70 | 80 |\n| Gamma | 90 | 100 |"
    )
    assert _row_shape_reconciliation_ok(words, complete)
