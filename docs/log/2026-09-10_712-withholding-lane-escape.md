# #712 — the withholding lane ships literal characters too

`fix/712-withholding-lane-escape`, off `main@147cdbf`.

## What was wrong

`_escaped_native_line` landed in #695 wired into one caller. `_all_native_text`
— the no-numeral lane, which runs only where every band is prose — escaped its
lines. `native_prose_floor_text`, the lane that runs whenever the page HAS a
numeric band to withhold, appended its prose lines raw.

Both lanes make the same promise: these are the page's own characters, verbatim,
under a banner that verifies nothing. One of them was emitting active markdown.
Round 12's reproduction therefore still fired on every withholding-shaped scan —
a native `<!--` line hides the sentence beneath it, and `# ...` becomes an `<h1>`
— in every CommonMark consumer of the shipped `.md`, not only in the review
viewer that #652 rounds 13 to 16 went on to fix.

## The fix

The same helper over the same two character sets, called at flush rather than
where the line is collected. That ordering is the whole subtlety:
`table_syntax_line_indices` has to read the page's raw pipes to find where a
table begins, and an escaped `\|` is not a pipe to it. Escaping at flush leaves
that analysis on the raw bands and changes only what is written out.

The docstring on `_NATIVE_BLOCK_ACTIVE` needed correcting too. It argued that an
ordered-list marker needs no rule because "this path runs only where every band
is prose", which was true of the single caller and is not true of the second.
The argument still holds, on a better premise: only PROSE bands are escaped in
either lane, and a prose band carries fewer than `row_shape_min` numeral-bearing
tokens -- counted by `bears_printed_numeral`, which matches any ASCII digit --
which at the shipping value of 1 means none, so no escaped line can open with an
ASCII digit. A line opening with an Arabic-Indic digit can still be prose, but
CommonMark's ordered-list marker is ASCII-only, so it is not a list either way.

## Fed 1989-11-14 p3

Byte-identical: 1,738 characters, 1,750 UTF-8 bytes, before and after. The
page's prose bands carry no character that would have been escaped. Measured by running the real fixture and
its real cached nougat attempt through selection against `HEAD`'s manifest and
against this branch's, not by inspecting the page. The same harness shows the
synthetic active-character page changing, so it is not blind.

## Pins

A withholding-shaped page whose prose band is `<!--`, `# ...`, a leading `>`, a
bullet and a line of inline syntax, checked through markdown-it-py and through
the review viewer's own JavaScript under Node, then through finalization and
through resume. Plus the control that escaping changed the prose and not what
ships: the numeric band stays behind its marker.

Four #649 tests moved from asserting a raw pipe-bearing sentence to asserting its
escaped form. Their point is that the line is not WITHHELD, which the escaped
form shows exactly as well.

One trap worth recording: `tests/conftest.py` patches `shutil.which` on the
module object for every test, so a node probe written inside a test body answers
`None` on a machine that has node and skips in silence. `_NODE` is resolved at
import for that reason.
