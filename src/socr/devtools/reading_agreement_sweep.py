"""Differential sweep for ``tables.reading_agreement`` number normalisation (#1237).

The comparator normalises numbers with a stack of order-dependent regexes. A change
that fixes one writing can quietly turn another from a dispute into a drop, and a
dropped wrong number is a false agreement. Hand-written rows did not catch that in
#1227: two of its three rounds introduced a false agreement that only a reviewer
found. This module holds a generated grid of percent writings and records what the
comparator does with each one, so a test can compare any later revision against the
recorded baseline.

The grid is the one used in #1227 round 3: numbers x gaps x wrappers x contexts, with
the ``\\%`` inside or outside the wrapper, read against a layer that prints a
DIFFERENT number (``wrong``) and against one that prints the SAME number (``right``).

Regenerate the checked-in baseline only when a change is MEANT to move results, and
review the diff::

    uv run socr-regenerate-reading-agreement-sweep

``--print-only`` reports what would change and does not write the fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

#: The fixture, relative to the repository root.
FIXTURE = Path("tests/fixtures/reading_agreement_sweep/baseline.json")

#: Every way #1227 and its reviews found a percent spaced from its number: nothing, a
#: blank, several blanks, a tie, LaTeX spaces, an escaped tie (an accent), a tab.
GAPS = ("", " ", "  ", "~", "\\ ", "\\,", "\\, ", " ~", "~~", "\\~", " \\~", "\t")
#: How a cell carries the number: bare, maths, an equation, brackets in and out of
#: maths, a stray dollar, a bracketed equation.
WRAPPERS = ("{}", "${}$", "$\\pi^* = {}$", "({})", "$({})$", "{}$", "$x = ({})$")
#: Where the cell sits: a pipe table row, prose, a LaTeX tabular.
CONTEXTS = (
    "| t | {} | 1.5 |",
    "T {} and 1.5",
    "\\begin{{tabular}}{{cc}} t & {} & 1.5 \\end{{tabular}}",
)
#: ``\\%`` written inside the wrapper (``$9 \\%$``) or after it (``$9$ \\%``).
PLACES = ("in", "out")

#: The layer prints 2; each reading ships a number that is NOT 2. A drop here is a
#: false agreement.
WRONG_LAYER = "T 2 % 1.5"
WRONG_NUMBERS = ("9", "(9)", "-9", ".9", "9.5", "x_9", "A-9", "1995")
#: The layer prints 2 and (2); each reading ships a writing of the same value. A new
#: dispute here is a false dispute unless the unspaced writing disputes too.
RIGHT_LAYER = "T 2 % (2) 1.5"
RIGHT_NUMBERS = ("2", "(2)", "2.0", "x_2", "A-2")


def cell(number: str, gap: str, wrapper: str, place: str) -> str:
    """One cell: ``number`` and its ``\\%`` separated by ``gap``, ``place`` saying
    whether the percent is inside the wrapper or after it."""
    if place == "in":
        return wrapper.format(number + gap + "\\%")
    return wrapper.format(number) + gap + "\\%"


def readings(numbers: tuple[str, ...]) -> Iterator[tuple[tuple[str, str, str, str], str]]:
    """``((number, wrapper, context, place), reading)`` for every gap, in grid order.
    The key leaves the gap out, so a reading's unspaced control shares its key."""
    for number, gap, wrapper, context, place in itertools.product(
        numbers, GAPS, WRAPPERS, CONTEXTS, PLACES
    ):
        yield (number, wrapper, context, place), context.format(cell(number, gap, wrapper, place))


def outcomes(compare: Callable, layer: str, numbers: tuple[str, ...]) -> dict[str, list[int]]:
    """``reading -> [compared, disputed]``. ``compare`` is ``compare_numbers`` with the
    own-characters provenance bound; it is passed in because this module may not import
    socr (``tests/test_package_layering.py``)."""
    out: dict[str, list[int]] = {}
    for _key, reading in readings(numbers):
        record = compare(layer, reading)
        out[reading] = [record.compared, len(record.disputed)]
    return out


def capture(compare: Callable) -> dict:
    """The fixture's content for ``compare``, without the revision stamp."""
    return {
        "wrong": {"layer": WRONG_LAYER, "outcomes": outcomes(compare, WRONG_LAYER, WRONG_NUMBERS)},
        "right": {"layer": RIGHT_LAYER, "outcomes": outcomes(compare, RIGHT_LAYER, RIGHT_NUMBERS)},
    }


#: Run in a child interpreter on ``<root>/src``, so the baseline describes THAT tree and
#: never the running package. Not seen by ``ast`` as an import.
_RUNNER = """
import functools, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import socr
assert Path(socr.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()), socr.__file__
from socr.devtools import reading_agreement_sweep as sweep
from socr.tables.reading_agreement import PROVENANCE_OWN_CHARACTERS, compare_numbers
compare = functools.partial(compare_numbers, provenance=PROVENANCE_OWN_CHARACTERS)
print(json.dumps(sweep.capture(compare)))
"""


def capture_from(src: Path) -> dict:
    """:func:`capture` for the socr package under ``src``, in an isolated child."""
    done = subprocess.run(
        [sys.executable, "-I", "-c", _RUNNER, str(src)],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode:
        raise SystemExit(f"capture failed:\n{done.stderr}")
    return json.loads(done.stdout)


def _revision(root: Path) -> str:
    """The commit, plus a digest of the comparator's own source: the file the baseline
    describes, which a dirty tree elsewhere does not change."""
    comparator = root / "src" / "socr" / "tables" / "reading_agreement.py"
    digest = hashlib.sha256(comparator.read_bytes()).hexdigest()[:12]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return f"{commit} reading_agreement.py sha256:{digest}"


def dumps(content: dict) -> str:
    """The fixture as text, one reading per line so a regenerated baseline diffs
    reading by reading."""
    lines = ["{", f'"revision": {json.dumps(content["revision"])},']
    for i, side in enumerate(("right", "wrong")):
        lines.append(f'"{side}": {{"layer": {json.dumps(content[side]["layer"])}, "outcomes": {{')
        items = sorted(content[side]["outcomes"].items())
        for j, (reading, result) in enumerate(items):
            comma = "," if j < len(items) - 1 else ""
            lines.append(f"{json.dumps(reading, ensure_ascii=False)}: {json.dumps(result)}{comma}")
        lines.append("}}" + ("," if i == 0 else ""))
    lines.append("}")
    return "\n".join(lines) + "\n"


def _repo_root() -> Path:
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / "pyproject.toml").is_file() and (parent / "tests").is_dir():
            return parent
    raise SystemExit("run from inside a socr checkout (no pyproject.toml + tests/ found)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--print-only", action="store_true", help="report changes, write nothing")
    args = parser.parse_args(argv)

    root = _repo_root()
    new = {"revision": _revision(root), **capture_from(root / "src")}
    path = root / FIXTURE
    old = json.loads(path.read_text()) if path.is_file() else {}
    for side in ("wrong", "right"):
        before = old.get(side, {}).get("outcomes", {})
        after = new[side]["outcomes"]
        moved = sum(1 for reading, result in after.items() if before.get(reading) != result)
        print(f"{side}: {len(after)} readings, {moved} differ from the checked-in baseline")
    if args.print_only:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(new))
    print(f"wrote {path} at {new['revision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
