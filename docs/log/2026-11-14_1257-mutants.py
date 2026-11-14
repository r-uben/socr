"""#1257 mutant check: every new evidence guard is removed in a COPY, its named test must FAIL.

CLAUDE.md guard rule.  Loaded: ``src``, ``tests`` and ``pyproject.toml`` are copied outside the
repo and ``socr.__file__`` is asserted to lie inside the copy.  Applied: each anchor is counted
UNCAPPED in the copied file and must occur exactly once, or the run aborts.  Load-bearing: the
named tests must fail on the mutant (and pass on the unmutated copy, run first).  Right suite: the
tests named are the ones that own the mutated function.

Run from the repo root::

    ~/venvs/socr/bin/python docs/log/2026-11-14_1257-mutants.py

Nothing is read from the command line.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = str(Path.home() / "venvs/socr/bin/python")
SRC_FILE = "src/socr/tables/row_binding_check.py"
EVIDENCE = "tests/test_gh1257_row_binding_evidence.py"
PIPELINE = "tests/test_gh1257_row_binding_pipeline.py"

SLOT = "                slot = sum(1 for e in edges if centre > e)  # 0 = label column\n"
FLAT_SLOT = (
    "                _t = w.text.translate(_UNICODE_MINUS)\n"
    "                slot = next((k + 1 for k, want in enumerate(wanted) if _t in want), 0)\n"
)

# (name, [(anchor, replacement)], [test selectors])
MUTANTS: list[tuple[str, list[tuple[str, str]], list[str]]] = [
    (
        "E1: drop the consecutive-baseline test (above and below)",
        [
            (
                "    while j >= 0 and free(lines[j]) and _consecutive(lines[j], nxt, bound):\n",
                "    while j >= 0 and free(lines[j]):\n",
            ),
            (
                "    while j < len(lines) and free(lines[j]) and _consecutive(prv, lines[j], bound):\n",
                "    while j < len(lines) and free(lines[j]):\n",
            ),
        ],
        [f"{EVIDENCE}::test_e1_adversary_a_fragment_that_is_not_consecutive_never_binds"],
    ),
    (
        "E1: drop 'no other row claims these fragments'",
        [
            (
                "            and all(w.x1 <= value_x0 for w in ln.words)\n"
                "            and not _claimed_elsewhere(ln, others)\n",
                "            and all(w.x1 <= value_x0 for w in ln.words)\n",
            )
        ],
        [
            f"{EVIDENCE}::test_e1_adversary_a_fragment_another_row_owns_never_binds",
            f"{EVIDENCE}::test_e1_adversary_a_shared_continuation_binds_neither_row",
        ],
    ),
    (
        "E1: allow a non-exact join (token equality instead of the strict gate)",
        [
            (
                '            if not _labels_exact(row.label, " ".join(parts), strict=True):\n',
                '            if not _same_label(_label_tokens(row.label), _label_tokens(" ".join(parts))):\n',
            )
        ],
        [f"{EVIDENCE}::test_e1_adversary_a_non_exact_join_never_binds"],
    ),
    (
        "E1: drop the left-edge test",
        [
            (
                "            if any(x < head_x0 - tol for x in starts):\n                continue\n",
                "",
            )
        ],
        [f"{EVIDENCE}::test_e1_adversary_a_continuation_left_of_the_label_never_binds"],
    ),
    (
        "E1: drop the label-column test (a fragment may reach into the value columns)",
        [("            and all(w.x1 <= value_x0 for w in ln.words)\n", "")],
        [f"{EVIDENCE}::test_e1_adversary_a_label_that_reaches_into_the_value_columns_never_binds"],
    ),
    (
        "E1: drop 'the line is not another row's line'",
        [
            (
                "            and ln.index not in anchored\n            and all(w.x1",
                "            and all(w.x1",
            )
        ],
        [f"{EVIDENCE}::test_e1_adversary_the_line_of_a_categorical_row_is_not_a_fragment"],
    ),
    (
        "E1: accept an ambiguous split (not exactly one)",
        [
            (
                "            hits.append(frozenset(ln.index for ln in (*top, *below[:b])))\n"
                "    return hits[0] if len(hits) == 1 else None",
                "            hits.append(frozenset(ln.index for ln in (*top, *below[:b])))\n"
                "    return hits[0] if hits else None",
            )
        ],
        [f"{EVIDENCE}::test_e1_adversary_two_splits_of_one_label_never_bind"],
    ),
    (
        "E1 group label: drop the consecutive-baseline test",
        [("                or (run and not _consecutive(run[-1], ln, bound))\n", "")],
        [f"{EVIDENCE}::test_e1_a_wrapped_group_label_whose_lines_are_a_section_apart_never_binds"],
    ),
    (
        "E3: drop the length equality (a prefix of the line passes)",
        [("        and len(shipped) == len(native)\n", "")],
        [f"{EVIDENCE}::test_e3_adversary_a_printed_word_the_row_left_out_never_binds"],
    ),
    (
        "E3/E1: drop the shared-line elimination",
        [
            (
                "            if frags and any(\n"
                "                frags & other[0] for k, other in found.items() if k != at and other[0]\n"
                "            ):\n"
                "                continue  # a shared line: nobody's\n",
                "",
            )
        ],
        [f"{EVIDENCE}::test_e3_adversary_two_identical_rows_for_one_printed_line_never_bind"],
    ),
    (
        "E2: drop the leftover-word test",
        [
            (
                "            len(got) == len(want) and all(_same_word(g, w) for g, w in zip(got, want))\n",
                "            all(_same_word(g, w) for g, w in zip(got, want))\n",
            )
        ],
        [f"{EVIDENCE}::test_e2_adversary_a_native_heading_word_left_over_never_binds"],
    ),
    (
        "E2: drop the consecutive-baseline test of the stack",
        [
            (
                "        if any(not _consecutive(a, b, bound) for a, b in zip(block, block[1:])):\n"
                "            break  # a longer block would only add a gap\n",
                "",
            )
        ],
        [f"{EVIDENCE}::test_e2_adversary_a_stack_that_is_not_consecutive_never_binds"],
    ),
    (
        "E2: membership by the flattened words instead of the column geometry",
        [(SLOT, FLAT_SLOT)],
        [
            f"{EVIDENCE}::test_e2_adversary_swapped_columns_never_bind",
            f"{EVIDENCE}::test_e2_adversary_words_moved_across_the_column_border_never_bind",
            f"{EVIDENCE}::test_e2_a_spanning_word_binds_only_in_the_column_its_centre_sits_over",
        ],
    ),
    (
        "E2: drop the shape test (heading cells = data columns)",
        [
            (
                "        if r.categories or len(r.numbers) != n_cols:\n            continue\n",
                "        if r.categories:\n            continue\n",
            )
        ],
        [f"{EVIDENCE}::test_e2_adversary_a_heading_with_fewer_cells_than_data_columns_never_binds"],
    ),
    (
        "completeness: bind a table that lost a row",
        [
            (
                "    if proposals and not _unclaimed_row_lines(\n"
                "        lines, anchored_lines | heading_lines, _value_columns(chosen)\n"
                "    ):\n",
                "    if proposals:\n",
            )
        ],
        [f"{EVIDENCE}::test_a_table_that_lost_a_row_keeps_its_mark_under_every_kind"],
    ),
    (
        "completeness: a running head with a date counts as a lost row (over-blocks)",
        [
            (
                "        if len(hit) >= MIN_VALUES_TO_CONVICT:\n            out.append(ln.index)\n",
                "        if hit or tail:\n            out.append(ln.index)\n",
            )
        ],
        [f"{EVIDENCE}::test_a_running_head_with_a_date_does_not_count_as_a_lost_row"],
    ),
    (
        "point repair: the repair gate uses the new evidence",
        [
            (
                "    base = check_row_binding(words, region, table_markdown, extra_evidence=False)\n",
                "    base = check_row_binding(words, region, table_markdown)\n",
            )
        ],
        [f"{EVIDENCE}::test_the_point_repair_gate_judges_by_the_same_line_rule"],
    ),
    (
        "record: no bound count and no evidence kind in the record",
        [
            (
                '        payload["rows_bound"] = result.bound_count()\n',
                "        pass\n",
            )
        ],
        [
            f"{EVIDENCE}::test_the_record_counts_bound_rows_and_names_the_evidence",
            f"{PIPELINE}::test_a_wrapped_label_table_loses_only_its_mark",
        ],
    ),
]


def run_pytest(copy: Path, selectors: list[str]) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": str(copy / "src"), "HOME": str(Path.home()), "PATH": "/usr/bin:/bin"}
    return subprocess.run(
        [PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *selectors],
        cwd=copy,
        env=env,
        capture_output=True,
        text=True,
    )


def fresh_copy(root: Path) -> Path:
    copy = Path(tempfile.mkdtemp(dir=root))
    for name in ("src", "tests"):
        shutil.copytree(REPO / name, copy / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(REPO / "pyproject.toml", copy / "pyproject.toml")
    probe = subprocess.run(
        [PY, "-c", "import socr; print(socr.__file__)"],
        cwd=copy,
        env={"PYTHONPATH": str(copy / "src"), "HOME": str(Path.home()), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    loaded = probe.stdout.strip()
    assert loaded.startswith(str(copy)), f"socr loads from {loaded}, not from the copy {copy}"
    return copy


def main() -> int:
    outside = Path(tempfile.mkdtemp(prefix="socr-1257-mutants-"))
    assert not str(outside).startswith(str(REPO))
    failures: list[str] = []
    try:
        base = fresh_copy(outside)
        every = sorted({sel for _, _, sels in MUTANTS for sel in sels})
        clean = run_pytest(base, every)
        print(f"unmutated copy: {'PASS' if clean.returncode == 0 else 'FAIL'}")
        if clean.returncode != 0:
            print(clean.stdout[-2000:])
            return 1
        for name, edits, selectors in MUTANTS:
            copy = fresh_copy(outside)
            target = copy / SRC_FILE
            text = target.read_text(encoding="utf-8")
            for anchor, replacement in edits:
                count = text.count(anchor)  # uncapped
                assert count == 1, f"{name}: anchor occurs {count} times: {anchor[:70]!r}"
                text = text.replace(anchor, replacement)
            target.write_text(text, encoding="utf-8")
            result = run_pytest(copy, selectors)
            killed = result.returncode != 0
            tail = [ln for ln in result.stdout.splitlines() if ln.startswith("FAILED")][:1]
            print(f"{'KILLED  ' if killed else 'SURVIVED'} {name}  {tail[0][:110] if tail else ''}")
            if not killed:
                failures.append(name)
            shutil.rmtree(copy, ignore_errors=True)
    finally:
        shutil.rmtree(outside, ignore_errors=True)
    print(f"{len(MUTANTS) - len(failures)}/{len(MUTANTS)} mutants killed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
