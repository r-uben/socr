"""Golden + mutant matrix for real-document native cell binding."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_golden_matrix_no_mutant_verifies() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "native_cell_golden_eval.py"),
            "--set",
            "extended",
            "--json",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    import json

    raw_out = proc.stdout
    start = raw_out.find("{")
    assert start >= 0, raw_out
    data = json.loads(raw_out[start:])
    for row in data["goldens"]:
        correct = row["correct"]["outcome"]
        assert correct in ("verified", "ambiguous"), (
            f"{row['page']}: correct transcription must not be CONTRADICTED, got {correct}"
        )
        for name, mut in row["mutants"].items():
            assert mut["outcome"] != "verified", f"{row['page']} mutant {name} must not VERIFY"
        if row["correct"]["outcome"] == "verified":
            for key in ("mut_model_digit", "mut_model_column_shift", "mut_model_minus_drop"):
                if key in row["mutants"]:
                    assert row["mutants"][key]["outcome"] != "verified", (
                        f"{row['page']} {key} must not VERIFY"
                    )
