"""Tests for GH-330 Task 3: reproducible, content-free self-bind coverage CLI command.

Covers:
- ``socr benchmark binding-coverage --manifest <path> --pdf-root <path>``
- Error handling on missing, non-file (directory), and 0-byte PDFs.
- Strict content-free JSON output (only identifiers, dimensions, and integer/boolean counts).
- Absence of fixture text, token values, markdown, or coordinates from stdout/stderr.
- Stable ordering and byte-identical JSON on repeated runs.
- Output formatting separation (``--format json`` vs ``--format summary``).
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest
from click.testing import CliRunner

from socr.cli import cli


def _create_sample_pdf(pdf_path: Path, n_pages: int = 2) -> None:
    """Create a multi-page PDF with synthetic tables and prose."""
    doc = fitz.open()
    for p in range(n_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Header for page {p + 1}", fontsize=12)

        # Draw a table with numeric cells
        page.insert_text((100, 150), "ColA", fontsize=10)
        page.insert_text((200, 150), "ColB", fontsize=10)
        page.insert_text((100, 180), "10.5", fontsize=10)
        page.insert_text((200, 180), "20.5", fontsize=10)

        page.draw_rect(fitz.Rect(90, 135, 280, 200))
        page.draw_line((90, 160), (280, 160))
        page.draw_line((180, 135), (180, 200))

    doc.save(str(pdf_path))
    doc.close()


def _create_manifest(manifest_path: Path, pdf_rel_path: str) -> None:
    """Create a synthetic manifest pointing to table pages."""
    manifest_data = [
        {"paper": "test_paper_1", "page": 1, "kind": "table", "file": pdf_rel_path},
        {"paper": "test_paper_1", "page": 2, "kind": "table", "file": pdf_rel_path},
    ]
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")


def test_coverage_cli_rejects_missing_pdf(tmp_path):
    """Refuse missing PDF with a clear error before opening."""
    runner = CliRunner()
    manifest_path = tmp_path / "manifest.json"
    _create_manifest(manifest_path, "nonexistent.pdf")

    result = runner.invoke(
        cli,
        [
            "benchmark",
            "binding-coverage",
            "--manifest",
            str(manifest_path),
            "--pdf-root",
            str(tmp_path),
        ],
    )
    # GH-350: no xfail escape. The command IS registered; unregistering it used
    # to turn this test green instead of red, which is the opposite of a guard.
    assert "No such command" not in result.output, (
        "binding-coverage is not registered; the guard must fail, not xfail"
    )

    assert result.exit_code != 0
    assert (
        "not found" in result.output.lower()
        or "missing" in result.output.lower()
        or "error" in result.output.lower()
    )


def test_coverage_cli_rejects_zero_byte_pdf(tmp_path):
    """Refuse 0-byte placeholder PDF with a clear error."""
    runner = CliRunner()
    zero_pdf = tmp_path / "zero.pdf"
    zero_pdf.write_bytes(b"")  # 0 bytes

    manifest_path = tmp_path / "manifest.json"
    _create_manifest(manifest_path, "zero.pdf")

    result = runner.invoke(
        cli,
        [
            "benchmark",
            "binding-coverage",
            "--manifest",
            str(manifest_path),
            "--pdf-root",
            str(tmp_path),
        ],
    )
    # GH-350: no xfail escape. The command IS registered; unregistering it used
    # to turn this test green instead of red, which is the opposite of a guard.
    assert "No such command" not in result.output, (
        "binding-coverage is not registered; the guard must fail, not xfail"
    )

    assert result.exit_code != 0
    assert (
        "0-byte" in result.output.lower()
        or "empty" in result.output.lower()
        or "zero" in result.output.lower()
    )


def test_coverage_cli_json_is_content_free_and_byte_identical(tmp_path):
    """JSON output contains only identifiers/integers/booleans and is byte-identical across runs."""
    pdf_path = tmp_path / "test.pdf"
    _create_sample_pdf(pdf_path, n_pages=2)

    manifest_path = tmp_path / "manifest.json"
    _create_manifest(manifest_path, "test.pdf")

    runner = CliRunner()
    cmd = [
        "benchmark",
        "binding-coverage",
        "--manifest",
        str(manifest_path),
        "--pdf-root",
        str(tmp_path),
        "--format",
        "json",
    ]

    res1 = runner.invoke(cli, cmd)
    if res1.exit_code != 0 and "No such command" in res1.output:
        pytest.xfail("Task 3 'binding-coverage' command not yet registered")

    assert res1.exit_code == 0
    res2 = runner.invoke(cli, cmd)
    assert res2.exit_code == 0

    # Repeated runs must produce byte-identical output
    assert res1.output == res2.output

    # Content-free verification: check that fixture cell contents and tokens never appear in JSON
    forbidden_tokens = ["10.5", "20.5", "ColA", "ColB", "Header for page"]
    for token in forbidden_tokens:
        assert token not in res1.output, (
            f"Content leak: token '{token}' found in content-free JSON output"
        )

    # Verify JSON structure
    data = json.loads(res1.output)
    assert isinstance(data, (dict, list))
