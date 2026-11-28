"""Offline re-finalization tools for finished socr runs.

Shared atomic I/O and table-count helpers live here so every ``socr.refinalize.*``
tool uses the same writers and recount logic instead of duplicating them.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["read_json", "recount_tables", "write_json", "write_text"]


def read_json(path: Path):
    """Read and parse a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    """Write *text* atomically as UTF-8 (temp file beside *path* + replace)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json(path: Path, data) -> None:
    """Write *data* atomically in socr's JSON format.

    Two-space indent, UTF-8 (``ensure_ascii=False``), and a trailing newline
    iff the file being replaced already ended with one (``manifest.json`` does).
    """
    newline = "\n" if path.is_file() and path.read_bytes().endswith(b"\n") else ""
    write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + newline)


def recount_tables(
    doc_dir: Path,
    meta_path: Path,
    meta: dict,
    *,
    sidecar_overrides: dict[int, dict] | None = None,
) -> None:
    """Recompute ``metadata.json``'s ``tables`` block from the sidecars as they now are,
    in both the document record and the run's root index."""
    from socr.core.table_counts import count_from_sidecars

    if "tables" not in meta:
        return
    counts = count_from_sidecars(doc_dir, sidecar_overrides=sidecar_overrides)
    if counts is None:
        return
    tables = counts.to_dict()
    if meta.get("tables") != tables:
        meta["tables"] = tables
        write_json(meta_path, meta)
    root_path = doc_dir.parent / "metadata.json"
    if root_path.is_file():
        root = read_json(root_path)
        entry = (root.get("files") or {}).get(meta.get("key"))
        if isinstance(entry, dict) and "tables" in entry and entry["tables"] != tables:
            entry["tables"] = tables
            write_json(root_path, root)
