"""Content-addressed blob store for OCR outputs.

The reproducibility mechanism for socr is NOT "re-run the engines and hope for
the same bytes" — VLM/LLM OCR is non-deterministic even at temperature 0, and
provider models drift silently. Instead, the winning ``PageOutput`` for a page
is frozen as a JSON blob, addressed by the SHA-256 of its own canonical bytes,
and stored here. ``socr replay`` reconstructs a document by fetching these blobs
— it never invokes an engine.

This module is deliberately dumb: it stores and retrieves bytes by their hash.
It knows nothing about pages, engines, or pipelines. The mapping from a page to
its blob lives in ``manifest.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from socr.core.result import PageOutput


def _canonical_bytes(obj: dict) -> bytes:
    """Deterministic JSON encoding so identical content yields an identical hash.

    ``sort_keys`` + compact separators + UTF-8 makes the byte stream a pure
    function of the dict's content, which is what content-addressing requires.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def blob_hash(obj: dict) -> str:
    """Content hash of a JSON-serializable object."""
    return hashlib.sha256(_canonical_bytes(obj)).hexdigest()


class BlobStore:
    """Filesystem content-addressed store. Keys are SHA-256 hex of the content.

    Layout: ``<root>/ab/abcdef...json`` (sharded by first 2 hex chars to avoid
    huge flat directories). Writes are atomic (temp file + rename) so a crash
    mid-write never leaves a half-written, wrong-hash blob.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _path_for(self, h: str) -> Path:
        return self.root / h[:2] / f"{h}.json"

    def put(self, obj: dict) -> str:
        """Store a JSON-serializable object; return its content hash.

        Idempotent: storing identical content twice is a no-op and returns the
        same hash.
        """
        h = blob_hash(obj)
        path = self._path_for(h)
        if path.exists():
            return h
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _canonical_bytes(obj)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return h

    def has(self, h: str) -> bool:
        return self._path_for(h).exists()

    def get(self, h: str) -> dict:
        """Fetch a blob by hash. Raises KeyError if absent (a broken manifest)."""
        path = self._path_for(h)
        if not path.exists():
            raise KeyError(f"blob {h} not in cache at {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    # --- PageOutput convenience -------------------------------------------

    def put_page(self, page: PageOutput) -> str:
        return self.put(page.to_dict())

    def get_page(self, h: str) -> PageOutput:
        return PageOutput.from_dict(self.get(h))
