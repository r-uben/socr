"""Metadata tracking for incremental batch processing.

Ported from sibling CLIs (gemini-ocr, marker-ocr, etc.).
Tracks SHA256 checksums so unchanged files are skipped on re-run.
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

METADATA_VERSION = "1"
METADATA_FILENAME = "metadata.json"


def file_checksum(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


class MetadataManager:
    """Manages metadata.json for incremental batch processing."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.metadata_path = self.output_dir / METADATA_FILENAME
        self._data: dict[str, Any] = {"version": METADATA_VERSION, "files": {}}
        self._load()

    def _load(self) -> None:
        if self.metadata_path.exists():
            try:
                self._data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                logger.debug(f"Loaded metadata with {len(self._data.get('files', {}))} entries")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load metadata, starting fresh: {e}")
                self._data = {"version": METADATA_VERSION, "files": {}}

    def save(self) -> None:
        """Atomic write: tmp file then rename."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.metadata_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(str(tmp_path), str(self.metadata_path))

    def is_processed(self, file_path: Path) -> bool:
        """Check if file was already processed (matching name + checksum)."""
        key = file_path.name
        entry = self._data.get("files", {}).get(key)
        if entry is None:
            return False
        if entry.get("status") != "completed":
            return False
        current_checksum = file_checksum(file_path)
        return entry.get("checksum") == current_checksum

    def record(
        self,
        file_path: Path,
        *,
        engine: str,
        processing_time: float,
        pages: int = 0,
        output_path: str = "",
    ) -> None:
        """Record a processed file and save incrementally."""
        key = file_path.name
        self._data["files"][key] = {
            "status": "completed",
            "engine": engine,
            "processing_time": round(processing_time, 2),
            "pages": pages,
            "timestamp": datetime.now().isoformat(),
            "output_path": output_path,
            "checksum": file_checksum(file_path),
        }
        self.save()

    @property
    def files(self) -> dict[str, Any]:
        return self._data.get("files", {})
