"""Names of the files a document's ``pages/`` folder holds beside the per-page sidecars (#1251).

``pages/NNNNN.json`` is a page SIDECAR and ``pages/NNNNN.md`` its fragment. A judge's replies sit
in ``pages/NNNNN.judge-calls``: a JSON list, not a sidecar. Every reader that globs ``*.json`` for
sidecars goes through :func:`page_sidecar_paths`, so the companion is never read as one (it would
count as an unreadable sidecar, or as a page called ``00001.judge``).
"""

from __future__ import annotations

from pathlib import Path

JUDGE_CALLS_SUFFIX = ".judge-calls"


def judge_calls_path(pages_dir: Path, page_num: int) -> Path:
    """The side file holding every page-judge call of page *page_num*."""
    return pages_dir / f"{page_num:05d}{JUDGE_CALLS_SUFFIX}"


def is_page_sidecar_name(name: str) -> bool:
    """``True`` for a ``*.json`` name that is a page sidecar and not a companion file."""
    return name.endswith(".json") and not name.endswith(JUDGE_CALLS_SUFFIX)


def page_sidecar_paths(pages_dir: Path, pattern: str = "*.json") -> list[Path]:
    """The page sidecars of *pages_dir* matching *pattern*, sorted, companions excluded."""
    return sorted(p for p in pages_dir.glob(pattern) if is_page_sidecar_name(p.name))
