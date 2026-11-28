"""Names of the files a document's ``pages/`` folder holds beside the per-page sidecars (#1251).

``pages/NNNNN.json`` is a page SIDECAR and ``pages/NNNNN.md`` its fragment. A judge's replies sit
in ``pages/NNNNN.judge-calls``: a JSON list, not a sidecar. Every reader that globs ``*.json`` for
sidecars goes through :func:`page_sidecar_paths`, so the companion is never read as one (it would
count as an unreadable sidecar, or as a page called ``00001.judge``).
"""

from __future__ import annotations

from pathlib import Path

JUDGE_CALLS_SUFFIX = ".judge-calls"
#: A judge-calls file that could not be read back, moved aside (kept) when new calls replace it.
JUDGE_CALLS_UNRESTORED_SUFFIX = ".unrestored"


def judge_calls_path(pages_dir: Path, page_num: int) -> Path:
    """The side file holding every page-judge call of page *page_num*."""
    return pages_dir / f"{page_num:05d}{JUDGE_CALLS_SUFFIX}"


def is_page_sidecar_name(name: str) -> bool:
    """``True`` for a ``*.json`` name that is a page sidecar and not a companion file."""
    return name.endswith(".json") and not name.endswith(JUDGE_CALLS_SUFFIX)


def page_sidecar_paths(pages_dir: Path, pattern: str = "*.json") -> list[Path]:
    """The page sidecars of *pages_dir* matching *pattern*, sorted, companions excluded."""
    return sorted(p for p in pages_dir.glob(pattern) if is_page_sidecar_name(p.name))


# Process-lifetime cache for the package source digest (matches orchestrator._socr_source_digest).
_SOURCE_DIGEST_CACHE: str | None = None


def page_blob_key(page_output_dict: dict) -> str:
    """Content-addressed ``page_fingerprint`` for a serialised PageOutput dict."""
    from socr.core.cache import blob_hash

    return "sha256:" + blob_hash(page_output_dict)


def socr_version() -> str:
    """socr's declared package version, read at call time."""
    import socr

    return str(getattr(socr, "__version__", ""))


def socr_source_digest() -> str:
    """SHA-256 over every shipped ``socr`` ``.py`` file, computed once per process.

    Matches ``socr.pipeline.orchestrator._socr_source_digest`` byte-for-byte so
    refinalized sidecars carry the identical code identity without cross-package
    private imports.
    """
    global _SOURCE_DIGEST_CACHE
    if _SOURCE_DIGEST_CACHE is not None:
        return _SOURCE_DIGEST_CACHE

    import hashlib

    import socr

    root = Path(socr.__file__).resolve().parent
    digest = hashlib.sha256()
    try:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(b"\x00")
            digest.update(path.read_bytes())
            digest.update(b"\x00")
        _SOURCE_DIGEST_CACHE = digest.hexdigest()
    except OSError:
        import uuid

        _SOURCE_DIGEST_CACHE = f"unreadable-source-tree:{uuid.uuid4().hex}"
    return _SOURCE_DIGEST_CACHE
