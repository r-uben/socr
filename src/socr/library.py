"""The papers library: processing driven by the library's own config (GH-964).

The library config (``~/papers/config.yaml``) is the single description of where
things live. Nothing here hardcodes a subfolder name: every path comes from the
config, is expanded and resolved against ``root``, and is validated on load.

Safety rules, each enforced in code rather than by convention:

* an existing text directory is never written into (``process_new``);
* promotion moves the old directory to the archive under a dated name and never
  deletes (``promote``);
* ``backup.rclone_remote`` is parsed and surfaced, never read from or written to.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from ocr_output_contract import FIGURES_DIRNAME, METADATA_FILENAME

DEFAULT_CONFIG_PATH = Path("~/papers/config.yaml")

#: Optional top-level config key naming the staging directory for ``--rerun``.
STAGING_KEY = "staging"
#: Used only when the config has no ``staging`` key: a directory of this name
#: inside ``index.dir``. Documented in the README; a config key overrides it.
DEFAULT_STAGING_NAME = "staging"

#: Page statuses that do not make a document unverified.
_CLEAN_PAGE_STATUSES = frozenset({"success", "skipped"})
#: The only document status that is trusted (ocr_output_contract.Status.COMPLETED).
_CLEAN_DOC_STATUS = "completed"
#: Filename pattern the pipeline writes; the config must agree with it.
_MARKDOWN_PATTERN = "{stem}.md"


class LibraryConfigError(ValueError):
    """The library config is missing a key, malformed, or unsafe. Never defaulted."""


class LibraryError(RuntimeError):
    """A refused library operation (nothing staged, already staged, ...)."""


@dataclass(frozen=True)
class LibraryConfig:
    root: Path
    pdf_dir: Path
    text_dir: Path
    markdown: str
    figures: str
    metadata: str
    index_dir: Path
    manifest: Path
    missing_text: Path
    documents: Path
    unverified: Path
    archive_dir: Path
    staging_dir: Path
    rclone_remote: str

    def text_doc_dir(self, stem: str) -> Path:
        return self.text_dir / stem

    def markdown_path(self, doc_dir: Path, stem: str) -> Path:
        return doc_dir / self.markdown.format(stem=stem)


def _require(data: Any, dotted: str) -> Any:
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node or node[part] in (None, ""):
            raise LibraryConfigError(f"library config: missing required key '{dotted}'")
        node = node[part]
    return node


def _require_str(data: Any, dotted: str) -> str:
    value = _require(data, dotted)
    if not isinstance(value, str):
        raise LibraryConfigError(f"library config: '{dotted}' must be a string, got {value!r}")
    return value


def _resolve(root: Path, raw: str, dotted: str) -> Path:
    """Expand ``~``, resolve relative to ``root``, refuse anything that escapes it."""
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = root / p
    p = Path(os.path.normpath(p))
    if not p.is_relative_to(root):
        raise LibraryConfigError(
            f"library config: '{dotted}' = {raw!r} resolves to {p}, outside root {root}"
        )
    # normpath is lexical; a symlink inside root could still lead out.
    if not p.resolve().is_relative_to(root.resolve()):
        raise LibraryConfigError(
            f"library config: '{dotted}' = {raw!r} resolves through a symlink outside root {root}"
        )
    return p


def _leaf(raw: str, dotted: str) -> str:
    """A bare file/dir name that must stay inside its parent."""
    if "/" in raw or "\\" in raw or raw in (".", "..") or Path(raw).is_absolute():
        raise LibraryConfigError(
            f"library config: '{dotted}' = {raw!r} must be a bare name inside its directory"
        )
    return raw


def load_library_config(path: Path | str = DEFAULT_CONFIG_PATH) -> LibraryConfig:
    cfg_path = Path(path).expanduser()
    try:
        text = cfg_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise LibraryConfigError(f"library config not found: {cfg_path}") from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise LibraryConfigError(f"library config {cfg_path}: invalid YAML: {e}") from None
    if not isinstance(data, dict):
        raise LibraryConfigError(f"library config {cfg_path}: top level must be a mapping")

    root = Path(os.path.normpath(Path(_require_str(data, "root")).expanduser()))
    if not root.is_absolute():
        raise LibraryConfigError(f"library config: 'root' must be absolute, got {str(root)!r}")

    pdf_dir = _resolve(root, _require_str(data, "input.pdf"), "input.pdf")
    text_dir = _resolve(root, _require_str(data, "output.text"), "output.text")
    markdown = _leaf(_require_str(data, "output.document.markdown"), "output.document.markdown")
    figures = _leaf(_require_str(data, "output.document.figures"), "output.document.figures")
    metadata = _leaf(_require_str(data, "output.document.metadata"), "output.document.metadata")
    # The pipeline writes fixed document names. A config that disagrees would be
    # silently ignored by the writer, so refuse it instead.
    for key, got, want in (
        ("output.document.markdown", markdown, _MARKDOWN_PATTERN),
        ("output.document.figures", figures, FIGURES_DIRNAME),
        ("output.document.metadata", metadata, METADATA_FILENAME),
    ):
        if got != want:
            raise LibraryConfigError(
                f"library config: '{key}' = {got!r}, but the OCR pipeline writes {want!r}; "
                "the layouts cannot be reconciled"
            )

    index_dir = _resolve(root, _require_str(data, "index.dir"), "index.dir")

    def index_file(name: str) -> Path:
        return index_dir / _leaf(_require_str(data, f"index.{name}"), f"index.{name}")

    manifest = index_file("manifest")
    missing_text = index_file("missing_text")
    documents = index_file("documents")
    unverified = index_file("unverified")
    archive_dir = _resolve(root, _require_str(data, "archive.dir"), "archive.dir")
    if STAGING_KEY in data:
        staging_dir = _resolve(root, _require_str(data, STAGING_KEY), STAGING_KEY)
    else:
        staging_dir = index_dir / DEFAULT_STAGING_NAME
    remote = _require_str(data, "backup.rclone_remote")

    dirs = {
        "input.pdf": pdf_dir,
        "output.text": text_dir,
        "index.dir": index_dir,
        "archive.dir": archive_dir,
        "staging": staging_dir,
    }
    names = list(dirs)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if dirs[a] == dirs[b]:
                raise LibraryConfigError(
                    f"library config: '{a}' and '{b}' resolve to the same directory {dirs[a]}"
                )
    for inner in ("staging",):
        for outer in ("input.pdf", "output.text", "archive.dir"):
            if dirs[inner].is_relative_to(dirs[outer]):
                raise LibraryConfigError(
                    f"library config: '{inner}' ({dirs[inner]}) lies inside '{outer}' "
                    f"({dirs[outer]})"
                )

    return LibraryConfig(
        root=root,
        pdf_dir=pdf_dir,
        text_dir=text_dir,
        markdown=markdown,
        figures=figures,
        metadata=metadata,
        index_dir=index_dir,
        manifest=manifest,
        missing_text=missing_text,
        documents=documents,
        unverified=unverified,
        archive_dir=archive_dir,
        staging_dir=staging_dir,
        rclone_remote=remote,
    )


# --- scanning -----------------------------------------------------------


def list_pdfs(cfg: LibraryConfig) -> list[Path]:
    """Top-level PDFs under ``input.pdf``, sorted. The text layout is flat by stem."""
    if not cfg.pdf_dir.is_dir():
        return []
    return sorted(
        (p for p in cfg.pdf_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: p.name,
    )


def has_text(cfg: LibraryConfig, stem: str) -> bool:
    return cfg.markdown_path(cfg.text_doc_dir(stem), stem).is_file()


def pending_pdfs(cfg: LibraryConfig) -> tuple[list[Path], list[Path]]:
    """Return (to_process, blocked).

    ``blocked`` are PDFs whose text directory exists but has no markdown (an
    interrupted or hand-edited directory). They are NOT processed: writing there
    would overwrite an existing text directory. ``--rerun`` handles them.
    """
    todo: list[Path] = []
    blocked: list[Path] = []
    for pdf in list_pdfs(cfg):
        if has_text(cfg, pdf.stem):
            continue
        (blocked if cfg.text_doc_dir(pdf.stem).exists() else todo).append(pdf)
    return todo, blocked


# --- status -------------------------------------------------------------


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def doc_status(cfg: LibraryConfig, doc_dir: Path) -> dict[str, Any]:
    """Status of one processed document directory, read from what the pipeline wrote."""
    meta = _read_json(doc_dir / cfg.metadata)
    status = meta.get("status") if isinstance(meta, dict) else None
    bad_pages: list[str] = []
    pages_dir = doc_dir / "pages"
    if pages_dir.is_dir():
        for sidecar in sorted(pages_dir.glob("*.json")):
            rec = _read_json(sidecar)
            page_status = rec.get("status") if isinstance(rec, dict) else None
            if page_status not in _CLEAN_PAGE_STATUSES:
                bad_pages.append(sidecar.stem)
    verified = status == _CLEAN_DOC_STATUS and not bad_pages
    return {
        "status": status if isinstance(status, str) else "unknown",
        "verified": verified,
        "bad_pages": bad_pages,
    }


def staged_stems(cfg: LibraryConfig) -> list[str]:
    if not cfg.staging_dir.is_dir():
        return []
    return sorted(
        d.name
        for d in cfg.staging_dir.iterdir()
        if d.is_dir() and cfg.markdown_path(d, d.name).is_file()
    )


# --- index --------------------------------------------------------------


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _lines(items: list[str]) -> str:
    return "".join(f"{i}\n" for i in items)


def refresh_index(cfg: LibraryConfig) -> dict[str, Any]:
    pdfs = list_pdfs(cfg)
    stems = sorted({p.stem for p in pdfs})
    awaiting = set(staged_stems(cfg))
    manifest: dict[str, Any] = {}
    missing: list[str] = []
    unverified: list[str] = []
    for stem in stems:
        if not has_text(cfg, stem):
            missing.append(stem)
            manifest[stem] = {"status": "missing_text", "awaiting_approval": stem in awaiting}
            continue
        info = doc_status(cfg, cfg.text_doc_dir(stem))
        if not info["verified"]:
            unverified.append(stem)
        manifest[stem] = {
            "status": info["status"],
            "verified": info["verified"],
            "bad_pages": info["bad_pages"],
            "awaiting_approval": stem in awaiting,
        }
    _atomic_write(cfg.documents, _lines([str(p.resolve()) for p in pdfs]))
    _atomic_write(cfg.missing_text, _lines(missing))
    _atomic_write(cfg.unverified, _lines(unverified))
    _atomic_write(
        cfg.manifest,
        json.dumps({"documents": manifest}, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    return {"documents": len(pdfs), "missing_text": missing, "unverified": unverified}


# --- processing ---------------------------------------------------------

#: ``process(pdf, out_root)``: runs one PDF through the pipeline into ``out_root``,
#: producing ``out_root/<stem>/<stem>.md``. The CLI passes the same
#: ``UnifiedPipeline.process`` that ``socr batch`` uses.
ProcessFn = Callable[[Path, Path], Any]


def process_new(cfg: LibraryConfig, process: ProcessFn, pdfs: list[Path]) -> list[tuple[str, Any]]:
    results = []
    for pdf in pdfs:
        doc_dir = cfg.text_doc_dir(pdf.stem)
        # NEVER-OVERWRITE guard: re-checked immediately before the write, not only
        # when the work list was built.
        if doc_dir.exists():
            raise LibraryError(f"refusing to write into existing text directory {doc_dir}")
        results.append((pdf.stem, process(pdf, cfg.text_dir)))
    return results


def rerun(cfg: LibraryConfig, process: ProcessFn, stem: str) -> Any:
    pdf = next((p for p in list_pdfs(cfg) if p.stem == stem), None)
    if pdf is None:
        raise LibraryError(f"no PDF with stem {stem!r} under {cfg.pdf_dir}")
    staged = cfg.staging_dir / stem
    if staged.exists():
        raise LibraryError(
            f"{staged} already exists: promote it (--promote {stem}) or move it aside; "
            "a staged run is never overwritten"
        )
    return process(pdf, cfg.staging_dir)


def promote(cfg: LibraryConfig, stem: str, now: datetime | None = None) -> tuple[Path | None, Path]:
    """Archive the current text dir under a dated name, then install the staged one.

    Returns (archived_path_or_None, installed_path). Nothing is deleted: both
    steps are renames. If the install fails after the archive step, the old
    directory is renamed back.
    """
    staged = cfg.staging_dir / stem
    if not (staged.is_dir() and cfg.markdown_path(staged, stem).is_file()):
        raise LibraryError(f"nothing staged for {stem!r} under {cfg.staging_dir}")
    target = cfg.text_doc_dir(stem)
    archived: Path | None = None
    if target.exists():
        stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
        archived = cfg.archive_dir / f"{stem}.{stamp}"
        n = 1
        while archived.exists():
            n += 1
            archived = cfg.archive_dir / f"{stem}.{stamp}.{n}"
        cfg.archive_dir.mkdir(parents=True, exist_ok=True)
        os.rename(target, archived)
    try:
        cfg.text_dir.mkdir(parents=True, exist_ok=True)
        os.rename(staged, target)
    except OSError:
        if archived is not None:
            os.rename(archived, target)
        raise
    return archived, target
