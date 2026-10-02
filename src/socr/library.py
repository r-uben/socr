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

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
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
#: directly under ``root`` (a sibling of the other library dirs, never nested in
#: them). Documented in the README; a config key overrides it.
DEFAULT_STAGING_NAME = ".socr-staging"
#: Control files kept in ``index.dir``. Index filenames may not collide with them.
LOCK_NAME = ".library.lock"
JOURNAL_NAME = ".promote.journal.json"

#: Page statuses that make a document unverified. Anything else (including an
#: absent or unreadable status) is not evidence of a problem.
_BAD_PAGE_STATUSES = frozenset({"warning", "error"})
#: The document status that is trusted (ocr_output_contract.Status.COMPLETED).
_CLEAN_DOC_STATUS = "completed"
#: Hand-placed marker in a text dir (named in the library config's comments).
#: Read as evidence only; socr never writes or deletes it.
UNVERIFIED_MARKER = "UNVERIFIED.txt"

#: Three states. A legacy run has no ``status`` in its metadata: that is
#: UNKNOWN, never UNVERIFIED (the real library has 342 such documents).
VERIFIED, UNVERIFIED, UNKNOWN = "verified", "unverified", "unknown"
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
        staging_dir = _resolve(root, DEFAULT_STAGING_NAME, STAGING_KEY)
    remote = _require_str(data, "backup.rclone_remote")

    dirs = {
        "input.pdf": pdf_dir,
        "output.text": text_dir,
        "index.dir": index_dir,
        "archive.dir": archive_dir,
        "staging": staging_dir,
    }
    # Compare after resolve(): a symlink alias inside root is an overlap too.
    real = {k: v.resolve() for k, v in dirs.items()}
    names = list(dirs)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if real[a].is_relative_to(real[b]) or real[b].is_relative_to(real[a]):
                raise LibraryConfigError(
                    f"library config: '{a}' ({dirs[a]}) and '{b}' ({dirs[b]}) are the same "
                    "directory or nest inside each other"
                )
    files = {
        "index.manifest": manifest,
        "index.missing_text": missing_text,
        "index.documents": documents,
        "index.unverified": unverified,
        "(lock file)": index_dir / LOCK_NAME,
        "(promotion journal)": index_dir / JOURNAL_NAME,
    }
    seen: dict[str, str] = {}
    for key, f in files.items():
        folded = f.name.casefold()
        if folded in seen:
            raise LibraryConfigError(
                f"library config: '{key}' and '{seen[folded]}' resolve to the same file name "
                f"{f.name!r} (compared case-insensitively)"
            )
        seen[folded] = key

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
            if page_status in _BAD_PAGE_STATUSES:
                bad_pages.append(sidecar.stem)
    has_status = isinstance(status, str) and bool(status)
    marker = (doc_dir / UNVERIFIED_MARKER).exists()
    if marker or bad_pages or (has_status and status != _CLEAN_DOC_STATUS):
        state = UNVERIFIED
    elif has_status:
        state = VERIFIED
    else:
        state = UNKNOWN
    return {
        "status": status if has_status else UNKNOWN,
        "state": state,
        "marker": marker,
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
    """Unique temp file in the same dir, fsync, then rename over the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise LibraryError(f"refusing to write through a symlink: {path}")
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        if tmp.is_symlink() or path.is_symlink():
            raise LibraryError(f"refusing to write through a symlink: {path}")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _lines(items: list[str]) -> str:
    return "".join(f"{i}\n" for i in items)


def _read_entries(path: Path) -> set[str]:
    """Entries of the curated list. Absent is empty; unreadable ABORTS the refresh."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return set()
    except (OSError, UnicodeDecodeError) as e:
        raise LibraryError(
            f"cannot read the curated list {path} ({e}); index refresh aborted, nothing written"
        ) from None
    return {ln.strip() for ln in text.splitlines() if ln.strip()}


def refresh_index(cfg: LibraryConfig, processed: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Rewrite the index files.

    ``processed`` are stems whose text dir socr itself wrote in this run. The
    unverified list is curated by hand as well: it is the UNION of the existing
    file and the newly computed entries, and a stem leaves it only when it is in
    ``processed`` and came out ``verified``.
    """
    pdfs = list_pdfs(cfg)
    stems = sorted({p.stem for p in pdfs})
    awaiting = set(staged_stems(cfg))
    manifest: dict[str, Any] = {}
    missing: list[str] = []
    computed: set[str] = set()
    cleared: set[str] = set()
    for stem in stems:
        if not has_text(cfg, stem):
            missing.append(stem)
            manifest[stem] = {"status": "missing_text", "awaiting_approval": stem in awaiting}
            continue
        info = doc_status(cfg, cfg.text_doc_dir(stem))
        if info["state"] == UNVERIFIED:
            computed.add(stem)
        elif info["state"] == VERIFIED and stem in processed:
            cleared.add(stem)
        manifest[stem] = {
            "status": info["status"],
            "state": info["state"],
            "bad_pages": info["bad_pages"],
            "awaiting_approval": stem in awaiting,
        }
    unverified = sorted((_read_entries(cfg.unverified) - cleared) | computed)
    for stem in unverified:
        if stem in manifest and manifest[stem].get("state") != UNVERIFIED:
            manifest[stem]["curated_unverified"] = True
    _atomic_write(cfg.documents, _lines([str(p.resolve()) for p in pdfs]))
    _atomic_write(cfg.missing_text, _lines(missing))
    _atomic_write(cfg.unverified, _lines(unverified))
    _atomic_write(
        cfg.manifest,
        json.dumps({"documents": manifest}, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    return {"documents": len(pdfs), "missing_text": missing, "unverified": unverified}


# --- locking, preflight --------------------------------------------------


@contextmanager
def library_lock(cfg: LibraryConfig):
    """Exclusive lock for the whole run (flock: released by the OS if we crash)."""
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.index_dir / LOCK_NAME
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o644)
    except OSError as e:
        raise LibraryError(f"cannot open lock file {path}: {e}") from None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LibraryError(
                f"another 'socr library' run holds {path}; wait for it to finish"
            ) from None
        yield
    finally:
        os.close(fd)


def check_stem_collisions(cfg: LibraryConfig) -> None:
    """Refuse when two PDFs map to the same stem, case-insensitively."""
    by_stem: dict[str, list[str]] = {}
    for pdf in list_pdfs(cfg):
        by_stem.setdefault(pdf.stem.casefold(), []).append(pdf.name)
    clashes = sorted(sorted(v) for v in by_stem.values() if len(v) > 1)
    if clashes:
        listing = "; ".join(" / ".join(c) for c in clashes)
        raise LibraryError(
            f"PDFs collide on the same stem (compared case-insensitively): {listing}. "
            "Rename one; nothing was processed."
        )


# --- processing ---------------------------------------------------------

#: ``process(pdf, out_root)``: runs one PDF through the pipeline into ``out_root``,
#: producing ``out_root/<stem>/<stem>.md``. The CLI passes the same
#: ``UnifiedPipeline.process`` that ``socr batch`` uses.
ProcessFn = Callable[[Path, Path], Any]

INSTALLED, FAILED, BLOCKED = "installed", "failed", "blocked"


def _rename_noreplace(src: Path, dst: Path) -> None:
    """Rename a directory, refusing to touch an existing target.

    Correct only under ``library_lock``: POSIX rename would silently replace an
    empty directory, and the check-then-rename gap is closed by the lock.
    """
    if os.path.lexists(dst):
        raise LibraryError(f"refusing to replace existing {dst}")
    os.rename(src, dst)


def install_staged(cfg: LibraryConfig, stem: str) -> Path:
    """Move a finished staged document into text/. Never replaces an existing dir."""
    src = cfg.staging_dir / stem
    dst = cfg.text_doc_dir(stem)
    if not cfg.markdown_path(src, stem).is_file():
        raise LibraryError(f"{src} has no {cfg.markdown.format(stem=stem)}; not installed")
    cfg.text_dir.mkdir(parents=True, exist_ok=True)
    _rename_noreplace(src, dst)
    return dst


def process_new(
    cfg: LibraryConfig, process: ProcessFn, pdfs: list[Path]
) -> list[tuple[str, str, Any]]:
    """Process each new PDF into staging, then move it into text/ if it produced markdown.

    Returns (stem, state, detail) with state INSTALLED / FAILED / BLOCKED. A crash or
    exception in one paper leaves its leftovers in staging and never touches text/.
    A run that finishes with status partial/failed but produced markdown IS
    installed (best available text); the index lists it as unverified.
    """
    out: list[tuple[str, str, Any]] = []
    for pdf in pdfs:
        stem = pdf.stem
        # NEVER-OVERWRITE guard, re-checked immediately before processing.
        if cfg.text_doc_dir(stem).exists():
            raise LibraryError(
                f"refusing to write into existing text directory {cfg.text_doc_dir(stem)}"
            )
        if (cfg.staging_dir / stem).exists():
            out.append((stem, BLOCKED, f"leftovers in {cfg.staging_dir / stem}; inspect them"))
            continue
        try:
            result = process(pdf, cfg.staging_dir)
        except Exception as e:  # leftovers stay in staging
            out.append((stem, FAILED, f"{type(e).__name__}: {e}"))
            continue
        try:
            install_staged(cfg, stem)
        except LibraryError as e:
            out.append((stem, FAILED, str(e)))
            continue
        out.append((stem, INSTALLED, result))
    return out


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


# --- promotion, with a crash journal ---------------------------------------


def _journal_path(cfg: LibraryConfig) -> Path:
    return cfg.index_dir / JOURNAL_NAME


def recover_promotion(cfg: LibraryConfig) -> str | None:
    """Finish an interrupted promotion before anything else. Call under the lock.

    The journal is written before the two renames and removed after. Recovery
    rolls FORWARD so the live text dir is never left absent. Returns a message
    when it acted, None when there was no journal.
    """
    jp = _journal_path(cfg)
    if not os.path.lexists(jp):
        return None
    try:
        j = json.loads(jp.read_text(encoding="utf-8"))
        target, staged = Path(j["target"]), Path(j["staged"])
        archived = Path(j["archived"]) if j.get("archived") else None
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise LibraryError(f"unreadable promotion journal {jp} ({e}); resolve it by hand") from None
    t, s = os.path.lexists(target), os.path.lexists(staged)
    a = archived is not None and os.path.lexists(archived)
    if s and not t and (a or archived is None):
        os.rename(staged, target)  # the interrupted step: finish it
        msg = f"recovered interrupted promotion of {j.get('stem')}: installed {target}"
    elif s and t and a is False and archived is not None:
        msg = f"discarded journal of a promotion that had not started ({j.get('stem')})"
    elif t and not s:
        msg = f"promotion of {j.get('stem')} had completed; cleared its journal"
    else:
        raise LibraryError(
            f"promotion journal {jp} does not match the disk (target={t}, staged={s}, "
            f"archived={a}); resolve it by hand"
        )
    jp.unlink()
    return msg


def promote(cfg: LibraryConfig, stem: str, now: datetime | None = None) -> tuple[Path | None, Path]:
    """Archive the current text dir under a dated name, then install the staged one.

    Call under ``library_lock`` after ``recover_promotion``. Both steps are renames
    (nothing is deleted) bracketed by a journal so a crash between them is
    finished by the next run instead of leaving the text dir absent.
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
        while os.path.lexists(archived):
            n += 1
            archived = cfg.archive_dir / f"{stem}.{stamp}.{n}"
        cfg.archive_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        _journal_path(cfg),
        json.dumps(
            {
                "stem": stem,
                "target": str(target),
                "staged": str(staged),
                "archived": str(archived) if archived else None,
            }
        ),
    )
    if archived is not None:
        _rename_noreplace(target, archived)
    cfg.text_dir.mkdir(parents=True, exist_ok=True)
    _rename_noreplace(staged, target)
    _journal_path(cfg).unlink()
    return archived, target
