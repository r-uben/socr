"""Re-audit finished runs for model preamble loops and fix or flag affected pages.

    socr-refinalize-boilerplate <out_dir> [--apply]

Pages whose ``winning_output.text`` is only repeated OCR preamble (no document
content after deterministic stripping) are marked ``audit_passed=false`` so a
resume re-OCRs them. Pages that merely leak a leading preamble before real text
get the prefix stripped in ``pages/NNN.md``, the document ``.md``, and the
sidecar — no model call.

Dry-run is the default: prints counts only. Pass ``--apply`` to write changes.

Resume after ``re_ocr`` demotions uses two gates:

* **Per page:** ``_load_terminal_page`` skips only when ``winning_output.audit_passed``
  is exactly ``True`` (``terminal=True`` alone is not enough).
* **Per document:** ``RootIndex.is_completed`` skips a finished doc unless its assembled
  ``*.md`` (the document stem beside ``pages/``, not ``pages/NNN.md``) is missing. On
  ``--apply``, when any page is demoted for ``re_ocr``, those assembled ``.md`` files
  are removed and ``metadata.json`` status is lowered from ``completed`` to ``partial``
  when present, matching the partial-run harness used in resume tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

LOG_NAME = "refinalized_boilerplate.json"


@dataclass(frozen=True)
class PageChange:
    doc: str
    page_num: int
    action: str  # "re_ocr" | "strip_preamble"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_json(path: Path, data) -> None:
    newline = "\n" if path.is_file() and path.read_bytes().endswith(b"\n") else ""
    _write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + newline)


def _invalidate_doc_resume_gate(doc_dir: Path) -> None:
    """Drop assembled document markdown and mark the doc partial so ``process()`` re-enters."""
    from ocr_output_contract import Status

    for md_path in doc_dir.glob("*.md"):
        md_path.unlink()
    meta_path = doc_dir / "metadata.json"
    if not meta_path.is_file():
        return
    meta = _read_json(meta_path)

    def _demote(record: dict) -> None:
        if record.get("status") == Status.COMPLETED.value:
            record["status"] = Status.PARTIAL.value

    _demote(meta)
    _write_json(meta_path, meta)
    root_path = doc_dir.parent / "metadata.json"
    if root_path.is_file():
        root = _read_json(root_path)
        entry = (root.get("files") or {}).get(meta.get("key"))
        if isinstance(entry, dict):
            _demote(entry)
            _write_json(root_path, root)


def _swap_once(doc_md: str, old: str, new: str, md_path: Path) -> str:
    count = doc_md.count(old)
    if count != 1:
        raise RuntimeError(f"{md_path}: expected exactly one occurrence, found {count}")
    return doc_md.replace(old, new, 1)


def classify_winning_text(text: str) -> str | None:
    """``re_ocr``, ``strip_preamble``, or ``None`` if the page needs no change."""
    from socr.audit.boilerplate import (
        is_boilerplate_only_output,
        strip_leading_ocr_boilerplate,
    )

    if not text or not text.strip():
        return None
    if is_boilerplate_only_output(text):
        return "re_ocr"
    stripped = strip_leading_ocr_boilerplate(text)
    if stripped != text.strip() and stripped.strip():
        return "strip_preamble"
    return None


def refinalize_page(sidecar: dict) -> tuple[str, str] | None:
    """``(action, new_text)`` when the sidecar should change, else ``None``."""
    if sidecar.get("terminal") is not True:
        return None
    winning = sidecar.get("winning_output") or {}
    text = winning.get("text") or ""
    action = classify_winning_text(text)
    if action is None:
        return None
    if action == "re_ocr":
        return action, text
    from socr.audit.boilerplate import strip_leading_ocr_boilerplate

    return action, strip_leading_ocr_boilerplate(text)


def refinalize_document(doc_dir: Path, *, apply: bool = False) -> list[PageChange]:
    from socr.core.verdict import model_reading_engines

    pages_dir = doc_dir / "pages"
    if not pages_dir.is_dir():
        return []
    md_paths = sorted(doc_dir.glob("*.md"))
    doc_md = md_paths[0].read_text(encoding="utf-8") if len(md_paths) == 1 else ""
    changes: list[PageChange] = []
    strip_writes: list[tuple[Path, Path, dict, str]] = []
    reocr_writes: list[tuple[Path, dict]] = []
    for path in sorted(pages_dir.glob("[0-9]*.json")):
        sidecar = _read_json(path)
        engine = (sidecar.get("winning_output") or {}).get("engine") or ""
        if engine not in model_reading_engines():
            continue
        done = refinalize_page(sidecar)
        if done is None:
            continue
        action, new_text = done
        page_num = int(sidecar["page_num"])
        changes.append(PageChange(doc=doc_dir.name, page_num=page_num, action=action))
        if not apply:
            continue
        if action == "strip_preamble":
            fragment_path = path.with_suffix(".md")
            old_text = (sidecar.get("winning_output") or {}).get("text") or ""
            old_fragment = fragment_path.read_text(encoding="utf-8")
            if old_fragment not in (old_text, new_text):
                raise RuntimeError(f"{fragment_path}: fragment is not its sidecar's text")
            if len(md_paths) == 1:
                heading = f"## Page {page_num}\n\n"
                doc_md = _swap_once(doc_md, heading + old_text, heading + new_text, md_paths[0])
            new_sidecar = dict(sidecar)
            winning = dict(new_sidecar.get("winning_output") or {})
            winning["text"] = new_text
            new_sidecar["winning_output"] = winning
            strip_writes.append((fragment_path, path, new_sidecar, new_text))
        else:
            new_sidecar = dict(sidecar)
            winning = dict(new_sidecar.get("winning_output") or {})
            winning["audit_passed"] = False
            new_sidecar["winning_output"] = winning
            new_sidecar["judge_rejected"] = True
            reocr_writes.append((path, new_sidecar))
    if not apply:
        return changes
    for fragment_path, sidecar_path, sidecar, text in strip_writes:
        _write_text(fragment_path, text)
        _write_json(sidecar_path, sidecar)
    for sidecar_path, sidecar in reocr_writes:
        _write_json(sidecar_path, sidecar)
    if reocr_writes:
        _invalidate_doc_resume_gate(doc_dir)
    if strip_writes and len(md_paths) == 1:
        _write_text(md_paths[0], doc_md)
    if changes:
        log_path = doc_dir / LOG_NAME
        log = (
            _read_json(log_path) if log_path.is_file() else {"tool": "socr-refinalize-boilerplate"}
        )
        pages = log.setdefault("pages", {})
        for change in changes:
            pages[str(change.page_num)] = {"action": change.action}
        _write_json(log_path, log)
    return changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="socr-refinalize-boilerplate", description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write sidecars/fragments (default is dry-run counts only)",
    )
    args = parser.parse_args(argv)
    counts: Counter[str] = Counter()
    for pages_dir in sorted(p for p in args.out_dir.rglob("pages") if p.is_dir()):
        doc_dir = pages_dir.parent
        for change in refinalize_document(doc_dir, apply=args.apply):
            counts[change.action] += 1
    mode = "apply" if args.apply else "dry-run"
    print(f"socr-refinalize-boilerplate ({mode})")
    for action, n in sorted(counts.items()):
        print(f"  {action}: {n}")
    if not counts:
        print("  (no flagged pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
