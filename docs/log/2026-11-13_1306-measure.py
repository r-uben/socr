"""#1306 measurement: apply the damaged-layer guard to every shipped page fragment of the
156-doc run 51270eb5_words, and run the words check with socr's own page analysis on the pages
a hidden footnote wrapper is found on. No OCR, no model, cached data only.

    PYTHONPATH=src ~/venvs/socr/bin/python docs/log/2026-11-13_1306-measure.py out.json

Stage 1 reads each ``pages/NNNNN.md`` fragment (the text the run shipped) and counts, before
and after the guard, the control bytes by code point and the hidden wrappers stripped and kept
(with the reason kept). A wrapper is stripped only on a native-text page and only when the
PDF's painted glyphs (``BornDigitalDetector._painted_layer_text``, the same function a run
uses) do not draw its frame; a page whose PDF is missing or unreadable keeps its wrappers and
is listed in ``skipped``. U+FFFD counted after the guard excludes those already in the
fragment (``pre_existing_replacement_chars``). Stage 2 rebuilds the words reference from the PDF for each page that
carries a wrapper and counts the words the check reports missing with the wrapper in place
(what run 51270eb5 shipped), with the wrapper stripped and not excused (what a guard without
the words-reference change would ship) and stripped and excused (what this change ships).
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

RUN = Path("/Volumes/Main/Library/socr-work/fed_rerun_51270eb5_words/unpacked")
PDFS = Path("/Volumes/Main/Library/Databases/socr/inputs/fed/pdfs")


def _pages():
    for md in sorted(RUN.glob("*/out/*/*/pages/*.md")):
        yield md.parents[1].name, md


def _painted(doc: str, page_num: int, cache: dict, skipped: list):
    """The page's painted glyphs, or None (recorded once per doc in *skipped*)."""
    from socr.core.born_digital import BornDigitalDetector

    if doc not in cache:
        pdf = PDFS / f"{doc}.pdf"
        try:
            import fitz

            cache[doc] = fitz.open(pdf)
        except Exception as exc:  # reported, never hidden
            cache[doc] = None
            skipped.append({"doc": doc, "skipped": f"PDF not opened: {exc!r}"})
    handle = cache[doc]
    if handle is None:
        return None
    painted = BornDigitalDetector()._painted_layer_text(handle[page_num - 1])
    if painted is None and not any(row["doc"] == doc for row in skipped):
        skipped.append({"doc": doc, "skipped": "painted glyph scan failed"})
    return painted


def _stage1():
    from socr.core.manifest import engine_ships_native_text
    from socr.core.damaged_layer import (
        CONTROL_BYTE_RE,
        FootnoteResult,
        REPLACEMENT_CHAR,
        replace_control_bytes,
        strip_hidden_footnotes,
    )

    out = {
        "pages": 0,
        "before_control": collections.Counter(),
        "after_control": collections.Counter(),
        "after_replacement_chars": 0,
        "pre_existing_replacement_chars": 0,
        "model_pages_with_wrapper": 0,
        "skipped": [],
        "control_pages": [],
        "wrapper_pages": [],
        "stripped": 0,
        "kept": collections.Counter(),
        "kept_examples": {},
        "pages_stripped": 0,
        "pages_with_kept": 0,
        "pages_stripped_and_kept": 0,
    }
    cache: dict = {}
    for doc, md in _pages():
        text = md.read_text(errors="replace")
        out["pages"] += 1
        out["pre_existing_replacement_chars"] += text.count(REPLACEMENT_CHAR)
        for m in CONTROL_BYTE_RE.findall(text):
            out["before_control"][f"U+{ord(m):04X}"] += 1
        repaired, found = replace_control_bytes(text)
        if found:
            out["control_pages"].append((doc, md.name, dict(found)))
        painted, native = None, True
        if "ootnote" in repaired.lower():
            side = json.loads(md.with_suffix(".json").read_text())
            native = engine_ships_native_text(side.get("engine"))
            painted = _painted(doc, int(md.stem), cache, out["skipped"]) if native else None
            if not native:
                out["model_pages_with_wrapper"] += 1
        footnotes = (
            strip_hidden_footnotes(repaired, painted) if native else FootnoteResult(repaired)
        )
        final = footnotes.text
        for m in CONTROL_BYTE_RE.findall(final):
            out["after_control"][f"U+{ord(m):04X}"] += 1
        out["after_replacement_chars"] += final.count(
            REPLACEMENT_CHAR
        )  # minus pre-existing: see below
        if footnotes.stripped or footnotes.kept:
            out["wrapper_pages"].append((doc, md.name))
        out["stripped"] += len(footnotes.stripped)
        for k in footnotes.kept:
            out["kept"][k.reason] += 1
            out["kept_examples"].setdefault(k.reason, (doc, md.name, k.note))
        out["pages_stripped"] += bool(footnotes.stripped)
        out["pages_with_kept"] += bool(footnotes.kept)
        out["pages_stripped_and_kept"] += bool(footnotes.stripped and footnotes.kept)
    out["after_replacement_chars"] -= out["pre_existing_replacement_chars"]
    return out


def _stage2(wrapper_pages):
    from socr.core.born_digital import BornDigitalDetector
    from socr.core.damaged_layer import strip_hidden_footnotes
    from socr.core.manifest import engine_ships_native_text
    from socr.core.result import PageOutput
    from socr.core.state import PageState
    from socr.core.verdict import page_missing_words

    by_doc = collections.defaultdict(list)
    for doc, name in wrapper_pages:
        by_doc[doc].append(name)
    rows = []
    for doc, names in sorted(by_doc.items()):
        pdf = PDFS / f"{doc}.pdf"
        if not pdf.exists():
            rows.append({"doc": doc, "skipped": "no PDF"})
            continue
        try:
            analysis = BornDigitalDetector().detect(pdf)
        except Exception as exc:  # reported, never hidden
            rows.append({"doc": doc, "skipped": f"page analysis failed: {exc!r}"})
            continue
        doc_dir = next(RUN.glob(f"*/out/*/{doc}"))
        for name in names:
            n = int(Path(name).stem)
            side = json.loads((doc_dir / "pages" / f"{n:05d}.json").read_text())
            text = (doc_dir / "pages" / name).read_text(errors="replace")
            pa = analysis.pages[n - 1]
            p = PageState(page_num=n)
            for f in (
                "own_layer_word_reference",
                "own_layer_figure_words",
                "is_born_digital",
                "has_unrecovered_symbol_glyphs",
            ):
                setattr(p, f, getattr(pa, f))
            p.prior_ocr_layer_suspect = bool(side.get("prior_ocr_layer_suspect"))
            output = PageOutput.from_dict(side.get("winning_output") or {})
            output.text = text
            before = page_missing_words(output, p)
            result = strip_hidden_footnotes(text, pa.painted_layer_text)
            stripped = PageOutput.from_dict(side.get("winning_output") or {})
            stripped.text = result.text
            plain = page_missing_words(stripped, p)  # no excuse recorded on p
            p.hidden_footnote_words = result.stripped_words
            excused = page_missing_words(stripped, p)
            rows.append(
                {
                    "doc": doc,
                    "page": n,
                    "engine": side.get("engine"),
                    "status": side.get("status"),
                    "native_lane": engine_ships_native_text(stripped.engine),
                    "stripped": len(result.stripped),
                    "kept": [k.reason for k in result.kept],
                    "missing_in_run": None if before is None else len(before),
                    "missing_stripped_not_excused": None if plain is None else len(plain),
                    "missing_stripped_excused": None if excused is None else len(excused),
                    "new_missing_words": sorted(set(excused or ()) - set(before or ())),
                }
            )
    return rows


def main(out_path: str) -> None:
    stage1 = _stage1()
    stage2 = _stage2(stage1["wrapper_pages"])
    json.dump({"stage1": stage1, "stage2": stage2}, open(out_path, "w"), indent=1, default=str)
    print(
        json.dumps({k: v for k, v in stage1.items() if k not in {"wrapper_pages"}}, default=str)[
            :4000
        ]
    )
    applied = [r for r in stage2 if "stripped" in r and r["stripped"]]
    print(
        "stage2 pages stripped:",
        len(applied),
        "new missing (excused vs run):",
        sum(1 for r in applied if r["new_missing_words"]),
        "extra missing if not excused:",
        sum(
            1
            for r in applied
            if (r["missing_stripped_not_excused"] or 0) > (r["missing_in_run"] or 0)
        ),
    )
    skipped = [r for r in stage2 if "skipped" in r] + stage1["skipped"]
    print("skipped docs:", len(skipped), [(r["doc"], r["skipped"]) for r in skipped])


if __name__ == "__main__":
    main(sys.argv[1])
