"""#1257 release rule replay: of the pages B1 frees, which carry independent page-scope evidence?

Reads the per-table replay (``2026-11-14_1257-replay.py``, run it first) and, for each page that
loses its row-binding mark under B1 alone, evaluates ``verdict.page_row_binding_release`` on the
page's shipped text of the cached run: the page analysis of its PDF supplies the layer the words and
numbers checks compare against (as ``socr-refinalize-own-text`` does), the sidecar supplies the
output and its audit events.  The marks socr itself put on the text (banners, table marks) are
removed first: the check ran on the text before them.  The run predates the pass trace, so the trace is
recomputed here, not read.  Provider state: with providers (the run had them).

Run from the repo root::

    PYTHONPATH=src ~/venvs/socr/bin/python docs/log/2026-11-14_1257-release-replay.py
"""

from __future__ import annotations

import collections
import importlib.util
import json
import re
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("replay1257", HERE / "2026-11-14_1257-replay.py")
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)

TABLE_MARK = re.compile(r"^> \[table unverified: [^\n]*\]\n\n?", re.MULTILINE)


def shipped_before_marks(md: str) -> str:
    from socr.core.manifest import without_page_top_banners

    return TABLE_MARK.sub("", without_page_top_banners(md, lambda banner: True))


def main() -> None:
    from socr.core.born_digital import BornDigitalDetector
    from socr.core.result import PageOutput
    from socr.core.verdict import page_row_binding_release
    from socr.pipeline.refinalize_own_text import _page_state

    items = replay.index_pages()
    by_key = {(i["doc"], i["page"]): i for i in items}
    replayed = [json.loads(line) for line in open(replay.OUT, encoding="utf-8")]
    freed = [
        r
        for r in replayed
        if any(t["old"] == "unverifiable" for t in r["tables"])
        and not any(t["new"] == "unverifiable" for t in r["tables"])
        and not any(t["old"] == "contradicted" for t in r["tables"])
    ]
    analyses: dict[str, object] = {}
    cat = collections.Counter()
    rows = []
    for r in freed:
        i = by_key[(r["doc"], r["page"])]
        sidecar = json.load(open(i["sidecar"], encoding="utf-8"))
        if r["doc"] not in analyses:
            analyses[r["doc"]] = BornDigitalDetector().detect(Path(f"{replay.PDFS}/{r['doc']}.pdf"))
        pa = analyses[r["doc"]].pages[r["page"] - 1]
        p = _page_state(sidecar, pa)
        md = Path(i["sidecar"][:-5] + ".md").read_text(encoding="utf-8")
        output = replace(
            PageOutput.from_dict(sidecar.get("winning_output") or {}), text=shipped_before_marks(md)
        )
        kinds = frozenset(e.get("kind", "") for e in sidecar.get("audit_events") or [])
        released = page_row_binding_release(output, p, kinds, providers_available=True)
        same = [g for g in i["gold"] if g["run"] == "51270eb5" and g["md_sha256"] == r["md_sha"]]
        if any(g["content"] == "defective" for g in same):
            c = "gold-defective on this exact markdown"
        elif any(g["content"] == "clean" for g in same):
            c = "gold-clean on this exact markdown"
        elif any(g["content"] == "defective" for g in i["gold"]):
            c = "gold-defective on the page (another reading of it)"
        elif any(g["content"] == "clean" for g in i["gold"]):
            c = "gold-clean on the page (another reading of it)"
        else:
            c = "no gold verdict"
        cat[(c, "freed" if released else "kept")] += 1
        rows.append(dict(doc=r["doc"], page=r["page"], gold=c, released=released))
    print(f"pages B1 frees (before the release rule): {len(freed)}")
    for (c, state), n in sorted(cat.items()):
        print(f"  {n:3d}  {c}  -> {state}")
    defective = sum(
        n for (c, s), n in cat.items() if s == "freed" and c.startswith("gold-defective")
    )
    clean = sum(n for (c, s), n in cat.items() if s == "freed" and c.startswith("gold-clean"))
    total = sum(n for (_c, s), n in cat.items() if s == "freed")
    print(f"freed now: {total}; gold-defective freed: {defective}; gold-clean freed: {clean}")
    out = Path(replay.OUT).with_name("socr-1257-release.jsonl")
    out.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
    print(f"per-page result written to {out}")


if __name__ == "__main__":
    main()
