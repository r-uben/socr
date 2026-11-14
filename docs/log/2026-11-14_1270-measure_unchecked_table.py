"""#1270 B4 review: of the pages classes 2 and 3 clear, how many ship a table with no ladder
terminal (they keep a table-scoped mark). Projected from the cached sidecars of run 51270eb5
plus the changed detector, as ``2026-11-14_1270-measure_c2.py`` does. Inputs: measure2.json."""

import glob
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, "/Users/rubenffuertes/.herdr/worktrees/socr/fix-issue-1270-b4/src")
from socr.core.born_digital import BornDigitalDetector
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.verdict import _text_layer_numbers_verdict, unchecked_table_clauses
from socr.judge.table_verdict import TABLE_LADDER_EVENT_KINDS

PDFS = "/Volumes/Main/Library/Databases/socr/inputs/fed/pdfs"
ROOT = "/Volumes/Main/Library/socr-work/fed_rerun_51270eb5_words/unpacked"
res = json.load(open("/private/tmp/claude-501/b4/measure2.json"))
det = BornDigitalDetector()
rows = []
for cls in ("c2", "c3"):
    for r in res[cls]:
        f = glob.glob(f"{ROOT}/*/out/*/{r['stem']}/pages/{r['page']:05d}.json")[0]
        d = json.load(open(f))
        wo = d["winning_output"]
        pa = det.detect_page(os.path.join(PDFS, r["stem"] + ".pdf"), r["page"])
        disp = d.get("table_ladder_disposition")
        p = SimpleNamespace(
            own_layer_text=pa.own_layer_text,
            own_layer_word_reference=pa.own_layer_word_reference,
            own_layer_figure_words=pa.own_layer_figure_words,
            is_born_digital=pa.is_born_digital,
            prior_ocr_layer_suspect=bool(
                pa.prior_ocr_layer
                or pa.prior_ocr_layer_scan_failed
                or pa.invisible_text_over_raster
            ),
            has_unrecovered_symbol_glyphs=pa.has_unrecovered_symbol_glyphs,
            table_ladder_disposition=FailureMode(disp) if disp else None,
            text_coverage_verdict=d.get("text_coverage", ""),
            has_figures=pa.has_figures,
            control_byte_digit_hits=0,
            control_byte_scan_failed=False,
            minus_as_digit_hits=0,
            minus_as_digit_scan_failed=False,
            has_encoding_hygiene_suspect=False,
            layer_refused_for_shape_alone=pa.layer_refused_for_shape_alone,
            layer_glyphs_undamaged=pa.layer_glyphs_undamaged,
            has_unmapped_math_glyphs=pa.has_unmapped_math_glyphs,
            has_corrupt_math=pa.has_corrupt_math,
        )
        o = PageOutput(
            page_num=r["page"],
            text=wo["text"],
            status=PageStatus.SUCCESS,
            engine=d.get("engine") or "qwen",
            failure_mode=FailureMode(d.get("failure_mode") or "none"),
        )
        clause, cleared = _text_layer_numbers_verdict(o, p)
        kinds = frozenset(e["kind"] for e in d.get("audit_events", []))
        tables = unchecked_table_clauses(o, p, kinds)
        has_terminal_event = bool(kinds & TABLE_LADDER_EVENT_KINDS)
        rows.append(
            dict(
                cls=cls,
                stem=r["stem"],
                page=r["page"],
                cleared=cleared,
                keeps_mark=bool(cleared and tables),
                tables=len(tables),
                has_terminal=bool(p.table_ladder_disposition) or has_terminal_event,
                nclauses=r["nclauses"],
            )
        )
json.dump(rows, open("/private/tmp/claude-501/b4/measure_unchecked_table.json", "w"))
for cls in ("c2", "c3"):
    rs = [x for x in rows if x["cls"] == cls]
    cl = [x for x in rs if x["cleared"]]
    keep = [x for x in cl if x["keeps_mark"]]
    sole = [x for x in rs if x["nclauses"] == 1]
    print(
        cls,
        "pages",
        len(rs),
        "cleared by the numbers rule",
        len(cl),
        "of which carry a table with no ladder terminal (keep a mark)",
        len(keep),
        "| sole-clause pages",
        len(sole),
        "cleared",
        sum(x["cleared"] for x in sole),
        "still unmarked",
        sum(1 for x in sole if x["cleared"] and not x["keeps_mark"]),
    )
