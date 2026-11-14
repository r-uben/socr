import json, sys, os

sys.path.insert(0, "/Users/rubenffuertes/.herdr/worktrees/socr/fix-issue-1270-b4/src")
from types import SimpleNamespace
from socr.core.born_digital import BornDigitalDetector
from socr.core.result import PageOutput, PageStatus
from socr.core.verdict import text_layer_numbers_clause

PDFS = "/Volumes/Main/Library/Databases/socr/inputs/fed/pdfs"
ROOT = "/Volumes/Main/Library/socr-work/fed_rerun_51270eb5_words/unpacked"
import glob

res = json.load(open("/private/tmp/claude-501/b4/measure2.json"))
det = BornDigitalDetector()
out = []
for r in res["c2"]:
    f = glob.glob(f"{ROOT}/*/out/*/{r['stem']}/pages/{r['page']:05d}.json")[0]
    d = json.load(open(f))
    wo = d["winning_output"]
    text = wo["text"]
    # strip socr marks the way a finalised body would: the clause reads the reading, marks are removed inside
    pa = det.detect_page(os.path.join(PDFS, r["stem"] + ".pdf"), r["page"])
    p = SimpleNamespace(
        own_layer_text=pa.own_layer_text,
        own_layer_word_reference=pa.own_layer_word_reference,
        own_layer_figure_words=pa.own_layer_figure_words,
        is_born_digital=pa.is_born_digital,
        prior_ocr_layer_suspect=bool(
            pa.prior_ocr_layer or pa.prior_ocr_layer_scan_failed or pa.invisible_text_over_raster
        ),
        has_unrecovered_symbol_glyphs=pa.has_unrecovered_symbol_glyphs,
        table_ladder_disposition=None,
        text_coverage_verdict=d.get("text_coverage", ""),
        has_figures=pa.has_figures,
        control_byte_digit_hits=0,
        control_byte_scan_failed=False,
        minus_as_digit_hits=0,
        minus_as_digit_scan_failed=False,
        has_encoding_hygiene_suspect=False,
        # review of PR #1325: the verdict now reads the detector's cleanliness evidence
        layer_refused_for_shape_alone=pa.layer_refused_for_shape_alone,
        layer_glyphs_undamaged=pa.layer_glyphs_undamaged,
        has_unmapped_math_glyphs=pa.has_unmapped_math_glyphs,
        has_corrupt_math=pa.has_corrupt_math,
    )
    o = PageOutput(
        page_num=r["page"],
        text=text,
        status=PageStatus.SUCCESS,
        engine=d.get("engine") or "qwen",
        failure_mode=__import__("socr.core.result", fromlist=["FailureMode"]).FailureMode(
            d.get("failure_mode") or "none"
        ),
    )
    cl = text_layer_numbers_clause(o, p)
    out.append(
        dict(
            stem=r["stem"],
            page=r["page"],
            nclauses=r["nclauses"],
            cleared=cl is None,
            other=r["other"],
            layer=pa.own_layer_text[:60],
            cov=p.text_coverage_verdict,
            fig=p.has_figures,
        )
    )
json.dump(out, open("/private/tmp/claude-501/b4/measure_c2.json", "w"))
print(
    "c2 total",
    len(out),
    "cleared",
    sum(o["cleared"] for o in out),
    "sole-clause pages",
    sum(1 for o in out if o["nclauses"] == 1),
    "sole cleared",
    sum(1 for o in out if o["nclauses"] == 1 and o["cleared"]),
)
for o in out:
    if o["nclauses"] == 1 and not o["cleared"]:
        print("kept sole:", o["stem"][-36:], o["page"], repr(o["layer"]), o["cov"], o["fig"])
