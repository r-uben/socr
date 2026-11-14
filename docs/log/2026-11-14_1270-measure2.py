import json, sys, os, collections, re

sys.path.insert(0, "/Users/rubenffuertes/.herdr/worktrees/socr/fix-issue-1270-b4/src")
import fitz
from socr.core.born_digital import BornDigitalDetector

PDFS = "/Volumes/Main/Library/Databases/socr/inputs/fed/pdfs"
pages = json.load(open("/private/tmp/claude-501/b4/pages.json"))
det = BornDigitalDetector()


def assess(stem, pn):
    pdf = os.path.join(PDFS, stem + ".pdf")
    return det.detect_page(pdf, pn)


PUA_R = "private-use codepoints in this page's native text layer are unrecovered mathematics"
NONUM = "the reading ships no number to check against the page's text layer"
AGREE = "the reading's numbers agree with this page's text layer, but"
res = collections.defaultdict(list)
for p in pages:
    for cl in p["clauses"]:
        key = None
        if cl["check"] == "native_glyphs" and cl["reason"] == PUA_R:
            key = "c1"
        elif cl["check"] == "text_layer_numbers" and cl["reason"] == NONUM:
            key = "c2"
        elif cl["check"] == "text_layer_numbers" and cl["reason"].startswith(AGREE):
            key = "c3"
        if key:
            try:
                pa = assess(p["stem"], p["page"])
                err = None
            except Exception as e:
                pa = None
                err = str(e)
            r = dict(
                stem=p["stem"],
                page=p["page"],
                status=p["status"],
                mode=p["mode"],
                nclauses=len(p["clauses"]),
                other=[(c["check"], c["result"]) for c in p["clauses"] if c is not cl],
                words=p["words"],
                cov=p["cov"],
                err=err,
            )
            if pa is not None:
                r.update(
                    flag_new=pa.has_unmapped_math_glyphs,
                    pua_raw=bool(re.search("[-]", pa.native_text or "")),
                    shape=pa.layer_refused_for_shape_alone,
                    has_fig=pa.has_figures,
                    born=pa.is_born_digital,
                    notes=pa.notes[:3],
                    char_count=pa.char_count,
                    unrec=pa.has_unrecovered_symbol_glyphs,
                )
            res[key].append(r)
json.dump(res, open("/private/tmp/claude-501/b4/measure2.json", "w"))
for k, v in res.items():
    print(k, len(v))
