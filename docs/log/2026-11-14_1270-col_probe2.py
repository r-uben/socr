import json, re, os, sys

sys.path.insert(0, "/Users/rubenffuertes/.herdr/worktrees/socr/fix-issue-1270-b4/src")
import fitz
from socr.core.born_digital import BornDigitalDetector

PDFS = "/Volumes/Main/Library/Databases/socr/inputs/fed/pdfs"
ROOT = "/Volumes/Main/Library/socr-work/fed_rerun_51270eb5_words/unpacked"
import glob

col = json.load(open("/private/tmp/claude-501/col68.json"))
Q = "['\"‘“]([^'\"’”]{1,40})['\"’”]"
PATS = [
    re.compile(
        r"missing (?:the |a |an )?(?:first |last |final )?" + Q + r" (?:sub-)?columns?", re.I
    ),
    re.compile(Q + r" columns? (?:is|are) missing", re.I),
    re.compile(r"missing (?:the )?(?:first |last |final )?(?:sub-)?columns? " + Q, re.I),
    re.compile(r"(?:missing|lacks?) (?:the |a )?" + Q + r" column header", re.I),
]
det = BornDigitalDetector()
out = []
for f, jr in col:
    stem = f.split("/")[-3]
    pn = int(f[-10:-5])
    claims = []
    for s in jr.split(";"):
        for p in PATS:
            m = p.search(s)
            if m:
                claims.append((m.group(1), s.strip()))
                break
    if not claims:
        continue
    d = json.load(open(glob.glob(f"{ROOT}/*/out/*/{stem}/pages/{pn:05d}.json")[0]))
    pdf = os.path.join(PDFS, stem + ".pdf")
    doc = fitz.open(pdf)
    page = doc[pn - 1]
    words = page.get_text("words")
    txt = " ".join(w[4] for w in words)
    cnt, boxes = det._detect_table_regions(page)

    def inbox(h, b):
        ws = [w[4] for w in words if fitz.Rect(w[:4]).intersects(fitz.Rect(*b))]
        return h.lower() in " ".join(ws).lower()

    for h, s in claims[:3]:
        out.append(
            dict(
                stem=stem,
                page=pn,
                H=h,
                in_page=h.lower() in txt.lower(),
                in_regions=[inbox(h, b) for b in boxes],
                nreg=cnt,
                rot=page.rotation,
                prior=d.get("prior_ocr_layer_suspect"),
                status=d["status"],
                claim=s[:160],
            )
        )
for o in out:
    print(json.dumps(o, ensure_ascii=False))
json.dump(out, open("/private/tmp/claude-501/b4/col_cands.json", "w"))
