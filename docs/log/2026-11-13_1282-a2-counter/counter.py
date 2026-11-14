"""Independent cell counter for #1282 (see README.md). Reads run + PDFs only."""

import json, re, sys, random, collections, unicodedata
from decimal import Decimal
from pathlib import Path
from multiprocessing import Pool
import fitz

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "src")
)  # the repo's src, from this file
from socr.benchmark.scoreboard import find_documents, _read_json

RUN = Path("/Volumes/Main/Library/socr-work/fed_rerun_51270eb5_words/unpacked")
PDFS = Path("/Volumes/Main/Library/Databases/socr/inputs/fed/pdfs")
INDEX = Path(
    "/Volumes/Main/Library/Databases/socr/audits/2026-11-11_fed_independent/opus/rerun-doc-index.json"
)
OUT = Path(__file__).parent
SKIP_ENGINES = {"native", "blank", "chart_asset"}
MINUS = "−–‒‐‑"  # U+2212, en dash, figure dash, hyphens
# en dash and figure dash glued to a number AFTER another number in the line: a range separator
# (`2.3 –2.6`), not a minus.  The hyphen-minus and U+2212 stay minus (a negative mid-row is real there).
RANGE_DASHES = "–‒"
SUP = "¹²³⁰⁴⁵⁶⁷⁸⁹*†‡"
PLACEHOLDERS = {"", "-", "—", "n.a.", "na", "n/a", "...", "…", "nm", "n.m.", "…."}
NUM = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)$")
ENUM = re.compile(r"^\d{1,3}\.$")


def norm_num(tok):
    """Canonical signed string, 'PH' for placeholder, None for non-numeric."""
    t = tok.strip()
    t = re.sub(r"</?su[pb]>", "", t)
    t = re.sub(r"[*_`]", "", t)  # markdown emphasis (`*1.2*`, `**1.2**`); a footnote `*` goes too
    for c in MINUS:
        t = t.replace(c, "-")
    t = t.rstrip(SUP).lstrip("$").rstrip("%").strip()
    t = t.rstrip(SUP)
    if t.lower() in PLACEHOLDERS:
        return "PH"
    neg = False
    m = re.fullmatch(r"\((.*)\)", t)
    if m:
        neg, t = True, m.group(1).strip()
    t = t.replace(",", "")
    if not NUM.match(t):
        return None
    if t[0] in "+-":
        neg = neg ^ (t[0] == "-")
        t = t[1:]
    if t.endswith("."):
        t = t[:-1]
    if t.startswith("0."):
        t = t[1:]
    elif re.fullmatch(r"0+\d", t):
        t = t.lstrip("0") or "0"
    t = re.sub(r"^0+(?=\d)", "", t)
    if neg and not re.fullmatch(r"\.?0*", t):  # zero carries no sign
        t = "-" + t
    return t


def footnote_glue(raw, sv, pv):
    """Shipped cell carries a superscript footnote mark that the text layer
    emits as a plain trailing digit: pv == sv + one digit, raw ends in a superscript."""
    return (
        raw.strip().rstrip("%").endswith(tuple(SUP))
        and len(pv) == len(sv) + 1
        and pv.startswith(sv)
        and pv[-1].isdigit()
    )


def norm_label(s):
    s = re.sub(r"[*_]", "", s)
    s = re.sub(r"</?su[pb]>", "", s)
    return re.sub(r"[^a-z0-9]", "", s.lower())


def pdf_lines(page):
    """Group words into baseline lines: a word joins a line when its vertical
    overlap with the line's band is >= half the smaller height (a baseline
    shift of a superscript/subscript is < half a glyph height; adjacent text
    rows overlap ~0). Then merge a lone minus glyph into the following number."""
    words = [w for w in page.get_text("words") if w[4].strip()]
    words.sort(key=lambda w: (w[1], w[0]))
    lines = []  # [y0,y1,[words]]
    for w in words:
        best = None
        for L in lines:
            ov = min(L[1], w[3]) - max(L[0], w[1])
            h = min(L[1] - L[0], w[3] - w[1])
            if h > 0 and ov / h >= 0.5:
                best = L
                break
        if best:
            best[0] = min(best[0], w[1])
            best[1] = max(best[1], w[3])
            best[2].append(w)
        else:
            lines.append([w[1], w[3], [w]])
    out = []
    for y0, y1, ws in lines:
        ws.sort(key=lambda w: w[0])
        toks = []
        i = 0
        while i < len(ws):
            t = ws[i][4]
            after_number = bool(toks) and norm_num(toks[-1]) not in (None, "PH")
            if after_number and t[:1] in RANGE_DASHES:
                t = t[1:]  # a range separator after a number: not a minus; the lone dash word drops
                if not t:
                    i += 1
                    continue
            elif (t in ("-",) or (len(t) == 1 and t in MINUS)) and i + 1 < len(ws):
                nx = ws[i + 1]
                bare = re.sub(r"[$()]", "", nx[4])  # `-$12` and `-(12)` are not numbers to norm_num
                if nx[0] - ws[i][2] < (ws[i][3] - ws[i][1]) and norm_num(bare) not in (None, "PH"):
                    toks.append("-" + bare)
                    i += 2
                    continue
            toks.append(t)
            i += 1
        out.append(toks)
    return out


def split_label_nums(toks):
    """-> (label_norm, [numeric canon]) or None when line has non-numeric after label."""
    i = 0
    while i < len(toks) and ENUM.match(toks[i]) and i == 0:
        i += 1
    lab = toks[:i]
    while (
        i < len(toks)
        and norm_num(toks[i]) in (None, "PH")
        and not (norm_num(toks[i]) == "PH" and lab and False)
    ):
        # placeholder tokens before the first number stay in the label only if non numeric
        if norm_num(toks[i]) == "PH" and toks[i].strip().lower() in PLACEHOLDERS - {""}:
            break
        lab.append(toks[i])
        i += 1
    nums = []
    for t in toks[i:]:
        v = norm_num(t)
        if v is None:
            return None
        if v != "PH":
            nums.append(v)
    return norm_label(" ".join(lab)), nums


def tables(md):
    cur, out = [], []
    for ln in md.split("\n"):
        if ln.lstrip().startswith("|"):
            cur.append(ln.strip())
        else:
            if cur:
                out.append(cur)
                cur = []
    if cur:
        out.append(cur)
    res = []
    for t in out:
        rows = []
        for ln in t:
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) or c == "" for c in cells) and any(cells):
                continue
            rows.append(cells)
        res.append(rows)
    return res


def classify(s, p):
    sn, pn = s.startswith("-"), p.startswith("-")
    sb, pb = s.lstrip("-"), p.lstrip("-")
    if sb == pb:
        return "sign_only"
    sd = sb.replace(".", "")
    pd = pb.replace(".", "")
    if sd == pd and "." not in sb and "." in pb:
        lead = pb.startswith(".")
        if lead:
            return "point_drop" if sn == pn else "point_and_sign"
        return "point_other"
    return "digit_change"


def do_doc(args):
    name, ddir = args
    res = {
        "cells": [],
        "pages_scanned": 0,
        "skips": collections.Counter(),
        "rows_attr": 0,
        "unattr": collections.Counter(),
        "equal_cells": 0,
        "rows_attr_pages": set(),
        "note": None,
    }
    pdf = PDFS / (name + ".pdf")
    try:
        doc = fitz.open(pdf)
    except Exception as e:
        res["note"] = f"pdf_open_fail {e}"
        res["skips"]["pdf_open_fail(doc)"] += 1
        return res
    pdir = Path(ddir) / "pages"
    for jf in sorted(pdir.glob("*.json")):
        sc = _read_json(jf)
        if not sc:
            res["skips"]["bad_sidecar"] += 1
            continue
        if sc.get("engine") in SKIP_ENGINES:
            res["skips"]["engine_" + str(sc.get("engine"))] += 1
            continue
        if sc.get("status") not in ("success", "warning"):
            res["skips"]["status_" + str(sc.get("status"))] += 1
            continue
        mdf = jf.with_suffix(".md")
        try:
            md = mdf.read_text()
        except OSError:
            res["skips"]["md_missing"] += 1
            continue
        pno = int(sc.get("page_num") or int(jf.stem))
        if re.search(r"\[page %d failed" % pno, md) or re.search(r"\[page \d+ failed", md):
            res["skips"]["withheld_page_failed_marker"] += 1
            continue
        tbs = tables(md)
        if not tbs:
            res["skips"]["no_pipe_table"] += 1
            continue
        if pno < 1 or pno > len(doc):
            res["skips"]["page_out_of_range"] += 1
            continue
        page = doc[pno - 1]
        lines = pdf_lines(page)
        if not lines:
            res["skips"]["pdf_page_no_text_layer"] += 1
            continue
        res["pages_scanned"] += 1
        idx = collections.defaultdict(list)
        for toks in lines:
            r = split_label_nums(toks)
            if r and r[0] and r[1]:
                idx[(r[0], len(r[1]))].append(r[1])
        shipped_cnt = collections.Counter()
        for rows in tbs:
            for cells in rows:
                vs = [norm_num(c) for c in cells[1:]]
                if None in vs:
                    continue
                shipped_cnt[(norm_label(cells[0]), len([v for v in vs if v != "PH"]))] += 1
        for ti, rows in enumerate(tbs, 1):
            for cells in rows:
                label = cells[0]
                if not norm_label(label) or norm_num(label) not in (None,):
                    res["unattr"]["no_label"] += 1
                    continue
                vals, cols = [], []
                bad = False
                for ci, c in enumerate(cells[1:], 1):
                    v = norm_num(c)
                    if v is None:
                        bad = True
                        break
                    if v == "PH":
                        continue
                    vals.append(v)
                    cols.append((ci, c))
                if bad:
                    res["unattr"]["non_numeric_cell_in_row"] += 1
                    continue
                if not vals:
                    res["unattr"]["no_numeric_cells"] += 1
                    continue
                if shipped_cnt[(norm_label(label), len(vals))] > 1:
                    res["unattr"]["ambiguous_duplicate_shipped_label_on_page"] += 1
                    continue
                cand = idx.get((norm_label(label), len(vals)), [])
                if not cand:
                    res["unattr"]["no_pdf_line_same_label_and_count"] += 1
                    continue
                if len(cand) > 1:
                    res["unattr"]["ambiguous_multiple_pdf_lines"] += 1
                    continue
                pv = cand[0]
                eq = sum(a == b for a, b in zip(vals, pv))
                if eq < 2:
                    res["unattr"]["fewer_than_2_positions_agree"] += 1
                    continue
                res["rows_attr"] += 1
                res["rows_attr_pages"].add(pno)
                diff = [
                    j
                    for j in range(len(vals))
                    if vals[j] != pv[j] and not footnote_glue(cols[j][1], vals[j], pv[j])
                ]
                res["equal_cells"] += len(vals) - len(diff)
                shift = None
                if len(diff) >= 2:
                    for k in (1, -1):
                        if all(0 <= j + k < len(pv) and pv[j + k] == vals[j] for j in diff):
                            shift = k
                            break
                for j in diff:
                    cl = "cell_shift" if shift else classify(vals[j], pv[j])
                    res["cells"].append(
                        {
                            "doc": name,
                            "page": pno,
                            "table": ti,
                            "row_label": label,
                            "column": cols[j][0],
                            "numeric_pos": j + 1,
                            "shipped": cols[j][1],
                            "shipped_norm": vals[j],
                            "layer": pv[j],
                            "class": cl,
                            "row_n": len(vals),
                            "row_diff": len(diff),
                        }
                    )
    res["rows_attr_pages"] = len(res["rows_attr_pages"])
    return res


def main():
    docs = find_documents(RUN)
    idx = json.loads(INDEX.read_text())
    jobs = [
        (n, str(RUN / p) if not Path(str(p)).is_absolute() else str(p)) for n, p in docs.items()
    ]
    missing = [n for n in idx if n not in docs]
    cells, skips, unattr = [], collections.Counter(), collections.Counter()
    ra = pages_scanned = eq = rap = 0
    notes = []
    with Pool(8) as pool:
        for r in pool.imap_unordered(do_doc, jobs, chunksize=1):
            cells += r["cells"]
            skips.update(r["skips"])
            unattr.update(r["unattr"])
            ra += r["rows_attr"]
            pages_scanned += r["pages_scanned"]
            eq += r["equal_cells"]
            rap += r["rows_attr_pages"]
            if r["note"]:
                notes.append(r["note"])
    cells.sort(key=lambda c: (c["doc"], c["page"], c["table"], c["column"]))
    with open(OUT / "cells.jsonl", "w") as f:
        for c in cells:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    counts = collections.Counter(c["class"] for c in cells)
    pg = collections.defaultdict(set)
    for c in cells:
        pg[c["class"]].add((c["doc"], c["page"]))
    summ = {
        "docs_in_index": len(idx),
        "docs_found": len(docs),
        "docs_missing_from_run": missing,
        "counts_by_class": dict(counts),
        "pages_by_class": {k: len(v) for k, v in pg.items()},
        "total_contradicting_cells": len(cells),
        "pages_with_any_contradiction": len({(c["doc"], c["page"]) for c in cells}),
        "rows_attributed": ra,
        "pages_with_attributed_row_sum": rap,
        "rows_unattributable": sum(unattr.values()),
        "unattributable_by_reason": dict(unattr),
        "agreeing_cells_in_attributed_rows": eq,
        "pages_scanned_with_pdf_text_and_table": pages_scanned,
        "pages_skipped_by_reason": dict(skips),
        "notes": notes,
    }
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    sample(cells)
    print(json.dumps(summ, indent=1))


def sample(cells):
    rng = random.Random(0)
    by = collections.defaultdict(list)
    for c in cells:
        by[c["class"]].append(c)
    classes = sorted(by)
    per = 30 // max(1, len(classes))
    pick = []
    for k in classes:
        pick += rng.sample(by[k], min(per, len(by[k])))
    rest = [c for c in cells if c not in pick]
    rng.shuffle(rest)
    pick += rest[: 30 - len(pick)]
    (OUT / "png").mkdir(exist_ok=True)
    with open(OUT / "sample.jsonl", "w") as f:
        for c in pick:
            png = f"png/{c['doc']}__p{c['page']}.png"
            if not (OUT / png).exists():
                d = fitz.open(PDFS / (c["doc"] + ".pdf"))
                d[c["page"] - 1].get_pixmap(dpi=150).save(OUT / png)
            f.write(json.dumps({**c, "png": str(OUT / png)}, ensure_ascii=False) + "\n")
