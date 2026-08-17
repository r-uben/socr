"""Generate a self-contained side-by-side review page: PDF page image vs extracted markdown.

GH-220. This is a *judgement instrument*, not a report. Two rules follow from that and
govern every choice below:

1. **It must never imply a page is correct.** socr records ``status: "success"`` on pages
   that carry audit warnings, and at least one observed page (EFO-Nov-2022 p58) fails its
   audit while recording no audit event at all. So the viewer never paints a page "clean";
   the strongest thing it says is *no machine signal was recorded*, which is not the same
   claim.
2. **Absent data must look absent.** A missing markdown fragment renders as a loud MISSING
   panel, never as an empty pane that reads like a blank page.

The page universe is taken from the **PDF**, not from ``pages/``, so a fragment that was
never written surfaces as a gap instead of silently shortening the document.
"""

from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- Rendering constants -------------------------------------------------------------
#
# Measured on the 68-page A4 EFO-Nov-2022 corpus document (2026-08-17):
#   scale 1.7 / q55 -> 1013x1432 px, 13.10 MB base64 for 68 pages
#   scale 2.0 / q55 -> 1191x1684 px, 16.60 MB base64  (overflows the cap)
# Grayscale was measured and rejected: these pages are already near-monochrome, so it
# saves almost nothing while degrading chart legibility.
RENDER_SCALE = 1.7
"""PyMuPDF matrix scale. Largest value that keeps a 68-page A4 document under the cap."""

JPEG_QUALITY = 55
"""JPEG quality. Table digits stay legible; CSS zoom covers the rest."""

ARTIFACT_BYTE_CAP = 16 * 1024 * 1024
"""Hard external limit: the artifact host refuses a rendered page larger than this."""

WRITE_REFUSAL_FLOOR = 15 * 1024 * 1024
"""Refuse to write above this, leaving headroom under the cap. Never silently drop pages."""

# Signals that mean "a machine noticed something". Presence of any of these makes a page
# suspect. Absence means only that nothing was recorded -- see rule 1 in the module docstring.
_BOOL_SIGNAL_KEYS = (
    "judge_rejected",
    "native_table_structure_failed",
    "native_table_unverifiable",
    "chart_asset_render_failed",
    "needs_ocr_enhancement",
)


@dataclass
class PageRecord:
    """One page's evidence. Every absence is explicit rather than defaulted away."""

    page_num: int
    image_b64: str = ""
    image_error: str = ""
    markdown: str | None = None
    """None means the fragment file does not exist. '' means it exists and is empty."""
    md_path_missing: bool = False
    json_path_missing: bool = False
    sidecar: dict[str, Any] = field(default_factory=dict)
    signals: list[str] = field(default_factory=list)
    reported_status: str = ""

    @property
    def suspect(self) -> bool:
        return bool(self.signals)

    @property
    def contradicts_itself(self) -> bool:
        """Page claims success while carrying at least one recorded signal."""
        return self.reported_status == "success" and self.suspect


@dataclass
class ReviewReport:
    """Document-level evidence plus per-page records."""

    pdf_name: str
    doc_status: str
    pages: list[PageRecord]
    untrusted_pages: list[int] = field(default_factory=list)
    table_flag_count: int = 0

    @property
    def suspect_count(self) -> int:
        return sum(1 for p in self.pages if p.suspect)

    @property
    def contradiction_count(self) -> int:
        return sum(1 for p in self.pages if p.contradicts_itself)


def _audit_event_kinds(sidecar: dict[str, Any]) -> list[str]:
    kinds: list[str] = []
    for event in sidecar.get("audit_events") or []:
        if isinstance(event, dict):
            kind = event.get("kind") or event.get("type") or event.get("event")
            kinds.append(str(kind) if kind else "unnamed_event")
        else:
            kinds.append(str(event))
    return kinds


def _untrusted_page_numbers(trust: dict[str, Any]) -> list[int]:
    """Extract page numbers from tables_trust.json, tolerating int or dict entries."""
    out: list[int] = []
    for entry in trust.get("untrusted_pages") or []:
        if isinstance(entry, dict):
            num = entry.get("page_num") or entry.get("page")
            if num is not None:
                out.append(int(num))
        elif isinstance(entry, int | str):
            try:
                out.append(int(entry))
            except (TypeError, ValueError):
                continue
    return sorted(set(out))


def _derive_signals(record: PageRecord, sidecar: dict[str, Any], untrusted: set[int]) -> list[str]:
    """Everything a machine noticed about this page, as human-readable labels."""
    signals: list[str] = []

    if record.md_path_missing:
        signals.append("markdown fragment missing")
    elif record.markdown is not None and not record.markdown.strip():
        signals.append("extract is empty")
    if record.json_path_missing:
        signals.append("sidecar missing")
    if record.image_error:
        signals.append(f"page image failed: {record.image_error}")

    signals.extend(_audit_event_kinds(sidecar))

    for key in _BOOL_SIGNAL_KEYS:
        if sidecar.get(key):
            signals.append(key)

    winning = sidecar.get("winning_output")
    if isinstance(winning, dict) and winning.get("audit_passed") is False:
        # The p58 case: audit failed, no audit_events recorded. Invisible without this.
        signals.append("audit_passed=false")

    if record.json_path_missing is False and sidecar.get("terminal") is not True:
        signals.append("page not terminal")

    if record.page_num in untrusted:
        signals.append("tables untrusted")

    return signals


def _render_page_image(pdf: Any, index: int, scale: float, quality: int) -> tuple[str, str]:
    """Return (base64 jpeg, error). Never raises -- a failed render must be visible."""
    try:
        import fitz

        pixmap = pdf.load_page(index).get_pixmap(matrix=fitz.Matrix(scale, scale))
        return base64.b64encode(pixmap.tobytes("jpeg", jpg_quality=quality)).decode(), ""
    except Exception as exc:  # noqa: BLE001 - surfacing beats propagating here
        return "", type(exc).__name__


def collect_pages(
    doc_dir: Path,
    pdf_path: Path,
    *,
    scale: float = RENDER_SCALE,
    quality: int = JPEG_QUALITY,
) -> ReviewReport:
    """Gather per-page evidence. The PDF defines the page universe, not ``pages/``."""
    import fitz

    metadata = _read_json(doc_dir / "metadata.json")
    trust = _read_json(doc_dir / "tables_trust.json")
    untrusted = set(_untrusted_page_numbers(trust))

    pdf = fitz.open(str(pdf_path))
    pages_dir = doc_dir / "pages"
    records: list[PageRecord] = []

    for index in range(pdf.page_count):
        page_num = index + 1
        md_path = pages_dir / f"{page_num:05d}.md"
        json_path = pages_dir / f"{page_num:05d}.json"

        record = PageRecord(page_num=page_num)
        record.md_path_missing = not md_path.exists()
        record.json_path_missing = not json_path.exists()
        if not record.md_path_missing:
            record.markdown = md_path.read_text(encoding="utf-8", errors="replace")

        sidecar = _read_json(json_path) if not record.json_path_missing else {}
        record.sidecar = sidecar
        record.reported_status = str(sidecar.get("status") or "")
        record.image_b64, record.image_error = _render_page_image(pdf, index, scale, quality)
        record.signals = _derive_signals(record, sidecar, untrusted)
        records.append(record)

    pdf.close()

    return ReviewReport(
        pdf_name=pdf_path.name,
        doc_status=str(metadata.get("status") or "unknown"),
        pages=records,
        untrusted_pages=sorted(untrusted),
        table_flag_count=int(trust.get("table_flags_n") or 0),
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _js_json(value: Any) -> str:
    """Serialize for embedding inside an inline ``<script>``.

    Escaping only ``</`` is not sufficient. Document text reaching this page is
    untrusted -- a PDF filename, a ``metadata.json`` status, or extracted markdown can
    contain anything -- and two sequences break out of a script element:
    a literal ``</script>`` ends it early, and ``<!--<script`` puts the parser into the
    script-data-double-escaped state so the real closing tag no longer closes it and the
    viewer silently stops working.

    Escaping every ``<`` as ``\\u003c`` closes both. The parsed JSON value is identical,
    since ``\\u003c`` is just how JSON spells ``<``.
    """
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


def build_review_html(report: ReviewReport, *, title: str | None = None) -> str:
    """Render the self-contained HTML. No external hosts: CSP blocks every one."""
    payload = [
        {
            "n": p.page_num,
            "img": p.image_b64,
            "imgErr": p.image_error,
            "md": p.markdown,
            "mdMissing": p.md_path_missing,
            "jsonMissing": p.json_path_missing,
            "status": p.reported_status,
            "engine": str(p.sidecar.get("engine") or ""),
            "provider": str(p.sidecar.get("provider") or ""),
            "cost": p.sidecar.get("cost_usd") or 0,
            "signals": p.signals,
            "contradicts": p.contradicts_itself,
        }
        for p in report.pages
    ]
    page_title = title or "socr Page Judge"

    # Count what is actually true rather than asserting it in the template. An
    # earlier revision hardcoded "every page reports success", which is a claim the
    # data does not establish -- and a viewer that states an unverified fact about
    # page status is the exact failure it exists to catch.
    reported_success = sum(1 for p in report.pages if p.reported_status == "success")
    reported_other = len(report.pages) - reported_success

    summary = {
        "pdf": report.pdf_name,
        "docStatus": report.doc_status,
        "pageCount": len(report.pages),
        "suspect": report.suspect_count,
        "contradictions": report.contradiction_count,
        "untrusted": len(report.untrusted_pages),
        "tableFlags": report.table_flag_count,
        "reportedSuccess": reported_success,
        "reportedOther": reported_other,
    }

    return (
        _TEMPLATE.replace("__TITLE__", html.escape(page_title))
        .replace("__SUMMARY__", _js_json(summary))
        .replace("__PAGES__", _js_json(payload))
    )


_TEMPLATE = r"""<meta charset="utf-8">
<title>__TITLE__</title>
<style>
/* Palette. Neutrals are warm-biased off the alarm hue rather than pure grey, so the
   page reads as ink-on-paper next to a scanned document. Semantic colours (alarm/warn)
   are deliberately separate from the navigation accent. */
:root{
  --bg:#fbfaf8; --panel:#ffffff; --ink:#1b1a18; --muted:#6a6862; --line:#e0ddd6;
  --alarm:#9e2410; --alarm-bg:#fbeeea; --warn:#7d5400; --warn-bg:#fbf4e4;
  --mark:#ffeea3; --accent:#1f5673;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --bg:#171614; --panel:#211f1d; --ink:#eeece7; --muted:#9b978e; --line:#37342f;
    --alarm:#ff9077; --alarm-bg:#3b1913; --warn:#e3b155; --warn-bg:#332810;
    --mark:#5e5220; --accent:#8fbdd8;
  }
}
:root[data-theme="dark"]{
  --bg:#171614; --panel:#211f1d; --ink:#eeece7; --muted:#9b978e; --line:#37342f;
  --alarm:#ff9077; --alarm-bg:#3b1913; --warn:#e3b155; --warn-bg:#332810;
  --mark:#5e5220; --accent:#8fbdd8;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);overflow-x:hidden;
  font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:9;background:var(--alarm-bg);color:var(--alarm);
  border-bottom:2px solid var(--alarm);padding:10px 14px}
header b{font-weight:700}
header .sub{color:var(--ink);opacity:.82;font-size:13px;margin-top:3px;max-width:80ch}
#rail{display:flex;flex-wrap:wrap;gap:3px;padding:8px 14px;background:var(--panel);
  border-bottom:1px solid var(--line);position:sticky;top:var(--htop,64px);z-index:8}
#rail button{min-width:30px;padding:0 4px;height:26px;border:1px solid var(--line);
  background:transparent;color:var(--muted);border-radius:3px;cursor:pointer;
  font:12px ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}
#rail button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
#rail button.cur{outline:2px solid var(--accent);color:var(--ink);font-weight:700}
body.reveal #rail button.sus{background:var(--alarm-bg);border-color:var(--alarm);color:var(--alarm)}
/* Absent evidence is shown unconditionally, even while judging cold: a page whose
   markdown, sidecar, or image is missing cannot be judged at all, and hiding that
   would make the instrument lie. Recorded warnings are what the 'w' toggle hides. */
#rail button.gone{border-color:var(--alarm);color:var(--alarm);border-style:dashed}
#rail .g{opacity:.85;font-weight:700}
/* The warning COUNT is part of what cold judging hides; the missing-evidence cross is not. */
body:not(.reveal) #rail .sigct{display:none}
main{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px;align-items:start}
@media (max-width:900px){main{grid-template-columns:1fr}}
section{background:var(--panel);border:1px solid var(--line);border-radius:6px;
  overflow:auto;max-height:calc(100vh - 150px)}
.ph{padding:7px 11px;border-bottom:1px solid var(--line);font-size:12px;color:var(--muted);
  position:sticky;top:0;background:var(--panel);z-index:2}
#imgwrap{padding:10px;text-align:center}
#imgwrap img{max-width:100%;height:auto;border:1px solid var(--line)}
#extract{padding:12px 14px}
#extract table{border-collapse:collapse;font-size:13px;margin:10px 0;
  font-variant-numeric:tabular-nums}
#extract th,#extract td{border:1px solid var(--line);padding:4px 7px;text-align:left}
#extract th{background:var(--bg);font-weight:600}
.scroller{overflow-x:auto;max-width:100%}
#extract pre{background:var(--bg);padding:9px;border-radius:4px;overflow-x:auto;font-size:12.5px}
#extract h1,#extract h2,#extract h3{margin:.7em 0 .3em;line-height:1.25;text-wrap:balance}
mark{background:var(--mark);color:var(--ink);padding:0 1px;border-radius:2px}
body:not(.nums) mark{background:transparent}
.void{margin:14px;padding:22px;border:2px dashed var(--alarm);color:var(--alarm);
  background:var(--alarm-bg);border-radius:6px;text-align:center;font-weight:700}
#sigbox{margin:0;padding:9px 12px;border-bottom:1px solid var(--line);background:var(--warn-bg);
  color:var(--ink)}
body:not(.reveal) #sigbox{display:none}
.chip{display:inline-block;background:var(--alarm-bg);color:var(--alarm);
  border:1px solid var(--alarm);border-radius:3px;padding:1px 7px;margin:2px 3px 2px 0;
  font-size:12px}
.hint{color:var(--muted);font-size:12px;padding:7px 12px;border-bottom:1px solid var(--line)}
body.reveal .hint.cold{display:none}
.strike{text-decoration:line-through;opacity:.65}
kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:3px;padding:0 4px;
  font:11px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--bg);color:var(--ink)}
</style>

<header id="hdr"></header>
<div id="rail"></div>
<main>
  <section id="left"><div class="ph" id="lph"></div><div id="imgwrap"></div></section>
  <section id="right">
    <div class="ph" id="rph"></div>
    <div id="sigbox"></div>
    <div class="hint cold">Judging cold: recorded warnings are hidden. Press <kbd>w</kbd> to reveal them for this page and the rail.</div>
    <div id="extract"></div>
  </section>
</main>

<script>
const SUMMARY = __SUMMARY__;
const PAGES = __PAGES__;
let cur = 0, raw = false, zoom = 100;

function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

// Numbers are the payload in a citation corpus, so they get marked for eye-scanning.
// Wrapped after escaping so the markup cannot be injected from document text.
function inline(s){
  s = esc(s);
  s = s.replace(/(-|−|\+)?\d[\d,&nbsp;]*(\.\d+)?%?/g, m => '<mark>'+m+'</mark>');
  s = s.replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>').replace(/(^|\W)\*([^*]+)\*/g, '$1<i>$2</i>');
  s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
  return s;
}

function renderMd(src){
  const lines = src.split('\n'); let out = '', i = 0;
  while(i < lines.length){
    const line = lines[i];
    if(/^```/.test(line)){
      let buf = []; i++;
      while(i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]);
      i++; out += '<pre>'+esc(buf.join('\n'))+'</pre>'; continue;
    }
    // A table block: consecutive pipe rows. Kept in its own scroller so a wide table
    // never makes the document body scroll sideways.
    if(/^\s*\|/.test(line)){
      let rows = [];
      while(i < lines.length && /^\s*\|/.test(lines[i])) rows.push(lines[i++]);
      const cells = r => r.replace(/^\s*\|/,'').replace(/\|\s*$/,'').split('|');
      const isSep = r => /^[\s|:-]+$/.test(r);
      let head = null, body = rows;
      if(rows.length > 1 && isSep(rows[1])){ head = cells(rows[0]); body = rows.slice(2); }
      let t = '<div class="scroller"><table>';
      if(head) t += '<thead><tr>'+head.map(c=>'<th>'+inline(c.trim())+'</th>').join('')+'</tr></thead>';
      t += '<tbody>'+body.map(r=>'<tr>'+cells(r).map(c=>'<td>'+inline(c.trim())+'</td>').join('')+'</tr>').join('')+'</tbody>';
      out += t + '</table></div>'; continue;
    }
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if(h){ const lv = Math.min(h[1].length,3); out += '<h'+lv+'>'+inline(h[2])+'</h'+lv+'>'; i++; continue; }
    if(/^\s*([-*+]|\d+\.)\s+/.test(line)){
      let items = [];
      while(i < lines.length && /^\s*([-*+]|\d+\.)\s+/.test(lines[i]))
        items.push(lines[i++].replace(/^\s*([-*+]|\d+\.)\s+/,''));
      out += '<ul>'+items.map(x=>'<li>'+inline(x)+'</li>').join('')+'</ul>'; continue;
    }
    if(!line.trim()){ i++; continue; }
    let para = [];
    while(i < lines.length && lines[i].trim() && !/^(#{1,6}\s|\s*\||```|\s*([-*+]|\d+\.)\s)/.test(lines[i]))
      para.push(lines[i++]);
    out += '<p>'+inline(para.join(' '))+'</p>';
  }
  return out;
}

function head(){
  const s = SUMMARY;
  document.getElementById('hdr').innerHTML =
    '<b>DOCUMENT: '+esc(s.docStatus.toUpperCase())+'</b> &nbsp;&middot;&nbsp; '+
    s.pageCount+' pages &nbsp;&middot;&nbsp; '+s.untrusted+' pages with untrusted tables &nbsp;&middot;&nbsp; '+
    s.tableFlags+' table flags'+
    '<div class="sub">'+s.reportedSuccess+' of '+s.pageCount+' page sidecars report '+
    '<b>success</b>'+(s.reportedOther?', '+s.reportedOther+' report something else':'')+
    '; '+s.contradictions+' of those carry recorded warnings. '+
    'A page with no recorded signal is <b>not verified</b> &mdash; it only means nothing was recorded. '+
    '<kbd>w</kbd> warnings &nbsp; <kbd>j</kbd>/<kbd>k</kbd> page &nbsp; <kbd>r</kbd> raw &nbsp; '+
    '<kbd>h</kbd> numbers &nbsp; <kbd>+</kbd>/<kbd>-</kbd> zoom</div>';
  document.documentElement.style.setProperty('--htop', document.getElementById('hdr').offsetHeight+'px');
}

function gone(p){ return p.mdMissing || p.jsonMissing || !p.img; }

function rail(){
  const r = document.getElementById('rail');
  r.innerHTML = PAGES.map((p,ix) => {
    // State is never colour-only: a glyph carries it too, and the aria-label spells it out.
    const missing = gone(p);
    const cls = (p.signals.length?'sus ':'') + (missing?'gone ':'') + (ix===cur?'cur':'');
    const glyph = missing ? '<span class="g">&times;</span>'
                : p.signals.length ? '<span class="g sigct">!'+p.signals.length+'</span>' : '';
    const label = missing ? 'page '+p.n+', evidence missing &mdash; cannot be judged'
                : p.signals.length ? 'page '+p.n+', '+p.signals.length+' recorded signal(s)'
                : 'page '+p.n+', no recorded signal, not verified';
    return '<button data-i="'+ix+'" class="'+cls+'" aria-label="'+label+'"'+
           (ix===cur?' aria-current="page"':'')+'>'+p.n+glyph+'</button>';
  }).join('');
  r.querySelectorAll('button').forEach(b =>
    b.onclick = () => { cur = +b.dataset.i; draw(); });
}

function draw(){
  const p = PAGES[cur];
  document.getElementById('lph').textContent =
    'PDF page '+p.n+' of '+SUMMARY.pageCount+'  &middot;  zoom '+zoom+'%';
  document.getElementById('imgwrap').innerHTML = p.img
    ? '<img style="width:'+zoom+'%" alt="page '+p.n+'" src="data:image/jpeg;base64,'+p.img+'">'
    : '<div class="void">NO IMAGE &mdash; PDF page '+p.n+' could not be rendered'+
      (p.imgErr?' ('+esc(p.imgErr)+')':'')+'</div>';

  const statusTxt = p.status ? 'sidecar: '+esc(p.status) : 'sidecar: (none)';
  document.getElementById('rph').innerHTML =
    '<span class="'+(p.contradicts?'strike':'')+'">'+statusTxt+'</span>'+
    (p.engine?'  &middot;  '+esc(p.engine):'')+(p.provider?' / '+esc(p.provider):'')+
    '  &middot;  $'+(p.cost||0)+(raw?'  &middot;  RAW':'');

  document.getElementById('sigbox').innerHTML = p.signals.length
    ? '<b>'+p.signals.length+' recorded signal(s)</b><br>'+
      p.signals.map(s=>'<span class="chip">'+esc(s)+'</span>').join('')
    : '<b>No recorded signal.</b> This is not a pass &mdash; nothing was recorded for this page.';

  const ex = document.getElementById('extract');
  if(p.mdMissing)      ex.innerHTML = '<div class="void">MISSING &mdash; pages/'+String(p.n).padStart(5,'0')+'.md was never written</div>';
  else if(!p.md || !p.md.trim()) ex.innerHTML = '<div class="void">EMPTY EXTRACT &mdash; the fragment exists but has no content</div>';
  else if(raw)         ex.innerHTML = '<pre>'+esc(p.md)+'</pre>';
  else                 ex.innerHTML = renderMd(p.md);

  document.querySelectorAll('#rail button').forEach((b,ix) => {
    b.classList.toggle('cur', ix===cur);
    if(ix===cur) b.setAttribute('aria-current','page'); else b.removeAttribute('aria-current');
  });
  const active = document.querySelector('#rail button.cur');
  if(active) active.scrollIntoView({block:'nearest'});
}

document.addEventListener('keydown', e => {
  if(e.metaKey||e.ctrlKey||e.altKey) return;
  const k = e.key;
  if(k==='j'||k==='ArrowRight'){ cur = Math.min(cur+1, PAGES.length-1); draw(); }
  else if(k==='k'||k==='ArrowLeft'){ cur = Math.max(cur-1, 0); draw(); }
  else if(k==='n'){ const nx = PAGES.findIndex((p,ix)=>ix>cur && p.signals.length);
                    if(nx>=0){ cur=nx; draw(); } }
  else if(k==='w'){ document.body.classList.toggle('reveal'); }
  else if(k==='r'){ raw = !raw; draw(); }
  else if(k==='h'){ document.body.classList.toggle('nums'); }
  else if(k==='+'||k==='='){ zoom = Math.min(zoom+25, 300); draw(); }
  else if(k==='-'){ zoom = Math.max(zoom-25, 50); draw(); }
  else if(k==='0'){ zoom = 100; draw(); }
});

document.body.classList.add('nums');
head(); rail(); draw();
</script>
"""
