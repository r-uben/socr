"""GH-225: deterministic, model-free provenance gate for URLs in emitted text.

Every other quality gate in this pipeline checks for LOSS — a dropped number, a
flattened table, a discarded heading.  This one checks for FABRICATION: an image
reference the model invented, shipped as page content with no marker separating
it from real output.  On the OBR Nov-2022 EFO run two pages carried
``![](https://i.imgur.com/…​.png)`` refs; the source is a UK government fiscal
document with no external image links at all, and one of those pages recorded
zero audit events.

A hyperlink is text, so it passes every gate the pipeline has: table geometry,
numeric-token preservation, word recall, structural shape.  And
``OutputNormalizer.strip_phantom_images`` explicitly keeps ``http(s)`` / ``data:``
refs — correctly, because a pure text normalizer has no way to tell a real
external image from an invented one.  Provenance is not a property of the text;
it is a relation between the text and the source document.  So the check lives
here, where the PDF is in scope, rather than in the normalizer.

The rule is derived from the document, never from a host allowlist:

* A **local asset** — a path that resolves to a file that exists under the
  document's own output directory — is legitimate: socr wrote it.
* An **http(s) URL that the source PDF itself contains** — in a link annotation
  or in its native text layer — is legitimate: the model transcribed it.
* Anything else is fabricated by construction.  That includes every ``data:``
  URI: a PDF cannot carry one in a link annotation, socr never emits one into
  page markdown (they are only ever sent TO a model), and a language model
  cannot produce real image bytes.

Reachability is deliberately NOT consulted.  A live URL is not a provenanced
one, and a network fetch would turn an attacker-controlled or coincidentally
registered host into a laundering channel for invented content.  No request is
ever made from this module.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ``![alt](target)`` — the same shape OutputNormalizer strips phantoms with.
_RE_MD_IMAGE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<target>[^)]+)\)")

# A URL as it appears in running text.  Deliberately permissive on the left
# (any scheme we could see in a PDF text layer) and greedy-but-bounded on the
# right; trailing sentence punctuation is trimmed by ``normalize_url``.
_RE_URL_IN_TEXT = re.compile(r"(?:https?|ftp)://[^\s<>\"'\]\)]+", re.IGNORECASE)

# Schemes that make a reference ABSOLUTE — i.e. not a path we can resolve
# against the document's own asset directory.  These are the refs this gate
# adjudicates; everything else is a path and stays with strip_phantom_images.
_ABSOLUTE_SCHEMES = ("http://", "https://", "data:", "file://", "ftp://")

# The text left in place of a removed reference.  Fixed, greppable, and it
# names WHY rather than merely that something was dropped: a reader who greps
# the corpus for this string finds every page where the model invented an
# asset.  The invented URL and any alt text ride in the audit event instead of
# here — an alt text is a caption for a figure that does not exist, so keeping
# it as prose would keep the fabrication while only removing its pointer.
FABRICATED_IMAGE_MARKER = (
    "[socr: fabricated image reference removed — not present in the source document]"
)


def normalize_url(raw: str) -> str:
    """Canonical form for comparing a URL in output against one in the source.

    Case-folds the scheme and host (both case-insensitive per RFC 3986) while
    leaving the path alone (it is case-SENSITIVE, and folding it would let
    ``/Report.pdf`` launder as ``/report.pdf``).  Trims the angle brackets and
    trailing sentence punctuation a text layer wraps URLs in, and drops the
    fragment, which never reaches a server and so cannot distinguish two refs.
    """
    s = raw.strip().strip("<>").strip()
    s = s.split("#", 1)[0]
    s = s.rstrip(".,;:!?)’'\"")
    if not s:
        return ""
    sep = "://"
    idx = s.find(sep)
    if idx == -1:
        return s
    scheme = s[:idx].lower()
    rest = s[idx + len(sep) :]
    slash = rest.find("/")
    if slash == -1:
        host, path = rest, ""
    else:
        host, path = rest[:slash], rest[slash:]
    path = path.rstrip("/")
    return f"{scheme}://{host.lower()}{path}"


def source_url_index(pdf_path: Path) -> frozenset[str]:
    """Every URL the SOURCE PDF itself contains, normalized for comparison.

    Two independent origins, unioned because either one alone has a blind spot:
    link annotations catch a clickable URL whose visible text is a caption
    ("see the OBR website"), and the native text layer catches a URL that is
    printed but not hyperlinked.  A scanned page contributes nothing from
    either, which is correct — a scan genuinely has no provenanced URLs, so
    every URL a VLM emits for it is fabricated.

    Best-effort by design: an unreadable or encrypted PDF yields the empty set,
    which makes the gate MAXIMALLY strict (nothing is provenanced) rather than
    silently permissive.  Failing open here would reintroduce exactly the
    silent-fabrication hole this module closes.
    """
    urls: set[str] = set()
    try:
        from socr.core.pdf import open_pdf

        with open_pdf(pdf_path) as doc:
            for page in doc:
                try:
                    for link in page.get_links() or []:
                        uri = link.get("uri") or ""
                        if uri:
                            norm = normalize_url(uri)
                            if norm:
                                urls.add(norm)
                except Exception as exc:  # one bad page must not blind the rest
                    logger.debug("GH-225: link annotations unreadable (%s)", exc)
                try:
                    layer = page.get_text() or ""
                except Exception as exc:
                    logger.debug("GH-225: text layer unreadable (%s)", exc)
                    continue
                for match in _RE_URL_IN_TEXT.finditer(layer):
                    norm = normalize_url(match.group(0))
                    if norm:
                        urls.add(norm)
    except Exception as exc:
        logger.debug("GH-225: could not index source URLs for %s (%s)", pdf_path, exc)
    return frozenset(urls)


def _is_local_asset(target: str, doc_dir: Path | None) -> bool:
    """True when *target* is a file socr itself wrote under the document's dir."""
    if doc_dir is None:
        return False
    try:
        p = Path(target)
        candidate = p if p.is_absolute() else doc_dir / p
        return candidate.is_file()
    except (OSError, ValueError):
        return False


def redact_fabricated_image_refs(
    text: str,
    *,
    source_urls: frozenset[str],
    doc_dir: Path | None = None,
) -> tuple[str, list[dict]]:
    """Replace image refs with no source provenance by an explicit marker.

    Returns ``(new_text, removed)`` where each ``removed`` entry records the
    invented target and its alt text for the audit event.  ``new_text is text``
    (unchanged) when nothing was fabricated, so a clean page costs nothing.

    Scope is markdown IMAGE refs only.  An inline LINK (``[label](url)``) is
    deliberately left alone: removing one would delete the label, which is real
    prose, and a URL a model mis-transcribed by one character is a corrupted
    reference rather than an invented asset.  An image ref has no such risk —
    it is a pure pointer, and a pointer to nothing carries no content.
    """
    if "![" not in text:
        return text, []

    removed: list[dict] = []

    def _replace(match: re.Match) -> str:
        target = (match.group("target") or "").strip().strip("<>")
        lowered = target.lower()
        if not lowered.startswith(_ABSOLUTE_SCHEMES):
            # A path, not an absolute reference: strip_phantom_images owns it
            # (it deletes the ones that do not resolve), and one that DOES
            # resolve is an asset socr wrote.
            return match.group(0)
        if _is_local_asset(target, doc_dir):
            return match.group(0)
        if lowered.startswith(("http://", "https://", "ftp://")):
            if normalize_url(target) in source_urls:
                return match.group(0)
        removed.append({"target": target, "alt": match.group("alt") or ""})
        return FABRICATED_IMAGE_MARKER

    new_text = _RE_MD_IMAGE.sub(_replace, text)
    if not removed:
        return text, []
    return new_text, removed
