"""GH-367: what counts as disproof of a ``bind()`` contradiction.

``tables/binding.py:bind()`` proves native geometry and emitted markdown
DISAGREE. It does not prove which side is wrong. GH-359 ruling 5 therefore
caps a contradicted table at ``TABLE_UNVERIFIED`` and forbids an ordinary
judge PASS from lifting that cap (the GH-273 failure: two frontier judges
blessed a binding shift). This module is the only path that may lift it.

Disproof is exact identity under bind()'s own normalizers, never a model
verdict. A contradiction is disproved when either:

1. **Encoding garbage (deterministic).** The native token contains a
   codepoint that is not decoded text — the same classes
   ``BornDigitalDetector._garbage_ratio`` already names as garbage
   (control except TAB/LF/CR, U+FFFD, BMP private-use, surrogates).
   Presence of any one such codepoint is enough; there is no ratio.
   Bind() compared a codebook/error string to markdown, so the
   comparison is invalid, not a content disagreement.
2. **Independent raster transcription (constrained model role).** A
   transcriber is shown ONLY the geometry-addressed cell crop. It never sees
   the markdown, the native string, the contradiction, or a PASS/FAIL
   schema. It returns a token. Disproof is: that token agrees with the
   markdown token AND disagrees with the native token, using the same
   ``_normalize_numeric_token`` / ``normalize_label`` bind() used to
   convict. Anything else (empty, infra, agrees with native, third
   string) is not a disproof.

A model that says "the native layer is corrupt" is not disproof — that
reintroduces the fallible-judge override the clamp exists to prevent.
The transcriber does not judge; arithmetic on three strings does.

Per-table, not partial: the constraint is "disproves EACH mechanical
contradiction". One remaining contradiction still withholds acceptance.
A prior lift recorded on the page (sidecar) applies only when the
markdown checksum and the contradiction signature set both match.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Callable, Literal

from socr.tables.binding import BindingResult, ContradictedCell, RowLabelContradiction
from socr.tables.native_rows import normalize_label
from socr.tables.native_verifier import (
    _normalize_numeric_token,
    is_numeric_token,
    strip_math_presentation,
)

DisproofKind = Literal["encoding_garbage", "raster_transcription", "prior_lift"]
ContradictionKind = Literal["cell", "row_label"]

#: BMP private-use area. Same class ``BornDigitalDetector._garbage_ratio``
#: counts as garbage: a ToUnicode hole, not a glyph transcription.
_PUA_BMP_START = 0xE000
_PUA_BMP_END = 0xF8FF
#: Unicode replacement character — the decoder's explicit "I do not know".
_REPLACEMENT_CHAR = 0xFFFD
#: UTF-16 surrogates; must not appear in valid decoded text.
_SURROGATE_START = 0xD800
_SURROGATE_END = 0xDFFF
#: Control characters that are not TAB / LF / CR. Same exclusion as
#: ``BornDigitalDetector._garbage_ratio``.
_TAB, _LF, _CR = 0x09, 0x0A, 0x0D


@dataclass(frozen=True)
class ContradictionItem:
    """One ``bind()`` conviction, with enough identity to resume against."""

    kind: ContradictionKind
    row_path: tuple[str, ...]
    col_path: tuple[str, ...]
    native_token: str
    model_token: str
    native_bbox: tuple[float, float, float, float] | None = None
    cell_bbox: tuple[float, float, float, float] | None = None
    address_source: str | None = None
    abstain_reason: str | None = None

    def signature(self) -> tuple[str, tuple[str, ...], tuple[str, ...], str, str]:
        return (self.kind, self.row_path, self.col_path, self.native_token, self.model_token)

    def to_record(self, disproof: DisproofKind | None) -> dict:
        return {
            "kind": self.kind,
            "row_path": list(self.row_path),
            "col_path": list(self.col_path),
            "native_token": self.native_token,
            "model_token": self.model_token,
            "disproof": disproof,
            "native_bbox": self.native_bbox,
            "cell_bbox": self.cell_bbox,
            "address_source": self.address_source,
            "abstain_reason": self.abstain_reason,
            "outcome": ItemOutcome(self, disproof).outcome,
        }


@dataclass(frozen=True)
class ItemOutcome:
    item: ContradictionItem
    disproof: DisproofKind | None

    @property
    def outcome(self) -> Literal["disproved", "held", "abstained"]:
        if self.disproof is not None:
            return "disproved"
        return "abstained" if self.item.cell_bbox is None else "held"


@dataclass(frozen=True)
class AdjudicationRecord:
    """JSON-ready per-table record persisted on the page sidecar."""

    status: Literal["lifted", "held"]
    markdown_sha256: str
    items: tuple[ItemOutcome, ...]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "markdown_sha256": self.markdown_sha256,
            "signatures": [list(outcome.item.signature()) for outcome in self.items],
            "items": [outcome.item.to_record(outcome.disproof) for outcome in self.items],
        }


def markdown_sha256(markdown: str) -> str:
    return hashlib.sha256(markdown.encode("utf-8")).hexdigest()


def token_is_encoding_garbage(token: str) -> bool:
    """True when *token* contains a codepoint that is not decoded text.

    Same character classes ``BornDigitalDetector._garbage_ratio`` already
    names as garbage. Boolean per token, not a ratio: one such codepoint
    means bind()'s native side is a codebook/error string.
    """
    for ch in token:
        cp = ord(ch)
        if cp < 0x20 and cp not in (_TAB, _LF, _CR):
            return True
        if cp == _REPLACEMENT_CHAR:
            return True
        if _PUA_BMP_START <= cp <= _PUA_BMP_END:
            return True
        if _SURROGATE_START <= cp <= _SURROGATE_END:
            return True
    return False


def tokens_agree(left: str, right: str, *, kind: ContradictionKind) -> bool:
    """Same equality bind() used to convict, including empty-key refusal."""
    if kind == "row_label":
        left_key = normalize_label(strip_math_presentation(left, label=True))
        right_key = normalize_label(strip_math_presentation(right, label=True))
        return bool(left_key) and left_key == right_key
    if is_numeric_token(left) and is_numeric_token(right):
        return _normalize_numeric_token(left) == _normalize_numeric_token(right)
    return left.strip() == right.strip()


def items_from_binding(result: BindingResult) -> tuple[ContradictionItem, ...]:
    items: list[ContradictionItem] = []
    for cell in result.contradicted_cells:
        items.append(_item_from_cell(cell))
    for row in result.row_label_contradictions:
        items.append(_item_from_row_label(row))
    return tuple(items)


def adjudication_text_lines(page, region) -> list[dict]:
    """Expose the geometry lane's lines for the adjudicator's column test.

    Keep locate's private representation inside the tables package; the
    orchestrator must use the same lines that established the row bands.
    """
    from socr.tables.locate import _text_lines_in_region

    return _text_lines_in_region(page, region)


def _item_from_cell(cell: ContradictedCell) -> ContradictionItem:
    return ContradictionItem(
        kind="cell",
        row_path=cell.row_path,
        col_path=cell.col_path,
        native_token=cell.native_token,
        model_token=cell.model_token or "",
        native_bbox=cell.native_bbox,
    )


def _item_from_row_label(row: RowLabelContradiction) -> ContradictionItem:
    native = row.row_path[-1] if row.row_path else ""
    return ContradictionItem(
        kind="row_label",
        row_path=row.row_path,
        col_path=(),
        native_token=native,
        model_token=row.candidate_label,
        native_bbox=row.native_bbox,
    )


def prior_lift_applies(prior: object, markdown: str, items: tuple[ContradictionItem, ...]) -> bool:
    """True when a sidecar record is a lift of *exactly* these contradictions."""
    if not isinstance(prior, dict) or prior.get("status") != "lifted":
        return False
    if not items:
        return False
    if prior.get("markdown_sha256") != markdown_sha256(markdown):
        return False
    raw_sigs = prior.get("signatures")
    if not isinstance(raw_sigs, list):
        return False
    try:
        # GH-388 review (cubic P1): SORTED SEQUENCES, not sets. Two
        # contradictions can share a signature -- the same native/model token
        # pair at two loci that normalize alike -- and a set collapses them.
        # A resumed run would then match a prior lift of ONE against a current
        # set of TWO and reuse it, clearing a contradiction nothing ever
        # disproved. The lift must require every contradiction, not every
        # distinct signature.
        prior_sigs = sorted(tuple(_coerce_signature(sig)) for sig in raw_sigs)
    except (TypeError, ValueError):
        return False
    current_sigs = sorted(item.signature() for item in items)
    return prior_sigs == current_sigs


def _coerce_signature(raw: object) -> tuple[str, tuple[str, ...], tuple[str, ...], str, str]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 5:
        raise ValueError("bad signature")
    kind, row_path, col_path, native, model = raw
    if kind not in ("cell", "row_label"):
        raise ValueError("bad kind")
    if not isinstance(row_path, (list, tuple)) or not isinstance(col_path, (list, tuple)):
        raise ValueError("bad paths")
    return (
        str(kind),
        tuple(str(p) for p in row_path),
        tuple(str(p) for p in col_path),
        str(native),
        str(model),
    )


def adjudicate(
    items: tuple[ContradictionItem, ...],
    *,
    markdown: str,
    prior: object = None,
    transcribe: Callable[[tuple[float, float, float, float]], str | None] | None = None,
) -> AdjudicationRecord:
    """Decide whether *every* contradiction is independently disproved.

    ``transcribe`` is given a geometry cell bbox and returns a token or None.
    It is not called for encoding-garbage items or when a matching prior
    lift already covers the whole set. A missing transcriber (None) is
    absence of evidence, not a conviction of either side.
    """
    digest = markdown_sha256(markdown)
    if not items:
        return AdjudicationRecord(status="held", markdown_sha256=digest, items=())

    items = tuple(
        replace(item, abstain_reason=item.abstain_reason or "no geometry address")
        if item.cell_bbox is None and not token_is_encoding_garbage(item.native_token)
        else item
        for item in items
    )
    if all(
        item.cell_bbox is not None or token_is_encoding_garbage(item.native_token) for item in items
    ) and prior_lift_applies(prior, markdown, items):
        outcomes = tuple(ItemOutcome(item=item, disproof="prior_lift") for item in items)
        return AdjudicationRecord(status="lifted", markdown_sha256=digest, items=outcomes)

    outcomes: list[ItemOutcome] = []
    for item in items:
        disproof = _disprove_one(item, transcribe)
        outcomes.append(ItemOutcome(item=item, disproof=disproof))

    status: Literal["lifted", "held"] = (
        "lifted" if outcomes and all(o.disproof is not None for o in outcomes) else "held"
    )
    return AdjudicationRecord(status=status, markdown_sha256=digest, items=tuple(outcomes))


def _disprove_one(
    item: ContradictionItem,
    transcribe: Callable[[tuple[float, float, float, float]], str | None] | None,
) -> DisproofKind | None:
    if token_is_encoding_garbage(item.native_token):
        return "encoding_garbage"
    if transcribe is None or item.cell_bbox is None:
        return None
    token = transcribe(item.cell_bbox)
    if not token:
        return None
    if not tokens_agree(token, item.model_token, kind=item.kind):
        return None
    if tokens_agree(token, item.native_token, kind=item.kind):
        return None
    return "raster_transcription"
