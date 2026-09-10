"""#713: the acceptance credential that can stand in for a timed-out page judge.

The bug this exists for: a candidate whose EVERY emitted table the table judge
ladder positively accepted, but whose PAGE-level judge call timed out, records
``audit_passed=False`` and is therefore refused by every strict selection pool.
The page then ships the fail-closed floor -- a verified reading turned into
total loss (BoE 2018 p1, #713).

Astra's ruling (2026-09-10) admits that candidate, demoted, ONLY against a
credential that is a positive proof rather than the absence of an adverse one.
"The audit log has a ``table_ladder_accepted`` event for this page" is not that
proof: the event names a table id, a rung trail and a witness scope, none of
which binds the surrounding PROSE, another attempt's bytes, or an edit made
after the verdict. So the credential binds, in one record:

* the exact candidate bytes evaluated (SHA-256 of the complete canonical
  page-candidate text, not a per-table markdown identity),
* the document checksum and the page number,
* the attempt identity (engine + provider id/model/backend),
* one entry per emitted table, each with the table's own markdown digest, the
  witness IMAGE's digest and scope, and the rung identities that executed,
* the judge/provider/configuration provenance under which the ladder ran.

Verification recomputes every one of those from what is about to ship. Any
mismatch -- one byte of the candidate, a different document, a table the
credential does not cover, a witness that was never rendered -- withholds the
stand-in and the page fails closed under its own reason.

Nothing here decides policy. ``manifest`` decides whether a verified credential
is enough (it also requires the page to carry no adverse ladder terminal and no
outstanding hard contradiction); this module only says whether the credential
truthfully describes the bytes in hand.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

#: Bumped whenever the bound fields change. A credential whose schema is not
#: this exact string is not verifiable by this code and is refused -- never
#: "best-effort" parsed. Widening what a credential proves must invalidate
#: every credential minted under the narrower promise.
CREDENTIAL_SCHEMA = "socr.page_judge_timeout_credential.v1"


def sha256_text(text: str) -> str:
    """SHA-256 of ``text`` as UTF-8, ``sha256:``-prefixed."""
    return "sha256:" + hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    """SHA-256 of a file's bytes, ``sha256:``-prefixed; ``""`` if unreadable.

    An unreadable witness image yields "" rather than a digest of nothing, and
    "" never verifies -- a credential with no witness identity for a table is
    exactly as inadmissible as one whose witness digest disagrees.
    """
    try:
        return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


@dataclass(frozen=True)
class TableAcceptance:
    """One emitted table the ladder ACCEPTED, and what was looked at to accept it."""

    #: ``p{page}-t{idx}`` -- the same id ``tables.witness`` assigns, derived
    #: from reading-order position among the candidate's own table blocks.
    table_id: str
    #: Digest of this table block's markdown as emitted by the candidate.
    markdown_sha256: str
    #: Digest of the witness IMAGE the ladder judged. "" means no image was
    #: rendered, which is never an acceptance this credential may record.
    witness_sha256: str
    #: ``located`` / ``page`` / ``none`` (``tables.witness.WitnessScope``).
    witness_scope: str
    #: The executing identity of every rung that ran for this table, in call
    #: order. Empty means nothing looked -- inadmissible.
    rungs: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "markdown_sha256": self.markdown_sha256,
            "witness_sha256": self.witness_sha256,
            "witness_scope": self.witness_scope,
            "rungs": list(self.rungs),
        }

    @classmethod
    def from_dict(cls, d: dict) -> TableAcceptance:
        return cls(
            table_id=str(d.get("table_id", "")),
            markdown_sha256=str(d.get("markdown_sha256", "")),
            witness_sha256=str(d.get("witness_sha256", "")),
            witness_scope=str(d.get("witness_scope", "")),
            rungs=tuple(str(r) for r in (d.get("rungs") or [])),
        )


@dataclass(frozen=True)
class TableAcceptanceCredential:
    """A page-scoped, attempt-bound proof that the ladder accepted every table."""

    page_num: int
    document_checksum: str
    #: SHA-256 of the COMPLETE canonical candidate text the ladder was run
    #: against -- the whole page, not the table blocks. This is what stops the
    #: credential from vouching for prose it never saw, for a sibling attempt,
    #: or for a post-judgment edit.
    candidate_sha256: str
    attempt_engine: str
    attempt_provider_id: str
    attempt_provider_model: str
    attempt_provider_backend: str
    #: The page judge's identity and the TYPED outcome it produced. A
    #: credential is minted only alongside a typed timeout, and verification
    #: re-checks that the attempt still carries that same outcome.
    judge_model: str
    judge_outcome: str
    #: The run fingerprint under which the ladder ran (binds rung identities,
    #: prompts, render DPI and every ladder flag in one value).
    run_fingerprint: str
    tables: tuple[TableAcceptance, ...] = ()
    schema: str = CREDENTIAL_SCHEMA
    #: Set at FINALIZATION, not at mint: the digest of the exact body that
    #: shipped (the candidate plus whatever flag note the manifest appended).
    #: "" on a freshly minted credential. Resume verifies the fragment on disk
    #: against this, which is the only way to check body identity for a body
    #: the credential's own ``candidate_sha256`` deliberately predates.
    finalized_sha256: str = ""

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "page_num": self.page_num,
            "document_checksum": self.document_checksum,
            "candidate_sha256": self.candidate_sha256,
            "attempt_engine": self.attempt_engine,
            "attempt_provider_id": self.attempt_provider_id,
            "attempt_provider_model": self.attempt_provider_model,
            "attempt_provider_backend": self.attempt_provider_backend,
            "judge_model": self.judge_model,
            "judge_outcome": self.judge_outcome,
            "run_fingerprint": self.run_fingerprint,
            "finalized_sha256": self.finalized_sha256,
            "tables": [t.to_dict() for t in self.tables],
        }

    @classmethod
    def from_dict(cls, d: dict) -> TableAcceptanceCredential | None:
        """Parse a persisted credential, or ``None`` if it is not one.

        Deliberately total and silent: a sidecar is untrusted parsed JSON, and
        every unparseable shape must reach the caller as "no credential", which
        fails closed, rather than as an exception that a broad ``except`` up the
        stack could turn into something else.
        """
        if not isinstance(d, dict) or d.get("schema") != CREDENTIAL_SCHEMA:
            return None
        try:
            return cls(
                page_num=int(d["page_num"]),
                document_checksum=str(d.get("document_checksum", "")),
                candidate_sha256=str(d.get("candidate_sha256", "")),
                attempt_engine=str(d.get("attempt_engine", "")),
                attempt_provider_id=str(d.get("attempt_provider_id", "")),
                attempt_provider_model=str(d.get("attempt_provider_model", "")),
                attempt_provider_backend=str(d.get("attempt_provider_backend", "")),
                judge_model=str(d.get("judge_model", "")),
                judge_outcome=str(d.get("judge_outcome", "")),
                run_fingerprint=str(d.get("run_fingerprint", "")),
                finalized_sha256=str(d.get("finalized_sha256", "")),
                tables=tuple(
                    TableAcceptance.from_dict(t)
                    for t in (d.get("tables") or [])
                    if isinstance(t, dict)
                ),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class CredentialVerdict:
    """Why a credential was or was not admitted. ``reason`` is "" when valid."""

    valid: bool
    reason: str = ""
    covered_tables: tuple[str, ...] = field(default=())


def _table_blocks_of(text: str) -> list[tuple[str, str]]:
    """``(table_id, block_markdown)`` for every table block the text emits.

    Reproduces ``tables.witness.prepare_table_witnesses``' own enumeration --
    ``find_table_blocks`` in reading order, ids ``p{page}-t{idx}`` -- so
    coverage is checked against the same blocks the ladder was handed, without
    needing the PDF. The page number is supplied by the caller because the id
    carries it.
    """
    from socr.tables.reconcile import find_table_blocks

    lines = (text or "").splitlines()
    blocks = find_table_blocks(text or "")
    return [(str(idx), "\n".join(lines[b.start : b.end + 1])) for idx, b in enumerate(blocks)]


def verify_credential(
    credential: dict | TableAcceptanceCredential | None,
    *,
    candidate_text: str,
    finalized: bool = False,
    page_num: int,
    document_checksum: str,
    engine: str,
    provider_id: str,
    provider_model: str,
    provider_backend: str,
    judge_outcome: str,
    run_fingerprint: str | None = None,
) -> CredentialVerdict:
    """Whether ``credential`` truthfully describes THIS candidate.

    Every bound field is recomputed from the caller's own view of what is about
    to ship; nothing is taken on the credential's word. In particular the
    coverage check enumerates the candidate's table blocks itself and requires
    an accepted entry, with a matching markdown digest and a real witness, for
    EVERY one of them -- a credential that covers two of three tables is not a
    partial credential, it is an invalid one.

    ``run_fingerprint`` is checked only when the caller supplies it: selection
    inside a live run has it, and a caller that does not know the run's identity
    must not be able to satisfy the check by omitting it either way.

    ``finalized=True`` says ``candidate_text`` is the SHIPPED body rather than the
    judged candidate -- the same bytes plus the disclosure notes finalization
    appends. The body digest is then checked against ``finalized_sha256``, which
    is stamped at finalization for exactly this, instead of against
    ``candidate_sha256``, which describes bytes that deliberately predate the
    notes. Coverage is unaffected: the notes are prose lines appended after the
    last table, so the block enumeration and its indices are identical either way.
    An empty ``finalized_sha256`` never verifies -- a credential that was never
    finalized cannot vouch for a finalized body.
    """
    cred = (
        credential
        if isinstance(credential, TableAcceptanceCredential)
        else TableAcceptanceCredential.from_dict(credential)
        if isinstance(credential, dict)
        else None
    )
    if cred is None:
        return CredentialVerdict(False, "no credential")
    if cred.schema != CREDENTIAL_SCHEMA:
        return CredentialVerdict(False, "credential schema mismatch")
    if cred.page_num != page_num:
        return CredentialVerdict(False, "credential page number mismatch")
    if not cred.document_checksum or cred.document_checksum != (document_checksum or ""):
        return CredentialVerdict(False, "credential document checksum mismatch")
    if finalized:
        if not cred.finalized_sha256 or cred.finalized_sha256 != sha256_text(candidate_text):
            return CredentialVerdict(False, "credential finalized body mismatch")
    elif cred.candidate_sha256 != sha256_text(candidate_text):
        return CredentialVerdict(False, "credential candidate bytes mismatch")
    if (
        cred.attempt_engine != (engine or "")
        or cred.attempt_provider_id != (provider_id or "")
        or cred.attempt_provider_model != (provider_model or "")
        or cred.attempt_provider_backend != (provider_backend or "")
    ):
        return CredentialVerdict(False, "credential attempt identity mismatch")
    if not cred.judge_outcome or cred.judge_outcome != (judge_outcome or ""):
        return CredentialVerdict(False, "credential judge outcome mismatch")
    if run_fingerprint is not None and cred.run_fingerprint != run_fingerprint:
        return CredentialVerdict(False, "credential run fingerprint mismatch")

    blocks = _table_blocks_of(candidate_text)
    if not blocks:
        # A credential vouches for TABLES. A candidate that emits none has
        # nothing for the ladder to have accepted, so the stand-in has no
        # evidence at all -- not "trivially satisfied".
        return CredentialVerdict(False, "candidate emits no table to vouch for")
    by_id = {t.table_id: t for t in cred.tables}
    covered: list[str] = []
    for idx, block_md in blocks:
        table_id = f"p{page_num}-t{idx}"
        entry = by_id.get(table_id)
        if entry is None:
            return CredentialVerdict(False, f"credential does not cover {table_id}")
        if entry.markdown_sha256 != sha256_text(block_md):
            return CredentialVerdict(False, f"credential table bytes mismatch for {table_id}")
        if not entry.witness_sha256 or entry.witness_scope in ("", "none"):
            return CredentialVerdict(False, f"credential has no witness for {table_id}")
        if not entry.rungs:
            return CredentialVerdict(False, f"credential records no rung for {table_id}")
        covered.append(table_id)
    if len(by_id) != len(covered):
        # The credential vouches for a table this candidate does not emit, so
        # it was minted against different bytes than the ones in hand.
        return CredentialVerdict(False, "credential covers a table the candidate does not emit")
    return CredentialVerdict(True, "", tuple(covered))
