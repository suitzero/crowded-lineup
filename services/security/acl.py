"""Principal identity, per-document ACL enforcement, and pre-filter builders."""

from __future__ import annotations

from enum import IntEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ClearanceLevel(IntEnum):
    PUBLIC = 0
    TEAM_INTERNAL = 1
    CONFIDENTIAL = 2
    STEWARD_RESTRICTED = 3


class PrincipalContext(BaseModel):
    principal_id: str
    tenant_id: str
    roles: list[str] = Field(default_factory=list)
    clearance: ClearanceLevel = ClearanceLevel.PUBLIC
    agent_id: str | None = None
    trace_id: str = Field(default_factory=lambda: uuid4().hex[:16])


class DocumentAclMeta(BaseModel):
    doc_id: str
    tenant_id: str
    classification: ClearanceLevel = ClearanceLevel.PUBLIC
    allowed_principals: list[str] = Field(default_factory=list)
    is_public: bool = False


DEV_PRINCIPALS: dict[str, PrincipalContext] = {
    "alice_alpha": PrincipalContext(
        principal_id="alice_alpha",
        tenant_id="team_alpha",
        roles=["analyst", "coach"],
        clearance=ClearanceLevel.CONFIDENTIAL,
    ),
    "bob_beta": PrincipalContext(
        principal_id="bob_beta",
        tenant_id="team_beta",
        roles=["analyst", "coach"],
        clearance=ClearanceLevel.CONFIDENTIAL,
    ),
    "steward_lead": PrincipalContext(
        principal_id="steward_lead",
        tenant_id="public_league",
        roles=["steward", "judge"],
        clearance=ClearanceLevel.STEWARD_RESTRICTED,
    ),
    "guest_public": PrincipalContext(
        principal_id="guest_public",
        tenant_id="public_league",
        roles=["viewer"],
        clearance=ClearanceLevel.PUBLIC,
    ),
}


def resolve_principal(principal_id: str | None, agent_id: str | None = None) -> PrincipalContext:
    """Resolve a caller principal from dev directory or fail closed to public guest."""
    pid = (principal_id or "guest_public").strip()
    base = DEV_PRINCIPALS.get(pid)
    if base is None:
        return PrincipalContext(
            principal_id=pid,
            tenant_id="unknown_untrusted",
            roles=[],
            clearance=ClearanceLevel.PUBLIC,
            agent_id=agent_id,
        )
    return base.model_copy(update={"agent_id": agent_id, "trace_id": uuid4().hex[:16]})


def can_principal_access(principal: PrincipalContext, acl: DocumentAclMeta) -> tuple[bool, str]:
    """Evaluate per-document ACL policy before returning any SQL row, chunk, or blob."""
    if acl.is_public and acl.classification == ClearanceLevel.PUBLIC:
        return True, "public_document"

    if (
        principal.principal_id in acl.allowed_principals
        and principal.clearance >= acl.classification
    ):
        return True, "explicit_principal_allowlist"

    if principal.tenant_id != acl.tenant_id:
        return False, f"tenant_mismatch:{principal.tenant_id}!={acl.tenant_id}"

    if principal.clearance < acl.classification:
        return False, f"insufficient_clearance:{principal.clearance}<{acl.classification}"

    if acl.allowed_principals and principal.principal_id not in acl.allowed_principals:
        return False, f"principal_not_in_allowlist:{principal.principal_id}"

    return True, "tenant_and_clearance_match"


def build_acl_prefilter(principal: PrincipalContext) -> dict[str, Any]:
    """Build structured pre-filter parameters for vector/search index and Postgres RLS."""
    return {
        "principal_id": principal.principal_id,
        "tenant_id": principal.tenant_id,
        "max_clearance": int(principal.clearance),
        "allow_public": True,
    }
