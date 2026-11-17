"""FastAPI dependencies for caller principal resolution and container access."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request

from services.api.state import ServiceContainer
from services.security.acl import PrincipalContext, resolve_principal


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container  # type: ignore[no-any-return]


def get_principal(
    x_principal_id: str | None = Header(default=None, alias="X-Principal-Id"),
    x_agent_id: str | None = Header(default=None, alias="X-Agent-Id"),
) -> PrincipalContext:
    return resolve_principal(x_principal_id, agent_id=x_agent_id)


ContainerDep = Annotated[ServiceContainer, Depends(get_container)]
PrincipalDep = Annotated[PrincipalContext, Depends(get_principal)]
