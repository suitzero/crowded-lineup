"""LLM provider abstraction for Steward Intelligence (Gemini HTTP + Deterministic CI Mock)."""

from __future__ import annotations

from typing import Protocol

import httpx

from services.api.agent.tools import Citation
from services.security.acl import PrincipalContext


class StewardLLMClient(Protocol):
    def synthesize(
        self,
        query: str,
        principal: PrincipalContext,
        hits_summary: list[str],
        heat_summary: dict[str, object] | None,
        leaderboard_summary: list[str],
        citations: list[Citation],
    ) -> str: ...


class DeterministicStewardLLM:
    """Zero-dependency deterministic synthesizer grounded strictly in authorized tool outputs."""

    def synthesize(
        self,
        query: str,
        principal: PrincipalContext,
        hits_summary: list[str],
        heat_summary: dict[str, object] | None,
        leaderboard_summary: list[str],
        citations: list[Citation],
    ) -> str:
        parts = [
            f"[Steward Analyst • Principal={principal.principal_id} ({principal.tenant_id})]",
        ]
        if hits_summary:
            parts.append("Authorized Corpus Evidence:")
            for line in hits_summary:
                parts.append(f"  - {line}")
        else:
            parts.append(
                "Authorized Corpus Evidence: No matching documents were authorized or found for this principal."
            )

        if heat_summary and heat_summary.get("found"):
            parts.append(
                f"Heat Telemetry ({heat_summary['heat_id']}): "
                f"{heat_summary['num_surfers']} surfers, "
                f"{heat_summary['total_waves']} waves, "
                f"{heat_summary['total_drop_ins']} drop-ins, "
                f"{heat_summary['total_collisions']} collisions."
            )

        if leaderboard_summary:
            parts.append("Top Leaderboard Agents: " + " | ".join(leaderboard_summary[:3]))

        if citations:
            refs = ", ".join(
                f"{c.source_id}" + (f" (p.{c.page_number})" if c.page_number else "")
                for c in citations
            )
            parts.append(f"Citations: {refs}")

        return "\n".join(parts)


class GeminiHttpStewardClient:
    """Gemini REST API client grounded exclusively in ACL-filtered context."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key
        self.model = model
        self._fallback = DeterministicStewardLLM()

    def synthesize(
        self,
        query: str,
        principal: PrincipalContext,
        hits_summary: list[str],
        heat_summary: dict[str, object] | None,
        leaderboard_summary: list[str],
        citations: list[Citation],
    ) -> str:
        prompt = (
            "You are the Crowded Lineup Steward & Strategy Analyst. "
            "Answer concisely using ONLY the authorized context below and cite doc_ids and page numbers.\n\n"
            f"Caller: {principal.principal_id} (tenant={principal.tenant_id})\n"
            f"Question: {query}\n"
            f"Authorized Hits:\n" + "\n".join(hits_summary) + "\n"
            f"Heat Summary: {heat_summary}\n"
            f"Leaderboard: {leaderboard_summary}\n"
        )
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(
                    url,
                    params={"key": self.api_key},
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return str(text)
        except Exception:
            return self._fallback.synthesize(
                query, principal, hits_summary, heat_summary, leaderboard_summary, citations
            )
