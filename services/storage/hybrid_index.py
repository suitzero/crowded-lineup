"""Hybrid BM25-style lexical + 64D vector index with strict pre-filter per-doc ACL."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from pydantic import BaseModel, Field

from services.security.acl import (
    DocumentAclMeta,
    PrincipalContext,
    build_acl_prefilter,
    can_principal_access,
)
from services.security.audit import AuditLogger

EMBEDDING_DIM = 64
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def hash_embed(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Deterministic local n-gram feature-hashing embedding normalized to unit L2 norm."""
    vec = [0.0] * dim
    tokens = tokenize(text)
    if not tokens:
        return vec
    for tok in tokens:
        h = int(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
        vec[idx] += sign
        # Add character trigrams for robust partial matching
        for i in range(max(0, len(tok) - 2)):
            tri = tok[i : i + 3]
            th = int(hashlib.blake2b(tri.encode("utf-8"), digest_size=8).hexdigest(), 16)
            vec[th % dim] += 0.35 * (1.0 if ((th >> 8) & 1) == 0 else -1.0)
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [round(v / norm, 6) for v in vec]
    return vec


def cosine_sim(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


class IndexedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    page_number: int
    content: str
    tags: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)
    acl: DocumentAclMeta
    source_uri: str
    custodian: str | None = None


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    page_number: int
    snippet: str
    score: float
    source_uri: str
    tags: list[str] = Field(default_factory=list)


class HybridAclSearchIndex:
    """Hybrid lexical + vector store that applies ACL pre-filtering BEFORE ranking."""

    def __init__(self, audit_logger: AuditLogger) -> None:
        self._chunks: dict[str, IndexedChunk] = {}
        self.audit = audit_logger

    def upsert_chunks(self, chunks: list[IndexedChunk]) -> int:
        for ch in chunks:
            if not ch.embedding:
                ch.embedding = hash_embed(ch.content)
            self._chunks[ch.chunk_id] = ch
        return len(chunks)

    def count(self) -> int:
        return len(self._chunks)

    def search(
        self,
        query: str,
        principal: PrincipalContext,
        top_k: int = 5,
        doc_id_filter: str | None = None,
    ) -> list[SearchHit]:
        """Execute pre-filtered hybrid search and emit mandatory audit log entries."""
        prefilter = build_acl_prefilter(principal)
        q_tokens = tokenize(query)
        q_tf = Counter(q_tokens)
        q_vec = hash_embed(query)

        # 1. Pre-filter strictly BEFORE computing IDF or ranking
        authorized: list[IndexedChunk] = []
        denied_doc_ids: set[str] = set()
        for ch in self._chunks.values():
            if doc_id_filter and ch.doc_id != doc_id_filter:
                continue
            ok, reason = can_principal_access(principal, ch.acl)
            if ok:
                authorized.append(ch)
            elif doc_id_filter and ch.doc_id == doc_id_filter:
                denied_doc_ids.add(ch.doc_id)
                self.audit.record(
                    principal,
                    "vector_search",
                    "document",
                    ch.doc_id,
                    "DENY",
                    reason,
                    query_text=query,
                    metadata={"prefilter": prefilter},
                )

        if not authorized:
            if not denied_doc_ids:
                self.audit.record(
                    principal,
                    "vector_search",
                    "corpus",
                    doc_id_filter or "*",
                    "ALLOW",
                    "no_authorized_chunks",
                    query_text=query,
                    metadata={"hits": 0, "prefilter": prefilter},
                )
            return []

        # 2. Score only authorized chunks (zero cross-tenant IDF leakage)
        n_docs = len(authorized)
        df: Counter[str] = Counter()
        doc_tokens_map: dict[str, list[str]] = {}
        for ch in authorized:
            toks = tokenize(ch.content + " " + " ".join(ch.tags) + " " + ch.title)
            doc_tokens_map[ch.chunk_id] = toks
            for t in set(toks):
                df[t] += 1

        scored: list[tuple[float, IndexedChunk]] = []
        for ch in authorized:
            toks = doc_tokens_map[ch.chunk_id]
            tf = Counter(toks)
            bm25 = 0.0
            dl = max(1, len(toks))
            for qt in q_tf:
                if qt not in tf:
                    continue
                idf = math.log(1.0 + (n_docs - df[qt] + 0.5) / (df[qt] + 0.5))
                freq = tf[qt]
                bm25 += idf * (freq * 2.2) / (freq + 1.2 * (0.25 + 0.75 * dl / 40.0))

            vec_sim = max(0.0, cosine_sim(q_vec, ch.embedding))
            hybrid_score = 0.65 * bm25 + 0.35 * vec_sim
            if hybrid_score > 0.01 or doc_id_filter:
                scored.append((hybrid_score, ch))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_items = scored[:top_k]
        hits = [
            SearchHit(
                chunk_id=ch.chunk_id,
                doc_id=ch.doc_id,
                title=ch.title,
                page_number=ch.page_number,
                snippet=ch.content[:320],
                score=round(score, 4),
                source_uri=ch.source_uri,
                tags=ch.tags,
            )
            for score, ch in top_items
        ]

        self.audit.record(
            principal,
            "vector_search",
            "corpus",
            doc_id_filter or "*",
            "ALLOW",
            "prefiltered_hybrid_search",
            query_text=query,
            metadata={
                "hits": len(hits),
                "doc_ids": sorted({h.doc_id for h in hits}),
                "prefilter": prefilter,
            },
        )
        return hits
