"""Shared service container wiring security, storage, streaming, and corpus bootstrap."""

from __future__ import annotations

import tempfile
from pathlib import Path

from data.sample_corpus.seed_corpus import generate_sample_corpus
from pipelines.batch.doc_ingest_beam import run_doc_ingest_beam
from pipelines.streaming.watermark_engine import WatermarkStreamEngine
from services.api.config import Settings
from services.security.audit import AuditLogger
from services.storage.blob import LocalBlobStore
from services.storage.hybrid_index import HybridAclSearchIndex
from services.storage.postgres import InMemoryArenaRepository


class ServiceContainer:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings(
            blob_root=Path(tempfile.mkdtemp(prefix="cl_blobs_")),
            corpus_work_dir=Path(tempfile.mkdtemp(prefix="cl_corpus_")),
        )
        self.audit = AuditLogger(jsonl_path=self.settings.audit_log_path)
        self.repo = InMemoryArenaRepository(self.audit)
        self.index = HybridAclSearchIndex(self.audit)
        self.blob = LocalBlobStore(self.settings.blob_root, self.audit)
        self.stream_engine = WatermarkStreamEngine()

        if self.settings.bootstrap_corpus:
            self.bootstrap_sample_data()

    def bootstrap_sample_data(self) -> None:
        loadfile = generate_sample_corpus(self.settings.corpus_work_dir)
        run_doc_ingest_beam(loadfile, self.repo, self.index, self.blob)
