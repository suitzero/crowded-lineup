"""Apache Beam batch document ingestion pipeline (PDF/OCR/load-file -> batch LLM tags -> ACL index)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from pipelines.batch.ocr import ExtractedPage, extract_image_ocr, extract_pdf_pages
from services.security.acl import ClearanceLevel, DocumentAclMeta
from services.storage.blob import BlobStore
from services.storage.hybrid_index import HybridAclSearchIndex, IndexedChunk, hash_embed
from services.storage.postgres import DocumentRecord, InMemoryArenaRepository


def batch_llm_extract_tags(text: str) -> list[str]:
    """Deterministic batch LLM tagger simulation (or pluggable Gemini batch inference hook)."""
    lowered = text.lower()
    tags: list[str] = []
    keyword_map = {
        "priority": "priority_rule",
        "drop-in": "drop_in_violation",
        "collision": "collision_rule",
        "stamina": "stamina_management",
        "peak": "wave_peak_strategy",
        "steward": "steward_incident",
        "ppo": "ppo_policy",
    }
    for kw, tag in keyword_map.items():
        if kw in lowered:
            tags.append(tag)
    return tags or ["general_corpus"]


def parse_loadfile_record(raw_line: str) -> dict[str, Any]:
    return dict(json.loads(raw_line))


def process_document_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Extract pages via pypdf/OCR, run batch tagger, and emit chunks + doc metadata."""
    file_path = Path(rec["file_path"])
    doc_type = rec.get("doc_type", "pdf")
    pages: list[ExtractedPage]
    if doc_type == "pdf":
        pages = extract_pdf_pages(file_path)
    else:
        pages = extract_image_ocr(file_path, sidecar_fallback=rec.get("ocr_sidecar_text"))

    chunks_out: list[dict[str, Any]] = []
    pages_map: dict[int, str] = {}
    for p in pages:
        pages_map[p.page_number] = p.text
        tags = batch_llm_extract_tags(p.text)
        chunks_out.append(
            {
                "chunk_id": f"{rec['doc_id']}-p{p.page_number}",
                "doc_id": rec["doc_id"],
                "title": rec["title"],
                "page_number": p.page_number,
                "content": p.text,
                "tags": tags,
                "embedding": hash_embed(p.text),
                "ocr_engine": p.engine,
                "custodian": rec.get("custodian"),
            }
        )

    return {
        "doc": rec,
        "file_bytes_path": str(file_path),
        "pages": pages_map,
        "chunks": chunks_out,
    }


def run_doc_ingest_beam(
    loadfile_path: Path,
    repo: InMemoryArenaRepository,
    hybrid_index: HybridAclSearchIndex,
    blob_store: BlobStore,
    output_manifest_path: Path | None = None,
) -> int:
    """Execute Apache Beam batch document pipeline and populate Postgres/Index/Blob stores."""
    lines = [ln for ln in loadfile_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    with tempfile.TemporaryDirectory(prefix="beam_doc_ingest_") as tmp_dir:
        out_records_path = Path(tmp_dir) / "processed_docs.jsonl"
        with beam.Pipeline(options=PipelineOptions(["--runner=DirectRunner"])) as pipeline:
            processed = (
                pipeline
                | "ReadLoadFile" >> beam.Create(lines)
                | "ParseLoadFileJson" >> beam.Map(parse_loadfile_record)
                | "ExtractOcrAndEmbed" >> beam.Map(process_document_record)
            )
            _ = (
                processed
                | "SerializeProcessed" >> beam.Map(json.dumps)
                | "WriteProcessedJsonl"
                >> beam.io.WriteToText(
                    str(out_records_path),
                    shard_name_template="",
                    num_shards=1,
                )
            )
            if output_manifest_path is not None:
                output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
                _ = (
                    processed
                    | "SerializeManifest"
                    >> beam.Map(
                        lambda item: json.dumps(
                            {"doc_id": item["doc"]["doc_id"], "chunks": len(item["chunks"])}
                        )
                    )
                    | "WriteManifest"
                    >> beam.io.WriteToText(
                        str(output_manifest_path),
                        shard_name_template="",
                        num_shards=1,
                    )
                )

        results = [
            json.loads(line)
            for line in out_records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    chunk_count = 0
    for item in results:
        meta = item["doc"]
        acl = DocumentAclMeta(
            doc_id=meta["doc_id"],
            tenant_id=meta["tenant_id"],
            classification=ClearanceLevel(int(meta["classification"])),
            allowed_principals=list(meta.get("allowed_principals", [])),
            is_public=bool(meta.get("is_public", False)),
        )
        raw_bytes = Path(item["file_bytes_path"]).read_bytes()
        blob_key = (
            f"corpus/{meta['tenant_id']}/{meta['doc_id']}/{Path(item['file_bytes_path']).name}"
        )
        source_uri = blob_store.put_bytes(blob_key, raw_bytes, acl)

        doc_record = DocumentRecord(
            doc_id=meta["doc_id"],
            tenant_id=meta["tenant_id"],
            title=meta["title"],
            source_uri=source_uri,
            classification=acl.classification,
            is_public=acl.is_public,
            allowed_principals=acl.allowed_principals,
            custodian=meta.get("custodian"),
            doc_type=meta.get("doc_type", "pdf"),
            pages=item["pages"],
        )
        repo.upsert_document(doc_record)

        indexed = [
            IndexedChunk(
                chunk_id=ch["chunk_id"],
                doc_id=ch["doc_id"],
                title=ch["title"],
                page_number=ch["page_number"],
                content=ch["content"],
                tags=ch["tags"],
                embedding=ch["embedding"],
                acl=acl,
                source_uri=source_uri,
                custodian=ch.get("custodian"),
            )
            for ch in item["chunks"]
        ]
        chunk_count += hybrid_index.upsert_chunks(indexed)

    return chunk_count
