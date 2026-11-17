"""ACL-enforced Blob Storage abstraction (Local FS fallback + MinIO/S3/GCS-interop)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import boto3
from botocore.client import Config as BotoConfig

from services.security.acl import DocumentAclMeta, PrincipalContext, can_principal_access
from services.security.audit import AuditLogger


class BlobAccessDeniedError(PermissionError):
    """Raised when a principal attempts to read a blob outside their ACL."""


class BlobStore(Protocol):
    def put_bytes(self, key: str, data: bytes, acl: DocumentAclMeta) -> str: ...
    def get_bytes(self, key: str, principal: PrincipalContext) -> bytes: ...
    def exists(self, key: str) -> bool: ...


class LocalBlobStore:
    """Filesystem blob store persisting `.acl.json` sidecars for strict read enforcement."""

    def __init__(self, root_dir: Path, audit_logger: AuditLogger) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.audit = audit_logger

    def _safe_path(self, key: str) -> Path:
        clean = key.lstrip("/").replace("..", "")
        return self.root_dir / clean

    def put_bytes(self, key: str, data: bytes, acl: DocumentAclMeta) -> str:
        target = self._safe_path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        acl_path = target.with_suffix(target.suffix + ".acl.json")
        acl_path.write_text(acl.model_dump_json(indent=2), encoding="utf-8")
        return f"local://{key.lstrip('/')}"

    def exists(self, key: str) -> bool:
        return self._safe_path(key).exists()

    def get_bytes(self, key: str, principal: PrincipalContext) -> bytes:
        target = self._safe_path(key)
        acl_path = target.with_suffix(target.suffix + ".acl.json")
        if not target.exists() or not acl_path.exists():
            self.audit.record(
                principal,
                "blob_read",
                "blob",
                key,
                "DENY",
                "blob_not_found",
            )
            raise FileNotFoundError(f"Blob not found: {key}")

        acl = DocumentAclMeta.model_validate_json(acl_path.read_text(encoding="utf-8"))
        allowed, reason = can_principal_access(principal, acl)
        self.audit.record(
            principal,
            "blob_read",
            "blob",
            key,
            "ALLOW" if allowed else "DENY",
            reason,
            metadata={"doc_id": acl.doc_id, "tenant_id": acl.tenant_id},
        )
        if not allowed:
            raise BlobAccessDeniedError(f"Access denied to blob {key}: {reason}")
        return target.read_bytes()


class S3MinioBlobStore:
    """S3 / MinIO / GCS-interop blob store storing ACL metadata in object tags/metadata."""

    def __init__(
        self,
        bucket: str,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        audit_logger: AuditLogger,
    ) -> None:
        self.bucket = bucket
        self.audit = audit_logger
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=BotoConfig(signature_version="s3v4"),
        )

    def put_bytes(self, key: str, data: bytes, acl: DocumentAclMeta) -> str:
        clean_key = key.lstrip("/")
        self.client.put_object(
            Bucket=self.bucket,
            Key=clean_key,
            Body=data,
            Metadata={"acl": json.dumps(acl.model_dump())},
        )
        return f"s3://{self.bucket}/{clean_key}"

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key.lstrip("/"))
            return True
        except Exception:
            return False

    def get_bytes(self, key: str, principal: PrincipalContext) -> bytes:
        clean_key = key.lstrip("/")
        resp = self.client.get_object(Bucket=self.bucket, Key=clean_key)
        raw_acl = resp.get("Metadata", {}).get("acl")
        if not raw_acl:
            self.audit.record(
                principal, "blob_read", "blob", clean_key, "DENY", "missing_acl_metadata"
            )
            raise BlobAccessDeniedError("Missing ACL metadata on blob")
        acl = DocumentAclMeta.model_validate_json(raw_acl)
        allowed, reason = can_principal_access(principal, acl)
        self.audit.record(
            principal,
            "blob_read",
            "blob",
            clean_key,
            "ALLOW" if allowed else "DENY",
            reason,
            metadata={"doc_id": acl.doc_id},
        )
        if not allowed:
            raise BlobAccessDeniedError(f"Access denied to blob {clean_key}: {reason}")
        return bytes(resp["Body"].read())
