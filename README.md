# Crowded Lineup — Multi-Agent RL Arena & Governed Data Intelligence Platform

Mobile-first pixel-art surfing strategy arena paired with a **Python (`uv` / `FastAPI` / `Pydantic`) data & infrastructure platform**:

- **Crowded Multi-Peak Simulation (`v2`, 4–40+ Surfers)**:
  - Deterministic seeded Mulberry32 PRNG (`dt = 0.1s`) with 3-tier staggered lineup spawns, multi-peak swells (`320x240`), `16D` continuous normalized observations (including left/center/right crowd-density sectors, inside-priority threat, and secondary peak awareness), archetype bots, tabular Q-learning (Web Worker), and real browser **PPO ONNX inference** (`onnxruntime-web`).
- **Cross-Runtime Parity & Multi-Agent PPO (`packages/python_sim`, `packages/python_rl`)**:
  - Vectorized PyTorch environment (`VectorizedLineupEnv`) verified bit-for-bit (`atol=1e-4`) against [`tests/fixtures/reference-scenarios.json`](tests/fixtures/reference-scenarios.json), plus 3-stage curriculum & historical league self-play PPO with opset-17 ONNX export.
- **Dual Batch & Streaming Pipelines (`pipelines/`)**:
  - **Streaming (`pipelines/streaming/`)**: Event-time watermark (`5s`) + `30s/10s` hopping windows for live congestion & drop-in spike alerts (with Apache Flink SQL parity topology).
  - **Batch (`pipelines/batch/`)**: **Apache Beam** pipelines for deterministic replay analytics and multi-tenant PDF/OCR/e-discovery load-file ingestion with batch tagging and embeddings.
- **Polyglot Storage, Per-Document ACL & Audit Logging (`services/security/`, `services/storage/`, `db/`)**:
  - **PostgreSQL 16 + `pgvector` Row-Level Security (RLS)**, **MinIO/GCS Blob ACLs**, **Pre-Filtered Hybrid BM25+Vector Search**, and an append-only `audit_logs` table (enforced by DB trigger) tracking every read, search, blob fetch, and LLM agent tool call.
- **Steward Intelligence Agent (`services/api/agent/`)**:
  - Typed Pydantic tools (`search_private_corpus`, `read_document_page`, `inspect_heat_replay`, `query_leaderboard`) answering queries with heat timestamps & document page citations strictly within caller ACLs.
- **Production IaC & CI/CD (`infra/`, `.github/workflows/ci.yml`, `docs/adr/`)**:
  - GCP **Terraform** (`Cloud Run`, `Cloud SQL Postgres`, `GCS`, `Pub/Sub`, `IAM`, `VPC Service Controls`), `docker-compose.yml`, Prometheus `/metrics`, Architecture Decision Records, and on-call runbooks.

---

## Quick Start

### 1. Frontend & Simulation (TypeScript + Vite + Vitest)
```bash
npm install
npm run test     # Vitest simulation, telemetry, audio, and reference fixture suite
npm run build    # TypeScript check + Vite bundle (with onnxruntime-web)
npm run dev      # Launch browser arena (4 / 16 / 24 / 40 surfer presets + Arena & Intel tab)
```

### 2. Python Platform, Parity, RL & Steward API (`uv`)
```bash
uv sync
uv run pytest                                  # Parity, PPO/ONNX, Beam, Streaming, ACL penetration tests
uv run ruff check . && uv run mypy packages services pipelines
uv run uvicorn services.api.main:app --reload  # FastAPI control plane on :8000
```

### 3. Train Multi-Agent PPO & Export ONNX for Browser Play
```bash
uv run python -m packages.python_rl.self_play --num-envs 32 --num-surfers 16 --updates 12 --out artifacts/ppo_crowded_v2.json
```
Import `artifacts/ppo_crowded_v2.json` via the **TRAIN AI -> Import JSON** button to play against the trained neural policy in the browser.

### 4. Local Full Infrastructure Stack & Terraform
```bash
docker compose -f infra/docker-compose.yml up --build
terraform -chdir=infra/terraform init -backend=false && terraform -chdir=infra/terraform validate
```

---

## Architecture & Trade-Off Docs
- [ADR-001: Batch vs. Streaming Pipelines](docs/adr/ADR-001-batch-vs-streaming.md)
- [ADR-002: Polyglot Storage Architecture](docs/adr/ADR-002-polyglot-storage.md)
- [ADR-003: Per-Document ACL Enforcement & Audit Logging](docs/adr/ADR-003-per-document-acl-and-rag.md)
- [On-Call Runbook: Streaming Lag & ACL Spikes](docs/runbooks/oncall-pipeline-lag.md)
- [GPU RL Migration Roadmap](docs/gpu-rl-roadmap.md)
