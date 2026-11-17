.PHONY: help test test-ts test-py lint parity train-ppo api web up down tf-validate

help:
	@echo "Crowded Lineup — Multi-Agent Arena & Data Intelligence Platform"
	@echo "  make test         - Run TypeScript Vitest + Python pytest suites"
	@echo "  make lint         - Run tsc --noEmit, ruff check/format, and mypy"
	@echo "  make parity       - Run TS <-> Python/PyTorch bit-exact parity tests"
	@echo "  make train-ppo    - Train crowded multi-agent PPO and export ONNX v2 policy"
	@echo "  make api          - Start FastAPI control plane (port 8000)"
	@echo "  make web          - Start Vite frontend (port 5173)"
	@echo "  make up           - Launch full Postgres+pgvector, MinIO, Redpanda, Redis, API stack"
	@echo "  make tf-validate  - Format-check and validate GCP Terraform modules"

test-ts:
	npm run test

test-py:
	uv run pytest

test: test-ts test-py

lint:
	npm run build
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy packages services pipelines

parity:
	uv run pytest packages/python_sim/tests/test_ts_parity.py

train-ppo:
	uv run python -m packages.python_rl.self_play --num-envs 32 --num-surfers 16 --updates 12 --out artifacts/ppo_crowded_v2.json

api:
	uv run uvicorn services.api.main:app --reload --port 8000

web:
	npm run dev

up:
	docker compose -f infra/docker-compose.yml up --build

down:
	docker compose -f infra/docker-compose.yml down

tf-validate:
	terraform -chdir=infra/terraform fmt -check -recursive
	terraform -chdir=infra/terraform init -backend=false
	terraform -chdir=infra/terraform validate
