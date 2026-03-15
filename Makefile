SHELL := /bin/bash

QUALITY_PATHS := src tests evaluation/resume_metrics.py
WEB_PACKAGE_MANAGER := npm ci
PYTHONPATH_EXPORT := PYTHONPATH=src

.PHONY: install fmt lint typecheck test test-web test-all build run api web dev ingest index eval load-test loadtest all docker-build docker-up docker-down stats eval-retrieval eval-grounding stability-60m

install:
	python -m pip install -U pip
	pip install -r requirements.txt
	pip install -e .
	cd web && $(WEB_PACKAGE_MANAGER)

fmt:
	ruff format .

lint:
	ruff check $(QUALITY_PATHS)

typecheck:
	mypy $(QUALITY_PATHS)

test:
	$(PYTHONPATH_EXPORT) pytest --cov=src --cov-report=term-missing --cov-report=xml -q

test-web:
	cd web && $(WEB_PACKAGE_MANAGER) && npm run test -- --coverage

test-all: test test-web

build:
	python -m build
	cd web && $(WEB_PACKAGE_MANAGER) && npm run build

run:
	$(PYTHONPATH_EXPORT) python -m scripts.run_api

api:
	$(PYTHONPATH_EXPORT) python -m scripts.run_api

web:
	cd web && $(WEB_PACKAGE_MANAGER) && npm run dev

dev:
	docker compose up --build

ingest:
	$(PYTHONPATH_EXPORT) python -m scripts.ingest_data

index:
	$(PYTHONPATH_EXPORT) python -m scripts.build_index

eval:
	$(PYTHONPATH_EXPORT) python -m scripts.measure_resume_metrics

load-test:
	$(PYTHONPATH_EXPORT) python -m scripts.load_test

loadtest: load-test

stats:
	curl -sS http://127.0.0.1:8000/stats | python3 -m json.tool

eval-retrieval:
	$(PYTHONPATH_EXPORT) python -m scripts.eval_retrieval \
		--eval_jsonl evaluation/datasets/retrieval_eval.jsonl \
		--hybrid_url http://127.0.0.1:8000/retrieve/hybrid \
		--bm25_url http://127.0.0.1:8000/retrieve/bm25 \
		--method POST --payload '{"top_k":5,"rewrite_query":false}' --k 5

eval-grounding:
	$(PYTHONPATH_EXPORT) python -m scripts.grounding_eval export \
		--query_url http://127.0.0.1:8000/query \
		--method POST \
		--payload '{"top_k":12,"rewrite_query":false}' \
		--grounding_queries grounding_queries_100.txt \
		--out_csv grounding_eval_100.csv \
		--concurrency 3 --timeout 60
	$(PYTHONPATH_EXPORT) python -m scripts.grounding_eval score --out_csv grounding_eval_100.csv

stability-60m:
	$(PYTHONPATH_EXPORT) python -m scripts.stability_test \
		--query_url http://127.0.0.1:8000/query \
		--method POST \
		--payload '{"top_k":12,"rewrite_query":false}' \
		--duration_minutes 60 --concurrency 20 \
		--stability_query "How many projects are there in the resume?" \
		--out_csv stability_60m.csv

all:
	$(PYTHONPATH_EXPORT) python -m scripts.run_all

docker-build:
	docker build -t rag-smart-qa:latest .

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v
