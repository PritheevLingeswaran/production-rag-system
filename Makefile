SHELL := /bin/bash

.PHONY: install fmt lint test run ingest index eval loadtest all docker-build docker-up docker-down

install:
	python -m pip install -U pip
	pip install -r requirements.txt
	pip install -e .

fmt:
	ruff format .

lint:
	ruff check .
	mypy .

test:
	PYTHONPATH=src pytest -q

run:
	PYTHONPATH=src python -m scripts.run_api

ingest:
	PYTHONPATH=src python -m scripts.ingest_data

index:
	PYTHONPATH=src python -m scripts.build_index

eval:
	PYTHONPATH=src python -m scripts.run_eval

loadtest:
	PYTHONPATH=src python -m scripts.load_test

all:
	PYTHONPATH=src python -m scripts.run_all

docker-build:
	docker build -t rag-smart-qa:latest .

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v
