.PHONY: help install test test-fast lint format docker-build docker-run clean

# Default target: show help
all: help

help:
	@echo "Available developer commands:"
	@echo "  install        Install dependencies including dev tools using uv"
	@echo "  test           Run all tests with uv"
	@echo "  test-fast      Run fast tests (excludes slow/gpu tests) with uv"
	@echo "  lint           Run lint checks (black, isort, flake8)"
	@echo "  format         Auto-format code (black, isort)"
	@echo "  docker-build   Build production Docker image"
	@echo "  docker-run     Run production Docker image locally"
	@echo "  clean          Remove temporary caches, virtualenv traces, and coverage data"

install:
	uv sync --all-extras

test:
	uv run pytest tests/ -v

test-fast:
	uv run pytest tests/ -v -m "not slow and not gpu"

lint:
	uv run black --check src/ tests/ scripts/
	uv run isort --check-only src/ tests/ scripts/

format:
	uv run black src/ tests/ scripts/
	uv run isort src/ tests/ scripts/

docker-build:
	docker build -f Dockerfile.prod -t unsloth-api:latest .

docker-run:
	docker run -d -p 8000:8000 --env-file .env unsloth-api:latest

clean:
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
