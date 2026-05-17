SHELL := /bin/bash
CONDA_ENV ?= st-dadk

# Initialize conda and activate environment
CONDA_BASE := $(shell conda info --base 2>/dev/null || echo "/opt/conda")
CONDA_ACTIVATE := source $(CONDA_BASE)/etc/profile.d/conda.sh && conda activate $(CONDA_ENV)
EXECUTABLE := $(CONDA_ACTIVATE) && poetry run

.PHONY: help clean install install-dev test test-cov lint format run-local-jupyter train kaust kaust-dry

.DEFAULT_GOAL := help

## Show this help message
help:
	@echo "DA-STDK Makefile Commands"
	@echo ""
	@echo "Environment Setup:"
	@echo "  make install          Install project dependencies with Poetry"
	@echo "  make install-dev      Install development dependencies in conda environment"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run linters (black, isort, mypy)"
	@echo "  make format           Format code with black and isort"
	@echo "  make test             Run tests"
	@echo "  make test-cov         Run tests with coverage report"
	@echo ""
	@echo "Training & Experiments:"
	@echo "  make train            Train model with default config"
	@echo "  make kaust            Run KAUST benchmark (4 scenarios x 2 models, with analysis)"
	@echo "  make kaust-dry        Print KAUST run commands without executing"
	@echo ""
	@echo "Utilities:"
	@echo "  make run-local-jupyter Start Jupyter Lab server"
	@echo "  make clean            Clean up temporary files"
	@echo ""
	@echo "For detailed script usage, see scripts/README.md"

## Install project dependencies with Poetry
install:
	@echo "Installing project dependencies with Poetry..."
	@$(CONDA_ACTIVATE) && poetry install --with dev

## Install development dependencies (conda)
install-dev:
	@echo "Installing development dependencies in conda env: $(CONDA_ENV)..."
	@$(SHELL) envs/conda/build_conda_env.sh -c $(CONDA_ENV)

## Run tests
test:
	@echo "Running tests..."
	@$(CONDA_ACTIVATE) && poetry run pytest tests/ -v

## Run tests with coverage
test-cov:
	@echo "Running tests with coverage..."
	@$(CONDA_ACTIVATE) && poetry run pytest tests/ -v --cov=da_stdk --cov-report=html --cov-report=term

## Run linters
lint:
	@echo "Running linters..."
	@$(CONDA_ACTIVATE) && poetry run python -m black --check da_stdk scripts
	@$(CONDA_ACTIVATE) && poetry run python -m isort --check-only da_stdk scripts
	@$(CONDA_ACTIVATE) && poetry run python -m mypy da_stdk --ignore-missing-imports || true

## Format code
format:
	@echo "Formatting code..."
	@$(CONDA_ACTIVATE) && poetry run python -m black da_stdk scripts
	@$(CONDA_ACTIVATE) && poetry run python -m isort da_stdk scripts

## Start Jupyter server locally
run-local-jupyter:
	@echo "Starting local Jupyter server..."
	@$(SHELL) envs/jupyter/start_jupyter_lab.sh --port 8501

## Train model with default config
train:
	@echo "Training model..."
	@$(CONDA_ACTIVATE) && poetry run python scripts/train_default.py

## Run KAUST benchmark (4 scenarios x STDK/DA-STDK) and analyze results
kaust:
	@echo "Running KAUST benchmark..."
	@$(CONDA_ACTIVATE) && poetry run python scripts/run_kaust_data.py \
		--config configs/config_default.yaml \
		--analyze

## Dry-run: print training commands for each scenario x model
kaust-dry:
	@$(CONDA_ACTIVATE) && poetry run python scripts/run_kaust_data.py \
		--config configs/config_default.yaml \
		--dry-run

## Clean up temporary files
clean:
	@echo "Cleaning up..."
	@find . -type f -name '*.py[co]' -delete
	@find . -type d -name '__pycache__' -delete
	@rm -rf build/ dist/ .eggs/ .pytest_cache
	@find . -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	@rm -f .coverage
