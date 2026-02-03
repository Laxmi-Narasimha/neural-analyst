.PHONY: help setup setup-analyst setup-validator dev-analyst dev-validator test-analyst

ROOT := $(shell pwd)

ANALYST_BACKEND_DIR := ai-data-analyst/backend
ANALYST_FRONTEND_DIR := ai-data-analyst/frontend
VALIDATOR_BACKEND_DIR := ai-data-validator/backend
VALIDATOR_FRONTEND_DIR := ai-data-validator/frontend

ANALYST_VENV := $(ANALYST_BACKEND_DIR)/.venv
VALIDATOR_VENV := $(VALIDATOR_BACKEND_DIR)/.venv

PYTHON_ANALYST ?= python3.11
PYTHON_VALIDATOR ?= python3.11

help:
	@echo "Targets:"
	@echo "  make setup           Create venvs and install dependencies"
	@echo "  make dev-analyst     Run FastAPI + Next.js (requires Node >= 20.9.0)"
	@echo "  make dev-validator   Run FastAPI + Streamlit"
	@echo "  make test-analyst    Run backend tests for ai-data-analyst"
	@echo ""
	@echo "See docs/SETUP.md for installing Python/Node/Postgres."

setup: setup-analyst setup-validator

setup-analyst:
	@./scripts/setup_analyst.sh "$(PYTHON_ANALYST)"

setup-validator:
	@./scripts/setup_validator.sh "$(PYTHON_VALIDATOR)"

dev-analyst:
	@./scripts/dev_analyst.sh

dev-validator:
	@./scripts/dev_validator.sh

test-analyst:
	@./scripts/test_analyst.sh "$(PYTHON_ANALYST)"

