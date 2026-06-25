# ASR_MeetingMinutes — developer entry points (mirrors quickstart.md)
#
# All targets are idempotent. The native Swift helper is only required for
# system-audio (US2); US1 (mic) and all unit/contract tests run without it.

PYTHON ?= python3
VENV   ?= .venv
PKG    := src/meeting_asr

.PHONY: help setup venv install build-native test test-fast lint format clean run app validate

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: venv install ## One-time: create venv + install deps (network allowed here)
	@echo "✓ setup complete (brew install portaudio for mic capture; swift build for system audio)"

venv: ## Create the local virtualenv
	$(PYTHON) -m venv $(VENV)

install: ## Install Python dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

build-native: ## Build the Core Audio Process-Tap Swift helper (macOS 14.4+)
	swift build -c release --package-path native/AudioTap
	@echo "✓ AudioTap helper built → $(shell swift build -c release --package-path native/AudioTap --show-bin-path)/AudioTap"

test: ## Run the full offline test suite (no network)
	$(PYTHON) -m pytest

test-fast: ## Run unit + contract only (skip slow/needs_models/needs_hardware)
	$(PYTHON) -m pytest tests/unit tests/contract -m "not slow and not needs_models and not needs_hardware"

lint: ## Lint with ruff
	$(PYTHON) -m ruff check $(PKG) tests || (echo "[ruff not installed; skipping]" && true)

format: ## Format with black + ruff
	$(PYTHON) -m black $(PKG) tests || true
	$(PYTHON) -m ruff check --fix $(PKG) tests || true

clean: ## Remove build/test artifacts (keeps venv + models cache)
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

run: ## Launch the desktop app (pywebview) — python -m app.main
	$(PYTHON) -m app.main

app: ## Build the double-click .app (PyInstaller + ad-hoc codesign)
	bash packaging/build_app.sh

validate: ## Run the accuracy-validation harness over all axes (needs models)
	$(PYTHON) -m validation --axis all
