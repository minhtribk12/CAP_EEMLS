.DEFAULT_GOAL := check
PY := poetry run
FAST := -m "not slow and not integration and not requires_gpu and not requires_docker"
EVERY := -m "not requires_gpu and not requires_docker"

.PHONY: install check check-fast lint format types spell test test-all integration coverage images images-local up run load down experiment clean release pin

PYTHON := $(shell poetry env info -p)/bin/python
PROJECTS := $(wildcard application/*) tools
FRAMEWORK_TESTS := tests/framework
OTHER_ROOT_TESTS := --ignore=$(FRAMEWORK_TESTS)
ROOT_TESTS := $(FRAMEWORK_TESTS) $(OTHER_ROOT_TESTS)
SOURCES := src application tests tools

install:
	poetry install --all-extras
	$(PY) pip install --no-deps $(addprefix --editable ,$(PROJECTS))

lint:
	$(PY) ruff check $(SOURCES)
	$(PY) ruff format --check $(SOURCES)

format:
	$(PY) ruff check --fix $(SOURCES)
	$(PY) ruff format $(SOURCES)

types:
	$(PY) mypy
	for project in $(PROJECTS); do (cd $$project && $(PYTHON) -m mypy) || exit 1; done

spell:
	$(PY) codespell $(SOURCES) config docs README.md

# Run pytest with $(1) in every project and with $(2) on the root parts $(3); no collected test passes.
define pytest_everywhere
for project in $(PROJECTS); do (cd $$project && $(PYTHON) -m pytest $(1)) || [ $$? -eq 5 ] || exit 1; done
for part in $(3); do $(PYTHON) -m pytest $(2) $$part || [ $$? -eq 5 ] || exit 1; done
endef

test:
	$(call pytest_everywhere,$(FAST),$(FAST),$(ROOT_TESTS))

test-all:
	$(call pytest_everywhere,$(EVERY),$(EVERY),$(ROOT_TESTS))

integration:
	$(call pytest_everywhere,-m integration,-m integration,$(ROOT_TESTS))

coverage:
	$(call pytest_everywhere,--cov --cov-report=term-missing --cov-fail-under=80 $(FAST),--cov=src/eemls --cov-report=term-missing --cov-fail-under=80 $(FAST),$(FRAMEWORK_TESTS))

check: lint types spell test

check-fast: lint test

release:
	$(PY) python tools/release.py $(VERSION)

pin:
	$(PY) python tools/pin.py $(APP) $(VERSION)

images:
	sh deploy/images.sh

images-local:
	REGISTRY=eemls.local $(PY) python tools/pin.py --bake --load local

up: images-local
	sh deploy/one_host/up.sh

run:
	sh deploy/one_host/run.sh

load:
	sh deploy/one_host/load_test.sh

down:
	sh deploy/one_host/down.sh

experiment: images-local
	sh deploy/one_host/experiment.sh

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage
	find src application tools tests \( -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache -o -name .coverage \) -prune -exec rm -rf {} +
