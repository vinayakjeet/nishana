BENCH_PYTHON := uv run python

.PHONY: help bench train-eval decontam status check test lint conventions

help:
	@echo "  bench        print the status numbers from committed artifacts"
	@echo "  train-eval   full evaluation sweep over every run in results/runs.yaml"
	@echo "  decontam     measure seed-vs-eval n-gram overlap (add APPLY=1 to drop)"
	@echo "  status       alias for bench"
	@echo "  check        ruff + conventions + pytest, what CI runs"

bench:
	$(BENCH_PYTHON) scripts/status.py

status: bench

train-eval:
	$(BENCH_PYTHON) scripts/train_eval.py
	$(BENCH_PYTHON) scripts/status.py

decontam:
ifdef APPLY
	$(BENCH_PYTHON) scripts/decontaminate.py --apply
else
	$(BENCH_PYTHON) scripts/decontaminate.py
endif

check: lint conventions test

lint:
	uv run ruff check .

conventions:
	$(BENCH_PYTHON) scripts/check_conventions.py

test:
	uv run pytest
