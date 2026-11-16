# Projekt D03 top-k. Pełne odtworzenie wyników: make all
SUBDIRS = reference openmp mpi cuda
PYTHON ?= python3

.PHONY: all build data test bench plots clean distclean

all: bench plots

build:
	@for d in $(SUBDIRS); do $(MAKE) -C $$d || exit 1; done

data:
	scripts/gen_data.sh

test: build
	$(PYTHON) tests/check.py --strict

# build + dane + testy + wszystkie pomiary z experiments.conf -> results/
bench:
	scripts/run_all.sh

# wykresy i tabele do sprawozdania z results/raw.csv
plots:
	$(PYTHON) scripts/plots.py

clean:
	@for d in $(SUBDIRS); do $(MAKE) -C $$d clean; done
	rm -rf tests/__pycache__ scripts/__pycache__

distclean: clean
	rm -rf data
