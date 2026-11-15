# Glowny Makefile D03 top-k.  make all = data + build + test + bench + plots
SUBDIRS  = reference openmp mpi cuda
BENCH    = openmp mpi cuda          # reference nie ma celu bench (patrz reference/benchmark.py)
PYTHON  ?= python3
FULL    ?= 0                        # FULL=1 => dodatkowo N=1M (~490 MB)

.PHONY: all data build test verify bench plots clean distclean

all: data build test bench plots

data:
	scripts/gen_data.sh $(if $(filter 1,$(FULL)),--full,)

build:
	@for d in $(SUBDIRS); do echo "== build $$d"; $(MAKE) -C $$d || exit 1; done

test: build
	$(PYTHON) tests/check.py

verify: test

bench: build data
	@for d in $(BENCH); do echo "== bench $$d"; $(MAKE) -C $$d bench || exit 1; done

plots:
	$(PYTHON) scripts/plots.py --root . --out plots

clean:
	@for d in $(SUBDIRS); do $(MAKE) -C $$d clean; done
	rm -rf plots tests/__pycache__ scripts/__pycache__

distclean: clean
	rm -rf data
