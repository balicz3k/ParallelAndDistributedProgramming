# Makefile — MPI top-k (wariant lokalny, jedna maszyna)
#
# Cele:
#   make            — zbuduj topk_mpi (czysty MPI) i topk_mpi_hybrid (MPI+OpenMP)
#   make data       — wygeneruj dane lokalne w ./data_mpi/ (nie dotyka ../data/)
#   make verify     — sprawdz zgodnosc rankingu MPI (1,2,4,8 proc.) z referencja
#   make bench      — pelny benchmark od zera: data + build + verify + sweep
#   make clean      — usun binarki i artefakty (NIE usuwa results_mpi.txt)
#
# UWAGA: wszystkie pomiary wykonywane LOKALNIE na jednej maszynie (8C/16T).
# To NIE jest pomiar skalowalnosci klastra (zgodnie z wymaganiami D03).

MPICXX   = mpicxx
CXXFLAGS = -O3 -std=c++17 -Wall
OMPFLAGS = -fopenmp

# Parametry domyslne benchmarku weryfikacji/danych
DATA_DIR = ./data_mpi
N        = 100000
D        = 128
Q        = 100
K        = 10
SEED     = 42

TARGET  = topk_mpi
HYBRID  = topk_mpi_hybrid

.PHONY: all data verify bench clean

all: $(TARGET) $(HYBRID)

$(TARGET): topk_mpi.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $<

$(HYBRID): topk_mpi.cpp
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

# Dane generowane lokalnie w DATA_DIR (wspolny generator z korzenia projektu,
# seed zapisany). Nie nadpisuje plikow w ../data/ innych wersji.
data: $(DATA_DIR)/base.bin

$(DATA_DIR)/base.bin:
	@mkdir -p $(DATA_DIR)
	python3 ../generate_data.py --n $(N) --d $(D) --q $(Q) \
	        --seed $(SEED) --out_dir $(DATA_DIR)

# Weryfikacja poprawnosci: ranking MPI musi byc identyczny z referencja.
verify: $(TARGET) data
	python3 run_bench.py --verify-only \
	    --data_dir $(DATA_DIR) --N $(N) --D $(D) --Q $(Q) --k $(K)

# Pelny benchmark od zera.
bench: $(TARGET) $(HYBRID) data
	python3 run_bench.py \
	    --data_dir $(DATA_DIR) --N $(N) --D $(D) --Q $(Q) --k $(K) --seed $(SEED)

clean:
	rm -f $(TARGET) $(HYBRID) verify_*.txt
