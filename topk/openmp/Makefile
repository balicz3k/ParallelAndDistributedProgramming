CXX      = g++
CXXFLAGS = -O3 -std=c++17 -fopenmp
TARGET   = topk_omp

# Dane domyslne (benchmark uzywa queries_bench.bin, Q=250)
DATA_DIR  = ../data/n100000
BASE      = $(DATA_DIR)/base.bin
QUERIES   = $(DATA_DIR)/queries_bench.bin
N         = 100000
D         = 128
Q         = 250
K         = 10
REPEAT    = 7

all: $(TARGET)

$(TARGET): topk_omp.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) topk_omp.cpp

# Pelny benchmark: skalowanie watkow S(p)/E(p), sweep N/k/Q, pamiec, poprawnosc.
# Odtwarza results_omp.txt w jednym spojnym formacie.
bench: $(TARGET)
	./run_bench_omp.sh

# Szybki pojedynczy przebieg (warmup + REPEAT powtorzen) na konfiguracji domyslnej.
run: $(TARGET)
	OMP_NUM_THREADS=16 ./$(TARGET) $(BASE) $(QUERIES) $(N) $(D) $(Q) $(K) 16 $(REPEAT)

# Weryfikacja poprawnosci vs referencja sekwencyjna dla 1/4/8/16 watkow.
verify: $(TARGET)
	@echo "=== Weryfikacja OMP vs REF (N=$(N) D=$(D) Q=$(Q) k=$(K)) ==="
	@cd ../reference && ./topk_ref $(BASE) $(QUERIES) $(N) $(D) $(Q) $(K) >/dev/null 2>&1 \
		&& cp results.txt /tmp/ref_verify_omp.txt
	@for t in 1 4 8 16; do \
		printf "threads=%-3s " $$t; \
		./$(TARGET) $(BASE) $(QUERIES) $(N) $(D) $(Q) $(K) $$t 1 /tmp/ref_verify_omp.txt \
			2>/dev/null | grep -E "VERIFY"; \
	done

clean:
	rm -f $(TARGET) results_omp_t*.txt

.PHONY: all bench run verify clean
