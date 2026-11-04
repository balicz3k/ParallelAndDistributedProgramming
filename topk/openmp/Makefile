CXX      = g++
CXXFLAGS = -O3 -std=c++17 -fopenmp
TARGET   = topk_omp

DATA_DIR  = ../data/n100000
BASE      = $(DATA_DIR)/base.bin
QUERIES   = $(DATA_DIR)/queries.bin
N         = 100000
D         = 128
Q         = 20
K         = 10

all: $(TARGET)

$(TARGET): topk_omp.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) topk_omp.cpp

bench: $(TARGET)
	@echo "=== Benchmark OpenMP top-k (N=$(N), D=$(D), Q=$(Q), k=$(K)) ===" | tee results_omp.txt
	@echo "" | tee -a results_omp.txt
	@for t in 1 2 4 8; do \
		echo "--- Watki: $$t ---" | tee -a results_omp.txt; \
		OMP_NUM_THREADS=$$t ./$(TARGET) $(BASE) $(QUERIES) $(N) $(D) $(Q) $(K) $$t \
			2>&1 | grep -E "(P95|Przepustowosc|Statystyki|Calkowity)" | tee -a results_omp.txt; \
		echo "" | tee -a results_omp.txt; \
	done

verify: $(TARGET)
	@echo "=== Weryfikacja poprawności OMP vs REF ==="
	@cd ../reference && ./topk_ref $(BASE) $(QUERIES) $(N) $(D) $(Q) $(K)
	@./$(TARGET) $(BASE) $(QUERIES) $(N) $(D) $(Q) $(K) 1 ../reference/results.txt | grep -E "(VERIFY|Wyniki)"
	@./$(TARGET) $(BASE) $(QUERIES) $(N) $(D) $(Q) $(K) 4 ../reference/results.txt | grep -E "(VERIFY)"
	@./$(TARGET) $(BASE) $(QUERIES) $(N) $(D) $(Q) $(K) 8 ../reference/results.txt | grep -E "(VERIFY)"

clean:
	rm -f $(TARGET) results_omp*.txt

.PHONY: all bench verify clean
