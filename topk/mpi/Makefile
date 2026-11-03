MPICXX   = mpicxx
CXXFLAGS = -O3 -std=c++17
TARGET   = topk_mpi

DATA_DIR = ../data/n100000
N        = 100000
D        = 128
Q        = 100
K        = 10

all: $(TARGET)

$(TARGET): topk_mpi.cpp
	$(MPICXX) $(CXXFLAGS) -o $(TARGET) topk_mpi.cpp

bench: $(TARGET)
	@echo "=== 1 proces ==="
	mpirun -np 1 ./$(TARGET) $(DATA_DIR)/base.bin $(DATA_DIR)/queries.bin $(N) $(D) $(Q) $(K)
	@echo ""
	@echo "=== 2 procesy ==="
	mpirun -np 2 ./$(TARGET) $(DATA_DIR)/base.bin $(DATA_DIR)/queries.bin $(N) $(D) $(Q) $(K)
	@echo ""
	@echo "=== 4 procesy ==="
	mpirun -np 4 ./$(TARGET) $(DATA_DIR)/base.bin $(DATA_DIR)/queries.bin $(N) $(D) $(Q) $(K)

clean:
	rm -f $(TARGET) results_mpi.txt

.PHONY: all bench clean
