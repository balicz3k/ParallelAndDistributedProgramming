# Pomiary MPI na jednym komputerze są lokalne (procesy dzielą rdzenie i pamięć).
MPICXX   = mpicxx
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -I../common

all: topk_mpi topk_mpi_hybrid

topk_mpi: topk_mpi.cpp ../common/common.h
	$(MPICXX) $(CXXFLAGS) topk_mpi.cpp -o topk_mpi

# wariant hybrydowy MPI + OpenMP (wątki dzielą zapytania partii)
topk_mpi_hybrid: topk_mpi.cpp ../common/common.h
	$(MPICXX) $(CXXFLAGS) -fopenmp topk_mpi.cpp -o topk_mpi_hybrid

clean:
	rm -f topk_mpi topk_mpi_hybrid

.PHONY: all clean
