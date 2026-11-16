# Te same flagi co w wersji OpenMP i MPI, żeby przyspieszenia nie wynikały z kompilatora
CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -I../common

topk_ref: topk_ref.cpp ../common/common.h
	$(CXX) $(CXXFLAGS) topk_ref.cpp -o topk_ref

clean:
	rm -f topk_ref

.PHONY: clean
