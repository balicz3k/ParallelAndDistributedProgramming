CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -fopenmp -I../common

topk_omp: topk_omp.cpp ../common/common.h
	$(CXX) $(CXXFLAGS) topk_omp.cpp -o topk_omp

clean:
	rm -f topk_omp

.PHONY: clean
