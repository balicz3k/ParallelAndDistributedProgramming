NVCC      = nvcc
NVCCFLAGS = -O3 -std=c++17 -arch=sm_89 -I../common

all: topk_cuda topk_cublas

topk_cuda: topk_cuda.cu ../common/common.h
	$(NVCC) $(NVCCFLAGS) topk_cuda.cu -o topk_cuda

# cuBLAS - dodatkowy punkt odniesienia wydajności
topk_cublas: topk_cublas.cu ../common/common.h
	$(NVCC) $(NVCCFLAGS) topk_cublas.cu -o topk_cublas -lcublas

clean:
	rm -f topk_cuda topk_cublas

.PHONY: all clean
