NVCC = nvcc
NVCCFLAGS = -O3 -std=c++17 -arch=sm_89
TARGET = topk_cuda

all: $(TARGET)

$(TARGET): topk_cuda.cu
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) topk_cuda.cu

bench: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all bench clean
