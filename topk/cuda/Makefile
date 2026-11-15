NVCC      = nvcc
NVCCFLAGS = -O3 -std=c++17 -arch=sm_89
TARGET    = topk_cuda
TARGET2   = topk_cublas

# Parametry benchmarku (nadpisywalne: make bench N=100000 k=10)
DATA   = ../data
D      ?= 128
Q      ?= 20
K      ?= 10
REPEAT ?= 5
WARMUP ?= 2

all: $(TARGET) $(TARGET2)

$(TARGET): topk_cuda.cu
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) topk_cuda.cu

$(TARGET2): topk_cublas.cu
	$(NVCC) $(NVCCFLAGS) -o $(TARGET2) topk_cublas.cu -lcublas

# Pelny sweep: N x {10k,100k,1M}, wlasny kernel + cuBLAS, zapis do results_cuda.txt
bench: all
	@echo "== Benchmark CUDA (wlasny kernel) + cuBLAS; D=$(D) Q=$(Q) k=$(K) =="
	./$(TARGET)  $(DATA)/n10000/base.bin   $(DATA)/n10000/queries.bin   10000   $(D) $(Q) $(K) --repeat $(REPEAT) --warmup $(WARMUP) --quiet --out results_cuda.txt --tag "N=10000"
	./$(TARGET)  $(DATA)/n100000/base.bin  $(DATA)/n100000/queries.bin  100000  $(D) $(Q) $(K) --repeat $(REPEAT) --warmup $(WARMUP) --quiet --append --out results_cuda.txt --tag "N=100000"
	./$(TARGET)  $(DATA)/n1000000/base.bin $(DATA)/n1000000/queries.bin 1000000 $(D) $(Q) $(K) --repeat $(REPEAT) --warmup $(WARMUP) --quiet --append --out results_cuda.txt --tag "N=1000000"
	./$(TARGET2) $(DATA)/n10000/base.bin   $(DATA)/n10000/queries.bin   10000   $(D) $(Q) $(K) --repeat $(REPEAT) --warmup $(WARMUP) --quiet --append --out results_cuda.txt --tag "N=10000"
	./$(TARGET2) $(DATA)/n100000/base.bin  $(DATA)/n100000/queries.bin  100000  $(D) $(Q) $(K) --repeat $(REPEAT) --warmup $(WARMUP) --quiet --append --out results_cuda.txt --tag "N=100000"
	./$(TARGET2) $(DATA)/n1000000/base.bin $(DATA)/n1000000/queries.bin 1000000 $(D) $(Q) $(K) --repeat $(REPEAT) --warmup $(WARMUP) --quiet --append --out results_cuda.txt --tag "N=1000000"
	@echo "== Gotowe: results_cuda.txt =="

# Sweep k (przy N=100k): k in {1,5,10,20,50}. Dla duzego k mniej watkow
# (limit shared memory sm_89 ~99 KiB: threads*k*8 B).
bench-k: $(TARGET)
	@for kk in 1 5 10 20 50; do \
	  th=256; if [ $$kk -ge 50 ]; then th=128; fi; \
	  ./$(TARGET) $(DATA)/n100000/base.bin $(DATA)/n100000/queries.bin 100000 $(D) $(Q) $$kk --threads $$th --repeat $(REPEAT) --warmup $(WARMUP) --quiet --append --out results_cuda.txt --tag "k=$$kk N=100000"; \
	done

# Weryfikacja zgodnosci z referencja (indeksy musza byc identyczne)
verify: $(TARGET)
	./$(TARGET) $(DATA)/n100000/base.bin $(DATA)/n100000/queries.bin 100000 $(D) $(Q) $(K) --repeat 1 --warmup 1 --quiet --out /tmp/cuda_verify.txt --tag verify

clean:
	rm -f $(TARGET) $(TARGET2)

.PHONY: all bench bench-k verify clean
