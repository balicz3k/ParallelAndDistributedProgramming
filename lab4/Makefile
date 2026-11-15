CC     = mpicc
CFLAGS = -O2 -Wall -Wextra
BIN    = prz_mpi ring bcast scatter_gather

.PHONY: all run_przyklad run_ring run_bcast run_scatter benchmark results clean

all: $(BIN)

# program przykładowy pod nazwą z treści zadania
prz_mpi: przyklad.c
	$(CC) $(CFLAGS) przyklad.c -o prz_mpi

ring: ring.c
	$(CC) $(CFLAGS) ring.c -o ring

bcast: bcast.c
	$(CC) $(CFLAGS) bcast.c -o bcast

scatter_gather: scatter_gather.c
	$(CC) $(CFLAGS) scatter_gather.c -o scatter_gather

run_przyklad: prz_mpi
	mpirun -np 7 --display-map ./prz_mpi

run_ring: ring
	printf "5\n7\n-1\n" | mpirun -np 4 ./ring

run_bcast: bcast
	printf "5\n7\n-1\n" | mpirun -np 4 ./bcast

run_scatter: scatter_gather
	mpirun -np 3 ./scatter_gather

# pomiary czasu dla p = 2..8 -> results_raw.csv
benchmark: scatter_gather
	./run_benchmark.sh

# zrzuty wyjścia programów do sprawozdania -> results.txt
results: $(BIN)
	{ echo "### mpirun -np 7 --display-map ./prz_mpi"; \
	  mpirun -np 7 --display-map ./prz_mpi; \
	  echo; echo "### mpirun -np 8 ./ring  (wejście: 5, 7, -1)"; \
	  printf "5\n7\n-1\n" | mpirun -np 8 ./ring; \
	  echo; echo "### mpirun -np 4 ./bcast  (wejście: 5, 7, -1)"; \
	  printf "5\n7\n-1\n" | mpirun -np 4 ./bcast; \
	  for p in 2 3 4 7 8; do \
	    echo; echo "### mpirun -np $$p ./scatter_gather"; \
	    mpirun -np $$p ./scatter_gather; \
	  done; } > results.txt 2>&1

clean:
	rm -f $(BIN)
