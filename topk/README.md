# PRiR D03 - wyszukiwanie wektorowe top-k

Dokladne top-k (odleglosc euklidesowa, float32, remis -> nizszy indeks).
Wersje: `reference/` (sekwencyjna), `openmp/`, `mpi/`, `cuda/`.

## Odtworzenie wynikow

```bash
make data     # dane do data/ (ziarno 42; FULL=1 dodaje N=1M)
make build    # kompilacja wszystkich wersji (g++, mpicxx, nvcc sm_89)
make test     # tests/check.py: zgodnosc z referencja numpy, przypadki brzegowe
make bench    # make bench w openmp/, mpi/, cuda/ -> results_*.txt
make plots    # scripts/plots.py -> plots/*.png
make all      # wszystko po kolei
```

Wymagania: g++ z OpenMP, OpenMPI, CUDA toolkit, Python 3 z numpy i matplotlib.

## Dane (`scripts/gen_data.sh`, `generate_data.py`)

Raw float32, row-major, bez naglowka; `data/n<N>[_d<D>][_q<Q>]/{base,queries}.bin`
(domyslnie D=128, Q=20). Zestawy: N=1k/10k/100k(/1M); D=32/64/256/512 (N=100k);
Q=1/100/1000 (N=100k). Ziarno w `SEED.txt` (zmiana: `SEED=7 make data`);
rozklad N(0,1). Wartosc k podaje sie przy uruchomieniu.

## Testy (`tests/check.py`)

Niezalezna referencja numpy (float64). Sprawdza kolejnosc, indeksy i odleglosci
(tolerancja 1e-4 abs + 1e-4 rel; zamiana indeksow dozwolona tylko dla remisu
w granicy tolerancji), dla kazdej implementacji i kilku p (watki/procesy).
Przypadki: k=1, k=N, k>N (czyste odrzucenie lub min(k,N) wynikow), N=1, N male, duplikaty,
wszystkie wektory rowne, wektor zerowy, kratka calkowita (remisy), N niepodzielne
przez p, D=1, D niepodzielne przez 4. Opcje: `--impl omp mpi`, `--quick`, `--strict`
(brak binarki = blad). Kod wyjscia != 0 przy niezgodnosci.

## Wykresy (`scripts/plots.py`)

Czyta `openmp/results_omp.txt` (wiersze `CSV,...`), `mpi/results_mpi.txt` (tabele
sekcji 2-7) i `cuda/results_cuda.txt` (bloki `## tag (N=..)` + `klucz=wartosc`).
Wyniki w `plots/`: `omp_speedup_efficiency.png`, `mpi_speedup_efficiency.png`,
`mpi_time_p95.png`, `qps_vs_N.png`, `p95_vs_N.png`, `memory_vs_N.png`,
`cuda_own_vs_cublas.png`, `sweep_k.png`. Brak pliku/danych => wykres pomijany.
Pomiar MPI na jednym komputerze jest lokalny - nie dowodzi skalowalnosci klastra.
Uwaga: implementacje mierzono przy roznym Q (OMP 250, MPI 100, CUDA 20) - QPS
miedzy implementacjami jest tylko orientacyjny.

## Uwagi do testow

CUDA zapisuje ranking tylko na stdout (do pliku trafiaja statystyki), wiec
`tests/check.py` parsuje stdout. Kernel CUDA ma limit k<=64 - dla wiekszego k
test akceptuje czyste odrzucenie (kod != 0, brak wynikow).

## Odtwarzanie samego MPI

`make -C mpi bench` (dane, build, weryfikacja P=1,2,4,8, sweepy, wariant
hybrydowy) -> `mpi/results_mpi.txt`. Nie uruchamiac rownolegle z innymi
benchmarkami (zaklocenia pomiarow).

## Format pliku wynikow implementacji

`query_id  rank  index  distance` (komentarze `#`), jak w `reference/results.txt`.

## Dokumentacja wersji referencyjnej

Szczegoly CLI `topk_ref`: `./reference/topk_ref base.bin queries.bin N D Q k`
(zapis `results.txt` w biezacym katalogu; P95 i QPS w statystykach).
