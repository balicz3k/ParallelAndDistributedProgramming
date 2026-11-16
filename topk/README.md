# PRiR D03 – wyszukiwanie wektorowe top-k

Dokładne top-k: odległość euklidesowa, `float32`, remis rozstrzyga mniejszy indeks.
Wersje: `reference/` (sekwencyjna), `openmp/`, `mpi/` (także hybryda MPI+OpenMP),
`cuda/` (własny kernel) i `cuda/topk_cublas` (cuBLAS jako dodatkowy punkt odniesienia).

## Odtworzenie wyników

```bash
make build    # kompilacja wszystkich wersji (g++, mpicxx, nvcc -arch=sm_89)
make data     # dane do data/ (scripts/gen_data.sh, ziarno 42)
make test     # tests/check.py: zgodność z niezależną referencją numpy
make bench    # scripts/run_all.sh: build + dane + testy + wszystkie pomiary -> results/
make plots    # scripts/plots.py: wykresy i tabele do sprawozdania z results/raw.csv
```

Wymagania: g++ z OpenMP, Open MPI, CUDA Toolkit (cuBLAS), Python 3 z numpy i matplotlib.
Pomiary uruchamiaj na nieobciążonym komputerze – `run_all.sh` przerywa, gdy GPU jest
zajęte przez inny proces. Części można liczyć osobno, np. `scripts/run_all.sh mpi k`
(dopisuje do `results/raw.csv`), a `scripts/run_all.sh --dry` robi szybką próbę.

## Konfiguracja pomiarów (`experiments.conf`)

Jeden plik z wszystkimi parametrami. Przypadek główny, wspólny dla wszystkich wersji:
**N = 100 000, D = 128, Q = 1000, k = 10**. Badane osobno: N, D, k, wielkość partii Q,
liczba wątków OpenMP, procesów MPI, wariant hybrydowy i liczba wątków w bloku CUDA.
Każdy pomiar to 2 przebiegi rozgrzewkowe i 7 powtórzeń.

## Dane

`data/n<N>_d<D>/base.bin` (N x D) i `queries.bin` (10 000 x D): surowe `float32`
wiersz po wierszu, bez nagłówka, rozkład N(0, 1), ziarno 42 (`generate_data.py`).
Programy czytają pierwsze Q zapytań, więc jeden plik zapytań wystarcza dla każdego Q.

## Uruchamianie jednej wersji

Wszystkie programy mają te same argumenty (`common/common.h`):

```
<program> <baza.bin> <zapytania.bin> <N> <D> <Q> <k> [opcje]
  --threads T   wątki OpenMP albo wątki w bloku CUDA (potęga 2)
  --warmup W    przebiegi rozgrzewkowe (domyślnie 2)
  --repeat R    mierzone powtórzenia (domyślnie 7)
  --lat L       ile zapytań na powtórzenie mierzyć pojedynczo, do P50/P95 (domyślnie 100)
  --csv         jeden wiersz "CSV,..." na powtórzenie
  --out PLIK    ranking: "zapytanie pozycja indeks odległość"
  --no-stream   tylko CUDA: bez strumieni
```

Przykłady:

```bash
d=data/n100000_d128
reference/topk_ref $d/base.bin $d/queries.bin 100000 128 1000 10
openmp/topk_omp    $d/base.bin $d/queries.bin 100000 128 1000 10 --threads 16
mpirun -np 8 mpi/topk_mpi $d/base.bin $d/queries.bin 100000 128 1000 10
mpirun -np 4 --bind-to none -x OMP_NUM_THREADS=4 mpi/topk_mpi_hybrid $d/base.bin $d/queries.bin 100000 128 1000 10
cuda/topk_cuda     $d/base.bin $d/queries.bin 100000 128 1000 10 --threads 256
cuda/topk_cublas   $d/base.bin $d/queries.bin 100000 128 1000 10
```

## Co jest mierzone

Kolumny `results/raw.csv` (jeden wiersz na powtórzenie):

| kolumna | znaczenie |
|---|---|
| `version` | `ref`, `omp`, `mpi`, `mpi_omp`, `cuda`, `cuda_bez_strumieni`, `cublas` |
| `threads`, `procs`, `block` | wątki OpenMP, procesy MPI, wątki w bloku CUDA |
| `io_ms` | wczytanie plików (raz, poza pomiarem partii) |
| `base_ms` | jednorazowe przygotowanie bazy: `MPI_Scatterv` albo kopia na GPU |
| `core_ms` | same obliczenia całej partii (MPI: maksimum po procesach, CUDA: kernel) |
| `full_ms` | cała partia z komunikacją (Bcast/Gather/scalanie) albo kopiami CPU-GPU |
| `qps` | Q / `full_ms` |
| `p50_us`, `p95_us` | opóźnienie zapytania obsługiwanego pojedynczo (L zapytań w powtórzeniu) |
| `rss_mb`, `vram_mb` | szczytowy RAM (MPI: suma procesów) i pamięć zajęta na GPU |

Pomiary MPI na jednym komputerze są **lokalne** – procesy dzielą rdzenie i pamięć,
więc nie pokazują skalowalności klastra. `results/env.txt` opisuje sprzęt i wersje
oprogramowania, `results/ranking_check.csv` – zgodność rankingów z referencją.

## Testy (`tests/check.py`)

Niezależna referencja numpy (`float64`). Sprawdza indeksy i odległości (tolerancja
1e-4 bezwzględnie + 1e-4 względnie; zamiana indeksów dopuszczona tylko przy remisie
w granicach tolerancji) dla każdej wersji i kilku liczb wątków/procesów. Przypadki:
k=1, k=N, k>N (program odmawia), N=1, małe N, duplikaty, wszystkie wektory równe,
wektor zerowy, kratka całkowita (wiele remisów), N niepodzielne przez p, D=1, D=517.
cuBLAS ma tolerancję 2e-3, bo wzór |q|² + |b|² − 2q·b traci dokładność dla prawie
identycznych wektorów.

## Ograniczenia

- Kernel CUDA i selekcja w cuBLAS: k ≤ 64 (`MAX_K`). Każdy wątek bloku trzyma listę
  k par (odległość, indeks) w pamięci współdzielonej: T·k·8 B + D·4 B na blok, a RTX 4060
  pozwala na 99 KiB. Dla k = 64 program sam zmniejsza blok z 256 do 128 wątków
  (zapisuje to w kolumnie `block`).
- cuBLAS liczy partię kawałkami tak, żeby macierz odległości zajmowała najwyżej 512 MiB VRAM.
