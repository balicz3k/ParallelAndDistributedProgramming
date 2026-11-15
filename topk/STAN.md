# Stan wyników projektu D03 (top-k) – diagnoza z 16.11.2026

Cel: zanim cokolwiek usunę lub zmierzę od nowa, spisuję, jakie pliki z wynikami są w repo,
skąd pochodzą, z jakimi parametrami i gdzie są używane. Na końcu propozycja jednej konfiguracji
pomiarów (do akceptacji).

## 1. Pliki wyników w repo

Daty: ostatni commit, który zmienił plik. Prawie wszystkie wyniki są z przebiegów 04.11
(nagłówki plików), a zacommitowano je 15.11 w commicie `bf3774e`.

| Plik | Skąd pochodzi | N / D / Q / k | Zapytania | Aktualny? | Gdzie użyty |
|---|---|---|---|---|---|
| `results.txt` (katalog główny) | ranking `topk_ref` (domyślne wyjście, skopiowane ręcznie) | 100k / 128 / 100 / 10 | `queries.bin` seed 42 | stary artefakt | nigdzie |
| `results_omp_t1/t8/t16.txt` (katalog główny) | ranking `topk_omp`, stary przebieg | 100k / 128 / 100 / 10 | seed 42 | stary artefakt, dubluje `openmp/` z innym Q | nigdzie |
| `openmp/results_omp.txt` | `openmp/run_bench_omp.sh` (`make -C openmp bench`), 04.11 19:41 | 100k / 128 / **250** / 10 (+ sweep N, k, Q≤250) | **`queries_bench.bin`, seed 20260404** | tak, ale inne zapytania niż reszta | `scripts/plots.py`, tabela 1 w sprawozdaniu |
| `openmp/results_omp_t1/t4/t8/t16.txt` | rankingi z `run_bench_omp.sh` | 100k / 128 / 250 / 10 | `queries_bench.bin` | tak | weryfikacja, nigdzie w raporcie |
| `mpi/results_mpi.txt` | `mpi/run_bench.py` (`make -C mpi bench`), 04.11 19:39 | 100k / 128 / **100** / 10 (+ sweep N, D, k, Q) | `mpi/data_mpi/…` seed 42 (te same bajty co `data/n100000_q100`) | tak | `scripts/plots.py`, tabele 1–2 w sprawozdaniu |
| `mpi/verify_p1/p2/p4/p8.txt` | rankingi z `run_bench.py --verify` | 100k / 128 / 100 / 10 | seed 42 | tak | weryfikacja |
| `cuda/results_cuda.txt` | `make -C cuda bench` + `bench-k` | 10k–1M / 128 / **20** / 1–50 | `data/n*/queries.bin` seed 42 | tak | `scripts/plots.py`, rys. CUDA w sprawozdaniu |
| `reference/results.txt` | ranking `topk_ref` (ostatnie uruchomienie) | 100k / 128 / 250 / 10 | `queries_bench.bin` | nie jest w repo (gitignore), ale z niego weryfikuje OMP | – |
| `reference/benchmark.py` + `benchmark_ref.png` | stary skrypt z 03.11 (N=10k/100k/1M, jeden przebieg) | – | – | przestarzały | nigdzie |
| `comparison_chart.png` | commit z 04.11 01:01 | – | – | przestarzały | nigdzie |
| `sprawozdanie.md` | stary raport Markdown (Q=20) | – | – | przestarzały, sprzeczny z LaTeX-em | – |
| `plots/*.png` (8 plików) | `scripts/plots.py` z trzech plików `results_*.txt` | mieszane Q | – | tak | 4 z nich skopiowane do `sprawozdanie/fig/` (identyczne) |
| `sprawozdanie/fig/*.png` (4) | kopie z `plots/` | mieszane Q | – | tak | `sprawozdanie.tex` |
| `sprawozdanie/{qps_compare,scaling,speedup}.png` | `sprawozdanie/make_plots.py` – **liczby wpisane ręcznie w skrypcie** (OMP QPS 169,2/315,8/…, MPI p≤4) | opis mówi Q=20 | – | nie pasują do żadnego pliku wyników | nigdzie (sprawozdanie ich nie dołącza) |

Dane: `data/` (gitignore) generuje `scripts/gen_data.sh` (seed 42, domyślnie Q=20; osobne katalogi
`n100000_q1/q100/q1000`, `_d32/64/256/512`). **`queries_bench.bin` nie jest generowany przez żaden
skrypt w repo** – nie da się go odtworzyć. MPI generuje własną kopię danych w `mpi/data_mpi/`.

## 2. Sprzeczności i braki

1. **Trzy różne Q:** OpenMP Q=250, MPI Q=100, CUDA Q=20. README sam pisze, że QPS „jest tylko
   orientacyjny”. Sprawozdanie zestawia w tabeli 1 OpenMP (Q=250) i MPI (Q=100) obok siebie.
2. **Dwa różne zbiory zapytań:** OpenMP mierzy na `queries_bench.bin` (seed 20260404),
   MPI/CUDA na `queries.bin` (seed 42). Wyniki rankingu OMP i MPI dotyczą innych zapytań.
3. **Tabela 2 sprawozdania („wspólne Q=100”) nie ma źródła w repo.** Liczby 596,4 / 80,3 / 8,95 /
   2,17 ms, QPS 11167 i 46064, P95 1917 µs nie występują w żadnym pliku wyników (CUDA ma tylko
   Q=20, OpenMP przy Q=100 ma tylko 16 wątków: 69,99 ms, a w tabeli jest 66,0). Na podstawie tej
   tabeli liczone są przyspieszenia 67× i 275×.
4. **Referencja bez powtórzeń:** `topk_ref` mierzy jeden przebieg (brak warmup/repeat), a jest
   bazą przyspieszeń w tabeli 2. Czas referencji (596,4 ms) też nie ma pliku źródłowego.
5. **Niespójności wewnątrz `mpi/results_mpi.txt`:** ten sam przypadek (N=100k, D=128, Q=100, k=10,
   P=1) daje 598 ms w sekcji 2, 603 ms w sweepie D i **905 ms** w sweepie k; P=8: 93 ms w sekcji 2,
   **156 ms** w sweepie D. Sweep k pokazuje, że k=1 jest wolniejsze od k=100 (1421 vs 609 ms) –
   to wygląda na zakłócenia (inne obciążenie maszyny), a nie na cechę algorytmu.
6. **Podobnie w OpenMP:** k=1 (353 ms, IQR 196 ms) wolniejsze niż k=10 (168 ms); dla 1 wątku
   max 3803 ms przy medianie 1533 ms (IQR 1174 ms) – pomiar z zakłóceniem, a od niego liczone S(p).
7. **CUDA: nagłówek pliku podaje inne liczby niż dane poniżej** (N=1M: „14,4 ms / 1,39k QPS” vs
   zmierzone 15,56 ms / 1285 QPS). Ten sam przypadek N=100k, k=10: 1,78 ms w sweepie N i 1,62 ms
   w sweepie k.
8. **P95 liczony różnie w każdej wersji** (OMP – czas zapytania liczonego przez 1 wątek w trybie
   „po zapytaniach”, MPI – czas zapytania na wszystkich procesach, CUDA – osobne uruchomienie
   jednego bloku na zapytanie, referencja – czas pętli dla jednego zapytania). Sprawozdanie
   to przyznaje, ale wtedy kolumna P95 w jednej tabeli nie nadaje się do porównań.
9. **Brak sweepu bloków CUDA** (`--threads` 64/128/256/512) – wymóg „wpływ konfiguracji bloków”.
10. **Różne formaty i trzy parsery:** `scripts/plots.py` parsuje trzy formaty tekstowe;
    `sprawozdanie/make_plots.py` ma liczby wpisane ręcznie. Dwa komplety wykresów (`plots/` i
    `sprawozdanie/fig/` + 3 PNG w `sprawozdanie/`).
11. **Dwie ścieżki uruchamiania pomiarów:** cele `bench` w trzech Makefile'ach (każdy z innymi
    parametrami) + główny `make bench`. Referencja nie ma celu `bench`.
12. **CLI niespójne:** `topk_ref` – tylko 6 argumentów pozycyjnych; `topk_omp` – argumenty
    pozycyjne `[threads] [repeat] [ref.txt]`; `topk_mpi` i `topk_cuda` – opcje `--repeat`,
    `--warmup`. Żadna wersja nie wypisuje CSV w tym samym formacie.
13. **Limit k ≤ 64 w kernelu CUDA** (`MAX_K`), więc k=100 z planu nie zadziała bez zmian.
14. **Historia git:** 3 duże commity z 15.11 i kilka z 04.11 w odstępach minut.

## 3. Propozycja jednej konfiguracji (do akceptacji)

Wspólne dla wszystkich wersji: euklides, float32, remis → mniejszy indeks, **jeden plik zapytań
na (N, D)** wygenerowany z seed 42 z Q_max = 10 000; mniejsze Q to pierwsze Q wierszy tego pliku
(wersje CPU czytają dokładnie Q·D liczb od początku pliku, więc wystarczy podać Q; wczytywanie
w wersjach CUDA sprawdzę przy B4.3). `queries_bench.bin` znika.

| Parametr | Wartości | Pozostałe parametry |
|---|---|---|
| **przypadek główny** | **N=100 000, D=128, Q=1000, k=10** | – |
| N | 10⁴, 10⁵, 10⁶ | D=128, Q=1000, k=10 |
| D | 32, 64, 128, 256, 512 | N=10⁵, Q=1000, k=10 |
| k | 1, 10, 50, 100 (CUDA: 1, 10, 50; k=100 raportowane jako „brak – limit shared memory”, chyba że podniosę limit przez mniej wątków) | N=10⁵, D=128, Q=1000 |
| Q (partia) | 1, 10, 100, 1000, 10 000 | N=10⁵, D=128, k=10 |
| wątki OpenMP | 1, 2, 4, 8, 16 | przypadek główny |
| procesy MPI | 1, 2, 4, 8 (+ hybryda 8×2, 4×4, 2×8) | przypadek główny |
| bloki CUDA | 64, 128, 256, 512 wątków | przypadek główny |
| powtórzenia | warmup 2 + repeat 7, mediana, IQR, min/max | wszystkie |

Każda wersja (także referencja) dostaje `--warmup`, `--repeat` i `--csv` i wypisuje jeden wiersz
CSV na powtórzenie do `results/raw.csv`. Jedna definicja P95: czas pojedynczego zapytania
w trybie, w którym każda wersja liczy zapytania w partii (opiszę dokładnie przy B4.3).

Szacunek czasu pełnego przebiegu (z czasów na zapytanie w obecnych plikach: referencja ok.
6 ms/zapytanie dla N=10⁵, D=128, czyli ok. 6 s na partię Q=1000): przypadki jednowątkowe
(referencja, OMP t=1, MPI p=1) dla N=10⁶ to ok. 60 s na partię × 9 przebiegów ≈ 9 min każdy, dla
D=512 ok. 5 min. Razem szacuję ok. 1–1,5 h; przed pełnym przebiegiem zrobię suchy przebieg
z repeat=1 i ewentualnie podzielę go na części.

## 4. Co usunę po akceptacji (osobny commit)

`results.txt`, `results_omp_t*.txt` (katalog główny), `comparison_chart.png`, `sprawozdanie.md`,
`openmp/results_omp*.txt`, `mpi/results_mpi.txt`, `mpi/verify_p*.txt`, `cuda/results_cuda.txt`,
`reference/benchmark.py`, `reference/benchmark_ref.png`, `plots/`, `sprawozdanie/make_plots.py`,
`sprawozdanie/{qps_compare,scaling,speedup}.png`, `openmp/run_bench_omp.sh` i `mpi/run_bench.py`
(zastąpione przez `scripts/run_all.sh`). Zostaną: kod, `tests/`, `scripts/` i nowe `results/`.

## 5. Pytania

1. Czy akceptujesz przypadek główny N=100k, D=128, Q=1000, k=10 i listę z punktu 3?
2. Czy mogę usunąć pliki z punktu 4?
3. k=100 w CUDA: zostawić limit 64 i opisać ograniczenie, czy podnieść limit kosztem mniejszej
   liczby wątków na blok?
