# Sprawozdanie: Równoległe wyszukiwanie top-k sąsiadów wektorowych

**Autor:** Jakub Balicki  
**Projekt:** D03 — Wyszukiwanie wektorowe top-k  
**Przedmiot:** Programowanie Równoległe i Rozproszone

---

## 1. Opis problemu

Wyszukiwanie top-k najbliższych sąsiadów (kNN) w przestrzeni wektorowej metodą brute-force.  
Dane wejściowe: baza N wektorów (float32, D wymiarów), Q zapytań, wynik: k najbliższych indeksów wg odległości euklidesowej.

Złożoność: O(N·D) na zapytanie → przy N=1M i D=128 to 128M operacji FP per zapytanie.

Parametry benchmarków: N=100 000, D=128, Q=20, k=10 (oraz N=1M dla porównania skali).

---

## 2. Implementacja referencyjna

Pojedynczy wątek, STL. Algorytm: dla każdego zapytania oblicz N odległości euklidesowych (kwadrat), następnie `std::partial_sort` wyznacza k najmniejszych, zwróć k indeksów.

**Wyniki (N=100k):**
- P95 latencja: 6120,6 µs
- QPS: 168,3

**Wyniki (N=1M):**
- P95 latencja: 62 370 µs (62,4 ms)
- QPS: 16,7

---

## 3. Wersja OpenMP

Strategia: zrównoleglenie pętli obliczania odległości przez `#pragma omp parallel for schedule(dynamic,64)`. Każdy wątek zapisuje odległości do przydzielonego zakresu wspólnej tablicy `dists[N]` (brak wyścigu danych). Po zakończeniu pętli sekwencyjny `std::partial_sort` wyznacza top-k. Kompilacja z `-fopenmp -O3`.

**Wyniki (N=100k, Q=20):**

| Wątki | P95 (µs) | QPS   | Speedup P95 | Speedup QPS | Efektywność |
|-------|----------|-------|-------------|-------------|-------------|
| 1     | 6 569    | 166,8 | 1,00x       | 1,00x       | 1,000       |
| 2     | 3 522    | 316,9 | 1,87x       | 1,90x       | 0,950       |
| 4     | 2 099    | 590,5 | 3,13x       | 3,54x       | 0,885       |
| 8     | 1 638    | 885,9 | 4,01x       | 5,31x       | 0,664       |

Poprawność: wyniki identyczne z referencją dla 1/2/4/8 wątków (VERIFY OK).

**Obserwacje:** Skalowanie wyraźnie sublinearne — efekt prawa Amdahla. Sekwencyjny `partial_sort` oraz narzut synchronizacji OMP ograniczają speedup. Efektywność równoległa spada z 0,95 przy 2 wątkach do 0,66 przy 8 wątkach. Pomiędzy 4 a 8 wątkami przyrost speedup QPS zmniejsza się (z 3,54x do 5,31x — mniejszy skok niż z 1 → 4 wątków), co wskazuje na wyczerpanie równoległości obliczeniowej na dostępnych rdzeniach.

---

## 4. Wersja MPI

Strategia: podział bazy wektorów między procesy przez `MPI_Scatter`. Każdy proces oblicza odległości dla swojego fragmentu bazy (N/P wektorów), buduje lokalny top-k, przesyła wyniki do procesu rank=0 przez `MPI_Gather`. Rank=0 wykonuje merge k·P kandydatów i wyznacza globalne top-k. Zapytanie rozgłaszane przez `MPI_Bcast`.

**Wyniki (N=100k, Q=20):**

| Procesy | P95 (µs) | QPS   | Czas obliczeń rank0 | Speedup (compute) |
|---------|----------|-------|---------------------|-------------------|
| 1       | 6 104    | 112,3 | 117,5 ms            | 1,0x              |
| 2       | 3 105    | 180,8 | 59,4 ms             | 1,97x             |
| 4       | 1 595    | 249,1 | 30,4 ms             | 3,87x             |

Czas merge: pomijalny (< 0,02 ms). Overhead komunikacji (Scatter/Bcast) ogranicza efektywne QPS względem OpenMP przy podobnej liczbie jednostek.

**Obserwacje:** Speedup czasu samych obliczeń bliski liniowemu (3,87x przy 4 procesach), jednak całkowite QPS niższe niż OpenMP z powodu narzutu MPI. MPI sensowne w środowiskach klastrowych (wiele węzłów); na jednej maszynie OpenMP jest korzystniejszy.

---

## 5. Wersja CUDA

Architektura kernela:
- 1 blok CUDA = 1 zapytanie (gridDim.x = Q), 256 wątków/blok
- Wątki przeglądają bazę grid-stride, utrzymując lokalny top-k w rejestrach
- Redukcja top-k przez shared memory (zapytanie w shared + bufor kandydatów)
- Odległość euklidesowa (kwadrat w kernelu, sqrt na końcu), przy remisie — mniejszy indeks

Sprzęt: RTX 4060, Ada Lovelace sm\_89, CUDA 12.4.

**Wyniki N=100k:**

| Tryb         | Czas (ms) | QPS    | Speedup vs ref |
|--------------|-----------|--------|----------------|
| Kernel only  | 5,856     | 3415,2 | ~20,3x         |
| Total (H↔D) | 13,801    | 1449,1 | ~8,6x          |
| Transfer H→D | 7,945     | —      | —              |

**Wyniki N=1M:**

| Tryb         | Czas (ms) | QPS   | Speedup vs ref (P95=62,4ms) |
|--------------|-----------|-------|-----------------------------|
| Kernel only  | 53,804    | 371,7 | 23,2x                       |
| Total (H↔D) | 132,034   | 151,5 | 9,4x                        |
| Transfer H→D | 78,230    | —     | —                           |

Poprawność: indeksy top-k identyczne z referencją dla N=100k i N=1M (VERIFY OK).

**Obserwacje:** Transfer danych H→D dominuje przy N=1M (78,2 ms z 132 ms całości). Przy stałej bazie w pamięci GPU (bez transferu) CUDA osiąga 23,2x przyspieszenia. RTX 4060 z 8 GB VRAM pozwala na przechowywanie bazy nawet dla N=1M (512 MB @ float32 × 128D).

---

## 6. Porównanie

Poniższa tabela prezentuje dwie metryki dla N=100k:

| Wersja         | P95 latencja (µs) | QPS    | Speedup QPS |
|----------------|-------------------|--------|-------------|
| Reference 1T   | 6 121             | 168,3  | 1,0x        |
| OpenMP 2T      | 3 522             | 316,9  | 1,90x       |
| OpenMP 4T      | 2 099             | 590,5  | 3,54x       |
| OpenMP 8T      | 1 638             | 885,9  | 5,31x       |
| MPI 2P         | 3 105             | 180,8  | 1,08x       |
| MPI 4P         | 1 595             | 249,1  | 1,48x       |
| CUDA kernel    | ~294 (5,856ms/20) | 3415,2 | 20,3x       |
| CUDA total     | 690               | 1449,1 | 8,6x        |

CUDA kernel wyprzedza wszystkie warianty CPU ponad 5-krotnie nawet względem OpenMP 8T. OpenMP przewyższa MPI na jednej maszynie — MPI traci na narzucie komunikacji.

---

## 7. Wnioski

1. **CUDA** osiąga najwyższe przyspieszenie (23,2x przy N=1M, kernel), co czyni ją optymalnym wyborem dla dużych baz statycznych przechowywanych w pamięci GPU.
2. **OpenMP** oferuje dobre skalowanie (3,80x QPS przy 8 wątkach) bez narzutu komunikacyjnego — praktyczne przy małych i średnich N oraz braku GPU.
3. **MPI** skaluje się niemal liniowo pod względem czasu obliczeń (3,87x przy 4 procesach), ale overhead Scatter/Bcast obniża realne QPS. Uzasadnione w środowiskach klastrowych z wieloma węzłami.
4. Bottleneck transferu H→D w CUDA (~78 ms dla N=1M) podkreśla znaczenie architektury z bazą rezydującą stale na GPU — kluczowe przy scenariuszach produkcyjnych (ANN serving).
5. Prawo Amdahla widoczne w OpenMP: przejście z 4 → 8 wątków daje mniejszy przyrost (3,54x → 5,31x QPS) niż 1 → 4 wątki (1,0x → 3,54x QPS). Efektywność spada od 95% przy 2 wątkach do 66% przy 8 wątkach.
