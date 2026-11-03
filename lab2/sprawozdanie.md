# Sprawozdanie – Laboratorium 2: OpenMP

**Przedmiot:** Programowanie Równoległe i Rozproszone  
**Temat:** Przetwarzanie chmury punktów 3D z użyciem OpenMP  
**Data:** 2026-11-03

---

## 1. Opis zadania i danych

Dane wejściowe to chmura punktów 3D w formacie tekstowym: trzy kolumny wartości zmiennoprzecinkowych (x, y, z), po jednym punkcie na wiersz. Plik `plik_3kolumny.txt` zawiera **N = 6 865 878 punktów** i zajmuje 175 MB na dysku. Dane reprezentują prawdziwy teren — współrzędne x, y to układ metryczny (zakres ~800 m × 600 m), z to wysokość (od ~180 do ~260 m n.p.m.).

Program `lab2_openmp.cpp` realizuje cztery zadania obliczeniowe na tej chmurze, każde z użyciem dyrektyw OpenMP. Pomiary czasu zbierano dla 1, 2, 4 i 8 wątków na tej samej maszynie — wyniki benchmarku pokazują, jak skaluje się każde z zadań.

---

## 2. Opis zadań i zastosowane dyrektywy OpenMP

### Zadanie 1 – Obliczenie centroidu

Centroid to punkt średni chmury: `cx = Σxᵢ/N`, analogicznie dla y i z. Wymaga trzykrotnej redukcji sumy po wszystkich N punktach.

**Dyrektywa:**
```cpp
#pragma omp parallel for schedule(static) \
    reduction(+:sum_x, sum_y, sum_z)
```

Klauzula `reduction(+:...)` zapewnia, że każdy wątek akumuluje częściowe sumy w prywatnych zmiennych, a na końcu pętli OpenMP scala je automatycznie. `schedule(static)` dzieli zakres iteracji równo — każdy wątek dostaje stały blok `N/num_threads` punktów, co jest optymalne przy jednakowym czasie przetwarzania każdej iteracji.

**Wynik:** centroid = (566836.22, 243447.88, 215.37)

---

### Zadanie 2 – Minimalna i maksymalna odległość od centroidu

Dla każdego punktu liczona jest odległość euklidesowa od centroidu, a następnie szukane globalne minimum i maksimum.

**Dyrektywa:**
```cpp
#pragma omp parallel for schedule(static) \
    reduction(min:dmin) reduction(max:dmax)
```

Redukcja `min` i `max` jest dostępna od OpenMP 3.1. Każdy wątek liczy lokalne minimum i maksimum, końcowe porównanie odbywa się na etapie łączenia częściowych wyników. Obliczenie `sqrt` w każdej iteracji powoduje, że zadanie 2 jest wyraźnie wolniejsze od zadania 1 — stąd lepiej widoczny speedup przy wielu wątkach.

**Wynik:** dmin = 2.48 m, dmax = 431.40 m

---

### Zadanie 3 – Normalizacja chmury do kostki [-1, 1]³

Każdy punkt jest przesuwany o centroid i skalowany przez `dmax`:
```
pts[i] = (pts[i] - centroid) / dmax
```
Operacja jest niezależna dla każdego punktu (embarrassingly parallel) — brak zależności danych między iteracjami.

**Dyrektywa:**
```cpp
#pragma omp parallel for schedule(static)
```

Brak klauzuli `reduction` — każda iteracja tylko zapisuje do `pts[i]`, bez współdzielenia między wątkami. `schedule(static)` sprawdza się tu dobrze, bo każda iteracja ma dokładnie taki sam koszt obliczeniowy.

---

### Zadanie 4 – Liczenie sąsiadów w promieniu r (próbka)

Dla każdego punktu z próbki sprawdzana jest odległość do wszystkich pozostałych punktów z tej samej próbki. Złożoność O(S²), gdzie S = 100 000. Zadanie czaso- i obliczeniochłonne, idealny kandydat do zrównoleglenia.

**Dyrektywa:**
```cpp
#pragma omp parallel for schedule(dynamic, 256)
```

`schedule(dynamic, 256)` zamiast `static` — choć każda iteracja ma porównywalny koszt, dynamiczne przydzielanie bloków po 256 iteracji zapobiega nierównomiernemu obciążeniu wątków przy cache misses i różnicach w rzeczywistym czasie wykonania. Zmienna `neighbor_count[i]` jest prywatna dla wątku (każdy wątek pisze tylko do swojego indeksu), więc nie ma wyścigu danych.

**Wynik:** średnia liczba sąsiadów w promieniu r=0.1 wynosi ~74 289 (po normalizacji), co oznacza gęste skupienie punktów.

---

## 3. Wyniki benchmarku

Pomiary na PC z 8 rdzeniami, kompilacja: `g++ -O2 -fopenmp -std=c++17`.

### Czasy wykonania [s]

| Zadanie                        | 1 wątek | 2 wątki | 4 wątki | 8 wątków |
|-------------------------------|---------|---------|---------|----------|
| 1 – Centroid                  | 0.0055  | 0.0036  | 0.0030  | 0.0032   |
| 2 – Min/max odległości        | 0.0135  | 0.0077  | 0.0041  | 0.0026   |
| 3 – Normalizacja              | 0.0103  | 0.0061  | 0.0040  | 0.0039   |
| 4 – Sąsiedzi (100k próbka)    | 10.2473 | 5.0686  | 2.6121  | 1.5027   |

### Speedup względem 1 wątku

| Zadanie                        | 1 wątek | 2 wątki | 4 wątki | 8 wątków |
|-------------------------------|---------|---------|---------|----------|
| 1 – Centroid                  | 1.00    | 1.53    | 1.83    | 1.72     |
| 2 – Min/max odległości        | 1.00    | 1.75    | 3.29    | 5.19     |
| 3 – Normalizacja              | 1.00    | 1.69    | 2.58    | 2.64     |
| 4 – Sąsiedzi (100k próbka)    | 1.00    | 2.02    | 3.92    | 6.82     |

---

## 4. Wnioski

Zadanie 4 (sąsiedzi) skaluje się najlepiej spośród wszystkich — speedup 6.82× przy 8 wątkach jest bliski idealnemu i wynika z tego, że czas obliczeń jest tu dominujący w stosunku do narzutu synchronizacji. Pozostałe trzy zadania operują na pełnych 6.8M punktach, ale ich łączny czas jest rzędu dziesiątek milisekund — przy tak krótkich operacjach narzut OpenMP (fork/join wątków, synchronizacja redukcji) stanowi istotną część całego czasu wykonania, co tłumaczy słabszy speedup, zwłaszcza przy przejściu z 4 do 8 wątków.

Zadanie 1 (centroid) przy 8 wątkach jest minimalnie wolniejsze niż przy 4 — czas rzędu 3 ms jest na tyle krótki, że losowe odchylenia timera i efekty cache dominują nad korzyścią z dodatkowych wątków. Zadania 2 i 4 korzystają z większej liczby wątków efektywnie, bo są bardziej obliczeniochłonne: zadanie 2 liczy pierwiastek kwadratowy dla każdego z 6.8M punktów, a zadanie 4 wykonuje ~10¹⁰ porównań dystansów na próbce 100k punktów.

Dobór `schedule(dynamic)` w zadaniu 4 w praktyce daje podobny wynik jak `static` przy równomiernej chmurze punktów, ale jest bezpieczniejszy dla danych z nierównomiernym rozkładem — klastry punktów powodowałyby różne czasy iteracji przy `static`, prowadząc do nierównomiernego obciążenia rdzeni.

---

## 5. Kompilacja i uruchomienie

**Wymagania:** `g++` z obsługą OpenMP i C++17 (`-fopenmp -std=c++17`)

```bash
# Kompilacja
make lab2
# lub ręcznie:
g++ -O2 -fopenmp -std=c++17 lab2_openmp.cpp -o lab2

# Uruchomienie z podaniem pliku i liczby wątków
./lab2 data/plik_3kolumny.txt 4

# Benchmark dla 1/2/4/8 wątków
make benchmark DATA=data/plik_3kolumny.txt
```

Plik danych nie jest częścią repozytorium (rozmiar 175 MB). Własny plik powinien zawierać trzy kolumny float oddzielone spacją, po jednym punkcie na wiersz.
