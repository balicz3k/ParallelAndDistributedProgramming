# Laboratorium 3 — Konwolucja 2D z OpenMP

**Przedmiot:** Programowanie Równoległe i Rozproszone  
**Data:** 2026-11-03

---

## 1. Opis konwolucji 2D

Konwolucja 2D to operacja matematyczna polegająca na przesuwaniu małej macierzy (jądra, ang. *kernel*) po większej macierzy wejściowej i obliczaniu sumy ważonej elementów sąsiedztwa dla każdej pozycji. Formalnie dla macierzy wejściowej `M[H×W]` i jądra `K[Kh×Kw]` wynik w punkcie `(i,j)` to:

```
wynik[i][j] = Σ_{m=0}^{Kh-1} Σ_{n=0}^{Kw-1} M[i+m-Kh/2][j+n-Kw/2] * K[m][n]
```

Przy pikselach poza granicami stosuje się *zero-padding* — wartości spoza macierzy traktuje się jako 0.

Konwolucja 2D jest fundamentem przetwarzania obrazów (rozmycie Gaussa, detekcja krawędzi metodą Sobela, wyostrzanie) i głębokiego uczenia maszynowego (warstwy konwolucyjne CNN). Złożoność obliczeniowa to O(H·W·Kh·Kw), co przy dużych macierzach i jądrach daje miliardy operacji zmiennoprzecinkowych.

---

## 2. Strategie zrównoleglenia

### 2.1 Podstawowy `parallel for` (OMP)

Dyrektywa `#pragma omp parallel for schedule(static)` rozdziela iteracje zewnętrznej pętli `i` między wątki. Każdy wątek przetwarza określoną liczbę wierszy macierzy wynikowej. Podział jest statyczny — każdy wątek dostaje `H/num_threads` wierszy. Metoda prosta i skuteczna dla równomiernych danych, bo każdy wiersz wymaga takiej samej liczby operacji.

### 2.2 `collapse(2)` — zrównoleglenie obu pętli

Klauzula `collapse(2)` łączy pętle po `i` i `j` w jedną przestrzeń iteracji o rozmiarze `H×W`. Zamiast `H` bloków pracy, wątki otrzymują `H*W` zadań jednostkowych. Pozwala to lepiej zbalansować obciążenie przy małych macierzach, gdzie `H` jest za małe, by efektywnie podzielić między wiele wątków. Dla dużych macierzy efekt jest podobny do wariantu podstawowego.

### 2.3 `schedule(dynamic, 64)` — dynamiczne przydzielanie

Harmonogram dynamiczny z rozmiarem bloku 64 oznacza, że każdy wątek po skończeniu swojego bloku wierszy pobiera kolejny z puli. Koszt synchronizacji jest niższy niż przy bloku rozmiaru 1, a jednocześnie lepiej radzi sobie z niejednorodnością obciążenia (np. przy różnych architekturach cache lub hyperthreadingu). Dla konwolucji, gdzie każdy wiersz wymaga identycznej liczby operacji, przewaga nad statycznym podziałem jest marginalna — ujawnia się przy większych macierzach i większej liczbie wątków.

---

## 3. Wyniki pomiarów

Środowisko: kompilator g++ -O2 -fopenmp -std=c++17, macierze float (single precision), zero-padding.

### 3.1 Czasy wykonania [s]

#### 512×512, jądro 3×3

| Wątki | Seq   | OMP   | Collapse | Dynamic |
|-------|-------|-------|----------|---------|
| 1     | 0.003 | 0.002 | 0.002    | 0.002   |
| 4     | 0.002 | 0.001 | 0.001    | 0.001   |
| 8     | 0.002 | 0.001 | 0.001    | 0.000   |

#### 2048×2048, jądro 7×7

| Wątki | Seq   | OMP   | Collapse | Dynamic |
|-------|-------|-------|----------|---------|
| 1     | 0.150 | 0.144 | 0.131    | 0.146   |
| 4     | 0.149 | 0.043 | 0.035    | 0.045   |
| 8     | 0.151 | 0.034 | 0.031    | 0.030   |

#### 4096×4096, jądro 15×15

| Wątki | Seq   | OMP   | Collapse | Dynamic |
|-------|-------|-------|----------|---------|
| 1     | 2.642 | 2.520 | 2.477    | 2.472   |
| 4     | 2.743 | 0.701 | 0.672    | 0.670   |
| 8     | 2.639 | 0.522 | 0.523    | 0.458   |

### 3.2 Speedup (wzrost prędkości względem Seq)

#### 512×512, jądro 3×3

| Wątki | OMP   | Collapse | Dynamic |
|-------|-------|----------|---------|
| 1     | 1.13x | 1.05x    | 1.36x   |
| 4     | 2.58x | 2.81x    | 2.42x   |
| 8     | 3.67x | 4.19x    | 4.69x   |

#### 2048×2048, jądro 7×7

| Wątki | OMP   | Collapse | Dynamic |
|-------|-------|----------|---------|
| 1     | 1.05x | 1.15x    | 1.03x   |
| 4     | 3.46x | 4.21x    | 3.34x   |
| 8     | 4.38x | 4.89x    | 5.06x   |

#### 4096×4096, jądro 15×15

| Wątki | OMP   | Collapse | Dynamic |
|-------|-------|----------|---------|
| 1     | 1.05x | 1.07x    | 1.07x   |
| 4     | 3.91x | 4.08x    | 4.09x   |
| 8     | 5.06x | 5.04x    | 5.76x   |

---

## 4. Wnioski

Zrównoleglenie konwolucji 2D przez OpenMP daje znaczące przyspieszenie już przy 4 wątkach — dla dużych macierzy (4096×4096, jądro 15×15) osiągnięto speedup rzędu 4x przy 4 wątkach i blisko 5–6x przy 8 wątkach. Efektywność spada poniżej liniowej z powodu narzutu synchronizacji, ograniczeń przepustowości pamięci i efektów cache (konwolucja przy dużych jądrach czyta dane z wielu odległych lokalizacji).

Wariant `collapse(2)` regularnie wypada nieco lepiej od podstawowego `parallel for` — szczególnie dla małych macierzy, gdzie sama liczba wierszy nie wystarczyła do pełnego nasycenia wątków. Dynamiczny harmonogram (`schedule(dynamic, 64)`) bywa najszybszy przy 8 wątkach i dużych macierzach, bo lepiej ukrywa nieregularności w dostępie do pamięci podręcznej między wątkami.

Dla małych macierzy (512×512) przyspieszenie jest skromniejsze — koszt uruchomienia puli wątków stanowi znaczącą część całkowitego czasu obliczeń. Próg opłacalności zrównoleglenia pojawia się gdzieś między 512×512 a 2048×2048: dla 2048×2048 speedup przy 8 wątkach przekracza 4.5x, co czyni parallelizację zdecydowanie uzasadnioną.
