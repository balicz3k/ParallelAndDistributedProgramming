# PRiR Laboratorium 2 — OpenMP: Przetwarzanie Chmury Punktów

## Opis zadań

Program przetwarza plik chmury punktów 3D w formacie `x y z` (float, jeden punkt na linię).  
Dla każdego zadania mierzony jest czas wykonania.

### Zadanie 1 — Centroid
Oblicza środek ciężkości chmury punktów używając redukcji OpenMP (`reduction(+:...)`).

### Zadanie 2 — Min/max odległości
Oblicza minimalną i maksymalną odległość euklidesową każdego punktu od centroidu.  
Używa `reduction(min:...)` i `reduction(max:...)`.

### Zadanie 3 — Normalizacja
Przesuwa chmurę punktów tak, by centroid był w [0,0,0], a następnie skaluje przez `dmax`,  
tak że wszystkie punkty mieszczą się w sferze jednostkowej (i kostce [-1,1]³).

### Zadanie 4 — Liczenie sąsiadów
Dla próbki 100 000 punktów liczy, ile innych punktów z tej próbki leży w promieniu `r = 0.1`  
(po normalizacji). Używa `schedule(dynamic, 256)`.  
Złożoność O(N²) — celowo ograniczona do próbki, żeby zmieścić się w rozsądnym czasie.

---

## Kompilacja

```bash
make lab2
# lub ręcznie:
g++ -O2 -fopenmp -std=c++17 lab2_openmp.cpp -o lab2
```

## Uruchomienie

```bash
# Plik danych jako argument, liczba wątków opcjonalna
./lab2 data/chmura_punktow.txt 8

# Lub przez zmienną środowiskową
OMP_NUM_THREADS=4 ./lab2 data/chmura_punktow.txt
```

## Benchmark (wszystkie wątki: 1, 2, 4, 8)

```bash
make benchmark
# Lub inny plik:
make benchmark DATA=/ścieżka/do/pliku.txt
```

## Wymagania

- `g++` z obsługą OpenMP (`-fopenmp`) i C++17
- System Linux/macOS

## Format danych

Plik tekstowy, jeden punkt na linię:
```
1.234 5.678 9.012
3.456 7.890 1.234
...
```
