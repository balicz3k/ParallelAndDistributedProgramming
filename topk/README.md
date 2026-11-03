# PRiR D03 — Top-k nearest neighbors

Projekt D03: brute-force top-k wyszukiwanie wektorowe (implementacja referencyjna).

## Struktura

```
topk/
  reference/
    topk_ref.cpp   — implementacja referencyjna (single-thread, brute-force)
    Makefile
    benchmark.py   — skrypt benchmarkowy, generuje benchmark_ref.png
  generate_data.py — generator wektorów float32
  data/            — wygenerowane pliki .bin (tworzony automatycznie)
  README.md
```

## Szybki start

```bash
# 1. Kompilacja
make -C reference

# 2. Generowanie danych testowych (N=10000, D=128, Q=10)
python3 generate_data.py --n 10000 --d 128 --q 10 --out_dir ./data

# 3. Uruchomienie top-k (k=5)
./reference/topk_ref data/base.bin data/queries.bin 10000 128 10 5

# 4. Benchmark (N = 10k, 100k, 1M)
python3 reference/benchmark.py
```

## Format pliku binarnego

Pliki `.bin` to raw float32, row-major (C order), bez nagłówka.
Każdy wiersz to jeden wektor D-wymiarowy.

Wczytanie w Pythonie:
```python
import numpy as np
base = np.fromfile("data/base.bin", dtype=np.float32).reshape(N, D)
```

## Parametry `topk_ref`

```
./topk_ref <baza.bin> <zapytania.bin> <N> <D> <Q> <k>
```

| Argument     | Opis                              |
|--------------|-----------------------------------|
| baza.bin     | Plik bazy wektorów (N × D float32)|
| zapytania.bin| Plik zapytań (Q × D float32)      |
| N            | Liczba wektorów w bazie           |
| D            | Liczba wymiarów                   |
| Q            | Liczba zapytań                    |
| k            | Liczba zwracanych sąsiadów        |

Wyniki zapisywane do `results.txt` w bieżącym katalogu.
