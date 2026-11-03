#!/usr/bin/env python3
"""
generate_data.py — generator float32 wektorów dla top-k benchmark

Użycie:
    python3 generate_data.py --n 1000000 --d 128 --q 100 --out_dir ./data

Generuje:
    <out_dir>/base.bin    — N x D wartości float32 (wektory bazy)
    <out_dir>/queries.bin — Q x D wartości float32 (zapytania)

Format binarny: raw float32, row-major (C order), bez nagłówka.
Wektory losowane z rozkładu N(0, 1) i normalizowane do długości 1
(unit sphere) — standard w benchmarkach ANN (np. SIFT, GloVe).
"""

import argparse
import os
import struct
import sys
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Generator wektorów float32 dla top-k")
    p.add_argument("--n",       type=int, default=10_000,
                   help="Liczba wektorów bazy (domyślnie 10000)")
    p.add_argument("--d",       type=int, default=128,
                   help="Liczba wymiarów (domyślnie 128)")
    p.add_argument("--q",       type=int, default=100,
                   help="Liczba zapytań (domyślnie 100)")
    p.add_argument("--seed",    type=int, default=42,
                   help="Ziarno losowości (domyślnie 42)")
    p.add_argument("--out_dir", type=str, default="./data",
                   help="Katalog wyjściowy (domyślnie ./data)")
    p.add_argument("--normalize", action="store_true",
                   help="Normalizuj wektory do długości 1 (unit sphere)")
    return p.parse_args()


def save_bin(path: str, arr: np.ndarray):
    """Zapisz tablicę float32 do pliku binarnego (bez nagłówka)."""
    assert arr.dtype == np.float32, f"Oczekiwano float32, dostano {arr.dtype}"
    arr.tofile(path)
    size_mb = arr.nbytes / 1024 / 1024
    print(f"  Zapisano: {path}  ({arr.shape[0]} x {arr.shape[1]}, {size_mb:.2f} MB)")


def generate(n: int, d: int, rng: np.random.Generator,
             normalize: bool) -> np.ndarray:
    """Generuj macierz n x d float32."""
    data = rng.standard_normal((n, d)).astype(np.float32)
    if normalize:
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # zabezpieczenie przed /0
        data /= norms
    return data


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Generator danych top-k")
    print(f"  N (baza)    = {args.n:,}")
    print(f"  D (wymiary) = {args.d}")
    print(f"  Q (zapytania) = {args.q}")
    print(f"  seed        = {args.seed}")
    print(f"  katalog     = {args.out_dir}")
    print(f"  normalizacja = {args.normalize}")
    print()

    rng = np.random.default_rng(args.seed)

    t0 = time.perf_counter()
    base = generate(args.n, args.d, rng, args.normalize)
    queries = generate(args.q, args.d, rng, args.normalize)
    t1 = time.perf_counter()

    print(f"Generowanie: {(t1 - t0)*1000:.1f} ms")

    base_path    = os.path.join(args.out_dir, "base.bin")
    queries_path = os.path.join(args.out_dir, "queries.bin")

    save_bin(base_path,    base)
    save_bin(queries_path, queries)

    print()
    print("Gotowe. Uruchom teraz:")
    print(f"  ./reference/topk_ref {base_path} {queries_path} "
          f"{args.n} {args.d} {args.q} 10")


if __name__ == "__main__":
    main()
