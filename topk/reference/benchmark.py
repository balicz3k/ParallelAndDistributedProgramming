#!/usr/bin/env python3
"""
benchmark.py — pomiar czasu referencyjna implementacja top-k dla różnych N

Generuje dane dla N = 10 000, 100 000, 1 000 000 i uruchamia topk_ref,
parsuje wyniki, rysuje wykres i zapisuje go do benchmark_ref.png.

Uruchomienie (z katalogu topk/):
    python3 reference/benchmark.py
"""

import os
import re
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")  # bez GUI (serwer/SSH)
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TOPK_DIR     = os.path.dirname(SCRIPT_DIR)          # katalog topk/
TOPK_REF     = os.path.join(SCRIPT_DIR, "topk_ref")
GEN_SCRIPT   = os.path.join(TOPK_DIR, "generate_data.py")
DATA_DIR     = os.path.join(TOPK_DIR, "data")

D   = 128   # liczba wymiarów (stała podczas benchmarku)
Q   = 20    # liczba zapytań na każdy pomiar
K   = 10    # liczba sąsiadów

NS  = [10_000, 100_000, 1_000_000]  # rozmiary bazy do przetestowania

# ---------------------------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------------------------

def run(cmd: list, cwd: str = None) -> str:
    """Uruchom polecenie i zwróć stdout jako string. Rzuć wyjątek przy błędzie."""
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"BLAD: {' '.join(cmd)}")
        print(result.stderr)
        sys.exit(1)
    return result.stdout


def generate_data(n: int) -> tuple[str, str]:
    """Wygeneruj dane dla danego N; zwróć (base_path, queries_path)."""
    out_dir = os.path.join(DATA_DIR, f"n{n}")
    base_path    = os.path.join(out_dir, "base.bin")
    queries_path = os.path.join(out_dir, "queries.bin")

    if os.path.exists(base_path) and os.path.exists(queries_path):
        print(f"  Dane dla N={n:,} już istnieją, pomijam generowanie.")
        return base_path, queries_path

    print(f"  Generowanie danych N={n:,} D={D} Q={Q} ...")
    run([sys.executable, GEN_SCRIPT,
         "--n", str(n), "--d", str(D), "--q", str(Q),
         "--out_dir", out_dir])
    return base_path, queries_path


def parse_stats(stdout: str) -> dict:
    """Wyciągnij liczby ze statystyk wypisywanych przez topk_ref."""
    stats = {}
    patterns = {
        "total_ms": r"Calkowity czas:\s+([\d.]+) ms",
        "avg_us":   r"Sredni czas / query:\s+([\d.]+) us",
        "p95_us":   r"P95 latency:\s+([\d.]+) us",
        "qps":      r"Przepustowosc:\s+([\d.]+) zapytan",
    }
    for key, pat in patterns.items():
        m = re.search(pat, stdout)
        if m:
            stats[key] = float(m.group(1))
    return stats


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(TOPK_REF):
        print(f"Brak pliku {TOPK_REF}. Kompiluj najpierw: make -C {SCRIPT_DIR}")
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)

    results = []
    for n in NS:
        print(f"\n--- N = {n:,} ---")
        base_path, queries_path = generate_data(n)

        print(f"  Uruchamianie topk_ref ...")
        stdout = run([TOPK_REF, base_path, queries_path,
                      str(n), str(D), str(Q), str(K)])
        stats = parse_stats(stdout)
        print(f"  total_ms={stats.get('total_ms'):.1f}  "
              f"avg_us={stats.get('avg_us'):.1f}  "
              f"p95_us={stats.get('p95_us'):.1f}  "
              f"qps={stats.get('qps'):.1f}")
        results.append({"n": n, **stats})

    # -----------------------------------------------------------------------
    # Wykres
    # -----------------------------------------------------------------------
    ns_arr  = np.array([r["n"]       for r in results])
    avg_ms  = np.array([r["avg_us"]  for r in results]) / 1000.0  # µs → ms
    p95_ms  = np.array([r["p95_us"]  for r in results]) / 1000.0
    qps_arr = np.array([r["qps"]     for r in results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Top-k brute-force (referencyjna) — D={D}, Q={Q}, k={K}",
                 fontsize=13)

    # Lewy panel: czas zapytania vs N
    ax1.plot(ns_arr, avg_ms, "o-", label="Avg latency")
    ax1.plot(ns_arr, p95_ms, "s--", label="P95 latency")
    ax1.set_xscale("log")
    ax1.set_xlabel("N (rozmiar bazy)")
    ax1.set_ylabel("Czas zapytania [ms]")
    ax1.set_title("Czas zapytania vs rozmiar bazy")
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.set_xticks(ns_arr)
    ax1.set_xticklabels([f"{n//1000}k" if n < 1_000_000 else "1M" for n in ns_arr])

    # Prawy panel: przepustowość vs N
    ax2.plot(ns_arr, qps_arr, "o-", color="tab:orange")
    ax2.set_xscale("log")
    ax2.set_xlabel("N (rozmiar bazy)")
    ax2.set_ylabel("Przepustowość [zapytań/s]")
    ax2.set_title("Przepustowość vs rozmiar bazy")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.set_xticks(ns_arr)
    ax2.set_xticklabels([f"{n//1000}k" if n < 1_000_000 else "1M" for n in ns_arr])

    plt.tight_layout()
    out_png = os.path.join(SCRIPT_DIR, "benchmark_ref.png")
    plt.savefig(out_png, dpi=150)
    print(f"\nWykres zapisany: {out_png}")


if __name__ == "__main__":
    main()
