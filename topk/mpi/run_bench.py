#!/usr/bin/env python3
"""
run_bench.py — sterownik benchmarku MPI top-k (wariant LOKALNY, jedna maszyna).

Wszystkie pomiary wykonywane lokalnie na jednej maszynie (8 rdzeni / 16 watkow).
NIE jest to pomiar skalowalnosci klastra (zgodnie z wymaganiami karty D03 i
sekcja "Pomiary i organizacja pracy": wynik MPI na jednym komputerze oznaczyc
jako lokalny).

Metodyka:
  - warmup + >=5 mierzonych powtorzen na konfiguracje (binarka robi to wewnetrznie),
  - mediana + rozrzut (min/max) czasu calkowitego,
  - P95 liczony z POJEDYNCZYCH zapytan zebranych ze WSZYSTKICH powtorzen,
  - osobno czas rdzenia obliczen vs czas z komunikacja (Bcast/Scatterv/Gather),
  - S(p)=T(1)/T(p), E(p)=S(p)/p (na medianie total proc),
  - peak RSS per rank (max i suma),
  - sweep po liczbie procesow {1,2,4,8} (+16 z --oversubscribe) i po N/D/k/Q.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "topk_mpi")
HYBRID = os.path.join(HERE, "topk_mpi_hybrid")
REF = os.path.join(HERE, "..", "reference", "topk_ref")
GEN = os.path.join(HERE, "..", "generate_data.py")

REPEAT = 7      # >=5 mierzonych powtorzen
WARMUP = 2
PROCS = [1, 2, 4, 8]
PROCS_OVERSUB = [16]   # >fizyczne rdzenie -> wymaga --oversubscribe


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def mpirun_cmd(np_, extra_mpi, binpath, args):
    cmd = ["mpirun", "-np", str(np_)]
    cmd += extra_mpi
    cmd += [binpath] + args
    return cmd


def parse_json(stdout):
    for line in stdout.splitlines():
        if line.startswith("JSON "):
            return json.loads(line[5:])
    return None


def gen_data(data_dir, n, d, q, seed):
    base = os.path.join(data_dir, "base.bin")
    # odrozniamy datasety po parametrach -> unikalny podkatalog
    if os.path.exists(base):
        return data_dir
    os.makedirs(data_dir, exist_ok=True)
    r = run([sys.executable, GEN, "--n", str(n), "--d", str(d),
             "--q", str(q), "--seed", str(seed), "--out_dir", data_dir])
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        sys.exit("Blad generowania danych")
    return data_dir


def dataset_dir(root, n, d, q, seed):
    return os.path.join(root, f"n{n}_d{d}_q{q}_s{seed}")


def run_reference(data_dir, n, d, q, k):
    """Uruchom referencje, zwroc sciezke do jej results.txt (ranking)."""
    base = os.path.join(data_dir, "base.bin")
    qf = os.path.join(data_dir, "queries.bin")
    r = run([REF, base, qf, str(n), str(d), str(q), str(k)], cwd=data_dir)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        sys.exit("Blad referencji")
    return os.path.join(data_dir, "results.txt")


def load_ranking(path):
    """Wczytaj ranking: dict[(q,rank)] = index (ignoruje dystans/FP)."""
    rk = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                q, rank, idx = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            rk[(q, rank)] = idx
    return rk


def verify(data_dir, n, d, q, k, procs):
    """Porownaj ranking MPI z referencja dla kazdej liczby procesow."""
    base = os.path.join(data_dir, "base.bin")
    qf = os.path.join(data_dir, "queries.bin")
    ref_path = run_reference(data_dir, n, d, q, k)
    ref = load_ranking(ref_path)
    results = []
    for np_ in procs:
        extra = ["--oversubscribe"] if np_ > 8 else []
        out = os.path.join(HERE, f"verify_p{np_}.txt")
        cmd = mpirun_cmd(np_, extra, BIN,
                         [base, qf, str(n), str(d), str(q), str(k),
                          "--repeat", "1", "--quiet",
                          "--verify-out", out])
        r = run(cmd, cwd=HERE)
        if r.returncode != 0:
            results.append((np_, "BLAD", r.stderr.strip()[:200]))
            continue
        got = load_ranking(out)
        # porownaj indeksy; remisy rozstrzygane po mniejszym indeksie,
        # wiec ranking powinien byc identyczny
        mism = sum(1 for key in ref if ref.get(key) != got.get(key))
        total = len(ref)
        ok = "OK" if mism == 0 else f"ROZNICE={mism}/{total}"
        results.append((np_, ok, f"{total-mism}/{total} zgodnych"))
    return results


def bench_config(binpath, n, d, q, k, np_, data_dir, extra_mpi=None,
                 omp_threads=None):
    base = os.path.join(data_dir, "base.bin")
    qf = os.path.join(data_dir, "queries.bin")
    extra_mpi = list(extra_mpi or [])
    env = dict(os.environ)
    if omp_threads is not None:
        env["OMP_NUM_THREADS"] = str(omp_threads)
    cmd = mpirun_cmd(np_, extra_mpi, binpath,
                     [base, qf, str(n), str(d), str(q), str(k),
                      "--repeat", str(REPEAT), "--warmup", str(WARMUP),
                      "--json", "--no-write", "--quiet"])
    r = run(cmd, cwd=HERE, env=env)
    if r.returncode != 0:
        print("STDERR:", r.stderr[:500])
        return None
    return parse_json(r.stdout)


def fmt_row(cols, widths):
    return "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data_mpi")
    ap.add_argument("--N", type=int, default=100000)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--Q", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.data_dir)
    os.makedirs(root, exist_ok=True)
    # dataset bazowy
    base_ds = gen_data(root, args.N, args.D, args.Q, args.seed)

    if args.verify_only:
        res = verify(base_ds, args.N, args.D, args.Q, args.k, PROCS)
        print("\n=== WERYFIKACJA POPRAWNOSCI (ranking vs reference) ===")
        for np_, status, info in res:
            print(f"  P={np_:>2}: {status}   ({info})")
        bad = [r for r in res if r[1] != "OK"]
        sys.exit(1 if bad else 0)

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    hw = subprocess.run(["bash", "-lc",
        "echo CPU:$(nproc) threads; "
        "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'; "
        "free -m | awk '/Mem:/{print \"RAM: \"$2\" MiB\"}'; "
        "mpirun --version 2>/dev/null | head -1 || true; "
        "g++ --version | head -1"],
        capture_output=True, text=True).stdout.strip()

    emit("=" * 72)
    emit("BENCHMARK MPI top-k — WYNIKI LOKALNE (JEDNA MASZYNA)")
    emit("=" * 72)
    emit("UWAGA: wszystkie pomiary wykonane LOKALNIE na jednej maszynie.")
    emit("To NIE jest pomiar skalowalnosci klastra (karta D03, sekcja Pomiary).")
    emit("Procesy MPI wspoldziela te same rdzenie CPU i magistrale pamieci.")
    emit("")
    emit("Konfiguracja sprzetu/oprogramowania:")
    for l in hw.splitlines():
        emit("  " + l)
    emit(f"  Data pomiaru: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    emit(f"  Metodyka: warmup={WARMUP}, powtorzenia={REPEAT} (mediana+rozrzut),")
    emit(f"            P95 z pojedynczych zapytan ze WSZYSTKICH powtorzen,")
    emit(f"            ziarno generatora={args.seed} (zapisane).")
    emit("")

    # --- WERYFIKACJA ---
    emit("-" * 72)
    emit("1) POPRAWNOSC — ranking MPI vs referencja sekwencyjna")
    emit(f"   (N={args.N} D={args.D} Q={args.Q} k={args.k})")
    emit("-" * 72)
    vres = verify(base_ds, args.N, args.D, args.Q, args.k, PROCS)
    for np_, status, info in vres:
        emit(f"   P={np_:>2}: {status:<14} ({info})")
    emit("")

    # --- SKALOWANIE PO PROCESACH (staly problem) ---
    emit("-" * 72)
    emit("2) SKALOWANIE PO PROCESACH — staly problem")
    emit(f"   N={args.N} D={args.D} Q={args.Q} k={args.k}")
    emit("   czas compute = sam rdzen (lokalne top-k, max po rankach)")
    emit("   czas total   = compute + komunikacja (Bcast/Scatterv/Gather)")
    emit("-" * 72)
    hdr = ["P", "compute_ms", "comm_ms", "total_ms", "rozrzut[min-max]",
           "QPS", "P95_us", "RSS_max", "RSS_sum", "S(p)", "E(p)"]
    w = [3, 11, 9, 9, 18, 8, 9, 8, 9, 6, 6]
    emit(fmt_row(hdr, w))
    t1 = None
    all_procs = PROCS + PROCS_OVERSUB
    for np_ in all_procs:
        extra = ["--oversubscribe"] if np_ > 8 else []
        j = bench_config(BIN, args.N, args.D, args.Q, args.k, np_, base_ds, extra)
        if j is None:
            emit(fmt_row([np_, "BLAD"] + [""]*9, w))
            continue
        if np_ == 1:
            t1 = j["total_ms_med"]
        sp = t1 / j["total_ms_med"] if t1 else 0
        ep = sp / np_
        note = " *oversub" if np_ > 8 else ""
        emit(fmt_row([
            np_,
            f"{j['compute_ms_med']:.2f}",
            f"{j['comm_ms_med']:.2f}",
            f"{j['total_ms_med']:.2f}",
            f"{j['total_ms_min']:.2f}-{j['total_ms_max']:.2f}",
            f"{j['qps_med']:.0f}",
            f"{j['p95_us']:.0f}",
            f"{j['rss_max_mib']:.0f}",
            f"{j['rss_sum_mib']:.0f}",
            f"{sp:.2f}",
            f"{ep:.2f}{note}",
        ], w))
    emit("")
    emit("   Uwaga interpretacyjna: na jednej maszynie memory-bound rdzen")
    emit("   dzieli przepustowosc pamieci miedzy procesy, wiec E(p) spada")
    emit("   przy p>liczby kanalow RAM; P=16 (oversubscribe) to 2 procesy")
    emit("   na rdzen -> spodziewany brak zysku / regresja.")
    emit("")

    # --- SWEEP N ---
    def sweep(title, param, values, base_params):
        emit("-" * 72)
        emit(title)
        emit("-" * 72)
        hh = [param, "P", "compute_ms", "comm_ms", "total_ms", "QPS",
              "P95_us", "RSS_sum_MiB"]
        ww = [10, 3, 11, 9, 9, 8, 9, 11]
        emit(fmt_row(hh, ww))
        for val in values:
            p = dict(base_params)
            p[param] = val
            ds = dataset_dir(root, p["N"], p["D"], p["Q"], args.seed)
            gen_data(ds, p["N"], p["D"], p["Q"], args.seed)
            for np_ in [1, 8]:
                j = bench_config(BIN, p["N"], p["D"], p["Q"], p["k"], np_, ds)
                if j is None:
                    emit(fmt_row([val, np_, "BLAD", "", "", "", "", ""], ww))
                    continue
                emit(fmt_row([
                    val, np_,
                    f"{j['compute_ms_med']:.2f}",
                    f"{j['comm_ms_med']:.2f}",
                    f"{j['total_ms_med']:.2f}",
                    f"{j['qps_med']:.0f}",
                    f"{j['p95_us']:.0f}",
                    f"{j['rss_sum_mib']:.0f}",
                ], ww))
        emit("")

    bp = {"N": args.N, "D": args.D, "Q": args.Q, "k": args.k}
    sweep("3) SWEEP N (rozmiar bazy) — P in {1,8}",
          "N", [10000, 100000, 1000000], bp)
    sweep("4) SWEEP D (wymiary) — P in {1,8}",
          "D", [32, 128, 512], bp)
    sweep("5) SWEEP k (liczba sasiadow) — P in {1,8}",
          "k", [1, 10, 100], bp)
    sweep("6) SWEEP Q (wielkosc partii) — P in {1,8}",
          "Q", [20, 100, 500], bp)

    # --- HYBRYDA MPI+OpenMP ---
    emit("-" * 72)
    emit("7) WARIANT HYBRYDOWY MPI+OpenMP (opcjonalny)")
    emit("   Rownoleglenie petli po zapytaniach (#pragma omp parallel for).")
    emit("   Lacznie proc*watki; tu dobrane tak, by proc*watki<=16 (fizyczne).")
    emit(f"   N={args.N} D={args.D} Q={args.Q} k={args.k}")
    emit("-" * 72)
    if os.path.exists(HYBRID):
        hh = ["proc", "watki/proc", "proc*watki", "compute_ms", "total_ms",
              "QPS", "P95_us"]
        ww = [5, 11, 11, 11, 9, 8, 9]
        emit(fmt_row(hh, ww))
        for np_, th in [(1, 16), (2, 8), (4, 4), (8, 2)]:
            j = bench_config(HYBRID, args.N, args.D, args.Q, args.k, np_,
                             base_ds, omp_threads=th)
            if j is None:
                emit(fmt_row([np_, th, np_*th, "BLAD", "", "", ""], ww))
                continue
            emit(fmt_row([
                np_, th, np_*th,
                f"{j['compute_ms_med']:.2f}",
                f"{j['total_ms_med']:.2f}",
                f"{j['qps_med']:.0f}",
                f"{j['p95_us']:.0f}",
            ], ww))
    else:
        emit("   (binarka hybrydowa niezbudowana — pomijam)")
    emit("")
    emit("=" * 72)
    emit("KONIEC. Wyniki surowe powyzej. Wszystko LOKALNE (1 maszyna).")
    emit("=" * 72)

    out_path = os.path.join(HERE, "results_mpi.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[zapisano] {out_path}")


if __name__ == "__main__":
    main()
