#!/usr/bin/env python3
"""Wykresy z openmp/results_omp.txt, mpi/results_mpi.txt, cuda/results_cuda.txt.

Parsery dopasowane do faktycznych formatow:
  OMP  - wiersze "CSV,N,D,Q,k,threads,repeat,med_ms,min_ms,max_ms,iqr_ms,p50_us,p95_us,qps,rss_kib"
  MPI  - tabele sekcji 2)-7) (kolumny wg naglowkow w pliku)
  CUDA - bloki "## [CUBLAS ]tag (N=.. D=.. Q=.. k=..)" + linie klucz=wartosc

Uzycie: python3 scripts/plots.py [--root .] [--out plots]
Brakujace pliki/dane => dany wykres jest pomijany z komunikatem.
"""
import argparse, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUM = r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?"
def fnum(s): return float(s.replace(",", "."))

# ---------------------------------------------------------------- OpenMP
def parse_omp(path):
    rows = {}
    for line in open(path, errors="replace"):
        if not line.startswith("CSV,"): continue
        t = line.strip().split(",")[1:]
        try:
            N, D, Q, k, th, rep = (int(x) for x in t[:6])
            med, mn, mx, iqr, p50, p95, qps, rss = (float(x) for x in t[6:14])
        except ValueError:
            continue
        # konfiguracja powtarza sie w wielu sekcjach - ostatni wpis wygrywa
        rows[(N, D, Q, k, th)] = dict(N=N, D=D, Q=Q, k=k, p=th, rep=rep, ms=med, ms_min=mn,
                                      ms_max=mx, p95=p95, qps=qps, rss=rss / 1024)
    return list(rows.values())

# ---------------------------------------------------------------- MPI
def parse_mpi(path):
    """-> dict sekcja -> lista rekordow. Sekcje: 'scal','N','D','k','Q','hyb'."""
    out = {"scal": [], "N": [], "D": [], "k": [], "Q": [], "hyb": []}
    sec = None
    names = {"2": "scal", "3": "N", "4": "D", "5": "k", "6": "Q", "7": "hyb"}
    for line in open(path, errors="replace"):
        m = re.match(r"^(\d)\) ", line)
        if m: sec = names.get(m.group(1)); continue
        if sec is None: continue
        t = line.split()
        if not t or not re.fullmatch(NUM, t[0]): continue
        try:
            if sec == "scal":   # P comp comm total min-max QPS P95 RSSmax RSSsum S E [*oversub]
                v = [fnum(x) for x in t[:4]] + [fnum(x) for x in t[5:11]]
                out[sec].append(dict(p=int(v[0]), comp=v[1], comm=v[2], ms=v[3], qps=v[4], p95=v[5],
                                     rss_max=v[6], rss_sum=v[7], S=v[8], E=v[9],
                                     over="oversub" in line))
            elif sec == "hyb":  # proc watki proc*watki comp total qps p95
                v = [fnum(x) for x in t[:7]]
                out[sec].append(dict(p=int(v[0]), th=int(v[1]), comp=v[3], ms=v[4], qps=v[5], p95=v[6]))
            else:               # x P comp comm total qps p95 rss_sum
                v = [fnum(x) for x in t[:8]]
                out[sec].append(dict(x=v[0], p=int(v[1]), comp=v[2], comm=v[3], ms=v[4],
                                     qps=v[5], p95=v[6], rss_sum=v[7]))
        except (ValueError, IndexError):
            continue
    return out

# ---------------------------------------------------------------- CUDA
def parse_cuda(path):
    recs, cur = [], None
    for line in open(path, errors="replace"):
        m = re.match(r"^##\s+(.*?)\s+\((N=.*)\)\s*$", line)
        if m:
            cur = {"tag": m.group(1), "impl": "cublas" if "CUBLAS" in m.group(1) else "cuda"}
            for k, v in re.findall(r"(\w+)=(" + NUM + ")", m.group(2)): cur[k] = fnum(v)
            recs.append(cur); continue
        if cur is None or line.startswith("#"): continue
        for k, v in re.findall(r"(\w+)=(" + NUM + ")", line): cur[k] = fnum(v)
    return recs

# ---------------------------------------------------------------- wykresy
def save(fig, out, name):
    fig.tight_layout(); fig.savefig(os.path.join(out, name), dpi=150); plt.close(fig)
    print("[ok]", os.path.join(out, name))

def scaling_plot(out, name, ps, S, E, title, xlabel, marks=()):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(ps, S, "o-", label="zmierzone"); axs[0].plot(ps, ps, "k--", lw=.8, label="idealne")
    axs[0].set_ylabel("S(p) = T(1)/T(p)")
    axs[1].plot(ps, E, "o-", label="zmierzone"); axs[1].axhline(1, color="k", ls="--", lw=.8, label="idealne")
    axs[1].set_ylabel("E(p) = S(p)/p"); axs[1].set_ylim(0, 1.15)
    for ax in axs:
        ax.set_xlabel(xlabel); ax.set_xticks(ps); ax.grid(alpha=.3); ax.legend()
        for p in marks: ax.axvline(p, color="r", ls=":", lw=.8)
    fig.suptitle(title); save(fig, out, name)

def plot_omp(omp, out):
    r = sorted((x for x in omp if (x["N"], x["D"], x["Q"], x["k"]) == (100000, 128, 250, 10)), key=lambda x: x["p"])
    if len(r) < 2 or r[0]["p"] != 1: print("[skip] OMP S/E: brak danych"); return
    ps = [x["p"] for x in r]; S = [r[0]["ms"] / x["ms"] for x in r]
    scaling_plot(out, "omp_speedup_efficiency.png", ps, S, [s / p for s, p in zip(S, ps)],
                 "OpenMP: N=100k D=128 Q=250 k=10 (CPU 8C/16T; p=16 = SMT)", "watki", marks=[8])

def plot_mpi(mpi, out):
    r = sorted(mpi["scal"], key=lambda x: x["p"])
    if len(r) < 2 or r[0]["p"] != 1: print("[skip] MPI S/E: brak danych"); return
    ps = [x["p"] for x in r]; S = [r[0]["ms"] / x["ms"] for x in r]
    scaling_plot(out, "mpi_speedup_efficiency.png", ps, S, [s / p for s, p in zip(S, ps)],
                 "MPI (LOKALNIE, 1 maszyna): N=100k D=128 Q=100 k=10; p=16 oversubscribe",
                 "procesy MPI", marks=[16])
    # czas: compute vs komunikacja + hybryda
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].bar([str(p) for p in ps], [x["comp"] for x in r], label="obliczenia (max po rankach)")
    axs[0].bar([str(p) for p in ps], [x["comm"] for x in r], bottom=[x["comp"] for x in r], label="komunikacja")
    axs[0].set_xlabel("procesy MPI"); axs[0].set_ylabel("czas [ms]"); axs[0].legend(); axs[0].grid(alpha=.3, axis="y")
    axs[1].plot(ps, [x["p95"] for x in r], "o-", label="czysty MPI")
    if mpi["hyb"]:
        h = sorted(mpi["hyb"], key=lambda x: x["p"])
        axs[1].plot([x["p"] for x in h], [x["p95"] for x in h], "s--", label="MPI+OpenMP (proc*watki=16)")
    axs[1].set_xlabel("procesy MPI"); axs[1].set_ylabel("P95 latencji zapytania [us]")
    axs[1].set_yscale("log"); axs[1].legend(); axs[1].grid(alpha=.3, which="both")
    fig.suptitle("MPI: podzial czasu i P95"); save(fig, out, "mpi_time_p95.png")

def plot_vs_N(omp, mpi, cuda, out):
    series = {}   # nazwa -> {N: (qps, p95, rss)}
    for th in (8, 16):
        d = {x["N"]: x for x in omp if (x["D"], x["Q"], x["k"], x["p"]) == (128, 250, 10, th)}
        if len(d) > 1:
            series[f"OpenMP p={th} (Q=250)"] = {n: (x["qps"], x["p95"], x["rss"]) for n, x in d.items()}
    for P in (1, 8):
        d = {int(x["x"]): x for x in mpi["N"] if x["p"] == P}
        if len(d) > 1:
            series[f"MPI P={P} (Q=100)"] = {n: (x["qps"], x["p95"], x["rss_sum"]) for n, x in d.items()}
    d = {int(x["N"]): x for x in cuda if x["impl"] == "cuda" and re.fullmatch(r"N=\d+", x["tag"])}
    if len(d) > 1:
        series["CUDA wlasny (Q=20)"] = {n: (x["qps_total_median"], x["p95"], x["ram_peak_rss_mb"])
                                        for n, x in d.items()}
    d = {int(x["N"]): x for x in cuda if x["impl"] == "cublas" and re.fullmatch(r"CUBLAS N=\d+", x["tag"])}
    if len(d) > 1:
        series["CUDA cuBLAS (Q=20, kernel)"] = {n: (x["qps_kernel_median"], None, x["ram_peak_rss_mb"]) for n, x in d.items()}
    if not series: print("[skip] vs N: brak danych"); return
    for idx, fn, yl in ((0, "qps_vs_N.png", "QPS [zapytania/s]"), (1, "p95_vs_N.png", "P95 latencji zapytania [us]")):
        fig, ax = plt.subplots(figsize=(7, 4.5)); n = 0
        for lab, s in series.items():
            xs = [x for x in sorted(s) if s[x][idx] is not None]
            if len(xs) > 1: ax.plot(xs, [s[x][idx] for x in xs], "o-", label=lab); n += 1
        if n:
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("N (rozmiar bazy)"); ax.set_ylabel(yl)
            ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
            save(fig, out, fn)
        else: plt.close(fig)
    # pamiec: RSS hosta + VRAM
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for lab, s in series.items():
        xs = sorted(s); axs[0].plot(xs, [s[x][2] for x in xs], "o-", label=lab)
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("N"); axs[0].set_ylabel("peak RSS [MiB] (MPI: suma rankow)")
    axs[0].grid(alpha=.3, which="both"); axs[0].legend(fontsize=7)
    for impl, lab in (("cuda", "wlasny kernel"), ("cublas", "cuBLAS")):
        d = sorted((int(x["N"]), x["vram_alloc_mb"]) for x in cuda if x["impl"] == impl and "vram_alloc_mb" in x
                   and re.fullmatch(r"(CUBLAS )?N=\d+", x["tag"]))
        if len(d) > 1: axs[1].plot(*zip(*d), "o-", label=lab)
    axs[1].set_xscale("log"); axs[1].set_yscale("log"); axs[1].set_xlabel("N"); axs[1].set_ylabel("VRAM zaalokowany [MiB]")
    axs[1].grid(alpha=.3, which="both"); axs[1].legend()
    fig.suptitle("Pamiec szczytowa"); save(fig, out, "memory_vs_N.png")

def plot_cuda_cmp(cuda, out):
    own = {int(x["N"]): x for x in cuda if x["impl"] == "cuda" and re.fullmatch(r"N=\d+", x["tag"])}
    cub = {int(x["N"]): x for x in cuda if x["impl"] == "cublas" and re.fullmatch(r"CUBLAS N=\d+", x["tag"])}
    ns = sorted(set(own) & set(cub))
    if not ns: print("[skip] CUDA vs cuBLAS: brak danych"); return
    fig, axs = plt.subplots(1, 2, figsize=(10, 4)); w = .38; xs = range(len(ns))
    for ax, key, yl in ((axs[0], "kernel_ms_median", "czas kernela [ms] (mediana)"),
                        (axs[1], "qps_kernel_median", "QPS (kernel)")):
        for off, dd, lab in ((-w / 2, own, "wlasny kernel"), (w / 2, cub, "cuBLAS (SGEMM)")):
            lo = [dd[n][key] - dd[n][key.replace("median", "min")] for n in ns]
            hi = [dd[n][key.replace("median", "max")] - dd[n][key] for n in ns]
            ax.bar([i + off for i in xs], [dd[n][key] for n in ns], w, label=lab, yerr=[lo, hi], capsize=3)
        ax.set_xticks(list(xs)); ax.set_xticklabels([str(n) for n in ns]); ax.set_xlabel("N"); ax.set_ylabel(yl)
        ax.set_yscale("log"); ax.grid(alpha=.3, axis="y", which="both"); ax.legend()
    fig.suptitle("CUDA: wlasny kernel vs cuBLAS (D=128 Q=20 k=10)"); save(fig, out, "cuda_own_vs_cublas.png")

def plot_sweep_k(omp, mpi, cuda, out):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4)); n = 0
    def add(d, lab):
        nonlocal n
        if len(d) > 1:
            for ax, i in zip(axs, (1, 2)): ax.plot([r[0] for r in d], [r[i] for r in d], "o-", label=lab)
            n += 1
    add(sorted((x["k"], x["qps"], x["p95"]) for x in omp
               if (x["N"], x["D"], x["Q"], x["p"]) == (100000, 128, 250, 16)), "OpenMP p=16")
    for P in (1, 8):
        add(sorted((x["x"], x["qps"], x["p95"]) for x in mpi["k"] if x["p"] == P), f"MPI P={P}")
    add(sorted((x["k"], x["qps_total_median"], x["p95"]) for x in cuda
               if x["impl"] == "cuda" and x["tag"].startswith("k=")), "CUDA wlasny")
    if not n: plt.close(fig); print("[skip] sweep k"); return
    for ax, yl in zip(axs, ("QPS", "P95 [us]")):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("k"); ax.set_ylabel(yl)
        ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    fig.suptitle("Wplyw k (N=100k D=128)"); save(fig, out, "sweep_k.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "plots"); os.makedirs(out, exist_ok=True)
    def find(sub, name):
        p = os.path.join(a.root, sub, name)
        return p if os.path.isfile(p) else None
    po, pm, pc = find("openmp", "results_omp.txt"), find("mpi", "results_mpi.txt"), find("cuda", "results_cuda.txt")
    omp = parse_omp(po) if po else []
    mpi = parse_mpi(pm) if pm else {k: [] for k in ("scal", "N", "D", "k", "Q", "hyb")}
    cuda = parse_cuda(pc) if pc else []
    print(f"[info] omp={len(omp)} rek., mpi scal={len(mpi['scal'])} N={len(mpi['N'])}, cuda={len(cuda)} blokow")
    plot_omp(omp, out); plot_mpi(mpi, out); plot_vs_N(omp, mpi, cuda, out)
    plot_cuda_cmp(cuda, out); plot_sweep_k(omp, mpi, cuda, out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
