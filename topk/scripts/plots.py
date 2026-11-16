#!/usr/bin/env python3
"""Wykresy i tabele do sprawozdania D03 - wyłącznie z results/raw.csv.

Wyniki: sprawozdanie/fig/*.pdf (+ .png), sprawozdanie/tables/*.tex
oraz sprawozdanie/tables/liczby.tex (makra z liczbami używanymi w tekście).
Każda wartość to mediana z powtórzeń danej konfiguracji; rozrzut to IQR.
Użycie: python3 scripts/plots.py [plik.csv]   (domyślnie results/raw.csv)
"""
import csv
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "raw.csv")
FIG = os.path.join(ROOT, "sprawozdanie", "fig")
TAB = os.path.join(ROOT, "sprawozdanie", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)


def read_conf():
    conf = {}
    for line in open(os.path.join(ROOT, "experiments.conf")):
        line = line.split("#")[0].strip()
        if "=" in line:
            key, val = line.split("=", 1)
            conf[key] = val.strip('"').split()
    return conf


CONF = read_conf()
MAIN = dict(N=int(CONF["MAIN_N"][0]), D=int(CONF["MAIN_D"][0]),
            Q=int(CONF["MAIN_Q"][0]), k=int(CONF["MAIN_K"][0]))
OMP_SWEEP = int(CONF["SWEEP_OMP_THREADS"][0])
MPI_SWEEP = int(CONF["SWEEP_MPI_PROCS"][0])

# ---------------------------------------------------------------- dane
KEY = ("version", "N", "D", "Q", "k", "threads", "procs", "block")
groups = {}
for row in csv.DictReader(open(SRC)):
    key = tuple(row[c] if c == "version" else int(row[c]) for c in KEY)
    groups.setdefault(key, []).append(row)


def q1q3(values):
    qs = statistics.quantiles(values, n=4) if len(values) > 1 else [values[0]] * 3
    return qs[0], qs[2]


def stats(rows):
    """Mediany i rozrzut jednej konfiguracji."""
    col = lambda c: [float(r[c]) for r in rows]
    full = col("full_ms")
    lo, hi = q1q3(full)
    Q = int(rows[0]["Q"])
    return dict(full=statistics.median(full), iqr=hi - lo, core=statistics.median(col("core_ms")),
                base=statistics.median(col("base_ms")), qps=Q / statistics.median(full) * 1000,
                p95=statistics.median(col("p95_us")), p50=statistics.median(col("p50_us")),
                rss=max(col("rss_mb")), vram=max(col("vram_mb")), n=len(rows))


def find(version, threads=None, procs=None, block=None, **params):
    """Statystyki konfiguracji; parametry N, D, Q, k domyślnie z przypadku głównego."""
    p = dict(MAIN, **params)
    for key, rows in groups.items():
        v, N, D, Q, k, t, pr, b = key
        if (v, N, D, Q, k) != (version, p["N"], p["D"], p["Q"], p["k"]):
            continue
        if threads is not None and t != threads:
            continue
        if procs is not None and pr != procs:
            continue
        if block is not None and b != block:
            continue
        return stats(rows)
    return None


# wersje porównywane w badaniach N, D, k, Q
SWEEP = [("ref", "referencja", dict(), "black", "x", ":"),
         ("omp", f"OpenMP, {OMP_SWEEP} wątków", dict(threads=OMP_SWEEP), "#1f4e79", "o", "-"),
         ("mpi", f"MPI, {MPI_SWEEP} procesów", dict(procs=MPI_SWEEP), "#c0392b", "s", "--"),
         ("cuda", "CUDA (własny kernel)", dict(block=256), "#2e8b57", "^", "-."),
         ("cublas", "cuBLAS", dict(), "#8e44ad", "v", "-")]

plt.rcParams.update({"font.size": 8, "text.usetex": False, "axes.grid": True,
                     "grid.alpha": 0.3, "font.family": "serif"})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".pdf"), metadata={"CreationDate": None})
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)


def fmt(x, digits=1):
    """Liczba z przecinkiem dziesiętnym; separator tysięcy jako cienka spacja."""
    s = f"{x:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", "\\,")
    return s


def write_table(name, header, rows, align):
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, name + ".tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


macros = {}

# ---------------------------------------------------------------- przypadek główny
ref = find("ref")
main_rows = [("ref", "referencja", "1 wątek", dict()),
             ("omp", "OpenMP", f"{OMP_SWEEP} wątków", dict(threads=OMP_SWEEP)),
             ("mpi", "MPI", f"{MPI_SWEEP} procesów", dict(procs=MPI_SWEEP)),
             ("cuda", "CUDA", "blok 256", dict(block=256)),
             ("cuda_bez_strumieni", "CUDA bez strumieni", "blok 256", dict()),
             ("cublas", "cuBLAS", "SGEMM + selekcja", dict())]
rows = []
for v, label, cfg, sel in main_rows:
    s = find(v, **sel)
    if s is None:
        continue
    rows.append([label, cfg, f"{fmt(s['full'], 1)} [{fmt(s['iqr'], 1)}]", fmt(s["core"], 1),
                 fmt(s["qps"], 0), fmt(s["p95"], 0), fmt(s["rss"], 0),
                 fmt(s["vram"], 0) if s["vram"] else "--", fmt(ref["full"] / s["full"], 1)])
write_table("tab_glowny", ["Wersja", "Konfiguracja", r"$T_\text{pełny}$ [ms] [IQR]",
                           r"$T_\text{rdzeń}$ [ms]", "QPS", r"P95 [\textmu s]", "RAM [MiB]",
                           "VRAM [MiB]", r"$T_\text{ref}/T$"], rows, "llrrrrrrr")
macros["refFull"] = fmt(ref["full"], 0)
macros["refQps"] = fmt(ref["qps"], 0)
macros["refPdziewiecdziesiatpiec"] = fmt(ref["p95"], 0)

# ---------------------------------------------------------------- OpenMP S(p), E(p)
omp_t = [int(t) for t in CONF["OMP_THREADS"]]
omp = {t: find("omp", threads=t) for t in omp_t}
T1 = omp[1]["full"]
rows = []
for t in omp_t:
    s = omp[t]
    rows.append([str(t), f"{fmt(s['full'], 1)} [{fmt(s['iqr'], 1)}]", fmt(T1 / s["full"], 2),
                 fmt(T1 / s["full"] / t, 2), fmt(ref["full"] / s["full"], 2), fmt(s["p95"], 0)])
write_table("tab_omp", ["Wątki $p$", r"$T(p)$ [ms] [IQR]", "$S(p)$", "$E(p)$",
                        r"$T_\text{ref}/T(p)$", r"P95 [\textmu s]"], rows, "rrrrrr")
macros["ompSmax"] = fmt(T1 / omp[omp_t[-1]]["full"], 1)
macros["ompEmax"] = fmt(T1 / omp[omp_t[-1]]["full"] / omp_t[-1], 2)
macros["ompSosiem"] = fmt(T1 / omp[8]["full"], 1) if 8 in omp else "--"
macros["ompJedenDoRef"] = fmt(ref["full"] / T1, 2)

fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
axs[0].plot(omp_t, omp_t, color="gray", ls=":", label="idealne")
axs[0].plot(omp_t, [T1 / omp[t]["full"] for t in omp_t], marker="o", color="#1f4e79",
            label="względem OpenMP, 1 wątek")
axs[0].plot(omp_t, [ref["full"] / omp[t]["full"] for t in omp_t], marker="x", ls="--",
            color="black", label="względem referencji")
axs[0].set_ylabel("przyspieszenie $S(p)$")
axs[0].legend(fontsize=7)
axs[1].plot(omp_t, [T1 / omp[t]["full"] / t for t in omp_t], marker="o", color="#1f4e79")
axs[1].axhline(1.0, color="gray", ls=":")
axs[1].set_ylabel("efektywność $E(p)$")
axs[1].set_ylim(0, 1.15)
for ax in axs:
    ax.set_xlabel("liczba wątków $p$")
    ax.set_xticks(omp_t)
save(fig, "fig_omp")

# ---------------------------------------------------------------- MPI S(p), E(p)
mpi_p = [int(p) for p in CONF["MPI_PROCS"]]
mpi = {p: find("mpi", procs=p) for p in mpi_p}
F1, C1 = mpi[1]["full"], mpi[1]["core"]
rows = []
for p in mpi_p:
    s = mpi[p]
    rows.append([f"{p}", "1", fmt(s["core"], 1), fmt(s["full"] - s["core"], 1),
                 f"{fmt(s['full'], 1)} [{fmt(s['iqr'], 1)}]", fmt(F1 / s["full"], 2),
                 fmt(F1 / s["full"] / p, 2), fmt(s["base"], 1), fmt(s["rss"], 0)])
for h in CONF["HYBRID"]:
    p, t = (int(x) for x in h.split("x"))
    s = find("mpi_omp", threads=t, procs=p)
    if s:
        rows.append([f"{p}", f"{t}", fmt(s["core"], 1), fmt(s["full"] - s["core"], 1),
                     f"{fmt(s['full'], 1)} [{fmt(s['iqr'], 1)}]", fmt(F1 / s["full"], 2),
                     fmt(F1 / s["full"] / (p * t), 2), fmt(s["base"], 1), fmt(s["rss"], 0)])
write_table("tab_mpi", ["Procesy", "Wątki", r"$T_\text{rdzeń}$ [ms]", "Komunikacja [ms]",
                        r"$T_\text{pełny}$ [ms] [IQR]", "$S(p)$", "$E(p)$",
                        "Scatterv bazy [ms]", "RAM [MiB]"], rows, "rrrrrrrrr")
pmax = mpi_p[-1]
macros["mpiSmax"] = fmt(F1 / mpi[pmax]["full"], 1)
macros["mpiEmax"] = fmt(F1 / mpi[pmax]["full"] / pmax, 2)
macros["mpiSrdzenmax"] = fmt(C1 / mpi[pmax]["core"], 1)
macros["mpiKomunikacjamax"] = fmt(mpi[pmax]["full"] - mpi[pmax]["core"], 1)
macros["mpiBazamax"] = fmt(mpi[pmax]["base"], 1)

fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
axs[0].plot(mpi_p, mpi_p, color="gray", ls=":", label="idealne")
axs[0].plot(mpi_p, [F1 / mpi[p]["full"] for p in mpi_p], marker="s", color="#c0392b",
            label="czas pełny (z komunikacją)")
axs[0].plot(mpi_p, [C1 / mpi[p]["core"] for p in mpi_p], marker="o", ls="--", color="#1f4e79",
            label="czas rdzenia")
axs[0].set_ylabel("przyspieszenie $S(p)$")
axs[0].legend(fontsize=7)
axs[1].plot(mpi_p, [F1 / mpi[p]["full"] / p for p in mpi_p], marker="s", color="#c0392b")
axs[1].plot(mpi_p, [C1 / mpi[p]["core"] / p for p in mpi_p], marker="o", ls="--", color="#1f4e79")
axs[1].axhline(1.0, color="gray", ls=":")
axs[1].set_ylabel("efektywność $E(p)$")
axs[1].set_ylim(0, 1.15)
for ax in axs:
    ax.set_xlabel("liczba procesów $p$ (pomiar lokalny, 1 komputer)")
    ax.set_xticks(mpi_p)
save(fig, "fig_mpi")

# ---------------------------------------------------------------- CUDA: bloki i porównanie z cuBLAS
blocks = [int(b) for b in CONF["CUDA_BLOCKS"]]
cb = {b: find("cuda", block=b) for b in blocks}
rows = [[str(b), fmt(cb[b]["core"], 2), fmt(cb[b]["full"], 2), fmt(cb[b]["qps"], 0), fmt(cb[b]["p95"], 0)]
        for b in blocks]
write_table("tab_cuda_bloki", ["Wątki w bloku", r"$T_\text{rdzeń}$ [ms]", r"$T_\text{pełny}$ [ms]",
                               "QPS", r"P95 [\textmu s]"], rows, "rrrrr")
cu, cn, cl = find("cuda", block=256), find("cuda_bez_strumieni"), find("cublas")
macros["cudaFull"] = fmt(cu["full"], 2)
macros["cudaCore"] = fmt(cu["core"], 2)
macros["cudaBase"] = fmt(cu["base"], 1)
macros["cudaNoStreamFull"] = fmt(cn["full"], 2)
macros["cudaDoRef"] = fmt(ref["full"] / cu["full"], 0)
macros["cublasFull"] = fmt(cl["full"], 2)
macros["cublasCore"] = fmt(cl["core"], 2)
macros["cublasBase"] = fmt(cl["base"], 1)
macros["cublasDoCuda"] = fmt(cu["full"] / cl["full"], 1)
macros["cudaVram"] = fmt(cu["vram"], 0)
macros["cublasVram"] = fmt(cl["vram"], 0)
best = min(blocks, key=lambda b: cb[b]["core"])
macros["cudaNajlepszyBlok"] = str(best)

fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
axs[0].plot(blocks, [cb[b]["core"] for b in blocks], marker="o", color="#2e8b57", label="rdzeń (kernel)")
axs[0].plot(blocks, [cb[b]["full"] for b in blocks], marker="s", ls="--", color="#1f4e79",
            label="pełny (z kopiami)")
axs[0].set_xscale("log", base=2)
axs[0].set_xticks(blocks)
axs[0].set_xticklabels([str(b) for b in blocks])
axs[0].set_xlabel("wątki w bloku")
axs[0].set_ylabel("czas partii [ms]")
axs[0].set_ylim(bottom=0)
axs[0].legend(fontsize=7)
names = ["CUDA", "CUDA\nbez strumieni", "cuBLAS"]
data = [cu, cn, cl]
x = range(len(names))
w = 0.27
axs[1].bar([i - w for i in x], [s["core"] for s in data], w, color="#2e8b57", label="rdzeń")
axs[1].bar(list(x), [s["full"] for s in data], w, color="#1f4e79", label="pełny")
axs[1].bar([i + w for i in x], [s["base"] for s in data], w, color="#bbbbbb", label="baza na GPU (raz)")
axs[1].set_xticks(list(x))
axs[1].set_xticklabels(names)
axs[1].set_ylabel("czas [ms]")
axs[1].set_ylim(0, 1.35 * max(s["full"] for s in data))
axs[1].legend(fontsize=7, ncol=3, loc="upper center")
save(fig, "fig_cuda")


# ---------------------------------------------------------------- badania N, D, k, Q
def find_sweep(v, sel, param, val):
    """Jak find, ale dla CUDA przy k=64 blok jest mniejszy niż 256 (limit shared memory)."""
    s = find(v, **sel, **{param: val})
    if s is None and "block" in sel:
        s = find(v, **{param: val})
    return s


def sweep_plot(param, values, name, xlabel, logx, logy, metrics, versions=SWEEP):
    fig, axs = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), 2.6))
    if len(metrics) == 1:
        axs = [axs]
    for ax, (metric, ylabel) in zip(axs, metrics):
        for v, label, sel, col, mk, ls in versions:
            xs, ys = [], []
            for val in values:
                s = find_sweep(v, sel, param, val)
                if s is not None and (metric != "vram" or s["vram"] > 0):
                    xs.append(val)
                    ys.append(s[metric])
            if xs:
                ax.plot(xs, ys, marker=mk, color=col, ls=ls, label=label, ms=4)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    axs[0].legend(fontsize=6.5)
    save(fig, name)


Ns = [int(n) for n in CONF["SWEEP_N"]]
Ds = [int(d) for d in CONF["SWEEP_D"]]
Ks = [int(k) for k in CONF["SWEEP_K"]]
Qs = [int(q) for q in CONF["SWEEP_Q"]]
sweep_plot("N", Ns, "fig_n", "rozmiar bazy $N$", True, True,
           [("qps", "zapytania/s (QPS)"), ("p95", r"P95 pojedynczego zapytania [µs]")])
sweep_plot("N", Ns, "fig_pamiec", "rozmiar bazy $N$", True, True,
           [("rss", "szczytowy RAM [MiB]"), ("vram", "pamięć GPU [MiB]")])
sweep_plot("D", Ds, "fig_d", "wymiar $D$", True, True, [("qps", "zapytania/s (QPS)")])
sweep_plot("k", Ks, "fig_k", "liczba sąsiadów $k$", True, True, [("qps", "zapytania/s (QPS)")])
sweep_plot("Q", Qs, "fig_q", "wielkość partii $Q$", True, True,
           [("qps", "zapytania/s (QPS)")], versions=SWEEP[1:])


def sweep_table(param, values, name, header, versions=SWEEP):
    rows = []
    for val in values:
        row = [fmt(val, 0)]
        for v, _, sel, *_ in versions:
            s = find_sweep(v, sel, param, val)
            row.append(fmt(s["qps"], 0) if s else "--")
        rows.append(row)
    write_table(name, [header] + [lab for _, lab, *_ in versions], rows, "r" * (len(versions) + 1))


sweep_table("N", Ns, "tab_n", "$N$")
sweep_table("D", Ds, "tab_d", "$D$")
sweep_table("k", Ks, "tab_k", "$k$")
sweep_table("Q", Qs, "tab_q", "$Q$", versions=SWEEP[1:])

# CUDA dla k=64: liczba wątków w bloku zmniejszona automatycznie
for key in groups:
    if key[0] == "cuda" and key[4] == max(Ks) and key[1:4] == (MAIN["N"], MAIN["D"], MAIN["Q"]):
        macros["cudaBlokKmax"] = str(key[7])

# CUDA: ile danych bazy kernel czyta na sekundę (każdy blok czyta całą bazę: Q*N*D*4 bajty)
for n, nm in [(MAIN["N"], "Glowny"), (max(Ns), "Nmax")]:
    s_ = find_sweep("cuda", dict(block=256), "N", n)
    if s_:
        macros["cudaGBs" + nm] = fmt(MAIN["Q"] * n * MAIN["D"] * 4 / (s_["core"] / 1000) / 1e9, 0)
        macros["cudaQps" + nm] = fmt(s_["qps"], 0)
# CPU przy największym N: ile danych bazy na sekundę przechodzi przez procesor
for v, nm, sel in [("omp", "omp", dict(threads=OMP_SWEEP)), ("mpi", "mpi", dict(procs=MPI_SWEEP))]:
    s_ = find(v, **sel, N=max(Ns))
    macros[nm + "GBsNmax"] = fmt(MAIN["Q"] * max(Ns) * MAIN["D"] * 4 / (s_["core"] / 1000) / 1e9, 0)

# P95 w przypadku głównym i wpływ k oraz D (wartości skrajne)
for v, nm, sel in [("ref", "Ref", dict()), ("omp", "Omp", dict(threads=OMP_SWEEP)),
                   ("mpi", "Mpi", dict(procs=MPI_SWEEP)), ("cuda", "Cuda", dict(block=256)),
                   ("cublas", "Cublas", dict())]:
    macros["p" + nm] = fmt(find(v, **sel)["p95"], 0)
    for param, values, tag in [("k", Ks, "K"), ("D", Ds, "D")]:
        lo, hi = find_sweep(v, sel, param, min(values)), find_sweep(v, sel, param, max(values))
        if lo and hi:
            macros[nm.lower() + "Qps" + tag + "min"] = fmt(lo["qps"], 0)
            macros[nm.lower() + "Qps" + tag + "max"] = fmt(hi["qps"], 0)
hyb = []
for h in CONF["HYBRID"]:
    p_, t_ = (int(x) for x in h.split("x"))
    s_ = find("mpi_omp", threads=t_, procs=p_)
    if s_:
        hyb.append((F1 / s_["full"], h))
if hyb:
    macros["hybrydaSmax"] = fmt(max(hyb)[0], 1)
    macros["hybrydaNajlepsza"] = max(hyb)[1].replace("x", r"\times")
macros["ompQpsQjeden"] = fmt(find("omp", threads=OMP_SWEEP, Q=1)["qps"], 0)

# Q: amortyzacja kopii i startu kernela na GPU
for v, nm in [("cuda", "cuda"), ("cublas", "cublas")]:
    sel = dict(block=256) if v == "cuda" else dict()
    lo, hi = find(v, Q=min(Qs), **sel), find(v, Q=max(Qs), **sel)
    if lo and hi:
        macros[nm + "QpsQmin"] = fmt(lo["qps"], 0)
        macros[nm + "QpsQmax"] = fmt(hi["qps"], 0)

# ---------------------------------------------------------------- zgodność rankingów
rank_path = os.path.join(os.path.dirname(SRC), "ranking_check.csv")
if os.path.isfile(rank_path):
    labels = {"omp": f"OpenMP, {OMP_SWEEP} wątków", "mpi": f"MPI, {MPI_SWEEP} procesów",
              "mpi_omp": "MPI+OpenMP, 4×4", "cuda": "CUDA", "cublas": "cuBLAS"}
    rows = []
    for r in csv.DictReader(open(rank_path)):
        diff = float(r["max_dist_diff"])
        diff_s = "0" if diff == 0 else f"{diff:.0e}".replace("e-0", r"\cdot 10^{-") + "}"
        rows.append([labels.get(r["version"], r["version"]), f"{r['identical_queries']} / {r['queries']}",
                     r["different_positions"], f"${diff_s}$"])
    write_table("tab_ranking", ["Wersja", "Zapytania identyczne z ref.", "Różne pozycje",
                                "Maks. różnica odległości"], rows, "lrrr")

# ---------------------------------------------------------------- środowisko (results/env.txt)
env_path = os.path.join(os.path.dirname(SRC), "env.txt")
if os.path.isfile(env_path):
    esc = lambda t: (t.replace("\\", "/").replace("_", r"\_").replace("%", r"\%")
                     .replace("&", r"\&").replace("#", r"\#"))
    skip = ("commit", "części")
    lines = [r"\begin{tabular}{@{}lp{10.5cm}@{}}", r"\toprule"]
    for line in open(env_path):
        key, _, val = line.strip().partition(": ")
        if key and key not in skip:
            lines.append(f"{esc(key)} & {esc(val)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "env.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

with open(os.path.join(TAB, "liczby.tex"), "w") as f:
    f.write("% wygenerowane przez scripts/plots.py z results/raw.csv\n")
    for name, val in macros.items():
        f.write(f"\\newcommand{{\\{name}}}{{{val}}}\n")

print(f"Wczytano {sum(len(r) for r in groups.values())} wierszy, {len(groups)} konfiguracji.")
print(f"Wykresy: {FIG}, tabele: {TAB}")
