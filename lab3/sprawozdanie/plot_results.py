#!/usr/bin/env python3
"""Wykresy do sprawozdania z lab. 3 (konwolucja 2D, OpenMP).
Dane sa wczytywane z results.txt (wyjscie programu lab3 z `make benchmark`)."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "results.txt"
rows = []  # (N, K, threads, {'seq':t, 'omp':(t,s), ...})
cur = None
for line in open(SRC, encoding="utf-8"):
    m = re.match(r"Rozmiar: (\d+)x\d+, J.dro: (\d+)x\d+, W.tki: (\d+)", line)
    if m:
        cur = dict(N=int(m[1]), K=int(m[2]), T=int(m[3]))
        rows.append(cur)
        continue
    m = re.match(r"Seq:\s+([\d.]+) s", line)
    if m and cur is not None:
        cur["seq"] = float(m[1]); continue
    m = re.match(r"(OMP|Collapse|Dynamic):\s+([\d.]+) s\s+speedup: ([\d.]+)x\s+poprawno.{1,3}: (\w+)", line)
    if m and cur is not None:
        cur[m[1]] = (float(m[2]), float(m[3]), m[4])

assert len(rows) == 9 and all(r[v][2] == "OK" for r in rows for v in ("OMP", "Collapse", "Dynamic"))
configs = sorted({(r["N"], r["K"]) for r in rows})
variants = [("OMP", r"\texttt{parallel for}", "o", "#1f4e79", "-"),
            ("Collapse", r"\texttt{collapse(2)}", "s", "#c0392b", "--"),
            ("Dynamic", r"\texttt{schedule(dynamic,\,64)}", "^", "#2e8b57", "-.")]
plt.rcParams.update({"font.size": 8, "text.usetex": False, "axes.grid": True,
                     "grid.alpha": 0.3, "font.family": "serif"})
labels = {"OMP": "parallel for", "Collapse": "collapse(2)", "Dynamic": "schedule(dynamic, 64)"}

# Rys. 1: speedup
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.7), sharey=True)
for ax, (N, K) in zip(axs, configs):
    sel = sorted([r for r in rows if (r["N"], r["K"]) == (N, K)], key=lambda r: r["T"])
    th = [r["T"] for r in sel]
    ax.plot([1, 8], [1, 8], color="gray", ls=":", label="idealne")
    for key, _, mk, col, ls in variants:
        ax.plot(th, [r[key][1] for r in sel], marker=mk, color=col, ls=ls, label=labels[key], ms=4)
    ax.set_xticks(th)
    ax.set_xlabel("liczba wątków")
    ax.set_title(f"{N}×{N}, jądro {K}×{K}", fontsize=8)
axs[0].set_ylabel("przyspieszenie $S$")
axs[0].legend(loc="upper left", fontsize=7)
fig.tight_layout()
fig.savefig("fig_speedup.pdf"); fig.savefig("fig_speedup.png", dpi=300)

# Rys. 2: czasy (2048 i 4096) ; seq jako punkty
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.7))
for ax, (N, K) in zip(axs, configs[1:]):
    sel = sorted([r for r in rows if (r["N"], r["K"]) == (N, K)], key=lambda r: r["T"])
    th = [r["T"] for r in sel]
    ax.plot(th, [r["seq"] for r in sel], marker="x", color="black", ls=":", label="sekwencyjnie", ms=5)
    for key, _, mk, col, ls in variants:
        ax.plot(th, [r[key][0] for r in sel], marker=mk, color=col, ls=ls, label=labels[key], ms=4)
    ax.set_xticks(th); ax.set_ylim(bottom=0)
    ax.set_xlabel("liczba wątków"); ax.set_title(f"{N}×{N}, jądro {K}×{K}", fontsize=8)
    ax.set_ylabel("czas [s]")
axs[0].legend(loc="center right", bbox_to_anchor=(1.0, 0.6), fontsize=7)
fig.tight_layout()
fig.savefig("fig_czas.pdf"); fig.savefig("fig_czas.png", dpi=300)
print("OK", len(rows))
