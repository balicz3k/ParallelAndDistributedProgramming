#!/usr/bin/env python3
"""
make_plots.py — wykresy do sprawozdania PRiR D03 (top-k).
Dane pochodzą z plików results*.txt (bloki threads=N / procesów / CUDA).
Zapis: PNG 300 dpi do bieżącego katalogu.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.autolayout": True,
})

# ---- Dane (z results*.txt, N=100k, D=128, Q=20, k=10) ----
# OpenMP: bloki threads=N z results_omp.txt
omp_t   = [1, 2, 4, 8]
omp_qps = [169.2, 315.8, 589.6, 972.1]
omp_spd = [q / omp_qps[0] for q in omp_qps]

# MPI: results_mpi.txt (QPS całkowite oraz czas obliczeń rank0)
mpi_p    = [1, 2, 4]
mpi_qps  = [112.3, 180.8, 249.1]
mpi_comp = [117.504, 59.423, 30.356]           # ms
mpi_spd_qps  = [q / mpi_qps[0] for q in mpi_qps]
mpi_spd_comp = [mpi_comp[0] / c for c in mpi_comp]

# ---- Wykres 1: przyspieszenie vs liczba jednostek ----
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ideal = list(range(1, 9))
ax.plot(ideal, ideal, "--", color="gray", label="idealne (liniowe)")
ax.plot(omp_t, omp_spd, "o-", color="#1f6fb4", label="OpenMP (QPS)")
ax.plot(mpi_p, mpi_spd_qps, "s-", color="#c0504d", label="MPI (QPS)")
ax.plot(mpi_p, mpi_spd_comp, "^--", color="#e08214",
        label="MPI (czas obliczeń rank 0)")
ax.set_xlabel("Liczba wątków / procesów")
ax.set_ylabel("Przyspieszenie względem 1 jednostki")
ax.set_xticks(ideal)
ax.legend()
fig.savefig("speedup.png", dpi=300)
plt.close(fig)

# ---- Wykres 2: QPS wszystkich wersji dla N=100k (skala log) ----
labels = ["Ref\n1T", "OpenMP\n8T", "MPI\n4P", "CUDA\ntotal", "CUDA\nkernel"]
qps    = [169.2, 972.1, 249.1, 1449.1, 3415.2]
colors = ["#7f7f7f", "#1f6fb4", "#c0504d", "#4a9c3f", "#2e7d2a"]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
bars = ax.bar(labels, qps, color=colors)
ax.set_yscale("log")
ax.set_ylabel("Przepustowość [zapytań/s] (skala log)")
for b, v in zip(bars, qps):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.05, f"{v:.0f}",
            ha="center", va="bottom", fontsize=9)
ax.set_ylim(100, 6000)
fig.savefig("qps_compare.png", dpi=300)
plt.close(fig)

# ---- Wykres 3: skalowanie czasu/zapytanie vs N (log-log) ----
# ref: 100k=5.91 ms/q, 1M=59.909 ms/q (z total_ms/Q)
N        = [1e5, 1e6]
ref_q    = [118.194 / 20, 1198.181452 / 20]     # ms/zapytanie
cuda_ker = [5.856 / 20, 53.804 / 20]            # ms/zapytanie (kernel)
cuda_tot = [13.801 / 20, 132.034 / 20]          # ms/zapytanie (z transferem)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.loglog(N, ref_q,    "o-", color="#7f7f7f", label="Referencja (1 wątek CPU)")
ax.loglog(N, cuda_tot, "s-", color="#4a9c3f", label="CUDA (z transferem H$\\to$D)")
ax.loglog(N, cuda_ker, "^-", color="#2e7d2a", label="CUDA (sam kernel)")
ax.set_xlabel("Rozmiar bazy N [wektorów]")
ax.set_ylabel("Czas na zapytanie [ms]")
ax.legend()
fig.savefig("scaling.png", dpi=300)
plt.close(fig)

print("Zapisano: speedup.png, qps_compare.png, scaling.png")
