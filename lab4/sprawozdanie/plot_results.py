#!/usr/bin/env python3
"""Wykres i tabela do sprawozdania z lab. 4 (MPI, Scatter/Gather).
Dane: ../results_raw.csv (N = 10000) i ../results_raw_duze.csv (N = 10^7),
oba z ./run_benchmark.sh. Wynik: fig_czas.pdf/.png oraz tabela.tex."""
import csv
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    """Zwraca {p: [czasy w s]}; przerywa, gdy któraś kontrola sumy nie jest OK."""
    times = {}
    for row in csv.DictReader(open(path, encoding="utf-8")):
        assert row["check"] == "OK", row
        times.setdefault(int(row["p"]), []).append(float(row["time_s"]))
    return times


small = load("../results_raw.csv")
big = load("../results_raw_duze.csv")
ps = sorted(small)
assert all(len(small[p]) >= 10 and len(big[p]) >= 10 for p in ps)

plt.rcParams.update({"font.size": 8, "text.usetex": False, "axes.grid": True,
                     "grid.alpha": 0.3, "font.family": "serif"})

fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
for ax, data, scale, unit, title, mk, col in [
        (axs[0], small, 1e6, "µs", "N = 10 000", "o", "#1f4e79"),
        (axs[1], big, 1e3, "ms", "N = 10 000 000", "s", "#c0392b")]:
    med = [statistics.median(data[p]) * scale for p in ps]
    lo = [m - min(data[p]) * scale for m, p in zip(med, ps)]
    hi = [max(data[p]) * scale - m for m, p in zip(med, ps)]
    ax.errorbar(ps, med, yerr=[lo, hi], marker=mk, color=col, ms=4, capsize=3,
                label="mediana (słupki: min–max)")
    ax.set_xticks(ps)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("liczba procesów $p$")
    ax.set_ylabel(f"czas [{unit}]")
    ax.set_title(title, fontsize=8)
axs[0].legend(loc="upper left", fontsize=7)
fig.tight_layout()
fig.savefig("fig_czas.pdf")
fig.savefig("fig_czas.png", dpi=300)


def fmt(x, digits):
    return f"{x:.{digits}f}".replace(".", ",")


# tabela: mediana, min, max dla obu rozmiarów
lines = []
for p in ps:
    s = [statistics.median(small[p]), min(small[p]), max(small[p])]
    b = [statistics.median(big[p]), min(big[p]), max(big[p])]
    lines.append(f"{p} & " + " & ".join(fmt(x * 1e6, 0) for x in s) + " & "
                 + " & ".join(fmt(x * 1e3, 2) for x in b) + r" \\")
head = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
        r" & \multicolumn{3}{c}{$N=10\,000$ [$\mu$s]} & \multicolumn{3}{c}{$N=10^7$ [ms]} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"$p$ & mediana & min & max & mediana & min & max \\", r"\midrule"]
tail = [r"\bottomrule", r"\end{tabular}"]
with open("tabela.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(head + lines + tail) + "\n")

# liczby używane w tekście sprawozdania (mediany dla p = 2 i p = 8)
macros = {
    "powtorzenia": str(len(small[ps[0]])),
    "malyPdwa": fmt(statistics.median(small[2]) * 1e6, 0),
    "malyPosiem": fmt(statistics.median(small[8]) * 1e6, 0),
    "duzyPdwa": fmt(statistics.median(big[2]) * 1e3, 1),
    "duzyPosiem": fmt(statistics.median(big[8]) * 1e3, 1),
}
with open("liczby.tex", "w", encoding="utf-8") as f:
    for name, val in macros.items():
        f.write(f"\\newcommand{{\\{name}}}{{{val}}}\n")

print("OK", {p: len(small[p]) for p in ps})
