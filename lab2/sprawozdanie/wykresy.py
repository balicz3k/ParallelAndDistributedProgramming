#!/usr/bin/env python3
# Wykresy do sprawozdania PRiR Lab2 na podstawie results_raw.csv.
# Generuje: speedup.pdf, synchronizacja.pdf, schedule.pdf
import csv, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "results_raw.csv"

rows = list(csv.DictReader(open(CSV)))
med = {}
for r in rows:
    key = (r["kind"], r["method"], r["sched"], r["chunk"], int(r["threads"]))
    med.setdefault(key, []).append(float(r["time_s"]))
med = {k: st.median(v) for k, v in med.items()}

T_seq = med[("seq", "seq", "-", "0", 1)]
threads = [1, 2, 4, 8, 16]

def times(method):
    return [med[("par", method, "static", "0", t)] for t in threads]

t_crit = times("critical")
t_atom = times("atomic")
t_priv = times("private")

# --- Wykres 1: speedup (vs sekwencyjna) ---
plt.figure(figsize=(6, 4))
for lab, t in [("critical", t_crit), ("atomic", t_atom), ("prywatne histogramy", t_priv)]:
    sp = [T_seq / x for x in t]
    plt.plot(threads, sp, "o-", label=lab)
plt.plot(threads, threads, "k--", lw=1, label="ideal $S_p=p$")
plt.xlabel("liczba watkow $p$")
plt.ylabel("przyspieszenie $S_p = T_{seq}/T_p$")
plt.title("Przyspieszenie wg sposobu synchronizacji histogramu")
plt.xticks(threads)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("speedup.pdf")
plt.savefig("speedup.png", dpi=130)
plt.close()

# --- Wykres 2: czas bezwzgledny (skala log) ---
plt.figure(figsize=(6, 4))
for lab, t in [("critical", t_crit), ("atomic", t_atom), ("prywatne histogramy", t_priv)]:
    plt.plot(threads, t, "o-", label=lab)
plt.axhline(T_seq, color="gray", ls=":", label="sekwencyjna")
plt.yscale("log")
plt.xlabel("liczba watkow $p$")
plt.ylabel("czas [s] (skala log)")
plt.title("Czas wykonania wg sposobu synchronizacji")
plt.xticks(threads)
plt.grid(alpha=0.3, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("synchronizacja.pdf")
plt.savefig("synchronizacja.png", dpi=130)
plt.close()

# --- Wykres 3: porownanie schedule (16 watkow, metoda private) ---
sched_cfgs = [
    ("static", "0", "static"),
    ("static", "1000", "static,1000"),
    ("static", "100000", "static,100000"),
    ("dynamic", "1000", "dynamic,1000"),
    ("dynamic", "100000", "dynamic,100000"),
    ("guided", "0", "guided"),
]
labels, vals = [], []
for s, c, lab in sched_cfgs:
    labels.append(lab)
    vals.append(med[("par", "private", s, c, 16)] * 1000.0)  # ms
plt.figure(figsize=(6.5, 4))
bars = plt.bar(range(len(vals)), vals, color="#4c78a8")
plt.xticks(range(len(vals)), labels, rotation=30, ha="right")
plt.ylabel("czas [ms]")
plt.title("Porownanie schedule (16 watkow, prywatne histogramy)")
for b, v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("schedule.pdf")
plt.savefig("schedule.png", dpi=130)
plt.close()

print("T_seq =", T_seq)
print("speedup private:", [round(T_seq / x, 2) for x in t_priv])
print("speedup atomic :", [round(T_seq / x, 2) for x in t_atom])
print("speedup critical:", [round(T_seq / x, 2) for x in t_crit])
print("Wykresy zapisane.")
