#!/usr/bin/env python3
"""Porównuje rankingi wersji z rankingiem referencji (ref.txt) w podanym katalogu.

Wypisuje CSV: wersja, liczba zapytań, zapytania z identyczną listą indeksów,
pozycje (zapytanie, miejsce) z innym indeksem, największa różnica odległości.
Użycie: python3 scripts/compare_rankings.py results/rankings > results/ranking_check.csv
"""
import os
import sys


def read(path):
    """Plik 'zapytanie pozycja indeks odległość' -> {zapytanie: [(indeks, odległość), ...]}"""
    res = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        q, _, idx, dist = line.split()
        res.setdefault(int(q), []).append((int(idx), float(dist)))
    return res


folder = sys.argv[1]
ref = read(os.path.join(folder, "ref.txt"))
print("version,queries,identical_queries,different_positions,max_dist_diff")
for name in sorted(os.listdir(folder)):
    if name == "ref.txt" or not name.endswith(".txt"):
        continue
    got = read(os.path.join(folder, name))
    assert got.keys() == ref.keys(), name
    same = diff_pos = 0
    max_diff = 0.0
    for q in ref:
        r, g = ref[q], got[q]
        diff_pos += sum(a[0] != b[0] for a, b in zip(r, g))
        same += all(a[0] == b[0] for a, b in zip(r, g))
        max_diff = max(max_diff, max(abs(a[1] - b[1]) for a, b in zip(r, g)))
    print(f"{name[:-4]},{len(ref)},{same},{diff_pos},{max_diff:.2e}")
