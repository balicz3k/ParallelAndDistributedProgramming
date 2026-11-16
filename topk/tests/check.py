#!/usr/bin/env python3
"""Test poprawnosci: porownuje wyniki kazdej implementacji z niezalezna
referencja numpy (float64, brute force, remis -> nizszy indeks).

Kryteria (euklides, float32 w implementacjach):
  * odleglosci zgodne z tolerancja: |d - d_ref| <= ATOL + RTOL*d_ref,
  * indeksy identyczne, chyba ze dwa kandydaci leza w granicy tolerancji
    (prawdziwy remis w float32) - wtedy akceptujemy zamiane, ale tylko jesli
    odleglosc elementu zwroconego jest rowna odleglosci oczekiwanego w tol.
  * dla k>N: implementacja albo konczy sie bledem (kod!=0, brak wynikow),
    albo zwraca min(k,N) wynikow na zapytanie.

Kazdy program uruchamiany jest z --warmup 0 --repeat 1 --out <plik> (wspolne
opcje z common/common.h), a ranking czytany z pliku.

Uzycie: python3 tests/check.py [--impl ref omp mpi cuda cublas] [--quick]
Kod wyjscia 0 = wszystko OK. Brak binarki => SKIP (nie blad), chyba ze --strict.
"""
import argparse, os, shutil, subprocess, sys, tempfile
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RTOL, ATOL = 1e-4, 1e-4  # na odleglosci (nie kwadraty); float32, D<=512

# nazwa -> (sciezka binarki, funkcja budujaca komende; p = watki/procesy)
def _cmd(exe, base, qry, N, D, Q, k, out):
    return [exe, base, qry, str(N), str(D), str(Q), str(k),
            "--warmup", "0", "--repeat", "1", "--lat", "1", "--out", out]

IMPLS = {
    "ref":  dict(exe="reference/topk_ref", ps=[None]),
    "omp":  dict(exe="openmp/topk_omp",    ps=[1, 4, 8], extra=lambda p: ["--threads", str(p)]),
    "mpi":  dict(exe="mpi/topk_mpi",       ps=[1, 2, 4],
                 wrap=lambda p: ["mpirun", "--oversubscribe", "-np", str(p)]),
    "cuda": dict(exe="cuda/topk_cuda",     ps=[None], max_k=64),  # kernel: k<=MAX_K=64
    # cuBLAS liczy d^2 = |q|^2 + |b|^2 - 2 q.b; dla prawie identycznych wektorow
    # odejmowanie duzych liczb daje blad d^2 rzedu 1e-6, czyli d rzedu 1e-3
    "cublas": dict(exe="cuda/topk_cublas", ps=[None], max_k=64, atol=2e-3),
}

def read_results(path):
    """Format: `q rank idx dist` (komentarze #). Zwraca dict q -> [(idx,dist)]."""
    res = {}
    for line in open(path):
        t = line.split()
        if len(t) == 4 and not line.lstrip().startswith("#"):
            try:
                q, r, i, d = int(t[0]), int(t[1]), int(t[2]), float(t[3])
            except ValueError:
                continue
            res.setdefault(q, []).append((r, i, d))
    return {q: [(i, d) for _, i, d in sorted(v)] for q, v in res.items()}

def truth(base, qry, k):
    kk = min(k, len(base))
    out = []
    b64 = base.astype(np.float64)
    for q in qry.astype(np.float64):
        d2 = ((b64 - q) ** 2).sum(1)
        idx = np.lexsort((np.arange(len(d2)), d2))[:kk]   # remis -> nizszy indeks
        out.append((idx, np.sqrt(d2[idx]), d2))
    return out

def compare(res, tr, Q, k, N, atol=ATOL):
    errs = []
    kk = min(k, N)
    for q in range(Q):
        got = res.get(q)
        if got is None or len(got) != kk:
            errs.append(f"q={q}: oczekiwano {kk} wynikow, jest {0 if got is None else len(got)}")
            continue
        idx, dist, d2 = tr[q]
        for r, (gi, gd) in enumerate(got):
            tol = atol + RTOL * dist[r]
            if abs(gd - dist[r]) > tol:
                errs.append(f"q={q} rank={r+1}: dist {gd} != {dist[r]:.6f}")
            if gi != idx[r]:
                ok = 0 <= gi < N and abs(np.sqrt(d2[gi]) - dist[r]) <= tol
                if not ok:
                    errs.append(f"q={q} rank={r+1}: idx {gi} != {idx[r]}")
                else:
                    errs.append(None)  # remis float32 - zamiana dopuszczona
        if len({i for i, _ in got}) != len(got):
            errs.append(f"q={q}: powtorzone indeksy w wyniku")
    return [e for e in errs if e], sum(1 for e in errs if e is None)

def make_cases(quick):
    rng = np.random.default_rng(12345)
    C = []  # (nazwa, base, queries, k)
    f = lambda *s: rng.standard_normal(s).astype(np.float32)
    C.append(("losowe N=2000 D=16 k=10", f(2000, 16), f(5, 16), 10))
    C.append(("k=1", f(500, 8), f(4, 8), 1))
    C.append(("Q=1", f(300, 32), f(1, 32), 5))
    C.append(("k=N", f(20, 8), f(3, 8), 20))
    C.append(("k>N", f(5, 8), f(2, 8), 10))
    C.append(("N=1 k=1", f(1, 4), f(2, 4), 1))
    C.append(("N male (N=3,k=2)", f(3, 4), f(2, 4), 2))
    b = f(50, 8); b[10:40] = b[10]          # 30 identycznych wektorow
    C.append(("duplikaty (remisy po indeksie)", b, b[[10, 3]].copy(), 12))
    b = np.zeros((100, 6), np.float32)       # wszystkie rowne -> czyste remisy
    C.append(("wszystkie rowne", b, np.zeros((2, 6), np.float32), 7))
    b = f(200, 8); b[17] = 0
    C.append(("wektor zerowy w bazie i zapytaniu", b, np.zeros((2, 8), np.float32), 5))
    b = np.round(f(400, 4) * 2).astype(np.float32)   # kratka calkowita -> duzo remisow
    C.append(("kratka calkowita (wiele remisow)", b, np.round(f(4, 4) * 2).astype(np.float32), 15))
    C.append(("N nie dzieli sie przez p (N=1001)", f(1001, 12), f(3, 12), 9))
    C.append(("D=1", f(300, 1), f(2, 1), 5))
    if not quick:
        C.append(("N=50000 D=128 k=100", f(50000, 128), f(8, 128), 100))
        C.append(("D=517 (nie wielokrotnosc 4)", f(3000, 517), f(3, 517), 10))
    return C

def run_impl(name, spec, case, tmp, strict):
    cname, base, qry, k = case
    N, D = base.shape; Q = len(qry)
    exe = os.path.join(ROOT, spec["exe"])
    if not os.path.isfile(exe):
        return "SKIP", ["brak binarki " + spec["exe"]]
    bp, qp = os.path.join(tmp, "base.bin"), os.path.join(tmp, "queries.bin")
    base.tofile(bp); qry.tofile(qp)
    tr = truth(base, qry, k)
    fails, ties = [], 0
    for p in spec["ps"]:
        wd = tempfile.mkdtemp(dir=tmp)
        out = os.path.join(wd, "ranking.txt")
        cmd = spec.get("wrap", lambda p: [])(p) + _cmd(exe, bp, qp, N, D, Q, k, out) \
              + (spec["extra"](p) if "extra" in spec and p else [])
        try:
            pr = subprocess.run(cmd, cwd=wd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            fails.append(f"p={p}: timeout"); continue
        res = read_results(out) if os.path.isfile(out) else {}
        tag = f"p={p}" if p else "-"
        if k > spec.get("max_k", 10**9) and pr.returncode != 0 and not res:
            continue  # znane ograniczenie implementacji (k > MAX_K) - czyste odrzucenie
        if k > N:
            if pr.returncode != 0 and not res:
                continue  # czyste odrzucenie - akceptowane
            if pr.returncode != 0:
                fails.append(f"{tag}: kod {pr.returncode} mimo wynikow"); continue
        elif pr.returncode != 0:
            fails.append(f"{tag}: kod wyjscia {pr.returncode}: {(pr.stderr or pr.stdout)[-120:].strip()}")
            continue
        e, t = compare(res, tr, Q, k, N, spec.get("atol", ATOL))
        ties += t
        fails += [f"{tag}: {x}" for x in e[:3]]
    return ("FAIL" if fails else "OK"), fails + ([f"(remisy float32 zaakceptowane: {ties})"] if ties and not fails else [])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", nargs="+", default=list(IMPLS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--strict", action="store_true", help="brak binarki = blad")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="topk_check_")
    bad = 0
    try:
        cases = make_cases(a.quick)
        print("przypadek".ljust(42) + "".join(f"{n:>8s}" for n in a.impl))
        for c in cases:
            row = f"{c[0]:42s}"; notes = []
            for n in a.impl:
                st, msg = run_impl(n, IMPLS[n], c, tmp, a.strict)
                if st == "SKIP" and a.strict: st = "FAIL"
                bad += st == "FAIL"
                row += f"{st:>8s}"
                notes += [f"   [{n}] {m}" for m in msg if st != "OK" or m.startswith("(")]
            print(row)
            for m in notes: print(m)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("WYNIK:", "BLEDY: %d" % bad if bad else "OK")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
