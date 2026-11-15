#!/usr/bin/env bash
# Generuje zestawy danych (stale ziarno) do data/. Istniejace katalogi pomija.
# Uzycie: scripts/gen_data.sh [--full]   (--full dodaje N=1M, ok. 490 MB)
# Katalog: data/n<N>[_d<D>][_q<Q>]  (D=128 i Q=20 sa domyslne i pomijane w nazwie)
# k jest parametrem uruchomienia, nie danych.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEED="${SEED:-42}"
FULL=0; [[ "${1:-}" == "--full" ]] && FULL=1

gen() { # N D Q
  local n=$1 d=$2 q=$3 name="n$1"
  [[ $d -ne 128 ]] && name+="_d$d"
  [[ $q -ne 20  ]] && name+="_q$q"
  local dir="$ROOT/data/$name"
  if [[ -f "$dir/base.bin" && -f "$dir/queries.bin" ]]; then
    echo "[skip] $name"; return
  fi
  echo "[gen ] $name (N=$n D=$d Q=$q seed=$SEED)"
  python3 "$ROOT/generate_data.py" --n "$n" --d "$d" --q "$q" --seed "$SEED" --out_dir "$dir" >/dev/null
  echo "seed=$SEED N=$n D=$d Q=$q" > "$dir/SEED.txt"
}

# skalowanie N (D=128, Q=20)
for n in 1000 10000 100000; do gen $n 128 20; done
[[ $FULL -eq 1 ]] && gen 1000000 128 20
# skalowanie D (N=100k)
for d in 32 64 256 512; do gen 100000 $d 20; done
# skalowanie Q / batch (N=100k)
for q in 1 100 1000; do gen 100000 128 $q; done
echo "Gotowe: $ROOT/data (ziarno $SEED)"
