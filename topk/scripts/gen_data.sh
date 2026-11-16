#!/usr/bin/env bash
# Generuje zestawy danych z experiments.conf do data/n<N>_d<D>/.
# Każdy zestaw ma base.bin (N x D) i queries.bin (Q_MAX x D); programy czytają
# tylko pierwsze Q zapytań, więc jeden plik zapytań wystarcza dla każdego Q.
# Istniejące zestawy są pomijane.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/experiments.conf"

gen() {  # N D
    local dir="$ROOT/data/n$1_d$2"
    if [[ -f "$dir/base.bin" && -f "$dir/queries.bin" ]]; then
        echo "[jest ] n$1_d$2"
        return
    fi
    echo "[tworze] n$1_d$2 (seed $SEED, Q=$Q_MAX)"
    python3 "$ROOT/generate_data.py" --n "$1" --d "$2" --q "$Q_MAX" \
        --seed "$SEED" --out_dir "$dir" > /dev/null
    echo "seed=$SEED N=$1 D=$2 Q=$Q_MAX" > "$dir/SEED.txt"
}

for n in $SWEEP_N; do gen "$n" "$MAIN_D"; done
for d in $SWEEP_D; do gen "$MAIN_N" "$d"; done
echo "Gotowe: $ROOT/data"
