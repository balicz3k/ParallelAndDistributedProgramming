#!/usr/bin/env bash
# Pełny przebieg pomiarów projektu D03 z parametrami z experiments.conf.
#
#   scripts/run_all.sh            wszystko od zera: build, dane, testy, wszystkie części
#   scripts/run_all.sh mpi k      tylko wybrane części, dopisywane do results/raw.csv
#   scripts/run_all.sh --dry ...  próba: 1 powtórzenie bez rozgrzewki -> results/dry.csv
#
# Części: omp mpi cuda n d k q rank
# Wyniki: results/raw.csv (jeden wiersz na powtórzenie), results/env.txt,
#         results/ranking_check.csv; log w results/run_all.log.
# Pomiary uruchamiane są po kolei, nigdy równolegle. Jeśli GPU jest zajęte
# przez inny proces (wykorzystanie > 10%), skrypt kończy się bez pomiarów
# (FORCE=1 pomija to sprawdzenie).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source experiments.conf

OUT=results/raw.csv
if [[ "${1:-}" == "--dry" ]]; then
    shift
    WARMUP=0; REPEAT=1; LAT_QUERIES=10
    OUT=results/dry.csv
fi
ALL_PARTS="omp mpi cuda n d k q rank"
PARTS="${*:-$ALL_PARTS}"
LOG=results/run_all.log
HEADER="version,N,D,Q,k,threads,procs,block,rep,io_ms,base_ms,core_ms,full_ms,qps,p50_us,p95_us,rss_mb,vram_mb"
mkdir -p results

gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
if (( gpu_util > 10 )) && [[ -z "${FORCE:-}" ]]; then
    echo "GPU jest zajęte (${gpu_util}%) - przerywam, żeby nie zafałszować pomiarów."
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    exit 1
fi

# pełny przebieg: build, dane, testy i nowy plik wyników
if [[ $# -eq 0 ]]; then
    make build
    scripts/gen_data.sh
    python3 tests/check.py --strict
    echo "$HEADER" > "$OUT"
    : > "$LOG"
fi
[[ -f "$OUT" ]] || echo "$HEADER" > "$OUT"

# środowisko pomiaru (nadpisywane przy każdym uruchomieniu)
{
    echo "data: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "commit: $(git rev-parse HEAD)"
    echo "części: $PARTS"
    echo "CPU: $(lscpu | sed -n 's/^Model name: *//p')"
    echo "rdzenie/wątki: $(nproc --all) wątków sprzętowych"
    echo "RAM: $(free -m | awk '/^Mem:/ {print $2}') MiB"
    echo "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
    echo "GPU przed pomiarami: wykorzystanie ${gpu_util}%, $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) zajęte"
    echo "obciążenie CPU (loadavg): $(cut -d' ' -f1-3 /proc/loadavg)"
    echo "system: $(uname -sr)"
    echo "g++: $(g++ --version | head -1)"
    echo "Open MPI: $(ompi_info | sed -n 's/^ *Open MPI: *//p' | head -1)"
    echo "nvcc: $(nvcc --version | tail -1)"
    echo "python: $(python3 --version), numpy $(python3 -c 'import numpy; print(numpy.__version__)')"
    echo "rozgrzewka/powtórzenia/zapytania do P95: $WARMUP/$REPEAT/$LAT_QUERIES"
} > results/env.txt

# measure N D Q k "polecenie" [opcje] - jeden program, wiersze CSV dopisywane do $OUT
measure() {
    local n=$1 d=$2 q=$3 k=$4 cmd=$5
    shift 5
    local dir="data/n${n}_d${d}"
    echo "[$(date +%H:%M:%S)] $cmd N=$n D=$d Q=$q k=$k $*" | tee -a "$LOG"
    $cmd "$dir/base.bin" "$dir/queries.bin" "$n" "$d" "$q" "$k" \
        --warmup "$WARMUP" --repeat "$REPEAT" --lat "$LAT_QUERIES" --csv "$@" < /dev/null \
        | tee -a "$LOG" | sed -n 's/^CSV,//p' >> "$OUT"
}

# wersje porównywane w badaniach N, D, k, Q
sweep_versions() {  # N D Q k [bez_ref]
    [[ "${5:-}" == "bez_ref" ]] || measure "$1" "$2" "$3" "$4" reference/topk_ref
    measure "$1" "$2" "$3" "$4" openmp/topk_omp --threads "$SWEEP_OMP_THREADS"
    measure "$1" "$2" "$3" "$4" "mpirun -np $SWEEP_MPI_PROCS mpi/topk_mpi"
    measure "$1" "$2" "$3" "$4" cuda/topk_cuda
    measure "$1" "$2" "$3" "$4" cuda/topk_cublas
}

M="$MAIN_N $MAIN_D $MAIN_Q $MAIN_K"
for part in $PARTS; do
    case $part in
    omp)   # referencja i skalowanie wątków OpenMP w przypadku głównym
        measure $M reference/topk_ref
        for t in $OMP_THREADS; do measure $M openmp/topk_omp --threads "$t"; done ;;
    mpi)   # skalowanie procesów MPI i wariant hybrydowy (bez wiązania z rdzeniem)
        for p in $MPI_PROCS; do measure $M "mpirun -np $p mpi/topk_mpi"; done
        for h in $HYBRID; do
            p=${h%x*}; t=${h#*x}
            measure $M "mpirun -np $p --bind-to none -x OMP_NUM_THREADS=$t mpi/topk_mpi_hybrid"
        done ;;
    cuda)  # konfiguracja bloków, strumienie, cuBLAS
        for b in $CUDA_BLOCKS; do measure $M cuda/topk_cuda --threads "$b"; done
        measure $M cuda/topk_cuda --no-stream
        measure $M cuda/topk_cublas ;;
    n) for n in $SWEEP_N; do [[ $n == "$MAIN_N" ]] || sweep_versions "$n" "$MAIN_D" "$MAIN_Q" "$MAIN_K"; done ;;
    d) for d in $SWEEP_D; do [[ $d == "$MAIN_D" ]] || sweep_versions "$MAIN_N" "$d" "$MAIN_Q" "$MAIN_K"; done ;;
    k) for k in $SWEEP_K; do [[ $k == "$MAIN_K" ]] || sweep_versions "$MAIN_N" "$MAIN_D" "$MAIN_Q" "$k"; done ;;
    # Q: bez referencji - liczy zapytania po kolei, więc jej QPS nie zależy od Q
    q) for q in $SWEEP_Q; do [[ $q == "$MAIN_Q" ]] || sweep_versions "$MAIN_N" "$MAIN_D" "$q" "$MAIN_K" bez_ref; done ;;
    rank)  # rankingi przypadku głównego i porównanie z referencją
        mkdir -p results/rankings
        dir="data/n${MAIN_N}_d${MAIN_D}"
        args="$dir/base.bin $dir/queries.bin $M --warmup 0 --repeat 1 --lat 0"
        reference/topk_ref $args --out results/rankings/ref.txt > /dev/null
        openmp/topk_omp $args --threads "$SWEEP_OMP_THREADS" --out results/rankings/omp.txt > /dev/null
        mpirun -np "$SWEEP_MPI_PROCS" mpi/topk_mpi $args --out results/rankings/mpi.txt < /dev/null > /dev/null
        mpirun -np 4 --bind-to none -x OMP_NUM_THREADS=4 mpi/topk_mpi_hybrid $args \
            --out results/rankings/mpi_omp.txt < /dev/null > /dev/null
        cuda/topk_cuda $args --out results/rankings/cuda.txt > /dev/null
        cuda/topk_cublas $args --out results/rankings/cublas.txt > /dev/null
        python3 scripts/compare_rankings.py results/rankings > results/ranking_check.csv
        cat results/ranking_check.csv ;;
    *) echo "Nieznana część: $part (dostępne: $ALL_PARTS)"; exit 1 ;;
    esac
done
echo "Gotowe: $OUT"
