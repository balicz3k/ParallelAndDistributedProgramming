#!/bin/bash
# Pomiar czasu scatter_gather dla p = 2..8 procesów.
# Każde powtórzenie to osobne uruchomienie mpirun; pierwsze (rozgrzewka) jest pomijane.
#   results_raw.csv       - N = 10000 (zadanie)
#   results_raw_duze.csv  - N = 10 000 000 (dodatkowo, dla porównania)
# Użycie: ./run_benchmark.sh [R]   (R - liczba powtórzeń, domyślnie 20)
set -e

R=${1:-20}

measure() {
    local n=$1 out=$2
    echo "p,rep,time_s,total_sum,check" > "$out"
    for p in 2 3 4 5 6 7 8; do
        mpirun -np $p ./scatter_gather $n csv > /dev/null      # rozgrzewka
        for rep in $(seq 1 $R); do
            line=$(mpirun -np $p ./scatter_gather $n csv)
            # program wypisuje p,time_s,total_sum,check; dopisuję numer powtórzenia
            echo "$line" | awk -F, -v r=$rep '{print $1 "," r "," $2 "," $3 "," $4}' >> "$out"
        done
        echo "N=$n p=$p gotowe"
    done
}

measure 10000 results_raw.csv
measure 10000000 results_raw_duze.csv
