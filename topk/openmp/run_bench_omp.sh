#!/usr/bin/env bash
# Benchmark harness dla OpenMP top-k. Uruchamia binarke topk_omp dla siatki
# konfiguracji, zbiera wiersze CSV ze stderr, liczy S(p)/E(p) i sklada
# jeden spojny plik results_omp.txt. Warmup + powtorzenia robi sama binarka.
set -u
cd "$(dirname "$0")"
BIN=./topk_omp
REP=7                      # niezaleznych powtorzen batcha (binarka robi >=1 warmup)
THREADS="1 2 4 8 16"
DATA=../data
REFBIN=../reference/topk_ref
OUT=results_omp.txt
RAW=/tmp/omp_csv.txt
: > "$RAW"

run() { # base queries N D Q k threads  ->  echo CSV line
  local base=$1 q=$2 N=$3 D=$4 Q=$5 k=$6 t=$7
  OMP_NUM_THREADS=$t $BIN "$base" "$q" "$N" "$D" "$Q" "$k" "$t" "$REP" 2>>"$RAW" >/dev/null
}

# tabela: wyciagnij pole z ostatniego CSV o danym (N,D,Q,k,threads)
field() { # key... uses awk on $RAW ; args: N D Q k t col(1-based within CSV after prefix)
  local N=$1 D=$2 Q=$3 k=$4 t=$5 col=$6
  awk -F, -v N=$N -v D=$D -v Q=$Q -v k=$k -v t=$t -v c=$col \
    '$1=="CSV" && $2==N && $3==D && $4==Q && $5==k && $6==t {v=$(c)} END{print v}' "$RAW"
}
# CSV columns (1-indexed): 1 CSV,2 N,3 D,4 Q,5 k,6 threads,7 repeat,
# 8 med_ms,9 min_ms,10 max_ms,11 iqr_ms,12 p50_us,13 p95_us,14 qps,15 rss_kib

{
echo "# ============================================================"
echo "# OpenMP top-k — wyniki benchmarku (jeden spojny format)"
echo "# Data: $(date -Iseconds)"
echo "# Sprzet: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
echo "#         rdzenie/watki: $(nproc) logicznych; RAM: $(free -g | awk '/Mem:/{print $2" GiB"}')"
echo "# Kompilator: $(g++ --version | head -1)"
echo "# Flagi: -O3 -std=c++17 -fopenmp"
echo "# Metryka czasu: RDZEN batcha (bez I/O wczytania plikow)."
echo "# Powtorzen batcha: $REP (po warmupie). Mediana + rozrzut (min/max/IQR)."
echo "# P95: 95. percentyl opoznienia POJEDYNCZEGO zapytania z Q*$REP obserwacji."
echo "# Metryka odleglosci: euklidesowa; remisy -> mniejszy indeks. Zgodne z topk_ref."
echo "# Zapytania benchmarku: data/n*/queries_bench.bin (Q=250, seed=20260404)."
echo "# ============================================================"
echo ""
} > "$OUT"

# ---------- CZESC 1: skalowanie watkow S(p)/E(p) przy stalym problemie -------
# Konfiguracja bazowa: N=100000 D=128 Q=250 k=10
BN=100000; BD=128; BQ=250; BK=10
BASE=$DATA/n100000/base.bin; QRY=$DATA/n100000/queries_bench.bin
for t in $THREADS; do run "$BASE" "$QRY" $BN $BD $BQ $BK $t; done

{
echo "## 1) Skalowanie liczby watkow (problem staly: N=$BN D=$BD Q=$BQ k=$BK)"
echo "#   S(p)=T(1)/T(p) wg mediany czasu batcha; E(p)=S(p)/p."
printf "# %-7s %-10s %-9s %-9s %-9s %-10s %-9s %-7s %-7s %-9s\n" \
  threads med_ms min_ms max_ms IQR_ms P95_us QPS "S(p)" "E(p)" RSS_MiB
T1=$(field $BN $BD $BQ $BK 1 8)
for t in $THREADS; do
  med=$(field $BN $BD $BQ $BK $t 8); mn=$(field $BN $BD $BQ $BK $t 9)
  mx=$(field $BN $BD $BQ $BK $t 10); iq=$(field $BN $BD $BQ $BK $t 11)
  p95=$(field $BN $BD $BQ $BK $t 13); qps=$(field $BN $BD $BQ $BK $t 14)
  rss=$(field $BN $BD $BQ $BK $t 15)
  sp=$(awk -v a=$T1 -v b=$med 'BEGIN{printf "%.3f", a/b}')
  ep=$(awk -v s=$sp -v p=$t 'BEGIN{printf "%.3f", s/p}')
  rm=$(awk -v r=$rss 'BEGIN{printf "%.1f", r/1024}')
  printf "  %-7s %-10s %-9s %-9s %-9s %-10s %-9s %-7s %-7s %-9s\n" \
    $t $med $mn $mx $iq $p95 $qps $sp $ep $rm
done
echo ""
} >> "$OUT"

# ---------- CZESC 2: sweep N (10k/100k/1M) przy 8 i 16 watkach ---------------
{
echo "## 2) Sweep rozmiaru bazy N (D=128 Q=250 k=10), watki=8 i 16"
printf "# %-9s %-7s %-10s %-10s %-9s %-9s\n" N threads med_ms P95_us QPS RSS_MiB
} >> "$OUT"
for Nd in 10000:n10000 100000:n100000 1000000:n1000000; do
  N=${Nd%%:*}; dir=${Nd##*:}
  B=$DATA/$dir/base.bin; Qf=$DATA/$dir/queries_bench.bin
  for t in 8 16; do
    run "$B" "$Qf" $N 128 250 10 $t
    med=$(field $N 128 250 10 $t 8); p95=$(field $N 128 250 10 $t 13)
    qps=$(field $N 128 250 10 $t 14); rss=$(field $N 128 250 10 $t 15)
    rm=$(awk -v r=$rss 'BEGIN{printf "%.1f", r/1024}')
    printf "  %-9s %-7s %-10s %-10s %-9s %-9s\n" $N $t $med $p95 $qps $rm >> "$OUT"
  done
done
echo "" >> "$OUT"

# ---------- CZESC 3: sweep k (1/10/50/100) przy N=100000, 16 watkow ----------
{
echo "## 3) Sweep k (N=100000 D=128 Q=250), watki=16"
printf "# %-6s %-10s %-10s %-9s %-9s\n" k med_ms P95_us QPS RSS_MiB
} >> "$OUT"
for k in 1 10 50 100; do
  run "$BASE" "$QRY" 100000 128 250 $k 16
  med=$(field 100000 128 250 $k 16 8); p95=$(field 100000 128 250 $k 16 13)
  qps=$(field 100000 128 250 $k 16 14); rss=$(field 100000 128 250 $k 16 15)
  rm=$(awk -v r=$rss 'BEGIN{printf "%.1f", r/1024}')
  printf "  %-6s %-10s %-10s %-9s %-9s\n" $k $med $p95 $qps $rm >> "$OUT"
done
echo "" >> "$OUT"

# ---------- CZESC 4: sweep Q (batch) przy N=100000 k=10, 16 watkow ----------
{
echo "## 4) Sweep wielkosci partii Q (N=100000 D=128 k=10), watki=16"
printf "# %-6s %-10s %-10s %-9s\n" Q med_ms P95_us QPS
} >> "$OUT"
for Q in 50 100 200 250; do
  run "$BASE" "$QRY" 100000 128 $Q 10 16
  med=$(field 100000 128 $Q 10 16 8); p95=$(field 100000 128 $Q 10 16 13)
  qps=$(field 100000 128 $Q 10 16 14)
  printf "  %-6s %-10s %-10s %-9s\n" $Q $med $p95 $qps >> "$OUT"
done
echo "" >> "$OUT"

# ---------- CZESC 5: poprawnosc vs referencja (kilka zestawow) ---------------
{
echo "## 5) Poprawnosc: porownanie indeksow top-k z referencja sekwencyjna"
} >> "$OUT"
verify_set() { # dir N Q k
  local dir=$1 N=$2 Q=$3 k=$4
  ( cd ../reference && ./topk_ref ../data/$dir/base.bin ../data/$dir/queries_bench.bin $N 128 $Q $k >/dev/null 2>&1 && cp results.txt /tmp/ref_verify.txt )
  local r
  for t in 1 8 16; do
    r=$($BIN $DATA/$dir/base.bin $DATA/$dir/queries_bench.bin $N 128 $Q $k $t 1 /tmp/ref_verify.txt 2>/dev/null | grep -c "VERIFY. OK")
    echo "  N=$N Q=$Q k=$k threads=$t : $([ "$r" = "1" ] && echo OK || echo BLAD)" >> "$OUT"
  done
}
verify_set n10000 10000 250 1
verify_set n10000 10000 250 50
verify_set n100000 100000 250 10
echo "" >> "$OUT"
echo "# Surowe wiersze CSV (kolumny: N,D,Q,k,threads,repeat,med_ms,min_ms,max_ms,iqr_ms,p50_us,p95_us,qps,rss_kib):" >> "$OUT"
grep '^CSV' "$RAW" >> "$OUT"
echo "DONE"
