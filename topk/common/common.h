/*
 * common.h - kod wspólny dla wszystkich wersji top-k (referencja, OpenMP,
 * MPI, CUDA, cuBLAS): opcje wiersza poleceń, wczytywanie danych, statystyki,
 * zapis rankingu i wiersza CSV z pomiarem.
 *
 * Wspólne wywołanie każdego programu:
 *   <program> <baza.bin> <zapytania.bin> <N> <D> <Q> <k> [opcje]
 *     --threads T    wątki OpenMP albo wątki na blok CUDA (0 = domyślne)
 *     --warmup W     przebiegi rozgrzewkowe, bez zapisu (domyślnie 2)
 *     --repeat R     mierzone powtórzenia (domyślnie 7)
 *     --lat L        ile zapytań w trybie pojedynczym na powtórzenie (P50/P95, domyślnie 100)
 *     --csv          po jednym wierszu "CSV,..." na każde powtórzenie
 *     --out PLIK     zapis rankingu: "zapytanie pozycja indeks odległość"
 *     --no-stream    (tylko CUDA) bez strumieni
 *
 * Pliki danych: surowe float32, wiersz po wierszu, bez nagłówka. Z pliku
 * zapytań czytane jest pierwsze Q wierszy.
 */
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <sys/resource.h>

struct Options {
    std::string base_path, query_path, out_path;
    int N = 0, D = 0, Q = 0, k = 0;
    int threads = 0;
    int warmup = 2, repeat = 7, lat_queries = 100;
    bool csv = false, no_stream = false;
};

// Zwraca false, gdy argumenty są błędne (program wypisuje wtedy sposób użycia).
static bool parse_options(int argc, char **argv, Options &o) {
    if (argc < 7) return false;
    o.base_path = argv[1];
    o.query_path = argv[2];
    o.N = std::atoi(argv[3]);
    o.D = std::atoi(argv[4]);
    o.Q = std::atoi(argv[5]);
    o.k = std::atoi(argv[6]);
    for (int i = 7; i < argc; ++i) {
        std::string a = argv[i];
        bool has_value = i + 1 < argc;
        if (a == "--threads" && has_value)     o.threads = std::atoi(argv[++i]);
        else if (a == "--warmup" && has_value) o.warmup = std::atoi(argv[++i]);
        else if (a == "--repeat" && has_value) o.repeat = std::atoi(argv[++i]);
        else if (a == "--lat" && has_value)    o.lat_queries = std::atoi(argv[++i]);
        else if (a == "--out" && has_value)    o.out_path = argv[++i];
        else if (a == "--csv")                 o.csv = true;
        else if (a == "--no-stream")           o.no_stream = true;
        else {
            std::fprintf(stderr, "Nieznana opcja: %s\n", a.c_str());
            return false;
        }
    }
    if (o.N < 1 || o.D < 1 || o.Q < 1 || o.k < 1 || o.repeat < 1 || o.warmup < 0) return false;
    if (o.k > o.N) {
        std::fprintf(stderr, "Błąd: k (%d) > N (%d)\n", o.k, o.N);
        return false;
    }
    o.lat_queries = std::max(0, std::min(o.lat_queries, o.Q));
    return true;
}

static void print_usage(const char *prog) {
    std::fprintf(stderr,
        "Użycie: %s <baza.bin> <zapytania.bin> <N> <D> <Q> <k> [--threads T]\n"
        "        [--warmup W] [--repeat R] [--lat L] [--csv] [--out PLIK] [--no-stream]\n",
        prog);
}

// Wczytuje count liczb float32 z początku pliku (plik może być dłuższy).
static bool load_floats(const std::string &path, size_t count, std::vector<float> &data) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "Błąd: nie można otworzyć %s\n", path.c_str());
        return false;
    }
    data.resize(count);
    f.read(reinterpret_cast<char *>(data.data()), (std::streamsize)(count * sizeof(float)));
    if (!f) {
        std::fprintf(stderr, "Błąd: plik %s jest za krótki\n", path.c_str());
        return false;
    }
    return true;
}

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Percentyl p (0-100) z interpolacją liniową między sąsiednimi obserwacjami.
static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double pos = p / 100.0 * (double)(v.size() - 1);
    size_t lo = (size_t)pos;
    size_t hi = std::min(lo + 1, v.size() - 1);
    double frac = pos - (double)lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

// Szczytowe zużycie RAM przez proces (Linux: ru_maxrss w KiB).
static double peak_rss_mb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss / 1024.0;
}

// Jeden pomiar (jedno powtórzenie). Czasy w ms, opóźnienia w mikrosekundach.
struct Measurement {
    double io_ms = 0;     // wczytanie plików (raz, poza pomiarem partii)
    double base_ms = 0;   // jednorazowe przygotowanie bazy: Scatterv (MPI), kopia na GPU (CUDA)
    double core_ms = 0;   // same obliczenia dla całej partii
    double full_ms = 0;   // cała partia: obliczenia + komunikacja / transfery
    std::vector<double> lat_us;  // opóźnienia zapytań obsługiwanych pojedynczo
    double rss_mb = 0, vram_mb = 0;
};

// Wiersz CSV (kolumny jak w nagłówku results/raw.csv, patrz scripts/run_all.sh):
// version,N,D,Q,k,threads,procs,block,rep,io_ms,base_ms,core_ms,full_ms,qps,p50_us,p95_us,rss_mb,vram_mb
static void print_csv(const char *version, const Options &o, int threads, int procs,
                      int block, int rep, const Measurement &m) {
    double qps = o.Q / m.full_ms * 1000.0;
    std::printf("CSV,%s,%d,%d,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.4f,%.4f,%.2f,%.2f,%.2f,%.1f,%.1f\n",
                version, o.N, o.D, o.Q, o.k, threads, procs, block, rep,
                m.io_ms, m.base_ms, m.core_ms, m.full_ms, qps,
                percentile(m.lat_us, 50.0), percentile(m.lat_us, 95.0), m.rss_mb, m.vram_mb);
}

// Krótkie podsumowanie na koniec (mediany z powtórzeń).
static void print_summary(const char *version, const Options &o,
                          const std::vector<Measurement> &ms) {
    std::vector<double> core, full, p95;
    for (const Measurement &m : ms) {
        core.push_back(m.core_ms);
        full.push_back(m.full_ms);
        p95.push_back(percentile(m.lat_us, 95.0));
    }
    double full_med = percentile(full, 50.0);
    std::printf("%s: N=%d D=%d Q=%d k=%d, powtórzeń %d (+%d rozgrzewki)\n",
                version, o.N, o.D, o.Q, o.k, o.repeat, o.warmup);
    std::printf("  partia: rdzeń %.3f ms, pełny %.3f ms (min %.3f, max %.3f), QPS %.1f\n",
                percentile(core, 50.0), full_med,
                *std::min_element(full.begin(), full.end()),
                *std::max_element(full.begin(), full.end()), o.Q / full_med * 1000.0);
    std::printf("  pojedyncze zapytanie: P95 %.1f us (mediana z powtórzeń, %d zapytań w każdym)\n",
                percentile(p95, 50.0), o.lat_queries);
    std::printf("  pamięć: RAM %.1f MiB, VRAM %.1f MiB, wczytanie plików %.1f ms\n",
                ms.back().rss_mb, ms.back().vram_mb, ms.back().io_ms);
}

// Zapis rankingu: idx i dist to tablice Q*k (dist = prawdziwa odległość, nie kwadrat).
static void write_ranking(const std::string &path, int Q, int k,
                          const std::vector<int> &idx, const std::vector<float> &dist) {
    FILE *f = std::fopen(path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "Błąd: nie można zapisać %s\n", path.c_str());
        return;
    }
    std::fprintf(f, "# zapytanie pozycja indeks odleglosc\n");
    for (int q = 0; q < Q; ++q)
        for (int i = 0; i < k; ++i)
            std::fprintf(f, "%d %d %d %.6f\n", q, i + 1, idx[(size_t)q * k + i],
                         dist[(size_t)q * k + i]);
    std::fclose(f);
}
