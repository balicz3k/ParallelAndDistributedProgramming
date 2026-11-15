/*
 * PRiR Laboratorium 2 — Równoległa analiza chmury punktów z OpenMP
 *
 * Realizacja polecenia "Laboratorium_OpenMP_Chmura_Punktow":
 *   - wczytanie chmury punktów XYZ do struktury Point{double x,y,z}
 *   - dla każdego punktu: angle = atan2(y-Y0, x-X0),
 *     r = sqrt(dx*dx+dy*dy), value = sin(angle)*cos(angle)*log(r+1)
 *   - statystyki globalne: min(value), max(value), sum(value)
 *   - podział na 100 sektorów kątowych i histogram liczby punktów
 *   - wersja sekwencyjna i równoległa, porównanie wyników (poprawność)
 *   - porównanie synchronizacji histogramu: critical / atomic / prywatne histogramy
 *   - porównanie strategii schedule (static/dynamic/guided, różne chunk)
 *   - pomiar czasu omp_get_wtime(), >=5 powtórzeń, mediana + rozrzut
 *
 * Kompilacja: g++ -O3 -fopenmp -std=c++17 -Wall -Wextra lab2_openmp.cpp -o lab2
 * Uruchomienie:
 *   ./lab2 <plik> correctness [threads]   — poprawność: seq vs par, 3 metody histogramu
 *   ./lab2 <plik> bench [powtorzenia]      — pełny benchmark (CSV na stdout)
 *   ./lab2 <plik>                          — skrót: correctness z maks. wątkami
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <cstdio>
#include <omp.h>

// ------------------------------------------------------------
// Parametry z polecenia
// ------------------------------------------------------------
static const double X0 = 566809.6;
static const double Y0 = 243450.1;
static const int    NSECT = 100;                 // liczba sektorów kątowych
static const double TWO_PI = 2.0 * M_PI;

struct Point { double x, y, z; };

// Wynik analizy: statystyki + histogram sektorów
struct Result {
    double vmin, vmax, vsum;
    std::vector<long long> hist;   // rozmiar NSECT
    Result() : vmin(0), vmax(0), vsum(0), hist(NSECT, 0) {}
};

// Numer sektora kątowego dla punktu. atan2 daje kąt w (-pi, pi];
// przesuwamy do [0, 2pi) i dzielimy na NSECT równych wycinków.
static inline int sector_of(double angle) {
    double a = angle;
    if (a < 0.0) a += TWO_PI;
    int s = static_cast<int>(a / TWO_PI * NSECT);
    if (s < 0) s = 0;
    if (s >= NSECT) s = NSECT - 1;      // zabezpieczenie dla a == 2pi
    return s;
}

// value dla pojedynczego punktu (wspólne dla wersji seq i par)
static inline double point_value(const Point& p, int& sector) {
    double dx = p.x - X0;
    double dy = p.y - Y0;
    double angle = std::atan2(dy, dx);
    double r = std::sqrt(dx * dx + dy * dy);
    sector = sector_of(angle);
    return std::sin(angle) * std::cos(angle) * std::log(r + 1.0);
}

// ------------------------------------------------------------
// Wczytywanie danych (jednowątkowe, poza pomiarem)
// ------------------------------------------------------------
std::vector<Point> load_points(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Blad: nie mozna otworzyc pliku: " << filename << "\n";
        std::exit(1);
    }
    std::vector<Point> pts;
    pts.reserve(7'000'000);
    Point p;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        if (ss >> p.x >> p.y >> p.z) pts.push_back(p);
    }
    return pts;
}

// ------------------------------------------------------------
// Wersja SEKWENCYJNA
// ------------------------------------------------------------
Result analyze_seq(const std::vector<Point>& pts) {
    Result R;
    R.vmin = std::numeric_limits<double>::max();
    R.vmax = std::numeric_limits<double>::lowest();
    R.vsum = 0.0;
    const long long N = static_cast<long long>(pts.size());
    for (long long i = 0; i < N; ++i) {
        int s;
        double v = point_value(pts[i], s);
        if (v < R.vmin) R.vmin = v;
        if (v > R.vmax) R.vmax = v;
        R.vsum += v;
        R.hist[s]++;
    }
    return R;
}

// ------------------------------------------------------------
// Wersja RÓWNOLEGŁA
// Statystyki: reduction(min/max/+). Histogram: 3 warianty synchronizacji.
//   mode 0 = critical, 1 = atomic, 2 = prywatne histogramy (reduction-ręczna)
// ------------------------------------------------------------
Result analyze_par(const std::vector<Point>& pts, int hist_mode,
                   const char* sched, int chunk) {
    Result R;
    double vmin = std::numeric_limits<double>::max();
    double vmax = std::numeric_limits<double>::lowest();
    double vsum = 0.0;
    const long long N = static_cast<long long>(pts.size());
    std::vector<long long>& hist = R.hist;   // wspolny histogram (mode 0,1)

    // Wybor klauzuli schedule jest kompilowany statycznie w kazdej galezi,
    // bo "schedule" nie moze byc zmienna. Makro skraca powtarzalny kod.
    #define LOOP_BODY \
        for (long long i = 0; i < N; ++i) { \
            int s; \
            double v = point_value(pts[i], s); \
            if (v < vmin) vmin = v; \
            if (v > vmax) vmax = v; \
            vsum += v; \
            if (hist_mode == 0) { _Pragma("omp critical") { hist[s]++; } } \
            else                { _Pragma("omp atomic")   hist[s]++;    } \
        }

    if (hist_mode == 2) {
        // Prywatne histogramy per watek + scalenie (ownhist / reduction reczna)
        #define PRIV_BODY \
            for (long long i = 0; i < N; ++i) { \
                int s; \
                double v = point_value(pts[i], s); \
                if (v < lmin) lmin = v; \
                if (v > lmax) lmax = v; \
                lsum += v; \
                local[s]++; \
            }
        #pragma omp parallel
        {
            std::vector<long long> local(NSECT, 0);
            double lmin = std::numeric_limits<double>::max();
            double lmax = std::numeric_limits<double>::lowest();
            double lsum = 0.0;
            std::string sc(sched);
            if (sc == "static") {
                #pragma omp for schedule(static) nowait
                PRIV_BODY
            } else if (sc == "dynamic") {
                int ch = chunk > 0 ? chunk : 4096;
                #pragma omp for schedule(dynamic, ch) nowait
                PRIV_BODY
            } else {
                #pragma omp for schedule(guided) nowait
                PRIV_BODY
            }
            #pragma omp critical
            {
                if (lmin < vmin) vmin = lmin;
                if (lmax > vmax) vmax = lmax;
                vsum += lsum;
                for (int k = 0; k < NSECT; ++k) hist[k] += local[k];
            }
        }
        #undef PRIV_BODY
    } else {
        // mode 0 (critical) lub 1 (atomic) na wspolnym histogramie.
        // Statystyki zawsze przez reduction.
        std::string sc(sched);
        if (sc == "static") {
            if (chunk > 0) {
                #pragma omp parallel for schedule(static, chunk) \
                    reduction(min:vmin) reduction(max:vmax) reduction(+:vsum)
                LOOP_BODY
            } else {
                #pragma omp parallel for schedule(static) \
                    reduction(min:vmin) reduction(max:vmax) reduction(+:vsum)
                LOOP_BODY
            }
        } else if (sc == "dynamic") {
            #pragma omp parallel for schedule(dynamic, chunk > 0 ? chunk : 4096) \
                reduction(min:vmin) reduction(max:vmax) reduction(+:vsum)
            LOOP_BODY
        } else { // guided
            #pragma omp parallel for schedule(guided) \
                reduction(min:vmin) reduction(max:vmax) reduction(+:vsum)
            LOOP_BODY
        }
    }
    #undef LOOP_BODY

    R.vmin = vmin; R.vmax = vmax; R.vsum = vsum;
    return R;
}

// ------------------------------------------------------------
// Mediana wektora czasów
// ------------------------------------------------------------
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n == 0) return 0.0;
    return (n % 2) ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}

bool results_equal(const Result& a, const Result& b) {
    if (std::fabs(a.vmin - b.vmin) > 1e-9) return false;
    if (std::fabs(a.vmax - b.vmax) > 1e-9) return false;
    if (std::fabs(a.vsum - b.vsum) > 1e-3) return false; // suma ~ milionow skladnikow
    for (int k = 0; k < NSECT; ++k) if (a.hist[k] != b.hist[k]) return false;
    return true;
}

// ------------------------------------------------------------
// Tryb POPRAWNOŚCI: seq vs par (3 metody histogramu), wypis statystyk i histogramu
// ------------------------------------------------------------
void run_correctness(const std::vector<Point>& pts, int threads) {
    omp_set_num_threads(threads);
    std::cout << "=== Poprawnosc (watkow: " << threads << ") ===\n";
    std::cout << "Punktow: " << pts.size() << "\n";

    Result seq = analyze_seq(pts);
    Result p0 = analyze_par(pts, 0, "static", 0);
    Result p1 = analyze_par(pts, 1, "static", 0);
    Result p2 = analyze_par(pts, 2, "static", 0);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Statystyki (sekwencyjnie):\n";
    std::cout << "  min(value) = " << seq.vmin << "\n";
    std::cout << "  max(value) = " << seq.vmax << "\n";
    std::cout << "  sum(value) = " << std::setprecision(4) << seq.vsum << "\n";

    std::cout << "Zgodnosc seq vs par:\n";
    std::cout << "  critical           : " << (results_equal(seq, p0) ? "OK" : "ROZNICA") << "\n";
    std::cout << "  atomic             : " << (results_equal(seq, p1) ? "OK" : "ROZNICA") << "\n";
    std::cout << "  prywatne histogramy: " << (results_equal(seq, p2) ? "OK" : "ROZNICA") << "\n";

    std::cout << "Histogram sektorow (sektor:liczba):\n";
    for (int k = 0; k < NSECT; ++k) {
        std::cout << std::setw(3) << k << ":" << std::setw(10) << seq.hist[k];
        std::cout << ((k % 5 == 4) ? "\n" : "  ");
    }
    std::cout << "\n";
}

// ------------------------------------------------------------
// Tryb BENCHMARK: CSV na stdout
//   kind,method,sched,chunk,threads,rep,time_s
// kind=seq|par; method=critical|atomic|private (tylko dla par)
// ------------------------------------------------------------
void run_bench(const std::vector<Point>& pts, int reps) {
    std::fprintf(stderr, "Benchmark: punktow=%zu, powtorzen=%d, maxthreads=%d\n",
                 pts.size(), reps, omp_get_max_threads());
    std::printf("kind,method,sched,chunk,threads,rep,time_s\n");

    const int thread_list[] = {1, 2, 4, 8, 16};
    const int nthr = 5;

    // 1) SEKWENCYJNIE (jeden watek, bez OpenMP-parallel) — baza czasu
    for (int rep = 0; rep < reps; ++rep) {
        double t0 = omp_get_wtime();
        volatile Result R = analyze_seq(pts);
        double t1 = omp_get_wtime();
        (void)R;
        std::printf("seq,seq,-,0,1,%d,%.6f\n", rep, t1 - t0);
    }

    // 2) RÓWNOLEGLE — 3 metody histogramu x schedule(static) x liczba watkow
    const char* methods[] = {"critical", "atomic", "private"};
    for (int m = 0; m < 3; ++m) {
        for (int ti = 0; ti < nthr; ++ti) {
            int t = thread_list[ti];
            omp_set_num_threads(t);
            for (int rep = 0; rep < reps; ++rep) {
                double t0 = omp_get_wtime();
                volatile Result R = analyze_par(pts, m, "static", 0);
                double t1 = omp_get_wtime();
                (void)R;
                std::printf("par,%s,static,0,%d,%d,%.6f\n", methods[m], t, rep, t1 - t0);
            }
        }
    }

    // 3) PORÓWNANIE SCHEDULE — metoda private (najszybsza/bez kontencji),
    //    pelna liczba watkow (16), warianty static/dynamic/guided + chunk
    omp_set_num_threads(16);
    struct SchedCfg { const char* s; int chunk; };
    SchedCfg cfgs[] = {
        {"static", 0}, {"static", 1000}, {"static", 100000},
        {"dynamic", 1000}, {"dynamic", 100000},
        {"guided", 0}
    };
    for (auto& c : cfgs) {
        for (int rep = 0; rep < reps; ++rep) {
            double t0 = omp_get_wtime();
            volatile Result R = analyze_par(pts, 2, c.s, c.chunk);
            double t1 = omp_get_wtime();
            (void)R;
            std::printf("par,private,%s,%d,16,%d,%.6f\n", c.s, c.chunk, rep, t1 - t0);
        }
    }

    std::fprintf(stderr, "Benchmark zakonczony.\n");
}

// ------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Uzycie: " << argv[0] << " <plik> [correctness|bench] [arg]\n";
        return 1;
    }
    std::string filename = argv[1];
    std::string mode = (argc >= 3) ? argv[2] : "correctness";

    std::vector<Point> pts = load_points(filename);
    if (pts.empty()) { std::cerr << "Blad: brak punktow.\n"; return 1; }

    if (mode == "bench") {
        int reps = (argc >= 4) ? std::atoi(argv[3]) : 5;
        if (reps < 1) reps = 5;
        run_bench(pts, reps);
    } else {
        int threads = (argc >= 4) ? std::atoi(argv[3]) : omp_get_max_threads();
        if (threads < 1) threads = omp_get_max_threads();
        run_correctness(pts, threads);
    }
    return 0;
}
