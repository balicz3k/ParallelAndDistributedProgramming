/*
 * topk_omp.cpp — brute-force top-k nearest neighbors (wersja OpenMP)
 *
 * Kompilacja: g++ -O3 -std=c++17 -fopenmp topk_omp.cpp -o topk_omp
 * Użycie:     ./topk_omp <baza.bin> <zapytania.bin> <N> <D> <Q> <k> [num_threads]
 *
 * Format plików wejściowych: raw float32, row-major (C order), bez nagłówka
 *   baza.bin     — N * D wartości float32
 *   zapytania.bin — Q * D wartości float32
 *
 * Algorytm: dla każdego zapytania oblicz odległość euklidesową do wszystkich
 * N punktów bazy równolegle (OpenMP), następnie znajdź k punktów o najmniejszej
 * odległości. Remisy rozstrzygane deterministycznie — przez mniejszy indeks.
 * Wyniki są identyczne z topk_ref.
 *
 * Strategia równoległości:
 *   - Pętla po N wektorach bazy jest zrównoleglona (#pragma omp parallel for)
 *   - Wątki zapisują wyniki do wspólnej tablicy dists[N] (każdy wątek do
 *     odrębnego zakresu indeksów — brak wyścigu danych, brak critical section)
 *   - Po obliczeniu wszystkich odległości sekwencyjny partial_sort wyznacza top-k
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <omp.h>

// ----------------------------------------------------------------------------
// Wczytaj plik binarny float32 do wektora (N * D elementów)
// ----------------------------------------------------------------------------
static std::vector<float> load_bin(const std::string &path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Blad: nie mozna otworzyc pliku: " << path << "\n";
        std::exit(1);
    }
    std::vector<float> data(count);
    f.read(reinterpret_cast<char *>(data.data()),
           static_cast<std::streamsize>(count * sizeof(float)));
    if (!f) {
        std::cerr << "Blad: za krotki plik: " << path << "\n";
        std::exit(1);
    }
    return data;
}

// ----------------------------------------------------------------------------
// Odległość euklidesowa (kwadrat) między wektorem a[] i b[] długości D.
// Zwracamy kwadrat, żeby uniknąć sqrt() — porównania są tak samo poprawne.
// ----------------------------------------------------------------------------
static inline float sq_dist(const float *a, const float *b, int D) {
    float sum = 0.0f;
    for (int i = 0; i < D; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// ----------------------------------------------------------------------------
// Znajdź top-k indeksów (0-based) o najmniejszej odległości euklidesowej.
// Wersja OpenMP: pętla po N wektorach zrównoleglona.
// Każdy wątek oblicza swój fragment, na końcu scalamy przez partial_sort.
// Remisy: mniejszy indeks wygrywa (deterministyczne — identyczne z ref).
// ----------------------------------------------------------------------------
static std::vector<std::pair<int, float>>
topk_query_omp(const float *query, const float *base, int N, int D, int k,
               int num_threads) {
    // Zbierz wszystkie odległości (kwadrat) razem z indeksem
    // Zrównolegel obliczanie odległości — każdy wątek liczy swój fragment
    std::vector<std::pair<float, int>> dists(N);

#pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads)
    for (int i = 0; i < N; ++i) {
        dists[i] = {sq_dist(query, base + (size_t)i * D, D), i};
    }

    // Częściowe sortowanie — tylko k najmniejszych (O(N + k log N))
    // To samo co referencja — gwarantuje identyczne wyniki
    std::partial_sort(dists.begin(), dists.begin() + k, dists.end(),
                      [](const std::pair<float, int> &a,
                         const std::pair<float, int> &b) {
                          if (a.first != b.first) return a.first < b.first;
                          return a.second < b.second;
                      });

    std::vector<std::pair<int, float>> result(k);
    for (int i = 0; i < k; ++i) {
        result[i] = {dists[i].second, std::sqrt(dists[i].first)};
    }
    return result;
}

// ----------------------------------------------------------------------------
// Percentyl p (0–100) z posortowanego wektora wartości mikrosekund
// ----------------------------------------------------------------------------
static double percentile(std::vector<double> &vals, double p) {
    if (vals.empty()) return 0.0;
    std::sort(vals.begin(), vals.end());
    double idx = (p / 100.0) * (static_cast<double>(vals.size()) - 1.0);
    size_t lo = static_cast<size_t>(idx);
    size_t hi = lo + 1 < vals.size() ? lo + 1 : lo;
    double frac = idx - static_cast<double>(lo);
    return vals[lo] * (1.0 - frac) + vals[hi] * frac;
}

// ----------------------------------------------------------------------------
// Weryfikacja poprawności: porównaj wyniki OMP z referencją (plik results.txt)
// Zwraca true jeśli wyniki są identyczne, false jeśli różnią się.
// ----------------------------------------------------------------------------
static bool verify_results(
    const std::vector<std::vector<std::pair<int, float>>> &omp_results,
    const std::string &ref_path, int Q, int k)
{
    std::ifstream ref(ref_path);
    if (!ref) {
        std::cerr << "[VERIFY] Brak pliku referencyjnego: " << ref_path
                  << " — pomijam weryfikację.\n";
        return true;
    }

    // Wczytaj wyniki ref: format "query_id rank index distance"
    std::vector<std::vector<std::pair<int,float>>> ref_results(Q,
        std::vector<std::pair<int,float>>(k));

    std::string line;
    while (std::getline(ref, line)) {
        if (line.empty() || line[0] == '#') continue;
        // Pomijamy sekcję statystyk (linie z '=')
        if (line.find('=') != std::string::npos && line[0] != ' ') continue;
        int qid, rank, idx;
        float dist;
        if (std::sscanf(line.c_str(), "%d %d %d %f", &qid, &rank, &idx, &dist) == 4) {
            if (qid < Q && rank - 1 < k) {
                ref_results[qid][rank - 1] = {idx, dist};
            }
        }
    }

    int mismatches = 0;
    for (int q = 0; q < Q; ++q) {
        for (int i = 0; i < k; ++i) {
            if (omp_results[q][i].first != ref_results[q][i].first) {
                if (mismatches < 5) {
                    std::cerr << "[VERIFY] Różnica: q=" << q << " rank=" << (i+1)
                              << " omp=" << omp_results[q][i].first
                              << " ref=" << ref_results[q][i].first << "\n";
                }
                ++mismatches;
            }
        }
    }

    if (mismatches == 0) {
        std::cout << "[VERIFY] OK — wyniki identyczne z referencją.\n";
        return true;
    } else {
        std::cerr << "[VERIFY] BLAD — " << mismatches << " niezgodności!\n";
        return false;
    }
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    if (argc < 7 || argc > 9) {
        std::cerr << "Uzycie: " << argv[0]
                  << " <baza.bin> <zapytania.bin> <N> <D> <Q> <k>"
                  << " [num_threads] [ref_results.txt]\n";
        return 1;
    }

    const std::string base_path  = argv[1];
    const std::string query_path = argv[2];
    const int N = std::stoi(argv[3]);
    const int D = std::stoi(argv[4]);
    const int Q = std::stoi(argv[5]);
    const int k = std::stoi(argv[6]);

    // Liczba wątków: argument programu > OMP_NUM_THREADS > omp_get_max_threads()
    int num_threads = omp_get_max_threads();
    if (argc >= 8) {
        num_threads = std::stoi(argv[7]);
    }
    // OMP_NUM_THREADS z środowiska ma priorytet jeśli nie podano argumentu
    const char *env_threads = std::getenv("OMP_NUM_THREADS");
    if (argc < 8 && env_threads) {
        num_threads = std::atoi(env_threads);
    }

    std::string ref_path = "";
    if (argc == 9) {
        ref_path = argv[8];
    }

    if (k > N) {
        std::cerr << "Blad: k (" << k << ") > N (" << N << ")\n";
        return 1;
    }

    std::cout << "Wczytywanie danych...\n";
    std::cout << "  baza:      " << base_path  << " (" << N << " x " << D << ")\n";
    std::cout << "  zapytania: " << query_path << " (" << Q << " x " << D << ")\n";
    std::cout << "  k = " << k << "\n";
    std::cout << "  watkow = " << num_threads << "\n\n";

    const std::vector<float> base    = load_bin(base_path,  (size_t)N * D);
    const std::vector<float> queries = load_bin(query_path, (size_t)Q * D);

    std::vector<std::vector<std::pair<int, float>>> all_results(Q);
    std::vector<double> query_times_us(Q);

    // ---- Przetwarzanie wsadowe -------------------------------------------
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int q = 0; q < Q; ++q) {
        auto t0 = std::chrono::high_resolution_clock::now();

        all_results[q] = topk_query_omp(queries.data() + (size_t)q * D,
                                        base.data(), N, D, k, num_threads);

        auto t1 = std::chrono::high_resolution_clock::now();
        query_times_us[q] =
            std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();
    // -----------------------------------------------------------------------

    double avg_us = std::accumulate(query_times_us.begin(),
                                    query_times_us.end(), 0.0) /
                    static_cast<double>(Q);
    double p95_us = percentile(query_times_us, 95.0);
    double qps    = (Q / total_ms) * 1000.0;

    // ---- Wyniki na stdout -------------------------------------------------
    std::cout << "=== Wyniki ===\n";
    for (int q = 0; q < Q; ++q) {
        std::cout << "Zapytanie " << q << ":\n";
        for (int i = 0; i < k; ++i) {
            std::cout << "  #" << (i + 1)
                      << "  indeks=" << all_results[q][i].first
                      << "  dist="   << all_results[q][i].second << "\n";
        }
    }

    // ---- Statystyki -------------------------------------------------------
    std::cout << "\n=== Statystyki (threads=" << num_threads << ") ===\n";
    std::printf("  Calkowity czas:        %.3f ms\n",  total_ms);
    std::printf("  Sredni czas / query:   %.3f us\n",  avg_us);
    std::printf("  P95 latency:           %.3f us\n",  p95_us);
    std::printf("  Przepustowosc:         %.1f zapytan/s\n", qps);

    // ---- Weryfikacja poprawności ------------------------------------------
    if (!ref_path.empty()) {
        verify_results(all_results, ref_path, Q, k);
    }

    // ---- Zapis do pliku results_omp.txt ------------------------------------
    std::string out_name = "results_omp_t" + std::to_string(num_threads) + ".txt";
    std::ofstream out(out_name);
    if (!out) {
        std::cerr << "Blad: nie mozna zapisac " << out_name << "\n";
        return 1;
    }
    out << "# topk_omp wyniki\n";
    out << "# N=" << N << " D=" << D << " Q=" << Q << " k=" << k
        << " threads=" << num_threads << "\n";
    out << "# format: query_id  rank  index  distance\n";
    for (int q = 0; q < Q; ++q) {
        for (int i = 0; i < k; ++i) {
            out << q << "  " << (i + 1) << "  "
                << all_results[q][i].first << "  "
                << all_results[q][i].second << "\n";
        }
    }
    out << "\n# Statystyki\n";
    out << std::fixed;
    out << "threads="   << num_threads << "\n";
    out << "total_ms="  << total_ms    << "\n";
    out << "avg_us="    << avg_us      << "\n";
    out << "p95_us="    << p95_us      << "\n";
    out << "qps="       << qps         << "\n";
    out.close();

    std::cout << "\nWyniki zapisane do " << out_name << "\n";
    return 0;
}
