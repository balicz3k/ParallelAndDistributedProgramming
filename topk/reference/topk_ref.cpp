/*
 * topk_ref.cpp — brute-force top-k nearest neighbors (referencyjna implementacja)
 *
 * Kompilacja: g++ -O2 -std=c++17 topk_ref.cpp -o topk_ref
 * Użycie:     ./topk_ref <baza.bin> <zapytania.bin> <N> <D> <Q> <k>
 *
 * Format plików wejściowych: raw float32, row-major (C order), bez nagłówka
 *   baza.bin     — N * D wartości float32
 *   zapytania.bin — Q * D wartości float32
 *
 * Algorytm: dla każdego zapytania oblicz odległość euklidesową do wszystkich
 * N punktów bazy, następnie znajdź k punktów o najmniejszej odległości.
 * Remisy rozstrzygane deterministycznie — przez mniejszy indeks.
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
// Zwracamy kwadrat, żeby uniknąć sqrt() — porównania są tak samo poprawne,
// a do wyników zapisujemy już prawdziwą odległość (sqrt na końcu).
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
// Remisy: mniejszy indeks wygrywa (deterministyczne sortowanie).
// Zwraca parę (indeks, odległość) posortowaną rosnąco po odległości.
// ----------------------------------------------------------------------------
static std::vector<std::pair<int, float>>
topk_query(const float *query, const float *base, int N, int D, int k) {
    // Zbierz wszystkie odległości (kwadrat) razem z indeksem
    std::vector<std::pair<float, int>> dists(N);
    for (int i = 0; i < N; ++i) {
        dists[i] = {sq_dist(query, base + (size_t)i * D, D), i};
    }

    // Częściowe sortowanie — tylko k najmniejszych (O(N + k log N))
    std::partial_sort(dists.begin(), dists.begin() + k, dists.end(),
                      [](const std::pair<float, int> &a,
                         const std::pair<float, int> &b) {
                          // Remis: mniejszy indeks wygrywa
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
// Percentyl p (0–100) z posortowanego wektora wartości nanosekund
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
// main
// ----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    if (argc != 7) {
        std::cerr << "Uzycie: " << argv[0]
                  << " <baza.bin> <zapytania.bin> <N> <D> <Q> <k>\n";
        return 1;
    }

    const std::string base_path  = argv[1];
    const std::string query_path = argv[2];
    const int N = std::stoi(argv[3]);  // liczba wektorów bazy
    const int D = std::stoi(argv[4]);  // liczba wymiarów
    const int Q = std::stoi(argv[5]);  // liczba zapytań
    const int k = std::stoi(argv[6]);  // liczba sąsiadów

    if (k > N) {
        std::cerr << "Blad: k (" << k << ") > N (" << N << ")\n";
        return 1;
    }

    std::cout << "Wczytywanie danych...\n";
    std::cout << "  baza:      " << base_path  << " (" << N << " x " << D << ")\n";
    std::cout << "  zapytania: " << query_path << " (" << Q << " x " << D << ")\n";
    std::cout << "  k = " << k << "\n\n";

    const std::vector<float> base   = load_bin(base_path,  (size_t)N * D);
    const std::vector<float> queries = load_bin(query_path, (size_t)Q * D);

    // Wyniki: dla każdego zapytania k par (indeks, odległość)
    std::vector<std::vector<std::pair<int, float>>> all_results(Q);
    std::vector<double> query_times_us(Q);  // czas każdego zapytania w µs

    // ---- Przetwarzanie wsadowe (batch) -----------------------------------
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int q = 0; q < Q; ++q) {
        auto t0 = std::chrono::high_resolution_clock::now();

        all_results[q] = topk_query(queries.data() + (size_t)q * D,
                                    base.data(), N, D, k);

        auto t1 = std::chrono::high_resolution_clock::now();
        query_times_us[q] =
            std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();
    // -----------------------------------------------------------------------

    double avg_us   = std::accumulate(query_times_us.begin(),
                                      query_times_us.end(), 0.0) /
                      static_cast<double>(Q);
    double p95_us   = percentile(query_times_us, 95.0);
    double qps      = (Q / total_ms) * 1000.0;

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
    std::cout << "\n=== Statystyki ===\n";
    std::printf("  Calkowity czas:        %.3f ms\n",  total_ms);
    std::printf("  Sredni czas / query:   %.3f us\n",  avg_us);
    std::printf("  P95 latency:           %.3f us\n",  p95_us);
    std::printf("  Przepustowosc:         %.1f zapytan/s\n", qps);

    // ---- Zapis do pliku results.txt ---------------------------------------
    std::ofstream out("results.txt");
    if (!out) {
        std::cerr << "Blad: nie mozna zapisac results.txt\n";
        return 1;
    }
    out << "# topk_ref wyniki\n";
    out << "# N=" << N << " D=" << D << " Q=" << Q << " k=" << k << "\n";
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
    out << "total_ms="   << total_ms << "\n";
    out << "avg_us="     << avg_us   << "\n";
    out << "p95_us="     << p95_us   << "\n";
    out << "qps="        << qps      << "\n";
    out.close();

    std::cout << "\nWyniki zapisane do results.txt\n";
    return 0;
}
