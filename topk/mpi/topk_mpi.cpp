/*
 * topk_mpi.cpp — rozproszone brute-force top-k nearest neighbors (MPI)
 *
 * Kompilacja: mpicxx -O3 -std=c++17 topk_mpi.cpp -o topk_mpi
 * Użycie:     mpirun -np <P> ./topk_mpi <baza.bin> <zapytania.bin> <N> <D> <Q> <k>
 *
 * Strategia:
 *   - Rank 0: wczytuje bazę i zapytania
 *   - MPI_Bcast: rozsyła zapytania + metadane do wszystkich
 *   - MPI_Scatter: dzieli bazę na P równych partycji (N/P wektorów na rank)
 *   - Każdy rank: brute-force top-k na swojej partycji (z globalnymi indeksami)
 *   - MPI_Gather: rank 0 zbiera lokalne top-k od wszystkich
 *   - Rank 0: merge globalny top-k ze wszystkich lokalnych wyników
 *
 * Poprawność: wyniki identyczne z topk_ref (deterministyczne remisy przez mniejszy indeks)
 */

#include <mpi.h>
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
// Wczytaj plik binarny float32
// ----------------------------------------------------------------------------
static std::vector<float> load_bin(const std::string &path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Blad: nie mozna otworzyc: " << path << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<float> data(count);
    f.read(reinterpret_cast<char *>(data.data()),
           static_cast<std::streamsize>(count * sizeof(float)));
    if (!f) {
        std::cerr << "Blad: za krotki plik: " << path << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return data;
}

// ----------------------------------------------------------------------------
// Odległość euklidesowa (kwadrat)
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
// Percentyl p (0-100) z wektora wartości
// ----------------------------------------------------------------------------
static double percentile(std::vector<double> vals, double p) {
    if (vals.empty()) return 0.0;
    std::sort(vals.begin(), vals.end());
    double idx = (p / 100.0) * (static_cast<double>(vals.size()) - 1.0);
    size_t lo = static_cast<size_t>(idx);
    size_t hi = lo + 1 < vals.size() ? lo + 1 : lo;
    double frac = idx - static_cast<double>(lo);
    return vals[lo] * (1.0 - frac) + vals[hi] * frac;
}

// ----------------------------------------------------------------------------
// Lokalne top-k na partycji bazy (z globalnymi indeksami)
// local_base  — wskaźnik do lokalnej partycji (local_N * D floatów)
// offset      — globalny offset pierwszego wektora w tej partycji
// Zwraca wektor par (globalny_indeks, dystans^2) rozmiaru k
// ----------------------------------------------------------------------------
static std::vector<std::pair<int, float>>
local_topk(const float *query, const float *local_base,
           int local_N, int D, int k, int offset) {
    std::vector<std::pair<float, int>> dists(local_N);
    for (int i = 0; i < local_N; ++i) {
        dists[i] = {sq_dist(query, local_base + (size_t)i * D, D),
                    offset + i};  // globalny indeks
    }
    int actual_k = std::min(k, local_N);
    std::partial_sort(dists.begin(), dists.begin() + actual_k, dists.end(),
                      [](const std::pair<float, int> &a,
                         const std::pair<float, int> &b) {
                          if (a.first != b.first) return a.first < b.first;
                          return a.second < b.second;
                      });
    std::vector<std::pair<int, float>> result(actual_k);
    for (int i = 0; i < actual_k; ++i) {
        result[i] = {dists[i].second, dists[i].first};  // (indeks, dist^2)
    }
    return result;
}

// ----------------------------------------------------------------------------
// Merge top-k z P * k kandydatów — wybiera globalne top-k
// candidates: wektor par (indeks, dist^2), rozmiar P*k
// ----------------------------------------------------------------------------
static std::vector<std::pair<int, float>>
merge_topk(std::vector<std::pair<int, float>> &candidates, int k) {
    std::partial_sort(candidates.begin(),
                      candidates.begin() + k,
                      candidates.end(),
                      [](const std::pair<int, float> &a,
                         const std::pair<int, float> &b) {
                          if (a.second != b.second) return a.second < b.second;
                          return a.first < b.first;
                      });
    return std::vector<std::pair<int, float>>(
        candidates.begin(), candidates.begin() + k);
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Czas całkowity (od MPI_Init, barier do Finalize)
    double t_total_start = MPI_Wtime();

    // ---- Metadane (rank 0 parsuje, potem broadcastuje) --------------------
    int params[4] = {0, 0, 0, 0};  // N, D, Q, k
    std::string base_path, query_path;

    if (rank == 0) {
        if (argc != 7) {
            std::cerr << "Uzycie: mpirun -np <P> " << argv[0]
                      << " <baza.bin> <zapytania.bin> <N> <D> <Q> <k>\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        base_path  = argv[1];
        query_path = argv[2];
        params[0]  = std::stoi(argv[3]);
        params[1]  = std::stoi(argv[4]);
        params[2]  = std::stoi(argv[5]);
        params[3]  = std::stoi(argv[6]);
    }

    // Broadcast parametrów N, D, Q, k
    MPI_Bcast(params, 4, MPI_INT, 0, MPI_COMM_WORLD);
    int N = params[0], D = params[1], Q = params[2], k = params[3];

    // ---- Podział bazy na partycje -----------------------------------------
    // Każdy rank dostaje base_local_N wektorów; ostatni rank dostaje resztę
    int base_per_rank = N / size;
    int remainder     = N % size;

    // Liczby wektorów dla każdego ranka i offsety (globalne indeksy)
    std::vector<int> counts(size), offsets(size);
    for (int r = 0; r < size; ++r) {
        counts[r]  = base_per_rank + (r < remainder ? 1 : 0);
        offsets[r] = r * base_per_rank + std::min(r, remainder);
    }
    int local_N    = counts[rank];
    int my_offset  = offsets[rank];

    // Przelicz na liczby floatów (dla MPI)
    std::vector<int> send_counts(size), send_disps(size);
    for (int r = 0; r < size; ++r) {
        send_counts[r] = counts[r] * D;
        send_disps[r]  = offsets[r] * D;
    }

    // ---- Wczytywanie danych przez rank 0 ----------------------------------
    std::vector<float> base_data, query_data;
    if (rank == 0) {
        base_data  = load_bin(base_path,  (size_t)N * D);
        query_data = load_bin(query_path, (size_t)Q * D);
        std::cout << "MPI top-k: N=" << N << " D=" << D
                  << " Q=" << Q << " k=" << k
                  << " P=" << size << "\n";
    }

    // ---- Broadcast zapytań ------------------------------------------------
    query_data.resize((size_t)Q * D);
    MPI_Bcast(query_data.data(), Q * D, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // ---- Scatter partycji bazy --------------------------------------------
    std::vector<float> local_base((size_t)local_N * D);
    MPI_Scatterv(rank == 0 ? base_data.data() : nullptr,
                 send_counts.data(), send_disps.data(), MPI_FLOAT,
                 local_base.data(), local_N * D, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    // Rank 0 może zwolnić pamięć bazy (już rozsiana)
    if (rank == 0) {
        base_data.clear();
        base_data.shrink_to_fit();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_compute_start = MPI_Wtime();

    // ---- Lokalne obliczenia top-k -----------------------------------------
    // Format wyników dla każdego zapytania: k par (indeks, dist^2)
    // Zbieramy flat: dla zapytania q, kandy z ranka r → pary (idx, dist2)
    // local_results[q * k + i] = {indeks, dist2}
    std::vector<int>   local_idx (Q * k, -1);
    std::vector<float> local_dist(Q * k, 0.0f);
    std::vector<double> query_times_us(Q);

    for (int q = 0; q < Q; ++q) {
        auto t0 = std::chrono::high_resolution_clock::now();

        auto res = local_topk(query_data.data() + (size_t)q * D,
                              local_base.data(), local_N, D, k, my_offset);
        // res może mieć mniej niż k elementów jeśli local_N < k — padujemy
        for (int i = 0; i < k; ++i) {
            if (i < (int)res.size()) {
                local_idx [q * k + i] = res[i].first;
                local_dist[q * k + i] = res[i].second;
            } else {
                local_idx [q * k + i] = -1;
                local_dist[q * k + i] = 1e30f;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        query_times_us[q] =
            std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    double t_compute_end = MPI_Wtime();
    double local_compute_ms = (t_compute_end - t_compute_start) * 1000.0;

    // ---- Gather wyników do rank 0 -----------------------------------------
    // Rank 0 zbiera od każdego ranka: Q*k indeksów + Q*k dystansów
    std::vector<int>   all_idx;
    std::vector<float> all_dist;
    if (rank == 0) {
        all_idx .resize((size_t)Q * k * size);
        all_dist.resize((size_t)Q * k * size);
    }

    MPI_Gather(local_idx.data(),  Q * k, MPI_INT,
               all_idx.data(),   Q * k, MPI_INT,   0, MPI_COMM_WORLD);
    MPI_Gather(local_dist.data(), Q * k, MPI_FLOAT,
               all_dist.data(),  Q * k, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double t_total_end = MPI_Wtime();
    double total_ms    = (t_total_end - t_total_start) * 1000.0;

    // ---- Rank 0: merge i wyniki -------------------------------------------
    if (rank == 0) {
        double t_merge_start = MPI_Wtime();

        // Zbierz czasy zapytań ze wszystkich ranków (do P95)
        // Używamy lokalnych czasów rank 0 jako przybliżenia
        double avg_us = std::accumulate(query_times_us.begin(),
                                        query_times_us.end(), 0.0) /
                        static_cast<double>(Q);
        double p95_us = percentile(query_times_us, 95.0);

        std::vector<std::vector<std::pair<int, float>>> final_results(Q);

        for (int q = 0; q < Q; ++q) {
            // Zbierz kandydatów od wszystkich ranków dla tego zapytania
            std::vector<std::pair<int, float>> candidates;
            candidates.reserve((size_t)size * k);
            for (int r = 0; r < size; ++r) {
                for (int i = 0; i < k; ++i) {
                    int idx    = all_idx [(size_t)r * Q * k + q * k + i];
                    float dist = all_dist[(size_t)r * Q * k + q * k + i];
                    if (idx >= 0) {
                        candidates.push_back({idx, dist});
                    }
                }
            }
            final_results[q] = merge_topk(candidates, k);
        }

        double t_merge_end = MPI_Wtime();
        double merge_ms = (t_merge_end - t_merge_start) * 1000.0;

        double qps = (Q / total_ms) * 1000.0;

        // ---- Wyniki na stdout ----------------------------------------------
        std::cout << "=== Wyniki ===\n";
        for (int q = 0; q < Q; ++q) {
            std::cout << "Zapytanie " << q << ":\n";
            for (int i = 0; i < k; ++i) {
                float true_dist = std::sqrt(final_results[q][i].second);
                std::cout << "  #" << (i + 1)
                          << "  indeks=" << final_results[q][i].first
                          << "  dist="   << true_dist << "\n";
            }
        }

        // ---- Statystyki ----------------------------------------------------
        std::cout << "\n=== Statystyki ===\n";
        std::printf("  Procesy MPI:           %d\n",   size);
        std::printf("  Calkowity czas:        %.3f ms\n", total_ms);
        std::printf("  Czas obliczen (rank0): %.3f ms\n", local_compute_ms);
        std::printf("  Czas merge:            %.3f ms\n", merge_ms);
        std::printf("  Sredni czas / query:   %.3f us\n", avg_us);
        std::printf("  P95 latency:           %.3f us\n", p95_us);
        std::printf("  Przepustowosc:         %.1f zapytan/s\n", qps);

        // ---- Zapis do pliku ------------------------------------------------
        std::ofstream out("results_mpi.txt");
        if (!out) {
            std::cerr << "Blad: nie mozna zapisac results_mpi.txt\n";
        } else {
            out << "# topk_mpi wyniki\n";
            out << "# N=" << N << " D=" << D << " Q=" << Q
                << " k=" << k << " P=" << size << "\n";
            out << "# format: query_id  rank  index  distance\n";
            for (int q = 0; q < Q; ++q) {
                for (int i = 0; i < k; ++i) {
                    float true_dist = std::sqrt(final_results[q][i].second);
                    out << q << "  " << (i + 1) << "  "
                        << final_results[q][i].first << "  "
                        << true_dist << "\n";
                }
            }
            out << "\n# Statystyki\n";
            out << std::fixed;
            out << "processes="   << size            << "\n";
            out << "total_ms="    << total_ms        << "\n";
            out << "compute_ms="  << local_compute_ms << "\n";
            out << "merge_ms="    << merge_ms        << "\n";
            out << "avg_us="      << avg_us          << "\n";
            out << "p95_us="      << p95_us          << "\n";
            out << "qps="         << qps             << "\n";
            out.close();
            std::cout << "\nWyniki zapisane do results_mpi.txt\n";
        }
    }

    MPI_Finalize();
    return 0;
}
