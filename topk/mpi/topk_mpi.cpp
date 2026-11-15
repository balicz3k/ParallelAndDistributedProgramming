/*
 * topk_mpi.cpp — rozproszone brute-force top-k nearest neighbors (MPI)
 *
 * Kompilacja (czysty MPI):   mpicxx -O3 -std=c++17 topk_mpi.cpp -o topk_mpi
 * Kompilacja (MPI+OpenMP):   mpicxx -O3 -std=c++17 -fopenmp topk_mpi.cpp -o topk_mpi_hybrid
 *
 * Użycie:
 *   mpirun -np <P> ./topk_mpi <baza.bin> <zapytania.bin> <N> <D> <Q> <k> [opcje]
 *
 * Opcje (flagi, kolejność dowolna):
 *   --repeat R        liczba mierzonych powtórzeń (domyślnie 1)
 *   --warmup W        liczba przebiegów rozgrzewkowych bez pomiaru (domyślnie 0)
 *   --json            wypisz blok JSON z surowymi danymi na stdout (dla harnessu)
 *   --no-write        nie zapisuj pliku z rankingiem
 *   --verify-out PATH sciezka pliku z rankingiem (domyslnie results_mpi.txt)
 *   --quiet           nie wypisuj listy wyników top-k (tylko statystyki)
 *
 * Strategia:
 *   - Rank 0: wczytuje bazę i zapytania
 *   - MPI_Bcast: rozsyła zapytania + metadane do wszystkich
 *   - MPI_Scatterv: dzieli bazę na partycje (N/P wektorów na rank, reszta N%P)
 *   - Każdy rank: brute-force top-k na swojej partycji (z globalnymi indeksami)
 *   - MPI_Gather: rank 0 zbiera lokalne top-k od wszystkich
 *   - Rank 0: merge globalny top-k ze wszystkich lokalnych wyników
 *
 * Poprawność: wyniki identyczne z topk_ref (deterministyczne remisy przez
 * mniejszy indeks; porównujemy kwadrat odległości).
 *
 * Pomiar czasu (MPI_Wtime, zsynchronizowany barierami):
 *   - t_compute  : sam rdzeń obliczeń (lokalne top-k), bez komunikacji
 *   - t_comm     : Bcast(queries) + Scatterv(base) + Gather(wyniki)
 *   - t_total_proc: compute + comm (bez I/O wczytania pliku i bez merge)
 * P95 liczony z czasów POJEDYNCZYCH zapytań zebranych ze WSZYSTKICH powtórzeń.
 * Pamięć szczytowa: getrusage(ru_maxrss) per rank, agregowana (max i suma).
 */

#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <sys/resource.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// Peak RSS tego procesu w KiB (Linux: ru_maxrss jest w KiB)
// ----------------------------------------------------------------------------
static long peak_rss_kib() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;
}

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
        std::cerr << "Blad: za krotki plik: " << path << " (oczekiwano "
                  << count << " float32)\n";
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
// Percentyl p (0-100) z wektora wartości (kopia + sort)
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

static double median(std::vector<double> vals) { return percentile(vals, 50.0); }

// ----------------------------------------------------------------------------
// Lokalne top-k na partycji bazy (z globalnymi indeksami).
// Zwraca wektor par (globalny_indeks, dist^2) rozmiaru min(k, local_N).
// ----------------------------------------------------------------------------
static std::vector<std::pair<int, float>>
local_topk(const float *query, const float *local_base,
           int local_N, int D, int k, int offset) {
    std::vector<std::pair<float, int>> dists(local_N);
    for (int i = 0; i < local_N; ++i) {
        dists[i] = {sq_dist(query, local_base + (size_t)i * D, D),
                    offset + i};
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
        result[i] = {dists[i].second, dists[i].first};
    }
    return result;
}

// ----------------------------------------------------------------------------
// Merge top-k z P*k kandydatów
// ----------------------------------------------------------------------------
static std::vector<std::pair<int, float>>
merge_topk(std::vector<std::pair<int, float>> &candidates, int k) {
    int kk = std::min<int>(k, (int)candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + kk,
                      candidates.end(),
                      [](const std::pair<int, float> &a,
                         const std::pair<int, float> &b) {
                          if (a.second != b.second) return a.second < b.second;
                          return a.first < b.first;
                      });
    return std::vector<std::pair<int, float>>(candidates.begin(),
                                              candidates.begin() + kk);
}

int main(int argc, char *argv[]) {
    int provided = 0;
#ifdef _OPENMP
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#else
    MPI_Init(&argc, &argv);
    (void)provided;
#endif

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---- Parsowanie argumentów (rank 0) -----------------------------------
    int params[4] = {0, 0, 0, 0};  // N, D, Q, k
    int opts[4]   = {1, 0, 0, 0};  // repeat, warmup, json, no_write(bit)+quiet
    int repeat = 1, warmup = 0, do_json = 0, no_write = 0, quiet = 0;
    std::string base_path, query_path, verify_out = "results_mpi.txt";

    if (rank == 0) {
        if (argc < 7) {
            std::cerr << "Uzycie: mpirun -np <P> " << argv[0]
                      << " <baza.bin> <zapytania.bin> <N> <D> <Q> <k>"
                      << " [--repeat R] [--warmup W] [--json] [--no-write] [--quiet]\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        base_path  = argv[1];
        query_path = argv[2];
        params[0]  = std::stoi(argv[3]);
        params[1]  = std::stoi(argv[4]);
        params[2]  = std::stoi(argv[5]);
        params[3]  = std::stoi(argv[6]);
        for (int a = 7; a < argc; ++a) {
            std::string s = argv[a];
            if (s == "--repeat" && a + 1 < argc) repeat = std::stoi(argv[++a]);
            else if (s == "--warmup" && a + 1 < argc) warmup = std::stoi(argv[++a]);
            else if (s == "--json") do_json = 1;
            else if (s == "--no-write") no_write = 1;
            else if (s == "--verify-out" && a + 1 < argc) verify_out = argv[++a];
            else if (s == "--quiet") quiet = 1;
            else { std::cerr << "Nieznana opcja: " << s << "\n"; MPI_Abort(MPI_COMM_WORLD, 1); }
        }
        opts[0] = repeat; opts[1] = warmup; opts[2] = do_json;
        opts[3] = (no_write ? 1 : 0) | (quiet ? 2 : 0);
    }

    MPI_Bcast(params, 4, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(opts,   4, MPI_INT, 0, MPI_COMM_WORLD);
    int N = params[0], D = params[1], Q = params[2], k = params[3];
    repeat = opts[0]; warmup = opts[1]; do_json = opts[2];
    no_write = opts[3] & 1; quiet = (opts[3] >> 1) & 1;
    if (repeat < 1) repeat = 1;

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    // ---- Podział bazy -----------------------------------------------------
    int base_per_rank = N / size;
    int remainder     = N % size;
    std::vector<int> counts(size), offsets(size);
    for (int r = 0; r < size; ++r) {
        counts[r]  = base_per_rank + (r < remainder ? 1 : 0);
        offsets[r] = r * base_per_rank + std::min(r, remainder);
    }
    int local_N   = counts[rank];
    int my_offset = offsets[rank];

    std::vector<int> send_counts(size), send_disps(size);
    for (int r = 0; r < size; ++r) {
        send_counts[r] = counts[r] * D;
        send_disps[r]  = offsets[r] * D;
    }

    // ---- Wczytanie danych (rank 0), poza pomiarem -------------------------
    std::vector<float> base_data, query_data;
    if (rank == 0) {
        base_data  = load_bin(base_path,  (size_t)N * D);
        query_data = load_bin(query_path, (size_t)Q * D);
    }
    query_data.resize((size_t)Q * D);

    std::vector<float> local_base((size_t)local_N * D);

    // Bufory wyników
    std::vector<int>   local_idx (Q * k, -1);
    std::vector<float> local_dist(Q * k, 1e30f);
    std::vector<int>   all_idx;
    std::vector<float> all_dist;
    if (rank == 0) {
        all_idx .resize((size_t)Q * k * size);
        all_dist.resize((size_t)Q * k * size);
    }

    // Akumulatory pomiarów (per powtórzenie) — tylko rank 0 raportuje
    std::vector<double> rep_compute_ms, rep_comm_ms, rep_total_ms, rep_qps;
    std::vector<double> all_query_us;          // wszystkie obserwacje per-query
    std::vector<std::vector<std::pair<int,float>>> final_results(Q);

    int total_runs = warmup + repeat;

    for (int run = 0; run < total_runs; ++run) {
        bool measured = (run >= warmup);

        // --- Komunikacja: Bcast zapytań ---
        MPI_Barrier(MPI_COMM_WORLD);
        double tc0 = MPI_Wtime();
        MPI_Bcast(query_data.data(), Q * D, MPI_FLOAT, 0, MPI_COMM_WORLD);
        // --- Komunikacja: Scatterv bazy ---
        MPI_Scatterv(rank == 0 ? base_data.data() : nullptr,
                     send_counts.data(), send_disps.data(), MPI_FLOAT,
                     local_base.data(), local_N * D, MPI_FLOAT,
                     0, MPI_COMM_WORLD);
        double tc1 = MPI_Wtime();
        double comm_scatter_ms = (tc1 - tc0) * 1000.0;

        // --- Obliczenia (rdzeń, bez komunikacji) ---
        std::vector<double> query_us(Q);
        MPI_Barrier(MPI_COMM_WORLD);
        double tk0 = MPI_Wtime();
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int q = 0; q < Q; ++q) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto res = local_topk(query_data.data() + (size_t)q * D,
                                  local_base.data(), local_N, D, k, my_offset);
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
            query_us[q] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        double tk1 = MPI_Wtime();
        double compute_ms = (tk1 - tk0) * 1000.0;

        // --- Komunikacja: Gather wyników ---
        double tg0 = MPI_Wtime();
        MPI_Gather(local_idx.data(),  Q * k, MPI_INT,
                   all_idx.data(),   Q * k, MPI_INT,   0, MPI_COMM_WORLD);
        MPI_Gather(local_dist.data(), Q * k, MPI_FLOAT,
                   all_dist.data(),  Q * k, MPI_FLOAT, 0, MPI_COMM_WORLD);
        double tg1 = MPI_Wtime();
        double comm_gather_ms = (tg1 - tg0) * 1000.0;

        // --- Merge globalny (rank 0); wykonujemy co run by mieć finalne wyniki ---
        if (rank == 0) {
            for (int q = 0; q < Q; ++q) {
                std::vector<std::pair<int, float>> cand;
                cand.reserve((size_t)size * k);
                for (int r = 0; r < size; ++r) {
                    for (int i = 0; i < k; ++i) {
                        int idx    = all_idx [(size_t)r * Q * k + q * k + i];
                        float dist = all_dist[(size_t)r * Q * k + q * k + i];
                        if (idx >= 0) cand.push_back({idx, dist});
                    }
                }
                final_results[q] = merge_topk(cand, k);
            }
        }

        // --- Agregacja czasów ---
        // compute: max po rankach (czas rdzenia = najwolniejszy rank)
        double max_compute = 0.0;
        MPI_Reduce(&compute_ms, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        double comm_ms = comm_scatter_ms + comm_gather_ms;
        double max_comm = 0.0;
        MPI_Reduce(&comm_ms, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (measured) {
            // zbierz czasy per-query ze wszystkich ranków dla P95 wielu obserwacji
            // each rank przetworzył wszystkie Q zapytań na swojej partycji;
            // opóźnienie zapytania = max po rankach (wszystkie muszą skończyć).
            std::vector<double> q_us_max(Q);
            MPI_Reduce(query_us.data(), q_us_max.data(), Q, MPI_DOUBLE,
                       MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                double total_proc_ms = max_compute + max_comm;
                rep_compute_ms.push_back(max_compute);
                rep_comm_ms.push_back(max_comm);
                rep_total_ms.push_back(total_proc_ms);
                rep_qps.push_back((Q / total_proc_ms) * 1000.0);
                for (int q = 0; q < Q; ++q) all_query_us.push_back(q_us_max[q]);
            }
        }
    }

    // ---- Pamięć szczytowa per rank -> agregacja ---------------------------
    long my_rss = peak_rss_kib();
    long rss_sum = 0, rss_max = 0;
    MPI_Reduce(&my_rss, &rss_sum, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_rss, &rss_max, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    // Zbierz też wektor RSS per rank (dla raportu JSON)
    std::vector<long> rss_all;
    if (rank == 0) rss_all.resize(size);
    MPI_Gather(&my_rss, 1, MPI_LONG, rank == 0 ? rss_all.data() : nullptr,
               1, MPI_LONG, 0, MPI_COMM_WORLD);

    // ---- Raport rank 0 ----------------------------------------------------
    if (rank == 0) {
        double med_compute = median(rep_compute_ms);
        double med_comm    = median(rep_comm_ms);
        double med_total   = median(rep_total_ms);
        double med_qps     = median(rep_qps);
        double min_total   = *std::min_element(rep_total_ms.begin(), rep_total_ms.end());
        double max_total   = *std::max_element(rep_total_ms.begin(), rep_total_ms.end());
        double p50_us      = percentile(all_query_us, 50.0);
        double p95_us      = percentile(all_query_us, 95.0);
        double p99_us      = percentile(all_query_us, 99.0);

        if (!quiet) {
            std::cout << "MPI top-k: N=" << N << " D=" << D << " Q=" << Q
                      << " k=" << k << " P=" << size
                      << " threads/rank=" << nthreads << "\n";
            std::cout << "=== Wyniki (top-" << k << " pierwsze 2 zapytania) ===\n";
            for (int q = 0; q < std::min(2, Q); ++q) {
                std::cout << "Zapytanie " << q << ":\n";
                for (int i = 0; i < (int)final_results[q].size(); ++i) {
                    float true_dist = std::sqrt(final_results[q][i].second);
                    std::cout << "  #" << (i + 1)
                              << "  indeks=" << final_results[q][i].first
                              << "  dist="   << true_dist << "\n";
                }
            }
        }

        std::cout << "\n=== Statystyki (lokalne, 1 maszyna) ===\n";
        std::printf("  Procesy MPI:              %d\n", size);
        std::printf("  Watki/rank (OpenMP):      %d\n", nthreads);
        std::printf("  Powtorzenia (+warmup):    %d (+%d)\n", repeat, warmup);
        std::printf("  Obserwacje zapytan (P95): %zu\n", all_query_us.size());
        std::printf("  Czas obliczen  (mediana): %.3f ms\n", med_compute);
        std::printf("  Czas komunikacji (med.):  %.3f ms\n", med_comm);
        std::printf("  Czas total proc (med.):   %.3f ms  [min %.3f / max %.3f]\n",
                    med_total, min_total, max_total);
        std::printf("  Przepustowosc (med.):     %.1f zapytan/s\n", med_qps);
        std::printf("  P50 / P95 / P99 latency:  %.1f / %.1f / %.1f us\n",
                    p50_us, p95_us, p99_us);
        std::printf("  Peak RSS max/rank:        %.1f MiB\n", rss_max / 1024.0);
        std::printf("  Peak RSS suma (wsz.rank): %.1f MiB\n", rss_sum / 1024.0);

        if (do_json) {
            std::printf("JSON {");
            std::printf("\"N\":%d,\"D\":%d,\"Q\":%d,\"k\":%d,\"P\":%d,\"threads\":%d,",
                        N, D, Q, k, size, nthreads);
            std::printf("\"repeat\":%d,\"warmup\":%d,", repeat, warmup);
            std::printf("\"compute_ms_med\":%.4f,\"comm_ms_med\":%.4f,\"total_ms_med\":%.4f,",
                        med_compute, med_comm, med_total);
            std::printf("\"total_ms_min\":%.4f,\"total_ms_max\":%.4f,", min_total, max_total);
            std::printf("\"qps_med\":%.3f,", med_qps);
            std::printf("\"p50_us\":%.3f,\"p95_us\":%.3f,\"p99_us\":%.3f,",
                        p50_us, p95_us, p99_us);
            std::printf("\"nquery_obs\":%zu,", all_query_us.size());
            std::printf("\"rss_max_mib\":%.3f,\"rss_sum_mib\":%.3f,",
                        rss_max / 1024.0, rss_sum / 1024.0);
            std::printf("\"rss_per_rank_mib\":[");
            for (int r = 0; r < size; ++r)
                std::printf("%s%.3f", r ? "," : "", rss_all[r] / 1024.0);
            std::printf("],");
            // surowe czasy total per powtórzenie
            std::printf("\"total_ms_runs\":[");
            for (size_t i = 0; i < rep_total_ms.size(); ++i)
                std::printf("%s%.4f", i ? "," : "", rep_total_ms[i]);
            std::printf("]}\n");
        }

        if (!no_write) {
            std::ofstream out(verify_out);
            if (out) {
                out << "# topk_mpi wyniki (ranking, do weryfikacji z referencja)\n";
                out << "# N=" << N << " D=" << D << " Q=" << Q
                    << " k=" << k << " P=" << size << "\n";
                out << "# format: query_id  rank  index  distance\n";
                out << std::fixed;
                for (int q = 0; q < Q; ++q) {
                    for (int i = 0; i < (int)final_results[q].size(); ++i) {
                        float true_dist = std::sqrt(final_results[q][i].second);
                        out << q << "  " << (i + 1) << "  "
                            << final_results[q][i].first << "  "
                            << true_dist << "\n";
                    }
                }
                out.close();
            }
        }
    }

    MPI_Finalize();
    return 0;
}
