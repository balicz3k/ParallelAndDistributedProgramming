/*
 * topk_mpi.cpp - dokładne top-k w MPI (baza podzielona między procesy).
 *
 * Kompilacja: mpicxx -O3 -std=c++17 -I../common topk_mpi.cpp -o topk_mpi
 *             (z -fopenmp powstaje wariant hybrydowy topk_mpi_hybrid)
 * Użycie:     mpirun -np P ./topk_mpi <baza.bin> <zapytania.bin> <N> <D> <Q> <k> [opcje z common.h]
 *
 * Przebieg:
 *  1. Proces 0 wczytuje bazę i zapytania.
 *  2. MPI_Scatterv dzieli bazę RAZ: proces r dostaje N/P wektorów, a pierwsze
 *     N % P procesów o jeden więcej (counts/displs w wektorach float).
 *  3. Dla każdej partii: MPI_Bcast zapytań, każdy proces liczy lokalne top-k
 *     na swojej części bazy (z globalnymi indeksami), MPI_Gather zbiera
 *     P*k kandydatów na zapytanie w procesie 0, który scala je w wynik.
 *     Przesyłane są tylko lokalne top-k (Q*k par na proces), nie odległości.
 * Remis: wygrywa mniejszy indeks - ranking identyczny z referencją.
 *
 * Czasy (MPI_Wtime, proces 0):
 *  base_ms - jednorazowe rozesłanie bazy (Scatterv),
 *  core_ms - same obliczenia lokalne (maksimum po procesach),
 *  full_ms - Bcast zapytań + obliczenia + Gather + scalanie.
 * P95: zapytania obsługiwane pojedynczo (Bcast jednego zapytania, lokalne
 * top-k, Gather, scalanie). Pamięć: suma szczytowego RSS wszystkich procesów.
 * W wariancie hybrydowym pętla po zapytaniach partii jest dzielona między wątki.
 */
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cmath>
#include <utility>
#include "common.h"

typedef std::pair<float, int> DistIdx;  // (kwadrat odległości, indeks) - porządek leksykograficzny

static float sq_dist(const float *a, const float *b, int D) {
    float sum = 0.0f;
    for (int i = 0; i < D; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Lokalne top-k jednego zapytania na części bazy; brakujące miejsca (gdy
// część ma mniej niż k wektorów) wypełnia indeks -1.
static void local_topk(const float *query, const float *part, int count, int offset,
                       int D, int k, int *idx, float *dist2) {
    std::vector<DistIdx> work(count);
    for (int i = 0; i < count; ++i)
        work[i] = {sq_dist(query, part + (size_t)i * D, D), offset + i};
    int kk = std::min(k, count);
    std::partial_sort(work.begin(), work.begin() + kk, work.end());
    for (int i = 0; i < k; ++i) {
        idx[i] = i < kk ? work[i].second : -1;
        dist2[i] = i < kk ? work[i].first : INFINITY;
    }
}

// Proces 0: z P*k kandydatów (cand_idx/cand_d2 w układzie [proces][zapytanie][k])
// wybiera k najlepszych dla zapytania q.
static void merge_topk(const std::vector<int> &cand_idx, const std::vector<float> &cand_d2,
                       int P, int nq, int q, int k, int *idx, float *dist2) {
    std::vector<DistIdx> c;
    for (int r = 0; r < P; ++r)
        for (int i = 0; i < k; ++i) {
            size_t pos = ((size_t)r * nq + q) * k + i;
            if (cand_idx[pos] >= 0) c.push_back({cand_d2[pos], cand_idx[pos]});
        }
    std::partial_sort(c.begin(), c.begin() + k, c.end());
    for (int i = 0; i < k; ++i) {
        idx[i] = c[i].second;
        dist2[i] = c[i].first;
    }
}

int main(int argc, char **argv) {
#ifdef _OPENMP
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    const int threads = omp_get_max_threads();
    const char *version = "mpi_omp";
#else
    MPI_Init(&argc, &argv);
    const int threads = 1;
    const char *version = "mpi";
#endif
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    // argumenty są dostępne we wszystkich procesach, więc każdy parsuje je sam
    Options o;
    if (!parse_options(argc, argv, o)) {
        if (rank == 0) print_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }
    const int N = o.N, D = o.D, Q = o.Q, k = o.k;

    // podział bazy: liczba wektorów i pierwszy wektor każdego procesu
    std::vector<int> counts(P), first(P), send_counts(P), send_displs(P);
    for (int r = 0, pos = 0; r < P; ++r) {
        counts[r] = N / P + (r < N % P ? 1 : 0);
        first[r] = pos;
        pos += counts[r];
        send_counts[r] = counts[r] * D;
        send_displs[r] = first[r] * D;
    }
    const int my_count = counts[rank];

    // 1. wczytanie danych w procesie 0
    std::vector<float> base, queries((size_t)Q * D);
    int ok = 1;
    double t0 = MPI_Wtime();
    if (rank == 0)
        ok = load_floats(o.base_path, (size_t)N * D, base) &&
             load_floats(o.query_path, (size_t)Q * D, queries);
    double io_ms = (MPI_Wtime() - t0) * 1000.0;
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok) {
        MPI_Finalize();
        return 1;
    }

    // 2. jednorazowy podział bazy
    std::vector<float> part((size_t)my_count * D);
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    MPI_Scatterv(base.data(), send_counts.data(), send_displs.data(), MPI_FLOAT,
                 part.data(), my_count * D, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double base_ms = (MPI_Wtime() - t0) * 1000.0;

    // bufory: lokalne top-k (Q*k), kandydaci w procesie 0 (P*Q*k), wynik (Q*k)
    std::vector<int> loc_idx((size_t)Q * k), cand_idx, idx((size_t)Q * k);
    std::vector<float> loc_d2((size_t)Q * k), cand_d2, dist2((size_t)Q * k);
    if (rank == 0) {
        cand_idx.resize((size_t)P * Q * k);
        cand_d2.resize((size_t)P * Q * k);
    }
    std::vector<float> one_query(D);
    std::vector<Measurement> results;

    for (int rep = -o.warmup; rep < o.repeat; ++rep) {
        Measurement m;
        m.io_ms = io_ms;
        m.base_ms = base_ms;

        // 3. partia zapytań
        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();
        MPI_Bcast(queries.data(), Q * D, MPI_FLOAT, 0, MPI_COMM_WORLD);
        double tc0 = MPI_Wtime();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int q = 0; q < Q; ++q)
            local_topk(&queries[(size_t)q * D], part.data(), my_count, first[rank], D, k,
                       &loc_idx[(size_t)q * k], &loc_d2[(size_t)q * k]);
        double my_core_ms = (MPI_Wtime() - tc0) * 1000.0;
        MPI_Gather(loc_idx.data(), Q * k, MPI_INT, cand_idx.data(), Q * k, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(loc_d2.data(), Q * k, MPI_FLOAT, cand_d2.data(), Q * k, MPI_FLOAT, 0, MPI_COMM_WORLD);
        if (rank == 0)
            for (int q = 0; q < Q; ++q)
                merge_topk(cand_idx, cand_d2, P, Q, q, k, &idx[(size_t)q * k], &dist2[(size_t)q * k]);
        m.full_ms = (MPI_Wtime() - t_start) * 1000.0;
        MPI_Reduce(&my_core_ms, &m.core_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // zapytania pojedynczo: Bcast jednego zapytania, lokalne top-k, Gather, scalenie
        for (int q = 0; q < o.lat_queries; ++q) {
            double tq = MPI_Wtime();
            if (rank == 0) std::copy_n(&queries[(size_t)q * D], D, one_query.begin());
            MPI_Bcast(one_query.data(), D, MPI_FLOAT, 0, MPI_COMM_WORLD);
            local_topk(one_query.data(), part.data(), my_count, first[rank], D, k,
                       loc_idx.data(), loc_d2.data());
            MPI_Gather(loc_idx.data(), k, MPI_INT, cand_idx.data(), k, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Gather(loc_d2.data(), k, MPI_FLOAT, cand_d2.data(), k, MPI_FLOAT, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                merge_topk(cand_idx, cand_d2, P, 1, 0, k, &idx[(size_t)q * k], &dist2[(size_t)q * k]);
                m.lat_us.push_back((MPI_Wtime() - tq) * 1e6);
            }
        }

        double my_rss = peak_rss_mb();
        MPI_Reduce(&my_rss, &m.rss_mb, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0 && rep >= 0) {
            if (o.csv) print_csv(version, o, threads, P, 0, rep, m);
            results.push_back(m);
        }
    }

    if (rank == 0) {
        std::printf("procesy MPI: %d, wątki na proces: %d (pomiar lokalny, 1 komputer)\n", P, threads);
        std::printf("rozesłanie bazy (raz): %.2f ms\n", base_ms);
        print_summary(version, o, results);
        if (!o.out_path.empty()) {
            std::vector<float> dist(dist2.size());
            for (size_t i = 0; i < dist2.size(); ++i) dist[i] = std::sqrt(dist2[i]);
            write_ranking(o.out_path, Q, k, idx, dist);
        }
    }
    MPI_Finalize();
    return 0;
}
