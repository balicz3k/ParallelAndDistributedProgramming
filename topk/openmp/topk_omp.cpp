/*
 * topk_omp.cpp - dokładne top-k w OpenMP.
 *
 * Kompilacja: g++ -O3 -std=c++17 -fopenmp -I../common topk_omp.cpp -o topk_omp
 * Użycie:     ./topk_omp <baza.bin> <zapytania.bin> <N> <D> <Q> <k> --threads T [opcje z common.h]
 *
 * Dwa sposoby podziału pracy:
 *  1) partia zapytań (Q >= liczba wątków): wątki dzielą się ZAPYTANIAMI
 *     (schedule(dynamic, 1)); każdy wątek liczy całe zapytanie sam, z własnym
 *     kopcem top-k, więc wątki nic nie współdzielą przy zapisie.
 *  2) pojedyncze zapytanie albo mała partia (Q < liczba wątków): wątki dzielą
 *     się BAZĄ (schedule(static)); każdy ma lokalny kopiec top-k swojej części,
 *     a po pętli wątek główny scala p*k kandydatów.
 * W żadnym wariancie nie ma sekcji krytycznej ani operacji atomowych.
 * Remis: wygrywa mniejszy indeks - ranking jest identyczny z referencją.
 *
 * Pomiar: partia w trybie 1) lub 2); P95 z zapytań liczonych pojedynczo
 * (zawsze tryb 2, czyli wszystkie wątki na jednym zapytaniu).
 */
#include <cmath>
#include <omp.h>
#include "common.h"

struct Cand {
    float dist;
    int idx;
};

// "a jest lepszy od b": mniejsza odległość, przy remisie mniejszy indeks
static inline bool better(const Cand &a, const Cand &b) {
    if (a.dist != b.dist) return a.dist < b.dist;
    return a.idx < b.idx;
}

static inline float sq_dist(const float *a, const float *b, int D) {
    float sum = 0.0f;
    for (int i = 0; i < D; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

// Kopiec maksymalny rozmiaru k: na szczycie najgorszy z dotychczasowych k kandydatów.
struct TopK {
    int k;
    std::vector<Cand> heap;
    explicit TopK(int k_) : k(k_) { heap.reserve(k_); }
    void clear() { heap.clear(); }
    void push(const Cand &c) {
        if ((int)heap.size() < k) {
            heap.push_back(c);
            std::push_heap(heap.begin(), heap.end(), better);
        } else if (better(c, heap.front())) {
            std::pop_heap(heap.begin(), heap.end(), better);
            heap.back() = c;
            std::push_heap(heap.begin(), heap.end(), better);
        }
    }
    // wynik posortowany od najlepszego
    void write(int *idx, float *dist2) const {
        std::vector<Cand> s = heap;
        std::sort(s.begin(), s.end(), better);
        for (size_t i = 0; i < s.size(); ++i) {
            idx[i] = s[i].idx;
            dist2[i] = s[i].dist;
        }
    }
};

// Tryb 2: jedno zapytanie, baza podzielona między wątki, potem scalanie.
static void query_split_base(const float *query, const float *base, int N, int D,
                             int threads, std::vector<TopK> &local, TopK &merged,
                             int *idx, float *dist2) {
#pragma omp parallel num_threads(threads)
    {
        TopK &my = local[omp_get_thread_num()];
        my.clear();
#pragma omp for schedule(static)
        for (int i = 0; i < N; ++i)
            my.push(Cand{sq_dist(query, base + (size_t)i * D, D), i});
    }
    merged.clear();
    for (const TopK &t : local)
        for (const Cand &c : t.heap) merged.push(c);
    merged.write(idx, dist2);
}

int main(int argc, char **argv) {
    Options o;
    if (!parse_options(argc, argv, o)) {
        print_usage(argv[0]);
        return 1;
    }
    const int N = o.N, D = o.D, Q = o.Q, k = o.k;
    const int threads = o.threads > 0 ? o.threads : omp_get_max_threads();

    std::vector<float> base, queries;
    double t0 = now_ms();
    if (!load_floats(o.base_path, (size_t)N * D, base) ||
        !load_floats(o.query_path, (size_t)Q * D, queries))
        return 1;
    double io_ms = now_ms() - t0;

    std::vector<int> idx((size_t)Q * k);
    std::vector<float> dist2((size_t)Q * k);
    std::vector<TopK> local(threads, TopK(k));
    TopK merged(k);
    const bool by_query = Q >= threads;
    std::vector<Measurement> results;

    for (int rep = -o.warmup; rep < o.repeat; ++rep) {
        Measurement m;
        m.io_ms = io_ms;

        double t_start = now_ms();
        if (by_query) {
#pragma omp parallel num_threads(threads)
            {
                TopK my(k);
#pragma omp for schedule(dynamic, 1)
                for (int q = 0; q < Q; ++q) {
                    const float *query = &queries[(size_t)q * D];
                    my.clear();
                    for (int i = 0; i < N; ++i)
                        my.push(Cand{sq_dist(query, base.data() + (size_t)i * D, D), i});
                    my.write(&idx[(size_t)q * k], &dist2[(size_t)q * k]);
                }
            }
        } else {
            for (int q = 0; q < Q; ++q)
                query_split_base(&queries[(size_t)q * D], base.data(), N, D, threads,
                                 local, merged, &idx[(size_t)q * k], &dist2[(size_t)q * k]);
        }
        m.core_ms = m.full_ms = now_ms() - t_start;

        for (int q = 0; q < o.lat_queries; ++q) {
            double tq = now_ms();
            query_split_base(&queries[(size_t)q * D], base.data(), N, D, threads,
                             local, merged, &idx[(size_t)q * k], &dist2[(size_t)q * k]);
            m.lat_us.push_back((now_ms() - tq) * 1000.0);
        }

        m.rss_mb = peak_rss_mb();
        if (rep >= 0) {
            if (o.csv) print_csv("omp", o, threads, 1, 0, rep, m);
            results.push_back(m);
        }
    }

    std::printf("wątki: %d, podział partii: %s\n", threads, by_query ? "po zapytaniach" : "po bazie");
    print_summary("omp", o, results);
    if (!o.out_path.empty()) {
        std::vector<float> dist(dist2.size());
        for (size_t i = 0; i < dist2.size(); ++i) dist[i] = std::sqrt(dist2[i]);
        write_ranking(o.out_path, Q, k, idx, dist);
    }
    return 0;
}
