/*
 * topk_ref.cpp - sekwencyjna wersja referencyjna dokładnego top-k.
 *
 * Kompilacja: g++ -O3 -std=c++17 -I../common topk_ref.cpp -o topk_ref
 * Użycie:     ./topk_ref <baza.bin> <zapytania.bin> <N> <D> <Q> <k> [opcje z common.h]
 *
 * Dla każdego zapytania liczony jest kwadrat odległości euklidesowej do
 * wszystkich N wektorów bazy, a k najmniejszych wybiera std::partial_sort.
 * Remis: wygrywa mniejszy indeks. Pierwiastek liczony jest dopiero dla wyniku.
 *
 * Pomiar: rozgrzewka + powtórzenia całej partii (core = full, bo nie ma
 * komunikacji), a P95 z zapytań liczonych po jednym (pierwsze --lat zapytań).
 * Ta wersja jest bazą dla przyspieszeń pozostałych wersji.
 */
#include <cmath>
#include <utility>
#include "common.h"

static float sq_dist(const float *a, const float *b, int D) {
    float sum = 0.0f;
    for (int i = 0; i < D; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// k najbliższych wektorów bazy dla jednego zapytania -> idx[k], dist2[k] (kwadraty)
static void topk_query(const float *query, const float *base, int N, int D, int k,
                       std::vector<std::pair<float, int>> &work, int *idx, float *dist2) {
    for (int i = 0; i < N; ++i)
        work[i] = {sq_dist(query, base + (size_t)i * D, D), i};
    // para (odległość, indeks) porównywana leksykograficznie => remis po indeksie
    std::partial_sort(work.begin(), work.begin() + k, work.end());
    for (int i = 0; i < k; ++i) {
        dist2[i] = work[i].first;
        idx[i] = work[i].second;
    }
}

int main(int argc, char **argv) {
    Options o;
    if (!parse_options(argc, argv, o)) {
        print_usage(argv[0]);
        return 1;
    }
    const int N = o.N, D = o.D, Q = o.Q, k = o.k;

    std::vector<float> base, queries;
    double t0 = now_ms();
    if (!load_floats(o.base_path, (size_t)N * D, base) ||
        !load_floats(o.query_path, (size_t)Q * D, queries))
        return 1;
    double io_ms = now_ms() - t0;

    std::vector<int> idx((size_t)Q * k);
    std::vector<float> dist2((size_t)Q * k);
    std::vector<std::pair<float, int>> work(N);
    std::vector<Measurement> results;

    for (int rep = -o.warmup; rep < o.repeat; ++rep) {
        Measurement m;
        m.io_ms = io_ms;

        // cała partia
        double t_start = now_ms();
        for (int q = 0; q < Q; ++q)
            topk_query(&queries[(size_t)q * D], base.data(), N, D, k, work,
                       &idx[(size_t)q * k], &dist2[(size_t)q * k]);
        m.core_ms = m.full_ms = now_ms() - t_start;

        // zapytania pojedynczo (opóźnienie)
        for (int q = 0; q < o.lat_queries; ++q) {
            double tq = now_ms();
            topk_query(&queries[(size_t)q * D], base.data(), N, D, k, work,
                       &idx[(size_t)q * k], &dist2[(size_t)q * k]);
            m.lat_us.push_back((now_ms() - tq) * 1000.0);
        }

        m.rss_mb = peak_rss_mb();
        if (rep >= 0) {
            if (o.csv) print_csv("ref", o, 1, 1, 0, rep, m);
            results.push_back(m);
        }
    }

    print_summary("ref", o, results);
    if (!o.out_path.empty()) {
        std::vector<float> dist(dist2.size());
        for (size_t i = 0; i < dist2.size(); ++i) dist[i] = std::sqrt(dist2[i]);
        write_ranking(o.out_path, Q, k, idx, dist);
    }
    return 0;
}
