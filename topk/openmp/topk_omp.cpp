/*
 * topk_omp.cpp — brute-force top-k nearest neighbors (wersja OpenMP)
 *
 * Kompilacja: g++ -O3 -std=c++17 -fopenmp topk_omp.cpp -o topk_omp
 * Użycie:     ./topk_omp <baza.bin> <zapytania.bin> <N> <D> <Q> <k>
 *                         [num_threads] [repeat] [ref_results.txt]
 *
 * Format plików wejściowych: raw float32, row-major (C order), bez nagłówka
 *   baza.bin      — N * D wartości float32
 *   zapytania.bin — Q * D wartości float32
 *
 * ALGORYTM (równoległy top-k):
 *   Dla każdego zapytania:
 *     1. Przestrzeń bazy N jest rozdzielana między wątki (schedule static).
 *     2. Każdy wątek utrzymuje WŁASNY lokalny bufor top-k (max-heap po parze
 *        (dist, index)) — brak sekcji krytycznych w pętli po N.
 *     3. Lokalne bufory scalane są na końcu (p*k elementów — pomijalnie małe).
 *   Eliminuje to sekwencyjny partial_sort po całym N z poprzedniej wersji.
 *   Remisy: mniejszy indeks wygrywa — wynik IDENTYCZNY z topk_ref.
 *
 * Porządek (ścisły, zgodny z referencją):
 *   a < b  <=>  (a.dist < b.dist) || (a.dist == b.dist && a.idx < b.idx)
 *
 * Pomiary: warmup + >=repeat powtórzeń; mediana i rozrzut (min/max,IQR) czasu
 *   batcha; P95 opóźnienia pojedynczego zapytania z WSZYSTKICH powtórzeń
 *   (Q*repeat obserwacji); peak RSS (getrusage); rozdzielenie I/O od rdzenia.
 */

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
#include <omp.h>
#include <sys/resource.h>

struct Cand { float dist; int idx; };

// "mniejszy = lepszy": najpierw mniejszy dist, przy remisie mniejszy indeks.
static inline bool better(const Cand &a, const Cand &b) {
    if (a.dist != b.dist) return a.dist < b.dist;
    return a.idx < b.idx;
}

static std::vector<float> load_bin(const std::string &path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Blad: nie mozna otworzyc: " << path << "\n"; std::exit(1); }
    std::vector<float> data(count);
    f.read(reinterpret_cast<char *>(data.data()),
           static_cast<std::streamsize>(count * sizeof(float)));
    if (!f) { std::cerr << "Blad: za krotki plik: " << path << "\n"; std::exit(1); }
    return data;
}

static inline float sq_dist(const float *a, const float *b, int D) {
    float sum = 0.0f;
    for (int i = 0; i < D; ++i) { float d = a[i] - b[i]; sum += d * d; }
    return sum;
}

// Lokalny bufor top-k jako max-heap: na szczycie AKTUALNIE NAJGORSZY element.
struct TopKBuffer {
    int k;
    std::vector<Cand> heap;
    explicit TopKBuffer(int k_) : k(k_) { heap.reserve(k_); }
    // komparator dla std::*_heap: na szczyt trafia najgorszy => "a<b" == better(a,b)
    static bool heap_less(const Cand &a, const Cand &b) { return better(a, b); }
    inline void push(const Cand &c) {
        if ((int)heap.size() < k) {
            heap.push_back(c);
            std::push_heap(heap.begin(), heap.end(), heap_less);
        } else if (better(c, heap.front())) {
            std::pop_heap(heap.begin(), heap.end(), heap_less);
            heap.back() = c;
            std::push_heap(heap.begin(), heap.end(), heap_less);
        }
    }
    inline void merge(const TopKBuffer &o) { for (const Cand &c : o.heap) push(c); }
};

// Sekwencyjny top-k jednego zapytania przez lokalny max-heap (rozmiar k).
// Uzywany: (a) jako jednostka pracy przy rownoleglosci PO ZAPYTANIACH,
//          (b) przez pojedynczy watek przy rownoleglosci PO BAZIE.
static inline void topk_one_serial(const float *query, const float *base,
                                   int N, int D, int k, TopKBuffer &buf,
                                   std::vector<Cand> &out) {
    buf.heap.clear();
    for (int i = 0; i < N; ++i)
        buf.push(Cand{ sq_dist(query, base + (size_t)i * D, D), i });
    out = buf.heap;
    std::sort(out.begin(), out.end(), better);
}

// Rownoleglosc PO BAZIE (gdy Q < liczba watkow): pojedyncze zapytanie, baza
// rozdzielona miedzy watki, kazdy ma lokalny top-k, potem scalanie. Jeden
// region parallel na zapytanie. Wynik identyczny z referencja.
static void topk_query_base_split(const float *query, const float *base,
                                   int N, int D, int k, int num_threads,
                                   std::vector<TopKBuffer> &locals,
                                   std::vector<Cand> &out)
{
    for (auto &lb : locals) lb.heap.clear();
#pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        TopKBuffer &lb = locals[tid];
#pragma omp for schedule(static)
        for (int i = 0; i < N; ++i)
            lb.push(Cand{ sq_dist(query, base + (size_t)i * D, D), i });
    }
    TopKBuffer merged(k);
    for (auto &lb : locals) merged.merge(lb);
    out = merged.heap;
    std::sort(out.begin(), out.end(), better);
}

static double percentile(std::vector<double> vals, double p) {
    if (vals.empty()) return 0.0;
    std::sort(vals.begin(), vals.end());
    double idx = (p / 100.0) * ((double)vals.size() - 1.0);
    size_t lo = (size_t)idx, hi = lo + 1 < vals.size() ? lo + 1 : lo;
    double frac = idx - (double)lo;
    return vals[lo] * (1.0 - frac) + vals[hi] * frac;
}
static double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n/2] : 0.5*(v[n/2-1]+v[n/2]);
}
static long peak_rss_kib() {
    struct rusage ru; getrusage(RUSAGE_SELF, &ru); return ru.ru_maxrss;
}

static bool verify_results(const std::vector<std::vector<Cand>> &results,
                           const std::string &ref_path, int Q, int k) {
    std::ifstream ref(ref_path);
    if (!ref) { std::cerr << "[VERIFY] Brak pliku ref: " << ref_path
                          << " — pomijam.\n"; return true; }
    std::vector<std::vector<std::pair<int,float>>> rr(
        Q, std::vector<std::pair<int,float>>(k, {-1, 0.0f}));
    std::string line;
    while (std::getline(ref, line)) {
        if (line.empty() || line[0] == '#') continue;
        int qid, rank, idx; float dist;
        if (std::sscanf(line.c_str(), "%d %d %d %f", &qid,&rank,&idx,&dist) == 4)
            if (qid>=0 && qid<Q && rank-1>=0 && rank-1<k)
                rr[qid][rank-1] = {idx, dist};
    }
    int mism = 0;
    for (int q=0;q<Q;++q) for (int i=0;i<k;++i) {
        if (rr[q][i].first < 0) continue;
        if (results[q][i].idx != rr[q][i].first) {
            if (mism<5) std::cerr << "[VERIFY] Roznica: q="<<q<<" rank="<<(i+1)
                << " omp="<<results[q][i].idx<<" ref="<<rr[q][i].first<<"\n";
            ++mism;
        }
    }
    if (mism==0){std::cout<<"[VERIFY] OK — indeksy identyczne z referencja.\n";return true;}
    std::cerr << "[VERIFY] BLAD — " << mism << " niezgodnosci!\n";
    return false;
}

int main(int argc, char *argv[]) {
    if (argc < 7 || argc > 10) {
        std::cerr << "Uzycie: " << argv[0]
            << " <baza.bin> <zapytania.bin> <N> <D> <Q> <k>"
            << " [num_threads] [repeat] [ref_results.txt]\n";
        return 1;
    }
    const std::string base_path = argv[1], query_path = argv[2];
    const int N=std::stoi(argv[3]), D=std::stoi(argv[4]);
    const int Q=std::stoi(argv[5]), k=std::stoi(argv[6]);

    int num_threads = omp_get_max_threads();
    const char *env_t = std::getenv("OMP_NUM_THREADS");
    if (argc >= 8) num_threads = std::stoi(argv[7]);
    else if (env_t) num_threads = std::atoi(env_t);

    int repeat = 5;
    if (argc >= 9) repeat = std::stoi(argv[8]);
    if (repeat < 1) repeat = 1;
    std::string ref_path = (argc == 10) ? argv[9] : "";

    if (k > N) { std::cerr << "Blad: k > N\n"; return 1; }

    std::cout << "Wczytywanie danych...\n";
    std::cout << "  baza:      " << base_path  << " (" << N << " x " << D << ")\n";
    std::cout << "  zapytania: " << query_path << " (" << Q << " x " << D << ")\n";
    std::cout << "  k=" << k << "  watkow=" << num_threads
              << "  powtorzen=" << repeat << "\n\n";

    const std::vector<float> base    = load_bin(base_path,  (size_t)N*D);
    const std::vector<float> queries = load_bin(query_path, (size_t)Q*D);

    std::vector<std::vector<Cand>> all_results(Q, std::vector<Cand>(k));
    std::vector<TopKBuffer> locals(num_threads, TopKBuffer(k));

    // Wybor strategii rownoleglosci:
    //  - Q >= num_threads: rownoleglosc PO ZAPYTANIACH (zgodnie z karta D03
    //    "rozdzielic zapytania") — JEDEN region parallel na caly batch, kazdy
    //    watek liczy cale zapytania sekwencyjnie. Minimalny narzut, dobre
    //    skalowanie przy duzym Q.
    //  - Q <  num_threads: rownoleglosc PO BAZIE (lokalne kopce + scalanie),
    //    zeby wykorzystac watki takze przy malym batchu.
    const bool by_query = (Q >= num_threads);
    const char *strategy = by_query ? "po zapytaniach" : "po bazie";

    auto run_batch = [&](std::vector<double> &pq) -> double {
        auto ts = std::chrono::high_resolution_clock::now();
        if (by_query) {
#pragma omp parallel num_threads(num_threads)
            {
                int tid = omp_get_thread_num();
                TopKBuffer buf(k);
#pragma omp for schedule(dynamic, 1)
                for (int q = 0; q < Q; ++q) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    topk_one_serial(queries.data()+(size_t)q*D, base.data(),
                                    N, D, k, buf, all_results[q]);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    pq[q] = std::chrono::duration<double,std::micro>(t1-t0).count();
                }
                (void)tid;
            }
        } else {
            for (int q=0;q<Q;++q) {
                auto t0 = std::chrono::high_resolution_clock::now();
                topk_query_base_split(queries.data()+(size_t)q*D, base.data(),
                                      N,D,k,num_threads, locals, all_results[q]);
                auto t1 = std::chrono::high_resolution_clock::now();
                pq[q] = std::chrono::duration<double,std::micro>(t1-t0).count();
            }
        }
        auto te = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double,std::milli>(te-ts).count();
    };

    { std::vector<double> warm(Q); run_batch(warm); }  // warmup

    std::vector<double> batch_ms(repeat);
    std::vector<double> all_q_us; all_q_us.reserve((size_t)Q*repeat);
    for (int r=0;r<repeat;++r) {
        std::vector<double> pq(Q);
        batch_ms[r] = run_batch(pq);
        all_q_us.insert(all_q_us.end(), pq.begin(), pq.end());
    }
    long rss = peak_rss_kib();

    double med_ms=median_of(batch_ms);
    double min_ms=*std::min_element(batch_ms.begin(),batch_ms.end());
    double max_ms=*std::max_element(batch_ms.begin(),batch_ms.end());
    double iqr_ms=percentile(batch_ms,75.0)-percentile(batch_ms,25.0);
    double p95_us=percentile(all_q_us,95.0);
    double p50_us=percentile(all_q_us,50.0);
    double qps=(Q/med_ms)*1000.0;

    std::cout << "=== Wyniki (pierwsze zapytanie) ===\n";
    for (int i=0;i<k;++i)
        std::cout << "  #"<<(i+1)<<"  indeks="<<all_results[0][i].idx
                  << "  dist="<<std::sqrt(all_results[0][i].dist)<<"\n";

    std::cout << "\n=== Statystyki (threads="<<num_threads
              <<", strategia="<<strategy<<") ===\n";
    std::printf("  Batch mediana:         %.3f ms  (min=%.3f max=%.3f IQR=%.3f)\n",
                med_ms,min_ms,max_ms,iqr_ms);
    std::printf("  P50 latency / query:   %.3f us\n", p50_us);
    std::printf("  P95 latency / query:   %.3f us  (%zu obserwacji)\n",
                p95_us, all_q_us.size());
    std::printf("  Przepustowosc:         %.1f zapytan/s\n", qps);
    std::printf("  Peak RSS:              %.2f MiB (%ld KiB)\n", rss/1024.0, rss);

    if (!ref_path.empty()) verify_results(all_results, ref_path, Q, k);

    std::string out_name = "results_omp_t"+std::to_string(num_threads)+".txt";
    std::ofstream out(out_name);
    if (out) {
        out << "# topk_omp wyniki\n";
        out << "# N="<<N<<" D="<<D<<" Q="<<Q<<" k="<<k
            << " threads="<<num_threads<<"\n";
        out << "# format: query_id  rank  index  distance\n";
        for (int q=0;q<Q;++q) for (int i=0;i<k;++i)
            out << q<<"  "<<(i+1)<<"  "<<all_results[q][i].idx<<"  "
                << std::sqrt(all_results[q][i].dist)<<"\n";
    }

    // CSV na stderr: N,D,Q,k,threads,repeat,med_ms,min_ms,max_ms,iqr_ms,p50_us,p95_us,qps,rss_kib
    std::fprintf(stderr,
        "CSV,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%ld\n",
        N,D,Q,k,num_threads,repeat,med_ms,min_ms,max_ms,iqr_ms,p50_us,p95_us,qps,rss);
    return 0;
}
