/*
 * topk_cuda.cu — brute-force top-k nearest neighbors (wersja CUDA)
 *
 * Kompilacja: nvcc -O3 -std=c++17 -arch=sm_89 -o topk_cuda topk_cuda.cu
 * Uzycie:
 *   ./topk_cuda <baza.bin> <zapytania.bin> <N> <D> <Q> <k> [opcje]
 *
 * Opcje (domyslne w nawiasach):
 *   --repeat R      liczba niezaleznych powtorzen pomiaru (5)
 *   --warmup W      liczba przebiegow rozgrzewkowych (2)
 *   --threads T     watkow na blok (256)
 *   --no-stream     wylacz strumienie / nakladanie transferu zapytan
 *   --quiet         nie wypisuj listy wynikow na stdout (tylko statystyki)
 *   --out PLIK      plik wynikowy (results_cuda.txt)
 *   --append        dopisz do pliku wynikowego zamiast nadpisywac
 *   --tag NAZWA     etykieta bloku wynikow (np. "N=100000")
 *
 * Format plikow wejsciowych identyczny z topk_ref.cpp:
 *   raw float32, row-major (C order), bez naglowka.
 *
 * Zgodnosc z referencja:
 *   - odleglosc euklidesowa (kwadrat w kernelu, sqrt na koncu),
 *   - remisy rozstrzygane deterministycznie przez MNIEJSZY indeks,
 *   - wyniki posortowane rosnaco po odleglosci; identyczne z topk_ref.
 *
 * Architektura GPU (wlasny rdzen):
 *   - jeden blok CUDA obsluguje jedno zapytanie (blockIdx.x == q),
 *   - wektor zapytania w shared memory, baza czytana grid-stride,
 *   - kazdy watek utrzymuje lokalny top-k w rejestrach (sortowana wstawka),
 *   - REDUKCJA TOP-K ROWNOLEGLA: redukcja drzewiasta po watkach bloku,
 *     scalanie dwoch posortowanych list top-k (merge). Brak sekwencyjnej
 *     petli w watku 0.
 *
 * Pomiary:
 *   - P95 opoznienia POJEDYNCZEGO zapytania: zdarzenia CUDA per zapytanie
 *     (kernel jednoblokowy na zapytanie), wiele obserwacji (Q * repeat).
 *   - Tryb batch (QPS): jeden kernel na caly wsad Q, przepustowosc.
 *   - czas rdzenia (kernel) vs czas pelny (z transferem H->D->H).
 *   - VRAM: cudaMemGetInfo (przed/po alokacji); RAM: getrusage peak RSS.
 *   - warmup + >=5 powtorzen, mediana i rozrzut (min/max/IQR).
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

#include <cuda_runtime.h>
#include <sys/resource.h>

#define MAX_K 64

// ----------------------------------------------------------------------------
// Obsluga bledow CUDA
// ----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "Blad CUDA %s:%d: %s\n", __FILE__, __LINE__,   \
                         cudaGetErrorString(err__));                            \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ----------------------------------------------------------------------------
// Wczytaj plik binarny float32 (count elementow)
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
// Wstaw kandydata (dist, idx) do posortowanej listy top-k (rosnaco po dist).
// Remis: mniejszy indeks wygrywa. Dziala w rejestrach.
// ----------------------------------------------------------------------------
__device__ __forceinline__ void topk_insert(float *d, int *ix, int k,
                                             float dist, int idx) {
    if (dist > d[k - 1] || (dist == d[k - 1] && idx >= ix[k - 1])) return;
    int p = k - 1;
    while (p > 0 && (d[p - 1] > dist ||
                     (d[p - 1] == dist && ix[p - 1] > idx))) {
        d[p]  = d[p - 1];
        ix[p] = ix[p - 1];
        --p;
    }
    d[p]  = dist;
    ix[p] = idx;
}

// ----------------------------------------------------------------------------
// Scal dwie posortowane listy top-k (a i b) w a[], zachowujac k najlepszych.
// Remis po mniejszym indeksie. Obie listy posortowane rosnaco po (dist, idx).
// ----------------------------------------------------------------------------
__device__ __forceinline__ void topk_merge(float *ad, int *ai,
                                            const float *bd, const int *bi,
                                            int k) {
    float rd[MAX_K];
    int   ri[MAX_K];
    int ia = 0, ib = 0;
    for (int o = 0; o < k; ++o) {
        float da = ad[ia], db = bd[ib];
        int   xa = ai[ia], xb = bi[ib];
        bool take_a = (da < db) || (da == db && xa <= xb);
        if (take_a) { rd[o] = da; ri[o] = xa; ++ia; }
        else        { rd[o] = db; ri[o] = xb; ++ib; }
    }
    for (int o = 0; o < k; ++o) { ad[o] = rd[o]; ai[o] = ri[o]; }
}

// ----------------------------------------------------------------------------
// Kernel: jeden blok = jedno zapytanie. q_base = offset pierwszego zapytania.
// ----------------------------------------------------------------------------
__global__ void topk_kernel(const float *__restrict__ base,
                            const float *__restrict__ queries,
                            int *__restrict__ out_idx,
                            float *__restrict__ out_dist,
                            int N, int D, int k, int q_base) {
    const int q   = q_base + blockIdx.x;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    extern __shared__ float smem[];
    float *s_query = smem;                 // D floatow
    float *s_dist  = smem + D;             // nth * k floatow
    int   *s_idx   = (int *)(s_dist + (size_t)nth * k);

    const float *query = queries + (size_t)q * D;
    for (int i = tid; i < D; i += nth) s_query[i] = query[i];
    __syncthreads();

    float loc_d[MAX_K];
    int   loc_i[MAX_K];
    for (int i = 0; i < k; ++i) { loc_d[i] = INFINITY; loc_i[i] = N; }

    // Grid-stride po bazie; wektoryzacja float4 gdy D%4==0.
    if ((D & 3) == 0) {
        const int D4 = D >> 2;
        const float4 *s_query4 = reinterpret_cast<const float4 *>(s_query);
        for (int b = tid; b < N; b += nth) {
            const float4 *vec4 =
                reinterpret_cast<const float4 *>(base + (size_t)b * D);
            float sum = 0.0f;
            #pragma unroll 4
            for (int i = 0; i < D4; ++i) {
                float4 a = s_query4[i];
                float4 c = vec4[i];
                float dx = a.x - c.x, dy = a.y - c.y;
                float dz = a.z - c.z, dw = a.w - c.w;
                sum += dx * dx + dy * dy + dz * dz + dw * dw;
            }
            topk_insert(loc_d, loc_i, k, sum, b);
        }
    } else {
        for (int b = tid; b < N; b += nth) {
            const float *vec = base + (size_t)b * D;
            float sum = 0.0f;
            for (int i = 0; i < D; ++i) {
                float diff = s_query[i] - vec[i];
                sum += diff * diff;
            }
            topk_insert(loc_d, loc_i, k, sum, b);
        }
    }

    for (int i = 0; i < k; ++i) {
        s_dist[(size_t)tid * k + i] = loc_d[i];
        s_idx[(size_t)tid * k + i]  = loc_i[i];
    }
    __syncthreads();

    // REDUKCJA DRZEWIASTA: watek t scala liste t z t+stride (merge dwoch
    // posortowanych list top-k w O(k)). Eliminuje sekwencyjna petle w tid==0.
    for (int stride = nth >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            topk_merge(&s_dist[(size_t)tid * k], &s_idx[(size_t)tid * k],
                       &s_dist[(size_t)(tid + stride) * k],
                       &s_idx[(size_t)(tid + stride) * k], k);
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int i = 0; i < k; ++i) {
            int idx = s_idx[i];
            out_idx[(size_t)q * k + i]  = (idx >= N) ? -1 : idx;
            out_dist[(size_t)q * k + i] =
                (idx >= N) ? INFINITY : sqrtf(s_dist[i]);
        }
    }
}

// ----------------------------------------------------------------------------
// Statystyki rozrzutu
// ----------------------------------------------------------------------------
struct Stats { double median, minv, maxv, q1, q3; };

static double pct(std::vector<double> &v, double p) {
    if (v.empty()) return 0.0;
    double idx = (p / 100.0) * ((double)v.size() - 1.0);
    size_t lo = (size_t)idx;
    size_t hi = lo + 1 < v.size() ? lo + 1 : lo;
    double fr = idx - (double)lo;
    return v[lo] * (1.0 - fr) + v[hi] * fr;
}

static Stats stats_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    Stats s;
    s.median = pct(v, 50.0);
    s.minv   = v.front();
    s.maxv   = v.back();
    s.q1     = pct(v, 25.0);
    s.q3     = pct(v, 75.0);
    return s;
}

// ----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    if (argc < 7) {
        std::cerr << "Uzycie: " << argv[0]
                  << " <baza.bin> <zapytania.bin> <N> <D> <Q> <k> [opcje]\n";
        return 1;
    }
    const std::string base_path  = argv[1];
    const std::string query_path = argv[2];
    const int N = std::stoi(argv[3]);
    const int D = std::stoi(argv[4]);
    const int Q = std::stoi(argv[5]);
    const int k = std::stoi(argv[6]);

    int repeat = 5, warmup = 2, threads = 256;
    bool use_stream = true, quiet = false, append = false;
    std::string out_path = "results_cuda.txt";
    std::string tag = "";
    for (int i = 7; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--repeat" && i + 1 < argc)      repeat = std::stoi(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc) warmup = std::stoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
        else if (a == "--no-stream")              use_stream = false;
        else if (a == "--quiet")                  quiet = true;
        else if (a == "--append")                 append = true;
        else if (a == "--out" && i + 1 < argc)    out_path = argv[++i];
        else if (a == "--tag" && i + 1 < argc)    tag = argv[++i];
        else { std::cerr << "Nieznana opcja: " << a << "\n"; return 1; }
    }

    if (k > N)     { std::cerr << "Blad: k > N\n"; return 1; }
    if (k > MAX_K) { std::cerr << "Blad: k > MAX_K (" << MAX_K << ")\n"; return 1; }

    std::cout << "Wczytywanie danych...\n";
    std::cout << "  baza:      " << base_path  << " (" << N << " x " << D << ")\n";
    std::cout << "  zapytania: " << query_path << " (" << Q << " x " << D << ")\n";
    std::cout << "  k=" << k << " repeat=" << repeat << " warmup=" << warmup
              << " threads=" << threads << " streamy=" << (use_stream ? 1 : 0)
              << "\n\n";

    const std::vector<float> base    = load_bin(base_path,  (size_t)N * D);
    const std::vector<float> queries = load_bin(query_path, (size_t)Q * D);

    size_t vram_free_before = 0, vram_total = 0;
    CUDA_CHECK(cudaMemGetInfo(&vram_free_before, &vram_total));

    float *d_base = nullptr, *d_queries = nullptr, *d_dist = nullptr;
    int   *d_idx  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_base,    (size_t)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_queries, (size_t)Q * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_idx,     (size_t)Q * k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dist,    (size_t)Q * k * sizeof(float)));

    size_t vram_free_after = 0;
    CUDA_CHECK(cudaMemGetInfo(&vram_free_after, &vram_total));
    double vram_alloc_mb =
        (double)(vram_free_before - vram_free_after) / (1024.0 * 1024.0);
    double vram_used_mb =
        (double)(vram_total - vram_free_after) / (1024.0 * 1024.0);
    double vram_total_mb = (double)vram_total / (1024.0 * 1024.0);

    float *h_queries_pin = nullptr;
    int   *h_idx_pin = nullptr;
    float *h_dist_pin = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_queries_pin, (size_t)Q * D * sizeof(float),
                             cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_idx_pin, (size_t)Q * k * sizeof(int),
                             cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_dist_pin, (size_t)Q * k * sizeof(float),
                             cudaHostAllocDefault));
    std::memcpy(h_queries_pin, queries.data(), (size_t)Q * D * sizeof(float));

    std::vector<int>   h_idx((size_t)Q * k);
    std::vector<float> h_dist((size_t)Q * k);

    const dim3 block(threads);
    const size_t shmem = (size_t)D * sizeof(float) +
                         (size_t)threads * k * (sizeof(float) + sizeof(int));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (shmem > prop.sharedMemPerBlock) {
        CUDA_CHECK(cudaFuncSetAttribute(
            topk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)shmem));
    }
    if (shmem > (size_t)prop.sharedMemPerBlockOptin) {
        std::cerr << "Blad: shared memory " << shmem << " B > limit "
                  << prop.sharedMemPerBlockOptin
                  << " B (zmniejsz threads lub k)\n";
        return 1;
    }
    std::cout << "Konfiguracja: bloki=" << Q << " watki=" << threads
              << " shared=" << shmem << " B\n";
    std::cout << "VRAM: alokacja=" << vram_alloc_mb << " MiB, uzyte="
              << vram_used_mb << " MiB / " << vram_total_mb << " MiB\n\n";

    // Baza transferowana RAZ (persystentna) — oddzielamy czas budowy bazy.
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(d_base, base.data(), (size_t)N * D * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaDeviceSynchronize());
    float base_upload_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&base_upload_ms, e0, e1));

    const int NSTREAMS = use_stream ? 4 : 1;
    std::vector<cudaStream_t> streams(NSTREAMS);
    for (int s = 0; s < NSTREAMS; ++s)
        CUDA_CHECK(cudaStreamCreate(&streams[s]));

    std::vector<cudaEvent_t> qs0(Q), qs1(Q);
    for (int q = 0; q < Q; ++q) {
        CUDA_CHECK(cudaEventCreate(&qs0[q]));
        CUDA_CHECK(cudaEventCreate(&qs1[q]));
    }

    // ---- BATCH (QPS): transfer zapytan nakladany na kernel strumieniami ---
    auto run_batch = [&](float &kernel_ms, float &total_ms) {
        cudaEvent_t t0, t1, kk0, kk1;
        CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventCreate(&kk0)); CUDA_CHECK(cudaEventCreate(&kk1));
        CUDA_CHECK(cudaEventRecord(t0, 0));
        if (use_stream) {
            int chunk = (Q + NSTREAMS - 1) / NSTREAMS;
            for (int s = 0; s < NSTREAMS; ++s) {
                int q0 = s * chunk, qn = std::min(chunk, Q - q0);
                if (qn <= 0) break;
                CUDA_CHECK(cudaMemcpyAsync(
                    d_queries + (size_t)q0 * D, h_queries_pin + (size_t)q0 * D,
                    (size_t)qn * D * sizeof(float), cudaMemcpyHostToDevice,
                    streams[s]));
            }
            CUDA_CHECK(cudaEventRecord(kk0, 0));
            for (int s = 0; s < NSTREAMS; ++s) {
                int q0 = s * chunk, qn = std::min(chunk, Q - q0);
                if (qn <= 0) break;
                topk_kernel<<<qn, block, shmem, streams[s]>>>(
                    d_base, d_queries, d_idx, d_dist, N, D, k, q0);
            }
            CUDA_CHECK(cudaEventRecord(kk1, 0));
            for (int s = 0; s < NSTREAMS; ++s) {
                int q0 = s * chunk, qn = std::min(chunk, Q - q0);
                if (qn <= 0) break;
                CUDA_CHECK(cudaMemcpyAsync(
                    h_idx_pin + (size_t)q0 * k, d_idx + (size_t)q0 * k,
                    (size_t)qn * k * sizeof(int), cudaMemcpyDeviceToHost,
                    streams[s]));
                CUDA_CHECK(cudaMemcpyAsync(
                    h_dist_pin + (size_t)q0 * k, d_dist + (size_t)q0 * k,
                    (size_t)qn * k * sizeof(float), cudaMemcpyDeviceToHost,
                    streams[s]));
            }
        } else {
            CUDA_CHECK(cudaMemcpy(d_queries, h_queries_pin,
                                  (size_t)Q * D * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(kk0, 0));
            topk_kernel<<<Q, block, shmem>>>(d_base, d_queries, d_idx, d_dist,
                                             N, D, k, 0);
            CUDA_CHECK(cudaEventRecord(kk1, 0));
            CUDA_CHECK(cudaMemcpy(h_idx_pin, d_idx,
                                  (size_t)Q * k * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_dist_pin, d_dist,
                                  (size_t)Q * k * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaEventRecord(t1, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kk0, kk1));
        CUDA_CHECK(cudaEventElapsedTime(&total_ms,  t0,  t1));
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaEventDestroy(kk0); cudaEventDestroy(kk1);
    };

    // ---- PER-QUERY latency (P95 pojedynczego zapytania) ------------------
    auto run_perquery = [&](std::vector<double> &lat_us) {
        CUDA_CHECK(cudaMemcpy(d_queries, h_queries_pin,
                              (size_t)Q * D * sizeof(float),
                              cudaMemcpyHostToDevice));
        for (int q = 0; q < Q; ++q) {
            cudaStream_t st = streams[q % NSTREAMS];
            CUDA_CHECK(cudaEventRecord(qs0[q], st));
            topk_kernel<<<1, block, shmem, st>>>(d_base, d_queries, d_idx,
                                                 d_dist, N, D, k, q);
            CUDA_CHECK(cudaEventRecord(qs1[q], st));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        for (int q = 0; q < Q; ++q) {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, qs0[q], qs1[q]));
            lat_us.push_back((double)ms * 1000.0);
        }
    };

    for (int w = 0; w < warmup; ++w) { float km, tm; run_batch(km, tm); }

    std::vector<double> kernel_ms_all, total_ms_all, qps_kernel_all, qps_total_all;
    for (int r = 0; r < repeat; ++r) {
        float km = 0.0f, tm = 0.0f;
        run_batch(km, tm);
        kernel_ms_all.push_back(km);
        total_ms_all.push_back(tm);
        qps_kernel_all.push_back((Q / (double)km) * 1000.0);
        qps_total_all.push_back((Q / (double)tm) * 1000.0);
    }
    std::memcpy(h_idx.data(), h_idx_pin, (size_t)Q * k * sizeof(int));
    std::memcpy(h_dist.data(), h_dist_pin, (size_t)Q * k * sizeof(float));

    std::vector<double> lat_us;
    for (int r = 0; r < repeat; ++r) run_perquery(lat_us);

    Stats s_kernel = stats_of(kernel_ms_all);
    Stats s_total  = stats_of(total_ms_all);
    Stats s_qpsk   = stats_of(qps_kernel_all);
    Stats s_qpst   = stats_of(qps_total_all);
    std::vector<double> lat_sorted = lat_us;
    std::sort(lat_sorted.begin(), lat_sorted.end());
    double p50 = pct(lat_sorted, 50.0);
    double p95 = pct(lat_sorted, 95.0);
    double p99 = pct(lat_sorted, 99.0);
    double lat_mean = std::accumulate(lat_us.begin(), lat_us.end(), 0.0) /
                      (double)lat_us.size();

    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    double peak_rss_mb = (double)ru.ru_maxrss / 1024.0;  // KiB -> MiB (Linux)

    if (!quiet) {
        std::cout << "=== Wyniki ===\n";
        for (int q = 0; q < Q; ++q) {
            std::cout << "Zapytanie " << q << ":\n";
            for (int i = 0; i < k; ++i)
                std::cout << "  #" << (i + 1)
                          << "  indeks=" << h_idx[(size_t)q * k + i]
                          << "  dist="   << h_dist[(size_t)q * k + i] << "\n";
        }
    }

    std::printf("\n=== Statystyki (repeat=%d, obserwacji per-query=%zu) ===\n",
                repeat, lat_us.size());
    std::printf("  Transfer bazy (H->D, raz):   %.3f ms\n", base_upload_ms);
    std::printf("  BATCH kernel_ms:   mediana=%.3f  [min=%.3f max=%.3f IQR=%.3f]\n",
                s_kernel.median, s_kernel.minv, s_kernel.maxv,
                s_kernel.q3 - s_kernel.q1);
    std::printf("  BATCH total_ms:    mediana=%.3f  [min=%.3f max=%.3f IQR=%.3f]\n",
                s_total.median, s_total.minv, s_total.maxv,
                s_total.q3 - s_total.q1);
    std::printf("  QPS (kernel):      mediana=%.1f  [min=%.1f max=%.1f]\n",
                s_qpsk.median, s_qpsk.minv, s_qpsk.maxv);
    std::printf("  QPS (total):       mediana=%.1f  [min=%.1f max=%.1f]\n",
                s_qpst.median, s_qpst.minv, s_qpst.maxv);
    std::printf("  Latencja 1 zapytania [us]: p50=%.2f p95=%.2f p99=%.2f "
                "srednia=%.2f\n", p50, p95, p99, lat_mean);
    std::printf("  VRAM: alokacja=%.1f MiB uzyte=%.1f/%.1f MiB\n",
                vram_alloc_mb, vram_used_mb, vram_total_mb);
    std::printf("  RAM peak RSS: %.1f MiB\n", peak_rss_mb);

    std::ofstream out(out_path, append ? std::ios::app : std::ios::trunc);
    out << std::fixed;
    if (!append) out << "# topk_cuda wyniki\n";
    out << "\n## " << (tag.empty() ? ("N=" + std::to_string(N)) : tag)
        << "  (N=" << N << " D=" << D << " Q=" << Q << " k=" << k
        << " threads=" << threads << " streamy=" << (use_stream ? 1 : 0)
        << " repeat=" << repeat << ")\n";
    out << "base_upload_ms="   << base_upload_ms << "\n";
    out << "kernel_ms_median=" << s_kernel.median << " kernel_ms_min="
        << s_kernel.minv << " kernel_ms_max=" << s_kernel.maxv
        << " kernel_ms_iqr=" << (s_kernel.q3 - s_kernel.q1) << "\n";
    out << "total_ms_median="  << s_total.median << " total_ms_min="
        << s_total.minv << " total_ms_max=" << s_total.maxv
        << " total_ms_iqr=" << (s_total.q3 - s_total.q1) << "\n";
    out << "qps_kernel_median=" << s_qpsk.median << " qps_kernel_min="
        << s_qpsk.minv << " qps_kernel_max=" << s_qpsk.maxv << "\n";
    out << "qps_total_median="  << s_qpst.median << " qps_total_min="
        << s_qpst.minv << " qps_total_max=" << s_qpst.maxv << "\n";
    out << "query_latency_us_p50=" << p50 << " p95=" << p95 << " p99=" << p99
        << " mean=" << lat_mean << " n_obs=" << lat_us.size() << "\n";
    out << "vram_alloc_mb=" << vram_alloc_mb << " vram_used_mb=" << vram_used_mb
        << " vram_total_mb=" << vram_total_mb << "\n";
    out << "ram_peak_rss_mb=" << peak_rss_mb << "\n";
    out.close();

    std::cout << "\nWyniki zapisane do " << out_path << "\n";

    for (int q = 0; q < Q; ++q) {
        cudaEventDestroy(qs0[q]); cudaEventDestroy(qs1[q]);
    }
    for (int s = 0; s < NSTREAMS; ++s) cudaStreamDestroy(streams[s]);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFreeHost(h_queries_pin); cudaFreeHost(h_idx_pin); cudaFreeHost(h_dist_pin);
    cudaFree(d_base); cudaFree(d_queries); cudaFree(d_idx); cudaFree(d_dist);
    return 0;
}
