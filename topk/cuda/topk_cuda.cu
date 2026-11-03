/*
 * topk_cuda.cu — brute-force top-k nearest neighbors (wersja CUDA)
 *
 * Kompilacja: nvcc -O3 -std=c++17 -arch=sm_89 -o topk_cuda topk_cuda.cu
 * Uzycie:     ./topk_cuda <baza.bin> <zapytania.bin> <N> <D> <Q> <k>
 *
 * Format plikow wejsciowych identyczny z topk_ref.cpp:
 *   raw float32, row-major (C order), bez naglowka.
 *
 * Algorytm (odpowiadajacy referencji):
 *   - odleglosc euklidesowa (kwadrat w kernelu, sqrt na koncu),
 *   - remisy rozstrzygane deterministycznie przez mniejszy indeks,
 *   - wyniki posortowane rosnaco po odleglosci.
 *
 * Architektura GPU:
 *   - jeden blok CUDA obsluguje jedno zapytanie (blockIdx.x == q),
 *   - watki bloku przegladaja baze rownolegle (grid-stride po N),
 *     kazdy watek utrzymuje lokalny top-k w rejestrach,
 *   - redukcja top-k w shared memory: watek 0 scala kandydatow
 *     z calego bloku i zapisuje wynik.
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

#include <cuda_runtime.h>

#define MAX_K 64
#define THREADS 256

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
// Remis: mniejszy indeks wygrywa. Dziala w rejestrach / shared memory.
// ----------------------------------------------------------------------------
__device__ __forceinline__ void topk_insert(float *d, int *ix, int k,
                                             float dist, int idx) {
    // Odrzuc, jesli gorszy od najgorszego w pelnej liscie
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
// Kernel: jeden blok = jedno zapytanie.
// ----------------------------------------------------------------------------
__global__ void topk_kernel(const float *__restrict__ base,
                            const float *__restrict__ queries,
                            int *__restrict__ out_idx,
                            float *__restrict__ out_dist,
                            int N, int D, int k) {
    const int q   = blockIdx.x;
    const int tid = threadIdx.x;

    // Wektor zapytania w shared memory (wspoldzielony przez watki bloku)
    extern __shared__ float smem[];
    float *s_query = smem;                 // D floatow
    // Bufor redukcji: THREADS * k par (dist, idx)
    float *s_dist  = smem + D;             // THREADS * k floatow
    int   *s_idx   = (int *)(s_dist + (size_t)THREADS * k);

    const float *query = queries + (size_t)q * D;
    for (int i = tid; i < D; i += blockDim.x) s_query[i] = query[i];
    __syncthreads();

    // Lokalny top-k w rejestrach
    float loc_d[MAX_K];
    int   loc_i[MAX_K];
    for (int i = 0; i < k; ++i) {
        loc_d[i] = INFINITY;
        loc_i[i] = N;  // wartownik: duzy indeks
    }

    // Grid-stride po bazie
    for (int b = tid; b < N; b += blockDim.x) {
        const float *vec = base + (size_t)b * D;
        float sum = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < D; ++i) {
            float diff = s_query[i] - vec[i];
            sum += diff * diff;
        }
        topk_insert(loc_d, loc_i, k, sum, b);
    }

    // Zapisz lokalny top-k do shared memory
    for (int i = 0; i < k; ++i) {
        s_dist[(size_t)tid * k + i] = loc_d[i];
        s_idx[(size_t)tid * k + i]  = loc_i[i];
    }
    __syncthreads();

    // Redukcja: watek 0 scala wszystkie lokalne listy
    if (tid == 0) {
        float fin_d[MAX_K];
        int   fin_i[MAX_K];
        for (int i = 0; i < k; ++i) { fin_d[i] = INFINITY; fin_i[i] = N; }

        int total = blockDim.x * k;
        for (int j = 0; j < total; ++j) {
            int idx = s_idx[j];
            if (idx >= N) continue;  // wartownik
            topk_insert(fin_d, fin_i, k, s_dist[j], idx);
        }

        for (int i = 0; i < k; ++i) {
            out_idx[(size_t)q * k + i]  = fin_i[i];
            out_dist[(size_t)q * k + i] = sqrtf(fin_d[i]);
        }
    }
}

// ----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    if (argc != 7) {
        std::cerr << "Uzycie: " << argv[0]
                  << " <baza.bin> <zapytania.bin> <N> <D> <Q> <k>\n";
        return 1;
    }
    const std::string base_path  = argv[1];
    const std::string query_path = argv[2];
    const int N = std::stoi(argv[3]);
    const int D = std::stoi(argv[4]);
    const int Q = std::stoi(argv[5]);
    const int k = std::stoi(argv[6]);

    if (k > N)  { std::cerr << "Blad: k > N\n"; return 1; }
    if (k > MAX_K) { std::cerr << "Blad: k > MAX_K (" << MAX_K << ")\n"; return 1; }

    std::cout << "Wczytywanie danych...\n";
    std::cout << "  baza:      " << base_path  << " (" << N << " x " << D << ")\n";
    std::cout << "  zapytania: " << query_path << " (" << Q << " x " << D << ")\n";
    std::cout << "  k = " << k << "\n\n";

    const std::vector<float> base    = load_bin(base_path,  (size_t)N * D);
    const std::vector<float> queries = load_bin(query_path, (size_t)Q * D);

    // ---- Alokacja GPU -----------------------------------------------------
    float *d_base = nullptr, *d_queries = nullptr, *d_dist = nullptr;
    int   *d_idx  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_base,    (size_t)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_queries, (size_t)Q * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_idx,     (size_t)Q * k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dist,    (size_t)Q * k * sizeof(float)));

    std::vector<int>   h_idx((size_t)Q * k);
    std::vector<float> h_dist((size_t)Q * k);

    // ---- Zdarzenia pomiarowe ---------------------------------------------
    cudaEvent_t e_total0, e_total1, e_kernel0, e_kernel1;
    CUDA_CHECK(cudaEventCreate(&e_total0));
    CUDA_CHECK(cudaEventCreate(&e_total1));
    CUDA_CHECK(cudaEventCreate(&e_kernel0));
    CUDA_CHECK(cudaEventCreate(&e_kernel1));

    const int   blocks = Q;
    const dim3  grid(blocks), block(THREADS);
    const size_t shmem = (size_t)D * sizeof(float) +
                         (size_t)THREADS * k * (sizeof(float) + sizeof(int));

    // Sprawdz limit shared memory
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (shmem > prop.sharedMemPerBlock) {
        // Podnies limit (opt-in) jesli architektura pozwala
        CUDA_CHECK(cudaFuncSetAttribute(
            topk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)shmem));
    }
    std::cout << "Konfiguracja: bloki=" << blocks << " watki=" << THREADS
              << " shared=" << shmem << " B\n\n";

    // ---- Pomiar CALKOWITY (H->D, kernel, D->H) ----------------------------
    CUDA_CHECK(cudaEventRecord(e_total0));

    CUDA_CHECK(cudaMemcpy(d_base, base.data(),
                          (size_t)N * D * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queries, queries.data(),
                          (size_t)Q * D * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(e_kernel0));
    topk_kernel<<<grid, block, shmem>>>(d_base, d_queries, d_idx, d_dist,
                                        N, D, k);
    CUDA_CHECK(cudaEventRecord(e_kernel1));

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_idx.data(), d_idx,
                          (size_t)Q * k * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dist.data(), d_dist,
                          (size_t)Q * k * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(e_total1));
    CUDA_CHECK(cudaDeviceSynchronize());

    float kernel_ms = 0.0f, total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, e_kernel0, e_kernel1));
    CUDA_CHECK(cudaEventElapsedTime(&total_ms,  e_total0,  e_total1));

    // ---- Statystyki -------------------------------------------------------
    // Czas per-query mierzony jako kernel/Q (kernel przetwarza wszystkie Q
    // rownolegle, wiec P95 liczymy jako efektywny czas na zapytanie).
    double avg_us_total  = (double)total_ms  * 1000.0 / Q;
    double avg_us_kernel = (double)kernel_ms * 1000.0 / Q;
    double qps_kernel = (Q / (double)kernel_ms) * 1000.0;
    double qps_total  = (Q / (double)total_ms)  * 1000.0;

    // ---- Wyniki na stdout -------------------------------------------------
    std::cout << "=== Wyniki ===\n";
    for (int q = 0; q < Q; ++q) {
        std::cout << "Zapytanie " << q << ":\n";
        for (int i = 0; i < k; ++i) {
            std::cout << "  #" << (i + 1)
                      << "  indeks=" << h_idx[(size_t)q * k + i]
                      << "  dist="   << h_dist[(size_t)q * k + i] << "\n";
        }
    }

    std::cout << "\n=== Statystyki ===\n";
    std::printf("  Czas kernela (GPU):    %.3f ms\n", kernel_ms);
    std::printf("  Czas calkowity (H->D->H): %.3f ms\n", total_ms);
    std::printf("  Transfer (total-kernel):  %.3f ms\n", total_ms - kernel_ms);
    std::printf("  Sredni czas/query (kernel):  %.3f us\n", avg_us_kernel);
    std::printf("  Sredni czas/query (total):   %.3f us\n", avg_us_total);
    std::printf("  QPS (kernel):          %.1f zapytan/s\n", qps_kernel);
    std::printf("  QPS (total):           %.1f zapytan/s\n", qps_total);

    // ---- Zapis do pliku ---------------------------------------------------
    std::ofstream out("results_cuda.txt");
    out << "# topk_cuda wyniki\n";
    out << "# N=" << N << " D=" << D << " Q=" << Q << " k=" << k << "\n";
    out << "# format: query_id  rank  index  distance\n";
    for (int q = 0; q < Q; ++q)
        for (int i = 0; i < k; ++i)
            out << q << "  " << (i + 1) << "  "
                << h_idx[(size_t)q * k + i] << "  "
                << h_dist[(size_t)q * k + i] << "\n";
    out << "\n# Statystyki\n";
    out << std::fixed;
    out << "kernel_ms=" << kernel_ms << "\n";
    out << "total_ms="  << total_ms  << "\n";
    out << "transfer_ms=" << (total_ms - kernel_ms) << "\n";
    out << "avg_us_kernel=" << avg_us_kernel << "\n";
    out << "avg_us_total="  << avg_us_total  << "\n";
    out << "qps_kernel="    << qps_kernel    << "\n";
    out << "qps_total="     << qps_total     << "\n";
    out.close();

    std::cout << "\nWyniki zapisane do results_cuda.txt\n";

    cudaFree(d_base); cudaFree(d_queries); cudaFree(d_idx); cudaFree(d_dist);
    cudaEventDestroy(e_total0); cudaEventDestroy(e_total1);
    cudaEventDestroy(e_kernel0); cudaEventDestroy(e_kernel1);
    return 0;
}
