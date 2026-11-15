/*
 * topk_cublas.cu — top-k nearest neighbors, DRUGA implementacja (referencja
 *                  wydajnosci oparta o zoptymalizowana biblioteke cuBLAS).
 *
 * Kompilacja:
 *   nvcc -O3 -std=c++17 -arch=sm_89 -o topk_cublas topk_cublas.cu -lcublas
 * Uzycie:
 *   ./topk_cublas <baza.bin> <zapytania.bin> <N> <D> <Q> <k> [opcje]
 *   (opcje: --repeat R, --warmup W, --quiet, --out PLIK, --append, --tag NAZWA)
 *
 * WKLAD WLASNY vs BIBLIOTEKA:
 *   - cuBLAS (SGEMM) liczy TYLKO iloczyn skalarny Q*B^T (czesc odleglosci).
 *   - Odleglosc euklidesowa kwadrat: ||q-b||^2 = ||q||^2 + ||b||^2 - 2 q.b
 *     Skladanie norm + selekcja top-k (z remisami po mniejszym indeksie)
 *     to WLASNE kernele. cuBLAS sluzy jako zoptymalizowany punkt odniesienia
 *     dla czesci obliczen odleglosci (zgodnie z karta D03).
 *
 * Zgodnosc z referencja: te same indeksy top-k (remis = mniejszy indeks),
 * odleglosci w ramach tolerancji FP (inna kolejnosc sumowania w SGEMM).
 */

#include <algorithm>
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
#include <cublas_v2.h>
#include <sys/resource.h>

#define MAX_K 64

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "Blad CUDA %s:%d: %s\n", __FILE__, __LINE__,   \
                         cudaGetErrorString(err__));                            \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st__ = (call);                                           \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                    \
            std::fprintf(stderr, "Blad cuBLAS %s:%d: kod %d\n", __FILE__,       \
                         __LINE__, (int)st__);                                  \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

static std::vector<float> load_bin(const std::string &path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Blad: nie mozna otworzyc: " << path << "\n"; std::exit(1); }
    std::vector<float> data(count);
    f.read(reinterpret_cast<char *>(data.data()),
           static_cast<std::streamsize>(count * sizeof(float)));
    if (!f) { std::cerr << "Blad: za krotki plik: " << path << "\n"; std::exit(1); }
    return data;
}

// Kwadraty norm wierszy macierzy (N x D) -> norms[N]  (WLASNY kernel)
__global__ void row_sqnorm(const float *__restrict__ M, float *__restrict__ nrm,
                           int N, int D) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N) return;
    const float *v = M + (size_t)r * D;
    float s = 0.0f;
    for (int i = 0; i < D; ++i) s += v[i] * v[i];
    nrm[r] = s;
}

// Zloz pelna odleglosc kwadrat z iloczynu skalarnego + norm (WLASNY kernel).
// dots: Q x N, wiersz q = iloczyny q.b_j. dist2[q*N+j] = qn[q] + bn[j] - 2 dot
__global__ void assemble_dist2(float *__restrict__ dots,
                               const float *__restrict__ qn,
                               const float *__restrict__ bn,
                               int Q, int N) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t tot = (size_t)Q * N;
    if (t >= tot) return;
    int q = (int)(t / N), j = (int)(t % N);
    float d = qn[q] + bn[j] - 2.0f * dots[t];
    dots[t] = d < 0.0f ? 0.0f : d;  // zabezpieczenie FP przed ujemna wartoscia
}

// Selekcja top-k dla jednego zapytania (1 blok = 1 zapytanie). WLASNY kernel.
// Redukcja drzewiasta po watkach, scalanie posortowanych list, remis = idx.
__device__ __forceinline__ void tk_insert(float *d, int *ix, int k, float v, int idx) {
    if (v > d[k - 1] || (v == d[k - 1] && idx >= ix[k - 1])) return;
    int p = k - 1;
    while (p > 0 && (d[p - 1] > v || (d[p - 1] == v && ix[p - 1] > idx))) {
        d[p] = d[p - 1]; ix[p] = ix[p - 1]; --p;
    }
    d[p] = v; ix[p] = idx;
}
__device__ __forceinline__ void tk_merge(float *ad, int *ai, const float *bd,
                                          const int *bi, int k) {
    float rd[MAX_K]; int ri[MAX_K];
    int ia = 0, ib = 0;
    for (int o = 0; o < k; ++o) {
        float da = ad[ia], db = bd[ib];
        int xa = ai[ia], xb = bi[ib];
        bool ta = (da < db) || (da == db && xa <= xb);
        if (ta) { rd[o] = da; ri[o] = xa; ++ia; }
        else    { rd[o] = db; ri[o] = xb; ++ib; }
    }
    for (int o = 0; o < k; ++o) { ad[o] = rd[o]; ai[o] = ri[o]; }
}

__global__ void select_topk(const float *__restrict__ dist2,
                            int *__restrict__ out_idx,
                            float *__restrict__ out_dist,
                            int N, int k) {
    const int q = blockIdx.x, tid = threadIdx.x, nth = blockDim.x;
    extern __shared__ float smem[];
    float *s_d = smem;
    int   *s_i = (int *)(s_d + (size_t)nth * k);

    float loc_d[MAX_K]; int loc_i[MAX_K];
    for (int i = 0; i < k; ++i) { loc_d[i] = INFINITY; loc_i[i] = N; }
    const float *row = dist2 + (size_t)q * N;
    for (int b = tid; b < N; b += nth) tk_insert(loc_d, loc_i, k, row[b], b);

    for (int i = 0; i < k; ++i) {
        s_d[(size_t)tid * k + i] = loc_d[i];
        s_i[(size_t)tid * k + i] = loc_i[i];
    }
    __syncthreads();
    for (int stride = nth >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            tk_merge(&s_d[(size_t)tid * k], &s_i[(size_t)tid * k],
                     &s_d[(size_t)(tid + stride) * k],
                     &s_i[(size_t)(tid + stride) * k], k);
        __syncthreads();
    }
    if (tid == 0)
        for (int i = 0; i < k; ++i) {
            int idx = s_i[i];
            out_idx[(size_t)q * k + i]  = (idx >= N) ? -1 : idx;
            out_dist[(size_t)q * k + i] = (idx >= N) ? INFINITY : sqrtf(s_d[i]);
        }
}

struct Stats { double median, minv, maxv, q1, q3; };
static double pct(std::vector<double> &v, double p) {
    if (v.empty()) return 0.0;
    double idx = (p / 100.0) * ((double)v.size() - 1.0);
    size_t lo = (size_t)idx, hi = lo + 1 < v.size() ? lo + 1 : lo;
    double fr = idx - (double)lo;
    return v[lo] * (1.0 - fr) + v[hi] * fr;
}
static Stats stats_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return { pct(v,50.0), v.front(), v.back(), pct(v,25.0), pct(v,75.0) };
}

int main(int argc, char *argv[]) {
    if (argc < 7) {
        std::cerr << "Uzycie: " << argv[0]
                  << " <baza.bin> <zapytania.bin> <N> <D> <Q> <k> [opcje]\n";
        return 1;
    }
    const std::string base_path = argv[1], query_path = argv[2];
    const int N = std::stoi(argv[3]), D = std::stoi(argv[4]);
    const int Q = std::stoi(argv[5]), k = std::stoi(argv[6]);
    int repeat = 5, warmup = 2, threads = 256;
    bool quiet = false, append = false;
    std::string out_path = "results_cuda.txt", tag = "";
    for (int i = 7; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--repeat" && i + 1 < argc)      repeat = std::stoi(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc) warmup = std::stoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
        else if (a == "--quiet")                  quiet = true;
        else if (a == "--append")                 append = true;
        else if (a == "--out" && i + 1 < argc)    out_path = argv[++i];
        else if (a == "--tag" && i + 1 < argc)    tag = argv[++i];
        else { std::cerr << "Nieznana opcja: " << a << "\n"; return 1; }
    }
    if (k > N || k > MAX_K) { std::cerr << "Blad: k > N lub k > MAX_K\n"; return 1; }

    std::cout << "topk_cublas (referencja wydajnosci): N=" << N << " D=" << D
              << " Q=" << Q << " k=" << k << " repeat=" << repeat << "\n\n";

    const std::vector<float> base    = load_bin(base_path,  (size_t)N * D);
    const std::vector<float> queries = load_bin(query_path, (size_t)Q * D);

    size_t vfree0 = 0, vtot = 0;
    CUDA_CHECK(cudaMemGetInfo(&vfree0, &vtot));

    float *d_base, *d_q, *d_dots, *d_bn, *d_qn, *d_dist;
    int   *d_idx;
    CUDA_CHECK(cudaMalloc(&d_base, (size_t)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q,    (size_t)Q * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dots, (size_t)Q * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bn,   (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_qn,   (size_t)Q * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_idx,  (size_t)Q * k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dist, (size_t)Q * k * sizeof(float)));

    size_t vfree1 = 0;
    CUDA_CHECK(cudaMemGetInfo(&vfree1, &vtot));
    double vram_alloc_mb = (double)(vfree0 - vfree1) / (1024.0 * 1024.0);
    double vram_used_mb  = (double)(vtot - vfree1) / (1024.0 * 1024.0);
    double vram_total_mb = (double)vtot / (1024.0 * 1024.0);

    CUDA_CHECK(cudaMemcpy(d_base, base.data(), (size_t)N * D * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, queries.data(), (size_t)Q * D * sizeof(float),
                          cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const size_t shmem = (size_t)threads * k * (sizeof(float) + sizeof(int));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (shmem > prop.sharedMemPerBlock)
        CUDA_CHECK(cudaFuncSetAttribute(select_topk,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));

    std::vector<int>   h_idx((size_t)Q * k);
    std::vector<float> h_dist((size_t)Q * k);

    const float alpha = 1.0f, beta = 0.0f;

    // Jeden pelny przebieg obliczen. Zwraca kernel_ms (bez transferu).
    auto run_once = [&](float &kernel_ms) {
        cudaEvent_t a, b;
        CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
        CUDA_CHECK(cudaEventRecord(a, 0));

        // Normy (wlasne kernele)
        row_sqnorm<<<(N + 255) / 256, 256>>>(d_base, d_bn, N, D);
        row_sqnorm<<<(Q + 255) / 256, 256>>>(d_q, d_qn, Q, D);

        // SGEMM (cuBLAS): dots[Q x N] = Q[Q x D] * B[N x D]^T
        // cuBLAS column-major. Liczymy C^T = (N x Q): C(j,q)=sum_d B(j,d)Q(q,d)
        // Traktujac wiersze (row-major) jako kolumny, uzywamy:
        //   C[N x Q] col-major = B^T? Wygodnie: result row-major (Q x N).
        // Uzyjemy: dots (row-major QxN) == col-major (N x Q).
        // col-major C(NxQ) = op(A=B, NxD row = DxN colmajor) ...
        // Prosto: cublasSgemm(N trans? ) -> stosujemy sprawdzony uklad:
        //   C(NxQ,colmajor) = Btilde(NxD rowmajor = DxN colmajor)^T * ...
        // Realizacja: C[j + q*N] = sum_d base[j*D+d]*q[q*D+d]
        //   = (base_colmajor D x N)^T . (q_colmajor D x Q)
        // base row-major NxD == col-major DxN (lda=D)
        // q    row-major QxD == col-major DxQ (lda=D)
        // C = base_cm^T (N x D) * q_cm (D x Q) -> N x Q col-major, ldc=N
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 N, Q, D, &alpha,
                                 d_base, D,   // A: DxN colmajor (lda=D)
                                 d_q,    D,   // B: DxQ colmajor (ldb=D)
                                 &beta, d_dots, N));  // C: NxQ colmajor (ldc=N)
        // d_dots col-major (N x Q) == row-major (Q x N): dots[q*N + j]. OK.

        size_t tot = (size_t)Q * N;
        assemble_dist2<<<(tot + 255) / 256, 256>>>(d_dots, d_qn, d_bn, Q, N);
        select_topk<<<Q, threads, shmem>>>(d_dots, d_idx, d_dist, N, k);

        CUDA_CHECK(cudaEventRecord(b, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, a, b));
        cudaEventDestroy(a); cudaEventDestroy(b);
    };

    for (int w = 0; w < warmup; ++w) { float km; run_once(km); }

    std::vector<double> kernel_ms_all, qps_all;
    for (int r = 0; r < repeat; ++r) {
        float km = 0.0f; run_once(km);
        kernel_ms_all.push_back(km);
        qps_all.push_back((Q / (double)km) * 1000.0);
    }
    CUDA_CHECK(cudaMemcpy(h_idx.data(), d_idx, (size_t)Q * k * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dist.data(), d_dist, (size_t)Q * k * sizeof(float),
                          cudaMemcpyDeviceToHost));

    Stats sk = stats_of(kernel_ms_all), sq = stats_of(qps_all);
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    double peak_rss_mb = (double)ru.ru_maxrss / 1024.0;

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

    std::printf("\n=== Statystyki cuBLAS (repeat=%d) ===\n", repeat);
    std::printf("  kernel_ms (SGEMM+selekcja): mediana=%.3f [min=%.3f max=%.3f IQR=%.3f]\n",
                sk.median, sk.minv, sk.maxv, sk.q3 - sk.q1);
    std::printf("  QPS (kernel): mediana=%.1f [min=%.1f max=%.1f]\n",
                sq.median, sq.minv, sq.maxv);
    std::printf("  VRAM: alokacja=%.1f MiB uzyte=%.1f/%.1f MiB\n",
                vram_alloc_mb, vram_used_mb, vram_total_mb);
    std::printf("  RAM peak RSS: %.1f MiB\n", peak_rss_mb);
    std::printf("  UWAGA: macierz odleglosci Q*N=%zu float (%.1f MiB) materializowana w VRAM\n",
                (size_t)Q * N, (double)((size_t)Q * N * sizeof(float)) / (1024.0 * 1024.0));

    std::ofstream out(out_path, append ? std::ios::app : std::ios::trunc);
    out << std::fixed;
    if (!append) out << "# topk_cublas wyniki\n";
    out << "\n## CUBLAS " << (tag.empty() ? ("N=" + std::to_string(N)) : tag)
        << "  (N=" << N << " D=" << D << " Q=" << Q << " k=" << k
        << " repeat=" << repeat << ")\n";
    out << "kernel_ms_median=" << sk.median << " kernel_ms_min=" << sk.minv
        << " kernel_ms_max=" << sk.maxv << " kernel_ms_iqr=" << (sk.q3 - sk.q1) << "\n";
    out << "qps_kernel_median=" << sq.median << " qps_kernel_min=" << sq.minv
        << " qps_kernel_max=" << sq.maxv << "\n";
    out << "vram_alloc_mb=" << vram_alloc_mb << " vram_used_mb=" << vram_used_mb
        << " vram_total_mb=" << vram_total_mb << "\n";
    out << "ram_peak_rss_mb=" << peak_rss_mb << "\n";
    out << "dist_matrix_mb=" << (double)((size_t)Q * N * sizeof(float)) / (1024.0 * 1024.0) << "\n";
    out.close();
    std::cout << "\nWyniki zapisane do " << out_path << "\n";

    cublasDestroy(handle);
    cudaFree(d_base); cudaFree(d_q); cudaFree(d_dots); cudaFree(d_bn);
    cudaFree(d_qn); cudaFree(d_idx); cudaFree(d_dist);
    return 0;
}
