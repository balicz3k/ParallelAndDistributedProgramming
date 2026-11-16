/*
 * topk_cublas.cu - top-k na GPU z cuBLAS jako dodatkowy punkt odniesienia.
 *
 * Kompilacja: nvcc -O3 -std=c++17 -arch=sm_89 -I../common topk_cublas.cu -o topk_cublas -lcublas
 * Użycie:     ./topk_cublas <baza.bin> <zapytania.bin> <N> <D> <Q> <k> [--threads T] [opcje z common.h]
 *
 * Kwadrat odległości rozpisany jako ||q - b||^2 = ||q||^2 + ||b||^2 - 2 q.b.
 *  - BIBLIOTEKA: tylko iloczyny q.b dla wszystkich par - jedno mnożenie
 *    macierzy cublasSgemm (zapytania x baza^T),
 *  - WŁASNE kernele: normy wierszy (row_sqnorm), złożenie odległości
 *    (assemble_dist2) i wybór top-k (select_topk - ten sam schemat co
 *    w topk_cuda.cu: lista k najlepszych na wątek + scalanie drzewiaste).
 * Normy bazy liczone są raz, razem z kopiowaniem bazy na GPU.
 * Macierz odległości ma rozmiar (zapytania x N), więc partia jest liczona
 * kawałkami po CHUNK zapytań, tak żeby macierz zajmowała najwyżej 512 MiB.
 * Wynik może różnić się od referencji kolejnością prawie równych odległości,
 * bo SGEMM sumuje w innej kolejności (różnice rzędu 1e-6).
 *
 * Czasy jak w topk_cuda.cu: core_ms - obliczenia partii na GPU,
 * full_ms - z kopiowaniem zapytań i wyników; P95 - zapytania pojedynczo.
 */
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "common.h"

#define MAX_K 64

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "Błąd CUDA %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err_));                           \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                          \
    do {                                                                            \
        if ((call) != CUBLAS_STATUS_SUCCESS) {                                      \
            std::fprintf(stderr, "Błąd cuBLAS %s:%d\n", __FILE__, __LINE__);        \
            std::exit(1);                                                           \
        }                                                                           \
    } while (0)

// norma^2 każdego wiersza macierzy rows x D
__global__ void row_sqnorm(const float *M, float *norm2, int rows, int D) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float s = 0.0f;
    for (int i = 0; i < D; ++i) s += M[(size_t)r * D + i] * M[(size_t)r * D + i];
    norm2[r] = s;
}

// dots[q*N + j] = q.b_j  ->  ||q||^2 + ||b_j||^2 - 2 q.b_j (ujemne zera z zaokrągleń -> 0)
__global__ void assemble_dist2(float *dots, const float *qn, const float *bn, int nq, int N) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nq * N) return;
    int q = (int)(t / N), j = (int)(t % N);
    float d = qn[q] + bn[j] - 2.0f * dots[t];
    dots[t] = d < 0.0f ? 0.0f : d;
}

__device__ __forceinline__ void topk_insert(float *d, int *ix, int k, float dist, int idx) {
    if (dist > d[k - 1] || (dist == d[k - 1] && idx >= ix[k - 1])) return;
    int p = k - 1;
    while (p > 0 && (d[p - 1] > dist || (d[p - 1] == dist && ix[p - 1] > idx))) {
        d[p] = d[p - 1];
        ix[p] = ix[p - 1];
        --p;
    }
    d[p] = dist;
    ix[p] = idx;
}

__device__ __forceinline__ void topk_merge(float *ad, int *ai, const float *bd, const int *bi, int k) {
    float rd[MAX_K];
    int ri[MAX_K];
    int ia = 0, ib = 0;
    for (int o = 0; o < k; ++o) {
        bool take_a = ad[ia] < bd[ib] || (ad[ia] == bd[ib] && ai[ia] <= bi[ib]);
        if (take_a) { rd[o] = ad[ia]; ri[o] = ai[ia]; ++ia; }
        else        { rd[o] = bd[ib]; ri[o] = bi[ib]; ++ib; }
    }
    for (int o = 0; o < k; ++o) {
        ad[o] = rd[o];
        ai[o] = ri[o];
    }
}

// Jeden blok = jeden wiersz macierzy odległości (jedno zapytanie).
__global__ void select_topk(const float *dist2, int *out_idx, float *out_dist, int N, int k) {
    const int q = blockIdx.x, tid = threadIdx.x, T = blockDim.x;
    extern __shared__ float smem[];
    float *s_dist = smem;
    int *s_idx = (int *)(s_dist + (size_t)T * k);

    float loc_d[MAX_K];
    int loc_i[MAX_K];
    for (int i = 0; i < k; ++i) {
        loc_d[i] = INFINITY;
        loc_i[i] = N;
    }
    for (int b = tid; b < N; b += T) topk_insert(loc_d, loc_i, k, dist2[(size_t)q * N + b], b);
    for (int i = 0; i < k; ++i) {
        s_dist[(size_t)tid * k + i] = loc_d[i];
        s_idx[(size_t)tid * k + i] = loc_i[i];
    }
    __syncthreads();
    for (int s = T / 2; s > 0; s /= 2) {
        if (tid < s)
            topk_merge(&s_dist[(size_t)tid * k], &s_idx[(size_t)tid * k],
                       &s_dist[(size_t)(tid + s) * k], &s_idx[(size_t)(tid + s) * k], k);
        __syncthreads();
    }
    if (tid == 0)
        for (int i = 0; i < k; ++i) {
            out_idx[(size_t)q * k + i] = s_idx[i];
            out_dist[(size_t)q * k + i] = sqrtf(s_dist[i]);
        }
}

int main(int argc, char **argv) {
    Options o;
    if (!parse_options(argc, argv, o) || o.k > MAX_K) {
        if (o.k > MAX_K) std::fprintf(stderr, "Błąd: k > MAX_K (%d)\n", MAX_K);
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

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int T = o.threads > 0 ? o.threads : 256;
    if (T < 32 || T > 1024 || (T & (T - 1)) != 0) {
        std::fprintf(stderr, "Błąd: liczba wątków w bloku musi być potęgą 2 z zakresu 32-1024\n");
        return 1;
    }
    auto shmem_for = [&](int t) { return (size_t)t * k * (sizeof(float) + sizeof(int)); };
    while (T > 32 && shmem_for(T) > prop.sharedMemPerBlockOptin) T /= 2;
    const size_t shmem = shmem_for(T);
    CUDA_CHECK(cudaFuncSetAttribute(select_topk, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));

    // ile zapytań naraz: macierz odległości CHUNK x N najwyżej 512 MiB
    const int CHUNK = (int)std::max<size_t>(1, std::min<size_t>(Q, (512u << 20) / sizeof(float) / N));

    size_t free_before, free_after, total;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
    float *d_base, *d_bn, *d_q, *d_qn, *d_dots, *d_dist;
    int *d_idx;
    CUDA_CHECK(cudaMalloc(&d_base, (size_t)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bn, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q, (size_t)Q * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_qn, (size_t)Q * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dots, (size_t)CHUNK * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_idx, (size_t)Q * k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dist, (size_t)Q * k * sizeof(float)));
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
    const double vram_mb = (double)(free_before - free_after) / (1024.0 * 1024.0);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // baza na GPU raz, razem z jej normami
    double tb = now_ms();
    CUDA_CHECK(cudaMemcpy(d_base, base.data(), (size_t)N * D * sizeof(float), cudaMemcpyHostToDevice));
    row_sqnorm<<<(N + 255) / 256, 256>>>(d_base, d_bn, N, D);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double base_ms = now_ms() - tb;

    // Obliczenia dla zapytań q0 .. q0+nq-1 (zapytania i ich normy są już na GPU).
    // cuBLAS liczy w układzie kolumnowym: baza N x D (wierszowo) to dla niego
    // macierz D x N, zapytania - D x nq. C = bazaᵀ * zapytania ma wymiar N x nq
    // kolumnowo, czyli w pamięci dokładnie dots[q*N + j] = q.b_j.
    const float alpha = 1.0f, beta = 0.0f;
    auto compute = [&](int q0, int nq) {
        for (int c0 = q0; c0 < q0 + nq; c0 += CHUNK) {
            int nc = std::min(CHUNK, q0 + nq - c0);
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, nc, D, &alpha,
                                     d_base, D, d_q + (size_t)c0 * D, D, &beta, d_dots, N));
            size_t cells = (size_t)nc * N;
            assemble_dist2<<<(unsigned)((cells + 255) / 256), 256>>>(d_dots, d_qn + c0, d_bn, nc, N);
            select_topk<<<nc, T, shmem>>>(d_dots, d_idx + (size_t)c0 * k, d_dist + (size_t)c0 * k, N, k);
        }
    };

    std::vector<int> idx((size_t)Q * k);
    std::vector<float> dist((size_t)Q * k);
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    std::vector<Measurement> results;

    for (int rep = -o.warmup; rep < o.repeat; ++rep) {
        Measurement m;
        m.io_ms = io_ms;
        m.base_ms = base_ms;
        m.vram_mb = vram_mb;

        double t_full = now_ms();
        CUDA_CHECK(cudaMemcpy(d_q, queries.data(), (size_t)Q * D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(e0, 0));
        row_sqnorm<<<(Q + 255) / 256, 256>>>(d_q, d_qn, Q, D);
        compute(0, Q);
        CUDA_CHECK(cudaEventRecord(e1, 0));
        CUDA_CHECK(cudaMemcpy(idx.data(), d_idx, (size_t)Q * k * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, (size_t)Q * k * sizeof(float), cudaMemcpyDeviceToHost));
        m.full_ms = now_ms() - t_full;
        CUDA_CHECK(cudaGetLastError());
        float core_ms;
        CUDA_CHECK(cudaEventElapsedTime(&core_ms, e0, e1));
        m.core_ms = core_ms;

        for (int q = 0; q < o.lat_queries; ++q) {
            double tq = now_ms();
            CUDA_CHECK(cudaMemcpy(d_q + (size_t)q * D, &queries[(size_t)q * D], D * sizeof(float),
                                  cudaMemcpyHostToDevice));
            row_sqnorm<<<1, 32>>>(d_q + (size_t)q * D, d_qn + q, 1, D);
            compute(q, 1);
            CUDA_CHECK(cudaMemcpy(&idx[(size_t)q * k], d_idx + (size_t)q * k, k * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&dist[(size_t)q * k], d_dist + (size_t)q * k, k * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            m.lat_us.push_back((now_ms() - tq) * 1000.0);
        }

        m.rss_mb = peak_rss_mb();
        if (rep >= 0) {
            if (o.csv) print_csv("cublas", o, 0, 1, T, rep, m);
            results.push_back(m);
        }
    }

    std::printf("GPU: %s, wątki w bloku (selekcja): %d, zapytań na kawałek: %d, kopia bazy + normy (raz): %.2f ms\n",
                prop.name, T, CHUNK, base_ms);
    print_summary("cublas", o, results);
    if (!o.out_path.empty()) write_ranking(o.out_path, Q, k, idx, dist);

    cublasDestroy(handle);
    cudaFree(d_base);
    cudaFree(d_bn);
    cudaFree(d_q);
    cudaFree(d_qn);
    cudaFree(d_dots);
    cudaFree(d_idx);
    cudaFree(d_dist);
    return 0;
}
