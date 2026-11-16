/*
 * topk_cuda.cu - dokładne top-k na GPU, własny kernel.
 *
 * Kompilacja: nvcc -O3 -std=c++17 -arch=sm_89 -I../common topk_cuda.cu -o topk_cuda
 * Użycie:     ./topk_cuda <baza.bin> <zapytania.bin> <N> <D> <Q> <k> [--threads T] [opcje z common.h]
 *
 * Kernel:
 *  - jeden blok obsługuje jedno zapytanie (blockIdx.x = numer zapytania),
 *  - wektor zapytania jest kopiowany do pamięci współdzielonej,
 *  - wątek t przegląda wektory bazy t, t+T, t+2T, ... (T = wątki w bloku),
 *    przy D podzielnym przez 4 czyta po float4,
 *  - każdy wątek trzyma własną posortowaną listę k najlepszych (tablice
 *    lokalne loc_d/loc_i, wstawianie przez przesuwanie),
 *  - listy wątków trafiają do pamięci współdzielonej i są scalane drzewiasto:
 *    w kroku o odstępie s wątek t scala swoją listę z listą t+s (log2 T kroków).
 * Pamięć współdzielona na blok: D*4 + T*k*(4+4) bajtów. RTX 4060 pozwala na
 * 99 KiB na blok, dlatego k <= MAX_K = 64, a gdy lista się nie mieści, program
 * zmniejsza T o połowę (np. k=64 -> 128 wątków) i zapisuje użyte T w CSV.
 *
 * Czasy: base_ms - jednorazowe skopiowanie bazy na GPU (cudaMemcpy),
 * full_ms - partia: zapytania H->D + kernele + wyniki D->H, podzielona na
 * 4 strumienie, żeby kopie jednej części nakładały się na obliczenia innej
 * (--no-stream: jeden strumień), core_ms - osobne uruchomienie samego kernela
 * dla całej partii (zapytania są już na GPU).
 * P95: zapytanie obsługiwane pojedynczo (kopia zapytania, kernel z 1 blokiem,
 * kopia wyniku), czas mierzony zegarem hosta.
 */
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
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

// Wstawia (dist, idx) do posortowanej rosnąco listy k najlepszych; remis po indeksie.
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

// Scala dwie posortowane listy (a i b) i zapisuje k najlepszych do a.
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

// Jeden blok = jedno zapytanie numer q_first + blockIdx.x.
__global__ void topk_kernel(const float *__restrict__ base, const float *__restrict__ queries,
                            int *__restrict__ out_idx, float *__restrict__ out_dist,
                            int N, int D, int k, int q_first) {
    const int q = q_first + blockIdx.x;
    const int tid = threadIdx.x;
    const int T = blockDim.x;

    extern __shared__ float smem[];
    float *s_query = smem;                          // D liczb
    float *s_dist = smem + D;                       // T*k odległości
    int *s_idx = (int *)(s_dist + (size_t)T * k);   // T*k indeksów

    for (int i = tid; i < D; i += T) s_query[i] = queries[(size_t)q * D + i];
    __syncthreads();

    float loc_d[MAX_K];
    int loc_i[MAX_K];
    for (int i = 0; i < k; ++i) {
        loc_d[i] = INFINITY;
        loc_i[i] = N;   // N = "brak kandydata", przegrywa każdy remis
    }

    if (D % 4 == 0) {
        const float4 *q4 = reinterpret_cast<const float4 *>(s_query);
        for (int b = tid; b < N; b += T) {
            const float4 *v4 = reinterpret_cast<const float4 *>(base + (size_t)b * D);
            float sum = 0.0f;
            for (int i = 0; i < D / 4; ++i) {
                float4 a = q4[i], c = v4[i];
                float dx = a.x - c.x, dy = a.y - c.y, dz = a.z - c.z, dw = a.w - c.w;
                sum += dx * dx + dy * dy + dz * dz + dw * dw;
            }
            topk_insert(loc_d, loc_i, k, sum, b);
        }
    } else {
        for (int b = tid; b < N; b += T) {
            const float *v = base + (size_t)b * D;
            float sum = 0.0f;
            for (int i = 0; i < D; ++i) {
                float diff = s_query[i] - v[i];
                sum += diff * diff;
            }
            topk_insert(loc_d, loc_i, k, sum, b);
        }
    }

    for (int i = 0; i < k; ++i) {
        s_dist[(size_t)tid * k + i] = loc_d[i];
        s_idx[(size_t)tid * k + i] = loc_i[i];
    }
    __syncthreads();

    // scalanie drzewiaste: T/2 scaleń, potem T/4, ..., aż wynik jest u wątku 0
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

    // liczba wątków w bloku: domyślnie 256, zmniejszana, gdy listy top-k nie mieszczą się w shared
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int T = o.threads > 0 ? o.threads : 256;
    auto shmem_for = [&](int t) { return (size_t)D * sizeof(float) + (size_t)t * k * (sizeof(float) + sizeof(int)); };
    if (T < 32 || T > 1024 || (T & (T - 1)) != 0) {
        std::fprintf(stderr, "Błąd: liczba wątków w bloku musi być potęgą 2 z zakresu 32-1024\n");
        return 1;
    }
    while (T > 32 && shmem_for(T) > prop.sharedMemPerBlockOptin) T /= 2;
    const size_t shmem = shmem_for(T);
    if (shmem > prop.sharedMemPerBlockOptin) {
        std::fprintf(stderr, "Błąd: za mało pamięci współdzielonej dla D=%d, k=%d\n", D, k);
        return 1;
    }
    CUDA_CHECK(cudaFuncSetAttribute(topk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));

    // pamięć GPU; VRAM = różnica wolnej pamięci przed i po alokacji
    size_t free_before, free_after, total;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
    float *d_base, *d_queries, *d_dist;
    int *d_idx;
    CUDA_CHECK(cudaMalloc(&d_base, (size_t)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_queries, (size_t)Q * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_idx, (size_t)Q * k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dist, (size_t)Q * k * sizeof(float)));
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
    const double vram_mb = (double)(free_before - free_after) / (1024.0 * 1024.0);

    // pamięć hosta przypięta (pinned) - wymagana do asynchronicznych kopii w strumieniach
    float *h_queries, *h_dist;
    int *h_idx;
    CUDA_CHECK(cudaMallocHost(&h_queries, (size_t)Q * D * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_idx, (size_t)Q * k * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_dist, (size_t)Q * k * sizeof(float)));
    std::memcpy(h_queries, queries.data(), (size_t)Q * D * sizeof(float));

    // baza kopiowana na GPU raz
    double tb = now_ms();
    CUDA_CHECK(cudaMemcpy(d_base, base.data(), (size_t)N * D * sizeof(float), cudaMemcpyHostToDevice));
    const double base_ms = now_ms() - tb;

    const int S = o.no_stream ? 1 : 4;
    std::vector<cudaStream_t> streams(S);
    for (cudaStream_t &s : streams) CUDA_CHECK(cudaStreamCreate(&s));
    cudaEvent_t ev_start, ev_k0, ev_k1, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_k0));
    CUDA_CHECK(cudaEventCreate(&ev_k1));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    std::vector<Measurement> results;
    const int chunk = (Q + S - 1) / S;   // zapytań na strumień

    for (int rep = -o.warmup; rep < o.repeat; ++rep) {
        Measurement m;
        m.io_ms = io_ms;
        m.base_ms = base_ms;
        m.vram_mb = vram_mb;

        // partia (czas pełny): w każdym strumieniu kopia części zapytań, kernel
        // i kopia wyników; kopie jednego strumienia nakładają się na kernele innych.
        // Zdarzenie na strumieniu domyślnym czeka na wszystkie strumienie
        // (strumienie z cudaStreamCreate synchronizują się ze strumieniem 0).
        CUDA_CHECK(cudaEventRecord(ev_start, 0));
        for (int s = 0; s < S; ++s) {
            int q0 = s * chunk, nq = std::min(chunk, Q - q0);
            if (nq <= 0) break;
            CUDA_CHECK(cudaMemcpyAsync(d_queries + (size_t)q0 * D, h_queries + (size_t)q0 * D,
                                       (size_t)nq * D * sizeof(float), cudaMemcpyHostToDevice, streams[s]));
            topk_kernel<<<nq, T, shmem, streams[s]>>>(d_base, d_queries, d_idx, d_dist, N, D, k, q0);
            CUDA_CHECK(cudaMemcpyAsync(h_idx + (size_t)q0 * k, d_idx + (size_t)q0 * k,
                                       (size_t)nq * k * sizeof(int), cudaMemcpyDeviceToHost, streams[s]));
            CUDA_CHECK(cudaMemcpyAsync(h_dist + (size_t)q0 * k, d_dist + (size_t)q0 * k,
                                       (size_t)nq * k * sizeof(float), cudaMemcpyDeviceToHost, streams[s]));
        }
        CUDA_CHECK(cudaEventRecord(ev_end, 0));

        // czas rdzenia: sam kernel dla całej partii (zapytania są już na GPU)
        CUDA_CHECK(cudaEventRecord(ev_k0, 0));
        topk_kernel<<<Q, T, shmem>>>(d_base, d_queries, d_idx, d_dist, N, D, k, 0);
        CUDA_CHECK(cudaEventRecord(ev_k1, 0));
        CUDA_CHECK(cudaEventSynchronize(ev_k1));
        CUDA_CHECK(cudaGetLastError());
        float core_ms, full_ms;
        CUDA_CHECK(cudaEventElapsedTime(&core_ms, ev_k0, ev_k1));
        CUDA_CHECK(cudaEventElapsedTime(&full_ms, ev_start, ev_end));
        m.core_ms = core_ms;
        m.full_ms = full_ms;

        // zapytania pojedynczo
        for (int q = 0; q < o.lat_queries; ++q) {
            double tq = now_ms();
            CUDA_CHECK(cudaMemcpy(d_queries + (size_t)q * D, h_queries + (size_t)q * D,
                                  D * sizeof(float), cudaMemcpyHostToDevice));
            topk_kernel<<<1, T, shmem>>>(d_base, d_queries, d_idx, d_dist, N, D, k, q);
            CUDA_CHECK(cudaMemcpy(h_idx + (size_t)q * k, d_idx + (size_t)q * k,
                                  k * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_dist + (size_t)q * k, d_dist + (size_t)q * k,
                                  k * sizeof(float), cudaMemcpyDeviceToHost));
            m.lat_us.push_back((now_ms() - tq) * 1000.0);
        }

        m.rss_mb = peak_rss_mb();
        if (rep >= 0) {
            if (o.csv) print_csv(o.no_stream ? "cuda_bez_strumieni" : "cuda", o, 0, 1, T, rep, m);
            results.push_back(m);
        }
    }

    std::printf("GPU: %s, wątki w bloku: %d, shared: %zu B, kopia bazy (raz): %.2f ms\n",
                prop.name, T, shmem, base_ms);
    print_summary("cuda", o, results);
    if (!o.out_path.empty()) {
        std::vector<int> idx(h_idx, h_idx + (size_t)Q * k);
        std::vector<float> dist(h_dist, h_dist + (size_t)Q * k);
        write_ranking(o.out_path, Q, k, idx, dist);
    }

    for (cudaStream_t &s : streams) cudaStreamDestroy(s);
    cudaFreeHost(h_queries);
    cudaFreeHost(h_idx);
    cudaFreeHost(h_dist);
    cudaFree(d_base);
    cudaFree(d_queries);
    cudaFree(d_idx);
    cudaFree(d_dist);
    return 0;
}
