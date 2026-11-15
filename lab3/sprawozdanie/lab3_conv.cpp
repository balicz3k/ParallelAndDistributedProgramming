#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>
#include <omp.h>

// ---------------------------------------------------------------
// Wypełnienie macierzy losowymi wartościami z rozkładu [-1, 1]
// ---------------------------------------------------------------
void fill_random(std::vector<float>& v, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

// ---------------------------------------------------------------
// Normalizacja jądra: dziel przez sumę bezwzględnych wartości
// (unika overflow przy dużych jądrach)
// ---------------------------------------------------------------
void normalize_kernel(std::vector<float>& kernel) {
    float s = 0.0f;
    for (float x : kernel) s += std::fabs(x);
    if (s > 1e-9f)
        for (float& x : kernel) x /= s;
}

// ---------------------------------------------------------------
// 1) Wersja sekwencyjna
// ---------------------------------------------------------------
void conv2d_seq(const std::vector<float>& matrix,
                const std::vector<float>& kernel,
                std::vector<float>& result,
                int H, int W, int Kh, int Kw)
{
    int hk = Kh / 2;
    int wk = Kw / 2;

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            float sum = 0.0f;
            for (int m = 0; m < Kh; ++m) {
                for (int n = 0; n < Kw; ++n) {
                    int x = i + m - hk;
                    int y = j + n - wk;
                    if (x >= 0 && x < H && y >= 0 && y < W) {
                        sum += matrix[x * W + y] * kernel[m * Kw + n];
                    }
                }
            }
            result[i * W + j] = sum;
        }
    }
}

// ---------------------------------------------------------------
// 2) Wersja równoległa — parallel for na pętli zewnętrznej (po i)
// ---------------------------------------------------------------
void conv2d_omp(const std::vector<float>& matrix,
                const std::vector<float>& kernel,
                std::vector<float>& result,
                int H, int W, int Kh, int Kw,
                int num_threads)
{
    int hk = Kh / 2;
    int wk = Kw / 2;

#pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            float sum = 0.0f;
            for (int m = 0; m < Kh; ++m) {
                for (int n = 0; n < Kw; ++n) {
                    int x = i + m - hk;
                    int y = j + n - wk;
                    if (x >= 0 && x < H && y >= 0 && y < W) {
                        sum += matrix[x * W + y] * kernel[m * Kw + n];
                    }
                }
            }
            result[i * W + j] = sum;
        }
    }
}

// ---------------------------------------------------------------
// 3) Wariant z collapse(2) — zrównoleglenie obu pętli zewnętrznych
// ---------------------------------------------------------------
void conv2d_omp_collapse(const std::vector<float>& matrix,
                         const std::vector<float>& kernel,
                         std::vector<float>& result,
                         int H, int W, int Kh, int Kw,
                         int num_threads)
{
    int hk = Kh / 2;
    int wk = Kw / 2;

#pragma omp parallel for collapse(2) num_threads(num_threads) schedule(static)
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            float sum = 0.0f;
            for (int m = 0; m < Kh; ++m) {
                for (int n = 0; n < Kw; ++n) {
                    int x = i + m - hk;
                    int y = j + n - wk;
                    if (x >= 0 && x < H && y >= 0 && y < W) {
                        sum += matrix[x * W + y] * kernel[m * Kw + n];
                    }
                }
            }
            result[i * W + j] = sum;
        }
    }
}

// ---------------------------------------------------------------
// 4) Wariant z schedule(dynamic, 64)
// ---------------------------------------------------------------
void conv2d_omp_schedule(const std::vector<float>& matrix,
                         const std::vector<float>& kernel,
                         std::vector<float>& result,
                         int H, int W, int Kh, int Kw,
                         int num_threads)
{
    int hk = Kh / 2;
    int wk = Kw / 2;

#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 64)
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            float sum = 0.0f;
            for (int m = 0; m < Kh; ++m) {
                for (int n = 0; n < Kw; ++n) {
                    int x = i + m - hk;
                    int y = j + n - wk;
                    if (x >= 0 && x < H && y >= 0 && y < W) {
                        sum += matrix[x * W + y] * kernel[m * Kw + n];
                    }
                }
            }
            result[i * W + j] = sum;
        }
    }
}

// ---------------------------------------------------------------
// Sprawdzenie poprawności — max różnica między dwoma wynikami
// ---------------------------------------------------------------
bool verify(const std::vector<float>& ref,
            const std::vector<float>& test,
            float tol = 1e-4f)
{
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::fabs(ref[i] - test[i]) > tol) return false;
    }
    return true;
}

// ---------------------------------------------------------------
// main
// ---------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Użycie: " << argv[0]
                  << " <rozmiar_macierzy> <rozmiar_jadra> [num_threads]\n"
                  << "Przykład: " << argv[0] << " 2048 7 8\n";
        return 1;
    }

    int N          = std::atoi(argv[1]);   // rozmiar macierzy N×N
    int K          = std::atoi(argv[2]);   // rozmiar jądra K×K
    int num_threads = (argc >= 4) ? std::atoi(argv[3]) : omp_get_max_threads();

    // Jądro musi być nieparzyste
    if (K % 2 == 0) { std::cerr << "Rozmiar jądra musi być nieparzysty.\n"; return 1; }

    int H = N, W = N, Kh = K, Kw = K;
    long long total = (long long)H * W;

    std::cout << "Rozmiar: " << H << "x" << W
              << ", Jądro: " << Kh << "x" << Kw
              << ", Wątki: " << num_threads << "\n";

    // Alokacja
    std::vector<float> matrix(total), kernel(Kh * Kw);
    std::vector<float> res_seq(total), res_omp(total),
                       res_collapse(total), res_dyn(total);

    fill_random(matrix, 42);
    fill_random(kernel, 7);
    normalize_kernel(kernel);

    // --- Sekwencja ---
    double t0 = omp_get_wtime();
    conv2d_seq(matrix, kernel, res_seq, H, W, Kh, Kw);
    double seq_time = omp_get_wtime() - t0;
    std::printf("Seq:     %8.3f s\n", seq_time);

    // --- OMP basic ---
    t0 = omp_get_wtime();
    conv2d_omp(matrix, kernel, res_omp, H, W, Kh, Kw, num_threads);
    double omp_time = omp_get_wtime() - t0;
    bool ok_omp = verify(res_seq, res_omp);
    std::printf("OMP:     %8.3f s  speedup: %.2fx  poprawność: %s\n",
                omp_time, seq_time / omp_time, ok_omp ? "OK" : "BLAD");

    // --- Collapse ---
    t0 = omp_get_wtime();
    conv2d_omp_collapse(matrix, kernel, res_collapse, H, W, Kh, Kw, num_threads);
    double col_time = omp_get_wtime() - t0;
    bool ok_col = verify(res_seq, res_collapse);
    std::printf("Collapse:%8.3f s  speedup: %.2fx  poprawność: %s\n",
                col_time, seq_time / col_time, ok_col ? "OK" : "BLAD");

    // --- Dynamic ---
    t0 = omp_get_wtime();
    conv2d_omp_schedule(matrix, kernel, res_dyn, H, W, Kh, Kw, num_threads);
    double dyn_time = omp_get_wtime() - t0;
    bool ok_dyn = verify(res_seq, res_dyn);
    std::printf("Dynamic: %8.3f s  speedup: %.2fx  poprawność: %s\n",
                dyn_time, seq_time / dyn_time, ok_dyn ? "OK" : "BLAD");

    return (ok_omp && ok_col && ok_dyn) ? 0 : 1;
}
