/*
 * PRiR Laboratorium 3 - rownolegla konwolucja dwuwymiarowa (OpenMP)
 *
 * Program generuje losowa macierz N x N oraz losowe jadro K x K (K nieparzyste),
 * liczy konwolucje z zerowym dopelnieniem poza obrazem i porownuje:
 *   seq      - wersja sekwencyjna (baza czasu i wynik wzorcowy)
 *   outer    - parallel for po wierszach i (schedule(static))
 *   collapse - parallel for collapse(2) po parach (i,j) (schedule(static))
 *   dynamic  - parallel for po wierszach, schedule(dynamic, 64)
 *   inner    - petla zewnetrzna i sekwencyjna, rownolegla jest wewnetrzna petla po j
 *              (tworzenie zespolu watkow dla kazdego wiersza)
 * Kazdy wariant rownolegly jest sprawdzany z wynikiem sekwencyjnym (maks. roznica).
 *
 * Kompilacja: g++ -O3 -fopenmp -std=c++17 -Wall -Wextra lab3_conv.cpp -o lab3
 * Uruchomienie: ./lab3 <N> <K> [powtorzenia=5]
 * Wyjscie (stdout): CSV  N,K,variant,threads,rep,time_s  oraz linie '#' z wynikiem
 * sprawdzenia poprawnosci. Dla kazdej konfiguracji wykonywana jest jedna rozgrzewka
 * (nie mierzona), potem <powtorzenia> pomiarow omp_get_wtime().
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <omp.h>

typedef std::vector<float> Mat;

// Wypelnienie wektora losowymi wartosciami z [-1, 1] (ziarno zapisane w wywolaniu)
static void fill_random(Mat& v, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

// Normalizacja jadra: dzielenie przez sume wartosci bezwzglednych
static void normalize_kernel(Mat& kernel) {
    float s = 0.0f;
    for (float x : kernel) s += std::fabs(x);
    if (s > 1e-9f)
        for (float& x : kernel) x /= s;
}

// Wynik konwolucji w jednym pikselu (i,j); kolejnosc sumowania taka sama we wszystkich
// wariantach, wiec wyniki rownolegle powinny byc identyczne z sekwencyjnym.
// noinline: wszystkie warianty (takze sekwencyjny) wywoluja ten sam skompilowany kod;
// inaczej kompilator inline'uje i rozwija petle inaczej w kazdym wariancie
// (wersja sekwencyjna byla wtedy ok. 30% wolniejsza od rownoleglej na 1 watku).
static __attribute__((noinline)) float pixel(const float* M, const float* J, int i, int j,
                          int H, int W, int K) {
    const int hk = K / 2;
    float sum = 0.0f;
    for (int m = 0; m < K; ++m) {
        for (int n = 0; n < K; ++n) {
            int x = i + m - hk;
            int y = j + n - hk;
            if (x >= 0 && x < H && y >= 0 && y < W)
                sum += M[x * W + y] * J[m * K + n];
        }
    }
    return sum;
}

// 1) Sekwencyjnie
static void conv_seq(const Mat& M, const Mat& J, Mat& R, int H, int W, int K) {
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
            R[i * W + j] = pixel(M.data(), J.data(), i, j, H, W, K);
}

// 2) Zrownoleglona petla zewnetrzna (wiersze), podzial statyczny
static void conv_outer(const Mat& M, const Mat& J, Mat& R, int H, int W, int K, int p) {
#pragma omp parallel for num_threads(p) schedule(static)
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
            R[i * W + j] = pixel(M.data(), J.data(), i, j, H, W, K);
}

// 3) collapse(2): obie petle po obrazie sklejone w jedna przestrzen H*W iteracji
static void conv_collapse(const Mat& M, const Mat& J, Mat& R, int H, int W, int K, int p) {
#pragma omp parallel for collapse(2) num_threads(p) schedule(static)
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
            R[i * W + j] = pixel(M.data(), J.data(), i, j, H, W, K);
}

// 4) Petla zewnetrzna z przydzialem dynamicznym (porcje po 64 wiersze)
static void conv_dynamic(const Mat& M, const Mat& J, Mat& R, int H, int W, int K, int p) {
#pragma omp parallel for num_threads(p) schedule(dynamic, 64)
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
            R[i * W + j] = pixel(M.data(), J.data(), i, j, H, W, K);
}

// 5) Zrownoleglona petla wewnetrzna (po j): zespol watkow tworzony dla kazdego wiersza
static void conv_inner(const Mat& M, const Mat& J, Mat& R, int H, int W, int K, int p) {
    for (int i = 0; i < H; ++i) {
#pragma omp parallel for num_threads(p) schedule(static)
        for (int j = 0; j < W; ++j)
            R[i * W + j] = pixel(M.data(), J.data(), i, j, H, W, K);
    }
}

// Maksymalna roznica bezwzgledna miedzy dwoma wynikami
static double max_diff(const Mat& a, const Mat& b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size(); ++i) d = std::max(d, (double)std::fabs(a[i] - b[i]));
    return d;
}

typedef void (*ParFn)(const Mat&, const Mat&, Mat&, int, int, int, int);

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Uzycie: " << argv[0] << " <N> <K> [powtorzenia]\n"
                  << "Przyklad: " << argv[0] << " 2048 7 5\n";
        return 1;
    }
    const int N = std::atoi(argv[1]);
    const int K = std::atoi(argv[2]);
    const int reps = (argc >= 4) ? std::atoi(argv[3]) : 5;
    if (N < 1 || K < 1 || K % 2 == 0 || reps < 1) {
        std::cerr << "Blad: N>=1, K nieparzyste, powtorzenia>=1.\n";
        return 1;
    }
    const int H = N, W = N;

    Mat M((size_t)H * W), J((size_t)K * K), ref((size_t)H * W), R((size_t)H * W);
    fill_random(M, 42);   // ziarno macierzy
    fill_random(J, 7);    // ziarno jadra
    normalize_kernel(J);

    std::printf("N,K,variant,threads,rep,time_s\n");

    // --- sekwencyjnie: rozgrzewka + pomiary (wynik ostatniego uruchomienia = wzorzec)
    conv_seq(M, J, ref, H, W, K);
    for (int r = 0; r < reps; ++r) {
        double t0 = omp_get_wtime();
        conv_seq(M, J, ref, H, W, K);
        std::printf("%d,%d,seq,1,%d,%.6f\n", N, K, r, omp_get_wtime() - t0);
    }

    struct Variant { const char* name; ParFn fn; };
    const Variant variants[] = {
        {"outer", conv_outer}, {"collapse", conv_collapse},
        {"dynamic", conv_dynamic}, {"inner", conv_inner}
    };
    const int threads[] = {1, 2, 4, 8, 16};

    bool all_ok = true;
    for (const Variant& v : variants) {
        for (int p : threads) {
            // rozgrzewka, jednoczesnie sprawdzenie poprawnosci wzgledem wersji sekwencyjnej
            std::fill(R.begin(), R.end(), 0.0f);
            v.fn(M, J, R, H, W, K, p);
            double d = max_diff(ref, R);
            bool ok = d <= 1e-4;
            all_ok = all_ok && ok;
            std::printf("# poprawnosc N=%d K=%d %s p=%d max|roznica|=%.3g %s\n",
                        N, K, v.name, p, d, ok ? "OK" : "BLAD");
            for (int r = 0; r < reps; ++r) {
                double t0 = omp_get_wtime();
                v.fn(M, J, R, H, W, K, p);
                std::printf("%d,%d,%s,%d,%d,%.6f\n", N, K, v.name, p, r,
                            omp_get_wtime() - t0);
            }
        }
    }
    return all_ok ? 0 : 1;
}
