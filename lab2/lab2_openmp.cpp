/*
 * PRiR Laboratorium 2 — OpenMP
 * Przetwarzanie chmury punktów 3D
 *
 * Kompilacja: g++ -O2 -fopenmp -std=c++17 lab2_openmp.cpp -o lab2
 * Uruchomienie: ./lab2 <plik_danych> [num_threads]
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <omp.h>

// ============================================================
// Struktury i typy
// ============================================================

struct Point {
    float x, y, z;
};

// ============================================================
// Wczytywanie danych
// ============================================================

std::vector<Point> load_points(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Błąd: nie można otworzyć pliku: " << filename << "\n";
        std::exit(1);
    }

    std::vector<Point> points;
    points.reserve(5'000'000); // wstępna rezerwacja ~5M punktów

    Point p;
    while (file >> p.x >> p.y >> p.z) {
        points.push_back(p);
    }

    std::cout << "Wczytano " << points.size() << " punktów z pliku: " << filename << "\n\n";
    return points;
}

// ============================================================
// Zadanie 1: Centroid
// ============================================================

void task1_centroid(const std::vector<Point>& pts,
                    double& cx, double& cy, double& cz) {
    std::cout << "=== Zadanie 1: Centroid ===\n";
    const long long N = static_cast<long long>(pts.size());

    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;

    double t0 = omp_get_wtime();

    #pragma omp parallel for schedule(static) \
        reduction(+:sum_x, sum_y, sum_z)
    for (long long i = 0; i < N; ++i) {
        sum_x += pts[i].x;
        sum_y += pts[i].y;
        sum_z += pts[i].z;
    }

    cx = sum_x / N;
    cy = sum_y / N;
    cz = sum_z / N;

    double t1 = omp_get_wtime();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Centroid: (" << cx << ", " << cy << ", " << cz << ")\n";
    std::cout << "Czas: " << std::setprecision(4) << (t1 - t0) << " s\n\n";
}

// ============================================================
// Zadanie 2: Min/max odległości od centroidu
// ============================================================

void task2_minmax(const std::vector<Point>& pts,
                  double cx, double cy, double cz,
                  double& dmin, double& dmax) {
    std::cout << "=== Zadanie 2: Min/max odległości od centroidu ===\n";
    const long long N = static_cast<long long>(pts.size());

    dmin = 1e30;
    dmax = 0.0;

    double t0 = omp_get_wtime();

    #pragma omp parallel for schedule(static) \
        reduction(min:dmin) reduction(max:dmax)
    for (long long i = 0; i < N; ++i) {
        double dx = pts[i].x - cx;
        double dy = pts[i].y - cy;
        double dz = pts[i].z - cz;
        double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (dist < dmin) dmin = dist;
        if (dist > dmax) dmax = dist;
    }

    double t1 = omp_get_wtime();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "dmin = " << dmin << "\n";
    std::cout << "dmax = " << dmax << "\n";
    std::cout << "Czas: " << std::setprecision(4) << (t1 - t0) << " s\n\n";
}

// ============================================================
// Zadanie 3: Normalizacja do kostki [-1, 1]^3
// ============================================================

void task3_normalize(std::vector<Point>& pts,
                     double cx, double cy, double cz, double dmax) {
    std::cout << "=== Zadanie 3: Normalizacja do kostki [-1,1]^3 ===\n";
    const long long N = static_cast<long long>(pts.size());

    // Każdy punkt: (xi - cx) / dmax  → wynik w [-1, 1]^3
    // dmax to maksymalna odległość od centroidu, więc dzieląc przez dmax
    // wszystkie punkty mieszczą się w sferze jednostkowej (a tym samym w kostce)

    double t0 = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < N; ++i) {
        pts[i].x = static_cast<float>((pts[i].x - cx) / dmax);
        pts[i].y = static_cast<float>((pts[i].y - cy) / dmax);
        pts[i].z = static_cast<float>((pts[i].z - cz) / dmax);
    }

    double t1 = omp_get_wtime();

    // Wypisz kilka przykładowych punktów po normalizacji
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Przykładowe znormalizowane punkty:\n";
    long long step = std::max(1LL, N / 5);
    for (long long i = 0; i < N && i < 5 * step; i += step) {
        std::cout << "  pts[" << i << "] = ("
                  << pts[i].x << ", "
                  << pts[i].y << ", "
                  << pts[i].z << ")\n";
    }
    std::cout << "Czas: " << std::setprecision(4) << (t1 - t0) << " s\n\n";
}

// ============================================================
// Zadanie 4: Liczenie sąsiadów w promieniu r (na próbce)
// ============================================================

void task4_neighbors(const std::vector<Point>& pts, double dmax) {
    std::cout << "=== Zadanie 4: Liczenie sąsiadów w promieniu r ===\n";

    // Próbka — O(N²) jest zbyt wolne dla kilku milionów punktów
    const long long SAMPLE = std::min((long long)pts.size(), 100'000LL);

    // r = 0.1 * dmax (punkty są już znormalizowane, więc dmax ≈ 1.0 po normalizacji)
    // Używamy stałego r = 0.1 (odpowiada 10% jednostkowego promienia)
    const double r = 0.1;
    const double r2 = r * r;

    std::cout << "Próbka: " << SAMPLE << " punktów\n";
    std::cout << "Promień r = " << r << "\n";

    std::vector<long long> neighbor_count(SAMPLE, 0);

    double t0 = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic, 256)
    for (long long i = 0; i < SAMPLE; ++i) {
        long long cnt = 0;
        double xi = pts[i].x, yi = pts[i].y, zi = pts[i].z;
        for (long long j = 0; j < SAMPLE; ++j) {
            if (i == j) continue;
            double dx = pts[j].x - xi;
            double dy = pts[j].y - yi;
            double dz = pts[j].z - zi;
            if (dx*dx + dy*dy + dz*dz <= r2) {
                ++cnt;
            }
        }
        neighbor_count[i] = cnt;
    }

    double t1 = omp_get_wtime();

    // Średnia liczba sąsiadów
    double avg = 0.0;
    for (long long i = 0; i < SAMPLE; ++i) avg += neighbor_count[i];
    avg /= SAMPLE;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Średnia liczba sąsiadów: " << avg << "\n";
    std::cout << "Czas: " << std::setprecision(4) << (t1 - t0) << " s\n\n";
}

// ============================================================
// main
// ============================================================

int main(int argc, char* argv[]) {
    // --- Argumenty ---
    if (argc < 2) {
        std::cerr << "Użycie: " << argv[0] << " <plik_danych> [num_threads]\n";
        return 1;
    }
    std::string filename = argv[1];

    if (argc >= 3) {
        int nt = std::atoi(argv[2]);
        if (nt > 0) omp_set_num_threads(nt);
    }

    std::cout << "============================================\n";
    std::cout << "PRiR Lab2 — Chmura Punktów (OpenMP)\n";
    std::cout << "============================================\n";
    std::cout << "Liczba wątków: " << omp_get_max_threads() << "\n\n";

    // --- Wczytaj dane ---
    std::vector<Point> pts = load_points(filename);
    if (pts.empty()) {
        std::cerr << "Błąd: brak punktów w pliku.\n";
        return 1;
    }

    // --- Zadania ---
    double cx, cy, cz;
    task1_centroid(pts, cx, cy, cz);

    double dmin, dmax;
    task2_minmax(pts, cx, cy, cz, dmin, dmax);

    task3_normalize(pts, cx, cy, cz, dmax);
    // Po normalizacji dmax ≈ 1.0 — przekazujemy 1.0 do zadania 4
    task4_neighbors(pts, 1.0);

    std::cout << "============================================\n";
    std::cout << "Wszystkie zadania zakończone.\n";
    std::cout << "============================================\n";

    return 0;
}
