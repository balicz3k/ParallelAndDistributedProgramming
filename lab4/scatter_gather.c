/* Lab 4, punkt 5: suma tablicy przez Scatter/Gather z pomiarem czasu.
 *
 * a) proces 0 tworzy tablicę N liczb losowych 0-100 (srand(42), domyślnie N = 10000),
 * b) tablica jest dzielona między p procesów; gdy N dzieli się przez p,
 *    używam MPI_Scatter, w przeciwnym razie MPI_Scatterv (pierwsze N % p
 *    procesów dostaje o jeden element więcej),
 * c) każdy proces liczy sumę swojej części,
 * d) MPI_Gather zbiera sumy częściowe w procesie 0,
 * e) proces 0 dodaje sumy częściowe,
 * f) proces 0 wypisuje sumę całkowitą, sumy z procesów i porównanie
 *    z sumą policzoną sekwencyjnie,
 * g) czas od MPI_Init do końca MPI_Gather (MPI_Wtime). Zaraz po MPI_Init jest
 *    MPI_Barrier, żeby wszystkie procesy zaczynały pomiar razem; jako wynik
 *    biorę najdłuższy czas spośród procesów (MPI_Reduce z MPI_MAX).
 *
 * Uruchomienie: mpirun -np 4 ./scatter_gather [N] [csv]
 *   N   - rozmiar tablicy (domyślnie 10000)
 *   csv - zamiast pełnego wydruku jedna linia: p,time_s,total_sum,check
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 10000;
    if (argc > 1)
        n = atoi(argv[1]);
    int csv = (argc > 2 && strcmp(argv[2], "csv") == 0);

    /* a) tablica tylko w procesie 0 */
    int *data = NULL;
    if (rank == 0) {
        data = malloc((size_t)n * sizeof(int));
        srand(42);
        for (int i = 0; i < n; i++)
            data[i] = rand() % 101;        /* 0..100 */
    }

    /* b) ile elementów dostaje każdy proces i od którego miejsca tablicy */
    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int offset = 0;
    for (int i = 0; i < size; i++) {
        counts[i] = n / size + (i < n % size ? 1 : 0);
        displs[i] = offset;
        offset += counts[i];
    }

    int my_count = counts[rank];
    int *part = malloc(my_count * sizeof(int));

    if (n % size == 0)
        MPI_Scatter(data, my_count, MPI_INT, part, my_count, MPI_INT, 0, MPI_COMM_WORLD);
    else
        MPI_Scatterv(data, counts, displs, MPI_INT, part, my_count, MPI_INT, 0, MPI_COMM_WORLD);

    /* c) suma lokalna */
    long long local_sum = 0;
    for (int i = 0; i < my_count; i++)
        local_sum += part[i];

    /* d) sumy częściowe do procesu 0 */
    long long *sums = NULL;
    if (rank == 0)
        sums = malloc(size * sizeof(long long));
    MPI_Gather(&local_sum, 1, MPI_LONG_LONG, sums, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    /* g) koniec komunikacji = koniec pomiaru */
    double my_time = MPI_Wtime() - start;
    double time;
    MPI_Reduce(&my_time, &time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        /* e) suma całkowita */
        long long total = 0;
        for (int i = 0; i < size; i++)
            total += sums[i];

        /* kontrola: suma policzona sekwencyjnie z całej tablicy */
        long long seq = 0;
        for (int i = 0; i < n; i++)
            seq += data[i];
        const char *check = (total == seq) ? "OK" : "BŁĄD";

        /* f) wydruk */
        if (csv) {
            printf("%d,%.9f,%lld,%s\n", size, time, total, check);
        } else {
            printf("Liczba procesów: %d, N = %d, %s\n", size, n,
                   n % size == 0 ? "MPI_Scatter" : "MPI_Scatterv");
            for (int i = 0; i < size; i++)
                printf("Proces %d: %d elementów, suma %lld\n", i, counts[i], sums[i]);
            printf("Suma całkowita: %lld (sekwencyjnie %lld) %s\n", total, seq, check);
            printf("Czas: %.6f s\n", time);
        }
        free(sums);
        free(data);
    }

    free(part);
    free(counts);
    free(displs);
    MPI_Finalize();
    return 0;
}
