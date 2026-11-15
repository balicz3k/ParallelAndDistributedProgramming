/* Lab 4, punkt 4: rozgłaszanie liczby przez MPI_Bcast.
 * Proces 0 wczytuje liczbę, MPI_Bcast przekazuje ją wszystkim procesom,
 * a każdy proces wypisuje swój numer i otrzymaną wartość. Pętla trwa,
 * dopóki liczba nie jest ujemna (ujemna też jest rozgłaszana, więc
 * wszystkie procesy kończą w tym samym obrocie pętli).
 *
 * Uruchomienie: printf "5\n7\n-1\n" | mpirun -np 4 ./bcast
 */
#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank, size, value;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    while (1) {
        if (rank == 0) {
            printf("Podaj liczbę:\n");
            fflush(stdout);
            if (scanf("%d", &value) != 1)
                value = -1;
        }

        MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (value < 0)
            break;

        /* kolejność wydruków z różnych procesów nie jest ustalona -
         * mpirun przekazuje je na ekran w kolejności nadejścia */
        printf("Proces %d z %d otrzymal liczbe %d\n", rank, size, value);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
