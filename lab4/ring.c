/* Lab 4, punkt 3: przekazywanie liczby w pierścieniu.
 * Proces 0 wczytuje liczbę i wysyła ją do procesu 1, proces i przekazuje ją
 * do i+1, a ostatni odsyła ją do procesu 0. Pętla trwa, dopóki liczba nie
 * jest ujemna. Liczba ujemna też obiega pierścień (bez wypisywania), dzięki
 * czemu każdy proces wie, że ma zakończyć działanie, i nikt nie czeka
 * w MPI_Recv.
 *
 * Uruchomienie: printf "5\n7\n-1\n" | mpirun -np 4 ./ring
 */
#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank, size, value;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0)
            printf("Potrzebne są co najmniej 2 procesy\n");
        MPI_Finalize();
        return 1;
    }

    int next = (rank + 1) % size;          /* do kogo wysyłam */
    int prev = (rank - 1 + size) % size;   /* od kogo odbieram */

    while (1) {
        if (rank == 0) {
            printf("Podaj liczbę:\n");
            fflush(stdout);
            if (scanf("%d", &value) != 1)
                value = -1;                /* koniec wejścia traktuję jak liczbę ujemną */

            MPI_Send(&value, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
            MPI_Recv(&value, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (value < 0)
                break;
            printf("Proces %d dostal %d od procesu %d\n", rank, value, prev);
            fflush(stdout);
        } else {
            MPI_Recv(&value, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (value >= 0) {
                printf("Proces %d dostal %d od procesu %d\n", rank, value, prev);
                fflush(stdout);
            }
            MPI_Send(&value, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
            if (value < 0)
                break;
        }
    }

    MPI_Finalize();
    return 0;
}
