/* Lab 4, punkt 2: program przykładowy MPI.
 * Każdy proces wypisuje swój numer (rank) i liczbę wszystkich procesów (size).
 *
 * Kompilacja:   mpicc przyklad.c -o prz_mpi
 * Uruchomienie: mpirun -np 7 --display-map ./prz_mpi
 */
#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank, size;

    MPI_Init(&argc, &argv);                  /* start MPI */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    /* mój numer */
    MPI_Comm_size(MPI_COMM_WORLD, &size);    /* ile jest procesów */

    printf("Moj numer %d, w grupie %d procesow\n", rank, size);

    MPI_Finalize();
    return 0;
}
