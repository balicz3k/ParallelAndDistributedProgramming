/* Lab 4, punkt 2: program przykładowy MPI.
 * Program „hello” z wykładu 3 (MPI_Init, MPI_Comm_rank, MPI_Comm_size),
 * dodatkowo wypisuje nazwę węzła, na którym działa proces.
 *
 * Kompilacja:   mpicc przyklad.c -o prz_mpi
 * Uruchomienie: mpirun -np 7 --display-map ./prz_mpi
 */
#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank, size, name_len;
    char name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(name, &name_len);

    printf("Moj numer %d, w grupie %d procesow, wezel %s\n", rank, size, name);

    MPI_Finalize();
    return 0;
}
