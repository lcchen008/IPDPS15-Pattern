#include "runtime.h"
#include <mpi.h>

void RuntimeInit(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
}

void RuntimeFinalize()
{
	MPI_Finalize();	
}
