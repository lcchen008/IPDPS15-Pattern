#include "lib/grid_mpi.h"
#include "lib/grid_view.h"
#include "lib/array.h"
#include <iostream>
#include "lib/util.h"
#include "lib/test/test_util.h"
using namespace std;

#define DIMS 3
#define N 1024 
#define NPROC 8

static void init_grid(GridMPI *g, int my_rank) {
  int idx = 0;
  float v = N * N * N* my_rank;
  if (g->num_dims() == 3) {
    for (int i = 0; i < g->my_size()[0]; ++i) {
      for (int j = 0; j < g->my_size()[1]; ++j) {
        for (int k = 0; k < g->my_size()[2]; ++k) {
          ((float*)(g->data()))[idx] = v;
          ++v;
          ++idx;
        }
      }
    }
  } else if  (g->num_dims() == 2) {
    for (int i = 0; i < g->my_size()[0]; ++i) {
      for (int j = 0; j < g->my_size()[1]; ++j) {
        ((float*)(g->data()))[idx] = v;
        ++v;
        ++idx;
      }
    }
  } else if  (g->num_dims() == 1) {
    for (int i = 0; i < g->my_size()[0]; ++i) {
      ((float*)(g->data()))[idx] = v;
      ++v;
      ++idx;
    }
  } else {
    exit(1);
  }
}

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	int my_rank;
  	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  	int num_procs;
  	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	IndexArray global_size(N, N, N);
	IntArray proc_size(2, 2, 2);

	Grid_view gv(DIMS, global_size, DIMS, proc_size, my_rank);	
	//cout<<"rank: "<<my_rank<<endl;
	GridMPI *g = gv.CreateGrid(sizeof(float), DIMS, global_size, 1);
	
	init_grid(g, my_rank);

	
	UnsignedArray halo(1, 1, 1);

	Width2 w = {halo, halo};

	gv.ExchangeBoundaries(g, w, false, false);

	//print_grid<float>(g, my_rank, cerr);

	MPI_Finalize();
}
