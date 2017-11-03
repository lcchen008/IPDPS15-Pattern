#include "lib/grid_mpi.h"
#include "lib/grid_view.h"
#include "lib/array.h"
#include <iostream>
#include "lib/util.h"
#include "lib/test/test_util.h"
#include "lib/stencil_runtime.h"
#include "lib/stencil_creator.h"
#include "irregular/irregular_runtime.h"
#include "include/runtime.h"

using namespace std;

#define DIMS 3
#define N 512 
#define NPROC 2 
#define W 1 

static void init_grid(GridMPI *g, int my_rank) {
  int idx = 0;
  double v = N * N * N* my_rank;
  if (g->num_dims() == 3) {
    for (int i = 0; i < g->my_size()[0]; ++i) {
      for (int j = 0; j < g->my_size()[1]; ++j) {
        for (int k = 0; k < g->my_size()[2]; ++k) {
          ((double*)(g->data()))[idx] = v;
          ++v;
          ++idx;
        }
      }
    }
  } else if  (g->num_dims() == 2) {
    for (int i = 0; i < g->my_size()[0]; ++i) {
      for (int j = 0; j < g->my_size()[1]; ++j) {
        ((double*)(g->data()))[idx] = v;
        ++v;
        ++idx;
      }
    }
  } else if  (g->num_dims() == 1) {
    for (int i = 0; i < g->my_size()[0]; ++i) {
      ((double*)(g->data()))[idx] = v;
      ++v;
      ++idx;
    }
  } else {
    exit(1);
  }
}

int main(int argc, char *argv[])
{
	IndexArray global_size(N, N, N);
	IntArray proc_size(1, 4, 8);

	double *parameters = (double *)malloc(sizeof(double)*5);

	RuntimeInit(argc, argv);
	StencilCreator<double> sc(3, global_size, 3, proc_size, W, 1, false, false, parameters, sizeof(double)*5);

	StencilRuntime *sr = sc.CreateRuntime();

	sr->StencilInit();

	//float *data = (float *)malloc(sizeof(float)*N*N*N);
	//gendata(data);
	//sr->copy_in(data);
	//sr->StencilBegin();
	//sr->copy_out(data);
	//sr->StencilFinalize();

	GridMPI *g = sr->grid(); 

	int my_rank = sr->my_rank(); 
	init_grid(g, my_rank);

	sr->StencilBegin();
	//sr->StencilFinalize();
	RuntimeFinalize();

	delete sr;
}
