#ifndef LIB_RUNTIME_H_
#define LIB_RUNTIME_H_
#include "array.h"
#include "grid_view.h"
#include "grid_mpi.h"
#include "grid_view_cuda.h"
#include "grid_view_cpu.h"
#include "grid_cpu.h"
#include "grid_cuda.h"
#include <vector>
using namespace std;

class StencilRuntime
{
public:
	StencilRuntime(int num_dims, int unit_size, const IndexArray &global_size, int proc_num_dims, IntArray proc_size, int stencil_width, int num_iters);

	void StencilInit();    	//do some MPI init and object initializations
	
	//splits the per-node grid to device-grids
	void create_grids();
	void clean_grids();
	void StencilBegin();
	void StencilFinalize(); // do some MPI finalize

	int num_dims(){return num_dims_;}
	int unit_size(){return unit_size_;}
	int stencil_width(){return stencil_width_;}

	GridCPU *grid_cpu(){return grid_cpu_;}
	GridCuda *grid_cuda(int id){return grid_cuda_[id];}
	GridMPI *grid(){return grid_;}
	Grid_view *gv(){return gv_;}
	int my_rank(){return my_rank_;}
	int along(){return along_;}
	void set_stencil_idx(int idx){stencil_idx_ = idx;};
	int get_stencil_idx(){return stencil_idx_;}

protected:
	int num_dims_;
	int unit_size_;
	IndexArray global_size_;
	int proc_num_dims_;
	IntArray proc_size_; 
	int stencil_width_;
	int my_rank_;
	int along_;
	int num_iters_;
	int current_iter_;

	///////////////device related/////////////////
	int num_devices_;
	int num_gpus_;

	Grid_view *gv_;
	Grid_view_cpu *gv_cpu_;
	Grid_view_cuda **gv_cuda_;

	GridMPI *grid_;
	GridCPU *grid_cpu_; 	//the part for cpu processing
	GridCuda **grid_cuda_; 	//cuda grids

	int stencil_idx_;

	vector<struct Tile> *internal_tiles;
	vector<struct Tile> *border_tiles;
	vector<MPI_Request> *requests;

	int *along_partitions_;	
	int *starts_;
	double *speeds_;

	static void *launch(void *arg);
	//partitions the per-node grid to cpu and gpu grids
	void partition();

	void along_dim();

	//split the grid_ and copy the sub data into sub grids
	void init_grids();

	void split();

	void tile_grids();

	void interdevice_exchange();

	void process_internal();

	void process_border();

	void profile_iter(); //first iteration used to profile speed of each device
};

#endif
