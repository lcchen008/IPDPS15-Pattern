#ifndef LIB_PARTITIONER_H_
#define LIB_PARTITIONER_H_

/**
 * Used to map one-D processes to 2D or 3D
 * Also used to partition global grid space
 * to local space
 *
 * */

#include "array.h"
#include "grid_mpi.h" 
#include <mpi.h>

class Grid_view 
{
public:
	int num_dims;        	//for global data
	IndexArray global_size; //for global data
	int proc_num_dims;	//organization of processes
	IndexArray proc_size;   //organization of processes
	int num_procs;
	int my_rank;		//my rank in linear ranking

	IndexArray my_idx;	//my process index in multi-dimension
	IndexArray my_size;
	IndexArray my_offset;	//my offset within the global grid
	IndexArray fw_neighbors;
	IndexArray bw_neighbors;

	int **partitions;  	//partition info for sub-grid on each processor
	int **offsets; 		//offset info for sub-grid on each processor
	std::vector<IndexArray> proc_indices;
	IndexArray min_partition;
	//MPI_Comm comm;

	int GetProcessRank(const IndexArray &proc_index) const;

	Grid_view(int num_dims, IndexArray &global_size, int proc_num_dims, IndexArray &proc_size, int my_rank);
	
	GridMPI *CreateGrid(int unit_size, int num_dims, const IndexArray &size, const int halo_width);

	char *GetHaloPeerBuf(int dim, bool fw, unsigned width);	

	void ExchangeBoundariesAsync(GridMPI *grid, 
			int dim, unsigned halo_fw_width, 
			unsigned halo_bw_width, bool diagonal, 
			bool periodic, std::vector<MPI_Request> &requests) const;

	void ExchangeBoundaries(GridMPI *grid,
                                  int dim,
                                  unsigned halo_fw_width,
                                  unsigned halo_bw_width,
                                  bool diagonal,
                                  bool periodic) const;

	void ExchangeBoundaries(GridMPI *g,
                                  const Width2 &halo_width,
                                  bool diagonal,
                                  bool periodic
                                  ) const;
	Grid_view(){}

};

#endif
