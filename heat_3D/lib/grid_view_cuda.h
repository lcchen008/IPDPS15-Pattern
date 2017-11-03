#ifndef LIB_GRID_VIEW_CUDA_H_
#define LIB_GRID_VIEW_CUDA_H_

#include "array.h"
#include "grid_cuda.h"
#include "grid_view.h"

class Grid_view_cuda:public Grid_view
{
public:
	GridCuda *CreateGrid(int elm_size, int num_dims, const IndexArray &my_offset, const IndexArray &my_size, const int halo_width, int num_devices, int device_id);		

	void ExchangeBoundariesAsync(GridCuda *grid, int dim, unsigned halo_fw_width, unsigned halo_bw_width);

	void ExchangeBoundaries(GridCuda *grid, int dim, unsigned halo_fw_width, unsigned halo_bw_width);
};

#endif
