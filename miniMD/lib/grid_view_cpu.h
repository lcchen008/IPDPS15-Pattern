#ifndef LIB_GRID_VIEW_CPU_H_
#define LIB_GRID_VIEW_CPU_H_

#include "grid_view.h"
#include "grid_view_cpu.h"
#include "grid_cpu.h"

class Grid_view_cpu:public Grid_view
{
public:
	char *GetHaloPeerBuf(int dim, bool fw, unsigned width);

	void ExchangeBoundariesAsync(GridCPU *grid, int dim, unsigned halo_fw_width, unsigned halo_bw_width);

	GridCPU *CreateGrid(int elm_size, int num_dims, const IndexArray &my_offset, const IndexArray &my_size, const int halo_width, int, int);
};

#endif
