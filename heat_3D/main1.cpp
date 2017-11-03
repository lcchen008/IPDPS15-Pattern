#include "lib/grid_cpu.h"
#include "lib/grid_view_cpu.h"
#include "lib/array.h"

int main()
{
	Grid_view_cpu *gv_cpu_ = new Grid_view_cpu();	

	IndexArray offset(6);

	GridCPU *grid_cpu_ = gv_cpu_->CreateGrid(1, 3, offset, offset, 6, 1, 0);

	delete gv_cpu_;
	delete grid_cpu_;

	gv_cpu_ = new Grid_view_cpu();

	grid_cpu_ = gv_cpu_->CreateGrid(1, 3, offset, offset, 6, 1, 0);

	delete gv_cpu_;
	delete grid_cpu_;
	delete grid_cpu_;

	return 0;
}
