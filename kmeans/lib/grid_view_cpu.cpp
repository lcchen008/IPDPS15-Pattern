#include "grid_view_cpu.h"
#include "array.h"
#include "grid_cpu.h"

GridCPU *Grid_view_cpu::CreateGrid(int elm_size, int num_dims, const IndexArray &my_offset, const IndexArray &my_size, const int halo_width, int num_devices, int device_id)
{
	IndexArray halo_fw;				
	halo_fw.Set(halo_width);
	IndexArray halo_bw;
	halo_bw.Set(halo_width);

	Width2 halo = {halo_bw, halo_fw};

	GridCPU *g = GridCPU::Create(elm_size, num_dims, my_size, my_offset, my_size, halo, num_devices, device_id); 

	return g;	
}
