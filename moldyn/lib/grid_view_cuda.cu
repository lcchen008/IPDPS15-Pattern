#include "grid_view_cuda.h"
#include "DS.h"

//Grid_view_cuda::Grid_view_cuda(int num_dims, IndexArray &global_size, int along, int my_share,int my_start, int num_devices, int device_id):num_dims_(num_dims), global_size(global_size), along_(along), my_share_(my_share), num_devices_(num_devices), device_id(device_id_)
//{
//	my_size_ = global_size;
//	my_size_[along_] = my_share_;
//	my_offset_.Set(0);
//	my_offset_[along_] = my_start; //the starting position along along_
//}

GridCuda* Grid_view_cuda::CreateGrid(int elm_size, int num_dims, const IndexArray &my_offset, const IndexArray &my_size, const int halo_width, int num_devices, int device_id)
{
	IndexArray halo_fw;
	halo_fw.Set(0);
	IndexArray halo_bw;
	halo_bw.Set(0);

	for(int i = 0; i < num_dims; i++)	
	{
		halo_fw[i] = halo_width;	
		halo_bw[i] = halo_width;	
	}

	Width2 halo = {halo_bw, halo_fw};

	GridCuda *g = GridCuda::Create(elm_size, num_dims, my_size, my_offset, my_size, halo, num_devices, device_id);

	return g;
}
