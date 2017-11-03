#include "grid_cpu.h"
#include "buffer.h"
#include <stdio.h>
#include "grid_cuda.h"
#include "data_util.h"
#include <string.h>
#include "macro.h"
#include <iostream>
#include "cu_util.h"
using namespace std;

void GridCPU::copy_host_to_host(IndexArray &my_offset, IndexArray &src_offset, BufferHost *src, IndexArray &size)
{
	buffer()->copy_in(src, my_offset, src_offset, size);
	//printf("buffer is: &&&&&&&&& %d %d\n", this->data_buffer_[0], buffer());
	//this->data_buffer_[0]->copy_in(src, my_offset, src_offset, size);
}

GridCPU::GridCPU(int unit_size, int num_dims,
	const IndexArray &size,
        const IndexArray &my_offset,
        const IndexArray &my_size, 
        const Width2 &halo, int num_devices, int device_id):
	GridMPI(unit_size, 
			num_dims, 
			size, 
			my_offset, 
			my_size, 
			halo)
{
	//this->my_size_ = my_size;
	//this->my_offset_ = my_offset;
	//this->halo_ = halo;
	my_real_size_ = my_size_;
	my_real_offset_ = my_offset_;
	//num_dims_ = num_dims;
	//unit_size_ = unit_size;
	num_devices_ = num_devices;
	device_id_ = device_id;
	for(int i = 0; i < num_dims; i++)
	{
		my_real_size_[i]+=halo.fw[i] + halo.bw[i];	
		my_real_offset_[i] -= halo.bw[i];
	}
}


GridCPU *GridCPU::Create(
		int unit_size, 
		int num_dims, 
		const IndexArray &size,
		const IndexArray &local_offset, 
		const IndexArray &local_size, 
		const Width2 &halo, int num_devices, int device_id)
{
	GridCPU *g = new GridCPU(unit_size, num_dims, size, local_offset, local_size, halo, num_devices, device_id);	
	g->InitBuffer();
	return g;
}

void GridCPU::InitBuffer()
{
	data_buffer_[0] = new BufferHost();
  	data_buffer_[0]->Allocate(num_dims_, unit_size_, my_real_size_);
  	data_[0] = (char*)data_buffer_[0]->Get();
  	data_buffer_[1] = new BufferHost();
	data_buffer_[1]->Allocate(num_dims_, unit_size_, my_real_size_);
  	data_[1] = (char*)data_buffer_[1]->Get();
}

void GridCPU::DeleteBuffers()
{
	Grid::DeleteBuffers();
}

void GridCPU::set_num_devices(int num_devices)
{
	this->num_devices_ = num_devices;
}

//used to send to neighbors in the same node
void GridCPU::send_to_neighbors(int dim, int stencil_width, GridMPI *grid_global, GridCuda **grid_cuda)
{
	IndexArray size = my_real_size();		
	size[dim] = stencil_width;

	size_t linear_size = size.accumulate(num_dims_);

	//---!first, send to lower neighbor, i.e., send to global out buffer	
	
	GridMPI *dst = grid_global;
	GridCPU *src = this;
	
	IndexArray dst_offset(0);	
	dst_offset[dim] = stencil_width;
	IndexArray src_offset(0);
	src_offset[dim] = stencil_width;

	memcpy(dst->data_in() + unit_size_ * GridCalcOffset3D(dst_offset, dst->my_real_size()), 
			src->data_in() + unit_size_ * GridCalcOffset3D(src_offset, src->my_real_size()), unit_size_ *linear_size);

	//---!next, send to upper neighbor, i.e., the first GPU grid
	GridCuda *dst_cuda = grid_cuda[0];			
	dst_offset[dim] = 0; 
	src_offset[dim] = src->my_real_size()[dim] - 2 * stencil_width;


	CUDA_SAFE_CALL(cudaSetDevice(0));

	//printf("Linear offset: %ld\n", GridCalcOffset3D(src_offset, src->my_real_size()));

	//void *test;

	//cudaMalloc((void **)&test, sizeof(int));
	
	//int test_h = 9;

	//CUDA_SAFE_CALL(cudaMemcpy(test, &test_h, sizeof(int), cudaMemcpyHostToDevice));
	//printf("linear size: %ld\n", linear_size);
	//checkCUDAError("I am checking here ####");		
	//printf("data in: %ld\n", dst_cuda->data_in());
	

	CUDA_SAFE_CALL(cudaMemcpy(dst_cuda->data_in(), 
				src->data_in() + unit_size_ * GridCalcOffset3D(src_offset, src->my_real_size()), 
				unit_size_ * linear_size, 
				cudaMemcpyHostToDevice));
	
	//CUDA_SAFE_CALL(cudaMemcpy(test, src->data_in() + unit_size_ * GridCalcOffset3D(src_offset, src->my_real_size()), 
	//		unit_size_ * linear_size, 
	//		cudaMemcpyHostToDevice));

}

void GridCPU::copy_from_bottom(int dim, int stencil_width, GridMPI *grid_global)
{
	IndexArray size = my_real_size();		
	size[dim] = stencil_width;

	size_t linear_size = size.accumulate(num_dims_);

	//---!first, send to lower neighbor, i.e., send to global out buffer	
	
	GridMPI *src = grid_global;
	GridCPU *dst = this;
	
	IndexArray dst_offset(0);	
	dst_offset[dim] = 0;
	IndexArray src_offset(0);
	src_offset[dim] = 0;

	memcpy(dst->data_in() + 
			unit_size_ * GridCalcOffset3D(dst_offset, dst->my_real_size()), 
			src->data_in() + 
			unit_size_ * GridCalcOffset3D(src_offset, src->my_real_size()), unit_size_ *linear_size);

}

//copy cpu border to global grid send buffer
void GridCPU::copy_to_global_grid(GridMPI *grid_global, int dim, int along_start, int along_length, unsigned width, bool fw)
{
	IndexArray my_offset(0);		
	my_offset[num_dims_-1] = width;
	
	if(fw)
	{
		my_offset[dim] = width;	
	}

	else
	{
		my_offset[dim] = my_real_size_[dim] - 2 * width;	
	}

	IndexArray global_halo_offset(0);

	global_halo_offset[num_dims_-1] = width;

	BufferHost *buf = buffer();//fw? halo_peer_fw_[dim]:halo_peer_bw_[dim];

	BufferHost *global_halo_buf = fw?grid_global->halo_self_fw()[dim]:grid_global->halo_self_bw()[dim];

	IndexArray copy_size = my_real_size_;
	copy_size[num_dims_-1] = along_length;
	copy_size[dim] = width;

	global_halo_buf->copy_in(buf, global_halo_offset, my_offset, copy_size);
}

void GridCPU::copy_from_global_grid(GridMPI *grid_global, int dim, int along_start, int along_length, unsigned width, bool fw)
{
	IndexArray my_offset(0);		
	my_offset[num_dims_-1] = width;
	
	if(fw)
	{
		my_offset[dim] = my_real_size_[dim] - width;	
	}

	IndexArray global_halo_offset(0);

	global_halo_offset[num_dims_-1] = width;

	BufferHost *buf = buffer();//fw? halo_peer_fw_[dim]:halo_peer_bw_[dim];

	BufferHost *global_halo_buf = fw?grid_global->halo_peer_fw()[dim]:grid_global->halo_peer_bw()[dim];

	IndexArray copy_size = my_real_size_;
	copy_size[num_dims_-1] = along_length;
	copy_size[dim] = width;

	global_halo_buf->copy_out(buf, global_halo_offset, my_offset, copy_size);
}
