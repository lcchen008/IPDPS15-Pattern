#include "grid_cuda.h"
#include "macro.h"
#include "data_util.h"
#include "grid_cpu.h"
#include <stdio.h>
#include "cu_util.h"
#include <iostream>
#include "time_util.h"
using namespace std;


GridCuda::GridCuda(int unit_size, 
int num_dims, 
const IndexArray &size, 
const IndexArray &my_offset, 
const IndexArray &my_size, 
const Width2 &halo,
int num_devices,
int device_id):
GridMPI(unit_size, 
num_dims, 
size, 
my_offset, 
my_size, 
halo)
//halo_self_fw_(NULL);
//halo_self_bw_(NULL),
//halo_peer_fw_(NULL),
//halo_peer_bw_(NULL)

{
	my_real_size_ = my_size_;	
	my_real_offset_ = my_offset_;
	num_devices_ = num_devices;
	device_id_ = device_id;

	for(int i = 0; i < num_dims; i++)
	{
		my_real_size_[i] += halo.fw[i] + halo.bw[i];
    		my_real_offset_[i] -= halo.bw[i];
	}

	cudaStreamCreate(&compute_stream_);
	cudaStreamCreate(&copy_out_stream_);
	cudaStreamCreate(&copy_in_stream_);
}

GridCuda *GridCuda::Create(int unit_size,
      		int num_dims, const IndexArray &size,
      		const IndexArray &my_offset,
      		const IndexArray &my_size,
      		const Width2 &halo, int num_devices, int device_id)
{
	GridCuda *g = new GridCuda(unit_size, num_dims, 
	size, my_offset, my_size, halo, num_devices, device_id);		
	g->InitBuffer();
	return g;
}

void GridCuda::InitBuffer()
{
	data_buffer_[0] = new BufferCUDADev();		
	data_buffer_[0] -> Allocate(num_dims_, unit_size_, my_real_size_);
	data_buffer_[1] = new BufferCUDADev();		
	data_buffer_[1] -> Allocate(num_dims_, unit_size_, my_real_size_);
	data_[0] = (char*)data_buffer_[0]->Get();
	data_[1] = (char*)data_buffer_[1]->Get();

  	InitHaloBuffers();
}

void GridCuda::InitHaloBuffers()
{
	halo_self_fw_ = new BufferCUDAHost*[num_dims_ - 1];			
	halo_self_bw_ = new BufferCUDAHost*[num_dims_ - 1];			
	halo_peer_fw_ = new BufferCUDAHost*[num_dims_ - 1];			
	halo_peer_bw_ = new BufferCUDAHost*[num_dims_ - 1];			


	//cout<<"#############halo fw: "<<halo_.fw<<endl;	
	//cout<<"#############halo bw: "<<halo_.bw<<endl;	

	for(int i = 0; i < num_dims_ - 1; i++)
	{
		halo_self_fw_[i] = halo_self_bw_[i] = NULL;	
		halo_peer_fw_[i] = halo_peer_bw_[i] = NULL;	
		if(halo_.fw[i])
		{
			IndexArray size = my_real_size_; 
			size[i] = halo_.fw[i]; 

			halo_self_fw_[i]= new BufferCUDAHost();	
			halo_peer_fw_[i]= new BufferCUDAHost();	
			halo_self_fw_[i]->Allocate(num_dims_, unit_size_, size);
			halo_peer_fw_[i]->Allocate(num_dims_, unit_size_, size);
		}

		if(halo_.bw[i])
		{
			IndexArray size = my_real_size_; 
			size[i] = halo_.bw[i]; 

			halo_self_bw_[i]= new BufferCUDAHost();	
			halo_peer_bw_[i]= new BufferCUDAHost();	
			halo_self_bw_[i]->Allocate(num_dims_, unit_size_, size);
			halo_peer_bw_[i]->Allocate(num_dims_, unit_size_, size);
		}
	}

	buffer_global_ = NULL;
	buffer_cpu_send_ = NULL;
	buffer_cpu_recv_ = NULL;
	//I am neighbor with CPU
	if(device_id_ == 1)
	{
		buffer_cpu_send_ = new BufferCUDAHost();
		buffer_cpu_recv_ = new BufferCUDAHost();

		IndexArray copy_size = my_real_size_;
		copy_size[num_dims_ - 1] = halo_.bw[num_dims_ - 1];

		buffer_cpu_send_->Allocate(num_dims_, unit_size_, copy_size);
		buffer_cpu_recv_->Allocate(num_dims_, unit_size_, copy_size);
	}

	//I need to copy data to global grid
	if(device_id_ == num_devices_ - 1)
	{
		buffer_global_ = new BufferCUDAHost();	
		IndexArray copy_size = my_real_size_;
		copy_size[num_dims_ - 1] = halo_.bw[num_dims_ - 1];

		buffer_global_->Allocate(num_dims_, unit_size_, copy_size);
	}
}

void GridCuda::DeleteBuffers()
{
	CUDA_SAFE_CALL(cudaSetDevice(device_id_ - 1));
	DeleteHaloBuffers();	
	Grid::DeleteBuffers();
}

void GridCuda::DeleteHaloBuffers()
{
	for (int i = 0; i < num_dims_ - 1; ++i) 
	{
    		if (halo_self_fw_) delete (halo_self_fw_[i]);
    		if (halo_self_bw_) delete (halo_self_bw_[i]);
    		if (halo_peer_fw_) delete (halo_peer_fw_[i]);
    		if (halo_peer_bw_) delete (halo_peer_bw_[i]);
  	}

  	PS_XDELETEA(halo_self_fw_);
  	PS_XDELETEA(halo_self_bw_);
  	PS_XDELETEA(halo_peer_fw_);
  	PS_XDELETEA(halo_peer_bw_);

	if(buffer_cpu_send_)
	{
		delete buffer_cpu_send_;
		delete buffer_cpu_recv_;
	}

	if(buffer_global_)
		delete buffer_global_;
}

void GridCuda::copy_host_to_device(IndexArray &my_offset,
			IndexArray &src_offset, 
			IndexArray &size, BufferHost *src)
{
	data_buffer_[0] -> copy_in_from_host(src, my_offset, src_offset, size);  		
}

void GridCuda::set_device_id(int id)
{
	this->device_id_ = id;
}

void GridCuda::set_num_devices(int num_devices)
{
	this->num_devices_ = num_devices;
}

//void GridCuda::copy_device_to_map(int dim, unsigned width, bool fw)
//{
//	if(dim == num_dims_ - 1)	
//		return;
//
//	IndexArray halo_offset(0); 		//my offset	
//	if(fw)
//	{
//		halo_offset[dim] = width;	
//	}
//	else
//	{
//		halo_offset[dim] = my_real_size()[dim] - 2 * width;	
//	}
//	
//	BufferCUDAHost *halo_buf = fw ? halo_self_fw_[dim] : halo_self_bw_[dim];	
//
//	IndexArray halo_size = my_real_size();
//
//	halo_size[dim] = width;
//
//	IndexArray map_offset(0);
//
//	double beforecopy = rtclock();
//	buffer()->copy_out_to_map(halo_buf,  map_offset, halo_offset, halo_size);		
//	double aftercopy = rtclock();
//
//	printf("copy device to map time: %f\n", aftercopy - beforecopy);
//}

//we can assume that the dim is not the highest
void GridCuda::copy_device_to_map(int dim, unsigned width, bool fw)
{
	if(dim == num_dims_ - 1)	
		return;

	IndexArray halo_offset(width, width, width); 		//my offset	

	if(!fw)
	{
		halo_offset[dim] = my_real_size()[dim] - 2 * width;	
	}
	
	BufferCUDAHost *halo_buf = fw ? halo_self_fw()[dim] : halo_self_bw()[dim];	

	IndexArray halo_size = my_size();

	halo_size[dim] = width;

	IndexArray map_offset(width, width, width);

	map_offset[dim] = 0;

	double beforecopy = rtclock();
	buffer()->copy_out_to_map(halo_buf,  map_offset, halo_offset, halo_size, compute_stream_);		
	double aftercopy = rtclock();

	//printf("copy device to map time: %f\n", aftercopy - beforecopy);
}

//we can assume that the dim is not the highest
//void GridCuda::copy_map_to_device(int dim, unsigned width, bool fw)
//{
//	if(dim == num_dims_ - 1)
//		return;
//
//	IndexArray halo_offset(0);
//
//	if(fw)
//	{
//		halo_offset[dim] = my_real_size_[dim] - width;	
//	}
//
//	BufferCUDAHost *halo_buf = fw ? halo_self_fw_[dim] : halo_self_bw_[dim];
//
//	IndexArray halo_size = my_real_size_;
//
//	halo_size[dim] = width;
//
//	IndexArray map_offset(0);
//
//	this->buffer()->copy_in_from_map(halo_buf, map_offset, halo_offset, halo_size);
//}

void GridCuda::copy_map_to_device(int dim, unsigned width, bool fw)
{
	if(dim == num_dims_ - 1)
		return;

	IndexArray halo_offset(width, width, width);

	if(fw)
	{
		halo_offset[dim] = my_real_size_[dim] - width;	
	}

	else
		halo_offset[dim] = 0;

	BufferCUDAHost *halo_buf = fw ? halo_peer_fw()[dim] : halo_peer_bw()[dim];

	IndexArray halo_size = my_size();

	halo_size[dim] = width;

	IndexArray map_offset(width,width,width);

	map_offset[dim] = 0;

	this->buffer()->copy_in_from_map(halo_buf, map_offset, halo_offset, halo_size, compute_stream_);
}


void GridCuda::copy_map_to_global_grid(GridMPI *grid_global, int dim, int along_start, int along_length, unsigned width, bool fw)
{
	IndexArray global_real_size = grid_global->my_real_size();
	IndexArray global_size = grid_global->my_size();
	
	int padding = (global_real_size[num_dims_ - 1] - global_size[num_dims_ - 1])/2;

	IndexArray global_offset(0);
	global_offset[num_dims_ - 1] = along_start + padding;

	IndexArray global_halo_size = grid_global->my_real_size();
	global_halo_size[dim] = width;

	IndexArray copy_size = my_real_size_;

	copy_size[num_dims_ - 1] = along_length;
	copy_size[dim] = width;

	IndexArray local_offset(0);
	local_offset[num_dims_-1] = width;
	IndexArray local_halo_size = my_real_size_;
	local_halo_size[dim] = width;

	size_t local_linear_offset = GridCalcOffset3D(local_offset, local_halo_size);
	size_t global_linear_offset = GridCalcOffset3D(global_offset, global_halo_size);

	size_t linear_size = copy_size.accumulate(num_dims_);

	char *local_halo_buf = (char *)(fw? halo_self_fw()[dim]->Get() : halo_self_bw()[dim]->Get());

	char *global_halo_buf = (char *)(fw? grid_global->halo_self_fw()[dim]->Get() : grid_global->halo_self_bw()[dim]->Get());

	memcpy(global_halo_buf + global_linear_offset * unit_size_, local_halo_buf + local_linear_offset * unit_size_, linear_size * unit_size_);

}

void GridCuda::copy_global_grid_to_map(GridMPI *grid_global, int dim, int along_start, int along_length, unsigned width, bool fw)
{
	IndexArray global_real_size = grid_global->my_real_size();
	IndexArray global_size = grid_global->my_size();
	
	int padding = (global_real_size[num_dims_ - 1] - global_size[num_dims_ - 1])/2;

	IndexArray global_offset(0);
	global_offset[num_dims_ - 1] = along_start + padding;

	IndexArray global_halo_size = grid_global->my_real_size();
	global_halo_size[dim] = width;

	IndexArray copy_size = my_real_size_;
	copy_size[num_dims_ - 1] = along_length;
	copy_size[dim] = width;

	IndexArray local_offset(0);
	local_offset[num_dims_-1] = width;
	IndexArray local_halo_size = my_real_size_;
	local_halo_size[dim] = width;

	size_t local_linear_offset = GridCalcOffset3D(local_offset, local_halo_size);
	size_t global_linear_offset = GridCalcOffset3D(global_offset, global_halo_size);

	size_t linear_size = copy_size.accumulate(num_dims_);

	char *local_halo_buf = (char *)(fw? halo_self_fw()[dim]->Get() : halo_self_bw()[dim]->Get());

	char *global_halo_buf = (char *)(fw? grid_global->halo_self_fw()[dim]->Get() : grid_global->halo_self_bw()[dim]->Get());

	memcpy(local_halo_buf + local_linear_offset * unit_size_, global_halo_buf + global_linear_offset * unit_size_, linear_size * unit_size_);
}

void GridCuda::copy_map_to_neighbor(int dim, int stencil_width, GridMPI *grid_global, GridCPU *grid_cpu)
{
	if((device_id_!=1)&&(device_id_!=num_devices_ - 1))
		return;

	IndexArray size = my_real_size();
	size[dim] = stencil_width;

	size_t linear_size = size.accumulate(num_dims_);

	//copy to local cpu grid
	if(device_id_ == 1)	
	{
		IndexArray cpu_offset(0); 	//dest
		cpu_offset[dim] = grid_cpu->my_real_size()[dim] - 2 * stencil_width;	
		IndexArray gpu_offset(0); 	//src	
		gpu_offset[dim] = stencil_width;

		memcpy(grid_cpu->data_in() + unit_size_ * GridCalcOffset3D(cpu_offset, grid_cpu->my_real_size()), buffer_cpu_send_->Get(), unit_size_ * linear_size);
	}

	//copy to global grid
	if(device_id_ == num_devices_ - 1)
	{
		GridMPI *dst = grid_global;
		IndexArray dst_offset(0);
		dst_offset[dim] = dst->my_real_size()[dim] - 2 * stencil_width;

		memcpy(dst->data_in() + unit_size_ * GridCalcOffset3D(dst_offset, dst->my_real_size()), buffer_global_->Get(), unit_size_ * linear_size);
	}
}

void GridCuda::copy_from_neighbor_map(int dim, int stencil_width, GridMPI *grid_global)
{
		IndexArray size = my_real_size();
		size[dim] = stencil_width;
		size_t linear_size = size.accumulate(num_dims_);

		int gpu_id = device_id_ - 1;
		GridCuda *dst = this;	

		IndexArray dst_offset(0);
		dst_offset[dim] = dst->my_real_size()[dim] - stencil_width;

		CUDA_SAFE_CALL(cudaSetDevice(gpu_id));

		CUDA_SAFE_CALL(cudaMemcpyAsync(dst->data_in() + unit_size_ * GridCalcOffset3D(dst_offset, dst->my_real_size()), buffer_global_->Get(), unit_size_ * linear_size, cudaMemcpyDeviceToHost, copy_out_stream_));
}

//---!TODO: add a stream
void GridCuda::send_to_neighbors(int dim, int stencil_width, GridMPI *grid_global, GridCPU *grid_cpu, GridCuda **grid_cuda)
{
	IndexArray size = my_real_size();
	size[dim] = stencil_width;
	size_t linear_size = size.accumulate(num_dims_);

	int gpu_id = device_id_ - 1;

	//---!first, send to lower neighbor
	//lower neighbor is CPU
	if(device_id_ == 1)	
	{
		IndexArray cpu_offset(0); 	//dest
		cpu_offset[dim] = grid_cpu->my_real_size()[dim] - 2 * stencil_width;	
		IndexArray gpu_offset(0); 	//src	
		gpu_offset[dim] = stencil_width;
				
		//cudaSetDevice(0);

		//CUDA_SAFE_CALL(cudaMemcpy(grid_cpu->data_in() + unit_size_ * GridCalcOffset3D(cpu_offset, grid_cpu->my_real_size()), 
		//this->data_in() + unit_size_ * GridCalcOffset3D(gpu_offset, this->my_real_size()), unit_size_ * linear_size, cudaMemcpyDeviceToHost));

		CUDA_SAFE_CALL(cudaMemcpyAsync(buffer_cpu_send_->Get(), this->data_in() + unit_size_ * GridCalcOffset3D(gpu_offset, this->my_real_size()), unit_size_ * linear_size, cudaMemcpyDeviceToHost, copy_out_stream_));
	}

	//---! lower neighbor is GPU
	else
	{	
		GridCuda *src = this;
		GridCuda *dst = grid_cuda[gpu_id - 1];

		IndexArray dst_offset(0);	
		dst_offset[dim] = dst->my_real_size()[dim] - stencil_width;

		IndexArray src_offset(0);
		src_offset[dim] = stencil_width;
		
		//cudaSetDevice(gpu_id);

		CUDA_SAFE_CALL(cudaMemcpyPeerAsync(dst->data_in() + unit_size_ * GridCalcOffset3D(dst_offset, dst->my_real_size()), 
		gpu_id - 1, src->data_in() + unit_size_ * GridCalcOffset3D(src_offset, src->my_real_size()), gpu_id, unit_size_ * linear_size, copy_out_stream_));
	}

	//---!second, send to upper neighbor
	//if I am the last device
	//I need to copy my border to global grid's top 
	if(device_id_ == num_devices_ - 1)
	{
		GridCuda *src = this;	
		GridMPI *dst = grid_global;

		IndexArray dst_offset(0);
		dst_offset[dim] = dst->my_real_size()[dim] - 2 * stencil_width;

		IndexArray  src_offset(0);
		src_offset[dim] = src->my_real_size()[dim] - 2 * stencil_width;

		//cudaSetDevice(gpu_id);

		CUDA_SAFE_CALL(cudaMemcpyAsync(buffer_global_->Get(), 
		this->data_in() + unit_size_ * GridCalcOffset3D(src_offset, src->my_real_size()), unit_size_ * linear_size, cudaMemcpyDeviceToHost, copy_out_stream_));
		//CUDA_SAFE_CALL(cudaMemcpy(dst->data_in() + unit_size_ * GridCalcOffset3D(dst_offset, dst->my_real_size()), 
		//src->data_in() + unit_size_ * GridCalcOffset3D(src_offset, src->my_real_size()), unit_size_ * linear_size, cudaMemcpyDeviceToHost));
	}

	//---!upper is GPU
	else
	{
		GridCuda *src = this;
		GridCuda *dst = grid_cuda[gpu_id + 1];

		IndexArray dst_offset(0);	

		IndexArray src_offset(0);
		src_offset[dim] = src->my_real_size()[dim] - 2 * stencil_width;
		
		//cudaSetDevice(gpu_id);

		CUDA_SAFE_CALL(cudaMemcpyPeerAsync(dst->data_in() + unit_size_ * GridCalcOffset3D(dst_offset, dst->my_real_size()), 
		gpu_id + 1, src->data_in() + unit_size_ * GridCalcOffset3D(src_offset, src->my_real_size()), gpu_id, unit_size_ * linear_size, copy_out_stream_));
	}
	
	checkCUDAError("cuda send neighbors#############");
}

void GridCuda::copy_from_top(int dim, int stencil_width, GridMPI *grid_global)
{
	//device id must indicate that I am the top device
	if(device_id_ == num_devices_ - 1)
	{
		IndexArray size = my_real_size();
		size[dim] = stencil_width;
		size_t linear_size = size.accumulate(num_dims_);

		//int gpu_id = device_id_ - 1;
		GridCuda *dst = this;	
		GridMPI *src = grid_global;

		IndexArray dst_offset(0);
		dst_offset[dim] = dst->my_real_size()[dim] - stencil_width;

		IndexArray  src_offset(0);
		src_offset[dim] = src->my_real_size()[dim] - stencil_width;

		//cudaSetDevice(gpu_id);

		//CUDA_SAFE_CALL(cudaMemcpy(dst->data_in() + unit_size_ * GridCalcOffset3D(dst_offset, dst->my_real_size()), 
		//src->data_in() + unit_size_ * GridCalcOffset3D(src_offset, src->my_real_size()), unit_size_ * linear_size, cudaMemcpyHostToDevice));
		//CUDA_SAFE_CALL(cudaMemcpyAsync());
		memcpy(buffer_global_->Get(), src->data_in() + unit_size_ * GridCalcOffset3D(src_offset, src->my_real_size()), unit_size_ * linear_size);
	}
}
