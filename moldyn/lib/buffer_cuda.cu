//#include "runtime_common_cuda.h"
#include "buffer_cuda.h"
#include "data_util.h"
#include <cstdio>
#include <assert.h>
#include <iostream>
#include "cu_util.h"
#include <stdio.h>
#include "time_util.h"
#include "util.h"

using namespace std;

//mapped host memory operation functions
BufferCUDAHost::BufferCUDAHost() 
	:Buffer(BufferCUDAHost::DeleteChunk)
{
}

BufferCUDAHost::~BufferCUDAHost() 
{
	Free();
}

void *BufferCUDAHost::GetChunk(IndexArray &size) 
{
	size_t s = size.accumulate(num_dims_); 
  	if (s == 0) return NULL;

	size_t linear_size = GetLinearSize(num_dims_, elm_size_, size);
	//printf("&&&&&&linear size: %ld, elm_size_: %ld&&&&&&\n", linear_size, elm_size_);
	//cout<<"multi dim size: "<<size<<endl;

  	void *ptr = NULL;  
  	CUDA_SAFE_CALL(cudaHostAlloc(&ptr, 
	linear_size, 
	cudaHostAllocPortable|cudaHostAllocMapped));

    	cudaHostGetDevicePointer((void **)&buf_d_, ptr, 0); 
	//printf("#############ptr is: %ld\n", ptr);

  	return ptr;
}

void BufferCUDAHost::DeleteChunk(void *ptr) 
{
	//printf("deleting #############ptr is: %ld\n", ptr);
  	CUDA_SAFE_CALL(cudaFreeHost(ptr));
  	return;
}

//device memory buffer operation functions
BufferCUDADev::BufferCUDADev()
    : Buffer(BufferCUDADev::DeleteChunk)
{	
	
}

BufferCUDADev::~BufferCUDADev() 
{
	Free();
}

void *BufferCUDADev::GetChunk(IndexArray &size) 
{
	//cout<<"GET CHUNK SIZE: "<<size<<endl;
	size_t s = size.accumulate(num_dims_); 
  	if (s == 0) return NULL;

	size_t linear_size = GetLinearSize(num_dims_, elm_size_, size);
	printf("&&&&&&linear size: %ld, elm_size_: %ld&&&&&&\n", linear_size, elm_size_);
	//cout<<"multi dim size: "<<size<<endl;

  	void *ptr = NULL;  
	
  	CUDA_SAFE_CALL(cudaMalloc(&ptr, linear_size));
  	return ptr;
}

void BufferCUDADev::DeleteChunk(void *ptr) 
{
  	CUDA_SAFE_CALL(cudaFree(ptr));
}

__device__ inline int get_linear_offset(size_t dim0,
size_t dim1,
size_t dim2,
size_t off0,
size_t off1,
size_t off2)
{
	return off0 + off1*dim0 + off2*dim0*dim1;	
}

template <typename  T>
__global__ void copy_halo_3D
(T *dst, 
size_t dst_dim0, 
size_t dst_dim1, 
size_t dst_dim2, 
size_t dst_off0, 
size_t dst_off1,
size_t dst_off2,
const T *src,
size_t src_dim0, 
size_t src_dim1, 
size_t src_dim2, 
size_t src_off0, 
size_t src_off1,
size_t src_off2,
size_t size0,
size_t size1,
size_t size2
)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;	
	int total_threads = blockDim.x * gridDim.x;
	int total = size0*size1*size2;

	int cur0, cur1, cur2;
	int src_abs0, src_abs1, src_abs2;
	int dst_abs0, dst_abs1, dst_abs2;

	for(int i = tid; i < total; i+= total_threads)
	{
		cur0 = i%size0;	
		cur1 = (i/size0)%size1;
		cur2 = i/size0/size1;

		src_abs0 = cur0 + src_off0;
		src_abs1 = cur1 + src_off1;
		src_abs2 = cur2 + src_off2;

		dst_abs0 = cur0 + dst_off0;
		dst_abs1 = cur1 + dst_off1;
		dst_abs2 = cur2 + dst_off2;

		int abs_off_dst = get_linear_offset
		(dst_dim0, 
		dst_dim1, 
		dst_dim2,
		dst_abs0,
		dst_abs1,
		dst_abs2);

		int abs_off_src = get_linear_offset
		(src_dim0, 
		src_dim1, 
		src_dim2,
		src_abs0,
		src_abs1,
		src_abs2);

		//printf("##############abs off dst: %d\n", abs_off_dst);
		*(dst + abs_off_dst) = *(src + abs_off_src);
	}
}

void BufferCUDADev::copy_in_from_map(BufferCUDAHost *source, 
			IndexArray &src_offset,
			IndexArray &my_offset, 
			IndexArray &size, cudaStream_t stream)
{

	//BufferCUDAHost &src = *source;
	dim3 grid(14, 1, 1);
    	dim3 block(64, 1, 1);

	if(elm_size_==4)
	{
		copy_halo_3D<float><<<grid, block, 0, stream>>>
		((float *)Get(), 
		size_[0], 
		size_[1], 
		size_[2], 
		my_offset[0], 
		my_offset[1],
		my_offset[2],
		(float *)(source->DeviceBuf()),
		(source->size())[0], 
		(source->size())[1], 
		(source->size())[2], 
		src_offset[0], 
		src_offset[1],
		src_offset[2],
		size[0],
		size[1],
		size[2]
		);

    		//cudaThreadSynchronize();
		checkCUDAError("copy in from map");
	}

	else if(elm_size_==8)
	{
		copy_halo_3D<double><<<grid, block, 0, stream>>>
		((double *)Get(), 
		size_[0], 
		size_[1], 
		size_[2], 
		my_offset[0], 
		my_offset[1],
		my_offset[2],
		(double *)(source->DeviceBuf()),
		(source->size())[0], 
		(source->size())[1], 
		(source->size())[2], 
		src_offset[0], 
		src_offset[1],
		src_offset[2],
		size[0],
		size[1],
		size[2]
		);
		//cudaThreadSynchronize();
		checkCUDAError("copy in from map");
	}

	else
		assert(0);

}

void BufferCUDADev::copy_out_to_map(BufferCUDAHost *dest, 
			IndexArray &dst_offset,
			IndexArray &my_offset, 
			IndexArray &size, cudaStream_t stream)
{

	//BufferCUDAHost dst = *dest;
	dim3 grid(14, 1, 1);
    	dim3 block(64, 1, 1);

	//printf("dst device ptr: %ld\n", dst.DeviceBuf());

	if(elm_size_==4)
	{
		copy_halo_3D<float><<<grid, block, 0, stream>>>
		((float *)(dest->DeviceBuf()), 
		(dest->size())[0], 
		(dest->size())[1], 
		(dest->size())[2], 
		dst_offset[0], 
		dst_offset[1],
		dst_offset[2],
		(float *)Get(),
		(size_)[0], 
		(size_)[1], 
		(size_)[2], 
		my_offset[0], 
		my_offset[1],
		my_offset[2],
		size[0],
		size[1],
		size[2]
		);

    		//cudaThreadSynchronize();
		checkCUDAError("calling copy 3D");
	}

	else if(elm_size_==8)
	{
		copy_halo_3D<double><<<grid, block, 0, stream>>>
		((double *)(dest->DeviceBuf()), 
		(dest->size())[0], 
		(dest->size())[1], 
		(dest->size())[2], 
		dst_offset[0], 
		dst_offset[1],
		dst_offset[2],
		(double *)Get(),
		(size_)[0], 
		(size_)[1], 
		(size_)[2], 
		my_offset[0], 
		my_offset[1],
		my_offset[2],
		size[0],
		size[1],
		size[2]
		);	

    		//cudaThreadSynchronize();
		checkCUDAError("calling copy 3D");
	}

	else
		assert(0);
}

void BufferCUDADev::copy_in_or_out(BufferHost *source, 
			IndexArray &my_offset,
			IndexArray &src_offset,
			IndexArray &size, bool in) 
{
	BufferHost &src = *source;
	char *dst_addr = (char *)Get();
	char *src_addr = (char *)src.Get();

	int num_dims = num_dims_;

	IndexArray offset_tmp(0);
	IndexArray src_start;
	IndexArray dst_start;

	if(num_dims==3)
	{
		for(int i = 0; i < size[2]; i++)	
		{
			offset_tmp[2] = i;
			for(int j = 0; j < size[1]; j++)
			{
				offset_tmp[1] = j;
				src_start = src_offset + offset_tmp;
				dst_start = my_offset + offset_tmp;
				intptr_t linear_offset_src = GridCalcOffset3D(src_start, src.size()) * elm_size_;
				intptr_t linear_offset_dst = GridCalcOffset3D(dst_start, size_)* elm_size_;
				if(in)	
				{
					CUDA_SAFE_CALL(cudaMemcpy((char *)Get()+linear_offset_dst, 
					(char *)src.Get() + linear_offset_src, size[0]*elm_size_, 
					cudaMemcpyHostToDevice));
				}

				else
				{
					CUDA_SAFE_CALL(cudaMemcpy((char *)src.Get() + 
					linear_offset_src, (char *)Get() +
					linear_offset_dst,size[0]*elm_size_, 
					cudaMemcpyDeviceToHost));
				}
			}
		}
	}

	else if(num_dims==2)
	{
		src_offset[2] = 0;
		my_offset[2] = 0;
		for(int i = 0; i < size[1]; i++)	
		{
			offset_tmp[1] = i;	
			src_start = src_offset + offset_tmp; 
			dst_start = my_offset + offset_tmp;

			intptr_t linear_offset_src = GridCalcOffset3D(src_start, src.size()) * elm_size_;
			intptr_t linear_offset_dst = GridCalcOffset3D(dst_start, size_) * elm_size_;
				
			if(in)
			{
				CUDA_SAFE_CALL(cudaMemcpy((char *)Get()+linear_offset_dst, 
				(char *)src.Get() + linear_offset_src, size[0]*elm_size_, 
				cudaMemcpyHostToDevice));
			}

			else
			{
				CUDA_SAFE_CALL(cudaMemcpy((char *)src.Get() + linear_offset_src, 
				(char *)Get()+linear_offset_dst, size[0]*elm_size_, 
				cudaMemcpyDeviceToHost));
			}
		}
	}
}


void BufferCUDADev::copy_in_from_host(BufferHost *src, 
			IndexArray &my_offset, 
			IndexArray &src_offset,
			IndexArray &size)
{
	copy_in_or_out(src, my_offset, src_offset, size, true);
}

void BufferCUDADev::copy_out_to_host(BufferHost *src, 
			IndexArray &my_offset, 
			IndexArray &src_offset,
			IndexArray &size)
{
	copy_in_or_out(src, my_offset, src_offset, size, false);
}
