#ifndef GPU_KERNEL
#define GPU_KERNEL
#endif


#include "CU_DS.h"
#include "../stencil.h"
#include <stdio.h>
using namespace FGPU;

__device__ int get_linear_size(int *size, int num_dims)
{
	if(num_dims == 2)	
	{
		return size[0]*size[1];	
	}
	else
		return size[0]*size[1]*size[2];
}

__device__ void get_relative_cor(int *cor, int *tile_cor, int *size, int dim, int relative_number)
{
	for(int i = 0; i < dim; i++)
	{
		cor[i] = tile_cor[i] + relative_number%size[i];	
		relative_number/=size[i];
	}
}

__global__ void compute_cuda_internal
(
	int num_tiles,
	Ltile *tiles,
	void *input,
	void *output,
	int num_dims,
	int unit_size,
	int *dims,     	//contains size info for current grid
	int stencil_idx
)
{
	const unsigned int tid = threadIdx.x;
    	const unsigned int bid = blockIdx.x;
	const unsigned int num_blocks = gridDim.x;

    	const unsigned int block_size = blockDim.x;

	__shared__ int sdims[3];
	__shared__ Ltile current_tile;
	__shared__ int linear_size;

	Ltile my_tile;

	if(tid == 0)
	{
		for(int i = 0; i < 3; i++)	
		{
			sdims[i] = dims[i];	
		}
	}

	for(int i = bid; i < num_tiles; i+= num_blocks)
	{
		if(tid==0)				
		{
			current_tile = tiles[i];	
			linear_size = get_linear_size(current_tile.size, num_dims);
			//printf("current tile: %d, %d, %d, size: %d, %d, %d\n", current_tile.offset[0], current_tile.offset[1], current_tile.offset[2], current_tile.size[0], current_tile.size[1], current_tile.size[2]);
		}

		__syncthreads();

		my_tile = current_tile;

		//process the tile by a thread block	
		int tile_size = linear_size;
		int offset[3];		
		int global_size[3];

		for(int i = 0; i < 3; i++)
		{
			global_size[i] = sdims[i];	
		}

		for(int j = tid; j < tile_size; j += block_size)		
		{
			get_relative_cor(offset, my_tile.offset, my_tile.size, num_dims, j);
			//printf("( %d, %d, %d )\n", offset[0], offset[1], offset[2]);	
			//if(offset[0]>=dims[0]||offset[1]>=dims[1]||offset[2]>=dims[2])
			//{
			//	printf("( %d, %d, %d )\n", offset[0], offset[1], offset[2]);	
			//}
			//printf("( %d, %d, %d )\n", offset[0], offset[1], offset[2]);
			//FGPU::stencil(input, output, offset, global_size);	
			stencil_function_table[stencil_idx](input, output, offset, global_size);	
		}

		__syncthreads();
	}
}
