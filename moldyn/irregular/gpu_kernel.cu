#ifndef G_KERNEL
#define G_KERNEL
//#ifndef GPU_KERNEL
//#define GPU_KERNEL
#define GPU_KERNEL

#include <stdio.h>
#include "parameters.h"
#include "data_type.h"
#include "gpu_kernel_tools.cu"
#include "ros.cu"
#include "rog.cu"
#include "../irregular.h"
using namespace FGPU;

__global__ void init_rog(Gobject *gobject)
{
	gobject->init();
}

__global__ void compute_gpu(
Part *parts,			//contains partition info
void *point_input,
int *part_index,
void *edge_input,
void *parameter,
int *task_offset,       //indicates the global task offset
int number_parts,
Gobject *object_g,
int map_idx,
int reduce_idx
)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int global_id = blockDim.x * blockIdx.x + tid;
    const unsigned int num_blocks = gridDim.x;


    __shared__ int task_index;      //task index within each SM
    __shared__ int has_taskl;
    __shared__ Sobject object_s;

    __shared__ irmap_fp map_ptrs;
    __shared__ irreduce_fp reduce_ptrs;

    if(tid == 0)
    {
        map_ptrs = map_function_table[map_idx];	 
        reduce_ptrs = reduce_function_table[reduce_idx];	 
    }

    object_s.first_init();

    __syncthreads();

    irmap_fp map_ptr = map_ptrs; 
    irreduce_fp reduce_ptr = reduce_ptrs;

    while(1)
    {
        __syncthreads();     

        if(tid == 0)
        {
	    has_taskl = 1;
            task_index = atomicAdd(task_offset, GPU_BLOCK_SIZE);     
            if(task_index>=number_parts)
            {
                has_taskl = 0;
            }
        }

        __syncthreads();

        if(has_taskl == 0)
            break;

        for(int j = task_index; (j < task_index + GPU_BLOCK_SIZE)&&(j < number_parts); j++)
        {
        	int start = parts[j].start;
        	int end = parts[j].end;
    		int part_id = parts[j].part_id;

		//if(tid==0)
		//	printf("start is: %d and end is: %d\n", start, end);

        	for(int i = start+tid; i < end; i+=GPU_THREADS)
        	{
        	    //FGPU::map(&object_s, point_input, part_index, edge_input, parameter, i, part_id, reduce_ptr); 
        	    map_ptr(&object_s, point_input, part_index, edge_input, parameter, i, part_id, reduce_ptr); 
        	}

    		object_g->merges(&object_s, reduce_ptr);

    		__syncthreads();

    		object_s.init();

    		__syncthreads();
        }
    }
}

__global__ void mergecgd(Gobject *object1, Gobject *object2, int reduce_idx)
{
	object1->mergeg(object2, reduce_function_table[reduce_idx]);
}

#endif
