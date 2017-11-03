#ifndef G_KERNEL
#define G_KERNEL

#ifndef GPU_KERNEL
#define GPU_KERNEL
#endif

#include "ros.cu"
#include "rog.cu"

#include <stdio.h>
#include "parameters.h"
#include "../regular.h"
#include "data_type.h"
#include "../lib/common.h"
//#define SHARED_SIZE 32768

__global__ void merged(GO *object1, GO *object2, int reduce_idx)
{
	object1->mergeg(object2, reduce_function_table[reduce_idx]);
}

template <class T1, class T2> 
__device__ void merge(T1 *dstobject, T2 *srcobject, reduce_fp reduce_ptr)
{
	for(int index = 0; index<srcobject->num_buckets; index++)
	{
		if((srcobject->buckets)[index]!=0)		
		{
			int key_size = srcobject->get_key_size(index);  	
			int value_size = srcobject->get_value_size(index);
			void *key = srcobject->get_key_address(index);	
			void *value = srcobject->get_value_address(index);
			dstobject->insert(key, key_size, value, value_size, reduce_ptr);
		}
	}
}

__global__ void compute_gpu(
void *input,
Offset *data_offset,      //used to split the input data
int *device_offset,     //offset within each device
int offset_number,
unsigned long long start,
GO *object_g, 
void *parameter,
int map_num,
int reduce_num
)
{
    __shared__ unsigned int num_groupss;
    const unsigned int tid = threadIdx.x;
    //const unsigned int global_id = blockDim.x * blockIdx.x + tid;
    __shared__ map_fp map_ptrs;
    __shared__ reduce_fp reduce_ptrs;

    if(tid == 0)
    {
    	num_groupss = get_num_groups();
	map_ptrs = map_function_table[map_num];
	reduce_ptrs = reduce_function_table[reduce_num];
    }

    __syncthreads();

    const int num_groups = num_groupss;//get_num_groups();//floor((double)SHARED_SIZE/sizeof(SO));
    const int group_id = get_group_id(num_groups, GPU_THREADS, tid);
    const int gid = get_gid(num_groups, GPU_THREADS, tid);
    const int group_size = get_group_size(num_groups, GPU_THREADS, group_id);

    map_fp map_ptr = map_ptrs; 
    reduce_fp reduce_ptr = reduce_ptrs;

    //if(global_id == 255)
    //{
    //    printf("num of groups: %d, group_size: %d, group_id: %d, gid: %d\n", num_groups, group_size, group_id, gid); 
    //    printf("size of double: %d\n", sizeof(double));
    //}

    __shared__ int task_index;      //task index within each SM
    __shared__ int has_taskl;

#ifdef USE_SHARED
    __shared__ char object_s[SHARED_SIZE];
    ((SO *)object_s + group_id)->oma_init(gid, group_size);
    //((SO *)object_s)->oma_init();
    //object_s.oma_init();
    __shared__ int do_merge;
    __shared__ int finished;
#endif

    object_g->oma_init();

    __syncthreads();

    while(1)
    {
        __syncthreads();     

        if(tid == 0)
        {
            task_index = atomicAdd(device_offset, R_GPU_BLOCK_SIZE);     

            if(task_index >= offset_number)
            {
                has_taskl = 0;
            }

	    else
		has_taskl = 1;
        }

        __syncthreads();

        if(has_taskl == 0)
	{
            break;
	}

#ifdef USE_SHARED

        if(tid == 0)
        {
            finished = 0;
            do_merge = 0;
        }
    
        __syncthreads();
    
        bool flag = true;
        int i = tid;
        
        while(finished != GPU_THREADS)
        {
            __syncthreads();     

            for(; i < R_GPU_BLOCK_SIZE && (i + task_index)<offset_number; i += GPU_THREADS)
            {
                if(do_merge)     
                    break;

                //bool success = FFGPU::map(((SO *)object_s + group_id), input, data_offset[i].offset, parameter, 1);
                bool success = map_ptr(((SO *)object_s + group_id), input, data_offset[i].offset, parameter, 1, reduce_ptr);
                //bool success = FFGPU::map(((SO *)object_s), input, data_offset[i].offset, parameter, 1);
    
                if(!success)
                {
                    do_merge = 1;
                    break;
                }
            }
    
            if(flag && ((i + task_index) >= offset_number || i >= R_GPU_BLOCK_SIZE))
            {
                flag = false;
                atomicAdd(&finished, 1);
            }
    
            __syncthreads();
    
            object_g->merge((SO *)object_s, reduce_ptr);

	    //if(gid == 0)
	    //{
	    //    merge(object_g, (SO *)object_s + group_id); 
	    //}
            
            __syncthreads();
    
            //((SO *)object_s + group_id)->oma_init(gid, group_size);
            //((SO *)object_s)->oma_init();
    
            if(tid == 0)
                do_merge = 0;
        }
#else
        for(int i = tid; (i + task_index) < offset_number; i += GPU_THREADS)
        {
            FFGPU::map(object_g, input, (char *)data_offset + unit_size * (start + i + task_index), parameter, 1); 
        }
#endif
    }
}
#endif
